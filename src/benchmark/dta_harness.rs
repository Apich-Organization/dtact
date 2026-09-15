//! Drives the real, unmodified [`crate::dta_scheduler::DtaScheduler`] over
//! synthetic closure-based tasks.
//!
//! This tests DTA's actual production queueing/mailbox/deflection/warehouse
//! code — not a reimplementation of it — while still avoiding the
//! fiber/`ContextPool` machinery, which is orthogonal to the
//! scheduling-algorithm question this benchmark asks (see module docs in
//! [`super`]).
//!
//! `DtaScheduler`'s core algorithm only ever moves an opaque `u32`
//! (`TaskIndex`) through mailboxes and local queues — it never touches
//! `ContextPool` itself (that only happens in `Worker::dispatch_loop`, which
//! this harness deliberately does not call). So this harness supplies its
//! own `TaskIndex -> boxed closure` side table ([`TaskSlab`]) and its own
//! worker loop (mirroring `DtaScheduler::run_worker_static`'s structure, but
//! calling [`crate::dta_scheduler::Worker::pop_local`] plus a synthetic
//! closure invocation in place of `Worker::dispatch_loop`'s real fiber
//! switch).

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::Scope;

use crate::common_types::TopologyMode;
use crate::dta_scheduler::{DtaScheduler, TaskIndex};

use super::instrumentation::AcquisitionMeter;
use super::numa_model::Topology;
use super::workloads::TaskSpawner;

/// A schedulable unit of work: mirrors [`super::work_stealing::Task`] so
/// both sides of the comparison store and invoke tasks identically.
pub type Task = Box<dyn FnOnce(&DtaHarness) + Send>;

/// Upper bound on how many free-list nodes one batch refill/donate moves
/// per CAS. Mirrors `MAX_LOCAL_BATCH` in `src/memory_management.rs`, kept
/// as a separate constant since this file deliberately doesn't depend on
/// production `ContextPool` internals (see `TaskSlab`'s doc comment).
const MAX_LOCAL_BATCH: usize = 32;
/// Maximum number of distinct worker ids that get a private batch cache.
/// Mirrors `MAX_CACHED_WORKERS` in `src/memory_management.rs`.
const MAX_CACHED_WORKERS: usize = 128;

/// A per-worker cache of free [`TaskSlab`] indices, sitting in front of the
/// shared free list to amortize its CAS cost — mirrors
/// `memory_management::LocalFreeCache`; kept as a separate type (rather
/// than reused across the module boundary) for the same reason `TaskSlab`
/// as a whole is a parallel implementation rather than a dependency on
/// production `ContextPool`: this benchmark's own per-task cost must stand
/// on its own, not accidentally inherit unrelated production code changes.
///
/// # Safety invariant
/// Never touched by more than one thread at a time: only ever accessed via
/// `TaskSlab::local_caches[worker_id]`, where `worker_id` comes from
/// [`crate::future_bridge::CURRENT_WORKER_ID`] — set once, for its whole
/// life, by [`DtaHarness::worker_loop`].
struct LocalFreeCache {
    slots: [u32; MAX_LOCAL_BATCH * 2],
    len: u32,
}

impl LocalFreeCache {
    const fn new() -> Self {
        Self {
            slots: [0; MAX_LOCAL_BATCH * 2],
            len: 0,
        }
    }
}

/// A fixed-capacity slot table mapping [`TaskIndex`] to a pending [`Task`]
/// closure, with a lock-free free-list — the same generation-free CAS-loop
/// free-list pattern `ContextPool::alloc_context`/`free_context` use in
/// production (`src/memory_management.rs`) for `FiberContext` slots,
/// applied here to closures instead, and for the same reason production
/// uses raw `UnsafeCell` access rather than a per-slot lock: a mutex here
/// would tax every DTA-side task cycle with a lock/unlock pair that
/// `crossbeam-deque` (storing its boxed task directly, lock-free) never
/// pays on the WS side, unfairly biasing the wall-clock comparison this
/// module exists to make. As of the per-worker `LocalFreeCache` above, it
/// also mirrors `ContextPool`'s batched-refill/donate optimization, so this
/// benchmark stays an accurate proxy for `ContextPool`'s real cost rather
/// than a stale, pessimistic one.
///
/// # Safety invariant
/// A slot index is, at every point in time, owned by exactly one of: the
/// free list (shared or a worker's local cache), or whichever call to
/// [`Self::store`] most recently produced it (until the matching
/// [`Self::take`] call, after which it returns to a free list). The
/// free-list CAS protocol below enforces this, so the `UnsafeCell` access
/// in `store`/`take` never races.
///
/// Deliberately excluded from the [`AcquisitionMeter`]: the preprint's β
/// model is about scheduling information-acquisition cost, not the
/// (analogous, but separate) context/memory allocation cost `ContextPool`
/// already pays in production.
struct TaskSlab {
    slots: Vec<UnsafeCell<Option<Task>>>,
    free_head: AtomicU64,
    next_free: Vec<AtomicU64>,
    bump: AtomicU64,
    local_caches: Box<[UnsafeCell<LocalFreeCache>]>,
    batch_size: u32,
}

// SAFETY: see the struct-level safety invariant — `UnsafeCell` access in
// `store`/`take` is protocol-exclusive, never concurrent on the same slot.
unsafe impl Sync for TaskSlab {}

impl TaskSlab {
    fn new(capacity: usize) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        let batch_size = ((capacity / 8) as u32).clamp(1, MAX_LOCAL_BATCH as u32);
        let local_caches = (0..MAX_CACHED_WORKERS)
            .map(|_| UnsafeCell::new(LocalFreeCache::new()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            slots: (0..capacity).map(|_| UnsafeCell::new(None)).collect(),
            free_head: AtomicU64::new(u64::MAX),
            next_free: (0..capacity).map(|_| AtomicU64::new(u64::MAX)).collect(),
            bump: AtomicU64::new(0),
            local_caches,
            batch_size,
        }
    }

    /// Stores `task` in a free slot and returns its index. Pops from this
    /// worker's local batch cache when possible (see [`LocalFreeCache`]),
    /// falling back to the shared free list directly (one CAS,
    /// [`Self::pop_global`]) for callers with no cache slot of their own,
    /// and finally bump-allocating a fresh slot when nothing has been
    /// freed yet.
    ///
    /// # Panics
    /// Panics if the slab's fixed capacity is exhausted (a run that needs
    /// more concurrently in-flight tasks than the harness was configured
    /// for — raise the capacity passed to [`DtaHarness::new`]).
    fn store(&self, task: Task) -> TaskIndex {
        let worker_id = crate::future_bridge::CURRENT_WORKER_ID.with(core::cell::Cell::get);
        let claimed = if worker_id < MAX_CACHED_WORKERS {
            // SAFETY: see `LocalFreeCache`'s doc comment — `worker_id`
            // uniquely identifies the one live OS thread that ever touches
            // this slot.
            let cache = unsafe { &mut *self.local_caches[worker_id].get() };
            if cache.len == 0 {
                self.refill_batch(cache);
            }
            if cache.len > 0 {
                cache.len -= 1;
                Some(cache.slots[cache.len as usize] as usize)
            } else {
                None
            }
        } else {
            self.pop_global()
        };

        let idx = claimed.unwrap_or_else(|| {
            // Free list (and this worker's cache) empty: bump-allocate a
            // fresh slot — never handed out before, so no other caller can
            // be touching it.
            let bumped = self.bump.fetch_add(1, Ordering::Relaxed);
            #[allow(clippy::cast_possible_truncation)]
            let bumped_usize = bumped as usize;
            assert!(
                bumped_usize < self.slots.len(),
                "DtaHarness TaskSlab exhausted ({} slots) — raise the capacity",
                self.slots.len()
            );
            bumped_usize
        });

        // SAFETY: `idx` was either just claimed exclusively from a free
        // list/cache, or freshly bump-allocated — either way, no other
        // caller can be touching this slot.
        unsafe { *self.slots[idx].get() = Some(task) };
        #[allow(clippy::cast_possible_truncation)]
        let idx_u32 = idx as TaskIndex;
        idx_u32
    }

    /// Uncached single-node pop directly from the shared free list, one CAS
    /// per call — today's original `store` free-list step, unchanged.
    fn pop_global(&self) -> Option<usize> {
        let mut head = self.free_head.load(Ordering::Acquire);
        loop {
            if head == u64::MAX {
                return None;
            }
            #[allow(clippy::cast_possible_truncation)]
            let idx = head as usize;
            let next = self.next_free[idx].load(Ordering::Relaxed);
            match self.free_head.compare_exchange_weak(
                head,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(idx),
                Err(actual) => head = actual,
            }
        }
    }

    /// Uncached single-node push directly onto the shared free list, one
    /// CAS per call — today's original `take` free-list step, unchanged.
    fn push_global(&self, index: TaskIndex) {
        let idx = index as usize;
        let mut head = self.free_head.load(Ordering::Relaxed);
        loop {
            self.next_free[idx].store(head, Ordering::Relaxed);
            match self.free_head.compare_exchange_weak(
                head,
                u64::from(index),
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => head = actual,
            }
        }
    }

    /// Refills `cache` from the shared free list: walks up to
    /// `self.batch_size` `next_free` links, then swings `free_head` past
    /// the whole run with **one** CAS. Mirrors
    /// `memory_management::ContextPool::refill_batch`.
    fn refill_batch(&self, cache: &mut LocalFreeCache) {
        let want = self.batch_size as usize;
        loop {
            let head = self.free_head.load(Ordering::Acquire);
            if head == u64::MAX {
                return; // shared list exhausted
            }
            let mut cursor = head;
            let mut collected = 0usize;
            while collected < want && cursor != u64::MAX {
                #[allow(clippy::cast_possible_truncation)]
                let cursor_u32 = cursor as u32;
                cache.slots[collected] = cursor_u32;
                collected += 1;
                #[allow(clippy::cast_possible_truncation)]
                let cursor_usize = cursor as usize;
                cursor = self.next_free[cursor_usize].load(Ordering::Relaxed);
            }
            // `cursor` is now the first node NOT claimed (possibly u64::MAX).
            let cas = self.free_head.compare_exchange_weak(
                head,
                cursor,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            if cas.is_ok() {
                #[allow(clippy::cast_possible_truncation)]
                let collected_u32 = collected as u32;
                cache.len = collected_u32;
                return;
            }
            // List changed under us — fall through to retry the walk from the fresh head.
        }
    }

    /// Donates `self.batch_size` slots from `cache` back to the shared free
    /// list with **one** CAS. Mirrors
    /// `memory_management::ContextPool::donate_batch`.
    fn donate_batch(&self, cache: &mut LocalFreeCache) {
        let n = self.batch_size as usize;
        debug_assert!(cache.len as usize >= n);
        let start = cache.len as usize - n;

        for i in start..start + n - 1 {
            let idx = cache.slots[i] as usize;
            self.next_free[idx].store(u64::from(cache.slots[i + 1]), Ordering::Relaxed);
        }
        let chain_head = cache.slots[start];
        let tail_idx = cache.slots[start + n - 1] as usize;

        let mut head = self.free_head.load(Ordering::Relaxed);
        loop {
            self.next_free[tail_idx].store(head, Ordering::Relaxed);
            match self.free_head.compare_exchange_weak(
                head,
                u64::from(chain_head),
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => head = actual,
            }
        }
        #[allow(clippy::cast_possible_truncation)]
        let n_u32 = n as u32;
        cache.len -= n_u32;
    }

    /// Takes ownership of the task at `index`, returning it to this
    /// worker's local batch cache (see [`LocalFreeCache`]) — donating a
    /// batch back to the shared free list when the cache fills up. Callers
    /// with no cache slot of their own fall back to [`Self::push_global`]
    /// directly.
    fn take(&self, index: TaskIndex) -> Task {
        let idx = index as usize;
        // SAFETY: the caller holds `index` because it was just popped from
        // the scheduler's queue, which only ever contains indices this
        // slab's `store` produced and that haven't been taken yet
        // (struct-level invariant) — so this is the slot's sole owner.
        let task = unsafe { &mut *self.slots[idx].get() }
            .take()
            .expect("DtaHarness TaskSlab: double-take on a task index");

        let worker_id = crate::future_bridge::CURRENT_WORKER_ID.with(core::cell::Cell::get);
        if worker_id < MAX_CACHED_WORKERS {
            // SAFETY: see `LocalFreeCache`'s doc comment.
            let cache = unsafe { &mut *self.local_caches[worker_id].get() };
            if cache.len as usize == 2 * self.batch_size as usize {
                self.donate_batch(cache);
            }
            cache.slots[cache.len as usize] = index;
            cache.len += 1;
        } else {
            self.push_global(index);
        }
        task
    }
}

/// Drives the real `DtaScheduler` over synthetic closure tasks.
pub struct DtaHarness {
    scheduler: DtaScheduler,
    slab: TaskSlab,
    topology: Topology,
    /// Real-time information-acquisition measurements for this run.
    pub meter: AcquisitionMeter,
    /// Per-worker completed-task counts — a separate empirical claim from
    /// the information-acquisition-rate model (`meter`): this measures
    /// load-balance *quality* (how evenly work ends up spread), which the
    /// preprint's stability/`Δq`-imbalance analysis is about, not the cost
    /// of finding work. See [`super::instrumentation::load_balance_stats`].
    pub per_worker_completed: Vec<AtomicU64>,
    shutdown: AtomicBool,
}

impl DtaHarness {
    /// Creates a harness whose `DtaScheduler` has `topology.total_cores()`
    /// workers, with a task slab sized for `task_capacity` concurrently
    /// in-flight (spawned but not yet completed) tasks.
    #[must_use]
    pub fn new(topology: Topology, task_capacity: usize) -> Self {
        let n = topology.total_cores();
        // `TopologyMode::Global`, not the production-default `P2PMesh`:
        // `P2PMesh` restricts `enqueue_deflect`'s target selection to the
        // source's own 8-core CCX group (`src/dta_scheduler.rs`,
        // `enqueue_deflect`'s final `else` branch) regardless of load —
        // for every worker count this harness sweeps (multiples of 16),
        // that CCX boundary always sits inside one declared virtual
        // socket, so DTA could *never* cross a socket under `P2PMesh` no
        // matter the deflection threshold. `Global` deflects across the
        // full `N`-worker range, matching what the preprint's dual-socket
        // worked example (`paper/main.tex` §`numa_concrete`) actually
        // analyzes: a scheduler whose deflection can reach the whole
        // declared topology, not a locality-pinned production default.
        Self {
            scheduler: DtaScheduler::new(n, TopologyMode::Global),
            slab: TaskSlab::new(task_capacity),
            topology,
            meter: AcquisitionMeter::new(n),
            per_worker_completed: (0..n).map(|_| AtomicU64::new(0)).collect(),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Sets the deflection-load threshold on every worker (default `80`,
    /// `src/dta_scheduler.rs`'s `Worker::new`). Lower values make DTA
    /// deflect more eagerly at a given queue depth — useful for a
    /// benchmark harness whose task volumes may not otherwise build up
    /// enough per-worker backlog to cross the production default.
    pub fn set_deflection_threshold(&self, threshold: u8) {
        for core in 0..self.worker_count() {
            let worker = unsafe { &*self.scheduler.workers[core].get() };
            worker
                .deflection_threshold
                .store(threshold, Ordering::Release);
        }
    }

    /// Number of workers in this harness's declared topology.
    #[must_use]
    pub const fn worker_count(&self) -> usize {
        self.scheduler.workers.len()
    }

    /// Publishes this harness's topology and meter as the active
    /// [`super::report_dta_hop`] target, spawns one OS thread per worker
    /// into `scope` running the synthetic dispatch loop, and returns.
    ///
    /// Callers must call [`Self::request_shutdown`], let the
    /// `std::thread::scope` block return (joining every worker thread),
    /// and only then call [`Self::end_measurement`] — in that order — so no
    /// worker thread can ever observe a retracted measurement target while
    /// still running.
    pub fn run_workers<'scope>(&'scope self, scope: &'scope Scope<'scope, '_>) {
        super::begin_measurement(self.topology, &self.meter);
        for core in 0..self.worker_count() {
            scope.spawn(move || self.worker_loop(core));
        }
    }

    /// Retracts this harness as the active measurement target. See
    /// [`Self::run_workers`] for the required call ordering.
    pub fn end_measurement(&self) {
        super::end_measurement();
    }

    /// Signals every worker thread to exit its dispatch loop once it next
    /// checks. Callers must join the `std::thread::scope` afterwards.
    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    fn worker_loop(&self, core: usize) {
        crate::future_bridge::CURRENT_WORKER_ID.with(|c| c.set(core));
        let mut idle_spins: u32 = 0;

        loop {
            if self.shutdown.load(Ordering::Acquire) {
                return;
            }

            let warehouse_busy = self.scheduler.warehouse.is_busy();
            let mut activity = if warehouse_busy {
                self.scheduler.drain_warehouse(core)
            } else {
                false
            };

            let worker = unsafe { &*self.scheduler.workers[core].get() };
            loop {
                // Real measured per-task SPSC-read charge (`β_DTA(N) =
                // Nλ·c_SPSC`): every dispatch, local or not, is one
                // SPSC-style observation in DTA's design (even the local
                // queue is an SPSC ring, not a free LIFO pop) — timing the
                // actual `pop_local` call, exactly like the WS baseline
                // times its own local `Deque::pop()`
                // (`work_stealing.rs::worker_loop`), rather than charging a
                // flat theoretical constant: a hardcoded charge would make
                // the "empirical vs. theoretical" ratio this benchmark
                // reports tautological for DTA specifically, always
                // reading exactly 1.0 regardless of what's actually
                // measured. `report_dta_hop` (called from
                // `dta_scheduler.rs` on an actual cross-worker push)
                // charges only the *additional* cross-socket penalty on
                // top of this measured base cost.
                let pop_timer = std::time::Instant::now();
                let popped = worker.pop_local();
                #[allow(clippy::cast_possible_truncation)]
                let pop_ns = (pop_timer.elapsed().as_nanos() as u64).max(1);
                let Some(task_idx) = popped else { break };
                activity = true;
                self.meter.record_spsc(core, pop_ns);
                let task = self.slab.take(task_idx);
                // `load_level` refresh during a same-core enqueue burst
                // happens inside the real, unmodified `Worker::push_local`
                // (`src/dta_scheduler.rs`, see `LOAD_REFRESH_PERIOD`) —
                // this harness routes every spawn through the real
                // `enqueue_deflect`, which calls `push_local` on its
                // same-core fast path, so no separate copy is needed here.
                task(self);
                self.meter.record_task_completed(core);
                self.per_worker_completed[core].fetch_add(1, Ordering::Relaxed);
            }

            if !warehouse_busy {
                activity |= self.scheduler.poll_mailboxes(core);
            }

            if activity {
                idle_spins = 0;
                continue;
            }

            idle_spins = idle_spins.saturating_add(1);
            if idle_spins < 512 {
                core::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// Measures the real, single-threaded (uncontended) cost of one
/// [`TaskSlab::store`] + [`TaskSlab::take`] round trip on the *cached* fast
/// path (worker id 0, mirroring `ContextPool`'s per-worker
/// `LocalFreeCache` optimization) — the indirection every DTA-harness task
/// pays that a directly-stored `crossbeam-deque` task (the WS baseline)
/// does not, and which the preprint's `β_DTA` formula (`paper/main.tex` eq.
/// `beta_dta_num`, a flat `c_SPSC` per task) does not model at all: the
/// paper's cost accounting is about moving a task *reference* between
/// workers' queues, not about resolving that reference to its payload.
/// This exists to let `benches/numa_information_cost.rs` attribute how
/// much of DTA's measured wall-clock overhead (over the WS baseline) this
/// specific indirection tax accounts for, separate from the
/// scheduling/mailbox traffic itself.
///
/// Uses a 64-slot capacity (`batch_size = 8`, see `TaskSlab::new`) so the
/// cached path's batching actually engages rather than degenerating to the
/// same per-op-CAS cost a tiny pool would clamp down to.
///
/// Returns the mean nanoseconds per store+take round trip over
/// `iterations` repetitions on the calling thread (single-threaded: the
/// free list's CAS loop never actually contends with itself here, so this
/// is TaskSlab's *best-case* cost — a lower bound on its real contribution
/// under concurrent load).
#[doc(hidden)]
#[must_use]
#[allow(unused_must_use)]
pub fn microbench_slab_roundtrip_ns(iterations: u32) -> f64 {
    let slab = TaskSlab::new(64);
    crate::future_bridge::CURRENT_WORKER_ID.with(|c| c.set(0));
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let idx = slab.store(Box::new(|_: &DtaHarness| {}));
        // The closure is never invoked: only the store+take round trip
        // (allocation/free-list bookkeeping) is under measurement here, not
        // task execution. `black_box` still forces the compiler to treat
        // the returned `Task` as observed so the store/take pair can't be
        // optimized away.
        core::hint::black_box(slab.take(idx));
    }
    let elapsed = start.elapsed();
    crate::future_bridge::CURRENT_WORKER_ID.with(|c| c.set(usize::MAX));
    #[allow(clippy::cast_precision_loss)]
    let ns = elapsed.as_secs_f64() * 1e9 / f64::from(iterations);
    ns
}

impl TaskSpawner for DtaHarness {
    fn spawn(&self, task: Task) {
        let idx = self.slab.store(task);
        let current = crate::future_bridge::CURRENT_WORKER_ID.with(core::cell::Cell::get);
        let n = self.worker_count();
        if current < n {
            // Called from within a running synthetic task on a worker
            // thread: route exactly as a real deflectable spawn would,
            // through the scheduler's own deflection policy.
            let _ = self.scheduler.enqueue_deflect(
                current,
                u64::from(idx),
                idx,
                crate::api::topology::Affinity::Any,
            );
        } else {
            // Called from the harness/main thread (not a worker OS thread).
            // Real production code (`src/api.rs`'s host-thread `spawn` path)
            // routes this exact case via the *calling thread's own* real
            // CPU core id (`topology::current().core_id % n`), not a fixed
            // worker — so different host threads (or repeated calls from
            // one host thread migrating across CPUs) naturally spread
            // across workers instead of funneling through a single one.
            // This harness must match that, not hardcode worker 0: this
            // path is exercised not just once (a single root-task spawn,
            // where the choice of target wouldn't matter) but potentially
            // many times per run — e.g. every arrival in an open-loop
            // Poisson-BoT workload (`workloads::run_poisson_bot`) is a
            // fresh off-worker `spawn` call. Hardcoding worker 0 there
            // would turn every arrival into a serialized bottleneck on one
            // worker's mailbox/queue before DTA's real deflection policy
            // ever gets a chance to run — a harness artifact, not real DTA
            // behavior, and not something WS's shared, N-way-stealable
            // `Injector` has an equivalent problem with.
            let core = crate::api::topology::current().core_id as usize % n;
            let _ = self.scheduler.enqueue_deflect(
                core,
                u64::from(idx),
                idx,
                crate::api::topology::Affinity::Any,
            );
        }
    }
}

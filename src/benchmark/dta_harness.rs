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

/// A fixed-capacity slot table mapping [`TaskIndex`] to a pending [`Task`]
/// closure, with a lock-free free-list — the same generation-free CAS-loop
/// free-list pattern `ContextPool::alloc_context`/`free_context` use in
/// production (`src/memory_management.rs`) for `FiberContext` slots,
/// applied here to closures instead, and for the same reason production
/// uses raw `UnsafeCell` access rather than a per-slot lock: a mutex here
/// would tax every DTA-side task cycle with a lock/unlock pair that
/// `crossbeam-deque` (storing its boxed task directly, lock-free) never
/// pays on the WS side, unfairly biasing the wall-clock comparison this
/// module exists to make.
///
/// # Safety invariant
/// A slot index is, at every point in time, owned by exactly one of: the
/// free list, or whichever call to [`Self::store`] most recently produced
/// it (until the matching [`Self::take`] call, after which it returns to
/// the free list). The free-list CAS protocol below enforces this, so the
/// `UnsafeCell` access in `store`/`take` never races.
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
}

// SAFETY: see the struct-level safety invariant — `UnsafeCell` access in
// `store`/`take` is protocol-exclusive, never concurrent on the same slot.
unsafe impl Sync for TaskSlab {}

impl TaskSlab {
    fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| UnsafeCell::new(None)).collect(),
            free_head: AtomicU64::new(u64::MAX),
            next_free: (0..capacity).map(|_| AtomicU64::new(u64::MAX)).collect(),
            bump: AtomicU64::new(0),
        }
    }

    /// Stores `task` in a free slot and returns its index.
    ///
    /// # Panics
    /// Panics if the slab's fixed capacity is exhausted (a run that needs
    /// more concurrently in-flight tasks than the harness was configured
    /// for — raise the capacity passed to [`DtaHarness::new`]).
    fn store(&self, task: Task) -> TaskIndex {
        // Try the free-list first (Treiber-stack pop).
        let mut head = self.free_head.load(Ordering::Acquire);
        loop {
            if head == u64::MAX {
                break;
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
                Ok(_) => {
                    // SAFETY: winning this CAS is what grants exclusive
                    // ownership of slot `idx` (struct-level invariant).
                    unsafe { *self.slots[idx].get() = Some(task) };
                    #[allow(clippy::cast_possible_truncation)]
                    return idx as TaskIndex;
                }
                Err(actual) => head = actual,
            }
        }
        // Free-list empty: bump-allocate a fresh slot.
        let idx = self.bump.fetch_add(1, Ordering::Relaxed);
        #[allow(clippy::cast_possible_truncation)]
        let idx_usize = idx as usize;
        assert!(
            idx_usize < self.slots.len(),
            "DtaHarness TaskSlab exhausted ({} slots) — raise the capacity",
            self.slots.len()
        );
        // SAFETY: a freshly bump-allocated index has never been handed out
        // before, so no other caller can be touching this slot.
        unsafe { *self.slots[idx_usize].get() = Some(task) };
        #[allow(clippy::cast_possible_truncation)]
        let idx_u32 = idx as TaskIndex;
        idx_u32
    }

    /// Takes ownership of the task at `index`, returning it to the free
    /// list for reuse.
    fn take(&self, index: TaskIndex) -> Task {
        let idx = index as usize;
        // SAFETY: the caller holds `index` because it was just popped from
        // the scheduler's queue, which only ever contains indices this
        // slab's `store` produced and that haven't been taken yet
        // (struct-level invariant) — so this is the slot's sole owner.
        let task = unsafe { &mut *self.slots[idx].get() }
            .take()
            .expect("DtaHarness TaskSlab: double-take on a task index");
        let mut head = self.free_head.load(Ordering::Relaxed);
        loop {
            self.next_free[idx].store(head, Ordering::Relaxed);
            match self.free_head.compare_exchange_weak(
                head,
                index.into(),
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => head = actual,
            }
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
            meter: AcquisitionMeter::new(),
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
            while let Some(task_idx) = worker.pop_local() {
                activity = true;
                // Flat per-task SPSC-read charge (`β_DTA(N) = Nλ·c_SPSC`):
                // every dispatch, local or not, is one SPSC-style
                // observation in DTA's design (even the local queue is an
                // SPSC ring, not a free LIFO pop). `report_dta_hop` (called
                // from `dta_scheduler.rs` on an actual cross-worker push)
                // charges only the *additional* cross-socket penalty on
                // top of this base cost, so a purely local dispatch still
                // registers DTA's constant baseline instead of reading as
                // zero acquisition cost.
                self.meter.record_spsc(super::numa_model::SPSC_COST_NS);
                let task = self.slab.take(task_idx);
                // `load_level` refresh during a same-core enqueue burst
                // happens inside the real, unmodified `Worker::push_local`
                // (`src/dta_scheduler.rs`, see `LOAD_REFRESH_PERIOD`) —
                // this harness routes every spawn through the real
                // `enqueue_deflect`, which calls `push_local` on its
                // same-core fast path, so no separate copy is needed here.
                task(self);
                self.meter.record_task_completed();
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
            // Called from the harness/main thread (seeding the root task):
            // route to worker 0, mirroring how a host thread's first spawn
            // has no "current core" of its own to prefer.
            let _ = self.scheduler.enqueue_deflect(
                0,
                u64::from(idx),
                idx,
                crate::api::topology::Affinity::Any,
            );
        }
    }
}

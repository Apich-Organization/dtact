//! A literature-faithful "pure" work-stealing (WS) baseline scheduler.
//!
//! "Pure" here means the same thing the preprint's theory means by WS
//! (`paper/main.tex`, Theorem "DTA vs. WS in the (α,β) plane"): a scheduler
//! with **no topology awareness** at all — every idle worker picks a victim
//! **uniformly at random among all `N` workers**, exactly like the classical
//! randomized work-stealing analyzed by Blumofe & Leiserson and by Arora,
//! Blumofe & Plaxton. This is deliberately *not* NUMA-aware stealing (that
//! would be a different, hybrid algorithm, and isn't what the preprint's
//! `β_WS(N) = Nλ·(δ̄ + Ω(log N)·c_CAS)` formula models) — the whole point of
//! the comparison is that DTA's hop-bounded, locality-first deflection
//! *is* topology-aware and WS's uniform-random stealing structurally is not.
//!
//! Built on `crossbeam-deque`'s Chase-Lev deque implementation (the same
//! lock-free deque family real-world work-stealing runtimes — Tokio, Rayon,
//! Java's `ForkJoinPool` — use), rather than a hand-rolled one, so the
//! comparison baseline is not weakened or strengthened by an amateur
//! reimplementation of a well-studied data structure.

use std::cell::{Cell, RefCell};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::Scope;
use std::time::Instant;

use crossbeam_deque::{Injector, Steal, Stealer, Worker as Deque};

use super::instrumentation::AcquisitionMeter;
use super::numa_model::{self, CAS_BASE_NS, Topology};
use super::workloads::TaskSpawner;

/// A schedulable unit of work: mirrors [`super::dta_harness`]'s task
/// representation so both sides of the comparison store and invoke tasks
/// identically.
pub type Task = Box<dyn FnOnce(&WsScheduler) + Send>;

thread_local! {
    /// This OS thread's own Chase-Lev deque half, if it is a WS worker
    /// thread. `None` on the harness/main thread and any other non-worker
    /// caller — those fall back to the shared injector in
    /// [`WsScheduler::spawn`].
    static LOCAL_DEQUE: RefCell<Option<Deque<Task>>> = const { RefCell::new(None) };
    /// This OS thread's worker index (core id) under the active
    /// [`WsScheduler`], or `usize::MAX` if it is not a worker thread.
    static CURRENT_CORE: Cell<usize> = const { Cell::new(usize::MAX) };
    /// Per-thread xorshift64 state for uniform random victim selection.
    static RNG_STATE: Cell<u64> = Cell::new(0);
}

fn thread_rng_next(bound: usize) -> usize {
    RNG_STATE.with(|cell| {
        let mut x = cell.get();
        if x == 0 {
            // Lazily seed from the thread id's hash and the current time so
            // distinct worker threads (and distinct runs) get distinct
            // streams without pulling in an external RNG crate.
            let addr = std::ptr::addr_of!(x) as u64;
            let t = u64::from(Instant::now().elapsed().subsec_nanos());
            x = addr ^ t ^ 0x2545_F491_4F6C_DD1D;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        cell.set(x);
        #[allow(clippy::cast_possible_truncation)]
        let r = (x % bound as u64) as usize;
        r
    })
}

/// A pure, topology-oblivious work-stealing scheduler over `N` workers,
/// instrumented with the same [`AcquisitionMeter`] protocol
/// [`super::dta_harness`] uses so the two are directly comparable.
pub struct WsScheduler {
    injector: Injector<Task>,
    stealers: Vec<Stealer<Task>>,
    /// Virtual NUMA topology used only to charge the synthetic cross-socket
    /// penalty on a steal — the scheduling *decision* itself never
    /// consults this (see module docs: WS is topology-oblivious by
    /// definition).
    topology: Topology,
    /// Real-time information-acquisition measurements for this run.
    pub meter: AcquisitionMeter,
    /// Per-worker completed-task counts — see
    /// [`super::dta_harness::DtaHarness::per_worker_completed`] for why
    /// this is tracked separately from `meter`.
    pub per_worker_completed: Vec<AtomicU64>,
    shutdown: AtomicBool,
    /// Each worker's owned deque half, taken exactly once by
    /// [`Self::run_workers`] and moved onto its owning OS thread. Deques
    /// cannot live directly in `Self` because `Deque<Task>` is `!Sync` —
    /// only the `Stealer` halves (in `stealers`) are meant to be shared.
    /// A `Mutex` (rather than `RefCell`) is required here specifically so
    /// `WsScheduler` itself stays `Sync` and can be shared as `&'scope
    /// WsScheduler` across the scoped worker threads.
    pending_deques: Mutex<Option<Vec<Deque<Task>>>>,
}

impl WsScheduler {
    /// Creates a new scheduler for `topology.total_cores()` workers. Call
    /// [`Self::run_workers`] inside a [`std::thread::scope`] before
    /// spawning any tasks.
    #[must_use]
    pub fn new(topology: Topology) -> Self {
        let n = topology.total_cores();
        let deques: Vec<Deque<Task>> = (0..n).map(|_| Deque::new_lifo()).collect();
        let stealers = deques.iter().map(Deque::stealer).collect();
        Self {
            injector: Injector::new(),
            stealers,
            topology,
            meter: AcquisitionMeter::new(),
            per_worker_completed: (0..n).map(|_| AtomicU64::new(0)).collect(),
            shutdown: AtomicBool::new(false),
            pending_deques: Mutex::new(Some(deques)),
        }
    }

    /// Spawns one OS thread per worker into `scope`, running the main
    /// steal loop on each. Must be called exactly once, before any task is
    /// spawned.
    ///
    /// # Panics
    /// Panics if called more than once on the same scheduler.
    pub fn run_workers<'scope>(&'scope self, scope: &'scope Scope<'scope, '_>) {
        let deques = self
            .pending_deques
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
            .expect("WsScheduler::run_workers called more than once");
        for (i, dq) in deques.into_iter().enumerate() {
            scope.spawn(move || {
                CURRENT_CORE.with(|c| c.set(i));
                LOCAL_DEQUE.with(|c| *c.borrow_mut() = Some(dq));
                worker_loop(self, i);
            });
        }
    }

    /// Signals every worker thread to exit its steal loop once it next
    /// checks. Callers must join the [`std::thread::scope`] afterwards to
    /// observe termination.
    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    /// Number of workers in this scheduler's declared topology.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.stealers.len()
    }
}

impl TaskSpawner for WsScheduler {
    fn spawn(&self, task: Task) {
        let on_worker = CURRENT_CORE.with(Cell::get) != usize::MAX;
        if on_worker {
            LOCAL_DEQUE.with(|c| {
                let guard = c.borrow();
                let dq = guard
                    .as_ref()
                    .expect("worker thread missing its local deque");
                dq.push(task);
            });
        } else {
            self.injector.push(task);
        }
    }
}

fn worker_loop(sched: &WsScheduler, my_core: usize) {
    let mut idle_spins: u32 = 0;

    loop {
        if sched.shutdown.load(Ordering::Acquire) {
            return;
        }

        let popped = LOCAL_DEQUE.with(|c| {
            let guard = c.borrow();
            guard.as_ref().and_then(Deque::pop)
        });
        if let Some(task) = popped {
            task(sched);
            sched.meter.record_task_completed();
            sched.per_worker_completed[my_core].fetch_add(1, Ordering::Relaxed);
            idle_spins = 0;
            continue;
        }

        // Global injector: a Treiber-stack-family MPMC structure, so
        // draining it is itself a CAS-class information-acquisition event.
        let inj_timer = Instant::now();
        let inj_result = LOCAL_DEQUE.with(|c| {
            let guard = c.borrow();
            let dq = guard.as_ref().expect("worker missing local deque");
            sched.injector.steal_batch_and_pop(dq)
        });
        #[allow(clippy::cast_possible_truncation)]
        let inj_ns = (inj_timer.elapsed().as_nanos() as u64).max(1);
        match inj_result {
            Steal::Success(task) => {
                sched.meter.record_cas(inj_ns, true);
                task(sched);
                sched.meter.record_task_completed();
                sched.per_worker_completed[my_core].fetch_add(1, Ordering::Relaxed);
                idle_spins = 0;
                continue;
            }
            Steal::Retry => {
                sched.meter.record_cas(inj_ns, false);
            }
            Steal::Empty => {}
        }

        // Uniform-random steal among all peers (topology-oblivious, per
        // module docs). Skipped entirely for a degenerate 1-worker run.
        let n = sched.stealers.len();
        if n > 1 {
            let mut victim = thread_rng_next(n);
            if victim == my_core {
                victim = (victim + 1) % n;
            }

            let read_timer = Instant::now();
            let steal_result = sched.stealers[victim].steal();
            #[allow(clippy::cast_possible_truncation)]
            let read_ns = (read_timer.elapsed().as_nanos() as u64).max(1);
            // The read itself: real measured cost of contacting the
            // victim's queue, plus the calibrated cross-socket penalty if
            // this victim happens to live in a different virtual socket —
            // together, this is the `δ̄` (topology-averaged SPSC-equivalent
            // read cost) term in the preprint's β_WS formula. Both
            // components are folded into one recorded event (rather than
            // charging the penalty separately) since they represent a
            // single logical observation.
            let penalty_ns =
                numa_model::charge_cross_socket_if_needed(&sched.topology, my_core, victim);
            sched.meter.record_spsc(read_ns + penalty_ns);

            match steal_result {
                Steal::Success(task) => {
                    sched.meter.record_cas(CAS_BASE_NS, true);
                    task(sched);
                    sched.meter.record_task_completed();
                    sched.per_worker_completed[my_core].fetch_add(1, Ordering::Relaxed);
                    idle_spins = 0;
                    continue;
                }
                Steal::Retry => {
                    // Lost a race for the same slot as another thief: the
                    // `Ω(log N)` CAS-contention term (paper's CAS-contention
                    // lemma) accumulates from events like this one.
                    sched.meter.record_cas(CAS_BASE_NS, false);
                }
                Steal::Empty => {}
            }
        }

        idle_spins = idle_spins.saturating_add(1);
        if idle_spins < 512 {
            core::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}

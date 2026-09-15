//! Academic comparison harness: DTA vs. pure work-stealing (WS) under a
//! synthetic NUMA cost model.
//!
//! This module exists to empirically validate the closed-form
//! information-acquisition-rate formulas derived in the DTA preprint
//! (`paper/main.tex`, secs. "Information Acquisition Rate" and "Concrete
//! Analysis on a Dual-Socket NUMA Topology"), rather than to re-run the
//! engineering-only Tokio comparison already in `benches/scheduler_efficiency.rs`
//! (which the preprint cites only as a secondary, reference-quality data
//! point, not as the paper's formal empirical claim).
//!
//! ## Method summary
//! * [`dta_harness`] drives the real, unmodified [`crate::dta_scheduler::DtaScheduler`]
//!   over synthetic closure-based tasks (decoupled from `ContextPool`/fiber
//!   execution, since the fiber-switch cost is orthogonal to the scheduling
//!   question and already characterised separately).
//! * [`work_stealing`] implements a literature-faithful "pure" work-stealing
//!   baseline (uniform-random victim selection, no NUMA awareness — the same
//!   definition of WS the preprint's theory assumes) on top of
//!   `crossbeam-deque`.
//! * [`numa_model`] supplies the calibrated synthetic NUMA distance model,
//!   using the preprint's own published constants
//!   (`δ_intra = 80ns`, `δ_inter = 300ns`, `c_CAS = 100ns`), since this
//!   development machine has no real second NUMA node to measure.
//! * [`instrumentation`] counts and times the actual information-acquisition
//!   events (SPSC reads, CAS attempts) each scheduler performs, so the
//!   empirical β can be compared directly against the preprint's formulas.
//! * [`workloads`] provides two literature-standard task-graph generators —
//!   fork-join Fibonacci and Unbalanced Tree Search (UTS) — in place of the
//!   ad-hoc "sum 100 numbers" task body used by the engineering benchmarks.
//!
//! Gated entirely behind the `benchmark` Cargo feature: none of this code
//! exists in a default build.

#![allow(clippy::missing_panics_doc)]

/// Real-`DtaScheduler` harness driving synthetic closure tasks.
pub mod dta_harness;
/// Instrumentation: counts and times information-acquisition events.
pub mod instrumentation;
/// Synthetic NUMA topology and calibrated cross-node cost model.
pub mod numa_model;
/// Pure work-stealing baseline scheduler (crossbeam-deque based).
pub mod work_stealing;
/// Literature-standard task-graph workloads (Fibonacci, UTS).
pub mod workloads;

use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// A process-global, swappable "active measurement" slot: lets the
/// unmodified `DtaScheduler` production code (`src/dta_scheduler.rs`)
/// report a cross-worker hop (mailbox push, deflection, warehouse park) to
/// whichever [`instrumentation::AcquisitionMeter`] the sweep harness is
/// currently measuring, without the production scheduler code taking a
/// hard dependency on this module, and without restricting the harness to
/// installing a hook only once per process (a plain `OnceLock` would force
/// one OS process per swept configuration; a small number of relaxed/
/// acquire-release atomics lets the whole sweep run in one process instead).
///
/// `sockets == 0` (the default, and the only possible state outside an
/// active `benchmark`-feature measurement) means "not being measured" —
/// [`report_dta_hop`] checks this with a single acquire load and no-ops if
/// unset, so normal (non-benchmark) use of `DtaScheduler` is entirely
/// unaffected. Call [`begin_measurement`] / [`end_measurement`] around each
/// swept configuration; `end_measurement` must only be called after every
/// worker thread that could call [`report_dta_hop`] has already joined
/// (e.g. after a `std::thread::scope` block returns), so no hop can ever
/// observe a dangling meter pointer.
static ACTIVE_SOCKETS: AtomicUsize = AtomicUsize::new(0);
static ACTIVE_CORES_PER_SOCKET: AtomicUsize = AtomicUsize::new(0);
static ACTIVE_METER: AtomicPtr<instrumentation::AcquisitionMeter> =
    AtomicPtr::new(core::ptr::null_mut());

/// Publishes `(topology, meter)` as the active measurement target for
/// subsequent [`report_dta_hop`] calls from any thread.
pub(crate) fn begin_measurement(
    topology: numa_model::Topology,
    meter: &instrumentation::AcquisitionMeter,
) {
    ACTIVE_METER.store(core::ptr::from_ref(meter).cast_mut(), Ordering::Relaxed);
    ACTIVE_CORES_PER_SOCKET.store(topology.cores_per_socket, Ordering::Relaxed);
    // Release: publishes the two relaxed stores above along with it: any
    // thread that observes `ACTIVE_SOCKETS != 0` via Acquire also observes
    // a fully-initialized meter pointer and core-per-socket count.
    ACTIVE_SOCKETS.store(topology.sockets, Ordering::Release);
}

/// Retracts the active measurement target. Callers must guarantee no
/// worker thread that could call [`report_dta_hop`] is still running.
pub(crate) fn end_measurement() {
    ACTIVE_SOCKETS.store(0, Ordering::Release);
    ACTIVE_METER.store(core::ptr::null_mut(), Ordering::Relaxed);
}

/// Called from `src/dta_scheduler.rs`'s successful cross-worker mailbox
/// push sites. Charges the synthetic cross-socket penalty on top of the
/// flat per-task SPSC cost [`dta_harness`] already recorded for this task's
/// dispatch — a no-op (charges nothing extra) for a same-socket hop, and a
/// no-op entirely unless a benchmark run is currently active.
///
/// The flat per-task cost is recorded once per dispatch regardless of
/// whether the task ever crosses a worker boundary (matching the
/// preprint's `β_DTA(N) = Nλ·c_SPSC` formula, which assumes every task
/// costs exactly one SPSC-style observation); this function only adds the
/// *additional* cost a real cross-socket hop would have incurred.
pub(crate) fn report_dta_hop(source: usize, target: usize) {
    let sockets = ACTIVE_SOCKETS.load(Ordering::Acquire);
    if sockets == 0 {
        return;
    }
    let cores_per_socket = ACTIVE_CORES_PER_SOCKET.load(Ordering::Relaxed);
    let meter_ptr = ACTIVE_METER.load(Ordering::Relaxed);
    if meter_ptr.is_null() {
        return;
    }
    let topology = numa_model::Topology {
        sockets,
        cores_per_socket,
    };
    let penalty_ns = numa_model::charge_cross_socket_if_needed(&topology, source, target);
    if penalty_ns > 0 {
        // SAFETY: non-null only while a measurement is active, and
        // `end_measurement` is only called after every worker thread that
        // could reach this function has already joined (see doc comment
        // above).
        let meter = unsafe { &*meter_ptr };
        // Attributed to `source`'s shard: that's the worker whose dispatch
        // (popping the task that's now being spawned/deflected) already
        // recorded this task's base SPSC charge, so the extra cross-socket
        // cost lands on the same shard it's extending.
        meter.record_extra_ns(source, penalty_ns);
    }
}

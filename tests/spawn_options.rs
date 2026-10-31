//! Tests for the spawn parameters that the C ABI, the macros and the
//! per-fiber topology override expose: everything here asserts an observable
//! scheduling or context effect, not merely that a setter accepted a value.

#![allow(dead_code)]

mod common;

use dtact::api::fiber::ClosureSpawnOptions;
use dtact::dta_scheduler::{DtaScheduler, LOCAL_QUEUE_HIGH_WATERMARK, TopologyMode};
use dtact::memory_management::SafetyLevel;
use dtact::{Affinity, Priority, WorkloadKind, dtact_await, fiber_info, spawn_with};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Two CCXs' worth of workers (`ccx_id = i / 8`), so a CCX-confined target
/// and a globally-chosen one are always distinguishable.
const WIDE_WORKERS: usize = 16;

/// Finds the one mailbox row `producer` filled, if any, and returns the
/// target worker index it went to.
fn mailbox_target(scheduler: &DtaScheduler, producer: usize) -> Option<usize> {
    let row = scheduler.mailboxes.get(producer)?;
    row.iter()
        .position(|mb| mb.head.load(Ordering::SeqCst) != mb.tail.load(Ordering::SeqCst))
}

#[test]
#[cfg_attr(miri, ignore)]
fn topology_mode_override_changes_the_deflection_neighbourhood() {
    // flow_id 50 gives h1 = 2, h2 = 7, so `source + h1 + h2 = 9`:
    //   P2PMesh confines the target to the source's CCX (`9 & 7 = 1`)
    //   Global may place it anywhere (`9 % 16 = 9`, i.e. the *other* CCX)
    let scheduler = DtaScheduler::new(WIDE_WORKERS, TopologyMode::P2PMesh);
    dtact::future_bridge::__set_current_worker_id_for_test(0);
    unsafe {
        let worker = &*scheduler.workers[0].get();
        worker.load_level.store(255, Ordering::SeqCst);
        worker.deflection_threshold.store(0, Ordering::SeqCst);
    }

    scheduler.enqueue_deflect(
        0,
        50,
        0,
        Affinity::Any,
        TopologyMode::P2PMesh,
        Priority::Normal,
    );
    assert_eq!(
        mailbox_target(&scheduler, 0),
        Some(1),
        "a P2PMesh fiber must be deflected inside its own CCX"
    );

    // Drain that mailbox so the second call's target is unambiguous.
    while scheduler.mailboxes[0][1].pop().is_some() {}

    scheduler.enqueue_deflect(
        0,
        50,
        1,
        Affinity::Any,
        TopologyMode::Global,
        Priority::Normal,
    );
    assert_eq!(
        mailbox_target(&scheduler, 0),
        Some(9),
        "a Global fiber must be deflected across CCX boundaries"
    );

    dtact::future_bridge::__set_current_worker_id_for_test(usize::MAX);
}

#[test]
#[cfg_attr(miri, ignore)]
fn pinned_mode_never_deflects() {
    let scheduler = DtaScheduler::new(4, TopologyMode::P2PMesh);
    dtact::future_bridge::__set_current_worker_id_for_test(0);
    unsafe {
        let worker = &*scheduler.workers[0].get();
        worker.load_level.store(255, Ordering::SeqCst);
        worker.deflection_threshold.store(0, Ordering::SeqCst);
    }

    assert!(scheduler.enqueue_deflect(
        0,
        0,
        0,
        Affinity::Any,
        TopologyMode::Pinned,
        Priority::Normal,
    ));
    assert!(
        mailbox_target(&scheduler, 0).is_none(),
        "a Pinned fiber must never be handed to a peer"
    );
    assert_eq!(
        unsafe { (*scheduler.workers[0].get()).local_queue_len() },
        1,
        "a Pinned fiber stays on its own worker's queue"
    );

    dtact::future_bridge::__set_current_worker_id_for_test(usize::MAX);
}

#[test]
#[cfg_attr(miri, ignore)]
fn chunk_priority_survives_a_mailbox_hop() {
    let scheduler = DtaScheduler::new(4, TopologyMode::P2PMesh);
    dtact::future_bridge::__set_current_worker_id_for_test(0);

    // Pinned cross-worker push: worker 0 -> worker 1's mailbox.
    assert!(scheduler.enqueue_pinned(1, 7, Priority::Critical));
    let chunk = scheduler.mailboxes[0][1]
        .pop()
        .expect("the pinned push must have landed in the target mailbox");
    assert_eq!(
        chunk.priority(),
        Priority::Critical,
        "the receiving worker must be able to see the priority"
    );
    assert_eq!(chunk.tasks[0], 7);
    assert_eq!(chunk.count, 1);

    // Default-constructed chunks (the only other source) read as Normal.
    assert_eq!(
        dtact::dta_scheduler::TaskChunk::default().priority(),
        Priority::Normal
    );

    dtact::future_bridge::__set_current_worker_id_for_test(usize::MAX);
}

/// Fills `worker`'s local queue to the high-water mark, so the next routed
/// chunk finds no room and has to take the deflect-or-park branch.
fn fill_local_queue(scheduler: &DtaScheduler, worker: usize) {
    let w = unsafe { &*scheduler.workers[worker].get() };
    while w.local_queue_len() < LOCAL_QUEUE_HIGH_WATERMARK {
        assert!(w.push_local(0), "filling the local queue must succeed");
    }
    assert_eq!(w.local_queue_len(), LOCAL_QUEUE_HIGH_WATERMARK);
}

#[test]
#[cfg_attr(miri, ignore)]
fn critical_chunk_parks_instead_of_hopping() {
    let scheduler = DtaScheduler::new(4, TopologyMode::P2PMesh);
    fill_local_queue(&scheduler, 1);

    // Worker 0 pushes a Critical fiber at worker 1, whose queue is full.
    dtact::future_bridge::__set_current_worker_id_for_test(0);
    assert!(scheduler.enqueue_pinned(1, 3, Priority::Critical));
    assert!(
        scheduler.mailboxes[0][1].pop().is_some(),
        "sanity: it arrived"
    );

    // Re-inject and let worker 1 route it with a full queue.
    assert!(scheduler.enqueue_pinned(1, 4, Priority::Critical));
    dtact::future_bridge::__set_current_worker_id_for_test(1);
    scheduler.poll_mailboxes(1);

    assert!(
        scheduler.warehouse.is_busy(),
        "a Critical chunk with no local room must be parked, not deflected onward"
    );
    assert!(
        mailbox_target(&scheduler, 1).is_none(),
        "a Critical chunk must not be handed to a third worker"
    );

    dtact::future_bridge::__set_current_worker_id_for_test(usize::MAX);
}

#[test]
#[cfg_attr(miri, ignore)]
fn normal_chunk_hops_instead_of_parking() {
    let scheduler = DtaScheduler::new(4, TopologyMode::P2PMesh);
    fill_local_queue(&scheduler, 2);

    dtact::future_bridge::__set_current_worker_id_for_test(0);
    assert!(scheduler.enqueue_pinned(2, 5, Priority::Normal));
    assert!(
        scheduler.mailboxes[0][2].pop().is_some(),
        "sanity: it arrived"
    );

    assert!(scheduler.enqueue_pinned(2, 6, Priority::Normal));
    dtact::future_bridge::__set_current_worker_id_for_test(2);
    scheduler.poll_mailboxes(2);

    // `(2 + 1 + 7*1) % 4 == 2` lands on the producer itself, which the
    // router bumps past -> worker 3.
    assert_eq!(
        mailbox_target(&scheduler, 2),
        Some(3),
        "a Normal chunk with no local room must be deflected to a peer"
    );
    assert!(
        !scheduler.warehouse.is_busy(),
        "a Normal chunk must not fall back to the warehouse while hops remain"
    );

    dtact::future_bridge::__set_current_worker_id_for_test(usize::MAX);
}

#[test]
#[cfg_attr(miri, ignore)]
fn closure_spawn_options_are_applied() {
    common::init_runtime();

    static STARTED: AtomicU32 = AtomicU32::new(0);
    static GATE: AtomicBool = AtomicBool::new(false);
    STARTED.store(0, Ordering::SeqCst);
    GATE.store(false, Ordering::SeqCst);

    let handle = dtact::api::fiber::spawn_closure(
        || {
            STARTED.store(1, Ordering::SeqCst);
            for _ in 0..20_000_000u32 {
                if GATE.load(Ordering::SeqCst) {
                    return;
                }
                std::thread::yield_now();
            }
        },
        ClosureSpawnOptions::new()
            .stack_size(48 * 1024)
            .priority(Priority::High)
            .safety(SafetyLevel::Safety1)
            .name("closure-probe"),
    );

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while STARTED.load(Ordering::SeqCst) == 0 {
        assert!(std::time::Instant::now() < deadline, "fiber never started");
        std::thread::yield_now();
    }

    let info = fiber_info(handle).expect("live fiber");
    assert_eq!(info.name, Some("closure-probe"));
    assert_eq!(info.priority, Priority::High);
    assert_eq!(info.stack_size, 48 * 1024);
    // The closure path is inherently pinned; the priority only decides
    // queue placement, never migration.
    assert_eq!(info.affinity, Affinity::SameCore);

    GATE.store(true, Ordering::SeqCst);
    dtact_await(handle);
}

/// The C ABI of `dtact_spawn_options_t` is `repr(C)` with four `u8`
/// discriminants, a `u32` stack size, then the safety byte — 12 bytes, with
/// `mStackSize` at offset 4 and `mSafety` at offset 8. Locking it down here
/// means a future field reordering cannot silently change what a C caller
/// compiled against the shipped `dtact.h` is talking to.
#[test]
fn c_spawn_options_layout_is_stable() {
    use dtact::c_ffi::dtact_spawn_options_t;
    assert_eq!(core::mem::size_of::<dtact_spawn_options_t>(), 12);
    assert_eq!(core::mem::align_of::<dtact_spawn_options_t>(), 4);
    assert_eq!(core::mem::offset_of!(dtact_spawn_options_t, priority), 0);
    assert_eq!(core::mem::offset_of!(dtact_spawn_options_t, affinity), 1);
    assert_eq!(core::mem::offset_of!(dtact_spawn_options_t, kind), 2);
    assert_eq!(core::mem::offset_of!(dtact_spawn_options_t, switcher), 3);
    assert_eq!(core::mem::offset_of!(dtact_spawn_options_t, stack_size), 4);
    assert_eq!(core::mem::offset_of!(dtact_spawn_options_t, safety), 8);
}

#[test]
#[cfg_attr(miri, ignore)]
fn c_spawn_options_carry_stack_and_safety() {
    use core::ffi::c_void;
    use dtact::c_ffi::{dtact_default_spawn_options, dtact_fiber_launch_ext, dtact_handle_t};

    common::init_runtime();

    static STARTED: AtomicU32 = AtomicU32::new(0);
    static GATE: AtomicBool = AtomicBool::new(false);
    STARTED.store(0, Ordering::SeqCst);
    GATE.store(false, Ordering::SeqCst);

    extern "C" fn gated_fiber(_arg: *mut c_void) {
        STARTED.store(1, Ordering::SeqCst);
        for _ in 0..20_000_000u32 {
            if GATE.load(Ordering::SeqCst) {
                return;
            }
            std::thread::yield_now();
        }
    }

    let defaults = dtact_default_spawn_options();
    assert_eq!(defaults.stack_size, 0, "0 must mean 'runtime default'");
    assert_eq!(defaults.safety, 0xFF, "0xFF must mean 'inherit'");

    let mut opts = defaults;
    opts.stack_size = 96 * 1024;
    opts.safety = 1;
    opts.priority = Priority::High.level();

    let handle: dtact_handle_t =
        unsafe { dtact_fiber_launch_ext(gated_fiber, core::ptr::null_mut(), &opts) };

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while STARTED.load(Ordering::SeqCst) == 0 {
        assert!(std::time::Instant::now() < deadline, "fiber never started");
        std::thread::yield_now();
    }

    let info = fiber_info(handle).expect("live fiber");
    assert_eq!(
        info.stack_size,
        96 * 1024,
        "dtact_spawn_options_t::stack_size must reach the fiber"
    );
    assert_eq!(info.priority, Priority::High);

    GATE.store(true, Ordering::SeqCst);
    dtact_await(handle);
}

// ---------------------------------------------------------------------------
// `#[export_fiber]` attribute arguments
// ---------------------------------------------------------------------------

static EXPORT_STARTED: AtomicU32 = AtomicU32::new(0);
static EXPORT_GATE: AtomicBool = AtomicBool::new(false);

#[dtact::export_fiber(
    stack = "80K",
    priority = "Critical",
    safety = "Safety1",
    name = "exported-worker"
)]
fn exported_worker(id: u32) {
    EXPORT_STARTED.store(id, Ordering::SeqCst);
    for _ in 0..20_000_000u32 {
        if EXPORT_GATE.load(Ordering::SeqCst) {
            return;
        }
        std::thread::yield_now();
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn export_fiber_attribute_arguments_are_applied() {
    common::init_runtime();
    EXPORT_STARTED.store(0, Ordering::SeqCst);
    EXPORT_GATE.store(false, Ordering::SeqCst);

    let handle = dtact_export_fiber_exported_worker(9);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while EXPORT_STARTED.load(Ordering::SeqCst) == 0 {
        assert!(std::time::Instant::now() < deadline, "fiber never started");
        std::thread::yield_now();
    }
    assert_eq!(EXPORT_STARTED.load(Ordering::SeqCst), 9);

    let info = fiber_info(handle).expect("live fiber");
    assert_eq!(info.name, Some("exported-worker"));
    assert_eq!(info.priority, Priority::Critical);
    assert_eq!(
        info.stack_size,
        80 * 1024,
        "`#[export_fiber(stack = ...)]` must reach the fiber"
    );

    EXPORT_GATE.store(true, Ordering::SeqCst);
    dtact_await(handle);
}

// ---------------------------------------------------------------------------
// `kind` seeds the adaptive spin budget
// ---------------------------------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn workload_kind_seeds_the_spin_budget() {
    assert_eq!(WorkloadKind::Compute.spin_budget(), 1000);
    assert_eq!(WorkloadKind::IO.spin_budget(), 100);
    assert_eq!(WorkloadKind::Memory.spin_budget(), 500);
    assert_eq!(WorkloadKind::System.spin_budget(), 200);
    for kind in [
        WorkloadKind::Compute,
        WorkloadKind::IO,
        WorkloadKind::Memory,
        WorkloadKind::System,
    ] {
        assert_eq!(WorkloadKind::from_level(kind.level()), kind);
    }
    assert_eq!(WorkloadKind::from_level(9), WorkloadKind::Compute);
}

#[test]
#[cfg_attr(miri, ignore)]
fn kind_and_priority_are_independent_knobs() {
    common::init_runtime();

    let gate = Arc::new(AtomicBool::new(false));
    let g = gate.clone();
    let handle = spawn_with()
        .kind(WorkloadKind::IO)
        .priority(Priority::Low)
        .spawn(async move {
            for _ in 0..200_000u32 {
                if g.load(Ordering::Acquire) {
                    return;
                }
                dtact::yield_now().await;
            }
        });

    let info = fiber_info(handle).expect("live fiber");
    assert_eq!(info.kind, WorkloadKind::IO);
    assert_eq!(info.priority, Priority::Low);

    gate.store(true, Ordering::Release);
    dtact_await(handle);
}

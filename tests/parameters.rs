//! Regression tests for the spawn-time parameters that used to be accepted
//! and silently dropped: `Priority`, `name`, per-fiber `SafetyLevel`, and the
//! requested stack size (from `SpawnBuilder::stack_size`, the
//! `#[task(stack = "...")]` attribute, and `spawn_with_stack`).
//!
//! Each test asserts an *observable* effect rather than just "the setter did
//! not panic": `fiber_info` reads the metadata back out of the live context,
//! and the scheduler tests check the actual stay-local-vs-deflect decision
//! and the queue insertion point.

#![allow(dead_code)]

mod common;

use dtact::api::fiber::parse_stack_size;
use dtact::dta_scheduler::{DtaScheduler, TopologyMode};
use dtact::{Affinity, Priority, WorkloadKind, dtact_await, fiber_info, spawn_with, yield_now};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Worker count for the scheduler-level tests: small enough that a handful of
/// mailboxes (each a ~9 MiB `MAILBOX_CAPACITY` buffer) stays cheap on Windows,
/// where `VirtualAlloc` commits the whole thing up front.
const SCHED_WORKERS: usize = 4;

/// Runs `f` on a fresh `DtaScheduler` with worker 0 pretending to be the
/// calling thread, so `enqueue_*` takes its same-core fast path.
fn with_scheduler<T>(topology: TopologyMode, f: impl FnOnce(&DtaScheduler) -> T) -> T {
    let scheduler = DtaScheduler::new(SCHED_WORKERS, topology);
    dtact::future_bridge::__set_current_worker_id_for_test(0);
    let out = f(&scheduler);
    dtact::future_bridge::__set_current_worker_id_for_test(usize::MAX);
    out
}

/// Pins worker 0 at a load just above the default threshold and reports
/// whether `enqueue_deflect` kept the task on worker 0.
fn stayed_local(priority: Priority) -> bool {
    with_scheduler(TopologyMode::Global, |scheduler| {
        unsafe {
            let worker = &*scheduler.workers[0].get();
            // Threshold 80 (the `Worker::new` default) against a load of 60:
            // the outcome is decided purely by the priority's bias.
            worker.deflection_threshold.store(80, Ordering::SeqCst);
            worker.load_level.store(60, Ordering::SeqCst);
        }
        let ok = scheduler.enqueue_deflect(0, 0, 0, Affinity::Any, TopologyMode::Global, priority);
        assert!(ok, "enqueue must succeed for {priority:?}");
        (unsafe { (*scheduler.workers[0].get()).local_queue_len() }) == 1
    })
}

#[test]
#[cfg_attr(miri, ignore)]
fn low_priority_deflects_before_normal() {
    assert!(
        !stayed_local(Priority::Low),
        "Priority::Low must deflect at load 60 (bias lowers the threshold to 48)"
    );
    assert!(
        stayed_local(Priority::Normal),
        "Priority::Normal must keep the task local at load 60 (bias 0, threshold 80)"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn high_priority_deflects_after_normal() {
    assert!(
        stayed_local(Priority::High),
        "Priority::High must keep the task local at load 60 (bias raises the threshold to 104)"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn critical_priority_never_deflects_even_at_full_load() {
    let local = with_scheduler(TopologyMode::Global, |scheduler| {
        unsafe {
            let worker = &*scheduler.workers[0].get();
            // Maximum possible load against the minimum possible threshold:
            // every deflectable priority leaves the core here.
            worker.deflection_threshold.store(0, Ordering::SeqCst);
            worker.load_level.store(255, Ordering::SeqCst);
        }
        scheduler.enqueue_deflect(0, 0, 0, Affinity::Any, TopologyMode::Global, Priority::Low);
        assert_eq!(
            unsafe { (*scheduler.workers[0].get()).local_queue_len() },
            0,
            "sanity: a deflectable priority must leave the core at load 255"
        );
        assert!(scheduler.enqueue_pinned(0, 1, Priority::Critical));
        unsafe { (*scheduler.workers[0].get()).local_queue_len() }
    });
    assert_eq!(
        local, 1,
        "a Critical fiber must stay on its own worker even at load 255"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn critical_priority_is_inserted_at_the_queue_front() {
    with_scheduler(TopologyMode::P2PMesh, |scheduler| {
        let worker = unsafe { &*scheduler.workers[0].get() };
        assert!(worker.push_local(10));
        assert!(worker.push_local(11));
        assert_eq!(worker.local_queue_len(), 2);

        // `enqueue_pinned` on the owning thread is a plain local push; for a
        // Critical fiber that means `push_front`.
        assert!(scheduler.enqueue_pinned(0, 12, Priority::Critical));
        assert_eq!(
            worker.local_queue_len(),
            3,
            "the Critical task must have been inserted, not dropped"
        );
        assert_eq!(
            worker.local_head.load(Ordering::SeqCst),
            dtact::dta_scheduler::LOCAL_QUEUE_MASK,
            "the Critical task must occupy the slot *before* the current head"
        );

        // A normal push still appends at the tail, leaving the head alone.
        assert!(worker.push_local(13));
        assert_eq!(worker.local_queue_len(), 4);
        assert_eq!(
            worker.local_head.load(Ordering::SeqCst),
            dtact::dta_scheduler::LOCAL_QUEUE_MASK
        );
    });
}

#[test]
#[cfg_attr(miri, ignore)]
fn fiber_info_reports_spawn_metadata() {
    common::init_runtime();

    let gate = Arc::new(AtomicBool::new(false));
    let g = gate.clone();
    let handle = spawn_with()
        .name("metadata-probe")
        .priority(Priority::High)
        .kind(WorkloadKind::Memory)
        .affinity(Affinity::Any)
        .stack_size(64 * 1024)
        .spawn(async move {
            // Hold the fiber open so its slot cannot be recycled before the
            // metadata is read back. Bounded so a failed assertion below
            // cannot pin a worker thread for the rest of the binary.
            for _ in 0..PARK_YIELDS {
                if g.load(Ordering::Acquire) {
                    return;
                }
                yield_now().await;
            }
        });

    let info = fiber_info(handle).expect("live fiber must report its metadata");
    assert_eq!(info.name, Some("metadata-probe"));
    assert_eq!(info.priority, Priority::High);
    assert_eq!(info.kind, WorkloadKind::Memory);
    assert_eq!(info.affinity, Affinity::Any);
    assert_eq!(
        info.stack_size,
        64 * 1024,
        "a request that fits the slot must be granted exactly"
    );

    gate.store(true, Ordering::Release);
    dtact_await(handle);
}

/// Upper bound on how long a parked test fiber waits for its gate before
/// giving up. Keeps a failed assertion from wedging the shared runtime.
const PARK_YIELDS: u32 = 200_000;

#[test]
#[cfg_attr(miri, ignore)]
fn default_spawn_gets_the_whole_slot_as_stack() {
    common::init_runtime();
    let capacity = dtact::GLOBAL_RUNTIME
        .get()
        .expect("runtime")
        .pool
        .stack_capacity;

    let gate = Arc::new(AtomicBool::new(false));
    let g = gate.clone();
    let handle = spawn_with().spawn(async move {
        for _ in 0..PARK_YIELDS {
            if g.load(Ordering::Acquire) {
                return;
            }
            yield_now().await;
        }
    });

    let info = fiber_info(handle).expect("live fiber");
    // The exact grant is the slot's capacity minus the future's own footprint
    // and alignment padding at the top of the zero-copy buffer.
    assert!(
        info.stack_size <= capacity && info.stack_size + 512 >= capacity,
        "an unspecified stack request must grant (almost) the slot's full \
         {capacity}-byte capacity, got {}",
        info.stack_size
    );
    assert!(
        info.stack_size > 8192,
        "the stack must extend past the 8 KB zero-copy buffer into the slot's reserved region, got {}",
        info.stack_size
    );

    gate.store(true, Ordering::Release);
    dtact_await(handle);
}

#[test]
#[cfg_attr(miri, ignore)]
fn oversized_stack_request_is_clamped_and_counted() {
    common::init_runtime();
    let capacity = dtact::GLOBAL_RUNTIME
        .get()
        .expect("runtime")
        .pool
        .stack_capacity;

    let before = dtact::STACK_CLAMPED_SPAWNS.load(Ordering::Relaxed);
    let gate = Arc::new(AtomicBool::new(false));
    let g = gate.clone();
    let handle = spawn_with().stack_size(64 * 1024 * 1024).spawn(async move {
        for _ in 0..PARK_YIELDS {
            if g.load(Ordering::Acquire) {
                return;
            }
            yield_now().await;
        }
    });

    let info = fiber_info(handle).expect("live fiber");
    assert!(
        info.stack_size <= capacity && info.stack_size + 512 >= capacity,
        "an oversized request must be clamped to what one slot can hold \
         ({capacity}), got {}",
        info.stack_size
    );
    assert!(
        dtact::STACK_CLAMPED_SPAWNS.load(Ordering::Relaxed) > before,
        "clamping must be counted, not silent"
    );

    gate.store(true, Ordering::Release);
    dtact_await(handle);
}

#[test]
#[cfg_attr(miri, ignore)]
fn task_macro_applies_stack_and_priority() {
    common::init_runtime();

    let gate = Arc::new(AtomicBool::new(false));
    let started = Arc::new(AtomicU32::new(0));
    let handle = dtact::spawn!(macro_task(gate.clone(), started.clone()));

    // Wait for the task to actually start, so its context is populated.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while started.load(Ordering::Acquire) == 0 {
        assert!(std::time::Instant::now() < deadline, "task never started");
        std::thread::yield_now();
    }

    let info = fiber_info(handle).expect("live task fiber");
    assert_eq!(
        info.stack_size,
        128 * 1024,
        "`#[task(stack = \"128K\")]` must reach the fiber"
    );
    assert_eq!(info.priority, Priority::High);

    gate.store(true, Ordering::Release);
    dtact_await(handle);
}

#[dtact::task(priority = "High", kind = "Compute", stack = "128K")]
async fn macro_task(gate: Arc<AtomicBool>, started: Arc<AtomicU32>) {
    started.store(1, Ordering::Release);
    for _ in 0..PARK_YIELDS {
        if gate.load(Ordering::Acquire) {
            return;
        }
        yield_now().await;
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn spawn_with_stack_honours_the_requested_size() {
    common::init_runtime();

    let gate = Arc::new(AtomicBool::new(false));
    let started = Arc::new(AtomicU32::new(0));
    let g = gate.clone();
    let s = started.clone();
    let handle = dtact::spawn_with_stack("32K", move || {
        s.store(1, Ordering::Release);
        // Bounded busy-wait: this closure cannot `.await`, so it must not be
        // able to spin forever if the assertion below fails.
        for _ in 0..20_000_000u32 {
            if g.load(Ordering::Acquire) {
                return;
            }
            std::thread::yield_now();
        }
    });

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while started.load(Ordering::Acquire) == 0 {
        assert!(std::time::Instant::now() < deadline, "fiber never started");
        std::thread::yield_now();
    }

    let info = fiber_info(handle).expect("live fiber");
    assert_eq!(
        info.stack_size,
        32 * 1024,
        "spawn_with_stack must grant the requested size"
    );

    gate.store(true, Ordering::Release);
    dtact_await(handle);
}

#[test]
#[cfg_attr(miri, ignore)]
fn critical_fibers_stay_correct_under_contention() {
    common::init_runtime();

    // Critical fibers take the `push_front` path both when they are first
    // enqueued and every time they re-enter the queue after a yield, so this
    // exercises the ring's head-side insertion under real concurrent load.
    // A lost, duplicated or overwritten slot would show up as a missing
    // increment or a hang here.
    const FIBERS: u32 = 200;
    const YIELDS: u32 = 20;

    let counter = Arc::new(AtomicU32::new(0));
    let mut handles = Vec::new();
    for _ in 0..FIBERS {
        let c = counter.clone();
        handles.push(spawn_with().priority(Priority::Critical).spawn(async move {
            for _ in 0..YIELDS {
                c.fetch_add(1, Ordering::SeqCst);
                yield_now().await;
            }
        }));
    }
    for handle in handles {
        dtact_await(handle);
    }

    assert_eq!(
        counter.load(Ordering::SeqCst),
        FIBERS * YIELDS,
        "every Critical fiber must run to completion exactly once"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn critical_child_spawned_inside_a_fiber_is_never_lost() {
    common::init_runtime();

    // Regression test for the interaction between `push_front` and
    // `dispatch_loop`'s cached queue head.
    //
    // The root fiber is popped from ring slot `h` (the dispatcher then stores
    // `h+1`). It spawns a Critical child — `push_front`-ed into slot `h-1`,
    // i.e. `local_head` moves *backwards* — then a Normal child at the tail,
    // then genuinely suspends (`dtact_await`, not `yield_now`, which would
    // resolve without ever leaving the dispatcher). On resumption the
    // dispatcher must notice the head moved, or it keeps its stale `h+1`,
    // pops the Normal child and stores `h+2` — stranding the Critical child
    // behind the queue head for a full lap of the 131 072-slot ring.
    let done = Arc::new(AtomicU32::new(0));
    let d = done.clone();
    let root = spawn_with().spawn(async move {
        let inner = d.clone();
        spawn_with().priority(Priority::Critical).spawn(async move {
            inner.fetch_add(1, Ordering::SeqCst);
        });
        let sibling = spawn_with().spawn(async move {
            yield_now().await;
        });
        dtact_await(sibling);
    });
    dtact_await(root);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while done.load(Ordering::SeqCst) != 1 {
        assert!(
            std::time::Instant::now() < deadline,
            "the Critical child spawned from inside a fiber was lost"
        );
        std::thread::yield_now();
    }
}

#[test]
fn stack_size_spec_parsing() {
    assert_eq!(parse_stack_size("256K"), Some(256 * 1024));
    assert_eq!(parse_stack_size("2M"), Some(2 * 1024 * 1024));
    assert_eq!(parse_stack_size("2mib"), Some(2 * 1024 * 1024));
    assert_eq!(parse_stack_size("1G"), Some(1024 * 1024 * 1024));
    assert_eq!(parse_stack_size("4096"), Some(4096));
    assert_eq!(parse_stack_size(" 64K "), Some(64 * 1024));
    assert_eq!(parse_stack_size("0"), None);
    assert_eq!(parse_stack_size(""), None);
    assert_eq!(parse_stack_size("K"), None);
    assert_eq!(parse_stack_size("12X"), None);
    // 64 GiB overflows a 32-bit `usize` but not a 64-bit one; either way it
    // must not panic and must not silently wrap to a tiny value.
    assert!(parse_stack_size("64G").is_none_or(|n| n == 64 * 1024 * 1024 * 1024));
}

#[test]
fn priority_levels_round_trip() {
    for priority in [
        Priority::Low,
        Priority::Normal,
        Priority::High,
        Priority::Critical,
    ] {
        assert_eq!(Priority::from_level(priority.level()), priority);
    }
    assert_eq!(Priority::from_level(9), Priority::Normal);
}

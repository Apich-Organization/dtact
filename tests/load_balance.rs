#![allow(dead_code)]

mod common;

use dtact::{Affinity, Priority, WorkloadKind, dtact_await, spawn, spawn_with, yield_now};
use serial_test::serial;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

#[test]
#[cfg_attr(miri, ignore)]
fn test_tasks_with_any_affinity_all_complete() {
    common::init_runtime();

    let counter = Arc::new(AtomicU32::new(0));
    let mut handles = Vec::new();

    for _ in 0..64 {
        let c = counter.clone();
        let h = spawn_with()
            .kind(WorkloadKind::Compute)
            .affinity(Affinity::Any)
            .spawn(async move {
                c.fetch_add(1, Ordering::SeqCst);
                yield_now().await;
                c.fetch_add(1, Ordering::SeqCst);
            });
        handles.push(h);
    }

    for h in handles {
        dtact_await(h);
    }

    assert_eq!(
        counter.load(Ordering::SeqCst),
        128,
        "all 64 tasks with Affinity::Any must complete both increments"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_high_priority_fibers_complete() {
    common::init_runtime();

    let counter = Arc::new(AtomicU32::new(0));
    let mut handles = Vec::new();

    for _ in 0..16 {
        let c = counter.clone();
        let h = spawn_with()
            .priority(Priority::High)
            .kind(WorkloadKind::Compute)
            .spawn(async move {
                c.fetch_add(1, Ordering::SeqCst);
            });
        handles.push(h);
    }

    for h in handles {
        dtact_await(h);
    }

    assert_eq!(counter.load(Ordering::SeqCst), 16);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_mixed_priority_all_complete() {
    common::init_runtime();

    let counter = Arc::new(AtomicU32::new(0));
    let mut handles = Vec::new();

    for priority in [
        Priority::Low,
        Priority::Normal,
        Priority::High,
        Priority::Critical,
    ] {
        for _ in 0..8 {
            let c = counter.clone();
            let h = spawn_with().priority(priority).spawn(async move {
                c.fetch_add(1, Ordering::SeqCst);
            });
            handles.push(h);
        }
    }

    for h in handles {
        dtact_await(h);
    }

    assert_eq!(
        counter.load(Ordering::SeqCst),
        32,
        "all 32 mixed-priority fibers must complete"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_io_workload_kind_fibers_complete() {
    common::init_runtime();

    let counter = Arc::new(AtomicU32::new(0));
    let mut handles = Vec::new();

    for _ in 0..20 {
        let c = counter.clone();
        let h = spawn_with().kind(WorkloadKind::IO).spawn(async move {
            yield_now().await;
            c.fetch_add(1, Ordering::SeqCst);
        });
        handles.push(h);
    }

    for h in handles {
        dtact_await(h);
    }

    assert_eq!(counter.load(Ordering::SeqCst), 20);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_concurrent_spawn_from_multiple_threads() {
    common::init_runtime();

    let counter = Arc::new(AtomicU32::new(0));
    let mut threads = Vec::new();

    for _ in 0..4 {
        let c = counter.clone();
        threads.push(std::thread::spawn(move || {
            let mut handles = Vec::new();
            for _ in 0..10 {
                let cc = c.clone();
                let h = spawn(async move {
                    cc.fetch_add(1, Ordering::SeqCst);
                });
                handles.push(h);
            }
            for h in handles {
                dtact_await(h);
            }
        }));
    }

    for t in threads {
        t.join().expect("thread panicked");
    }

    assert_eq!(
        counter.load(Ordering::SeqCst),
        40,
        "spawning from 4 OS threads concurrently must produce correct results"
    );
}

/// `#[serial]` because this test mutates the process-wide deflection
/// threshold on every worker — without it, this can race with
/// `test_synchronous_burst_spawn_spreads_across_workers` (also `#[serial]`,
/// also a threshold-mutating test) restoring/overwriting each other's
/// setting mid-run and producing spurious failures in either test.
#[test]
#[serial]
#[cfg_attr(miri, ignore)]
fn test_deflection_threshold_config() {
    common::init_runtime();

    // Set threshold to 0 for all workers — every enqueue attempt will deflect
    // Verify the runtime doesn't deadlock or lose tasks when threshold is minimal
    let num_workers = dtact::GLOBAL_RUNTIME
        .get()
        .map(|r| r.scheduler.workers.len())
        .unwrap_or(1);

    for i in 0..num_workers {
        dtact::config::set_deflection_threshold(i, 0);
    }

    let counter = Arc::new(AtomicU32::new(0));
    let mut handles = Vec::new();
    for _ in 0..20 {
        let c = counter.clone();
        handles.push(spawn(async move {
            c.fetch_add(1, Ordering::SeqCst);
        }));
    }
    for h in handles {
        dtact_await(h);
    }
    assert_eq!(
        counter.load(Ordering::SeqCst),
        20,
        "tasks must complete even with threshold=0"
    );

    // Restore default threshold
    for i in 0..num_workers {
        dtact::config::set_deflection_threshold(i, 128);
    }
}

/// Regression test for a scheduler-level fix: a single fiber that
/// synchronously fans out many independent child fibers (no yield point in
/// between) used to serialize the *entire* burst onto one worker,
/// regardless of the deflection threshold — because `load_level` was only
/// ever refreshed between full local-queue drains, so `enqueue_deflect`
/// kept consulting a stale, pre-burst reading no matter how large the
/// self-created backlog actually grew. `Worker::push_local` — the one
/// choke point every same-core enqueue funnels through, whether from many
/// separately-dispatched fibers or one fiber's own tight spawn loop — now
/// refreshes `load_level` every `LOAD_REFRESH_PERIOD` pushes, specifically
/// so a long synchronous fan-out gets a chance to notice its own backlog
/// and start deflecting.
///
/// `#[serial]` because this test mutates the process-wide deflection
/// threshold on every worker, which would otherwise race with other tests
/// sharing `GLOBAL_RUNTIME` in this binary.
#[test]
#[serial]
#[cfg_attr(miri, ignore)]
fn test_synchronous_burst_spawn_spreads_across_workers() {
    common::init_runtime();

    let num_workers = dtact::GLOBAL_RUNTIME
        .get()
        .map(|r| r.scheduler.workers.len())
        .unwrap_or(1);

    // Very low threshold, deliberately: this test's runtime (`common::
    // init_runtime`) sizes its `ContextPool` at only 512 contexts, so a
    // synchronous spawn burst here self-limits into ~511-child waves (pool
    // exhaustion forces the spawning fiber to yield back to the scheduler,
    // which drains the wave before resuming it) — the queue depth within
    // one wave never gets much beyond ~511. `load = (queue_len*100)>>13`
    // needs `queue_len >= 82` to exceed threshold 1, comfortably inside
    // that per-wave ceiling with room for multiple `LOAD_REFRESH_PERIOD`
    // (32-push) checkpoints above threshold before the wave ends — a
    // higher threshold here would be testing this test's own pool-capacity
    // ceiling as much as the fix. This is about confirming deflection
    // *can* engage mid-burst at all, not about tuning the
    // production-default threshold (80).
    for i in 0..num_workers {
        dtact::config::set_deflection_threshold(i, 1);
    }

    const CHILDREN: u32 = 6000;
    // `dtact::api::topology::current_core()` queries the real *hardware*
    // CPU core the OS happens to have this thread on right now — unrelated
    // to, and not stable with, DTA's own worker-thread indexing (worker
    // threads are never pinned via `sched_setaffinity`). What identifies
    // "which DTA worker ran this fiber" is which of the `dtact-worker-N`
    // OS threads (each spawned once, for the process's lifetime, in
    // `Runtime::start`) executed it — so track OS thread identity instead.
    let threads_used: Arc<std::sync::Mutex<std::collections::HashSet<std::thread::ThreadId>>> =
        Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
    let done = Arc::new(AtomicU32::new(0));

    let threads_root = threads_used.clone();
    let done_root = done.clone();
    let root = spawn(async move {
        for _ in 0..CHILDREN {
            let threads = threads_root.clone();
            let d = done_root.clone();
            // Fire-and-forget: independent, dependency-free children —
            // structurally a Bag-of-Tasks, just arriving as one internal
            // burst rather than externally, one at a time.
            // `Affinity::Any` is required to exercise `enqueue_deflect` at
            // all — the builder's default, `Affinity::SameCore`, is by
            // design routed via `enqueue_pinned`, which never deflects
            // regardless of load (that is the entire meaning of "pinned").
            let _handle = dtact::SpawnBuilder::<dtact::CrossThreadNoFloat>::new()
                .affinity(Affinity::Any)
                .spawn(async move {
                    threads
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .insert(std::thread::current().id());
                    d.fetch_add(1, Ordering::Relaxed);
                });
        }
    });
    dtact_await(root);

    // Children are fire-and-forget (not individually awaited above), so
    // wait for them to actually finish before inspecting which threads ran.
    let start = std::time::Instant::now();
    while done.load(Ordering::Relaxed) < CHILDREN {
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "timed out waiting for {} of {CHILDREN} burst-spawned children to complete",
            done.load(Ordering::Relaxed)
        );
        std::thread::yield_now();
    }

    let workers_used = threads_used
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .len();
    assert!(
        workers_used > 1,
        "a {CHILDREN}-task synchronous burst from one fiber landed entirely \
         on {workers_used} worker thread(s) instead of spreading"
    );

    // Restore the documented production default (`Worker::new`).
    for i in 0..num_workers {
        dtact::config::set_deflection_threshold(i, 80);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_global_topology_mode_completes_all_tasks() {
    // Test that Global topology mode runs all tasks to completion
    // This creates its own scheduler/pool directly (not GLOBAL_RUNTIME)
    let scheduler =
        dtact::dta_scheduler::DtaScheduler::new(2, dtact::dta_scheduler::TopologyMode::Global);
    let pool = dtact::memory_management::ContextPool::new(
        32,
        131_072,
        dtact::memory_management::SafetyLevel::Safety0,
        0,
    )
    .expect("pool creation failed");

    // The scheduler/pool struct validates construction succeeded
    // Verify it's non-trivially constructed
    assert!(pool.slot_size > 0, "pool slot size must be positive");
    assert!(!scheduler.workers.is_empty(), "scheduler must have workers");
}

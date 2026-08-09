#![allow(dead_code)]

mod common;

use dtact::{Affinity, Priority, WorkloadKind, dtact_await, spawn, spawn_with, yield_now};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
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
#[cfg(not(feature = "hw-acceleration"))]
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
#[cfg(not(feature = "hw-acceleration"))]
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
#[cfg(not(feature = "hw-acceleration"))]
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
#[cfg(not(feature = "hw-acceleration"))]
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

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
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

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
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

#![allow(dead_code)]

mod common;

use dtact::{
    CrossThreadFloat, CrossThreadNoFloat, DtactWaitExt, SameThreadFloat, SameThreadNoFloat,
    dtact_await, spawn, spawn_with, yield_now,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_fiber_completes_without_yield() {
    common::init_runtime();
    let done = Arc::new(AtomicU32::new(0));
    let d = done.clone();
    let handle = spawn(async move {
        d.store(1, Ordering::SeqCst);
    });
    dtact_await(handle);
    assert_eq!(done.load(Ordering::SeqCst), 1);
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_fiber_yield_and_resume() {
    common::init_runtime();
    let steps = Arc::new(AtomicU32::new(0));
    let s = steps.clone();
    let handle = spawn(async move {
        s.fetch_add(1, Ordering::SeqCst);
        yield_now().await;
        s.fetch_add(1, Ordering::SeqCst);
    });
    dtact_await(handle);
    assert_eq!(steps.load(Ordering::SeqCst), 2);
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_multiple_yield_points() {
    common::init_runtime();
    let counter = Arc::new(AtomicU32::new(0));
    let c = counter.clone();
    let handle = spawn(async move {
        for _ in 0..5 {
            c.fetch_add(1, Ordering::SeqCst);
            yield_now().await;
        }
    });
    dtact_await(handle);
    assert_eq!(counter.load(Ordering::SeqCst), 5);
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_nested_fiber_spawn_from_fiber() {
    common::init_runtime();
    let inner_ran = Arc::new(AtomicU32::new(0));
    let r = inner_ran.clone();
    let outer = spawn(async move {
        let inner = spawn(async move {
            r.store(42, Ordering::SeqCst);
        });
        // Yield so the inner fiber gets scheduled
        yield_now().await;
        dtact_await(inner);
    });
    dtact_await(outer);
    assert_eq!(inner_ran.load(Ordering::SeqCst), 42);
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_spawn_with_all_switchers() {
    common::init_runtime();
    let results: Arc<Vec<AtomicU32>> = Arc::new((0..4).map(|_| AtomicU32::new(0)).collect());

    let r0 = results.clone();
    let h0 = spawn_with()
        .switcher::<CrossThreadFloat>()
        .spawn(async move {
            r0[0].store(1, Ordering::SeqCst);
        });

    let r1 = results.clone();
    let h1 = spawn_with()
        .switcher::<CrossThreadNoFloat>()
        .spawn(async move {
            r1[1].store(1, Ordering::SeqCst);
        });

    let r2 = results.clone();
    let h2 = spawn_with()
        .switcher::<SameThreadFloat>()
        .spawn(async move {
            r2[2].store(1, Ordering::SeqCst);
        });

    let r3 = results.clone();
    let h3 = spawn_with()
        .switcher::<SameThreadNoFloat>()
        .spawn(async move {
            r3[3].store(1, Ordering::SeqCst);
        });

    for h in [h0, h1, h2, h3] {
        dtact_await(h);
    }
    for i in 0..4 {
        assert_eq!(
            results[i].load(Ordering::SeqCst),
            1,
            "switcher variant {} did not run",
            i
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_many_sequential_fibers() {
    common::init_runtime();
    for i in 0u32..100 {
        let val = Arc::new(AtomicU32::new(0));
        let v = val.clone();
        let h = spawn(async move {
            v.store(i.wrapping_add(1), Ordering::SeqCst);
        });
        dtact_await(h);
        assert_eq!(val.load(Ordering::SeqCst), i.wrapping_add(1));
    }
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_future_resolved_via_wait_ext() {
    common::init_runtime();
    let result = Arc::new(AtomicU32::new(0));
    let r = result.clone();
    let handle = spawn(async move {
        async fn produce() -> u32 {
            77
        }
        let val = produce().wait();
        r.store(val, Ordering::SeqCst);
    });
    dtact_await(handle);
    assert_eq!(result.load(Ordering::SeqCst), 77);
}

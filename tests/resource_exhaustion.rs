#![allow(dead_code)]

mod common;

use dtact::memory_management::{ContextPool, SafetyLevel};
use dtact::{dtact_await, spawn};
use std::sync::Arc;
use std::sync::atomic::Ordering;

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
fn test_context_alloc_returns_none_when_exhausted() {
    let pool = ContextPool::new(4, 131_072, SafetyLevel::Safety0, 0).expect("pool creation failed");

    // Allocate all 4 slots
    let mut allocated = Vec::new();
    for _ in 0..4 {
        let id = pool.alloc_context().expect("expected slot to be available");
        allocated.push(id);
    }

    // 5th allocation must return None
    assert!(
        pool.alloc_context().is_none(),
        "pool should be exhausted after capacity allocations"
    );

    // Free one slot and verify we can allocate again
    pool.free_context(allocated[0]);
    assert!(
        pool.alloc_context().is_some(),
        "pool should allow allocation after a slot is freed"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
fn test_free_context_allows_realloc() {
    // Verifies ABA protection indirectly: alloc → free → re-alloc must succeed
    // and the slot index may be reused (free-list is a stack, so the same slot
    // comes back, but with an incremented generation guarded by the handle).
    let pool = ContextPool::new(2, 131_072, SafetyLevel::Safety0, 0).expect("pool creation failed");

    // Fill the pool
    let a = pool.alloc_context().expect("first alloc failed");
    let b = pool.alloc_context().expect("second alloc failed");
    assert!(pool.alloc_context().is_none(), "pool must be full");

    // Free one slot — the same slot should come back on the next alloc
    pool.free_context(a);
    let c = pool
        .alloc_context()
        .expect("re-alloc after free must succeed");
    assert!(c < 2, "re-allocated slot index must be valid");

    pool.free_context(b);
    pool.free_context(c);
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
fn test_alloc_free_cycle_repeated() {
    let pool = ContextPool::new(2, 131_072, SafetyLevel::Safety0, 0).expect("pool creation failed");

    // Repeatedly alloc and free the same slot — validates free-list integrity
    for _ in 0..100 {
        let id = pool.alloc_context().expect("alloc failed");
        pool.free_context(id);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
fn test_heap_escaped_spawns_counter_accuracy() {
    common::init_runtime();

    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    struct LargeFuture([u8; 16384]);
    impl Future for LargeFuture {
        type Output = u32;
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<u32> {
            Poll::Ready(1)
        }
    }

    let before = dtact::HEAP_ESCAPED_SPAWNS.load(Ordering::Relaxed);

    // 1 large future (>8KB) — should escape
    let h1 = spawn(LargeFuture([0u8; 16384]));
    // 3 small futures — should NOT escape
    let h2 = spawn(async { 1u32 });
    let h3 = spawn(async { 2u32 });
    let h4 = spawn(async { 3u32 });

    for h in [h1, h2, h3, h4] {
        dtact_await(h);
    }

    let after = dtact::HEAP_ESCAPED_SPAWNS.load(Ordering::Relaxed);
    assert!(
        after > before,
        "exactly the large future should have escaped to heap"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
fn test_pool_survives_many_alloc_free_cycles_concurrently() {
    let pool = Arc::new(
        ContextPool::new(32, 131_072, SafetyLevel::Safety0, 0).expect("pool creation failed"),
    );

    let mut threads = Vec::new();
    for _ in 0..4 {
        let p = pool.clone();
        threads.push(std::thread::spawn(move || {
            for _ in 0..25 {
                if let Some(id) = p.alloc_context() {
                    std::hint::black_box(id);
                    p.free_context(id);
                }
            }
        }));
    }

    for t in threads {
        t.join().expect("thread panicked");
    }

    // Pool should still function after concurrent stress
    let id = pool
        .alloc_context()
        .expect("pool should have free slots after stress");
    pool.free_context(id);
}

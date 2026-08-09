#![allow(dead_code)]

mod common;

use dtact::{dtact_await, spawn};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_panic_in_fiber_does_not_crash_runtime() {
    common::init_runtime();

    // Spawn a fiber that panics — fiber_entry_point wraps it in catch_unwind
    let bad = spawn(async {
        panic!("intentional test panic");
    });

    // dtact_await returns normally because fiber_entry_point sets Finished after catching the panic
    dtact_await(bad);

    // Runtime is still alive: a subsequent fiber runs correctly
    let result = Arc::new(AtomicU32::new(0));
    let r = result.clone();
    let good = spawn(async move {
        r.store(1, Ordering::SeqCst);
    });
    dtact_await(good);
    assert_eq!(
        result.load(Ordering::SeqCst),
        1,
        "runtime should remain responsive after a fiber panic"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_panic_fiber_slot_is_recycled() {
    common::init_runtime();

    // Exhaust a few allocations with panicking fibers and verify the pool
    // remains usable: the panicked fiber's slot must be returned to the free list.
    for _ in 0..10 {
        let bad = spawn(async {
            panic!("slot-recycle panic");
        });
        dtact_await(bad);
    }

    // All slots recycled — this fiber must still be allocatable
    let alive = Arc::new(AtomicU32::new(0));
    let a = alive.clone();
    let h = spawn(async move {
        a.store(1, Ordering::SeqCst);
    });
    dtact_await(h);
    assert_eq!(
        alive.load(Ordering::SeqCst),
        1,
        "slot must be recycled after panic"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_multiple_concurrent_panics() {
    common::init_runtime();

    // Spawn 8 panicking fibers simultaneously
    let handles: Vec<_> = (0..8)
        .map(|i| {
            spawn(async move {
                panic!("concurrent panic {}", i);
            })
        })
        .collect();

    for h in handles {
        dtact_await(h);
    }

    // Runtime survives: spawn and run 8 valid fibers
    let counter = Arc::new(AtomicU32::new(0));
    let valid_handles: Vec<_> = (0..8)
        .map(|_| {
            let c = counter.clone();
            spawn(async move {
                c.fetch_add(1, Ordering::SeqCst);
            })
        })
        .collect();

    for h in valid_handles {
        dtact_await(h);
    }
    assert_eq!(counter.load(Ordering::SeqCst), 8);
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_panic_does_not_corrupt_sibling_fibers() {
    common::init_runtime();

    let counter = Arc::new(AtomicU32::new(0));

    let ca = counter.clone();
    let fiber_a = spawn(async move {
        ca.fetch_add(1, Ordering::SeqCst);
    });

    let fiber_b = spawn(async {
        panic!("sibling corruption test");
    });

    let cc = counter.clone();
    let fiber_c = spawn(async move {
        cc.fetch_add(1, Ordering::SeqCst);
    });

    dtact_await(fiber_a);
    dtact_await(fiber_b);
    dtact_await(fiber_c);

    assert_eq!(
        counter.load(Ordering::SeqCst),
        2,
        "fibers A and C must complete despite fiber B panicking"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_panic_with_string_payload() {
    common::init_runtime();

    // Verify String panic payload (non-trivial type) is handled without memory issues
    let after = Arc::new(AtomicU32::new(0));
    let a = after.clone();

    let bad = spawn(async {
        let msg = String::from("heap-allocated panic payload");
        panic!("{}", msg);
    });
    dtact_await(bad);

    let good = spawn(async move {
        a.store(99, Ordering::SeqCst);
    });
    dtact_await(good);
    assert_eq!(after.load(Ordering::SeqCst), 99);
}

#![allow(dead_code)]

mod common;

use dtact::{TaskOutcome, dtact_await, outcome, spawn};
use serial_test::serial;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

// All tests in this file are `#[serial]`: `outcome()` reads a slot's
// terminal state, which `ContextPool::claim_context` only preserves until
// that slot is next claimed by *any* fiber (see `outcome`'s doc comment
// for why it must work this way). With near-instant panicking fibers and
// a small shared pool (`common::init_runtime`), a sibling test spawning
// its own fibers concurrently can — and, empirically, reliably did —
// claim and recycle the exact slot an `outcome()` call in this file was
// about to query, before that call ever ran, well before any race
// internal to a single `outcome()` call could be the cause. `#[serial]`
// (already used elsewhere in this crate's suite for tests that share
// process-wide state) removes that specific source of contention.
#[test]
#[serial]
#[cfg_attr(miri, ignore)]
fn test_panic_in_fiber_does_not_crash_runtime() {
    common::init_runtime();

    // Spawn a fiber that panics — fiber_entry_point wraps it in catch_unwind
    let bad = spawn(async {
        panic!("intentional test panic");
    });

    // `outcome` already blocks until termination (it is a join, not a
    // poll) and captures the status atomically with detecting it — calling
    // `dtact_await` first and `outcome` after would reopen exactly the
    // race this design avoids (see `outcome`'s doc comment), so call only
    // `outcome` here.
    assert_eq!(
        outcome(bad).map(|(o, _)| o),
        Some(TaskOutcome::Panicked),
        "a panicking fiber must be observably distinct from a normal completion"
    );

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
#[serial]
#[cfg_attr(miri, ignore)]
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
#[serial]
#[cfg_attr(miri, ignore)]
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
#[serial]
#[cfg_attr(miri, ignore)]
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

    // Query outcome directly rather than `dtact_await` first — see the
    // comment in `test_panic_in_fiber_does_not_crash_runtime`.
    let outcome_a = outcome(fiber_a).map(|(o, _)| o);
    let outcome_b = outcome(fiber_b).map(|(o, _)| o);
    let outcome_c = outcome(fiber_c).map(|(o, _)| o);

    assert_eq!(
        counter.load(Ordering::SeqCst),
        2,
        "fibers A and C must complete despite fiber B panicking"
    );
    assert_eq!(outcome_a, Some(TaskOutcome::Finished));
    assert_eq!(outcome_b, Some(TaskOutcome::Panicked));
    assert_eq!(outcome_c, Some(TaskOutcome::Finished));
}

#[test]
#[serial]
#[cfg_attr(miri, ignore)]
fn test_panic_with_string_payload() {
    common::init_runtime();

    // Verify String panic payload (non-trivial type) is handled without memory issues
    let after = Arc::new(AtomicU32::new(0));
    let a = after.clone();

    let bad = spawn(async {
        let msg = String::from("heap-allocated panic payload");
        panic!("{}", msg);
    });
    let (result, message) = outcome(bad).expect("panicked fiber must report an outcome");
    assert_eq!(result, TaskOutcome::Panicked);
    assert_eq!(message.as_deref(), Some("heap-allocated panic payload"));

    let good = spawn(async move {
        a.store(99, Ordering::SeqCst);
    });
    dtact_await(good);
    assert_eq!(after.load(Ordering::SeqCst), 99);
}

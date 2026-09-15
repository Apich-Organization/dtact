#![allow(dead_code)]

mod common;

use dtact::{TaskOutcome, cancel, dtact_await, outcome, spawn};
use serial_test::serial;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::task::{Context, Poll};

/// Never resolves on its own and never re-wakes itself, so a fiber
/// awaiting it genuinely parks (`Suspending` -> `Yielded`) instead of
/// looping through `wait_pinned`'s adaptive-spin fast path. This is the
/// deterministic way to test `cancel()` against a truly suspended fiber,
/// as opposed to one still spinning inside a single `wait_pinned` call.
struct Forever;

impl Future for Forever {
    type Output = ();
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        Poll::Pending
    }
}

// Every test below calls `outcome(handle)` alone rather than
// `dtact_await(handle)` followed by a separate `outcome(handle)` call.
// `outcome` already blocks until termination and captures the status
// atomically with detecting it (see its doc comment); calling
// `dtact_await` first would let the slot be reclaimed and reused in the
// gap before the separate `outcome` call starts its own wait from
// scratch — reopening the exact race this design closes. This was found
// the hard way: an earlier version of these tests called both and was
// reproducibly flaky under `--test-threads` > 1.

#[test]
#[serial]
#[cfg_attr(miri, ignore)]
fn test_cancel_parked_fiber_reports_cancelled() {
    common::init_runtime();

    let ran_past_wait = Arc::new(AtomicU32::new(0));
    let r = ran_past_wait.clone();
    let handle = spawn(async move {
        Forever.await;
        // Must never execute: cancellation unwinds out of the `.await`.
        r.store(1, Ordering::SeqCst);
    });

    // Give the fiber a chance to actually reach Forever and park.
    std::thread::sleep(std::time::Duration::from_millis(50));

    cancel(handle);
    assert_eq!(
        outcome(handle).map(|(o, _)| o),
        Some(TaskOutcome::Cancelled),
        "a cancelled fiber must report TaskOutcome::Cancelled, not Finished or Panicked"
    );
    assert_eq!(
        ran_past_wait.load(Ordering::SeqCst),
        0,
        "code after the cancelled await point must never run"
    );
}

#[test]
#[serial]
#[cfg_attr(miri, ignore)]
fn test_cancel_before_first_poll() {
    common::init_runtime();

    let ran = Arc::new(AtomicU32::new(0));
    let r = ran.clone();
    let handle = spawn(async move {
        r.store(1, Ordering::SeqCst);
        Forever.await;
    });

    // Cancel immediately — this may race the fiber's first dispatch, which
    // is fine: either it never runs at all, or it runs up to `Forever` and
    // is cancelled there. Both land in Cancelled with `ran` possibly 0 or 1;
    // what must never happen is Finished or a hang.
    cancel(handle);
    assert_eq!(
        outcome(handle).map(|(o, _)| o),
        Some(TaskOutcome::Cancelled)
    );
}

#[test]
#[serial]
#[cfg_attr(miri, ignore)]
fn test_cancel_on_finished_handle_is_a_safe_noop() {
    common::init_runtime();

    let handle = spawn(async {});
    assert_eq!(outcome(handle).map(|(o, _)| o), Some(TaskOutcome::Finished));

    // The slot may already be recycled for something else; cancelling a
    // stale handle must not corrupt whatever now lives in that slot.
    cancel(handle);

    let after = Arc::new(AtomicU32::new(0));
    let a = after.clone();
    let next = spawn(async move {
        a.store(7, Ordering::SeqCst);
    });
    dtact_await(next);
    assert_eq!(after.load(Ordering::SeqCst), 7);
}

#[test]
#[serial]
#[cfg_attr(miri, ignore)]
fn test_cancel_does_not_affect_sibling_fibers() {
    common::init_runtime();

    let counter = Arc::new(AtomicU32::new(0));

    let ca = counter.clone();
    let fiber_a = spawn(async move {
        ca.fetch_add(1, Ordering::SeqCst);
    });

    let fiber_b = spawn(Forever);

    let cc = counter.clone();
    let fiber_c = spawn(async move {
        cc.fetch_add(1, Ordering::SeqCst);
    });

    std::thread::sleep(std::time::Duration::from_millis(50));
    cancel(fiber_b);

    let outcome_a = outcome(fiber_a).map(|(o, _)| o);
    let outcome_b = outcome(fiber_b).map(|(o, _)| o);
    let outcome_c = outcome(fiber_c).map(|(o, _)| o);

    assert_eq!(counter.load(Ordering::SeqCst), 2);
    assert_eq!(outcome_a, Some(TaskOutcome::Finished));
    assert_eq!(outcome_b, Some(TaskOutcome::Cancelled));
    assert_eq!(outcome_c, Some(TaskOutcome::Finished));
}

#[test]
#[serial]
#[cfg_attr(miri, ignore)]
fn test_cancel_slot_reuse_does_not_leak_cancellation_flag() {
    common::init_runtime();

    // Cancel several fibers so their slots are recycled with
    // `cancel_requested` having been set at least once each.
    for _ in 0..8 {
        let h = spawn(Forever);
        std::thread::sleep(std::time::Duration::from_millis(10));
        cancel(h);
        assert_eq!(outcome(h).map(|(o, _)| o), Some(TaskOutcome::Cancelled));
    }

    // A freshly spawned fiber reusing one of those slots must not inherit
    // a stale cancel_requested flag and unwind on its very first await.
    let completed = Arc::new(AtomicU32::new(0));
    let c = completed.clone();
    let h = spawn(async move {
        dtact::yield_now().await;
        c.store(1, Ordering::SeqCst);
    });
    assert_eq!(outcome(h).map(|(o, _)| o), Some(TaskOutcome::Finished));
    assert_eq!(completed.load(Ordering::SeqCst), 1);
}

/// Races `cancel()` against a target that completes (and has its slot
/// reclaimed and reused) essentially immediately, with no delay between
/// spawn and cancel. Targets the write-side counterpart of the read-side
/// race `dtact_await_observe` had to be fixed for: `cancel` reads
/// `generation`, and *then* stores `cancel_requested = true` — if the
/// slot is reclaimed by an unrelated new fiber in the gap between that
/// read and that store, the store could land on the wrong occupant. If
/// this ever regresses, a later fresh fiber spawned into the same
/// recycled slot would spuriously report `Cancelled` instead of
/// `Finished`.
#[test]
#[serial]
#[cfg_attr(miri, ignore)]
fn test_cancel_racing_natural_completion_never_hits_a_different_fiber() {
    common::init_runtime();

    for _ in 0..500 {
        // Completes essentially instantly — no yield points — so it is
        // likely to already be finished (and its slot potentially
        // reclaimed) by the time `cancel` below runs.
        let h = spawn(async {});
        cancel(h);
        // Whatever `h` itself reports, it must be Finished or Cancelled —
        // never silently missing (None would mean the generation check
        // failed to recognise a still-valid, if racy, target) — and, more
        // importantly, the *next* spawn (below) must never be corrupted.
        let this_outcome = outcome(h).map(|(o, _)| o);
        assert!(
            matches!(
                this_outcome,
                Some(TaskOutcome::Finished) | Some(TaskOutcome::Cancelled)
            ),
            "unexpected outcome for a racily-cancelled instant fiber: {this_outcome:?}"
        );

        // A completely unrelated, freshly spawned fiber must never report
        // Cancelled — that would mean the racing `cancel` call above
        // landed on this slot instead of (or as well as) its real target.
        let next = spawn(async {});
        assert_eq!(
            outcome(next).map(|(o, _)| o),
            Some(TaskOutcome::Finished),
            "a fresh, uncancelled fiber must never inherit a stray cancellation"
        );
    }
}

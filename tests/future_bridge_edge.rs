#![allow(dead_code)]

mod common;

use dtact::{DtactWaitExt, dtact_await, spawn, yield_now};
use serial_test::serial;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::task::{Context, Poll};

// A future that returns Pending on its first poll, then Ready on the second.
struct OnceDelayed {
    polled_once: bool,
    value: u32,
}

impl Future for OnceDelayed {
    type Output = u32;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
        if self.polled_once {
            Poll::Ready(self.value)
        } else {
            self.polled_once = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

#[test]
#[serial]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_wait_panics_outside_fiber_context() {
    // wait() must panic when called from a plain OS thread (no fiber context)
    let result = std::panic::catch_unwind(|| {
        let _ = async { 42u32 }.wait();
    });
    assert!(result.is_err(), "wait() outside a fiber must panic");
}

#[test]
#[serial]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_wait_resolves_immediately_ready_future() {
    common::init_runtime();
    let result = Arc::new(AtomicU32::new(0));
    let r = result.clone();
    let h = spawn(async move {
        let val = async { 99u32 }.wait();
        r.store(val, Ordering::SeqCst);
    });
    dtact_await(h);
    assert_eq!(result.load(Ordering::SeqCst), 99);
}

#[test]
#[serial]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_wait_resolves_pending_then_ready_future() {
    common::init_runtime();
    let result = Arc::new(AtomicU32::new(0));
    let r = result.clone();
    let h = spawn(async move {
        let val = OnceDelayed {
            polled_once: false,
            value: 55,
        }
        .wait();
        r.store(val, Ordering::SeqCst);
    });
    dtact_await(h);
    assert_eq!(result.load(Ordering::SeqCst), 55);
}

#[test]
#[serial]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_wait_chained_futures() {
    common::init_runtime();
    let result = Arc::new(AtomicU32::new(0));
    let r = result.clone();
    let h = spawn(async move {
        let a = async { 10u32 }.wait();
        let b = async { 20u32 }.wait();
        r.store(a + b, Ordering::SeqCst);
    });
    dtact_await(h);
    assert_eq!(result.load(Ordering::SeqCst), 30);
}

#[test]
#[serial]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_large_future_heap_escape_increments_counter() {
    common::init_runtime();

    struct LargeFuture([u8; 16384]);
    impl Future for LargeFuture {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            Poll::Ready(())
        }
    }

    let before = dtact::HEAP_ESCAPED_SPAWNS.load(Ordering::Relaxed);
    let h = spawn(LargeFuture([0u8; 16384]));
    dtact_await(h);
    let after = dtact::HEAP_ESCAPED_SPAWNS.load(Ordering::Relaxed);
    assert!(
        after > before,
        "future exceeding 8KB must escape to heap and increment counter"
    );
}

#[test]
#[serial]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_zero_sized_future() {
    common::init_runtime();
    let done = Arc::new(AtomicU32::new(0));
    let d = done.clone();

    struct Zst;
    impl Future for Zst {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            Poll::Ready(())
        }
    }

    let h = spawn(async move {
        Zst.wait();
        d.store(1, Ordering::SeqCst);
    });
    dtact_await(h);
    assert_eq!(done.load(Ordering::SeqCst), 1);
}

#[test]
#[serial]
#[cfg_attr(miri, ignore)]
#[cfg(not(feature = "hw-acceleration"))]
#[cfg(not(feature = "hw-acceleration"))]
fn test_yield_now_is_rescheduled() {
    common::init_runtime();
    // Verify yield_now().await doesn't deadlock and execution resumes
    let steps = Arc::new(AtomicU32::new(0));
    let s = steps.clone();
    let h = spawn(async move {
        s.fetch_add(1, Ordering::SeqCst);
        yield_now().await;
        yield_now().await;
        yield_now().await;
        s.fetch_add(1, Ordering::SeqCst);
    });
    dtact_await(h);
    assert_eq!(steps.load(Ordering::SeqCst), 2);
}

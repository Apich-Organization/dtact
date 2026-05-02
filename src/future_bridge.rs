#![allow(unsafe_code)]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use core::sync::atomic::Ordering;

use crate::memory_management::{FiberContext, FiberStatus};

/// `VTable` for the Zero-Cost Dtact Waker.
/// 
/// This waker bypasses the standard `Arc` reference counting overhead by
/// pinning wakes directly to the arena-managed `FiberContext`. Since the
/// context is persistent until the fiber terminates, the waker pointer
/// is always valid.
static DTACT_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    clone_waker,
    wake_impl,
    wake_by_ref_impl,
    drop_waker,
);

#[inline(always)]
unsafe fn clone_waker(data: *const ()) -> RawWaker {
    RawWaker::new(data, &DTACT_WAKER_VTABLE)
}

#[inline(always)]
unsafe fn wake_impl(data: *const ()) {
    unsafe { wake_by_ref_impl(data) }
}

#[inline(always)]
unsafe fn wake_by_ref_impl(data: *const ()) {
    let ctx = unsafe { &*data.cast::<FiberContext>() };

    // CAS-free topology resumption:
    // Mark the fiber as ready for execution.
    ctx.state.store(FiberStatus::Running as u8, Ordering::Release);

    // Inject the fiber back into the P2P Mailbox Mesh.
    // The scheduler will handle whether this is a local push or a cross-core signal.
    crate::wake_fiber(ctx.origin_core as usize, ctx.fiber_index);
}

#[inline(always)]
const unsafe fn drop_waker(_data: *const ()) {
    // No-op. The FiberContext is persistently managed by the lock-free ContextPool.
}

/// The Trampoline: Assembly Switch.
/// 
/// Suspends the current fiber natively, saving its state and returning
/// execution to the scheduler's dispatch loop.
#[inline(always)]
unsafe fn dtact_asm_fiber_suspend(ctx: *mut FiberContext) {
    unsafe {
        ((*ctx).switch_fn)(
            &raw mut (*ctx).regs,
            &raw const (*ctx).executor_regs,
        );
    };
}

thread_local! {
    /// Tracks the active executing fiber on the current hardware thread.
    pub(crate) static CURRENT_FIBER: core::cell::Cell<*mut FiberContext> = const { core::cell::Cell::new(core::ptr::null_mut()) };
}

/// The core execution bridge between Rust Futures and Dtact Fibers.
/// 
/// This function executes a future on the fiber's stack. If the future yields
/// (returns `Poll::Pending`), this function enters an adaptive spin loop
/// before natively suspending the fiber if the task remains unresolved.
/// 
/// ## Features
/// - **Zero-Cost Waking**: Uses a direct pointer to `FiberContext` instead of `Arc`.
/// - **Adaptive Spinning**: Dynamically adjusts spinning duration based on
///   historical resolution latency.
/// - **Thread-Migration Guard**: Detects and panics if the OS migrates the 
///   fiber's thread while it's executing a stack-pinned future.
/// 
/// # Panics
/// - Panics if called outside of a DTA-V3 Fiber context.
/// - Panics if illegal OS thread migration is detected, as it would violate stack-pinned invariants.
#[inline(always)]
pub fn wait<F: Future>(mut fut: F) -> F::Output {
    let ctx_ptr = CURRENT_FIBER.with(std::cell::Cell::get);
    assert!(!ctx_ptr.is_null(), "dtact::wait() invoked outside of a DTA-V3 Fiber Execution Context. Thread migration forbidden.");

    let ctx = unsafe { &mut *ctx_ptr };
    
    // Thread Migration Guard
    let tid = crate::utils::get_thread_id();
    if ctx.last_os_thread_id == 0 {
        ctx.last_os_thread_id = tid;
    } else if ctx.last_os_thread_id != tid {
        panic!("DTA-V3 Critical: Illegal OS Thread Migration detected for Fiber {}. Stack-pinned invariants violated.", ctx.fiber_index);
    }
    
    // Construct the Lock-Free, Zero-Cost Waker
    let raw_waker = RawWaker::new(ctx_ptr as *const (), &DTACT_WAKER_VTABLE);
    let waker = unsafe { Waker::from_raw(raw_waker) };
    let mut cx = Context::from_waker(&waker);

    // Pin the future to the local fiber stack footprint safely.
    let mut fut_pinned = unsafe { Pin::new_unchecked(&mut fut) };

    loop {
        match fut_pinned.as_mut().poll(&mut cx) {
            Poll::Ready(output) => {
                // Task resolved! Reward the adaptive spin loop budget.
                ctx.adaptive_spin_count = (ctx.adaptive_spin_count + 1).min(200);
                ctx.spin_failure_count = ctx.spin_failure_count.saturating_sub(1);
                return output;
            },
            Poll::Pending => {
                let current_spin = ctx.adaptive_spin_count;
                let failure_count = ctx.spin_failure_count;

                // Adaptive Cooldown: If we've failed many times, yield immediately.
                if failure_count < 10 {
                    for i in 0..current_spin {
                        core::hint::spin_loop();
                        
                        // Sparse Polling: Reduce L1 pressure by only polling every 8 hints.
                        if i.trailing_zeros() >= 3
                            && let Poll::Ready(output) = fut_pinned.as_mut().poll(&mut cx) {
                                ctx.adaptive_spin_count = (current_spin + 2).min(200);
                                ctx.spin_failure_count = failure_count.saturating_sub(1);
                                return output;
                            }
                    }
                }

                // Spin failed. Penalize budget and yield.
                ctx.spin_failure_count = failure_count.saturating_add(1);
                ctx.adaptive_spin_count = current_spin.saturating_sub(5).max(5);

                ctx.state.store(FiberStatus::Yielded as u8, Ordering::Release);
                
                unsafe { dtact_asm_fiber_suspend(ctx_ptr) };
            }
        }
    }
}

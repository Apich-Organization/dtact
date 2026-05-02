#![allow(unsafe_code)]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use core::sync::atomic::Ordering;

use crate::memory_management::{FiberContext, FiberStatus};

/// VTable for the Zero-Cost Dtact Waker.
/// Bypasses `Arc` counting overhead by pinning wakes directly to the Arena-managed `FiberContext`.
static DTACT_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    clone_waker,
    wake_impl,
    wake_by_ref_impl,
    drop_waker,
);

unsafe fn clone_waker(data: *const ()) -> RawWaker {
    RawWaker::new(data, &DTACT_WAKER_VTABLE)
}

unsafe fn wake_impl(data: *const ()) {
    unsafe { wake_by_ref_impl(data) }
}

unsafe fn wake_by_ref_impl(data: *const ()) {
    let ctx = unsafe { &*(data as *const FiberContext) };

    // CAS-free topology resumption:
    // Mark as ready.
    ctx.state.store(FiberStatus::Running as u8, Ordering::Release);

    // Inject into the P2P Mailbox Mesh via the global scheduler interface.
    // The scheduler will handle whether this is a local push or a cross-core CLDEMOTE push.
    crate::wake_fiber(ctx.origin_core as usize, ctx.fiber_index);
}

unsafe fn drop_waker(_data: *const ()) {
    // No-op. The FiberContext is persistently managed by the lock-free ContextPool.
}

/// The Trampoline: Assembly Switch
/// Suspends the current fiber natively without `mprotect` or Heavy OS intervention.
/// Saves ONLY callee-saved registers and swaps the stack pointer to return to the scheduler.
#[inline(always)]
unsafe fn dtact_asm_fiber_suspend(ctx: *mut FiberContext) {
    unsafe {
        ((*ctx).switch_fn)(
            &mut (*ctx).regs,
            &(*ctx).executor_regs,
        )
    };
}

thread_local! {
    /// Tracks the active executing fiber on the current hardware thread.
    pub(crate) static CURRENT_FIBER: core::cell::Cell<*mut FiberContext> = const { core::cell::Cell::new(core::ptr::null_mut()) };
}

/// Ultra-Fast execution path.
/// Bridges the asynchronous `Future` ecosystem with the DTA-V3 stackful Fiber topology.
/// If the future yields, the fiber natively suspends into the hardware mesh.
#[inline(always)]
pub fn wait<F: Future>(mut fut: F) -> F::Output {
    let ctx_ptr = CURRENT_FIBER.with(|c| c.get());
    if ctx_ptr.is_null() {
        panic!("dtact::wait() invoked outside of a DTA-V3 Fiber Execution Context. Thread migration forbidden.");
    }

    let ctx = unsafe { &mut *ctx_ptr };
    
    // Construct the Lock-Free, Zero-Cost Waker
    let raw_waker = RawWaker::new(ctx_ptr as *const (), &DTACT_WAKER_VTABLE);
    let waker = unsafe { Waker::from_raw(raw_waker) };
    let mut cx = Context::from_waker(&waker);

    // Pin the future to the local fiber stack footprint safely.
    let mut fut_pinned = unsafe { Pin::new_unchecked(&mut fut) };

    loop {
        match fut_pinned.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => {
                // Future is waiting on I/O or an external event.
                // Yield the physical CPU core to the next fiber in the mailbox.
                ctx.state.store(FiberStatus::Yielded as u8, Ordering::Release);
                
                // Jump out to `dispatch_loop_asm`
                unsafe { dtact_asm_fiber_suspend(ctx_ptr) };
                
                // The fiber has been resumed! The Wake impl fired and the Scheduler popped us.
                // Loop around and rapidly poll the future again.
            }
        }
    }
}

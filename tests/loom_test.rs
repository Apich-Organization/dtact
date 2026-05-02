use dtact::FiberStatus;
use loom::sync::Arc;
use loom::sync::atomic::{AtomicU8, Ordering};
use loom::thread;

struct MockFiberContext {
    state: AtomicU8,
}

#[test]
fn test_fiber_state_transitions() {
    loom::model(|| {
        let ctx = Arc::new(MockFiberContext {
            state: AtomicU8::new(FiberStatus::Initial as u8),
        });

        // Simulating dispatch loop picking it up
        ctx.state
            .store(FiberStatus::Running as u8, Ordering::Release);

        let ctx_clone = ctx.clone();

        // Waker thread
        let waker = thread::spawn(move || {
            // Simulated wake_by_ref_impl
            let prev = ctx_clone
                .state
                .swap(FiberStatus::Notified as u8, Ordering::AcqRel);
            if prev == FiberStatus::Yielded as u8 {
                // Enqueue
            }
        });

        // wait_pinned thread
        let waiter = thread::spawn(move || {
            // Simulating a loop iteration
            ctx.state
                .store(FiberStatus::Running as u8, Ordering::Release);

            // Simulating poll returning Pending

            // Try to suspend
            if ctx
                .state
                .compare_exchange(
                    FiberStatus::Running as u8,
                    FiberStatus::Yielded as u8,
                    Ordering::Release,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                // suspended
            }
        });

        waker.join().unwrap();
        waiter.join().unwrap();
    });
}

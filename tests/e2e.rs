use dtact::{
    ContextPool, DtactWaitExt, FiberStatus, Priority, SafetyLevel, WorkloadKind, spawn, spawn_with,
    yield_now,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;

#[dtact::dtact_init(workers = 4, capacity = 2048, safety = "Safety1")]
#[test]
fn test_dtact_comprehensive_e2e() {
    // 1. Test Fiber Execution
    {
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        spawn(async move {
            c.fetch_add(1, Ordering::SeqCst);
            yield_now().await;
            c.fetch_add(1, Ordering::SeqCst);
        });
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    // 2. Test Workload Kind and Priority
    {
        let counter = Arc::new(AtomicU32::new(0));
        let c_high = counter.clone();
        spawn_with()
            .priority(Priority::High)
            .kind(WorkloadKind::Compute)
            .name("high-priority-compute")
            .spawn(async move {
                c_high.fetch_add(10, Ordering::SeqCst);
            });

        let c_low = counter.clone();
        spawn_with()
            .priority(Priority::Low)
            .kind(WorkloadKind::IO)
            .name("low-priority-io")
            .spawn(async move {
                c_low.fetch_add(1, Ordering::SeqCst);
            });

        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(counter.load(Ordering::SeqCst) >= 11);
    }

    // 3. Test Zero-Copy Fallback
    {
        struct LargeFuture([u8; 16384]);
        impl core::future::Future for LargeFuture {
            type Output = ();
            fn poll(
                self: core::pin::Pin<&mut Self>,
                _: &mut core::task::Context<'_>,
            ) -> core::task::Poll<Self::Output> {
                core::task::Poll::Ready(())
            }
        }

        let initial_escaped = dtact::HEAP_ESCAPED_SPAWNS.load(Ordering::Relaxed);
        spawn(LargeFuture([0; 16384]));
        std::thread::sleep(std::time::Duration::from_millis(100));
        let final_escaped = dtact::HEAP_ESCAPED_SPAWNS.load(Ordering::Relaxed);
        assert!(
            final_escaped > initial_escaped,
            "Large future should have escaped to heap"
        );
    }

    // 4. Test DtactWaitExt Bridge
    {
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();
        async fn some_async_val() -> u32 {
            100
        }
        spawn(async move {
            let val = some_async_val().wait();
            c.store(val, Ordering::SeqCst);
        });
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }
}

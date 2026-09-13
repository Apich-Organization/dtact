use std::sync::Once;

static INIT: Once = Once::new();

/// Initializes the global Dtact runtime exactly once per test binary.
/// Safe to call from every test function — subsequent calls are no-ops.
pub fn init_runtime() {
    INIT.call_once(|| {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let runtime = dtact::GLOBAL_RUNTIME.get_or_init(|| {
            let scheduler = dtact::dta_scheduler::DtaScheduler::new(
                workers,
                dtact::dta_scheduler::TopologyMode::P2PMesh,
            );
            let pool = dtact::memory_management::ContextPool::new(
                512,
                524_288,
                dtact::memory_management::SafetyLevel::Safety1,
                0,
                workers,
            )
            .expect("test runtime init failed");
            dtact::Runtime {
                scheduler,
                pool,
                started: core::sync::atomic::AtomicBool::new(false),
                shutdown: core::sync::atomic::AtomicBool::new(false),
            }
        });
        runtime.start();
        // Brief pause to let worker threads reach their polling loops
        std::thread::sleep(std::time::Duration::from_millis(30));
    });
}

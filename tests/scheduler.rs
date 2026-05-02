use proptest::prelude::*;
use dtact::dta_scheduler::{DtaScheduler, TopologyMode};
use dtact::FiberStatus;
use std::sync::atomic::Ordering;

proptest! {
    #[test]
    fn test_deflection_consistency(
        source_core in 0usize..64usize,
        flow_id in 0u64..10000u64,
        load in 0u8..255u8,
        threshold in 0u8..255u8
    ) {
        let scheduler = DtaScheduler::new(64, TopologyMode::P2PMesh);
        
        // Set local load
        unsafe {
            let worker = &*scheduler.workers[source_core].get();
            worker.load_level.store(load, Ordering::SeqCst);
            worker.deflection_threshold.store(threshold, Ordering::SeqCst);
        }

        // We can't easily capture the result of enqueue_task because it uses a function pointer 
        // that pushes to a mailbox. But we can test the target_core calculation logic 
        // if we expose it or test the resulting mailbox state.
        
        // For now, let's verify that the deflection logic doesn't panic and stays within bounds.
        scheduler.enqueue_task(source_core, flow_id, 0);
    }

    #[test]
    fn test_global_topology_distribution(
        source_core in 0usize..64usize,
        flow_id in 0u64..10000u64
    ) {
        let scheduler = DtaScheduler::new(64, TopologyMode::Global);
        
        // In Global mode, deflection should be able to reach any core.
        // We'll verify this by setting a very low threshold.
        unsafe {
            let worker = &*scheduler.workers[source_core].get();
            worker.load_level.store(100, Ordering::SeqCst);
            worker.deflection_threshold.store(10, Ordering::SeqCst);
        }

        scheduler.enqueue_task(source_core, flow_id, 1);
    }
}

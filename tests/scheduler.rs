use dtact::dta_scheduler::{DtaScheduler, TopologyMode};
use proptest::prelude::*;
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
            let worker = &mut *scheduler.workers[source_core].get();
            worker.load_level.store(load, Ordering::SeqCst);
            worker.deflection_threshold.store(threshold, Ordering::SeqCst);
        }

        scheduler.enqueue_task(source_core, flow_id, 0);

        // Verify task is successfully enqueued somewhere
        let mut total_tasks = 0;
        unsafe {
            for i in 0..64 {
                let worker = &*scheduler.workers[i].get();
                total_tasks += worker.local_queue_len();

                for j in 0..64 {
                    let mailbox = &scheduler.mailboxes[i][j];
                    let head = mailbox.head.load(Ordering::SeqCst);
                    let tail = mailbox.tail.load(Ordering::SeqCst);
                    // DTACT_MAILBOX_MASK is not public, but we can just check if tail != head
                    if tail != head {
                        total_tasks += 1; // 1 TaskChunk
                    }
                }
            }
        }
        assert_eq!(total_tasks, 1, "Task must be enqueued exactly once");
    }

    #[test]
    fn test_global_topology_distribution(
        source_core in 0usize..64usize,
        flow_id in 0u64..10000u64
    ) {
        let scheduler = DtaScheduler::new(64, TopologyMode::Global);

        unsafe {
            let worker = &mut *scheduler.workers[source_core].get();
            worker.load_level.store(100, Ordering::SeqCst);
            worker.deflection_threshold.store(10, Ordering::SeqCst);
        }

        scheduler.enqueue_task(source_core, flow_id, 1);

        let mut total_tasks = 0;
        unsafe {
            for i in 0..64 {
                let worker = &*scheduler.workers[i].get();
                total_tasks += worker.local_queue_len();

                for j in 0..64 {
                    let mailbox = &scheduler.mailboxes[i][j];
                    if mailbox.tail.load(Ordering::SeqCst) != mailbox.head.load(Ordering::SeqCst) {
                        total_tasks += 1;
                    }
                }
            }
        }
        assert_eq!(total_tasks, 1, "Task must be enqueued exactly once in Global mode");
    }
}

use dtact::Affinity::Any;
use dtact::dta_scheduler::{DtaScheduler, TopologyMode};
use proptest::prelude::*;
use std::sync::atomic::Ordering;

proptest! {
    #[cfg_attr(miri, ignore)]
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

        let _ = scheduler.enqueue_deflect(source_core, flow_id, 0, Any);

        // Verify task is successfully enqueued somewhere
        let mut total_tasks = 0;
        unsafe {
            for i in 0..64 {
                let worker = &*scheduler.workers[i].get();
                total_tasks += worker.local_queue_len();

                for mailbox in scheduler.mailboxes[i].iter().take(64) {
                    let head = mailbox.head.load(Ordering::SeqCst);
                    let tail = mailbox.tail.load(Ordering::SeqCst);
                    if tail != head {
                        total_tasks += 1; // 1 TaskChunk
                    }
                }

                let ext_mailbox = &scheduler.external_mailboxes[i];
                if ext_mailbox.tail.load(Ordering::SeqCst) != ext_mailbox.head.load(Ordering::SeqCst) {
                    total_tasks += 1;
                }
            }
        }
        assert_eq!(total_tasks, 1, "Task must be enqueued exactly once");
    }

    #[cfg_attr(miri, ignore)]
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

        let _ = scheduler.enqueue_deflect(source_core, flow_id, 1, Any);

        let mut total_tasks = 0;
        unsafe {
            for i in 0..64 {
                let worker = &*scheduler.workers[i].get();
                total_tasks += worker.local_queue_len();

                for mailbox in scheduler.mailboxes[i].iter().take(64) {
                    if mailbox.tail.load(Ordering::SeqCst) != mailbox.head.load(Ordering::SeqCst) {
                        total_tasks += 1;
                    }
                }

                let ext_mailbox = &scheduler.external_mailboxes[i];
                if ext_mailbox.tail.load(Ordering::SeqCst) != ext_mailbox.head.load(Ordering::SeqCst) {
                    total_tasks += 1;
                }
            }
        }
        assert_eq!(total_tasks, 1, "Task must be enqueued exactly once in Global mode");
    }
}

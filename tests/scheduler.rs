use dtact::Affinity::Any;
use dtact::dta_scheduler::{DtaScheduler, TopologyMode};
use proptest::prelude::*;
use std::sync::atomic::Ordering;

/// Worker count for these tests.
///
/// `DtaScheduler::new(n)` allocates `n * n + n` mailboxes, each backed by a
/// ~9 MiB buffer (`MAILBOX_CAPACITY` x 144-byte `TaskChunk`). On Unix that
/// buffer is a lazily-faulted `MAP_NORESERVE` mapping, so 64 workers
/// (~37 GiB of address space) is fine. On Windows `VirtualAlloc` uses
/// `MEM_COMMIT`, which charges the full size against the system commit
/// limit up front and fails long before 4,160 mailboxes. 16 workers
/// (272 mailboxes, ~2.4 GiB) still spans two CCXs (`ccx_id = i / 8`).
const WORKERS: usize = if cfg!(windows) { 16 } else { 64 };

proptest! {
    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_deflection_consistency(
        source_core in 0usize..WORKERS,
        flow_id in 0u64..10000u64,
        load in 0u8..255u8,
        threshold in 0u8..255u8
    ) {
        let scheduler = DtaScheduler::new(WORKERS, TopologyMode::P2PMesh);

        // Set local load
        unsafe {
            let worker = &mut *scheduler.workers[source_core].get();
            worker.load_level.store(load, Ordering::SeqCst);
            worker.deflection_threshold.store(threshold, Ordering::SeqCst);
        }

        let _ = scheduler.enqueue_deflect(
            source_core,
            flow_id,
            0,
            Any,
            TopologyMode::P2PMesh,
            dtact::Priority::Normal,
        );

        // Verify task is successfully enqueued somewhere
        let mut total_tasks = 0;
        unsafe {
            for i in 0..WORKERS {
                let worker = &*scheduler.workers[i].get();
                total_tasks += worker.local_queue_len();

                for mailbox in scheduler.mailboxes[i].iter().take(WORKERS) {
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
        source_core in 0usize..WORKERS,
        flow_id in 0u64..10000u64
    ) {
        let scheduler = DtaScheduler::new(WORKERS, TopologyMode::Global);

        unsafe {
            let worker = &mut *scheduler.workers[source_core].get();
            worker.load_level.store(100, Ordering::SeqCst);
            worker.deflection_threshold.store(10, Ordering::SeqCst);
        }

        let _ = scheduler.enqueue_deflect(
            source_core,
            flow_id,
            1,
            Any,
            TopologyMode::Global,
            dtact::Priority::Normal,
        );

        let mut total_tasks = 0;
        unsafe {
            for i in 0..WORKERS {
                let worker = &*scheduler.workers[i].get();
                total_tasks += worker.local_queue_len();

                for mailbox in scheduler.mailboxes[i].iter().take(WORKERS) {
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

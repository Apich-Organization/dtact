#![no_main]

use arbitrary::Arbitrary;
use dtact::dta_scheduler::{DtaScheduler, TopologyMode};
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::Ordering;

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    source_core: u8,
    flow_id: u32,
    priority: u8,
    local_load: u8,
    deflection_threshold: u8,
    is_global_mode: bool,
}

fuzz_target!(|data: FuzzInput| {
    let source_core = (data.source_core as usize) % 64;
    let mode = if data.is_global_mode {
        TopologyMode::Global
    } else {
        TopologyMode::P2PMesh
    };

    let scheduler = DtaScheduler::new(64, mode);

    unsafe {
        let worker = &mut *scheduler.workers[source_core].get();
        worker.load_level.store(data.local_load, Ordering::SeqCst);
        worker
            .deflection_threshold
            .store(data.deflection_threshold, Ordering::SeqCst);
    }

    scheduler.enqueue_task(source_core, data.flow_id as u64, data.priority as u32);

    let mut total_tasks = 0;
    unsafe {
        for i in 0..64 {
            let worker = &*scheduler.workers[i].get();
            total_tasks += worker.local_queue_len();

            for j in 0..64 {
                let mailbox = &scheduler.mailboxes[i][j];
                let head = mailbox.head.load(Ordering::SeqCst);
                let tail = mailbox.tail.load(Ordering::SeqCst);
                if tail != head {
                    total_tasks += 1;
                }
            }
        }
    }

    assert_eq!(total_tasks, 1, "Task must be enqueued exactly once");
});

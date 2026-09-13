use dtact::{ContextPool, SafetyLevel};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::thread;

#[cfg_attr(miri, ignore)]
#[test]
fn test_concurrent_context_allocation() {
    let pool = Arc::new(ContextPool::new(1024, 65536, SafetyLevel::Safety1, 0, 1).unwrap());
    let mut handles = vec![];

    for _ in 0..8 {
        let p = pool.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                if let Some(idx) = p.alloc_context() {
                    let ctx = p.get_context_ptr(idx);
                    unsafe {
                        assert_eq!((*ctx).fiber_index, idx);
                    }
                    p.free_context(idx);
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

/// Same shape as `test_concurrent_context_allocation`, but each thread
/// claims a distinct worker id first via the test-only
/// `__set_current_worker_id_for_test` hook, so this actually exercises
/// `ContextPool`'s per-worker `LocalFreeCache` fast path (batching +
/// refill/donate CAS) instead of the uncached global path that test covers
/// via the default `usize::MAX` sentinel.
#[cfg_attr(miri, ignore)]
#[test]
fn test_concurrent_context_allocation_with_per_worker_batch_cache() {
    const WORKERS: usize = 8;
    const CAPACITY: u32 = 512;
    let pool =
        Arc::new(ContextPool::new(CAPACITY, 65536, SafetyLevel::Safety1, 0, WORKERS).unwrap());
    let outstanding: Arc<Mutex<HashSet<u32>>> = Arc::new(Mutex::new(HashSet::new()));
    let mut handles = vec![];

    for worker_id in 0..WORKERS {
        let p = pool.clone();
        let outstanding = outstanding.clone();
        handles.push(thread::spawn(move || {
            dtact::future_bridge::__set_current_worker_id_for_test(worker_id);
            for _ in 0..2000 {
                if let Some(idx) = p.alloc_context() {
                    assert!(
                        outstanding.lock().unwrap().insert(idx),
                        "index {idx} allocated twice concurrently"
                    );
                    let ctx = p.get_context_ptr(idx);
                    unsafe {
                        assert_eq!((*ctx).fiber_index, idx);
                    }
                    outstanding.lock().unwrap().remove(&idx);
                    p.free_context(idx);
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Every one of the `CAPACITY` slots must still be recoverable: some sit
    // in the shared global list, the rest are resident in one of the
    // `WORKERS` per-worker caches this test used. A live worker's cache is
    // never "lost" capacity in production (workers run for the runtime's
    // whole life), it's just not visible to a caller outside that worker
    // id — so recovery here explicitly walks every id that could be
    // holding a slot, rather than draining from a single neutral thread.
    let mut recovered = HashSet::new();
    for worker_id in (0..WORKERS).chain(std::iter::once(usize::MAX)) {
        dtact::future_bridge::__set_current_worker_id_for_test(worker_id);
        while let Some(idx) = pool.alloc_context() {
            recovered.insert(idx);
        }
    }
    dtact::future_bridge::__set_current_worker_id_for_test(usize::MAX);
    assert_eq!(
        recovered.len(),
        CAPACITY as usize,
        "some capacity was unrecoverable after concurrent batch-cache use"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_guard_page_isolation() {
    // This test verifies that Safety2 (per-context guard pages) correctly
    // catches overflows.
    let pool = ContextPool::new(64, 4096, SafetyLevel::Safety2, 0, 1).unwrap();
    let idx = pool.alloc_context().expect("Should alloc");
    let _ctx_ptr = pool.get_context_ptr(idx);

    // The read buffer is 8KB, and above it is the stack.
    // If we write far below the context pointer (into the guard page), it should fault.
    // We can't easily catch a segfault in a test, but we can verify the memory layout.
    let (_base, slot_sz, guard_sz, _context_offset) = pool.get_dispatch_layout();

    #[cfg(unix)]
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    #[cfg(windows)]
    let page_size = unsafe {
        let mut info = std::mem::zeroed();
        windows_sys::Win32::System::SystemInformation::GetSystemInfo(&raw mut info);
        info.dwPageSize as usize
    };

    assert_eq!(guard_sz, page_size);
    let requested_stack_sz = 4096;
    assert!(slot_sz > requested_stack_sz + 8192);
}

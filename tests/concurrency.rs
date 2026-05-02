use dtact::{ContextPool, SafetyLevel, FiberStatus};
use std::sync::Arc;
use std::thread;
use std::sync::atomic::Ordering;

#[test]
fn test_concurrent_context_allocation() {
    let pool = Arc::new(ContextPool::new(1024, 65536, SafetyLevel::Safety1, 0).unwrap());
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

#[test]
fn test_guard_page_isolation() {
    // This test verifies that Safety2 (per-context guard pages) correctly 
    // catches overflows.
    let pool = ContextPool::new(64, 4096, SafetyLevel::Safety2, 0).unwrap();
    let idx = pool.alloc_context().expect("Should alloc");
    let ctx_ptr = pool.get_context_ptr(idx);
    
    // The read buffer is 8KB, and above it is the stack.
    // If we write far below the context pointer (into the guard page), it should fault.
    // We can't easily catch a segfault in a test, but we can verify the memory layout.
    let (base, slot_sz, guard_sz) = pool.get_dispatch_layout();
    assert_eq!(guard_sz, 4096);
    assert!(slot_sz > 4096 + 8192);
}

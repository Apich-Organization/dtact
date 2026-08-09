#![allow(dead_code)]

mod common;

use core::ffi::c_void;
use dtact::c_ffi::{
    dtact_await, dtact_default_config, dtact_default_spawn_options, dtact_fiber_launch,
    dtact_fiber_launch_ext, dtact_fiber_launch_with_cleanup,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

// ---- C-style fiber functions ------------------------------------------------

extern "C" fn noop_fiber(_arg: *mut c_void) {}

extern "C" fn increment_fiber(arg: *mut c_void) {
    let counter = unsafe { Arc::from_raw(arg as *const AtomicU32) };
    counter.fetch_add(1, Ordering::SeqCst);
    // Leak the Arc so the caller's clone remains valid
    let _ = Arc::into_raw(counter);
}

extern "C" fn long_running_fiber(arg: *mut c_void) {
    // Loops to simulate a fiber that keeps the worker busy
    let counter = unsafe { Arc::from_raw(arg as *const AtomicU32) };
    for _ in 0..1000 {
        counter.fetch_add(1, Ordering::Relaxed);
    }
    let _ = Arc::into_raw(counter);
}

unsafe extern "C" fn cleanup_callback(arg: *mut c_void) {
    let flag = unsafe { Arc::from_raw(arg as *const AtomicU32) };
    flag.fetch_add(1, Ordering::SeqCst);
    let _ = Arc::into_raw(flag);
}

// ---- Default config/options tests ------------------------------------------

#[test]
fn test_dtact_default_config_values() {
    let cfg = dtact_default_config();
    assert_eq!(cfg.workers, 0, "default workers should be 0 (auto-detect)");
    assert_eq!(
        cfg.safety_level, 1,
        "default safety level should be 1 (Safety1)"
    );
    assert_eq!(
        cfg.topology_mode, 0,
        "default topology should be 0 (P2PMesh)"
    );
    assert_eq!(
        cfg.fiber_capacity, 0,
        "default capacity should be 0 (use runtime default)"
    );
    assert_eq!(
        cfg.stack_size, 0,
        "default stack size should be 0 (use runtime default)"
    );
}

#[test]
fn test_dtact_default_spawn_options_values() {
    let opts = dtact_default_spawn_options();
    assert_eq!(opts.priority, 1, "default priority should be Normal (1)");
    assert_eq!(opts.affinity, 0, "default affinity should be SameCore (0)");
    assert_eq!(opts.kind, 0, "default kind should be Compute (0)");
    assert_eq!(
        opts.switcher, 0,
        "default switcher should be CrossThreadFloat (0)"
    );
}

// ---- Fiber launch tests -----------------------------------------------------

#[test]
#[cfg_attr(miri, ignore)]
fn test_c_ffi_fiber_launch_basic() {
    common::init_runtime();

    let counter = Arc::new(AtomicU32::new(0));
    let raw = Arc::into_raw(counter.clone()) as *mut c_void;

    let handle = unsafe { dtact_fiber_launch(increment_fiber, raw) };
    dtact_await(handle);

    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_c_ffi_fiber_launch_noop() {
    common::init_runtime();
    let handle = unsafe { dtact_fiber_launch(noop_fiber, core::ptr::null_mut()) };
    dtact_await(handle);
    // No assertion needed — just must not deadlock or crash
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_c_ffi_fiber_launch_ext_all_priorities() {
    common::init_runtime();

    let counter = Arc::new(AtomicU32::new(0));

    for priority in 0u8..=3 {
        let raw = Arc::into_raw(counter.clone()) as *mut c_void;
        let mut opts = dtact_default_spawn_options();
        opts.priority = priority;

        let handle = unsafe { dtact_fiber_launch_ext(increment_fiber, raw, &opts) };
        dtact_await(handle);
    }

    assert_eq!(
        counter.load(Ordering::SeqCst),
        4,
        "all 4 priority-level fibers must complete"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_c_ffi_fiber_launch_ext_all_kinds() {
    common::init_runtime();
    let counter = Arc::new(AtomicU32::new(0));

    for kind in 0u8..=3 {
        let raw = Arc::into_raw(counter.clone()) as *mut c_void;
        let mut opts = dtact_default_spawn_options();
        opts.kind = kind;

        let handle = unsafe { dtact_fiber_launch_ext(increment_fiber, raw, &opts) };
        dtact_await(handle);
    }

    assert_eq!(counter.load(Ordering::SeqCst), 4);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_c_ffi_fiber_launch_with_cleanup_called() {
    common::init_runtime();

    // fiber_ran tracks execution; cleanup_ran tracks cleanup invocation
    let fiber_ran = Arc::new(AtomicU32::new(0));
    let cleanup_ran = Arc::new(AtomicU32::new(0));

    // The fiber arg is the fiber_ran counter
    let fiber_arg = Arc::into_raw(fiber_ran.clone()) as *mut c_void;
    // The cleanup callback receives the same arg pointer
    let cleanup_arg = Arc::into_raw(cleanup_ran.clone()) as *mut c_void;

    // We encode both pointers by wrapping in a small struct
    struct TwoFlags {
        fiber: *mut c_void,
        cleanup: *mut c_void,
    }
    unsafe impl Send for TwoFlags {}

    extern "C" fn two_flag_fiber(arg: *mut c_void) {
        let flags = unsafe { &*(arg as *const TwoFlags) };
        let f = unsafe { Arc::from_raw(flags.fiber as *const AtomicU32) };
        f.fetch_add(1, Ordering::SeqCst);
        let _ = Arc::into_raw(f);
    }

    unsafe extern "C" fn two_flag_cleanup(arg: *mut c_void) {
        let flags = unsafe { Box::from_raw(arg as *mut TwoFlags) };
        let c = unsafe { Arc::from_raw(flags.cleanup as *const AtomicU32) };
        c.fetch_add(1, Ordering::SeqCst);
        let _ = Arc::into_raw(c);
    }

    let flags = Box::new(TwoFlags {
        fiber: fiber_arg,
        cleanup: cleanup_arg,
    });
    let flags_ptr = Box::into_raw(flags) as *mut c_void;

    let handle =
        unsafe { dtact_fiber_launch_with_cleanup(two_flag_fiber, flags_ptr, two_flag_cleanup) };
    dtact_await(handle);

    assert_eq!(
        fiber_ran.load(Ordering::SeqCst),
        1,
        "fiber body must have run"
    );
    assert_eq!(
        cleanup_ran.load(Ordering::SeqCst),
        1,
        "cleanup callback must have been called"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_c_ffi_multiple_fibers_complete() {
    common::init_runtime();

    let counter = Arc::new(AtomicU32::new(0));
    let mut handles = Vec::new();

    for _ in 0..8 {
        let raw = Arc::into_raw(counter.clone()) as *mut c_void;
        let h = unsafe { dtact_fiber_launch(long_running_fiber, raw) };
        handles.push(h);
    }

    for h in handles {
        dtact_await(h);
    }

    // Each fiber increments 1000 times; 8 fibers = 8000 total
    assert_eq!(counter.load(Ordering::SeqCst), 8000);
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_c_ffi_handle_sentinel_bit_set() {
    common::init_runtime();
    let handle = unsafe { dtact_fiber_launch(noop_fiber, core::ptr::null_mut()) };
    dtact_await(handle);
    // The valid sentinel bit (bit 63) must be set on a freshly returned handle
    assert_ne!(handle.0 & (1 << 63), 0, "handle sentinel bit must be set");
}

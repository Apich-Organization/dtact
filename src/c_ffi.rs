use core::ffi::c_void;
use crate::memory_management::SafetyLevel;
use crate::dta_scheduler::TopologyMode;

/// Opaque handle representing a spawned Dtact fiber.
#[allow(non_camel_case_types)]
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct dtact_handle_t(pub u64);

/// Configuration structure for initializing the Dtact runtime from C.
#[repr(C)]
pub struct dtact_config_t {
    /// Number of hardware worker threads. Set to 0 for auto-detection.
    pub workers: u32,
    /// Memory safety level (0-2).
    pub safety_level: u8,
    /// Topology mode (0: P2PMesh, 1: Global).
    pub topology_mode: u8,
}

/// Returns the recommended default configuration for the Dtact runtime.
#[unsafe(no_mangle)]
pub extern "C" fn dtact_default_config() -> dtact_config_t {
    dtact_config_t {
        workers: 0, // Auto-detect
        safety_level: 1, // Safety1
        topology_mode: 0, // P2PMesh
    }
}

/// Initializes the global Dtact runtime singleton.
/// 
/// # Safety
/// This function should be called once at application startup.
#[unsafe(no_mangle)]
pub extern "C" fn dtact_init(cfg: *const dtact_config_t) -> *mut c_void {
    let cfg = unsafe { &*cfg };
    let _workers = if cfg.workers == 0 { 
        crate::api::topology::current().core_id as usize + 1 
    } else { 
        cfg.workers as usize 
    };

    let safety = match cfg.safety_level {
        0 => SafetyLevel::Safety0,
        1 => SafetyLevel::Safety1,
        2 => SafetyLevel::Safety2,
        _ => SafetyLevel::Safety1,
    };

    let topology = match cfg.topology_mode {
        0 => TopologyMode::P2PMesh,
        1 => TopologyMode::Global,
        _ => TopologyMode::P2PMesh,
    };

    crate::GLOBAL_RUNTIME.get_or_init(|| {
        let scheduler = crate::dta_scheduler::DtaScheduler::new(128, topology);
        let pool = crate::memory_management::ContextPool::new(
            16384,
            2 * 1024 * 1024,
            safety,
            4,
        ).expect("DTA-V3 FFI Initialization Failed");
        crate::Runtime { scheduler, pool }
    });

    // Return a dummy pointer as "runtime handle" for C
    core::ptr::null_mut()
}

/// Critical failure handler. Aborts the process if a fiber attempts to 
/// return without properly terminating via the runtime.
#[unsafe(no_mangle)]
pub extern "C" fn dtact_abort() -> ! {
    eprintln!("DTA-V3 Critical: Fiber attempted to 'return' instead of yielding. Stack corrupted.");
    std::process::abort();
}

/// Frees an argument pointer previously allocated for a fiber.
#[unsafe(no_mangle)]
pub extern "C" fn dtact_free_arg(arg: *mut c_void) {
    if !arg.is_null() {
        unsafe {
            // Assumes standard C allocator (malloc/free)
            libc::free(arg);
        }
    }
}

/// Launches a C-function as a DTA-V3 stackful Fiber.
/// 
/// # Safety
/// * `func` must be a valid function pointer.
/// * `arg` must point to memory that remains valid for the entire duration of the fiber's execution.
///   Since the fiber is launched asynchronously, the caller's stack may return before the fiber starts.
///   **Critical**: `arg` must be heap-allocated (and freed within the fiber) or static.
#[unsafe(no_mangle)]
pub extern "C" fn dtact_fiber_launch(
    func: extern "C" fn(*mut c_void),
    arg: *mut c_void,
) -> dtact_handle_t {
    let runtime = crate::GLOBAL_RUNTIME.get().expect("Dtact Runtime not initialized");
    let pool = &runtime.pool;
    let ctx_id = pool.alloc_context().expect("Context pool exhausted - OOM");
    
    let ctx_ptr = pool.get_context_ptr(ctx_id);
    let current_core = crate::api::topology::current().core_id as usize;

    unsafe {
        (*ctx_ptr).state.store(crate::memory_management::FiberStatus::Running as u8, core::sync::atomic::Ordering::Release);
        (*ctx_ptr).origin_core = current_core as u16;
        (*ctx_ptr).fiber_index = ctx_id;
        (*ctx_ptr).switch_fn = crate::context_switch::switch_context_cross_thread_float;
        
        (*ctx_ptr).closure_ptr = arg as *mut ();
        (*ctx_ptr).trampoline = core::mem::transmute(func);
        
        // Unified Trampoline for C-Functions
        (*ctx_ptr).invoke_closure = |ptr| {
            let ctx = &mut *(ptr as *mut crate::memory_management::FiberContext);
            let f: extern "C" fn(*mut c_void) = core::mem::transmute(ctx.trampoline);
            f(ctx.closure_ptr as *mut c_void);
        };
        
        // 3. ABI-Compliant Stack Alignment & Poisoning
        // We leave 64 bytes for Shadow Space (Windows) and Future safety.
        let stack_top = (ctx_ptr as usize & !0xF) - 64;
        let stack_top_ptr = stack_top as *mut u64;
        
        // Place a "poison" return address on the stack.
        // If the fiber function ever attempts to 'ret', it will jump here and abort.
        core::ptr::write(stack_top_ptr, dtact_abort as *const () as u64);
        
        let stack_top = stack_top as *mut u8;

        #[cfg(target_arch = "x86_64")]
        {
            (*ctx_ptr).regs.gprs[0] = stack_top as u64; // RSP
            (*ctx_ptr).regs.gprs[7] = crate::api::fiber_entry_point as *const () as u64; // RIP
        }
        #[cfg(target_arch = "aarch64")]
        {
            (*ctx_ptr).regs.gprs[12] = stack_top as u64; // SP
            (*ctx_ptr).regs.gprs[11] = crate::api::fiber_entry_point as u64; // x30 (LR)
        }
        #[cfg(target_arch = "riscv64")]
        {
            (*ctx_ptr).regs.gprs[0] = stack_top as u64; // SP
            (*ctx_ptr).regs.gprs[13] = crate::api::fiber_entry_point as u64; // RA
        }
        (*ctx_ptr).cleanup_fn = None;
    }

    crate::wake_fiber(current_core, ctx_id);
    dtact_handle_t((ctx_id as u64) | ((current_core as u64) << 32))
}

/// Launches a C-function as a DTA-V3 stackful Fiber with an ownership cleanup callback.
/// 
/// # Safety
/// * `func` and `cleanup` must be valid function pointers.
/// * `cleanup` will be called with `arg` once the fiber has finished execution.
#[unsafe(no_mangle)]
pub extern "C" fn dtact_fiber_launch_with_cleanup(
    func: extern "C" fn(*mut c_void),
    arg: *mut c_void,
    cleanup: unsafe extern "C" fn(*mut c_void),
) -> dtact_handle_t {
    let runtime = crate::GLOBAL_RUNTIME.get().expect("Dtact Runtime not initialized");
    let pool = &runtime.pool;
    let ctx_id = pool.alloc_context().expect("Context pool exhausted - OOM");
    
    let ctx_ptr = pool.get_context_ptr(ctx_id);
    let current_core = crate::api::topology::current().core_id as usize;

    unsafe {
        (*ctx_ptr).state.store(crate::memory_management::FiberStatus::Running as u8, core::sync::atomic::Ordering::Release);
        (*ctx_ptr).origin_core = current_core as u16;
        (*ctx_ptr).fiber_index = ctx_id;
        (*ctx_ptr).switch_fn = crate::context_switch::switch_context_cross_thread_float;
        
        (*ctx_ptr).closure_ptr = arg as *mut ();
        (*ctx_ptr).trampoline = core::mem::transmute(func);
        (*ctx_ptr).cleanup_fn = Some(core::mem::transmute(cleanup));
        
        // Unified Trampoline for C-Functions
        (*ctx_ptr).invoke_closure = |ptr| {
            let ctx = &mut *(ptr as *mut crate::memory_management::FiberContext);
            let f: extern "C" fn(*mut c_void) = core::mem::transmute(ctx.trampoline);
            f(ctx.closure_ptr as *mut c_void);
        };
        
        // ABI-Compliant Stack Alignment & Poisoning
        let stack_top = (ctx_ptr as usize & !0xF) - 64;
        let stack_top_ptr = stack_top as *mut u64;
        core::ptr::write(stack_top_ptr, dtact_abort as *const () as u64);
        
        let stack_top = stack_top as *mut u8;

        #[cfg(target_arch = "x86_64")]
        {
            (*ctx_ptr).regs.gprs[0] = stack_top as u64; // RSP
            (*ctx_ptr).regs.gprs[7] = crate::api::fiber_entry_point as *const () as u64; // RIP
        }
        #[cfg(target_arch = "aarch64")]
        {
            (*ctx_ptr).regs.gprs[12] = stack_top as u64; // SP
            (*ctx_ptr).regs.gprs[11] = crate::api::fiber_entry_point as u64; // x30 (LR)
        }
        #[cfg(target_arch = "riscv64")]
        {
            (*ctx_ptr).regs.gprs[0] = stack_top as u64; // SP
            (*ctx_ptr).regs.gprs[13] = crate::api::fiber_entry_point as u64; // RA
        }
    }

    crate::wake_fiber(current_core, ctx_id);
    dtact_handle_t((ctx_id as u64) | ((current_core as u64) << 32))
}

/// Blocks the current thread until the specified fiber terminates.
/// 
/// If called from a Dtact fiber, this will natively yield the physical core.
/// If called from a non-managed thread (e.g., C main), this uses a tiered
/// spin-loop and futex-wait strategy for zero-CPU idling.
#[unsafe(no_mangle)]
pub extern "C" fn dtact_await(handle: dtact_handle_t) {
    let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(|c| c.get());
    if ctx_ptr.is_null() {
        // UNIVERSAL WAIT: If called from a non-Fiber thread (e.g., C++ main), 
        // we use a tiered strategy: spin-loop -> futex_wait.
        let target_ctx_id = (handle.0 & 0xFFFFFFFF) as u32;
        let runtime = crate::GLOBAL_RUNTIME.get().expect("Runtime not initialized");
        let pool = &runtime.pool;
        let target_ctx = pool.get_context_ptr(target_ctx_id);
        
        let mut spins = 0;
        loop {
            let status = unsafe { (*target_ctx).state.load(core::sync::atomic::Ordering::Acquire) };
            if status == crate::memory_management::FiberStatus::Initial as u8 {
                break;
            }
            if spins < 100 {
                core::hint::spin_loop();
                spins += 1;
            } else {
                // Perform a zero-overhead OS-level wait until the Fiber finishes
                unsafe { crate::utils::futex_wait(&(*target_ctx).state, status) };
            }
        }
        return;
    }

    let target_ctx_id = (handle.0 & 0xFFFFFFFF) as u32;
    let runtime = crate::GLOBAL_RUNTIME.get().expect("Runtime not initialized");
    let pool = &runtime.pool;
    let target_ctx = pool.get_context_ptr(target_ctx_id);

    loop {
        let status = unsafe { (*target_ctx).state.load(core::sync::atomic::Ordering::Acquire) };
        if status == crate::memory_management::FiberStatus::Initial as u8 {
            break;
        }

        // Yield the current fiber natively to the scheduler
        unsafe {
            let ctx = &mut *ctx_ptr;
            ctx.state.store(crate::memory_management::FiberStatus::Yielded as u8, core::sync::atomic::Ordering::Release);
            
            // Invoke the assembly trampoline to swap stacks back to the scheduler
            ((*ctx).switch_fn)(
                &mut (*ctx).regs,
                &(*ctx).executor_regs,
            );
        }
    }
}

/// Spawns the hardware worker threads and starts the Dtact runtime.
/// This call blocks until all worker threads terminate.
#[unsafe(no_mangle)]
pub extern "C" fn dtact_run(_rt: *mut c_void) {
    let runtime = crate::GLOBAL_RUNTIME.get().expect("Dtact Runtime not initialized");
    let scheduler = &runtime.scheduler;
    let pool = &runtime.pool;
    let (base, sz, guard_sz) = pool.get_dispatch_layout();
    
    let workers_count = scheduler.workers.len();
    let mut handles = alloc::vec::Vec::with_capacity(workers_count);
    
    for i in 0..workers_count {
        // Capture raw pointers to avoid lifetime issues across thread boundaries
        let scheduler_ptr = scheduler as *const _ as usize;
        let base_ptr = base as usize;
        
        let handle = std::thread::spawn(move || {
            let s = unsafe { &*(scheduler_ptr as *const crate::dta_scheduler::DtaScheduler) };
            let b = base_ptr as *mut u8;
            s.run_worker(i, b, sz, guard_sz);
        });
        handles.push(handle);
    }
    
    // Block until all hardware worker threads terminate
    for h in handles {
        let _ = h.join();
    }
}

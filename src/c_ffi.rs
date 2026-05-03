use crate::dta_scheduler::TopologyMode;
use crate::memory_management::SafetyLevel;
use core::ffi::c_void;

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
    /// Maximum number of concurrent fibers.
    pub contexts: u32,
    /// Stack size per fiber in bytes.
    pub stack_size: u64,
}

/// Returns the recommended default configuration for the Dtact runtime.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dtact_default_config(cfg: *mut dtact_config_t) {
    let cfg = unsafe { &mut *cfg };
    *cfg = dtact_config_t {
        workers: 0,       // Auto-detect
        safety_level: 1,  // Safety1
        topology_mode: 0, // P2PMesh
        contexts: 4096,
        stack_size: 2 * 1024 * 1024,
    };
}

/// Initializes the global Dtact runtime singleton.
///
/// # Safety
/// * This function should be called once at application startup.
/// * `cfg` must be a valid, non-null pointer to a `dtact_config_t` structure.
///
/// # Panics
/// * Panics if the runtime is already initialized or if memory allocation fails.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dtact_init(cfg: *const dtact_config_t) -> *mut c_void {
    let cfg = unsafe { &*cfg };
    let _workers = if cfg.workers == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        cfg.workers as usize
    };

    let safety = match cfg.safety_level {
        0 => SafetyLevel::Safety0,
        2 => SafetyLevel::Safety2,
        _ => SafetyLevel::Safety1,
    };

    let topology = match cfg.topology_mode {
        1 => TopologyMode::Global,
        _ => TopologyMode::P2PMesh,
    };

    unsafe {
        if !crate::GLOBAL_RUNTIME.is_null() {
            return crate::GLOBAL_RUNTIME as *mut c_void;
        }

        let scheduler = crate::dta_scheduler::DtaScheduler::new(_workers, topology);
        let pool = crate::memory_management::ContextPool::new(
            cfg.contexts,
            cfg.stack_size as usize,
            safety,
            0,
        )
        .expect("DTA-V3 FFI Initialization Failed");

        let runtime = Box::new(crate::Runtime {
            scheduler: Box::new(scheduler),
            pool: Box::new(pool),
            started: core::sync::atomic::AtomicBool::new(false),
            shutdown: core::sync::atomic::AtomicBool::new(false),
        });

        let ptr = Box::into_raw(runtime);
        crate::GLOBAL_RUNTIME = ptr;
        ptr as *mut c_void
    }
}

/// Critical failure handler. Aborts the process if a fiber attempts to
/// return without properly terminating via the runtime.
#[unsafe(no_mangle)]
pub extern "C" fn dtact_abort() -> ! {
    eprintln!("DTA-V3 Critical: Fiber attempted to 'return' instead of yielding. Stack corrupted.");
    std::process::abort();
}

/// Frees an argument pointer previously allocated for a fiber.
///
/// # Safety
/// * `arg` must be a valid pointer previously allocated by the C allocator (e.g. `malloc`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dtact_free_arg(arg: *mut c_void) {
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
///
/// # Panics
/// * Panics if the runtime is not initialized.
/// * Panics if the context pool is exhausted.
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern "C" fn dtact_fiber_launch(
    func: extern "C" fn(*mut c_void),
    arg: *mut c_void,
) -> dtact_handle_t {
    let runtime = unsafe {
        crate::GLOBAL_RUNTIME
            .as_ref()
            .expect("Dtact Runtime not initialized")
    };
    let pool = &runtime.pool;
    let ctx_id = pool.alloc_context().expect("Context pool exhausted - OOM");

    let ctx_ptr = pool.get_context_ptr(ctx_id);
    #[allow(clippy::cast_possible_truncation)]
    let current_core = crate::api::topology::current().core_id as usize;

    unsafe {
        (*ctx_ptr).state.store(
            crate::memory_management::FiberStatus::Running as u8,
            core::sync::atomic::Ordering::Release,
        );
        (*ctx_ptr).origin_core = current_core as u16;
        (*ctx_ptr).fiber_index = ctx_id;
        (*ctx_ptr).switch_fn = crate::context_switch::switch_context_cross_thread_float;

        (*ctx_ptr).closure_ptr = arg.cast::<()>();
        (*ctx_ptr).trampoline =
            core::mem::transmute::<extern "C" fn(*mut c_void), unsafe extern "C" fn()>(func);

        // Unified Trampoline for C-Functions
        (*ctx_ptr).invoke_closure = |ptr| {
            // In C-FFI, we don't need the ctx here, we just call the trampoline with ptr (which is closure_ptr)
            let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(std::cell::Cell::get);
            if let Some(ctx) = unsafe { ctx_ptr.as_ref() } {
                let f: extern "C" fn(*mut c_void) = unsafe { core::mem::transmute(ctx.trampoline) };
                f(ptr.cast::<c_void>());
            }
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

    let r#gen = unsafe {
        (*ctx_ptr)
            .generation
            .load(core::sync::atomic::Ordering::Acquire)
    } as u64;
    crate::wake_fiber(current_core, ctx_id);

    // Handle Layout: [16-bit Generation | 16-bit CoreID | 32-bit ContextID]
    let handle_val = u64::from(ctx_id) | ((current_core as u64) << 32) | ((r#gen & 0xFFFF) << 48);
    dtact_handle_t(handle_val)
}

/// Launches a C-function as a DTA-V3 stackful Fiber with an ownership cleanup callback.
///
/// # Safety
/// * `func` and `cleanup` must be valid function pointers.
/// * `cleanup` will be called with `arg` once the fiber has finished execution.
///
/// # Panics
/// * Panics if the runtime is not initialized.
/// * Panics if the context pool is exhausted.
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation)]
pub unsafe extern "C" fn dtact_fiber_launch_with_cleanup(
    func: extern "C" fn(*mut c_void),
    arg: *mut c_void,
    cleanup: unsafe extern "C" fn(*mut c_void),
) -> dtact_handle_t {
    let runtime = unsafe {
        crate::GLOBAL_RUNTIME
            .as_ref()
            .expect("Dtact Runtime not initialized")
    };
    let pool = &runtime.pool;
    let ctx_id = pool.alloc_context().expect("Context pool exhausted - OOM");

    let ctx_ptr = pool.get_context_ptr(ctx_id);
    #[allow(clippy::cast_possible_truncation)]
    let current_core = crate::api::topology::current().core_id as usize;

    unsafe {
        (*ctx_ptr).state.store(
            crate::memory_management::FiberStatus::Running as u8,
            core::sync::atomic::Ordering::Release,
        );
        (*ctx_ptr).origin_core = current_core as u16;
        (*ctx_ptr).fiber_index = ctx_id;
        (*ctx_ptr).switch_fn = crate::context_switch::switch_context_cross_thread_float;

        (*ctx_ptr).closure_ptr = arg.cast::<()>();
        (*ctx_ptr).trampoline =
            core::mem::transmute::<extern "C" fn(*mut c_void), unsafe extern "C" fn()>(func);
        (*ctx_ptr).cleanup_fn = Some(core::mem::transmute::<
            unsafe extern "C" fn(*mut c_void),
            unsafe extern "C" fn(*mut ()),
        >(cleanup));

        // Unified Trampoline for C-Functions
        (*ctx_ptr).invoke_closure = |ptr| {
            let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(std::cell::Cell::get);
            if let Some(ctx) = unsafe { ctx_ptr.as_ref() } {
                let f: extern "C" fn(*mut c_void) = unsafe {
                    core::mem::transmute::<unsafe extern "C" fn(), extern "C" fn(*mut c_void)>(
                        ctx.trampoline,
                    )
                };
                f(ptr.cast::<c_void>());
            }
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
    dtact_handle_t(u64::from(ctx_id) | ((current_core as u64) << 32))
}

/// Blocks the current thread until the specified fiber terminates.
///
/// If called from a Dtact fiber, this will natively yield the physical core.
/// If called from a non-managed thread (e.g., C main), this uses a tiered
/// spin-loop and futex-wait strategy for zero-CPU idling.
///
/// # Panics
/// * Panics if the runtime is not initialized.
#[unsafe(no_mangle)]
pub extern "C" fn dtact_await(handle: dtact_handle_t) {
    let handle_val = handle.0 & !(1 << 63); // Strip sentinel bit
    let target_ctx_id = (handle_val & 0xFFFF_FFFF) as u32;
    let handle_gen = (handle_val >> 48) as u16;
    let runtime = unsafe {
        crate::GLOBAL_RUNTIME
            .as_ref()
            .expect("Runtime not initialized")
    };
    let pool = &runtime.pool;
    let target_ctx = pool.get_context_ptr(target_ctx_id);

    let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(std::cell::Cell::get);

    if ctx_ptr.is_null() {
        // ===== NON-FIBER PATH (C main thread, host thread) =====
        // Tiered strategy: spin-loop -> futex_wait
        let mut spins = 0u32;
        loop {
            let current_gen = unsafe {
                (*target_ctx)
                    .generation
                    .load(core::sync::atomic::Ordering::Acquire)
            } as u16;
            let status = unsafe {
                (*target_ctx)
                    .state
                    .load(core::sync::atomic::Ordering::Acquire)
            };

            if current_gen != handle_gen
                || status == crate::memory_management::FiberStatus::Initial as u8
                || status == crate::memory_management::FiberStatus::Finished as u8
            {
                break;
            }

            if spins < 100 {
                core::hint::spin_loop();
                spins += 1;
            } else {
                unsafe { crate::utils::futex_wait(&raw const (*target_ctx).state, status) };
            }
        }
        return;
    }

    // ===== FIBER PATH (called from within a running fiber) =====
    loop {
        // Clear Notified state before starting check
        unsafe {
            (*ctx_ptr)
                .state
                .store(crate::memory_management::FiberStatus::Running as u8, core::sync::atomic::Ordering::Release);
        }

        // 0. Check target state and generation
        let current_gen = unsafe {
            (*target_ctx)
                .generation
                .load(core::sync::atomic::Ordering::Acquire)
        } as u16;
        let status = unsafe {
            (*target_ctx)
                .state
                .load(core::sync::atomic::Ordering::Acquire)
        };
        // unsafe { libc::printf(b"[Dtact] Fiber %u checking Target %u: state=%u\n\0".as_ptr().cast(), (*ctx_ptr).fiber_index, target_ctx_id, status); }

        if current_gen != handle_gen
            || status == crate::memory_management::FiberStatus::Initial as u8
            || status == crate::memory_management::FiberStatus::Finished as u8
        {
            // Target already finished (or context recycled), clear waiter and break
            unsafe {
                (*target_ctx)
                    .waiter_handle
                    .store(0, core::sync::atomic::Ordering::Relaxed);
            }
            break;
        }

        // 1. Register the current fiber as a waiter for the target fiber
        let current_worker = crate::future_bridge::CURRENT_WORKER_ID.with(|c| c.get());
        let current_ctx_id = unsafe { (*ctx_ptr).fiber_index as u64 };
        let my_handle = current_ctx_id | ((current_worker as u64) << 32);

        unsafe {
            (*target_ctx)
                .waiter_handle
                .store(my_handle, core::sync::atomic::Ordering::Release);
        }

        // Full memory barrier: waiter_handle store must be visible before state re-check
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);

        // 2. Double-check target state after registering waiter
        let current_gen = unsafe {
            (*target_ctx)
                .generation
                .load(core::sync::atomic::Ordering::Acquire)
        } as u16;
        let status = unsafe {
            (*target_ctx)
                .state
                .load(core::sync::atomic::Ordering::Acquire)
        };

        if current_gen != handle_gen
            || status == crate::memory_management::FiberStatus::Initial as u8
            || status == crate::memory_management::FiberStatus::Finished as u8
        {
            // Completed between check and waiter registration
            unsafe {
                (*target_ctx)
                    .waiter_handle
                    .store(0, core::sync::atomic::Ordering::Relaxed);
            }
            break;
        }

        // 3. Try to transition to Yielded and suspend
        unsafe {
            let ctx = &mut *ctx_ptr;
            if ctx
                .state
                .compare_exchange(
                    crate::memory_management::FiberStatus::Running as u8,
                    crate::memory_management::FiberStatus::Yielded as u8,
                    core::sync::atomic::Ordering::Release,
                    core::sync::atomic::Ordering::Acquire,
                )
                .is_ok()
            {
                (ctx.switch_fn)(&raw mut ctx.regs, &raw const ctx.executor_regs);
            }
        }
    }
}

/// Signals all worker threads to shutdown and waits for them to terminate.
/// This call blocks until all hardware worker threads have exited.
///
/// # Panics
/// * Panics if the runtime is not initialized.#[unsafe(no_mangle)]
pub unsafe extern "C" fn dtact_init(cfg: *const dtact_config_t) -> *mut c_void {
    let cfg = unsafe { &*cfg };
    let _workers = if cfg.workers == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        cfg.workers as usize
    };

    let safety = match cfg.safety_level {
        0 => crate::memory_management::SafetyLevel::Safety0,
        1 => crate::memory_management::SafetyLevel::Safety1,
        2 => crate::memory_management::SafetyLevel::Safety2,
        _ => crate::memory_management::SafetyLevel::Safety1,
    };

    let topology = match cfg.topology_mode {
        0 => crate::dta_scheduler::TopologyMode::P2PMesh,
        1 => crate::dta_scheduler::TopologyMode::Global,
        _ => crate::dta_scheduler::TopologyMode::P2PMesh,
    };

    let scheduler = crate::dta_scheduler::DtaScheduler::new(_workers, topology);
    let pool = crate::memory_management::ContextPool::new(
        cfg.contexts,
        cfg.stack_size as usize,
        safety,
        0,
    )
    .expect("DTA-V3 FFI Initialization Failed");

    let mut runtime = Box::new(crate::Runtime {
        scheduler,
        pool,
        started: core::sync::atomic::AtomicBool::new(false),
        shutdown: core::sync::atomic::AtomicBool::new(false),
        handles: Vec::new(),
    });

    // Spawn threads early in init
    let workers_count = runtime.scheduler.workers.len();
    for i in 0..workers_count {
        let handle = std::thread::spawn(move || {
            if let Some(runtime) = unsafe { crate::GLOBAL_RUNTIME.as_ref() } {
                crate::dta_scheduler::DtaScheduler::run_worker_static(
                    &runtime.scheduler,
                    i,
                    &runtime.pool,
                    &runtime.shutdown,
                );
            }
        });
        runtime.handles.push(handle);
    }

    let ptr = Box::into_raw(runtime);
    crate::GLOBAL_RUNTIME = ptr;
    ptr as *mut c_void
}

#[unsafe(no_mangle)]
pub extern "C" fn dtact_run(_rt: *mut c_void) {
    if let Some(runtime) = unsafe { crate::GLOBAL_RUNTIME.as_ref() } {
        runtime.started.store(true, core::sync::atomic::Ordering::Release);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dtact_shutdown() {
    unsafe {
        if let Some(runtime_ptr) = crate::GLOBAL_RUNTIME.as_mut() {
            runtime_ptr.shutdown.store(true, core::sync::atomic::Ordering::SeqCst);
            
            // Join threads
            let handles = core::mem::take(&mut runtime_ptr.handles);
            for h in handles {
                let _ = h.join();
            }

            // Drop runtime
            let _ = Box::from_raw(crate::GLOBAL_RUNTIME);
            crate::GLOBAL_RUNTIME = core::ptr::null_mut();
        }
    }
}

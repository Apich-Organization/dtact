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
    /// Topology mode (0: `P2PMesh`, 1: Global).
    pub topology_mode: u8,
    /// NUMA node for memory allocation. Set to 0 for default (local node).
    pub numa: u8,
    /// Maximum number of concurrent fibers. Set to 0 for default (4096).
    pub fiber_capacity: u32,
    /// Stack size per fiber in bytes. Set to 0 for default (512KB).
    pub stack_size: u32,
}

/// Advanced options for spawning a fiber from C FFI.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct dtact_spawn_options_t {
    /// 0: Low, 1: Normal, 2: High, 3: Critical
    pub priority: u8,
    /// 0: `SameCore`, 1: `SameCCX`, 2: `SameNUMA`, 3: Any
    pub affinity: u8,
    /// 0: Compute, 1: IO, 2: Memory, 3: System
    pub kind: u8,
    /// 0: `CrossThreadFloat`, 1: `CrossThreadNoFloat`, 2: `SameThreadFloat`, 3: `SameThreadNoFloat`
    pub switcher: u8,
}

/// Returns the recommended default options for the Dtact runtime.
#[unsafe(no_mangle)]
pub const extern "C" fn dtact_default_spawn_options() -> dtact_spawn_options_t {
    dtact_spawn_options_t {
        priority: 1, // Normal
        affinity: 0, // SameCore
        kind: 0,     // Compute
        switcher: 0, // CrossThreadFloat
    }
}

/// Returns the recommended default configuration for the Dtact runtime.
#[unsafe(no_mangle)]
pub const extern "C" fn dtact_default_config() -> dtact_config_t {
    dtact_config_t {
        workers: 0,        // Auto-detect
        safety_level: 1,   // Safety1
        topology_mode: 0,  // P2PMesh
        numa: 0,           // Default local node
        fiber_capacity: 0, // Default 4096
        stack_size: 0,     // Default 512KB
    }
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
    let workers = if cfg.workers == 0 {
        std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
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

    let capacity = if cfg.fiber_capacity == 0 {
        4096
    } else {
        cfg.fiber_capacity
    };
    let stack_size = if cfg.stack_size == 0 {
        512 * 1024
    } else {
        cfg.stack_size as usize
    };

    crate::GLOBAL_RUNTIME.get_or_init(|| {
        let scheduler = crate::dta_scheduler::DtaScheduler::new(workers, topology);
        let pool = crate::memory_management::ContextPool::new(
            capacity,
            stack_size,
            safety,
            cfg.numa as usize,
        )
        .expect("DTA-V3 FFI Initialization Failed");
        crate::Runtime {
            scheduler,
            pool,
            started: crate::sync::atomic::AtomicBool::new(false),
            shutdown: crate::sync::atomic::AtomicBool::new(false),
        }
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
    let Some(runtime) = crate::GLOBAL_RUNTIME.get() else {
        return dtact_handle_t(0);
    };
    let pool = &runtime.pool;
    let Some(ctx_id) = pool.alloc_context() else {
        return dtact_handle_t(0);
    };

    let ctx_ptr = pool.get_context_ptr(ctx_id);
    #[allow(clippy::cast_possible_truncation)]
    let current_core = crate::api::topology::current().core_id as usize;

    unsafe {
        (*ctx_ptr).state.store(
            crate::memory_management::FiberStatus::Running as u32,
            core::sync::atomic::Ordering::Release,
        );
        (*ctx_ptr).origin_core = current_core as u16;
        (*ctx_ptr).fiber_index = ctx_id;
        (*ctx_ptr).switch_fn = crate::context_switch::switch_context_cross_thread_float;
        (*ctx_ptr).kind = crate::common_types::WorkloadKind::Compute;
        (*ctx_ptr).mode = crate::common_types::TopologyMode::P2PMesh;
        (*ctx_ptr).affinity = crate::api::topology::Affinity::SameCore;

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
        // We leave 72 bytes for Shadow Space (Windows) and Future safety.
        // -72 ensures that RSP is 16-byte aligned + 8, which is required by the Windows x64 ABI.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let stack_top = (ctx_ptr as usize & !0xF) - 72;
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let stack_top = (ctx_ptr as usize & !0xF) - 80;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let stack_top_ptr = stack_top as *mut u64;

        // Place a "poison" return address on the stack.
        // If the fiber function ever attempts to 'ret', it will jump here and abort.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        core::ptr::write(stack_top_ptr, dtact_abort as *const () as u64);

        let stack_top = stack_top as *mut u8;

        #[cfg(target_arch = "x86_64")]
        {
            (*ctx_ptr).regs.gprs[0] = stack_top as u64; // RSP
            (*ctx_ptr).regs.gprs[7] = crate::api::fiber_entry_point as *const () as u64; // RIP
            #[cfg(windows)]
            {
                let limit = (ctx_ptr as usize).saturating_sub(pool.slot_size);
                (*ctx_ptr).regs.gprs[10] = ctx_ptr as u64; // Stack Base
                (*ctx_ptr).regs.gprs[11] = limit as u64; // Stack Limit
                (*ctx_ptr).regs.gprs[12] = limit as u64; // DeallocationStack
                (*ctx_ptr).regs.gprs[13] = !0; // ExceptionList
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            (*ctx_ptr).regs.gprs[12] = stack_top as u64; // SP
            (*ctx_ptr).regs.gprs[11] = crate::api::fiber_entry_point as *const () as u64; // x30 (LR)
            #[cfg(windows)]
            {
                let limit = (ctx_ptr as usize).saturating_sub(pool.slot_size);
                (*ctx_ptr).regs.gprs[13] = ctx_ptr as u64; // Stack Base
                (*ctx_ptr).regs.gprs[14] = limit as u64; // Stack Limit
                (*ctx_ptr).regs.gprs[15] = limit as u64; // DeallocationStack
            }
        }
        #[cfg(target_arch = "riscv64")]
        {
            (*ctx_ptr).regs.gprs[0] = stack_top as u64; // SP
            (*ctx_ptr).regs.gprs[13] = crate::api::fiber_entry_point as *const () as u64; // RA
        }
        (*ctx_ptr).cleanup_fn = None;
    }

    let r#gen = u64::from(unsafe {
        (*ctx_ptr)
            .generation
            .load(core::sync::atomic::Ordering::Acquire)
    });
    crate::wake_fiber(current_core, ctx_id);

    // Handle Layout: [1-bit Valid | 15-bit Generation | 16-bit CoreID | 32-bit ContextID]
    dtact_handle_t(
        u64::from(ctx_id) | ((current_core as u64) << 32) | ((r#gen & 0x7FFF) << 48) | (1 << 63),
    )
}

/// Launches a C-function as a DTA-V3 stackful Fiber with advanced options.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::too_many_lines)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dtact_fiber_launch_ext(
    func: extern "C" fn(*mut c_void),
    arg: *mut c_void,
    options: *const dtact_spawn_options_t,
) -> dtact_handle_t {
    let Some(runtime) = crate::GLOBAL_RUNTIME.get() else {
        return dtact_handle_t(0);
    };
    let pool = &runtime.pool;
    let Some(ctx_id) = pool.alloc_context() else {
        return dtact_handle_t(0);
    };

    let ctx_ptr = pool.get_context_ptr(ctx_id);
    let current_core = crate::api::topology::current().core_id as usize;

    let opts = if options.is_null() {
        dtact_default_spawn_options()
    } else {
        unsafe { *options }
    };

    unsafe {
        (*ctx_ptr).state.store(
            crate::memory_management::FiberStatus::Running as u32,
            core::sync::atomic::Ordering::Release,
        );
        (*ctx_ptr).origin_core = current_core as u16;
        (*ctx_ptr).fiber_index = ctx_id;

        (*ctx_ptr).switch_fn = match opts.switcher {
            1 => crate::context_switch::switch_context_cross_thread_no_float,
            2 => crate::context_switch::switch_context_same_thread_float,
            3 => crate::context_switch::switch_context_same_thread_no_float,
            _ => crate::context_switch::switch_context_cross_thread_float,
        };

        let allow_deflection = opts.switcher == 0 || opts.switcher == 1;
        (*ctx_ptr).mode = if allow_deflection {
            let topology = runtime.scheduler.topology;
            match topology {
                crate::dta_scheduler::TopologyMode::Global => {
                    crate::common_types::TopologyMode::Global
                }
                _ => crate::common_types::TopologyMode::P2PMesh,
            }
        } else {
            crate::common_types::TopologyMode::Pinned
        };

        (*ctx_ptr).affinity = match opts.affinity {
            1 => crate::api::topology::Affinity::SameCCX,
            2 => crate::api::topology::Affinity::SameNUMA,
            3 => crate::api::topology::Affinity::Any,
            _ => crate::api::topology::Affinity::SameCore,
        };

        (*ctx_ptr).kind = match opts.kind {
            1 => crate::common_types::WorkloadKind::IO,
            2 => crate::common_types::WorkloadKind::Memory,
            3 => crate::common_types::WorkloadKind::System,
            _ => crate::common_types::WorkloadKind::Compute,
        };

        (*ctx_ptr).adaptive_spin_count = match (*ctx_ptr).kind {
            crate::common_types::WorkloadKind::Compute => 1000,
            crate::common_types::WorkloadKind::IO => 100,
            crate::common_types::WorkloadKind::Memory => 500,
            crate::common_types::WorkloadKind::System => 200,
        };

        (*ctx_ptr).closure_ptr = arg.cast::<()>();
        (*ctx_ptr).trampoline =
            core::mem::transmute::<extern "C" fn(*mut c_void), unsafe extern "C" fn()>(func);

        (*ctx_ptr).invoke_closure = |ptr| {
            let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(std::cell::Cell::get);
            if let Some(ctx) = unsafe { ctx_ptr.as_ref() } {
                let f: extern "C" fn(*mut c_void) = unsafe { core::mem::transmute(ctx.trampoline) };
                f(ptr.cast::<c_void>());
            }
        };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let stack_top = (ctx_ptr as usize & !0xF) - 72;
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let stack_top = (ctx_ptr as usize & !0xF) - 80;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let stack_top_ptr = stack_top as *mut u64;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        core::ptr::write(stack_top_ptr, dtact_abort as *const () as u64);
        let stack_top = stack_top as *mut u8;

        #[cfg(target_arch = "x86_64")]
        {
            (*ctx_ptr).regs.gprs[0] = stack_top as u64;
            (*ctx_ptr).regs.gprs[7] = crate::api::fiber_entry_point as *const () as u64;
            #[cfg(windows)]
            {
                let limit = (ctx_ptr as usize).saturating_sub(pool.slot_size);
                (*ctx_ptr).regs.gprs[10] = ctx_ptr as u64;
                (*ctx_ptr).regs.gprs[11] = limit as u64;
                (*ctx_ptr).regs.gprs[12] = limit as u64;
                (*ctx_ptr).regs.gprs[13] = !0;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            (*ctx_ptr).regs.gprs[12] = stack_top as u64;
            (*ctx_ptr).regs.gprs[11] = crate::api::fiber_entry_point as *const () as u64;
            #[cfg(windows)]
            {
                let limit = (ctx_ptr as usize).saturating_sub(pool.slot_size);
                (*ctx_ptr).regs.gprs[13] = ctx_ptr as u64;
                (*ctx_ptr).regs.gprs[14] = limit as u64;
                (*ctx_ptr).regs.gprs[15] = limit as u64;
            }
        }
        #[cfg(target_arch = "riscv64")]
        {
            (*ctx_ptr).regs.gprs[0] = stack_top as u64;
            (*ctx_ptr).regs.gprs[13] = crate::api::fiber_entry_point as *const () as u64;
        }
        (*ctx_ptr).cleanup_fn = None;
    }

    let r#gen = u64::from(unsafe {
        (*ctx_ptr)
            .generation
            .load(core::sync::atomic::Ordering::Acquire)
    });
    crate::wake_fiber(current_core, ctx_id);

    dtact_handle_t(
        u64::from(ctx_id) | ((current_core as u64) << 32) | ((r#gen & 0x7FFF) << 48) | (1 << 63),
    )
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
    let Some(runtime) = crate::GLOBAL_RUNTIME.get() else {
        return dtact_handle_t(0);
    };
    let pool = &runtime.pool;
    let Some(ctx_id) = pool.alloc_context() else {
        return dtact_handle_t(0);
    };

    let ctx_ptr = pool.get_context_ptr(ctx_id);
    #[allow(clippy::cast_possible_truncation)]
    let current_core = crate::api::topology::current().core_id as usize;

    unsafe {
        (*ctx_ptr).state.store(
            crate::memory_management::FiberStatus::Running as u32,
            core::sync::atomic::Ordering::Release,
        );
        (*ctx_ptr).origin_core = current_core as u16;
        (*ctx_ptr).fiber_index = ctx_id;
        (*ctx_ptr).switch_fn = crate::context_switch::switch_context_cross_thread_float;
        (*ctx_ptr).kind = crate::common_types::WorkloadKind::Compute;
        (*ctx_ptr).mode = crate::common_types::TopologyMode::P2PMesh;
        (*ctx_ptr).affinity = crate::api::topology::Affinity::SameCore;

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
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let stack_top = (ctx_ptr as usize & !0xF) - 72;
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let stack_top = (ctx_ptr as usize & !0xF) - 80;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let stack_top_ptr = stack_top as *mut u64;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        core::ptr::write(stack_top_ptr, dtact_abort as *const () as u64);

        let stack_top = stack_top as *mut u8;

        #[cfg(target_arch = "x86_64")]
        {
            (*ctx_ptr).regs.gprs[0] = stack_top as u64; // RSP
            (*ctx_ptr).regs.gprs[7] = crate::api::fiber_entry_point as *const () as u64; // RIP
            #[cfg(windows)]
            {
                let limit = (ctx_ptr as usize).saturating_sub(pool.slot_size);
                (*ctx_ptr).regs.gprs[10] = ctx_ptr as u64; // Stack Base
                (*ctx_ptr).regs.gprs[11] = limit as u64; // Stack Limit
                (*ctx_ptr).regs.gprs[12] = limit as u64; // DeallocationStack
                (*ctx_ptr).regs.gprs[13] = !0; // ExceptionList
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            (*ctx_ptr).regs.gprs[12] = stack_top as u64; // SP
            (*ctx_ptr).regs.gprs[11] = crate::api::fiber_entry_point as *const () as u64; // x30 (LR)
            #[cfg(windows)]
            {
                let limit = (ctx_ptr as usize).saturating_sub(pool.slot_size);
                (*ctx_ptr).regs.gprs[13] = ctx_ptr as u64; // Stack Base
                (*ctx_ptr).regs.gprs[14] = limit as u64; // Stack Limit
                (*ctx_ptr).regs.gprs[15] = limit as u64; // DeallocationStack
            }
        }
        #[cfg(target_arch = "riscv64")]
        {
            (*ctx_ptr).regs.gprs[0] = stack_top as u64; // SP
            (*ctx_ptr).regs.gprs[13] = crate::api::fiber_entry_point as *const () as u64; // RA
        }
    }

    let r#gen = u64::from(unsafe {
        (*ctx_ptr)
            .generation
            .load(core::sync::atomic::Ordering::Acquire)
    });

    crate::wake_fiber(current_core, ctx_id);
    dtact_handle_t(
        u64::from(ctx_id) | ((current_core as u64) << 32) | ((r#gen & 0x7FFF) << 48) | (1 << 63),
    )
}

/// Launches a C-function as a DTA-V3 stackful Fiber with an ownership cleanup callback and options.
#[unsafe(no_mangle)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::too_many_lines)]
pub unsafe extern "C" fn dtact_fiber_launch_with_cleanup_ext(
    func: extern "C" fn(*mut c_void),
    arg: *mut c_void,
    cleanup: unsafe extern "C" fn(*mut c_void),
    options: *const dtact_spawn_options_t,
) -> dtact_handle_t {
    let Some(runtime) = crate::GLOBAL_RUNTIME.get() else {
        return dtact_handle_t(0);
    };
    let pool = &runtime.pool;
    let Some(ctx_id) = pool.alloc_context() else {
        return dtact_handle_t(0);
    };

    let ctx_ptr = pool.get_context_ptr(ctx_id);
    let current_core = crate::api::topology::current().core_id as usize;

    let opts = if options.is_null() {
        dtact_default_spawn_options()
    } else {
        unsafe { *options }
    };

    unsafe {
        (*ctx_ptr).state.store(
            crate::memory_management::FiberStatus::Running as u32,
            core::sync::atomic::Ordering::Release,
        );
        (*ctx_ptr).origin_core = current_core as u16;
        (*ctx_ptr).fiber_index = ctx_id;

        (*ctx_ptr).switch_fn = match opts.switcher {
            1 => crate::context_switch::switch_context_cross_thread_no_float,
            2 => crate::context_switch::switch_context_same_thread_float,
            3 => crate::context_switch::switch_context_same_thread_no_float,
            _ => crate::context_switch::switch_context_cross_thread_float,
        };

        let allow_deflection = opts.switcher == 0 || opts.switcher == 1;
        (*ctx_ptr).mode = if allow_deflection {
            let topology = runtime.scheduler.topology;
            match topology {
                crate::dta_scheduler::TopologyMode::Global => {
                    crate::common_types::TopologyMode::Global
                }
                _ => crate::common_types::TopologyMode::P2PMesh,
            }
        } else {
            crate::common_types::TopologyMode::Pinned
        };

        (*ctx_ptr).affinity = match opts.affinity {
            1 => crate::api::topology::Affinity::SameCCX,
            2 => crate::api::topology::Affinity::SameNUMA,
            3 => crate::api::topology::Affinity::Any,
            _ => crate::api::topology::Affinity::SameCore,
        };

        (*ctx_ptr).kind = match opts.kind {
            1 => crate::common_types::WorkloadKind::IO,
            2 => crate::common_types::WorkloadKind::Memory,
            3 => crate::common_types::WorkloadKind::System,
            _ => crate::common_types::WorkloadKind::Compute,
        };

        (*ctx_ptr).adaptive_spin_count = match (*ctx_ptr).kind {
            crate::common_types::WorkloadKind::Compute => 1000,
            crate::common_types::WorkloadKind::IO => 100,
            crate::common_types::WorkloadKind::Memory => 500,
            crate::common_types::WorkloadKind::System => 200,
        };

        (*ctx_ptr).closure_ptr = arg.cast::<()>();
        (*ctx_ptr).trampoline =
            core::mem::transmute::<extern "C" fn(*mut c_void), unsafe extern "C" fn()>(func);
        (*ctx_ptr).cleanup_fn = Some(core::mem::transmute::<
            unsafe extern "C" fn(*mut c_void),
            unsafe extern "C" fn(*mut ()),
        >(cleanup));

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

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let stack_top = (ctx_ptr as usize & !0xF) - 72;
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let stack_top = (ctx_ptr as usize & !0xF) - 80;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let stack_top_ptr = stack_top as *mut u64;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        core::ptr::write(stack_top_ptr, dtact_abort as *const () as u64);
        let stack_top = stack_top as *mut u8;

        #[cfg(target_arch = "x86_64")]
        {
            (*ctx_ptr).regs.gprs[0] = stack_top as u64;
            (*ctx_ptr).regs.gprs[7] = crate::api::fiber_entry_point as *const () as u64;
            #[cfg(windows)]
            {
                let limit = (ctx_ptr as usize).saturating_sub(pool.slot_size);
                (*ctx_ptr).regs.gprs[10] = ctx_ptr as u64;
                (*ctx_ptr).regs.gprs[11] = limit as u64;
                (*ctx_ptr).regs.gprs[12] = limit as u64;
                (*ctx_ptr).regs.gprs[13] = !0;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            (*ctx_ptr).regs.gprs[12] = stack_top as u64;
            (*ctx_ptr).regs.gprs[11] = crate::api::fiber_entry_point as *const () as u64;
            #[cfg(windows)]
            {
                let limit = (ctx_ptr as usize).saturating_sub(pool.slot_size);
                (*ctx_ptr).regs.gprs[13] = ctx_ptr as u64;
                (*ctx_ptr).regs.gprs[14] = limit as u64;
                (*ctx_ptr).regs.gprs[15] = limit as u64;
            }
        }
        #[cfg(target_arch = "riscv64")]
        {
            (*ctx_ptr).regs.gprs[0] = stack_top as u64;
            (*ctx_ptr).regs.gprs[13] = crate::api::fiber_entry_point as *const () as u64;
        }
    }

    let r#gen = u64::from(unsafe {
        (*ctx_ptr)
            .generation
            .load(core::sync::atomic::Ordering::Acquire)
    });
    crate::wake_fiber(current_core, ctx_id);

    dtact_handle_t(
        u64::from(ctx_id) | ((current_core as u64) << 32) | ((r#gen & 0x7FFF) << 48) | (1 << 63),
    )
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
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::too_many_lines)]
pub extern "C" fn dtact_await(handle: dtact_handle_t) {
    let handle_val = handle.0 & !(1 << 63); // Strip sentinel bit
    let target_ctx_id = (handle_val & 0xFFFF_FFFF) as u32;
    let handle_gen = ((handle_val >> 48) & 0x7FFF) as u16; // Mask out sentinel bit
    let Some(runtime) = crate::GLOBAL_RUNTIME.get() else {
        return;
    };
    let pool = &runtime.pool;
    let target_ctx = pool.get_context_ptr(target_ctx_id);

    let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(std::cell::Cell::get);

    if ctx_ptr.is_null() {
        // ===== NON-FIBER PATH (C main thread, host thread) =====
        // Three consecutive Acquire loads bracket the state read — no SeqCst fence needed.
        // On AArch64, consecutive `ldar` instructions cannot be reordered by the hardware.
        let mut spins = 0u32;
        loop {
            let (_current_gen, status) = unsafe {
                let g1 = ((*target_ctx)
                    .generation
                    .load(core::sync::atomic::Ordering::Acquire) as u16)
                    & 0x7FFF;
                let status = (*target_ctx)
                    .state
                    .load(core::sync::atomic::Ordering::Acquire);
                let g2 = ((*target_ctx)
                    .generation
                    .load(core::sync::atomic::Ordering::Acquire) as u16)
                    & 0x7FFF;

                if g1 != handle_gen || g2 != handle_gen {
                    break;
                }
                (g1, status)
            };

            if status == crate::memory_management::FiberStatus::Finished as u32 {
                break;
            }

            if spins < 4000 {
                core::hint::spin_loop();
                spins += 1;
            } else {
                // Host thread has nothing else to do — OS yield is acceptable here.
                // Generation double-check above prevents ABA-induced permanent stalls.
                std::thread::yield_now();
                spins = 2000;
            }
        }
        return;
    }

    // ===== FIBER PATH (called from within a running fiber) =====
    loop {
        // Ensure we are in Running state before checking target.
        // We use swap to safely clear any Notified state from a previous wake event.
        unsafe {
            (*ctx_ptr).state.swap(
                crate::memory_management::FiberStatus::Running as u32,
                core::sync::atomic::Ordering::AcqRel,
            );
        }

        // 0. Check target state and generation
        let (current_gen, status) = unsafe {
            (
                ((*target_ctx)
                    .generation
                    .load(core::sync::atomic::Ordering::Acquire) as u16)
                    & 0x7FFF,
                (*target_ctx)
                    .state
                    .load(core::sync::atomic::Ordering::Acquire),
            )
        };

        if current_gen != handle_gen
            || status == crate::memory_management::FiberStatus::Finished as u32
            || status == crate::memory_management::FiberStatus::Initial as u32
        {
            // Target already finished (or context recycled/freed), break
            break;
        }

        // 1. Register the current fiber as a waiter for the target fiber
        let current_worker = crate::future_bridge::CURRENT_WORKER_ID.with(std::cell::Cell::get);
        let current_ctx_id = unsafe { u64::from((*ctx_ptr).fiber_index) };
        let my_handle = current_ctx_id | ((current_worker as u64) << 32) | (1 << 63);

        // Register the current fiber as a waiter.
        // AcqRel: Release ensures our preceding loads are visible before the swap;
        // Acquire ensures subsequent state re-check sees the fiber's Release stores.
        // AcqRel is sufficient — SeqCst is not needed on any architecture here.
        unsafe {
            (*target_ctx)
                .waiter_handle
                .swap(my_handle, core::sync::atomic::Ordering::AcqRel);
        }

        // 2. Double-check target state after registering waiter
        let (current_gen_post, status_post) = unsafe {
            (
                ((*target_ctx)
                    .generation
                    .load(core::sync::atomic::Ordering::Acquire) as u16)
                    & 0x7FFF,
                (*target_ctx)
                    .state
                    .load(core::sync::atomic::Ordering::Acquire),
            )
        };

        if current_gen_post != handle_gen
            || status_post == crate::memory_management::FiberStatus::Finished as u32
            || status_post == crate::memory_management::FiberStatus::Initial as u32
        {
            // Completed between check and waiter registration
            unsafe {
                let _ = (*target_ctx).waiter_handle.compare_exchange(
                    my_handle,
                    0,
                    core::sync::atomic::Ordering::Relaxed,
                    core::sync::atomic::Ordering::Relaxed,
                );
            }
            break;
        }

        // 3. Try to transition to Suspending and suspend
        unsafe {
            let ctx = &mut *ctx_ptr;
            if ctx
                .state
                .compare_exchange(
                    crate::memory_management::FiberStatus::Running as u32,
                    crate::memory_management::FiberStatus::Suspending as u32,
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
/// * Panics if the runtime is not initialized.
#[unsafe(no_mangle)]
pub extern "C" fn dtact_run(_rt: *mut c_void) {
    let Some(runtime) = crate::GLOBAL_RUNTIME.get() else {
        return;
    };
    let scheduler = &runtime.scheduler;
    let workers_count = scheduler.workers.len();
    let mut handles = alloc::vec::Vec::with_capacity(workers_count);

    for i in 0..workers_count {
        let handle = std::thread::spawn(move || {
            if let Some(runtime) = crate::GLOBAL_RUNTIME.get() {
                crate::dta_scheduler::DtaScheduler::run_worker_static(
                    &runtime.scheduler,
                    i,
                    &runtime.pool,
                    &runtime.shutdown,
                );
            }
        });
        handles.push(handle);
    }

    for h in handles {
        let _ = h.join();
    }
}

/// Signals the cooperative shutdown of all Dtact worker threads.
#[unsafe(no_mangle)]
pub extern "C" fn dtact_shutdown() {
    if let Some(runtime) = crate::GLOBAL_RUNTIME.get() {
        // SeqCst ensures the shutdown flag write is ordered before all subsequent reads.
        runtime
            .shutdown
            .store(true, crate::sync::atomic::Ordering::SeqCst);

        // Bump event_signal to break workers out of WFE/umwait standby.
        // Workers spinning in non-hw-accel mode will see shutdown on their next loop iteration.
        for i in 0..runtime.scheduler.workers.len() {
            let worker = unsafe { &*runtime.scheduler.workers[i].get() };
            worker
                .event_signal
                .fetch_add(1, crate::sync::atomic::Ordering::Release);
            // futex_wake is a no-op if no worker is futex-sleeping (they no longer do),
            // but kept as belt-and-suspenders for any platform-specific edge cases.
            unsafe {
                crate::utils::futex_wake(
                    (&raw const worker.event_signal).cast::<crate::sync::atomic::AtomicU32>(),
                );
            }
        }
    }
}

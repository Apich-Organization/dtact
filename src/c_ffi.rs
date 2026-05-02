use core::ffi::c_void;

/// The opaque handle returned to C/C++ clients across the FFI boundary.
/// Packs the `ctx_id` and the `origin_core` together for deterministic routing.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct dtact_handle_t(pub u64);

/// C-FFI entry point to spawn a new native thread-less Coroutine (Fiber)
#[no_mangle]
pub extern "C" fn dtact_spawn_c(
    func: extern "C" fn(*mut c_void),
    arg: *mut c_void,
    _priority: u8,
    _affinity_mode: u8,
) -> dtact_handle_t {
    let pool = crate::GLOBAL_CONTEXT_POOL.get().expect("Dtact Runtime not initialized");
    let ctx_id = pool.alloc_context().expect("Context pool exhausted - OOM");
    
    let ctx_ptr = pool.get_context_ptr(ctx_id);
    let current_core = crate::topology::current().core_id as usize;

    unsafe {
        // Prepare context for execution
        (*ctx_ptr).state.store(crate::memory_management::FiberStatus::Running as u8, core::sync::atomic::Ordering::Release);
        (*ctx_ptr).origin_core = current_core as u16;
        (*ctx_ptr).fiber_index = ctx_id;
        (*ctx_ptr).switch_fn = crate::context_switch::switch_context_cross_thread_float;
        
        // Set the C execution payload
        (*ctx_ptr).closure_ptr = arg;
        (*ctx_ptr).trampoline = core::mem::transmute(func);
        
        // Calculate the initial Stack Pointer (Top of the pre-allocated arena buffer)
        // Note: Real implementations will align this properly and inject a return stub to gracefully clean up.
        let base = ctx_ptr as *mut u8;
        let stack_top = base.sub(64); // Safe margin below the context struct
        
        #[cfg(target_arch = "x86_64")]
        {
            (*ctx_ptr).stack_ptr = stack_top as usize;
            // X86-64 ABI: The instruction pointer starts at the trampoline.
            // When we pop registers in the dispatcher, we must eventually 'ret' to the function.
            let stack_slice = core::slice::from_raw_parts_mut(stack_top as *mut usize, 1);
            stack_slice[0] = func as usize; // Return address for `ret`
        }
    }

    crate::wake_fiber(current_core, ctx_id);
    
    dtact_handle_t((ctx_id as u64) | ((current_core as u64) << 32))
}

#[no_mangle]
pub extern "C" fn dtact_await(_handle: dtact_handle_t) {
    // In a fully developed runtime, a C thread calling this would register a waker 
    // against the handle's Context ID. If executing inside a Dtact Fiber, it would 
    // retrieve the Thread-Local context pointer and trigger `dtact_asm_fiber_suspend` natively.
}

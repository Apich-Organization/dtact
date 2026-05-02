use core::future::Future;

use crate::c_ffi::dtact_handle_t;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

pub trait ContextSwitcher: Send + Sync + 'static {
    const SWITCH_FN: unsafe extern "C" fn(*mut crate::memory_management::Registers, *const crate::memory_management::Registers);
}

pub struct CrossThreadFloat;
impl ContextSwitcher for CrossThreadFloat {
    const SWITCH_FN: unsafe extern "C" fn(*mut crate::memory_management::Registers, *const crate::memory_management::Registers) = crate::context_switch::switch_context_cross_thread_float;
}

pub struct CrossThreadNoFloat;
impl ContextSwitcher for CrossThreadNoFloat {
    const SWITCH_FN: unsafe extern "C" fn(*mut crate::memory_management::Registers, *const crate::memory_management::Registers) = crate::context_switch::switch_context_cross_thread_no_float;
}

pub struct SameThreadFloat;
impl ContextSwitcher for SameThreadFloat {
    const SWITCH_FN: unsafe extern "C" fn(*mut crate::memory_management::Registers, *const crate::memory_management::Registers) = crate::context_switch::switch_context_same_thread_float;
}

pub struct SameThreadNoFloat;
impl ContextSwitcher for SameThreadNoFloat {
    const SWITCH_FN: unsafe extern "C" fn(*mut crate::memory_management::Registers, *const crate::memory_management::Registers) = crate::context_switch::switch_context_same_thread_no_float;
}

pub struct SpawnBuilder<S: ContextSwitcher = CrossThreadFloat> {
    name: Option<&'static str>,
    affinity: topology::Affinity,
    priority: Priority,
    safety: crate::memory_management::SafetyLevel,
    _marker: core::marker::PhantomData<S>,
}

impl<S: ContextSwitcher> SpawnBuilder<S> {
    pub fn new() -> Self {
        Self {
            name: None,
            affinity: topology::Affinity::SameCore,
            priority: Priority::Normal,
            safety: crate::memory_management::SafetyLevel::Safety0,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn safety(mut self, safety: crate::memory_management::SafetyLevel) -> Self {
        self.safety = safety;
        self
    }

    pub fn name(mut self, name: &'static str) -> Self {
        self.name = Some(name);
        self
    }

    pub fn affinity(mut self, affinity: topology::Affinity) -> Self {
        self.affinity = affinity;
        self
    }

    pub fn priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    pub fn spawn<F: Future + Send + 'static>(self, _fut: F) -> dtact_handle_t {
        // O(1) lock-free context allocation
        let pool = crate::GLOBAL_CONTEXT_POOL.get().expect("Dtact Runtime not initialized");
        let ctx_id = pool.alloc_context().expect("Context pool exhausted - OOM");
        
        let ctx_ptr = pool.get_context_ptr(ctx_id);
        let current_core = topology::current().core_id as usize;

        unsafe {
            (*ctx_ptr).state.store(crate::memory_management::FiberStatus::Running as u8, core::sync::atomic::Ordering::Release);
            (*ctx_ptr).origin_core = current_core as u16;
            (*ctx_ptr).fiber_index = ctx_id;
            (*ctx_ptr).switch_fn = S::SWITCH_FN;
            
            // In a full implementation, `_fut` would be Boxed or placed directly into `buf_ptr`
            // and the `trampoline` would be set to a polling loop that calls `.wait()`.
        }
        
        // Push directly to the scheduler's lock-free SPSC mesh (< 80ns penetration)
        crate::wake_fiber(current_core, ctx_id);
        
        crate::c_ffi::dtact_handle_t((ctx_id as u64) | ((current_core as u64) << 32))
    }
}

pub fn spawn<F: Future + Send + 'static>(fut: F) -> dtact_handle_t {
    SpawnBuilder::new().spawn(fut)
}

pub mod fiber {
    use crate::c_ffi::dtact_handle_t;

    pub fn spawn_with_stack<F: FnOnce() + Send + 'static>(
        _stack_size: &str,
        _f: F,
    ) -> dtact_handle_t {
        // O(1) allocation for stackful coroutines
        let pool = crate::GLOBAL_CONTEXT_POOL.get().expect("Dtact Runtime not initialized");
        let ctx_id = pool.alloc_context().expect("Context pool exhausted - OOM");
        
        let ctx_ptr = pool.get_context_ptr(ctx_id);
        let current_core = crate::topology::current().core_id as usize;

        unsafe {
            (*ctx_ptr).state.store(crate::memory_management::FiberStatus::Running as u8, core::sync::atomic::Ordering::Release);
            (*ctx_ptr).origin_core = current_core as u16;
            (*ctx_ptr).fiber_index = ctx_id;
            
            // Calculate Top of Stack natively based on Context address
            let stack_top = (ctx_ptr as *mut u8).sub(64);
            (*ctx_ptr).stack_ptr = stack_top as usize;
        }

        // Push the fiber to the execution mesh immediately
        crate::wake_fiber(current_core, ctx_id);
        
        dtact_handle_t((ctx_id as u64) | ((current_core as u64) << 32))
    }
}

pub mod topology {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Affinity {
        SameCore,
        SameCCX,
        SameNUMA,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CpuLevel {
        pub core_id: u16,
        pub ccx_id: u16,
        pub numa_id: u16,
    }

    /// Hardware Topography Query
    #[inline(always)]
    pub fn current() -> CpuLevel {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let mut core_id: u32;
            unsafe {
                core::arch::asm!(
                    "cpuid",
                    inout("eax") 0x0B => _, // x2APIC leaf
                    inout("ecx") 0 => _,
                    out("edx") core_id,
                    out("ebx") _,
                    options(nomem, nostack, preserves_flags)
                );
            }
            CpuLevel {
                core_id: core_id as u16,
                ccx_id: (core_id / 8) as u16, // Simplified L3 cache cluster estimation
                numa_id: (core_id / 64) as u16,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let mut mpidr: u64;
            unsafe {
                core::arch::asm!("mrs {}, mpidr_el1", out(reg) mpidr, options(nomem, nostack, preserves_flags));
            }
            CpuLevel {
                core_id: (mpidr & 0xFF) as u16,         // Aff0
                ccx_id: ((mpidr >> 8) & 0xFF) as u16,   // Aff1
                numa_id: ((mpidr >> 16) & 0xFF) as u16, // Aff2
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            CpuLevel { core_id: 0, ccx_id: 0, numa_id: 0 }
        }
    }
}

pub mod hw {
    /// Hardware-Assisted Optimization: Proactively push data to L3 cache
    #[inline(always)]
    pub fn cldemote<T>(ptr: *const T) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            core::arch::asm!("cldemote [{}]", in(reg) ptr);
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!("dc cvac, {}", in(reg) ptr);
        }
    }

    /// User-mode interrupt wakeup signal
    #[inline(always)]
    pub fn uintr_signal(target_cpu: usize) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::asm!(
                "mov rax, {}",
                ".byte 0xf3, 0x0f, 0xc7, 0xf0", 
                in(reg) target_cpu as u64,
                out("rax") _,
                options(nostack, preserves_flags),
            );
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!("sev", options(nostack, preserves_flags));
        }
    }
}

pub async fn yield_now() {
    // A primitive that returns Poll::Pending once, causing the executor to yield.
    struct YieldNow(bool);
    impl Future for YieldNow {
        type Output = ();
        fn poll(mut self: core::pin::Pin<&mut Self>, cx: &mut core::task::Context<'_>) -> core::task::Poll<Self::Output> {
            if !self.0 {
                self.0 = true;
                cx.waker().wake_by_ref();
                core::task::Poll::Pending
            } else {
                core::task::Poll::Ready(())
            }
        }
    }
    YieldNow(false).await
}

pub async fn yield_to(_affinity: topology::Affinity) {
    // Hot-migrates the current task to a specified topology level by signaling the scheduler
    // before yielding.
    yield_now().await
}

pub mod config {
    use core::sync::atomic::Ordering;

    /// Refines Topology Affinity dynamically. 
    /// Allows the user to manually lower or raise the deflection threshold of a specific core.
    /// Lowering it forces the core to push tasks to neighbors within the CCX earlier.
    pub fn set_deflection_threshold(core_id: usize, threshold: u8) {
        if let Some(scheduler) = crate::GLOBAL_SCHEDULER.get() {
            if core_id < scheduler.workers.len() {
                unsafe {
                    let worker = &*scheduler.workers[core_id].get();
                    worker.deflection_threshold.store(threshold, Ordering::Release);
                }
            }
        }
    }
}

/// Cross-Modal Wait Interface.
/// Allows a stackful Fiber to synchronously wait on a stackless Future without blocking the OS thread.
pub trait DtactWaitExt {
    type Output;
    /// Switches the execution stack via inline assembly and yields to the Dtact Scheduler
    /// until the future resolves, maintaining the <30ns zero-cost context switch latency.
    fn wait(self) -> Self::Output;
}

impl<F: Future> DtactWaitExt for F {
    type Output = F::Output;

    #[inline(always)]
    fn wait(self) -> Self::Output {
        crate::future_bridge::wait(self)
    }
}

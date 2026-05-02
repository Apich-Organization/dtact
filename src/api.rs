use core::future::Future;
use crate::memory_management::FiberContext;
use crate::memory_management::{TopologyMode, WorkloadKind};

pub use crate::c_ffi::dtact_handle_t;
pub use topology::Affinity;

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
    kind: WorkloadKind,
    mode: TopologyMode,
    safety: crate::memory_management::SafetyLevel,
    _marker: core::marker::PhantomData<S>,
}

impl<S: ContextSwitcher> SpawnBuilder<S> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            name: None,
            affinity: topology::Affinity::SameCore,
            priority: Priority::Normal,
            kind: WorkloadKind::Compute,
            mode: TopologyMode::P2PMesh,
            safety: crate::memory_management::SafetyLevel::Safety0,
            _marker: core::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub fn kind(mut self, kind: WorkloadKind) -> Self {
        self.kind = kind;
        self
    }

    #[inline(always)]
    pub fn topology_mode(mut self, mode: TopologyMode) -> Self {
        self.mode = mode;
        self
    }

    #[inline(always)]
    pub fn safety(mut self, safety: crate::memory_management::SafetyLevel) -> Self {
        self.safety = safety;
        self
    }

    #[inline(always)]
    pub fn name(mut self, name: &'static str) -> Self {
        self.name = Some(name);
        self
    }

    #[inline(always)]
    pub fn affinity(mut self, affinity: topology::Affinity) -> Self {
        self.affinity = affinity;
        self
    }

    #[inline(always)]
    pub fn priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    #[inline]
    pub fn spawn<F: Future + Send + 'static + core::marker::Unpin>(self, fut: F) -> dtact_handle_t {
        let runtime = crate::GLOBAL_RUNTIME.get().expect("Dtact Runtime not initialized");
        let pool = &runtime.pool;
        let ctx_id = pool.alloc_context().expect("Context pool exhausted - OOM");
        
        let ctx_ptr = pool.get_context_ptr(ctx_id);
        let current_core = topology::current().core_id as usize;

        unsafe {
            (*ctx_ptr).state.store(crate::memory_management::FiberStatus::Running as u8, core::sync::atomic::Ordering::Release);
            (*ctx_ptr).kind = self.kind;
            (*ctx_ptr).mode = self.mode;
            (*ctx_ptr).origin_core = current_core as u16;
            (*ctx_ptr).fiber_index = ctx_id;
            (*ctx_ptr).switch_fn = S::SWITCH_FN;
            
            // 1. Aligned Zero-Copy Future Migration (with Heap Fallback)
            let align = core::mem::align_of::<F>();
            let fut_size = core::mem::size_of::<F>();
            let buffer_start = (*ctx_ptr).read_buffer_ptr as usize;
            let buffer_end = buffer_start + 8192;
            let aligned_fut_addr = (buffer_end - fut_size) & !(align - 1);
            
            // Point 4: Footprint Check (Start and End must be within the 8KB buffer)
            if aligned_fut_addr < buffer_start || (aligned_fut_addr + fut_size) > buffer_end {
                // Future exceeds or misaligns outside the pre-allocated 8KB buffer. 
                // Fallback to heap allocation to maintain stability (slight performance hit).
                crate::HEAP_ESCAPED_SPAWNS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                
                #[cfg(debug_assertions)]
                {
                    static WARNED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
                    if !WARNED.swap(true, core::sync::atomic::Ordering::Relaxed) {
                        eprintln!("DTA-V3 WARNING: Future exceeds or misaligns 8KB zero-copy buffer. Switching to heap-allocation mode.");
                    }
                }
                
                let boxed = Box::new(fut);
                let fut_ptr = Box::into_raw(boxed);
                (*ctx_ptr).closure_ptr = fut_ptr as *mut ();
                (*ctx_ptr).invoke_closure = |ptr| {
                    unsafe {
                        let mut f = Box::from_raw(ptr as *mut F);
                        crate::future_bridge::wait(&mut *f);
                    }
                };
                (*ctx_ptr).cleanup_fn = None; 
            } else {
                let fut_ptr = aligned_fut_addr as *mut F;
                core::ptr::write(fut_ptr, fut);
                
                (*ctx_ptr).invoke_closure = |ptr| {
                    let f_ptr = ptr as *mut F;
                    unsafe {
                        crate::future_bridge::wait(&mut *f_ptr);
                        core::ptr::drop_in_place(f_ptr);
                    }
                };
                (*ctx_ptr).closure_ptr = fut_ptr as *mut ();
            }
            
            // 3. Windows ABI Compliance (Shadow Space) & Stack Alignment
            // x86_64 ABI Rule: RSP must be 16-byte aligned BEFORE a CALL.
            // Point 1: Shadow Space Separation (Stack MUST start BELOW the 8KB Future buffer)
            let mut stack_top = (buffer_start as usize & !0xF) - 64;
            let stack_top_ptr = stack_top as *mut u64;
            
            // Point 4: "Return-to-Nowhere" Protection
            // Place a poison return address on the stack.
            core::ptr::write(stack_top_ptr, crate::c_ffi::dtact_abort as u64);
            
            let stack_top = stack_top as *mut u8;

            #[cfg(target_arch = "x86_64")]
            {
                (*ctx_ptr).regs.gprs[0] = stack_top as u64; // RSP
                (*ctx_ptr).regs.gprs[7] = fiber_entry_point as u64; // RIP
            }
            #[cfg(target_arch = "aarch64")]
            {
                (*ctx_ptr).regs.gprs[12] = stack_top as u64; // SP
                (*ctx_ptr).regs.gprs[11] = fiber_entry_point as u64; // x30 (LR)
            }
            #[cfg(target_arch = "riscv64")]
            {
                (*ctx_ptr).regs.gprs[0] = stack_top as u64; // SP
                (*ctx_ptr).regs.gprs[13] = fiber_entry_point as u64; // RA
            }
        }

        crate::wake_fiber(current_core, ctx_id);
        dtact_handle_t((ctx_id as u64) | ((current_core as u64) << 32))
    }
}

pub(crate) unsafe extern "C" fn fiber_entry_point() {
    let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(|c| c.get());
    if ctx_ptr.is_null() { return; }
    
    let ctx = unsafe { &mut *ctx_ptr };
    let invoke = ctx.invoke_closure;
    let arg = ctx.closure_ptr;
    
    // Execute the task payload with SEH/Panic protection
    let _ = std::panic::catch_unwind(core::panic::AssertUnwindSafe(move || {
        unsafe { invoke(arg) };
    }));
    
    // Mark context as free for the lock-free pool
    ctx.state.store(crate::memory_management::FiberStatus::Initial as u8, core::sync::atomic::Ordering::Release);
    
    // Execute cleanup if present (e.g. FFI arg free)
    if let Some(cleanup) = ctx.cleanup_fn.take() {
        unsafe { cleanup(ctx.closure_ptr) };
    }
    
    unsafe {
        ((*ctx).switch_fn)(
            &mut (*ctx).regs,
            &(*ctx).executor_regs,
        );
    }
}

pub static TOPOLOGY_EPOCH: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

pub mod topology {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Affinity {
        SameCore,
        SameCCX,
        SameNUMA,
        Any,
    }

    #[inline(always)]
    pub fn current_core() -> u16 {
        current().core_id
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CpuLevel {
        pub core_id: u16,
        pub ccx_id: u16,
        pub numa_id: u16,
    }

    #[inline(always)]
    pub fn current() -> CpuLevel {
        thread_local! {
            static CACHED: core::cell::Cell<(CpuLevel, u64)> = const { 
                core::cell::Cell::new((CpuLevel { core_id: 0, ccx_id: 0, numa_id: 0 }, 0)) 
            };
        }
        
        let (mut cpu, mut last_refresh) = CACHED.with(|c| c.get());
        let (now, cpu_id) = crate::utils::get_tick_with_cpu();
        
        // Refresh every 100k cycles OR if Core ID mismatch (vCPU migration)
        if now.wrapping_sub(last_refresh) > 100_000 || (cpu.core_id as u32) != cpu_id {
            let next_cpu = current_raw();
            if next_cpu != cpu {
                crate::TOPOLOGY_EPOCH.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                cpu = next_cpu;
            }
            last_refresh = now;
            CACHED.with(|c| c.set((cpu, last_refresh)));
        }
        cpu
    }

    #[inline(always)]
    pub fn current_raw() -> CpuLevel {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let (mut x2apic_id, mut core_shift, mut package_shift): (u32, u32, u32);
            
            unsafe {
                let (mut eax, mut _ebx, mut _ecx, mut edx): (u32, u32, u32, u32);
                core::arch::asm!(
                    "push rbx",
                    "cpuid",
                    "mov {ebx_out:e}, rbx",
                    "pop rbx",
                    ebx_out = out(reg) _ebx,
                    inout("eax") 0x0B => eax,
                    inout("ecx") 0 => _ecx,
                    out("edx") edx,
                );
                core_shift = eax;
                x2apic_id = edx;
                
                let (mut eax_p, mut _ebx_p, mut _ecx_p, mut _edx_p): (u32, u32, u32, u32);
                core::arch::asm!(
                    "push rbx",
                    "cpuid",
                    "mov {ebx_out:e}, rbx",
                    "pop rbx",
                    ebx_out = out(reg) _ebx_p,
                    inout("eax") 0x0B => eax_p,
                    inout("ecx") 1 => _ecx_p,
                    out("edx") _edx_p,
                );
                package_shift = eax_p;
            }
            
            let core_id = x2apic_id & ((1 << core_shift) - 1);
            let ccx_id = (x2apic_id >> core_shift) & ((1 << (package_shift - core_shift)) - 1);
            let numa_id = x2apic_id >> package_shift;

            return CpuLevel {
                core_id: core_id as u16,
                ccx_id: ccx_id as u16,
                numa_id: numa_id as u16,
            };
        }

        #[cfg(target_arch = "aarch64")]
        {
            let mut mpidr: u64;
            unsafe {
                core::arch::asm!("mrs {}, mpidr_el1", out(reg) mpidr, options(nomem, nostack, preserves_flags));
            }
            return CpuLevel {
                core_id: (mpidr & 0xFF) as u16,         
                ccx_id: ((mpidr >> 8) & 0xFF) as u16,   
                numa_id: ((mpidr >> 16) & 0xFF) as u16, 
            };
        }

        #[cfg(target_arch = "riscv64")]
        {
            let mut hart_id: u64;
            unsafe {
                core::arch::asm!("csrr {}, mhartid", out(reg) hart_id, options(nomem, nostack, preserves_flags));
            }
            return CpuLevel {
                core_id: (hart_id & 0xFFFF) as u16,
                ccx_id: (hart_id >> 16) as u16,
                numa_id: 0,
            };
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64")))]
        {
            CpuLevel { core_id: 0, ccx_id: 0, numa_id: 0 }
        }
    }
}

#[inline(always)]
pub fn spawn<F: Future + Send + 'static + core::marker::Unpin>(fut: F) -> dtact_handle_t {
    SpawnBuilder::<CrossThreadFloat>::new().spawn(fut)
}

pub mod spawn {
    use super::*;
    #[inline(always)]
    pub fn builder() -> SpawnBuilder<CrossThreadFloat> {
        SpawnBuilder::new()
    }
}

pub mod fiber {
    use super::*;
    #[inline]
    pub fn spawn_with_stack<F: FnOnce() + Send + 'static>(_stack_size_str: &str, f: F) -> dtact_handle_t {
        let runtime = crate::GLOBAL_RUNTIME.get().expect("Dtact Runtime not initialized");
        let pool = &runtime.pool;
        let ctx_id = pool.alloc_context().expect("Context pool exhausted - OOM");
        let ctx_ptr = pool.get_context_ptr(ctx_id);
        let current_core = topology::current().core_id as usize;

        unsafe {
            (*ctx_ptr).state.store(crate::memory_management::FiberStatus::Running as u8, core::sync::atomic::Ordering::Release);
            (*ctx_ptr).origin_core = current_core as u16;
            (*ctx_ptr).fiber_index = ctx_id;
            (*ctx_ptr).switch_fn = crate::context_switch::switch_context_same_thread_no_float;
            
            let f_ptr = (*ctx_ptr).read_buffer_ptr as *mut F;
            core::ptr::write(f_ptr, f);
            (*ctx_ptr).invoke_closure = |ptr| {
                let f = core::ptr::read(ptr as *mut F);
                f();
            };
            (*ctx_ptr).closure_ptr = f_ptr as *mut ();
            
            // Point 1: Shadow Space Separation (Stack MUST start BELOW the 8KB Future buffer)
            let buffer_start = (*ctx_ptr).read_buffer_ptr as usize;
            let mut stack_top = (buffer_start as usize & !0xF) - 64;
            let stack_top_ptr = stack_top as *mut u64;
            
            // Point 4: "Return-to-Nowhere" Protection
            core::ptr::write(stack_top_ptr, crate::c_ffi::dtact_abort as u64);
            
            let stack_top = stack_top as *mut u8;
            
            #[cfg(target_arch = "x86_64")]
            {
                (*ctx_ptr).regs.gprs[0] = stack_top as u64; // RSP
                (*ctx_ptr).regs.gprs[7] = super::fiber_entry_point as u64; // RIP
            }
            #[cfg(target_arch = "aarch64")]
            {
                (*ctx_ptr).regs.gprs[12] = stack_top as u64; // SP
                (*ctx_ptr).regs.gprs[11] = super::fiber_entry_point as u64; // x30 (LR)
            }
            #[cfg(target_arch = "riscv64")]
            {
                (*ctx_ptr).regs.gprs[0] = stack_top as u64; // SP
                (*ctx_ptr).regs.gprs[13] = super::fiber_entry_point as u64; // RA
            }
        }

        crate::wake_fiber(current_core, ctx_id);
        dtact_handle_t((ctx_id as u64) | ((current_core as u64) << 32))
    }

    /// Yields execution directly to another fiber.
    /// Note: This is a hint to the scheduler.
    #[inline(always)]
    pub fn yield_to(handle: dtact_handle_t) {
        let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(|c| c.get());
        if ctx_ptr.is_null() { return; } 
        
        let target_ctx_id = (handle.0 & 0xFFFFFFFF) as u32;
        let target_core_id = (handle.0 >> 32) as usize;
        
        crate::wake_fiber(target_core_id, target_ctx_id);
        
        unsafe {
            let ctx = &mut *ctx_ptr;
            ctx.state.store(crate::memory_management::FiberStatus::Yielded as u8, core::sync::atomic::Ordering::Release);
            (ctx.switch_fn)(&mut ctx.regs, &ctx.executor_regs);
        }
    }
}

#[cfg(feature = "hw-acceleration")]
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
        #[cfg(target_arch = "riscv64")]
        unsafe {
            core::arch::asm!("cbo.clean 0({0})", in(reg) ptr);
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
        #[cfg(target_arch = "riscv64")]
        unsafe {
            core::arch::asm!("csrw uipi, {0}", in(reg) target_cpu);
        }
    }
}

pub async fn yield_now() {
    struct YieldNow(bool);
    impl Future for YieldNow {
        type Output = ();
        #[inline(always)]
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

/// Yields execution to another fiber handle asynchronously.
#[inline(always)]
pub async fn yield_to(handle: dtact_handle_t) {
    let target_ctx_id = (handle.0 & 0xFFFFFFFF) as u32;
    let target_core_id = (handle.0 >> 32) as usize;
    crate::wake_fiber(target_core_id, target_ctx_id);
    yield_now().await;
}

pub mod config {
    use core::sync::atomic::Ordering;
    #[inline(always)]
    pub fn set_deflection_threshold(core_id: usize, threshold: u8) {
        if let Some(runtime) = crate::GLOBAL_RUNTIME.get() {
            if core_id < runtime.scheduler.workers.len() {
                unsafe {
                    let worker = &*runtime.scheduler.workers[core_id].get();
                    worker.deflection_threshold.store(threshold, Ordering::Release);
                }
            }
        }
    }
}

pub trait DtactWaitExt {
    type Output;
    #[inline(always)]
    fn wait(self) -> Self::Output;
}

impl<F: Future> DtactWaitExt for F {
    type Output = F::Output;
    #[inline(always)]
    fn wait(self) -> Self::Output {
        crate::future_bridge::wait(self)
    }
}

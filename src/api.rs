pub use crate::c_ffi::dtact_handle_t;
pub use crate::common_types::{TopologyMode, WorkloadKind};
pub use crate::memory_management::{ContextPool, FiberContext, FiberStatus, SafetyLevel};
use core::future::Future;
use core::pin::Pin;
pub use topology::Affinity;

/// Scheduling Priority for fibers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    /// Background tasks with no latency requirements.
    Low,
    /// Standard application tasks.
    Normal,
    /// Latency-sensitive tasks that should preempt normal work.
    High,
    /// Critical real-time tasks that must run as soon as possible.
    Critical,
}

/// Interface for custom context switching logic.
pub trait ContextSwitcher: Send + Sync + 'static {
    /// The raw assembly function used for switching to/from this fiber.
    const SWITCH_FN: unsafe extern "C" fn(
        *mut crate::memory_management::Registers,
        *const crate::memory_management::Registers,
    );
}

/// Standard switcher that saves/restores floating-point state and supports cross-thread migration.
pub struct CrossThreadFloat;
impl ContextSwitcher for CrossThreadFloat {
    const SWITCH_FN: unsafe extern "C" fn(
        *mut crate::memory_management::Registers,
        *const crate::memory_management::Registers,
    ) = crate::context_switch::switch_context_cross_thread_float;
}

/// Lightweight switcher that skips floating-point state but supports cross-thread migration.
pub struct CrossThreadNoFloat;
impl ContextSwitcher for CrossThreadNoFloat {
    const SWITCH_FN: unsafe extern "C" fn(
        *mut crate::memory_management::Registers,
        *const crate::memory_management::Registers,
    ) = crate::context_switch::switch_context_cross_thread_no_float;
}

/// Optimized switcher for fibers pinned to a single thread, saving/restoring floating-point state.
pub struct SameThreadFloat;
impl ContextSwitcher for SameThreadFloat {
    const SWITCH_FN: unsafe extern "C" fn(
        *mut crate::memory_management::Registers,
        *const crate::memory_management::Registers,
    ) = crate::context_switch::switch_context_same_thread_float;
}

/// The fastest possible switcher: pins to one thread and ignores floating-point state.
pub struct SameThreadNoFloat;
impl ContextSwitcher for SameThreadNoFloat {
    const SWITCH_FN: unsafe extern "C" fn(
        *mut crate::memory_management::Registers,
        *const crate::memory_management::Registers,
    ) = crate::context_switch::switch_context_same_thread_no_float;
}

/// Fluent builder for configuring and launching fibers.
pub struct SpawnBuilder<S: ContextSwitcher = CrossThreadFloat> {
    name: Option<&'static str>,
    affinity: topology::Affinity,
    priority: Priority,
    kind: WorkloadKind,
    mode: TopologyMode,
    safety: crate::memory_management::SafetyLevel,
    _marker: core::marker::PhantomData<S>,
}

impl<S: ContextSwitcher> Default for SpawnBuilder<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<S: ContextSwitcher> SpawnBuilder<S> {
    /// Creates a new builder with default settings:
    /// Normal priority, Compute kind, P2P Mesh mode, and Safety0 (raw performance).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
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

    /// Sets the workload kind (Compute or IO).
    #[inline(always)]
    #[must_use]
    pub const fn kind(mut self, kind: WorkloadKind) -> Self {
        self.kind = kind;
        self
    }

    /// Sets the topology mode (P2P Mesh or Local Queue).
    #[inline(always)]
    #[must_use]
    pub const fn topology_mode(mut self, mode: TopologyMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the hardware safety level (0-2).
    #[inline(always)]
    #[must_use]
    pub const fn safety(mut self, safety: crate::memory_management::SafetyLevel) -> Self {
        self.safety = safety;
        self
    }

    /// Sets a descriptive name for the fiber (useful for telemetry).
    #[inline(always)]
    #[must_use]
    pub const fn name(mut self, name: &'static str) -> Self {
        self.name = Some(name);
        self
    }

    /// Sets the core affinity (`SameCore`, `SameNUMA`, etc.).
    #[inline(always)]
    #[must_use]
    pub const fn affinity(mut self, affinity: topology::Affinity) -> Self {
        self.affinity = affinity;
        self
    }

    /// Sets the scheduling priority.
    #[inline(always)]
    #[must_use]
    pub const fn priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Switches the context-switching strategy (e.g. `SameThreadNoFloat`).
    #[inline(always)]
    #[must_use]
    pub const fn switcher<NewS: ContextSwitcher>(self) -> SpawnBuilder<NewS> {
        SpawnBuilder {
            name: self.name,
            affinity: self.affinity,
            priority: self.priority,
            kind: self.kind,
            mode: self.mode,
            safety: self.safety,
            _marker: core::marker::PhantomData,
        }
    }

    /// Finalizes and launches the fiber into the runtime.
    ///
    /// This performs the critical "Zero-Copy" layout calculation:
    /// 1. Attempts to place the Future directly at the top of the fiber stack.
    /// 2. If the Future is too large (>8KB), falls back to heap allocation.
    /// 3. Configures the assembly trampoline for the selected `ContextSwitcher`.
    ///
    /// # Panics
    /// * Panics if the runtime is not initialized.
    /// * Panics if the context pool is exhausted.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::useless_let_if_seq)]
    pub fn spawn<F: Future + Send + 'static>(self, fut: F) -> dtact_handle_t {
        let runtime = crate::GLOBAL_RUNTIME
            .get()
            .expect("Dtact Runtime not initialized");
        let pool = &runtime.pool;
        let ctx_id = pool.alloc_context().expect("Context pool exhausted - OOM");

        let ctx_ptr = pool.get_context_ptr(ctx_id);
        #[allow(clippy::cast_possible_truncation)]
        let current_core = topology::current().core_id as usize % runtime.scheduler.workers.len();

        unsafe {
            (*ctx_ptr).state.store(
                crate::memory_management::FiberStatus::Running as u8,
                core::sync::atomic::Ordering::Release,
            );
            (*ctx_ptr).kind = self.kind;
            (*ctx_ptr).mode = self.mode;
            (*ctx_ptr).origin_core = current_core as u16;
            (*ctx_ptr).fiber_index = ctx_id;
            (*ctx_ptr).switch_fn = S::SWITCH_FN;

            // Set adaptive spin count based on workload kind
            (*ctx_ptr).adaptive_spin_count = match self.kind {
                WorkloadKind::Compute => 1000,
                WorkloadKind::IO => 100,
                WorkloadKind::Memory => 500,
                WorkloadKind::System => 200,
            };

            // Aligned Zero-Copy Future Migration
            let align = core::mem::align_of::<F>();
            let fut_size = core::mem::size_of::<F>();
            let buffer_start = (*ctx_ptr).read_buffer_ptr as usize;
            let buffer_end = buffer_start + 8192;
            let aligned_fut_addr = (buffer_end - fut_size) & !(align - 1);

            // Determine where the stack region ends (just below the future).
            // The stack grows DOWNWARD from this address toward buffer_start.
            let stack_limit: usize;

            if aligned_fut_addr < buffer_start || (aligned_fut_addr + fut_size) > buffer_end {
                // Future exceeds pre-allocated 8KB buffer. Fallback to heap.
                crate::HEAP_ESCAPED_SPAWNS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

                #[cfg(debug_assertions)]
                {
                    static WARNED: core::sync::atomic::AtomicBool =
                        core::sync::atomic::AtomicBool::new(false);
                    if !WARNED.swap(true, core::sync::atomic::Ordering::Relaxed) {
                        eprintln!(
                            "DTA-V3 WARNING: Future exceeds or misaligns 8KB zero-copy buffer. Switching to heap-allocation mode."
                        );
                    }
                }

                let boxed = Box::new(fut);
                let fut_ptr = Box::into_raw(boxed);
                (*ctx_ptr).closure_ptr = fut_ptr.cast::<()>();
                (*ctx_ptr).invoke_closure = |ptr| unsafe {
                    let mut f = Box::from_raw(ptr.cast::<F>());
                    let f_pinned = Pin::new_unchecked(&mut *f);
                    crate::future_bridge::wait_pinned(f_pinned);
                };
                (*ctx_ptr).cleanup_fn = None;

                // Heap path: entire 8KB buffer is available as stack
                stack_limit = buffer_end;
            } else {
                let fut_ptr = aligned_fut_addr as *mut F;
                core::ptr::write(fut_ptr, fut);

                (*ctx_ptr).invoke_closure = |ptr| {
                    let f_ptr = ptr.cast::<F>();
                    unsafe {
                        let f_pinned = Pin::new_unchecked(&mut *f_ptr);
                        crate::future_bridge::wait_pinned(f_pinned);
                        core::ptr::drop_in_place(f_ptr);
                    }
                };
                (*ctx_ptr).closure_ptr = fut_ptr.cast::<()>();

                // Inline path: stack lives below the future
                stack_limit = aligned_fut_addr;
            }

            // Stack setup: RSP starts at the top of the available stack region.
            // 16-byte aligned, minus 8 for the poison return address (to satisfy
            // x86_64 ABI requirement that RSP is 16n+8 at function entry after call).
            let stack_top = (stack_limit & !0xF) - 8;
            let stack_top_ptr = stack_top as *mut u64;

            // Poison return address (dtact_abort) — if fiber_entry_point ever returns,
            // this triggers a controlled abort instead of undefined behavior.
            core::ptr::write(stack_top_ptr, crate::c_ffi::dtact_abort as *const () as u64);

            let stack_top = stack_top as *mut u8;

            #[cfg(target_arch = "x86_64")]
            {
                (*ctx_ptr).regs.gprs[0] = stack_top as u64; // RSP
                (*ctx_ptr).regs.gprs[7] = fiber_entry_point as *const () as u64; // RIP
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
        dtact_handle_t(u64::from(ctx_id) | ((current_core as u64) << 32))
    }
}

pub(crate) unsafe extern "C" fn fiber_entry_point() {
    let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(std::cell::Cell::get);
    if ctx_ptr.is_null() {
        return;
    }

    let ctx = unsafe { &mut *ctx_ptr };
    let invoke = ctx.invoke_closure;
    let arg = ctx.closure_ptr;

    // Execute the task payload with SEH/Panic protection
    let _ = std::panic::catch_unwind(core::panic::AssertUnwindSafe(move || {
        unsafe { invoke(arg) };
    }));

    // Return context to pool (marks as Initial, wakes futex waiters, returns to free list)
    let ctx_id = ctx.fiber_index;
    if let Some(runtime) = crate::GLOBAL_RUNTIME.get() {
        runtime.pool.free_context(ctx_id);
    }

    // Wake up any fiber waiting for this one (FFI join)
    let waiter = ctx
        .waiter_handle
        .swap(0, core::sync::atomic::Ordering::SeqCst);
    if waiter != 0 {
        let waiter_ctx_id = (waiter & 0xFFFF_FFFF) as u32;
        let waiter_core_id = (waiter >> 32) as usize;
        crate::wake_fiber(waiter_core_id, waiter_ctx_id);
    }

    // Execute cleanup if present (e.g. FFI arg free)
    if let Some(cleanup) = ctx.cleanup_fn.take() {
        unsafe { cleanup(ctx.closure_ptr) };
    }

    unsafe {
        (ctx.switch_fn)(&raw mut ctx.regs, &raw const ctx.executor_regs);
    }
}

/// Global epoch counter for hardware topology changes.
/// Incremented whenever a thread migration across CCX/NUMA boundaries is detected.
pub static TOPOLOGY_EPOCH: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Hardware Topology Discovery and Affinity Management.
pub mod topology {
    /// Resumption affinity hints for the P2P Mesh scheduler.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Affinity {
        /// Resume on the same physical CPU core.
        SameCore,
        /// Resume on any core within the same Core Complex (CCX).
        SameCCX,
        /// Resume on any core within the same NUMA node.
        SameNUMA,
        /// No affinity preference.
        Any,
    }

    /// Returns the Core ID of the currently executing hardware thread.
    #[inline(always)]
    #[must_use]
    pub fn current_core() -> u16 {
        current().core_id
    }

    /// Hierarchical representation of a CPU core's location.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CpuLevel {
        /// Logical Core ID.
        pub core_id: u16,
        /// Core Complex (L3 boundary) ID.
        pub ccx_id: u16,
        /// Non-Uniform Memory Access (NUMA) node ID.
        pub numa_id: u16,
    }

    /// Returns the hierarchical topology information for the current core.
    ///
    /// This function utilizes thread-local caching and adaptive refresh
    /// intervals to minimize the overhead of hardware discovery (e.g., CPUID).
    #[inline(always)]
    pub fn current() -> CpuLevel {
        thread_local! {
            static CACHED: core::cell::Cell<(CpuLevel, u64)> = const {
                core::cell::Cell::new((CpuLevel { core_id: 0, ccx_id: 0, numa_id: 0 }, 0))
            };
        }

        let (mut cpu, mut last_refresh) = CACHED.with(std::cell::Cell::get);
        let (now, cpu_id) = crate::utils::get_tick_with_cpu();

        // Refresh every 100k cycles OR if Core ID mismatch (vCPU migration)
        if now.wrapping_sub(last_refresh) > 100_000 || u32::from(cpu.core_id) != cpu_id {
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

    /// Performs a raw hardware topology discovery via CPUID/MPIDR.
    #[inline(always)]
    #[must_use]
    pub fn current_raw() -> CpuLevel {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let (x2apic_id, core_shift, package_shift): (u32, u32, u32);

            unsafe {
                let (mut eax, mut edx_v): (u32, u32);
                core::arch::asm!(
                    "push rbx",
                    "cpuid",
                    "mov {ebx_out:e}, ebx",
                    "pop rbx",
                    ebx_out = out(reg) _,
                    inout("eax") 0x0B => eax,
                    inout("ecx") 0 => _,
                    out("edx") edx_v,
                );
                core_shift = eax;
                x2apic_id = edx_v;

                let eax_p: u32;
                core::arch::asm!(
                    "push rbx",
                    "cpuid",
                    "mov {ebx_out:e}, ebx",
                    "pop rbx",
                    ebx_out = out(reg) _,
                    inout("eax") 0x0B => eax_p,
                    inout("ecx") 1 => _,
                    out("edx") _,
                );
                package_shift = eax_p;
            }

            let core_id = x2apic_id & ((1 << core_shift) - 1);
            let ccx_id = (x2apic_id >> core_shift) & ((1 << (package_shift - core_shift)) - 1);
            let numa_id = x2apic_id >> package_shift;

            CpuLevel {
                core_id: (core_id & 0xFFFF) as u16,
                ccx_id: (ccx_id & 0xFFFF) as u16,
                numa_id: (numa_id & 0xFFFF) as u16,
            }
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

        #[cfg(not(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )))]
        {
            CpuLevel {
                core_id: 0,
                ccx_id: 0,
                numa_id: 0,
            }
        }
    }
}

/// Spawns a new fiber and returns a handle for synchronization.
#[inline(always)]
pub fn spawn<F: Future + Send + 'static>(fut: F) -> dtact_handle_t {
    SpawnBuilder::<CrossThreadFloat>::new().spawn(fut)
}

/// Returns a new `SpawnBuilder` for configuring a fiber.
#[inline(always)]
#[must_use]
pub const fn spawn_with() -> SpawnBuilder<CrossThreadFloat> {
    SpawnBuilder::new()
}

/// Fiber configuration and construction utilities.
#[doc(hidden)]
pub mod spawn {
    use super::{CrossThreadFloat, SpawnBuilder};
    /// Returns a new `SpawnBuilder` with default settings.
    #[inline(always)]
    #[must_use]
    #[doc(hidden)]
    pub const fn builder() -> SpawnBuilder<CrossThreadFloat> {
        SpawnBuilder::new()
    }
}

/// Fiber-local execution and synchronization utilities.
pub mod fiber {
    use super::{dtact_handle_t, topology};
    /// Spawns a fiber from a closure with a specific stack configuration.
    ///
    /// # Panics
    /// * Panics if the runtime is not initialized.
    /// * Panics if the context pool is exhausted.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub fn spawn_with_stack<F: FnOnce() + Send + 'static>(
        _stack_size_str: &str,
        f: F,
    ) -> dtact_handle_t {
        let runtime = crate::GLOBAL_RUNTIME
            .get()
            .expect("Dtact Runtime not initialized");
        let pool = &runtime.pool;
        let ctx_id = pool.alloc_context().expect("Context pool exhausted - OOM");
        let ctx_ptr = pool.get_context_ptr(ctx_id);
        #[allow(clippy::cast_possible_truncation)]
        let current_core = topology::current().core_id as usize;

        unsafe {
            (*ctx_ptr).state.store(
                crate::memory_management::FiberStatus::Running as u8,
                core::sync::atomic::Ordering::Release,
            );
            (*ctx_ptr).origin_core = current_core as u16;
            (*ctx_ptr).fiber_index = ctx_id;
            (*ctx_ptr).switch_fn = crate::context_switch::switch_context_same_thread_no_float;

            let f_ptr = (*ctx_ptr).read_buffer_ptr.cast::<F>();
            core::ptr::write(f_ptr, f);
            (*ctx_ptr).invoke_closure = |ptr| {
                let f = core::ptr::read(ptr.cast::<F>());
                f();
            };
            (*ctx_ptr).closure_ptr = f_ptr.cast::<()>();

            // Point 1: Shadow Space Separation (Stack MUST start BELOW the 8KB Future buffer)
            let buffer_start = (*ctx_ptr).read_buffer_ptr as usize;
            let stack_top = (buffer_start & !0xF) - 64;
            let stack_top_ptr = stack_top as *mut u64;

            // Point 4: "Return-to-Nowhere" Protection
            core::ptr::write(stack_top_ptr, crate::c_ffi::dtact_abort as *const () as u64);

            let stack_top = stack_top as *mut u8;

            #[cfg(target_arch = "x86_64")]
            {
                (*ctx_ptr).regs.gprs[0] = stack_top as u64; // RSP
                (*ctx_ptr).regs.gprs[7] = super::fiber_entry_point as *const () as u64; // RIP
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
        dtact_handle_t(u64::from(ctx_id) | ((current_core as u64) << 32))
    }

    /// Yields execution directly to another fiber.
    /// Note: This is a hint to the scheduler.
    #[inline(always)]
    pub fn yield_to(handle: dtact_handle_t) {
        let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(std::cell::Cell::get);
        if ctx_ptr.is_null() {
            return;
        }

        let target_ctx_id = (handle.0 & 0xFFFF_FFFF) as u32;
        let target_core_id = (handle.0 >> 32) as usize;

        crate::wake_fiber(target_core_id, target_ctx_id);

        unsafe {
            let ctx = &mut *ctx_ptr;
            ctx.state.store(
                crate::memory_management::FiberStatus::Yielded as u8,
                core::sync::atomic::Ordering::Release,
            );
            (ctx.switch_fn)(&raw mut ctx.regs, &raw const ctx.executor_regs);
        }
    }
}

/// Advanced Hardware Acceleration primitives.
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

/// Yields execution to the scheduler.
pub async fn yield_now() {
    struct YieldNow(bool);
    impl Future for YieldNow {
        type Output = ();
        #[inline(always)]
        fn poll(
            mut self: core::pin::Pin<&mut Self>,
            cx: &mut core::task::Context<'_>,
        ) -> core::task::Poll<Self::Output> {
            if self.0 {
                core::task::Poll::Ready(())
            } else {
                self.0 = true;
                cx.waker().wake_by_ref();
                core::task::Poll::Pending
            }
        }
    }
    YieldNow(false).await;
}

/// Yields execution to another fiber handle asynchronously.
#[inline(always)]
pub async fn yield_to(handle: dtact_handle_t) {
    let target_ctx_id = (handle.0 & 0xFFFF_FFFF) as u32;
    let target_core_id = (handle.0 >> 32) as usize;
    crate::wake_fiber(target_core_id, target_ctx_id);
    yield_now().await;
}

/// Global Runtime Configuration and Telemetry.
pub mod config {
    use core::sync::atomic::Ordering;
    /// Sets the work-deflection threshold for a specific hardware worker.
    #[inline(always)]
    pub fn set_deflection_threshold(core_id: usize, threshold: u8) {
        if let Some(runtime) = crate::GLOBAL_RUNTIME.get()
            && core_id < runtime.scheduler.workers.len()
        {
            unsafe {
                let worker = &*runtime.scheduler.workers[core_id].get();
                worker
                    .deflection_threshold
                    .store(threshold, Ordering::Release);
            }
        }
    }
}

/// Extension trait for blocking on asynchronous futures from within a fiber.
pub trait DtactWaitExt {
    /// The type of value produced by the future.
    type Output;
    /// Blocks the current fiber until the future resolves.

    fn wait(self) -> Self::Output;
}

impl<F: Future> DtactWaitExt for F {
    type Output = F::Output;
    #[inline(always)]
    fn wait(self) -> Self::Output {
        crate::future_bridge::wait(self)
    }
}

pub use crate::c_ffi::dtact_handle_t;
pub use crate::common_types::{TopologyMode, WorkloadKind};
pub use crate::memory_management::{
    ContextPool, FiberContext, FiberStatus, SafetyLevel, TaskOutcome,
};
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

impl Priority {
    /// Packed discriminant, used for telemetry and for the C-FFI round-trip.
    #[inline(always)]
    #[must_use]
    pub const fn level(self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
            Self::Critical => 3,
        }
    }

    /// Inverse of [`Self::level`]; anything out of range maps to `Normal`.
    #[inline(always)]
    #[must_use]
    pub const fn from_level(level: u8) -> Self {
        match level {
            0 => Self::Low,
            2 => Self::High,
            3 => Self::Critical,
            _ => Self::Normal,
        }
    }

    /// Signed bias added to a worker's adaptive `deflection_threshold`
    /// before the load comparison in `DtaScheduler::enqueue_deflect`.
    ///
    /// Negative bias = deflect sooner (the fiber is cheap to move and not
    /// latency-sensitive); positive bias = deflect later. `Normal` is
    /// exactly `0`, so the default spawn path computes the identical
    /// comparison it always did.
    #[inline(always)]
    pub(crate) const fn deflection_bias(self) -> i8 {
        match self {
            Self::Low => -32,
            Self::High => 24,
            // `Normal` must reproduce the pre-priority comparison exactly,
            // and `Critical` never reaches the threshold comparison at all
            // (see `may_deflect`) — so both are a zero bias.
            Self::Normal | Self::Critical => 0,
        }
    }

    /// `false` for [`Self::Critical`], which is routed as pinned: it is
    /// never handed to another worker, so it can neither be delayed by a
    /// remote queue nor migrated away from its cache-warm core.
    #[inline(always)]
    pub(crate) const fn may_deflect(self) -> bool {
        !matches!(self, Self::Critical)
    }

    /// `true` when a fiber of this priority is inserted at the *front* of
    /// the local queue instead of the back.
    #[inline(always)]
    pub(crate) const fn jumps_queue(self) -> bool {
        matches!(self, Self::Critical)
    }
}

/// Interface for custom context switching logic.
///
/// `ALLOW_DEFLECTION` reports whether fibers using this switcher may be
/// migrated across worker threads by the scheduler. `SameThread` variants set
/// this to `false` because their assembly switch routines do not preserve
/// per-thread state (TLS, TIB, FS/GS); deflecting such a fiber would
/// silently corrupt it.
pub trait ContextSwitcher: Send + Sync + 'static {
    /// The raw assembly function used for switching to/from this fiber.
    const SWITCH_FN: unsafe extern "C" fn(
        *mut crate::memory_management::Registers,
        *const crate::memory_management::Registers,
    );
    /// `true` if the scheduler is allowed to deflect/migrate this fiber across cores.
    const ALLOW_DEFLECTION: bool;
}

/// Standard switcher that saves/restores floating-point state and supports cross-thread migration.
pub struct CrossThreadFloat;
impl ContextSwitcher for CrossThreadFloat {
    const SWITCH_FN: unsafe extern "C" fn(
        *mut crate::memory_management::Registers,
        *const crate::memory_management::Registers,
    ) = crate::context_switch::switch_context_cross_thread_float;
    const ALLOW_DEFLECTION: bool = true;
}

/// Lightweight switcher that skips floating-point state but supports cross-thread migration.
pub struct CrossThreadNoFloat;
impl ContextSwitcher for CrossThreadNoFloat {
    const SWITCH_FN: unsafe extern "C" fn(
        *mut crate::memory_management::Registers,
        *const crate::memory_management::Registers,
    ) = crate::context_switch::switch_context_cross_thread_no_float;
    const ALLOW_DEFLECTION: bool = true;
}

/// Optimized switcher for fibers pinned to a single thread, saving/restoring floating-point state.
pub struct SameThreadFloat;
impl ContextSwitcher for SameThreadFloat {
    const SWITCH_FN: unsafe extern "C" fn(
        *mut crate::memory_management::Registers,
        *const crate::memory_management::Registers,
    ) = crate::context_switch::switch_context_same_thread_float;
    const ALLOW_DEFLECTION: bool = false;
}

/// The fastest possible switcher: pins to one thread and ignores floating-point state.
pub struct SameThreadNoFloat;
impl ContextSwitcher for SameThreadNoFloat {
    const SWITCH_FN: unsafe extern "C" fn(
        *mut crate::memory_management::Registers,
        *const crate::memory_management::Registers,
    ) = crate::context_switch::switch_context_same_thread_no_float;
    const ALLOW_DEFLECTION: bool = false;
}

/// Fluent builder for configuring and launching fibers.
pub struct SpawnBuilder<S: ContextSwitcher = CrossThreadFloat> {
    name: Option<&'static str>,
    affinity: topology::Affinity,
    priority: Priority,
    kind: WorkloadKind,
    /// Per-fiber deflection scope. `None` inherits the runtime's topology;
    /// an explicit value overrides it for this fiber only.
    mode: Option<TopologyMode>,
    safety: crate::memory_management::SafetyLevel,
    stack_size: Option<usize>,
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
            mode: None,
            safety: crate::memory_management::SafetyLevel::Safety0,
            stack_size: None,
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

    /// Sets this fiber's deflection scope, overriding the runtime's topology
    /// for this fiber only.
    ///
    /// * `P2PMesh` — deflection is confined to the fibre's own CCX
    ///   neighbourhood (the default on a `P2PMesh` runtime).
    /// * `Global` — deflection may target any worker in the mesh, even on a
    ///   `P2PMesh` runtime. Useful for long-running, cache-cold work that
    ///   benefits from being placed wherever there is room.
    /// * `Pinned` — never deflected at all; the fiber stays on its origin
    ///   worker (also forced for `SameThread` switchers, which cannot
    ///   survive migration).
    ///
    /// Leaving this unset inherits the runtime's topology, which is what
    /// every fiber got before this override existed.
    #[inline(always)]
    #[must_use]
    pub const fn topology_mode(mut self, mode: TopologyMode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Sets the hardware safety level (0-2).
    ///
    /// This is the *per-fiber* stack-overflow policy, and is independent of
    /// the pool-wide `SafetyLevel` passed to the runtime initializer (which
    /// only decides the arena's address-space layout):
    /// * `Safety0` — no canary; the fiber gets the whole slot stack.
    /// * `Safety1` — an 8-byte canary is planted below the granted stack and
    ///   verified when the fiber terminates.
    /// * `Safety2` — as `Safety1`, and additionally verified at every
    ///   suspension point, so an overflow is reported at the first `.await`
    ///   that follows it rather than only at termination.
    #[inline(always)]
    #[must_use]
    pub const fn safety(mut self, safety: crate::memory_management::SafetyLevel) -> Self {
        self.safety = safety;
        self
    }

    /// Requests `bytes` of stack for this fiber.
    ///
    /// The request is clamped to what one slot can actually provide (the
    /// runtime's configured `stack_size` plus the 8 KB zero-copy buffer);
    /// an oversized request is recorded in [`crate::STACK_CLAMPED_SPAWNS`]
    /// rather than silently ignored. Leaving this unset grants the fiber
    /// the whole slot, which is what every fiber got before this knob
    /// existed.
    #[inline(always)]
    #[must_use]
    pub const fn stack_size(mut self, bytes: usize) -> Self {
        self.stack_size = Some(bytes);
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
            stack_size: self.stack_size,
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
    #[allow(clippy::too_many_lines)]
    pub fn spawn<F: Future + Send + 'static>(self, fut: F) -> dtact_handle_t {
        let runtime = crate::GLOBAL_RUNTIME
            .get()
            .expect("Dtact Runtime not initialized");
        let pool = &runtime.pool;
        let mut fixed_spins: u32 = 0;

        let ctx_id = 'alloc: loop {
            if let Some(id) = pool.alloc_context() {
                // If we are in a fiber, reward the success
                let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(std::cell::Cell::get);
                if !ctx_ptr.is_null() {
                    unsafe {
                        let ctx = &mut *ctx_ptr;
                        ctx.adaptive_spin_count = (ctx.adaptive_spin_count + 1).min(2000);
                        ctx.spin_failure_count = ctx.spin_failure_count.saturating_sub(1);
                    }
                }
                break 'alloc id;
            }

            let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(std::cell::Cell::get);
            if ctx_ptr.is_null() {
                // HOST-THREAD SPINNING — spin hard first, OS yield only as last resort
                if fixed_spins < 8000 {
                    core::hint::spin_loop();
                    fixed_spins += 1;

                    // Sparse Polling for host threads too
                    if fixed_spins.trailing_zeros() >= 3
                        && let Some(id) = pool.alloc_context()
                    {
                        break 'alloc id;
                    }
                } else {
                    std::thread::yield_now();
                    fixed_spins = 4000; // Keep partial spin budget
                }
            } else {
                // FIBER-AWARE ADAPTIVE SPINNING
                unsafe {
                    let ctx = &mut *ctx_ptr;
                    let current_spin = ctx.adaptive_spin_count;
                    let failure_count = ctx.spin_failure_count;

                    // Only spin if failure count is low
                    if failure_count < 20 {
                        for i in 0..current_spin {
                            core::hint::spin_loop();

                            // Sparse Polling: only check the pool every 8 iterations to reduce L1 pressure
                            if i.trailing_zeros() >= 3
                                && let Some(id) = pool.alloc_context()
                            {
                                ctx.adaptive_spin_count = (current_spin + 2).min(2000);
                                ctx.spin_failure_count = failure_count.saturating_sub(1);
                                break 'alloc id;
                            }
                        }
                    }

                    // Spin failed: Penalize budget and yield
                    ctx.spin_failure_count = failure_count.saturating_add(1);
                    ctx.adaptive_spin_count = current_spin.saturating_sub(100).max(200);

                    ctx.state.store(
                        crate::memory_management::FiberStatus::Notified as u32,
                        core::sync::atomic::Ordering::Release,
                    );
                    (ctx.switch_fn)(&raw mut ctx.regs, &raw const ctx.executor_regs);
                }
            }
        };

        let ctx_ptr = pool.get_context_ptr(ctx_id);
        let current_core = crate::future_bridge::CURRENT_WORKER_ID.with(|c| {
            let id = c.get();
            if id < runtime.scheduler.workers.len() {
                id
            } else {
                topology::current().core_id as usize % runtime.scheduler.workers.len()
            }
        });

        unsafe {
            (*ctx_ptr).state.store(
                crate::memory_management::FiberStatus::Running as u32,
                core::sync::atomic::Ordering::Release,
            );
            (*ctx_ptr).kind = self.kind;
            // Switcher policy overrides user mode: SameThread switchers can never deflect.
            // Compile-time const, dead-code-eliminated to a single branch by the optimizer.
            // An unset mode inherits the runtime's topology, so a `Pinned`
            // runtime pins every fiber and a `Global` one makes Global the
            // default scope without any per-fiber configuration.
            (*ctx_ptr).mode = if S::ALLOW_DEFLECTION {
                self.mode.unwrap_or(runtime.scheduler.topology)
            } else {
                TopologyMode::Pinned
            };
            (*ctx_ptr).affinity = self.affinity;
            (*ctx_ptr).origin_core = current_core as u16;
            (*ctx_ptr).fiber_index = ctx_id;
            (*ctx_ptr).switch_fn = S::SWITCH_FN;
            (*ctx_ptr).last_os_thread_id = 0; // Reset for new fiber execution
            (*ctx_ptr).priority = self.priority;
            (*ctx_ptr).name = self.name;

            // Seed the adaptive spin budget from the workload kind.
            (*ctx_ptr).adaptive_spin_count = self.kind.spin_budget();

            // Aligned Zero-Copy Future Migration
            let align = core::mem::align_of::<F>();
            let fut_size = core::mem::size_of::<F>();
            let buffer_start = (*ctx_ptr).read_buffer_ptr as usize;
            let buffer_end = buffer_start + 8192;
            let aligned_fut_addr = (buffer_end - fut_size) & !(align - 1);

            // Determine where the stack region ends (just below the future).
            // The stack grows DOWNWARD from this address toward the slot's
            // own bottom.
            let stack_ceiling: usize = if aligned_fut_addr < buffer_start
                || (aligned_fut_addr + fut_size) > buffer_end
            {
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
                buffer_end
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
                aligned_fut_addr
            };

            // Hand the fiber its stack: from the ceiling just computed down
            // to the slot's own bottom, honouring the requested size and
            // planting the overflow canary if this fiber asked for one.
            //
            // Before this, the stack was confined to the 8 KB zero-copy
            // buffer even though every slot also reserves a full
            // `stack_size` region below it — so a runtime configured with
            // 512 KB stacks gave its fibers 8 KB, and the rest of the slot
            // was address space nobody could reach.
            let stack_bottom = pool.stack_bottom(ctx_id);
            if let Some(requested) = self.stack_size
                && requested > pool.stack_capacity
            {
                crate::STACK_CLAMPED_SPAWNS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            }
            let _stack_limit = crate::memory_management::grant_stack(
                ctx_ptr,
                stack_ceiling,
                stack_bottom,
                self.stack_size,
                self.safety,
            );

            // ABI-compliant stack alignment
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            let stack_top = (stack_ceiling & !0xF) - 8;
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            let stack_top = stack_ceiling & !0xF;

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            let stack_top_ptr = stack_top as *mut u64;

            // Poison return address (dtact_abort) — if fiber_entry_point ever returns,
            // this triggers a controlled abort instead of undefined behavior.
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            core::ptr::write(stack_top_ptr, crate::c_ffi::dtact_abort as *const () as u64);

            let stack_top = stack_top as *mut u8;

            #[cfg(target_arch = "x86_64")]
            {
                (*ctx_ptr).regs.gprs[0] = stack_top as u64; // RSP
                (*ctx_ptr).regs.gprs[7] = fiber_entry_point as *const () as u64; // RIP
                #[cfg(windows)]
                {
                    // The TIB's three stack bounds are now exactly the ones
                    // `grant_stack` just granted: base = the ceiling the
                    // stack grows down from, limit = the lowest byte it may
                    // touch, deallocation stack = the slot's own bottom.
                    (*ctx_ptr).regs.gprs[10] = stack_ceiling as u64; // Stack Base (top)
                    (*ctx_ptr).regs.gprs[11] = stack_limit as u64; // Stack Limit (bottom)
                    (*ctx_ptr).regs.gprs[12] = stack_bottom as u64; // DeallocationStack
                    (*ctx_ptr).regs.gprs[13] = 0; // ExceptionList
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                let lr = fiber_entry_point as *const () as u64;
                let sp = stack_top as u64;
                #[allow(unused)]
                let mut signed_lr = lr;
                #[cfg(all(target_arch = "aarch64", unix, feature = "security-hardened",))]
                core::arch::asm!(
                    "mov x16, {lr}",
                    "mov x17, {sp}",
                    ".inst 0xDAC10230", // pacia x16, x17
                    "mov {lr}, x16",
                    lr = inout(reg) signed_lr,
                    sp = in(reg) sp,
                    out("x16") _, out("x17") _,
                );
                (*ctx_ptr).regs.gprs[12] = sp; // SP
                (*ctx_ptr).regs.gprs[11] = signed_lr; // Signed x30 (LR)
                #[cfg(windows)]
                {
                    (*ctx_ptr).regs.gprs[13] = stack_ceiling as u64; // Stack Base (top)
                    (*ctx_ptr).regs.gprs[14] = stack_limit as u64; // Stack Limit (bottom)
                    (*ctx_ptr).regs.gprs[15] = stack_bottom as u64; // DeallocationStack
                }
            }
            #[cfg(target_arch = "riscv64")]
            {
                (*ctx_ptr).regs.gprs[0] = stack_top as u64; // SP
                (*ctx_ptr).regs.gprs[13] = fiber_entry_point as *const () as u64; // RA
            }
        }

        let r#gen = u64::from(unsafe {
            (*ctx_ptr)
                .generation
                .load(core::sync::atomic::Ordering::Acquire)
        });

        crate::wake_fiber(current_core, ctx_id);

        // Handle Layout: [1-bit Valid | 15-bit Generation | 16-bit CoreID | 32-bit ContextID]
        dtact_handle_t(
            u64::from(ctx_id)
                | ((current_core as u64) << 32)
                | ((r#gen & 0x7FFF) << 48)
                | (1 << 63),
        )
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

    // Execute the task payload with SEH/Panic protection. This catch_unwind
    // spans the fiber's *entire* multi-suspension lifetime, not just one
    // resumption: `invoke` drives the user future/closure across however
    // many switch-out/switch-in cycles it takes via `wait_pinned`, and each
    // resumption re-enters this same still-live stack frame, so a panic (or
    // a `cancel()`-triggered unwind, see `future_bridge::DtaCancellation`)
    // raised from arbitrarily deep inside any of those resumptions is still
    // caught here.
    let result = std::panic::catch_unwind(core::panic::AssertUnwindSafe(move || {
        unsafe { invoke(arg) };
    }));

    // Execute cleanup if present (e.g. FFI arg free) — MUST happen before we lose the context
    if let Some(cleanup) = ctx.cleanup_fn.take() {
        unsafe { cleanup(ctx.closure_ptr) };
    }

    // Stack-overflow canary (Safety1/Safety2). Checked here, once, at the
    // single point where the fiber's stack usage is complete — the canary
    // word sits immediately below the granted limit, so a tripped canary
    // means this fiber's deepest frame reached past the stack it was given.
    ctx.check_stack_guard();

    let final_state = match result {
        Ok(()) => crate::memory_management::FiberStatus::Finished,
        Err(payload) => {
            // Cold: this runs once per fiber, at termination — but for the
            // overwhelming majority of fibers in any real workload, that
            // termination is `Ok`, not `Err`.
            core::hint::cold_path();
            if payload
                .downcast_ref::<crate::future_bridge::DtaCancellation>()
                .is_some()
            {
                crate::memory_management::FiberStatus::Cancelled
            } else {
                // A named fiber gets its name into the log alongside the
                // panic hook's own message: without it, the only thing tying
                // a panic to a fiber is the pool index in `outcome`, which
                // nobody has at the moment the panic is printed. Unnamed
                // fibers stay silent here — the default hook already said
                // everything useful about them.
                if let Some(name) = ctx.name {
                    eprintln!(
                        "DTA-V3: fiber '{name}' ({}) panicked; payload is available from dtact::outcome.",
                        ctx.fiber_index
                    );
                }
                // Double-boxed so the fat `Box<dyn Any + Send>` pointer fits
                // in `panic_payload_ptr`'s thin `*mut ()`; retrieved by
                // `crate::api::outcome`, and otherwise dropped by
                // `ContextPool::free_context` when this slot is recycled.
                ctx.panic_payload_ptr = Box::into_raw(Box::new(payload)).cast::<()>();
                crate::memory_management::FiberStatus::Panicked
            }
        }
    };

    // Mark as terminal. The scheduler will return this context to the pool
    // AFTER we switch back, preventing use-after-free races.
    ctx.state
        .store(final_state as u32, core::sync::atomic::Ordering::Release);
    // No futex_wake needed: dtact_await host-thread path uses spin+yield_now, not futex.

    // Wake up any fiber waiting for this one (FFI join).
    // AcqRel: Release ensures state=Finished is visible before we read waiter_handle;
    // Acquire syncs with the waiter's AcqRel swap that registered the handle.
    let waiter = ctx
        .waiter_handle
        .swap(0, core::sync::atomic::Ordering::AcqRel);
    if waiter != 0 {
        // Centralised wake routing — reads the waiter's mode and dispatches
        // through enqueue_pinned or enqueue_deflect with full warehouse fallback.
        crate::wake_waiter_handle(waiter);
    }

    // Switch back to the scheduler. The scheduler's dispatch_loop will see
    // state == Finished and call free_context on our behalf.
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

        // `mrs mpidr_el1` is an EL1-privileged system register read. Linux is
        // the only major OS that traps it from EL0 and emulates a sane value
        // (`emulate_mrs` in arch/arm64/kernel/sys.c). macOS, Windows-on-ARM
        // and the BSDs do not emulate it — the bare instruction raises an
        // illegal-instruction fault (SIGILL on Unix, STATUS_ILLEGAL_INSTRUCTION
        // on Windows). Restrict the read to Linux and fall back to the null
        // topology elsewhere; the scheduler treats that as a single group.
        #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
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

        // `mhartid` is a Machine-mode privileged register. Reading it from User-mode
        // (U-mode) will raise an illegal instruction exception. Since Dtact is a
        // user-space library, we fall back to a single-core topology on RISC-V
        // until a stable platform-specific syscall for topology is integrated.
        #[cfg(all(target_arch = "riscv64", feature = "kernel"))]
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

        #[cfg(any(
            all(target_arch = "aarch64", not(target_os = "linux")),
            all(target_arch = "riscv64", not(feature = "kernel")),
            not(any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "aarch64",
                target_arch = "riscv64",
            )),
        ))]
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

    /// Parses a human-readable stack size such as `"256K"`, `"2M"`, `"64KiB"`
    /// or a bare byte count into bytes.
    ///
    /// Accepts an optional binary suffix (`K`/`KiB`, `M`/`MiB`, `G`/`GiB`,
    /// case-insensitive) applied to a decimal number. Returns `None` for
    /// anything it cannot parse, for a zero/empty value, and on overflow —
    /// callers treat that as "no explicit request" and let the slot's own
    /// capacity stand, rather than failing a spawn over a typo.
    ///
    /// Cold path by construction: it runs once per spawn of a
    /// `spawn_with_stack` fiber, never per task. The `#[task]` attribute
    /// resolves its stack size at compile time instead.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn parse_stack_size(spec: &str) -> Option<usize> {
        let spec = spec.trim();
        let split = spec
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(spec.len());
        let (digits, suffix) = spec.split_at(split);
        let value: usize = digits.parse().ok()?;
        if value == 0 {
            return None;
        }
        let unit = match suffix.trim().to_ascii_uppercase().as_str() {
            "" | "B" => 1usize,
            "K" | "KB" | "KIB" => 1024,
            "M" | "MB" | "MIB" => 1024 * 1024,
            "G" | "GB" | "GIB" => 1024 * 1024 * 1024,
            _ => return None,
        };
        value.checked_mul(unit)
    }

    /// Spawn-time options for the closure-based spawn path
    /// ([`spawn_closure`] / [`spawn_with_stack`]).
    ///
    /// The closure path exists because a plain `fn`/`FnOnce` cannot be a
    /// future; it needs the same knobs the future path gets from
    /// [`super::SpawnBuilder`], or a fiber spawned from C could never be
    /// given a stack size, a priority or a name. This struct is that
    /// parity, and `#[dtact::export_fiber]` builds one from its attribute
    /// arguments.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ClosureSpawnOptions {
        /// Stack bytes to grant. `None` = the slot's full capacity.
        pub stack_size: Option<usize>,
        /// Scheduling priority (see [`super::Priority`]).
        pub priority: super::Priority,
        /// Per-fiber stack-overflow policy.
        pub safety: crate::memory_management::SafetyLevel,
        /// Optional static name, surfaced in diagnostics and by
        /// [`super::fiber_info`].
        pub name: Option<&'static str>,
    }

    impl ClosureSpawnOptions {
        /// Defaults: no explicit stack request, `Normal` priority,
        /// `Safety0`, unnamed — i.e. exactly what `spawn_with_stack` did
        /// before the other knobs existed.
        #[must_use]
        pub const fn new() -> Self {
            Self {
                stack_size: None,
                priority: super::Priority::Normal,
                safety: crate::memory_management::SafetyLevel::Safety0,
                name: None,
            }
        }

        /// Requests `bytes` of stack (clamped to the slot's capacity).
        #[inline(always)]
        #[must_use]
        pub const fn stack_size(mut self, bytes: usize) -> Self {
            self.stack_size = Some(bytes);
            self
        }

        /// Sets the scheduling priority.
        #[inline(always)]
        #[must_use]
        pub const fn priority(mut self, priority: super::Priority) -> Self {
            self.priority = priority;
            self
        }

        /// Sets the per-fiber stack-overflow policy.
        #[inline(always)]
        #[must_use]
        pub const fn safety(mut self, safety: crate::memory_management::SafetyLevel) -> Self {
            self.safety = safety;
            self
        }

        /// Sets a static name for diagnostics.
        #[inline(always)]
        #[must_use]
        pub const fn name(mut self, name: &'static str) -> Self {
            self.name = Some(name);
            self
        }
    }

    impl Default for ClosureSpawnOptions {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Spawns a fiber from a closure with a specific stack configuration.
    ///
    /// `stack_size_str` accepts the same syntax as
    /// [`parse_stack_size`] (`"256K"`, `"2M"`, ...) and is the stack this
    /// closure is actually granted: the stack starts immediately below the
    /// closure's own slot in the zero-copy buffer and grows down through
    /// the slot's reserved stack region, stopping at the granted limit. An
    /// unparseable or oversized request falls back to the slot's full
    /// capacity (an oversized one is also counted in
    /// [`crate::STACK_CLAMPED_SPAWNS`]).
    ///
    /// Shorthand for [`spawn_closure`] with
    /// `ClosureSpawnOptions::new().stack_size(..)`; use that function
    /// directly to also set the priority, per-fiber safety level or name.
    ///
    /// # Panics
    /// * Panics if the runtime is not initialized.
    /// * Panics if the context pool is exhausted.
    #[inline]
    pub fn spawn_with_stack<F: FnOnce() + Send + 'static>(
        stack_size_str: &str,
        f: F,
    ) -> dtact_handle_t {
        let opts = parse_stack_size(stack_size_str)
            .map_or_else(ClosureSpawnOptions::new, |bytes| {
                ClosureSpawnOptions::new().stack_size(bytes)
            });
        spawn_closure(f, opts)
    }

    /// Spawns a fiber from a closure with the full set of spawn-time options.
    ///
    /// The closure path is inherently pinned to the spawning thread
    /// (`SameThreadNoFloat` switcher): a plain `FnOnce` has no suspension
    /// points, so it never needs to migrate, and its switcher could not
    /// survive migration anyway. `opts.priority` therefore only affects
    /// queue placement (a `Critical` closure jumps the local queue), and
    /// `opts.name`/`opts.safety`/`opts.stack_size` apply exactly as they do
    /// for a future-based fiber.
    ///
    /// # Panics
    /// * Panics if the runtime is not initialized.
    /// * Panics if the context pool is exhausted.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::too_many_lines)]
    pub fn spawn_closure<F: FnOnce() + Send + 'static>(
        f: F,
        opts: ClosureSpawnOptions,
    ) -> dtact_handle_t {
        let runtime = crate::GLOBAL_RUNTIME
            .get()
            .expect("Dtact Runtime not initialized");
        let pool = &runtime.pool;
        let ctx_id = pool.alloc_context().expect("Context pool exhausted - OOM");
        let ctx_ptr = pool.get_context_ptr(ctx_id);
        #[allow(clippy::cast_possible_truncation)]
        let current_core = topology::current().core_id as usize;

        let requested = opts.stack_size;
        if let Some(bytes) = requested
            && bytes > pool.stack_capacity
        {
            crate::STACK_CLAMPED_SPAWNS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        }

        unsafe {
            (*ctx_ptr).state.store(
                crate::memory_management::FiberStatus::Running as u32,
                core::sync::atomic::Ordering::Release,
            );
            (*ctx_ptr).origin_core = current_core as u16;
            (*ctx_ptr).fiber_index = ctx_id;
            (*ctx_ptr).switch_fn = crate::context_switch::switch_context_same_thread_no_float;
            // A same-thread switcher can never be deflected; make that
            // explicit here too so the wake path does not have to infer it.
            (*ctx_ptr).mode = crate::common_types::TopologyMode::Pinned;
            (*ctx_ptr).affinity = topology::Affinity::SameCore;
            (*ctx_ptr).priority = opts.priority;
            (*ctx_ptr).name = opts.name;
            (*ctx_ptr).kind = crate::common_types::WorkloadKind::Compute;
            (*ctx_ptr).adaptive_spin_count =
                crate::common_types::WorkloadKind::Compute.spin_budget();

            let f_ptr = (*ctx_ptr).read_buffer_ptr.cast::<F>();
            core::ptr::write(f_ptr, f);
            (*ctx_ptr).invoke_closure = |ptr| {
                let f = core::ptr::read(ptr.cast::<F>());
                f();
            };
            (*ctx_ptr).closure_ptr = f_ptr.cast::<()>();

            // Point 1: Shadow Space Separation (Stack MUST start BELOW the 8KB Future buffer)
            let buffer_start = (*ctx_ptr).read_buffer_ptr as usize;
            let stack_ceiling = (buffer_start & !0xF) - 72;
            let stack_bottom = pool.stack_bottom(ctx_id);

            // Point 4: "Return-to-Nowhere" Protection
            let stack_top_ptr = stack_ceiling as *mut u64;
            core::ptr::write(stack_top_ptr, crate::c_ffi::dtact_abort as *const () as u64);

            // Grant the requested stack. The closure occupies the bottom of
            // the 8 KB buffer, so the whole granted region lives in the
            // slot's reserved stack area below it — exactly what
            // `ContextPool::new`'s `stack_size` argument reserved.
            let _stack_limit = crate::memory_management::grant_stack(
                ctx_ptr,
                stack_ceiling,
                stack_bottom,
                requested,
                opts.safety,
            );

            let stack_top = stack_ceiling as *mut u8;

            #[cfg(target_arch = "x86_64")]
            {
                (*ctx_ptr).regs.gprs[0] = stack_top as u64; // RSP
                (*ctx_ptr).regs.gprs[7] = super::fiber_entry_point as *const () as u64; // RIP
                #[cfg(windows)]
                {
                    (*ctx_ptr).regs.gprs[10] = stack_ceiling as u64; // Stack Base (top)
                    (*ctx_ptr).regs.gprs[11] = stack_limit as u64; // Stack Limit (bottom)
                    (*ctx_ptr).regs.gprs[12] = stack_bottom as u64; // DeallocationStack
                    (*ctx_ptr).regs.gprs[13] = 0; // ExceptionList
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                let lr = super::fiber_entry_point as *const () as u64;
                let sp = stack_top as u64;
                let mut signed_lr = lr;
                #[cfg(all(target_arch = "aarch64", unix, feature = "security-hardened",))]
                core::arch::asm!(
                    "mov x16, {lr}",
                    "mov x17, {sp}",
                    ".inst 0xDAC10230", // pacia x16, x17
                    "mov {lr}, x16",
                    lr = inout(reg) signed_lr,
                    sp = in(reg) sp,
                    out("x16") _, out("x17") _,
                );
                (*ctx_ptr).regs.gprs[12] = sp; // SP
                (*ctx_ptr).regs.gprs[11] = signed_lr; // Signed x30 (LR)
                #[cfg(windows)]
                {
                    (*ctx_ptr).regs.gprs[13] = stack_ceiling as u64; // Stack Base (top)
                    (*ctx_ptr).regs.gprs[14] = stack_limit as u64; // Stack Limit (bottom)
                    (*ctx_ptr).regs.gprs[15] = stack_bottom as u64; // DeallocationStack
                }
            }
            #[cfg(target_arch = "riscv64")]
            {
                (*ctx_ptr).regs.gprs[0] = stack_top as u64; // SP
                (*ctx_ptr).regs.gprs[13] = super::fiber_entry_point as *const () as u64; // RA
            }
        }

        let r#gen = u64::from(unsafe {
            (*ctx_ptr)
                .generation
                .load(core::sync::atomic::Ordering::Acquire)
        });
        crate::wake_fiber(current_core, ctx_id);
        // Same handle layout as every other spawn path — `[1-bit Valid |
        // 15-bit Generation | 16-bit CoreID | 32-bit ContextID]`. The
        // generation and valid bit are what let `dtact_await`/`fiber_info`
        // tell this fiber apart from a later occupant of the same slot.
        dtact_handle_t(
            u64::from(ctx_id)
                | ((current_core as u64) << 32)
                | ((r#gen & 0x7FFF) << 48)
                | (1 << 63),
        )
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
        let target_core_id = ((handle.0 >> 32) & 0xFFFF) as usize;

        // State-guarded wake — see `awaken_fiber_by_index` for the protocol
        // and the double-dispatch race it prevents on deflectable fibers.
        crate::awaken_fiber_by_index(target_core_id, target_ctx_id);

        unsafe {
            let ctx = &mut *ctx_ptr;
            ctx.state.store(
                crate::memory_management::FiberStatus::Suspending as u32,
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
#[inline(always)]
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
    let handle_val = handle.0 & !(1 << 63); // Strip sentinel bit
    let target_ctx_id = (handle_val & 0xFFFF_FFFF) as u32;
    let target_core_id = ((handle_val >> 32) & 0xFFFF) as usize;
    // State-guarded wake — see `awaken_fiber_by_index` for the protocol
    // and the double-dispatch race it prevents on deflectable fibers.
    crate::awaken_fiber_by_index(target_core_id, target_ctx_id);
    yield_now().await;
}

/// Requests cooperative cancellation of the fiber referred to by `handle`.
///
/// This only sets a flag and, if the target is currently parked
/// (`Yielded`), wakes it so it gets a chance to observe that flag — it
/// does not itself block or confirm cancellation. The target unwinds the
/// next time it resumes through any DTA suspension point (`wait_now`,
/// `yield_now`, a mailbox/mutex wait, ...), landing in
/// [`FiberStatus::Cancelled`], observable via [`outcome`].
///
/// # Cooperative, not preemptive
/// A fiber that never suspends through a DTA primitive — a pure
/// CPU-bound loop with no `.await` inside it — never observes this flag
/// and cannot be cancelled until it does, exactly as with any
/// cooperative scheduler (DTA does not preempt `Running` fibers).
///
/// A stale handle (the fiber already finished and its slot was reused
/// for an unrelated task) is silently ignored: the generation check
/// below guards against cancelling the wrong task — including the case
/// where the slot is reclaimed *during* this call (see the comment on
/// the post-store re-check).
#[allow(clippy::cast_possible_truncation)]
#[inline]
pub fn cancel(handle: dtact_handle_t) {
    let Some(runtime) = crate::GLOBAL_RUNTIME.get() else {
        return;
    };
    let handle_val = handle.0 & !(1 << 63); // Strip sentinel bit
    let target_ctx_id = (handle_val & 0xFFFF_FFFF) as u32;
    let target_core_id = ((handle_val >> 32) & 0xFFFF) as usize;
    let handle_gen = ((handle_val >> 48) & 0x7FFF) as u16;

    let ctx_ptr = runtime.pool.get_context_ptr(target_ctx_id);
    // Acquire, not Relaxed: unlike `cancel_requested` below, these two
    // generation reads gate a real correctness decision (whether to undo
    // the store past the race window), and `cancel` is called rarely
    // enough — once per cancellation, not once per task — that there is
    // no meaningful cost to paying for a timely, correctly-ordered read
    // here rather than trusting Relaxed's weaker eventual-visibility.
    let gen_before = unsafe {
        ((*ctx_ptr)
            .generation
            .load(core::sync::atomic::Ordering::Acquire) as u16)
            & 0x7FFF
    };
    if gen_before != handle_gen {
        // Handle refers to a since-recycled slot; nothing to cancel.
        return;
    }

    // `cancel_requested` itself is a pure signal with no payload riding
    // along with it (unlike `state`/`generation`, nothing downstream
    // depends on *what else* was visible at the moment it flips) —
    // Relaxed is sufficient, same reasoning as the warehouse `backlog`
    // counter's ordering (see `paper/main.tex`'s memory-order audit).
    unsafe {
        (*ctx_ptr)
            .cancel_requested
            .store(true, core::sync::atomic::Ordering::Relaxed);
    }

    // Re-check generation after the store: if it moved, `claim_context`
    // reclaimed this slot for a *different* fiber somewhere between our
    // read above and the store just now, meaning the store above may have
    // just set `cancel_requested` on that unrelated new occupant instead
    // of our actual target (which, for generation to have moved at all,
    // must itself already be done). Undo it rather than leave a stray
    // cancellation armed against a fiber that never asked for one — found
    // by the same kind of concurrent-load stress testing that caught the
    // read-side race in `dtact_await_observe`, not merely theorized.
    let gen_after = unsafe {
        ((*ctx_ptr)
            .generation
            .load(core::sync::atomic::Ordering::Acquire) as u16)
            & 0x7FFF
    };
    if gen_after != gen_before {
        unsafe {
            (*ctx_ptr)
                .cancel_requested
                .store(false, core::sync::atomic::Ordering::Relaxed);
        }
        return;
    }

    // State-guarded wake, same protocol as `yield_to` — only actually
    // enqueues if the target was parked (`Yielded`); a `Running`/
    // `Suspending` fiber will observe the flag on its own next resumption.
    crate::awaken_fiber_by_index(target_core_id, target_ctx_id);
}

/// Blocks until the fiber referred to by `handle` terminates, then returns
/// its outcome.
///
/// A join, not a poll: this has the same blocking behaviour as
/// `dtact_await` (natively yields if called from within a fiber, spins
/// then parks if called from a host thread), plus the classification.
///
/// `None` only if `handle` was already stale (its slot had been reused for
/// a different fiber) at the moment this call started waiting — not "not
/// yet terminated," since this blocks until it is.
///
/// # Why this blocks instead of polling
/// An earlier version of this function read `state` directly, non-blocking,
/// meant to be called any time after a separate `dtact_await` had already
/// confirmed termination. That was racy in a way this project initially
/// underestimated: `ContextPool::claim_context` finalizes a slot's
/// `state`/`generation` the moment `alloc_context` next hands that slot to
/// a **different** fiber, and under real concurrent load (many threads
/// sharing one pool, as in this crate's own test suite) that next claim
/// can — and empirically did, reproducibly — happen before a second,
/// separate call got back around to reading it, even one issued
/// immediately after the first confirmed termination. Blocking here and
/// capturing the status in the exact same read that detects termination
/// (see [`crate::c_ffi::dtact_await_observe`]) closes that window
/// entirely: there is no second read to race.
///
/// For [`memory_management::TaskOutcome::Panicked`], also returns the
/// panic payload's message when it was an ordinary `&str`/`String`
/// payload (the common case for `panic!("...")`). The classification
/// (`Panicked` itself) is captured atomically with termination and is
/// therefore race-free; the message is a second, separate read taken
/// immediately after and carries the same — far narrower — residual risk
/// as the rest of this function's outcome used to: an extremely tight
/// race against a concurrent reclaim could occasionally lose the message
/// (never the classification) back to `None`.
#[must_use]
#[inline]
pub fn outcome(
    handle: dtact_handle_t,
) -> Option<(crate::memory_management::TaskOutcome, Option<String>)> {
    let (status, ctx_ptr) = crate::c_ffi::dtact_await_observe(handle)?;

    if status == FiberStatus::Finished as u32 {
        Some((crate::memory_management::TaskOutcome::Finished, None))
    } else if status == FiberStatus::Cancelled as u32 {
        Some((crate::memory_management::TaskOutcome::Cancelled, None))
    } else if status == FiberStatus::Panicked as u32 {
        // `ctx_ptr` is exactly what `dtact_await_observe` already resolved
        // to detect termination — no need to re-derive `target_ctx_id` and
        // re-look-up `GLOBAL_RUNTIME`/`get_context_ptr` for it here.
        core::hint::cold_path();
        let ctx = unsafe { &mut *ctx_ptr };
        let payload_ptr = core::mem::replace(&mut ctx.panic_payload_ptr, core::ptr::null_mut());
        let message = if payload_ptr.is_null() {
            None
        } else {
            let payload =
                *unsafe { Box::from_raw(payload_ptr.cast::<Box<dyn core::any::Any + Send>>()) };
            payload
                .downcast_ref::<&str>()
                .map(|s| String::from(*s))
                .or_else(|| payload.downcast_ref::<String>().cloned())
        };
        Some((crate::memory_management::TaskOutcome::Panicked, message))
    } else {
        None
    }
}

/// Runtime-visible metadata of a fiber, as supplied at spawn time.
///
/// This is what the `name`, `priority`, `kind`, `affinity` and stack-size
/// knobs on [`SpawnBuilder`] actually turn into: the scheduler reads
/// `priority`/`affinity` from the same context on every enqueue, and this
/// struct exposes the whole set for telemetry, debugging and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiberInfo {
    /// The `&'static str` passed to [`SpawnBuilder::name`], if any.
    pub name: Option<&'static str>,
    /// The scheduling priority this fiber is dispatched with.
    pub priority: Priority,
    /// The workload kind used to seed its adaptive spin budget.
    pub kind: WorkloadKind,
    /// The affinity hint used by the wake/deflection path.
    pub affinity: Affinity,
    /// Stack bytes actually granted, after clamping to the slot.
    pub stack_size: usize,
    /// The worker index this fiber was spawned from.
    pub origin_core: u16,
}

/// Reads back the spawn-time metadata of the fiber `handle` refers to.
///
/// Returns `None` when the runtime is not initialized or the handle's
/// generation no longer matches (its slot has since been recycled), so a
/// stale handle reports "gone" instead of describing an unrelated fiber.
#[must_use]
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub fn fiber_info(handle: dtact_handle_t) -> Option<FiberInfo> {
    let runtime = crate::GLOBAL_RUNTIME.get()?;
    let handle_val = handle.0 & !(1 << 63); // Strip sentinel bit
    #[allow(clippy::cast_possible_truncation)]
    let ctx_id = (handle_val & 0xFFFF_FFFF) as u32;
    let handle_gen = ((handle_val >> 48) & 0x7FFF) as u16;

    let ctx = runtime.pool.get_context_ptr(ctx_id);
    // SAFETY: every index below `capacity` maps to an initialized
    // `FiberContext` for the pool's whole lifetime.
    unsafe {
        if ((*ctx)
            .generation
            .load(core::sync::atomic::Ordering::Acquire) as u16)
            & 0x7FFF
            != handle_gen
        {
            return None;
        }
        Some(FiberInfo {
            name: (*ctx).name,
            priority: (*ctx).priority,
            kind: (*ctx).kind,
            affinity: (*ctx).affinity,
            stack_size: (*ctx).stack_size,
            origin_core: (*ctx).origin_core,
        })
    }
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

/// A wrapper for futures generated by the `task` macro, carrying metadata.
pub struct TaskFuture<F, S> {
    /// The underlying future representing the task's body.
    pub future: F,
    /// The scheduling priority of the task.
    pub priority: Priority,
    /// The core affinity hints.
    pub affinity: Affinity,
    /// The workload category/kind.
    pub kind: WorkloadKind,
    /// Stack bytes requested by the `#[task(stack = "...")]` attribute, if
    /// any. Resolved to a byte count at compile time, so spawning a task
    /// costs no parsing.
    pub stack_size: Option<usize>,
    /// Phantom data for the context switcher.
    pub _marker: core::marker::PhantomData<S>,
}

impl<F: Future + Send + 'static, S: ContextSwitcher> Future for TaskFuture<F, S> {
    type Output = F::Output;

    #[inline(always)]
    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        // Safe projection to the inner future.
        unsafe { self.map_unchecked_mut(|s| &mut s.future).poll(cx) }
    }
}

/// Spawner tags and traits for macro-based task dispatch.
#[doc(hidden)]
pub mod spawner_traits {
    use super::{ContextSwitcher, Future, SpawnBuilder, TaskFuture, dtact_handle_t};

    /// Sentinel type for selecting the correct spawn implementation via method resolution.
    pub struct SpawnerTag;

    impl SpawnerTag {
        /// Spawn a macro-wrapped task future with its configured metadata and switcher.
        #[inline(always)]
        pub fn spawn<F: Future + Send + 'static, S: ContextSwitcher>(
            self,
            task: TaskFuture<F, S>,
        ) -> dtact_handle_t {
            let builder = SpawnBuilder::<S>::new()
                .priority(task.priority)
                .affinity(task.affinity)
                .kind(task.kind);
            // `stack_size` is only overridden when the attribute asked for
            // one; otherwise the fiber gets the slot's full capacity, same
            // as a plain `spawn()`.
            match task.stack_size {
                Some(bytes) => builder.stack_size(bytes).spawn(task.future),
                None => builder.spawn(task.future),
            }
        }
    }

    /// Fallback trait for standard futures (not wrapped by `#[task]`).
    pub trait FallbackSpawner<F> {
        /// Spawn a standard future.
        fn spawn(self, fut: F) -> dtact_handle_t;
    }

    impl<F: Future + Send + 'static> FallbackSpawner<F> for &SpawnerTag {
        #[inline(always)]
        fn spawn(self, fut: F) -> dtact_handle_t {
            crate::api::spawn(fut)
        }
    }
}

#![allow(unsafe_code)]
#![allow(non_snake_case)]

use crate::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Safety policies for context pool memory layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyLevel {
    /// Raw performance: No guard pages, minimal overhead.
    Safety0,
    /// Balanced: Guard pages every 32 contexts to catch massive overflows.
    Safety1,
    /// Strict: Per-context hardware guard pages for maximum isolation.
    Safety2,
}

pub use crate::common_types::{TopologyMode, WorkloadKind};

/// Machine-specific registers for context switching.
///
/// Aligned to 64 bytes to prevent cache line splits and ensure atomic
/// context updates on supported architectures.
#[repr(C, align(64))]
#[derive(Debug)]
pub(crate) struct Registers {
    /// General Purpose Registers (GPRs).
    pub(crate) gprs: [u64; 16],
    /// SIMD / Extended state (e.g. AVX, Neon).
    pub(crate) extended_state: [u8; 512],
}

impl Registers {
    /// Creates a new register set, with a sane default MXCSR on `x86_64`.
    ///
    /// # The bug this fixes
    /// The `*_float` switchers in `context_switch.rs` (both cross-thread
    /// and same-thread; the `*_no_float` switchers never touch this state
    /// at all — that is the entire point of "no float") store the SSE
    /// control word (MXCSR) inside `gprs` on `x86_64` — Unix at `gprs[8]`
    /// (byte offset 64), Windows at `gprs[14]` (byte offset 112, alongside
    /// the extra TIB/XMM6-15 state that ABI requires) — and `ldmxcsr` it
    /// into the live CPU register on every switch. MXCSR's six
    /// exception-mask bits use **inverted** polarity from what a zeroed
    /// word usually means: `1` = masked (disabled, the safe default), `0`
    /// = unmasked (that exception class traps immediately on the next
    /// occurrence). A context that has never yet been switched into — i.e.
    /// every freshly allocated `FiberContext` slot, the common case for
    /// any newly spawned fiber — previously carried a zeroed `gprs`, so
    /// its first dispatch loaded MXCSR `0x00000000`: every exception class
    /// unmasked, including "precision" (inexact), which fires on nearly
    /// every non-exact floating-point result. The very first non-trivial
    /// float operation that fiber (or, transitively, anything running on
    /// that same OS thread before it switches back) performs then
    /// reliably raises `SIGFPE`.
    ///
    /// `0x1F80` is the standard SSE reset value: all six exception classes
    /// masked, round-to-nearest, denormals-are-zero/flush-to-zero off —
    /// the same default every thread on the process's main stack already
    /// runs under, so a first-ever fiber dispatch now behaves identically
    /// to ordinary code instead of silently arming a crash.
    ///
    /// `AArch64`'s FPCR and RISC-V's `fcsr` do not have this hazard: `AArch64`
    /// FPCR's trap-enable bits default to *disabled* at `0` (the opposite
    /// polarity from MXCSR, so zero-init is already safe there), and
    /// RISC-V's F/D extension does not trap floating-point exceptions at
    /// all in the base ISA (they only accumulate as sticky `fflags` bits,
    /// checked by software, never delivered as a signal) — so neither
    /// needs the equivalent of this fix.
    #[must_use]
    #[inline(always)]
    pub(crate) const fn new() -> Self {
        // `mut` is only exercised on `x86_64` (the two `cfg` blocks below);
        // AArch64 and RISC-V need no special-cased slot, since neither has
        // this hazard (see the module-level doc comment above).
        #[allow(unused_mut)]
        let mut gprs = [0u64; 16];
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            gprs[8] = 0x0000_0000_0000_1F80;
        }
        #[cfg(all(target_arch = "x86_64", windows))]
        {
            gprs[14] = 0x0000_0000_0000_1F80;
        }
        Self {
            gprs,
            extended_state: [0; 512],
        }
    }
}

impl Default for Registers {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Lifecycle state of a fiber.
#[repr(u32)]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiberStatus {
    /// The fiber is newly created and has not yet been polled.
    Initial = 0,
    /// The fiber is currently being executed by a worker core.
    Running = 1,
    /// The fiber is suspended and waiting for an event (e.g. I/O or Mutex).
    Yielded = 2,
    /// The fiber has successfully completed its execution.
    Finished = 3,
    /// Terminated due to an unhandled panic.
    Panicked = 4,
    /// The fiber was woken up by a waker.
    Notified = 5,
    /// The fiber is currently transitioning to a suspended state.
    Suspending = 6,
}

/// The hardware-level execution context for a stackful fiber.
///
/// This structure is strictly aligned to 64 bytes to ensure that all
/// register state and future data reside within a single cache line (or contiguous lines)
/// to minimize L1/L2 misses during context switches.
#[repr(C, align(64))]
#[doc(hidden)]
pub struct FiberContext {
    /// Standard CPU registers (rax, rbx, etc.)
    pub regs: Registers,
    /// Return address for the scheduler dispatch loop.
    pub executor_regs: Registers,
    /// Fiber identification index.
    pub fiber_index: u32,
    /// The OS thread ID where this fiber was last executed.
    pub last_os_thread_id: u64,
    /// The hardware core ID where this fiber was originally spawned.
    pub origin_core: u16,
    /// Current execution state.
    pub state: AtomicU32,
    /// Pointer to the assembly context-switch function.
    pub switch_fn: unsafe extern "C" fn(*mut Registers, *const Registers),
    /// Pointer to the fiber's entry-point closure or future.
    pub closure_ptr: *mut (),
    /// Trampoline address for C-FFI or Rust closure invocation.
    pub trampoline: unsafe extern "C" fn(),
    /// Internal wrapper to drive the closure or poll the future.
    pub invoke_closure: fn(*mut ()),
    /// Optional cleanup callback (used for C-FFI ownership management).
    pub cleanup_fn: Option<unsafe extern "C" fn(*mut ())>,
    /// Pointer to the fiber's 8KB read/stack buffer.
    pub read_buffer_ptr: *mut u8,
    /// Metadata: Workload Hint.
    pub kind: WorkloadKind,
    /// Metadata: Topology Strategy.
    pub mode: TopologyMode,
    /// Metadata: Core Affinity Hint for wake routing.
    pub affinity: crate::api::topology::Affinity,
    /// Statistics: Adaptive Spin Budget.
    pub adaptive_spin_count: u32,
    /// Statistics: Recent Spin Failures.
    pub spin_failure_count: u32,

    /// Current stack pointer for this fiber.
    pub(crate) stack_ptr: usize,
    /// Saved stack pointer of the executor thread.
    pub(crate) scheduler_stack_ptr: usize,
    /// OS-specific TIB stack limit (Windows).
    pub(crate) tib_stack_limit: usize,
    /// OS-specific TIB stack base (Windows).
    pub(crate) tib_stack_base: usize,
    /// Thread ID of a non-fiber waiter (for C-FFI join).
    pub(crate) waiter_thread_id: AtomicU64,
    /// Handle of a fiber waiter (for C-FFI join).
    pub(crate) waiter_handle: AtomicU64,
    /// Generation counter to prevent ABA in handles.
    pub(crate) generation: AtomicU32,
    /// Link to the next available context in the free list.
    pub(crate) next_free: AtomicU32,
    /// Pointer to panic payload if the fiber crashed.
    pub(crate) panic_payload_ptr: *mut (),
    /// Pointer to the result of the fiber.
    pub(crate) result_ptr: *mut (),
    /// Opaque pointer for reader bridge.
    pub(crate) reader_ptr: *mut (),
    /// Reference to a shared buffer.
    pub(crate) buf_ptr: *mut [u8],
}

impl FiberContext {
    /// Creates a new, blank `FiberContext`.
    ///
    /// `const fn` on normal builds; plain `fn` under `cfg(loom)` because
    /// loom's atomic types do not have `const` constructors.
    #[cfg(loom)]
    pub fn new() -> Self {
        Self {
            stack_ptr: 0,
            scheduler_stack_ptr: 0,
            tib_stack_limit: 0,
            tib_stack_base: 0,
            state: AtomicU32::new(FiberStatus::Initial as u32),
            kind: WorkloadKind::Compute,
            mode: TopologyMode::P2PMesh,
            affinity: crate::api::topology::Affinity::SameCore,
            origin_core: 0,
            fiber_index: 0,
            waiter_thread_id: AtomicU64::new(0),
            waiter_handle: AtomicU64::new(0),
            generation: AtomicU32::new(0),
            regs: Registers::new(),
            executor_regs: Registers::new(),
            next_free: AtomicU32::new(u32::MAX),
            panic_payload_ptr: core::ptr::null_mut(),
            trampoline: dummy_trampoline,
            invoke_closure: dummy_invoke,
            closure_ptr: core::ptr::null_mut(),
            result_ptr: core::ptr::null_mut(),
            reader_ptr: core::ptr::null_mut(),
            buf_ptr: core::ptr::slice_from_raw_parts_mut(core::ptr::null_mut(), 0),
            read_buffer_ptr: core::ptr::null_mut(),
            switch_fn: crate::context_switch::switch_context_cross_thread_float,
            cleanup_fn: None,
            adaptive_spin_count: 50,
            spin_failure_count: 0,
            last_os_thread_id: 0,
        }
    }
    /// Creates a new, blank `FiberContext`.
    ///
    /// `const fn` on normal builds; plain `fn` under `cfg(loom)` because
    /// loom's atomic types do not have `const` constructors.
    #[cfg(not(loom))]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            stack_ptr: 0,
            scheduler_stack_ptr: 0,
            tib_stack_limit: 0,
            tib_stack_base: 0,
            state: AtomicU32::new(FiberStatus::Initial as u32),
            kind: WorkloadKind::Compute,
            mode: TopologyMode::P2PMesh,
            affinity: crate::api::topology::Affinity::SameCore,
            origin_core: 0,
            fiber_index: 0,
            waiter_thread_id: AtomicU64::new(0),
            waiter_handle: AtomicU64::new(0),
            generation: AtomicU32::new(0),
            regs: Registers::new(),
            executor_regs: Registers::new(),
            next_free: AtomicU32::new(u32::MAX),
            panic_payload_ptr: core::ptr::null_mut(),
            trampoline: dummy_trampoline,
            invoke_closure: dummy_invoke,
            closure_ptr: core::ptr::null_mut(),
            result_ptr: core::ptr::null_mut(),
            reader_ptr: core::ptr::null_mut(),
            buf_ptr: core::ptr::slice_from_raw_parts_mut(core::ptr::null_mut(), 0),
            read_buffer_ptr: core::ptr::null_mut(),
            switch_fn: crate::context_switch::switch_context_cross_thread_float,
            cleanup_fn: None,
            adaptive_spin_count: 50,
            spin_failure_count: 0,
            last_os_thread_id: 0,
        }
    }
}

#[cfg(not(loom))]
impl Default for FiberContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(loom)]
impl Default for FiberContext {
    fn default() -> Self {
        Self::new()
    }
}

const unsafe extern "C" fn dummy_trampoline() {}
const fn dummy_invoke(_: *mut ()) {}

/// A page-aligned arena for managing fiber stacks and control blocks.
///
/// The `ContextPool` ensures O(1) allocation and hardware-level isolation
/// through tiered safety levels and OS memory protection primitives.
#[allow(dead_code)]
pub struct ContextPool {
    base_ptr: *mut u8,
    total_size: usize,
    /// Size of each context slot in bytes.
    pub slot_size: usize,
    /// OS page size resolved at construction time. macOS arm64 uses 16 KiB
    /// pages, so this must be queried via `sysconf` rather than hardcoded —
    /// the slot layout depends on it for guard-page offsets.
    page_size: usize,
    #[allow(dead_code)]
    capacity: u32,
    safety: SafetyLevel,
    free_head: AtomicU64,
    /// Byte offset from slot start to the `FiberContext` within each slot.
    ///
    /// Pre-computed once as `slot_size − ceil_align(size_of::<FiberContext>(), 64)`
    /// and cached here so `get_context_ptr` — called on every fiber dispatch —
    /// never re-executes the alignment arithmetic at runtime.
    pub context_end_offset: usize,
}

unsafe impl Send for ContextPool {}
unsafe impl Sync for ContextPool {}

impl ContextPool {
    /// Creates a new `ContextPool` with the specified capacity and safety.
    ///
    /// This function performs the initial bulk allocation (via mmap or
    /// `VirtualAlloc`) and configures any requested hardware guard pages.
    ///
    /// # Errors
    /// Returns an error if the OS fails to allocate the requested memory region
    /// or if hardware protection cannot be applied to the guard pages.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[inline(never)]
    pub fn new(
        capacity: u32,
        stack_size: usize,
        safety: SafetyLevel,
        numa: usize,
    ) -> Result<Self, &'static str> {
        #[cfg(unix)]
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        #[cfg(windows)]
        let page_size = unsafe {
            let mut info = core::mem::zeroed();
            windows_sys::Win32::System::SystemInformation::GetSystemInfo(&raw mut info);
            info.dwPageSize as usize
        };

        #[allow(clippy::items_after_statements)]
        const ALIGN: usize = 64;
        let context_sz = (core::mem::size_of::<FiberContext>() + ALIGN - 1) & !(ALIGN - 1);

        // Slot Size: [ Stack Space | 8KB Read Buffer | FiberContext ]
        let slot_size = (stack_size + context_sz + 8192 + page_size - 1) & !(page_size - 1);
        // Pre-compute the intra-slot byte offset to FiberContext, eliminating
        // a subtract on every get_context_ptr call in the dispatch hot path.
        let context_end_offset = slot_size - context_sz;

        let total_size = match safety {
            SafetyLevel::Safety0 => capacity as usize * slot_size,
            SafetyLevel::Safety1 => {
                let num_groups = (capacity as usize).div_ceil(32);
                capacity as usize * slot_size + num_groups * page_size
            }
            SafetyLevel::Safety2 => capacity as usize * (slot_size + page_size),
        };

        // Add 4KB for SEH/Metadata
        let total_size_with_meta = total_size + 4096;

        unsafe {
            let base_ptr = Self::allocate_arena(total_size_with_meta, safety, numa)?;

            // PRE-PROTECT Guard Pages
            if safety == SafetyLevel::Safety1 {
                for i in 0..capacity.div_ceil(32) {
                    let guard_ptr = base_ptr.add(i as usize * (slot_size * 32 + page_size));
                    Self::apply_hardware_protection(guard_ptr, page_size)?;
                }
            } else if safety == SafetyLevel::Safety2 {
                for i in 0..capacity {
                    let guard_ptr = base_ptr.add(i as usize * (slot_size + page_size));
                    Self::apply_hardware_protection(guard_ptr, page_size)?;
                }
            }

            let pool = Self {
                base_ptr,
                total_size: total_size_with_meta,
                slot_size,
                page_size,
                capacity,
                safety,
                free_head: AtomicU64::new(0),
                context_end_offset,
            };

            for i in 0..capacity {
                let ctx_ptr = pool.get_context_ptr(i);
                core::ptr::write(ctx_ptr, FiberContext::new());
                (*ctx_ptr).fiber_index = i;

                // Robust Aligned Read Buffer (64-byte aligned)
                let raw_read_buf = ctx_ptr.cast::<u8>().sub(8192);
                (*ctx_ptr).read_buffer_ptr = (raw_read_buf as usize & !63) as *mut u8;

                (*ctx_ptr).next_free.store(i + 1, Ordering::Relaxed);
            }

            let last_ctx = pool.get_context_ptr(capacity - 1);
            (*last_ctx).next_free.store(u32::MAX, Ordering::Relaxed);

            // Windows SEH Registration
            #[cfg(windows)]
            {
                use windows_sys::Win32::System::Diagnostics::Debug::{
                    IMAGE_RUNTIME_FUNCTION_ENTRY, RtlAddFunctionTable,
                };

                #[repr(C, packed)]
                struct UnwindInfo {
                    version_and_flags: u8,
                    prolog_size: u8,
                    unwind_code_count: u8,
                    frame_register_and_offset: u8,
                }

                let meta_base = base_ptr.add(total_size);
                let unwind_info_ptr = meta_base.cast::<UnwindInfo>();
                core::ptr::write(
                    unwind_info_ptr,
                    UnwindInfo {
                        version_and_flags: 0x01,
                        prolog_size: 0,
                        unwind_code_count: 0,
                        frame_register_and_offset: 0,
                    },
                );

                #[allow(clippy::cast_ptr_alignment)]
                let function_table_ptr = meta_base
                    .add(core::mem::size_of::<UnwindInfo>())
                    .cast::<IMAGE_RUNTIME_FUNCTION_ENTRY>();
                core::ptr::write(
                    function_table_ptr,
                    IMAGE_RUNTIME_FUNCTION_ENTRY {
                        BeginAddress: 0,
                        EndAddress: total_size as u32,
                        Anonymous: windows_sys::Win32::System::Diagnostics::Debug::IMAGE_RUNTIME_FUNCTION_ENTRY_0 {
                            UnwindData: (unwind_info_ptr as usize - base_ptr as usize) as u32,
                        },
                    },
                );

                let base = base_ptr as usize;
                RtlAddFunctionTable(function_table_ptr.cast_const(), 1, base as u64);
            }

            Ok(pool)
        }
    }

    #[inline(always)]
    fn apply_hardware_protection(ptr: *mut u8, size: usize) -> Result<(), &'static str> {
        #[cfg(unix)]
        unsafe {
            if libc::mprotect(ptr.cast(), size, libc::PROT_NONE) != 0 {
                return Err("mprotect failed");
            }
        }
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, VirtualProtect};
            let mut old = 0;
            if VirtualProtect(ptr.cast(), size, PAGE_NOACCESS, &raw mut old) == 0 {
                return Err("VirtualProtect failed");
            }
        }
        Ok(())
    }

    #[inline(always)]
    #[allow(clippy::useless_let_if_seq)]
    #[allow(clippy::cast_possible_truncation)]
    unsafe fn allocate_arena(
        size: usize,
        safety: SafetyLevel,
        numa: usize,
    ) -> Result<*mut u8, &'static str> {
        unsafe {
            #[cfg(unix)]
            {
                let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
                let mut ptr = libc::MAP_FAILED;

                // Try HugeTLB for Safety0 (best perf), fall back to standard pages
                if safety == SafetyLevel::Safety0 {
                    ptr = unsafe {
                        libc::mmap(
                            core::ptr::null_mut(),
                            size,
                            libc::PROT_READ | libc::PROT_WRITE,
                            flags | 0x40000, // MAP_HUGETLB
                            -1,
                            0,
                        )
                    };
                }

                if ptr == libc::MAP_FAILED {
                    // Add MAP_NORESERVE so virtual mapping succeeds under
                    // strict overcommit accounting (containers, QEMU CI).
                    // Physical pages are demand-faulted on first write —
                    // exactly what we want for a sparsely-used context arena.
                    ptr = unsafe {
                        libc::mmap(
                            core::ptr::null_mut(),
                            size,
                            libc::PROT_READ | libc::PROT_WRITE,
                            flags | libc::MAP_NORESERVE,
                            -1,
                            0,
                        )
                    };
                }

                if ptr == libc::MAP_FAILED {
                    return Err("mmap failed");
                }

                // Linux-only: hint THP for the arena if we fell back to plain
                // mmap (Safety1/2 or HUGETLB-exhausted Safety0).  Reduces TLB
                // misses across the lifetime of the runtime.  Safety0 with
                // successful HUGETLB already gets explicit huge pages.
                #[cfg(target_os = "linux")]
                {
                    const MADV_HUGEPAGE: libc::c_int = 14;
                    libc::madvise(ptr, size, MADV_HUGEPAGE);
                }

                #[cfg(target_os = "linux")]
                if numa > 0 {
                    let mask: usize = 1 << (numa % 64);
                    // MPOL_BIND = 2
                    libc::syscall(libc::SYS_mbind, ptr, size, 2, &raw const mask, 64, 0);
                }

                Ok(ptr.cast::<u8>())
            }
            #[cfg(windows)]
            {
                use windows_sys::Win32::System::Memory::{
                    MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
                };
                let mut flags = MEM_COMMIT | MEM_RESERVE;
                if safety == SafetyLevel::Safety0 {
                    flags |= 0x2000_0000;
                } // MEM_LARGE_PAGES

                let mut ptr = if numa > 0 {
                    windows_sys::Win32::System::Memory::VirtualAllocExNuma(
                        windows_sys::Win32::System::Threading::GetCurrentProcess(),
                        core::ptr::null_mut(),
                        size,
                        flags,
                        PAGE_READWRITE,
                        numa as u32,
                    )
                } else {
                    VirtualAlloc(core::ptr::null_mut(), size, flags, PAGE_READWRITE)
                };

                if ptr.is_null() && (flags & 0x2000_0000) != 0 {
                    let fallback_flags = flags & !0x2000_0000;
                    ptr = if numa > 0 {
                        windows_sys::Win32::System::Memory::VirtualAllocExNuma(
                            windows_sys::Win32::System::Threading::GetCurrentProcess(),
                            core::ptr::null_mut(),
                            size,
                            fallback_flags,
                            PAGE_READWRITE,
                            numa as u32,
                        )
                    } else {
                        VirtualAlloc(core::ptr::null_mut(), size, fallback_flags, PAGE_READWRITE)
                    };
                }

                if ptr.is_null() {
                    return Err("VirtualAlloc failed");
                }
                Ok(ptr.cast::<u8>())
            }
        }
    }

    /// Returns a raw pointer to a context based on its index.
    ///
    /// Hot path: called once per fiber dispatch. `context_end_offset` is
    /// pre-computed at pool construction so this function executes a single
    /// multiply + two adds + one pointer cast — no alignment arithmetic.
    ///
    /// Guard-page offsets use the page size captured at construction so the
    /// layout is always consistent with `new()`, even on platforms where the
    /// OS page size is not 4 KiB (e.g. macOS arm64 = 16 KiB).
    #[inline(always)]
    pub const fn get_context_ptr(&self, index: u32) -> *mut FiberContext {
        let guard_offset = match self.safety {
            SafetyLevel::Safety0 => 0,
            // `>> 5` == `/ 32` for the group index; avoids a division on every call.
            SafetyLevel::Safety1 => ((index as usize >> 5) + 1) * self.page_size,
            SafetyLevel::Safety2 => (index as usize + 1) * self.page_size,
        };

        unsafe {
            #[allow(clippy::cast_ptr_alignment)]
            self.base_ptr
                .add(index as usize * self.slot_size + guard_offset + self.context_end_offset)
                .cast::<FiberContext>()
        }
    }

    /// O(1) Pop from the free list with ABA protection.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn alloc_context(&self) -> Option<u32> {
        let mut head = self.free_head.load(Ordering::Acquire);
        loop {
            let index = head as u32;
            let r#gen = (head >> 32) as u32;
            if index == u32::MAX {
                return None;
            }

            let ctx = self.get_context_ptr(index);
            let next = unsafe { (*ctx).next_free.load(Ordering::Relaxed) };

            let new_head = (u64::from(r#gen.wrapping_add(1)) << 32) | u64::from(next);

            // Under loom use strong CAS to avoid spurious-failure branch explosion.
            #[cfg(not(loom))]
            let cas = self.free_head.compare_exchange_weak(
                head,
                new_head,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            #[cfg(loom)]
            let cas = self.free_head.compare_exchange(
                head,
                new_head,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            match cas {
                Ok(_) => return Some(index),
                Err(latest) => head = latest,
            }
        }
    }

    /// Returns a context to the free list.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn free_context(&self, index: u32) {
        let ctx = self.get_context_ptr(index);

        // Reset state to Initial and notify any waiting host threads.
        unsafe {
            (*ctx)
                .state
                .store(FiberStatus::Initial as u32, Ordering::Release);
            (*ctx).generation.fetch_add(1, Ordering::AcqRel);
        };

        let mut head = self.free_head.load(Ordering::Relaxed);
        loop {
            let current_idx = head as u32;
            let r#gen = (head >> 32) as u32;
            unsafe { (*ctx).next_free.store(current_idx, Ordering::Relaxed) };
            let new_head = (u64::from(r#gen.wrapping_add(1)) << 32) | u64::from(index);
            // Strong CAS under loom eliminates spurious-failure branches.
            #[cfg(not(loom))]
            let cas = self.free_head.compare_exchange_weak(
                head,
                new_head,
                Ordering::Release,
                Ordering::Relaxed,
            );
            #[cfg(loom)]
            let cas = self.free_head.compare_exchange(
                head,
                new_head,
                Ordering::Release,
                Ordering::Relaxed,
            );
            match cas {
                Ok(_) => break,
                Err(h) => head = h,
            }
        }
    }

    /// Returns the base pointer and layout metadata for direct dispatcher access.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn get_dispatch_layout(&self) -> (*mut u8, usize, usize, usize) {
        // Use the page size captured at construction so the layout the
        // dispatcher sees always matches the layout the arena was built with.
        let page_size = self.page_size;
        let align = 64;
        let context_sz = (core::mem::size_of::<FiberContext>() + align - 1) & !(align - 1);
        let guard_size = if self.safety == SafetyLevel::Safety0 {
            0
        } else {
            page_size
        };
        // context_offset: byte offset within each slot where FiberContext begins
        let context_offset = self.slot_size - context_sz;
        (self.base_ptr, self.slot_size, guard_size, context_offset)
    }
}

impl Drop for ContextPool {
    #[inline(always)]
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.base_ptr.cast(), self.total_size);
        }
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            VirtualFree(self.base_ptr.cast(), 0, MEM_RELEASE);
        }
    }
}

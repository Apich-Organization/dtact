#![allow(unsafe_code)]
#![allow(non_snake_case)]

use core::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};

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
    /// Creates a new, zeroed register set.
    #[must_use]
    #[inline(always)]
    pub(crate) const fn new() -> Self {
        Self {
            gprs: [0; 16],
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[doc(hidden)]
pub enum FiberStatus {
    /// The fiber is initialized but has not yet started.
    Initial = 0,
    /// The fiber is currently executing on a physical core.
    Running = 1,
    /// The fiber is suspended and waiting for an event (e.g. I/O or Mutex).
    Yielded = 2,
    /// The fiber has successfully completed its execution.
    Finished = 3,
    /// Terminated due to an unhandled panic.
    Panicked = 4,
    /// The fiber was woken up by a waker.
    Notified = 5,
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
    pub state: AtomicU8,
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
    pub(crate) const fn new() -> Self {
        Self {
            stack_ptr: 0,
            scheduler_stack_ptr: 0,
            tib_stack_limit: 0,
            tib_stack_base: 0,
            state: AtomicU8::new(FiberStatus::Initial as u8),
            kind: WorkloadKind::Compute,
            mode: TopologyMode::P2PMesh,
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

const unsafe extern "C" fn dummy_trampoline() {}
const fn dummy_invoke(_: *mut ()) {}

/// A page-aligned arena for managing fiber stacks and control blocks.
///
/// The `ContextPool` ensures O(1) allocation and hardware-level isolation
/// through tiered safety levels and OS memory protection primitives.
pub struct ContextPool {
    base_ptr: *mut u8,
    total_size: usize,
    /// Size of each context slot in bytes.
    pub slot_size: usize,
    #[allow(dead_code)]
    capacity: u32,
    safety: SafetyLevel,
    free_head: AtomicU64,
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
            windows_sys::Win32::System::SystemInformation::GetSystemInfo(&mut info);
            info.dwPageSize as usize
        };

        let align = 64;
        let context_sz = (core::mem::size_of::<FiberContext>() + align - 1) & !(align - 1);

        // Slot Size: [ Stack Space | 8KB Read Buffer | FiberContext ]
        let slot_size = (stack_size + context_sz + 8192 + page_size - 1) & !(page_size - 1);

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
                capacity,
                safety,
                free_head: AtomicU64::new(0),
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
                let unwind_info_ptr = meta_base as *mut UnwindInfo;
                core::ptr::write(
                    unwind_info_ptr,
                    UnwindInfo {
                        version_and_flags: 0x01,
                        prolog_size: 0,
                        unwind_code_count: 0,
                        frame_register_and_offset: 0,
                    },
                );

                let function_table_ptr =
                    unwind_info_ptr.add(1) as *mut IMAGE_RUNTIME_FUNCTION_ENTRY;
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
                RtlAddFunctionTable(function_table_ptr as *const _, 1, base as u64);
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
            if VirtualProtect(ptr as *mut _, size, PAGE_NOACCESS, &mut old) == 0 {
                return Err("VirtualProtect failed");
            }
        }
        Ok(())
    }

    #[inline(always)]
    #[allow(clippy::useless_let_if_seq)]
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
                    ptr = unsafe {
                        libc::mmap(
                            core::ptr::null_mut(),
                            size,
                            libc::PROT_READ | libc::PROT_WRITE,
                            flags,
                            -1,
                            0,
                        )
                    };
                }

                if ptr == libc::MAP_FAILED {
                    return Err("mmap failed");
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
                    flags |= 0x20000000;
                } // MEM_LARGE_PAGES

                let ptr = if numa > 0 {
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

                if ptr.is_null() {
                    return Err("VirtualAlloc failed");
                }
                Ok(ptr as *mut u8)
            }
        }
    }

    /// Returns a raw pointer to a context based on its index.
    #[inline(always)]
    pub const fn get_context_ptr(&self, index: u32) -> *mut FiberContext {
        let page_size = 4096;
        let align = 64;
        let context_sz = (core::mem::size_of::<FiberContext>() + align - 1) & !(align - 1);

        let guard_offset = match self.safety {
            SafetyLevel::Safety0 => 0,
            SafetyLevel::Safety1 => (index as usize / 32 + 1) * page_size,
            SafetyLevel::Safety2 => (index as usize + 1) * page_size,
        };

        unsafe {
            let slot_base = self
                .base_ptr
                .add(index as usize * self.slot_size + guard_offset);
            #[allow(clippy::cast_ptr_alignment)]
            slot_base
                .add(self.slot_size - context_sz)
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

            match self.free_head.compare_exchange_weak(
                head,
                new_head,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
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

        // Reset state to Initial and notify any waiting host threads
        unsafe {
            (*ctx)
                .state
                .store(FiberStatus::Initial as u8, Ordering::Release);
            (*ctx).generation.fetch_add(1, Ordering::AcqRel);
            crate::utils::futex_wake(&raw const (*ctx).state);
        };

        let mut head = self.free_head.load(Ordering::Relaxed);
        loop {
            let current_idx = head as u32;
            let r#gen = (head >> 32) as u32;
            unsafe { (*ctx).next_free.store(current_idx, Ordering::Relaxed) };
            let new_head = (u64::from(r#gen.wrapping_add(1)) << 32) | u64::from(index);
            match self.free_head.compare_exchange_weak(
                head,
                new_head,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
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
        #[cfg(unix)]
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        #[cfg(windows)]
        let page_size = unsafe {
            let mut info = core::mem::zeroed();
            windows_sys::Win32::System::SystemInformation::GetSystemInfo(&mut info);
            info.dwPageSize as usize
        };
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
            VirtualFree(self.base_ptr as *mut _, 0, MEM_RELEASE);
        }
    }
}

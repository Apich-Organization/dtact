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

/// Hints to the scheduler about the expected behavior of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadKind {
    /// CPU-bound logic (default).
    Compute,
    /// Heavily blocked on external I/O.
    IO,
    /// Memory-intensive scanning or bulk transfers.
    Memory,
}

/// Topology Strategy for the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyMode {
    /// Peer-to-Peer Mesh: Tasks are deflected to neighbors based on load.
    P2PMesh,
    /// Global: Tasks are shared across all cores via a common pool.
    Global,
}

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
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum FiberStatus {
    /// Not currently allocated.
    Initial = 0,
    /// Actively scheduled or running.
    Running = 1,
    /// Voluntarily yielded to another fiber.
    Yielded = 2,
    /// Execution finished successfully.
    #[allow(dead_code)]
    Finished = 3,
    /// Terminated due to an unhandled panic.
    #[allow(dead_code)]
    Panicked = 4,
}

/// The control block and metadata for a single Dtact Fiber.
///
/// This struct is placed at the top of each context slot, immediately
/// below the 8KB zero-copy buffer.
#[repr(C, align(64))]
pub(crate) struct FiberContext {
    /// Current stack pointer for this fiber.
    pub(crate) stack_ptr: usize,
    /// Saved stack pointer of the executor thread.
    pub(crate) scheduler_stack_ptr: usize,
    /// OS-specific TIB stack limit (Windows).
    pub(crate) tib_stack_limit: usize,
    /// OS-specific TIB stack base (Windows).
    pub(crate) tib_stack_base: usize,
    /// Current execution state.
    pub(crate) state: AtomicU8,
    /// Configured workload kind.
    pub(crate) kind: WorkloadKind,
    /// Configured topology mode.
    pub(crate) mode: TopologyMode,
    /// Core ID where this fiber was originally spawned.
    pub(crate) origin_core: u16,
    /// Unique index in the `ContextPool`.
    pub(crate) fiber_index: u32,
    /// Thread ID of a non-fiber waiter (for C-FFI join).
    pub(crate) waiter_thread_id: AtomicU64,
    /// Reserved GPR state.
    pub(crate) regs: Registers,
    /// Saved executor register state.
    pub(crate) executor_regs: Registers,
    /// Link to the next available context in the free list.
    pub(crate) next_free: AtomicU32,
    /// Pointer to panic payload if the fiber crashed.
    pub(crate) panic_payload_ptr: *mut (),
    /// Assembly entry point.
    pub(crate) trampoline: unsafe extern "C" fn(),
    /// Closure/Future invocation wrapper.
    pub(crate) invoke_closure: unsafe fn(*mut ()),
    /// Pointer to the closure or future.
    pub(crate) closure_ptr: *mut (),
    /// Pointer to the result of the fiber.
    pub(crate) result_ptr: *mut (),
    /// Opaque pointer for reader bridge.
    pub(crate) reader_ptr: *mut (),
    /// Reference to a shared buffer.
    pub(crate) buf_ptr: *mut [u8],
    /// The 8KB Zero-Copy buffer located just below this struct.
    pub(crate) read_buffer_ptr: *mut u8,
    /// Target assembly function for context switching.
    pub(crate) switch_fn: unsafe extern "C" fn(*mut Registers, *const Registers),
    /// Cleanup logic called when the fiber is destroyed.
    pub(crate) cleanup_fn: Option<unsafe extern "C" fn(*mut ())>,
    /// Adaptive spin count for futex synchronization.
    pub(crate) adaptive_spin_count: u32,
    /// Number of consecutive spin failures.
    pub(crate) spin_failure_count: u32,
    /// Last OS thread ID that executed this fiber.
    pub(crate) last_os_thread_id: u64,
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
const unsafe fn dummy_invoke(_: *mut ()) {}

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
    pub fn new(
        capacity: u32,
        stack_size: usize,
        safety: SafetyLevel,
        numa: usize,
    ) -> Result<Self, &'static str> {
        let page_size = 4096;
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
                total_size,
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
                    RUNTIME_FUNCTION, RtlAddFunctionTable,
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

                let function_table_ptr = unwind_info_ptr.add(1) as *mut RUNTIME_FUNCTION;
                core::ptr::write(
                    function_table_ptr,
                    RUNTIME_FUNCTION {
                        BeginAddress: 0,
                        EndAddress: total_size as u32,
                        UnwindData: (unwind_info_ptr as usize - base_ptr as usize) as u32,
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
    unsafe fn allocate_arena(
        size: usize,
        safety: SafetyLevel,
        numa: usize,
    ) -> Result<*mut u8, &'static str> {
        unsafe {
            #[cfg(unix)]
            {
                let mut flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
                if safety == SafetyLevel::Safety0 {
                    flags |= 0x40000;
                } // MAP_HUGETLB
                let ptr = unsafe {
                    libc::mmap(
                        core::ptr::null_mut(),
                        size,
                        libc::PROT_READ | libc::PROT_WRITE,
                        flags,
                        -1,
                        0,
                    )
                };
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
    pub fn get_dispatch_layout(&self) -> (*mut u8, usize, usize) {
        let page_size = 4096;
        let guard_size = if self.safety == SafetyLevel::Safety0 {
            0
        } else {
            page_size
        };
        (self.base_ptr, self.slot_size, guard_size)
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

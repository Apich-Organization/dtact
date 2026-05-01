#![allow(unsafe_code)]
#![allow(clippy::vec_box)]
// For Windows Development traditions, we allow non_snake_case
#![allow(non_snake_case)]
// Please notice this module do not support async operations that evolves AVX-256/AVX-512/AArch64 SVE/RISC-V V, and if evolved, shall be saved by the caller per ABI standards.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::pin::Pin;
use std::arch::naked_asm;
use std::cell::RefCell;
use std::future::Future;
use std::task::Context;
use std::task::Poll;
use std::sync::atomic::AtomicU8;
use std::task::RawWaker;

/// Safety policies for context pool memory layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyLevel {
    /// Allocates large linear memory arenas via Huge-Pages without Guard Pages.
    Safety0,
    /// Organizes every 32 stack slots into a Segment, with an mprotect Guard Page at the end.
    Safety1,
    /// Each stack slot occupies a dedicated virtual memory page with its own independent Guard Page.
    Safety2,
}

/// Machine-specific registers for context switching.
#[repr(C, align(64))]
#[derive(Debug)]
pub(crate) struct Registers {
    /// General purpose registers.
    pub(crate) gprs: [u64; 16],
    /// Extended state (e.g., FPU/SIMD for `x86_64`, NEON for Aarch64, fp for RISC-V).
    pub(crate) extended_state: [u8; 512],
}

impl Registers {
    /// Create a zero-initialized register state.
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

/// Represents the current execution status of a Fiber.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum FiberStatus {
    Initial = 0,
    Running = 1,
    Yielded = 2,
    Finished = 3,
    Panicked = 4,
}

pub(crate) enum StackMemory {
    Safety0 {
        base: *mut u8,
        usable_size: usize,
    },
    Safety1 {
        base: *mut u8,
        usable_size: usize,
        is_guard: bool,
    },
    Safety2(GuardedStack),
}

pub(crate) struct ContextPool {
    safety: SafetyLevel,
    usable_size: usize,
    free_list: crossbeam::queue::ArrayQueue<Box<FiberContext>>, // Assuming lock-free queue is needed, we'll use a simple approach here for demonstration if crossbeam isn't available, but the docs mention lock-free free-list.
}

impl StackMemory {
    #[cfg(unix)]
    pub(crate) fn allocate_safety0(usable_size: usize) -> Result<Self, crate::error::DecodeError> {
        let usable_size = (usable_size + page_size() - 1) & !(page_size() - 1);
        unsafe {
            // Use huge pages if possible (MAP_HUGETLB is Linux specific, but we map anonymous memory)
            let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | 0x40000; // MAP_HUGETLB is usually 0x40000
            let base = libc::mmap(
                core::ptr::null_mut(),
                usable_size,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                -1,
                0,
            );
            if base == libc::MAP_FAILED {
                // Fallback to normal pages
                let base = libc::mmap(
                    core::ptr::null_mut(),
                    usable_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                );
                if base == libc::MAP_FAILED {
                    return crate::error::cold_decode_error_other("mmap failed for safety 0");
                }
                return Ok(Self::Safety0 {
                    base: base.cast::<u8>(),
                    usable_size,
                });
            }
            Ok(Self::Safety0 {
                base: base.cast::<u8>(),
                usable_size,
            })
        }
    }

    #[cfg(windows)]
    pub(crate) fn allocate_safety0(usable_size: usize) -> Result<Self, crate::error::DecodeError> {
        let usable_size = (usable_size + page_size() - 1) & !(page_size() - 1);
        unsafe {
            let base = winapi_shim::VirtualAlloc(
                core::ptr::null_mut(),
                usable_size,
                winapi_shim::MEM_COMMIT | winapi_shim::MEM_RESERVE | 0x20000000, // MEM_LARGE_PAGES
                winapi_shim::PAGE_READWRITE,
            );
            if base.is_null() {
                // Fallback to normal pages
                let base = winapi_shim::VirtualAlloc(
                    core::ptr::null_mut(),
                    usable_size,
                    winapi_shim::MEM_COMMIT | winapi_shim::MEM_RESERVE,
                    winapi_shim::PAGE_READWRITE,
                );
                if base.is_null() {
                    return crate::error::cold_decode_error_other("VirtualAlloc failed for safety 0");
                }
                return Ok(Self::Safety0 {
                    base: base.cast::<u8>(),
                    usable_size,
                });
            }
            Ok(Self::Safety0 {
                base: base.cast::<u8>(),
                usable_size,
            })
        }
    }

    #[inline(always)]
    pub(crate) fn top(&self) -> u64 {
        match self {
            Self::Safety0 { base, usable_size } => {
                let raw = *base as u64 + *usable_size as u64;
                raw & !15
            }
            Self::Safety1 { base, usable_size, .. } => {
                let raw = *base as u64 + *usable_size as u64;
                raw & !15
            }
            Self::Safety2(stack) => stack.top(),
        }
    }

    #[inline(always)]
    pub(crate) fn bottom(&self) -> u64 {
        match self {
            Self::Safety0 { base, .. } => *base as u64,
            Self::Safety1 { base, .. } => *base as u64,
            Self::Safety2(stack) => stack.bottom(),
        }
    }

    #[inline(always)]
    pub(crate) fn allocation_base(&self) -> u64 {
        match self {
            Self::Safety0 { base, .. } => *base as u64,
            Self::Safety1 { base, .. } => *base as u64,
            Self::Safety2(stack) => stack.allocation_base(),
        }
    }
}

/// A fiber stack with an `mmap`-backed guard page at the bottom.
#[allow(dead_code)]
pub(crate) struct GuardedStack {
    base: *mut u8,
    total_len: usize,
    page_size: usize,
}

impl GuardedStack {
    #[inline]
    #[cfg(unix)]
    pub(crate) fn new(usable_size: usize) -> core::result::Result<Self, crate::error::DecodeError> {
        let page_size = page_size();
        let usable_size = (usable_size + page_size - 1) & !(page_size - 1);
        let total_len = page_size + usable_size;

        unsafe {
            let base = libc::mmap(
                core::ptr::null_mut(),
                total_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            if base == libc::MAP_FAILED {
                return crate::error::cold_decode_error_other("mmap failed for fiber stack");
            }

            let rc = libc::mprotect(base, page_size, libc::PROT_NONE);
            if rc != 0 {
                libc::munmap(base, total_len);
                return crate::error::cold_decode_error_other("mprotect failed for guard page");
            }

            Ok(Self {
                base: base.cast::<u8>(),
                total_len,
                page_size,
            })
        }
    }

    #[inline]
    #[cfg(windows)]
    pub(crate) fn new(usable_size: usize) -> core::result::Result<Self, crate::error::DecodeError> {
        let page_size = page_size();
        let usable_size = (usable_size + page_size - 1) & !(page_size - 1);
        let total_len = page_size + usable_size;

        unsafe {
            let base = winapi_shim::VirtualAlloc(
                core::ptr::null_mut(),
                total_len,
                winapi_shim::MEM_COMMIT | winapi_shim::MEM_RESERVE,
                winapi_shim::PAGE_READWRITE,
            );
            if base.is_null() {
                return crate::error::cold_decode_error_other(
                    "VirtualAlloc failed for fiber stack",
                );
            }

            let mut old_protect = 0;
            let rc = winapi_shim::VirtualProtect(
                base,
                page_size,
                winapi_shim::PAGE_NOACCESS,
                &mut old_protect,
            );
            if rc == 0 {
                winapi_shim::VirtualFree(base, 0, winapi_shim::MEM_RELEASE);
                return crate::error::cold_decode_error_other(
                    "VirtualProtect failed for guard page",
                );
            }

            Ok(Self {
                base: base.cast::<u8>(),
                total_len,
                page_size,
            })
        }
    }

    #[inline(always)]
    #[must_use]
    #[allow(dead_code)]
    pub(crate) const fn usable(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(
                self.base.add(self.page_size),
                self.total_len - self.page_size,
            )
        }
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) const fn usable_mut(&mut self) -> &mut [u8] {
        unsafe {
            core::slice::from_raw_parts_mut(
                self.base.add(self.page_size),
                self.total_len - self.page_size,
            )
        }
    }

    #[inline(always)]
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn top(&self) -> u64 {
        let raw = self.base as u64 + self.total_len as u64;
        raw & !15
    }

    #[inline(always)]
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn bottom(&self) -> u64 {
        self.base as u64 + self.page_size as u64
    }

    #[inline(always)]
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn allocation_base(&self) -> u64 {
        self.base as u64
    }
}

impl Drop for GuardedStack {
    #[inline(always)]
    #[cfg(unix)]
    fn drop(&mut self) {
        unsafe {
            let rc = libc::munmap(self.base.cast::<libc::c_void>(), self.total_len);
            debug_assert!(rc == 0, "munmap failed for fiber stack");
        }
    }

    #[inline(always)]
    #[cfg(windows)]
    fn drop(&mut self) {
        unsafe {
            let rc = winapi_shim::VirtualFree(
                self.base.cast::<core::ffi::c_void>(),
                0,
                winapi_shim::MEM_RELEASE,
            );
            debug_assert!(rc != 0, "VirtualFree failed for fiber stack");
        }
    }
}

unsafe impl Send for GuardedStack {}

#[cfg(unix)]
#[inline(always)]
fn page_size() -> usize {
    static PAGE_SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGE_SIZE.get_or_init(|| unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize })
}

#[cfg(windows)]
#[inline(always)]
fn page_size() -> usize {
    static PAGE_SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGE_SIZE.get_or_init(|| {
        let mut info = core::mem::MaybeUninit::uninit();
        unsafe {
            winapi_shim::GetSystemInfo(info.as_mut_ptr());
            info.assume_init().dwPageSize as usize
        }
    })
}

#[cfg(windows)]
mod winapi_shim {
    #[repr(C)]
    pub(crate) struct SYSTEM_INFO {
        pub(crate) wProcessorArchitecture: u16,
        pub(crate) wReserved: u16,
        pub(crate) dwPageSize: u32,
        pub(crate) lpMinimumApplicationAddress: *mut core::ffi::c_void,
        pub(crate) lpMaximumApplicationAddress: *mut core::ffi::c_void,
        pub(crate) dwActiveProcessorMask: usize,
        pub(crate) dwNumberOfProcessors: u32,
        pub(crate) dwProcessorType: u32,
        pub(crate) dwAllocationGranularity: u32,
        pub(crate) wProcessorLevel: u16,
        pub(crate) wProcessorRevision: u16,
    }
    pub(crate) const MEM_COMMIT: u32 = 0x00001000;
    pub(crate) const MEM_RESERVE: u32 = 0x00002000;
    pub(crate) const MEM_RELEASE: u32 = 0x00008000;
    pub(crate) const PAGE_NOACCESS: u32 = 0x01;
    pub(crate) const PAGE_READWRITE: u32 = 0x04;

    unsafe extern "system" {
        pub(crate) fn VirtualAlloc(
            lpAddress: *mut core::ffi::c_void,
            dwSize: usize,
            flAllocationType: u32,
            flProtect: u32,
        ) -> *mut core::ffi::c_void;
        pub(crate) fn VirtualFree(
            lpAddress: *mut core::ffi::c_void,
            dwSize: usize,
            dwFreeType: u32,
        ) -> i32;
        pub(crate) fn VirtualProtect(
            lpAddress: *mut core::ffi::c_void,
            dwSize: usize,
            flNewProtect: u32,
            lpflOldProtect: *mut u32,
        ) -> i32;
        pub(crate) fn GetSystemInfo(lpSystemInfo: *mut SYSTEM_INFO);
    }
}

/// Context metadata for an executing fiber, managing stacks, registers, and closure passing.
#[repr(C, align(64))]
pub(crate) struct FiberContext {
    /// RSP at the time of suspension
    pub(crate) stack_ptr: usize,
    /// Scheduler RSP before entering this Fiber
    pub(crate) scheduler_stack_ptr: usize,
    /// Ready, Running, Waiting, Dead
    pub(crate) state: AtomicU8,
    /// Topology Information
    pub(crate) origin_core: u16,
    pub(crate) fiber_index: u32,
    /// Custom P2P wakeup logic implemented via RawWaker
    pub(crate) static_raw_waker: RawWaker,

    /// The memory region mapping the stack space in memory.
    pub(crate) stack: StackMemory,
    /// Registers for the fiber state.
    pub(crate) regs: Registers,
    /// Registers for the executor state (where the fiber yields back to).
    pub(crate) executor_regs: Registers,
    /// The lifecycle status of the fiber.
    pub(crate) status: FiberStatus,
    /// Holds panic payload if a panic occurred within the fiber.
    pub(crate) panic_payload: Option<Box<dyn std::any::Any + Send>>,
    /// Fixed landing assembly to launch closures.
    pub(crate) trampoline: unsafe extern "C" fn(),
    /// Closure thunk invocation pointer.
    pub(crate) invoke_closure: unsafe fn(*mut ()),
    /// Opaque pointer to closure state.
    pub(crate) closure_ptr: *mut (),
    /// Opaque pointer to the result slot.
    pub(crate) result_ptr: *mut (),
    /// Opaque pointer to the `AsyncRead` structure.
    pub(crate) reader_ptr: *mut (),
    /// Byte slice actively being used as data source for parsing.
    pub(crate) buf_ptr: *mut [u8],
    /// 8KB heap-allocated IO staging buffer.
    pub(crate) read_buffer: Box<[u8]>,
}

std::thread_local! {
    static CONTEXT_POOL: RefCell<Vec<Box<FiberContext>>> = const { RefCell::new(Vec::new()) };
    static CURRENT_FIBER: std::cell::Cell<*mut FiberContext> = const { std::cell::Cell::new(core::ptr::null_mut()) };
}

const MAX_POOLED_CONTEXTS: usize = 8_192;

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
    /// Fiber has been allocated but not yet started executing.
    Initial,
    /// Fiber is currently executing or ready to execute.
    Running,
    /// Fiber was suspended while waiting for additional async I/O data.
    Yielded,
    /// Fiber has completed its closure execution.
    Finished,
    /// Fiber experienced a panic that was caught.
    Panicked,
}

/// A fiber stack with an `mmap`-backed guard page at the bottom.
///
/// Layout (low → high):
/// ```text
/// [ guard page  |  usable stack memory ]
///   PAGE_SIZE       stack_size
/// ```
///
/// The guard page is mapped `PROT_NONE`, so any access (stack overflow) will
/// trigger a hardware fault (SIGSEGV / SIGBUS) instead of silently corrupting
/// adjacent heap memory.
#[allow(dead_code)]
pub(crate) struct GuardedStack {
    /// Base pointer returned by `mmap` (start of guard page).
    base: *mut u8,
    /// Total allocation length (guard + usable).
    total_len: usize,
    /// Page size used for the guard.
    page_size: usize,
}

impl GuardedStack {
    /// Allocate a new guarded stack of *at least* `usable_size` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `mmap` or `mprotect` fails.
    #[inline]
    #[cfg(unix)]
    pub(crate) fn new(usable_size: usize) -> core::result::Result<Self, crate::error::DecodeError> {
        let page_size = page_size();
        // Round usable_size up to page boundary.
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

            // Protect the first page as a guard (PROT_NONE → any access faults).
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

    /// Allocate a new guarded stack on Windows.
    ///
    /// # Errors
    ///
    /// Returns an error if `VirtualAlloc` or `VirtualProtect` fails.
    #[inline]
    #[cfg(windows)]
    pub(crate) fn new(usable_size: usize) -> core::result::Result<Self, crate::error::DecodeError> {
        let page_size = page_size();
        let usable_size = (usable_size + page_size - 1) & !(page_size - 1);
        let total_len = page_size + usable_size;

        unsafe {
            // Reserve and commit the entire stack
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

            // Guard the first page
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

    /// Usable stack region (excludes guard page).
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

    /// Usable stack region (mutable).
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

    /// The top of the usable stack (highest address), 16-byte aligned.
    #[inline(always)]
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn top(&self) -> u64 {
        let raw = self.base as u64 + self.total_len as u64;
        raw & !15 // 16-byte align
    }

    /// The bottom of the usable stack (lowest usable address, just above guard page).
    #[inline(always)]
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn bottom(&self) -> u64 {
        self.base as u64 + self.page_size as u64
    }

    /// The absolute base of the mapping (potentially including guard page).
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
                0, // dwSize must be 0 for MEM_RELEASE
                winapi_shim::MEM_RELEASE,
            );
            debug_assert!(rc != 0, "VirtualFree failed for fiber stack");
        }
    }
}

// GuardedStack owns a unique mmap region — safe to move between threads.
unsafe impl Send for GuardedStack {}

#[cfg(unix)]
#[inline(always)]
fn page_size() -> usize {
    // Cached via a static to avoid repeated syscalls.
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
#[repr(C, align(16))]
pub(crate) struct FiberContext {
    /// The `GuardedStack` which maps the actual stack space in memory.
    pub(crate) stack: GuardedStack,
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

// FiberContext is intentionally NOT Send/Sync.
// It is only accessed through BridgeFuture, which manually implements
// Send/Sync with the correct safety invariants.
std::thread_local! {
    static CONTEXT_POOL: RefCell<Vec<Box<FiberContext>>> = const { RefCell::new(Vec::new()) };
    static CURRENT_FIBER: std::cell::Cell<*mut FiberContext> = const { std::cell::Cell::new(core::ptr::null_mut()) };
}

/// Cap the number of contexts pooled per thread to prevent unbounded mmap accumulation.
const MAX_POOLED_CONTEXTS: usize = 8_192;



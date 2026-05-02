#![allow(unsafe_code)]
#![allow(non_snake_case)]

use core::sync::atomic::{AtomicU32, AtomicU8, Ordering};
use std::task::RawWaker;

/// Safety policies for context pool memory layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyLevel {
    Safety0,
    Safety1,
    Safety2,
}

/// Machine-specific registers for context switching.
#[repr(C, align(64))]
#[derive(Debug)]
pub(crate) struct Registers {
    pub(crate) gprs: [u64; 16],
    pub(crate) extended_state: [u8; 512],
}

impl Registers {
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

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum FiberStatus {
    Initial = 0,
    Running = 1,
    Yielded = 2,
    Finished = 3,
    Panicked = 4,
}

/// Context metadata for an executing fiber.
/// MUST be `repr(C)` and its fields matched exactly by inline assembly in the Scheduler!
#[repr(C, align(64))]
pub(crate) struct FiberContext {
    // Offset 0: RSP when fiber is suspended
    pub(crate) stack_ptr: usize,
    // Offset 8: RSP of the scheduler before entering this fiber
    pub(crate) scheduler_stack_ptr: usize,
    
    // Windows TIB Pre-population for __chkstk safety
    pub(crate) tib_stack_limit: usize,
    pub(crate) tib_stack_base: usize,
    
    // Topology & Lock-free states
    pub(crate) state: AtomicU8,
    pub(crate) origin_core: u16,
    pub(crate) fiber_index: u32,
    pub(crate) static_raw_waker: RawWaker,

    pub(crate) regs: Registers,
    pub(crate) executor_regs: Registers,
    
    // Lock-free free-list next pointer
    pub(crate) next_free: AtomicU32,
    
    // Pure POD payload tracking (Eliminates Box/Heap overhead)
    pub(crate) panic_payload_ptr: *mut (),
    pub(crate) trampoline: unsafe extern "C" fn(),
    pub(crate) invoke_closure: unsafe fn(*mut ()),
    pub(crate) closure_ptr: *mut (),
    pub(crate) result_ptr: *mut (),
    pub(crate) reader_ptr: *mut (),
    
    // Pre-allocated IO buffer within the stack space
    pub(crate) buf_ptr: *mut [u8],
    pub(crate) read_buffer_ptr: *mut u8,
}

impl FiberContext {
    pub(crate) const fn new() -> Self {
        Self {
            stack_ptr: 0,
            scheduler_stack_ptr: 0,
            tib_stack_limit: 0,
            tib_stack_base: 0,
            state: AtomicU8::new(FiberStatus::Initial as u8),
            origin_core: 0,
            fiber_index: 0,
            static_raw_waker: RawWaker::new(core::ptr::null(), unsafe { core::mem::transmute(0usize) }),
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
        }
    }
}

unsafe extern "C" fn dummy_trampoline() {}
unsafe fn dummy_invoke(_: *mut ()) {}

/// Global linear memory arena managing all fiber contexts and their stacks.
pub struct ContextPool {
    base_ptr: *mut u8,
    total_size: usize,
    pub slot_size: usize,
    capacity: usize,
    safety: SafetyLevel,
    // Free list using a 64-bit atomic to pack (generation_counter: u32 | index: u32)
    // This absolutely eliminates the ABA problem in lock-free concurrency.
    free_head: core::sync::atomic::AtomicU64,
}

unsafe impl Send for ContextPool {}
unsafe impl Sync for ContextPool {}

impl ContextPool {
    pub fn new(capacity: usize, stack_size: usize, safety: SafetyLevel, num_numa_nodes: usize) -> Result<Self, crate::errors::DtactError> {
        let page_size = page_size();
        let align = 64; 

        // Context size rounded up to 64 bytes
        let context_sz = (core::mem::size_of::<FiberContext>() + align - 1) & !(align - 1);
        let mut slot_size = stack_size + context_sz + 8192; // 8KB read buffer included in slot
        slot_size = (slot_size + page_size - 1) & !(page_size - 1);
        
        if safety == SafetyLevel::Safety2 {
            slot_size += page_size;
        }

        let mut total_size = slot_size * capacity;
        if safety == SafetyLevel::Safety1 {
            total_size += (capacity / 32) * page_size;
        }

        unsafe {
            let base_ptr = Self::allocate_arena(total_size, safety)?;
            
            // NUMA Topology Partitioning: mbind
            Self::apply_numa_mbind(base_ptr, capacity, slot_size, num_numa_nodes)?;
            
            Self::apply_guards(base_ptr, capacity, slot_size, page_size, safety)?;

            // Initialize contexts and build the Lock-Free Free-List
            for i in 0..capacity {
                let slot_base = base_ptr.add(i * slot_size);
                
                // Stack layout: [ Guard (opt) | Stack | 8KB Read Buffer | FiberContext ]
                let ctx_ptr = slot_base.add(slot_size - context_sz) as *mut FiberContext;
                let read_buf_ptr = (ctx_ptr as *mut u8).sub(8192);
                
                let limit = if safety == SafetyLevel::Safety2 || (safety == SafetyLevel::Safety1 && i % 32 == 31) {
                    slot_base.add(page_size) as usize
                } else {
                    slot_base as usize
                };
                let base = read_buf_ptr as usize;

                core::ptr::write(ctx_ptr, FiberContext::new());
                let ctx = &mut *ctx_ptr;
                ctx.fiber_index = i as u32;
                ctx.tib_stack_limit = limit;
                ctx.tib_stack_base = base;
                ctx.read_buffer_ptr = read_buf_ptr;
                
                // Link free list
                ctx.next_free.store((i + 1) as u32, Ordering::Relaxed);
            }

            // Cap the list
            let last_ctx_ptr = base_ptr.add((capacity - 1) * slot_size + slot_size - context_sz) as *mut FiberContext;
            (*last_ctx_ptr).next_free.store(u32::MAX, Ordering::Relaxed);

            Ok(Self {
                base_ptr,
                total_size,
                slot_size,
                capacity,
                safety,
                free_head: core::sync::atomic::AtomicU64::new(0), // generation 0, index 0
            })
        }
    }

    unsafe fn allocate_arena(size: usize, safety: SafetyLevel) -> Result<*mut u8, crate::errors::DtactError> {
        #[cfg(unix)]
        {
            let mut flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
            if safety == SafetyLevel::Safety0 {
                flags |= 0x40000; // MAP_HUGETLB
            }
            
            let mut ptr = unsafe {
                libc::mmap(core::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0)
            };
            if ptr == libc::MAP_FAILED && safety == SafetyLevel::Safety0 {
                ptr = unsafe { libc::mmap(core::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
            }
            if ptr == libc::MAP_FAILED {
                return crate::errors::cold_dtact_error_mmap_failed();
            }
            Ok(ptr as *mut u8)
        }

        #[cfg(windows)]
        {
            let mut flags = winapi_shim::MEM_COMMIT | winapi_shim::MEM_RESERVE;
            if safety == SafetyLevel::Safety0 { flags |= 0x20000000; }
            let mut ptr = winapi_shim::VirtualAlloc(core::ptr::null_mut(), size, flags, winapi_shim::PAGE_READWRITE);
            if ptr.is_null() && safety == SafetyLevel::Safety0 {
                ptr = winapi_shim::VirtualAlloc(core::ptr::null_mut(), size, winapi_shim::MEM_COMMIT | winapi_shim::MEM_RESERVE, winapi_shim::PAGE_READWRITE);
            }
            if ptr.is_null() { return crate::errors::cold_dtact_error_virtual_alloc_failed(); }
            Ok(ptr as *mut u8)
        }
    }

    unsafe fn apply_numa_mbind(base: *mut u8, capacity: usize, slot_size: usize, num_numa_nodes: usize) -> Result<(), crate::errors::DtactError> {
        #[cfg(target_os = "linux")]
        {
            if num_numa_nodes > 1 {
                let slots_per_node = capacity / num_numa_nodes;
                let size_per_node = slots_per_node * slot_size;
                for i in 0..num_numa_nodes {
                    let node_mask: libc::c_ulong = 1 << i;
                    let ptr = unsafe {
                        base.add(i * size_per_node)
                    };
                    // MPOL_BIND = 2
                    unsafe {
                    libc::syscall(
                        libc::SYS_mbind,
                        ptr,
                        size_per_node,
                        2, // MPOL_BIND
                        &node_mask as *const _,
                        core::mem::size_of::<libc::c_ulong>() * 8,
                        0
                    )};
                }
            }
        }
        Ok(())
    }

    unsafe fn apply_guards(base: *mut u8, capacity: usize, slot_size: usize, page_size: usize, safety: SafetyLevel) -> Result<(), crate::errors::DtactError> {
        if safety == SafetyLevel::Safety0 { return Ok(()); }
        
        for i in 0..capacity {
            let guard_ptr = if safety == SafetyLevel::Safety1 {
                if (i % 32) != 31 { continue; }
                // Safety1: Group Guard placed after every 32 slots.
                unsafe { base.add((i + 1) * slot_size + (i / 32) * page_size) }
            } else {
                // Safety2: Individual Guard placed at the end of each slot.
                unsafe { base.add(i * slot_size + slot_size - page_size) }
            };
            
            #[cfg(unix)]
            {
                if unsafe { libc::mprotect(guard_ptr as *mut libc::c_void, page_size, libc::PROT_NONE) } != 0 {
                    return crate::errors::cold_dtact_error_mprotect_failed();
                }
            }
            #[cfg(windows)]
            {
                let mut old_protect = 0;
                if winapi_shim::VirtualProtect(guard_ptr as *mut core::ffi::c_void, page_size, winapi_shim::PAGE_NOACCESS, &mut old_protect) == 0 {
                    return crate::errors::cold_dtact_error_virtual_protect_failed();
                }
            }
        }
        Ok(())
    }

    #[inline(always)]
    pub fn get_dispatch_layout(&self) -> (*mut u8, usize) {
        let align = 64;
        let context_sz = (core::mem::size_of::<FiberContext>() + align - 1) & !(align - 1);
        let first_ctx_ptr = unsafe { self.base_ptr.add(self.slot_size - context_sz) };
        (first_ctx_ptr, self.slot_size)
    }

    /// Lock-Free Allocation (O(1) Pop with ABA protection)
    #[inline(always)]
    pub fn alloc_context(&self) -> Option<u32> {
        let mut current_head = self.free_head.load(Ordering::Acquire);
        loop {
            let index = (current_head & 0xFFFFFFFF) as u32;
            let generation = (current_head >> 32) as u32;
            if index == u32::MAX { return None; }
            
            let ctx_ptr = unsafe {
                let (_, slot_sz) = self.get_dispatch_layout();
                self.base_ptr.add(index as usize * slot_sz + slot_sz - ((core::mem::size_of::<FiberContext>() + 63) & !63)) as *const FiberContext
            };
            let next_index = unsafe { (*ctx_ptr).next_free.load(Ordering::Relaxed) };
            
            // Advance generation counter to prevent ABA vulnerability
            let new_head = ((generation.wrapping_add(1) as u64) << 32) | (next_index as u64);
            
            match self.free_head.compare_exchange_weak(current_head, new_head, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(index),
                Err(h) => current_head = h,
            }
        }
    }

    /// Lock-Free Reclamation (O(1) Push with ABA protection)
    #[inline(always)]
    pub fn free_context(&self, index: u32) {
        let ctx_ptr = unsafe {
            let (_, slot_sz) = self.get_dispatch_layout();
            self.base_ptr.add(index as usize * slot_sz + slot_sz - ((core::mem::size_of::<FiberContext>() + 63) & !63)) as *mut FiberContext
        };
        
        let mut current_head = self.free_head.load(Ordering::Relaxed);
        loop {
            let current_index = (current_head & 0xFFFFFFFF) as u32;
            let generation = (current_head >> 32) as u32;
            
            unsafe { (*ctx_ptr).next_free.store(current_index, Ordering::Relaxed) };
            
            // Advance generation counter on push as well
            let new_head = ((generation.wrapping_add(1) as u64) << 32) | (index as u64);
            
            match self.free_head.compare_exchange_weak(current_head, new_head, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => break,
                Err(h) => current_head = h,
            }
        }
    }
}

impl Drop for ContextPool {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe { libc::munmap(self.base_ptr as *mut libc::c_void, self.total_size); }
        #[cfg(windows)]
        unsafe { winapi_shim::VirtualFree(self.base_ptr as *mut core::ffi::c_void, 0, winapi_shim::MEM_RELEASE); }
    }
}

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
        unsafe { winapi_shim::GetSystemInfo(info.as_mut_ptr()); info.assume_init().dwPageSize as usize }
    })
}

#[cfg(windows)]
mod winapi_shim {
    #[repr(C)]
    pub(crate) struct SYSTEM_INFO {
        pub(crate) wProcessorArchitecture: u16, pub(crate) wReserved: u16, pub(crate) dwPageSize: u32,
        pub(crate) lpMinimumApplicationAddress: *mut core::ffi::c_void, pub(crate) lpMaximumApplicationAddress: *mut core::ffi::c_void,
        pub(crate) dwActiveProcessorMask: usize, pub(crate) dwNumberOfProcessors: u32, pub(crate) dwProcessorType: u32,
        pub(crate) dwAllocationGranularity: u32, pub(crate) wProcessorLevel: u16, pub(crate) wProcessorRevision: u16,
    }
    pub(crate) const MEM_COMMIT: u32 = 0x00001000; pub(crate) const MEM_RESERVE: u32 = 0x00002000; pub(crate) const MEM_RELEASE: u32 = 0x00008000;
    pub(crate) const PAGE_NOACCESS: u32 = 0x01; pub(crate) const PAGE_READWRITE: u32 = 0x04;
    unsafe extern "system" {
        pub(crate) fn VirtualAlloc(lpAddress: *mut core::ffi::c_void, dwSize: usize, flAllocationType: u32, flProtect: u32) -> *mut core::ffi::c_void;
        pub(crate) fn VirtualFree(lpAddress: *mut core::ffi::c_void, dwSize: usize, dwFreeType: u32) -> i32;
        pub(crate) fn VirtualProtect(lpAddress: *mut core::ffi::c_void, dwSize: usize, flNewProtect: u32, lpflOldProtect: *mut u32) -> i32;
        pub(crate) fn GetSystemInfo(lpSystemInfo: *mut SYSTEM_INFO);
    }
}

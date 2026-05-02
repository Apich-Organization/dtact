#![allow(unsafe_code)]
#![allow(non_snake_case)]

use core::sync::atomic::{AtomicU32, AtomicU8, AtomicU64, Ordering};
// use crate::dta_scheduler::TaskIndex; // Reserved for zero-copy flow tracking
use std::task::RawWaker;

/// Safety policies for context pool memory layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyLevel {
    Safety0,
    Safety1,
    Safety2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadKind {
    Compute,
    IO,
    Memory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyMode {
    P2PMesh,
    Global,
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

#[repr(C, align(64))]
pub(crate) struct FiberContext {
    pub(crate) stack_ptr: usize,
    pub(crate) scheduler_stack_ptr: usize,
    pub(crate) tib_stack_limit: usize,
    pub(crate) tib_stack_base: usize,
    pub(crate) state: AtomicU8,
    pub(crate) kind: WorkloadKind,
    pub(crate) mode: TopologyMode,
    pub(crate) origin_core: u16,
    pub(crate) fiber_index: u32,
    pub(crate) waiter_thread_id: AtomicU64,
    pub(crate) static_raw_waker: RawWaker,
    pub(crate) regs: Registers,
    pub(crate) executor_regs: Registers,
    pub(crate) next_free: AtomicU32,
    pub(crate) panic_payload_ptr: *mut (),
    pub(crate) trampoline: unsafe extern "C" fn(),
    pub(crate) invoke_closure: unsafe fn(*mut ()),
    pub(crate) closure_ptr: *mut (),
    pub(crate) result_ptr: *mut (),
    pub(crate) reader_ptr: *mut (),
    pub(crate) buf_ptr: *mut [u8],
    pub(crate) read_buffer_ptr: *mut u8,
    pub(crate) switch_fn: unsafe extern "C" fn(*mut Registers, *const Registers),
    pub(crate) cleanup_fn: Option<unsafe extern "C" fn(*mut ())>,
    pub(crate) adaptive_spin_count: u32,
    pub(crate) spin_failure_count: u32,
    pub(crate) last_os_thread_id: u64,
}

impl FiberContext {
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
            switch_fn: crate::context_switch::switch_context_cross_thread_float,
            cleanup_fn: None,
            adaptive_spin_count: 50,
            spin_failure_count: 0,
            last_os_thread_id: 0,
        }
    }
}

unsafe extern "C" fn dummy_trampoline() {}
unsafe fn dummy_invoke(_: *mut ()) {}

pub struct ContextPool {
    base_ptr: *mut u8,
    total_size: usize,
    pub slot_size: usize,
    capacity: u32,
    safety: SafetyLevel,
    free_head: AtomicU64,
}

unsafe impl Send for ContextPool {}
unsafe impl Sync for ContextPool {}

impl ContextPool {
    pub fn new(capacity: u32, stack_size: usize, safety: SafetyLevel, _numa: usize) -> Result<Self, &'static str> {
        let page_size = 4096;
        let align = 64;
        let context_sz = (core::mem::size_of::<FiberContext>() + align - 1) & !(align - 1);
        
        // Slot Size: [ Stack Space | 8KB Read Buffer | FiberContext ]
        // Enforce page alignment for the entire slot to ensure mprotect compatibility
        let slot_size = (stack_size + context_sz + 8192 + page_size - 1) & !(page_size - 1);
        
        let total_size = match safety {
            SafetyLevel::Safety0 => capacity as usize * slot_size,
            SafetyLevel::Safety1 => {
                let num_groups = (capacity as usize + 31) / 32;
                capacity as usize * slot_size + num_groups * page_size
            }
            SafetyLevel::Safety2 => capacity as usize * (slot_size + page_size),
        };
        
        // Add 4KB for SEH/Metadata (Windows RVA safety)
        let total_size_with_meta = total_size + 4096;

        unsafe {
            let base_ptr = Self::allocate_arena(total_size_with_meta, safety)?;
            
            // PRE-PROTECT Guard Pages (Low Address / Bottom of stack)
            if safety == SafetyLevel::Safety1 {
                for i in 0..((capacity + 31) / 32) {
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

            // Build the Lock-Free Free-List
            for i in 0..capacity {
                let ctx_ptr = pool.get_context_ptr(i);
                core::ptr::write(ctx_ptr, FiberContext::new());
                (*ctx_ptr).fiber_index = i;
                
                // Robust Aligned Read Buffer (64-byte aligned)
                let raw_read_buf = (ctx_ptr as *mut u8).sub(8192);
                (*ctx_ptr).read_buffer_ptr = (raw_read_buf as usize & !63) as *mut u8;
                
                // Link next available context
                (*ctx_ptr).next_free.store(i + 1, Ordering::Relaxed);
            }

            // Sentinel for the end of the free list
            let last_ctx = pool.get_context_ptr(capacity - 1);
            (*last_ctx).next_free.store(u32::MAX, Ordering::Relaxed);

            // Point 2: Windows SEH Registration (Full Production Unwind Info)
            #[cfg(windows)]
            {
                use windows_sys::Win32::System::Diagnostics::Debug::{RtlAddFunctionTable, RUNTIME_FUNCTION};
                
                #[repr(C, packed)]
                struct UnwindInfo {
                    version_and_flags: u8,
                    prolog_size: u8,
                    unwind_code_count: u8,
                    frame_register_and_offset: u8,
                }
                
                // Metadata is at the end of the arena (last 4KB page)
                let meta_base = base_ptr.add(total_size);
                let unwind_info_ptr = meta_base as *mut UnwindInfo;
                core::ptr::write(unwind_info_ptr, UnwindInfo {
                    version_and_flags: 0x01, 
                    prolog_size: 0,
                    unwind_code_count: 0,
                    frame_register_and_offset: 0,
                });

                let function_table_ptr = unwind_info_ptr.add(1) as *mut RUNTIME_FUNCTION;
                core::ptr::write(function_table_ptr, RUNTIME_FUNCTION {
                    BeginAddress: 0,
                    EndAddress: total_size as u32,
                    UnwindData: (unwind_info_ptr as usize - base_ptr as usize) as u32,
                });
                
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
            if libc::mprotect(ptr as *mut _, size, libc::PROT_NONE) != 0 { return Err("mprotect failed"); }
        }
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::System::Memory::{VirtualProtect, PAGE_NOACCESS};
            let mut old = 0;
            if VirtualProtect(ptr as *mut _, size, PAGE_NOACCESS, &mut old) == 0 { return Err("VirtualProtect failed"); }
        }
        Ok(())
    }

    #[inline(always)]
    unsafe fn allocate_arena(size: usize, safety: SafetyLevel) -> Result<*mut u8, &'static str> {
        #[cfg(unix)]
        {
            let mut flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
            if safety == SafetyLevel::Safety0 { flags |= 0x40000; } // MAP_HUGETLB
            let ptr = unsafe { libc::mmap(core::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0) };
            if ptr == libc::MAP_FAILED { return Err("mmap failed"); }
            Ok(ptr as *mut u8)
        }
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE};
            let mut flags = MEM_COMMIT | MEM_RESERVE;
            if safety == SafetyLevel::Safety0 { flags |= 0x20000000; } // MEM_LARGE_PAGES
            let ptr = VirtualAlloc(core::ptr::null_mut(), size, flags, PAGE_READWRITE);
            if ptr.is_null() { return Err("VirtualAlloc failed"); }
            Ok(ptr as *mut u8)
        }
    }

    #[inline(always)]
    pub fn get_context_ptr(&self, index: u32) -> *mut FiberContext {
        let page_size = 4096;
        let align = 64;
        let context_sz = (core::mem::size_of::<FiberContext>() + align - 1) & !(align - 1);
        
        let guard_offset = match self.safety {
            SafetyLevel::Safety0 => 0,
            SafetyLevel::Safety1 => (index as usize / 32 + 1) * page_size,
            SafetyLevel::Safety2 => (index as usize + 1) * page_size,
        };

        unsafe {
            let slot_base = self.base_ptr.add(index as usize * self.slot_size + guard_offset);
            slot_base.add(self.slot_size - context_sz) as *mut FiberContext
        }
    }

    /// O(1) Pop with ABA protection
    #[inline(always)]
    pub fn alloc_context(&self) -> Option<u32> {
        let mut head = self.free_head.load(Ordering::Acquire);
        loop {
            let index = head as u32;
            let r#gen = (head >> 32) as u32;
            if index == u32::MAX { return None; }
            
            let ctx = self.get_context_ptr(index);
            let next = unsafe { (*ctx).next_free.load(Ordering::Relaxed) };
            
            let new_head = ((r#gen.wrapping_add(1) as u64) << 32) | (next as u64);
            
            match self.free_head.compare_exchange_weak(head, new_head, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(index),
                Err(latest) => head = latest,
            }
        }
    }

    #[inline(always)]
    pub fn free_context(&self, index: u32) {
        let ctx = self.get_context_ptr(index);
        
        // Reset state to Initial and notify any waiting host threads
        unsafe { 
            (*ctx).state.store(FiberStatus::Initial as u8, Ordering::Release); 
            crate::utils::futex_wake(&(*ctx).state);
        };

        let mut head = self.free_head.load(Ordering::Relaxed);
        loop {
            let current_idx = head as u32;
            let r#gen = (head >> 32) as u32;
            unsafe { (*ctx).next_free.store(current_idx, Ordering::Relaxed) };
            let new_head = ((r#gen.wrapping_add(1) as u64) << 32) | (index as u64);
            match self.free_head.compare_exchange_weak(head, new_head, Ordering::Release, Ordering::Relaxed) {
                Ok(_) => break,
                Err(h) => head = h,
            }
        }
    }

    #[inline(always)]
    pub fn get_dispatch_layout(&self) -> (*mut u8, usize, usize) {
        let page_size = 4096;
        let guard_size = if self.safety == SafetyLevel::Safety0 { 0 } else { page_size };
        (self.base_ptr, self.slot_size, guard_size)
    }
}

impl Drop for ContextPool {
    #[inline(always)]
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe { libc::munmap(self.base_ptr as *mut _, self.total_size); }
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
            VirtualFree(self.base_ptr as *mut _, 0, MEM_RELEASE);
        }
    }
}

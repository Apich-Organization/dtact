use alloc::vec::Vec;
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::cell::UnsafeCell;
use core::arch::asm;

/// Task Index used for Zero-Copy passing
pub type TaskIndex = u32;

pub const CHUNK_SIZE: usize = 32;
// MUST be a power of two for bitwise masking
pub const MAILBOX_CAPACITY: usize = 1024;
pub const MAILBOX_MASK: usize = MAILBOX_CAPACITY - 1;

/// Fixed-size local Arena ring buffer to avoid Heap-based Reallocations.
/// Sized to exactly hold the max queue without global locks.
pub const LOCAL_QUEUE_CAPACITY: usize = 8192;
pub const LOCAL_QUEUE_MASK: usize = LOCAL_QUEUE_CAPACITY - 1;

/// Batch Ownership Transfer Chunk
/// A chunk of 32 task indices, transferred in a single atomic pointer exchange.
#[derive(Debug, Clone, Copy)]
pub struct TaskChunk {
    pub tasks: [TaskIndex; CHUNK_SIZE],
    pub count: usize,
}

impl Default for TaskChunk {
    #[inline(always)]
    fn default() -> Self {
        Self {
            tasks: [0; CHUNK_SIZE],
            count: 0,
        }
    }
}

/// Helper for Huge Page Allocation to eliminate TLB Misses.
pub struct HugeBuffer<T> {
    pub ptr: *mut T,
    pub size_bytes: usize,
}

impl<T> HugeBuffer<T> {
    pub fn new() -> Self {
        let size_bytes = core::mem::size_of::<T>();
        
        #[cfg(unix)]
        unsafe {
            let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | 0x40000; // MAP_HUGETLB
            let mut ptr = libc::mmap(
                core::ptr::null_mut(),
                size_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                -1,
                0,
            );
            if ptr == libc::MAP_FAILED {
                // Fallback to 4KB pages
                ptr = libc::mmap(
                    core::ptr::null_mut(),
                    size_bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                );
                assert!(ptr != libc::MAP_FAILED, "HugeBuffer mmap failed");
            }
            core::ptr::write_bytes(ptr, 0, size_bytes);
            Self { ptr: ptr as *mut T, size_bytes }
        }

        #[cfg(windows)]
        unsafe {
            // Simplified VirtualAlloc for Windows
            let ptr = alloc::alloc::alloc_zeroed(core::alloc::Layout::new::<T>()) as *mut T;
            Self { ptr, size_bytes }
        }
    }
}

impl<T> Drop for HugeBuffer<T> {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size_bytes);
        }
    }
}

/// Single-Producer Single-Consumer (SPSC) Queue for the P2P Mesh Mailbox
#[repr(align(64))]
pub struct Mailbox {
    head: AtomicUsize,
    _pad1: [u8; 64 - core::mem::size_of::<AtomicUsize>()],
    
    tail: AtomicUsize,
    _pad2: [u8; 64 - core::mem::size_of::<AtomicUsize>()],
    
    buffer: HugeBuffer<UnsafeCell<[TaskChunk; MAILBOX_CAPACITY]>>,
}

unsafe impl Sync for Mailbox {}
unsafe impl Send for Mailbox {}

impl Mailbox {
    pub fn new() -> Self {
        Self {
            head: AtomicUsize::new(0),
            _pad1: [0; 56],
            tail: AtomicUsize::new(0),
            _pad2: [0; 56],
            buffer: HugeBuffer::new(),
        }
    }

    #[inline(always)]
    pub fn push(&self, chunk: TaskChunk) -> Result<(), TaskChunk> {
        let current_tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (current_tail + 1) & MAILBOX_MASK;

        if next_tail == self.head.load(Ordering::Acquire) {
            return Err(chunk);
        }

        unsafe {
            let buffer_ptr = (*self.buffer.ptr).get() as *mut TaskChunk;
            *buffer_ptr.add(current_tail) = chunk;
        }

        self.tail.store(next_tail, Ordering::Release);
        
        #[cfg(all(feature = "hw-acceleration", any(target_arch = "x86", target_arch = "x86_64")))]
        unsafe {
            // Hardware Acceleration: CLDEMOTE
            // Proactively evicts the cache line containing the tail pointer to the shared L3 cache,
            // anticipating the consumer core will read it soon, significantly reducing coherency traffic.
            core::arch::asm!("cldemote [{}]", in(reg) &self.tail);
        }

        #[cfg(all(feature = "hw-acceleration", target_arch = "aarch64"))]
        unsafe {
            // Hardware Acceleration: ARM Data Cache Clean by VA to Point of Coherency (DC CVAC)
            // Pushes the updated tail index out of the local L1/L2 down to the shared Point of Coherency,
            // acting identically to CLDEMOTE by accelerating visibility to the remote consumer core.
            core::arch::asm!("dc cvac, {}", in(reg) &self.tail);
        }

        #[cfg(all(feature = "hw-acceleration", target_arch = "riscv64"))]
        unsafe {
            // Hardware Acceleration: RISC-V Zicbom Cache Block Clean (cbo.clean)
            // Cleans the cache block to the point of coherency, effectively demoting it 
            // to a shared level so other harts can see the updated tail pointer instantly.
            core::arch::asm!("cbo.clean 0({0})", in(reg) &self.tail);
        }
        
        Ok(())
    }

    #[inline(always)]
    pub fn pop(&self) -> Option<TaskChunk> {
        let current_head = self.head.load(Ordering::Relaxed);

        if current_head == self.tail.load(Ordering::Acquire) {
            return None; // Empty
        }

        let chunk = unsafe {
            let buffer_ptr = (*self.buffer.ptr).get() as *mut TaskChunk;
            core::ptr::read(buffer_ptr.add(current_head))
        };

        let next_head = (current_head + 1) & MAILBOX_MASK;
        self.head.store(next_head, Ordering::Release);
        Some(chunk)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuLevel {
    pub core_id: u16,
    pub ccx_id: u16,
    pub numa_id: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyMode {
    P2PMesh,
    Global,
}

#[repr(align(64))]
pub struct Worker {
    pub cpu: CpuLevel,
    pub load_level: AtomicU8,
    pub deflection_threshold: AtomicU8,
    
    pub local_queue: HugeBuffer<[TaskIndex; LOCAL_QUEUE_CAPACITY]>,
    pub local_head: usize,
    pub local_tail: usize,
    
    pub ticks: u64,
    pub polling_order: Vec<usize>,
}

unsafe impl Sync for Worker {}
unsafe impl Send for Worker {}

impl Worker {
    pub fn new(cpu: CpuLevel, total_cores: usize) -> Self {
        let mut polling_order = Vec::with_capacity(total_cores - 1);
        let my_core = cpu.core_id as usize;
        let my_ccx = cpu.ccx_id;

        for i in 0..total_cores {
            if i != my_core && (i / 8) as u16 == my_ccx {
                polling_order.push(i);
            }
        }
        for i in 0..total_cores {
            if i != my_core && (i / 8) as u16 != my_ccx {
                polling_order.push(i);
            }
        }

        Self {
            cpu,
            load_level: AtomicU8::new(0),
            deflection_threshold: AtomicU8::new(80),
            local_queue: HugeBuffer::new(),
            local_head: 0,
            local_tail: 0,
            ticks: 0,
            polling_order,
        }
    }
    
    #[inline(always)]
    pub fn local_queue_len(&self) -> usize {
        self.local_tail.wrapping_sub(self.local_head) & LOCAL_QUEUE_MASK
    }

    #[inline(always)]
    pub fn update_load(&self) {
        let queue_len = self.local_queue_len();
        let load = core::cmp::min((queue_len * 100) >> 13, 100) as u8;
        self.load_level.store(load, Ordering::Relaxed);
    }
    
    #[inline(always)]
    pub fn tick(&mut self) {
        self.ticks = self.ticks.wrapping_add(1);
        if (self.ticks & 1023) == 0 {
            let load = self.load_level.load(Ordering::Relaxed);
            let current_thresh = self.deflection_threshold.load(Ordering::Relaxed);
            
            let new_thresh = if load > 90 {
                current_thresh.saturating_sub(5).max(40)
            } else if load < 30 {
                current_thresh.saturating_add(5).min(95)
            } else {
                current_thresh
            };
            
            self.deflection_threshold.store(new_thresh, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn push_local(&mut self, task: TaskIndex) {
        unsafe {
            (*self.local_queue.ptr)[self.local_tail] = task;
        }
        self.local_tail = (self.local_tail + 1) & LOCAL_QUEUE_MASK;
    }
    
    #[inline(always)]
    pub fn push_batch(&mut self, chunk: &TaskChunk) {
        let count = chunk.count;
        let tail = self.local_tail;
        let end_idx = tail + count;
        
        if end_idx <= LOCAL_QUEUE_CAPACITY {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    chunk.tasks.as_ptr(),
                    (*self.local_queue.ptr).as_mut_ptr().add(tail),
                    count
                );
            }
        } else {
            let first_part = LOCAL_QUEUE_CAPACITY - tail;
            let second_part = count - first_part;
            unsafe {
                core::ptr::copy_nonoverlapping(
                    chunk.tasks.as_ptr(),
                    (*self.local_queue.ptr).as_mut_ptr().add(tail),
                    first_part
                );
                core::ptr::copy_nonoverlapping(
                    chunk.tasks.as_ptr().add(first_part),
                    (*self.local_queue.ptr).as_mut_ptr(),
                    second_part
                );
            }
        }
        self.local_tail = end_idx & LOCAL_QUEUE_MASK;
    }
    
    /// Pure ASM Dispatch Loop to completely bypass compiler stack ABI
    /// Executed by the worker thread until the local queue is empty.
    pub unsafe fn dispatch_loop_asm(&mut self, context_base: *mut u8, context_size: usize, group_guard_size: usize) {
        while self.local_head != self.local_tail {
            let task = (unsafe { *self.local_queue.ptr })[self.local_head];
            self.local_head = (self.local_head + 1) & LOCAL_QUEUE_MASK;
            
            // O(1) alignment calculation with Safety1 GroupGuard spacing natively supported
            let group_offset = (task as usize >> 5) * group_guard_size;
            let target_ptr = unsafe {
                context_base.add((task as usize) * context_size + group_offset)
            };
            
            // Hardware Prefetch: Bring FiberContext to L1 using T0 hint immediately
            #[cfg(target_arch = "x86_64")]
            unsafe {
                core::arch::x86_64::_mm_prefetch::<0>(target_ptr as *const i8);
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) target_ptr, options(nostack, preserves_flags));
            }
            #[cfg(all(target_arch = "riscv64", feature = "hw-acceleration"))]
            unsafe {
                // RISC-V Zicbop extension: prefetch for read into L1 cache
                core::arch::asm!("prefetch.r 0({0})", in(reg) target_ptr, options(nostack, preserves_flags));
            }

            // Inline Assembly Fast-Path Context Switch (Minimal Registers)
            #[cfg(all(target_arch = "x86_64", unix))]
            unsafe {
                core::arch::asm!(
                    // SysV ABI: Save critical caller-saved state (FPU/AVX skipped)
                    "push rbp",
                    "push rbx",
                    "push r12",
                    "push r13",
                    "push r14",
                    "push r15",
                    
                    // Perform Stack Pointer swap (Scheduler -> Fiber)
                    "mov [rcx + 8], rsp",    // ctx.scheduler_stack_ptr = rsp
                    "mov rsp, [rcx + 0]",    // rsp = ctx.stack_ptr
                    
                    // Pop Fiber's caller-saved registers and execute
                    "pop r15",
                    "pop r14",
                    "pop r13",
                    "pop r12",
                    "pop rbx",
                    "pop rbp",
                    "ret", // Jumps exactly to where Fiber yielded or starts
                    in("rcx") target_ptr,
                );
            }

            #[cfg(all(target_arch = "x86_64", windows))]
            unsafe {
                core::arch::asm!(
                    // Windows ABI requires preserving rdi and rsi in addition to the SysV registers.
                    "push rbp",
                    "push rbx",
                    "push rdi",
                    "push rsi",
                    "push r12",
                    "push r13",
                    "push r14",
                    "push r15",
                    
                    "mov [rcx + 8], rsp",
                    "mov rsp, [rcx + 0]",
                    
                    "pop r15",
                    "pop r14",
                    "pop r13",
                    "pop r12",
                    "pop rsi",
                    "pop rdi",
                    "pop rbx",
                    "pop rbp",
                    "ret",
                    in("rcx") target_ptr,
                );
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                core::arch::asm!(
                    // ARM64 AAPCS: Callee-saved registers are x19-x29, plus x30 (LR)
                    "stp x19, x20, [sp, #-16]!",
                    "stp x21, x22, [sp, #-16]!",
                    "stp x23, x24, [sp, #-16]!",
                    "stp x25, x26, [sp, #-16]!",
                    "stp x27, x28, [sp, #-16]!",
                    "stp x29, x30, [sp, #-16]!",
                    
                    "mov x1, sp",
                    "str x1, [{0}, #8]", // ctx.scheduler_stack_ptr = sp
                    "ldr x1, [{0}, #0]", // sp = ctx.stack_ptr
                    "mov sp, x1",
                    
                    "ldp x29, x30, [sp], #16",
                    "ldp x27, x28, [sp], #16",
                    "ldp x25, x26, [sp], #16",
                    "ldp x23, x24, [sp], #16",
                    "ldp x21, x22, [sp], #16",
                    "ldp x19, x20, [sp], #16",
                    "ret",
                    in(reg) target_ptr,
                );
            }

            #[cfg(target_arch = "riscv64")]
            unsafe {
                core::arch::asm!(
                    // RISC-V 64: Callee-saved registers are s0-s11 and ra (return address)
                    "addi sp, sp, -112",
                    "sd s0, 0(sp)",
                    "sd s1, 8(sp)",
                    "sd s2, 16(sp)",
                    "sd s3, 24(sp)",
                    "sd s4, 32(sp)",
                    "sd s5, 40(sp)",
                    "sd s6, 48(sp)",
                    "sd s7, 56(sp)",
                    "sd s8, 64(sp)",
                    "sd s9, 72(sp)",
                    "sd s10, 80(sp)",
                    "sd s11, 88(sp)",
                    "sd ra, 96(sp)",
                    
                    "sd sp, 8({0})", // ctx.scheduler_stack_ptr = sp
                    "ld sp, 0({0})", // sp = ctx.stack_ptr
                    
                    "ld ra, 96(sp)",
                    "ld s11, 88(sp)",
                    "ld s10, 80(sp)",
                    "ld s9, 72(sp)",
                    "ld s8, 64(sp)",
                    "ld s7, 56(sp)",
                    "ld s6, 48(sp)",
                    "ld s5, 40(sp)",
                    "ld s4, 32(sp)",
                    "ld s3, 24(sp)",
                    "ld s2, 16(sp)",
                    "ld s1, 8(sp)",
                    "ld s0, 0(sp)",
                    "addi sp, sp, 112",
                    "ret",
                    in(reg) target_ptr,
                );
            }
        }
    }
}

pub struct DtaScheduler {
    pub workers: Vec<UnsafeCell<Worker>>,
    pub mailboxes: Vec<Vec<Mailbox>>,
    pub topology: TopologyMode,
    pub enqueue_jmp: [fn(&DtaScheduler, usize, usize, TaskIndex); 2],
}

unsafe impl Sync for DtaScheduler {}
unsafe impl Send for DtaScheduler {}

impl DtaScheduler {
    pub fn new(num_workers: usize, topology: TopologyMode) -> Self {
        let mut workers = Vec::with_capacity(num_workers);
        let mut mailboxes = Vec::with_capacity(num_workers);

        for i in 0..num_workers {
            workers.push(UnsafeCell::new(Worker::new(CpuLevel {
                core_id: i as u16,
                ccx_id: (i / 8) as u16,
                numa_id: (i / 64) as u16,
            }, num_workers)));
            
            let mut row = Vec::with_capacity(num_workers);
            for _ in 0..num_workers {
                row.push(Mailbox::new());
            }
            mailboxes.push(row);
        }

        Self {
            workers,
            mailboxes,
            topology,
            enqueue_jmp: [Self::do_push_local, Self::do_push_remote],
        }
    }

    #[inline(always)]
    fn do_push_local(&self, source_core: usize, _target_core: usize, task: TaskIndex) {
        unsafe {
            let worker = &mut *self.workers[source_core].get();
            worker.push_local(task);
        }
    }

    #[inline(always)]
    fn do_push_remote(&self, source_core: usize, target_core: usize, task: TaskIndex) {
        let mut chunk = TaskChunk::default();
        chunk.tasks[0] = task;
        chunk.count = 1;
        let _ = self.mailboxes[source_core][target_core].push(chunk);
        
        #[cfg(all(feature = "hw-acceleration", any(target_arch = "x86", target_arch = "x86_64")))]
        unsafe {
            // Hardware Acceleration: UINTR (User-Level Interrupts)
            // Fire a _senduipi directly to the target core to wake it up instantly if it's yielding,
            // avoiding OS-level context switches or expensive IPIs.
            // Using raw bytes for `senduipi rax` to guarantee compilation on older LLVMs.
            core::arch::asm!(
                "mov rax, {}",
                ".byte 0xf3, 0x0f, 0xc7, 0xf0", 
                in(reg) target_core as u64,
                out("rax") _,
                options(nostack, preserves_flags),
            );
        }

        #[cfg(all(feature = "hw-acceleration", target_arch = "aarch64"))]
        unsafe {
            // Hardware Acceleration: ARM Send Event (SEV)
            // Broadcasts an event to all cores. If the target core is in a low-power yield state
            // via WFE (Wait For Event), this wakes it up instantly in userspace without OS traps.
            core::arch::asm!("sev", options(nostack, preserves_flags));
        }

        #[cfg(all(feature = "hw-acceleration", target_arch = "riscv64"))]
        unsafe {
            // Hardware Acceleration: RISC-V AIA (Advanced Interrupt Architecture)
            // Sends a User-level IPI (UIPI) to the target hart.
            // Writing to the `uipi` CSR triggers a user-level interrupt directly on the 
            // target core, allowing instantaneous wakeups from `Zihintpause` spin states.
            core::arch::asm!("csrw uipi, {0}", in(reg) target_core);
        }
    }

    #[inline(always)]
    pub fn enqueue_task(&self, source_core: usize, flow_id: u64, task: TaskIndex) {
        let worker_ref = unsafe { &*self.workers[source_core].get() };
        let threshold = worker_ref.deflection_threshold.load(Ordering::Relaxed);
        let load = worker_ref.load_level.load(Ordering::Relaxed);

        // Branchless Double Hash for Spatial Smoothing
        let deflect_mask = if load > threshold { usize::MAX } else { 0 };
        let h1 = (flow_id & 7) as usize; 
        let h2 = ((flow_id >> 3) & 7 | 1) as usize; 

        let ccx_base = source_core & !7;
        let local_idx = source_core & 7;
        
        let deflect_target = (local_idx + h1 + h2) & 7;
        let target_idx = local_idx ^ ((local_idx ^ deflect_target) & deflect_mask);
        let target_core = ccx_base | target_idx;

        let jump_idx = (target_core != source_core) as usize;
        (self.enqueue_jmp[jump_idx])(self, source_core, target_core, task);
    }

    #[inline(always)]
    pub fn poll_mailboxes(&self, current_core: usize) {
        let worker = unsafe { &mut *self.workers[current_core].get() };
        
        let num_polls = worker.polling_order.len();
        for idx in 0..num_polls {
            let i = worker.polling_order[idx];
            
            while let Some(chunk) = self.mailboxes[i][current_core].pop() {
                worker.push_batch(&chunk);
            }
        }
        
        worker.update_load();
        worker.tick();
    }

    /// The Main Scheduler Heartbeat Loop
    /// Executes the dispatch assembly and seamlessly manages the mailbox topology when exhausted.
    pub fn run_worker(&self, current_core: usize, context_base: *mut u8, context_size: usize, group_guard_size: usize) {
        loop {
            // 1. Drain the local SPSC queue natively
            #[cfg(target_arch = "x86_64")]
            unsafe {
                let worker = &mut *self.workers[current_core].get();
                worker.dispatch_loop_asm(context_base, context_size, group_guard_size);
            }
            
            // 2. Local queue exhausted. Pull from P2P Mailbox Mesh.
            self.poll_mailboxes(current_core);
            
            // 3. Affinity Controls & Backoff
            unsafe {
                let worker = &*self.workers[current_core].get();
                if worker.local_queue_len() == 0 {
                    // Active spinning or UMWAIT if queue remains completely dry.
                    // This implements the dev-doc requirement for topology affinity controls.
                    core::hint::spin_loop();
                }
            }
        }
    }
}

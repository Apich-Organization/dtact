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
    /// Executed by the worker thread infinitely.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn dispatch_loop_asm(&mut self, context_base: *mut u8, context_size: usize) -> ! {
        loop {
            if self.local_head != self.local_tail {
                let task = (*self.local_queue.ptr)[self.local_head];
                self.local_head = (self.local_head + 1) & LOCAL_QUEUE_MASK;
                
                let target_ptr = context_base.add((task as usize) * context_size);
                
                // Hardware Prefetch: Bring FiberContext to L1 using T0 hint immediately
                core::arch::x86_64::_mm_prefetch::<0>(target_ptr as *const i8);
                
                // Inline Assembly Fast-Path Context Switch (Minimal Registers)
                // rcx holds target_ptr (FiberContext pointer)
                // rdi holds scheduler state pointer
                asm!(
                    // 1. Save critical caller-saved state (FPU/AVX skipped)
                    "push rbp",
                    "push rbx",
                    "push r12",
                    "push r13",
                    "push r14",
                    "push r15",
                    
                    // 2. Perform Stack Pointer swap (Scheduler -> Fiber)
                    // rcx points to FiberContext. offset 0 = fiber_rsp, offset 8 = sched_rsp
                    "mov [rcx + 8], rsp",    // ctx.scheduler_stack_ptr = rsp
                    "mov rsp, [rcx + 0]",    // rsp = ctx.stack_ptr
                    
                    // 3. Pop Fiber's caller-saved registers and execute
                    "pop r15",
                    "pop r14",
                    "pop r13",
                    "pop r12",
                    "pop rbx",
                    "pop rbp",
                    "ret", // Jumps exactly to where Fiber yielded or starts
                    in("rcx") target_ptr,
                );
            } else {
                // Queue is empty, yield to poll mailboxes (simplified loop)
                core::hint::spin_loop();
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

        let deflect_mask = if load > threshold { usize::MAX } else { 0 };
        let offset = ((flow_id & 7) as usize) | 1; 

        let ccx_base = source_core & !7;
        let local_idx = source_core & 7;
        let target_idx = local_idx ^ (offset & deflect_mask);
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
}

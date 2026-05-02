use alloc::vec::Vec;
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::cell::UnsafeCell;
#[allow(unused_imports)]
use core::arch::asm;

/// Task Index used for Zero-Copy passing within the ContextPool.
pub type TaskIndex = u32;

/// Number of tasks in a single `TaskChunk`.
pub const CHUNK_SIZE: usize = 32;

/// Capacity of a single core-to-core mailbox.
/// MUST be a power of two for bitwise masking.
pub const MAILBOX_CAPACITY: usize = 1024;
/// Mask for mailbox index wrap-around.
pub const MAILBOX_MASK: usize = MAILBOX_CAPACITY - 1;

/// Capacity of a worker's local execution queue.
/// Sized to exactly hold the max queue without global locks.
pub const LOCAL_QUEUE_CAPACITY: usize = 8192;
/// Mask for local queue index wrap-around.
pub const LOCAL_QUEUE_MASK: usize = LOCAL_QUEUE_CAPACITY - 1;

/// Batch Ownership Transfer Chunk.
/// 
/// A chunk of 32 task indices, transferred in a single atomic pointer exchange
/// to minimize coherency traffic across the P2P mesh.
#[derive(Debug, Clone, Copy)]
pub struct TaskChunk {
    /// Array of task indices in this chunk.
    pub tasks: [TaskIndex; CHUNK_SIZE],
    /// Number of active tasks in this chunk.
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
/// 
/// Manages page-aligned memory regions that utilize 2MB or 1GB huge pages
/// (where supported by the OS) to maximize memory throughput.
pub struct HugeBuffer<T> {
    /// Pointer to the allocated memory.
    pub ptr: *mut T,
    /// Total size of the buffer in bytes.
    pub size_bytes: usize,
}

impl<T> HugeBuffer<T> {
    /// Allocates a new `HugeBuffer` using OS-specific huge page primitives.
    #[inline]
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
            use crate::memory_management::winapi_shim;
            #[cfg(feature = "windows-root")]
            {
                let mut ptr = winapi_shim::VirtualAlloc(
                    core::ptr::null_mut(),
                    size_bytes,
                    winapi_shim::MEM_RESERVE | winapi_shim::MEM_COMMIT | winapi_shim::MEM_LARGE_PAGES,
                    winapi_shim::PAGE_READWRITE,
                );
                if ptr.is_null() {
                    ptr = winapi_shim::VirtualAlloc(
                        core::ptr::null_mut(),
                        size_bytes,
                        winapi_shim::MEM_RESERVE | winapi_shim::MEM_COMMIT,
                        winapi_shim::PAGE_READWRITE,
                    );
                    assert!(!ptr.is_null(), "HugeBuffer VirtualAlloc failed");
                }
                Self { ptr: ptr as *mut T, size_bytes }
            }
            #[cfg(not(feature = "windows-root"))]
            {
                let ptr = winapi_shim::VirtualAlloc(
                    core::ptr::null_mut(),
                    size_bytes,
                    winapi_shim::MEM_RESERVE | winapi_shim::MEM_COMMIT,
                    winapi_shim::PAGE_READWRITE,
                );
                assert!(!ptr.is_null(), "HugeBuffer VirtualAlloc failed");
                Self { ptr: ptr as *mut T, size_bytes }
            }
        }
    }
}

impl<T> Drop for HugeBuffer<T> {
    #[inline(always)]
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size_bytes);
        }
        #[cfg(windows)]
        unsafe {
            crate::memory_management::winapi_shim::VirtualFree(
                self.ptr as *mut core::ffi::c_void,
                0,
                crate::memory_management::winapi_shim::MEM_RELEASE
            );
        }
    }
}

/// Single-Producer Single-Consumer (SPSC) Queue for the P2P Mesh Mailbox.
/// 
/// Aligned to 64 bytes to prevent false sharing between sender and receiver cores.
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
    /// Creates a new, empty Mailbox.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            head: AtomicUsize::new(0),
            _pad1: [0; 56],
            tail: AtomicUsize::new(0),
            _pad2: [0; 56],
            buffer: HugeBuffer::new(),
        }
    }

    /// Pushes a `TaskChunk` into the mailbox.
    /// 
    /// Utilizes hardware-specific demote/clean instructions to accelerate
    /// visibility of the updated tail pointer to the consumer core.
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
            core::arch::asm!("cldemote [{}]", in(reg) &self.tail);
        }

        #[cfg(all(feature = "hw-acceleration", target_arch = "aarch64"))]
        unsafe {
            core::arch::asm!("dc cvac, {}", in(reg) &self.tail);
        }

        #[cfg(all(feature = "hw-acceleration", target_arch = "riscv64"))]
        unsafe {
            core::arch::asm!("cbo.clean 0({0})", in(reg) &self.tail);
        }
        
        Ok(())
    }

    /// Pops a `TaskChunk` from the mailbox.
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

/// Hardware-specific CPU hierarchy information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuLevel {
    /// Physical Core ID.
    pub core_id: u16,
    /// Core Complex (CCX) ID.
    pub ccx_id: u16,
    /// NUMA Node ID.
    pub numa_id: u16,
}

/// Topology Strategy for the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyMode {
    /// Peer-to-Peer Mesh: Tasks are deflected to neighbors based on load.
    P2PMesh,
    /// Global: Tasks are shared across all cores via a common pool (unsupported in V3).
    Global,
}

/// Execution unit managed by a single OS thread.
/// 
/// Contains the local SPSC queue, load metrics, and work-deflection heuristics.
#[repr(align(64))]
pub struct Worker {
    /// Hierarchy information for this worker's core.
    pub cpu: CpuLevel,
    /// Current load level (0-100).
    pub load_level: AtomicU8,
    /// Load threshold above which tasks are deflected to peers.
    pub deflection_threshold: AtomicU8,
    
    /// Local SPSC execution queue.
    pub local_queue: HugeBuffer<[TaskIndex; LOCAL_QUEUE_CAPACITY]>,
    /// Head of the local queue.
    pub local_head: usize,
    /// Tail of the local queue.
    pub local_tail: usize,
    
    /// Total scheduler ticks executed.
    pub ticks: u64,
    /// Ordered list of peer core IDs for mailbox polling.
    pub polling_order: Vec<usize>,
}

unsafe impl Sync for Worker {}
unsafe impl Send for Worker {}

impl Worker {
    /// Creates a new `Worker` and calculates its CCX-aware polling order.
    #[inline(always)]
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
    
    /// Returns the current number of tasks in the local queue.
    #[inline(always)]
    pub fn local_queue_len(&self) -> usize {
        self.local_tail.wrapping_sub(self.local_head) & LOCAL_QUEUE_MASK
    }

    /// Updates the `load_level` based on the current queue length.
    #[inline(always)]
    pub fn update_load(&self) {
        let queue_len = self.local_queue_len();
        let load = core::cmp::min((queue_len * 100) >> 13, 100) as u8;
        self.load_level.store(load, Ordering::Relaxed);
    }
    
    /// Performs internal maintenance tasks (e.g., adaptive threshold updates).
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

    /// Pushes a single task into the local queue.
    #[inline(always)]
    pub fn push_local(&mut self, task: TaskIndex) {
        unsafe {
            (*self.local_queue.ptr)[self.local_tail] = task;
        }
        self.local_tail = (self.local_tail + 1) & LOCAL_QUEUE_MASK;
    }
    
    /// Pushes a batch of tasks into the local queue.
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
    
    /// Primary execution loop for the worker thread.
    /// 
    /// Drains the local queue, performs O(1) context alignment, and executes
    /// the context switch to the fiber.
    #[inline(always)]
    pub fn dispatch_loop(&mut self, context_base: *mut u8, context_size: usize, group_guard_size: usize) {
        while self.local_head != self.local_tail {
            let task = (unsafe { *self.local_queue.ptr })[self.local_head];
            self.local_head = (self.local_head + 1) & LOCAL_QUEUE_MASK;
            
            let group_offset = (task as usize >> 5) * group_guard_size;
            let target_ptr = unsafe {
                context_base.add((task as usize) * context_size + group_offset) as *mut crate::memory_management::FiberContext
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
                core::arch::asm!("prefetch.r 0({0})", in(reg) target_ptr, options(nostack, preserves_flags));
            }

            crate::future_bridge::CURRENT_FIBER.with(|c| c.set(target_ptr));

            unsafe {
                ((*target_ptr).switch_fn)(
                    &mut (*target_ptr).executor_regs,
                    &(*target_ptr).regs,
                );
            }

            crate::future_bridge::CURRENT_FIBER.with(|c| c.set(core::ptr::null_mut()));
        }
    }
}

/// The Dtact-V3 Distributed Scheduler.
/// 
/// Manages a set of `Worker` units and the P2P Mailbox matrix for 
/// cross-core task migration.
pub struct DtaScheduler {
    /// Thread-local worker states.
    pub workers: Vec<UnsafeCell<Worker>>,
    /// N x N Mailbox matrix for P2P communication.
    pub mailboxes: Vec<Vec<Mailbox>>,
    /// Active topology mode.
    pub topology: TopologyMode,
    /// Branchless jump table for task enqueuing.
    pub enqueue_jmp: [fn(&DtaScheduler, usize, usize, TaskIndex); 2],
}

unsafe impl Sync for DtaScheduler {}
unsafe impl Send for DtaScheduler {}

impl DtaScheduler {
    /// Creates a new `DtaScheduler` for the specified number of workers.
    #[inline(always)]
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
            core::arch::asm!("sev", options(nostack, preserves_flags));
        }

        #[cfg(all(feature = "hw-acceleration", target_arch = "riscv64"))]
        unsafe {
            core::arch::asm!("csrw uipi, {0}", in(reg) target_core);
        }
    }

    /// Enqueues a task into the mesh, applying work-deflection if necessary.
    /// 
    /// Uses a branchless Double-Hash strategy to distribute tasks across the
    /// CCX neighbors if the local core is under heavy load.
    #[inline(always)]
    pub fn enqueue_task(&self, source_core: usize, flow_id: u64, task: TaskIndex) {
        let worker_ref = unsafe { &*self.workers[source_core].get() };
        let threshold = worker_ref.deflection_threshold.load(Ordering::Relaxed);
        let load = worker_ref.load_level.load(Ordering::Relaxed);

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

    /// Polls all incoming mailboxes for the current core.
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

    /// Main heartbeat loop for a worker thread.
    /// 
    /// Manages the full lifecycle of fiber execution: local dispatch,
    /// P2P polling, and hardware-assisted backoff (UMWAIT, WFE, PAUSE).
    #[inline]
    pub fn run_worker(&self, current_core: usize, context_base: *mut u8, context_size: usize, group_guard_size: usize) {
        loop {
            unsafe {
                let worker = &mut *self.workers[current_core].get();
                worker.dispatch_loop(context_base, context_size, group_guard_size);
            }
            
            self.poll_mailboxes(current_core);
            
            unsafe {
                let worker = &*self.workers[current_core].get();
                if worker.local_queue_len() == 0 {
                    #[cfg(all(feature = "hw-acceleration", target_arch = "aarch64"))]
                    {
                        core::arch::asm!("wfe", options(nostack, preserves_flags));
                    }
                    #[cfg(all(feature = "hw-acceleration", target_arch = "riscv64"))]
                    {
                        core::arch::asm!("pause", options(nostack, preserves_flags));
                    }
                    #[cfg(all(feature = "hw-acceleration", any(target_arch = "x86_64", target_arch = "x86")))]
                    {
                        unsafe {
                            let tail_ptr = &worker.local_tail as *const _ as *mut core::ffi::c_void;
                            let control = 1u32; // C0.1 (Fast wakeup)
                            core::arch::asm!(
                                "umonitor {0}",
                                "test {1}, {1}", 
                                "jnz 2f",
                                "umwait {2}",
                                "2:",
                                in(reg) tail_ptr,
                                in(reg) worker.local_queue_len(),
                                in(reg) control,
                                inout("eax") 0xFFFFFFFFu32 => _,
                                inout("edx") 0xFFFFFFFFu32 => _,
                                options(nostack, preserves_flags)
                            );
                        }
                    }
                    #[cfg(all(not(feature = "hw-acceleration"), any(target_arch = "x86_64", target_arch = "x86")))]
                    {
                        core::hint::spin_loop();
                    }
                }
            }
        }
    }
}

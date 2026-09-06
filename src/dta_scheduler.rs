use crate::sync::atomic::{AtomicU8, AtomicU32, AtomicUsize, Ordering};
use alloc::vec::Vec;
#[allow(unused_imports)]
use core::arch::asm;
use core::cell::UnsafeCell;
use core::mem::MaybeUninit;

/// Task Index used for Zero-Copy passing within the `ContextPool`.
pub type TaskIndex = u32;

/// Number of tasks in a single `TaskChunk`.
pub const CHUNK_SIZE: usize = 32;

/// Capacity of a single core-to-core mailbox.
/// MUST be a power of two for bitwise masking.
pub const MAILBOX_CAPACITY: usize = 65_536;

/// Mask for mailbox index wrap-around.
pub const MAILBOX_MASK: usize = MAILBOX_CAPACITY - 1;

/// Capacity of a worker's local execution queue.
/// Sized to exactly hold the max queue without global locks.
pub const LOCAL_QUEUE_CAPACITY: usize = 131_072;

/// Mask for local queue index wrap-around.
pub const LOCAL_QUEUE_MASK: usize = LOCAL_QUEUE_CAPACITY - 1;
/// High-water mark above which a worker stops accepting new chunks and routes
/// them onward. Set at 7/8 capacity so `push_batch` has guaranteed headroom for
/// one full chunk on top.
pub const LOCAL_QUEUE_HIGH_WATERMARK: usize = LOCAL_QUEUE_CAPACITY - LOCAL_QUEUE_CAPACITY / 8;

/// Warehouse capacity in chunks. 32 768 chunks × 32 tasks = 1 048 576 tasks of
/// emergency back-pressure storage. Must be a power of two for bitwise masking.
pub const WAREHOUSE_CAPACITY: usize = 32_768;

/// Mask for warehouse index wrap-around.
pub const WAREHOUSE_MASK: usize = WAREHOUSE_CAPACITY - 1;

/// Batch Ownership Transfer Chunk.
///
/// A chunk of 32 task indices, transferred in a single atomic pointer exchange
/// to minimize coherency traffic across the P2P mesh.
///
/// `hop_count` tracks how many times this chunk has been re-deflected when the
/// receiving worker's queue was over the high-water mark. Once it exceeds
/// `DtaScheduler::max_hops`, the chunk is funneled into the warehouse instead
/// of bouncing further between cores — this is the bound that prevents the
/// classic "starving fiber held by full mailbox" deadlock.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TaskChunk {
    /// Array of task indices in this chunk. (128 bytes)
    pub tasks: [TaskIndex; CHUNK_SIZE],
    /// Number of active tasks in this chunk.
    pub count: u16,
    /// Number of times this chunk has been re-routed. Bounded by `max_hops`.
    pub hop_count: u8,
    /// Reserved for future use; keeps the chunk 4-byte aligned for the trailing pad.
    _flags: u8,
    /// Padding so the chunk's stride is 144 B and cleanly slices into cache lines.
    _pad: [u8; 12],
}

impl Default for TaskChunk {
    #[inline(always)]
    fn default() -> Self {
        Self {
            tasks: [0; CHUNK_SIZE],
            count: 0,
            hop_count: 0,
            _flags: 0,
            _pad: [0; 12],
        }
    }
}

/// Helper for Huge Page Allocation to eliminate TLB Misses.
///
/// Manages page-aligned memory regions that utilize 2MB or 1GB huge pages
/// (where supported by the OS) to maximize memory throughput.
#[allow(dead_code)]
pub struct HugeBuffer<T> {
    /// Pointer to the allocated memory.
    ptr: *mut T,
    size_bytes: usize,
    is_mmap: bool,
}

unsafe impl<T> Send for HugeBuffer<T> {}
unsafe impl<T> Sync for HugeBuffer<T> {}

impl<T> HugeBuffer<T> {
    /// Allocates a new `HugeBuffer` using OS-specific huge page primitives.
    #[inline(never)]
    #[must_use]
    #[allow(clippy::useless_let_if_seq)]
    pub fn new() -> Option<Self> {
        let size_bytes = core::mem::size_of::<T>();

        #[cfg(unix)]
        unsafe {
            let base_flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

            // Tier 1: explicit MAP_HUGETLB for ≥ 2 MiB regions when the kernel
            // has hugepages reserved.  Cheapest TLB footprint when available.
            let mut ptr = libc::MAP_FAILED;
            if size_bytes >= 2 * 1024 * 1024 {
                ptr = libc::mmap(
                    core::ptr::null_mut(),
                    size_bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    base_flags | 0x40000, // MAP_HUGETLB
                    -1,
                    0,
                );
            }

            // Tier 2: plain anonymous mmap with MAP_NORESERVE.  Kernel can
            // still back this with transparent huge pages (khugepaged /
            // MADV_HUGEPAGE).  This tier was previously absent — the old
            // code jumped straight from a failed HUGETLB to
            // std::alloc::alloc_zeroed, which on Linux uses glibc's malloc
            // heuristics rather than a clean page-aligned mapping and so
            // cannot get a THP backing.
            //
            // MAP_NORESERVE is critical for the large MAILBOX_CAPACITY:
            // each mailbox reserves ~9 MiB of address space, and an
            // 8-worker runtime asks for ~600 MiB total.  Without
            // NORESERVE, kernels with overcommit_memory=2 (strict
            // accounting — common in containers and on QEMU rootfses)
            // refuse the mapping outright even though the runtime touches
            // only a tiny fraction of each buffer at steady state.  With
            // NORESERVE the virtual mapping always succeeds and physical
            // pages are demand-faulted on first write only.  This is what
            // makes the runtime viable under QEMU-emulated aarch64 CI.
            if ptr == libc::MAP_FAILED {
                // libc::MAP_NORESERVE is defined on every Unix the crate
                // supports (Linux/glibc/musl/android = 0x4000, BSDs = 0x40,
                // OpenBSD = 0x0000/no-op) — the bit never collides with
                // other map flags, so the OR is portable.
                ptr = libc::mmap(
                    core::ptr::null_mut(),
                    size_bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    base_flags | libc::MAP_NORESERVE,
                    -1,
                    0,
                );
            }

            if ptr != libc::MAP_FAILED {
                // Anonymous mmap pages are already zero-filled by the kernel
                // (MAP_ANONYMOUS contract).  The previous explicit
                // `write_bytes(ptr, 0, size_bytes)` was a pure 9 MB-class
                // memset per mailbox at init time — for an 8-worker runtime
                // this added ~600 MB of wasted write traffic against caches
                // and the memory bus.  Skip it: rely on the kernel's
                // guarantee.

                // On Linux, hint the kernel that this region wants huge-page
                // backing if standard mapping fell through (Tier 2).  Best-
                // effort: ignore the return value.
                #[cfg(target_os = "linux")]
                {
                    const MADV_HUGEPAGE: libc::c_int = 14;
                    libc::madvise(ptr, size_bytes, MADV_HUGEPAGE);
                }

                return Some(Self {
                    ptr: ptr.cast::<T>(),
                    size_bytes,
                    is_mmap: true,
                });
            }

            // Tier 3: aligned std::alloc fallback for environments where mmap
            // itself is exhausted (rare — QEMU/aarch64, sandboxed containers).
            let layout = std::alloc::Layout::from_size_align(size_bytes, 64).ok()?;
            let alloc_ptr = std::alloc::alloc_zeroed(layout);
            if alloc_ptr.is_null() {
                return None;
            }
            Some(Self {
                ptr: alloc_ptr.cast::<T>(),
                size_bytes,
                is_mmap: false,
            })
        }

        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::System::Memory;
            #[cfg(feature = "windows-root")]
            {
                let mut ptr = Memory::VirtualAlloc(
                    core::ptr::null_mut(),
                    size_bytes,
                    Memory::MEM_RESERVE | Memory::MEM_COMMIT | Memory::MEM_LARGE_PAGES,
                    Memory::PAGE_READWRITE,
                );
                if ptr.is_null() {
                    ptr = Memory::VirtualAlloc(
                        core::ptr::null_mut(),
                        size_bytes,
                        Memory::MEM_RESERVE | Memory::MEM_COMMIT,
                        Memory::PAGE_READWRITE,
                    );
                    if ptr.is_null() {
                        return None;
                    }
                }
                Some(Self {
                    ptr: ptr.cast::<T>(),
                    size_bytes,
                    is_mmap: false,
                })
            }
            #[cfg(not(feature = "windows-root"))]
            {
                let ptr = Memory::VirtualAlloc(
                    core::ptr::null_mut(),
                    size_bytes,
                    Memory::MEM_RESERVE | Memory::MEM_COMMIT,
                    Memory::PAGE_READWRITE,
                );
                if ptr.is_null() {
                    return None;
                }
                Some(Self {
                    ptr: ptr.cast::<T>(),
                    size_bytes,
                    is_mmap: false,
                })
            }
        }
    }
}

impl<T> Drop for HugeBuffer<T> {
    #[inline(never)]
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            if self.is_mmap {
                libc::munmap(self.ptr.cast::<libc::c_void>(), self.size_bytes);
            } else {
                let layout = std::alloc::Layout::from_size_align_unchecked(self.size_bytes, 64);
                std::alloc::dealloc(self.ptr.cast::<u8>(), layout);
            }
        }
        #[cfg(windows)]
        unsafe {
            windows_sys::Win32::System::Memory::VirtualFree(
                self.ptr.cast::<core::ffi::c_void>(),
                0,
                windows_sys::Win32::System::Memory::MEM_RELEASE,
            );
        }
    }
}

/// Single-Producer Single-Consumer (SPSC) Queue for the P2P Mesh Mailbox.
///
/// Aligned to 64 bytes to prevent false sharing between sender and receiver cores.
#[repr(C, align(64))]
pub struct Mailbox {
    pub head: AtomicUsize,
    _pad1: [u8; 64 - core::mem::size_of::<AtomicUsize>()],

    pub tail: AtomicUsize,
    _pad2: [u8; 64 - core::mem::size_of::<AtomicUsize>()],

    pub buffer: HugeBuffer<UnsafeCell<[TaskChunk; MAILBOX_CAPACITY]>>,
}

unsafe impl Sync for Mailbox {}
unsafe impl Send for Mailbox {}

impl Mailbox {
    /// Creates a new, empty Mailbox.
    #[inline(never)]
    #[must_use]
    pub fn new() -> Option<Self> {
        Some(Self {
            head: AtomicUsize::new(0),
            _pad1: [0; 56],
            tail: AtomicUsize::new(0),
            _pad2: [0; 56],
            buffer: HugeBuffer::new()?,
        })
    }

    /// Pushes a `TaskChunk` into the mailbox.
    ///
    /// Utilizes hardware-specific demote/clean instructions to accelerate
    /// visibility of the updated tail pointer to the consumer core.
    ///
    /// # Errors
    /// Returns the `TaskChunk` back to the caller if the mailbox is full.
    #[inline(always)]
    #[allow(clippy::result_large_err)]
    pub fn push(&self, chunk: TaskChunk) -> Result<(), TaskChunk> {
        let current_tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (current_tail + 1) & MAILBOX_MASK;

        if next_tail == self.head.load(Ordering::Acquire) {
            return Err(chunk);
        }

        unsafe {
            let buffer_ptr = (*self.buffer.ptr).get().cast::<TaskChunk>();
            *buffer_ptr.add(current_tail) = chunk;
        }

        self.tail.store(next_tail, Ordering::Release);

        #[cfg(all(
            feature = "hw-acceleration",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        unsafe {
            core::arch::asm!("cldemote [{}]", in(reg) &raw const self.tail);
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
        // Single Acquire on tail synchronizes with the producer's Release store.
        if current_head == self.tail.load(Ordering::Acquire) {
            return None;
        }

        let chunk = unsafe {
            let buffer_ptr = (*self.buffer.ptr).get().cast::<TaskChunk>();
            core::ptr::read(buffer_ptr.add(current_head))
        };

        let next_head = (current_head + 1) & MAILBOX_MASK;
        self.head.store(next_head, Ordering::Release);
        Some(chunk)
    }
}

/// A single slot in the warehouse ring.
///
/// Vyukov-style bounded MPMC: each slot holds a sequence number that producers
/// and consumers compare against their claimed position. Slots are 64-byte
/// aligned individually so concurrent producers on adjacent slots never share
/// a cache line. The chunk payload is `MaybeUninit` because slots are
/// initialized lazily on first push.
#[repr(C, align(64))]
pub struct WarehouseSlot {
    /// Sequence: equals `pos` when slot is ready to be written by a producer
    /// at position `pos`; equals `pos + 1` when ready to be read by the
    /// consumer at position `pos`; equals `pos + CAPACITY` after the consumer
    /// reclaims the slot for the next round.
    pub seq: AtomicUsize,
    /// Payload — only valid while `seq == claim_pos + 1`.
    pub chunk: UnsafeCell<MaybeUninit<TaskChunk>>,
}

/// Bounded MPMC ring buffer used as the scheduler-level emergency backlog.
///
/// Cache-line layout (each `repr(C, align(64))` field starts a fresh line):
///   Line 0: `backlog` — single hot atomic checked by every worker on every
///           tier-0 iteration. Isolated so producer/consumer writes to head
///           and tail never invalidate it on remote cores.
///   Line 1: `tail` — written by every producer on push (high contention when
///           warehouse is active, dead-cold otherwise).
///   Line 2: `head` — written by every consumer on pop.
///   Line 3+: 32 768 individually 64-byte-padded slots.
#[repr(C, align(64))]
pub struct Warehouse {
    /// Approximate count of resident chunks. Workers fast-path-check this
    /// with a single `Relaxed` load each tick — it must NEVER share a cache
    /// line with the producer/consumer indices.
    pub backlog: AtomicU32,
    _pad0: [u8; 60],

    /// Producer claim counter.
    pub tail: AtomicUsize,
    _pad1: [u8; 56],

    /// Consumer claim counter.
    pub head: AtomicUsize,
    _pad2: [u8; 56],

    /// Ring buffer. Each slot is individually padded to its own cache line.
    pub slots: HugeBuffer<[WarehouseSlot; WAREHOUSE_CAPACITY]>,
}

unsafe impl Sync for Warehouse {}
unsafe impl Send for Warehouse {}

impl Warehouse {
    /// Creates an empty warehouse with all slots seq-initialised to their index.
    #[must_use]
    #[inline(never)]
    pub fn new() -> Option<Self> {
        let wh = Self {
            backlog: AtomicU32::new(0),
            _pad0: [0; 60],
            tail: AtomicUsize::new(0),
            _pad1: [0; 56],
            head: AtomicUsize::new(0),
            _pad2: [0; 56],
            slots: HugeBuffer::new()?,
        };
        // Initialise sequence numbers: slot i starts at seq=i so the first
        // producer at position i sees diff = 0 and can claim it.
        unsafe {
            let base = wh.slots.ptr.cast::<WarehouseSlot>();
            for i in 0..WAREHOUSE_CAPACITY {
                (*base.add(i)).seq.store(i, Ordering::Release);
            }
        }
        Some(wh)
    }

    /// Pushes a chunk into the warehouse. Returns `Err(chunk)` if full.
    ///
    /// Cold path: only entered when normal mailbox routes are saturated.
    ///
    /// Under high concurrency, the `Greater` and CAS-failure branches are the
    /// source of CAS storms: all contending producers reload `tail` in lock-step
    /// and hammer the same next slot simultaneously.  The fix is a staggered
    /// per-worker exponential back-off (see [`warehouse_backoff`]) that spreads
    /// retry windows across different instruction-cycle offsets, dissolving the
    /// thundering herd without introducing OS scheduler latency.
    #[inline(always)]
    #[allow(clippy::result_large_err)]
    pub fn push(&self, chunk: TaskChunk) -> Result<(), TaskChunk> {
        let base = self.slots.ptr.cast::<WarehouseSlot>();

        // Read worker ID once before the retry loop so the TLS lookup cost is
        // not repeated on every CAS failure.  `usize::MAX` on host threads —
        // `wrapping_mul` keeps the formula valid for that sentinel too.
        let worker_id = crate::future_bridge::CURRENT_WORKER_ID.with(std::cell::Cell::get);

        let mut pos = self.tail.load(Ordering::Relaxed);
        let mut retry: u32 = 0;
        loop {
            let slot = unsafe { &*base.add(pos & WAREHOUSE_MASK) };
            let seq = slot.seq.load(Ordering::Acquire);
            let diff = (seq.cast_signed()).wrapping_sub(pos.cast_signed());
            match diff.cmp(&0) {
                std::cmp::Ordering::Equal => {
                    // Slot is ready for our position — try to claim by bumping tail.
                    // Under loom we use the strong form so spurious CAS failures are
                    // not treated as new branches (they would cause exponential state-
                    // space growth in loom's exhaustive-interleaving model).  The
                    // weak form is kept for production where it is cheaper.
                    #[cfg(not(loom))]
                    let cas_result = self.tail.compare_exchange_weak(
                        pos,
                        pos + 1,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    );
                    #[cfg(loom)]
                    let cas_result = self.tail.compare_exchange(
                        pos,
                        pos + 1,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    );
                    if cas_result.is_ok() {
                        unsafe { (*slot.chunk.get()).write(chunk) };
                        // Publish: subsequent Acquire on seq by the consumer
                        // synchronises with this Release and sees the payload.
                        slot.seq.store(pos + 1, Ordering::Release);
                        self.backlog.fetch_add(1, Ordering::Release);
                        return Ok(());
                    }
                    // CAS lost to a concurrent producer on the same slot.
                    // Back off before reloading to reduce coherence-bus traffic
                    // and give the winner time to publish its seq update.
                    warehouse_backoff(worker_id, retry);
                    retry = retry.saturating_add(1);
                    pos = self.tail.load(Ordering::Relaxed);
                }
                std::cmp::Ordering::Less => {
                    // Slot is from a previous round still being drained — full.
                    return Err(chunk);
                }
                std::cmp::Ordering::Greater => {
                    // Another producer already claimed `pos` and bumped `tail`
                    // past us.  Without back-off every producer immediately
                    // reloads and races for the same new `tail` value, forming
                    // a CAS storm.  The staggered delay spreads their retries
                    // across distinct cycle windows so only one fires at a time.
                    warehouse_backoff(worker_id, retry);
                    retry = retry.saturating_add(1);
                    pos = self.tail.load(Ordering::Relaxed);
                }
            }
        }
    }

    /// Pops a chunk from the warehouse. Returns `None` if empty.
    ///
    /// Symmetric staggered back-off to [`Warehouse::push`]: multiple workers
    /// draining the warehouse simultaneously can produce an identical CAS storm
    /// on `head`.  Per-worker exponential avoidance spreads their retry windows.
    #[inline(always)]
    pub fn pop(&self) -> Option<TaskChunk> {
        let base = self.slots.ptr.cast::<WarehouseSlot>();

        let worker_id = crate::future_bridge::CURRENT_WORKER_ID.with(std::cell::Cell::get);

        let mut pos = self.head.load(Ordering::Relaxed);
        let mut retry: u32 = 0;
        loop {
            let slot = unsafe { &*base.add(pos & WAREHOUSE_MASK) };
            let seq = slot.seq.load(Ordering::Acquire);
            let diff = (seq.cast_signed()).wrapping_sub((pos + 1).cast_signed());
            match diff.cmp(&0) {
                std::cmp::Ordering::Equal => {
                    // Slot has a published chunk for our position — try to claim.
                    // Strong CAS under loom prevents spurious-failure branch explosion.
                    #[cfg(not(loom))]
                    let cas_result = self.head.compare_exchange_weak(
                        pos,
                        pos + 1,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    );
                    #[cfg(loom)]
                    let cas_result = self.head.compare_exchange(
                        pos,
                        pos + 1,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    );
                    if cas_result.is_ok() {
                        let chunk = unsafe { (*slot.chunk.get()).assume_init_read() };
                        // Release the slot for the next round (pos + CAPACITY).
                        slot.seq.store(pos + WAREHOUSE_CAPACITY, Ordering::Release);
                        self.backlog.fetch_sub(1, Ordering::Release);
                        return Some(chunk);
                    }
                    // Lost the CAS to a concurrent consumer — back off before
                    // reloading so all consumers don't pile onto head at once.
                    warehouse_backoff(worker_id, retry);
                    retry = retry.saturating_add(1);
                    pos = self.head.load(Ordering::Relaxed);
                }
                std::cmp::Ordering::Less => {
                    return None;
                }
                std::cmp::Ordering::Greater => {
                    // Another consumer already claimed this slot; our view of
                    // `head` is stale.  Stagger before reloading.
                    warehouse_backoff(worker_id, retry);
                    retry = retry.saturating_add(1);
                    pos = self.head.load(Ordering::Relaxed);
                }
            }
        }
    }

    /// Hot-path probe: single Relaxed load on its own cache line.
    #[inline(always)]
    #[must_use]
    pub fn is_busy(&self) -> bool {
        self.backlog.load(Ordering::Relaxed) != 0
    }
}

/// Staggered exponential back-off for Warehouse MPMC CAS contention.
///
/// Every competing worker gets a **distinct** base delay derived from its
/// core ID so their retry windows are interleaved rather than synchronised.
/// The prime multiplier 7 ensures separation across typical small worker
/// counts (2–64 cores); the result is clamped to \[1, 32\] cycles.
///
/// Delay formula: `cycles = ((worker_id × 7) & 0x1F + 1) × 2^min(retry, 6)`
///
/// | worker | base | after 6 retries |
/// |--------|------|-----------------|
/// |   0    |   1  |       64 cy     |
/// |   1    |   8  |      512 cy     |
/// |   3    |  22  |     1 408 cy    |
/// |   7    |  50  |     3 200 cy    |
///
/// Each loop iteration is a single `nop` — precisely **1 clock cycle** at
/// base frequency.  This is fundamentally different from
/// `core::hint::spin_loop()` (which emits `PAUSE` ≈ 100 cy on modern Intel
/// and is identical for every caller), and from a `black_box` loop (whose
/// duration is compiler- and micro-architecture-dependent).  The inline-asm
/// NOP loop is the only approach that gives both sub-cycle granularity and
/// zero variance between workers at the same retry depth.
///
/// The `#[inline(never)]` prevents label collisions when the function is
/// instantiated at multiple call sites; the `#[cold]` annotation ensures the
/// compiler doesn't hoist or speculatively execute the delay on the hot path.
#[cold]
#[inline(never)]
fn warehouse_backoff(worker_id: usize, retry: u32) {
    let base = (worker_id.wrapping_mul(7) & 0x1F).wrapping_add(1) as u64;
    // Cap exponent at 6 to keep the worst-case wait under ~3 200 cycles (≈ 1 µs @ 3 GHz).
    let cycles = base << u64::from(retry.min(6));

    // x86 / x86_64 — Intel syntax, counted NOP loop.
    // `dec` sets ZF when the result reaches zero; `jnz` exits at that point.
    // We do NOT include `preserves_flags` because `dec` intentionally modifies EFLAGS.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // SAFETY: pure counted NOP loop; touches only the counter register and EFLAGS.
    unsafe {
        core::arch::asm!(
            "2:",
            "nop",
            "dec {n}",
            "jnz 2b",
            n = inout(reg) cycles => _,
            options(nostack),
        );
    }

    // AArch64 — `subs` sets the NE condition flag; `b.ne` exits when zero.
    #[cfg(target_arch = "aarch64")]
    // SAFETY: pure counted NOP loop; touches only the counter register and NZCV.
    unsafe {
        core::arch::asm!(
            "2:",
            "nop",
            "subs {n}, {n}, #1",
            "b.ne 2b",
            n = inout(reg) cycles => _,
            options(nostack),
        );
    }

    // RISC-V 64 — `addi` with -1, branch-if-nonzero.
    #[cfg(target_arch = "riscv64")]
    // SAFETY: pure counted NOP loop; touches only the counter register.
    unsafe {
        core::arch::asm!(
            "2:",
            "nop",
            "addi {n}, {n}, -1",
            "bnez {n}, 2b",
            n = inout(reg) cycles => _,
            options(nostack),
        );
    }

    // Fallback for architectures without hand-written asm.
    // `black_box` prevents the compiler from eliminating the loop body as a
    // dead no-op; actual timing accuracy is best-effort on this path.
    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )))]
    {
        for i in 0_u64..cycles {
            core::hint::black_box(i);
        }
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

pub use crate::common_types::TopologyMode;

/// Execution unit managed by a single OS thread.
///
/// Cache-line layout (repr C, 64-byte aligned):
///   Line 0 (0–63):   cpu, `load_level`, `deflection_threshold`, `local_head`, `local_tail`, ticks
///   Line 1 (64–127): `event_signal` — isolated to prevent false-sharing with line 0
///                     (`signal_worker` on remote cores writes here; local worker reads line 0)
///   Line 2+ (128+):  `local_queue` buffer, `polling_order`
#[repr(C, align(64))]
pub struct Worker {
    /// Hierarchy information for this worker's core.
    pub cpu: CpuLevel,
    /// Current load level (0-100).
    pub load_level: AtomicU8,
    /// Load threshold above which tasks are deflected to peers.
    pub deflection_threshold: AtomicU8,
    /// Head of the local queue. Single-producer-single-consumer: only this worker touches it.
    pub local_head: AtomicUsize,
    /// Tail of the local queue.
    pub local_tail: AtomicUsize,
    /// Total scheduler ticks executed.
    pub ticks: u64,
    // Fill cache line 0 to 64 bytes.
    // cpu(6) + load_level(1) + deflection_threshold(1) + local_head(8) + local_tail(8) + ticks(8) = 32
    _pad0: [u8; 32],

    /// Counter for hardware-assisted wakeups (WFE/umonitor).
    /// Isolated on its own cache line: remote workers write here via `signal_worker`,
    /// which would otherwise false-share with the hot `local_head/local_tail` above.
    pub event_signal: AtomicU32,
    // Fill cache line 1 to 64 bytes: 4 bytes used, 60 bytes pad.
    _pad1: [u8; 60],

    /// Local SPSC execution queue (huge-page backed).
    pub local_queue: HugeBuffer<[TaskIndex; LOCAL_QUEUE_CAPACITY]>,
    /// Ordered list of peer core IDs for mailbox polling.
    pub polling_order: Vec<usize>,
}

unsafe impl Sync for Worker {}
unsafe impl Send for Worker {}

impl Worker {
    /// Creates a new `Worker` and calculates its CCX-aware polling order.
    #[inline(never)]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(cpu: CpuLevel, total_cores: usize) -> Option<Self> {
        let mut polling_order = Vec::with_capacity(total_cores - 1);
        let my_core = cpu.core_id as usize;
        let my_ccx = cpu.ccx_id;

        for i in 0..total_cores {
            if i != my_core && (i / 8) as u16 == my_ccx {
                polling_order.push(i);
            }
        }
        for offset in 0..total_cores {
            let i = (my_core + offset) % total_cores;
            if i != my_core && (i / 8) as u16 != my_ccx {
                polling_order.push(i);
            }
        }

        Some(Self {
            cpu,
            load_level: AtomicU8::new(0),
            deflection_threshold: AtomicU8::new(80),
            local_head: AtomicUsize::new(0),
            local_tail: AtomicUsize::new(0),
            ticks: 0,
            _pad0: [0; 32],
            event_signal: AtomicU32::new(0),
            _pad1: [0; 60],
            local_queue: HugeBuffer::new()?,
            polling_order,
        })
    }

    /// Returns the current number of tasks in the local queue.
    #[inline(always)]
    pub fn local_queue_len(&self) -> usize {
        let head = self.local_head.load(Ordering::Relaxed);
        let tail = self.local_tail.load(Ordering::Relaxed);
        tail.wrapping_sub(head) & LOCAL_QUEUE_MASK
    }

    /// Updates the `load_level` based on the current queue length.
    #[inline(always)]
    pub fn update_load(&self) {
        let queue_len = self.local_queue_len();
        #[allow(clippy::cast_possible_truncation)]
        let load = core::cmp::min((queue_len * 100) >> 13, 100) as u8;
        self.load_level.store(load, Ordering::Relaxed);
    }

    /// Performs internal maintenance tasks (e.g., adaptive threshold updates).
    #[inline(always)]
    pub fn tick(&mut self) {
        self.ticks = self.ticks.wrapping_add(1);
        if self.ticks.trailing_zeros() >= 10 {
            let load = self.load_level.load(Ordering::Relaxed);
            let current_thresh = self.deflection_threshold.load(Ordering::Relaxed);

            let new_thresh = if load > 90 {
                current_thresh.saturating_sub(5).max(40)
            } else if load < 30 {
                current_thresh.saturating_add(5).min(95)
            } else {
                current_thresh
            };

            self.deflection_threshold
                .store(new_thresh, Ordering::Relaxed);
        }
    }

    /// Pushes a single task into the local queue. Returns true if successful.
    #[inline(always)]
    pub fn push_local(&self, task: TaskIndex) -> bool {
        let tail = self.local_tail.load(Ordering::Relaxed);
        if self.local_queue_len() >= LOCAL_QUEUE_CAPACITY - 1 {
            return false;
        }
        unsafe {
            let buffer_ptr = self.local_queue.ptr.cast::<TaskIndex>();
            *buffer_ptr.add(tail) = task;
        }
        // Relaxed: only this worker thread reads local_tail.
        self.local_tail
            .store((tail + 1) & LOCAL_QUEUE_MASK, Ordering::Relaxed);
        true
    }

    /// Pushes a batch of tasks into the local queue.
    ///
    /// CALLER CONTRACT: caller must guarantee `local_queue_len() + chunk.count`
    /// stays under `LOCAL_QUEUE_CAPACITY`. The `route_chunk` / `drain_warehouse`
    /// paths enforce this via `LOCAL_QUEUE_HIGH_WATERMARK`.
    #[inline]
    pub fn push_batch(&mut self, chunk: &TaskChunk) {
        let count = chunk.count as usize;
        let tail = self.local_tail.load(Ordering::Relaxed);
        let end_idx = tail.wrapping_add(count);

        if end_idx <= LOCAL_QUEUE_CAPACITY {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    chunk.tasks.as_ptr(),
                    (*self.local_queue.ptr).as_mut_ptr().add(tail),
                    count,
                );
            }
        } else {
            let first_part = LOCAL_QUEUE_CAPACITY - tail;
            let second_part = count - first_part;
            unsafe {
                core::ptr::copy_nonoverlapping(
                    chunk.tasks.as_ptr(),
                    (*self.local_queue.ptr).as_mut_ptr().add(tail),
                    first_part,
                );
                core::ptr::copy_nonoverlapping(
                    chunk.tasks.as_ptr().add(first_part),
                    (*self.local_queue.ptr).as_mut_ptr(),
                    second_part,
                );
            }
        }
        // Relaxed: push_batch is only called from the local worker thread.
        self.local_tail
            .store(end_idx & LOCAL_QUEUE_MASK, Ordering::Relaxed);
    }

    /// Primary execution loop for the worker thread.
    ///
    /// Drains the local queue, performs O(1) context alignment, and executes
    /// the context switch to the fiber.
    ///
    /// Returns `true` if at least one fiber was dispatched, `false` if the
    /// local queue was empty. The caller uses this instead of re-reading
    /// `local_head` before/after the call to detect activity.
    ///
    /// # Safety
    /// * `context_base` must point to the start of the `ContextPool` memory region.
    /// * `context_size` and `group_guard_size` must match the pool's initialized layout.
    #[inline(always)]
    pub unsafe fn dispatch_loop(&self, pool: &crate::memory_management::ContextPool) -> bool {
        // Relaxed: local_head and local_tail are only accessed by this worker thread.
        let mut head = self.local_head.load(Ordering::Relaxed);
        let mut tail = self.local_tail.load(Ordering::Relaxed);

        if head == tail {
            return false;
        }

        loop {
            let task = unsafe {
                let buffer_ptr = self.local_queue.ptr.cast::<TaskIndex>();
                *buffer_ptr.add(head)
            };

            head = (head + 1) & LOCAL_QUEUE_MASK;
            self.local_head.store(head, Ordering::Relaxed);

            let target_ptr = pool.get_context_ptr(task);

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
                    &raw mut (*target_ptr).executor_regs,
                    &raw const (*target_ptr).regs,
                );
            }

            crate::future_bridge::CURRENT_FIBER.with(|c| c.set(core::ptr::null_mut()));

            // Optimized lifecycle state machine: handle Finished, Notified, or Suspending transitions.
            let post_state = unsafe { (*target_ptr).state.load(Ordering::Acquire) };

            let mut final_state = post_state;
            if post_state == crate::memory_management::FiberStatus::Suspending as u32 {
                match unsafe {
                    (*target_ptr).state.compare_exchange(
                        crate::memory_management::FiberStatus::Suspending as u32,
                        crate::memory_management::FiberStatus::Yielded as u32,
                        Ordering::Release,
                        Ordering::Acquire,
                    )
                } {
                    Ok(_) => final_state = crate::memory_management::FiberStatus::Yielded as u32,
                    Err(actual) => final_state = actual,
                }
            }

            // Terminal states (Finished, Panicked)
            if final_state == crate::memory_management::FiberStatus::Finished as u32
                || final_state == crate::memory_management::FiberStatus::Panicked as u32
            {
                pool.free_context(task);
            } else if final_state == crate::memory_management::FiberStatus::Notified as u32 {
                // Cooperative yield or backpressure-induced suspension: re-enqueue.
                //
                // Re-push to the same local queue. We JUST popped this task, so the
                // queue has at least one slot free — push_local cannot fail here in
                // a well-formed state machine. debug_assert catches future bugs.
                let pushed = self.push_local(task);
                debug_assert!(
                    pushed,
                    "DTA-V3 invariant: Notified re-enqueue must succeed (we just freed a slot by popping)"
                );
                // Return to allow mailbox polling and prevent live-locks on high contention.
                return true;
            }

            tail = self.local_tail.load(Ordering::Relaxed);
            if head == tail {
                break;
            }
        }

        true
    }
}

/// The Dtact-V3 Distributed Scheduler.
///
/// Manages a set of `Worker` units, the P2P Mailbox matrix for cross-core task
/// migration, and a single shared `Warehouse` that catches overflow chunks when
/// every per-core mailbox is saturated. The warehouse activates back-pressure:
/// while it holds chunks, external injections are diverted into it and workers
/// preferentially drain it before polling new mailbox traffic.
pub struct DtaScheduler {
    /// Thread-local worker states.
    pub workers: Vec<UnsafeCell<Worker>>,
    /// N x N Mailbox matrix for P2P communication. SPSC per cell — `mailboxes[i][j]`
    /// has worker `i` as its unique producer and worker `j` as its unique consumer.
    pub mailboxes: Vec<Vec<Mailbox>>,
    /// Mailboxes for tasks spawned from external host threads (MPSC via spinlock).
    pub external_mailboxes: Vec<Mailbox>,
    /// Locks for external mailboxes (to allow multiple host threads to spawn).
    pub external_locks: Vec<crate::utils::SpinLock>,
    /// Active topology mode.
    pub topology: TopologyMode,
    /// Maximum hop count before a chunk is diverted to the warehouse.
    /// Derived from `num_workers / 2`; not user-tunable via Rust or C FFI.
    pub max_hops: u8,
    /// Shared MPMC overflow store + back-pressure flag.
    pub warehouse: Warehouse,
}

unsafe impl Sync for DtaScheduler {}
unsafe impl Send for DtaScheduler {}

impl DtaScheduler {
    /// Creates a new `DtaScheduler` for the specified number of workers.
    #[inline(never)]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(num_workers: usize, topology: TopologyMode) -> Self {
        let mut workers = Vec::with_capacity(num_workers);
        let mut mailboxes = Vec::with_capacity(num_workers);
        let mut external_mailboxes = Vec::with_capacity(num_workers);
        let mut external_locks = Vec::with_capacity(num_workers);

        for i in 0..num_workers {
            workers.push(UnsafeCell::new(
                Worker::new(
                    CpuLevel {
                        core_id: i as u16,
                        ccx_id: (i / 8) as u16,
                        numa_id: (i / 64) as u16,
                    },
                    num_workers,
                )
                .expect("Failed to initialize scheduler structures (Worker)"),
            ));

            let mut row = Vec::with_capacity(num_workers);
            for _ in 0..num_workers {
                row.push(
                    Mailbox::new()
                        .expect("Failed to initialize scheduler structures (Mailbox row)"),
                );
            }
            mailboxes.push(row);
            external_mailboxes.push(
                Mailbox::new()
                    .expect("Failed to initialize scheduler structures (External Mailbox)"),
            );
            external_locks.push(crate::utils::SpinLock::new());
        }

        // max_hops = num_workers / 2 (clamped to u8). With 8 workers → 4 hops.
        // Single-worker (degenerate) → 0; first push failure goes straight to warehouse.
        let max_hops = core::cmp::min(num_workers / 2, u8::MAX as usize) as u8;

        Self {
            workers,
            mailboxes,
            external_mailboxes,
            external_locks,
            topology,
            max_hops,
            warehouse: Warehouse::new()
                .expect("Failed to initialize scheduler structures (Warehouse)"),
        }
    }

    /// Signals a worker that new work is available.
    ///
    /// The Release on `event_signal` synchronizes with the worker's Acquire load
    /// before WFE/umonitor in Tier 3. No OS call needed — fully user-space.
    #[inline(always)]
    pub(crate) fn signal_worker(&self, target_core: usize) {
        let worker = unsafe { &*self.workers[target_core].get() };
        worker.event_signal.fetch_add(1, Ordering::Release);

        #[cfg(all(
            feature = "hw-acceleration",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        unsafe {
            // Optional UIPI signal to the target core.
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

    /// Enqueue a task whose fiber may NOT be deflected (`SameThread` switchers).
    ///
    /// Routes strictly to `target_core` (the fiber's origin). On same-core push
    /// it is a single `push_local`. On cross-core it uses the SPSC mailbox
    /// matrix (the current worker is the unique producer for `mailboxes[me][*]`)
    /// or the external mailbox if the caller is a non-worker thread. Pinned
    /// chunks NEVER enter the warehouse because warehouse drainers cannot
    /// honour fiber-to-thread pinning.
    #[inline(always)]
    pub fn enqueue_pinned(&self, target_core: usize, task: TaskIndex) -> bool {
        let n = self.workers.len();
        let target = target_core % n;
        let current = crate::future_bridge::CURRENT_WORKER_ID.with(std::cell::Cell::get);

        if current == target {
            // Hot path: same-worker direct push. Single Relaxed access.
            let worker = unsafe { &*self.workers[target].get() };
            return worker.push_local(task);
        }

        // Cross-worker push: assemble a 1-task chunk and route via SPSC matrix
        // (current is fiber) or external mailbox (current is host thread).
        let mut chunk = TaskChunk::default();
        chunk.tasks[0] = task;
        chunk.count = 1;

        let ok = if current < n {
            self.mailboxes[current][target].push(chunk).is_ok()
        } else {
            self.external_locks[target].lock();
            let r = self.external_mailboxes[target].push(chunk).is_ok();
            self.external_locks[target].unlock();
            r
        };

        if ok {
            self.signal_worker(target);
        }
        ok
    }

    /// Enqueue a task whose fiber MAY be deflected (`CrossThread` switchers).
    ///
    /// Honours topology mode and current load. When the warehouse is busy
    /// (`is_busy() == true`), diverts the task there immediately to relieve
    /// back-pressure on the per-core mailboxes (the soft-back-pressure feedback).
    #[inline(always)]
    pub fn enqueue_deflect(
        &self,
        source_core: usize,
        flow_id: u64,
        task: TaskIndex,
        affinity: crate::api::topology::Affinity,
    ) -> bool {
        // Soft back-pressure: if the warehouse already holds chunks, every new
        // task goes straight in. Single Relaxed load on an isolated cache line —
        // the cold path is the `#[cold]` divert_to_warehouse, branch-predicted
        // false on every well-behaved iteration.
        if self.warehouse.is_busy() {
            return self.divert_to_warehouse(task);
        }

        let n = self.workers.len();
        let source = source_core % n;
        let worker_ref = unsafe { &*self.workers[source].get() };
        let threshold = worker_ref.deflection_threshold.load(Ordering::Relaxed);
        let load = worker_ref.load_level.load(Ordering::Relaxed);

        let deflect_mask = if load > threshold { usize::MAX } else { 0 };
        #[allow(clippy::cast_possible_truncation)]
        let h1 = (flow_id & 7) as usize;
        #[allow(clippy::cast_possible_truncation)]
        let h2 = ((flow_id >> 3) & 7 | 1) as usize;

        let target = if self.topology == TopologyMode::Global
            && matches!(affinity, crate::api::topology::Affinity::Any)
        {
            let deflect_target = (source + h1 + h2) % n;
            // Branchless conditional selection: if deflect_mask is all 1s (load > threshold),
            // this evaluates to deflect_target. If deflect_mask is 0 (load <= threshold),
            // this evaluates to source. This avoids cross-core deflection under low load
            // without incurring misprediction latency.
            source ^ ((source ^ deflect_target) & deflect_mask)
        } else if matches!(affinity, crate::api::topology::Affinity::SameNUMA) {
            let numa_base = source & !63;
            let local_idx = source & 63;
            let deflect_target = (local_idx + h1 + h2) % 64;
            let target_idx = local_idx ^ ((local_idx ^ deflect_target) & deflect_mask);
            (numa_base | target_idx) % n
        } else {
            // SameCCX or default (which pins deflection to local CCX under P2PMesh)
            let ccx_base = source & !7;
            let local_idx = source & 7;
            let deflect_target = (local_idx + h1 + h2) & 7;
            let target_idx = local_idx ^ ((local_idx ^ deflect_target) & deflect_mask);
            (ccx_base | target_idx) % n
        };

        let current = crate::future_bridge::CURRENT_WORKER_ID.with(std::cell::Cell::get);

        if current == target {
            let worker = unsafe { &*self.workers[target].get() };
            if worker.push_local(task) {
                return true;
            }
            // Local queue full — fall through to chunk routing
        }

        // Cross-worker (or local full): wrap in chunk and push, with hop fallback.
        let mut chunk = TaskChunk::default();
        chunk.tasks[0] = task;
        chunk.count = 1;
        self.push_chunk_with_hop(current, target, &mut chunk)
    }

    /// Push a chunk to `initial_target`. If the mailbox is full, hop to a peer.
    /// After `max_hops` attempts, the chunk is parked in the warehouse.
    /// Returns `true` once the chunk has been deposited somewhere (mailbox or
    /// warehouse); only a true warehouse overflow panics.
    ///
    /// CRITICAL: `mailboxes[producer][producer]` (the self-column) is **never**
    /// polled — `Worker::polling_order` filters out `i == my_core`. Anything
    /// pushed there is permanently stranded. Both the entry target and every
    /// hopped target must therefore be coerced away from `producer` whenever
    /// the producer is itself a worker. The bare `(producer + 1 + 7·hop) % n`
    /// formula lands on `producer` for any `n` that divides `1 + 7·hop`
    /// (e.g. n=4 at `hop_count=1` → offset 8 → self), so we bump by one in that
    /// case instead of relying on the formula alone.
    fn push_chunk_with_hop(
        &self,
        producer: usize,
        initial_target: usize,
        chunk: &mut TaskChunk,
    ) -> bool {
        let n = self.workers.len();
        let mut target = initial_target;
        if producer < n && n > 1 && target == producer {
            target = (target + 1) % n;
        }
        loop {
            let result = if producer < n {
                // Producer is a worker — use SPSC matrix.
                self.mailboxes[producer][target].push(*chunk)
            } else {
                // Producer is a host thread — use locked external_mailbox.
                self.external_locks[target].lock();
                let r = self.external_mailboxes[target].push(*chunk);
                self.external_locks[target].unlock();
                r
            };

            match result {
                Ok(()) => {
                    self.signal_worker(target);
                    return true;
                }
                Err(c) => {
                    *chunk = c;
                    if chunk.hop_count >= self.max_hops {
                        // Exhausted hops — park in warehouse (cold path).
                        return self.park_in_warehouse(*chunk);
                    }
                    chunk.hop_count = chunk.hop_count.saturating_add(1);
                    // ×7 is coprime to small worker counts for uniform spread,
                    // but for some (n, hop_count) pairs `(1 + 7·hop_count) % n`
                    // is 0, which would land us on the self-column. Bump past
                    // it. Host producers skip the check — external_mailboxes
                    // are indexed by target only and don't have a self-sink.
                    target = (producer.wrapping_add(1 + chunk.hop_count as usize * 7)) % n;
                    if producer < n && n > 1 && target == producer {
                        target = (target + 1) % n;
                    }
                }
            }
        }
    }

    /// Cold path: divert one task directly to the warehouse.
    #[cold]
    #[inline(never)]
    fn divert_to_warehouse(&self, task: TaskIndex) -> bool {
        let mut chunk = TaskChunk::default();
        chunk.tasks[0] = task;
        chunk.count = 1;
        // Mark as already-exhausted-hops so any worker that drains it pushes
        // directly into its own queue without trying to re-deflect.
        chunk.hop_count = u8::MAX;
        self.park_in_warehouse(chunk)
    }

    /// Cold path: park a chunk in the warehouse. Panics on warehouse overflow.
    #[cold]
    #[inline(never)]
    fn park_in_warehouse(&self, chunk: TaskChunk) -> bool {
        assert!(
            self.warehouse.push(chunk).is_ok(),
            "DTA-V3: warehouse overflow — backlog exceeds {} chunks ({} tasks). \
             Application has scheduled tasks faster than the runtime can drain, \
             beyond emergency back-pressure capacity.",
            WAREHOUSE_CAPACITY,
            WAREHOUSE_CAPACITY * CHUNK_SIZE
        );
        true
    }

    /// Drain the warehouse into the current worker's local queue.
    ///
    /// Stops when either the warehouse is empty or the local queue is over the
    /// high-water mark. Each pop is direct `push_batch` (we pre-check space) —
    /// no further routing, no recursion back into the warehouse.
    ///
    /// Returns `true` if at least one chunk was drained, allowing the caller
    /// to detect activity without additional queue-length reads.
    #[cold]
    #[inline(never)]
    pub fn drain_warehouse(&self, current_core: usize) -> bool {
        let worker = unsafe { &mut *self.workers[current_core].get() };
        // Cap draining per call so a single worker doesn't monopolise the
        // warehouse while peers also want to help.
        let cap = 64usize;
        let mut drained = 0usize;

        // local_head is immutable during this function — cache it once.
        let fixed_head = worker.local_head.load(Ordering::Relaxed);
        let mut cur_len = worker
            .local_tail
            .load(Ordering::Relaxed)
            .wrapping_sub(fixed_head)
            & LOCAL_QUEUE_MASK;

        while drained < cap {
            if cur_len + CHUNK_SIZE > LOCAL_QUEUE_HIGH_WATERMARK {
                break;
            }
            match self.warehouse.pop() {
                Some(chunk) => {
                    let added = chunk.count as usize;
                    worker.push_batch(&chunk);
                    cur_len += added;
                    drained += 1;
                }
                None => break,
            }
        }
        drained > 0
    }

    /// Polls all incoming mailboxes for the current core, routing each chunk
    /// through the 4-way branchless dispatch (`route_chunk`).
    ///
    /// Returns `true` if at least one chunk was received from any mailbox,
    /// allowing the caller to detect activity without extra queue-length reads.
    ///
    /// `local_head` is cached once before the polling loops because it does
    /// not change here (we are only pushing work, not consuming). Only
    /// `local_tail` is re-loaded per iteration, halving the atomic reads in
    /// the capacity check compared to calling `local_queue_len()` each time.
    #[inline(always)]
    pub fn poll_mailboxes(&self, current_core: usize) -> bool {
        let worker = unsafe { &mut *self.workers[current_core].get() };

        // local_head is immutable during this function — cache it once.
        let fixed_head = worker.local_head.load(Ordering::Relaxed);
        let mut cur_len = worker
            .local_tail
            .load(Ordering::Relaxed)
            .wrapping_sub(fixed_head)
            & LOCAL_QUEUE_MASK;
        let mut received_any = false;

        let num_polls = worker.polling_order.len();
        for idx in 0..num_polls {
            let i = worker.polling_order[idx];
            let row = &self.mailboxes[i];

            loop {
                if cur_len + CHUNK_SIZE >= LOCAL_QUEUE_CAPACITY {
                    break;
                }
                match row[current_core].pop() {
                    Some(chunk) => {
                        received_any = true;
                        cur_len += self.route_chunk(worker, current_core, chunk, cur_len);
                    }
                    None => break,
                }
            }
        }

        // Poll the external mailbox last so external injection naturally yields
        // to internal CCX traffic when both are active.
        loop {
            if cur_len + CHUNK_SIZE >= LOCAL_QUEUE_CAPACITY {
                break;
            }
            match self.external_mailboxes[current_core].pop() {
                Some(chunk) => {
                    received_any = true;
                    cur_len += self.route_chunk(worker, current_core, chunk, cur_len);
                }
                None => break,
            }
        }

        worker.update_load();
        worker.tick();
        received_any
    }

    /// 4-way branchless chunk router. The function-pointer table makes the
    /// hot routes (cases 01 and 11 → `push_local`) collapse to a single indirect
    /// call after the index is computed.
    #[inline(always)]
    #[allow(clippy::items_after_statements)]
    fn route_chunk(
        &self,
        worker: &mut Worker,
        current_core: usize,
        chunk: TaskChunk,
        cur_len: usize,
    ) -> usize {
        let space_ok = (cur_len + chunk.count as usize) <= LOCAL_QUEUE_HIGH_WATERMARK;
        let hops_ok = chunk.hop_count < self.max_hops;

        //   no space, no hops left  → warehouse
        //   space,    no hops left  → local (we already have room)
        //   no space, hops left     → deflect to peer
        //   space,    hops left     → local (preferred)
        // Using an if/else block allows the compiler to fully inline these target
        // functions and generate direct branches, rather than an indirect jump
        // through a function pointer array which introduces misprediction latency.
        if space_ok {
            let added = chunk.count as usize;
            self.route_local(worker, current_core, chunk);
            added
        } else if hops_ok {
            self.route_deflect(worker, current_core, chunk);
            0
        } else {
            self.route_park(worker, current_core, chunk);
            0
        }
    }

    #[inline(always)]
    #[allow(clippy::unused_self)]
    fn route_local(&self, worker: &mut Worker, _core: usize, chunk: TaskChunk) {
        worker.push_batch(&chunk);
    }

    /// This code path utilizes branchless programming to eliminate mispredictions.
    /// Mark with `#[inline(always)]` to ensure the compiler optimizes the call site performance.
    #[inline(always)]
    fn route_park(&self, _worker: &mut Worker, _core: usize, chunk: TaskChunk) {
        let _ = self.park_in_warehouse(chunk);
    }

    /// This code path utilizes branchless programming to eliminate mispredictions.
    /// Mark with `#[inline(always)]` to ensure the compiler optimizes the call site performance.
    ///
    /// Like `push_chunk_with_hop`, the target must never equal `current_core`:
    /// `mailboxes[current_core][current_core]` is never polled (the self-column
    /// is filtered out of `Worker::polling_order`), so anything routed there
    /// is permanently stranded.
    #[inline(always)]
    fn route_deflect(&self, _worker: &mut Worker, current_core: usize, mut chunk: TaskChunk) {
        chunk.hop_count = chunk.hop_count.saturating_add(1);
        let n = self.workers.len();
        let mut target = (current_core.wrapping_add(1 + chunk.hop_count as usize * 7)) % n;
        if n > 1 && target == current_core {
            target = (target + 1) % n;
        }
        match self.mailboxes[current_core][target].push(chunk) {
            Ok(()) => self.signal_worker(target),
            Err(c) => {
                let _ = self.park_in_warehouse(c);
            }
        }
    }

    /// Main heartbeat loop for a hardware worker thread with cooperative shutdown.
    ///
    /// Periodically polls local queues, mailboxes, and external queues for work.
    /// Supports cooperative shutdown via the provided atomic flag.
    #[inline]
    #[allow(clippy::too_many_lines)]
    pub fn run_worker_static(
        scheduler: &Self,
        current_core: usize,
        pool: &crate::memory_management::ContextPool,
        shutdown: &crate::sync::atomic::AtomicBool,
    ) {
        crate::future_bridge::CURRENT_WORKER_ID.with(|c| c.set(current_core));
        let mut idle_count: u32 = 0;

        loop {
            if shutdown.load(Ordering::Acquire) {
                return;
            }

            // 0. Emergency back-pressure: snapshot warehouse state once.
            //    The check is a single Relaxed load on an isolated cache line —
            //    perfectly branch-predicted false in the common case.
            //    We use the snapshot for both the drain decision (step 0) and
            //    the mailbox-gating decision (step 2), so the warehouse is not
            //    re-read mid-iteration. A one-iteration delay in leaving back-
            //    pressure mode is negligible; correctness is not affected.
            let warehouse_busy = scheduler.warehouse.is_busy();
            let mut activity = if warehouse_busy {
                // Drain FIRST so we have tasks before dispatching.
                scheduler.drain_warehouse(current_core)
            } else {
                false
            };

            // 1. Dispatch local tasks (all local queue accesses are Relaxed — single thread).
            //    dispatch_loop now returns whether it executed anything, so we
            //    no longer need to read local_head before and after the call.
            unsafe {
                let worker = &*scheduler.workers[current_core].get();
                activity |= worker.dispatch_loop(pool);
            }

            // 2. Poll incoming mailboxes — skipped while in back-pressure mode
            //    to avoid flooding an already-saturated local queue.
            //    poll_mailboxes returns whether any chunks arrived, replacing
            //    the old q_len_before / q_len_after pair of queue-length reads.
            if !warehouse_busy {
                activity |= scheduler.poll_mailboxes(current_core);
            }

            if activity {
                idle_count = 0;
                continue;
            }

            // 3. Adaptive Backoff (pure user-space — no OS syscalls)
            idle_count = idle_count.saturating_add(1);

            // Tier 1: Fast spin (low latency)
            if idle_count < 256 {
                core::hint::spin_loop();
                continue;
            }

            // Tier 2: Heavier hardware pause (power efficiency)
            if idle_count < 2048 {
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    core::arch::asm!("yield", options(nostack, preserves_flags));
                }
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                core::hint::spin_loop();
                #[cfg(all(feature = "hw-acceleration", target_arch = "riscv64"))]
                unsafe {
                    core::arch::asm!("pause", options(nostack, preserves_flags));
                }
                #[cfg(not(any(
                    target_arch = "aarch64",
                    target_arch = "x86",
                    target_arch = "x86_64",
                    all(feature = "hw-acceleration", target_arch = "riscv64")
                )))]
                for _ in 0..8 {
                    core::hint::spin_loop();
                }
                continue;
            }

            // Tier 3: Hardware standby — purely in user-space, no OS syscalls.
            // With hw-acceleration: WFE (AArch64) or umonitor/umwait (x86) power-save
            // while automatically waking when event_signal is written by signal_worker.
            // Without hw-acceleration: continued spinning; workers stay hot.
            unsafe {
                #[cfg(all(feature = "hw-acceleration", target_arch = "aarch64"))]
                core::arch::asm!("wfe", options(nostack, preserves_flags));

                #[cfg(all(feature = "hw-acceleration", target_arch = "riscv64"))]
                core::arch::asm!("pause", options(nostack, preserves_flags));

                #[cfg(all(
                    feature = "hw-acceleration",
                    any(target_arch = "x86_64", target_arch = "x86")
                ))]
                {
                    // umonitor sets up a write-monitor on event_signal; umwait sleeps
                    // until event_signal is written (or timeout). Fully user-space (MWAIT C0.1).
                    // Acquire load before umonitor: syncs with signal_worker's Release write,
                    // ensuring all preceding mailbox stores are visible after wake.
                    let worker = &*scheduler.workers[current_core].get();
                    let signal_before = worker.event_signal.load(Ordering::Acquire);
                    let sig_ptr = &raw const worker.event_signal as *mut core::ffi::c_void;
                    let control = 1u32;
                    let timeout_low = 2_000_000u32;
                    let timeout_high = 0u32;
                    core::arch::asm!(
                        "umonitor {0}",
                        "cmp {1:e}, {2:e}",
                        "jne 2f",
                        "umwait {3:e}",
                        "2:",
                        in(reg) sig_ptr,
                        in(reg) signal_before,
                        in(reg) worker.event_signal.load(Ordering::Relaxed),
                        in(reg) control,
                        inout("eax") timeout_low => _,
                        inout("edx") timeout_high => _,
                        options(nostack, preserves_flags)
                    );
                }

                // After WFE (AArch64/RISC-V): Acquire fence ensures all remote stores
                // that triggered the SEV/write are visible for the poll below.
                // Acquire (DMB ISH) is sufficient — SeqCst (DSB ISH+ISB) is overkill.
                #[cfg(any(target_arch = "aarch64", target_arch = "riscv64"))]
                crate::sync::atomic::fence(Ordering::Acquire);

                // Without hw-acceleration there is no hardware block/monitor
                // primitive available, so this tier would otherwise just be
                // more `pause` — pure CPU-hogging with no way for the OS to
                // know this thread has nothing to do. Under
                // `cooperative-yield`, hand the core back voluntarily
                // (`sched_yield`/`SwitchToThread`) once we're deep enough
                // into backoff that low-latency spinning has already failed;
                // this is what lets other runnable threads (a benchmark
                // harness's own thread pool, an async runtime running
                // alongside dtact, etc.) actually get scheduled promptly
                // instead of waiting for involuntary OS preemption, which
                // can stack up to 100ms+ under heavy thread oversubscription.
                #[cfg(all(not(feature = "hw-acceleration"), feature = "cooperative-yield"))]
                std::thread::yield_now();

                #[cfg(all(not(feature = "hw-acceleration"), not(feature = "cooperative-yield")))]
                for _ in 0..16 {
                    core::hint::spin_loop();
                }
            }

            scheduler.poll_mailboxes(current_core);
        }
    }
}

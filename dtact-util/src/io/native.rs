use super::{Future, Pin};
use std::cell::RefCell;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::sync::OnceLock;
use std::sync::atomic::{
    AtomicBool, AtomicI32, AtomicPtr, AtomicU32, AtomicUsize, Ordering, fence,
};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

// Latency-breakdown tracing (DTACT_IO_TRACE=1) — shared with the
// Windows backend, see `crate::io::trace`'s module doc.
use crate::io::trace::io_trace;
#[allow(unused_imports)]
use crate::io::trace::trace_now_us;

static WORKER_ROUND_ROBIN: AtomicUsize = AtomicUsize::new(0);

// =========================================================================
// 1-2. TreiberStack / BufferPool — moved to `crate::lockfree`
// =========================================================================
// Both were previously private copies living in this module; they're now
// shared with `fs`'s native backends (see `fs::pool`), which adopted the
// exact same "preallocated arena + TreiberStack free-list" shape instead
// of allocating a fresh `Box`/`Arc` per filesystem op. This module keeps
// its own `chunk_owners` array (below) layered on top for the
// thread-local slab caching this specific reactor wants — that part is
// genuinely io-specific and stays here rather than in the shared type.
use crate::lockfree::{BufferPool, SpscQueue, TreiberStack};

// =========================================================================
// 3. THREAD-LOCAL SLAB ALLOCATOR & RETURN PATH
// =========================================================================
#[repr(align(64))]
struct LocalAllocator {
    thread_idx: usize,
    local_chunks: Vec<u32>,
}

thread_local! {
    static LOCAL_ALLOCATOR: RefCell<Option<LocalAllocator>> = const { RefCell::new(None) };
    static THREAD_ID: usize = {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    };
}

#[doc(hidden)]
#[must_use]
pub fn get_local_thread_id() -> usize {
    THREAD_ID.with(|id| *id)
}

static THREAD_RETURNED_STACKS: OnceLock<Box<[TreiberStack]>> = OnceLock::new();
static GLOBAL_BUFFER_POOL: OnceLock<BufferPool> = OnceLock::new();
/// Which thread's local cache last owned a given chunk, `u32::MAX` if
/// none/unknown — sized and populated alongside `GLOBAL_BUFFER_POOL` in
/// `init_runtime`. Kept as a separate array (rather than folded into
/// the shared `crate::lockfree::BufferPool`) because this thread-local
/// return-path optimization is specific to this reactor's per-worker
/// slab caching, not something `fs`'s simpler global-free-stack usage
/// of the same `BufferPool` type needs.
static CHUNK_OWNERS: OnceLock<Box<[AtomicU32]>> = OnceLock::new();

fn get_or_init_local_allocator() -> Option<usize> {
    LOCAL_ALLOCATOR.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if borrow.is_none() {
            let idx = get_local_thread_id();
            if idx < 512 {
                *borrow = Some(LocalAllocator {
                    thread_idx: idx,
                    local_chunks: Vec::new(),
                });
            }
        }
        borrow.as_ref().map(|alloc| alloc.thread_idx)
    })
}

#[doc(hidden)]
pub fn allocate_buffer() -> Option<u32> {
    let t_idx_opt = get_or_init_local_allocator();
    if let Some(t_idx) = t_idx_opt {
        LOCAL_ALLOCATOR.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let alloc = borrow.as_mut().unwrap();

            // 1. Try local cache
            if let Some(idx) = alloc.local_chunks.pop() {
                return Some(idx);
            }
            // 2. Try thread-specific returned stack
            if let Some(stacks) = THREAD_RETURNED_STACKS.get()
                && let Some(stack) = stacks.get(t_idx)
            {
                while let Some(idx) = stack.pop() {
                    alloc.local_chunks.push(idx);
                }
                if let Some(idx) = alloc.local_chunks.pop() {
                    return Some(idx);
                }
            }
            // 3. Fallback to global pool
            if let Some(pool) = GLOBAL_BUFFER_POOL.get()
                && let Some(idx) = pool.acquire()
            {
                if let Some(owners) = CHUNK_OWNERS.get() {
                    owners[idx as usize].store(t_idx as u32, Ordering::Release);
                }
                return Some(idx);
            }
            None
        })
    } else if let Some(pool) = GLOBAL_BUFFER_POOL.get() {
        if let Some(idx) = pool.acquire() {
            if let Some(owners) = CHUNK_OWNERS.get() {
                owners[idx as usize].store(u32::MAX, Ordering::Release);
            }
            return Some(idx);
        }
        None
    } else {
        None
    }
}

#[doc(hidden)]
pub fn free_buffer(idx: u32) {
    if let Some(pool) = GLOBAL_BUFFER_POOL.get() {
        let owner = CHUNK_OWNERS.get().map_or(u32::MAX, |owners| {
            owners[idx as usize].load(Ordering::Acquire)
        });
        if owner == u32::MAX {
            pool.release(idx);
            return;
        }

        let current_thread_idx = get_or_init_local_allocator();
        if Some(owner as usize) == current_thread_idx {
            LOCAL_ALLOCATOR.with(|cell| {
                if let Some(alloc) = cell.borrow_mut().as_mut() {
                    alloc.local_chunks.push(idx);
                }
            });
        } else if let Some(stacks) = THREAD_RETURNED_STACKS.get() {
            if let Some(stack) = stacks.get(owner as usize) {
                stack.push(idx);
            } else {
                pool.release(idx);
            }
        } else {
            pool.release(idx);
        }
    }
}

#[doc(hidden)]
#[repr(align(64))]
pub struct BufferSlice {
    pub buf_idx: u32,
    pub read_pos: usize,
    pub write_pos: usize,
}

impl BufferSlice {
    #[must_use]
    pub const fn new(buf_idx: u32, len: usize) -> Self {
        Self {
            buf_idx,
            read_pos: 0,
            write_pos: len,
        }
    }

    /// Raw pointer to this slice's backing chunk in the global buffer pool.
    ///
    /// # Panics
    ///
    /// Panics if called before [`init_runtime`]/[`init`] has initialized
    /// the global buffer pool.
    #[inline]
    pub fn data(&self) -> *mut u8 {
        GLOBAL_BUFFER_POOL.get().unwrap().get_ptr(self.buf_idx)
    }

    #[inline]
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.write_pos.saturating_sub(self.read_pos)
    }
}

impl Drop for BufferSlice {
    fn drop(&mut self) {
        free_buffer(self.buf_idx);
    }
}

// =========================================================================
// 4. CACHE-ALIGNED LOCK-FREE SPSC RINGBUFFER — moved to `crate::lockfree`
// =========================================================================
// Was a private copy here; now shared with `stream`'s native duplex-pipe
// backend via `crate::lockfree::SpscQueue` (imported above alongside
// `BufferPool`/`TreiberStack`).

// =========================================================================
// 5. IO ENGINE WORKERS AND EVENTS DEFINITIONS
// =========================================================================
/// Which async operation a [`DtactIoFuture`] represents — mirrors the
/// Windows backend's `OpCode` of the same name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpCode {
    /// A socket read.
    Read,
    /// A socket write.
    Write,
    /// Accept a new connection on a listening socket.
    Accept,
    /// Connect to a remote address.
    Connect,
    /// Connectionless UDP send to an explicit peer. Carries a caller-owned
    /// `msghdr` (see `DtactUdpSocket::send_to`); `io_uring` uses `SendMsg`,
    /// the mio fallback uses `sendmsg(2)`.
    SendTo,
    /// Connectionless UDP receive recording the peer address. Carries a
    /// caller-owned `msghdr`; `io_uring` uses `RecvMsg`, the mio fallback uses
    /// `recvmsg(2)`.
    RecvFrom,
}

/// A single io-worker request, submitted across an [`SpscQueue`].
///
/// Sent from a fiber's poll to the worker thread that owns the underlying
/// reactor (`io_uring` ring or mio `Poll`). Not constructed directly by
/// callers — built internally from a [`DtactIoFuture`]'s fields on first
/// poll.
#[derive(Clone, Copy)]
pub enum IoRequest {
    /// Read into `buf_ptr[..len]` at `offset` (or the current file
    /// position for sockets, where `offset` is ignored).
    Read {
        /// The raw fd to read from.
        fd: u32,
        /// `io_uring` direct/fixed-file index, or `u32::MAX` if `fd` isn't
        /// registered as one.
        direct_fd_idx: u32,
        /// Destination buffer.
        buf_ptr: *mut u8,
        /// Length of the buffer at `buf_ptr`.
        len: usize,
        /// Positional read offset (ignored for plain socket reads).
        offset: i64,
        /// This op's slot in the owning worker's op-slot table.
        slot_idx: usize,
    },
    /// Write `buf_ptr[..len]` at `offset`.
    Write {
        /// The raw fd to write to.
        fd: u32,
        /// `io_uring` direct/fixed-file index, or `u32::MAX` if `fd` isn't
        /// registered as one.
        direct_fd_idx: u32,
        /// Source buffer.
        buf_ptr: *const u8,
        /// Length of the buffer at `buf_ptr`.
        len: usize,
        /// Positional write offset (ignored for plain socket writes).
        offset: i64,
        /// This op's slot in the owning worker's op-slot table.
        slot_idx: usize,
    },
    /// Accept a new connection on listening socket `fd`.
    Accept {
        /// The listening socket's raw fd.
        fd: u32,
        /// `io_uring` direct/fixed-file index, or `u32::MAX` if `fd` isn't
        /// registered as one.
        direct_fd_idx: u32,
        /// This op's slot in the owning worker's op-slot table.
        slot_idx: usize,
    },
    /// Connect socket `fd` to `addr`.
    Connect {
        /// The socket's raw fd.
        fd: u32,
        /// `io_uring` direct/fixed-file index, or `u32::MAX` if `fd` isn't
        /// registered as one.
        direct_fd_idx: u32,
        /// The remote address to connect to.
        addr: libc::sockaddr_storage,
        /// Byte length of the valid prefix of `addr`.
        addr_len: libc::socklen_t,
        /// This op's slot in the owning worker's op-slot table.
        slot_idx: usize,
    },
    /// Connectionless UDP send to an explicit peer.
    SendTo {
        /// The UDP socket's raw fd.
        fd: u32,
        /// `io_uring` direct/fixed-file index, or `u32::MAX` if `fd` isn't
        /// registered as one.
        direct_fd_idx: u32,
        /// Caller-owned `msghdr` (see `DtactUdpSocket::send_to`), valid until
        /// the op completes.
        msg_ptr: *mut libc::msghdr,
        /// This op's slot in the owning worker's op-slot table.
        slot_idx: usize,
    },
    /// Connectionless UDP receive, recording the peer address.
    RecvFrom {
        /// The UDP socket's raw fd.
        fd: u32,
        /// `io_uring` direct/fixed-file index, or `u32::MAX` if `fd` isn't
        /// registered as one.
        direct_fd_idx: u32,
        /// Caller-owned `msghdr` whose `msg_name` the kernel fills with the
        /// sender's address; valid until the op completes.
        msg_ptr: *mut libc::msghdr,
        /// This op's slot in the owning worker's op-slot table.
        slot_idx: usize,
    },
    /// Register `fd` as an `io_uring` direct/fixed file, returning its
    /// direct-fd index as the op's result.
    RegisterFile {
        /// The raw fd to register.
        fd: RawFd,
        /// This op's slot in the owning worker's op-slot table.
        slot_idx: usize,
    },
    /// Release a previously-registered direct/fixed file.
    UnregisterFile {
        /// The direct/fixed-file index to release.
        direct_fd_idx: u32,
        /// This op's slot in the owning worker's op-slot table.
        slot_idx: usize,
    },
}

/// Lock-free waker slot.
///
/// `waker` is written by the fiber (before the SPSC push) and read+cleared
/// by the io-worker (after the SPSC pop, under the Acquire that observes the
/// Release from the SPSC push).  Since only one fiber owns a slot at a time
/// and the io-worker reads only after the ordering guarantee, there is no
/// data race — no Mutex needed.
// `#[repr(align(64))]` on the slot itself (not just the backing array) is
// what actually prevents false sharing: with a bare struct, adjacent slots
// in `Box<[WakerSlot]>` pack tightly and two io-worker threads completing
// unrelated ops in neighbouring slots end up bouncing the same cache line.
#[repr(align(64))]
struct WakerSlot {
    /// Stores the raw `data` pointer of a fiber `Waker` (`*const FiberContext`).
    waker_data: AtomicPtr<()>,
    /// Stores the raw `vtable` pointer of a fiber `Waker` (`*const RawWakerVTable`).
    /// Combined, these allow zero-cost reconstruction of the `RawWaker` without clone/drop overhead.
    waker_vtable: AtomicPtr<RawWakerVTable>,
    waker_lock: AtomicBool,
    result: AtomicI32,
    completed: AtomicBool,
    dropped: AtomicBool,
    /// fd this op was issued against, recorded so the owning worker thread
    /// can cancel/clean up the op purely from `slot_idx` (see `cancel_queue`)
    /// without the dropping thread needing to touch backend state itself.
    origin_fd: AtomicU32,
}

#[repr(align(64))]
struct WaitSlot {
    waker_data: AtomicPtr<()>,
    waker_vtable: AtomicPtr<RawWakerVTable>,
}

impl WakerSlot {
    #[inline(always)]
    fn lock_waker(&self) {
        while self
            .waker_lock
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
    }

    #[inline(always)]
    fn unlock_waker(&self) {
        self.waker_lock.store(false, Ordering::Release);
    }
}

#[inline(always)]
fn wake_next_waiting_fiber(state: &WorkerState) {
    if let Some(wait_idx) = state.waiting_queue.pop() {
        let wait_slot = &state.wait_slots[wait_idx as usize];
        let data = wait_slot
            .waker_data
            .swap(std::ptr::null_mut(), Ordering::Relaxed);
        let vtable = wait_slot
            .waker_vtable
            .swap(std::ptr::null_mut(), Ordering::Relaxed);
        state.free_wait_slots.push(wait_idx);

        if !data.is_null() && !vtable.is_null() {
            let raw = RawWaker::new(data.cast_const(), unsafe { &*vtable });
            let w = unsafe { Waker::from_raw(raw) };
            w.wake();
        }
    }
}

/// Per-worker reactor state.
///
/// Holds the `io_uring` ring (Linux) or mio `Poll` (other Unix), its
/// op-slot table, and the lock-free queues fibers use to submit/cancel
/// requests. One of these exists per io-worker thread — see
/// [`init_runtime`].
pub struct WorkerState {
    #[cfg(target_os = "linux")]
    ring: std::cell::UnsafeCell<io_uring::IoUring>,
    #[cfg(not(target_os = "linux"))]
    poll: std::cell::UnsafeCell<mio::Poll>,

    queues: Box<[SpscQueue<IoRequest>]>,
    slots: Box<[WakerSlot]>,
    free_slots: TreiberStack,

    wait_slots: Box<[WaitSlot]>,
    free_wait_slots: TreiberStack,
    waiting_queue: TreiberStack,
    is_sleeping: AtomicBool,

    /// Slot indices whose owning `DtactIoFuture` was dropped before the op
    /// completed. Multiple arbitrary threads may push here (whichever
    /// thread drops the future), so this must be MP-safe — `TreiberStack`
    /// already is (CAS-based), unlike the per-thread SPSC `queues`, which
    /// must never receive a push from more than one producer thread.
    /// Only the owning io-worker thread pops from it.
    cancel_queue: TreiberStack,

    #[cfg(target_os = "linux")]
    wake_eventfd: RawFd,
    #[cfg(target_os = "linux")]
    sqpoll_enabled: bool,
    #[cfg(not(target_os = "linux"))]
    waker: std::sync::Arc<mio::Waker>,

    direct_fd_free: TreiberStack,
}

// SAFETY: `queues: Box<[SpscQueue<IoRequest>]>` holds `IoRequest`s
// containing raw pointers (`buf_ptr`/`msg_ptr`/etc.), which aren't
// automatically `Send`. Those pointers are always ownership-transferred,
// never shared: a fiber pushes an `IoRequest` onto exactly one
// `SpscQueue`, and only this `WorkerState`'s own single owning io-worker
// thread ever pops from it (see the `cancel_queue` doc above for the one
// deliberate MP exception, which doesn't carry raw buffer pointers). The
// pointed-to buffers themselves are guaranteed to outlive the op by the
// same contract every `DtactIoFuture` (also `unsafe impl Send`) already
// relies on.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl Send for WorkerState {}
unsafe impl Sync for WorkerState {}

/// Runtime config consulted after startup. Only the two fields read on the
/// steady-state path live here: `workers` (op fan-out across io-worker
/// threads) and `pin_cpus` (per-worker core affinity). The
/// `buffer_pool_size`/`chunk_size`/`ring_depth` knobs from
/// [`init_runtime`] are consumed eagerly during startup to size the buffer
/// pool, per-op slot tables, and ring depth — they are never read back
/// afterwards, so they are deliberately not retained here (retaining them
/// was dead state that tripped `dead_code`).
struct GlobalConfig {
    workers: usize,
    pin_cpus: Vec<usize>,
}

static GLOBAL_CONFIG: OnceLock<GlobalConfig> = OnceLock::new();
static WORKERS: OnceLock<Box<[WorkerState]>> = OnceLock::new();
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

#[cfg(target_os = "linux")]
fn pin_thread_to_cpu(cpu_id: usize) -> Result<(), &'static str> {
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu_id, &mut cpuset);
        let thread = libc::pthread_self();
        let res = libc::pthread_setaffinity_np(
            thread,
            std::mem::size_of::<libc::cpu_set_t>(),
            &raw const cpuset,
        );
        if res == 0 {
            Ok(())
        } else {
            Err("pthread_setaffinity_np failed")
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_thread_to_cpu(_cpu_id: usize) -> Result<(), &'static str> {
    Ok(())
}

/// Registering `n` direct/fixed files with `io_uring` requires the
/// process's `RLIMIT_NOFILE` soft limit to cover `n` (on top of whatever
/// fds are already open) — unconditionally asking for 4096 slots panics
/// with EMFILE on any environment with a lower default soft limit (1024
/// is a common distro default). Raise our own soft limit to the hard
/// limit first (always permitted for a non-privileged process to do to
/// itself), then size the direct-fd table to what's actually available,
/// capped at the desired maximum and leaving headroom for real sockets.
#[cfg(target_os = "linux")]
fn pick_direct_fd_count(desired_max: usize) -> usize {
    unsafe {
        let mut lim: libc::rlimit = std::mem::zeroed();
        if libc::getrlimit(libc::RLIMIT_NOFILE, &raw mut lim) == 0 {
            if lim.rlim_cur < lim.rlim_max {
                let raised = libc::rlimit {
                    rlim_cur: lim.rlim_max,
                    rlim_max: lim.rlim_max,
                };
                let _ = libc::setrlimit(libc::RLIMIT_NOFILE, &raw const raised);
                let _ = libc::getrlimit(libc::RLIMIT_NOFILE, &raw mut lim);
            }
            let headroom = 256usize;
            let available = (lim.rlim_cur as usize).saturating_sub(headroom);
            return available.clamp(64, desired_max);
        }
    }
    // getrlimit itself failed — fall back to a conservative count rather
    // than the original unconditional 4096.
    64
}

// =========================================================================
// 6. RUNTIME INITIALIZATION
// =========================================================================
/// Start the native io reactor.
///
/// Spins up `workers` io-worker threads (`io_uring` on Linux, kqueue/mio
/// elsewhere), a `buffer_pool_size`-chunk arena sliced into `chunk_size`-
/// byte buffers, a `ring_depth`-deep in-flight-op slot table per worker,
/// and optional `pin_cpus` core affinity (index `i` pins worker `i`; a
/// shorter/empty slice leaves the rest unpinned).
///
/// Idempotent: only the first call takes effect, later calls are no-ops
/// (mirrors [`crate::fs::init_fs`]/[`crate::process::init_process`]).
///
/// # Panics
///
/// Panics if the OS refuses to create an `eventfd` (Linux) for the
/// worker-wake mechanism, or if a worker thread fails to spawn — both are
/// treated as fatal startup failures.
///
/// Argument order — `(workers, ring_depth, buffer_pool_size, chunk_size,
/// pin_cpus)` — matches every other native backend's five-knob init
/// function in this crate; see the crate-level doc comment in `crate` for
/// the full init-API shape this is part of.
// One-time startup sequencing (config → buffer pool → per-worker ring/
// slot-table/queue allocation → thread spawn) reads more clearly as one
// linear function than split across several that would each need most of
// the same local state threaded through as parameters; this is also
// Linux-only io_uring setup code that's expensive to verify a refactor of
// without a Linux box in the loop, so left as-is rather than restructured
// speculatively.
#[allow(clippy::too_many_lines)]
pub fn init_runtime(
    workers: usize,
    ring_depth: u32,
    buffer_pool_size: usize,
    chunk_size: usize,
    pin_cpus: &[usize],
) {
    let config = GlobalConfig {
        workers,
        pin_cpus: pin_cpus.to_vec(),
    };
    if GLOBAL_CONFIG.set(config).is_err() {
        return;
    }

    let pool = BufferPool::new(buffer_pool_size, chunk_size);
    let _ = GLOBAL_BUFFER_POOL.set(pool);
    let owners: Vec<AtomicU32> = (0..buffer_pool_size)
        .map(|_| AtomicU32::new(u32::MAX))
        .collect();
    let _ = CHUNK_OWNERS.set(owners.into_boxed_slice());

    let mut returned_stacks = Vec::with_capacity(512);
    for _ in 0..512 {
        returned_stacks.push(TreiberStack::new(0));
    }
    let _ = THREAD_RETURNED_STACKS.set(returned_stacks.into_boxed_slice());

    let mut worker_states = Vec::with_capacity(workers);
    for _worker_idx in 0..workers {
        let mut queues = Vec::with_capacity(512);
        for _ in 0..512 {
            queues.push(SpscQueue::new(256));
        }
        let queues = queues.into_boxed_slice();

        let mut slots = Vec::with_capacity(ring_depth as usize);
        for _ in 0..ring_depth {
            slots.push(WakerSlot {
                waker_data: AtomicPtr::new(std::ptr::null_mut()),
                waker_vtable: AtomicPtr::new(std::ptr::null_mut()),
                waker_lock: AtomicBool::new(false),
                result: AtomicI32::new(0),
                completed: AtomicBool::new(false),
                dropped: AtomicBool::new(false),
                origin_fd: AtomicU32::new(u32::MAX),
            });
        }
        let slots = slots.into_boxed_slice();
        let free_slots = TreiberStack::new(ring_depth as usize);
        for i in 0..ring_depth {
            free_slots.push(i);
        }
        let cancel_queue = TreiberStack::new(ring_depth as usize);

        let wait_slots_depth = 65536;
        let mut wait_slots = Vec::with_capacity(wait_slots_depth);
        for _ in 0..wait_slots_depth {
            wait_slots.push(WaitSlot {
                waker_data: AtomicPtr::new(std::ptr::null_mut()),
                waker_vtable: AtomicPtr::new(std::ptr::null_mut()),
            });
        }
        let wait_slots = wait_slots.into_boxed_slice();
        let free_wait_slots = TreiberStack::new(wait_slots_depth);
        for i in 0..wait_slots_depth {
            free_wait_slots.push(i as u32);
        }
        let waiting_queue = TreiberStack::new(wait_slots_depth);
        let is_sleeping = AtomicBool::new(false);

        #[cfg(target_os = "linux")]
        let direct_fd_count = pick_direct_fd_count(4096);
        #[cfg(not(target_os = "linux"))]
        let direct_fd_count = 4096usize;

        let direct_fd_free = TreiberStack::new(direct_fd_count);
        for i in 0..direct_fd_count as u32 {
            direct_fd_free.push(i);
        }

        #[cfg(target_os = "linux")]
        {
            let wake_eventfd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
            assert!(wake_eventfd >= 0, "Failed to create eventfd");

            let (ring, sqpoll_enabled) = io_uring::IoUring::builder()
                .setup_sqpoll(2000)
                .build(ring_depth)
                .map_or_else(
                    |_| {
                        (
                            io_uring::IoUring::new(ring_depth)
                                .expect("Failed to initialize io_uring fallback"),
                            false,
                        )
                    },
                    |r| (r, true),
                );

            let initial_fds = vec![-1; direct_fd_count];
            ring.submitter()
                .register_files(&initial_fds)
                .expect("Failed to register direct FDs");

            worker_states.push(WorkerState {
                ring: std::cell::UnsafeCell::new(ring),
                queues,
                slots,
                free_slots,
                wait_slots,
                free_wait_slots,
                waiting_queue,
                is_sleeping,
                cancel_queue,
                wake_eventfd,
                sqpoll_enabled,
                direct_fd_free,
            });
        }

        #[cfg(not(target_os = "linux"))]
        {
            let poll = mio::Poll::new().expect("Failed to initialize mio Poll");
            let waker = std::sync::Arc::new(
                mio::Waker::new(poll.registry(), mio::Token(0))
                    .expect("Failed to create mio waker"),
            );

            worker_states.push(WorkerState {
                poll: std::cell::UnsafeCell::new(poll),
                queues,
                slots,
                free_slots,
                wait_slots,
                free_wait_slots,
                waiting_queue,
                is_sleeping,
                cancel_queue,
                waker,
                direct_fd_free,
            });
        }
    }

    let worker_states = worker_states.into_boxed_slice();
    let _ = WORKERS.set(worker_states);

    for worker_idx in 0..workers {
        std::thread::Builder::new()
            .name(format!("dtact-io-worker-{worker_idx}"))
            .spawn(move || {
                LOCAL_ALLOCATOR.with(|cell| {
                    *cell.borrow_mut() = Some(LocalAllocator {
                        thread_idx: worker_idx,
                        local_chunks: Vec::new(),
                    });
                });

                let state = &WORKERS.get().unwrap()[worker_idx];

                #[cfg(target_os = "linux")]
                run_linux_worker_loop(worker_idx, state);

                #[cfg(not(target_os = "linux"))]
                run_mio_worker_loop(worker_idx, state);
            })
            .expect("Failed to spawn dtact-io worker thread");
    }
}

/// Shorthand initialiser: `workers` io-worker threads with sane defaults.
///
/// 64 MiB buffer pool split into 4 KiB chunks, no CPU pinning, a 1024-deep
/// per-worker op-slot ring. Equivalent to
/// `init_runtime(workers, 1024, 65536, 4096, &[])`. Matches
/// [`crate::fs::init`]/[`crate::process::init`]'s shape.
pub fn init(workers: usize) {
    init_runtime(workers, 1024, 65536, 4096, &[]);
}

/// Signal every io-worker thread to stop and unblock its reactor wait
/// (`eventfd` on Linux, the `mio::Waker` elsewhere) so it can observe the
/// shutdown flag and exit. Does not join the worker threads.
pub fn shutdown_runtime() {
    SHUTDOWN.store(true, Ordering::Release);
    if let Some(workers) = WORKERS.get() {
        for state in workers {
            #[cfg(target_os = "linux")]
            let _ = unsafe {
                libc::write(
                    state.wake_eventfd,
                    std::ptr::from_ref::<u64>(&1u64).cast::<libc::c_void>(),
                    8,
                )
            };
            #[cfg(not(target_os = "linux"))]
            state.waker.wake();
        }
    }
}

// =========================================================================
// 7. LINUX SYSTEM CALL DRIVER (io_uring)
// =========================================================================
// The per-completion dispatch (decode result → route to slot vs. cancel
// vs. eventfd wake → wake the right waiter) is inherently one state
// machine over one `for cqe in cq` loop; splitting it would just move the
// same branches behind indirection without reducing real complexity, and
// this is Linux-only io_uring code that's expensive to verify a refactor
// of without a Linux box in the loop.
#[allow(clippy::too_many_lines)]
#[cfg(target_os = "linux")]
fn run_linux_worker_loop(worker_idx: usize, state: &WorkerState) {
    if let Some(config) = GLOBAL_CONFIG.get()
        && let Some(&cpu_id) = config.pin_cpus.get(worker_idx)
    {
        let _ = pin_thread_to_cpu(cpu_id);
    }

    let ring = unsafe { &mut *state.ring.get() };
    let mut eventfd_buf = 0u64;
    let mut eventfd_submitted = false;
    // Consecutive idle iterations, only tracked/consulted under the `spin`
    // feature — escalates how long `adaptive_idle_spin` is willing to spin
    // before falling back to a blocking wait, and resets to 0 the moment
    // real work shows up.
    #[cfg(feature = "spin")]
    let mut idle_streak: u32 = 0;

    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            break;
        }

        if !eventfd_submitted {
            let sqe = io_uring::opcode::Read::new(
                io_uring::types::Fd(state.wake_eventfd),
                (&raw mut eventfd_buf).cast::<u8>(),
                8,
            )
            .build()
            .user_data(u64::MAX);

            unsafe {
                if ring.submission().push(&sqe).is_ok() {
                    eventfd_submitted = true;
                }
            }
        }

        let mut pushed_sqe = false;
        for q in &state.queues {
            while let Some(req) = q.pop() {
                pushed_sqe = true;
                let _ = submit_linux_request(state, &req);
            }
        }

        while let Some(slot_idx) = state.cancel_queue.pop() {
            pushed_sqe = true;
            let sqe = io_uring::opcode::AsyncCancel::new(u64::from(slot_idx))
                .build()
                .user_data(u64::MAX - 1);
            unsafe {
                let _ = push_sqe(ring, &sqe);
            }
        }

        // `push()` only writes into the *local* SQE array and tail — it is
        // not visible to the SQPOLL kernel thread (which polls the shared
        // mmap'd ring) until the tail is published. `submit()` publishes
        // it as a side effect of the io_uring_enter syscall, but that
        // syscall is exactly what SQPOLL exists to avoid. So: always
        // `sync()` (cheap, no syscall — just a store-release into shared
        // memory) so an actively-spinning kernel thread sees new entries
        // immediately, and only pay for `io_uring_enter` (via `submit()`)
        // when the kernel thread has actually gone to sleep and asked to
        // be woken (`need_wakeup`), or when SQPOLL isn't in use at all.
        let any_pending = state.queues.iter().any(|q| !q.is_empty());

        // When we just submitted new work and there is nothing else
        // queued behind it, we already know the very next thing this
        // loop will do is block waiting for that work to complete (the
        // `!pushed_sqe && !has_completions` branch below, one iteration
        // later) — that used to cost a *second* `io_uring_enter` syscall
        // (`submit()` now, `submit_and_wait(1)` next loop). Fold both
        // into a single blocking enter here instead.
        let mut folded_wait = false;
        if pushed_sqe || eventfd_submitted {
            ring.submission().sync();
            if pushed_sqe && !any_pending {
                state.is_sleeping.store(true, Ordering::SeqCst);
                // Dekker-style re-check: a producer that pushed to a queue
                // and observed `is_sleeping == false` just before our store
                // above (a StoreLoad reorder is otherwise legal even on
                // x86-TSO) would skip the eventfd wakeup, leaving us to
                // block forever on a request nobody drained. Re-scan the
                // queues now that `is_sleeping` is published; if anything
                // landed, bail out of the blocking wait and let the top of
                // the loop drain it next iteration instead.
                fence(Ordering::SeqCst);
                let missed = state.queues.iter().any(|q| !q.is_empty());
                if missed {
                    state.is_sleeping.store(false, Ordering::SeqCst);
                    let sr = ring.submit();
                    io_trace!(
                        "[dtact-io] t={} loop submit(folded-missed) result={:?}",
                        trace_now_us(),
                        sr
                    );
                } else {
                    let sr = ring.submit_and_wait(1);
                    state.is_sleeping.store(false, Ordering::Release);
                    io_trace!(
                        "[dtact-io] t={} loop submit_and_wait(folded) result={:?}",
                        trace_now_us(),
                        sr
                    );
                }
                folded_wait = true;
            } else {
                let should_enter = if state.sqpoll_enabled {
                    ring.submission().need_wakeup()
                } else {
                    true
                };
                if should_enter {
                    let sr = ring.submit();
                    io_trace!(
                        "[dtact-io] t={} loop submit() result={:?}",
                        trace_now_us(),
                        sr
                    );
                }
            }
        }

        let mut has_completions = false;
        let mut cq = ring.completion();
        cq.sync();
        let cq_len = cq.len();
        if cq_len > 0 {
            io_trace!("[dtact-io] t={} loop cq_len={}", trace_now_us(), cq_len);
        }
        for cqe in cq {
            has_completions = true;
            let user_data = cqe.user_data();
            let res = cqe.result();

            if user_data == u64::MAX {
                eventfd_submitted = false;
            } else if user_data == u64::MAX - 1 {
                // Cancel event completion, do nothing
            } else {
                process_linux_completion(state, user_data as usize, res);
            }
        }

        #[cfg(feature = "spin")]
        if folded_wait || pushed_sqe || has_completions {
            idle_streak = 0;
        }

        if !folded_wait && !pushed_sqe && !has_completions {
            // A bounded busy-poll was tried here (spin on the completion
            // ring/request queues before committing to a blocking
            // `submit_and_wait`, on the theory that a dedicated io-worker
            // thread can safely spin without blocking any fiber). Measured
            // on real, non-idle hardware (background load from unrelated
            // processes competing for the same cores) it was a severe net
            // regression — UDP roundtrip latency went from ~20µs to
            // 1-18ms, apparently because a spinning thread doesn't get
            // scheduled contiguously under contention the way a blocking
            // `submit_and_wait` (which yields immediately) does. Reverted
            // by default; do not reintroduce unconditionally without
            // validating under real background load, not just an idle
            // benchmark box.
            //
            // Under the opt-in `spin` feature (off by default, see
            // `Cargo.toml`) we retry the same regression with two changes
            // meant to avoid it: the spin is *adaptive* (escalates only
            // after repeated idle iterations, so a lightly-loaded worker
            // never spins) and tightly *bounded* (a few thousand
            // `spin_loop` hints, roughly single-digit microseconds, not a
            // busy-wait until a timeout) — a short enough window that a
            // contended core still yields back promptly via the fallback
            // blocking `submit_and_wait` below.
            #[cfg(feature = "spin")]
            {
                if adaptive_idle_spin(state, idle_streak) {
                    continue;
                }
            }
            #[cfg(feature = "spin")]
            {
                idle_streak = idle_streak.saturating_add(1);
            }

            state.is_sleeping.store(true, Ordering::SeqCst);
            // Same Dekker-style re-check as the folded-wait path above:
            // `any_pending` was computed earlier in this iteration and may
            // be stale by now. Without this re-scan + fence, a producer's
            // push-then-check-is_sleeping (also StoreLoad-ordered) can
            // race with our store-then-block here and neither side sends
            // a wakeup — the classic lost-wakeup deadlock.
            fence(Ordering::SeqCst);
            let missed = state.queues.iter().any(|q| !q.is_empty());
            if !any_pending && !missed {
                let sr = ring.submit_and_wait(1);
                io_trace!(
                    "[dtact-io] t={} loop submit_and_wait(idle) result={:?}",
                    trace_now_us(),
                    sr
                );
            } else if missed {
                let sr = ring.submit();
                io_trace!(
                    "[dtact-io] t={} loop submit(idle-missed) result={:?}",
                    trace_now_us(),
                    sr
                );
            }
            state.is_sleeping.store(false, Ordering::Release);
        }
    }
}

/// Opt-in (`spin` feature) adaptive busy-poll tried right before an
/// io-worker would otherwise commit to a blocking `submit_and_wait`.
///
/// Deliberately conservative on both axes that made the earlier
/// unconditional version regress (see the call site's comment): it only
/// spins once the worker has already been idle for a few consecutive
/// iterations (`idle_streak`) — a worker that's finding real work every
/// loop never spins at all — and each spin attempt is capped at a few
/// thousand `spin_loop` hints (roughly single-digit microseconds), so a
/// core under contention gets back to yielding via the blocking wait
/// almost as quickly as it would without this feature.
///
/// Returns `true` if new work showed up in a submission queue during the
/// spin, in which case the caller should skip the blocking wait and loop
/// back around to drain it immediately.
#[cfg(all(target_os = "linux", feature = "spin"))]
fn adaptive_idle_spin(state: &WorkerState, idle_streak: u32) -> bool {
    // Don't spin at all until the worker has genuinely gone idle a few
    // times in a row — a worker that's busy every iteration should never
    // pay the spin cost.
    const ARM_AFTER: u32 = 2;
    if idle_streak < ARM_AFTER {
        return false;
    }

    // Escalate the spin budget with sustained idleness, capped low enough
    // that even the maximum budget is a few-microsecond affair, not a
    // real busy-wait.
    let budget = 256u32.saturating_mul(idle_streak.saturating_sub(ARM_AFTER) + 1);
    let budget = budget.min(4096);

    for _ in 0..budget {
        if state.queues.iter().any(|q| !q.is_empty()) {
            return true;
        }
        core::hint::spin_loop();
    }
    false
}

#[cfg(target_os = "linux")]
unsafe fn push_sqe(
    ring: &mut io_uring::IoUring,
    sqe: &io_uring::squeue::Entry,
) -> Result<(), &'static str> {
    loop {
        let res = unsafe { ring.submission().push(sqe) };
        if res == Ok(()) {
            return Ok(());
        }
        let _ = ring.submit();
        core::hint::spin_loop();
    }
}

// One match arm per `IoRequest` variant, each translating that op into an
// `io_uring` SQE — naturally as many lines as there are variants times a
// few lines of setup each; splitting per-variant into helper functions
// (as `Connect` already does via `submit_connect`) is reasonable future
// cleanup but not done wholesale here to keep this pass's diff focused on
// UDP support plus the lint/Send fixes, on Linux-only code that's
// expensive to verify a refactor of without a Linux box in the loop.
#[allow(clippy::too_many_lines)]
#[cfg(target_os = "linux")]
fn submit_linux_request(state: &WorkerState, req: &IoRequest) -> Result<(), &'static str> {
    let ring = unsafe { &mut *state.ring.get() };

    let sqe = match *req {
        IoRequest::Read {
            fd,
            direct_fd_idx,
            buf_ptr,
            len,
            offset,
            slot_idx,
        } => {
            let use_fixed = direct_fd_idx != u32::MAX;
            let target_fd = if use_fixed {
                direct_fd_idx as i32
            } else {
                fd as i32
            };
            let mut s =
                io_uring::opcode::Read::new(io_uring::types::Fd(target_fd), buf_ptr, len as u32)
                    .offset(offset as u64)
                    .build()
                    .user_data(slot_idx as u64);
            if use_fixed {
                s = s.flags(io_uring::squeue::Flags::FIXED_FILE);
            }
            s
        }
        IoRequest::Write {
            fd,
            direct_fd_idx,
            buf_ptr,
            len,
            offset,
            slot_idx,
        } => {
            let use_fixed = direct_fd_idx != u32::MAX;
            let target_fd = if use_fixed {
                direct_fd_idx as i32
            } else {
                fd as i32
            };
            let mut s =
                io_uring::opcode::Write::new(io_uring::types::Fd(target_fd), buf_ptr, len as u32)
                    .offset(offset as u64)
                    .build()
                    .user_data(slot_idx as u64);
            if use_fixed {
                s = s.flags(io_uring::squeue::Flags::FIXED_FILE);
            }
            s
        }
        IoRequest::Accept {
            fd,
            direct_fd_idx,
            slot_idx,
        } => {
            let use_fixed = direct_fd_idx != u32::MAX;
            let target_fd = if use_fixed {
                direct_fd_idx as i32
            } else {
                fd as i32
            };
            let mut s = io_uring::opcode::Accept::new(
                io_uring::types::Fd(target_fd),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
            .build()
            .user_data(slot_idx as u64);
            if use_fixed {
                s = s.flags(io_uring::squeue::Flags::FIXED_FILE);
            }
            s
        }
        IoRequest::Connect {
            fd,
            direct_fd_idx,
            addr,
            addr_len,
            slot_idx,
        } => {
            // `addr` lives inside the IoRequest enum on the io-worker's stack.
            // io_uring copies the sockaddr into the kernel during push_sqe /
            // io_uring_enter, so a stack pointer is safe for the duration of
            // submit_linux_request.  No Mutex required.
            let addr_ptr = (&raw const addr).cast::<libc::sockaddr>();

            let use_fixed = direct_fd_idx != u32::MAX;
            let target_fd = if use_fixed {
                direct_fd_idx as i32
            } else {
                fd as i32
            };
            let mut s =
                io_uring::opcode::Connect::new(io_uring::types::Fd(target_fd), addr_ptr, addr_len)
                    .build()
                    .user_data(slot_idx as u64);
            if use_fixed {
                s = s.flags(io_uring::squeue::Flags::FIXED_FILE);
            }
            s
        }
        IoRequest::SendTo {
            fd,
            direct_fd_idx,
            msg_ptr,
            slot_idx,
        } => {
            let use_fixed = direct_fd_idx != u32::MAX;
            let target_fd = if use_fixed {
                direct_fd_idx as i32
            } else {
                fd as i32
            };
            let mut s = io_uring::opcode::SendMsg::new(
                io_uring::types::Fd(target_fd),
                msg_ptr.cast_const(),
            )
            .build()
            .user_data(slot_idx as u64);
            if use_fixed {
                s = s.flags(io_uring::squeue::Flags::FIXED_FILE);
            }
            s
        }
        IoRequest::RecvFrom {
            fd,
            direct_fd_idx,
            msg_ptr,
            slot_idx,
        } => {
            let use_fixed = direct_fd_idx != u32::MAX;
            let target_fd = if use_fixed {
                direct_fd_idx as i32
            } else {
                fd as i32
            };
            let mut s = io_uring::opcode::RecvMsg::new(io_uring::types::Fd(target_fd), msg_ptr)
                .build()
                .user_data(slot_idx as u64);
            if use_fixed {
                s = s.flags(io_uring::squeue::Flags::FIXED_FILE);
            }
            s
        }
        IoRequest::RegisterFile { fd, slot_idx } => {
            if let Some(direct_idx) = state.direct_fd_free.pop() {
                let fds = [fd];
                let res = ring.submitter().register_files_update(direct_idx, &fds);
                let out_res = match res {
                    Ok(_) => direct_idx as i32,
                    Err(e) => -(e.raw_os_error().unwrap_or(libc::EINVAL)),
                };
                process_linux_completion(state, slot_idx, out_res);
            } else {
                process_linux_completion(state, slot_idx, -libc::ENFILE);
            }
            return Ok(());
        }
        IoRequest::UnregisterFile {
            direct_fd_idx,
            slot_idx,
        } => {
            let fds = [-1];
            let res = ring.submitter().register_files_update(direct_fd_idx, &fds);
            state.direct_fd_free.push(direct_fd_idx);
            let out_res = match res {
                Ok(_) => 0,
                Err(e) => -(e.raw_os_error().unwrap_or(libc::EINVAL)),
            };
            process_linux_completion(state, slot_idx, out_res);
            return Ok(());
        }
    };

    let user_data = sqe.get_user_data();
    let r = unsafe { push_sqe(ring, &sqe) };
    io_trace!(
        "[dtact-io] t={} slot={} submit_linux_request pushed_local ok={}",
        trace_now_us(),
        user_data,
        r.is_ok()
    );
    r
}

#[cfg(target_os = "linux")]
fn process_linux_completion(state: &WorkerState, slot_idx: usize, res: i32) {
    let slot = &state.slots[slot_idx];

    io_trace!(
        "[dtact-io] t={} slot={} res={} B_kernel_complete",
        trace_now_us(),
        slot_idx,
        res
    );

    slot.result.store(res, Ordering::Release);

    // Extract (and fully detach) the waker BEFORE publishing `completed`.
    // If `completed` were published first, a concurrently spin-polling
    // fiber (see `wait_pinned`'s adaptive spin) could observe it, free
    // this slot, and have a *brand new* op reuse the same slot index and
    // install a fresh waker into the very fields we're about to swap out
    // here — we'd then wake (or null out) the new op's waker instead of
    // the one this completion actually belongs to, permanently losing
    // the new op's wakeup. Waker extraction is a self-contained lock+swap,
    // so doing it first is safe regardless of publication order.
    slot.lock_waker();
    let data = slot
        .waker_data
        .swap(std::ptr::null_mut(), Ordering::Relaxed);
    let vtable = slot
        .waker_vtable
        .swap(std::ptr::null_mut(), Ordering::Relaxed);
    slot.unlock_waker();

    slot.completed.store(true, Ordering::Release);

    if slot.dropped.load(Ordering::Acquire) {
        state.free_slots.push(slot_idx as u32);
        wake_next_waiting_fiber(state);
    } else if !data.is_null() && !vtable.is_null() {
        let raw = RawWaker::new(data.cast_const(), unsafe { &*vtable });
        let w = unsafe { Waker::from_raw(raw) };
        w.wake();
    }
}

// =========================================================================
// 8. FALLBACK MULTIPLEXER (mio REACTOR) FOR OTHER PLATFORMS
// =========================================================================
#[cfg(not(target_os = "linux"))]
struct FdState {
    reader_waker: Option<Waker>,
    writer_waker: Option<Waker>,
    /// Which `WakerSlot` each waker came from, so `cancel_queue` draining
    /// can find and clear the right side without an O(n) scan.
    reader_slot: Option<usize>,
    writer_slot: Option<usize>,
    /// Interest last handed to `reregister` for this fd, so callers can
    /// skip the syscall entirely when the newly-computed interest is
    /// identical (e.g. a Write request arriving while a Read is already
    /// registered doesn't need to touch epoll/kqueue at all).
    registered_interest: Option<mio::Interest>,
}

#[cfg(not(target_os = "linux"))]
impl FdState {
    const fn new() -> Self {
        Self {
            reader_waker: None,
            writer_waker: None,
            reader_slot: None,
            writer_slot: None,
            registered_interest: None,
        }
    }
}

/// Grow `fd_states` on demand instead of preallocating a fixed-size table
/// that silently drops events for any fd beyond it.
#[cfg(not(target_os = "linux"))]
fn ensure_fd_state(fd_states: &mut Vec<FdState>, fd: usize) {
    if fd_states.len() <= fd {
        fd_states.resize_with(fd + 1, FdState::new);
    }
}

/// Install `fd_state`'s currently-desired waker for `fd` (reader or
/// writer side, matching `is_reader`) and reregister with the OS poller
/// only when the resulting interest set actually changed. Returns
/// `false` if `reregister` failed — in which case the just-installed
/// waker has already been woken immediately (rather than left parked
/// waiting for an event that, given the broken registration, may never
/// arrive) and cleared back out of `fd_state`.
#[cfg(not(target_os = "linux"))]
fn install_interest(
    state: &WorkerState,
    fd_state: &mut FdState,
    fd: u32,
    slot_idx: usize,
    is_reader: bool,
) -> bool {
    let slot = &state.slots[slot_idx];
    slot.lock_waker();
    let data = slot
        .waker_data
        .swap(std::ptr::null_mut(), Ordering::Relaxed);
    let vtable = slot
        .waker_vtable
        .swap(std::ptr::null_mut(), Ordering::Relaxed);
    slot.unlock_waker();

    let waker = if !data.is_null() && !vtable.is_null() {
        let raw = RawWaker::new(data as *const (), unsafe { &*vtable });
        Some(unsafe { Waker::from_raw(raw) })
    } else {
        None
    };

    if is_reader {
        fd_state.reader_waker = waker;
        fd_state.reader_slot = if fd_state.reader_waker.is_some() {
            Some(slot_idx)
        } else {
            None
        };
    } else {
        fd_state.writer_waker = waker;
        fd_state.writer_slot = if fd_state.writer_waker.is_some() {
            Some(slot_idx)
        } else {
            None
        };
    }

    let interest = get_mio_interest(fd_state);
    if fd_state.registered_interest == Some(interest) {
        return true;
    }

    let res = unsafe {
        let poll = &mut *state.poll.get();
        poll.registry().reregister(
            &mut mio::unix::SourceFd(&(fd as i32)),
            mio::Token(fd as usize),
            interest,
        )
    };

    match res {
        Ok(()) => {
            fd_state.registered_interest = Some(interest);
            true
        }
        Err(e) => {
            io_trace!(
                "[dtact-io] t={} fd={} reregister failed: {e}",
                trace_now_us(),
                fd
            );
            // Registration is broken for this fd — don't leave the fiber
            // parked waiting for an event that may never come; wake it
            // immediately so it retries the syscall directly and
            // surfaces a real error through the existing WouldBlock path.
            let woken = if is_reader {
                fd_state.reader_waker.take()
            } else {
                fd_state.writer_waker.take()
            };
            if is_reader {
                fd_state.reader_slot = None;
            } else {
                fd_state.writer_slot = None;
            }
            if let Some(w) = woken {
                w.wake();
            }
            false
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn get_mio_interest(fd_state: &FdState) -> mio::Interest {
    let r = fd_state.reader_waker.is_some();
    let w = fd_state.writer_waker.is_some();
    if r && w {
        mio::Interest::READABLE | mio::Interest::WRITABLE
    } else if r {
        mio::Interest::READABLE
    } else if w {
        mio::Interest::WRITABLE
    } else {
        mio::Interest::READABLE
    }
}

#[cfg(not(target_os = "linux"))]
fn run_mio_worker_loop(worker_idx: usize, state: &WorkerState) {
    if let Some(config) = GLOBAL_CONFIG.get() {
        if let Some(&cpu_id) = config.pin_cpus.get(worker_idx) {
            let _ = pin_thread_to_cpu(cpu_id);
        }
    }

    let poll = unsafe { &mut *state.poll.get() };
    let mut events = mio::Events::with_capacity(256);
    // Starts small and grows on demand via `ensure_fd_state` — no fixed
    // upper bound on fd numbers, unlike the old preallocated 65536-entry
    // table (which silently dropped events for anything beyond it).
    let mut fd_states: Vec<FdState> = Vec::with_capacity(256);

    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            break;
        }

        let mut processed_any = false;
        for q in state.queues.iter() {
            while let Some(req) = q.pop() {
                processed_any = true;
                process_mio_request(state, &mut fd_states, req);
            }
        }

        // Drained *after* the per-thread request queues above, so a
        // Cancel for a slot whose original request was still sitting in
        // its SPSC queue at the start of this iteration is guaranteed to
        // be processed after that request installs its waker — never
        // before, which would free/reuse the slot out from under it.
        while let Some(slot_idx) = state.cancel_queue.pop() {
            processed_any = true;
            cancel_mio_slot(state, &mut fd_states, slot_idx as usize);
        }

        state.is_sleeping.store(true, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        let mut any_pending = false;
        for q in state.queues.iter() {
            if !q.is_empty() {
                any_pending = true;
                break;
            }
        }

        let poll_res = if !any_pending {
            poll.poll(&mut events, Some(std::time::Duration::from_millis(10)))
        } else {
            poll.poll(&mut events, Some(std::time::Duration::from_millis(0)))
        };
        state.is_sleeping.store(false, Ordering::Release);

        if poll_res.is_err() {
            continue;
        }

        for event in events.iter() {
            let token = event.token();
            if token == mio::Token(0) {
                continue;
            }
            let fd = token.0;
            process_mio_event(
                state,
                &mut fd_states,
                fd,
                event.is_readable(),
                event.is_writable(),
            );
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn process_mio_request(state: &WorkerState, fd_states: &mut Vec<FdState>, req: IoRequest) {
    match req {
        IoRequest::Read { fd, slot_idx, .. }
        | IoRequest::Accept { fd, slot_idx, .. }
        | IoRequest::RecvFrom { fd, slot_idx, .. } => {
            ensure_fd_state(fd_states, fd as usize);
            install_interest(state, &mut fd_states[fd as usize], fd, slot_idx, true);
        }
        IoRequest::Write { fd, slot_idx, .. }
        | IoRequest::Connect { fd, slot_idx, .. }
        | IoRequest::SendTo { fd, slot_idx, .. } => {
            ensure_fd_state(fd_states, fd as usize);
            install_interest(state, &mut fd_states[fd as usize], fd, slot_idx, false);
        }
        IoRequest::RegisterFile { fd, slot_idx } => {
            let res = unsafe {
                let poll = &mut *state.poll.get();
                poll.registry().register(
                    &mut mio::unix::SourceFd(&fd),
                    mio::Token(fd as usize),
                    mio::Interest::READABLE | mio::Interest::WRITABLE,
                )
            };
            match res {
                Ok(()) => complete_mio_slot(state, slot_idx, fd),
                Err(e) => {
                    let os_err = e.raw_os_error().unwrap_or(libc::EINVAL);
                    complete_mio_slot(state, slot_idx, -os_err);
                }
            }
        }
        IoRequest::UnregisterFile {
            direct_fd_idx,
            slot_idx,
        } => {
            let _ = unsafe {
                let poll = &mut *state.poll.get();
                poll.registry()
                    .deregister(&mut mio::unix::SourceFd(&(direct_fd_idx as i32)))
            };
            if let Some(fd_state) = fd_states.get_mut(direct_fd_idx as usize) {
                fd_state.reader_waker = None;
                fd_state.writer_waker = None;
                fd_state.reader_slot = None;
                fd_state.writer_slot = None;
                fd_state.registered_interest = None;
            }
            complete_mio_slot(state, slot_idx, 0);
        }
    }
}

/// Handle a slot whose owning `DtactIoFuture` was dropped before its op
/// completed (see `Drop for DtactIoFuture`). Clears whichever side of the
/// fd's interest this slot owns (if the request had already been
/// processed into `fd_states`) and recycles the slot.
#[cfg(not(target_os = "linux"))]
fn cancel_mio_slot(state: &WorkerState, fd_states: &mut Vec<FdState>, slot_idx: usize) {
    let slot = &state.slots[slot_idx];
    let fd = slot.origin_fd.load(Ordering::Relaxed);

    if fd != u32::MAX {
        ensure_fd_state(fd_states, fd as usize);
        let fd_state = &mut fd_states[fd as usize];
        let mut touched = false;
        if fd_state.reader_slot == Some(slot_idx) {
            fd_state.reader_waker = None;
            fd_state.reader_slot = None;
            touched = true;
        }
        if fd_state.writer_slot == Some(slot_idx) {
            fd_state.writer_waker = None;
            fd_state.writer_slot = None;
            touched = true;
        }
        if touched {
            let interest = get_mio_interest(fd_state);
            if fd_state.registered_interest != Some(interest) {
                let res = unsafe {
                    let poll = &mut *state.poll.get();
                    poll.registry().reregister(
                        &mut mio::unix::SourceFd(&(fd as i32)),
                        mio::Token(fd as usize),
                        interest,
                    )
                };
                if res.is_ok() {
                    fd_state.registered_interest = Some(interest);
                }
            }
        }
    }

    state.free_slots.push(slot_idx as u32);
    wake_next_waiting_fiber(state);
}

#[cfg(not(target_os = "linux"))]
fn process_mio_event(
    _state: &WorkerState,
    fd_states: &mut Vec<FdState>,
    fd: usize,
    readable: bool,
    writable: bool,
) {
    ensure_fd_state(fd_states, fd);
    let fd_state = &mut fd_states[fd];

    if readable {
        if let Some(w) = fd_state.reader_waker.take() {
            w.wake();
        }
        fd_state.reader_slot = None;
    }
    if writable {
        if let Some(w) = fd_state.writer_waker.take() {
            w.wake();
        }
        fd_state.writer_slot = None;
    }

    let interest = get_mio_interest(fd_state);
    if fd_state.registered_interest != Some(interest) {
        let res = unsafe {
            let poll = &mut *_state.poll.get();
            poll.registry().reregister(
                &mut mio::unix::SourceFd(&(fd as i32)),
                mio::Token(fd),
                interest,
            )
        };
        if res.is_ok() {
            fd_state.registered_interest = Some(interest);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn complete_mio_slot(state: &WorkerState, slot_idx: usize, res: i32) {
    let slot = &state.slots[slot_idx];
    slot.result.store(res, Ordering::Release);

    // See the matching comment in `process_linux_completion`: extract the
    // waker before publishing `completed`, so a slot reused immediately
    // after another thread observes `completed` can never have its freshly
    // installed waker clobbered by this call.
    slot.lock_waker();
    let data = slot
        .waker_data
        .swap(std::ptr::null_mut(), Ordering::Relaxed);
    let vtable = slot
        .waker_vtable
        .swap(std::ptr::null_mut(), Ordering::Relaxed);
    slot.unlock_waker();

    slot.completed.store(true, Ordering::Release);

    if !data.is_null() && !vtable.is_null() {
        let raw = RawWaker::new(data as *const (), unsafe { &*vtable });
        let w = unsafe { Waker::from_raw(raw) };
        w.wake();
    }
}

// =========================================================================
// 9. DtactIoFuture INTERFACE
// =========================================================================
/// A single in-flight async socket op (read/write/accept/connect).
///
/// Dispatched to the io-worker for `worker_idx` and polled to completion.
/// Mirrors the Windows backend's `DtactIoFuture` field-for-field so
/// higher-level types (`DtactTcpStream`/`DtactTcpListener`) don't need
/// backend-specific code.
pub struct DtactIoFuture {
    /// Index of the io-worker (and its `WORKERS` slot) this op runs on.
    pub worker_idx: usize,
    /// The raw fd this op is issued against.
    pub fd: u32,
    /// `io_uring` direct/fixed-file index, or `u32::MAX` if `fd` is not
    /// registered as one.
    pub direct_fd_idx: u32,
    /// Which operation this future performs.
    pub op: OpCode,
    /// Read/Write only: pointer to the caller-supplied buffer.
    pub buf_ptr: *mut u8,
    /// Read/Write only: length of the buffer at `buf_ptr`.
    pub len: usize,
    /// Positional read/write offset (ignored for plain socket ops).
    pub offset: i64,
    /// Connect only: the remote address to connect to.
    pub addr: Option<libc::sockaddr_storage>,
    /// Connect only: byte length of `addr`.
    pub addr_len: libc::socklen_t,
    /// Slot index in the owning worker's op-slot table once the op has
    /// been submitted; `None` before the first `poll`.
    pub slot_idx: Option<usize>,
    /// `SendTo`/`RecvFrom` only: caller-owned `msghdr` (see
    /// `DtactUdpSocket`), null for every other op.
    pub msg_ptr: *mut libc::msghdr,
}

unsafe impl Send for DtactIoFuture {}
unsafe impl Sync for DtactIoFuture {}

impl DtactIoFuture {
    /// Construct a not-yet-submitted op. Submission happens on first
    /// `poll`, not here — see `impl Future for DtactIoFuture`.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        worker_idx: usize,
        fd: u32,
        direct_fd_idx: u32,
        op: OpCode,
        buf_ptr: *mut u8,
        len: usize,
        offset: i64,
        addr: Option<libc::sockaddr_storage>,
        addr_len: libc::socklen_t,
        slot_idx: Option<usize>,
    ) -> Self {
        Self {
            worker_idx,
            fd,
            direct_fd_idx,
            op,
            buf_ptr,
            len,
            offset,
            addr,
            addr_len,
            slot_idx,
            msg_ptr: std::ptr::null_mut(),
        }
    }

    const fn create_io_request(&self, slot_idx: usize) -> IoRequest {
        match self.op {
            OpCode::SendTo => IoRequest::SendTo {
                fd: self.fd,
                direct_fd_idx: self.direct_fd_idx,
                msg_ptr: self.msg_ptr,
                slot_idx,
            },
            OpCode::RecvFrom => IoRequest::RecvFrom {
                fd: self.fd,
                direct_fd_idx: self.direct_fd_idx,
                msg_ptr: self.msg_ptr,
                slot_idx,
            },
            OpCode::Read => IoRequest::Read {
                fd: self.fd,
                direct_fd_idx: self.direct_fd_idx,
                buf_ptr: self.buf_ptr,
                len: self.len,
                offset: self.offset,
                slot_idx,
            },
            OpCode::Write => IoRequest::Write {
                fd: self.fd,
                direct_fd_idx: self.direct_fd_idx,
                buf_ptr: self.buf_ptr,
                len: self.len,
                offset: self.offset,
                slot_idx,
            },
            OpCode::Accept => IoRequest::Accept {
                fd: self.fd,
                direct_fd_idx: self.direct_fd_idx,
                slot_idx,
            },
            OpCode::Connect => IoRequest::Connect {
                fd: self.fd,
                direct_fd_idx: self.direct_fd_idx,
                addr: self.addr.unwrap(),
                addr_len: self.addr_len,
                slot_idx,
            },
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn execute_syscall(&self) -> std::io::Result<usize> {
        let res = match self.op {
            OpCode::Read => {
                let buf_ptr = self.buf_ptr;
                let len = self.len;
                unsafe { libc::read(self.fd as i32, buf_ptr as *mut libc::c_void, len) }
            }
            OpCode::Write => {
                let buf_ptr = self.buf_ptr;
                let len = self.len;
                unsafe { libc::write(self.fd as i32, buf_ptr as *const libc::c_void, len) }
            }
            OpCode::Accept => unsafe {
                libc::accept(self.fd as i32, std::ptr::null_mut(), std::ptr::null_mut()) as isize
            },
            OpCode::SendTo => unsafe { libc::sendmsg(self.fd as i32, self.msg_ptr, 0) },
            OpCode::RecvFrom => unsafe { libc::recvmsg(self.fd as i32, self.msg_ptr, 0) },
            OpCode::Connect => {
                let addr_ptr =
                    &self.addr.unwrap() as *const libc::sockaddr_storage as *const libc::sockaddr;
                let res = unsafe { libc::connect(self.fd as i32, addr_ptr, self.addr_len) };
                if res < 0 {
                    let err = std::io::Error::last_os_error();
                    if err.raw_os_error() == Some(libc::EISCONN) {
                        return Ok(0);
                    }
                    return Err(err);
                }
                res as isize
            }
        };

        if res < 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(res as usize)
        }
    }
}

impl Future for DtactIoFuture {
    type Output = std::io::Result<usize>;

    // Covers every op's first-poll submission, in-flight re-poll, and
    // op-slot-pool-exhausted/waiting-list fallback path in one state
    // machine — the natural shape for a hand-written `Future::poll`, and
    // Linux-only code that's expensive to verify a refactor of without a
    // Linux box in the loop.
    #[allow(clippy::too_many_lines)]
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        #[cfg(target_os = "linux")]
        {
            let slot_idx = if let Some(idx) = self.slot_idx {
                idx
            } else {
                let state = &WORKERS.get().unwrap()[self.worker_idx];
                let idx = match state.free_slots.pop() {
                    Some(i) => i as usize,
                    None => {
                        if let Some(wait_idx) = state.free_wait_slots.pop() {
                            let wait_slot = &state.wait_slots[wait_idx as usize];
                            wait_slot
                                .waker_data
                                .store(cx.waker().data().cast_mut(), Ordering::Relaxed);
                            wait_slot.waker_vtable.store(
                                std::ptr::from_ref::<RawWakerVTable>(cx.waker().vtable())
                                    .cast_mut(),
                                Ordering::Relaxed,
                            );
                            state.waiting_queue.push(wait_idx);

                            if let Some(i) = state.free_slots.pop() {
                                wait_slot
                                    .waker_data
                                    .store(std::ptr::null_mut(), Ordering::Relaxed);
                                wait_slot
                                    .waker_vtable
                                    .store(std::ptr::null_mut(), Ordering::Relaxed);
                                i as usize
                            } else {
                                return Poll::Pending;
                            }
                        } else {
                            cx.waker().wake_by_ref();
                            return Poll::Pending;
                        }
                    }
                };

                let slot = &state.slots[idx];
                slot.completed.store(false, Ordering::Relaxed);
                slot.dropped.store(false, Ordering::Relaxed);
                slot.origin_fd.store(self.fd, Ordering::Relaxed);
                // Store the raw waker details.
                slot.lock_waker();
                slot.waker_data
                    .store(cx.waker().data().cast_mut(), Ordering::Relaxed);
                slot.waker_vtable.store(
                    std::ptr::from_ref::<RawWakerVTable>(cx.waker().vtable()).cast_mut(),
                    Ordering::Relaxed,
                );
                slot.unlock_waker();

                let req = self.create_io_request(idx);
                let q_idx = get_or_init_local_allocator().unwrap_or(0);
                let queue = &state.queues[q_idx];

                if queue.push(req).is_err() {
                    // Queue full — reset slot and retry next poll.
                    slot.lock_waker();
                    slot.waker_data
                        .store(std::ptr::null_mut(), Ordering::Relaxed);
                    slot.waker_vtable
                        .store(std::ptr::null_mut(), Ordering::Relaxed);
                    slot.unlock_waker();
                    state.free_slots.push(idx as u32);
                    wake_next_waiting_fiber(state);
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }

                // Paired with the io-worker's Dekker-style re-check
                // (see `run_linux_worker_loop`): this must be a
                // SeqCst load with a fence between the queue push
                // above and this load, otherwise the push and this
                // load can be observed out of order (StoreLoad
                // reorder) and we could skip the wakeup right as the
                // io-worker is about to go to sleep without ever
                // seeing the new queue entry — a permanent lost
                // wakeup / deadlock.
                fence(Ordering::SeqCst);
                if state.is_sleeping.load(Ordering::SeqCst) {
                    unsafe {
                        let _ = libc::write(
                            state.wake_eventfd,
                            std::ptr::from_ref::<u64>(&1u64).cast::<libc::c_void>(),
                            8,
                        );
                    }
                }

                io_trace!(
                    "[dtact-io] t={} slot={} fd={} op={:?} A_submit",
                    trace_now_us(),
                    idx,
                    self.fd,
                    self.op
                );

                self.slot_idx = Some(idx);
                idx
            };

            let state = &WORKERS.get().unwrap()[self.worker_idx];
            let slot = &state.slots[slot_idx];

            if slot.completed.load(Ordering::Acquire) {
                let res = slot.result.load(Ordering::Acquire);
                io_trace!(
                    "[dtact-io] t={} slot={} res={} C_fiber_resumed",
                    trace_now_us(),
                    slot_idx,
                    res
                );
                // Clear the waker
                slot.lock_waker();
                slot.waker_data
                    .store(std::ptr::null_mut(), Ordering::Relaxed);
                slot.waker_vtable
                    .store(std::ptr::null_mut(), Ordering::Relaxed);
                slot.unlock_waker();
                state.free_slots.push(slot_idx as u32);
                self.slot_idx = None;

                wake_next_waiting_fiber(state);

                if res < 0 {
                    Poll::Ready(Err(std::io::Error::from_raw_os_error(-res)))
                } else {
                    Poll::Ready(Ok(res as usize))
                }
            } else {
                // Still pending — update the waker if the waker changed
                // (e.g. the fiber migrated to a different scheduler core).
                let new_data = cx.waker().data().cast_mut();
                let new_vtable =
                    std::ptr::from_ref::<RawWakerVTable>(cx.waker().vtable()).cast_mut();

                slot.lock_waker();
                let old_data = slot.waker_data.load(Ordering::Relaxed);
                let old_vtable = slot.waker_vtable.load(Ordering::Relaxed);
                if old_data != new_data || old_vtable != new_vtable {
                    slot.waker_data.store(new_data, Ordering::Relaxed);
                    slot.waker_vtable.store(new_vtable, Ordering::Relaxed);
                }
                slot.unlock_waker();
                Poll::Pending
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            let res = self.execute_syscall();
            if self.slot_idx.is_some()
                && !matches!(res, Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock)
            {
                let state = &WORKERS.get().unwrap()[self.worker_idx];
                state.free_slots.push(self.slot_idx.unwrap() as u32);
                self.slot_idx = None;
                wake_next_waiting_fiber(state);
            }

            match res {
                Ok(n) => Poll::Ready(Ok(n)),
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    let slot_idx = match self.slot_idx {
                        Some(idx) => idx,
                        None => {
                            let state = &WORKERS.get().unwrap()[self.worker_idx];
                            let idx = match state.free_slots.pop() {
                                Some(i) => i as usize,
                                None => {
                                    if let Some(wait_idx) = state.free_wait_slots.pop() {
                                        let wait_slot = &state.wait_slots[wait_idx as usize];
                                        wait_slot
                                            .waker_data
                                            .store(cx.waker().data() as *mut (), Ordering::Relaxed);
                                        wait_slot.waker_vtable.store(
                                            cx.waker().vtable() as *const RawWakerVTable as *mut _,
                                            Ordering::Relaxed,
                                        );
                                        state.waiting_queue.push(wait_idx);

                                        if let Some(i) = state.free_slots.pop() {
                                            wait_slot
                                                .waker_data
                                                .store(std::ptr::null_mut(), Ordering::Relaxed);
                                            wait_slot
                                                .waker_vtable
                                                .store(std::ptr::null_mut(), Ordering::Relaxed);
                                            i as usize
                                        } else {
                                            return Poll::Pending;
                                        }
                                    } else {
                                        cx.waker().wake_by_ref();
                                        return Poll::Pending;
                                    }
                                }
                            };

                            let slot = &state.slots[idx];
                            slot.completed.store(false, Ordering::Relaxed);
                            slot.dropped.store(false, Ordering::Relaxed);
                            slot.origin_fd.store(self.fd, Ordering::Relaxed);
                            let raw = cx.waker().as_raw();
                            slot.lock_waker();
                            slot.waker_data
                                .store(raw.data() as *mut (), Ordering::Relaxed);
                            slot.waker_vtable.store(
                                raw.vtable() as *const RawWakerVTable as *mut _,
                                Ordering::Relaxed,
                            );
                            slot.unlock_waker();

                            let req = self.create_io_request(idx);
                            let q_idx = get_or_init_local_allocator().unwrap_or(0);
                            let queue = &state.queues[q_idx];

                            if queue.push(req).is_err() {
                                slot.lock_waker();
                                slot.waker_data
                                    .store(std::ptr::null_mut(), Ordering::Relaxed);
                                slot.waker_vtable
                                    .store(std::ptr::null_mut(), Ordering::Relaxed);
                                slot.unlock_waker();
                                state.free_slots.push(idx as u32);
                                wake_next_waiting_fiber(state);
                                cx.waker().wake_by_ref();
                                return Poll::Pending;
                            }

                            fence(Ordering::SeqCst);
                            if state.is_sleeping.load(Ordering::SeqCst) {
                                state.waker.wake();
                            }
                            self.slot_idx = Some(idx);
                            idx
                        }
                    };

                    let state = &WORKERS.get().unwrap()[self.worker_idx];
                    let slot = &state.slots[slot_idx];
                    let raw = cx.waker().as_raw();
                    let new_data = raw.data() as *mut ();
                    let new_vtable = raw.vtable() as *const RawWakerVTable as *mut _;

                    slot.lock_waker();
                    let old_data = slot.waker_data.load(Ordering::Relaxed);
                    let old_vtable = slot.waker_vtable.load(Ordering::Relaxed);
                    let mut changed = false;
                    if old_data != new_data || old_vtable != new_vtable {
                        slot.waker_data.store(new_data, Ordering::Relaxed);
                        slot.waker_vtable.store(new_vtable, Ordering::Relaxed);
                        changed = true;
                    }
                    slot.unlock_waker();

                    if changed {
                        let req = self.create_io_request(slot_idx);
                        let q_idx = get_or_init_local_allocator().unwrap_or(0);
                        let _ = state.queues[q_idx].push(req);
                        fence(Ordering::SeqCst);
                        if state.is_sleeping.load(Ordering::SeqCst) {
                            state.waker.wake();
                        }
                    }
                    Poll::Pending
                }
                Err(e) => Poll::Ready(Err(e)),
            }
        }
    }
}

impl Drop for DtactIoFuture {
    fn drop(&mut self) {
        let Some(idx) = self.slot_idx else { return };
        let Some(state) = WORKERS.get().and_then(|w| w.get(self.worker_idx)) else {
            return;
        };

        // Clear the waker so the io-worker won't try to wake a fiber that
        // is no longer polling this future.
        let slot = &state.slots[idx];
        slot.lock_waker();
        slot.waker_data
            .store(std::ptr::null_mut(), Ordering::Relaxed);
        slot.waker_vtable
            .store(std::ptr::null_mut(), Ordering::Relaxed);
        slot.unlock_waker();

        if slot.completed.load(Ordering::Acquire) {
            // The op already finished (CQE/event observed) and nobody will
            // touch this slot again — safe to recycle right away, from any
            // thread, since it never gets accessed again after this point.
            state.free_slots.push(idx as u32);
            wake_next_waiting_fiber(state);
            return;
        }

        // The op may still be in flight (submitted to the kernel / queued
        // for the io-worker / registered with the OS reactor). We must NOT
        // touch backend state (the io_uring `ring` or the mio `poll`)
        // from here — `Drop::drop` can run on an arbitrary thread, not
        // necessarily the io-worker thread that owns that `UnsafeCell`.
        // Instead, mark the slot dropped and hand cancellation off to the
        // owning worker via `cancel_queue`, which — unlike the per-thread
        // SPSC `queues` — is safe to push to from any thread.
        slot.dropped.store(true, Ordering::Release);
        state.cancel_queue.push(idx as u32);
        fence(Ordering::SeqCst);

        #[cfg(target_os = "linux")]
        {
            if state.is_sleeping.load(Ordering::SeqCst) {
                unsafe {
                    let _ = libc::write(
                        state.wake_eventfd,
                        std::ptr::from_ref::<u64>(&1u64).cast::<libc::c_void>(),
                        8,
                    );
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            if state.is_sleeping.load(Ordering::SeqCst) {
                state.waker.wake();
            }
        }
    }
}

// =========================================================================
// 10. HIGH-LEVEL API: DtactTcpStream AND DtactTcpListener
// =========================================================================
/// A lock-free, non-blocking TCP stream registered with the dtact-io
/// driver. Mirrors the Windows backend's `DtactTcpStream` API so callers
/// can switch platforms without code changes.
pub struct DtactTcpStream {
    inner: std::net::TcpStream,
    direct_fd_idx: u32,
    worker_idx: usize,
}

impl DtactTcpStream {
    /// Register an existing non-blocking `TcpStream` with the dtact-io driver.
    ///
    /// Registration is **synchronous and lock-free on the hot path** — it calls
    /// `io_uring_register_files_update` directly under a per-worker mutex rather
    /// than going through the SPSC queue, which would require a spin-wait and
    /// could deadlock when called from within a dtact fiber.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if `set_nonblocking`/`set_nodelay` fails, or
    /// if `io_uring`'s direct-file registration fails (e.g. the per-worker
    /// direct-fd table is exhausted).
    ///
    /// # Panics
    ///
    /// Panics if called before [`init_runtime`]/[`init`] has been called
    /// (`WORKERS` not yet initialized).
    pub fn from_std(stream: std::net::TcpStream) -> std::io::Result<Self> {
        let fd = stream.as_raw_fd();
        stream.set_nonblocking(true)?;
        // Nagle's algorithm batches small writes waiting for the peer's
        // ACK; combined with the peer's delayed-ACK timer (~40-200ms on
        // Linux) this stalls exactly the small request/response traffic
        // this driver targets. Every consumer of this async driver wants
        // low latency, not bandwidth-optimised batching, so disable it
        // unconditionally rather than leaving it as a footgun.
        stream.set_nodelay(true)?;

        let num_workers = GLOBAL_CONFIG.get().map_or(1, |c| c.workers);
        let worker_idx = fd as usize % num_workers;
        let state = &WORKERS.get().unwrap()[worker_idx];

        let direct_fd_idx = register_fd_sync(state, fd);

        Ok(Self {
            inner: stream,
            direct_fd_idx,
            worker_idx,
        })
    }

    /// Read into `buf`, returning `Ok(0)` immediately for an empty buffer
    /// without issuing a syscall.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `read(2)`/`io_uring` read
    /// completion reports one; `Ok(0)` signals EOF, not an error.
    pub async fn read(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // Try the syscall directly, exactly once, before paying for an
        // io_uring round trip — this only helps when data is already
        // available. A previous version of this function busy-spun the
        // OS thread for up to 4000 iterations (issuing the syscall every
        // 128 spins, ~31 raw syscalls) before falling back to async: on a
        // cooperative fiber scheduler that's actively harmful whenever
        // the data *isn't* ready yet (the common case for a server
        // request/response loop) — it blocks the OS thread from running
        // any other fiber and burns dozens of guaranteed-EAGAIN syscalls
        // per await instead of yielding immediately.
        let res = unsafe {
            let r = libc::read(
                self.inner.as_raw_fd(),
                buf.as_mut_ptr().cast::<libc::c_void>(),
                buf.len(),
            );
            match r.cmp(&0) {
                std::cmp::Ordering::Greater => Ok(r as usize),
                std::cmp::Ordering::Equal => Ok(0), // EOF
                std::cmp::Ordering::Less => Err(std::io::Error::last_os_error()),
            }
        };

        match res {
            Ok(n) => return Ok(n),
            Err(e) => {
                if e.kind() != std::io::ErrorKind::WouldBlock {
                    return Err(e);
                }
            }
        }

        // 100% Zerocopy, Lockless Direct path using DtactIoFuture
        DtactIoFuture {
            worker_idx: self.worker_idx,
            fd: self.inner.as_raw_fd() as u32,
            direct_fd_idx: self.direct_fd_idx,
            op: OpCode::Read,
            buf_ptr: buf.as_mut_ptr(),
            len: buf.len(),
            offset: 0,
            addr: None,
            addr_len: 0,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        }
        .await
        .map(|n| n.min(buf.len()))
    }

    /// Write `buf`, returning `Ok(0)` immediately for an empty buffer
    /// without issuing a syscall.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `write(2)`/`io_uring`
    /// write completion reports one (e.g. `BrokenPipe` if the peer closed
    /// the connection).
    pub async fn write(&self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // One direct attempt before going async — see the comment in
        // `read()` above for why this is no longer a busy-spin loop.
        let res = unsafe {
            let r = libc::write(
                self.inner.as_raw_fd(),
                buf.as_ptr().cast::<libc::c_void>(),
                buf.len(),
            );
            if r >= 0 {
                Ok(r as usize)
            } else {
                Err(std::io::Error::last_os_error())
            }
        };

        match res {
            Ok(n) => return Ok(n),
            Err(e) => {
                if e.kind() != std::io::ErrorKind::WouldBlock {
                    return Err(e);
                }
            }
        }

        DtactIoFuture {
            worker_idx: self.worker_idx,
            fd: self.inner.as_raw_fd() as u32,
            direct_fd_idx: self.direct_fd_idx,
            op: OpCode::Write,
            buf_ptr: buf.as_ptr().cast_mut(),
            len: buf.len(),
            offset: 0,
            addr: None,
            addr_len: 0,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        }
        .await
    }

    /// Create a new non-blocking socket and connect to `addr`, registering
    /// the result with the dtact-io driver.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if socket creation, `set_nonblocking`/
    /// `set_nodelay`, the `connect(2)` syscall (or its `io_uring`
    /// completion), or direct-file registration fails — e.g.
    /// `ConnectionRefused` if nothing is listening at `addr`.
    ///
    /// # Panics
    ///
    /// Panics if called before [`init_runtime`]/[`init`] has been called
    /// (`WORKERS` not yet initialized).
    // Socket creation, connect submission (both io_uring's `Connect`
    // opcode and the poll-based non-Linux fallback), and direct-file
    // registration are all inherently part of one linear "create the
    // connected stream" sequence; this is also Linux/Unix-only code that's
    // expensive to verify a refactor of without a Linux box in the loop.
    #[allow(clippy::too_many_lines)]
    pub async fn connect(addr: std::net::SocketAddr) -> std::io::Result<Self> {
        let domain = match addr {
            std::net::SocketAddr::V4(_) => libc::AF_INET,
            std::net::SocketAddr::V6(_) => libc::AF_INET6,
        };
        let fd = unsafe {
            libc::socket(
                domain,
                libc::SOCK_STREAM | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
                0,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }

        // `from_raw_fd` takes ownership; the socket is closed on Drop.
        let stream = unsafe { std::net::TcpStream::from_raw_fd(fd) };
        // See the comment in `from_std` — Nagle + the peer's delayed ACK
        // otherwise stalls small request/response traffic by ~40-200ms.
        stream.set_nodelay(true)?;
        let num_workers = GLOBAL_CONFIG.get().map_or(1, |c| c.workers);
        let worker_idx = fd as usize % num_workers;
        let state = &WORKERS.get().unwrap()[worker_idx];

        // register_fd_sync returns u32::MAX (raw-fd mode) — no queue, no spin,
        // no deadlock risk when called from within a dtact fiber.
        let direct_fd_idx = register_fd_sync(state, fd);

        let (libc_addr, addr_len) = socket_addr_to_libc(addr);

        // Try direct connect first!
        let connect_res = unsafe {
            libc::connect(
                fd,
                (&raw const libc_addr).cast::<libc::sockaddr>(),
                addr_len,
            )
        };
        if connect_res == 0 {
            return Ok(Self {
                inner: stream,
                direct_fd_idx,
                worker_idx,
            });
        }
        let err = std::io::Error::last_os_error();
        #[cfg(target_os = "windows")]
        let is_in_progress = err.raw_os_error()
            == Some(windows_sys::Win32::Networking::WinSock::WSAEWOULDBLOCK as i32);
        #[cfg(not(target_os = "windows"))]
        let is_in_progress = err.raw_os_error() == Some(libc::EINPROGRESS);

        if !is_in_progress {
            return Err(err);
        }

        // One non-blocking `poll` check before going async — see the
        // comment in `read()` above for why this is no longer a
        // busy-spin loop (connect latency is dominated by the network
        // round trip anyway, so spinning here never helps).
        let mut pollfd = libc::pollfd {
            fd,
            events: libc::POLLOUT,
            revents: 0,
        };
        let poll_res = unsafe { libc::poll(&raw mut pollfd, 1, 0) };
        if poll_res > 0 {
            if (pollfd.revents & libc::POLLOUT) != 0 {
                let mut err_code: libc::c_int = 0;
                let mut err_len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
                let sockopt_res = unsafe {
                    libc::getsockopt(
                        fd,
                        libc::SOL_SOCKET,
                        libc::SO_ERROR,
                        (&raw mut err_code).cast::<libc::c_void>(),
                        &raw mut err_len,
                    )
                };
                if sockopt_res == 0 && err_code == 0 {
                    return Ok(Self {
                        inner: stream,
                        direct_fd_idx,
                        worker_idx,
                    });
                }
                let os_err = if err_code != 0 {
                    err_code
                } else {
                    libc::ECONNREFUSED
                };
                return Err(std::io::Error::from_raw_os_error(os_err));
            } else if (pollfd.revents & (libc::POLLERR | libc::POLLHUP)) != 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionRefused,
                    "connect failed",
                ));
            }
        }

        let connect_res = DtactIoFuture {
            worker_idx,
            fd: fd as u32,
            direct_fd_idx,
            op: OpCode::Connect,
            buf_ptr: std::ptr::null_mut(),
            len: 0,
            offset: 0,
            addr: Some(libc_addr),
            addr_len,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        }
        .await;

        match connect_res {
            Ok(_) => Ok(Self {
                inner: stream,
                direct_fd_idx,
                worker_idx,
            }),
            Err(e) => Err(e),
        }
    }
}

impl Drop for DtactTcpStream {
    fn drop(&mut self) {
        if let Some(workers) = WORKERS.get()
            && let Some(state) = workers.get(self.worker_idx)
        {
            unregister_fd_sync(state, self.direct_fd_idx);
        }
    }
}

impl crate::io::AsyncRead for DtactTcpStream {
    async fn read(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.read(buf).await
    }
}

impl crate::io::AsyncWrite for DtactTcpStream {
    async fn write(&self, buf: &[u8]) -> std::io::Result<usize> {
        self.write(buf).await
    }
}

/// A lock-free, non-blocking TCP listener registered with the dtact-io
/// driver. Mirrors the Windows backend's `DtactTcpListener` API so callers
/// can switch platforms without code changes.
pub struct DtactTcpListener {
    inner: std::net::TcpListener,
    direct_fd_idx: u32,
    worker_idx: usize,
}

impl DtactTcpListener {
    /// Register an existing non-blocking `TcpListener` with the dtact-io
    /// driver (see `DtactTcpStream::from_std` for the registration
    /// mechanics).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if `set_nonblocking` fails or if
    /// `io_uring`'s direct-file registration fails.
    ///
    /// # Panics
    ///
    /// Panics if called before [`init_runtime`]/[`init`] has been called
    /// (`WORKERS` not yet initialized).
    pub fn from_std(listener: std::net::TcpListener) -> std::io::Result<Self> {
        let fd = listener.as_raw_fd();
        listener.set_nonblocking(true)?;

        let num_workers = GLOBAL_CONFIG.get().map_or(1, |c| c.workers);
        let worker_idx = fd as usize % num_workers;
        let state = &WORKERS.get().unwrap()[worker_idx];

        let direct_fd_idx = register_fd_sync(state, fd);

        Ok(Self {
            inner: listener,
            direct_fd_idx,
            worker_idx,
        })
    }

    /// Accept a new connection, registering the accepted stream with the
    /// dtact-io driver.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `accept(2)`/`io_uring`
    /// accept completion reports one, or if registering the new stream
    /// with the driver fails.
    pub async fn accept(&self) -> std::io::Result<(DtactTcpStream, std::net::SocketAddr)> {
        // One direct attempt before going async — see the comment in
        // `read()` above for why this is no longer a busy-spin loop. An
        // accept() in particular has no reason to expect a pending
        // connection at any given instant, so spinning here was pure
        // waste on every call that didn't already have one queued.
        let res = unsafe {
            let mut addr: libc::sockaddr_storage = std::mem::zeroed();
            let mut len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
            let r = libc::accept(
                self.inner.as_raw_fd(),
                (&raw mut addr).cast::<libc::sockaddr>(),
                &raw mut len,
            );
            if r >= 0 {
                Ok((r, addr, len))
            } else {
                Err(std::io::Error::last_os_error())
            }
        };

        match res {
            Ok((client_fd, addr, len)) => {
                // Parse peer addr directly from the sockaddr we already have —
                // no extra getpeername() syscall needed.
                let peer_addr = sockaddr_storage_to_socketaddr(&addr, len);
                // Set nonblocking on the client fd.
                unsafe { libc::fcntl(client_fd, libc::F_SETFL, libc::O_NONBLOCK) };
                let stream = unsafe { std::net::TcpStream::from_raw_fd(client_fd) };
                let client_stream = DtactTcpStream::from_std(stream)?;
                return Ok((client_stream, peer_addr));
            }
            Err(e) => {
                if e.kind() != std::io::ErrorKind::WouldBlock {
                    return Err(e);
                }
            }
        }

        let res = DtactIoFuture {
            worker_idx: self.worker_idx,
            fd: self.inner.as_raw_fd() as u32,
            direct_fd_idx: self.direct_fd_idx,
            op: OpCode::Accept,
            buf_ptr: std::ptr::null_mut(),
            len: 0,
            offset: 0,
            addr: None,
            addr_len: 0,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        }
        .await?;

        let client_fd = res as RawFd;
        // Set nonblocking on the accepted fd.
        unsafe { libc::fcntl(client_fd, libc::F_SETFL, libc::O_NONBLOCK) };
        let stream = unsafe { std::net::TcpStream::from_raw_fd(client_fd) };
        let peer_addr = stream.peer_addr()?;
        let client_stream = DtactTcpStream::from_std(stream)?;
        Ok((client_stream, peer_addr))
    }
}

impl Drop for DtactTcpListener {
    fn drop(&mut self) {
        if let Some(workers) = WORKERS.get()
            && let Some(state) = workers.get(self.worker_idx)
        {
            unregister_fd_sync(state, self.direct_fd_idx);
        }
    }
}

// =========================================================================
// 10b. HIGH-LEVEL API: DtactUdpSocket
// =========================================================================

/// Async UDP socket driven by the native backend (`io_uring` `SendMsg`/`RecvMsg`
/// on Linux, `sendmsg`/`recvmsg` via the mio/kqueue reactor elsewhere).
///
/// Supports the connectionless (`send_to`/`recv_from`) and connected
/// (`connect`/`send`/`recv`) patterns, mirroring `std::net::UdpSocket`'s and
/// `tokio::net::UdpSocket`'s API shape. The connected `send`/`recv` reuse the
/// same `Write`/`Read` submission machinery as [`DtactTcpStream`].
pub struct DtactUdpSocket {
    inner: std::net::UdpSocket,
    direct_fd_idx: u32,
    worker_idx: usize,
}

impl DtactUdpSocket {
    /// Bind a new UDP socket to `addr` and register it with the driver.
    ///
    /// # Errors
    /// Returns any error from binding the OS socket or registering it.
    pub async fn bind(addr: std::net::SocketAddr) -> std::io::Result<Self> {
        let sock = std::net::UdpSocket::bind(addr)?;
        Self::from_std(sock)
    }

    /// Register an existing (already-bound) `std::net::UdpSocket`, taking
    /// ownership.
    ///
    /// # Errors
    /// Returns any error from switching the socket to non-blocking mode or
    /// registering it with the driver.
    ///
    /// # Panics
    /// Panics if called before [`init_runtime`]/[`init`] has been called
    /// (`WORKERS` not yet initialized).
    pub fn from_std(socket: std::net::UdpSocket) -> std::io::Result<Self> {
        let fd = socket.as_raw_fd();
        socket.set_nonblocking(true)?;
        let num_workers = GLOBAL_CONFIG.get().map_or(1, |c| c.workers);
        let worker_idx = fd as usize % num_workers;
        let state = &WORKERS.get().unwrap()[worker_idx];
        let direct_fd_idx = register_fd_sync(state, fd);
        Ok(Self {
            inner: socket,
            direct_fd_idx,
            worker_idx,
        })
    }

    /// The local address this socket is bound to.
    ///
    /// # Errors
    /// Returns any error from the underlying `getsockname` call.
    pub fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.inner.local_addr()
    }

    /// Send `buf` as a single datagram to `target`, returning the number of
    /// bytes sent.
    ///
    /// # Errors
    /// Returns any error from the underlying `sendmsg`.
    pub async fn send_to(
        &self,
        buf: &[u8],
        target: std::net::SocketAddr,
    ) -> std::io::Result<usize> {
        // `libc::iovec`/`libc::msghdr` embed a `*mut c_void`, which isn't
        // `Send` by default — the compiler can't tell that the pointee is
        // exclusively owned by this op and never touched concurrently, so
        // without this wrapper the future this `async fn` desugars to
        // wouldn't be `Send` (needed for a fiber's future to migrate
        // between dtact's worker threads). `SendToState` bundles the
        // locals that must stay put across the `.await` below (the kernel
        // reads through raw pointers inside `msg` for as long as the op is
        // in flight) so we can assert `Send` once, in one place, rather
        // than have it fail opaquely at the whole future's boundary.
        struct SendToState {
            storage: libc::sockaddr_storage,
            iov: libc::iovec,
            msg: libc::msghdr,
        }
        // SAFETY: `storage`/`iov`/`msg` are exclusively owned by this
        // future (part of its own generated state machine, stable once
        // pinned); the raw pointers they contain point at `storage` and at
        // the caller's `buf`, never at anything a second thread could
        // concurrently touch — the same ownership-transfer-via-submission
        // contract every other `IoRequest` variant already relies on (see
        // `unsafe impl Send for DtactIoFuture`, above).
        unsafe impl Send for SendToState {}

        let (storage, addr_len) = socket_addr_to_libc(target);
        let mut state = SendToState {
            storage,
            iov: libc::iovec {
                iov_base: buf.as_ptr().cast_mut().cast::<libc::c_void>(),
                iov_len: buf.len(),
            },
            msg: unsafe { std::mem::zeroed() },
        };
        state.msg.msg_name = std::ptr::addr_of_mut!(state.storage).cast::<libc::c_void>();
        state.msg.msg_namelen = addr_len;
        state.msg.msg_iov = &raw mut state.iov;
        state.msg.msg_iovlen = 1;

        // One direct attempt before going async — see `DtactTcpStream::write`.
        let r = unsafe { libc::sendmsg(self.inner.as_raw_fd(), &raw const state.msg, 0) };
        if r >= 0 {
            return Ok(r as usize);
        }
        let e = std::io::Error::last_os_error();
        if e.kind() != std::io::ErrorKind::WouldBlock {
            return Err(e);
        }

        let mut fut = DtactIoFuture::new(
            self.worker_idx,
            self.inner.as_raw_fd() as u32,
            self.direct_fd_idx,
            OpCode::SendTo,
            std::ptr::null_mut(),
            0,
            0,
            None,
            0,
            None,
        );
        fut.msg_ptr = &raw mut state.msg;
        fut.await
    }

    /// Receive a single datagram into `buf`, returning the byte count and the
    /// peer address it came from.
    ///
    /// # Errors
    /// Returns any error from the underlying `recvmsg`.
    pub async fn recv_from(
        &self,
        buf: &mut [u8],
    ) -> std::io::Result<(usize, std::net::SocketAddr)> {
        // See `send_to`'s `SendToState` for why this wrapper (and its
        // `unsafe impl Send`) exists.
        struct RecvFromState {
            storage: libc::sockaddr_storage,
            iov: libc::iovec,
            msg: libc::msghdr,
        }
        // SAFETY: same reasoning as `send_to`'s `SendToState` — exclusively
        // owned by this future, pointers only ever point at its own
        // fields/the caller's `buf`.
        unsafe impl Send for RecvFromState {}

        let mut state = RecvFromState {
            storage: unsafe { std::mem::zeroed() },
            iov: libc::iovec {
                iov_base: buf.as_mut_ptr().cast::<libc::c_void>(),
                iov_len: buf.len(),
            },
            msg: unsafe { std::mem::zeroed() },
        };
        state.msg.msg_name = std::ptr::addr_of_mut!(state.storage).cast::<libc::c_void>();
        state.msg.msg_namelen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
        state.msg.msg_iov = &raw mut state.iov;
        state.msg.msg_iovlen = 1;

        let r = unsafe { libc::recvmsg(self.inner.as_raw_fd(), &raw mut state.msg, 0) };
        if r >= 0 {
            let from = sockaddr_storage_to_socketaddr(&state.storage, state.msg.msg_namelen);
            return Ok((r as usize, from));
        }
        let e = std::io::Error::last_os_error();
        if e.kind() != std::io::ErrorKind::WouldBlock {
            return Err(e);
        }

        let mut fut = DtactIoFuture::new(
            self.worker_idx,
            self.inner.as_raw_fd() as u32,
            self.direct_fd_idx,
            OpCode::RecvFrom,
            std::ptr::null_mut(),
            0,
            0,
            None,
            0,
            None,
        );
        fut.msg_ptr = &raw mut state.msg;
        let n = fut.await?;
        let from = sockaddr_storage_to_socketaddr(&state.storage, state.msg.msg_namelen);
        Ok((n, from))
    }

    /// Connect this socket to `addr` so [`send`](Self::send)/[`recv`](Self::recv)
    /// can omit the peer address. UDP `connect` is a local operation, so it
    /// completes without a round trip.
    ///
    /// # Errors
    /// Returns any error from the underlying `connect`.
    pub async fn connect(&self, addr: std::net::SocketAddr) -> std::io::Result<()> {
        self.inner.connect(addr)
    }

    /// Send `buf` to the connected peer, returning the number of bytes sent.
    ///
    /// # Errors
    /// Returns any error from the underlying send.
    pub async fn send(&self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let r = unsafe {
            libc::send(
                self.inner.as_raw_fd(),
                buf.as_ptr().cast::<libc::c_void>(),
                buf.len(),
                0,
            )
        };
        if r >= 0 {
            return Ok(r as usize);
        }
        let e = std::io::Error::last_os_error();
        if e.kind() != std::io::ErrorKind::WouldBlock {
            return Err(e);
        }
        DtactIoFuture {
            worker_idx: self.worker_idx,
            fd: self.inner.as_raw_fd() as u32,
            direct_fd_idx: self.direct_fd_idx,
            op: OpCode::Write,
            buf_ptr: buf.as_ptr().cast_mut(),
            len: buf.len(),
            offset: 0,
            addr: None,
            addr_len: 0,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        }
        .await
    }

    /// Receive a datagram from the connected peer into `buf`, returning the
    /// byte count.
    ///
    /// # Errors
    /// Returns any error from the underlying recv.
    pub async fn recv(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let r = unsafe {
            libc::recv(
                self.inner.as_raw_fd(),
                buf.as_mut_ptr().cast::<libc::c_void>(),
                buf.len(),
                0,
            )
        };
        if r >= 0 {
            return Ok(r as usize);
        }
        let e = std::io::Error::last_os_error();
        if e.kind() != std::io::ErrorKind::WouldBlock {
            return Err(e);
        }
        DtactIoFuture {
            worker_idx: self.worker_idx,
            fd: self.inner.as_raw_fd() as u32,
            direct_fd_idx: self.direct_fd_idx,
            op: OpCode::Read,
            buf_ptr: buf.as_mut_ptr(),
            len: buf.len(),
            offset: 0,
            addr: None,
            addr_len: 0,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        }
        .await
    }
}

impl Drop for DtactUdpSocket {
    fn drop(&mut self) {
        if let Some(workers) = WORKERS.get()
            && let Some(state) = workers.get(self.worker_idx)
        {
            unregister_fd_sync(state, self.direct_fd_idx);
        }
    }
}

// =========================================================================
// 10c. HIGH-LEVEL API: DtactUnixStream / DtactUnixListener
// =========================================================================
// Unix-domain-socket counterpart to `DtactTcpStream`/`DtactTcpListener` —
// same `Read`/`Write`/`Accept`/`Connect` `DtactIoFuture` submission
// machinery (it only ever cared about the raw fd and an `OpCode`, never
// the address family), same fast-path-syscall-before-going-async shape.
// The only real differences from TCP are address handling (a filesystem
// path via `libc::sockaddr_un`, not `sockaddr_in`/`sockaddr_in6`) and that
// there's no `TCP_NODELAY` equivalent to disable — Unix domain sockets
// have no Nagle's-algorithm-style batching to begin with.
//
// Available on every Unix this file compiles for (this whole file is
// already `cfg(all(feature = "native", unix))`, per `io::mod`) — both the
// Linux `io_uring` path and the mio/kqueue fallback path drive
// `DtactIoFuture` purely off a raw fd + `OpCode`, so nothing here needed
// Linux-specific syscalls beyond what `unix_path_to_libc` already handles
// portably (it reads `sockaddr_un::sun_path`'s actual length from the
// platform's own `libc` struct rather than hardcoding Linux's 108-byte
// value, so it's correct on macOS/BSD's shorter `sun_path` too). Windows
// has no Unix-domain-socket analogue at all; use
// `crate::io::DtactNamedPipe` there instead.

/// A lock-free, non-blocking Unix-domain-socket stream registered with
/// the dtact-io driver. Mirrors [`DtactTcpStream`]'s API.
pub struct DtactUnixStream {
    inner: std::os::unix::net::UnixStream,
    direct_fd_idx: u32,
    worker_idx: usize,
    read_backpressured: std::sync::atomic::AtomicBool,
    write_backpressured: std::sync::atomic::AtomicBool,
}

impl DtactUnixStream {
    /// Register an existing non-blocking `UnixStream` with the dtact-io
    /// driver. See [`DtactTcpStream::from_std`] for the registration
    /// mechanics (identical here, minus the `TCP_NODELAY` call — Unix
    /// domain sockets have nothing analogous to disable).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if `set_nonblocking` fails.
    ///
    /// # Panics
    ///
    /// Panics if called before [`init_runtime`]/[`init`] has been called
    /// (`WORKERS` not yet initialized).
    pub fn from_std(stream: std::os::unix::net::UnixStream) -> std::io::Result<Self> {
        let fd = stream.as_raw_fd();
        stream.set_nonblocking(true)?;

        let num_workers = GLOBAL_CONFIG.get().map_or(1, |c| c.workers);
        let worker_idx = WORKER_ROUND_ROBIN.fetch_add(1, Ordering::Relaxed) % num_workers;
        let state = &WORKERS.get().unwrap()[worker_idx];

        let direct_fd_idx = register_fd_sync(state, fd);

        Ok(Self {
            inner: stream,
            direct_fd_idx,
            worker_idx,
            read_backpressured: std::sync::atomic::AtomicBool::new(false),
            write_backpressured: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Read into `buf`, returning `Ok(0)` immediately for an empty buffer
    /// without issuing a syscall. See [`DtactTcpStream::read`] for the
    /// fast-path-syscall-before-going-async rationale (identical here).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying read reports one; `Ok(0)`
    /// signals EOF, not an error.
    pub async fn read(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        if !self.read_backpressured.load(Ordering::Relaxed) {
            let res = unsafe {
                let r = libc::read(
                    self.inner.as_raw_fd(),
                    buf.as_mut_ptr().cast::<libc::c_void>(),
                    buf.len(),
                );
                match r.cmp(&0) {
                    std::cmp::Ordering::Greater => Ok(r as usize),
                    std::cmp::Ordering::Equal => Ok(0), // EOF
                    std::cmp::Ordering::Less => Err(std::io::Error::last_os_error()),
                }
            };

            match res {
                Ok(n) => return Ok(n),
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::WouldBlock {
                        return Err(e);
                    }
                }
            }
            self.read_backpressured.store(true, Ordering::Relaxed);
        }

        let future = DtactIoFuture {
            worker_idx: self.worker_idx,
            fd: self.inner.as_raw_fd() as u32,
            direct_fd_idx: self.direct_fd_idx,
            op: OpCode::Read,
            buf_ptr: buf.as_mut_ptr(),
            len: buf.len(),
            offset: 0,
            addr: None,
            addr_len: 0,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        }
        .await;

        self.read_backpressured.store(false, Ordering::Relaxed);
        future.map(|n| n.min(buf.len()))
    }

    /// Write `buf`, returning `Ok(0)` immediately for an empty buffer
    /// without issuing a syscall. See [`DtactTcpStream::write`] for the
    /// fast-path rationale (identical here).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying write reports one (e.g.
    /// `BrokenPipe` if the peer closed the connection).
    pub async fn write(&self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        if !self.write_backpressured.load(Ordering::Relaxed) {
            let res = unsafe {
                let r = libc::write(
                    self.inner.as_raw_fd(),
                    buf.as_ptr().cast::<libc::c_void>(),
                    buf.len(),
                );
                if r >= 0 {
                    Ok(r as usize)
                } else {
                    Err(std::io::Error::last_os_error())
                }
            };

            match res {
                Ok(n) => return Ok(n),
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::WouldBlock {
                        return Err(e);
                    }
                }
            }
            self.write_backpressured.store(true, Ordering::Relaxed);
        }

        let future = DtactIoFuture {
            worker_idx: self.worker_idx,
            fd: self.inner.as_raw_fd() as u32,
            direct_fd_idx: self.direct_fd_idx,
            op: OpCode::Write,
            buf_ptr: buf.as_ptr().cast_mut(),
            len: buf.len(),
            offset: 0,
            addr: None,
            addr_len: 0,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        };

        self.write_backpressured.store(false, Ordering::Relaxed);
        future.await
    }

    /// Create a new non-blocking Unix domain socket and connect to the
    /// filesystem path `path`, registering the result with the dtact-io
    /// driver. See [`DtactTcpStream::connect`] for the direct-connect /
    /// `EINPROGRESS` / async-fallback sequencing (identical here, modulo
    /// address family).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if socket creation, `set_nonblocking`, the
    /// `connect(2)` syscall (or its `io_uring` completion), or `path`
    /// itself (e.g. too long for `sockaddr_un::sun_path`) fails — e.g.
    /// `ConnectionRefused`/`NotFound` if nothing is listening at `path`.
    ///
    /// # Panics
    ///
    /// Panics if called before [`init_runtime`]/[`init`] has been called
    /// (`WORKERS` not yet initialized).
    #[allow(clippy::too_many_lines)]
    pub async fn connect(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let (libc_addr, addr_len) = unix_path_to_libc(path.as_ref())?;

        let fd = unsafe {
            libc::socket(
                libc::AF_UNIX,
                libc::SOCK_STREAM | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
                0,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }

        // `from_raw_fd` takes ownership; the socket is closed on Drop.
        let stream = unsafe { std::os::unix::net::UnixStream::from_raw_fd(fd) };
        let num_workers = GLOBAL_CONFIG.get().map_or(1, |c| c.workers);
        let worker_idx = fd as usize % num_workers;
        let state = &WORKERS.get().unwrap()[worker_idx];
        let direct_fd_idx = register_fd_sync(state, fd);

        // Try direct connect first.
        let connect_res = unsafe {
            libc::connect(
                fd,
                (&raw const libc_addr).cast::<libc::sockaddr>(),
                addr_len,
            )
        };
        if connect_res == 0 {
            return Ok(Self {
                inner: stream,
                direct_fd_idx,
                worker_idx,
                read_backpressured: std::sync::atomic::AtomicBool::new(false),
                write_backpressured: std::sync::atomic::AtomicBool::new(false),
            });
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EINPROGRESS) {
            return Err(err);
        }

        // One non-blocking `poll` check before going async — see the
        // comment in `DtactTcpStream::connect` for why (connect latency
        // for a local Unix socket is dominated by the peer's own accept
        // loop scheduling, not spinnable away).
        let mut pollfd = libc::pollfd {
            fd,
            events: libc::POLLOUT,
            revents: 0,
        };
        let poll_res = unsafe { libc::poll(&raw mut pollfd, 1, 0) };
        if poll_res > 0 {
            if (pollfd.revents & libc::POLLOUT) != 0 {
                let mut err_code: libc::c_int = 0;
                let mut err_len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
                let sockopt_res = unsafe {
                    libc::getsockopt(
                        fd,
                        libc::SOL_SOCKET,
                        libc::SO_ERROR,
                        (&raw mut err_code).cast::<libc::c_void>(),
                        &raw mut err_len,
                    )
                };
                if sockopt_res == 0 && err_code == 0 {
                    return Ok(Self {
                        inner: stream,
                        direct_fd_idx,
                        worker_idx,
                        read_backpressured: std::sync::atomic::AtomicBool::new(false),
                        write_backpressured: std::sync::atomic::AtomicBool::new(false),
                    });
                }
                let os_err = if err_code != 0 {
                    err_code
                } else {
                    libc::ECONNREFUSED
                };
                return Err(std::io::Error::from_raw_os_error(os_err));
            } else if (pollfd.revents & (libc::POLLERR | libc::POLLHUP)) != 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionRefused,
                    "connect failed",
                ));
            }
        }

        let connect_res = DtactIoFuture {
            worker_idx,
            fd: fd as u32,
            direct_fd_idx,
            op: OpCode::Connect,
            buf_ptr: std::ptr::null_mut(),
            len: 0,
            offset: 0,
            addr: Some(libc_addr),
            addr_len,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        }
        .await;

        match connect_res {
            Ok(_) => Ok(Self {
                inner: stream,
                direct_fd_idx,
                worker_idx,
                read_backpressured: std::sync::atomic::AtomicBool::new(false),
                write_backpressured: std::sync::atomic::AtomicBool::new(false),
            }),
            Err(e) => Err(e),
        }
    }

    /// The connected peer's credentials (PID/UID/GID), as recorded by the
    /// kernel at `connect(2)`/`accept(2)` time — not queryable/spoofable
    /// after the fact by the peer itself, which is what makes this useful
    /// for authorization over a local socket.
    ///
    /// A plain synchronous syscall (`getsockopt(SO_PEERCRED)` on Linux,
    /// `getpeereid(2)` elsewhere), not routed through the driver — same
    /// "cheap enough to not need the ring" judgment call as
    /// `DtactTcpStream`'s `local_addr`-shaped helpers.
    ///
    /// # Errors
    /// Returns an `io::Error` if the underlying syscall fails (e.g. the
    /// socket was closed concurrently).
    pub fn peer_cred(&self) -> std::io::Result<DtactUCred> {
        peer_cred_impl(self.inner.as_raw_fd())
    }
}

impl Drop for DtactUnixStream {
    fn drop(&mut self) {
        if let Some(workers) = WORKERS.get()
            && let Some(state) = workers.get(self.worker_idx)
        {
            unregister_fd_sync(state, self.direct_fd_idx);
        }
    }
}

impl crate::io::AsyncRead for DtactUnixStream {
    async fn read(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.read(buf).await
    }
}

impl crate::io::AsyncWrite for DtactUnixStream {
    async fn write(&self, buf: &[u8]) -> std::io::Result<usize> {
        self.write(buf).await
    }
}

/// A lock-free, non-blocking Unix-domain-socket listener registered with
/// the dtact-io driver. Mirrors [`DtactTcpListener`]'s API.
pub struct DtactUnixListener {
    inner: std::os::unix::net::UnixListener,
    direct_fd_idx: u32,
    worker_idx: usize,
}

impl DtactUnixListener {
    /// Bind a new listener to the filesystem path `path` and register it
    /// with the driver. `path` must not already exist — like
    /// `std::os::unix::net::UnixListener::bind`, this does not remove a
    /// stale socket file left behind by a previous run.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `bind(2)` fails (e.g.
    /// `AddrInUse` if `path` already exists) or registration fails.
    pub fn bind(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let listener = std::os::unix::net::UnixListener::bind(path)?;
        Self::from_std(listener)
    }

    /// Register an existing non-blocking `UnixListener` with the dtact-io
    /// driver (see [`DtactTcpListener::from_std`] for the mechanics).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if `set_nonblocking` fails.
    ///
    /// # Panics
    ///
    /// Panics if called before [`init_runtime`]/[`init`] has been called
    /// (`WORKERS` not yet initialized).
    pub fn from_std(listener: std::os::unix::net::UnixListener) -> std::io::Result<Self> {
        let fd = listener.as_raw_fd();
        listener.set_nonblocking(true)?;

        let num_workers = GLOBAL_CONFIG.get().map_or(1, |c| c.workers);
        let worker_idx = WORKER_ROUND_ROBIN.fetch_add(1, Ordering::Relaxed) % num_workers;
        let state = &WORKERS.get().unwrap()[worker_idx];

        let direct_fd_idx = register_fd_sync(state, fd);

        Ok(Self {
            inner: listener,
            direct_fd_idx,
            worker_idx,
        })
    }

    /// Accept a new connection, registering the accepted stream with the
    /// dtact-io driver.
    ///
    /// Unlike [`DtactTcpListener::accept`], the peer address is fetched
    /// via a `getpeername`-equivalent (`UnixStream::peer_addr`) rather
    /// than hand-decoded from the raw `accept(2)`/`io_uring` result —
    /// Unix domain socket peer addresses are frequently unnamed (a client
    /// that didn't `bind()` before `connect()`, the common case), so
    /// there's no meaningful "avoid an extra syscall" win to chase here
    /// the way there is for TCP's always-populated IP/port.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `accept(2)`/`io_uring`
    /// accept completion reports one, or if registering the new stream
    /// with the driver fails.
    pub async fn accept(
        &self,
    ) -> std::io::Result<(DtactUnixStream, std::os::unix::net::SocketAddr)> {
        // 1. Direct opportunistic check using accept4 natively to avoid later fcntl
        let res = unsafe {
            libc::accept4(
                self.inner.as_raw_fd(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
            )
        };
        if res >= 0 {
            let stream = unsafe { std::os::unix::net::UnixStream::from_raw_fd(res) };
            let peer_addr = stream.peer_addr()?;
            let client_stream = DtactUnixStream::from_std(stream)?;
            return Ok((client_stream, peer_addr));
        }
        let err = std::io::Error::last_os_error();
        if err.kind() != std::io::ErrorKind::WouldBlock {
            return Err(err);
        }

        // 2. Async path: The driver's OpCode::Accept MUST pass SOCK_NONBLOCK | SOCK_CLOEXEC
        let res = DtactIoFuture {
            worker_idx: self.worker_idx,
            fd: self.inner.as_raw_fd() as u32,
            direct_fd_idx: self.direct_fd_idx,
            op: OpCode::Accept,
            buf_ptr: std::ptr::null_mut(),
            len: 0,
            offset: 0,
            addr: None,
            addr_len: 0,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        }
        .await?;

        let client_fd = res as RawFd;
        // Zero extra syscalls here. The fd is already non-blocking!
        let stream = unsafe { std::os::unix::net::UnixStream::from_raw_fd(client_fd) };
        let peer_addr = stream.peer_addr()?;
        let client_stream = DtactUnixStream::from_std(stream)?;
        Ok((client_stream, peer_addr))
    }
}

impl Drop for DtactUnixListener {
    fn drop(&mut self) {
        if let Some(workers) = WORKERS.get()
            && let Some(state) = workers.get(self.worker_idx)
        {
            unregister_fd_sync(state, self.direct_fd_idx);
        }
    }
}

/// Peer credentials (PID/UID/GID) of a Unix-domain-socket peer, as
/// reported by [`DtactUnixStream::peer_cred`].
///
/// `pid` is `None` on platforms whose peer-credential syscall doesn't
/// report one (anything but Linux — `getpeereid(2)` elsewhere only
/// yields uid/gid).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtactUCred {
    uid: u32,
    gid: u32,
    pid: Option<i32>,
}

impl DtactUCred {
    /// The peer's user ID.
    #[must_use]
    pub const fn uid(&self) -> u32 {
        self.uid
    }

    /// The peer's group ID.
    #[must_use]
    pub const fn gid(&self) -> u32 {
        self.gid
    }

    /// The peer's process ID, where the platform's peer-credential
    /// syscall reports one (Linux only — see this type's doc).
    #[must_use]
    pub const fn pid(&self) -> Option<i32> {
        self.pid
    }
}

#[cfg(target_os = "linux")]
fn peer_cred_impl(fd: RawFd) -> std::io::Result<DtactUCred> {
    let mut cred: libc::ucred = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    let r = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&raw mut cred).cast::<libc::c_void>(),
            &raw mut len,
        )
    };
    if r != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(DtactUCred {
        uid: cred.uid,
        gid: cred.gid,
        pid: Some(cred.pid),
    })
}

#[cfg(not(target_os = "linux"))]
fn peer_cred_impl(fd: RawFd) -> std::io::Result<DtactUCred> {
    let mut uid: libc::uid_t = 0;
    let mut gid: libc::gid_t = 0;
    let r = unsafe { libc::getpeereid(fd, &raw mut uid, &raw mut gid) };
    if r != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(DtactUCred {
        uid,
        gid,
        pid: None,
    })
}

/// Build a `libc::sockaddr_un` (returned inside a `sockaddr_storage`, like
/// every other address helper in this module) for `path`.
///
/// # Errors
///
/// Returns `InvalidInput` if `path` doesn't fit in `sockaddr_un::sun_path`
/// (108 bytes on Linux, shorter on macOS/BSD, including the NUL
/// terminator this function adds).
fn unix_path_to_libc(
    path: &std::path::Path,
) -> std::io::Result<(libc::sockaddr_storage, libc::socklen_t)> {
    use std::os::unix::ffi::OsStrExt;
    let bytes = path.as_os_str().as_bytes();
    let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    // SAFETY: `sockaddr_storage` is sized/aligned to hold any sockaddr
    // variant this platform supports, `sockaddr_un` included; casting its
    // address to `*mut sockaddr_un` and writing through it is exactly what
    // every other `*_to_libc` helper in this module does for its own
    // sockaddr variant.
    let sun_ptr = (&raw mut storage).cast::<libc::sockaddr_un>();
    let sun_path_cap = unsafe { (*sun_ptr).sun_path.len() };
    // Reserve one byte for the NUL terminator `sockaddr_un`'s `sun_path`
    // conventionally carries (matching `std::os::unix::net`'s own
    // behavior), so `bytes.len() == sun_path_cap` is still rejected.
    if bytes.len() >= sun_path_cap {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "unix socket path too long for sockaddr_un::sun_path",
        ));
    }
    unsafe {
        (*sun_ptr).sun_family = libc::AF_UNIX as libc::sa_family_t;
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            (*sun_ptr).sun_path.as_mut_ptr().cast::<u8>(),
            bytes.len(),
        );
    }
    let len = (std::mem::size_of::<libc::sa_family_t>() + bytes.len() + 1) as libc::socklen_t;
    Ok((storage, len))
}

/// A Unix-domain-socket peer address as reported by `recvfrom(2)`.
///
/// Either the filesystem path the peer `bind()`-ed to, or unnamed (a
/// socket that never called `bind()` before `sendto()`, the common
/// client-side case).
///
/// Not `std::os::unix::net::SocketAddr`: that type has no public
/// constructor from raw `sockaddr_un` bytes (only from a live syscall
/// result, e.g. `UnixListener::accept`'s own internal plumbing), and
/// `recvfrom`'s peer address has to come from the kernel-filled
/// `msghdr::msg_name` of the completed op — there's no separate syscall
/// this crate could use to fetch a `std`-constructed one instead the way
/// [`DtactUnixListener::accept`] does via `UnixStream::peer_addr`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DtactUnixSocketAddr(Option<std::path::PathBuf>);

impl DtactUnixSocketAddr {
    /// The path the peer was bound to, if any.
    #[must_use]
    pub fn as_pathname(&self) -> Option<&std::path::Path> {
        self.0.as_deref()
    }

    /// `true` if the peer never `bind()`-ed before sending (the common
    /// case for a datagram socket that only ever calls `send_to`).
    #[must_use]
    pub const fn is_unnamed(&self) -> bool {
        self.0.is_none()
    }
}

/// Parse a `recvfrom`-filled `sockaddr_un` (inside the generic
/// `sockaddr_storage` every address helper in this module uses) into a
/// [`DtactUnixSocketAddr`].
fn sockaddr_un_to_addr(
    storage: &libc::sockaddr_storage,
    len: libc::socklen_t,
) -> DtactUnixSocketAddr {
    let family_len = std::mem::size_of::<libc::sa_family_t>();
    if (len as usize) <= family_len {
        return DtactUnixSocketAddr(None); // unnamed: no path bytes at all
    }
    // SAFETY: `storage` is sized/aligned to hold any sockaddr variant,
    // `sockaddr_un` included, and `len` (from the completed `recvfrom`)
    // bounds how much of it the kernel actually filled in.
    let sun = unsafe { &*std::ptr::from_ref(storage).cast::<libc::sockaddr_un>() };
    let path_len = (len as usize) - family_len;
    let path_len = path_len.min(sun.sun_path.len());
    // SAFETY: `path_len` was just clamped to `sun_path`'s own length.
    let bytes = unsafe { std::slice::from_raw_parts(sun.sun_path.as_ptr().cast::<u8>(), path_len) };
    // `sun_path` is conventionally NUL-terminated for a pathname address;
    // trim at the first NUL rather than trusting `path_len` to already
    // exclude it.
    let bytes = bytes.split(|&b| b == 0).next().unwrap_or(&[]);
    if bytes.is_empty() {
        DtactUnixSocketAddr(None)
    } else {
        use std::os::unix::ffi::OsStrExt;
        DtactUnixSocketAddr(Some(std::path::PathBuf::from(std::ffi::OsStr::from_bytes(
            bytes,
        ))))
    }
}

// =========================================================================
// 10d. HIGH-LEVEL API: DtactUnixDatagram
// =========================================================================

/// Async Unix-domain datagram socket.
///
/// Connectionless counterpart to [`DtactUnixStream`], directly analogous
/// to [`DtactUdpSocket`] (same connectionless `send_to`/`recv_from` and
/// connected `connect`/`send`/`recv` pattern, same `SendTo`/`RecvFrom`/
/// `Read`/`Write` submission machinery — only the address family and the
/// `libc::sockaddr_un` construction differ).
pub struct DtactUnixDatagram {
    inner: std::os::unix::net::UnixDatagram,
    direct_fd_idx: u32,
    worker_idx: usize,
    read_backpressured: std::sync::atomic::AtomicBool,
    write_backpressured: std::sync::atomic::AtomicBool,
}

impl DtactUnixDatagram {
    /// Bind a new datagram socket to the filesystem path `path` and
    /// register it with the driver. `path` must not already exist —
    /// like `std::os::unix::net::UnixDatagram::bind`, this does not
    /// remove a stale socket file left behind by a previous run.
    ///
    /// # Errors
    /// Returns any error from binding the OS socket or registering it.
    pub fn bind(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let sock = std::os::unix::net::UnixDatagram::bind(path)?;
        Self::from_std(sock)
    }

    /// Create an unbound datagram socket (matches
    /// `std::os::unix::net::UnixDatagram::unbound`) — usable for
    /// `send_to`/`connect` immediately, has no path of its own until (if
    /// ever) explicitly bound.
    ///
    /// # Errors
    /// Returns any error from creating the OS socket or registering it.
    pub fn unbound() -> std::io::Result<Self> {
        let sock = std::os::unix::net::UnixDatagram::unbound()?;
        Self::from_std(sock)
    }

    /// Register an existing `std::os::unix::net::UnixDatagram`, taking
    /// ownership.
    ///
    /// # Errors
    /// Returns any error from switching the socket to non-blocking mode
    /// or registering it with the driver.
    ///
    /// # Panics
    /// Panics if called before [`init_runtime`]/[`init`] has been called
    /// (`WORKERS` not yet initialized).
    pub fn from_std(socket: std::os::unix::net::UnixDatagram) -> std::io::Result<Self> {
        let fd = socket.as_raw_fd();
        socket.set_nonblocking(true)?;
        let num_workers = GLOBAL_CONFIG.get().map_or(1, |c| c.workers);
        let worker_idx = fd as usize % num_workers;
        let state = &WORKERS.get().unwrap()[worker_idx];
        let direct_fd_idx = register_fd_sync(state, fd);
        Ok(Self {
            inner: socket,
            direct_fd_idx,
            worker_idx,
            read_backpressured: std::sync::atomic::AtomicBool::new(false),
            write_backpressured: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Send `buf` as a single datagram to the socket bound at `target`,
    /// returning the number of bytes sent.
    ///
    /// # Errors
    /// Returns any error from the underlying `sendmsg`, `target` path
    /// resolution included (e.g. too long for `sockaddr_un::sun_path`).
    pub async fn send_to(
        &self,
        buf: &[u8],
        target: impl AsRef<std::path::Path>,
    ) -> std::io::Result<usize> {
        struct SendToState {
            storage: libc::sockaddr_storage,
            iov: libc::iovec,
            msg: libc::msghdr,
        }
        unsafe impl Send for SendToState {}

        if !self.write_backpressured.load(Ordering::Relaxed) {
            let (storage, addr_len) = unix_path_to_libc(target.as_ref())?;
            let mut state = SendToState {
                storage,
                iov: libc::iovec {
                    iov_base: buf.as_ptr().cast_mut().cast::<libc::c_void>(),
                    iov_len: buf.len(),
                },
                msg: unsafe { std::mem::zeroed() },
            };
            state.msg.msg_name = std::ptr::addr_of_mut!(state.storage).cast::<libc::c_void>();
            state.msg.msg_namelen = addr_len;
            state.msg.msg_iov = &raw mut state.iov;
            state.msg.msg_iovlen = 1;

            let r = unsafe { libc::sendmsg(self.inner.as_raw_fd(), &raw const state.msg, 0) };
            if r >= 0 {
                return Ok(r as usize);
            }

            let e = std::io::Error::last_os_error();
            if e.kind() != std::io::ErrorKind::WouldBlock {
                return Err(e);
            }
            self.write_backpressured.store(true, Ordering::Relaxed);
        }

        // Re-generate the state layout if falling back to the driver ring
        let (storage, addr_len) = unix_path_to_libc(target.as_ref())?;
        let mut state = SendToState {
            storage,
            iov: libc::iovec {
                iov_base: buf.as_ptr().cast_mut().cast::<libc::c_void>(),
                iov_len: buf.len(),
            },
            msg: unsafe { std::mem::zeroed() },
        };
        state.msg.msg_name = std::ptr::addr_of_mut!(state.storage).cast::<libc::c_void>();
        state.msg.msg_namelen = addr_len;
        state.msg.msg_iov = &raw mut state.iov;
        state.msg.msg_iovlen = 1;

        let mut fut = DtactIoFuture::new(
            self.worker_idx,
            self.inner.as_raw_fd() as u32,
            self.direct_fd_idx,
            OpCode::SendTo,
            std::ptr::null_mut(),
            0,
            0,
            None,
            0,
            None,
        );
        fut.msg_ptr = &raw mut state.msg;
        let res = fut.await;
        self.write_backpressured.store(false, Ordering::Relaxed);
        res
    }

    /// Receive a single datagram into `buf`, returning the byte count and
    /// the peer address it came from.
    ///
    /// # Errors
    /// Returns any error from the underlying `recvmsg`.
    pub async fn recv_from(&self, buf: &mut [u8]) -> std::io::Result<(usize, DtactUnixSocketAddr)> {
        struct RecvFromState {
            storage: libc::sockaddr_storage,
            iov: libc::iovec,
            msg: libc::msghdr,
        }
        unsafe impl Send for RecvFromState {}

        if !self.read_backpressured.load(Ordering::Relaxed) {
            let mut state = RecvFromState {
                storage: unsafe { std::mem::zeroed() },
                iov: libc::iovec {
                    iov_base: buf.as_mut_ptr().cast::<libc::c_void>(),
                    iov_len: buf.len(),
                },
                msg: unsafe { std::mem::zeroed() },
            };
            state.msg.msg_name = std::ptr::addr_of_mut!(state.storage).cast::<libc::c_void>();
            state.msg.msg_namelen =
                std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
            state.msg.msg_iov = &raw mut state.iov;
            state.msg.msg_iovlen = 1;

            let r = unsafe { libc::recvmsg(self.inner.as_raw_fd(), &raw mut state.msg, 0) };
            if r >= 0 {
                let from = sockaddr_un_to_addr(&state.storage, state.msg.msg_namelen);
                return Ok((r as usize, from));
            }
            let e = std::io::Error::last_os_error();
            if e.kind() != std::io::ErrorKind::WouldBlock {
                return Err(e);
            }
            self.read_backpressured.store(true, Ordering::Relaxed);
        }

        let mut state = RecvFromState {
            storage: unsafe { std::mem::zeroed() },
            iov: libc::iovec {
                iov_base: buf.as_mut_ptr().cast::<libc::c_void>(),
                iov_len: buf.len(),
            },
            msg: unsafe { std::mem::zeroed() },
        };
        state.msg.msg_name = std::ptr::addr_of_mut!(state.storage).cast::<libc::c_void>();
        state.msg.msg_namelen = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
        state.msg.msg_iov = &raw mut state.iov;
        state.msg.msg_iovlen = 1;

        let mut fut = DtactIoFuture::new(
            self.worker_idx,
            self.inner.as_raw_fd() as u32,
            self.direct_fd_idx,
            OpCode::RecvFrom,
            std::ptr::null_mut(),
            0,
            0,
            None,
            0,
            None,
        );
        fut.msg_ptr = &raw mut state.msg;
        let n = fut.await?;
        self.read_backpressured.store(false, Ordering::Relaxed);
        let from = sockaddr_un_to_addr(&state.storage, state.msg.msg_namelen);
        Ok((n, from))
    }

    /// Connect this socket to the path `target` so
    /// [`send`](Self::send)/[`recv`](Self::recv) can omit the peer
    /// address.
    ///
    /// # Errors
    /// Returns any error from the underlying `connect`.
    pub async fn connect(&self, target: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.inner.connect(target)
    }

    /// Send `buf` to the connected peer, returning the number of bytes
    /// sent.
    ///
    /// # Errors
    /// Returns any error from the underlying send.
    pub async fn send(&self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let r = unsafe {
            libc::send(
                self.inner.as_raw_fd(),
                buf.as_ptr().cast::<libc::c_void>(),
                buf.len(),
                0,
            )
        };
        if r >= 0 {
            return Ok(r as usize);
        }
        let e = std::io::Error::last_os_error();
        if e.kind() != std::io::ErrorKind::WouldBlock {
            return Err(e);
        }
        DtactIoFuture {
            worker_idx: self.worker_idx,
            fd: self.inner.as_raw_fd() as u32,
            direct_fd_idx: self.direct_fd_idx,
            op: OpCode::Write,
            buf_ptr: buf.as_ptr().cast_mut(),
            len: buf.len(),
            offset: 0,
            addr: None,
            addr_len: 0,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        }
        .await
    }

    /// Receive a datagram from the connected peer into `buf`, returning
    /// the byte count.
    ///
    /// # Errors
    /// Returns any error from the underlying recv.
    pub async fn recv(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let r = unsafe {
            libc::recv(
                self.inner.as_raw_fd(),
                buf.as_mut_ptr().cast::<libc::c_void>(),
                buf.len(),
                0,
            )
        };
        if r >= 0 {
            return Ok(r as usize);
        }
        let e = std::io::Error::last_os_error();
        if e.kind() != std::io::ErrorKind::WouldBlock {
            return Err(e);
        }
        DtactIoFuture {
            worker_idx: self.worker_idx,
            fd: self.inner.as_raw_fd() as u32,
            direct_fd_idx: self.direct_fd_idx,
            op: OpCode::Read,
            buf_ptr: buf.as_mut_ptr(),
            len: buf.len(),
            offset: 0,
            addr: None,
            addr_len: 0,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        }
        .await
    }
}

impl Drop for DtactUnixDatagram {
    fn drop(&mut self) {
        if let Some(workers) = WORKERS.get()
            && let Some(state) = workers.get(self.worker_idx)
        {
            unregister_fd_sync(state, self.direct_fd_idx);
        }
    }
}

// =========================================================================
// 10b. FIFO (named-pipe) read/write ends — the Unix counterpart to
// `named_pipe_windows`'s server/client handles.
// =========================================================================
// Does not create the FIFO itself (`mkfifo(2)` is out of scope, matching
// `tokio::net::unix::pipe`'s own scope — it only opens an already-`mkfifo`'d
// path); create the FIFO externally (the `mkfifo` shell command, or
// `libc::mkfifo`) before opening either end here. Both ends reuse the
// exact same reactor registration (`register_fd_sync`) and `DtactIoFuture`
// read/write path as `DtactUnixStream` above — a FIFO fd is just as
// poll/io_uring-able as a socket fd once opened non-blocking.

/// The read end of a Unix FIFO. Open with [`open_fifo_read`].
pub struct DtactFifoReader {
    inner: std::fs::File,
    direct_fd_idx: u32,
    worker_idx: usize,
    backpressured: std::sync::atomic::AtomicBool,
}

unsafe impl Send for DtactFifoReader {}
unsafe impl Sync for DtactFifoReader {}

impl DtactFifoReader {
    /// Read into `buf`, returning `0` at EOF (every writer end closed).
    ///
    /// # Errors
    /// Returns an `io::Error` if the underlying read reports one.
    pub async fn read(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if !self.backpressured.load(Ordering::Relaxed) {
            let res = unsafe {
                let r = libc::read(
                    self.inner.as_raw_fd(),
                    buf.as_mut_ptr().cast::<libc::c_void>(),
                    buf.len(),
                );
                match r.cmp(&0) {
                    std::cmp::Ordering::Greater => Ok(r as usize),
                    std::cmp::Ordering::Equal => Ok(0),
                    std::cmp::Ordering::Less => Err(std::io::Error::last_os_error()),
                }
            };
            match res {
                Ok(n) => return Ok(n),
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::WouldBlock {
                        return Err(e);
                    }
                }
            }
            self.backpressured.store(true, Ordering::Relaxed);
        }
        let future = DtactIoFuture {
            worker_idx: self.worker_idx,
            fd: self.inner.as_raw_fd() as u32,
            direct_fd_idx: self.direct_fd_idx,
            op: OpCode::Read,
            buf_ptr: buf.as_mut_ptr(),
            len: buf.len(),
            offset: 0,
            addr: None,
            addr_len: 0,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        }
        .await;
        self.backpressured.store(false, Ordering::Relaxed);
        future.map(|n| n.min(buf.len()))
    }
}

impl Drop for DtactFifoReader {
    fn drop(&mut self) {
        if let Some(workers) = WORKERS.get()
            && let Some(state) = workers.get(self.worker_idx)
        {
            unregister_fd_sync(state, self.direct_fd_idx);
        }
    }
}

/// The write end of a Unix FIFO. Open with [`open_fifo_write`].
pub struct DtactFifoWriter {
    inner: std::fs::File,
    direct_fd_idx: u32,
    worker_idx: usize,
    backpressured: std::sync::atomic::AtomicBool,
}

unsafe impl Send for DtactFifoWriter {}
unsafe impl Sync for DtactFifoWriter {}

impl DtactFifoWriter {
    /// Write `buf`, returning the byte count written.
    ///
    /// # Errors
    /// Returns an `io::Error` if the underlying write reports one (e.g.
    /// `BrokenPipe` once every reader end has closed).
    pub async fn write(&self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if !self.backpressured.load(Ordering::Relaxed) {
            let res = unsafe {
                let r = libc::write(
                    self.inner.as_raw_fd(),
                    buf.as_ptr().cast::<libc::c_void>(),
                    buf.len(),
                );
                if r >= 0 {
                    Ok(r as usize)
                } else {
                    Err(std::io::Error::last_os_error())
                }
            };
            match res {
                Ok(n) => return Ok(n),
                Err(e) => {
                    if e.kind() != std::io::ErrorKind::WouldBlock {
                        return Err(e);
                    }
                }
            }
            self.backpressured.store(true, Ordering::Relaxed);
        }
        let future = DtactIoFuture {
            worker_idx: self.worker_idx,
            fd: self.inner.as_raw_fd() as u32,
            direct_fd_idx: self.direct_fd_idx,
            op: OpCode::Write,
            buf_ptr: buf.as_ptr().cast_mut(),
            len: buf.len(),
            offset: 0,
            addr: None,
            addr_len: 0,
            slot_idx: None,
            msg_ptr: std::ptr::null_mut(),
        }
        .await;
        self.backpressured.store(false, Ordering::Relaxed);
        future.map(|n| n.min(buf.len()))
    }
}

impl Drop for DtactFifoWriter {
    fn drop(&mut self) {
        if let Some(workers) = WORKERS.get()
            && let Some(state) = workers.get(self.worker_idx)
        {
            unregister_fd_sync(state, self.direct_fd_idx);
        }
    }
}

fn register_fifo_fd(fd: RawFd) -> (u32, usize) {
    let num_workers = GLOBAL_CONFIG.get().map_or(1, |c| c.workers);
    let worker_idx = fd as usize % num_workers;
    let state = &WORKERS.get().unwrap()[worker_idx];
    (register_fd_sync(state, fd), worker_idx)
}

/// Open the read end of the FIFO at `path` (which must already exist —
/// see the module-doc note above on why this doesn't `mkfifo` it).
///
/// Non-blocking: unlike a blocking `open(2)` on a FIFO's read end (which
/// waits for a writer), this returns immediately regardless of whether a
/// writer is currently open, matching `tokio::net::unix::pipe::OpenOptions`.
///
/// # Errors
/// Returns an `io::Error` if `open(2)` fails (e.g. `path` doesn't exist
/// or isn't a FIFO) or if registering the fd with the reactor fails.
///
/// # Panics
/// Panics if called before [`init_runtime`]/[`init`] has been called.
pub async fn open_fifo_read(
    path: impl Into<std::path::PathBuf>,
) -> std::io::Result<DtactFifoReader> {
    use std::os::unix::fs::OpenOptionsExt;
    let path = path.into();
    // `O_NONBLOCK` makes `open(2)` itself non-blocking for a FIFO's read
    // end per POSIX semantics (it returns immediately regardless of
    // whether a writer is open), so there's no actual blocking syscall
    // here to hand off to a blocking-pool thread the way `fs::DtactFile`
    // needs to for ordinary file I/O.
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&path)?;
    let (direct_fd_idx, worker_idx) = register_fifo_fd(file.as_raw_fd());
    Ok(DtactFifoReader {
        inner: file,
        direct_fd_idx,
        worker_idx,
        backpressured: AtomicBool::new(false),
    })
}

/// Open the write end of the FIFO at `path` (which must already exist,
/// and — per POSIX FIFO semantics — already have at least one reader end
/// open, or this fails with `ENXIO` rather than blocking).
///
/// # Errors
/// Returns an `io::Error` if `open(2)` fails (`ENXIO` with no reader
/// present is the common case, not a driver bug) or if registering the
/// fd with the reactor fails.
///
/// # Panics
/// Panics if called before [`init_runtime`]/[`init`] has been called.
pub async fn open_fifo_write(
    path: impl Into<std::path::PathBuf>,
) -> std::io::Result<DtactFifoWriter> {
    use std::os::unix::fs::OpenOptionsExt;
    let path = path.into();
    // See `open_fifo_read`'s comment on why `O_NONBLOCK` means this
    // doesn't need a blocking-pool hand-off either.
    let file = std::fs::OpenOptions::new()
        .write(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&path)?;
    let (direct_fd_idx, worker_idx) = register_fifo_fd(file.as_raw_fd());
    Ok(DtactFifoWriter {
        inner: file,
        direct_fd_idx,
        worker_idx,
        backpressured: AtomicBool::new(false),
    })
}

// =========================================================================
// 11. FILE-REGISTRATION HELPERS
// =========================================================================

/// Register `fd` with the dtact-io driver.
///
/// We intentionally skip `io_uring` fixed-file registration here.
/// `register_files_update` (`io_uring_register`) returns EBUSY under SQPOLL
/// when called concurrently with the io worker's submit/wait loop, and
/// serialising it with a mutex would either deadlock (if called from inside
/// a fiber) or severely harm throughput.  Fixed files provide only ~5%
/// throughput gain; correctness takes priority.
///
/// `u32::MAX` is the sentinel the io-path already uses for "raw fd" mode.
const fn register_fd_sync(_state: &WorkerState, _fd: RawFd) -> u32 {
    u32::MAX
}

/// Nothing to release when we aren't using fixed files.
const fn unregister_fd_sync(_state: &WorkerState, _direct_fd_idx: u32) {}

// =========================================================================
// 12. HELPER CONVERTER FUNCTIONS
// =========================================================================
const fn socket_addr_to_libc(
    addr: std::net::SocketAddr,
) -> (libc::sockaddr_storage, libc::socklen_t) {
    let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let len = match addr {
        std::net::SocketAddr::V4(a) => {
            let sin = libc::sockaddr_in {
                sin_family: libc::AF_INET as libc::sa_family_t,
                sin_port: a.port().to_be(),
                sin_addr: libc::in_addr {
                    s_addr: u32::from_ne_bytes(a.ip().octets()),
                },
                sin_zero: [0; 8],
            };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&raw const sin).cast::<u8>(),
                    (&raw mut storage).cast::<u8>(),
                    std::mem::size_of::<libc::sockaddr_in>(),
                );
            }
            std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t
        }
        std::net::SocketAddr::V6(a) => {
            let sin6 = libc::sockaddr_in6 {
                sin6_family: libc::AF_INET6 as libc::sa_family_t,
                sin6_port: a.port().to_be(),
                sin6_flowinfo: a.flowinfo(),
                sin6_addr: libc::in6_addr {
                    s6_addr: a.ip().octets(),
                },
                sin6_scope_id: a.scope_id(),
            };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&raw const sin6).cast::<u8>(),
                    (&raw mut storage).cast::<u8>(),
                    std::mem::size_of::<libc::sockaddr_in6>(),
                );
            }
            std::mem::size_of::<libc::sockaddr_in6>() as libc::socklen_t
        }
    };
    (storage, len)
}

/// Parse a `libc::sockaddr_storage` (returned by `libc::accept`) into a
/// `std::net::SocketAddr` without issuing an extra `getpeername` syscall.
fn sockaddr_storage_to_socketaddr(
    storage: &libc::sockaddr_storage,
    _len: libc::socklen_t,
) -> std::net::SocketAddr {
    match libc::c_int::from(storage.ss_family) {
        libc::AF_INET => {
            // Safety: ss_family confirmed to be AF_INET.
            let sin = unsafe { &*std::ptr::from_ref(storage).cast::<libc::sockaddr_in>() };
            let ip = std::net::Ipv4Addr::from(u32::from_be(sin.sin_addr.s_addr));
            let port = u16::from_be(sin.sin_port);
            std::net::SocketAddr::V4(std::net::SocketAddrV4::new(ip, port))
        }
        libc::AF_INET6 => {
            // Safety: ss_family confirmed to be AF_INET6.
            let sin6 = unsafe { &*std::ptr::from_ref(storage).cast::<libc::sockaddr_in6>() };
            let ip = std::net::Ipv6Addr::from(sin6.sin6_addr.s6_addr);
            let port = u16::from_be(sin6.sin6_port);
            std::net::SocketAddr::V6(std::net::SocketAddrV6::new(
                ip,
                port,
                sin6.sin6_flowinfo,
                sin6.sin6_scope_id,
            ))
        }
        _ => {
            panic!("Unsupported address family: {}", storage.ss_family);
        }
    }
}

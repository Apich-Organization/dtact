//! Windows native named pipes: `DtactNamedPipeServer`/`DtactNamedPipeClient`
//! — the Windows IPC counterpart to [`super::DtactUnixStream`] on Unix
//! (see that type's doc for why there's no cross-platform unification:
//! named pipes and Unix domain sockets are different enough OS primitives
//! that a shared abstraction would either leak platform details or lose
//! capability on one side).
//!
//! Uses real overlapped `ReadFile`/`WriteFile`/`ConnectNamedPipe` against
//! a dedicated IOCP — the same "one completion port + one worker thread +
//! preallocated `OpState` slot pool" shape as `fs::iocp_windows`, copied
//! rather than shared with it: named pipe handles behave like file
//! handles for I/O purposes (this module reuses the exact `ReadFile`/
//! `WriteFile`+`OVERLAPPED` pattern), but a *separate* port/worker/pool
//! keeps this module self-contained and matches `io::windows`'s own
//! socket IOCP already being separate from `fs::iocp_windows`'s file
//! IOCP — a third independent one for pipes is consistent with that
//! existing split, not a new pattern.
//!
//! **No pipe-instance pooling / listener abstraction.** Each
//! [`DtactNamedPipeServer::create`] call creates exactly one pipe
//! instance; accepting N concurrent clients means calling `create` N
//! times (typically in a loop, creating the next instance right after
//! the previous one connects) — this matches
//! `tokio::net::windows::named_pipe::ServerOptions::create`'s own
//! per-instance semantics exactly, rather than inventing a
//! `DtactTcpListener`-style persistent listener Windows named pipes don't
//! naturally have.

use super::windows::GLOBAL_CONFIG;
use crate::lockfree::{AtomicWakerSlot, TreiberStack};
use std::ffi::c_void;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, Ordering};
use std::task::{Context, Poll};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_IO_PENDING, ERROR_PIPE_BUSY, ERROR_PIPE_CONNECTED, GENERIC_READ,
    GENERIC_WRITE, GetLastError, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, FILE_FLAG_OVERLAPPED, OPEN_EXISTING, PIPE_ACCESS_DUPLEX, ReadFile, WriteFile,
};
use windows_sys::Win32::System::IO::{
    CreateIoCompletionPort, GetQueuedCompletionStatusEx, OVERLAPPED, OVERLAPPED_ENTRY,
};
use windows_sys::Win32::System::Pipes::{
    ConnectNamedPipe, CreateNamedPipeW, PIPE_READMODE_BYTE, PIPE_TYPE_BYTE,
    PIPE_UNLIMITED_INSTANCES, PIPE_WAIT, WaitNamedPipeW,
};

const WAKE_KEY: usize = 0;
const PIPE_KEY: usize = 1;
const PENDING: i64 = i64::MIN;

#[repr(align(64))]
struct Port {
    handle: HANDLE,
}
unsafe impl Send for Port {}
unsafe impl Sync for Port {}

static PORT: OnceLock<Port> = OnceLock::new();

fn port() -> HANDLE {
    let p = PORT.get_or_init(|| {
        let handle = unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, ptr::null_mut(), 0, 1) };
        assert!(
            !handle.is_null(),
            "dtact-io: named-pipe CreateIoCompletionPort failed"
        );
        std::thread::Builder::new()
            .name("dtact-io-namedpipe-iocp".into())
            .spawn(worker_loop)
            .expect("failed to spawn dtact-io named-pipe IOCP worker thread");
        Port { handle }
    });
    p.handle
}

#[repr(C)]
struct OpState {
    overlapped: OVERLAPPED,
    result: AtomicI64,
    waker: AtomicWakerSlot,
}

impl OpState {
    const fn fresh() -> Self {
        Self {
            overlapped: unsafe { std::mem::zeroed() },
            result: AtomicI64::new(PENDING),
            waker: AtomicWakerSlot::new(),
        }
    }
}

// SAFETY: same reasoning as `fs::iocp_windows::OpState` — `overlapped` is
// opaque kernel-visible scratch memory written once at submission and
// otherwise only touched by the OS/IOCP worker; `result`/`waker` are
// already atomics.
unsafe impl Send for OpState {}
unsafe impl Sync for OpState {}

#[allow(dead_code)]
#[repr(align(64))]
struct SlotPool {
    slots: Box<[OpState]>,
    free: TreiberStack,
}

static RING_DEPTH: OnceLock<usize> = OnceLock::new();
static SLOT_POOL: OnceLock<SlotPool> = OnceLock::new();

fn slot_pool() -> &'static SlotPool {
    SLOT_POOL.get_or_init(|| {
        let depth = *RING_DEPTH.get_or_init(|| 256);
        let mut slots = Vec::with_capacity(depth);
        for _ in 0..depth {
            slots.push(OpState::fresh());
        }
        let free = TreiberStack::new(depth);
        for i in 0..depth as u32 {
            free.push(i);
        }
        SlotPool {
            slots: slots.into_boxed_slice(),
            free,
        }
    })
}

/// Start the named-pipe IOCP subsystem. Idempotent — later calls are
/// no-ops. `ring_depth` sizes the preallocated op-slot pool (see the
/// module doc's slot-pool paragraph); called automatically (with a
/// default depth) by every constructor below, so most callers never need
/// to call this directly.
pub fn init(ring_depth: u32) {
    let _ = RING_DEPTH.set(ring_depth.max(1) as usize);
    let _ = slot_pool();
    let _ = port();
}

fn ensure_init() {
    if RING_DEPTH.get().is_none() {
        init(256);
    }
}

const fn encode_ok(n: usize) -> i64 {
    n as i64
}

const fn encode_err(win32_code: u32) -> i64 {
    -(win32_code as i64)
}

fn decode(result: i64) -> io::Result<usize> {
    if result >= 0 {
        Ok(result as usize)
    } else {
        Err(io::Error::from_raw_os_error(-result as i32))
    }
}

fn worker_loop() {
    let iocp = port();
    let mut entries: [OVERLAPPED_ENTRY; 64] = unsafe { std::mem::zeroed() };
    loop {
        let mut removed: u32 = 0;
        let ok = unsafe {
            GetQueuedCompletionStatusEx(
                iocp,
                entries.as_mut_ptr(),
                entries.len() as u32,
                &raw mut removed,
                u32::MAX,
                0,
            )
        };
        if ok == 0 {
            continue;
        }
        for entry in &entries[..removed as usize] {
            if entry.lpCompletionKey == WAKE_KEY {
                continue;
            }
            // `lpOverlapped` is the very `overlapped` field this op was
            // submitted with; `overlapped` is `OpState`'s first field, so
            // the round-trip back to `OpState` is exact. `OpState` is
            // 64-byte aligned (pulled up by `AtomicWakerSlot`), which is
            // *stricter* than the 8-byte `OVERLAPPED` clippy sees — but
            // every `OpState` in this module lives in a `Box<[OpState]>`
            // allocation, and `Box`/`Vec` always align to `align_of::<T>()`,
            // so the cast is sound. Clippy cannot see that invariant.
            #[allow(clippy::cast_ptr_alignment)]
            let op_ptr = entry.lpOverlapped.cast::<OpState>();
            if op_ptr.is_null() {
                continue;
            }
            debug_assert_eq!(
                (op_ptr as usize) % core::mem::align_of::<OpState>(),
                0,
                "IOCP returned an OpState pointer that is not 64-byte aligned"
            );
            let op = unsafe { &*op_ptr };
            let bytes = entry.dwNumberOfBytesTransferred as usize;
            op.result.store(encode_ok(bytes), Ordering::Release);
            op.waker.take_and_wake();
        }
    }
}

#[repr(align(64))]
enum Slot {
    Pooled { shard_idx: usize, slot_idx: u32 },
    Heap(Box<OpState>),
}

#[repr(align(64))]
struct Shard {
    slots: Box<[OpState]>,
    free: TreiberStack,
}

#[repr(align(64))]
struct ShardedSlotPool {
    shards: Box<[Shard]>,
}

static SHARDED_POOL: OnceLock<ShardedSlotPool> = OnceLock::new();

fn init_sharded_pool() -> &'static ShardedSlotPool {
    SHARDED_POOL.get_or_init(|| {
        let num_workers = GLOBAL_CONFIG.get().map_or(1, |c| c.workers).max(1);
        let depth_per_shard = *RING_DEPTH.get_or_init(|| 256);

        let mut shards = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let mut slots = Vec::with_capacity(depth_per_shard);
            for _ in 0..depth_per_shard {
                slots.push(OpState::fresh());
            }
            let free = TreiberStack::new(depth_per_shard);
            for i in 0..depth_per_shard as u32 {
                free.push(i);
            }
            shards.push(Shard {
                slots: slots.into_boxed_slice(),
                free,
            });
        }

        ShardedSlotPool {
            shards: shards.into_boxed_slice(),
        }
    })
}

fn get_local_shard_idx(num_shards: usize) -> usize {
    thread_local! {
        // Lazily initialized once per thread execution lifetime
        static LAZY_SHARD_IDX: usize = {
            static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            COUNTER.fetch_add(1, Ordering::Relaxed)
        };
    }
    LAZY_SHARD_IDX.with(|&idx| idx % num_shards)
}

fn acquire_slot() -> Slot {
    let pool = init_sharded_pool();
    let num_shards = pool.shards.len();

    // Thread-safe sampling at checkout time
    let shard_idx = get_local_shard_idx(num_shards);
    let shard = &pool.shards[shard_idx];

    shard.free.pop().map_or_else(
        || Slot::Heap(Box::new(OpState::fresh())),
        |idx| {
            shard.slots[idx as usize]
                .result
                .store(PENDING, Ordering::Relaxed);

            Slot::Pooled {
                shard_idx,
                slot_idx: idx,
            }
        },
    )
}

#[repr(align(64))]
struct IoOp {
    slot: Slot,
}

impl IoOp {
    #[inline]
    fn state(&self) -> &OpState {
        match &self.slot {
            Slot::Pooled {
                shard_idx,
                slot_idx,
            } => &init_sharded_pool().shards[*shard_idx].slots[*slot_idx as usize],
            Slot::Heap(b) => b,
        }
    }

    #[inline]
    fn overlapped_ptr(&self) -> *mut OVERLAPPED {
        match &self.slot {
            Slot::Pooled {
                shard_idx,
                slot_idx,
            } => {
                let state_ref = &init_sharded_pool().shards[*shard_idx].slots[*slot_idx as usize];
                std::ptr::from_ref::<OpState>(state_ref).cast_mut().cast()
            }
            Slot::Heap(b) => std::ptr::from_ref::<OpState>(b.as_ref()).cast_mut().cast(),
        }
    }
}

impl Future for IoOp {
    type Output = io::Result<usize>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<usize>> {
        let r = self.state().result.load(Ordering::Acquire);
        if r != PENDING {
            return Poll::Ready(decode(r));
        }
        self.state().waker.register(cx.waker());
        let r = self.state().result.load(Ordering::Acquire);
        if r != PENDING {
            return Poll::Ready(decode(r));
        }
        Poll::Pending
    }
}

impl Drop for IoOp {
    fn drop(&mut self) {
        if let Slot::Pooled {
            shard_idx,
            slot_idx,
        } = self.slot
        {
            let pool = init_sharded_pool();
            let shard = &pool.shards[shard_idx];
            let done = shard.slots[slot_idx as usize]
                .result
                .load(Ordering::Acquire)
                != PENDING;
            if done {
                shard.free.push(slot_idx);
            }
            // If the kernel operation is still running/stalled on the port,
            // the slot leaks safely to prevent cross-stack pointer corruption.
        }
    }
}

#[repr(align(64))]
enum IoOpResult {
    Ready(io::Result<usize>),
    Pending(IoOp),
}

fn issue_read(handle: HANDLE, buf: &mut [u8]) -> IoOpResult {
    let slot = acquire_slot();
    let op = IoOp { slot };
    let ov_ptr = op.overlapped_ptr();

    let mut bytes_transferred: u32 = 0;
    let ok = unsafe {
        ReadFile(
            handle,
            buf.as_mut_ptr(),
            buf.len() as u32,
            &raw mut bytes_transferred,
            ov_ptr,
        )
    };

    if ok != 0 {
        // Fast path: completed immediately synchronous!
        // We set the result state so Drop doesn't track it as leaked/stale,
        // then immediately return the payload size.
        op.state()
            .result
            .store(encode_ok(bytes_transferred as usize), Ordering::Relaxed);
        return IoOpResult::Ready(Ok(bytes_transferred as usize));
    }

    let err = unsafe { GetLastError() };
    if err == ERROR_IO_PENDING {
        IoOpResult::Pending(op)
    } else {
        // Immediate failure path
        op.state().result.store(encode_err(err), Ordering::Relaxed);
        IoOpResult::Ready(Err(io::Error::from_raw_os_error(err as i32)))
    }
}

fn issue_write(handle: HANDLE, buf: &[u8]) -> IoOpResult {
    let slot = acquire_slot();
    let op = IoOp { slot };
    let ov_ptr = op.overlapped_ptr();

    let mut bytes_transferred: u32 = 0;
    let ok = unsafe {
        WriteFile(
            handle,
            buf.as_ptr(),
            buf.len() as u32,
            &raw mut bytes_transferred,
            ov_ptr,
        )
    };

    if ok != 0 {
        // Fast path: written immediately without pausing task execution
        op.state()
            .result
            .store(encode_ok(bytes_transferred as usize), Ordering::Relaxed);
        return IoOpResult::Ready(Ok(bytes_transferred as usize));
    }

    let err = unsafe { GetLastError() };
    if err == ERROR_IO_PENDING {
        IoOpResult::Pending(op)
    } else {
        op.state().result.store(encode_err(err), Ordering::Relaxed);
        IoOpResult::Ready(Err(io::Error::from_raw_os_error(err as i32)))
    }
}

/// Submit `ConnectNamedPipe` (wait for a client to connect to a
/// just-created server instance) as an overlapped op through the same
/// IOCP/slot-pool machinery as reads/writes.
fn issue_connect(handle: HANDLE) -> IoOp {
    let op = IoOp {
        slot: acquire_slot(),
    };
    let ov_ptr = op.overlapped_ptr();
    let ok = unsafe { ConnectNamedPipe(handle, ov_ptr) };
    if ok == 0 {
        let err = unsafe { GetLastError() };
        // A client that connected in the (tiny) window between
        // `CreateNamedPipeW` and this call is reported as
        // `ERROR_PIPE_CONNECTED`, not `ERROR_IO_PENDING` — that's success,
        // not a real error, and no completion packet will ever arrive for
        // it since the op never actually went "pending" at the kernel
        // level.
        if err == ERROR_PIPE_CONNECTED {
            op.state().result.store(encode_ok(0), Ordering::Release);
        } else if err != ERROR_IO_PENDING {
            op.state().result.store(encode_err(err), Ordering::Release);
        }
    }
    op
}

fn to_wide(s: &str) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    std::ffi::OsStr::new(s)
        .encode_wide()
        .chain(Some(0))
        .collect()
}

/// One named-pipe connection — the read/write half shared by
/// [`DtactNamedPipeServer`] (after a client connects) and
/// [`DtactNamedPipeClient`].
#[repr(align(64))]
pub struct DtactNamedPipeHandle {
    handle: HANDLE,
}

unsafe impl Send for DtactNamedPipeHandle {}
unsafe impl Sync for DtactNamedPipeHandle {}

impl DtactNamedPipeHandle {
    /// Read into `buf`, returning the number of bytes read (`0` = the
    /// peer closed its end).
    ///
    /// # Errors
    /// Returns an `io::Error` if the underlying `ReadFile`/IOCP
    /// completion reports one.
    pub async fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        match issue_read(self.handle, buf) {
            IoOpResult::Ready(res) => res,
            IoOpResult::Pending(fut) => fut.await,
        }
    }

    /// Write from `buf`, returning the number of bytes written.
    ///
    /// # Errors
    /// Returns an `io::Error` if the underlying `WriteFile`/IOCP
    /// completion reports one (e.g. the peer closed its end).
    pub async fn write(&self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        match issue_write(self.handle, buf) {
            IoOpResult::Ready(res) => res,
            IoOpResult::Pending(fut) => fut.await,
        }
    }
}

impl Drop for DtactNamedPipeHandle {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

impl crate::io::AsyncRead for DtactNamedPipeHandle {
    async fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        self.read(buf).await
    }
}

impl crate::io::AsyncWrite for DtactNamedPipeHandle {
    async fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.write(buf).await
    }
}

/// A single named-pipe server instance, before a client has connected.
///
/// Create one per client you intend to accept — see the module doc's
/// "no pipe-instance pooling" paragraph for why there's no persistent
/// listener type the way TCP has [`super::DtactTcpListener`].
pub struct DtactNamedPipeServer {
    handle: HANDLE,
}

unsafe impl Send for DtactNamedPipeServer {}
unsafe impl Sync for DtactNamedPipeServer {}

impl DtactNamedPipeServer {
    /// Create a new duplex, byte-mode, overlapped pipe instance named
    /// `name` (e.g. `r"\\.\pipe\my-app"`).
    ///
    /// # Errors
    /// Returns an `io::Error` if `CreateNamedPipeW` fails (e.g. `name` is
    /// malformed) or if associating the handle with the IOCP fails.
    pub fn create(name: &str) -> io::Result<Self> {
        ensure_init();
        let wide = to_wide(name);
        let handle = unsafe {
            CreateNamedPipeW(
                wide.as_ptr(),
                PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                PIPE_UNLIMITED_INSTANCES,
                4096,
                4096,
                0,
                ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        let iocp = port();
        let assoc = unsafe { CreateIoCompletionPort(handle, iocp, PIPE_KEY, 0) };
        if assoc.is_null() {
            let e = io::Error::last_os_error();
            unsafe { CloseHandle(handle) };
            return Err(e);
        }
        Ok(Self { handle })
    }

    /// Wait for a client to connect to this pipe instance.
    ///
    /// # Errors
    /// Returns an `io::Error` if the underlying `ConnectNamedPipe`/IOCP
    /// completion reports one.
    pub async fn connect(self) -> io::Result<DtactNamedPipeHandle> {
        issue_connect(self.handle).await?;
        let handle = self.handle;
        std::mem::forget(self); // ownership of `handle` moves to the returned `DtactNamedPipeHandle`
        Ok(DtactNamedPipeHandle { handle })
    }
}

impl Drop for DtactNamedPipeServer {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

/// A named-pipe client. Connects to an already-`create`d server instance
/// by name.
pub struct DtactNamedPipeClient;

impl DtactNamedPipeClient {
    /// Connect to the server pipe instance named `name`, retrying (via
    /// `WaitNamedPipeW`) while every existing instance is busy — matches
    /// `tokio::net::windows::named_pipe::ClientOptions::open`'s own
    /// retry-on-`ERROR_PIPE_BUSY` behavior, since a brand-new server
    /// instance can take a moment to be created after the previous
    /// client disconnected.
    ///
    /// # Errors
    /// Returns an `io::Error` if `CreateFileW`/`WaitNamedPipeW` fails for
    /// any reason other than transient busy (e.g. `NotFound` if no server
    /// is listening at `name` at all), or if associating the resulting
    /// handle with the IOCP fails.
    ///
    /// # Panics
    /// Panics if the OS refuses to spawn the connect-retry thread (fatal
    /// resource exhaustion) — same class of failure every other native
    /// backend in this crate treats as unrecoverable at thread-spawn time.
    pub async fn connect(name: &str) -> io::Result<DtactNamedPipeHandle> {
        ensure_init();
        let name_owned = name.to_string();
        // `CreateFileW`/`WaitNamedPipeW` have no async variant — hand the
        // (briefly) blocking retry loop to a throwaway thread rather than
        // stalling the calling task's OS thread, same rationale as
        // `crate::io::lookup_host`. `HANDLE` (`*mut c_void`) isn't `Send`
        // by default, so it's carried across the channel wrapped in
        // `SendHandle` — sound because a `HANDLE` is just an opaque
        // kernel-assigned integer/pointer with no thread-affinity
        // requirement of its own (unlike e.g. a GUI window handle).
        let (tx, rx) = crate::sync::oneshot::channel();
        std::thread::Builder::new()
            .name("dtact-io-namedpipe-connect".into())
            .spawn(move || {
                let _ = tx.send(connect_blocking(&name_owned).map(SendHandle));
            })
            .expect("failed to spawn dtact-io named-pipe connect thread");
        let handle = rx
            .await
            .unwrap_or_else(|_| {
                Err(io::Error::other(
                    "dtact-io: named-pipe connect thread panicked before sending a result",
                ))
            })?
            .0;

        let iocp = port();
        let assoc = unsafe { CreateIoCompletionPort(handle, iocp, PIPE_KEY, 0) };
        if assoc.is_null() {
            let e = io::Error::last_os_error();
            unsafe { CloseHandle(handle) };
            return Err(e);
        }
        Ok(DtactNamedPipeHandle { handle })
    }
}

/// Wraps a raw `HANDLE` solely to carry it across the `connect_blocking`
/// resolver thread's `oneshot` channel — see the `# Safety`-equivalent
/// note at its one call site for why this is sound.
struct SendHandle(HANDLE);
unsafe impl Send for SendHandle {}

fn connect_blocking(name: &str) -> io::Result<HANDLE> {
    let wide = to_wide(name);
    loop {
        let handle = unsafe {
            CreateFileW(
                wide.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                0,
                ptr::null_mut(),
                OPEN_EXISTING,
                FILE_FLAG_OVERLAPPED,
                ptr::null_mut::<c_void>() as HANDLE,
            )
        };
        if handle != INVALID_HANDLE_VALUE {
            return Ok(handle);
        }
        let err = unsafe { GetLastError() };
        if err != ERROR_PIPE_BUSY {
            return Err(io::Error::last_os_error());
        }
        // Every existing instance is busy — wait (up to 5s) for one to
        // free up, then retry `CreateFileW`. `WaitNamedPipeW` itself has
        // no overlapped/async form.
        unsafe { WaitNamedPipeW(wide.as_ptr(), 5000) };
    }
}

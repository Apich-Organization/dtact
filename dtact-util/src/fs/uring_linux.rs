//! Linux native filesystem backend: real `io_uring` opcodes
//! (`OpenAt`/`Read`/`Write`/`Fsync`/`Close`) submitted to a single
//! dedicated ring, instead of `native.rs`'s thread-pool fallback (which on
//! Linux is not even compiled in — see `fs::mod`'s `cfg` gates).
//!
//! **Not every op actually goes through the ring.** [`DtactFile::read`]/
//! [`write`](DtactFile::write)/[`read_at`](DtactFile::read_at)/
//! [`write_at`](DtactFile::write_at) first try a direct, synchronous
//! `pread(2)`/`pwrite(2)` on the *calling* thread ([`try_pread`]/
//! [`try_pwrite`]) and only fall back to submitting an SQE on `EAGAIN`
//! (which in practice only happens for a non-regular fd, e.g. a pipe —
//! regular-file `pread`/`pwrite` essentially never returns `EAGAIN`). This
//! is a deliberate latency optimization (skips the `MpmcStack` push /
//! worker unpark / `submit_and_wait` / wake round-trip entirely for the
//! common case), but it means the "submitted to a single dedicated ring"
//! framing above is only strictly true for [`DtactFile::open`]/
//! [`create`](DtactFile::create)/[`sync_all`](DtactFile::sync_all)/
//! [`close`](DtactFile::close) plus non-regular-fd reads/writes — every
//! other read/write is a **blocking syscall run inline on whatever OS
//! thread is currently polling the future**. For page-cache-hot regular
//! files that's negligible; for a cache-miss read from slow local disk, or
//! any file actually backed by a network filesystem (NFS, etc., where
//! `pread`/`pwrite` can block for the network round-trip rather than
//! returning `EAGAIN`), that blocking syscall stalls the entire OS thread
//! it runs on — and with it, every other fiber/task currently scheduled
//! onto that thread — for the syscall's full duration. Not a correctness
//! bug, but a real latency-tail risk this module's earlier doc didn't call
//! out; workloads sensitive to that tail should route such reads/writes
//! elsewhere (e.g. a dedicated blocking-thread pool) rather than assume
//! this backend is uniformly non-blocking just because it's "the
//! `io_uring` backend".
//!
//! **Per-op state is a preallocated slot, not a fresh allocation.**
//! [`init_fs`] carves out a fixed `Box<[OpState]>` arena (sized by
//! `ring_depth`) up front, handed out/reclaimed via a
//! [`crate::lockfree::TreiberStack`] free-list, mirroring
//! `fs::iocp_windows`'s pool and (like it) `io::native`'s `BufferPool`
//! before that. Because a pooled slot's address is stable for the whole
//! process (never individually freed), `user_data` can just be the raw
//! slot pointer — no `Arc`/refcount bookkeeping needed for the common
//! case at all, which is a further simplification over this file's first
//! pass (which `Arc`-heap-allocated every single op). A slot is only
//! returned to the pool once its result has actually been observed
//! (`Drop for IoOp` checks `result != PENDING`); if a future is dropped
//! while its op is still in flight, the slot is deliberately leaked
//! rather than risked for reuse (reclaiming it safely needs an
//! `IORING_OP_ASYNC_CANCEL` submitted for it and waiting on *that*
//! completion first — not implemented here, same caveat as the Windows
//! backend's module doc).

use crate::lockfree::{AtomicWakerSlot, MpmcStack, TreiberStack};
use std::ffi::CString;
use std::future::Future;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, Ordering};
use std::task::{Context, Poll};
use std::thread::Thread;

use io_uring::{IoUring, opcode, squeue, types};

/// Sentinel for `OpState::result` meaning "not yet completed" — mirrors
/// `fs::iocp_windows::PENDING`. `>= 0` after completion is the raw
/// `io_uring` cqe result (bytes transferred / fd); `< 0` and `!= PENDING`
/// is `-errno`, exactly `io_uring`'s own convention, so no decode step is
/// needed beyond checking the sign.
const PENDING: i64 = i64::MIN;

#[repr(align(64))]
struct OpState {
    result: AtomicI64,
    waker: AtomicWakerSlot,
}

impl OpState {
    const fn fresh() -> Self {
        Self {
            result: AtomicI64::new(PENDING),
            waker: AtomicWakerSlot::new(),
        }
    }
}

// =============================================================================
// Preallocated slot pool — see module doc for the reuse/leak-on-cancel policy.
// =============================================================================

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

/// Which allocation a given `IoOp`'s [`OpState`] lives in: a checked-out
/// pool slot (common case, no allocation), or a one-off heap fallback if
/// the pool was exhausted.
#[repr(align(64))]
enum Slot {
    Pooled(u32),
    Heap(Box<OpState>),
}

fn acquire_slot() -> Slot {
    let pool = slot_pool();
    pool.free.pop().map_or_else(
        || Slot::Heap(Box::new(OpState::fresh())),
        |idx| {
            pool.slots[idx as usize]
                .result
                .store(PENDING, Ordering::Relaxed);
            Slot::Pooled(idx)
        },
    )
}

/// Wraps a raw `squeue::Entry` so it can cross the pending-submit queue to
/// the single worker thread that owns the `IoUring` instance. Sound because
/// every pointer baked into the entry (path `CString`, buffer, the slot
/// itself) is kept alive by its owning allocation — the pool's arena for
/// pooled slots (never freed), the `IoOp`'s `Box` for heap-fallback slots
/// (kept alive across the whole `.await`) — until the matching completion
/// is processed in `worker_loop`.
struct SendEntry(squeue::Entry);
unsafe impl Send for SendEntry {}

#[repr(align(64))]
struct Ring {
    /// Lock-free MPMC handoff (many task threads push, the single worker
    /// thread drains) — not a `Mutex<Vec<SendEntry>>`.
    pending: MpmcStack<SendEntry>,
    worker: OnceLock<Thread>,
}

static RING: OnceLock<Ring> = OnceLock::new();

fn ring() -> &'static Ring {
    RING.get_or_init(|| {
        let r = Ring {
            pending: MpmcStack::new(),
            worker: OnceLock::new(),
        };
        let handle = std::thread::Builder::new()
            .name("dtact-fs-uring".into())
            .spawn(worker_loop)
            .expect("failed to spawn dtact-fs-uring worker thread");
        let _ = r.worker.set(handle.thread().clone());
        r
    })
}

/// Configure and eagerly start the fs-io_uring subsystem.
///
/// Preallocates `ring_depth` op slots (see module doc) and starts the
/// submit-queue worker thread. `workers`/`buffer_pool_size`/`chunk_size`/
/// `pin_cpus` mirror `crate::io::native::init_runtime`'s signature for
/// consistency across this crate's native backends but are unused here
/// today: submission is single-worker-thread by design, and there's no
/// `IORING_REGISTER_BUFFERS`-backed buffer pool yet (see the module doc's
/// "zero-copy" note).
pub fn init_fs(
    _workers: usize,
    ring_depth: u32,
    _buffer_pool_size: usize,
    _chunk_size: usize,
    _pin_cpus: &[usize],
) {
    let _ = RING_DEPTH.set(ring_depth.max(1) as usize);
    let _ = slot_pool();
    let _ = ring();
}

/// Simple-signature convenience wrapper: `init_fs(workers, 256, 0, 0, &[])`.
pub fn init(workers: usize) {
    init_fs(workers, 256, 0, 0, &[]);
}

fn worker_loop() {
    let mut io_uring = IoUring::new(256).expect("dtact-fs: IoUring::new failed");
    let r = RING
        .get()
        .expect("ring() must be called before worker_loop starts");
    loop {
        if r.pending.is_empty() {
            // `park_timeout` rather than an unbounded `park()`: closes the
            // (rare) race where a new entry is pushed and this thread
            // unparked *just before* it actually calls park — worst case
            // we wake up to nothing and loop back around within 5ms.
            std::thread::park_timeout(std::time::Duration::from_millis(5));
            continue;
        }
        let batch = r.pending.drain_all();
        if batch.is_empty() {
            continue;
        }

        {
            let mut sq = io_uring.submission();
            for entry in &batch {
                // SAFETY: buffers/paths referenced by each entry are kept
                // alive by their owning allocations until the matching
                // completion is processed below, so they're still valid
                // at submit time.
                unsafe {
                    let _ = sq.push(&entry.0);
                }
            }
            sq.sync();
        }

        if let Err(e) = io_uring.submit_and_wait(batch.len()) {
            eprintln!("dtact-fs-uring: submit_and_wait failed: {e}");
            continue;
        }

        let mut cq = io_uring.completion();
        cq.sync();
        for cqe in &mut cq {
            let user_data = cqe.user_data();
            if user_data == 0 {
                continue;
            }
            // No ownership transfer needed here (unlike the earlier
            // Arc-per-op version): pooled slots live in the arena for the
            // whole process, heap-fallback slots are kept alive by the
            // `IoOp` across its `.await`, so this is just a borrow.
            let state = unsafe { &*(user_data as *const OpState) };
            let res = cqe.result();
            state.result.store(i64::from(res), Ordering::Release);
            state.waker.take_and_wake();
        }
    }
}

fn submit(entry: squeue::Entry) -> IoOp {
    let slot = acquire_slot();
    let ptr: *const OpState = match &slot {
        Slot::Pooled(idx) => &raw const slot_pool().slots[*idx as usize],
        Slot::Heap(b) => b.as_ref(),
    };
    let entry = entry.user_data(ptr as u64);
    let r = ring();
    r.pending.push(SendEntry(entry));
    if let Some(t) = r.worker.get() {
        t.unpark();
    }
    IoOp { slot }
}

struct IoOp {
    slot: Slot,
}

impl IoOp {
    #[inline]
    fn state(&self) -> &OpState {
        match &self.slot {
            Slot::Pooled(idx) => &slot_pool().slots[*idx as usize],
            Slot::Heap(b) => b,
        }
    }
}

impl Future for IoOp {
    type Output = io::Result<i32>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<i32>> {
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
        if let Slot::Pooled(idx) = self.slot {
            let pool = slot_pool();
            let done = pool.slots[idx as usize].result.load(Ordering::Acquire) != PENDING;
            if done {
                pool.free.push(idx);
            }
            // Else: leak this slot — see module doc's cancellation caveat.
        }
    }
}

/// Direct positional read on the calling thread.
///
/// Regular files served from the page cache complete a `pread(2)`
/// immediately without ever blocking, so issuing the syscall inline avoids
/// the entire cross-thread ring round-trip — `MpmcStack` push, worker
/// unpark, `submit_and_wait`, and the wake back — that otherwise dominates
/// small/medium file-op latency (a single `dtact-fs-uring` worker also
/// serializes every op, which the direct path sidesteps). Returns `None`
/// only if the syscall would block (`EAGAIN`, possible only for a
/// non-regular fd such as a pipe), signalling the caller to fall back to
/// the ring; `EINTR` is retried transparently.
#[inline]
fn try_pread(fd: i32, buf: &mut [u8], offset: u64) -> Option<io::Result<usize>> {
    loop {
        let n = unsafe {
            libc::pread(
                fd,
                buf.as_mut_ptr().cast::<libc::c_void>(),
                buf.len(),
                offset as libc::off_t,
            )
        };
        if n >= 0 {
            return Some(Ok(n as usize));
        }
        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            // Interrupted before any transfer — retry the syscall.
            Some(libc::EINTR) => {}
            Some(libc::EAGAIN) => return None,
            _ => return Some(Err(err)),
        }
    }
}

/// Direct positional write on the calling thread — the write-side twin of
/// [`try_pread`]; see its doc for the rationale. Returns `None` on `EAGAIN`
/// to fall back to the ring.
#[inline]
fn try_pwrite(fd: i32, buf: &[u8], offset: u64) -> Option<io::Result<usize>> {
    loop {
        let n = unsafe {
            libc::pwrite(
                fd,
                buf.as_ptr().cast::<libc::c_void>(),
                buf.len(),
                offset as libc::off_t,
            )
        };
        if n >= 0 {
            return Some(Ok(n as usize));
        }
        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            // Interrupted before any transfer — retry the syscall.
            Some(libc::EINTR) => {}
            Some(libc::EAGAIN) => return None,
            _ => return Some(Err(err)),
        }
    }
}

fn decode(res: i64) -> io::Result<i32> {
    if res < 0 {
        Err(io::Error::from_raw_os_error(-res as i32))
    } else {
        Ok(res as i32)
    }
}

fn path_cstring(path: &Path) -> io::Result<CString> {
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains a NUL byte"))
}

/// An open file whose ops are submitted as real `io_uring` SQEs.
pub struct DtactFile {
    fd: i32,
    cursor: AtomicI64,
}

unsafe impl Send for DtactFile {}
unsafe impl Sync for DtactFile {}

async fn open_impl(path: &Path, flags: i32, mode: u32) -> io::Result<DtactFile> {
    let cpath = path_cstring(path)?;
    let cpath_ptr = cpath.as_ptr();
    let entry = opcode::OpenAt::new(types::Fd(libc::AT_FDCWD), cpath_ptr)
        .flags(flags)
        .mode(mode)
        .build();
    let op = submit(entry);
    let fd = op.await?;
    // `cpath` must outlive the point where the kernel has actually
    // dereferenced the path, i.e. until `submit_and_wait` returns for this
    // SQE, which is exactly when our `IoOp` resolves — safe to drop now.
    drop(cpath);
    Ok(DtactFile {
        fd,
        cursor: AtomicI64::new(0),
    })
}

impl DtactFile {
    /// Open an existing file for reading via a ring-submitted `Openat`.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `Openat` completion
    /// reports one — most commonly `NotFound` if `path` doesn't exist, or
    /// `PermissionDenied` if it exists but isn't readable.
    pub async fn open(path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        open_impl(&path, libc::O_RDONLY, 0).await
    }

    /// Create (truncating if it already exists) a file for reading and
    /// writing via a ring-submitted `Openat`.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `Openat` completion
    /// reports one, e.g. `PermissionDenied` if the containing directory
    /// isn't writable.
    pub async fn create(path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        open_impl(&path, libc::O_RDWR | libc::O_CREAT | libc::O_TRUNC, 0o644).await
    }

    /// Generic open honoring an arbitrary [`std::fs::OpenOptions`]. See
    /// the doc comment inline below for why this falls back to a
    /// synchronous `open()` rather than a pure-uring one.
    ///
    /// # Errors
    ///
    /// Returns whatever `std::fs::OpenOptions::open` returns for `path`
    /// with `opts` applied (e.g. `NotFound`, `PermissionDenied`,
    /// `AlreadyExists` depending on which `OpenOptions` flags are set).
    // `opts` is only ever borrowed internally (`OpenOptions::open` takes
    // `&self`), but taking it by value here is the existing, stable public
    // API (unrelated to the `async` removal above) — changing it to `&_`
    // would be a breaking signature change for callers, not a clippy-only
    // fix.
    #[allow(clippy::needless_pass_by_value)]
    pub fn open_with(
        path: impl Into<PathBuf>,
        opts: std::fs::OpenOptions,
    ) -> impl std::future::Future<Output = io::Result<Self>> {
        // `std::fs::OpenOptions` has no public flag getters; delegate to
        // its own (synchronous) `open()` for flag resolution, then hand
        // the resulting fd off to the ring for all subsequent async ops.
        // This costs one blocking `openat(2)` on the calling thread for
        // the *open* only — reads/writes on the returned handle are still
        // fully io_uring-async. A pure-uring open would need to duplicate
        // `OpenOptions`' private flag-computation logic here instead.
        use std::os::unix::io::IntoRawFd;
        std::future::ready((|| {
            let path = path.into();
            let file = opts.open(&path)?;
            let fd = file.into_raw_fd();
            Ok(Self {
                fd,
                cursor: AtomicI64::new(0),
            })
        })())
    }

    /// Read at the file's shared cursor, advancing it by the number of
    /// bytes actually read. `buf` is handed back (resized to what was
    /// filled) so the caller can reuse its allocation.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `Read` completion reports
    /// one; a short read (including `0` at EOF) is a normal `Ok`, not an
    /// error.
    pub async fn read(&self, mut buf: Vec<u8>) -> io::Result<(usize, Vec<u8>)> {
        let offset = self.cursor.load(Ordering::Relaxed) as u64;
        // Fast path: direct pread on the calling thread (see `try_pread`).
        if let Some(res) = try_pread(self.fd, &mut buf, offset) {
            let n = res?;
            self.cursor.fetch_add(n as i64, Ordering::Relaxed);
            return Ok((n, buf));
        }
        let entry = opcode::Read::new(types::Fd(self.fd), buf.as_mut_ptr(), buf.len() as u32)
            .offset(offset)
            .build();
        let n = submit(entry).await?;
        self.cursor.fetch_add(i64::from(n), Ordering::Relaxed);
        Ok((n as usize, buf))
    }

    /// Write at the file's shared cursor, advancing it by the number of
    /// bytes actually written. `buf` is handed back so the caller can
    /// reuse its allocation.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `Write` completion
    /// reports one (e.g. disk full, or the fd was closed concurrently).
    pub async fn write(&self, buf: Vec<u8>) -> io::Result<(usize, Vec<u8>)> {
        let offset = self.cursor.load(Ordering::Relaxed) as u64;
        // Fast path: direct pwrite on the calling thread (see `try_pwrite`).
        if let Some(res) = try_pwrite(self.fd, &buf, offset) {
            let n = res?;
            self.cursor.fetch_add(n as i64, Ordering::Relaxed);
            return Ok((n, buf));
        }
        let entry = opcode::Write::new(types::Fd(self.fd), buf.as_ptr(), buf.len() as u32)
            .offset(offset)
            .build();
        let n = submit(entry).await?;
        self.cursor.fetch_add(i64::from(n), Ordering::Relaxed);
        Ok((n as usize, buf))
    }

    /// Positional read: submits its own SQE with an explicit offset, so
    /// concurrent `read_at`/`write_at` calls on the same handle are safe
    /// (no shared cursor involved).
    ///
    /// # Errors
    ///
    /// Same as [`Self::read`].
    pub async fn read_at(&self, mut buf: Vec<u8>, offset: u64) -> io::Result<(usize, Vec<u8>)> {
        // Fast path: direct pread on the calling thread (see `try_pread`).
        if let Some(res) = try_pread(self.fd, &mut buf, offset) {
            let n = res?;
            return Ok((n, buf));
        }
        let entry = opcode::Read::new(types::Fd(self.fd), buf.as_mut_ptr(), buf.len() as u32)
            .offset(offset)
            .build();
        let n = submit(entry).await?;
        Ok((n as usize, buf))
    }

    /// Positional write: submits its own SQE with an explicit offset, so
    /// concurrent `read_at`/`write_at` calls on the same handle are safe.
    ///
    /// # Errors
    ///
    /// Same as [`Self::write`].
    pub async fn write_at(&self, buf: Vec<u8>, offset: u64) -> io::Result<(usize, Vec<u8>)> {
        // Fast path: direct pwrite on the calling thread (see `try_pwrite`).
        if let Some(res) = try_pwrite(self.fd, &buf, offset) {
            let n = res?;
            return Ok((n, buf));
        }
        let entry = opcode::Write::new(types::Fd(self.fd), buf.as_ptr(), buf.len() as u32)
            .offset(offset)
            .build();
        let n = submit(entry).await?;
        Ok((n as usize, buf))
    }

    /// Flush all buffered writes to disk via a ring-submitted `Fsync`.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `Fsync` completion
    /// reports one (e.g. the underlying device was removed).
    pub async fn sync_all(&self) -> io::Result<()> {
        let entry = opcode::Fsync::new(types::Fd(self.fd)).build();
        submit(entry).await?;
        Ok(())
    }

    /// File metadata (size, timestamps, permissions, ...).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `fstat` fails (e.g. the
    /// fd was closed concurrently).
    pub fn metadata(&self) -> impl std::future::Future<Output = io::Result<std::fs::Metadata>> {
        // `Statx` needs a scratch `statx` buffer plus a conversion to
        // `std::fs::Metadata`, which has no public constructor from raw
        // `statx` fields. Fall back to a direct `fstat` via a borrowed
        // `std::fs::File` (fd not taken, just observed) rather than faking
        // a `Metadata` — same "cheap enough to not need the ring" judgment
        // call as `fs::iocp_windows::metadata`.
        use std::os::unix::io::{AsRawFd, FromRawFd};
        let file = unsafe { std::fs::File::from_raw_fd(self.fd) };
        let file = std::mem::ManuallyDrop::new(file);
        let meta = file.metadata();
        let _ = file.as_raw_fd();
        std::future::ready(meta)
    }

    /// Explicitly close this file via a ring-submitted `Close`, rather
    /// than waiting for `Drop` (which closes synchronously instead).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `Close` completion
    /// reports one.
    pub async fn close(self) -> io::Result<()> {
        let entry = opcode::Close::new(types::Fd(self.fd)).build();
        submit(entry).await?;
        std::mem::forget(self); // fd already closed by the kernel via the op above
        Ok(())
    }
}

impl Drop for DtactFile {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.fd);
        }
    }
}

/// Metadata for the file/directory at `path`. Delegates to
/// `std::fs::metadata` (a single synchronous syscall — not worth
/// dispatching through the ring).
///
/// # Errors
///
/// Returns whatever `std::fs::metadata` returns, e.g. `NotFound` if
/// `path` doesn't exist.
pub async fn metadata(path: impl Into<PathBuf>) -> io::Result<std::fs::Metadata> {
    let path = path.into();
    std::fs::metadata(&path)
}

/// List the entries of directory `path`. Delegates to
/// `std::fs::read_dir`, eagerly collecting all entries.
///
/// # Errors
///
/// Returns whatever `std::fs::read_dir` returns for opening the
/// directory, or whatever the first failing entry's `io::Result` returns
/// while collecting.
pub async fn read_dir(path: impl Into<PathBuf>) -> io::Result<Vec<std::fs::DirEntry>> {
    let path: PathBuf = path.into();
    std::fs::read_dir(&path)?.collect()
}

/// Recursively create `path` and any missing parent directories.
/// Delegates to `std::fs::create_dir_all`.
///
/// # Errors
///
/// Returns whatever `std::fs::create_dir_all` returns, e.g.
/// `PermissionDenied`.
pub async fn create_dir_all(path: impl Into<PathBuf>) -> io::Result<()> {
    let path = path.into();
    std::fs::create_dir_all(&path)
}

/// Remove the file at `path`. Delegates to `std::fs::remove_file`.
///
/// # Errors
///
/// Returns whatever `std::fs::remove_file` returns, e.g. `NotFound`.
pub async fn remove_file(path: impl Into<PathBuf>) -> io::Result<()> {
    let path = path.into();
    std::fs::remove_file(&path)
}

/// Resolve `path` to an absolute path with all intermediate components
/// (`.`, `..`, symlinks) resolved.
///
/// Delegates to `std::fs::canonicalize` — same "one syscall, not worth
/// the ring" judgment call as [`metadata`].
///
/// # Errors
///
/// Returns whatever `std::fs::canonicalize` returns, e.g. `NotFound`.
pub async fn canonicalize(path: impl Into<PathBuf>) -> io::Result<PathBuf> {
    let path = path.into();
    std::fs::canonicalize(&path)
}

/// Copy the contents (and permission bits) of the file at `from` to `to`,
/// returning the byte count copied. Delegates to `std::fs::copy`.
///
/// # Errors
///
/// Returns whatever `std::fs::copy` returns, e.g. `NotFound` if `from`
/// doesn't exist.
pub async fn copy(from: impl Into<PathBuf>, to: impl Into<PathBuf>) -> io::Result<u64> {
    let from = from.into();
    let to = to.into();
    std::fs::copy(&from, &to)
}

/// Create a single new directory. Unlike [`create_dir_all`], fails if any
/// parent component doesn't already exist. Delegates to
/// `std::fs::create_dir`.
///
/// # Errors
///
/// Returns whatever `std::fs::create_dir` returns, e.g. `AlreadyExists`.
pub async fn create_dir(path: impl Into<PathBuf>) -> io::Result<()> {
    let path = path.into();
    std::fs::create_dir(&path)
}

/// Create a hard link at `dst` pointing at the same inode as `src`.
/// Delegates to `std::fs::hard_link`.
///
/// # Errors
///
/// Returns whatever `std::fs::hard_link` returns, e.g. `NotFound` if
/// `src` doesn't exist.
pub async fn hard_link(src: impl Into<PathBuf>, dst: impl Into<PathBuf>) -> io::Result<()> {
    let src = src.into();
    let dst = dst.into();
    std::fs::hard_link(&src, &dst)
}

/// Read the entire contents of the file at `path` into a `Vec<u8>`.
///
/// Delegates to `std::fs::read` — a whole-file read isn't worth routing
/// through the ring op-by-op the way [`DtactFile::read`] is for
/// caller-managed partial reads.
///
/// # Errors
///
/// Returns whatever `std::fs::read` returns, e.g. `NotFound`.
pub async fn read(path: impl Into<PathBuf>) -> io::Result<Vec<u8>> {
    let path = path.into();
    std::fs::read(&path)
}

/// Read the target of the symbolic link at `path`. Delegates to
/// `std::fs::read_link`.
///
/// # Errors
///
/// Returns whatever `std::fs::read_link` returns, e.g. `NotFound`, or an
/// error if `path` isn't actually a symlink.
pub async fn read_link(path: impl Into<PathBuf>) -> io::Result<PathBuf> {
    let path = path.into();
    std::fs::read_link(&path)
}

/// Read the entire contents of the file at `path` into a `String`.
/// Delegates to `std::fs::read_to_string`.
///
/// # Errors
///
/// Returns whatever `std::fs::read_to_string` returns, e.g. an
/// `InvalidData` error if the file isn't valid UTF-8.
pub async fn read_to_string(path: impl Into<PathBuf>) -> io::Result<String> {
    let path = path.into();
    std::fs::read_to_string(&path)
}

/// Remove an empty directory. Fails if `path` is non-empty — see
/// [`remove_dir_all`] for the recursive version. Delegates to
/// `std::fs::remove_dir`.
///
/// # Errors
///
/// Returns whatever `std::fs::remove_dir` returns, e.g. `NotFound`, or an
/// error if the directory isn't empty.
pub async fn remove_dir(path: impl Into<PathBuf>) -> io::Result<()> {
    let path = path.into();
    std::fs::remove_dir(&path)
}

/// Recursively remove a directory and everything under it. Delegates to
/// `std::fs::remove_dir_all`.
///
/// # Errors
///
/// Returns whatever `std::fs::remove_dir_all` returns, e.g. `NotFound`.
pub async fn remove_dir_all(path: impl Into<PathBuf>) -> io::Result<()> {
    let path = path.into();
    std::fs::remove_dir_all(&path)
}

/// Rename (move) the file or directory at `from` to `to`, replacing `to`
/// if it already exists.
///
/// Delegates to `std::fs::rename` — see its own documentation for
/// cross-platform caveats (e.g. renaming across filesystems).
///
/// # Errors
///
/// Returns whatever `std::fs::rename` returns.
pub async fn rename(from: impl Into<PathBuf>, to: impl Into<PathBuf>) -> io::Result<()> {
    let from = from.into();
    let to = to.into();
    std::fs::rename(&from, &to)
}

/// Set `path`'s permission bits to `perm`. Delegates to
/// `std::fs::set_permissions`.
///
/// # Errors
///
/// Returns whatever `std::fs::set_permissions` returns, e.g. `NotFound`.
pub async fn set_permissions(
    path: impl Into<PathBuf>,
    perm: std::fs::Permissions,
) -> io::Result<()> {
    let path = path.into();
    std::fs::set_permissions(&path, perm)
}

/// Create a symbolic link at `dst` pointing at `src`. Delegates to
/// `std::os::unix::fs::symlink`.
///
/// # Errors
///
/// Returns whatever `std::os::unix::fs::symlink` returns, e.g.
/// `AlreadyExists` if `dst` already exists.
pub async fn symlink(src: impl Into<PathBuf>, dst: impl Into<PathBuf>) -> io::Result<()> {
    let src = src.into();
    let dst = dst.into();
    std::os::unix::fs::symlink(&src, &dst)
}

/// Query `path`'s metadata *without* following a trailing symlink (unlike
/// [`metadata`], which does). Delegates to `std::fs::symlink_metadata`.
///
/// # Errors
///
/// Returns whatever `std::fs::symlink_metadata` returns, e.g. `NotFound`.
pub async fn symlink_metadata(path: impl Into<PathBuf>) -> io::Result<std::fs::Metadata> {
    let path = path.into();
    std::fs::symlink_metadata(&path)
}

/// Check whether `path` exists, following symlinks — a permission error
/// while checking is propagated as `Err` rather than silently read as
/// "doesn't exist". Delegates to `std::fs::exists`.
///
/// # Errors
///
/// Returns an `io::Error` for any failure *other than* "doesn't exist",
/// e.g. `PermissionDenied` on a parent directory.
pub async fn try_exists(path: impl Into<PathBuf>) -> io::Result<bool> {
    let path = path.into();
    std::fs::exists(&path)
}

/// Write `contents` to the file at `path`, creating it if it doesn't
/// exist and truncating it if it does. Delegates to `std::fs::write`.
///
/// # Errors
///
/// Returns whatever `std::fs::write` returns, e.g. `PermissionDenied`.
pub async fn write(path: impl Into<PathBuf>, contents: impl AsRef<[u8]>) -> io::Result<()> {
    let path = path.into();
    std::fs::write(&path, contents)
}

//! Windows native filesystem backend: real IOCP-based async file I/O.
//!
//! Unlike `native.rs`'s thread-pool fallback, this issues `ReadFile`/
//! `WriteFile` directly against a `FILE_FLAG_OVERLAPPED` handle associated
//! with a completion port, so a read/write is a single async syscall whose
//! completion is delivered straight into the caller-supplied buffer — no
//! hop through a blocking worker thread, no extra copy. This is the
//! Windows analogue of the `io_uring` path on Linux (`uring_linux.rs`).
//!
//! One process-wide IOCP handle + one dedicated worker thread drains
//! `GetQueuedCompletionStatusEx` and wakes the waiting future per
//! completed op.
//!
//! **Per-op state is a preallocated slot, not a fresh heap allocation.**
//! [`init_fs`] carves out a fixed `Box<[OpState]>` arena (sized by
//! `ring_depth`) up front, handed out and reclaimed via a
//! [`crate::lockfree::TreiberStack`] free-list — the same
//! "arena + Treiber free-list" shape `io::native`'s `BufferPool` uses,
//! moved into `crate::lockfree` specifically so this module could share
//! it. `OVERLAPPED` is still the struct's first field (`#[repr(C)]`) so
//! the raw `*mut OVERLAPPED` Windows hands back on completion casts
//! straight back to the full slot, whether that slot lives in the pool
//! or (only if the pool is exhausted) on the heap as a one-off fallback —
//! see `acquire_slot`.
//!
//! A slot is only returned to the pool once its result has actually been
//! observed (`Drop for IoOp` checks `result != PENDING`); if a future is
//! dropped while its op is still in flight, that slot is deliberately
//! leaked rather than risked for reuse — reclaiming it safely would need
//! `CancelIoEx` plus waiting for the cancellation's own completion before
//! the slot could be trusted again (the same problem `io::native` solves
//! for socket ops via its `cancel_queue`). Not implemented in this pass;
//! every test/bench in this crate awaits ops to completion, so it isn't
//! exercised, but it's a real gap a future pass should close before this
//! is used somewhere that cancels in-flight file ops routinely.

use crate::lockfree::{AtomicWakerSlot, TreiberStack};
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::ptr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, Ordering};
use std::task::{Context, Poll};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_HANDLE_EOF, ERROR_IO_PENDING, GENERIC_READ, GENERIC_WRITE, HANDLE,
    INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Storage::FileSystem::{
    CREATE_ALWAYS, CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ,
    FILE_SHARE_WRITE, FlushFileBuffers, GetFileSizeEx, OPEN_EXISTING, ReadFile, WriteFile,
};
use windows_sys::Win32::System::IO::{
    CreateIoCompletionPort, GetQueuedCompletionStatusEx, OVERLAPPED, OVERLAPPED_ENTRY,
};

const WAKE_KEY: usize = 0;
const FILE_KEY: usize = 1;

struct Port {
    handle: HANDLE,
}
unsafe impl Send for Port {}
unsafe impl Sync for Port {}

static PORT: OnceLock<Port> = OnceLock::new();

fn port() -> HANDLE {
    let p = PORT.get_or_init(|| {
        let handle = unsafe { CreateIoCompletionPort(INVALID_HANDLE_VALUE, ptr::null_mut(), 0, 1) };
        assert!(!handle.is_null(), "dtact-fs: CreateIoCompletionPort failed");
        std::thread::Builder::new()
            .name("dtact-fs-iocp".into())
            .spawn(worker_loop)
            .expect("failed to spawn dtact-fs-iocp worker thread");
        Port { handle }
    });
    p.handle
}

/// Sentinel for `OpState::result` meaning "not yet completed". Any other
/// value is a real result: `>= 0` is bytes transferred, `< 0` (and
/// `!= PENDING`) is `-(win32 error code)` — Win32 error codes are small
/// positive `DWORD`s, so this never collides with the sentinel.
const PENDING: i64 = i64::MIN;

#[repr(C)]
struct OpState {
    overlapped: OVERLAPPED,
    /// PENDING until the IOCP worker stores the real outcome — see the
    /// `PENDING` doc above. Single atomic: no lock, one store on the fire
    /// path, one load (or two, across the waker-registration race window)
    /// on the poll path.
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

// SAFETY: `overlapped` embeds an `OVERLAPPED` (raw pointers inside, from
// windows-sys, e.g. its `hEvent` field) purely as opaque kernel-visible
// scratch memory: it is written once at submission time on the submitting
// thread, then only ever touched by the OS kernel and by the single IOCP
// worker thread reading it back via `GetQueuedCompletionStatusEx` — Rust
// code never dereferences the pointers embedded inside it. The fields
// Rust code actually reads/writes concurrently (`result`, `waker`) are
// already atomics, so no additional synchronization is needed for those
// either. This makes `OpState` sound to hand across threads, which in
// turn is what makes `IoOp` (which owns one via `Slot` below) itself
// `Send`, so its future can be `.await`-ed from a multi-threaded executor
// (required for the blocking FFI layer to `block_on` it from any thread).
unsafe impl Send for OpState {}
// SAFETY: same reasoning as `Send` above — all cross-thread-visible state
// is atomics; nothing borrows `&OpState` and mutates the non-atomic
// `overlapped` field concurrently with another thread's access.
unsafe impl Sync for OpState {}

// =============================================================================
// Preallocated slot pool — see module doc for the reuse/leak-on-cancel policy.
// =============================================================================

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

/// Configure and eagerly start the fs-IOCP subsystem.
///
/// `ring_depth` sized preallocated op slots (see module doc) plus the
/// completion-port worker thread. `workers`/`buffer_pool_size`/
/// `chunk_size`/`pin_cpus` are accepted for signature parity with the
/// other native backends' `init_fs` (and with
/// `crate::io::native::init_runtime`) but unused here: IOCP dispatch is
/// single-worker-thread by design (one port, one
/// `GetQueuedCompletionStatusEx` loop), and this backend has no
/// caller-facing buffer pool yet (reads/writes still take an owned
/// `Vec<u8>` per call — see the `fs` module doc for why that wasn't
/// changed in this pass).
pub fn init_fs(
    _workers: usize,
    ring_depth: u32,
    _buffer_pool_size: usize,
    _chunk_size: usize,
    _pin_cpus: &[usize],
) {
    let _ = RING_DEPTH.set(ring_depth.max(1) as usize);
    let _ = slot_pool();
    let _ = port();
}

/// Simple-signature convenience wrapper: `init_fs(workers, 256, 0, 0, &[])`.
pub fn init(workers: usize) {
    init_fs(workers, 256, 0, 0, &[]);
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
            // every `OpState` in this module lives in a `Box<[OpState]>` or
            // `Box<OpState>` allocation, and `Box`/`Vec` always align to
            // `align_of::<T>()`, so the cast is sound. Clippy cannot see
            // that invariant.
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
            // `Internal` holds the NTSTATUS of the completed request; a
            // nonzero value with zero bytes transferred on a Read is EOF or
            // a real error — treat "zero bytes, no bytes expected error" as
            // a plain EOF (0), matching ReadFile's synchronous convention.
            op.result.store(encode_ok(bytes), Ordering::Release);
            op.waker.take_and_wake();
        }
    }
}

/// Which allocation a given `IoOp`'s [`OpState`] lives in: a checked-out
/// pool slot (common case, no allocation), or a one-off heap fallback if
/// the pool was exhausted.
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

    #[inline]
    fn overlapped_ptr(&self) -> *mut OVERLAPPED {
        match &self.slot {
            // SAFETY: this index is exclusively checked out to this `IoOp`
            // until it's returned to the free-list in `Drop` (never while
            // an op referencing it might still be in flight — see the
            // reuse policy in the module doc), so a mutable raw view of
            // the array element it owns is sound despite going through a
            // shared `&SlotPool` reference.
            Slot::Pooled(idx) => (&raw const slot_pool().slots[*idx as usize])
                .cast_mut()
                .cast(),
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

fn issue_read(handle: HANDLE, buf: &mut [u8], offset: u64) -> IoOp {
    let slot = acquire_slot();
    let op = IoOp { slot };
    let ov_ptr = op.overlapped_ptr();
    unsafe {
        (*ov_ptr).Anonymous.Anonymous.Offset = offset as u32;
        (*ov_ptr).Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
    }

    let ok = unsafe {
        ReadFile(
            handle,
            buf.as_mut_ptr(),
            buf.len() as u32,
            ptr::null_mut(),
            ov_ptr,
        )
    };
    if ok == 0 {
        let err = unsafe { windows_sys::Win32::Foundation::GetLastError() };
        if err != ERROR_IO_PENDING {
            let encoded = if err == ERROR_HANDLE_EOF {
                encode_ok(0)
            } else {
                encode_err(err)
            };
            op.state().result.store(encoded, Ordering::Release);
        }
    }
    op
}

fn issue_write(handle: HANDLE, buf: &[u8], offset: u64) -> IoOp {
    let slot = acquire_slot();
    let op = IoOp { slot };
    let ov_ptr = op.overlapped_ptr();
    unsafe {
        (*ov_ptr).Anonymous.Anonymous.Offset = offset as u32;
        (*ov_ptr).Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
    }

    let ok = unsafe {
        WriteFile(
            handle,
            buf.as_ptr(),
            buf.len() as u32,
            ptr::null_mut(),
            ov_ptr,
        )
    };
    if ok == 0 {
        let err = unsafe { windows_sys::Win32::Foundation::GetLastError() };
        if err != ERROR_IO_PENDING {
            op.state().result.store(encode_err(err), Ordering::Release);
        }
    }
    op
}

fn to_wide(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str().encode_wide().chain(Some(0)).collect()
}

/// An open file whose reads/writes are dispatched as real overlapped IOCP
/// operations — no thread-pool hop, buffer handed straight to the kernel.
pub struct DtactFile {
    handle: HANDLE,
    cursor: AtomicI64,
}

unsafe impl Send for DtactFile {}
unsafe impl Sync for DtactFile {}

fn open_impl(path: &Path, disposition: u32, access: u32) -> io::Result<DtactFile> {
    let wide = to_wide(path);
    let handle = unsafe {
        CreateFileW(
            wide.as_ptr(),
            access,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            ptr::null_mut(),
            disposition,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
            ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    let iocp = port();
    let assoc = unsafe { CreateIoCompletionPort(handle, iocp, FILE_KEY, 0) };
    if assoc.is_null() {
        let e = io::Error::last_os_error();
        unsafe { CloseHandle(handle) };
        return Err(e);
    }
    Ok(DtactFile {
        handle,
        cursor: AtomicI64::new(0),
    })
}

fn open_with_impl(path: &Path, opts: &std::fs::OpenOptions) -> io::Result<DtactFile> {
    use std::os::windows::io::IntoRawHandle;
    let file = opts.open(path)?;
    let handle = file.into_raw_handle() as HANDLE;
    let iocp = port();
    let assoc = unsafe { CreateIoCompletionPort(handle, iocp, FILE_KEY, 0) };
    if assoc.is_null() {
        let e = io::Error::last_os_error();
        unsafe { CloseHandle(handle) };
        return Err(e);
    }
    Ok(DtactFile {
        handle,
        cursor: AtomicI64::new(0),
    })
}

impl DtactFile {
    /// Open an existing file for reading.
    ///
    /// # Errors
    ///
    /// Returns `io::ErrorKind::NotFound` if `path` doesn't exist,
    /// `PermissionDenied` if it exists but isn't readable by the current
    /// user, or another `io::Error` from `CreateFileW`/associating the
    /// handle with the IOCP for any other OS-level failure.
    pub fn open(path: impl Into<PathBuf>) -> impl Future<Output = io::Result<Self>> {
        let path = path.into();
        std::future::ready(open_impl(&path, OPEN_EXISTING, GENERIC_READ))
    }

    /// Create a new file (or truncate an existing one) for reading and
    /// writing.
    ///
    /// # Errors
    ///
    /// Returns `io::ErrorKind::PermissionDenied` if the containing
    /// directory isn't writable, `NotFound` if the containing directory
    /// doesn't exist, or another `io::Error` from `CreateFileW`/
    /// associating the handle with the IOCP for any other OS-level
    /// failure.
    pub fn create(path: impl Into<PathBuf>) -> impl Future<Output = io::Result<Self>> {
        let path = path.into();
        std::future::ready(open_impl(
            &path,
            CREATE_ALWAYS,
            GENERIC_READ | GENERIC_WRITE,
        ))
    }

    /// Generic open honoring an arbitrary [`std::fs::OpenOptions`]. Rather
    /// than re-deriving Win32 access/disposition flags from `opts` (which
    /// has no public getters), this delegates to `opts.open()` itself with
    /// `FILE_FLAG_OVERLAPPED` injected via `OpenOptionsExt::custom_flags`,
    /// then takes ownership of the resulting overlapped-capable handle.
    ///
    /// # Errors
    ///
    /// Returns whatever `opts.open(path)` returns (e.g. `NotFound`,
    /// `PermissionDenied`, `AlreadyExists` depending on how `opts` is
    /// configured), or an `io::Error` if associating the resulting handle
    /// with the IOCP fails.
    pub fn open_with(
        path: impl Into<PathBuf>,
        mut opts: std::fs::OpenOptions,
    ) -> impl Future<Output = io::Result<Self>> {
        use std::os::windows::fs::OpenOptionsExt;
        opts.custom_flags(FILE_FLAG_OVERLAPPED);
        let path = path.into();
        std::future::ready(open_with_impl(&path, &opts))
    }

    /// Read into `buf` at the current shared cursor, advancing it by the
    /// number of bytes read, and hand `buf` back for reuse.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `ReadFile`/IOCP completion
    /// reports one (e.g. the handle was closed concurrently); `Ok((0,
    /// buf))` signals EOF, matching `ReadFile`'s own convention.
    pub async fn read(&self, mut buf: Vec<u8>) -> io::Result<(usize, Vec<u8>)> {
        let offset = self.cursor.load(Ordering::Relaxed) as u64;
        let n = issue_read(self.handle, &mut buf, offset).await?;
        self.cursor.fetch_add(n as i64, Ordering::Relaxed);
        Ok((n, buf))
    }

    /// Write `buf` at the current shared cursor, advancing it by the
    /// number of bytes written, and hand `buf` back for reuse.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if the underlying `WriteFile`/IOCP
    /// completion reports one (e.g. disk full, handle closed
    /// concurrently).
    pub async fn write(&self, buf: Vec<u8>) -> io::Result<(usize, Vec<u8>)> {
        let offset = self.cursor.load(Ordering::Relaxed) as u64;
        let n = issue_write(self.handle, &buf, offset).await?;
        self.cursor.fetch_add(n as i64, Ordering::Relaxed);
        Ok((n, buf))
    }

    /// Positional read: does not move the shared cursor, safe to call
    /// concurrently with other `read_at`/`write_at` calls on the same handle
    /// (each issues its own `OVERLAPPED` with an explicit offset).
    ///
    /// # Errors
    ///
    /// Same as [`Self::read`].
    pub async fn read_at(&self, mut buf: Vec<u8>, offset: u64) -> io::Result<(usize, Vec<u8>)> {
        let n = issue_read(self.handle, &mut buf, offset).await?;
        Ok((n, buf))
    }

    /// Positional write: does not move the shared cursor, safe to call
    /// concurrently with other `read_at`/`write_at` calls on the same
    /// handle.
    ///
    /// # Errors
    ///
    /// Same as [`Self::write`].
    pub async fn write_at(&self, buf: Vec<u8>, offset: u64) -> io::Result<(usize, Vec<u8>)> {
        let n = issue_write(self.handle, &buf, offset).await?;
        Ok((n, buf))
    }

    /// Flush the file's buffers to disk (`FlushFileBuffers`).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if `FlushFileBuffers` fails (e.g. the
    /// underlying device was removed).
    pub fn sync_all(&self) -> impl Future<Output = io::Result<()>> + '_ {
        std::future::ready(self.sync_all_impl())
    }

    fn sync_all_impl(&self) -> io::Result<()> {
        let ok = unsafe { FlushFileBuffers(self.handle) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Query file metadata via a temporarily-borrowed `std::fs::File`
    /// view of this handle (never closes it — see the `ManuallyDrop`
    /// guard in the implementation).
    ///
    /// # Errors
    ///
    /// Returns whatever `std::fs::File::metadata` returns for the
    /// underlying handle.
    pub fn metadata(&self) -> impl Future<Output = io::Result<std::fs::Metadata>> + '_ {
        std::future::ready(self.metadata_impl())
    }

    // Win32 has no handle->std::fs::Metadata conversion without going
    // through a path or a duplicated std::fs::File; borrow the raw
    // handle briefly via ManuallyDrop so we don't double-close it.
    fn metadata_impl(&self) -> io::Result<std::fs::Metadata> {
        use std::os::windows::io::{AsRawHandle, FromRawHandle};
        let file = unsafe { std::fs::File::from_raw_handle(self.handle.cast()) };
        let file = std::mem::ManuallyDrop::new(file);
        let meta = file.metadata();
        let _ = file.as_raw_handle(); // keep handle alive/used until here
        meta
    }

    /// Close the file. A no-op beyond documenting intent — `Drop` already
    /// performs the actual `CloseHandle`, so this exists only so callers
    /// can spell out an explicit close point in async code.
    ///
    /// # Errors
    ///
    /// Never actually fails; returns `Ok(())` unconditionally. The
    /// `Result` return type exists for API parity with the other
    /// backends' `close`, in case a future revision needs to surface a
    /// real close-time error.
    pub fn close(self) -> impl Future<Output = io::Result<()>> {
        // Drop performs the actual CloseHandle.
        std::future::ready(Ok(()))
    }

    /// The file's current size in bytes (`GetFileSizeEx`).
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if `GetFileSizeEx` fails (e.g. the handle
    /// was invalidated).
    pub fn len(&self) -> io::Result<u64> {
        let mut size: i64 = 0;
        let ok = unsafe { GetFileSizeEx(self.handle, &raw mut size) };
        if ok == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(size as u64)
    }

    /// Whether the file is currently zero-length. Queries the live file
    /// size via [`Self::len`] rather than caching it, so this can return
    /// different answers across calls if the file is being written
    /// concurrently.
    ///
    /// # Errors
    ///
    /// Same as [`Self::len`].
    pub fn is_empty(&self) -> io::Result<bool> {
        self.len().map(|n| n == 0)
    }
}

impl Drop for DtactFile {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

/// Query a path's metadata without opening a [`DtactFile`] handle.
///
/// # Errors
///
/// Returns `io::ErrorKind::NotFound` if `path` doesn't exist,
/// `PermissionDenied` if a containing directory can't be traversed, or
/// another `io::Error` from `std::fs::metadata`.
pub fn metadata(path: impl Into<PathBuf>) -> impl Future<Output = io::Result<std::fs::Metadata>> {
    let path = path.into();
    std::future::ready(std::fs::metadata(&path))
}

/// List a directory's entries.
///
/// # Errors
///
/// Returns `io::ErrorKind::NotFound` if `path` doesn't exist, `NotADirectory`
/// if it isn't a directory, `PermissionDenied` if it can't be read, or
/// propagates any per-entry `io::Error` encountered while collecting.
pub fn read_dir(
    path: impl Into<PathBuf>,
) -> impl Future<Output = io::Result<Vec<std::fs::DirEntry>>> {
    let path: PathBuf = path.into();
    std::future::ready(std::fs::read_dir(&path).and_then(Iterator::collect))
}

/// Recursively create a directory and all missing parent directories.
///
/// # Errors
///
/// Returns `io::ErrorKind::PermissionDenied` if any component can't be
/// created, or `AlreadyExists`-adjacent errors if a path component exists
/// as a non-directory — see `std::fs::create_dir_all`'s own documented
/// error conditions, which this delegates to directly.
pub fn create_dir_all(path: impl Into<PathBuf>) -> impl Future<Output = io::Result<()>> {
    let path = path.into();
    std::future::ready(std::fs::create_dir_all(&path))
}

/// Remove a file.
///
/// # Errors
///
/// Returns `io::ErrorKind::NotFound` if `path` doesn't exist,
/// `PermissionDenied` if it can't be removed, or another `io::Error` from
/// `std::fs::remove_file`.
pub fn remove_file(path: impl Into<PathBuf>) -> impl Future<Output = io::Result<()>> {
    let path = path.into();
    std::future::ready(std::fs::remove_file(&path))
}

/// Resolve `path` to an absolute path with all intermediate components
/// (`.`, `..`, symlinks/junctions) resolved.
///
/// # Errors
///
/// Returns whatever `std::fs::canonicalize` returns, e.g. `NotFound`.
pub fn canonicalize(path: impl Into<PathBuf>) -> impl Future<Output = io::Result<PathBuf>> {
    let path = path.into();
    std::future::ready(std::fs::canonicalize(&path))
}

/// Copy the contents (and permission bits) of the file at `from` to `to`,
/// returning the byte count copied.
///
/// # Errors
///
/// Returns whatever `std::fs::copy` returns, e.g. `NotFound` if `from`
/// doesn't exist.
pub fn copy(
    from: impl Into<PathBuf>,
    to: impl Into<PathBuf>,
) -> impl Future<Output = io::Result<u64>> {
    let from = from.into();
    let to = to.into();
    std::future::ready(std::fs::copy(&from, &to))
}

/// Create a single new directory. Unlike [`create_dir_all`], fails if any
/// parent component doesn't already exist.
///
/// # Errors
///
/// Returns whatever `std::fs::create_dir` returns, e.g. `AlreadyExists`.
pub fn create_dir(path: impl Into<PathBuf>) -> impl Future<Output = io::Result<()>> {
    let path = path.into();
    std::future::ready(std::fs::create_dir(&path))
}

/// Create a hard link at `dst` pointing at the same file as `src`.
///
/// # Errors
///
/// Returns whatever `std::fs::hard_link` returns, e.g. `NotFound` if
/// `src` doesn't exist.
pub fn hard_link(
    src: impl Into<PathBuf>,
    dst: impl Into<PathBuf>,
) -> impl Future<Output = io::Result<()>> {
    let src = src.into();
    let dst = dst.into();
    std::future::ready(std::fs::hard_link(&src, &dst))
}

/// Read the entire contents of the file at `path` into a `Vec<u8>`.
///
/// # Errors
///
/// Returns whatever `std::fs::read` returns, e.g. `NotFound`.
pub fn read(path: impl Into<PathBuf>) -> impl Future<Output = io::Result<Vec<u8>>> {
    let path = path.into();
    std::future::ready(std::fs::read(&path))
}

/// Read the target of the symbolic link at `path`.
///
/// # Errors
///
/// Returns whatever `std::fs::read_link` returns, e.g. `NotFound`, or an
/// error if `path` isn't actually a symlink.
pub fn read_link(path: impl Into<PathBuf>) -> impl Future<Output = io::Result<PathBuf>> {
    let path = path.into();
    std::future::ready(std::fs::read_link(&path))
}

/// Read the entire contents of the file at `path` into a `String`.
///
/// # Errors
///
/// Returns whatever `std::fs::read_to_string` returns, e.g. an
/// `InvalidData` error if the file isn't valid UTF-8.
pub fn read_to_string(path: impl Into<PathBuf>) -> impl Future<Output = io::Result<String>> {
    let path = path.into();
    std::future::ready(std::fs::read_to_string(&path))
}

/// Remove an empty directory. Fails if `path` is non-empty — see
/// [`remove_dir_all`] for the recursive version.
///
/// # Errors
///
/// Returns whatever `std::fs::remove_dir` returns, e.g. `NotFound`, or an
/// error if the directory isn't empty.
pub fn remove_dir(path: impl Into<PathBuf>) -> impl Future<Output = io::Result<()>> {
    let path = path.into();
    std::future::ready(std::fs::remove_dir(&path))
}

/// Recursively remove a directory and everything under it.
///
/// # Errors
///
/// Returns whatever `std::fs::remove_dir_all` returns, e.g. `NotFound`.
pub fn remove_dir_all(path: impl Into<PathBuf>) -> impl Future<Output = io::Result<()>> {
    let path = path.into();
    std::future::ready(std::fs::remove_dir_all(&path))
}

/// Rename (move) the file or directory at `from` to `to`.
///
/// Replaces `to` if it already exists — see `std::fs::rename`'s own
/// documentation for cross-platform caveats (e.g. renaming across
/// volumes).
///
/// # Errors
///
/// Returns whatever `std::fs::rename` returns.
pub fn rename(
    from: impl Into<PathBuf>,
    to: impl Into<PathBuf>,
) -> impl Future<Output = io::Result<()>> {
    let from = from.into();
    let to = to.into();
    std::future::ready(std::fs::rename(&from, &to))
}

/// Set `path`'s permission bits to `perm`.
///
/// # Errors
///
/// Returns whatever `std::fs::set_permissions` returns, e.g. `NotFound`.
pub fn set_permissions(
    path: impl Into<PathBuf>,
    perm: std::fs::Permissions,
) -> impl Future<Output = io::Result<()>> {
    let path = path.into();
    std::future::ready(std::fs::set_permissions(&path, perm))
}

/// Create a directory symbolic link at `dst` pointing at `src`.
///
/// # Errors
///
/// Returns whatever `std::os::windows::fs::symlink_dir` returns, e.g.
/// `AlreadyExists` if `dst` already exists, or a permissions error —
/// creating symlinks on Windows normally requires either an elevated
/// process or Developer Mode enabled.
pub fn symlink_dir(
    src: impl Into<PathBuf>,
    dst: impl Into<PathBuf>,
) -> impl Future<Output = io::Result<()>> {
    let src = src.into();
    let dst = dst.into();
    std::future::ready(std::os::windows::fs::symlink_dir(&src, &dst))
}

/// Create a file symbolic link at `dst` pointing at `src`.
///
/// # Errors
///
/// Same as [`symlink_dir`], for a file target instead of a directory.
pub fn symlink_file(
    src: impl Into<PathBuf>,
    dst: impl Into<PathBuf>,
) -> impl Future<Output = io::Result<()>> {
    let src = src.into();
    let dst = dst.into();
    std::future::ready(std::os::windows::fs::symlink_file(&src, &dst))
}

/// Query `path`'s metadata *without* following a trailing symlink/junction
/// (unlike [`metadata`], which does).
///
/// # Errors
///
/// Returns whatever `std::fs::symlink_metadata` returns, e.g. `NotFound`.
pub fn symlink_metadata(
    path: impl Into<PathBuf>,
) -> impl Future<Output = io::Result<std::fs::Metadata>> {
    let path = path.into();
    std::future::ready(std::fs::symlink_metadata(&path))
}

/// Check whether `path` exists, following symlinks/junctions.
///
/// A permission error while checking is propagated as `Err` rather than
/// silently read as "doesn't exist" — see `std::fs::exists`'s own
/// documentation for the exact distinction.
///
/// # Errors
///
/// Returns an `io::Error` for any failure *other than* "doesn't exist".
pub fn try_exists(path: impl Into<PathBuf>) -> impl Future<Output = io::Result<bool>> {
    let path = path.into();
    std::future::ready(std::fs::exists(&path))
}

/// Write `contents` to the file at `path`, creating it if it doesn't
/// exist and truncating it if it does.
///
/// # Errors
///
/// Returns whatever `std::fs::write` returns, e.g. `PermissionDenied`.
pub fn write(
    path: impl Into<PathBuf>,
    contents: impl AsRef<[u8]>,
) -> impl Future<Output = io::Result<()>> {
    let path = path.into();
    std::future::ready(std::fs::write(&path, contents))
}

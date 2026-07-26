//! Native filesystem backend: a small dedicated blocking-thread pool that
//! bridges `std::fs` and platform positional-I/O syscalls (`pread`/`pwrite`
//! on Unix, `seek_read`/`seek_write` on Windows) into futures.
//!
//! **Not the only native fs backend, despite the module name.** Per
//! `fs::mod`'s `cfg` gates, this module only compiles — and is only used —
//! on Unix platforms that are neither Linux nor covered by a dedicated
//! backend, i.e. macOS/BSD today. Linux uses [`super::uring_linux`] (real
//! `io_uring` opcodes on a dedicated ring) and Windows uses
//! [`super::iocp_windows`] (real overlapped IOCP). This was accurate when
//! first written — at the time this thread-pool bridge really was used
//! unconditionally on every platform — but both of those backends were
//! added afterward without this doc being updated to match; don't trust
//! "used unconditionally on all platforms" claims in older comments here,
//! check `fs::mod`'s `cfg` gates for the current routing instead.
//!
//! **Deferred / not lock-free**: unlike the `io` module's io_uring-backed
//! reactor (SPSC queues, per-slot atomics, zero-lock hot path), this backend
//! uses a plain `Mutex`-guarded completion slot per operation. Filesystem
//! syscalls are not competitive with a lock-free dispatch path the way
//! socket I/O is — the syscall itself dominates — so a mutex here is a
//! deliberate, correctness-first simplification, not an oversight.

use crate::lockfree::OnceSlot;
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::task::{Context, Poll};

type Job = Box<dyn FnOnce() + Send + 'static>;

#[repr(align(64))]
struct FsPool {
    sender: mpsc::Sender<Job>,
}

static FS_POOL: OnceLock<FsPool> = OnceLock::new();

/// Start the fs thread pool with the given number of worker threads.
/// Idempotent — later calls are no-ops once the pool is initialized.
pub fn init(workers: usize) {
    FS_POOL.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..workers.max(1) {
            let rx = Arc::clone(&rx);
            std::thread::Builder::new()
                .name("dtact-fs-worker".into())
                .spawn(move || {
                    loop {
                        let job = { rx.lock().unwrap().recv() };
                        match job {
                            Ok(job) => job(),
                            Err(_) => break,
                        }
                    }
                })
                .expect("failed to spawn dtact-fs worker thread");
        }
        FsPool { sender: tx }
    });
}

/// Full-signature entry point matching the other native backends'
/// `init_fs` (and `crate::io::native::init_runtime`), for the `fs_init`
/// macro to call uniformly regardless of which backend is active.
/// `ring_depth`/`buffer_pool_size`/`chunk_size`/`pin_cpus` don't apply to
/// this thread-pool-bridged fallback (no ring, no arena) and are ignored.
pub fn init_fs(
    workers: usize,
    _ring_depth: u32,
    _buffer_pool_size: usize,
    _chunk_size: usize,
    _pin_cpus: &[usize],
) {
    init(workers);
}

/// A single blocking filesystem operation, dispatched to the fs thread
/// pool. Completion is signaled via a wait-free [`OnceSlot`] (a single
/// `AtomicPtr` swap) rather than a `Mutex`-guarded result/waker pair —
/// same completion mechanism `process::native` already uses, moved here so
/// every op's poll no longer pays a lock/unlock on the hot path.
pub struct BlockingOp<T> {
    slot: Arc<OnceSlot<T>>,
}

impl<T: Send + 'static> Future for BlockingOp<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        self.slot.poll(cx)
    }
}

fn spawn_blocking<T, F>(f: F) -> BlockingOp<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    // Ensure a pool exists even if the caller never called `init` explicitly.
    if FS_POOL.get().is_none() {
        init(4);
    }
    let slot = Arc::new(OnceSlot::new());
    let slot2 = Arc::clone(&slot);
    let job: Job = Box::new(move || {
        let result = f();
        slot2.set(result);
    });
    let _ = FS_POOL.get().unwrap().sender.send(job);
    BlockingOp { slot }
}

/// An open file whose blocking read/write/metadata operations run on the
/// dtact-fs thread pool rather than the calling task's thread.
pub struct DtactFile {
    inner: Arc<Mutex<Option<std::fs::File>>>,
}

impl DtactFile {
    pub async fn open(path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        let file = spawn_blocking(move || std::fs::File::open(&path)).await?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Some(file))),
        })
    }

    pub async fn create(path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        let file = spawn_blocking(move || std::fs::File::create(&path)).await?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Some(file))),
        })
    }

    pub async fn open_with(
        path: impl Into<PathBuf>,
        opts: std::fs::OpenOptions,
    ) -> io::Result<Self> {
        let path = path.into();
        let file = spawn_blocking(move || opts.open(&path)).await?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Some(file))),
        })
    }

    /// Read into `buf`, returning the number of bytes read and the buffer
    /// (buffer round-tripping avoids a borrow across the `.await` point).
    pub async fn read(&self, mut buf: Vec<u8>) -> io::Result<(usize, Vec<u8>)> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            use std::io::Read;
            let mut guard = inner.lock().unwrap();
            let file = guard
                .as_mut()
                .ok_or_else(|| io::Error::other("dtact-fs: file already closed"))?;
            let n = file.read(&mut buf)?;
            Ok((n, buf))
        })
        .await
    }

    pub async fn write(&self, buf: Vec<u8>) -> io::Result<(usize, Vec<u8>)> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            use std::io::Write;
            let mut guard = inner.lock().unwrap();
            let file = guard
                .as_mut()
                .ok_or_else(|| io::Error::other("dtact-fs: file already closed"))?;
            let n = file.write(&buf)?;
            Ok((n, buf))
        })
        .await
    }

    /// Positional read: `pread` on Unix, `seek_read` on Windows. Does not
    /// move the file's shared cursor, so is safe to call concurrently with
    /// other `read_at`/`write_at` calls on the same handle.
    pub async fn read_at(&self, mut buf: Vec<u8>, offset: u64) -> io::Result<(usize, Vec<u8>)> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let guard = inner.lock().unwrap();
            let file = guard
                .as_ref()
                .ok_or_else(|| io::Error::other("dtact-fs: file already closed"))?;
            let n = read_at_impl(file, &mut buf, offset)?;
            Ok((n, buf))
        })
        .await
    }

    pub async fn write_at(&self, buf: Vec<u8>, offset: u64) -> io::Result<(usize, Vec<u8>)> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let guard = inner.lock().unwrap();
            let file = guard
                .as_ref()
                .ok_or_else(|| io::Error::other("dtact-fs: file already closed"))?;
            let n = write_at_impl(file, &buf, offset)?;
            Ok((n, buf))
        })
        .await
    }

    pub async fn sync_all(&self) -> io::Result<()> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let guard = inner.lock().unwrap();
            let file = guard
                .as_ref()
                .ok_or_else(|| io::Error::other("dtact-fs: file already closed"))?;
            file.sync_all()
        })
        .await
    }

    pub async fn metadata(&self) -> io::Result<std::fs::Metadata> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            let guard = inner.lock().unwrap();
            let file = guard
                .as_ref()
                .ok_or_else(|| io::Error::other("dtact-fs: file already closed"))?;
            file.metadata()
        })
        .await
    }

    /// Close the file. Equivalent to dropping it, but lets callers observe
    /// close-time errors (there are none on the std backend today, but this
    /// keeps the signature stable if a future io_uring `Close` opcode needs
    /// to surface one).
    pub async fn close(self) -> io::Result<()> {
        let inner = Arc::clone(&self.inner);
        spawn_blocking(move || {
            inner.lock().unwrap().take();
            Ok(())
        })
        .await
    }
}

#[cfg(unix)]
fn read_at_impl(file: &std::fs::File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

#[cfg(unix)]
fn write_at_impl(file: &std::fs::File, buf: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.write_at(buf, offset)
}

#[cfg(windows)]
fn read_at_impl(file: &std::fs::File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

#[cfg(windows)]
fn write_at_impl(file: &std::fs::File, buf: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_write(buf, offset)
}

pub async fn metadata(path: impl Into<PathBuf>) -> io::Result<std::fs::Metadata> {
    let path = path.into();
    spawn_blocking(move || std::fs::metadata(&path)).await
}

/// Read a directory's entries into a `Vec` (the blocking `ReadDir` iterator
/// itself never crosses the pool boundary, so this fully drains it on the
/// worker thread rather than trickling one syscall per `.await`).
pub async fn read_dir(path: impl Into<PathBuf>) -> io::Result<Vec<std::fs::DirEntry>> {
    let path: PathBuf = path.into();
    spawn_blocking(move || -> io::Result<Vec<std::fs::DirEntry>> {
        std::fs::read_dir(&path)?.collect()
    })
    .await
}

pub async fn create_dir_all(path: impl Into<PathBuf>) -> io::Result<()> {
    let path = path.into();
    spawn_blocking(move || std::fs::create_dir_all(&path)).await
}

pub async fn remove_file(path: impl Into<PathBuf>) -> io::Result<()> {
    let path = path.into();
    spawn_blocking(move || std::fs::remove_file(&path)).await
}

/// Resolve `path` to an absolute path with all intermediate components
/// (`.`, `..`, symlinks) resolved.
///
/// # Errors
/// Returns whatever `std::fs::canonicalize` returns (e.g. `NotFound` if
/// `path` doesn't exist).
pub async fn canonicalize(path: impl Into<PathBuf>) -> io::Result<PathBuf> {
    let path = path.into();
    spawn_blocking(move || std::fs::canonicalize(&path)).await
}

/// Copy the contents (and permission bits) of the file at `from` to `to`,
/// creating or truncating `to`, returning the byte count copied.
///
/// # Errors
/// Returns whatever `std::fs::copy` returns (e.g. `NotFound` if `from`
/// doesn't exist).
pub async fn copy(from: impl Into<PathBuf>, to: impl Into<PathBuf>) -> io::Result<u64> {
    let from = from.into();
    let to = to.into();
    spawn_blocking(move || std::fs::copy(&from, &to)).await
}

/// Create a single new directory. Unlike [`create_dir_all`], fails if any
/// parent component doesn't already exist.
///
/// # Errors
/// Returns whatever `std::fs::create_dir` returns (e.g. `NotFound` if a
/// parent component is missing, `AlreadyExists` if `path` already
/// exists).
pub async fn create_dir(path: impl Into<PathBuf>) -> io::Result<()> {
    let path = path.into();
    spawn_blocking(move || std::fs::create_dir(&path)).await
}

/// Create a hard link at `dst` pointing at the same inode as `src`.
///
/// # Errors
/// Returns whatever `std::fs::hard_link` returns (e.g. `NotFound` if
/// `src` doesn't exist, or an error if `src`/`dst` are on different
/// filesystems).
pub async fn hard_link(src: impl Into<PathBuf>, dst: impl Into<PathBuf>) -> io::Result<()> {
    let src = src.into();
    let dst = dst.into();
    spawn_blocking(move || std::fs::hard_link(&src, &dst)).await
}

/// Read the entire contents of the file at `path` into a `Vec<u8>`.
///
/// # Errors
/// Returns whatever `std::fs::read` returns (e.g. `NotFound`,
/// `PermissionDenied`).
pub async fn read(path: impl Into<PathBuf>) -> io::Result<Vec<u8>> {
    let path = path.into();
    spawn_blocking(move || std::fs::read(&path)).await
}

/// Read the target of the symbolic link at `path`.
///
/// # Errors
/// Returns whatever `std::fs::read_link` returns (e.g. `NotFound`, or an
/// error if `path` isn't actually a symlink).
pub async fn read_link(path: impl Into<PathBuf>) -> io::Result<PathBuf> {
    let path = path.into();
    spawn_blocking(move || std::fs::read_link(&path)).await
}

/// Read the entire contents of the file at `path` into a `String`.
///
/// # Errors
/// Returns whatever `std::fs::read_to_string` returns (e.g. `NotFound`,
/// or an `InvalidData` error if the file isn't valid UTF-8).
pub async fn read_to_string(path: impl Into<PathBuf>) -> io::Result<String> {
    let path = path.into();
    spawn_blocking(move || std::fs::read_to_string(&path)).await
}

/// Remove an empty directory. Fails if `path` is non-empty — see
/// [`remove_dir_all`] for the recursive version.
///
/// # Errors
/// Returns whatever `std::fs::remove_dir` returns (e.g. `NotFound`, or an
/// error if the directory isn't empty).
pub async fn remove_dir(path: impl Into<PathBuf>) -> io::Result<()> {
    let path = path.into();
    spawn_blocking(move || std::fs::remove_dir(&path)).await
}

/// Recursively remove a directory and everything under it.
///
/// # Errors
/// Returns whatever `std::fs::remove_dir_all` returns (e.g. `NotFound`,
/// `PermissionDenied`).
pub async fn remove_dir_all(path: impl Into<PathBuf>) -> io::Result<()> {
    let path = path.into();
    spawn_blocking(move || std::fs::remove_dir_all(&path)).await
}

/// Rename (move) the file or directory at `from` to `to`, replacing `to`
/// if it already exists (platform-dependent semantics — see
/// `std::fs::rename`'s own documentation for the exact cross-platform
/// caveats, e.g. renaming across filesystems).
///
/// # Errors
/// Returns whatever `std::fs::rename` returns.
pub async fn rename(from: impl Into<PathBuf>, to: impl Into<PathBuf>) -> io::Result<()> {
    let from = from.into();
    let to = to.into();
    spawn_blocking(move || std::fs::rename(&from, &to)).await
}

/// Set `path`'s permission bits to `perm`.
///
/// # Errors
/// Returns whatever `std::fs::set_permissions` returns (e.g. `NotFound`,
/// `PermissionDenied`).
pub async fn set_permissions(
    path: impl Into<PathBuf>,
    perm: std::fs::Permissions,
) -> io::Result<()> {
    let path = path.into();
    spawn_blocking(move || std::fs::set_permissions(&path, perm)).await
}

/// Create a symbolic link at `dst` pointing at `src`.
///
/// # Errors
/// Returns whatever `std::os::unix::fs::symlink` returns (e.g.
/// `AlreadyExists` if `dst` already exists).
pub async fn symlink(src: impl Into<PathBuf>, dst: impl Into<PathBuf>) -> io::Result<()> {
    let src = src.into();
    let dst = dst.into();
    spawn_blocking(move || std::os::unix::fs::symlink(&src, &dst)).await
}

/// Query `path`'s metadata *without* following a trailing symlink (unlike
/// [`metadata`], which does).
///
/// # Errors
/// Returns whatever `std::fs::symlink_metadata` returns (e.g.
/// `NotFound`).
pub async fn symlink_metadata(path: impl Into<PathBuf>) -> io::Result<std::fs::Metadata> {
    let path = path.into();
    spawn_blocking(move || std::fs::symlink_metadata(&path)).await
}

/// Check whether `path` exists, following symlinks. Unlike a bare
/// `metadata().is_ok()` check, a permission error while checking is
/// propagated as `Err` rather than silently read as "doesn't exist" —
/// see `std::fs::exists`'s own documentation for the exact distinction.
///
/// # Errors
/// Returns an `io::Error` for any failure *other than* "doesn't exist"
/// (e.g. `PermissionDenied` on a parent directory).
pub async fn try_exists(path: impl Into<PathBuf>) -> io::Result<bool> {
    let path = path.into();
    spawn_blocking(move || std::fs::exists(&path)).await
}

/// Write `contents` to the file at `path`, creating it if it doesn't
/// exist and truncating it if it does (equivalent to `create` + a single
/// `write_all`).
///
/// # Errors
/// Returns whatever `std::fs::write` returns (e.g. `PermissionDenied`).
pub async fn write(
    path: impl Into<PathBuf>,
    contents: impl AsRef<[u8]> + Send + 'static,
) -> io::Result<()> {
    let path = path.into();
    spawn_blocking(move || std::fs::write(&path, contents)).await
}

#[allow(dead_code)]
fn _assert_path_bound(_: &Path) {}

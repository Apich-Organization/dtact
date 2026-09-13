//! dtact-io native Unix-domain-socket *persistent-connection round-trip*
//! vs `tokio::net::UnixStream` benchmark — the Unix-socket counterpart to
//! `tcp_roundtrip.rs` (see its module doc for the ping-pong-vs-bulk-echo
//! rationale, identical here).
//!
//! Unix-only, matching `DtactUnixStream`/`DtactUnixListener`'s own
//! `cfg(unix)` gate. Every item below is individually `cfg`-gated (rather
//! than nested in one `#[cfg(...)] mod`) because `criterion_main!`
//! expands to a `fn main` that must live at the crate root to be found as
//! the binary's actual entry point — a private `fn main` nested in a
//! module doesn't count, and re-exporting it runs into the same privacy
//! wall. Builds and runs a no-op on Windows instead of failing the build.
//!
//! Run:  cargo bench --bench unix_socket_roundtrip --features native
//! Test: cargo bench --bench unix_socket_roundtrip --features native -- --test

#[cfg(unix)]
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
#[cfg(unix)]
use dtact_util::io::{DtactUnixListener, DtactUnixStream, init_runtime, shutdown_runtime};
#[cfg(unix)]
use std::hint::black_box;
#[cfg(unix)]
use std::os::unix::net::UnixListener;
#[cfg(unix)]
use std::path::PathBuf;
#[cfg(unix)]
use std::sync::Arc;
#[cfg(unix)]
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

#[cfg(unix)]
static INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();

#[cfg(unix)]
fn ensure_runtimes() {
    INIT.get_or_init(|| {
        let _ = dtact::GLOBAL_RUNTIME.get_or_init(|| {
            let sched = dtact::dta_scheduler::DtaScheduler::new(
                2,
                dtact::dta_scheduler::TopologyMode::P2PMesh,
            );
            let pool = dtact::memory_management::ContextPool::new(
                4096,
                64 * 1024,
                dtact::memory_management::SafetyLevel::Safety0,
                0,
                2,
            )
            .expect("dtact pool init failed");
            dtact::Runtime {
                scheduler: sched,
                pool,
                started: core::sync::atomic::AtomicBool::new(false),
                shutdown: core::sync::atomic::AtomicBool::new(false),
            }
        });
        if let Some(rt) = dtact::GLOBAL_RUNTIME.get() {
            rt.start();
        }
        init_runtime(1, 4096, 16_384, 4096, &[]);
    });
}

#[cfg(unix)]
fn fresh_socket_path(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "dtact-uds-bench-{tag}-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = std::fs::remove_file(&path);
    path
}

/// Persistent dtact-io echo server: accepts one connection and echoes
/// each `plen`-byte request back on the same stream, forever.
#[cfg(unix)]
fn spawn_dtact_server(plen: usize, stop: Arc<AtomicBool>) -> PathBuf {
    let path = fresh_socket_path("dtact");
    let listener = UnixListener::bind(&path).unwrap();
    let dtact_listener = Arc::new(DtactUnixListener::from_std(listener).unwrap());

    dtact::spawn(async move {
        while !stop.load(Ordering::Relaxed) {
            if let Ok((stream, _)) = dtact_listener.accept().await {
                let mut buf = vec![0u8; plen];
                'conn: while !stop.load(Ordering::Relaxed) {
                    let mut pos = 0usize;
                    while pos < plen {
                        match stream.read(&mut buf[pos..]).await {
                            Ok(0) | Err(_) => break 'conn,
                            Ok(n) => pos += n,
                        }
                    }
                    let mut sent = 0usize;
                    while sent < pos {
                        match stream.write(&buf[sent..pos]).await {
                            Ok(0) | Err(_) => break 'conn,
                            Ok(n) => sent += n,
                        }
                    }
                }
            }
        }
    });

    path
}

/// Persistent tokio echo server, same shape.
#[cfg(unix)]
fn spawn_tokio_server(rt: &tokio::runtime::Runtime, plen: usize, stop: Arc<AtomicBool>) -> PathBuf {
    let path = fresh_socket_path("tokio");
    let listener = UnixListener::bind(&path).unwrap();
    listener.set_nonblocking(true).unwrap();

    let path2 = path.clone();
    rt.spawn(async move {
        let listener = tokio::net::UnixListener::from_std(listener).unwrap();
        while !stop.load(Ordering::Relaxed) {
            if let Ok((mut stream, _)) = listener.accept().await {
                let stop = stop.clone();
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = vec![0u8; plen];
                    'conn: while !stop.load(Ordering::Relaxed) {
                        let mut pos = 0usize;
                        while pos < plen {
                            match stream.read(&mut buf[pos..]).await {
                                Ok(0) | Err(_) => break 'conn,
                                Ok(n) => pos += n,
                            }
                        }
                        if stream.write_all(&buf[..pos]).await.is_err() {
                            break 'conn;
                        }
                    }
                });
            }
        }
        let _ = std::fs::remove_file(&path2);
    });

    path
}

#[cfg(unix)]
fn bench_roundtrip(c: &mut Criterion) {
    ensure_runtimes();

    let tokio_rt = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_io()
            .build()
            .unwrap(),
    );

    for &size in &[64usize, 1_024, 8_192] {
        let label = match size {
            s if s < 1024 => format!("{s} B"),
            s => format!("{} KB", s / 1024),
        };
        let payload = vec![0xABu8; size];

        let mut group = c.benchmark_group(format!("Unix Socket Roundtrip ({label})"));
        group.throughput(Throughput::Bytes(size as u64));

        // ── dtact-io native ───────────────────────────────────────────────
        {
            let stop = Arc::new(AtomicBool::new(false));
            let server_path = spawn_dtact_server(size, stop.clone());
            std::thread::sleep(std::time::Duration::from_millis(20));

            group.bench_with_input(BenchmarkId::new("dtact-io", &label), &payload, |b, p| {
                // One persistent connection reused across every batch.
                let connected = Arc::new(std::sync::Mutex::new(None::<Arc<DtactUnixStream>>));
                let c2 = connected.clone();
                let done = Arc::new(AtomicU8::new(0));
                let d2 = done.clone();
                let path = server_path.clone();
                dtact::spawn(async move {
                    let s = DtactUnixStream::connect(&path).await.unwrap();
                    *c2.lock().unwrap() = Some(Arc::new(s));
                    d2.store(1, Ordering::Release);
                });
                while done.load(Ordering::Acquire) == 0 {
                    std::hint::spin_loop();
                }
                let stream = connected.lock().unwrap().clone().unwrap();

                b.iter_custom(|iters| {
                    let done = Arc::new(AtomicU8::new(0));
                    let done2 = done.clone();
                    let pdata = p.to_vec();
                    let stream = stream.clone();
                    let start = std::time::Instant::now();
                    dtact::spawn(async move {
                        let mut buf = vec![0u8; pdata.len()];
                        for _ in 0..iters {
                            let mut sent = 0usize;
                            while sent < pdata.len() {
                                match stream.write(&pdata[sent..]).await {
                                    Ok(0) | Err(_) => break,
                                    Ok(n) => sent += n,
                                }
                            }
                            let mut pos = 0usize;
                            while pos < pdata.len() {
                                match stream.read(&mut buf[pos..]).await {
                                    Ok(0) | Err(_) => break,
                                    Ok(n) => pos += n,
                                }
                            }
                            black_box(&buf[..pos]);
                        }
                        done2.store(1, Ordering::Release);
                    });
                    while done.load(Ordering::Acquire) == 0 {
                        std::hint::spin_loop();
                    }
                    start.elapsed()
                });
            });

            stop.store(true, Ordering::Release);
            let _ = std::os::unix::net::UnixStream::connect(&server_path);
            std::thread::sleep(std::time::Duration::from_millis(30));
            let _ = std::fs::remove_file(&server_path);
        }

        // ── raw tokio ─────────────────────────────────────────────────────
        {
            let stop = Arc::new(AtomicBool::new(false));
            let server_path = spawn_tokio_server(&tokio_rt, size, stop.clone());
            std::thread::sleep(std::time::Duration::from_millis(20));

            let tokio_rt2 = tokio_rt.clone();
            group.bench_with_input(BenchmarkId::new("tokio", &label), &payload, move |b, p| {
                let rt = tokio_rt2.clone();
                let path = server_path.clone();
                let stream = Arc::new(rt.block_on(async move {
                    let s = tokio::net::UnixStream::connect(&path).await.unwrap();
                    tokio::sync::Mutex::new(s)
                }));
                b.iter_custom(|iters| {
                    let pdata = p.to_vec();
                    let rt = tokio_rt2.clone();
                    let stream = stream.clone();
                    let start = std::time::Instant::now();
                    rt.block_on(async move {
                        use tokio::io::{AsyncReadExt, AsyncWriteExt};
                        let mut s = stream.lock().await;
                        let mut buf = vec![0u8; pdata.len()];
                        for _ in 0..iters {
                            let _ = s.write_all(&pdata).await;
                            let mut pos = 0usize;
                            while pos < pdata.len() {
                                match s.read(&mut buf[pos..]).await {
                                    Ok(0) | Err(_) => break,
                                    Ok(n) => pos += n,
                                }
                            }
                            black_box(&buf[..pos]);
                        }
                    });
                    start.elapsed()
                });
            });

            stop.store(true, Ordering::Release);
            std::thread::sleep(std::time::Duration::from_millis(30));
        }

        group.finish();
    }

    shutdown_runtime();
}

#[cfg(unix)]
criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(30);
    targets = bench_roundtrip
);

#[cfg(unix)]
criterion_main!(benches);

#[cfg(not(unix))]
fn main() {
    eprintln!(
        "unix_socket_roundtrip bench is Unix-only (Windows has no Unix-domain-socket \
         analogue) — skipping."
    );
}

//! dtact-io native TCP *persistent-connection round-trip* vs
//! `tokio::net::TcpStream` benchmark.
//!
//! Unlike `io_performance.rs` (which reconnects every iteration and only
//! ever tests payloads >= 1 KiB), this benchmark connects **once** and then
//! loops `write` + `read` in a tight `b.iter_custom` — a strict ping-pong,
//! exactly like `udp_performance.rs`'s roundtrip test — across the same
//! small payload sizes the UDP benchmark uses (64 B, 1 KiB, 8 KiB).
//!
//! This isolates the per-op dispatch overhead of a single outstanding
//! request/response from the bulk-transfer amortisation the echo benchmark
//! measures, and makes the TCP and UDP numbers directly comparable.
//!
//! Run:  cargo bench --bench tcp_roundtrip --features native
//! Test: cargo bench --bench tcp_roundtrip --features native -- --test

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dtact_util::io::{DtactTcpListener, DtactTcpStream, init_runtime, shutdown_runtime};
use std::hint::black_box;
use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

static INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();

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

/// Persistent dtact-io echo server: accepts one connection and echoes each
/// `plen`-byte request back on the same stream, forever.
fn spawn_dtact_server(plen: usize, stop: Arc<AtomicBool>) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let dtact_listener = Arc::new(DtactTcpListener::from_std(listener).unwrap());

    dtact::spawn(async move {
        while !stop.load(Ordering::Relaxed) {
            if let Ok((stream, _)) = dtact_listener.accept().await {
                let mut buf = vec![0u8; plen];
                // Echo loop on this persistent connection.
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

    addr
}

/// Persistent tokio echo server, same shape.
fn spawn_tokio_server(
    rt: &tokio::runtime::Runtime,
    plen: usize,
    stop: Arc<AtomicBool>,
) -> SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let addr = listener.local_addr().unwrap();

    rt.spawn(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap();
        while !stop.load(Ordering::Relaxed) {
            if let Ok((mut stream, _)) = listener.accept().await {
                let stop = stop.clone();
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    stream.set_nodelay(true).ok();
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
    });

    addr
}

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
            s if s < 1024 => format!("{} B", s),
            s => format!("{} KB", s / 1024),
        };
        let payload = vec![0xABu8; size];

        let mut group = c.benchmark_group(format!("TCP Roundtrip ({label})"));
        group.throughput(Throughput::Bytes(size as u64));

        // ── dtact-io native ───────────────────────────────────────────────
        {
            let stop = Arc::new(AtomicBool::new(false));
            let server_addr = spawn_dtact_server(size, stop.clone());
            std::thread::sleep(std::time::Duration::from_millis(20));

            group.bench_with_input(BenchmarkId::new("dtact-io", &label), &payload, |b, p| {
                // One persistent connection reused across every batch.
                let connected = Arc::new(std::sync::Mutex::new(None::<Arc<DtactTcpStream>>));
                let c2 = connected.clone();
                let done = Arc::new(AtomicU8::new(0));
                let d2 = done.clone();
                dtact::spawn(async move {
                    let s = DtactTcpStream::connect(server_addr).await.unwrap();
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
            let _ = std::net::TcpStream::connect(server_addr);
            std::thread::sleep(std::time::Duration::from_millis(30));
        }

        // ── raw tokio ─────────────────────────────────────────────────────
        {
            let stop = Arc::new(AtomicBool::new(false));
            let server_addr = spawn_tokio_server(&tokio_rt, size, stop.clone());
            std::thread::sleep(std::time::Duration::from_millis(20));

            let tokio_rt2 = tokio_rt.clone();
            group.bench_with_input(BenchmarkId::new("tokio", &label), &payload, move |b, p| {
                let rt = tokio_rt2.clone();
                let stream = Arc::new(rt.block_on(async {
                    let s = tokio::net::TcpStream::connect(server_addr).await.unwrap();
                    s.set_nodelay(true).ok();
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
            let _ = std::net::TcpStream::connect(server_addr);
            std::thread::sleep(std::time::Duration::from_millis(30));
        }

        group.finish();
    }

    shutdown_runtime();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(8))
        .sample_size(30);
    targets = bench_roundtrip
);
criterion_main!(benches);

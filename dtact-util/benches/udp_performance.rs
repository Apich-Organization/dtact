//! dtact-io native UDP vs `tokio::net::UdpSocket` round-trip benchmark.
//!
//! One persistent echo server per backend loops `recv_from` → `send_to`
//! back to the sender; the measured client sends a datagram and waits for
//! the echo. Native (IOCP/io_uring/kqueue) and raw tokio are benchmarked
//! side by side in one criterion group per payload size — not mutually
//! feature-gated — mirroring `io_performance.rs`.
//!
//! Run:  cargo bench --bench udp_performance --features native
//! Test: cargo bench --bench udp_performance --features native -- --test

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dtact_util::io::{DtactUdpSocket, init_runtime, shutdown_runtime};
use std::hint::black_box;
use std::net::SocketAddr;
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

/// Spawn a persistent dtact-io UDP echo server. Returns its bound address.
fn spawn_dtact_server(plen: usize, stop: Arc<AtomicBool>) -> SocketAddr {
    let addr_slot = Arc::new(std::sync::Mutex::new(None));
    let addr_slot2 = addr_slot.clone();
    dtact::spawn(async move {
        let sock = DtactUdpSocket::bind("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        *addr_slot2.lock().unwrap() = Some(sock.local_addr().unwrap());
        let mut buf = vec![0u8; plen];
        while !stop.load(Ordering::Relaxed) {
            if let Ok((n, from)) = sock.recv_from(&mut buf).await {
                let _ = sock.send_to(&buf[..n], from).await;
            }
        }
    });
    loop {
        if let Some(a) = *addr_slot.lock().unwrap() {
            break a;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

/// Spawn a persistent tokio UDP echo server. Returns its bound address.
fn spawn_tokio_server(
    rt: &tokio::runtime::Runtime,
    plen: usize,
    stop: Arc<AtomicBool>,
) -> SocketAddr {
    let sock = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
    let addr = sock.local_addr().unwrap();
    sock.set_nonblocking(true).unwrap();
    rt.spawn(async move {
        let sock = tokio::net::UdpSocket::from_std(sock).unwrap();
        let mut buf = vec![0u8; plen];
        while !stop.load(Ordering::Relaxed) {
            if let Ok((n, from)) = sock.recv_from(&mut buf).await {
                let _ = sock.send_to(&buf[..n], from).await;
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

        let mut group = c.benchmark_group(format!("UDP Roundtrip ({label})"));
        group.throughput(Throughput::Bytes(size as u64));

        // ── dtact-io native ───────────────────────────────────────────────
        {
            let stop = Arc::new(AtomicBool::new(false));
            let server_addr = spawn_dtact_server(size, stop.clone());
            std::thread::sleep(std::time::Duration::from_millis(10));

            group.bench_with_input(BenchmarkId::new("dtact-io", &label), &payload, |b, p| {
                b.iter_custom(|iters| {
                    let done = Arc::new(AtomicU8::new(0));
                    let done2 = done.clone();
                    let pdata = p.to_vec();
                    let start = std::time::Instant::now();
                    dtact::spawn(async move {
                        let client = DtactUdpSocket::bind("127.0.0.1:0".parse().unwrap())
                            .await
                            .unwrap();
                        let mut buf = vec![0u8; pdata.len()];
                        for _ in 0..iters {
                            let _ = client.send_to(&pdata, server_addr).await;
                            if let Ok((n, _)) = client.recv_from(&mut buf).await {
                                black_box(&buf[..n]);
                            }
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
            // Nudge the server out of its recv so it observes `stop`.
            let nudge = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
            let _ = nudge.send_to(&payload, server_addr);
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        // ── raw tokio ─────────────────────────────────────────────────────
        {
            let stop = Arc::new(AtomicBool::new(false));
            let server_addr = spawn_tokio_server(&tokio_rt, size, stop.clone());
            std::thread::sleep(std::time::Duration::from_millis(10));

            let tokio_rt2 = tokio_rt.clone();
            group.bench_with_input(BenchmarkId::new("tokio", &label), &payload, move |b, p| {
                b.iter_custom(|iters| {
                    let pdata = p.to_vec();
                    let rt = tokio_rt2.clone();
                    let start = std::time::Instant::now();
                    rt.block_on(async move {
                        let client = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
                        let mut buf = vec![0u8; pdata.len()];
                        for _ in 0..iters {
                            let _ = client.send_to(&pdata, server_addr).await;
                            if let Ok((n, _)) = client.recv_from(&mut buf).await {
                                black_box(&buf[..n]);
                            }
                        }
                    });
                    start.elapsed()
                });
            });

            stop.store(true, Ordering::Release);
            let nudge = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
            let _ = nudge.send_to(&payload, server_addr);
            std::thread::sleep(std::time::Duration::from_millis(20));
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

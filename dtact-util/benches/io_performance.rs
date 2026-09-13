/// dtact-io vs tokio TCP echo benchmark
///
/// Payload sizes: 1 KB, 64 KB, 1 MB
///
/// Each benchmark group creates one persistent server that loops accepting
/// connections and echoing data back.  This avoids ephemeral-port exhaustion
/// from creating thousands of new listeners in a tight loop.
///
/// Run:  cargo bench --bench io_performance
/// Test: cargo bench --bench io_performance -- --test
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dtact_util::io::{DtactTcpListener, DtactTcpStream, init_runtime, shutdown_runtime};
use std::hint::black_box;
use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

// ─── one-time runtime init ────────────────────────────────────────────────────

static INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();

fn ensure_runtimes() {
    INIT.get_or_init(|| {
        // Dtact scheduler: 2 workers (diagnostic: reduced from 4 to cut
        // thread oversubscription on an 8-logical-core box while
        // investigating scheduler wake latency)
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

        // dtact-io: 1 IO workers, 16 384 pool buffers × 4 KB = 64 MB arena,
        // ring_depth = 4096.
        init_runtime(1, 4096, 16_384, 4096, &[]);
    });
}

// ─── dtact-io: persistent echo server ────────────────────────────────────────

/// Spawns a persistent dtact-io echo server.  Returns its `SocketAddr`.
/// Pass `stop` an `Arc<AtomicBool>` and set it to `true` to shut it down.
fn spawn_dtact_server(plen: usize, stop: Arc<AtomicBool>) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let dtact_listener = Arc::new(DtactTcpListener::from_std(listener).unwrap());

    // Pre-spawn 4 acceptor/worker fibers (one per dtact worker thread).
    // Only one connection is active at a time in this benchmark, so 4 is
    // more than enough and avoids 1024 concurrent io_uring accept ops.
    for _ in 0..4 {
        let listener = dtact_listener.clone();
        let stop = stop.clone();
        let plen2 = plen;
        dtact::spawn(async move {
            // Allocate the echo buffer once per fiber, reused across connections.
            let mut buf = vec![0u8; plen2];
            while !stop.load(Ordering::Relaxed) {
                if let Ok((stream, _)) = listener.accept().await {
                    let mut pos = 0usize;
                    while pos < plen2 {
                        match stream.read(&mut buf[pos..]).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                debug_assert!(n <= plen2 - pos);
                                pos += n;
                            }
                        }
                    }
                    let mut sent = 0usize;
                    while sent < pos {
                        match stream.write(&buf[sent..pos]).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => sent += n,
                        }
                    }
                }
            }
        });
    }

    addr
}

fn spawn_dtact_server_concurrent(plen: usize, stop: Arc<AtomicBool>) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let dtact_listener = Arc::new(DtactTcpListener::from_std(listener).unwrap());

    let listener = dtact_listener.clone();
    dtact::spawn(async move {
        while !stop.load(Ordering::Relaxed) {
            if let Ok((stream, _)) = listener.accept().await {
                let plen2 = plen;
                dtact::spawn(async move {
                    let mut buf = vec![0u8; plen2];
                    let mut pos = 0usize;
                    while pos < plen2 {
                        match stream.read(&mut buf[pos..]).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => pos += n,
                        }
                    }
                    let mut sent = 0usize;
                    while sent < pos {
                        match stream.write(&buf[sent..pos]).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => sent += n,
                        }
                    }
                });
            }
        }
    });

    addr
}

// ─── tokio: persistent echo server ───────────────────────────────────────────

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
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt, AsyncWriteExt};
                    let mut buf = vec![0u8; plen];
                    let mut pos = 0usize;
                    while pos < plen {
                        match stream.read(&mut buf[pos..]).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => pos += n,
                        }
                    }
                    let _ = stream.write_all(&buf[..pos]).await;
                });
            }
        }
    });

    addr
}

// ─── benchmark ───────────────────────────────────────────────────────────────

fn bench_echo(c: &mut Criterion) {
    ensure_runtimes();

    let tokio_rt = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_io()
            .build()
            .unwrap(),
    );

    // payload sizes: 1 KB, 64 KB, 1 MB
    for &size in &[
        1_024usize,
        65_536,
        1_048_576,
        4_194_304,
        16_777_216,
        67_108_864,
        268_435_456,
    ] {
        let label = match size {
            s if s < 10 * 1024 => format!("{} B", s),
            s if s < 1024 * 1024 => format!("{} KB", s / 1024),
            s => format!("{} MB", s / (1024 * 1024)),
        };
        let payload = vec![0xABu8; size];

        let mut group = c.benchmark_group(format!("TCP Echo ({label})"));
        group.throughput(Throughput::Bytes(size as u64));

        // ── dtact-io ─────────────────────────────────────────────────────────
        {
            let stop = Arc::new(AtomicBool::new(false));
            let server_addr = spawn_dtact_server(size, stop.clone());
            // small warm-up: let the server fibers get scheduled
            std::thread::sleep(std::time::Duration::from_millis(10));

            group.bench_with_input(BenchmarkId::new("dtact-io", &label), &payload, |b, p| {
                b.iter_custom(|iters| {
                    let done = Arc::new(AtomicU8::new(0));
                    let done2 = done.clone();
                    let pdata = p.to_vec();

                    let start = std::time::Instant::now();
                    dtact::spawn(async move {
                        for _ in 0..iters {
                            if let Ok(stream) = DtactTcpStream::connect(server_addr).await {
                                let mut sent = 0usize;
                                while sent < pdata.len() {
                                    match stream.write(&pdata[sent..]).await {
                                        Ok(0) | Err(_) => break,
                                        Ok(n) => sent += n,
                                    }
                                }
                                let mut buf = vec![0u8; pdata.len()];
                                let mut pos = 0usize;
                                while pos < pdata.len() {
                                    match stream.read(&mut buf[pos..]).await {
                                        Ok(0) | Err(_) => break,
                                        Ok(n) => {
                                            debug_assert!(n <= pdata.len() - pos);
                                            pos += n;
                                        }
                                    }
                                }
                                black_box(&buf[..pos]);
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
            // one final connect to unblock the accept, draining the server
            let _ = std::net::TcpStream::connect(server_addr);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        // ── tokio ─────────────────────────────────────────────────────────────
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
                        for _ in 0..iters {
                            if let Ok(mut stream) =
                                tokio::net::TcpStream::connect(server_addr).await
                            {
                                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                                let _ = stream.write_all(&pdata).await;
                                let mut buf = vec![0u8; pdata.len()];
                                let mut pos = 0usize;
                                while pos < pdata.len() {
                                    match stream.read(&mut buf[pos..]).await {
                                        Ok(0) | Err(_) => break,
                                        Ok(n) => pos += n,
                                    }
                                }
                                black_box(buf);
                            }
                        }
                    });
                    start.elapsed()
                });
            });

            stop.store(true, Ordering::Release);
            let _ = std::net::TcpStream::connect(server_addr);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        group.finish();
    }

    // NOTE: deliberately no `shutdown_runtime()` here — `bench_concurrent`
    // runs next in the same process (see `criterion_group!` below) and
    // still needs the dtact-io driver alive. Calling it here used to kill
    // the io-worker thread while `ensure_runtimes()`'s `OnceLock` silently
    // no-ops on the next call, leaving every later dtact-io op parked
    // forever waiting on a driver that no longer exists — the "hangs
    // without a debugger" bug (a debugger session was just too slow to
    // ever reach `bench_concurrent` and observe it). Teardown happens once,
    // for real, at the end of `bench_concurrent` instead.
}

fn bench_concurrent(c: &mut Criterion) {
    ensure_runtimes();

    #[cfg(unix)]
    unsafe {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) == 0 {
            limit.rlim_cur = limit.rlim_max;
            libc::setrlimit(libc::RLIMIT_NOFILE, &limit);
        }
    }

    let tokio_rt = Arc::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_io()
            .build()
            .unwrap(),
    );

    for &num_conns in &[100usize, 1000, 10000] {
        let mut group = c.benchmark_group(format!("Concurrent Connections ({num_conns})"));
        group.warm_up_time(std::time::Duration::from_secs(2));
        group.measurement_time(std::time::Duration::from_secs(10));
        group.sample_size(10);

        let size = 64usize;
        let payload = vec![0xABu8; size];

        // ── dtact-io ─────────────────────────────────────────────────────────
        {
            let stop = Arc::new(AtomicBool::new(false));
            let server_addr = spawn_dtact_server_concurrent(size, stop.clone());
            std::thread::sleep(std::time::Duration::from_millis(10));

            group.bench_function(BenchmarkId::new("dtact-io", num_conns), |b| {
                b.iter_custom(|iters| {
                    let start = std::time::Instant::now();
                    for _ in 0..iters {
                        let active = Arc::new(std::sync::atomic::AtomicUsize::new(num_conns));
                        let done = Arc::new(AtomicBool::new(false));

                        for _ in 0..num_conns {
                            let active = active.clone();
                            let done = done.clone();
                            let payload = payload.clone();
                            dtact::spawn(async move {
                                if let Ok(stream) = DtactTcpStream::connect(server_addr).await {
                                    let mut sent = 0usize;
                                    while sent < payload.len() {
                                        match stream.write(&payload[sent..]).await {
                                            Ok(0) | Err(_) => break,
                                            Ok(n) => sent += n,
                                        }
                                    }
                                    let mut buf = vec![0u8; payload.len()];
                                    let mut pos = 0usize;
                                    while pos < payload.len() {
                                        match stream.read(&mut buf[pos..]).await {
                                            Ok(0) | Err(_) => break,
                                            Ok(n) => pos += n,
                                        }
                                    }
                                    black_box(buf);
                                }
                                if active.fetch_sub(1, Ordering::SeqCst) == 1 {
                                    done.store(true, Ordering::Release);
                                }
                            });
                        }

                        while !done.load(Ordering::Acquire) {
                            std::hint::spin_loop();
                        }
                    }
                    start.elapsed()
                });
            });

            stop.store(true, Ordering::Release);
            let _ = std::net::TcpStream::connect(server_addr);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        // ── tokio ─────────────────────────────────────────────────────────────
        {
            let stop = Arc::new(AtomicBool::new(false));
            let server_addr = spawn_tokio_server(&tokio_rt, size, stop.clone());
            std::thread::sleep(std::time::Duration::from_millis(10));

            let tokio_rt2 = tokio_rt.clone();
            group.bench_function(BenchmarkId::new("tokio", num_conns), |b| {
                b.iter_custom(|iters| {
                    let rt = tokio_rt2.clone();
                    let payload_ref = &payload;
                    let start = std::time::Instant::now();
                    rt.block_on(async move {
                        for _ in 0..iters {
                            let mut handles = Vec::with_capacity(num_conns);
                            for _ in 0..num_conns {
                                let payload = payload_ref.clone();
                                handles.push(tokio::spawn(async move {
                                    if let Ok(mut stream) =
                                        tokio::net::TcpStream::connect(server_addr).await
                                    {
                                        use tokio::io::{AsyncReadExt, AsyncWriteExt};
                                        let _ = stream.write_all(&payload).await;
                                        let mut buf = vec![0u8; payload.len()];
                                        let mut pos = 0usize;
                                        while pos < payload.len() {
                                            match stream.read(&mut buf[pos..]).await {
                                                Ok(0) | Err(_) => break,
                                                Ok(n) => pos += n,
                                            }
                                        }
                                        black_box(buf);
                                    }
                                }));
                            }
                            for h in handles {
                                let _ = h.await;
                            }
                        }
                    });
                    start.elapsed()
                });
            });

            stop.store(true, Ordering::Release);
            let _ = std::net::TcpStream::connect(server_addr);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        group.finish();
    }

    shutdown_runtime();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(20))
        .sample_size(50);
    targets = bench_echo, bench_concurrent
);
criterion_main!(benches);

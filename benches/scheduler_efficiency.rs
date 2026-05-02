use criterion::{Criterion, criterion_group, criterion_main};
use dtact::yield_now;
use std::hint::black_box;

/// Initializes the Dtact runtime with 4 workers.
/// Called once before starting benchmarks.
fn init_dtact() {
    let _ = dtact::GLOBAL_RUNTIME.get_or_init(|| {
        let workers_count = 4;
        let scheduler = dtact::dta_scheduler::DtaScheduler::new(
            workers_count,
            dtact::dta_scheduler::TopologyMode::P2PMesh,
        );
        let pool = dtact::memory_management::ContextPool::new(
            65536,
            64 * 1024,
            dtact::memory_management::SafetyLevel::Safety0,
            0,
        )
        .expect("DTA-V3 Hardware Initialization Failed");

        dtact::Runtime {
            scheduler,
            pool,
            started: core::sync::atomic::AtomicBool::new(false),
            shutdown: core::sync::atomic::AtomicBool::new(false),
        }
    });
    if let Some(rt) = dtact::GLOBAL_RUNTIME.get() {
        rt.start();
    }
}

/// Benchmark 1: Pure Spawning Overhead
/// Spawns N tasks that do minimal work and joins them.
fn bench_spawn_efficiency(c: &mut Criterion) {
    init_dtact();
    let tokio_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Spawn Efficiency (1M tasks)");
    let num_tasks = 1_000_000;

    group.bench_function("Tokio", |b| {
        b.to_async(&tokio_rt).iter(|| async {
            let mut handles = Vec::with_capacity(num_tasks);
            for _ in 0..num_tasks {
                handles.push(tokio::spawn(async move {
                    black_box(1);
                }));
            }
            for h in handles {
                let _ = h.await;
            }
        });
    });

    group.bench_function("Dtact", |b| {
        b.iter(|| {
            let handle =
                dtact::api::SpawnBuilder::<dtact::SameThreadNoFloat>::new().spawn(async move {
                    let mut handles = Vec::with_capacity(num_tasks);
                    for _ in 0..num_tasks {
                        handles.push(
                            dtact::api::SpawnBuilder::<dtact::SameThreadNoFloat>::new().spawn(
                                async move {
                                    black_box(1);
                                },
                            ),
                        );
                    }
                    for h in handles {
                        dtact::c_ffi::dtact_await(h);
                    }
                });
            dtact::c_ffi::dtact_await(handle);
        });
    });
    group.finish();
}

/// Benchmark 2: Cooperative Yielding Latency
/// Spawns a few tasks that yield many times to test context switch overhead.
fn bench_yield_efficiency(c: &mut Criterion) {
    init_dtact();
    let tokio_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Yield Efficiency (10 tasks x 100 yields)");
    let num_yields = 100;
    let num_tasks = 10;

    group.bench_function("Tokio", |b| {
        b.to_async(&tokio_rt).iter(|| async {
            let mut handles = Vec::with_capacity(num_tasks);
            for _ in 0..num_tasks {
                handles.push(tokio::spawn(async move {
                    for _ in 0..num_yields {
                        tokio::task::yield_now().await;
                    }
                }));
            }
            for h in handles {
                let _ = h.await;
            }
        });
    });

    group.bench_function("Dtact", |b| {
        b.iter(|| {
            let handle =
                dtact::api::SpawnBuilder::<dtact::SameThreadNoFloat>::new().spawn(async move {
                    let mut handles = Vec::with_capacity(num_tasks);
                    for _ in 0..num_tasks {
                        handles.push(
                            dtact::api::SpawnBuilder::<dtact::SameThreadNoFloat>::new().spawn(
                                async move {
                                    for _ in 0..num_yields {
                                        yield_now().await;
                                    }
                                },
                            ),
                        );
                    }
                    for h in handles {
                        dtact::c_ffi::dtact_await(h);
                    }
                });
            dtact::c_ffi::dtact_await(handle);
        });
    });
    group.finish();
}

/// Benchmark 3: Work Deflection (Load Balancing)
/// One task spawns many workers to test how well the scheduler distributes load.
fn bench_deflection_efficiency(c: &mut Criterion) {
    init_dtact();
    let tokio_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .build()
        .unwrap();

    let mut group = c.benchmark_group("Work Deflection (Hot Core)");
    let num_tasks = 10_000_000;

    group.bench_function("Tokio", |b| {
        b.to_async(&tokio_rt).iter(|| async {
            let mut handles = Vec::with_capacity(num_tasks);
            for _ in 0..num_tasks {
                handles.push(tokio::spawn(async move {
                    let mut sum = 0;
                    for i in 0..100 {
                        sum += black_box(i);
                    }
                    sum
                }));
            }
            for h in handles {
                let _ = h.await;
            }
        });
    });

    group.bench_function("Dtact", |b| {
        b.iter(|| {
            let handle =
                dtact::api::SpawnBuilder::<dtact::CrossThreadNoFloat>::new().spawn(async move {
                    let mut handles = Vec::with_capacity(num_tasks);
                    for _ in 0..num_tasks {
                        handles.push(
                            dtact::api::SpawnBuilder::<dtact::CrossThreadNoFloat>::new().spawn(
                                async move {
                                    let mut sum = 0;
                                    for i in 0..100 {
                                        sum += black_box(i);
                                    }
                                    sum
                                },
                            ),
                        );
                    }
                    for h in handles {
                        dtact::c_ffi::dtact_await(h);
                    }
                });
            dtact::c_ffi::dtact_await(handle);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_spawn_efficiency,
    bench_yield_efficiency,
    bench_deflection_efficiency
);
criterion_main!(benches);

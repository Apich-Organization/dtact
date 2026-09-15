//! General-purpose engineering benchmarks for the Dtact scheduler.
//!
//! Distinct from the `benchmark`-feature-gated benches (`dta_load_ratio`,
//! `dta_forkjoin_bound`, `numa_information_cost`, ...), which exist to
//! validate specific numeric claims in `paper/main.tex` against a
//! synthetic NUMA cost model. This file has no academic claim behind it —
//! it exists to answer ordinary engineering questions ("did this change
//! regress spawn throughput", "what does cancellation actually cost") and
//! runs unconditionally (no `benchmark` feature required).
//!
//! Run with `cargo bench --bench scheduler_efficiency`.
//!
//! ## A note on the yield benchmarks specifically
//! `wait_pinned` (`src/future_bridge.rs`) re-polls a still-`Pending` future
//! up to `adaptive_spin_count` times (default 50) *before* ever performing
//! a real assembly context switch. `yield_now()`'s future resolves
//! unconditionally on its second poll, so in the common case — nothing
//! else needs this worker's core right now — a self `yield_now()` never
//! reaches the real suspend/resume path at all; it costs one atomic swap
//! and a couple of extra poll calls. `bench_yield_now_loop` below measures
//! exactly that fast path. `bench_fiber_pingpong` measures the other
//! thing "yield" can mean: `yield_to()`'s explicit handoff to a named
//! peer fiber, which *does* go through the real wake protocol and, when
//! the peer is genuinely not already running, the real switch. Treat
//! these as two different costs, not two measurements of the same thing.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dtact::{TaskOutcome, cancel, outcome, yield_now};
use std::future::Future;
use std::hint::black_box;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::task::{Context, Poll};

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
            8192,
            64 * 1024,
            dtact::memory_management::SafetyLevel::Safety0,
            0,
            workers_count,
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

/// Runs `f` with the default panic hook replaced by a silent one, then
/// restores whatever hook was previously installed.
///
/// `fiber_entry_point` (`src/api.rs`) catches every panic inside a fiber
/// via `catch_unwind` and safely classifies it — but Rust's panic *hook*
/// (the "thread '...' panicked at ...:" printout) fires unconditionally
/// before unwinding even starts, regardless of whether the panic is later
/// caught. `bench_cancellation` and `bench_panic_vs_normal_completion`
/// each deliberately trigger thousands of *expected* panics per sample
/// (`DtaCancellation`'s unwind, and the literal `panic!("benchmark
/// panic")`) — left at the default hook, one real run of this file
/// produced over 750,000 lines of "panicked at" noise, which is not just
/// unreadable but was large enough to blow past this session's own
/// output-capture limit before the later benchmarks ever ran. Scoped
/// (not a permanent global override) so a genuinely unexpected panic in
/// any *other* benchmark in this file still prints normally.
fn with_silent_panic_hook<T>(f: impl FnOnce() -> T) -> T {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = f();
    std::panic::set_hook(previous);
    result
}

fn tokio_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .build()
        .unwrap()
}

/// Task counts covering three orders of magnitude — enough to see how a
/// change's cost scales without the multi-minute-per-sample runtime the
/// previous 1M/10M configurations required for routine iteration.
const TASK_COUNTS: &[usize] = &[1_000, 10_000, 100_000];

/// Benchmark 1: pure spawn+join throughput, Dtact vs. Tokio.
fn bench_spawn_join(c: &mut Criterion) {
    init_dtact();
    let rt = tokio_runtime();
    let mut group = c.benchmark_group("Spawn+Join Throughput");

    for &n in TASK_COUNTS {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("Dtact", n), &n, |b, &n| {
            b.iter(|| {
                let handle = dtact::api::SpawnBuilder::<dtact::CrossThreadNoFloat>::new().spawn(
                    async move {
                        let mut handles = Vec::with_capacity(n);
                        for _ in 0..n {
                            handles.push(
                                dtact::api::SpawnBuilder::<dtact::CrossThreadNoFloat>::new().spawn(
                                    async move {
                                        black_box(1);
                                    },
                                ),
                            );
                        }
                        for h in handles {
                            dtact::c_ffi::dtact_await(h);
                        }
                    },
                );
                dtact::c_ffi::dtact_await(handle);
            });
        });

        group.bench_with_input(BenchmarkId::new("Tokio", n), &n, |b, &n| {
            b.to_async(&rt).iter(|| async move {
                let mut handles = Vec::with_capacity(n);
                for _ in 0..n {
                    handles.push(tokio::spawn(async move {
                        black_box(1);
                    }));
                }
                for h in handles {
                    let _ = h.await;
                }
            });
        });
    }

    group.finish();
}

/// Benchmark 2: cooperative-yield fast path (see module doc comment for
/// why this specifically measures the no-real-switch case).
fn bench_yield_now_loop(c: &mut Criterion) {
    init_dtact();
    let rt = tokio_runtime();
    let mut group = c.benchmark_group("Yield Fast Path (10 tasks x 100 yields)");
    let num_yields = 100;
    let num_tasks = 10;

    group.bench_function("Dtact", |b| {
        b.iter(|| {
            let handle =
                dtact::api::SpawnBuilder::<dtact::CrossThreadNoFloat>::new().spawn(async move {
                    let mut handles = Vec::with_capacity(num_tasks);
                    for _ in 0..num_tasks {
                        handles.push(
                            dtact::api::SpawnBuilder::<dtact::CrossThreadNoFloat>::new().spawn(
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

    group.bench_function("Tokio", |b| {
        b.to_async(&rt).iter(|| async {
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

    group.finish();
}

/// Benchmark 3: cross-fiber `yield_to()` handoff cost, between two
/// genuinely distinct fibers (A repeatedly hands off to B, B hands back to
/// A), not a self-directed `YieldNow` (Benchmark 2). Exercises `yield_to`'s
/// full wake protocol (`FiberContext::try_notify`'s conditional swap and,
/// whenever the peer happens to be parked, the real enqueue) against a
/// *named peer* — but is not a guaranteed measurement of a real assembly
/// context switch specifically: each side's own suspension after the
/// handoff is still driven by `yield_now()`, which (see Benchmark 2's
/// note) can itself resolve via the adaptive-spin fast path depending on
/// scheduling. Treat this as "the handoff-protocol cost", not as an
/// isolated switch-cost number.
///
/// # Why this is safe to run with no coordination between the two sides
/// Each fiber independently fires `handoffs` `yield_to` calls at the
/// other with no guarantee both finish at the same time — earlier, this
/// surfaced a genuine, pre-existing hazard in `awaken_fiber_by_index`
/// (unconditionally swapping a wake target's `state` to `Notified` with
/// no check that the target was still alive, corrupting a
/// just-terminated peer's terminal state and hanging its joiner forever
/// — reproduced with a minimal standalone repro, independent of
/// Criterion). That has since been fixed at the root
/// (`FiberContext::try_notify`, `src/memory_management.rs`): a `yield_to`
/// aimed at an already-terminated peer is now a safe no-op instead of
/// corrupting anything, so the two fibers below need no explicit
/// coordination to avoid it.
fn bench_fiber_pingpong(c: &mut Criterion) {
    init_dtact();
    let mut group = c.benchmark_group("Fiber Ping-Pong (1000 handoffs)");
    let handoffs = 1000u32;

    group.bench_function("Dtact yield_to", |b| {
        b.iter(|| {
            // Lock-free handle publication (see Benchmark 4's comment on
            // why not a `Mutex`): `0` is never a valid `dtact_handle_t`
            // (bit 63 is always set — see `SpawnBuilder::spawn`'s "Handle
            // Layout" comment), so it doubles as the "not yet published"
            // sentinel.
            let a_cell = Arc::new(std::sync::atomic::AtomicU64::new(0));
            let b_cell = Arc::new(std::sync::atomic::AtomicU64::new(0));
            let b_for_a = b_cell.clone();
            let a_for_b = a_cell.clone();

            let fiber_a =
                dtact::api::SpawnBuilder::<dtact::CrossThreadNoFloat>::new().spawn(async move {
                    let peer = loop {
                        let v = b_for_a.load(Ordering::Acquire);
                        if v != 0 {
                            break dtact::dtact_handle_t(v);
                        }
                        yield_now().await;
                    };
                    for _ in 0..handoffs {
                        dtact::yield_to_async(peer).await;
                    }
                });
            a_cell.store(fiber_a.0, Ordering::Release);

            let fiber_b =
                dtact::api::SpawnBuilder::<dtact::CrossThreadNoFloat>::new().spawn(async move {
                    let peer = loop {
                        let v = a_for_b.load(Ordering::Acquire);
                        if v != 0 {
                            break dtact::dtact_handle_t(v);
                        }
                        yield_now().await;
                    };
                    for _ in 0..handoffs {
                        dtact::yield_to_async(peer).await;
                    }
                });
            b_cell.store(fiber_b.0, Ordering::Release);

            dtact::c_ffi::dtact_await(fiber_a);
            dtact::c_ffi::dtact_await(fiber_b);
        });
    });

    group.finish();
}

/// Benchmark 4: cancellation overhead — spawn N fibers parked indefinitely,
/// cancel and join all of them. Answers "what does tearing down a fiber
/// that never gets to run to completion actually cost", distinct from
/// Benchmark 1's normal-completion cost.
fn bench_cancellation(c: &mut Criterion) {
    init_dtact();
    let mut group = c.benchmark_group("Cancellation Throughput");

    struct Forever;
    impl Future for Forever {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            Poll::Pending
        }
    }

    with_silent_panic_hook(|| {
        for &n in &[100usize, 1_000] {
            group.throughput(Throughput::Elements(n as u64));
            group.bench_with_input(BenchmarkId::new("Dtact", n), &n, |b, &n| {
                b.iter(|| {
                    let handles: Vec<_> = (0..n)
                        .map(|_| {
                            dtact::api::SpawnBuilder::<dtact::CrossThreadNoFloat>::new()
                                .spawn(Forever)
                        })
                        .collect();
                    // Give every fiber a chance to reach its first suspension
                    // point before requesting cancellation.
                    for h in &handles {
                        cancel(*h);
                    }
                    for h in handles {
                        black_box(outcome(h));
                    }
                });
            });
        }
    });

    group.finish();
}

/// Benchmark 5: panicking-completion overhead vs. normal completion.
/// `fiber_entry_point` (`src/api.rs`) now classifies `catch_unwind`'s
/// `Ok`/`Err` instead of unconditionally reporting `Finished`; this
/// answers whether that classification (plus the boxed-payload storage on
/// the panic path) is measurable.
fn bench_panic_vs_normal_completion(c: &mut Criterion) {
    init_dtact();
    let mut group = c.benchmark_group("Completion Path Overhead (1000 tasks)");
    let n = 1000;

    group.bench_function("Normal completion", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..n)
                .map(|_| {
                    dtact::api::SpawnBuilder::<dtact::CrossThreadNoFloat>::new().spawn(async move {
                        black_box(1);
                    })
                })
                .collect();
            for h in handles {
                black_box(outcome(h));
            }
        });
    });

    with_silent_panic_hook(|| {
        group.bench_function("Panicking completion", |b| {
            b.iter(|| {
                let handles: Vec<_> = (0..n)
                    .map(|_| {
                        dtact::api::SpawnBuilder::<dtact::CrossThreadNoFloat>::new().spawn(
                            async move {
                                panic!("benchmark panic");
                            },
                        )
                    })
                    .collect();
                for h in handles {
                    let result = outcome(h);
                    debug_assert_eq!(
                        result.as_ref().map(|(o, _)| *o),
                        Some(TaskOutcome::Panicked)
                    );
                    black_box(result);
                }
            });
        });
    });

    group.finish();
}

/// Benchmark 6: work deflection / load balancing under a hot-core scenario
/// (one fiber fans out many small tasks), Dtact vs. Tokio.
fn bench_deflection(c: &mut Criterion) {
    init_dtact();
    let rt = tokio_runtime();
    let mut group = c.benchmark_group("Work Deflection (Hot Core)");

    for &n in TASK_COUNTS {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("Dtact", n), &n, |b, &n| {
            b.iter(|| {
                let handle = dtact::api::SpawnBuilder::<dtact::CrossThreadNoFloat>::new().spawn(
                    async move {
                        let mut handles = Vec::with_capacity(n);
                        for _ in 0..n {
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
                    },
                );
                dtact::c_ffi::dtact_await(handle);
            });
        });

        group.bench_with_input(BenchmarkId::new("Tokio", n), &n, |b, &n| {
            b.to_async(&rt).iter(|| async move {
                let mut handles = Vec::with_capacity(n);
                for _ in 0..n {
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
    }

    group.finish();
}

criterion_group!(
    name = benches;
    // Right-sized for routine engineering iteration rather than an
    // exhaustive nightly comparison run: the previous 10s warm-up / 600s
    // measurement / 200-sample configuration, combined with the old
    // 1M/10M-task benchmarks, made a single `cargo bench` run take hours.
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5))
        .sample_size(50)
        .noise_threshold(0.05);
    targets = bench_spawn_join,
    bench_yield_now_loop,
    bench_fiber_pingpong,
    bench_cancellation,
    bench_panic_vs_normal_completion,
    bench_deflection,
);
criterion_main!(benches);

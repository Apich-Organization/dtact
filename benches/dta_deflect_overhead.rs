//! Isolated, single-threaded (uncontended) microbenchmark of
//! `DtaScheduler::enqueue_deflect` itself.
//!
//! Follow-up to `dta_task_overhead.rs` and `numa_information_cost.rs`:
//! fixing `ContextPool`/`TaskSlab`'s CAS-based allocator overhead (see
//! `dta_task_overhead.rs`'s `context_pool`/`slab` rows, and
//! `src/memory_management.rs`'s per-worker `LocalFreeCache`) made DTA's
//! isolated local-dispatch path faster than WS's, but did **not** close
//! the wall-clock gap `numa_information_cost.rs` shows in real
//! multi-threaded runs with actual fib/UTS work — meaning the allocator
//! was never the dominant cost once tasks do real work (its ~15ns/task
//! saving is noise against a ~20ms/35000-task run). This isolates the next
//! candidate: `dta_task_overhead.rs`'s `dta_local_queue` row measures raw
//! `Worker::push_local`/`pop_local` directly, bypassing the deflection
//! *decision* and the chunk-wrapped mailbox push `enqueue_deflect` adds on
//! top of that — every single spawn goes through `enqueue_deflect`, not
//! through `push_local` directly, so its cost (not `push_local`'s) is what
//! actually gates DTA's real per-task throughput.
//!
//! `paper/main.tex` is a **draft preprint**; this isolates a concrete,
//! checkable implementation cost, not a claim about the theory.
//!
//! Academic-only: requires `cargo bench --bench dta_deflect_overhead
//! --features benchmark`.

use dtact::api::topology::Affinity;
use dtact::dta_scheduler::{DtaScheduler, TopologyMode};
use std::sync::atomic::Ordering;

const ITERATIONS: u32 = 2_000_000;

/// `enqueue_deflect`'s cost when it always takes the local fast path
/// (`push_local`) — i.e. `deflection_threshold` is never exceeded — paired
/// with `pop_local` so the queue never fills and falls into the "local
/// queue full" chunk-routing fallback.
fn bench_deflect_stays_local_ns() -> f64 {
    let sched = DtaScheduler::new(2, TopologyMode::Global);
    for w in &sched.workers {
        unsafe {
            (*w.get())
                .deflection_threshold
                .store(255, Ordering::Relaxed)
        };
    }
    dtact::future_bridge::__set_current_worker_id_for_test(0);

    for i in 0..1024u32 {
        sched.enqueue_deflect(0, u64::from(i), 0, Affinity::Any);
        unsafe { (*sched.workers[0].get()).pop_local() };
    }
    let start = std::time::Instant::now();
    for i in 0..ITERATIONS {
        core::hint::black_box(sched.enqueue_deflect(0, u64::from(i), 0, Affinity::Any));
        core::hint::black_box(unsafe { (*sched.workers[0].get()).pop_local() });
    }
    let elapsed = start.elapsed();
    dtact::future_bridge::__set_current_worker_id_for_test(usize::MAX);
    #[allow(clippy::cast_precision_loss)]
    let ns = elapsed.as_secs_f64() * 1e9 / f64::from(ITERATIONS);
    ns
}

/// `enqueue_deflect`'s cost when it always deflects to a peer worker
/// (`TaskChunk` construction + mailbox push), paired with the receiving
/// worker draining its mailbox (`poll_mailboxes`) and local queue
/// (`pop_local`) each iteration so nothing overflows into the hop-search
/// or warehouse fallback paths.
fn bench_deflect_cross_worker_ns() -> f64 {
    let sched = DtaScheduler::new(2, TopologyMode::Global);
    for w in &sched.workers {
        unsafe {
            let worker = &*w.get();
            worker.deflection_threshold.store(0, Ordering::Relaxed);
            worker.load_level.store(255, Ordering::Relaxed);
        }
    }
    dtact::future_bridge::__set_current_worker_id_for_test(0);

    // `enqueue_deflect`'s target formula (Global topology, Affinity::Any)
    // is `(source + h1 + h2) % n` where `h1 = flow_id & 7` and
    // `h2 = ((flow_id >> 3) & 7) | 1` (always odd). Shifting `i` left by 3
    // keeps `h1 == 0` (even) always, so `h1 + h2` is always odd, forcing
    // `target = (0 + odd) % 2 == 1` — deterministically never `source`
    // itself — every time `deflect` is true (guaranteed above via
    // `load_level=255 > deflection_threshold=0`).
    let flow_id_for = |i: u32| u64::from(i) << 3;

    for i in 0..1024u32 {
        sched.enqueue_deflect(0, flow_id_for(i), 0, Affinity::Any);
        sched.poll_mailboxes(1);
        unsafe { (*sched.workers[1].get()).pop_local() };
    }
    let start = std::time::Instant::now();
    for i in 0..ITERATIONS {
        core::hint::black_box(sched.enqueue_deflect(0, flow_id_for(i), 0, Affinity::Any));
        core::hint::black_box(sched.poll_mailboxes(1));
        core::hint::black_box(unsafe { (*sched.workers[1].get()).pop_local() });
    }
    let elapsed = start.elapsed();
    dtact::future_bridge::__set_current_worker_id_for_test(usize::MAX);
    #[allow(clippy::cast_precision_loss)]
    let ns = elapsed.as_secs_f64() * 1e9 / f64::from(ITERATIONS);
    ns
}

fn main() {
    println!(
        "# Single-threaded, uncontended enqueue_deflect round-trip costs \
         ({ITERATIONS} iterations each)."
    );
    println!("{:<30} {:>14}", "component", "ns/roundtrip");

    let local_ns = bench_deflect_stays_local_ns();
    println!("{:<30} {:>14.2}", "enqueue_deflect (local)", local_ns);

    let cross_ns = bench_deflect_cross_worker_ns();
    println!(
        "{:<30} {:>14.2}",
        "enqueue_deflect (cross-worker)", cross_ns
    );

    println!(
        "\n# For reference, dta_task_overhead.rs's own isolated measurements on this machine:"
    );
    println!("#   raw Worker::push_local + pop_local (no deflect decision): ~2-3 ns");
    println!("#   real ContextPool::alloc_context + free_context (cached):  ~8-10 ns");
    println!("#   crossbeam_deque push + pop (WS local path):                ~17-20 ns");
}

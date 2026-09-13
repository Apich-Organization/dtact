//! Isolated, single-threaded (uncontended) microbenchmark attributing
//! *where* DTA's per-task fixed overhead over the WS baseline actually
//! comes from — a follow-up to the wall-clock gap `numa_information_cost`
//! found between the two schedulers even at N <= physical-core-count (no
//! oversubscription), which the paper's own closed-form acquisition-cost
//! prediction (`paper/main.tex` eq. `beta_dta_num`/`beta_ws_num`) does not
//! predict: that formula says DTA should be *cheaper* per task
//! (`c_SPSC = 80ns` flat vs `δ̄ + 100·log2(N)` for WS), yet real wall-clock
//! runs showed DTA consistently slower.
//!
//! `paper/main.tex` is a **draft preprint**; nothing here is trying to
//! settle whether its model is "right". What this measures is a specific,
//! checkable structural difference the paper's abstract cost model does not
//! account for at all: DTA's harness (mirroring DTA's real production
//! design — fibers are fixed-capacity `ContextPool` slots referenced by
//! index, never moved by value) stores each task behind a `TaskIndex` in a
//! CAS-based free-list slab (`DtaHarness`'s internal `TaskSlab`), while the
//! WS baseline (`crossbeam-deque`) stores the boxed task closure directly in
//! the deque slot with no separate indirection. The paper's `β` accounting
//! (`paper/main.tex` eq. `beta_def`/`cdot`) is entirely about the cost of
//! moving a task *reference* between workers' queues — it has nothing to
//! say about the cost of resolving that reference to its payload, because
//! the abstract model treats "the task" as an opaque unit with no storage
//! model of its own.
//!
//! Three round-trip costs are measured independently, single-threaded, so
//! none of them include any real cross-thread contention (a lower bound on
//! each mechanism's real contribution under concurrent load):
//! 1. **slab**: `TaskSlab::store` + `TaskSlab::take` (DTA harness only).
//! 2. **dta_local_queue**: `Worker::push_local` + `Worker::pop_local` (DTA's
//!    real production SPSC local-queue code — no mailbox, no slab).
//! 3. **ws_deque**: `crossbeam_deque::Worker::push` + `.pop()` (the WS
//!    baseline's real steady-state local path).
//!
//! `slab + dta_local_queue` approximates the DTA harness's real per-task
//! fixed cost on the all-local fast path (every task pays exactly one slab
//! store+take and one local-queue push+pop); `ws_deque` alone is WS's
//! equivalent, since it never pays a separate indirection cost.
//!
//! Academic-only: requires `cargo bench --bench dta_task_overhead --features
//! benchmark`.

use dtact::dta_scheduler::{CpuLevel, Worker};
use dtact::memory_management::{ContextPool, SafetyLevel};

const ITERATIONS: u32 = 2_000_000;

fn cpu0() -> CpuLevel {
    CpuLevel {
        core_id: 0,
        ccx_id: 0,
        numa_id: 0,
    }
}

/// `Worker::push_local` + `Worker::pop_local` round-trip, single-threaded.
fn bench_dta_local_queue_ns() -> f64 {
    let worker = Worker::new(cpu0(), 1).expect("Worker::new failed");
    // Warm up (first touches of the huge-page-backed local_queue buffer).
    for i in 0..1024u32 {
        assert!(worker.push_local(i));
        assert!(worker.pop_local().is_some());
    }
    let start = std::time::Instant::now();
    for i in 0..ITERATIONS {
        core::hint::black_box(worker.push_local(i));
        core::hint::black_box(worker.pop_local());
    }
    let elapsed = start.elapsed();
    #[allow(clippy::cast_precision_loss)]
    let ns = elapsed.as_secs_f64() * 1e9 / f64::from(ITERATIONS);
    ns
}

/// The real, production `ContextPool::alloc_context` + `free_context`
/// round trip, single-threaded, on the cached fast path (worker id 0,
/// `memory_management::LocalFreeCache`) — the production allocator
/// `TaskSlab`'s `slab` row above mirrors. 64-slot capacity so batching
/// (`batch_size = 8`) actually engages.
fn bench_context_pool_ns() -> f64 {
    let pool = ContextPool::new(64, 8192, SafetyLevel::Safety0, 0, 1).expect("pool init");
    dtact::future_bridge::__set_current_worker_id_for_test(0);
    // Warm up (first touches of the pool's mmap'd arena).
    for _ in 0..1024u32 {
        let idx = pool.alloc_context().expect("pool has capacity");
        pool.free_context(idx);
    }
    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        let idx = pool.alloc_context().expect("pool has capacity");
        core::hint::black_box(idx);
        pool.free_context(idx);
    }
    let elapsed = start.elapsed();
    dtact::future_bridge::__set_current_worker_id_for_test(usize::MAX);
    #[allow(clippy::cast_precision_loss)]
    let ns = elapsed.as_secs_f64() * 1e9 / f64::from(ITERATIONS);
    ns
}

/// `crossbeam_deque::Worker` (LIFO) push + pop round-trip, single-threaded
/// — WS's real steady-state local path, storing the actual boxed task
/// directly (no separate indirection layer).
fn bench_ws_deque_ns() -> f64 {
    let deque: crossbeam_deque::Worker<Box<dyn FnOnce() + Send>> =
        crossbeam_deque::Worker::new_lifo();
    for _ in 0..1024u32 {
        deque.push(Box::new(|| {}));
        core::hint::black_box(deque.pop());
    }
    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        deque.push(Box::new(|| {}));
        core::hint::black_box(deque.pop());
    }
    let elapsed = start.elapsed();
    #[allow(clippy::cast_precision_loss)]
    let ns = elapsed.as_secs_f64() * 1e9 / f64::from(ITERATIONS);
    ns
}

fn main() {
    println!(
        "# Single-threaded, uncontended round-trip costs ({ITERATIONS} iterations each) — \
         isolates DTA's task-indirection tax from real scheduling/mailbox traffic. See module \
         docs for what each row does and doesn't measure."
    );
    println!("{:<24} {:>14}", "component", "ns/roundtrip");

    let slab_ns = dtact::benchmark::dta_harness::microbench_slab_roundtrip_ns(ITERATIONS);
    println!("{:<24} {:>14.2}", "slab (store+take)", slab_ns);

    let context_pool_ns = bench_context_pool_ns();
    println!("{:<24} {:>14.2}", "context_pool (real)", context_pool_ns);

    let dta_local_ns = bench_dta_local_queue_ns();
    println!("{:<24} {:>14.2}", "dta_local_queue", dta_local_ns);

    let ws_deque_ns = bench_ws_deque_ns();
    println!("{:<24} {:>14.2}", "ws_deque", ws_deque_ns);

    let dta_full_local_ns = slab_ns + dta_local_ns;
    println!(
        "\n# DTA harness full local-path cost (slab + dta_local_queue): {dta_full_local_ns:.2} ns/task"
    );
    println!(
        "# DTA *production* full local-path cost (context_pool + dta_local_queue): {:.2} ns/task",
        context_pool_ns + dta_local_ns
    );
    println!("# WS full local-path cost (ws_deque only):            {ws_deque_ns:.2} ns/task");
    if ws_deque_ns > 0.0 {
        println!(
            "# ratio, harness slab (dta_full / ws_full): {:.2}x",
            dta_full_local_ns / ws_deque_ns
        );
        println!(
            "# ratio, real ContextPool (dta_full / ws_full): {:.2}x",
            (context_pool_ns + dta_local_ns) / ws_deque_ns
        );
    }
    println!(
        "# NOTE: paper/main.tex is a draft preprint. Its beta_DTA/beta_WS closed forms model \
         only the cross-worker task-reference-movement cost, not this indirection tax — a \
         mismatch here is a gap in what the model accounts for, not necessarily an error in it."
    );
}

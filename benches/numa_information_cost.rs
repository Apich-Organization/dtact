//! Real-execution comparison of the DTA scheduler against a pure
//! work-stealing (WS) baseline, exercised against two of the preprint's
//! draft empirical claims — NOT a re-run of the engineering-only Tokio
//! comparison in `scheduler_efficiency.rs`.
//!
//! `paper/main.tex` is a **draft preprint** with acknowledged analysis
//! limitations of its own (see its "Future Work" / open-problems sections);
//! refining that analysis is future work, not something this benchmark is
//! trying to settle. What it *can* do is put the paper's closed-form
//! predictions next to what the real, unmodified scheduler code actually
//! does under a controlled, documented synthetic NUMA model — a stronger
//! signal than the paper's existing pure-Python simulation scripts
//! (`paper/script/simulate_*.py`), but still a comparison point for a draft
//! theory, not a validation of it. A mismatch is exactly the kind of finding
//! that should feed back into revising the theory, not evidence of a
//! benchmark bug (though it can be that too — check both).
//!
//! Two claim areas are covered here:
//! * **Information-acquisition cost** (`paper/main.tex`, secs.
//!   "Information Acquisition Rate" and "Concrete Analysis on a
//!   Dual-Socket NUMA Topology"): the empirical per-task acquisition cost
//!   (`ns/task`) against the closed-form `β_DTA`/`β_WS` predictions.
//! * **Load-balance quality** (the paper's stability / `Δq`-imbalance
//!   discussion): how evenly completed tasks actually land across workers,
//!   via [`dtact::benchmark::instrumentation::load_balance_stats`] — a
//!   separate claim from acquisition cost, measured independently.
//!
//! Academic-only: requires `cargo bench --bench numa_information_cost
//! --features benchmark`. See `src/benchmark/mod.rs` for the full method
//! description and `src/benchmark/numa_model.rs` for why this is a
//! calibrated synthetic NUMA model rather than a real- or virtualized-
//! hardware measurement (no KVM is available in this development
//! environment, and the underlying machine has no real second NUMA node in
//! any case).

use dtact::benchmark::dta_harness::DtaHarness;
use dtact::benchmark::instrumentation::{self, LoadBalanceStats, Snapshot};
use dtact::benchmark::numa_model::{self, Topology};
use dtact::benchmark::work_stealing::WsScheduler;
use dtact::benchmark::workloads::{self, TaskSpawner, UtsParams};

use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
enum Workload {
    Fib { n: u64, cutoff: u64 },
    Uts(UtsParams),
}

impl Workload {
    const fn label(self) -> &'static str {
        match self {
            Self::Fib { .. } => "fib",
            Self::Uts(_) => "uts",
        }
    }

    fn run_on<S: TaskSpawner>(self, spawner: &S) {
        match self {
            Self::Fib { n, cutoff } => {
                let run = workloads::spawn_fib(spawner, n, cutoff);
                run.countdown.wait_zero();
            }
            Self::Uts(params) => {
                let run = workloads::spawn_uts(spawner, 0x5EED_5EED, params);
                run.countdown.wait_zero();
            }
        }
    }
}

struct RunResult {
    snapshot: Snapshot,
    elapsed: Duration,
    balance: LoadBalanceStats,
}

/// Deliberately below `DtaScheduler`'s production default (80): this
/// harness's task volumes wouldn't otherwise build up enough per-worker
/// backlog to trigger deflection at all (see `DtaHarness::new`'s doc
/// comment on why `TopologyMode::Global` is also required for this).
/// Chosen to make cross-worker (and thus, under the declared virtual
/// topology, sometimes cross-socket) traffic a routine occurrence rather
/// than a rare edge case, which is the load regime the preprint's
/// asymptotic analysis (`λ_steal ≈ λ_deflect` "at high load") concerns.
const DEFLECTION_THRESHOLD: u8 = 15;

fn run_dta(topology: Topology, workload: Workload) -> RunResult {
    let harness = DtaHarness::new(topology, 1 << 20);
    harness.set_deflection_threshold(DEFLECTION_THRESHOLD);
    let start = Instant::now();
    std::thread::scope(|scope| {
        harness.run_workers(scope);
        workload.run_on(&harness);
        harness.request_shutdown();
    });
    let elapsed = start.elapsed();
    // Only safe to retract the measurement target now: every worker thread
    // that could call `report_dta_hop` has already joined (the `scope`
    // block above only returns once they have).
    harness.end_measurement();
    RunResult {
        snapshot: harness.meter.snapshot(),
        elapsed,
        balance: instrumentation::load_balance_stats(&harness.per_worker_completed),
    }
}

fn run_ws(topology: Topology, workload: Workload) -> RunResult {
    let sched = WsScheduler::new(topology);
    let start = Instant::now();
    std::thread::scope(|scope| {
        sched.run_workers(scope);
        workload.run_on(&sched);
        sched.request_shutdown();
    });
    let elapsed = start.elapsed();
    RunResult {
        snapshot: sched.meter.snapshot(),
        elapsed,
        balance: instrumentation::load_balance_stats(&sched.per_worker_completed),
    }
}

/// The preprint's closed-form per-task acquisition cost prediction,
/// `paper/main.tex` eq. `beta_dta_num`: a flat `c_SPSC = 80ns`, independent
/// of `N`. Draft prediction, offered as a comparison point — see module docs.
fn theoretical_dta_ns_per_task() -> f64 {
    numa_model::SPSC_COST_NS as f64
}

/// The preprint's closed-form per-task acquisition cost prediction,
/// `paper/main.tex` eq. `beta_ws_num`: `δ̄ + 100·log2(N)` ns, where `δ̄ →
/// (δ_intra + δ_inter)/2 = 190ns` as `N → ∞` (eq. `delta_bar_numa`). Uses
/// the same large-`N` limit the paper's own summary table
/// (§`numa_concrete`) uses, rather than the exact finite-`N` weighted
/// average, so results are directly comparable to that table. Draft
/// prediction, offered as a comparison point — see module docs.
fn theoretical_ws_ns_per_task(n: usize) -> f64 {
    let delta_bar = f64::midpoint(
        numa_model::DELTA_INTRA_NS as f64,
        numa_model::DELTA_INTER_NS as f64,
    );
    #[allow(clippy::cast_precision_loss)]
    let log2n = (n as f64).log2();
    delta_bar + 100.0 * log2n
}

fn print_header() {
    println!(
        "{:<6} {:<12} {:<6} {:<10} {:>10} {:>12} {:>10} {:>14} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "N",
        "topology",
        "sched",
        "workload",
        "tasks",
        "elapsed_ms",
        "tasks/ms",
        "ns/task(emp)",
        "ns/task(th)",
        "ratio",
        "cas/succ",
        "bal_cv",
        "bal_max/mn"
    );
}

#[allow(clippy::too_many_arguments)]
fn print_row(
    n: usize,
    topo_name: &str,
    sched_name: &str,
    workload: Workload,
    r: &RunResult,
    theoretical: f64,
) {
    let snap = r.snapshot;
    #[allow(clippy::cast_precision_loss)]
    let elapsed_ms = r.elapsed.as_secs_f64() * 1000.0;
    let throughput = if elapsed_ms > 0.0 {
        #[allow(clippy::cast_precision_loss)]
        let t = snap.tasks_completed as f64 / elapsed_ms;
        t
    } else {
        0.0
    };
    let empirical = snap.ns_per_task();
    let ratio = if theoretical > 0.0 {
        empirical / theoretical
    } else {
        0.0
    };
    println!(
        "{:<6} {:<12} {:<6} {:<10} {:>10} {:>12.2} {:>10.2} {:>14.2} {:>10.2} {:>10.3} {:>10.2} {:>10.3} {:>10.3}",
        n,
        topo_name,
        sched_name,
        workload.label(),
        snap.tasks_completed,
        elapsed_ms,
        throughput,
        empirical,
        theoretical,
        ratio,
        snap.mean_cas_attempts_per_success(),
        r.balance.cv,
        r.balance.max_over_mean,
    );
}

fn main() {
    let local_ns = numa_model::calibrate_local_atomic_load_ns();
    println!(
        "# This machine's real uncontended atomic-load latency: {local_ns:.2} ns \
         (paper's assumed δ_intra = {} ns — diagnostic only, not substituted into the cost model)",
        numa_model::DELTA_INTRA_NS
    );
    println!(
        "# Synthetic NUMA model: δ_intra={}ns δ_inter={}ns c_CAS={}ns (paper/main.tex Sec. numa_concrete)",
        numa_model::DELTA_INTRA_NS,
        numa_model::DELTA_INTER_NS,
        numa_model::CAS_BASE_NS
    );
    println!(
        "# CAUTION: N > physical core count (8 on this machine) is OS-thread oversubscription, \
         not real additional parallelism — throughput numbers at those N are noisier and the \
         acquisition-cost ratio is the primary signal to trust, not raw throughput."
    );
    println!(
        "# NOTE: paper/main.tex is a draft preprint with acknowledged analysis limitations. \
         'ratio' columns compare against its current closed-form predictions as a reference \
         point, not a pass/fail validation — deviations are findings to feed back into the \
         theory, not necessarily benchmark defects. bal_cv/bal_max_mn measure a *separate* \
         claim (load-balance quality across workers), independent of acquisition cost."
    );

    type TopologyFactory = fn(usize) -> Topology;

    let worker_counts = [8usize, 16, 32, 64];
    let topologies: [(&str, TopologyFactory); 2] = [
        ("flat", Topology::flat),
        ("dual_socket", Topology::dual_socket),
    ];
    let workloads = [
        Workload::Fib { n: 30, cutoff: 10 },
        Workload::Uts(UtsParams::balanced()),
        Workload::Uts(UtsParams::unbalanced()),
    ];

    print_header();
    for &n in &worker_counts {
        for (topo_name, topo_fn) in topologies {
            let topology = topo_fn(n);
            for workload in workloads {
                let dta = run_dta(topology, workload);
                print_row(
                    n,
                    topo_name,
                    "dta",
                    workload,
                    &dta,
                    theoretical_dta_ns_per_task(),
                );

                let ws = run_ws(topology, workload);
                print_row(
                    n,
                    topo_name,
                    "ws",
                    workload,
                    &ws,
                    theoretical_ws_ns_per_task(n),
                );
            }
        }
    }
}

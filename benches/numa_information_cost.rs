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
//!
//! ## On the `ns/task` ratio (the `β_WS/β_DTA` claim) needing the median
//!
//! An earlier version of this benchmark reported only the mean `ns/task`
//! across [`REPEATS`] runs, and that mean swung 2-6x between otherwise
//! identical full-binary reruns — noise-dominated, not a usable signal.
//! The cause: this development machine's per-run timings are right-skewed
//! by occasional large outliers (a background VM/host scheduling stall,
//! not small jitter), which more repeats alone does not fix — a mean is
//! not robust to a handful of stalled runs dragging it well above what
//! most runs actually measure (confirmed directly: `REPEATS = 50` on a
//! scoped flat/fib-only sweep still showed ranges like `[82-3044]` at
//! N=32). Switching the reported ratio to the **median** (see
//! [`AggResult::ns_per_task_median`]) fixed this: three independent
//! full-binary reruns at `REPEATS = 15` gave a WS/DTA median-ratio of
//! `2.19, 3.78, 5.15` (mean across the 3 reruns) at `N = 8, 16, 32` —
//! reproducible to within roughly 1.1-1.7x across reruns, vs. 2-6x before.
//! That is a real, evidenced finding, not noise: a clear, monotonically
//! increasing trend with `N`, directionally consistent with the paper's
//! `β_WS/β_DTA = Ω(log N)` claim, but sitting well below the paper's own
//! predicted numeric table (`≈6.3, 7.3, 8.3` at those `N`) — a genuine gap
//! to feed back into the theory now that it isn't just measurement noise.
//! `N = 64` stays unreliable even under this fix (this machine has 8
//! logical CPUs; `N = 64` is 8x oversubscribed) and shouldn't be trusted
//! as a clean data point.

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

type TopologyFactory = fn(usize) -> Topology;

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

/// Independent repetitions per (N, topology, scheduler, workload)
/// configuration. Earlier single-shot sweeps showed the `ns/task(emp)`
/// ratio at a fixed configuration swing by 2-6x between otherwise-identical
/// repeated runs (short absolute elapsed times at these task volumes mean
/// OS-scheduling/VM jitter dominates a single sample) — [`AggResult`]
/// reports the mean *and* the observed min/max range across `REPEATS` runs
/// so that spread is visible directly in the output, instead of requiring
/// several full external re-runs of this binary to notice it.
const REPEATS: u32 = 15;

fn run_dta_once(topology: Topology, workload: Workload) -> RunResult {
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

fn run_ws_once(topology: Topology, workload: Workload) -> RunResult {
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

/// Mean, median, and (for `ns_per_task`) min/max range of [`RunResult`]'s
/// derived scalars across [`REPEATS`] independent runs.
///
/// Both mean *and* median are tracked deliberately: this benchmark's
/// per-run timings turned out to be right-skewed by occasional large
/// outliers (a background VM/host scheduling stall stealing tens to
/// hundreds of milliseconds mid-run, not small jitter around a stable
/// mean) — repeating `numa_information_cost`'s scoped flat/fib sweep at
/// `REPEATS = 50` showed some `ns/task[min-max]` ranges still spanning
/// 20-37x (e.g. N=32 WS: `[82-3044]`), which does *not* shrink by simply
/// averaging more samples the way `dta_forkjoin_bound.rs`'s much
/// shorter-duration trials converged under more `TRIALS`. A mean is not
/// robust to that: a handful of stalled runs among many fast ones drag it
/// well above what most runs actually measure. The median is the standard
/// fix for exactly this (the same reason widely-used benchmarking tools
/// report medians, not means, for wall-clock timings) — `ratio` below is
/// computed from the median, with the mean kept alongside for visibility
/// into how much the outliers are actually distorting it.
struct AggResult {
    tasks_completed: u64,
    elapsed_ms_mean: f64,
    throughput_mean: f64,
    ns_per_task_mean: f64,
    ns_per_task_median: f64,
    ns_per_task_min: f64,
    ns_per_task_max: f64,
    cas_per_success_mean: f64,
    bal_cv_mean: f64,
    bal_max_over_mean_mean: f64,
}

fn mean(xs: &[f64]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = xs.len() as f64;
    xs.iter().sum::<f64>() / n
}

/// Median of `xs`. `xs` is sorted in place — callers must not rely on its
/// original order afterward.
fn median_sorted(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        f64::midpoint(xs[n / 2 - 1], xs[n / 2])
    }
}

fn aggregate(results: &[RunResult]) -> AggResult {
    let elapsed_ms: Vec<f64> = results
        .iter()
        .map(|r| r.elapsed.as_secs_f64() * 1000.0)
        .collect();
    let mut ns_per_task: Vec<f64> = results.iter().map(|r| r.snapshot.ns_per_task()).collect();
    let throughput: Vec<f64> = results
        .iter()
        .zip(&elapsed_ms)
        .map(|(r, &ms)| {
            if ms > 0.0 {
                #[allow(clippy::cast_precision_loss)]
                let t = r.snapshot.tasks_completed as f64 / ms;
                t
            } else {
                0.0
            }
        })
        .collect();
    let cas: Vec<f64> = results
        .iter()
        .map(|r| r.snapshot.mean_cas_attempts_per_success())
        .collect();
    let bal_cv: Vec<f64> = results.iter().map(|r| r.balance.cv).collect();
    let bal_mm: Vec<f64> = results.iter().map(|r| r.balance.max_over_mean).collect();
    let ns_per_task_min = ns_per_task.iter().copied().fold(f64::INFINITY, f64::min);
    let ns_per_task_max = ns_per_task
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let ns_per_task_median = median_sorted(&mut ns_per_task);
    AggResult {
        tasks_completed: results
            .last()
            .expect("REPEATS >= 1")
            .snapshot
            .tasks_completed,
        elapsed_ms_mean: mean(&elapsed_ms),
        throughput_mean: mean(&throughput),
        ns_per_task_mean: mean(&ns_per_task),
        ns_per_task_median,
        ns_per_task_min,
        ns_per_task_max,
        cas_per_success_mean: mean(&cas),
        bal_cv_mean: mean(&bal_cv),
        bal_max_over_mean_mean: mean(&bal_mm),
    }
}

fn run_dta(topology: Topology, workload: Workload) -> AggResult {
    let results: Vec<RunResult> = (0..REPEATS)
        .map(|_| run_dta_once(topology, workload))
        .collect();
    aggregate(&results)
}

fn run_ws(topology: Topology, workload: Workload) -> AggResult {
    let results: Vec<RunResult> = (0..REPEATS)
        .map(|_| run_ws_once(topology, workload))
        .collect();
    aggregate(&results)
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
        "{:<6} {:<12} {:<6} {:<10} {:>10} {:>12} {:>10} {:>14} {:>14} {:>18} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "N",
        "topology",
        "sched",
        "workload",
        "tasks",
        "elapsed_ms",
        "tasks/ms",
        "ns/task(mean)",
        "ns/task(med)",
        "ns/task[min-max]",
        "ns/task(th)",
        "ratio(med)",
        "cas/succ",
        "bal_cv",
        "bal_max/mn"
    );
    println!(
        "  ({REPEATS} independent repeats per row. 'ratio(med)' uses the MEDIAN, not the \
         mean, of ns/task(emp) — these timings are right-skewed by occasional large \
         outliers (background VM/host scheduling stalls), so the mean is shown for \
         visibility into how much those outliers distort it, but is not the number to \
         trust. [min-max] is the full observed range across repeats — a wide range means \
         this configuration is noise-dominated even after averaging.)"
    );
}

#[allow(clippy::too_many_arguments)]
fn print_row(
    n: usize,
    topo_name: &str,
    sched_name: &str,
    workload: Workload,
    r: &AggResult,
    theoretical: f64,
) {
    let ratio_med = if theoretical > 0.0 {
        r.ns_per_task_median / theoretical
    } else {
        0.0
    };
    let range = format!("[{:.0}-{:.0}]", r.ns_per_task_min, r.ns_per_task_max);
    println!(
        "{:<6} {:<12} {:<6} {:<10} {:>10} {:>12.2} {:>10.2} {:>14.2} {:>14.2} {:>18} {:>10.2} {:>10.3} {:>10.2} {:>10.3} {:>10.3}",
        n,
        topo_name,
        sched_name,
        workload.label(),
        r.tasks_completed,
        r.elapsed_ms_mean,
        r.throughput_mean,
        r.ns_per_task_mean,
        r.ns_per_task_median,
        range,
        theoretical,
        ratio_med,
        r.cas_per_success_mean,
        r.bal_cv_mean,
        r.bal_max_over_mean_mean,
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

    let topologies: [(&str, TopologyFactory); 2] = [
        ("flat", Topology::flat),
        ("dual_socket", Topology::dual_socket),
    ];
    let workloads = [
        Workload::Fib { n: 30, cutoff: 10 },
        Workload::Uts(UtsParams::balanced()),
        Workload::Uts(UtsParams::unbalanced()),
    ];

    let physical_cores = std::thread::available_parallelism().map_or(8, std::num::NonZero::get);

    println!(
        "\n# === Section 1: N <= {physical_cores} (this machine's physical core count) ===\n\
         # Every worker gets its own real core: 1:1 with the paper's implicit\n\
         # assumption that a \"worker\" is a hardware execution context, not an\n\
         # oversubscribed OS thread. This is the primary comparison for validating\n\
         # the acquisition-cost closed forms (beta_DTA, beta_WS) and the gamma*\n\
         # crossover claim."
    );
    let in_core_counts: Vec<usize> = [2usize, 4, 6, 8]
        .into_iter()
        .filter(|&n| n <= physical_cores)
        .collect();
    print_header();
    run_sweep(&in_core_counts, &topologies, &workloads);

    println!(
        "\n# === Section 2: N > {physical_cores} (oversubscription stress test) ===\n\
         # OS-thread oversubscription, not real additional parallelism. Included\n\
         # to characterize how each scheduler's *implementation* (not the paper's\n\
         # cost model, which only speaks to acquisition cost per completed task)\n\
         # degrades when there are more schedulable workers than hardware\n\
         # contexts to run them on. A large gap between the two schedulers here\n\
         # is a scalability finding about the code, separate from the\n\
         # information-acquisition-rate claim Section 1 addresses."
    );
    let oversubscribed_counts = [16usize, 32, 64];
    print_header();
    run_sweep(&oversubscribed_counts, &topologies, &workloads);
}

fn run_sweep(
    worker_counts: &[usize],
    topologies: &[(&str, TopologyFactory)],
    workloads: &[Workload],
) {
    for &n in worker_counts {
        for &(topo_name, topo_fn) in topologies {
            let topology = topo_fn(n);
            for &workload in workloads {
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

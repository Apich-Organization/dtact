//! Open-loop, Poisson-arrival load-ratio (`ρ_0`) sweep — tests the
//! preprint's explicit scope-of-applicability claim (`paper/main.tex`
//! §`sec:dta_scope`): "DTA is most advantageous at high but subcritical
//! load, `ρ_0 ∈ [0.6, 0.95]`".
//!
//! Every other benchmark in this harness (`numa_information_cost`,
//! `dta_task_overhead`, `dta_deflect_overhead`) is *closed-loop*: every task
//! is fired as fast as possible with no arrival-rate control, so `ρ_0` is
//! undefined for those runs — they cannot speak to this specific claim at
//! all. This benchmark instead uses
//! [`dtact::benchmark::workloads::run_poisson_bot`] to pace task arrivals
//! to a target system-wide rate `λ`, holding per-task service time fixed so
//! `ρ_0 = λ / (N·μ)` is a controlled independent variable, and reports
//! per-task latency (mean/p50/p99/max) and makespan/throughput across a
//! sweep of `ρ_0` values spanning below, inside, and above the paper's
//! claimed `[0.6, 0.95]` sweet spot — so the claim is either corroborated
//! (DTA's advantage over WS peaks in that band) or not (a finding to feed
//! back into the theory, not a benchmark defect by itself — see
//! `numa_information_cost.rs`'s module docs for the same caveat).
//!
//! Academic-only: requires `cargo bench --bench dta_load_ratio --features
//! benchmark`.

use dtact::benchmark::dta_harness::DtaHarness;
use dtact::benchmark::numa_model::Topology;
use dtact::benchmark::work_stealing::WsScheduler;
use dtact::benchmark::workloads::{self, PoissonBotParams};

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// Per-task service time: fine enough to let the sweep exercise a wide
/// range of `λ` without runs taking implausibly long, coarse enough that
/// real measured latencies aren't swamped by the ~ns-scale timer-read
/// noise `Instant::now()` itself carries.
const SERVICE_NS: u64 = 5_000;
/// Tasks per run: large enough for stable mean/p99 estimates, small enough
/// that even the lowest swept `ρ_0` (longest generation time) finishes in
/// well under a second.
const TASK_COUNT: u64 = 4_000;
/// Matches `numa_information_cost.rs`'s choice: low enough that this
/// harness's task volumes build up real per-worker backlog and actually
/// exercise deflection, rather than the production default (80) which
/// these task volumes wouldn't otherwise reach.
const DEFLECTION_THRESHOLD: u8 = 15;

struct LoadRunResult {
    elapsed: Duration,
    latencies_ns: Vec<u64>,
}

impl LoadRunResult {
    fn mean_ns(&self) -> f64 {
        if self.latencies_ns.is_empty() {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let sum: f64 = self.latencies_ns.iter().map(|&x| x as f64).sum();
        sum / self.latencies_ns.len() as f64
    }

    fn percentile_ns(&self, p: f64) -> u64 {
        if self.latencies_ns.is_empty() {
            return 0;
        }
        #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
        let idx = (((self.latencies_ns.len() - 1) as f64) * p).round() as usize;
        self.latencies_ns[idx.min(self.latencies_ns.len() - 1)]
    }

    fn max_ns(&self) -> u64 {
        self.latencies_ns.iter().copied().max().unwrap_or(0)
    }

    fn throughput_tasks_per_ms(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let elapsed_ms = self.elapsed.as_secs_f64() * 1000.0;
        if elapsed_ms > 0.0 {
            #[allow(clippy::cast_precision_loss)]
            let t = self.latencies_ns.len() as f64 / elapsed_ms;
            t
        } else {
            0.0
        }
    }
}

fn collect_and_sort(latencies: &[std::sync::atomic::AtomicU64]) -> Vec<u64> {
    let mut v: Vec<u64> = latencies
        .iter()
        .map(|a| a.load(Ordering::Relaxed))
        .collect();
    v.sort_unstable();
    v
}

fn run_dta_load(topology: Topology, params: PoissonBotParams) -> LoadRunResult {
    let harness = DtaHarness::new(topology, 1 << 16);
    harness.set_deflection_threshold(DEFLECTION_THRESHOLD);
    let start = Instant::now();
    let latencies = std::thread::scope(|scope| {
        harness.run_workers(scope);
        let run = workloads::run_poisson_bot(&harness, params);
        run.countdown.wait_zero();
        harness.request_shutdown();
        run.latencies_ns
    });
    let elapsed = start.elapsed();
    // Safe only now: `std::thread::scope` above has already joined every
    // worker thread that could call `report_dta_hop`.
    harness.end_measurement();
    LoadRunResult {
        elapsed,
        latencies_ns: collect_and_sort(&latencies),
    }
}

fn run_ws_load(topology: Topology, params: PoissonBotParams) -> LoadRunResult {
    let sched = WsScheduler::new(topology);
    let start = Instant::now();
    let latencies = std::thread::scope(|scope| {
        sched.run_workers(scope);
        let run = workloads::run_poisson_bot(&sched, params);
        run.countdown.wait_zero();
        sched.request_shutdown();
        run.latencies_ns
    });
    let elapsed = start.elapsed();
    LoadRunResult {
        elapsed,
        latencies_ns: collect_and_sort(&latencies),
    }
}

fn print_header() {
    println!(
        "{:<6} {:<6} {:<10} {:>10} {:>12} {:>10} {:>12} {:>12} {:>12} {:>12}",
        "N",
        "sched",
        "rho_0",
        "tasks",
        "elapsed_ms",
        "tasks/ms",
        "lat_mean_ns",
        "lat_p50_ns",
        "lat_p99_ns",
        "lat_max_ns"
    );
}

#[allow(clippy::too_many_arguments)]
fn print_row(n: usize, sched_name: &str, rho_0: f64, r: &LoadRunResult) {
    #[allow(clippy::cast_precision_loss)]
    let elapsed_ms = r.elapsed.as_secs_f64() * 1000.0;
    println!(
        "{:<6} {:<6} {:<10.3} {:>10} {:>12.2} {:>10.2} {:>12.1} {:>12} {:>12} {:>12}",
        n,
        sched_name,
        rho_0,
        r.latencies_ns.len(),
        elapsed_ms,
        r.throughput_tasks_per_ms(),
        r.mean_ns(),
        r.percentile_ns(0.50),
        r.percentile_ns(0.99),
        r.max_ns(),
    );
}

fn run_sweep(worker_counts: &[usize], rho_0_values: &[f64]) {
    print_header();
    for &n in worker_counts {
        let topology = Topology::flat(n);
        for &rho_0 in rho_0_values {
            let params = PoissonBotParams::for_rho_0(rho_0, n, SERVICE_NS, TASK_COUNT);

            let dta = run_dta_load(topology, params);
            print_row(n, "dta", rho_0, &dta);

            let ws = run_ws_load(topology, params);
            print_row(n, "ws", rho_0, &ws);
        }
    }
}

fn main() {
    println!(
        "# Open-loop Poisson-arrival load-ratio sweep (paper/main.tex sec:dta_scope: \
         'DTA is most advantageous at high but subcritical load, rho_0 in [0.6, 0.95]')."
    );
    println!(
        "# service_ns={SERVICE_NS} (fixed per-task service time), tasks per run={TASK_COUNT}. \
         rho_0 = lambda / (N * mu), controlled via generator inter-arrival pacing \
         (dtact::benchmark::workloads::run_poisson_bot)."
    );
    println!(
        "# NOTE: paper/main.tex is a draft preprint. This sweep is a comparison point for \
         its scope-of-applicability claim, not a pass/fail validation — see \
         numa_information_cost.rs's module docs for the same caveat, which applies here too."
    );

    let physical_cores = std::thread::available_parallelism().map_or(8, std::num::NonZero::get);
    let worker_counts: Vec<usize> = [4usize, 8, 16]
        .into_iter()
        .filter(|&n| n <= physical_cores)
        .collect();

    // Spans below (0.1-0.5), inside (0.6-0.95), and above/at (0.99) the
    // paper's claimed sweet spot, so the sweep can show whether DTA's
    // relative advantage over WS actually peaks where the paper claims it
    // does, rather than only sampling the claimed band itself.
    let rho_0_values = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99];

    run_sweep(&worker_counts, &rho_0_values);
}

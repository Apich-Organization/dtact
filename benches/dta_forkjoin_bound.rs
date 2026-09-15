//! Fork-join makespan bound check against `paper/main.tex`
//! `Theorem~\ref{thm:fj_first}`:
//!
//! ```text
//! E[C_max^FJ] <= T_1/N + d*H_K/mu
//! ```
//!
//! where `d` is DAG depth (number of synchronised levels), `K` is the
//! join fan-in (tasks per level, all required before the next level may
//! start), `T_1 = (d*K)/mu` is total sequential work, `N` is worker count,
//! and `H_K` is the `K`-th harmonic number. The theorem's proof (ii) relies
//! specifically on each level's `K` tasks having i.i.d. `Exp(mu)` service
//! times (so the level's completion time is `E[max of K iid Exp(mu)] =
//! H_K/mu` by a birth-death first-passage argument) — [`run_level_sync_fj`]
//! below reproduces exactly that structure: `d` sequential barrier-synced
//! levels of `K` parallel exponentially-distributed tasks each, rather than
//! the deterministic-duration Fibonacci/UTS task graphs the other
//! benchmarks use (which would make the `H_K` term untestable, since a
//! deterministic level's completion time doesn't depend on `K` at all).
//!
//! ## The raw first-order bound is not the whole claim
//!
//! Measurement showed the raw first-order bound gets violated by *both*
//! DTA and WS, not just DTA, whenever `K > N` (more join fan-in than
//! workers). This is not a benchmark-fairness bug: the paper's own proof
//! of part (ii) models each level's completion as a birth-death chain with
//! transition rate `m·mu` at state `m` (`m` predecessors remaining) —
//! implicitly assuming all `m` remaining tasks can run *concurrently*,
//! which only holds while `m ≤ N`. When `K > N`, real hardware can only
//! service `N` of the `m` remaining tasks at once, so the true completion
//! rate is `min(m, N)·mu`, not `m·mu`, and the level takes systematically
//! longer than `H_K/mu` predicts — a gap in the theorem's applicability
//! domain (it doesn't state `K ≤ N` as a precondition, but its proof
//! needs it), not an implementation defect.
//!
//! Separately, the paper's own `Remark~\ref{rem:dag_combined}` ("Combined
//! bound and high-load asymptotics") explains a second, independent gap:
//! `Theorem~\ref{thm:fj_first}`'s first-order processing term implicitly
//! assumes each level starts from the *steady-state* (NESS) queue
//! distribution, but a genuinely level-synchronous DAG like this one
//! starts every level from a *batch-arrival* initial condition (all of a
//! level's tasks appear simultaneously) — `Proposition~\ref{prop:dag_2B}`
//! ("incomplete relaxation") bounds that deviation by `δ^(B) = 2d/γ`,
//! which the paper itself says "the first-order approximation becomes
//! unreliable near [this]" as `γ → 0`. [`FjParams::combined_bound_ns`]
//! computes the paper's own full bound (`eq:fj_full`: first-order +
//! `δ^(A)` Gumbel-variance term + `δ^(B)`), which is what should actually
//! hold for this workload shape — both bounds are reported below so the
//! gap between them is visible, not just whether either one passes.
//!
//! ## The `bal_max/mn` column: apparent DTA "serialization" is not a bug
//!
//! With `K` small (2, 4, 8 — under `dta_scheduler::LOAD_REFRESH_PERIOD`,
//! 32), DTA routinely runs an *entire* level-synchronous DAG on a single
//! worker (`bal_max/mn == N` — every task lands on one worker, `N-1`
//! workers do nothing). The mechanism is real: `Worker::push_local` only
//! refreshes `load_level` every `LOAD_REFRESH_PERIOD` *cumulative* pushes
//! (not reset per burst), so `enqueue_deflect`'s stay-local-vs-deflect
//! read can stay pinned to a stale, low pre-run value for an entire trial
//! whenever a chain of same-worker relay bursts never cumulatively crosses
//! that threshold — which every `K < 32` configuration here does.
//!
//! This was tested directly by lowering `DEFLECTION_THRESHOLD` (1 and 4,
//! vs. the default 15) to force earlier deflection: balance improved
//! (`bal_max/mn` dropped from `N` to ~1.3-2.4) but **mean makespan got
//! worse** in nearly every row, sometimes by 2-4x. For bursts this small,
//! the cross-worker mailbox-push-plus-wake cost of deflecting a task
//! outweighs the ~20us of work it saves running it elsewhere — DTA
//! defaulting to "stay local" here is a reasonable, not a defective,
//! outcome; forcing it to spread the work only adds overhead. The
//! `bal_max/mn` column is left in specifically so this is visible, not to
//! imply every high value is a problem.
//!
//! ## `d=8, K=16` (the heaviest config) needed `TRIALS = 400` to converge
//!
//! The `d=8, K=16` row (the deepest, widest-fan-in configuration, and the
//! only one where deflection partially engages at all — see above) is the
//! one most sensitive to sample size. At `TRIALS = 20` its mean swung
//! roughly 2-3x between otherwise-identical full-binary reruns (observed:
//! ~4800-10900us at `N=4`; ~7300-19600us at `N=6`; ~7600-22800us at `N=8`),
//! and whether it passed the combined bound looked like a coin flip at
//! every `N` — an earlier version of this comment drew a "clean
//! monotonic-in-`N`" conclusion from one such noisy run, which a repeat
//! promptly contradicted (`N=6` measuring *worse* than `N=8`). Raising
//! `TRIALS` to `400` (still well under a minute total runtime) resolved
//! this: three independent full-binary reruns at `TRIALS = 400` gave
//! `N=4 ∈ [4849, 6430]us` (bound 9300us, comfortably inside every time),
//! `N=6 ∈ [7586, 8194]us` (bound 9087us, inside every time), and
//! `N=8 ∈ [9022, 9555]us` (bound 8980us, **outside every time**, by
//! 0.5-6%) — a small, narrow, but now consistently reproduced violation
//! specific to `N=8`. This matches this machine's logical CPU count: `N=8`
//! workers plus this benchmark's own host/measurement thread is 9 threads
//! contending for 8 hardware contexts, while `N=6` (7 on 8) and `N=4` (5 on
//! 8) both have headroom — `N=6` is included alongside `4` and `8`
//! specifically to bracket this transition. Treat any `N ≥` this machine's
//! logical CPU count as measuring host contention layered on top of
//! whatever the schedulers themselves do, not a clean reading of either.
//! The broader lesson generalizes beyond this one row: this benchmark's
//! task volumes are small enough that `TRIALS` matters a lot — don't trust
//! a single low-`TRIALS` run's pass/fail verdict on the more demanding
//! configurations without checking it reproduces.
//!
//! **Cross-check against mean/median skew.** `numa_information_cost.rs`'s
//! beta-ratio investigation separately found this machine's per-run
//! timings are right-skewed by occasional large host/VM stalls, which
//! inflates a *mean* without moving the *median* — a different failure
//! mode than plain sample-size noise, worth checking independently before
//! trusting `TRIALS = 400`'s convergence. [`BoundCheck::median_ns`] and the
//! `outlier?` column (`mean > median * 1.3`) do exactly that. Result: every
//! `d=8, K=16` DTA row comes back `outlier? == false` — mean and median
//! agree closely there, so the `N=8` violation above is *not* a
//! mean-statistic artifact. The small-`K` rows (`K ∈ {2, 4, 8}`, the fully
//! serialized ones — see above) *are* flagged `outlier? == true` (mean
//! sometimes 2-3x the median), but this doesn't change any pass/fail
//! verdict for them since both statistics sit comfortably under their very
//! loose combined bound either way.
//!
//! This is *not* the same DAG shape as [`workloads::spawn_fib`] (unbounded
//! binary recursion, no explicit level barriers) — it is a purpose-built
//! level-synchronous generator matching the theorem's own model, sitting
//! directly in this bench file rather than in `workloads.rs` since it
//! exists only to exercise this one theorem.
//!
//! The theorem's protocol assumption ("DTA scheduling at each level" under
//! the two-phase protocol, `Definition~\ref{def:two_phase}`) is specific to
//! DTA; running the same generator through the WS baseline is still
//! reported for reference (the bound is a *DTA* claim, not a general
//! scheduler-agnostic one, so a WS row exceeding it is not a violation of
//! anything — only a DTA row exceeding it would be).
//!
//! Academic-only: requires `cargo bench --bench dta_forkjoin_bound
//! --features benchmark`.

use dtact::benchmark::dta_harness::DtaHarness;
use dtact::benchmark::instrumentation;
use dtact::benchmark::numa_model::{self, Topology};
use dtact::benchmark::work_stealing::WsScheduler;
use dtact::benchmark::workloads::{Countdown, TaskSpawner};

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const DEFLECTION_THRESHOLD: u8 = 15;
/// Repeated trials per configuration: the theorem bounds an *expectation*
/// (`E[C_max^FJ]`), so a single sample is not the right thing to compare
/// against it — this averages several independent runs instead. `400`, not
/// a smaller round number, because it was empirically tested: at `20` the
/// heaviest configuration (`d=8, K=16`) swung 2-3x between full-binary
/// reruns and its combined-bound pass/fail verdict didn't reproduce; `400`
/// converges it to a tight, reproducible range across reruns while the
/// whole sweep still finishes in well under a minute (see module docs,
/// "`d=8, K=16` needed `TRIALS = 400` to converge").
const TRIALS: u32 = 400;

#[derive(Clone, Copy, Debug)]
struct FjParams {
    /// DAG depth: number of sequential, barrier-synced levels.
    d: usize,
    /// Join fan-in: parallel tasks per level, all required before the next
    /// level may begin.
    k: usize,
    /// Mean per-task service time (`1/mu`), in nanoseconds. Actual
    /// per-task durations are drawn i.i.d. `Exp(mu)`, not fixed at this
    /// value, matching the theorem's assumption.
    mean_service_ns: u64,
}

impl FjParams {
    /// `H_K`, the `K`-th harmonic number.
    fn harmonic_k(self) -> f64 {
        (1..=self.k).map(|j| 1.0 / j as f64).sum()
    }

    /// `T_1`, total sequential work across all `d*K` tasks, in nanoseconds.
    fn t1_ns(self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let total_tasks = (self.d * self.k) as f64;
        total_tasks * self.mean_service_ns as f64
    }

    /// The theorem's first-order bound, `T_1/N + d*H_K/mu`, in nanoseconds.
    fn bound_ns(self, n: usize) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let n_f = n as f64;
        #[allow(clippy::cast_precision_loss)]
        let d_f = self.d as f64;
        self.t1_ns() / n_f + d_f * self.harmonic_k() * self.mean_service_ns as f64
    }

    /// Second-order correction `δ^(A)` (`paper/main.tex` Prop. `dag_2A`,
    /// "Gumbel fluctuation"): `z_{1-α}·π·√d / (√6·μ)`, the confidence-width
    /// term accounting for the *variance* of the K-way join time (the
    /// first-order bound only uses its mean, `H_K/mu`). Uses `α = 0.05`
    /// (one-sided `z ≈ 1.645`), a standard choice — the paper leaves `α`
    /// as a free parameter.
    fn delta_a_ns(self) -> f64 {
        const Z_95: f64 = 1.645;
        #[allow(clippy::cast_precision_loss)]
        let d_f = self.d as f64;
        #[allow(clippy::cast_precision_loss)]
        let mean_service_ns = self.mean_service_ns as f64;
        Z_95 * core::f64::consts::PI * d_f.sqrt() / 6.0f64.sqrt() * mean_service_ns
    }

    /// Second-order correction `δ^(B)` (`paper/main.tex` Prop. `dag_2B`,
    /// "incomplete relaxation"): `2d/γ`. Accounts for each DAG level
    /// starting from a *batch-arrival* initial condition (all of a level's
    /// tasks submitted simultaneously, exactly what [`spawn_level`]
    /// produces) rather than the steady-state (NESS) condition
    /// `Theorem~\ref{thm:fj_first}`'s own first-order processing term
    /// implicitly assumes. `γ` is the per-worker birth-death chain's
    /// spectral gap; the paper does not give a general closed form for it,
    /// only a concrete numeric example — `γ ≈ 0.04μ` "in the
    /// light-to-moderate load regime (`ϱ_0 ≤ 0.8`)" (Remark
    /// `dag_combined`, citing its own "Performance IV" Fig. `stab_L2`).
    /// This uses that paper-supplied value verbatim rather than
    /// independently re-deriving `γ` for this workload/hardware — treat
    /// the resulting combined bound as the paper's own best concrete
    /// instantiation, not an independently-validated number.
    fn delta_b_ns(self) -> f64 {
        const GAMMA_OVER_MU: f64 = 0.04;
        #[allow(clippy::cast_precision_loss)]
        let d_f = self.d as f64;
        #[allow(clippy::cast_precision_loss)]
        let mean_service_ns = self.mean_service_ns as f64;
        2.0 * d_f / GAMMA_OVER_MU * mean_service_ns
    }

    /// The paper's full combined bound (`eq:fj_full`, Remark
    /// `dag_combined`): first-order bound plus both second-order
    /// corrections. This — not the raw first-order bound alone — is what
    /// the paper itself says should hold for a genuinely batch-arrival,
    /// level-synchronous DAG like the one this benchmark generates.
    fn combined_bound_ns(self, n: usize) -> f64 {
        self.bound_ns(n) + self.delta_a_ns() + self.delta_b_ns()
    }
}

const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic per-task `Exp(mu)` duration draw, keyed by `(run_seed,
/// level, slot)` so concurrently-running sibling tasks never share mutable
/// RNG state (each derives its own draw independently, no synchronization
/// needed).
fn task_duration_ns(run_seed: u64, level: usize, slot: usize, mean_service_ns: u64) -> u64 {
    #[allow(clippy::cast_possible_truncation)]
    let key = run_seed ^ ((level as u64) << 32) ^ (slot as u64) ^ 0xABCD_EF01_1234_5678;
    let r = splitmix64(key);
    #[allow(clippy::cast_precision_loss)]
    let unit = ((r >> 11) as f64 * (1.0 / (1u64 << 53) as f64)).max(f64::MIN_POSITIVE);
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    let dur_ns = ((-unit.ln()) * mean_service_ns as f64) as u64;
    dur_ns.max(1)
}

/// Spawns level `level`'s `K` tasks (on the calling worker's fast on-worker
/// path — `s` is only ever `&S` from inside an already-running task here,
/// never the harness/main thread — see [`run_level_sync_fj`]). Whichever of
/// the `K` tasks happens to be the *last* to finish (`remaining` hits zero)
/// recurses to spawn level `level + 1` directly, again from on-worker: this
/// is exactly the theorem's "representative task waits for its K
/// predecessors" synchronisation, expressed as a continuation rather than
/// an external barrier, so no worker OS thread ever blocks waiting on
/// children (matching how [`workloads::spawn_fib`]'s `fib_task` structures
/// fork-join: every task returns immediately after spawning, the join is
/// implicit in the last-decrement race, not an explicit wait).
fn spawn_level<S: TaskSpawner>(
    spawner: &S,
    level: usize,
    params: FjParams,
    run_seed: u64,
    done: Arc<Countdown>,
) {
    if level >= params.d {
        done.done_one();
        return;
    }
    let remaining = Arc::new(AtomicU64::new(params.k as u64));
    for slot in 0..params.k {
        let dur_ns = task_duration_ns(run_seed, level, slot, params.mean_service_ns);
        let remaining = Arc::clone(&remaining);
        let done = Arc::clone(&done);
        spawner.spawn(Box::new(move |s: &S| {
            numa_model::burn_ns(dur_ns);
            // `fetch_sub` returns the *previous* value: `1` means this call
            // just took the counter from 1 to 0, i.e. this is the last of
            // the K siblings to finish.
            if remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                spawn_level(s, level + 1, params, run_seed, done);
            }
        }));
    }
}

/// Runs one `d`-level, `K`-way-join level-synchronous fork-join DAG on
/// `spawner`, blocking until it fully drains, and returns the measured
/// wall-clock makespan (`C_max^FJ`).
///
/// Exactly **one** task is spawned off-worker (the root, matching
/// [`workloads::spawn_fib`]'s precedent) — every subsequent task, at every
/// level, is spawned from within an already-running worker task via
/// [`spawn_level`]'s on-worker fast path. An earlier version of this
/// generator instead spawned every level's `K` tasks directly from this
/// (non-worker) function, `d` times — measurement showed that paid DTA's
/// off-worker dispatch latency (a real, but off-topic, mailbox round-trip
/// cost) `d*K` times per run instead of once, which dominated the makespan
/// and had nothing to do with the theorem under test.
fn run_level_sync_fj<S: TaskSpawner>(spawner: &S, params: FjParams, seed: u64) -> Duration {
    let done = Arc::new(Countdown::new());
    done.add(1);
    let start = Instant::now();
    let done_root = Arc::clone(&done);
    spawner.spawn(Box::new(move |s: &S| {
        spawn_level(s, 0, params, seed, done_root);
    }));
    done.wait_zero();
    start.elapsed()
}

struct BoundCheck {
    /// Mean elapsed makespan across all `TRIALS` runs — the statistic
    /// `Theorem~\ref{thm:fj_first}` actually bounds (`E[C_max^FJ]` is
    /// explicitly an expectation), so this is what `within_bound` and
    /// `within_combined` are checked against.
    mean_ns: f64,
    /// Median elapsed makespan across all `TRIALS` runs — reported
    /// alongside the mean because `numa_information_cost.rs`'s beta-ratio
    /// investigation found this development machine's per-run timings are
    /// right-skewed by occasional large outliers (background VM/host
    /// scheduling stalls), which inflate a mean but not a median. A large
    /// `mean_ns`/`median_ns` gap is a sign this row's `within_bound`
    /// verdict is being driven by a handful of stalled trials rather than
    /// the algorithm's own steady-state behavior — check it before trusting
    /// a `false` verdict at face value.
    median_ns: f64,
    bound_ns: f64,
    combined_bound_ns: f64,
    within_bound: bool,
    within_combined: bool,
    /// Mean, across all `TRIALS` runs, of `max completed / mean completed`
    /// across workers (see [`instrumentation::load_balance_stats`]) — `1.0`
    /// is perfectly even; values approaching `N` mean the work concentrated
    /// on roughly one worker instead of spreading across all `N`. Tracked
    /// to test the hypothesis that a burst smaller than
    /// `dta_scheduler::LOAD_REFRESH_PERIOD` (32) never gets a mid-burst
    /// `load_level` refresh, so `enqueue_deflect` can end up reading a
    /// stale, pre-burst load for every child in the burst and never
    /// deflecting any of them.
    bal_max_over_mean: f64,
}

fn median_sorted(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        f64::midpoint(xs[n / 2 - 1], xs[n / 2])
    }
}

/// Shared bound bookkeeping: turns per-trial makespan/balance samples into
/// a [`BoundCheck`] against both the first-order and combined bounds.
fn make_bound_check(
    topology: Topology,
    params: FjParams,
    mut samples_ns: Vec<f64>,
    total_bal: f64,
) -> BoundCheck {
    #[allow(clippy::cast_precision_loss)]
    let trials_f = samples_ns.len() as f64;
    let mean_ns = samples_ns.iter().sum::<f64>() / trials_f;
    let median_ns = median_sorted(&mut samples_ns);
    let bound_ns = params.bound_ns(topology.total_cores());
    let combined_bound_ns = params.combined_bound_ns(topology.total_cores());
    BoundCheck {
        mean_ns,
        median_ns,
        bound_ns,
        combined_bound_ns,
        within_bound: mean_ns <= bound_ns,
        within_combined: mean_ns <= combined_bound_ns,
        bal_max_over_mean: total_bal / trials_f,
    }
}

fn measure_dta(topology: Topology, params: FjParams) -> BoundCheck {
    let mut samples_ns = Vec::with_capacity(TRIALS as usize);
    let mut total_bal = 0.0;
    for trial in 0..TRIALS {
        let harness = DtaHarness::new(topology, 4096);
        harness.set_deflection_threshold(DEFLECTION_THRESHOLD);
        let elapsed = std::thread::scope(|scope| {
            harness.run_workers(scope);
            let elapsed = run_level_sync_fj(&harness, params, u64::from(trial));
            harness.request_shutdown();
            elapsed
        });
        harness.end_measurement();
        samples_ns.push(elapsed.as_secs_f64() * 1e9);
        total_bal +=
            instrumentation::load_balance_stats(&harness.per_worker_completed).max_over_mean;
    }
    make_bound_check(topology, params, samples_ns, total_bal)
}

fn measure_ws(topology: Topology, params: FjParams) -> BoundCheck {
    let mut samples_ns = Vec::with_capacity(TRIALS as usize);
    let mut total_bal = 0.0;
    for trial in 0..TRIALS {
        let sched = WsScheduler::new(topology);
        let elapsed = std::thread::scope(|scope| {
            sched.run_workers(scope);
            let elapsed = run_level_sync_fj(&sched, params, u64::from(trial));
            sched.request_shutdown();
            elapsed
        });
        samples_ns.push(elapsed.as_secs_f64() * 1e9);
        total_bal += instrumentation::load_balance_stats(&sched.per_worker_completed).max_over_mean;
    }
    // The raw first-order theorem is a DTA-specific claim (see module
    // docs), so WS's `within_bound`/`within_combined` here are
    // informational reference points, not violation checks.
    make_bound_check(topology, params, samples_ns, total_bal)
}

fn print_header() {
    println!(
        "{:<6} {:<6} {:<6} {:<6} {:>10} {:>14} {:>14} {:>14} {:>14} {:>14} {:<10} {:<10} {:<12} {:>10}",
        "N",
        "sched",
        "d",
        "K",
        "tasks",
        "T1_us",
        "bound_us",
        "combined_us",
        "emp_us(mean)",
        "emp_us(med)",
        "outlier?",
        "within_1st",
        "within_comb",
        "bal_max/mn"
    );
}

#[allow(clippy::too_many_arguments)]
fn print_row(n: usize, sched_name: &str, params: FjParams, check: &BoundCheck) {
    // Flags a mean noticeably above the median — the signature of a few
    // trials stalling on host/VM scheduling jitter rather than a genuine
    // shift in the algorithm's own steady-state behavior (see
    // `BoundCheck::median_ns`'s doc comment).
    let outlier_skewed = check.mean_ns > check.median_ns * 1.3;
    println!(
        "{:<6} {:<6} {:<6} {:<6} {:>10} {:>14.2} {:>14.2} {:>14.2} {:>14.2} {:>14.2} {:<10} {:<10} {:<12} {:>10.3}",
        n,
        sched_name,
        params.d,
        params.k,
        params.d * params.k,
        params.t1_ns() / 1000.0,
        check.bound_ns / 1000.0,
        check.combined_bound_ns / 1000.0,
        check.mean_ns / 1000.0,
        check.median_ns / 1000.0,
        outlier_skewed,
        check.within_bound,
        check.within_combined,
        check.bal_max_over_mean,
    );
}

fn main() {
    println!(
        "# Fork-join makespan bound check against paper/main.tex Theorem fj_first: \
         E[C_max^FJ] <= T_1/N + d*H_K/mu."
    );
    println!(
        "# Level-synchronous d-deep, K-way-join DAG with i.i.d. Exp(mu) per-task service \
         times (matching the theorem's own model) — NOT the same DAG shape as the \
         Fibonacci/UTS workloads used elsewhere in this harness."
    );
    println!(
        "# {TRIALS} trials averaged per configuration (the theorem bounds an expectation). \
         'within_1st' checks the raw Theorem fj_first bound (T1/N + d*H_K/mu); 'within_comb' \
         checks the paper's own combined bound (eq. fj_full = first-order + Gumbel-variance \
         term delta^(A) + batch-arrival/NESS-mismatch term delta^(B), using the paper's own \
         concrete gamma ~= 0.04*mu). Both are DTA-specific checks (the theorem's protocol \
         assumption is DTA-at-each-level); the WS rows are reference-only, not violations."
    );
    println!(
        "# NOTE: paper/main.tex is a draft preprint — see numa_information_cost.rs's module \
         docs for the same 'comparison point, not pass/fail validation' caveat."
    );

    let physical_cores = std::thread::available_parallelism().map_or(8, std::num::NonZero::get);
    let worker_counts: Vec<usize> = [4usize, 6, 8, 16]
        .into_iter()
        .filter(|&n| n <= physical_cores)
        .collect();

    let configs = [
        FjParams {
            d: 4,
            k: 2,
            mean_service_ns: 20_000,
        },
        FjParams {
            d: 4,
            k: 8,
            mean_service_ns: 20_000,
        },
        FjParams {
            d: 8,
            k: 4,
            mean_service_ns: 20_000,
        },
        FjParams {
            d: 8,
            k: 16,
            mean_service_ns: 20_000,
        },
    ];

    print_header();
    for &n in &worker_counts {
        let topology = Topology::flat(n);
        for &params in &configs {
            let dta = measure_dta(topology, params);
            print_row(n, "dta", params, &dta);

            let ws = measure_ws(topology, params);
            print_row(n, "ws", params, &ws);
        }
    }
}

//! Literature-standard task-graph workloads for the DTA-vs-WS comparison.
//!
//! Replaces the ad-hoc "sum 100 numbers" task body used by the
//! engineering-only benchmarks in `benches/scheduler_efficiency.rs`.
//!
//! Two workloads are provided:
//! * [`fib`]: fork-join Fibonacci, the canonical balanced-tree microbenchmark
//!   from the work-stealing literature (Blumofe et al., "Cilk: An Efficient
//!   Multithreaded Runtime System", 1995). Produces a regular, predictable
//!   task graph — good for measuring raw scheduling throughput in the
//!   "balanced bag-of-tasks" regime the preprint's `Δq → 0` analysis covers.
//! * [`uts`]: an Unbalanced-Tree-Search-inspired binomial tree (after
//!   Olivier et al., "UTS: An Unbalanced Tree Search Benchmark", 2007), the
//!   standard benchmark in the HPC/parallel-runtime literature specifically
//!   designed to stress dynamic load balancers with irregular,
//!   unpredictable parallelism. Produces genuine load imbalance (`Δq > 0`),
//!   which is exactly the regime the preprint's information-gap analysis
//!   (`eq:info_gap_2socket`) is about.
//!
//! Both are written generically against the [`TaskSpawner`] trait so the
//! exact same task graph runs, unmodified, on both the DTA harness and the
//! work-stealing baseline — the only thing that differs between the two
//! runs is the scheduler underneath.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// The minimal capability a scheduler-under-test must expose to run these
/// workloads: the ability to spawn a new, independently schedulable task
/// from within a currently-running one.
///
/// Deliberately *not* object-safe (`spawn` takes `impl FnOnce`, not
/// `Box<dyn FnOnce>`, at the call site) is avoided here specifically so
/// implementors can choose their own boxed-task representation; both
/// [`super::dta_harness`] and [`super::work_stealing`] implement this over
/// `Box<dyn FnOnce(&Self) + Send>`.
pub trait TaskSpawner: Sized + 'static {
    /// Schedules `task` to run at some point on some worker, without
    /// blocking the caller.
    fn spawn(&self, task: Box<dyn FnOnce(&Self) + Send>);
}

/// Shared outstanding-task counter used to detect when a fork-join task
/// graph has fully drained, without needing real join handles.
///
/// Protocol: the spawner increments the count *before* a new task becomes
/// reachable by any worker, and each task decrements it exactly once, after
/// it has finished spawning any children of its own — so the count never
/// observes a spurious zero while children are still being registered.
#[derive(Debug, Default)]
pub struct Countdown(AtomicU64);

impl Countdown {
    /// Creates a countdown starting at zero.
    #[must_use]
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Registers `n` additional outstanding tasks.
    #[inline]
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::AcqRel);
    }

    /// Marks one task as finished.
    #[inline]
    pub fn done_one(&self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }

    /// Busy-waits (with backoff) until every registered task has finished.
    pub fn wait_zero(&self) {
        let mut spins: u32 = 0;
        while self.0.load(Ordering::Acquire) != 0 {
            if spins < 4000 {
                core::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// Sequential (unforked) Fibonacci, used both as the recursion base case and
/// as the sequential reference implementation.
#[must_use]
pub const fn fib_seq(n: u64) -> u64 {
    if n < 2 {
        n
    } else {
        fib_seq(n - 1) + fib_seq(n - 2)
    }
}

/// Result and bookkeeping shared across one fork-join Fibonacci run.
pub struct FibRun {
    /// Accumulated sum of leaf Fibonacci values (NOT `fib(n)` itself — the
    /// task graph intentionally sums leaves independently rather than
    /// combining results pairwise, so no task needs to wait on its
    /// children's *results*, only on the graph draining — see module docs).
    pub sum: Arc<AtomicU64>,
    /// Outstanding-task tracker for this run.
    pub countdown: Arc<Countdown>,
}

/// Spawns a fork-join Fibonacci computation of `fib(n)`.
///
/// Forks down to `cutoff` (below which a task computes sequentially instead
/// of forking further — standard practice to bound task-creation overhead,
/// matching how real Cilk/TBB/Rayon fib benchmarks are parameterised).
///
/// Returns immediately; call [`Countdown::wait_zero`] on the returned
/// [`FibRun::countdown`] to block until the computation has fully drained,
/// then read [`FibRun::sum`].
pub fn spawn_fib<S: TaskSpawner>(spawner: &S, n: u64, cutoff: u64) -> FibRun {
    let run = FibRun {
        sum: Arc::new(AtomicU64::new(0)),
        countdown: Arc::new(Countdown::new()),
    };
    run.countdown.add(1);
    let sum = Arc::clone(&run.sum);
    let countdown = Arc::clone(&run.countdown);
    spawner.spawn(Box::new(move |s: &S| {
        fib_task(s, n, cutoff, &sum, &countdown);
    }));
    run
}

fn fib_task<S: TaskSpawner>(
    spawner: &S,
    n: u64,
    cutoff: u64,
    sum: &Arc<AtomicU64>,
    countdown: &Arc<Countdown>,
) {
    if n <= cutoff {
        sum.fetch_add(fib_seq(n), Ordering::Relaxed);
    } else {
        countdown.add(2);
        let sum_a = Arc::clone(sum);
        let sum_b = Arc::clone(sum);
        let cd_a = Arc::clone(countdown);
        let cd_b = Arc::clone(countdown);
        spawner.spawn(Box::new(move |s: &S| {
            fib_task(s, n - 1, cutoff, &sum_a, &cd_a);
        }));
        spawner.spawn(Box::new(move |s: &S| {
            fib_task(s, n - 2, cutoff, &sum_b, &cd_b);
        }));
    }
    countdown.done_one();
}

/// `SplitMix64` — a fast, well-distributed, non-cryptographic hash.
///
/// Used here purely to derive a reproducible per-node pseudo-random stream
/// from a `(parent_seed, child_index)` pair. The canonical UTS benchmark
/// spec uses SHA-1 for this; `SplitMix64` is used here instead to avoid a
/// cryptographic dependency for a property (uniform, reproducible per-node
/// branching) that doesn't require cryptographic strength — only the
/// branching *statistics* (each node's child count ~ Binomial(m, q)) matter
/// for stressing a load balancer, not the specific generator.
#[must_use]
const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Parameters for a Binomial-type UTS tree (Olivier et al. 2007).
///
/// Also carries a hard maximum depth kept as a termination safety net for a
/// benchmark context — the canonical spec relies purely on `m·q < 1`
/// (subcritical branching) for almost-sure finiteness, which holds for the
/// default parameters below, but a benchmark harness should never depend on
/// a probabilistic tail bound alone for termination.
#[derive(Debug, Clone, Copy)]
pub struct UtsParams {
    /// Root branching factor (children of the tree root).
    pub root_children: u32,
    /// Maximum candidate children per non-root node (the Binomial `m`).
    pub max_children: u32,
    /// Per-candidate-child survival probability (the Binomial `q`).
    /// Must satisfy `max_children as f64 * q < 1.0` for the tree to be
    /// subcritical (finite in expectation).
    pub q: f64,
    /// Hard depth cap; nodes at this depth never generate children
    /// regardless of `q`.
    pub max_depth: u32,
    /// Calibrated synthetic per-node work, in nanoseconds (busy-wait),
    /// representing task granularity independent of tree shape.
    pub node_work_ns: u64,
}

impl UtsParams {
    /// A "balanced-ish" configuration: subcritical branching
    /// (`max_children · q = 0.8 < 1`) with a wide root so the *expected*
    /// total population — `root_children / (1 - μ)` for a Galton-Watson
    /// process with per-node offspring mean `μ` — lands around 20,000
    /// nodes, enough to distribute meaningfully across up to ~64 workers,
    /// while staying safely subcritical (bounded variance, no risk of a
    /// depth-capped near-critical process ballooning unpredictably).
    #[must_use]
    pub const fn balanced() -> Self {
        Self {
            root_children: 4_000,
            max_children: 4,
            q: 0.2,
            max_depth: 16,
            node_work_ns: 200,
        }
    }

    /// A deliberately less-subcritical configuration (`μ = 0.9`), stressing
    /// dynamic load balancing harder via higher relative variance in
    /// subtree sizes (closer to the critical point) — the regime the
    /// preprint's information-gap analysis is most interested in (`Δq`
    /// large). Expected total population is similarly ~20,000 nodes
    /// (`root_children / (1 - μ)`), but with markedly heavier-tailed
    /// subtree-size variance than [`Self::balanced`].
    #[must_use]
    pub const fn unbalanced() -> Self {
        Self {
            root_children: 2_000,
            max_children: 6,
            q: 0.15,
            max_depth: 18,
            node_work_ns: 200,
        }
    }
}

/// Bookkeeping shared across one UTS run.
pub struct UtsRun {
    /// Total number of tree nodes visited (the standard UTS output metric).
    pub nodes_visited: Arc<AtomicU64>,
    /// Outstanding-task tracker for this run.
    pub countdown: Arc<Countdown>,
}

/// Spawns a UTS binomial-tree search rooted at `seed`. See [`UtsParams`] for
/// the tree-shape/imbalance knobs. Returns immediately; wait on
/// [`UtsRun::countdown`], then read [`UtsRun::nodes_visited`].
pub fn spawn_uts<S: TaskSpawner>(spawner: &S, seed: u64, params: UtsParams) -> UtsRun {
    let run = UtsRun {
        nodes_visited: Arc::new(AtomicU64::new(0)),
        countdown: Arc::new(Countdown::new()),
    };
    run.countdown.add(1);
    let nodes = Arc::clone(&run.nodes_visited);
    let countdown = Arc::clone(&run.countdown);
    spawner.spawn(Box::new(move |s: &S| {
        uts_task(s, seed, 0, params, &nodes, &countdown);
    }));
    run
}

fn uts_task<S: TaskSpawner>(
    spawner: &S,
    seed: u64,
    depth: u32,
    params: UtsParams,
    nodes: &Arc<AtomicU64>,
    countdown: &Arc<Countdown>,
) {
    super::numa_model::burn_ns(params.node_work_ns);
    nodes.fetch_add(1, Ordering::Relaxed);

    let children: u32 = if depth == 0 {
        params.root_children
    } else if depth >= params.max_depth {
        0
    } else {
        let mut count = 0u32;
        for i in 0..params.max_children {
            let h = splitmix64(seed ^ (u64::from(i).wrapping_mul(0x0001_0001_0001_0001)));
            // Standard "53 random bits -> f64 in [0,1)" recipe: `h >> 11`
            // keeps the top 53 bits (an f64 mantissa's worth), then scale
            // by 2^-53. `1u64 << 53` is exactly representable as f64 (a
            // power of two well within the 52-bit mantissa), so this cast
            // loses no precision despite the lint.
            #[allow(clippy::cast_precision_loss)]
            let unit = (h >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            if unit < params.q {
                count += 1;
            }
        }
        count
    };

    if children > 0 {
        countdown.add(u64::from(children));
        for i in 0..children {
            let child_seed = splitmix64(seed.wrapping_add(u64::from(i)).wrapping_mul(0x9E37_79B1));
            let nodes_c = Arc::clone(nodes);
            let cd_c = Arc::clone(countdown);
            spawner.spawn(Box::new(move |s: &S| {
                uts_task(s, child_seed, depth + 1, params, &nodes_c, &cd_c);
            }));
        }
    }
    countdown.done_one();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    type InlineTask = Box<dyn FnOnce(&Inline) + Send>;

    /// A trivial single-threaded, immediate-execution `TaskSpawner` used
    /// only to unit-test the workload generators' correctness in isolation
    /// from any real scheduler.
    struct Inline {
        queue: Mutex<Vec<InlineTask>>,
    }

    impl TaskSpawner for Inline {
        fn spawn(&self, task: InlineTask) {
            self.queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(task);
        }
    }

    impl Inline {
        fn drain(&self) {
            loop {
                let next = self
                    .queue
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .pop();
                match next {
                    Some(task) => task(self),
                    None => break,
                }
            }
        }
    }

    #[test]
    fn fib_fork_join_matches_sequential() {
        let inline = Inline {
            queue: Mutex::new(Vec::new()),
        };
        let run = spawn_fib(&inline, 15, 8);
        inline.drain();
        run.countdown.wait_zero();
        assert_eq!(run.sum.load(Ordering::Relaxed), fib_seq(15));
    }

    #[test]
    fn uts_terminates_and_visits_at_least_root() {
        let inline = Inline {
            queue: Mutex::new(Vec::new()),
        };
        let run = spawn_uts(&inline, 42, UtsParams::balanced());
        inline.drain();
        run.countdown.wait_zero();
        assert!(run.nodes_visited.load(Ordering::Relaxed) >= 1);
    }

    /// Sanity-checks the branching generator's qualitative statistical
    /// behavior — a process nearer the critical point (`μ → 1`) has higher
    /// *relative* variance in total population than one further below it —
    /// using small, fast, deliberately few-branch parameters chosen only to
    /// make that property visible quickly. This is independent of whether
    /// [`UtsParams::balanced`]/[`UtsParams::unbalanced`] (tuned instead for
    /// realistic sweep task counts, where many-root-branch averaging would
    /// otherwise wash out the same effect via the central limit theorem)
    /// individually produce a visible CV gap at their own scale.
    #[test]
    fn uts_branching_closer_to_critical_has_higher_relative_variance() {
        const FEW_BRANCHES: UtsParams = UtsParams {
            root_children: 6,
            max_children: 4,
            q: 0.15, // μ = 0.6
            max_depth: 20,
            node_work_ns: 0,
        };
        const NEAR_CRITICAL: UtsParams = UtsParams {
            root_children: 6,
            max_children: 6,
            q: 0.1583, // μ ≈ 0.95
            max_depth: 40,
            node_work_ns: 0,
        };

        fn run_sizes(params: UtsParams, seeds: u64) -> Vec<f64> {
            (0..seeds)
                .map(|seed| {
                    let inline = Inline {
                        queue: Mutex::new(Vec::new()),
                    };
                    let run = spawn_uts(&inline, seed, params);
                    inline.drain();
                    #[allow(clippy::cast_precision_loss)]
                    let n = run.nodes_visited.load(Ordering::Relaxed) as f64;
                    n
                })
                .collect()
        }
        fn coefficient_of_variation(samples: &[f64]) -> f64 {
            #[allow(clippy::cast_precision_loss)]
            let n = samples.len() as f64;
            let mean = samples.iter().sum::<f64>() / n;
            let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            variance.sqrt() / mean
        }

        let low_cv = coefficient_of_variation(&run_sizes(FEW_BRANCHES, 60));
        let high_cv = coefficient_of_variation(&run_sizes(NEAR_CRITICAL, 60));
        assert!(
            high_cv > low_cv,
            "expected near-critical CV ({high_cv}) > subcritical CV ({low_cv})"
        );
    }
}

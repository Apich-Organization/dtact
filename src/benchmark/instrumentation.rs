//! Counts and times the real information-acquisition events each scheduler
//! performs.
//!
//! This lets the empirical information-acquisition rate `β_π`
//! (`paper/main.tex`, Definition "Information acquisition rate", eq.
//! `beta_def`) be computed directly from a live run rather than assumed.
//!
//! The paper's discrete-to-continuous correspondence is:
//! `C_π(T) = #_SPSC(T) · c_SPSC + Σ_(CAS event k) c_CAS(n_k)`.
//! This module accumulates exactly those two terms — a count/cost for SPSC
//! reads, and a count/cost for CAS attempts (split into successes and
//! failed/retried attempts, since the CAS-contention lemma's `Θ(log n)`
//! prediction is about the *retry* count) — from real timed events.

use core::sync::atomic::{AtomicU64, Ordering};

/// Accumulates real timed information-acquisition events for one scheduler
/// under test over the course of one benchmark run.
///
/// All fields are independent monotonic counters updated with `Relaxed`
/// ordering from many worker threads concurrently; correctness only
/// requires that the final totals (read after all workers have joined) are
/// self-consistent, not that any two fields are observed atomically
/// together mid-run.
#[derive(Debug, Default)]
pub struct AcquisitionMeter {
    /// Number of SPSC-read information-acquisition events observed.
    pub spsc_reads: AtomicU64,
    /// Total nanoseconds charged against SPSC-read events (real measured
    /// local cost plus any cross-socket penalty from [`super::numa_model`]).
    pub spsc_ns: AtomicU64,
    /// Number of CAS attempts observed (successes and failures/retries).
    pub cas_attempts: AtomicU64,
    /// Number of CAS attempts that succeeded on the first try.
    pub cas_successes: AtomicU64,
    /// Total nanoseconds charged against CAS events.
    pub cas_ns: AtomicU64,
    /// Total number of tasks completed (the `Nλ` normalizer: dividing
    /// accumulated cost by this count yields per-task acquisition cost).
    pub tasks_completed: AtomicU64,
}

impl AcquisitionMeter {
    /// Creates a fresh, zeroed meter.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            spsc_reads: AtomicU64::new(0),
            spsc_ns: AtomicU64::new(0),
            cas_attempts: AtomicU64::new(0),
            cas_successes: AtomicU64::new(0),
            cas_ns: AtomicU64::new(0),
            tasks_completed: AtomicU64::new(0),
        }
    }

    /// Records one SPSC-read information-acquisition event costing `ns`
    /// nanoseconds (real measured cost plus any charged cross-socket
    /// penalty).
    #[inline]
    pub fn record_spsc(&self, ns: u64) {
        self.spsc_reads.fetch_add(1, Ordering::Relaxed);
        self.spsc_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Records one CAS attempt costing `ns` nanoseconds, noting whether it
    /// succeeded.
    #[inline]
    pub fn record_cas(&self, ns: u64, success: bool) {
        self.cas_attempts.fetch_add(1, Ordering::Relaxed);
        if success {
            self.cas_successes.fetch_add(1, Ordering::Relaxed);
        }
        self.cas_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Adds `ns` of extra charged cost to the running SPSC-cost total
    /// without counting it as an additional read event — used when a
    /// penalty (e.g. [`super::numa_model::charge_cross_socket_if_needed`])
    /// applies on top of a read that a caller already recorded via
    /// [`Self::record_spsc`] as its own event.
    #[inline]
    pub fn record_extra_ns(&self, ns: u64) {
        self.spsc_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Records that one task ran to completion (the `Nλ` normalizer).
    #[inline]
    pub fn record_task_completed(&self) {
        self.tasks_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// A point-in-time, non-atomic-across-fields snapshot of the meter,
    /// suitable for reporting once all workers have joined.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            spsc_reads: self.spsc_reads.load(Ordering::Relaxed),
            spsc_ns: self.spsc_ns.load(Ordering::Relaxed),
            cas_attempts: self.cas_attempts.load(Ordering::Relaxed),
            cas_successes: self.cas_successes.load(Ordering::Relaxed),
            cas_ns: self.cas_ns.load(Ordering::Relaxed),
            tasks_completed: self.tasks_completed.load(Ordering::Relaxed),
        }
    }
}

/// A resolved, immutable snapshot of an [`AcquisitionMeter`]'s counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct Snapshot {
    /// See [`AcquisitionMeter::spsc_reads`].
    pub spsc_reads: u64,
    /// See [`AcquisitionMeter::spsc_ns`].
    pub spsc_ns: u64,
    /// See [`AcquisitionMeter::cas_attempts`].
    pub cas_attempts: u64,
    /// See [`AcquisitionMeter::cas_successes`].
    pub cas_successes: u64,
    /// See [`AcquisitionMeter::cas_ns`].
    pub cas_ns: u64,
    /// See [`AcquisitionMeter::tasks_completed`].
    pub tasks_completed: u64,
}

impl Snapshot {
    /// Total accumulated information-acquisition cost, in nanoseconds:
    /// `C_π(T) = #_SPSC · c_SPSC-measured + Σ c_CAS-measured`
    /// (paper eq. "cdot"/discrete correspondence).
    #[must_use]
    pub const fn total_ns(&self) -> u64 {
        self.spsc_ns + self.cas_ns
    }

    /// Empirical information-acquisition rate `β_π`, in nanoseconds of
    /// acquisition cost per completed task — the discrete analogue of
    /// `β_π = lim_{T→∞} C_π(T)/T` normalized by task throughput so that
    /// runs of different lengths/task counts are comparable, and directly
    /// comparable to the paper's per-task closed forms
    /// (`β_DTA(N) = Nλ·80ns`, `β_WS(N) = Nλ·(190+100·log2 N)ns`, which are
    /// themselves total-system rates; dividing by `N·λ` — here,
    /// `tasks_completed / T` — recovers the same per-task quantity).
    ///
    /// Returns `0.0` if no tasks completed (avoids a division by zero).
    #[must_use]
    pub fn ns_per_task(&self) -> f64 {
        if self.tasks_completed == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let ratio = self.total_ns() as f64 / self.tasks_completed as f64;
            ratio
        }
    }

    /// Mean CAS retry count implied by this snapshot:
    /// `cas_attempts / cas_successes`. The CAS-contention lemma
    /// (`paper/main.tex`, Lemma "CAS contention cost") predicts this grows
    /// as `Θ(log N)` for the work-stealing baseline under contention.
    /// Returns `0.0` if there were no successful CAS events.
    #[must_use]
    pub fn mean_cas_attempts_per_success(&self) -> f64 {
        if self.cas_successes == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let ratio = self.cas_attempts as f64 / self.cas_successes as f64;
            ratio
        }
    }
}

/// Real-execution load-balance-quality statistics computed from each
/// worker's completed-task count.
///
/// This is a separate empirical claim from the information-acquisition-rate
/// model above — the preprint's "Load Balance" / stability analysis
/// (`paper/main.tex`, secs. on stability metrics and `Δq` queue imbalance)
/// is about how *evenly* work ends up spread across workers, not about the
/// cost of finding it. Computed from real per-worker task counts recorded
/// during an actual run — not modeled or assumed.
#[derive(Debug, Clone, Copy)]
pub struct LoadBalanceStats {
    /// Number of workers.
    pub workers: usize,
    /// Total tasks completed across all workers.
    pub total_tasks: u64,
    /// Busiest worker's completed-task count.
    pub max_tasks: u64,
    /// Idlest worker's completed-task count.
    pub min_tasks: u64,
    /// Mean completed-task count per worker.
    pub mean_tasks: f64,
    /// Coefficient of variation (stddev / mean) of per-worker task counts —
    /// `0` for perfectly even distribution, larger for more imbalance.
    pub cv: f64,
    /// Busiest worker's share of the mean (`max / mean`) — `1.0` for
    /// perfectly even distribution.
    pub max_over_mean: f64,
}

/// Computes [`LoadBalanceStats`] from each worker's completed-task counter.
#[must_use]
pub fn load_balance_stats(per_worker: &[AtomicU64]) -> LoadBalanceStats {
    let counts: Vec<u64> = per_worker
        .iter()
        .map(|c| c.load(Ordering::Relaxed))
        .collect();
    let workers = counts.len();
    let total_tasks: u64 = counts.iter().sum();
    let max_tasks = counts.iter().copied().max().unwrap_or(0);
    let min_tasks = counts.iter().copied().min().unwrap_or(0);
    #[allow(clippy::cast_precision_loss)]
    let mean_tasks = if workers == 0 {
        0.0
    } else {
        total_tasks as f64 / workers as f64
    };
    let cv = if mean_tasks > 0.0 {
        #[allow(clippy::cast_precision_loss)]
        let variance = counts
            .iter()
            .map(|&c| (c as f64 - mean_tasks).powi(2))
            .sum::<f64>()
            / workers as f64;
        variance.sqrt() / mean_tasks
    } else {
        0.0
    };
    #[allow(clippy::cast_precision_loss)]
    let max_over_mean = if mean_tasks > 0.0 {
        max_tasks as f64 / mean_tasks
    } else {
        0.0
    };
    LoadBalanceStats {
        workers,
        total_tasks,
        max_tasks,
        min_tasks,
        mean_tasks,
        cv,
        max_over_mean,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_snapshots_correctly() {
        let meter = AcquisitionMeter::new();
        meter.record_spsc(80);
        meter.record_spsc(380); // e.g. one cross-socket-penalized read
        meter.record_cas(100, true);
        meter.record_cas(100, false);
        meter.record_task_completed();
        meter.record_task_completed();

        let snap = meter.snapshot();
        assert_eq!(snap.spsc_reads, 2);
        assert_eq!(snap.spsc_ns, 460);
        assert_eq!(snap.cas_attempts, 2);
        assert_eq!(snap.cas_successes, 1);
        assert_eq!(snap.cas_ns, 200);
        assert_eq!(snap.tasks_completed, 2);
        assert_eq!(snap.total_ns(), 660);
        assert!((snap.ns_per_task() - 330.0).abs() < f64::EPSILON);
        assert!((snap.mean_cas_attempts_per_success() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ns_per_task_avoids_div_by_zero() {
        let meter = AcquisitionMeter::new();
        assert!((meter.snapshot().ns_per_task() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn load_balance_stats_perfectly_even() {
        let counters: Vec<AtomicU64> = (0..4).map(|_| AtomicU64::new(10)).collect();
        let stats = load_balance_stats(&counters);
        assert_eq!(stats.workers, 4);
        assert_eq!(stats.total_tasks, 40);
        assert_eq!(stats.max_tasks, 10);
        assert_eq!(stats.min_tasks, 10);
        assert!((stats.cv - 0.0).abs() < f64::EPSILON);
        assert!((stats.max_over_mean - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn load_balance_stats_detects_imbalance() {
        let counters = vec![
            AtomicU64::new(100),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        ];
        let stats = load_balance_stats(&counters);
        assert_eq!(stats.total_tasks, 100);
        assert!(stats.cv > 1.0, "expected high CV, got {}", stats.cv);
        assert!((stats.max_over_mean - 4.0).abs() < f64::EPSILON);
    }
}

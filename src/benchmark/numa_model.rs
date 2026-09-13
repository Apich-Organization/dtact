//! Synthetic dual-socket NUMA topology and calibrated cross-node cost model.
//!
//! This development machine (see the session's hardware survey) is a
//! single-socket, single-NUMA-node 8-core laptop part with no KVM
//! acceleration available (Xen guest, no VT-x exposed) — so there is no way
//! to obtain genuine cross-socket latency asymmetry here, whether on bare
//! metal or inside a VM. Full-system virtualization does not change this:
//! `qemu -numa dist,...` only fabricates ACPI SRAT/SLIT tables for the guest
//! kernel to read — it does not make the emulator actually delay memory
//! accesses accordingly, so it cannot manufacture real latency asymmetry
//! that the underlying host memory system doesn't have.
//!
//! The only methodologically honest option is a **synthetic, calibrated cost
//! model**: real hardware supplies the true *local* (intra-socket) timings
//! (measured live, not injected), and a documented, literature-derived
//! penalty is added on top of that real measurement whenever a scheduling
//! decision would, on real dual-socket hardware, have crossed a socket
//! boundary. This mirrors standard practice in NUMA-simulation research
//! (e.g. gem5-style fixed remote-access penalties) when the target topology
//! isn't physically available.
//!
//! The constants below are taken **verbatim** from the DTA preprint's own
//! worked example (`paper/main.tex`, §"Concrete Analysis on a Dual-Socket
//! NUMA Topology"), which in turn cites them as "illustrative estimates
//! drawn from published hardware latency figures... not latencies measured
//! on our own testbed" for a representative modern dual-socket server (e.g.
//! AMD EPYC 7763 / Intel Xeon Platinum 8380). Using the same numbers here —
//! rather than inventing new ones — means this benchmark's results are
//! directly comparable to the paper's existing closed-form predictions
//! (Eq. `beta_dta_num`, `beta_ws_num`, `gamma_star`).

/// Intra-socket SPSC latency `δ_intra`, in nanoseconds (paper §`numa_concrete`).
pub const DELTA_INTRA_NS: u64 = 80;
/// Cross-socket NUMA latency `δ_inter`, in nanoseconds (paper §`numa_concrete`).
pub const DELTA_INTER_NS: u64 = 300;
/// Uncontended CAS latency `c_CAS^(0)`, in nanoseconds (paper §`numa_concrete`).
pub const CAS_BASE_NS: u64 = 100;
/// SPSC read cost `c_SPSC` (paper sets this equal to `δ_intra`).
pub const SPSC_COST_NS: u64 = DELTA_INTRA_NS;

/// Additive cross-socket penalty: `δ_inter - δ_intra`.
///
/// Charged on top of a real, locally-measured operation when that
/// operation crosses a virtual socket boundary — in addition to (not
/// instead of) the real measured local cost, so the model never claims to
/// replace real timing; it only approximates the extra cost real
/// dual-socket hardware would have added.
pub const CROSS_SOCKET_PENALTY_NS: u64 = DELTA_INTER_NS - DELTA_INTRA_NS;

/// A declared virtual NUMA topology: `sockets` sockets, `cores_per_socket`
/// worker cores each, for a total of `sockets * cores_per_socket` workers.
///
/// The paper's concrete worked example uses the simplest non-trivial case,
/// `sockets = 2` (its group-theoretic model is
/// `G = Z_2 × Z_{N/2}`, §`numa_concrete`) — that is the default topology used
/// by the sweep harness, but the type is general so a flat, single-socket
/// topology (`sockets = 1`, the "no NUMA effect" control condition) can be
/// swept alongside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Topology {
    /// Number of virtual sockets (NUMA nodes) in the declared topology.
    pub sockets: usize,
    /// Worker cores assigned to each socket.
    pub cores_per_socket: usize,
}

impl Topology {
    /// A flat, single-socket topology: the "no NUMA effect" control
    /// condition, where every worker is local to every other worker.
    #[must_use]
    pub const fn flat(total_cores: usize) -> Self {
        Self {
            sockets: 1,
            cores_per_socket: total_cores,
        }
    }

    /// The paper's worked example: two sockets, `total_cores / 2` cores
    /// each (`G = Z_2 × Z_{N/2}`, §`numa_concrete`). `total_cores` must be
    /// even; odd inputs round the second socket down by one core.
    #[must_use]
    pub const fn dual_socket(total_cores: usize) -> Self {
        Self {
            sockets: 2,
            cores_per_socket: total_cores / 2,
        }
    }

    /// Total worker count this topology declares.
    #[must_use]
    pub const fn total_cores(&self) -> usize {
        self.sockets * self.cores_per_socket
    }

    /// The virtual socket (NUMA node) id a worker core belongs to.
    #[must_use]
    pub const fn socket_of(&self, core: usize) -> usize {
        match core.checked_div(self.cores_per_socket) {
            Some(socket) => socket,
            None => 0,
        }
    }

    /// Whether two worker cores are local to each other (same virtual
    /// socket) under this topology.
    #[must_use]
    pub const fn same_socket(&self, a: usize, b: usize) -> bool {
        self.socket_of(a) == self.socket_of(b)
    }
}

/// Actively burns wall-clock time for approximately `nanos` nanoseconds.
///
/// Uses a calibrated busy-wait so the synthetic cost is genuinely incurred
/// by the calling thread (consuming pipeline/cache resources the way a real
/// stalled remote memory access would) rather than merely recorded after
/// the fact. A sleeping wait would understate contention effects between
/// concurrently "penalized" threads; a real spin does not.
///
/// No-ops when `nanos == 0` (the common case for intra-socket operations).
#[inline]
pub fn burn_ns(nanos: u64) {
    if nanos == 0 {
        return;
    }
    let start = std::time::Instant::now();
    let target = std::time::Duration::from_nanos(nanos);
    while start.elapsed() < target {
        core::hint::spin_loop();
    }
}

/// Charges the cross-socket penalty if `source` and `target` differ in
/// virtual socket, on top of whatever real local cost the caller measured.
///
/// Returns the number of nanoseconds actually charged (`0` if same-socket),
/// so callers can fold it into what they record in an
/// [`super::instrumentation::AcquisitionMeter`]. This is the single
/// function every scheduler-under-test calls at its cross-worker
/// information-acquisition points, so both sides of the comparison are
/// held to identically calibrated costs.
#[inline]
#[must_use]
pub fn charge_cross_socket_if_needed(topology: &Topology, source: usize, target: usize) -> u64 {
    if topology.same_socket(source, target) {
        0
    } else {
        burn_ns(CROSS_SOCKET_PENALTY_NS);
        CROSS_SOCKET_PENALTY_NS
    }
}

/// Measures this real machine's own uncontended atomic-load latency.
///
/// A tight loop of relaxed loads on a hot, uncontended `AtomicU64`, purely
/// for diagnostic reporting alongside the sweep results — e.g. "how does
/// this machine's real local cost compare to the paper's assumed 80ns
/// `δ_intra`?" This value is never substituted into the cost model itself:
/// doing so would make results incomparable to the paper's own published
/// table, which is exactly the comparison this benchmark exists to produce.
#[must_use]
pub fn calibrate_local_atomic_load_ns() -> f64 {
    use core::sync::atomic::{AtomicU64, Ordering};
    const WARMUP: u32 = 10_000;
    const SAMPLES: u32 = 1_000_000;
    let counter = AtomicU64::new(0);
    for _ in 0..WARMUP {
        core::hint::black_box(counter.load(Ordering::Relaxed));
    }
    let start = std::time::Instant::now();
    for _ in 0..SAMPLES {
        core::hint::black_box(counter.load(Ordering::Relaxed));
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1e9 / f64::from(SAMPLES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_socket_splits_evenly() {
        let topo = Topology::dual_socket(64);
        assert_eq!(topo.sockets, 2);
        assert_eq!(topo.cores_per_socket, 32);
        assert_eq!(topo.total_cores(), 64);
        assert!(topo.same_socket(0, 31));
        assert!(!topo.same_socket(0, 32));
    }

    #[test]
    fn flat_topology_is_always_local() {
        let topo = Topology::flat(16);
        assert!(topo.same_socket(0, 15));
    }

    #[test]
    fn burn_ns_actually_waits() {
        let start = std::time::Instant::now();
        burn_ns(1_000_000); // 1ms
        assert!(start.elapsed() >= std::time::Duration::from_micros(900));
    }
}

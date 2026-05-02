/// Topology Strategy for the scheduler and fibers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyMode {
    /// Peer-to-Peer Mesh: Tasks are deflected to neighbors based on load.
    P2PMesh,
    /// Global: Tasks are shared across all cores via a common pool (unsupported in V3).
    Global,
}

/// Metadata: Workload Hint for scheduling decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadKind {
    /// Latency-sensitive compute tasks.
    Compute,
    /// Throughput-oriented I/O tasks.
    IO,
    /// Memory-intensive scanning or bulk transfers.
    Memory,
    /// Background maintenance or telemetry tasks.
    System,
}

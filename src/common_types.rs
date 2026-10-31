/// Topology Strategy for the scheduler and fibers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyMode {
    /// Peer-to-Peer Mesh: Tasks are deflected to neighbors based on load.
    P2PMesh,
    /// Global: Tasks are shared across all cores via a common pool.
    Global,
    /// Pinned: Fiber must stay on its origin worker. No cross-core deflection.
    /// Set automatically when the fiber's switcher is a `SameThread` variant —
    /// `SameThread` context-switch assumes no thread migration, so deflection
    /// would violate the switcher's invariants.
    Pinned,
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

impl WorkloadKind {
    /// Initial adaptive spin budget for a fiber of this kind, in
    /// `spin_loop` iterations, used by
    /// [`crate::future_bridge::wait_pinned`].
    ///
    /// This is the *starting* value of the fiber's self-tuning spin budget:
    /// a compute fiber is worth spinning on for a long time (it is likely
    /// to become ready on the very next poll), while an I/O fiber that just
    /// returned `Pending` is probably waiting on the kernel and is better
    /// handed back to the scheduler promptly so the worker can run something
    /// else. The loop then adapts from there in both directions, so this is
    /// a starting point rather than a hard cap.
    ///
    /// Single source of truth: both the Rust spawn paths and the C FFI
    /// (which receives the kind as a `u8`) resolve through this, so the two
    /// cannot drift apart.
    #[inline(always)]
    #[must_use]
    pub const fn spin_budget(self) -> u32 {
        match self {
            Self::Compute => 1000,
            Self::IO => 100,
            Self::Memory => 500,
            Self::System => 200,
        }
    }

    /// Packed discriminant matching the C ABI's `kind` field.
    #[inline(always)]
    #[must_use]
    pub const fn level(self) -> u8 {
        match self {
            Self::Compute => 0,
            Self::IO => 1,
            Self::Memory => 2,
            Self::System => 3,
        }
    }

    /// Inverse of [`Self::level`]; anything out of range maps to `Compute`.
    #[inline(always)]
    #[must_use]
    pub const fn from_level(level: u8) -> Self {
        match level {
            1 => Self::IO,
            2 => Self::Memory,
            3 => Self::System,
            _ => Self::Compute,
        }
    }
}

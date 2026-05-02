//! # Dtact-V3: Distributed Task-Aware Coroutine Toolkit
//! 
//! Dtact is a high-performance, low-latency asynchronous runtime designed for systems-level
//! programming across heterogeneous architectures (x86_64, AArch64, RISC-V).
//! 
//! ## Core Architecture
//! 1. **Lock-Free Arena**: A page-aligned memory pool for fiber contexts, providing O(1) allocation
//!    and hardware-level guard pages for memory safety.
//! 2. **P2P Scheduler Mesh**: A distributed work-stealing/deflection scheduler that minimizes L3 
//!    cache thrashing and maximizes NUMA-local execution.
//! 3. **Zero-Copy Migration**: Leveraging self-referential futures and direct stack-top injection 
//!    to move running tasks across cores without heap allocation.
//! 
//! ## Safety & Safety Levels
//! Dtact provides tiered safety levels (0-2) allowing developers to trade off between raw 
//! performance and hardware-enforced isolation (e.g., guard pages and SEH registration).

// =========================================================================
// RUST LINT CONFIGURATION: dtact
// =========================================================================

// -------------------------------------------------------------------------
// LEVEL 1: CRITICAL ERRORS (Deny)
// -------------------------------------------------------------------------
#![deny(
    unreachable_code,
    improper_ctypes_definitions,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms,
    clippy::perf,
    clippy::correctness,
    clippy::suspicious,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::missing_safety_doc,
    clippy::same_item_push,
    clippy::implicit_clone,
    clippy::all,
    clippy::pedantic,
    missing_docs,
    clippy::nursery,
    clippy::single_call_fn,
)]
// -------------------------------------------------------------------------
// LEVEL 2: STYLE WARNINGS (Warn)
// -------------------------------------------------------------------------
#![warn(
    dead_code,
    warnings,
    clippy::dbg_macro,
    clippy::todo,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::unnecessary_safety_comment
)]
// -------------------------------------------------------------------------
// LEVEL 3: ALLOW/IGNORABLE (Allow)
// -------------------------------------------------------------------------
#![allow(
    unsafe_code,
    unused_unsafe,
    private_interfaces,
    clippy::restriction,
    clippy::inline_always,
    unused_doc_comments,
    clippy::empty_line_after_doc_comments
)]
#![crate_name = "dtact"]

extern crate alloc;

/// Bridge for polling futures within a FiberContext.
pub mod future_bridge;
/// Low-level assembly-based context switching primitives.
pub mod context_switch;
/// Distributed P2P Mesh scheduler implementation.
pub mod dta_scheduler;
/// Lock-free arena and OS-level memory management.
pub mod memory_management;
/// C-compatible FFI boundary for cross-language integration.
pub mod c_ffi;
/// Timing, topology, and OS-specific primitives.
pub mod utils;
/// Standard error types for the Dtact runtime.
pub mod errors;
/// Public user-facing API for spawning and managing fibers.
pub mod api;

pub use api::*;

/// DTA-V3 Runtime Environment.
/// 
/// Consolidates the distributed scheduler and the memory pool into a single
/// unit to ensure architectural consistency across all worker threads.
pub struct Runtime {
    /// The distributed P2P work-deflection scheduler.
    pub(crate) scheduler: dta_scheduler::DtaScheduler,
    /// The lock-free arena for managing fiber stacks and contexts.
    pub(crate) pool: memory_management::ContextPool,
}

/// Global Singleton for the Runtime Environment.
/// 
/// This is initialized exactly once per process via `dtact_init` or
/// implicit autostart triggers in the proc-macro layer.
pub(crate) static GLOBAL_RUNTIME: std::sync::OnceLock<Runtime> = std::sync::OnceLock::new();

/// Telemetry: Tracks fibers that failed the 8KB zero-copy check and fell back to heap allocation.
/// 
/// A high value indicates that captured future sizes exceed the pre-allocated
/// stack-top buffer, causing a performance cliff due to heap traffic.
pub static HEAP_ESCAPED_SPAWNS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Awakens a suspended fiber by pushing it onto the DTA-V3 Scheduler mesh.
/// 
/// This function is the primary signaling mechanism for cross-thread wakeups.
/// It uses the fiber's index as a flow-id for deterministic load distribution
/// across the worker cores.
/// 
/// # Arguments
/// * `origin_core` - The core ID where the fiber was originally spawned.
/// * `fiber_index` - The unique identifier of the fiber in the context pool.
#[inline(always)]
pub(crate) fn wake_fiber(origin_core: usize, fiber_index: u32) {
    if let Some(runtime) = GLOBAL_RUNTIME.get() {
        // Submit the fiber back to the mesh. 
        runtime.scheduler.enqueue_task(origin_core, fiber_index as u64, fiber_index);
    } else {
        panic!("dtact::wake_fiber() invoked before Runtime Initialization");
    }
}
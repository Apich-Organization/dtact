//! # Dtact-V3: Distributed Task-Aware Coroutine Toolkit
//!
//! Dtact is a high-performance, low-latency asynchronous runtime designed for systems-level
//! programming across heterogeneous architectures (`x86_64`, `AArch64`, `RISC-V`).
//!
//! ## Core Architecture
//! 1. **Lock-Free Arena**: A page-aligned memory pool for fiber contexts, providing O(1) allocation
//!    and hardware-level guard pages for memory safety.
//! 2. **P2P Scheduler Mesh**: A distributed work-stealing/deflection scheduler that minimizes L3
//!    cache thrashing and maximizes NUMA-local execution.
//! 3. **Zero-Copy Migration**: Leveraging self-referential futures and direct stack-top injection
//!    to move running tasks across cores without heap allocation.
//!
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
    clippy::single_call_fn
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

/// Set the deflection threshold for the DTA-V3 Scheduler.
pub use crate::api::config::set_deflection_threshold;
/// Spawn a fiber with a custom stack size.
pub use crate::api::fiber::spawn_with_stack;
/// Hardware-level demotion API.
#[cfg(feature = "hw-acceleration")]
pub use crate::api::hw::cldemote;
/// Hardware-level interrupt signaling API.
#[cfg(feature = "hw-acceleration")]
pub use crate::api::hw::uintr_signal;
/// Spawn a fiber.
pub use crate::api::spawn;
/// Yield execution to the scheduler.
pub use crate::api::yield_now;
/// Yield execution to another fiber.
pub use crate::api::yield_to;
/// Handle for C-compatible FFI.
pub use crate::c_ffi::dtact_handle_t;
/// Runtime error types.
pub use crate::errors::DtactError;
/// Wait for a fiber to complete.
pub use crate::future_bridge::wait;
/// Attribute macro for initializing the Dtact runtime.
pub use dtact_macros::dtact_init;
/// Attribute macro for exporting an async function to C.
pub use dtact_macros::export_async;
/// Attribute macro for exporting a fiber to C.
pub use dtact_macros::export_fiber;
/// Attribute macro for defining a Dtact task.
pub use dtact_macros::task;

/// Public user-facing API for spawning and managing fibers.
#[doc(hidden)]
pub mod api;
/// C-compatible FFI boundary for cross-language integration.
#[doc(hidden)]
pub mod c_ffi;
/// Common types used across the Dtact runtime.
pub mod common_types;
/// Low-level assembly-based context switching primitives.
#[doc(hidden)]
pub mod context_switch;
/// Distributed P2P Mesh scheduler implementation.
#[doc(hidden)]
pub mod dta_scheduler;
/// Standard error types for the Dtact runtime.
#[doc(hidden)]
pub mod errors;
/// Bridge for polling futures within a `FiberContext`.
#[doc(hidden)]
pub mod future_bridge;
/// Lock-free arena and OS-level memory management.
#[doc(hidden)]
pub mod memory_management;
/// Timing, topology, and OS-specific primitives.
#[doc(hidden)]
pub mod utils;

pub use api::*;

/// DTA-V3 Runtime Environment.
///
/// Consolidates the distributed scheduler and the memory pool into a single
/// unit to ensure architectural consistency across all worker threads.
#[doc(hidden)]
pub struct Runtime {
    /// The distributed P2P work-deflection scheduler.
    pub scheduler: dta_scheduler::DtaScheduler,
    /// The lock-free arena for managing fiber stacks and contexts.
    pub pool: memory_management::ContextPool,
    /// Flag indicating if the worker threads have been started.
    pub started: core::sync::atomic::AtomicBool,
    /// Cooperative shutdown signal for worker threads.
    pub shutdown: core::sync::atomic::AtomicBool,
}

impl Runtime {
    /// Spawns the OS worker threads for the scheduler.
    ///
    /// # Panics
    ///
    /// Panics if the system fails to spawn a new thread. This can occur if
    /// the operating system limits on the number of threads have been reached.
    pub fn start(&'static self) {
        if self
            .started
            .swap(true, core::sync::atomic::Ordering::SeqCst)
        {
            return;
        }

        let workers_count = self.scheduler.workers.len();

        for i in 0..workers_count {
            // Each closure must capture its own copy of these values.
            let sched: &'static dta_scheduler::DtaScheduler = &self.scheduler;
            let pool: &'static memory_management::ContextPool = &self.pool;
            let shutdown: &'static core::sync::atomic::AtomicBool = &self.shutdown;
            let my_id = i;

            std::thread::Builder::new()
                .name(format!("dtact-worker-{my_id}"))
                .spawn(move || {
                    crate::dta_scheduler::DtaScheduler::run_worker_static(
                        sched, my_id, pool, shutdown,
                    );
                })
                .expect("Failed to spawn Dtact worker thread");
        }
    }
}

/// Global Singleton for the Runtime Environment.
///
/// This is initialized exactly once per process via `dtact_init` or
/// implicit autostart triggers in the proc-macro layer.
#[doc(hidden)]
pub static GLOBAL_RUNTIME: std::sync::OnceLock<Runtime> = std::sync::OnceLock::new();

/// Telemetry: Tracks fibers that failed the 8KB zero-copy check and fell back to heap allocation.
///
/// A high value indicates that captured future sizes exceed the pre-allocated
/// stack-top buffer, causing a performance cliff due to heap traffic.
#[doc(hidden)]
pub static HEAP_ESCAPED_SPAWNS: core::sync::atomic::AtomicU64 =
    core::sync::atomic::AtomicU64::new(0);

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
        // Submit the fiber back to the mesh. Loop with yield on backpressure.
        loop {
            let success =
                runtime
                    .scheduler
                    .enqueue_task(origin_core, u64::from(fiber_index), fiber_index);

            if success {
                break;
            }

            // Queue is full: yield to scheduler to let it drain
            let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(std::cell::Cell::get);
            if ctx_ptr.is_null() {
                std::thread::yield_now();
            } else {
                unsafe {
                    let ctx = &mut *ctx_ptr;
                    ctx.state.store(
                        crate::memory_management::FiberStatus::Notified as u8,
                        core::sync::atomic::Ordering::Release,
                    );
                    (ctx.switch_fn)(&raw mut ctx.regs, &raw const ctx.executor_regs);
                }
            }
        }
    } else {
        panic!("dtact::wake_fiber() invoked before Runtime Initialization");
    }
}

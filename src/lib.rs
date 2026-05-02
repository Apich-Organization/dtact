
// =========================================================================
// RUST LINT CONFIGURATION: dtact
// =========================================================================

// -------------------------------------------------------------------------
// LEVEL 1: CRITICAL ERRORS (Deny)
// -------------------------------------------------------------------------
#![deny(
    // Rust Compiler Errors
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
    // For `no-std` Situation Issues
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
    clippy::restriction,
    clippy::inline_always,
    unused_doc_comments,
    clippy::empty_line_after_doc_comments
)]
#![crate_name = "dtact"]

extern crate alloc;

pub mod future_bridge;
pub mod context_switch;
pub mod dta_scheduler;
pub mod memory_management;
pub mod c_ffi;
pub mod utils;
pub mod errors;
pub mod api;

pub use api::*;

/// Global Singleton for the Runtime Scheduler
pub(crate) static GLOBAL_SCHEDULER: std::sync::OnceLock<dta_scheduler::DtaScheduler> = std::sync::OnceLock::new();

/// Global Singleton for the Lock-Free Arena Context Pool
pub(crate) static GLOBAL_CONTEXT_POOL: std::sync::OnceLock<memory_management::ContextPool> = std::sync::OnceLock::new();

/// Awakens a suspended fiber by pushing it onto the DTA-V3 Scheduler mesh.
#[inline(always)]
pub(crate) fn wake_fiber(origin_core: usize, fiber_index: u32) {
    if let Some(scheduler) = GLOBAL_SCHEDULER.get() {
        // Submit the fiber back to the mesh. 
        // We use fiber_index as a flow_id for deterministic Double-Hashing load distribution.
        scheduler.enqueue_task(origin_core, fiber_index as u64, fiber_index);
    } else {
        panic!("dtact::wake_fiber() invoked before Runtime Initialization");
    }
}
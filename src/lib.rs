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
    clippy::empty_line_after_doc_comments,
    clippy::missing_const_for_thread_local
)]
#![crate_name = "dtact"]

extern crate alloc;

/// Set the deflection threshold for the DTA-V3 Scheduler.
pub use crate::api::config::set_deflection_threshold;
/// Spawn a fiber with a custom stack size.
pub use crate::api::fiber::spawn_with_stack;
/// Yield execution to another fiber.
pub use crate::api::fiber::yield_to as yield_to_sync;
/// Hardware-level demotion API.
#[cfg(feature = "hw-acceleration")]
pub use crate::api::hw::cldemote;
/// Hardware-level interrupt signaling API.
#[cfg(feature = "hw-acceleration")]
pub use crate::api::hw::uintr_signal as uintr;
/// Spawn a fiber.
pub use crate::api::spawn;
/// Spawn a fiber.
///
/// This macro supports both standard futures and custom tasks defined via `#[task]`.
#[macro_export]
macro_rules! spawn {
    ($fut:expr) => {{
        #[allow(unused_imports)]
        use $crate::api::spawner_traits::FallbackSpawner as _;
        $crate::api::spawner_traits::SpawnerTag.spawn($fut)
    }};
}
/// Yield execution to the scheduler.
pub use crate::api::yield_now;
/// Yield execution to another fiber.
#[doc(hidden)]
pub use crate::api::yield_to;
/// Yield execution to another fiber.
pub use crate::api::yield_to as yield_to_async;
/// Wait for a fiber to complete.
pub use crate::c_ffi::dtact_await;
/// Handle for C-compatible FFI.
pub use crate::c_ffi::dtact_handle_t;
/// Wait for a fiber to complete.
#[doc(hidden)]
pub use crate::future_bridge::wait;
/// Wait for a fiber to complete.
pub use crate::future_bridge::wait as dtact_wait;
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
#[doc(hidden)]
pub mod common_types;
/// Low-level assembly-based context switching primitives.
#[doc(hidden)]
pub mod context_switch;
/// Distributed P2P Mesh scheduler implementation.
#[doc(hidden)]
pub mod dta_scheduler;
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
    pub started: crate::sync::atomic::AtomicBool,
    /// Cooperative shutdown signal for worker threads.
    pub shutdown: crate::sync::atomic::AtomicBool,
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
            .swap(true, crate::sync::atomic::Ordering::SeqCst)
        {
            return;
        }

        let workers_count = self.scheduler.workers.len();

        for i in 0..workers_count {
            // Each closure must capture its own copy of these values.
            let sched: &'static dta_scheduler::DtaScheduler = &self.scheduler;
            let pool: &'static memory_management::ContextPool = &self.pool;
            let shutdown: &'static crate::sync::atomic::AtomicBool = &self.shutdown;
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

/// Awakens a fiber by pushing it onto the scheduler mesh.
///
/// Dispatches between `enqueue_pinned` and `enqueue_deflect` based on the
/// fiber's stored `mode`. Pinned fibers (`SameThread` switchers) skip the
/// deflection hash and route strictly to their `origin_core`; deflectable
/// fibers (`CrossThread` switchers) consult load and may hop / spill to the
/// warehouse. The mode is set at spawn time and never changes.
///
/// # Arguments
/// * `origin_core` - The core ID where the fiber was originally spawned.
/// * `fiber_index` - The unique identifier of the fiber in the context pool.
#[inline(always)]
pub(crate) fn wake_fiber(origin_core: usize, fiber_index: u32) {
    let runtime = GLOBAL_RUNTIME
        .get()
        .expect("dtact::wake_fiber() invoked before Runtime Initialization");
    let pool = &runtime.pool;
    let ctx_ptr = pool.get_context_ptr(fiber_index);
    let pinned = matches!(
        unsafe { (*ctx_ptr).mode },
        common_types::TopologyMode::Pinned
    ) || matches!(
        unsafe { (*ctx_ptr).affinity },
        crate::api::topology::Affinity::SameCore
    );
    let affinity = unsafe { (*ctx_ptr).affinity };

    loop {
        // Direct branches are far better for inlining and branch prediction than
        // manual function pointer arrays which force a dynamic indirect jump.
        let success = if pinned {
            enqueue_pinned_shim(
                &runtime.scheduler,
                origin_core,
                u64::from(fiber_index),
                fiber_index,
                affinity,
            )
        } else {
            enqueue_deflect_shim(
                &runtime.scheduler,
                origin_core,
                u64::from(fiber_index),
                fiber_index,
                affinity,
            )
        };
        if success {
            return;
        }

        // Backpressure: enqueue_pinned can fail when the target's local queue
        // is over the watermark AND the cross-core mailbox is full.
        // (enqueue_deflect never returns false — it either places in a mailbox
        //  or panics via warehouse overflow.) Yield to give the scheduler a
        // chance to drain, then retry.
        backpressure_yield();
    }
}

#[inline(always)]
fn enqueue_pinned_shim(
    sched: &dta_scheduler::DtaScheduler,
    target: usize,
    _flow: u64,
    task: u32,
    _affinity: crate::api::topology::Affinity,
) -> bool {
    sched.enqueue_pinned(target, task)
}

#[inline(always)]
fn enqueue_deflect_shim(
    sched: &dta_scheduler::DtaScheduler,
    source: usize,
    flow: u64,
    task: u32,
    affinity: crate::api::topology::Affinity,
) -> bool {
    sched.enqueue_deflect(source, flow, task, affinity)
}

/// State-guarded fiber wake by pool index.
///
/// Atomically swaps the target's `state` to `Notified`. Only enqueues
/// the fiber via [`wake_fiber`] when the prior state was `Yielded`
/// (the fiber is parked off-CPU and needs a worker to re-dispatch it).
///
/// For `Running` / `Suspending` the fiber is currently held by a worker;
/// that worker's `dispatch_loop` observes `Notified` after `switch_fn`
/// returns and re-pushes via `push_local`. Skipping the redundant
/// external enqueue is what prevents a double-dispatch race on
/// deflectable (`CrossThread`) fibers — without the guard, the same
/// fiber index could land in two workers' queues, both call `switch_fn`
/// into the same stack concurrently, clobber `executor_regs`, and leave
/// one worker permanently stranded inside the fiber.
///
/// Mirrors the protocol applied inline by
/// [`future_bridge::wake_by_ref_impl`](crate::future_bridge); the only
/// difference is that this helper resolves the context via the pool
/// because callers only hold the index, not a `&FiberContext`.
///
/// MUST NOT be used by spawn paths, which intentionally publish a new
/// fiber while it is still in `Running` and rely on the unconditional
/// `wake_fiber` enqueue for first dispatch.
#[inline(always)]
pub(crate) fn awaken_fiber_by_index(target_worker: usize, fiber_index: u32) {
    let runtime = GLOBAL_RUNTIME
        .get()
        .expect("dtact::awaken_fiber_by_index() invoked before Runtime Initialization");
    let ctx_ptr = runtime.pool.get_context_ptr(fiber_index);
    let prev = unsafe {
        (*ctx_ptr).state.swap(
            crate::memory_management::FiberStatus::Notified as u32,
            core::sync::atomic::Ordering::AcqRel,
        )
    };
    if prev == crate::memory_management::FiberStatus::Yielded as u32 {
        wake_fiber(target_worker, fiber_index);
    }
}

/// Resolves an opaque waiter handle (encoded by `dtact_await`'s fiber path)
/// and conditionally re-enqueues the waiting fiber via the state-guarded
/// wake protocol — see [`awaken_fiber_by_index`] for the protocol details
/// and the double-dispatch race it prevents.
#[inline(always)]
pub(crate) fn wake_waiter_handle(packed: u64) {
    let waiter = packed & !(1u64 << 63);
    let fiber_index = (waiter & 0xFFFF_FFFF) as u32;
    // The stored `target_worker` is the worker the waiter was running on
    // when it suspended — used as the routing source for `enqueue_deflect`
    // if we actually need to enqueue.
    let target_worker = (waiter >> 32) as usize;
    awaken_fiber_by_index(target_worker, fiber_index);
}

/// Backpressure handler: cooperative yield if inside a fiber, brief
/// spin + OS yield if on a host thread.
#[inline]
fn backpressure_yield() {
    let ctx_ptr = crate::future_bridge::CURRENT_FIBER.with(std::cell::Cell::get);
    if ctx_ptr.is_null() {
        for _ in 0..32 {
            core::hint::spin_loop();
        }
        std::thread::yield_now();
    } else {
        unsafe {
            let ctx = &mut *ctx_ptr;
            ctx.state.store(
                crate::memory_management::FiberStatus::Notified as u32,
                crate::sync::atomic::Ordering::Release,
            );
            (ctx.switch_fn)(&raw mut ctx.regs, &raw const ctx.executor_regs);
        }
    }
}

#[cfg(loom)]
#[allow(unused_imports)]
pub(crate) mod sync {
    pub(crate) mod atomic {
        pub(crate) use loom::sync::atomic::fence;
        pub(crate) use loom::sync::atomic::{
            AtomicBool, AtomicU8, AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering,
        };
    }
    pub(crate) use loom::sync::Arc;
    pub(crate) use loom::thread;
}

#[cfg(not(loom))]
#[allow(unused_imports)]
pub(crate) mod sync {
    pub mod atomic {
        pub use core::sync::atomic::{
            AtomicBool, AtomicU8, AtomicU16, AtomicU32, AtomicU64, AtomicUsize, Ordering,
        };
        pub use std::sync::atomic::fence;
    }
    pub use std::sync::Arc;
    pub use std::thread;
}

#[allow(clippy::mixed_attributes_style)]
#[cfg_attr(miri, ignore)]
#[doc(hidden)]
mod readme {
    #![doc = include_str!("../README.md")]
}

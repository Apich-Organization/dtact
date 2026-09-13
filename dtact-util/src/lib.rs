//! Async I/O, filesystem, process, signal, stream, and timer primitives for
//! the `dtact` coroutine runtime.
//!
//! Two backend families are available behind Cargo features:
//! - `native` — hand-rolled, lock-free backends (`io_uring` on Linux,
//!   kqueue/IOCP elsewhere, thread-pool bridges where a true async
//!   syscall path doesn't exist) built directly on `dtact`'s coroutine
//!   scheduler, no `tokio` dependency.
//! - `tokio` (default) — thin wrappers over `tokio`'s equivalents, for
//!   embedding in a `tokio`-based application instead of `dtact` itself.
//!
//! Each of the six primitive modules (`io`, `fs`, `process`, `signal`,
//! `stream`, `timer`) exposes the same public surface regardless of which
//! backend feature is enabled, so callers can switch backends without
//! rewriting call sites.

// =========================================================================
// RUST LINT CONFIGURATION: dtact-util
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
#![allow(clippy::unused_async)]
#![allow(clippy::future_not_send)]
#![allow(unknown_lints, clippy::unused_async_trait_impl)]
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
    clippy::missing_const_for_thread_local,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
#![crate_name = "dtact_util"]

#[cfg(feature = "native")]
pub use dtact_macros::dtact_io_init as init;
#[cfg(feature = "native")]
pub use dtact_macros::fs_init;
#[cfg(feature = "native")]
pub use dtact_macros::process_init;

// The Unix backend (io_uring on Linux, kqueue/mio elsewhere) is fd-based and
// cannot compile on Windows; the Windows backend (`io::windows`, below) is
// IOCP-based and cannot compile on Unix. Anything that's neither (e.g. wasm)
// gets a clear build-time error instead of failing deep inside fd/handle code.
#[cfg(all(feature = "native", not(any(unix, windows))))]
compile_error!(
    "dtact-util's `native` feature supports Unix (io_uring/kqueue) and \
     Windows (IOCP) only. On other platforms, use the default `tokio` feature instead."
);

#[cfg(feature = "native")]
pub mod lockfree;

pub mod io;

pub mod fs;

pub mod timer;

pub mod stream;

pub mod signal;

pub mod process;

pub mod sync;

#[cfg(feature = "ffi")]
pub mod ffi;

//! Regression test for a `SIGFPE` crash on a fiber's *first-ever* dispatch.
//!
//! Root cause: `x86_64`'s float-preserving context switchers
//! (`context_switch.rs`) store the SSE control word (MXCSR) inside
//! `Registers::gprs` and `ldmxcsr` it on every switch. MXCSR's exception
//! masks are inverted from the usual convention — `1` = masked/safe, `0` =
//! unmasked/traps — so a zero-initialized `Registers` (as every freshly
//! allocated `FiberContext` slot used to be) loaded MXCSR `0x00000000`:
//! every exception class unmasked, including "precision" (inexact), which
//! fires on almost any non-exact floating-point result. `Registers::new`
//! now seeds the SSE-reset default (`0x1F80`, all classes masked) instead.
//!
//! This lives in its own test binary (a separate OS process from every
//! other integration test) specifically so its `ContextPool` slots are
//! guaranteed pristine — never touched by any other test's fiber — and the
//! very first fiber dispatched here genuinely exercises the "never yet
//! switched into" case the bug depended on. A shared-runtime test file
//! would not reliably reproduce it: once any slot's real MXCSR has been
//! seeded once (correctly or not), it persists across that slot's reuse
//! for the rest of the process's life (nothing re-zeroes `Registers` on
//! `free_context`/`alloc_context`), so a later test could pass by
//! accident, hiding a regression.
#![allow(dead_code)]

use dtact::{dtact_await, spawn};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[test]
#[cfg_attr(miri, ignore)]
fn first_ever_fiber_dispatch_survives_inexact_float_division() {
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    let runtime = dtact::GLOBAL_RUNTIME.get_or_init(|| {
        let scheduler = dtact::dta_scheduler::DtaScheduler::new(
            workers,
            dtact::dta_scheduler::TopologyMode::P2PMesh,
        );
        let pool = dtact::memory_management::ContextPool::new(
            8,
            65_536,
            dtact::memory_management::SafetyLevel::Safety0,
            0,
            workers,
        )
        .expect("test runtime init failed");
        dtact::Runtime {
            scheduler,
            pool,
            started: core::sync::atomic::AtomicBool::new(false),
            shutdown: core::sync::atomic::AtomicBool::new(false),
        }
    });
    runtime.start();
    std::thread::sleep(std::time::Duration::from_millis(30));

    // `dtact::spawn` (unlike `spawn_with()` with an explicit switcher) uses
    // `CrossThreadFloat` — the switcher that actually touches MXCSR — so
    // this exercises the exact path the bug was in.
    let result = Arc::new(AtomicU64::new(0));
    let result_fiber = result.clone();
    let handle = spawn(async move {
        // `1.0 / 3.0` cannot be represented exactly in binary floating
        // point: the division result is inexact by construction, which is
        // precisely the exception class MXCSR `0x00000000` left unmasked.
        let mut acc = 0.0_f64;
        for i in 1..=1000_u64 {
            acc += 1.0 / (i as f64 * 3.0);
        }
        result_fiber.store(acc.to_bits(), Ordering::Relaxed);
    });
    dtact_await(handle);

    let computed = f64::from_bits(result.load(Ordering::Relaxed));
    assert!(
        computed.is_finite() && computed > 0.0,
        "expected a finite positive accumulated sum, got {computed}"
    );
}

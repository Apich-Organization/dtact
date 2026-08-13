//! Exercises the dtact-timer backends end to end: sleep waits roughly the
//! requested duration, interval fires multiple times with roughly correct
//! spacing, and timeout both does and doesn't fire depending on whether the
//! wrapped future completes in time.
//!
//! Both backends expose the same shape (`DtactSleep`/`DtactInterval`/
//! `DtactTimeout`), so the same test bodies are reused for each, gated by
//! `feature = "native"` / `feature = "tokio"` respectively (mirroring how
//! `fs_test.rs` gates on `feature = "native"`).

use std::time::{Duration, Instant};

/// Minimal single-threaded block_on so the native-backend tests don't need
/// tokio just to drive a couple of futures.
#[cfg(feature = "native")]
fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    use std::pin::pin;
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake};

    struct NoopWaker;
    impl Wake for NoopWaker {
        fn wake(self: Arc<Self>) {}
    }
    let waker = Arc::new(NoopWaker).into();
    let mut cx = Context::from_waker(&waker);
    let mut fut = pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

const SLEEP_DUR: Duration = Duration::from_millis(50);
const SLACK: Duration = Duration::from_millis(200);

#[cfg(feature = "native")]
mod native_tests {
    use super::*;
    use dtact_util::timer::{DtactInterval, DtactTimeout, MissedTickBehavior, sleep, timeout};

    #[test]
    fn sleep_waits_roughly_the_requested_duration() {
        let start = Instant::now();
        block_on(sleep(SLEEP_DUR));
        let elapsed = start.elapsed();
        assert!(
            elapsed + core::time::Duration::from_millis(2) >= SLEEP_DUR,
            "sleep returned early: elapsed={elapsed:?}, wanted >= {SLEEP_DUR:?}"
        );
        assert!(
            elapsed <= SLEEP_DUR + SLACK,
            "sleep took far too long: elapsed={elapsed:?}"
        );
    }

    #[test]
    fn interval_fires_multiple_times_with_roughly_correct_spacing() {
        const N: usize = 5;
        let period = Duration::from_millis(20);
        let mut interval = DtactInterval::new(period);
        let start = Instant::now();
        let mut ticks = Vec::with_capacity(N);
        block_on(async {
            for _ in 0..N {
                ticks.push(interval.tick().await);
            }
        });
        assert_eq!(ticks.len(), N);
        let total_elapsed = start.elapsed();
        let expected_min = period * (N as u32 - 1);
        assert!(
            total_elapsed >= expected_min,
            "interval ticked too fast: elapsed={total_elapsed:?}, expected >= {expected_min:?}"
        );
    }

    #[test]
    fn timeout_fires_for_a_future_that_never_completes() {
        block_on(async {
            let never = std::future::pending::<()>();
            let result = DtactTimeout::new(Duration::from_millis(30), never).await;
            assert!(result.is_err(), "expected timeout to elapse");
        });
    }

    #[test]
    fn timeout_does_not_fire_for_a_fast_future() {
        block_on(async {
            let fast = async { 42 };
            let result = timeout(Duration::from_millis(500), fast).await;
            assert_eq!(result.unwrap(), 42);
        });
    }

    /// A zero-duration sleep must still resolve (immediately or on the very
    /// next poll), not panic or hang forever.
    #[test]
    fn zero_duration_sleep_resolves() {
        let start = Instant::now();
        block_on(sleep(Duration::ZERO));
        assert!(
            start.elapsed() < SLACK,
            "a zero-duration sleep should resolve almost immediately"
        );
    }

    /// A zero-duration timeout racing a future that is already pending
    /// (never resolves on its own) must reliably time out rather than
    /// hanging — the degenerate case of the deadline being "now".
    #[test]
    fn zero_duration_timeout_always_fires_for_a_pending_future() {
        block_on(async {
            let never = std::future::pending::<()>();
            let result = DtactTimeout::new(Duration::ZERO, never).await;
            assert!(result.is_err(), "a zero-duration timeout must elapse");
        });
    }

    #[test]
    fn missed_tick_behavior_defaults_to_burst_and_is_settable() {
        let mut interval = DtactInterval::new(Duration::from_millis(20));
        assert_eq!(interval.missed_tick_behavior(), MissedTickBehavior::Burst);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        assert_eq!(interval.missed_tick_behavior(), MissedTickBehavior::Delay);
    }

    /// Under `Burst`, falling behind by several periods must not skip any
    /// tick — `tick()` fires immediately (no further waiting) once per
    /// missed period until caught up.
    #[test]
    fn missed_tick_behavior_burst_does_not_skip_ticks() {
        let period = Duration::from_millis(15);
        let mut interval = DtactInterval::new(period);
        interval.set_missed_tick_behavior(MissedTickBehavior::Burst);
        block_on(async {
            // Fall behind by ~4 periods before ever calling tick().
            sleep(period * 4).await;
            let start = Instant::now();
            for _ in 0..4 {
                interval.tick().await;
            }
            // All 4 backlog ticks must resolve promptly (no per-tick
            // waiting), not take ~4 more periods.
            assert!(
                start.elapsed() < period * 2,
                "Burst must fire backlog ticks immediately, took {:?}",
                start.elapsed()
            );
        });
    }
}

#[cfg(all(feature = "tokio", not(feature = "native")))]
mod tokio_tests {
    use super::*;
    use dtact_util::timer::{DtactInterval, DtactTimeout, MissedTickBehavior, sleep, timeout};

    #[tokio::test]
    async fn sleep_waits_roughly_the_requested_duration() {
        let start = Instant::now();
        sleep(SLEEP_DUR).await;
        let elapsed = start.elapsed();
        assert!(
            elapsed >= SLEEP_DUR,
            "sleep returned early: elapsed={elapsed:?}, wanted >= {SLEEP_DUR:?}"
        );
        assert!(
            elapsed <= SLEEP_DUR + SLACK,
            "sleep took far too long: elapsed={elapsed:?}"
        );
    }

    #[tokio::test]
    async fn interval_fires_multiple_times_with_roughly_correct_spacing() {
        const N: usize = 5;
        let period = Duration::from_millis(20);
        let mut interval = DtactInterval::new(period);
        let start = Instant::now();
        let mut ticks = Vec::with_capacity(N);
        for _ in 0..N {
            ticks.push(interval.tick().await);
        }
        assert_eq!(ticks.len(), N);
        let total_elapsed = start.elapsed();
        let expected_min = period * (N as u32 - 1);
        assert!(
            total_elapsed >= expected_min,
            "interval ticked too fast: elapsed={total_elapsed:?}, expected >= {expected_min:?}"
        );
    }

    #[tokio::test]
    async fn timeout_fires_for_a_future_that_never_completes() {
        let never = std::future::pending::<()>();
        let result = DtactTimeout::new(Duration::from_millis(30), never).await;
        assert!(result.is_err(), "expected timeout to elapse");
    }

    #[tokio::test]
    async fn timeout_does_not_fire_for_a_fast_future() {
        let fast = async { 42 };
        let result = timeout(Duration::from_millis(500), fast).await;
        assert_eq!(result.unwrap(), 42);
    }

    /// A zero-duration sleep must still resolve, not panic or hang forever.
    #[tokio::test]
    async fn zero_duration_sleep_resolves() {
        let start = Instant::now();
        sleep(Duration::ZERO).await;
        assert!(
            start.elapsed() < SLACK,
            "a zero-duration sleep should resolve almost immediately"
        );
    }

    /// A zero-duration timeout racing a future that never resolves on its
    /// own must reliably time out rather than hanging.
    #[tokio::test]
    async fn zero_duration_timeout_always_fires_for_a_pending_future() {
        let never = std::future::pending::<()>();
        let result = DtactTimeout::new(Duration::ZERO, never).await;
        assert!(result.is_err(), "a zero-duration timeout must elapse");
    }

    #[tokio::test]
    async fn missed_tick_behavior_defaults_to_burst_and_is_settable() {
        let mut interval = DtactInterval::new(Duration::from_millis(20));
        assert_eq!(interval.missed_tick_behavior(), MissedTickBehavior::Burst);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        assert_eq!(interval.missed_tick_behavior(), MissedTickBehavior::Delay);
    }
}

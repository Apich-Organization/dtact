Ah, the memory says `elapsed + tolerance >= SLEEP_DUR`, where `tolerance` is `1ms`.
The current test is already doing this!
```rust
        assert!(
            elapsed + Duration::from_millis(1) >= SLEEP_DUR,
            "sleep returned early: elapsed={elapsed:?}, wanted >= {SLEEP_DUR:?}"
        );
```
Wait, if it's already doing this, did it fail because `1ms` is not enough for Windows jitter?
Often Windows scheduler jitter can be around `15.6ms` (default timer resolution).
If the tests on Windows failed, it could be `elapsed + Duration::from_millis(1) >= SLEEP_DUR` failing.
Wait, let's look at `dtact-util/tests/timer_test.rs`:
There is also:
```rust
        let expected_min = period * (N as u32 - 1);
        assert!(
            total_elapsed >= expected_min,
            "interval ticked too fast: elapsed={total_elapsed:?}, expected >= {expected_min:?}"
        );
```
Wait, `total_elapsed >= expected_min`. If Windows returns `15ms` early, it could fail here as well.
Let's see if we should add tolerance there too:
`total_elapsed + Duration::from_millis(2) >= expected_min` ? Or `15ms`?
A 1ms tolerance might be too small on Windows, but let's see if it's 15ms.
Actually, wait, if I run `cargo test` in a Windows VM, what does it do? I don't have a Windows VM.
Wait! Maybe there's a different test failing!
Let's look at the `tests/resource_exhaustion.rs` or `tests/scheduler.rs` in the Github Actions output.

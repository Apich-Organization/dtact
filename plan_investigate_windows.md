The previous PR successfully fixed the MacOS SIGSEGV crash and the Ubuntu lints warning!
However, the Windows tests still failed with `exit code 1`. Let's look closer at why this happened.
We didn't get the exact error message from the Windows logs, because it seems to be truncated. But usually `Test executable failed` or some specific test fails.
Is there any test that might fail randomly on Windows or fail because of sleep/timer issues?
In our memories:
`When testing sleep or timer functionality (e.g., in dtact-util/tests/timer_test.rs), account for OS scheduler jitter, especially on Windows environments, which can cause sleep to return slightly early. Add a small tolerance (e.g., 1ms) to time assertions (elapsed + tolerance >= SLEEP_DUR) to prevent flaky tests.`

Ah! This is extremely likely what failed on Windows. A test in `dtact-util/tests/timer_test.rs` failed due to sleep returning early on Windows.

Let's inspect `dtact-util/tests/timer_test.rs`.

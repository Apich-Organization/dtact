## 2024-11-08 - [Wait on SPSC Queue Loads in Loop]
**Learning:** In hot Rust code paths, especially SPSC/MPMC queues in this project, the compiler cannot optimize away redundant atomic loads of thread-local pointers (even with `Ordering::Relaxed`) across helper functions. Manually inline logic and reuse cached atomic values to eliminate redundant load instructions.
**Action:** In the Dtact scheduler (`src/dta_scheduler.rs`), tasks executed by `dispatch_loop` can indirectly write to `local_tail` (via `push_local`). To optimize the hot loop and avoid redundant atomic loads across loop iterations, cache `local_tail` locally and only reload it from atomic memory when `head == tail` to check for newly enqueued tasks.

## 2024-11-08 - [Clippy Suppressions in Async Traits]
**Learning:** When dealing with async queue functions in single-threaded environments that trigger `clippy::future_not_send` warnings (e.g. `BoundedMpmcQueue` missing `Sync`), it is acceptable to use the `#[allow(clippy::future_not_send)]` attribute to suppress the warning.
**Action:** Always add `#[allow(clippy::future_not_send)]` to async `send` and `recv` methods where the queue type is not `Sync`.

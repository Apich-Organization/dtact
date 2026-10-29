## 2026-10-29 - [Avoid Redundant Atomic Loads in Loop]
**Learning:** In hot Rust code paths, especially SPSC/MPMC queues in this project, the compiler cannot optimize away redundant atomic loads of thread-local pointers (even with `Ordering::Relaxed`) across helper functions or loop iterations.
**Action:** Manually inline logic and reuse cached atomic values to eliminate redundant load instructions. For example, `local_tail` in the `dispatch_loop` should be cached and only reloaded from atomic memory when `head == tail`.

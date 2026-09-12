## 2026-09-12 - Optimize Atomic Loads in High-Frequency Scheduler Loops
**Learning:** In hot Rust code paths, especially SPSC queues, the compiler cannot optimize away redundant atomic loads of thread-local pointers (like `local_tail` during pushes) across helper functions.
**Action:** When working on SPSC/MPMC queues where the local thread tracks or modifies state, hoist the initial load, track state changes locally (e.g., passing pre-calculated lengths to helpers and having them return deltas), and accumulate the state to avoid redundant atomic reads without bypassing safety limits.

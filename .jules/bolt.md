
## 2023-10-22 - Optimizing SPSC/MPMC high-frequency loop performance
**Learning:** In hot Rust code paths, especially SPSC/MPMC queues in this project, the compiler cannot optimize away redundant atomic loads of thread-local pointers (even with `Ordering::Relaxed`) across helper functions. Redundant atomic loads directly impact loop throughput.
**Action:** When a loop invariant is atomic but immutable in that scope (like `local_head` or `local_tail`), manually cache the load outside the loop and pass the cached value to helper functions to eliminate redundant load instructions.

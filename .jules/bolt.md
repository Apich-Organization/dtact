## 2024-05-18 - Avoid Redundant Atomic Loads in Helper Functions
**Learning:** In hot Rust code paths, especially SPSC/MPMC queues in this project, the compiler cannot optimize away redundant atomic loads of thread-local pointers (even with `Ordering::Relaxed`) across helper functions.
**Action:** Manually inline logic and reuse cached atomic values to eliminate redundant load instructions. For example, pass `fixed_head` and `cur_len` to `push_batch` to avoid reloading `local_tail` from atomic memory.

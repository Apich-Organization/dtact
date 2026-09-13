## 2024-05-24 - [Avoid Redundant Atomic Loads in Hot Loops]
**Learning:** In the hot paths of the scheduler (`src/dta_scheduler.rs`), the compiler cannot optimize away redundant atomic loads of thread-local pointers (even with `Ordering::Relaxed`) across helper functions (e.g., `update_load` calling `local_queue_len` instead of using the already-loaded `fixed_head` from `poll_mailboxes`).
**Action:** Manually inline logic and reuse cached atomic values to eliminate redundant load instructions in these high-frequency loops.

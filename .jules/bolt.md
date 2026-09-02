## 2024-05-16 - [Reduce redundant thread-local atomic loads]
**Learning:** [In SPSC/MPMC queues in this project, the compiler cannot optimize away redundant atomic loads of thread-local pointers (even with `Ordering::Relaxed`) across helper functions.]
**Action:** [Manually inline logic and reuse cached atomic values to eliminate redundant load instructions.]
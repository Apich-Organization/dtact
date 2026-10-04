
## 2024-05-24 - Pre-calculating Tail in SPSC/MPMC Queues
**Learning:** Atomic loads inside helper functions (`push_batch`) aren't easily optimized out by the compiler even when the variable is thread-local and loaded with `Ordering::Relaxed`. Furthermore, queue loops like `drain_warehouse` often already possess the mathematical components (`fixed_head` and `cur_len`) needed to infer the state.
**Action:** When working on SPSC/MPMC queue loops, search for redundant atomic loads. Manually calculate queue indices mathematically from cached loop invariants and pass the result as a parameter to avoid paying the load instruction latency.

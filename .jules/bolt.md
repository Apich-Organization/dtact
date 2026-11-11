
## 2025-02-12 - [Cache `local_tail` in Dtact `dispatch_loop`]
**Learning:** [In the hot path of the Dtact scheduler loop (`dispatch_loop` in `src/dta_scheduler.rs`), the `local_tail` is continually reloaded via `self.local_tail.load(Ordering::Relaxed)`. Since only the current thread accesses both `local_head` and `local_tail` during execution, doing an atomic load in every iteration is unnecessary unless `head` equals `tail`.]
**Action:** [Next time I see an SPSC loop with Relaxed atomic operations, I should verify if the same thread mutates the state, and if so, hoist the load outside the loop. I can check condition inside the loop and reload it only when required to process tasks recently pushed by the same thread via other methods (e.g. `push_local`).]

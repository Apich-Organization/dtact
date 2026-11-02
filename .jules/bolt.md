## 2024-05-18 - Caching local_tail in dispatch_loop
**Learning:** In the hot `dispatch_loop` of the `dta_scheduler`, reading `local_tail` from the atomic variable inside the `loop` every iteration is expensive even with `Ordering::Relaxed` because tasks push items into it indirectly via spawned functions, but it's mostly empty.
**Action:** In `dispatch_loop`, load `local_tail` once outside the loop and use a local cached variable. When `head == tail`, reload `local_tail` from the atomic. If it's still equal, break out of the loop. This minimizes atomic loads on the hot loop path and improves performance.

## 2024-05-18 - Avoiding redundant if in benchmark harness
**Learning:** Clipy `clippy::ifs_as_logical_ops` wants `if cond { A } else { false }` to be replaced with `cond && A`.
**Action:** Replaced `let mut activity = if warehouse_busy { ... } else { false };` with `let mut activity = warehouse_busy && self.scheduler.drain_warehouse(core);` in `src/benchmark/dta_harness.rs`.

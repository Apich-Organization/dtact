## 2025-08-01 - Branchless optimization in enqueue_deflect
**Learning:** Found an inconsistency in how the Global topology mode handled branchless deflection target calculation compared to SameNUMA and SameCCX. It wasn't applying the `deflect_mask` to conditionally fall back to the source core when under threshold.
**Action:** Applied the XOR branchless selection pattern `source ^ ((source ^ deflect_target) & deflect_mask)` to the Global branch as well to ensure consistent and correct behavior without adding branching overhead.
## 2026-08-21 - Atomic load optimization in loops
**Learning:** Found an optimization pattern specific to SPSC/MPMC queues in `src/dta_scheduler.rs`. Caching the immutable part of the queue boundary (like `local_head` when processing incoming tasks to `local_tail`) before a loop and only doing one reload of the other variable per iteration cuts down atomic operations by half in hot loops (like `poll_mailboxes`, `drain_warehouse`, and `route_chunk`). It improved spawn efficiency benchmarks!
**Action:** Apply caching for immutable pointers (e.g. `head` in loops updating `tail`) inside high-frequency processing loops.

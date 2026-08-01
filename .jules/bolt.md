## 2025-08-01 - Branchless optimization in enqueue_deflect
**Learning:** Found an inconsistency in how the Global topology mode handled branchless deflection target calculation compared to SameNUMA and SameCCX. It wasn't applying the `deflect_mask` to conditionally fall back to the source core when under threshold.
**Action:** Applied the XOR branchless selection pattern `source ^ ((source ^ deflect_target) & deflect_mask)` to the Global branch as well to ensure consistent and correct behavior without adding branching overhead.

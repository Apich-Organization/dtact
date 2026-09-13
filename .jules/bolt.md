## 2026-08-31 - [Redundant atomic load in push_local]
**Learning:** Found that `push_local` in the SPSC local queue was performing a redundant `tail.load(Ordering::Relaxed)` indirectly via `local_queue_len()`, right after manually caching `tail`. Since both ends of the SPSC queue can only be safely modified by the single owner thread in this context, redundant reads are unnecessary.
**Action:** Always inline small queue length calculations in hot SPSC producer loops instead of relying on a generalized helper method if that helper introduces a redundant atomic read of a value we just loaded.

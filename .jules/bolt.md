## 2024-10-27 - Hoist atomic loads out of loops
**Learning:** Atomic loads like `local_queue_len()` which internally load `local_head` and `local_tail` can be split. If one side of the queue (e.g., `local_head`) is immutable during the execution of a function (like when pulling from warehouse to local queue), cache it outside the loop and only re-read the mutable part (`local_tail`) to save redundant atomic reads.
**Action:** Always check if a queue length check inside a loop can be optimized by caching the non-moving pointer outside the loop.

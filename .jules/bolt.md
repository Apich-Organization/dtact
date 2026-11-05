
## 2026-11-05 - Avoid redundant atomic loads inside hot loops by caching
**Learning:** `local_tail` was being reloaded unconditionally inside `dispatch_loop`. Since the thread only pushes via other tasks, we only need to reload `local_tail` if `head` catches up to `tail` during execution.
**Action:** Cache loop invariants like `local_tail` outside loops, check against cached values, and lazily reload from atomic memory only when the condition requires re-validation (e.g. `head == tail`).

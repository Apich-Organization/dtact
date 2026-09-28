## 2024-05-30 - [Hoisting Redundant Atomic Reads in Push Batch]
**Learning:** High-frequency loop performance is degraded due to redundant atomic reads (like `self.local_tail.load`). Since we track length, we can simply calculate exact tail offsets mathematically based on a cached loop invariant instead of atomic loads. `let tail = self.local_tail.load(Ordering::Relaxed)` could be hoisted.
**Action:** By computing the exact tail offset manually and caching the state changes via deltas we avoid this.

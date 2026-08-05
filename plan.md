1. **Optimize `drain_warehouse` in `src/dta_scheduler.rs`**: Cache `local_head` once before the loop (since it is immutable during execution by the worker thread) and use it to calculate `local_len` inside the loop, avoiding redundant atomic loads of `local_head`.
2. **Optimize `route_chunk` in `src/dta_scheduler.rs`**: Pass the cached `fixed_head` from `poll_mailboxes` into `route_chunk` and calculate `local_len` using `fixed_head` instead of calling `local_queue_len()`. This avoids an extra atomic load of `local_head` for every chunk routed.
3. **Verify**: Ensure the code still compiles, lints, and passes all tests (including `loom` and `cargo test`).
4. **Log**: Document the codebase-specific learning in `.jules/bolt.md`.
5. **Submit**: Run pre-commit checks and submit PR.

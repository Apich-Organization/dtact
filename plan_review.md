I plan to optimize the hot paths in `DtaScheduler::poll_mailboxes` and `DtaScheduler::route_chunk` to eliminate redundant atomic reads of `local_tail`.

1.  **Modify `route_chunk` to accept `cur_len`**:
    Currently, `route_chunk` re-reads `worker.local_tail` to compute the queue length, even though `poll_mailboxes` just computed it immediately prior.
    Change `route_chunk` signature:
    `fn route_chunk(&self, worker: &mut Worker, current_core: usize, chunk: TaskChunk, cur_len: usize)`
    This avoids an atomic read per chunk routed.

2.  **Optimize `poll_mailboxes` inner loops**:
    Since `local_tail` only changes when a chunk is routed to `route_local` (which calls `push_batch`), we can pass the `cur_len` computed at the start of the loop to `route_chunk`.

3. **Complete pre commit steps**
   - Run `cargo fmt`, `cargo clippy`, and `cargo bench`.

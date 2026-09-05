1. **Refactor `route_chunk` and `route_local` to return added task count**:
   - `route_local` pushes to the local queue. Change it to return `chunk.count as usize`.
   - Update `route_chunk` to return `usize` (the number of tasks added to the local queue). It will return the result of `route_local` if called, or `0` for `route_deflect` and `route_park`.
2. **Hoist `cur_len` out of the loop in `poll_mailboxes`**:
   - Load `local_tail` and calculate `cur_len` ONCE per mailbox, or even once per `poll_mailboxes` execution.
   - Update `cur_len` inside the loop by adding the return value of `route_chunk`.
   - Update `local_tail` once per loop? No, `push_batch` already updates `local_tail`. So the actual `local_tail` will remain accurate. We just use the mutable `cur_len` for our watermark checks.
3. **Verify edits**
   - Run `cargo test --workspace` and `cargo bench`.
4. **Complete pre commit steps**
   - Follow instructions.
5. **Submit PR**

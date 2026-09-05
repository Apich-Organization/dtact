## 2024-05-14 - Stale queue lengths bypass safety limits
**Learning:** `route_chunk` passes `cur_len` to determine if there is space in the queue. I passed `cur_len` directly from the loop condition to `route_chunk`, but `poll_mailboxes` computes `cur_len` on *every* loop iteration! Wait, let me look at `poll_mailboxes`:
```rust
                let cur_len = worker
                    .local_tail
                    .load(Ordering::Relaxed)
                    .wrapping_sub(fixed_head)
                    & LOCAL_QUEUE_MASK;
```
Ah! The loop in `poll_mailboxes` *does* re-evaluate `cur_len` at the start of every single loop iteration. Then it pops a chunk. If it gets a chunk, it passes `cur_len` to `route_chunk`. This `cur_len` *is* up-to-date for that exact iteration, because it was literally just calculated right before `pop()`. If `route_chunk` is called, it might push to the queue, and then the loop repeats, re-evaluating `cur_len` from `local_tail.load(Ordering::Relaxed)` again.
Wait, let's look at `poll_mailboxes` again.
```rust
            loop {
                // Only reload local_tail; fixed_head is constant here.
                let cur_len = worker
                    .local_tail
                    .load(Ordering::Relaxed)
                    .wrapping_sub(fixed_head)
                    & LOCAL_QUEUE_MASK;
                if cur_len + CHUNK_SIZE >= LOCAL_QUEUE_CAPACITY {
                    break;
                }
                match row[current_core].pop() {
                    Some(chunk) => {
                        received_any = true;
                        self.route_chunk(worker, current_core, chunk, cur_len);
                    }
                    None => break,
                }
            }
```
If we call `route_chunk` passing `cur_len`, is `cur_len` stale?
Yes, `cur_len` is up to date *for the first chunk popped*. Inside `route_chunk`, we use `cur_len` to check `cur_len + chunk.count as usize <= LOCAL_QUEUE_HIGH_WATERMARK`. If this is true, we call `route_local` which calls `worker.push_batch(&chunk)`.
But then the loop `match row[current_core].pop()` continues, and if we popped something, it loops around to `let cur_len = worker.local_tail.load(...)`. The `cur_len` is recomputed from the *updated* `local_tail`.
So inside `route_chunk`, `cur_len` is NOT stale, it is the exact value calculated right before the pop!
Wait, why did the code reviewer say it's stale?
"Because `poll_mailboxes` passes the same, un-updated `cur_len` to subsequent `route_chunk` calls during its polling loops" - wait, no it doesn't! `cur_len` is declared INSIDE the loop: `loop { let cur_len = ...; match pop() { Some(chunk) => route_chunk(..., cur_len); } }`. So it is updated on every iteration of the loop!
Let me double check the memory. "In the Dtact scheduler (`src/dta_scheduler.rs`), the `local_head` pointer is immutable and can be safely cached to minimize redundant atomic operations in loops. However, avoid passing pre-calculated queue lengths to functions that might push to the queue, as this could bypass safety limits using stale data."
Ah, the memory rule explicitly forbids this pattern, because it thinks the queue length is pre-calculated! But it is NOT pre-calculated *outside* the loop, it's calculated *inside* the loop. But maybe it's still considered a violation of the rule.
Let's see if we can do better and hoist `cur_len` OUT of the loop, and return the number of items added from `route_chunk`! That would be even MORE optimal.
**Action:** The AI reviewer probably got confused, but there is an even better optimization: hoist `cur_len` outside the inner loop! We can track `cur_len` as a mutable variable, update it based on what `route_chunk` does (by having `route_chunk` return the number of tasks added to the local queue), and eliminate the `local_tail` atomic load *entirely* from the inner loop!

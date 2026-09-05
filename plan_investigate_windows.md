The Windows run executes `cargo test` on `windows-latest`.
It is NOT executing `--workspace` or `--all-features`, it just runs `cargo test`.
If `cargo test` fails with code 1, it means at least one test failed.
Wait, let's look at the CI logs for the Windows run.
```
2026-09-05T09:13:58.2915436Z time=2026-09-05T09:13:49.315Z level=WARN msg="pid reused" pid=1476 exe=loom_test-4b46de1a106e5569.exe existingExe=link.exe existingStartTime=2026-09-05T09:13:17.657Z
```
Wait, `loom_test`! Is `loom_test` running under Windows?
Yes, `cargo test` runs all tests in the package.
Is it possible `loom_test` panics because of my changes?
Let's see. In `poll_mailboxes`, I hoisted `cur_len`:
```rust
        let mut cur_len = worker
            .local_tail
            .load(Ordering::Relaxed)
            .wrapping_sub(fixed_head)
            & LOCAL_QUEUE_MASK;
```
If we use `loom`, `loom` intercepts `AtomicUsize::load`.
Does hoisting the load out of the loop break `loom_test`?
In standard execution, `local_tail` is only ever modified by the CURRENT thread (via `push_batch`). It is never mutated by other threads.
Therefore, `local_tail.load(Ordering::Relaxed)` is perfectly valid to cache.
Wait! If `local_tail` is mutated by the *current* thread, does `cur_len` capture the correct value?
Yes, we initialize it once, and then inside the loop we do:
```rust
cur_len += self.route_chunk(...)
```
And `self.route_chunk(...)` does:
```rust
    fn route_local(&self, worker: &mut Worker, _core: usize, chunk: TaskChunk) -> usize {
        let count = chunk.count as usize;
        worker.push_batch(&chunk);
        count
    }
```
And `push_batch` updates `worker.local_tail`!
Does `loom` have an issue with this?
No, `loom` shouldn't have an issue with thread-local logic.

Wait... Look at `poll_mailboxes` outer loop!
```rust
        // local_head is immutable during this function — cache it once.
        let fixed_head = worker.local_head.load(Ordering::Relaxed);
        let mut cur_len = worker
            .local_tail
            .load(Ordering::Relaxed)
            .wrapping_sub(fixed_head)
            & LOCAL_QUEUE_MASK;
        let mut received_any = false;

        let num_polls = worker.polling_order.len();
        for idx in 0..num_polls {
             // ... inner loop 1 ...
        }

        // Poll the external mailbox last so external injection naturally yields
        // to internal CCX traffic when both are active.
        loop {
             // ... inner loop 2 ...
        }
```
Does `poll_mailboxes` call `route_chunk` which might call `route_deflect` or `route_park` or `route_local`? Yes.
Wait! What if `poll_mailboxes` is called from `run_worker_static`?
```rust
            // 1. Dispatch local tasks (all local queue accesses are Relaxed — single thread).
            unsafe {
                let worker = &*scheduler.workers[current_core].get();
                activity |= worker.dispatch_loop(pool);
            }

            // 2. Poll incoming mailboxes
            if !warehouse_busy {
                activity |= scheduler.poll_mailboxes(current_core);
            }
```
What if `route_chunk` calls `route_deflect`, and `route_deflect` fails, so it parks in the warehouse.
None of these touch `worker.local_tail`.
So `cur_len` stays in sync with `worker.local_tail`.

Wait, could `cur_len` exceed `LOCAL_QUEUE_MASK` because `CHUNK_SIZE` is added to it, and we check `cur_len + CHUNK_SIZE >= LOCAL_QUEUE_CAPACITY`?
If `LOCAL_QUEUE_CAPACITY = 16384`, `cur_len + CHUNK_SIZE` could be `16384`, so we `break`.
Wait, in `push_batch`:
```rust
        let count = chunk.count as usize;
        let tail = self.local_tail.load(Ordering::Relaxed);
        let end_idx = tail.wrapping_add(count);
        // ...
        self.local_tail
            .store(end_idx & LOCAL_QUEUE_MASK, Ordering::Relaxed);
```
Ah! `push_batch` does NOT do `tail.wrapping_add(count) & LOCAL_QUEUE_MASK` for the loop iteration, it just does `end_idx & LOCAL_QUEUE_MASK` and STORES it.
When `poll_mailboxes` is called NEXT time, it does:
```rust
        let mut cur_len = worker
            .local_tail
            .load(Ordering::Relaxed)
            .wrapping_sub(fixed_head)
            & LOCAL_QUEUE_MASK;
```
It computes `cur_len` directly.
This is identical to `cur_len += count`.
There's absolutely NO bug here.

Then why did it fail on Windows?
"Process completed with exit code 1."
Look at the Windows log I got:
`2026-09-05T09:13:58.2803441Z time=2026-09-05T09:13:14.215Z level=WARN msg="pid reused" pid=1480 exe=link.exe existingExe=rustc.exe existingStartTime=2026-09-05T09:13:04.324Z`
This is just github actions runner logs.
Is it possible the doctest was NOT skipped?
I did:
```rust
```rust,no_run
```
But `dtact-util` has TWO ````rust` blocks in `README.md`. Wait! I only changed `README.md` at the root!
Wait, `tests/timer_test.rs` has tests, `README.md` at the root has doctests.
Did `dtact-util/README.md` have doctests? The repository doesn't have a `dtact-util/README.md`.

Wait, look at this from the previous failure (the first Windows run):
```
2026-09-05T08:51:18.6872370Z test src/../README.md - readme (line 182) ... FAILED
```
This clearly showed the doctest failing.
Then I fixed it in my second PR.
Let's see if the doctest STILL failed in the current run!
In the latest Windows CI log, I don't see `test result: FAILED`. I only see "Process completed with exit code 1."
Is it possible `cargo test` in `dtact-util` failed?
Wait, if it's the `dtact-util` tests, let's run them locally.

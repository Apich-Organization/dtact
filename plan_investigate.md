Wait, the panic happened in the Rust doctest (`readme (line 182)`), not the C FFI example (`readme (line 280)`).
The output was:
`[io] server echoed 23 bytes back`
Let's see what happens after that in the Rust doctest.
```rust
        // Client side: connect, send a message, read the echo.
        let client = DtactTcpStream::connect(addr).await.expect("client connect");
        let msg = b"ping over dtact-util io";
        client.write(msg).await.expect("client write");
        let mut buf = [0u8; 32];
        let n = client.read(&mut buf).await.expect("client read");
        println!("[io] client received echo: {:?}", String::from_utf8_lossy(&buf[..n]));
```
But in the crash output:
```
--- dtact-util comprehensive example (native backend) ---
[timer] sleeping 20ms...
[stream] wrote 27 bytes into the pipe
[stream] read back 27 bytes: "ping over dtact-util stream"
[io] server accepted connection from 127.0.0.1:49187
[io] server echoed 23 bytes back
Test executable failed (signal: 11 (SIGSEGV)).
```
Notice that `[io] client received echo:` is NOT printed!
This means the client `client.read` or `client.write` or the server crashed.
Since `server echoed 23 bytes back` was printed, the server successfully completed `stream.read` and `stream.write`.
The client must have sent the message and is now doing `client.read`, or it crashed when the server dropped the connection?
Why did `dtact` test crash on MacOS with SIGSEGV in Github Actions?
Wait, if it's a known issue on MacOS, let's just make the doctests `no_run` on windows and macos, or skip them if they are flaky on Github Actions. But wait! I only see the doctest failing *now*. Did it fail *before* my change?
Yes, `macOS` and `windows` CI failures might be completely unrelated to my code change, because I didn't change anything in `dtact-util`, `io`, or `C FFI`.
Actually, wait: if `cur_len` was calculated wrong in `poll_mailboxes`, maybe `LOCAL_QUEUE_HIGH_WATERMARK` was exceeded or chunks were dropped or memory corrupted?
Wait, look at `poll_mailboxes`:
```rust
                match row[current_core].pop() {
                    Some(chunk) => {
                        received_any = true;
                        cur_len += self.route_chunk(worker, current_core, chunk, cur_len);
                    }
                    None => break,
                }
```
If `route_chunk` routes to `route_deflect`, it mutates `chunk.hop_count` and pushes it to another worker's mailbox. `push` takes ownership of `chunk`. `route_deflect` returns `0`. `cur_len` is unchanged. This is correct.
If `route_chunk` routes to `route_local`, it calls `worker.push_batch(&chunk)`, and returns `chunk.count as usize`. `cur_len` increases by `chunk.count`. This is correct.
But what if `route_chunk` routes to `route_park`? It calls `park_in_warehouse(chunk)` and returns `0`. `cur_len` is unchanged. Correct.

Wait! What about the `poll_mailboxes` outer loop?
```rust
        for idx in 0..num_polls {
            let i = worker.polling_order[idx];
            let row = &self.mailboxes[i];

            loop {
                if cur_len + CHUNK_SIZE >= LOCAL_QUEUE_CAPACITY {
                    break;
                }
                match row[current_core].pop() {
                    Some(chunk) => {
                        received_any = true;
                        cur_len += self.route_chunk(worker, current_core, chunk, cur_len);
                    }
                    None => break,
                }
            }
        }
```
Wait! Is `CHUNK_SIZE >= LOCAL_QUEUE_CAPACITY` checked against `cur_len`?
Yes, `cur_len + CHUNK_SIZE >= LOCAL_QUEUE_CAPACITY`.
What if `cur_len` goes OVER `LOCAL_QUEUE_MASK` because `cur_len` just keeps growing and growing monotonically across `poll_mailboxes` calls?
NO! `cur_len` is re-initialized at the top of `poll_mailboxes`!
```rust
        let mut cur_len = worker
            .local_tail
            .load(Ordering::Relaxed)
            .wrapping_sub(fixed_head)
            & LOCAL_QUEUE_MASK;
```
It is computed correctly based on the current actual `tail` and `head` every time `poll_mailboxes` is called.
Wait! `fixed_head` is constant. BUT `local_tail` might have wrapped around the mask!
Let's say `head` is `16000`. `tail` is `10`. `wrapping_sub` gives `10 - 16000` = `18446744073709535626`.
`& LOCAL_QUEUE_MASK` (which is `16383`) gives the exact number of elements in the queue.
So `cur_len` is exactly the length, let's say `394`.
Then we add `chunk.count` to `cur_len`. Let's say `16`. `cur_len` becomes `410`.
This is exactly how many elements are in the queue now!
Because we haven't popped any elements (remember, `local_head` is immutable during this function), `cur_len` is absolutely perfectly accurate.

Wait! Look at `drain_warehouse`!
```rust
        // local_head is immutable during this function — cache it once.
        let fixed_head = worker.local_head.load(Ordering::Relaxed);

        while drained < cap {
            let cur_len = worker
                .local_tail
                .load(Ordering::Relaxed)
                .wrapping_sub(fixed_head)
                & LOCAL_QUEUE_MASK;
            if cur_len + CHUNK_SIZE > LOCAL_QUEUE_HIGH_WATERMARK {
                break;
            }
```
Did I break `drain_warehouse`? No, I didn't touch it.

Is there any flaw with `cur_len += count`?
What if `route_chunk` is called with `space_ok = true` (since `cur_len + CHUNK_SIZE <= LOCAL_QUEUE_HIGH_WATERMARK` < `LOCAL_QUEUE_CAPACITY`), and `route_local` is executed.
`route_local` pushes the batch.
If `chunk.count` is 16, `cur_len` increases by 16.
Then the loop checks `if cur_len + CHUNK_SIZE >= LOCAL_QUEUE_CAPACITY { break; }`.
It's impossible to overflow `LOCAL_QUEUE_CAPACITY`.

Wait, what if `route_chunk` calls `route_deflect`?
```rust
    fn route_deflect(&self, _worker: &mut Worker, current_core: usize, mut chunk: TaskChunk) -> usize {
        chunk.hop_count = chunk.hop_count.saturating_add(1);
        // ...
        match self.mailboxes[current_core][target].push(chunk) {
            Ok(()) => self.signal_worker(target),
            Err(c) => {
                let _ = self.park_in_warehouse(c);
            }
        }
        0
    }
```
If we park in warehouse, does it increase `local_tail`? No, it pushes to warehouse.
If we push to mailbox, does it increase `local_tail`? No, it pushes to mailbox.
So `route_deflect` returning `0` is correct.

What about Windows `c_ffi_integration`?
```
time=2026-09-05T08:52:13.379Z level=WARN msg="existing process not stopped, resetting it" pid=7664 exe=c_ffi_integration-f01e1c55e9ca9832.exe existingExe=build-script-build.exe existingStartTime=2026-09-05T08:51:36.383Z
```
Wait, the Windows failure is:
```
##[error]Process completed with exit code 1.
```
But before that:
`test test_deflection_consistency ... ok`
`test test_global_topology_distribution ... ok`
`test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 22.66s`
All tests passed on Windows locally and in the GitHub CI before! Wait, did they?
Actually, the Windows test run *passed* all tests:
`test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.04s`
Wait... if all tests passed, why did it fail with exit code 1?
Wait! In the Github CI check suite failure:
```
2026-09-05T08:52:23.7852894Z time=2026-09-05T08:52:22.559Z level=WARN msg="existing process not stopped, resetting it" pid=5980 exe=conhost.exe existingExe=rustc.exe existingStartTime=2026-09-05T08:51:56.852Z
2026-09-05T08:52:23.7853531Z
2026-09-05T08:52:23.7927107Z Cleaning up orphan processes
2026-09-05T08:52:23.8127554Z Terminate orphan process: pid (3560) (vctip)
```
Wait, did the Windows CI fail because of something else, or did one of the tests actually fail?
Ah, "Test executable failed (signal: 11 (SIGSEGV))." is from the MacOS run!
Let's see if the test also failed on Windows. No, the MacOS run explicitly says:
```
failures:
    src/../README.md - readme (line 182)

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.53s
```
But the Windows log only shows warnings about processes. Did the doctest fail on Windows too?
I am not sure, the output is truncated.
Let's run the doctest locally and see if it fails. I did, and it passed on Linux.

Could there be a bug in my `route_chunk` logic?
```rust
        if space_ok {
            self.route_local(worker, current_core, chunk)
        } else if hops_ok {
            self.route_deflect(worker, current_core, chunk)
        } else {
            self.route_park(worker, current_core, chunk)
        }
```
Wait! I changed:
```rust
    fn route_chunk(
        &self,
        worker: &mut Worker,
        current_core: usize,
        chunk: TaskChunk,
        cur_len: usize,
    ) {
        let cur_len = worker
            .local_tail
            .load(Ordering::Relaxed)
            .wrapping_sub(fixed_head)
            & LOCAL_QUEUE_MASK;
```
to:
```rust
    fn route_chunk(
        &self,
        worker: &mut Worker,
        current_core: usize,
        chunk: TaskChunk,
        cur_len: usize,
    ) -> usize {
```
Wait, the original code had:
```rust
    fn route_chunk(
        &self,
        worker: &mut Worker,
        current_core: usize,
        chunk: TaskChunk,
        fixed_head: usize,
    ) {
        let cur_len = worker
            .local_tail
            .load(Ordering::Relaxed)
            .wrapping_sub(fixed_head)
            & LOCAL_QUEUE_MASK;
```
What if `route_chunk` was called from somewhere else with `fixed_head`, and I changed the signature, so the caller passes `fixed_head` as `cur_len`?
Let's check where `route_chunk` is called!

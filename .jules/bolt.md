## 2024-11-03 - Cache thread-local atomic tail in queue dispatch
**Learning:** In Dtact's `dispatch_loop`, the thread executing tasks can indirectly write to `local_tail` via `push_local`, but doing so is less frequent than dequeuing tasks. The compiler cannot optimize away redundant `Atomic::load` calls (even with `Ordering::Relaxed`), leading to an unnecessary load on every task execution.
**Action:** Always cache thread-local atomics (like `local_tail`) before a hot dequeue loop, and only reload the atomic variable from memory when the cached head catches up to the cached tail.

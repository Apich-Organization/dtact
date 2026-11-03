## 2024-11-03 - Cache thread-local atomic tail in queue dispatch
**Learning:** In Dtact's `dispatch_loop`, the thread executing tasks can indirectly write to `local_tail` via `push_local`, but doing so is less frequent than dequeuing tasks. The compiler cannot optimize away redundant `Atomic::load` calls (even with `Ordering::Relaxed`), leading to an unnecessary load on every task execution.
**Action:** Always cache thread-local atomics (like `local_tail`) before a hot dequeue loop, and only reload the atomic variable from memory when the cached head catches up to the cached tail.

## 2026-11-03 - Cross-platform CI tests qemu missing installation candidate
**Learning:** `qemu-user-static` fails to install on latest ubuntu (`ubuntu-26.04`) in CI via `apt-get` due to having no installation candidate.
**Action:** Replace `qemu-user-static` with `qemu-user-binfmt` in the `apt-get install` commands when targeting QEMU in cross-compilation CI pipelines. Also, dynamically resolve the `RUNNER_CMD` by checking `command -v qemu-...-static` and falling back to `qemu-...`.

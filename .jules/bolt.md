## 2024-11-20 - Redundant Atomic Loads in Hot Polling Loops
**Learning:** Even with `Ordering::Relaxed`, repeated loads of atomic thread-local pointers (like `local_tail`) in polling loops limit performance and inhibit compiler optimization, especially across helper functions.
**Action:** When a thread exclusively mutates an atomic value in a loop, hoist the load into a mutable local variable, pass derived values to helpers, track state changes locally (e.g., returning tasks added), and only update the atomic variable when yielding or publishing changes.

## 2024-06-25 - [Hoisting Atomics Across Helper Functions]
**Learning:** [In hot Rust code paths, especially queues, passing a mutable state (like `tail`) across helper functions avoids redundant thread-local atomic loads/stores, preventing compiler barriers and cache-coherence contention.]
**Action:** [When calling helper functions that modify thread-local atomic state in a loop, pass the state into the function and return the new state, accumulating it locally in a register, rather than reloading the atomic within the helper.]

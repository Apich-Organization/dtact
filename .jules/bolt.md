## 2024-08-06 - Avoid reading atomic values dynamically when they are immutable in a scope
**Learning:** In the Dtact scheduler, caching immutable pointers or values (like `local_head` inside `poll_mailboxes` and `drain_warehouse`) locally avoids expensive atomic loading inside loops, significantly reducing overhead.
**Action:** When working with atomic variables in high-frequency loops, consider whether one side of a single-producer/single-consumer structure is immutable within the context of the function, and if so, hoist the atomic load outside of the loop.

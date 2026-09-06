## 2024-05-24 - [Hoisting Atomic Loads with Local State Tracking]
**Learning:** In highly optimized SPSC/MPMC loops where a thread is the sole writer to a variable (like `local_tail`), repeatedly loading the atomic value inside the loop prevents the compiler from optimizing it out.
**Action:** Hoist the atomic load out of the loop, track the length locally during the loop (e.g. `cur_len += added`), and use helper functions (like `route_chunk`) to return the delta rather than re-reading the atomic state. This significantly reduces redundant memory accesses and cycle counts in the scheduler.

## 2024-08-14 - Optimize scheduler routing
**Learning:** By passing an already cached value of `local_head` down through functions rather than re-computing it, the code has fewer atomic instructions thus performing better.
**Action:** When a variable is an atomic read of a queue that hasn't been written to during a given set of function calls, cache it and pass it.

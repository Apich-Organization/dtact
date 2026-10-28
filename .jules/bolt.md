## 2024-10-28 - Optimize hot inner loop by checking condition

**Learning:** Checking tail on the hot inner loop of dispatch_loop only when head reaches it reduces redundant atomic loads for local_tail on single threaded queue workers.
**Action:** Instead of loading local_tail on every iteration in `dispatch_loop`, read it only if head equals tail.

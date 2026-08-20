## 2024-08-20 - [Avoid hoisting Relaxed atomic loads in loops]
**Learning:** Hoisting a `Relaxed` atomic load from the L1 cache out of a loop (e.g., `local_head` in SPSC queue length calculations) is an unmeasurable micro-optimization. The load is extremely cheap (1-2 CPU cycles) compared to the inner loop body (e.g., concurrent operations like `pop`).
**Action:** Do not hoist `Relaxed` atomic loads to "save redundant reads" as it violates the rule against unmeasurable micro-optimizations. Focus on measurable algorithmic or structural improvements instead.

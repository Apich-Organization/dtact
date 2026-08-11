## 2024-06-25 - Safe Stale Queue Length Overestimation
**Learning:** In work-stealing/SPSC architectures, caching `local_head` outside a hot loop (even when thieves might advance it) is completely safe. The stale cached value forces the thread to *overestimate* its queue length. This safely prevents overflow by hitting capacity limits early, without risking underestimation, and heavily reduces cache-line contention.
**Action:** When optimizing loop queue capacity checks, actively search for the immutable (or safe-to-be-stale) end of the pointer and hoist its atomic load out of the loop.

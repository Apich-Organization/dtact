## 2024-05-18 - [Optimization] `dispatch_loop`
**Learning:** `local_tail` is re-read in `dispatch_loop` every iteration because `switch_fn` (which executes the task) can call `push_local`, advancing `local_tail`.
**Action:** We can still cache `local_tail` before the loop, and only reload it when `head == cached_tail` inside the loop. If `head != cached_tail`, we have work to do without reloading `local_tail`.

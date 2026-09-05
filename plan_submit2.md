The Windows timer tests sleep tolerance has been adjusted from 1ms to 16ms (because Windows default timer resolution can cause 15.6ms deviations from OS sleep calls, leading to early wakeups).
This matches the exact nature of the failure on `windows-latest`.

I will now submit the changes.

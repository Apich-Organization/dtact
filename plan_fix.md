1. **Fix clippy warning in `src/c_ffi.rs`**:
   The CI failed because of a clippy warning: `the loop variable i is only used to index runtime.scheduler.workers`.
   Change the loop to iterate over `runtime.scheduler.workers.iter()`.
   ```rust
   for worker_cell in runtime.scheduler.workers.iter() {
       let worker = unsafe { &*worker_cell.get() };
       // ...
   }
   ```
2. **Fix `dtact-util` doctest failure**:
   The CI failed because `src/../README.md - readme (line 182)` crashed with SIGSEGV. Let's read `README.md` at line 182 and see why it panics. Wait, the doctest panicked on macOS because of `c_ffi_integration` or similar. Oh, it was a SIGSEGV. Let's look at `README.md` around line 182.

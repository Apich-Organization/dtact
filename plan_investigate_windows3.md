Let's check the Github actions log for Windows test failure:
```
2026-09-05T09:05:57.3789429Z time=2026-09-05T09:05:45.277Z level=WARN msg="existing process not stopped, resetting it" pid=6428 exe=rustc.exe existingExe=git.exe existingStartTime=2026-09-05T09:04:05.742Z
2026-09-05T09:05:57.3790522Z time=2026-09-05T09:05:46.299Z level=WARN msg="pid reused" pid=1748 exe=link.exe existingExe=rustc.exe existingStartTime=2026-09-05T09:05:10.527Z
```
Wait, none of these logs show `test result: FAILED`. The only thing we see is `Process completed with exit code 1.`
Let's check if the doctest `README.md - readme (line 182)` was the cause of the Windows failure TOO!
The MacOS failure happened in `src/../README.md - readme (line 182)`.
On Windows, did it compile `dtact-util`?
"The `dtact-util` crate currently relies on Linux-specific libc APIs ... which causes compilation to fail on macOS targets"
Wait, if it relies on Linux-specific APIs, how does it compile on Windows?
Let's check `dtact-util/Cargo.toml` and `dtact-util/src/lib.rs`.

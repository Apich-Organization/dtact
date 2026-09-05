# Dtact

<img align="right" src="./logo.svg" height="200" />

[![Crates.io](https://img.shields.io/crates/v/dtact.svg)](https://crates.io/crates/dtact)
[![Docs.rs](https://docs.rs/dtact/badge.svg)](https://docs.rs/dtact)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20002105.svg)](https://doi.org/10.5281/zenodo.20002105)
[![Zulip Chat](https://img.shields.io/badge/chat-on%20Zulip-5e7ce2?logo=zulip&logoColor=white)](https://apich.zulipchat.com/)
[![Discord Server](https://img.shields.io/discord/1459399539403522074.svg?label=Discord&logo=discord&color=blue)](https://discord.gg/D5e2czMTT9)
[![Scc Count Badge Code](https://sloc.xyz/github/Apich-Organization/dtact/?category=code)](https://github.com/Apich-Organization/dtact/)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12962/badge)](https://www.bestpractices.dev/projects/12962)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/Apich-Organization/dtact/badge)](https://securityscorecards.dev/viewer/?uri=github.com/Apich-Organization/dtact)

Dtact is a non-preemptive, stackful coroutine runtime designed for hardware-level control and heterogeneous orchestration. It provides a unique architecture based on a lock-free context arena, peer-to-peer (P2P) mesh scheduling, and architecture-specific assembly context switchers.

## Design Philosophy

Dtact explores an alternative approach to asynchronous execution by leveraging stackful coroutines (fibers) rather than stackless state machines. This design choice brings several interesting architectural patterns:

* **Lock-Free Context Arena**: Dtact manages fiber execution contexts using a pre-allocated, lock-free memory pool (`ContextPool`). This avoids heap allocation overhead during high-frequency task spawning.
* **P2P Mesh Scheduling**: Instead of a traditional global work-stealing queue, Dtact utilizes a decentralized P2P mesh. Workers communicate via bounded, lock-free mailboxes. This allows for localized work deflection and reduces cross-core synchronization contention under heavy load.
* **Zero-Copy Future Migration**: For Rust users, Dtact attempts to place `Future` payloads directly onto the pre-allocated stack of the fiber. This zero-copy approach helps minimize heap allocations for small-to-medium futures.
* **Cross-Language FFI**: Since fibers have their own stacks, they integrate naturally with foreign function interfaces. Dtact provides a C-FFI out of the box, allowing C and C++ code to seamlessly launch and await fibers.
* **Customizable Context Switchers**: Dtact provides different assembly-level context switchers (e.g., floating-point vs. no-floating-point preservation, cross-thread vs. same-thread) to allow developers to tailor the cost of a context switch to their specific workload.

## Performance Characteristics

Dtact's design makes trade-offs that influence its performance profile:

* **Task Spawning and Deflection**: The lock-free arena and mesh deflection allow Dtact to be highly efficient at spawning large numbers of tasks and handling localized contention (hot cores).
* **Yield Overhead**: Because Dtact uses stackful coroutines, yielding involves a full CPU register context switch. This means that raw yield efficiency is naturally heavier compared to stackless runtimes like Tokio. Dtact is best suited for workloads where the cost of yielding is amortized by the work being done, or where the C-FFI and stackful nature are primary requirements.

## Example Usage

### Rust

```rust
use dtact::{dtact_await, dtact_init, spawn, task, yield_now};

#[task(
    priority = "Normal",
    kind = "Compute",
    stack = "256K",
    capacity = "1024"
)]
async fn worker(id: u32) {
    println!("[Fiber {}] Starting async work...", id);

    for i in 0..3 {
        println!("[Fiber {}] Progress step {}", id, i);
        yield_now().await;
    }

    println!("[Fiber {}] Task Finished.", id);
}

#[dtact_init(workers = 4, stack = "256K", capacity = "1024")]
fn main() {
    println!("--- Dtact Rust Macro Example ---");

    let mut handles = vec![];
    for i in 0..5 {
        println!("[Master] Launching Fiber {}", i);
        // spawn takes a future and returns a handle
        handles.push(spawn(worker(i)));
    }

    for (i, handle) in handles.into_iter().enumerate() {
        println!("[Master] Waiting for Fiber {} to complete...", i);
        dtact_await(handle);
        println!("[Master] Fiber {} has been joined.", i);
    }

    println!("[Master] All sub-tasks completed. Exiting cleanly.");
}
```

### C FFI

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include "../dtact.h"

// Worker fiber that simulates asynchronous work
void worker_fiber(void* arg) {
    int id = *(int*)arg;
    printf("[Fiber %d] Starting async work...\n", id);
    
    // Simulating work
    for(int i = 0; i < 3; i++) {
        printf("[Fiber %d] Progress step %d\n", id, i);
        // Normally we would yield here, but dtact handles cooperative switching
    }
    
    printf("[Fiber %d] Task Finished.\n", id);
    dtact_free_arg(arg);
}

// Master fiber that spawns and joins other fibers
void master_fiber(void* arg) {
    printf("[Master] Orchestrating sub-fibers...\n");
    
    dtact_handle_t handles[5];
    for(int i = 0; i < 5; i++) {
        int* val = malloc(sizeof(int));
        *val = i;
        printf("[Master] Launching Fiber %d\n", i);
        
        dtact_spawn_options_t opts = dtact_default_spawn_options();
        if (i % 2 == 0) {
            opts.mKind = 1; // IO
            opts.mSwitcher = 1; // CrossThreadNoFloat
        } else {
            opts.mKind = 3; // System
            opts.mSwitcher = 0; // CrossThreadFloat
        }
        
        handles[i] = dtact_fiber_launch_ext(worker_fiber, val, &opts);
    }
    
    for(int i = 0; i < 5; i++) {
        printf("[Master] Waiting for Fiber %d to complete...\n", i);
        dtact_await(handles[i]);
        printf("[Master] Fiber %d has been joined.\n", i);
    }
    
    printf("[Master] All sub-tasks completed. Signaling shutdown.\n");
    dtact_shutdown();
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("--- Dtact C-FFI Example ---\n");
    
    // 1. Initialize Runtime
    dtact_config_t cfg = dtact_default_config();
    cfg.mWorkers = 4;
    cfg.mFiberCapacity = 1024; // Limit to 1024 fibers for this example
    cfg.mStackSize = 256 * 1024; // 256KB stacks are sufficient
    void* rt = dtact_init(&cfg);
    
    // 2. Launch Initial Root Fiber
    dtact_fiber_launch(master_fiber, NULL);
    
    // 3. Start Execution
    // This call blocks the main thread and starts 4 worker threads.
    // It returns when dtact_shutdown() is called.
    printf("Entering Runtime execution loop...\n");
    dtact_run(rt);
    
    printf("Runtime exited cleanly.\n");
    return 0;
}
```
## `dtact-util`: async I/O for the pieces of a workload DTA fits

The Design Philosophy and Performance Characteristics sections above are about the DTA scheduling algorithm itself: a decentralized, mesh-deflection scheduler over stackful fibers. That algorithm is a good fit for **Bag-of-Tasks (BoT) workloads at moderate-to-high load** and for **parallel batch computation** — Dtact doubles as a parallelization accelerator in that mode, not just an async I/O runtime. Its competitive-ratio analysis for workloads shaped like an arbitrary **DAG of dependencies is not yet complete**, and the combination of stackful fibers with decentralized, peer-to-peer scheduling decisions can put it at a modest I/O-throughput disadvantage relative to a centralized, stackless scheduler purpose-built for I/O fan-out.

`dtact-util` exists to give you a real choice at the I/O layer instead of forcing DTA's tradeoffs onto every workload:

* **`native` (default)** — A hand-rolled, lock-free driver (`io_uring` on Linux, IOCP on Windows, kqueue via `mio` elsewhere) built directly on Dtact's own fiber scheduler. Best when your I/O is **long-lived and/or highly concurrent**—many sustained connections, steady-state servers, or workloads where you want Dtact's mesh scheduling driving I/O-bound fibers alongside compute fibers without a second runtime.
* **`tokio`** — A thin wrapper over `tokio`'s own reactor (`tokio::net`/`fs`/`process`/`signal`/`time`), for embedding in (or alongside) a `tokio`-based application. Prefer this when your workload is dominated by **large numbers of short-lived connections**: Tokio's centralized, stackless reactor amortizes connection churn more efficiently than a decentralized, stackful design.
* **`sync` (standalone)** — A lightweight, high-performance module providing specialized asynchronous and synchronous primitives. Unlike the others, this module is backend-agnostic and does not support the `tokio` mode, focusing instead on pure, high-throughput utility primitives.

### Design Philosophy & OS Considerations

Dtact is built on the pillars of **self-organizing traffic scheduling with limited information**, **strict module decoupling** (the scheduler should remain agnostic of I/O state), and a **user-space-first execution model**.

This contrasts sharply with operating systems (notably Windows) that prioritize "suspension-first" philosophies to favor GUI interactivity. Consequently, you may observe performance variance in certain asynchronous primitives on these platforms—a reality shared by other runtimes like Tokio. Since our primary development and deployment target is Linux, we prioritize the integrity and performance of our core design over "patching" the erratic behaviors of OS-level I/O stacks.

Both backends expose the same public surface — `io`, `fs`, `process`, `signal`, `stream`, and `timer` modules — so switching backends is a `Cargo.toml` feature flip, not a rewrite. An optional `ffi` feature exposes all six as a blocking/synchronous C ABI (see [`dtact-util/dtact_util.h`](https://www.google.com/search?q=dtact-util/dtact_util.h)) for embedding from C/C++.

See [`examples/rust_dtact_util.rs`](https://www.google.com/search?q=examples/rust_dtact_util.rs) and [`examples/c_dtact_util.c`](https://www.google.com/search?q=examples/c_dtact_util.c) for end-to-end examples exercising `io`/`fs`/`stream`/`timer`/`process` together (build/run via `examples/Makefile`'s `run_rust_util` / `run_util` targets).

### Rust

```rust,no_run
use dtact::{dtact_await, dtact_init};
use dtact_util::io::{DtactTcpListener, DtactTcpStream, init as io_init};
use dtact_util::stream;
use dtact_util::{fs, timer};
use std::time::Duration;

#[dtact_init(workers = 4, stack = "256K", capacity = "1024")]
fn main() {
    println!("--- dtact-util comprehensive example (native backend) ---");

    // `io`/`fs` each own a small dedicated worker-thread pool, independent
    // of the dta_scheduler fiber workers above; start both once up front.
    io_init(2);
    fs::init(1);

    let timer_handle = dtact::spawn(async move {
        println!("[timer] sleeping 20ms...");
        timer::sleep(Duration::from_millis(20)).await;
        println!("[timer] awake.");
    });

    let fs_handle = dtact::spawn(async move {
        let dir = std::env::temp_dir().join(format!("dtact-util-example-{}", std::process::id()));
        fs::create_dir_all(&dir)
            .await
            .expect("create example temp dir");
        let path = dir.join("hello.txt");

        let file = fs::DtactFile::create(&path)
            .await
            .expect("create temp file");
        let (n, _buf) = file
            .write(b"hello from dtact-util fs".to_vec())
            .await
            .expect("write temp file");
        println!("[fs] wrote {n} bytes to {}", path.display());
        file.sync_all().await.expect("fsync temp file");
        drop(file);

        let file = fs::DtactFile::open(&path).await.expect("reopen temp file");
        let (n, buf) = file.read(vec![0u8; 64]).await.expect("read temp file");
        println!("[fs] read back: {:?}", String::from_utf8_lossy(&buf[..n]));

        let _ = fs::remove_file(&path).await;
    });

    let stream_handle = dtact::spawn(async move {
        let (a, b) = stream::pair(64);
        let msg = b"ping over dtact-util stream";
        let written = a.write(msg).await.expect("write to stream pair");
        println!("[stream] wrote {written} bytes into the pipe");

        let mut buf = vec![0u8; msg.len()];
        let read = b.read(&mut buf).await.expect("read from stream pair");
        println!(
            "[stream] read back {read} bytes: {:?}",
            String::from_utf8_lossy(&buf[..read])
        );
    });

    let io_handle = dtact::spawn(async move {
        let std_listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind std listener");
        let addr = std_listener.local_addr().expect("read local addr");
        let listener = DtactTcpListener::from_std(std_listener).expect("adopt std listener");

        // Server side: accept once, echo whatever it receives.
        let server = dtact::spawn(async move {
            let (stream, peer) = listener.accept().await.expect("accept connection");
            println!("[io] server accepted connection from {peer}");
            let mut buf = [0u8; 32];
            let n = stream.read(&mut buf).await.expect("server read");
            let n2 = stream.write(&buf[..n]).await.expect("server echo write");
            println!("[io] server echoed {n2} bytes back");
        });

        // Client side: connect, send a message, read the echo.
        let client = DtactTcpStream::connect(addr).await.expect("client connect");
        let msg = b"ping over dtact-util io";
        client.write(msg).await.expect("client write");
        let mut buf = [0u8; 32];
        let n = client.read(&mut buf).await.expect("client read echo");
        println!(
            "[io] client received echo: {:?}",
            String::from_utf8_lossy(&buf[..n])
        );

        dtact_await(server);
    });

    for (name, handle) in [
        ("timer", timer_handle),
        ("fs", fs_handle),
        ("stream", stream_handle),
        ("io", io_handle),
    ] {
        dtact_await(handle);
        println!("[master] {name} fiber joined.");
    }

    println!("--- all dtact-util primitives exercised successfully ---");
}
```

### C FFI

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../dtact-util/dtact_util.h"

#if defined(_WIN32)
#include <process.h>
#define getpid() ((int)_getpid())
#else
#include <unistd.h>
#endif

static void check_last_error(const char *what) {
    const char *msg = dtact_util_last_error_message();
    if (msg) {
        fprintf(stderr, "[error] %s: %s\n", what, msg);
    } else {
        fprintf(stderr, "[error] %s: (no error message recorded)\n", what);
    }
}

// --- timer -------------------------------------------------------------
static void demo_timer(void) {
    printf("[timer] sleeping 15ms via dtact_util_timer_sleep_ms...\n");
    dtact_util_timer_sleep_ms(15);

    DtactInterval *iv = dtact_util_timer_interval_create(5);
    if (!iv) {
        check_last_error("timer_interval_create");
        return;
    }
    for (int i = 0; i < 2; i++) {
        dtact_util_timer_interval_tick(iv);
        printf("[timer] interval tick %d\n", i);
    }
    dtact_util_timer_interval_free(iv);
}

// --- fs ------------------------------------------------------------------
static void demo_fs(void) {
    dtact_util_fs_init(1);

    char path[512];
    snprintf(path, sizeof(path), "dtact-util-c-example-%d.txt", (int)getpid());

    DtactFile *f = dtact_util_fs_file_create(path);
    if (!f) {
        check_last_error("fs_file_create");
        return;
    }
    const char *payload = "hello from the dtact-util C example";
    ptrdiff_t written = dtact_util_fs_file_write(f, (const uint8_t *)payload, strlen(payload));
    printf("[fs] wrote %lld bytes to %s\n", (long long)written, path);
    dtact_util_fs_file_sync(f);
    dtact_util_fs_file_close(f);

    DtactFile *f2 = dtact_util_fs_file_open(path);
    if (!f2) {
        check_last_error("fs_file_open");
        return;
    }
    uint8_t buf[128] = {0};
    ptrdiff_t got = dtact_util_fs_file_read(f2, buf, sizeof(buf) - 1);
    printf("[fs] read back %lld bytes: %s\n", (long long)got, (const char *)buf);
    dtact_util_fs_file_close(f2);

    remove(path);
}

// --- stream (in-process duplex pipe) -------------------------------------
static void demo_stream(void) {
    DtactStream *a = NULL;
    DtactStream *b = NULL;
    if (dtact_util_stream_pair_create(64, &a, &b) != 0) {
        check_last_error("stream_pair_create");
        return;
    }

    const char *msg = "ping over dtact-util stream";
    ptrdiff_t w = dtact_util_stream_write(a, (const uint8_t *)msg, strlen(msg));
    printf("[stream] wrote %lld bytes into the pipe\n", (long long)w);

    uint8_t buf[64] = {0};
    ptrdiff_t r = dtact_util_stream_read(b, buf, sizeof(buf) - 1);
    printf("[stream] read back %lld bytes: %s\n", (long long)r, (const char *)buf);

    dtact_util_stream_free(a);
    dtact_util_stream_free(b);
}

// --- io (loopback TCP echo) ----------------------------------------------
static void demo_io(void) {
    dtact_util_io_init(1);

    // Discover a free loopback port the same way the crate's own ffi_test
    // does: bind with std/libc, close it, then hand the fixed address to
    // dtact_util. There is no FFI accessor for a listener's bound address.
    // The C example keeps this simple by using a fixed high port instead of
    // an OS-assigned ephemeral one.
    const char *addr = "127.0.0.1:38213";

    DtactTcpListener *listener = dtact_util_io_listener_bind(addr);
    if (!listener) {
        check_last_error("io_listener_bind");
        return;
    }

    DtactTcpStream *client = dtact_util_io_stream_connect(addr);
    if (!client) {
        check_last_error("io_stream_connect");
        dtact_util_io_listener_close(listener);
        return;
    }

    DtactTcpStream *server_side = dtact_util_io_listener_accept(listener);
    if (!server_side) {
        check_last_error("io_listener_accept");
        dtact_util_io_stream_close(client);
        dtact_util_io_listener_close(listener);
        return;
    }

    const char *msg = "ping over dtact-util io";
    dtact_util_io_stream_write(client, (const uint8_t *)msg, strlen(msg));

    uint8_t buf[64] = {0};
    ptrdiff_t n = dtact_util_io_stream_read(server_side, buf, sizeof(buf) - 1);
    printf("[io] server received: %.*s\n", (int)n, buf);
    dtact_util_io_stream_write(server_side, buf, (size_t)n);

    ptrdiff_t echoed = dtact_util_io_stream_read(client, buf, sizeof(buf) - 1);
    printf("[io] client received echo: %.*s\n", (int)echoed, buf);

    dtact_util_io_stream_close(server_side);
    dtact_util_io_stream_close(client);
    dtact_util_io_listener_close(listener);
}

// --- process ---------------------------------------------------------------
static void demo_process(void) {
    dtact_util_process_init(1);

#if defined(_WIN32)
    const char *prog = "cmd";
    const char *argv[] = {"/C", "echo", "hello-from-child", NULL};
    size_t argc = 3;
#else
    const char *prog = "sh";
    const char *argv[] = {"-c", "printf hello-from-child", NULL};
    size_t argc = 2;
#endif

    DtactChild *child = dtact_util_process_spawn(prog, argv, argc, DTACT_STDOUT_PIPE);
    if (!child) {
        check_last_error("process_spawn");
        return;
    }

    DtactChildStdout *out = dtact_util_process_child_take_stdout(child);
    if (out) {
        uint8_t buf[128] = {0};
        ptrdiff_t total = 0;
        for (;;) {
            ptrdiff_t n = dtact_util_process_stdout_read(out, buf + total, sizeof(buf) - 1 - (size_t)total);
            if (n <= 0) break;
            total += n;
        }
        printf("[process] child stdout: %.*s\n", (int)total, buf);
        dtact_util_process_stdout_free(out);
    }

    int32_t exit_code = 0;
    dtact_util_process_child_wait(child, &exit_code);
    printf("[process] child exited with code %d\n", exit_code);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("--- dtact-util comprehensive C FFI example ---\n");

    demo_timer();
    demo_fs();
    demo_stream();
    demo_io();
    demo_process();

    printf("--- all dtact-util primitives exercised successfully ---\n");
    return 0;
}
```

## License

This project is licensed under either of the MIT license or the Apache License (Version 2.0).

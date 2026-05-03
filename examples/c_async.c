#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

// dtact FFI types
typedef struct {
    uint64_t handle;
} dtact_handle_t;

typedef struct {
    uint32_t workers;
    uint8_t safety_level;
    uint8_t topology_mode;
    uint32_t contexts;
    uint64_t stack_size;
} dtact_config_t;

// Dtact FFI Prototypes
extern void dtact_default_config(dtact_config_t* cfg);
extern void* dtact_init(const dtact_config_t* cfg);
extern dtact_handle_t dtact_fiber_launch(void (*func)(void*), void* arg);
extern void dtact_await(dtact_handle_t handle);
extern void dtact_run(void* rt);
extern void dtact_shutdown();
extern void dtact_free_arg(void* arg);

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
        handles[i] = dtact_fiber_launch(worker_fiber, val);
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
    dtact_config_t cfg;
    dtact_default_config(&cfg);
    cfg.workers = 4;
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

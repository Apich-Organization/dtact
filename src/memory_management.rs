#![allow(unsafe_code)]
#![allow(non_snake_case)]

use core::cell::UnsafeCell;

use crate::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// Defensive upper cap on how many per-worker batch caches a single
/// `ContextPool` will ever allocate, regardless of the `num_workers` its
/// constructor is given. A worker id at or beyond `min(num_workers,
/// MAX_CACHED_WORKERS)` falls back to the safe global CAS path — still
/// fully correct, just without the fast path. Exists only to bound
/// allocation size against a pathological caller-supplied `num_workers`;
/// comfortably above any realistic core count.
const MAX_CACHED_WORKERS: usize = 4096;

/// Upper bound on how many free-list nodes one batch refill/donate moves
/// per CAS. Matches `CHUNK_SIZE` (`src/dta_scheduler.rs`) — the same "how
/// many indices per shared-pointer exchange" convention this codebase
/// already uses for the scheduler's own mailbox chunks.
const MAX_LOCAL_BATCH: u32 = 32;

/// A per-worker cache of free [`ContextPool`] slot indices, sitting in
/// front of the shared CAS-protected free list to amortize its cost:
/// `alloc_context`/`free_context` pop/push this cache directly (no atomics)
/// in the common case, only touching the shared list once every
/// `batch_size` operations.
///
/// No slot "ownership" is tracked: unlike a page-based allocator (e.g.
/// mimalloc, which must return memory to its originating page for OS-level
/// reclamation), every `ContextPool` slot is interchangeable and lives in
/// one pre-allocated arena for the pool's whole lifetime — a freed slot can
/// go into whichever worker's cache frees it, regardless of who originally
/// allocated it.
///
/// # Safety invariant
/// Never touched by more than one thread at a time: it is only ever
/// accessed via `ContextPool::local_caches[worker_id]`, where `worker_id`
/// comes from [`crate::future_bridge::CURRENT_WORKER_ID`] — a value set
/// exactly once, for the life of the OS thread, by the one worker thread
/// `DtaScheduler::run_worker_static` spawns for that id
/// (`src/dta_scheduler.rs`), and read (never set) everywhere else. No other
/// thread ever reads or writes this specific cache slot. Mirrors the same
/// "protocol-exclusive `UnsafeCell`" reasoning already used for `Worker` in
/// `src/dta_scheduler.rs` and `TaskSlab` in `src/benchmark/dta_harness.rs`.
struct LocalFreeCache {
    slots: [u32; (MAX_LOCAL_BATCH * 2) as usize],
    len: u32,
}

impl LocalFreeCache {
    const fn new() -> Self {
        Self {
            slots: [0; (MAX_LOCAL_BATCH * 2) as usize],
            len: 0,
        }
    }
}

/// Safety policies for context pool memory layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyLevel {
    /// Raw performance: No guard pages, minimal overhead.
    Safety0,
    /// Balanced: Guard pages every 32 contexts to catch massive overflows.
    Safety1,
    /// Strict: Per-context hardware guard pages for maximum isolation.
    Safety2,
}

pub use crate::common_types::{TopologyMode, WorkloadKind};

/// Machine-specific registers for context switching.
///
/// Aligned to 64 bytes to prevent cache line splits and ensure atomic
/// context updates on supported architectures.
#[repr(C, align(64))]
#[derive(Debug)]
pub(crate) struct Registers {
    /// General Purpose Registers (GPRs).
    pub(crate) gprs: [u64; 16],
    /// SIMD / Extended state (e.g. AVX, Neon).
    pub(crate) extended_state: [u8; 512],
}

impl Registers {
    /// Creates a new register set, with a sane default MXCSR on `x86_64`.
    ///
    /// # The bug this fixes
    /// The `*_float` switchers in `context_switch.rs` (both cross-thread
    /// and same-thread; the `*_no_float` switchers never touch this state
    /// at all — that is the entire point of "no float") store the SSE
    /// control word (MXCSR) inside `gprs` on `x86_64` — Unix at `gprs[8]`
    /// (byte offset 64), Windows at `gprs[14]` (byte offset 112, alongside
    /// the extra TIB/XMM6-15 state that ABI requires) — and `ldmxcsr` it
    /// into the live CPU register on every switch. MXCSR's six
    /// exception-mask bits use **inverted** polarity from what a zeroed
    /// word usually means: `1` = masked (disabled, the safe default), `0`
    /// = unmasked (that exception class traps immediately on the next
    /// occurrence). A context that has never yet been switched into — i.e.
    /// every freshly allocated `FiberContext` slot, the common case for
    /// any newly spawned fiber — previously carried a zeroed `gprs`, so
    /// its first dispatch loaded MXCSR `0x00000000`: every exception class
    /// unmasked, including "precision" (inexact), which fires on nearly
    /// every non-exact floating-point result. The very first non-trivial
    /// float operation that fiber (or, transitively, anything running on
    /// that same OS thread before it switches back) performs then
    /// reliably raises `SIGFPE`.
    ///
    /// `0x1F80` is the standard SSE reset value: all six exception classes
    /// masked, round-to-nearest, denormals-are-zero/flush-to-zero off —
    /// the same default every thread on the process's main stack already
    /// runs under, so a first-ever fiber dispatch now behaves identically
    /// to ordinary code instead of silently arming a crash.
    ///
    /// `AArch64`'s FPCR and RISC-V's `fcsr` do not have this hazard: `AArch64`
    /// FPCR's trap-enable bits default to *disabled* at `0` (the opposite
    /// polarity from MXCSR, so zero-init is already safe there), and
    /// RISC-V's F/D extension does not trap floating-point exceptions at
    /// all in the base ISA (they only accumulate as sticky `fflags` bits,
    /// checked by software, never delivered as a signal) — so neither
    /// needs the equivalent of this fix.
    #[must_use]
    #[inline(always)]
    pub(crate) const fn new() -> Self {
        // `mut` is only exercised on `x86_64` (the two `cfg` blocks below);
        // AArch64 and RISC-V need no special-cased slot, since neither has
        // this hazard (see the module-level doc comment above).
        #[allow(unused_mut)]
        let mut gprs = [0u64; 16];
        #[cfg(all(target_arch = "x86_64", unix))]
        {
            gprs[8] = 0x0000_0000_0000_1F80;
        }
        #[cfg(all(target_arch = "x86_64", windows))]
        {
            gprs[14] = 0x0000_0000_0000_1F80;
        }
        Self {
            gprs,
            extended_state: [0; 512],
        }
    }
}

impl Default for Registers {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Lifecycle state of a fiber.
#[repr(u32)]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FiberStatus {
    /// The fiber is newly created and has not yet been polled.
    Initial = 0,
    /// The fiber is currently being executed by a worker core.
    Running = 1,
    /// The fiber is suspended and waiting for an event (e.g. I/O or Mutex).
    Yielded = 2,
    /// The fiber has successfully completed its execution.
    Finished = 3,
    /// Terminated due to an unhandled panic.
    Panicked = 4,
    /// The fiber was woken up by a waker.
    Notified = 5,
    /// The fiber is currently transitioning to a suspended state.
    Suspending = 6,
    /// Terminated cooperatively in response to `cancel()`, distinct from
    /// an unhandled panic even though both unwind the fiber's stack.
    Cancelled = 7,
}

/// Terminal outcomes of a fiber, as observed by a caller after joining it.
///
/// Distinct from [`FiberStatus`]: this is the small, public subset of
/// terminal states relevant to a caller of [`crate::api::outcome`], not
/// the full internal lifecycle enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskOutcome {
    /// The fiber ran to completion without panicking or being cancelled.
    Finished,
    /// The fiber unwound due to an unhandled panic in its body.
    Panicked,
    /// The fiber unwound because [`crate::api::cancel`] was called on it.
    Cancelled,
}

/// The hardware-level execution context for a stackful fiber.
///
/// Cache-line layout (repr C, 64-byte aligned):
///   Lines 0–19: `regs`, `executor_regs` (640 B / 10 lines each) — untouched
///               by this layout's reasoning, see [`Registers`].
///   Line 20:    `state`, `cancel_requested` — both read *and* written by a
///               remote thread (`try_notify`'s wake, `cancel`'s flag) *and*
///               checked every iteration of this fiber's own
///               `wait_pinned` loop. Isolated together so a wake touches
///               exactly one line, and so neither shares a line with...
///   Line 21:    `adaptive_spin_count`, `spin_failure_count` — also
///               touched every `wait_pinned` iteration, but *only* by this
///               fiber's own thread (self-tuning, never written remotely).
///               An earlier layout packed these onto the same line as
///               `state` and, later, `cancel_requested`: every remote wake
///               or `cancel()` call would then invalidate this purely
///               thread-local spin-budget data for no reason, and every
///               local spin-budget update would invalidate the line a
///               remote wake was about to read. Splitting them is the
///               actual fix; the rest of this struct (setup-once fields,
///               read after spawn but essentially never rewritten) is far
///               less contended and does not need the same treatment.
///   Line 22+:   everything else — `fiber_index`, the closure/trampoline
///               pointers, `waiter_thread_id`/`waiter_handle`, etc.
#[repr(C, align(64))]
#[doc(hidden)]
pub struct FiberContext {
    /// Standard CPU registers (rax, rbx, etc.)
    pub regs: Registers,
    /// Return address for the scheduler dispatch loop.
    pub executor_regs: Registers,

    /// Current execution state.
    pub state: AtomicU32,
    /// Set by [`crate::api::cancel`]; observed by this fiber itself the
    /// next time it resumes through [`crate::future_bridge::wait_pinned`].
    /// Cooperative only: a fiber that never suspends through a DTA
    /// primitive never observes this flag.
    pub(crate) cancel_requested: AtomicBool,
    // Fill cache line 20 to 64 bytes: state(4) + cancel_requested(1) = 5.
    _pad_sync: [u8; 59],

    /// Statistics: Adaptive Spin Budget. Thread-local — see the struct-level
    /// doc comment for why this is deliberately not on the same line as
    /// `state`/`cancel_requested` above.
    pub adaptive_spin_count: u32,
    /// Statistics: Recent Spin Failures.
    pub spin_failure_count: u32,
    // Fill cache line 21 to 64 bytes: 8 bytes used, 56 bytes pad.
    _pad_spin: [u8; 56],

    /// Fiber identification index.
    pub fiber_index: u32,
    /// The OS thread ID where this fiber was last executed.
    pub last_os_thread_id: u64,
    /// The hardware core ID where this fiber was originally spawned.
    pub origin_core: u16,
    /// Pointer to the assembly context-switch function.
    pub switch_fn: unsafe extern "C" fn(*mut Registers, *const Registers),
    /// Pointer to the fiber's entry-point closure or future.
    pub closure_ptr: *mut (),
    /// Trampoline address for C-FFI or Rust closure invocation.
    pub trampoline: unsafe extern "C" fn(),
    /// Internal wrapper to drive the closure or poll the future.
    pub invoke_closure: fn(*mut ()),
    /// Optional cleanup callback (used for C-FFI ownership management).
    pub cleanup_fn: Option<unsafe extern "C" fn(*mut ())>,
    /// Pointer to the fiber's 8KB read/stack buffer.
    pub read_buffer_ptr: *mut u8,
    /// Metadata: Workload Hint.
    pub kind: WorkloadKind,
    /// Metadata: Topology Strategy.
    pub mode: TopologyMode,
    /// Metadata: Core Affinity Hint for wake routing.
    pub affinity: crate::api::topology::Affinity,

    /// Current stack pointer for this fiber.
    pub(crate) stack_ptr: usize,
    /// Saved stack pointer of the executor thread.
    pub(crate) scheduler_stack_ptr: usize,
    /// OS-specific TIB stack limit (Windows).
    pub(crate) tib_stack_limit: usize,
    /// OS-specific TIB stack base (Windows).
    pub(crate) tib_stack_base: usize,
    /// Thread ID of a non-fiber waiter (for C-FFI join).
    pub(crate) waiter_thread_id: AtomicU64,
    /// Handle of a fiber waiter (for C-FFI join).
    pub(crate) waiter_handle: AtomicU64,
    /// Generation counter to prevent ABA in handles.
    pub(crate) generation: AtomicU32,
    /// Link to the next available context in the free list.
    pub(crate) next_free: AtomicU32,
    /// Pointer to panic payload if the fiber crashed.
    pub(crate) panic_payload_ptr: *mut (),
    /// Pointer to the result of the fiber.
    pub(crate) result_ptr: *mut (),
    /// Opaque pointer for reader bridge.
    pub(crate) reader_ptr: *mut (),
    /// Reference to a shared buffer.
    pub(crate) buf_ptr: *mut [u8],
}

impl FiberContext {
    /// Creates a new, blank `FiberContext`.
    ///
    /// `const fn` on normal builds; plain `fn` under `cfg(loom)` because
    /// loom's atomic types do not have `const` constructors.
    #[cfg(loom)]
    pub fn new() -> Self {
        Self {
            stack_ptr: 0,
            scheduler_stack_ptr: 0,
            tib_stack_limit: 0,
            tib_stack_base: 0,
            state: AtomicU32::new(FiberStatus::Initial as u32),
            kind: WorkloadKind::Compute,
            mode: TopologyMode::P2PMesh,
            affinity: crate::api::topology::Affinity::SameCore,
            origin_core: 0,
            fiber_index: 0,
            waiter_thread_id: AtomicU64::new(0),
            waiter_handle: AtomicU64::new(0),
            generation: AtomicU32::new(0),
            regs: Registers::new(),
            executor_regs: Registers::new(),
            next_free: AtomicU32::new(u32::MAX),
            panic_payload_ptr: core::ptr::null_mut(),
            trampoline: dummy_trampoline,
            invoke_closure: dummy_invoke,
            closure_ptr: core::ptr::null_mut(),
            result_ptr: core::ptr::null_mut(),
            reader_ptr: core::ptr::null_mut(),
            buf_ptr: core::ptr::slice_from_raw_parts_mut(core::ptr::null_mut(), 0),
            read_buffer_ptr: core::ptr::null_mut(),
            switch_fn: crate::context_switch::switch_context_cross_thread_float,
            cleanup_fn: None,
            adaptive_spin_count: 50,
            spin_failure_count: 0,
            last_os_thread_id: 0,
            cancel_requested: AtomicBool::new(false),
            _pad_sync: [0; 59],
            _pad_spin: [0; 56],
        }
    }
    /// Creates a new, blank `FiberContext`.
    ///
    /// `const fn` on normal builds; plain `fn` under `cfg(loom)` because
    /// loom's atomic types do not have `const` constructors.
    #[cfg(not(loom))]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            stack_ptr: 0,
            scheduler_stack_ptr: 0,
            tib_stack_limit: 0,
            tib_stack_base: 0,
            state: AtomicU32::new(FiberStatus::Initial as u32),
            kind: WorkloadKind::Compute,
            mode: TopologyMode::P2PMesh,
            affinity: crate::api::topology::Affinity::SameCore,
            origin_core: 0,
            fiber_index: 0,
            waiter_thread_id: AtomicU64::new(0),
            waiter_handle: AtomicU64::new(0),
            generation: AtomicU32::new(0),
            regs: Registers::new(),
            executor_regs: Registers::new(),
            next_free: AtomicU32::new(u32::MAX),
            panic_payload_ptr: core::ptr::null_mut(),
            trampoline: dummy_trampoline,
            invoke_closure: dummy_invoke,
            closure_ptr: core::ptr::null_mut(),
            result_ptr: core::ptr::null_mut(),
            reader_ptr: core::ptr::null_mut(),
            buf_ptr: core::ptr::slice_from_raw_parts_mut(core::ptr::null_mut(), 0),
            read_buffer_ptr: core::ptr::null_mut(),
            switch_fn: crate::context_switch::switch_context_cross_thread_float,
            cleanup_fn: None,
            adaptive_spin_count: 50,
            spin_failure_count: 0,
            last_os_thread_id: 0,
            cancel_requested: AtomicBool::new(false),
            _pad_sync: [0; 59],
            _pad_spin: [0; 56],
        }
    }

    /// Attempts to transition `state` to `Notified` in response to a wake,
    /// succeeding only if the fiber is currently in one of the "live,
    /// possibly-waiting" states (`Running`, `Suspending`, `Yielded`).
    ///
    /// A wake — whether from `crate::api::cancel`/`yield_to`
    /// (`crate::awaken_fiber_by_index`) or a stored [`core::task::Waker`]
    /// firing after the fiber it was created for has already moved on
    /// (`future_bridge::wake_by_ref_impl`) — can legitimately arrive
    /// after the target has already terminated (`Finished`/`Panicked`/
    /// `Cancelled`) or had its slot reclaimed (`Initial`). Both call
    /// sites used to swap `state` to `Notified` unconditionally,
    /// silently overwriting a terminal value — and since nothing ever
    /// transitions a slot back out of `Notified` once its owning fiber
    /// is gone, any `dtact_await`/`crate::api::outcome` still waiting on
    /// that handle would then hang forever. Not a theoretical concern:
    /// reproduced with a minimal standalone program (two fibers, one
    /// `yield_to`-ing the other after it had already finished).
    ///
    /// Returns `true` iff the previous state was `Yielded` — the fiber
    /// was fully parked off any run queue and the caller must enqueue it
    /// itself; for `Running`/`Suspending` the fiber's own worker will
    /// notice `Notified` when it resumes from its context switch, and
    /// for anything else (already terminal, `Initial`, or already
    /// `Notified`) this is correctly a no-op.
    #[inline(always)]
    pub(crate) fn try_notify(&self) -> bool {
        self.state
            .try_update(Ordering::AcqRel, Ordering::Acquire, |s| {
                if s == FiberStatus::Running as u32
                    || s == FiberStatus::Suspending as u32
                    || s == FiberStatus::Yielded as u32
                {
                    Some(FiberStatus::Notified as u32)
                } else {
                    None
                }
            })
            == Ok(FiberStatus::Yielded as u32)
    }
}

#[cfg(not(loom))]
impl Default for FiberContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(loom)]
impl Default for FiberContext {
    fn default() -> Self {
        Self::new()
    }
}

const unsafe extern "C" fn dummy_trampoline() {}
const fn dummy_invoke(_: *mut ()) {}

/// A page-aligned arena for managing fiber stacks and control blocks.
///
/// The `ContextPool` ensures O(1) allocation and hardware-level isolation
/// through tiered safety levels and OS memory protection primitives.
#[allow(dead_code)]
pub struct ContextPool {
    base_ptr: *mut u8,
    total_size: usize,
    /// Size of each context slot in bytes.
    pub slot_size: usize,
    /// OS page size resolved at construction time. macOS arm64 uses 16 KiB
    /// pages, so this must be queried via `sysconf` rather than hardcoded —
    /// the slot layout depends on it for guard-page offsets.
    page_size: usize,
    #[allow(dead_code)]
    capacity: u32,
    safety: SafetyLevel,
    free_head: AtomicU64,
    /// Byte offset from slot start to the `FiberContext` within each slot.
    ///
    /// Pre-computed once as `slot_size − ceil_align(size_of::<FiberContext>(), 64)`
    /// and cached here so `get_context_ptr` — called on every fiber dispatch —
    /// never re-executes the alignment arithmetic at runtime.
    pub context_end_offset: usize,
    /// Per-(pool, worker-id) batch caches amortizing `free_head`'s CAS cost.
    /// See [`LocalFreeCache`] and `alloc_context`/`free_context`.
    local_caches: Box<[UnsafeCell<LocalFreeCache>]>,
    /// How many free-list nodes one refill/donate moves per CAS, scaled to
    /// this pool's own capacity (see `new()`) so a batch can never claim a
    /// disproportionate share of a small pool.
    batch_size: u32,
}

unsafe impl Send for ContextPool {}
unsafe impl Sync for ContextPool {}

impl ContextPool {
    /// Builds this pool's empty per-worker batch caches and computes the
    /// batch size they refill/donate by.
    ///
    /// Sized against `num_workers`, not just `capacity`: every worker's
    /// cache can independently grow up to `2 * batch_size` before it must
    /// donate, so the *worst case* amount of the pool's capacity sitting
    /// idle in caches (invisible to any other caller) is `num_workers * 2 *
    /// batch_size`. Choosing `batch_size = capacity / (8 * num_workers)`
    /// keeps that worst case at `capacity / 4` — at most a quarter of the
    /// pool can ever be cache-resident at once, leaving the rest always
    /// reachable through the shared global list. Getting this wrong is not
    /// a performance footnote: sizing `batch_size` off `capacity` alone
    /// (ignoring `num_workers`) previously let real worker caches
    /// collectively hoard up to 100% of a pool's capacity, starving any
    /// other caller (e.g. a host thread's `alloc_context_global`) — a real,
    /// reproduced-under-stress-testing livelock, not a theoretical concern.
    ///
    /// Also scaled down for small pools so one worker's batch can't claim a
    /// disproportionate share of total capacity (e.g. capacity=2 in several
    /// tests clamps to `batch_size=1`, exactly degenerating to the uncached
    /// per-node behavior — zero risk, zero benefit, the right tradeoff for
    /// pools that tiny).
    fn new_local_caches(
        capacity: u32,
        num_workers: usize,
    ) -> (Box<[UnsafeCell<LocalFreeCache>]>, u32) {
        let effective_workers = num_workers.clamp(1, MAX_CACHED_WORKERS);
        #[allow(clippy::cast_possible_truncation)]
        let divisor = (8 * effective_workers) as u32;
        let batch_size = (capacity / divisor).clamp(1, MAX_LOCAL_BATCH);
        let local_caches = (0..effective_workers)
            .map(|_| UnsafeCell::new(LocalFreeCache::new()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        (local_caches, batch_size)
    }

    /// Creates a new `ContextPool` with the specified capacity and safety.
    ///
    /// `num_workers` is the number of real scheduler workers that will ever
    /// call [`Self::alloc_context`]/[`Self::free_context`] from their own
    /// dispatch thread (i.e. whatever is passed to the paired
    /// `DtaScheduler::new`) — it sizes and bounds the per-worker batch
    /// caches (see [`Self::new_local_caches`]) so they can never
    /// collectively hoard more than a bounded fraction of `capacity`. Pass
    /// `1` for a pool with no real scheduler workers attached (e.g. a
    /// standalone allocator-only test): every caller then takes the
    /// uncached global path, identical to this pool's pre-batch-cache
    /// behavior.
    ///
    /// This function performs the initial bulk allocation (via mmap or
    /// `VirtualAlloc`) and configures any requested hardware guard pages.
    ///
    /// # Errors
    /// Returns an error if the OS fails to allocate the requested memory region
    /// or if hardware protection cannot be applied to the guard pages.
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[inline(never)]
    pub fn new(
        capacity: u32,
        stack_size: usize,
        safety: SafetyLevel,
        numa: usize,
        num_workers: usize,
    ) -> Result<Self, &'static str> {
        #[cfg(unix)]
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        #[cfg(windows)]
        let page_size = unsafe {
            let mut info = core::mem::zeroed();
            windows_sys::Win32::System::SystemInformation::GetSystemInfo(&raw mut info);
            info.dwPageSize as usize
        };

        #[allow(clippy::items_after_statements)]
        const ALIGN: usize = 64;
        let context_sz = (core::mem::size_of::<FiberContext>() + ALIGN - 1) & !(ALIGN - 1);

        // Slot Size: [ Stack Space | 8KB Read Buffer | FiberContext ]
        let slot_size = (stack_size + context_sz + 8192 + page_size - 1) & !(page_size - 1);
        // Pre-compute the intra-slot byte offset to FiberContext, eliminating
        // a subtract on every get_context_ptr call in the dispatch hot path.
        let context_end_offset = slot_size - context_sz;

        let total_size = match safety {
            SafetyLevel::Safety0 => capacity as usize * slot_size,
            SafetyLevel::Safety1 => {
                let num_groups = (capacity as usize).div_ceil(32);
                capacity as usize * slot_size + num_groups * page_size
            }
            SafetyLevel::Safety2 => capacity as usize * (slot_size + page_size),
        };

        // Add 4KB for SEH/Metadata
        let total_size_with_meta = total_size + 4096;

        unsafe {
            let base_ptr = Self::allocate_arena(total_size_with_meta, safety, numa)?;

            // PRE-PROTECT Guard Pages
            if safety == SafetyLevel::Safety1 {
                for i in 0..capacity.div_ceil(32) {
                    let guard_ptr = base_ptr.add(i as usize * (slot_size * 32 + page_size));
                    Self::apply_hardware_protection(guard_ptr, page_size)?;
                }
            } else if safety == SafetyLevel::Safety2 {
                for i in 0..capacity {
                    let guard_ptr = base_ptr.add(i as usize * (slot_size + page_size));
                    Self::apply_hardware_protection(guard_ptr, page_size)?;
                }
            }

            let (local_caches, batch_size) = Self::new_local_caches(capacity, num_workers);

            let pool = Self {
                base_ptr,
                total_size: total_size_with_meta,
                slot_size,
                page_size,
                capacity,
                safety,
                free_head: AtomicU64::new(0),
                context_end_offset,
                local_caches,
                batch_size,
            };

            for i in 0..capacity {
                let ctx_ptr = pool.get_context_ptr(i);
                core::ptr::write(ctx_ptr, FiberContext::new());
                (*ctx_ptr).fiber_index = i;

                // Robust Aligned Read Buffer (64-byte aligned)
                let raw_read_buf = ctx_ptr.cast::<u8>().sub(8192);
                (*ctx_ptr).read_buffer_ptr = (raw_read_buf as usize & !63) as *mut u8;

                (*ctx_ptr).next_free.store(i + 1, Ordering::Relaxed);
            }

            let last_ctx = pool.get_context_ptr(capacity - 1);
            (*last_ctx).next_free.store(u32::MAX, Ordering::Relaxed);

            // Windows SEH Registration
            #[cfg(windows)]
            {
                use windows_sys::Win32::System::Diagnostics::Debug::{
                    IMAGE_RUNTIME_FUNCTION_ENTRY, RtlAddFunctionTable,
                };

                #[repr(C, packed)]
                struct UnwindInfo {
                    version_and_flags: u8,
                    prolog_size: u8,
                    unwind_code_count: u8,
                    frame_register_and_offset: u8,
                }

                let meta_base = base_ptr.add(total_size);
                let unwind_info_ptr = meta_base.cast::<UnwindInfo>();
                core::ptr::write(
                    unwind_info_ptr,
                    UnwindInfo {
                        version_and_flags: 0x01,
                        prolog_size: 0,
                        unwind_code_count: 0,
                        frame_register_and_offset: 0,
                    },
                );

                #[allow(clippy::cast_ptr_alignment)]
                let function_table_ptr = meta_base
                    .add(core::mem::size_of::<UnwindInfo>())
                    .cast::<IMAGE_RUNTIME_FUNCTION_ENTRY>();
                core::ptr::write(
                    function_table_ptr,
                    IMAGE_RUNTIME_FUNCTION_ENTRY {
                        BeginAddress: 0,
                        EndAddress: total_size as u32,
                        Anonymous: windows_sys::Win32::System::Diagnostics::Debug::IMAGE_RUNTIME_FUNCTION_ENTRY_0 {
                            UnwindData: (unwind_info_ptr as usize - base_ptr as usize) as u32,
                        },
                    },
                );

                let base = base_ptr as usize;
                RtlAddFunctionTable(function_table_ptr.cast_const(), 1, base as u64);
            }

            Ok(pool)
        }
    }

    #[inline(always)]
    fn apply_hardware_protection(ptr: *mut u8, size: usize) -> Result<(), &'static str> {
        #[cfg(unix)]
        unsafe {
            if libc::mprotect(ptr.cast(), size, libc::PROT_NONE) != 0 {
                return Err("mprotect failed");
            }
        }
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, VirtualProtect};
            let mut old = 0;
            if VirtualProtect(ptr.cast(), size, PAGE_NOACCESS, &raw mut old) == 0 {
                return Err("VirtualProtect failed");
            }
        }
        Ok(())
    }

    #[inline(always)]
    #[allow(clippy::useless_let_if_seq)]
    #[allow(clippy::cast_possible_truncation)]
    unsafe fn allocate_arena(
        size: usize,
        safety: SafetyLevel,
        numa: usize,
    ) -> Result<*mut u8, &'static str> {
        unsafe {
            #[cfg(unix)]
            {
                let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
                let mut ptr = libc::MAP_FAILED;

                // Try HugeTLB for Safety0 (best perf), fall back to standard pages
                if safety == SafetyLevel::Safety0 {
                    ptr = unsafe {
                        libc::mmap(
                            core::ptr::null_mut(),
                            size,
                            libc::PROT_READ | libc::PROT_WRITE,
                            flags | 0x40000, // MAP_HUGETLB
                            -1,
                            0,
                        )
                    };
                }

                if ptr == libc::MAP_FAILED {
                    // Add MAP_NORESERVE so virtual mapping succeeds under
                    // strict overcommit accounting (containers, QEMU CI).
                    // Physical pages are demand-faulted on first write —
                    // exactly what we want for a sparsely-used context arena.
                    ptr = unsafe {
                        libc::mmap(
                            core::ptr::null_mut(),
                            size,
                            libc::PROT_READ | libc::PROT_WRITE,
                            flags | libc::MAP_NORESERVE,
                            -1,
                            0,
                        )
                    };
                }

                if ptr == libc::MAP_FAILED {
                    return Err("mmap failed");
                }

                // Linux-only: hint THP for the arena if we fell back to plain
                // mmap (Safety1/2 or HUGETLB-exhausted Safety0).  Reduces TLB
                // misses across the lifetime of the runtime.  Safety0 with
                // successful HUGETLB already gets explicit huge pages.
                #[cfg(target_os = "linux")]
                {
                    const MADV_HUGEPAGE: libc::c_int = 14;
                    libc::madvise(ptr, size, MADV_HUGEPAGE);
                }

                #[cfg(target_os = "linux")]
                if numa > 0 {
                    let mask: usize = 1 << (numa % 64);
                    // MPOL_BIND = 2
                    libc::syscall(libc::SYS_mbind, ptr, size, 2, &raw const mask, 64, 0);
                }

                Ok(ptr.cast::<u8>())
            }
            #[cfg(windows)]
            {
                use windows_sys::Win32::System::Memory::{
                    MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
                };
                let mut flags = MEM_COMMIT | MEM_RESERVE;
                if safety == SafetyLevel::Safety0 {
                    flags |= 0x2000_0000;
                } // MEM_LARGE_PAGES

                let mut ptr = if numa > 0 {
                    windows_sys::Win32::System::Memory::VirtualAllocExNuma(
                        windows_sys::Win32::System::Threading::GetCurrentProcess(),
                        core::ptr::null_mut(),
                        size,
                        flags,
                        PAGE_READWRITE,
                        numa as u32,
                    )
                } else {
                    VirtualAlloc(core::ptr::null_mut(), size, flags, PAGE_READWRITE)
                };

                if ptr.is_null() && (flags & 0x2000_0000) != 0 {
                    let fallback_flags = flags & !0x2000_0000;
                    ptr = if numa > 0 {
                        windows_sys::Win32::System::Memory::VirtualAllocExNuma(
                            windows_sys::Win32::System::Threading::GetCurrentProcess(),
                            core::ptr::null_mut(),
                            size,
                            fallback_flags,
                            PAGE_READWRITE,
                            numa as u32,
                        )
                    } else {
                        VirtualAlloc(core::ptr::null_mut(), size, fallback_flags, PAGE_READWRITE)
                    };
                }

                if ptr.is_null() {
                    return Err("VirtualAlloc failed");
                }
                Ok(ptr.cast::<u8>())
            }
        }
    }

    /// Returns a raw pointer to a context based on its index.
    ///
    /// Hot path: called once per fiber dispatch. `context_end_offset` is
    /// pre-computed at pool construction so this function executes a single
    /// multiply + two adds + one pointer cast — no alignment arithmetic.
    ///
    /// Guard-page offsets use the page size captured at construction so the
    /// layout is always consistent with `new()`, even on platforms where the
    /// OS page size is not 4 KiB (e.g. macOS arm64 = 16 KiB).
    #[inline(always)]
    pub const fn get_context_ptr(&self, index: u32) -> *mut FiberContext {
        let guard_offset = match self.safety {
            SafetyLevel::Safety0 => 0,
            // `>> 5` == `/ 32` for the group index; avoids a division on every call.
            SafetyLevel::Safety1 => ((index as usize >> 5) + 1) * self.page_size,
            SafetyLevel::Safety2 => (index as usize + 1) * self.page_size,
        };

        unsafe {
            #[allow(clippy::cast_ptr_alignment)]
            self.base_ptr
                .add(index as usize * self.slot_size + guard_offset + self.context_end_offset)
                .cast::<FiberContext>()
        }
    }

    /// O(1) pop from this worker's local batch cache (no atomics),
    /// refilling from the shared free list with one CAS per `batch_size`
    /// slots when the cache is empty. Callers with no cache slot of their
    /// own (see [`LocalFreeCache`]'s doc comment) fall back to
    /// [`Self::alloc_context_global`] directly.
    #[inline(always)]
    pub fn alloc_context(&self) -> Option<u32> {
        let worker_id = crate::future_bridge::CURRENT_WORKER_ID.with(core::cell::Cell::get);
        if worker_id >= self.local_caches.len() {
            // Cold: only host threads / callers outside a real scheduler
            // worker ever take this branch (see `LocalFreeCache`'s doc
            // comment) — every real worker's own dispatch thread always has
            // `worker_id < self.local_caches.len()`.
            core::hint::cold_path();
            return self.alloc_context_global();
        }
        // SAFETY: see `LocalFreeCache`'s doc comment — `worker_id` uniquely
        // identifies the one live OS thread that ever touches this slot.
        let cache = unsafe { &mut *self.local_caches[worker_id].get() };
        if cache.len == 0 {
            self.refill_batch(cache);
        }
        if cache.len == 0 {
            // Cold: the shared free list itself was exhausted on refill —
            // true pool exhaustion, not the routine empty-cache case above.
            core::hint::cold_path();
            return None;
        }
        cache.len -= 1;
        // SAFETY: `cache.len` never exceeds `2 * self.batch_size`, itself
        // clamped to `MAX_LOCAL_BATCH` at construction — always in bounds
        // of the fixed-size `slots` array. The compiler can't see that
        // invariant across `refill_batch`/`donate_batch`'s mutations of
        // `cache.len`, so without this hint it inserts a bounds check here.
        unsafe { core::hint::assert_unchecked((cache.len as usize) < cache.slots.len()) };
        let index = cache.slots[cache.len as usize];
        self.claim_context(index);
        Some(index)
    }

    /// Finalizes a slot at the point a fresh fiber claims it: bumps
    /// `generation`, clears `cancel_requested`, and drops any leftover
    /// panic payload from whichever fiber previously occupied this slot.
    ///
    /// All three are deliberately done *here* (on claim) rather than in
    /// [`Self::free_context`] (on release), which is the opposite of
    /// what "ABA-safety must happen right away" might suggest. The
    /// reason: `crate::api::outcome` reads a just-terminated fiber's
    /// `state`/`panic_payload_ptr` (and `generation`, to know it's
    /// still reading the fiber its caller thinks it is) *after* that
    /// caller's `dtact_await` has already returned — necessarily some
    /// time after `free_context` ran on a different thread. Finalizing
    /// at free time raced that read and lost it almost every time under
    /// any real concurrent load (reproduced empirically, not merely
    /// theorized); claiming instead of releasing is the only point that
    /// is actually exclusive to one fiber's setup, so it is the only
    /// point these can safely happen without racing a joiner.
    /// `SpawnBuilder::spawn` and every C-FFI spawn path already
    /// unconditionally overwrite `state` to `Running` on claim
    /// (untouched by this change) — this extends the same "claim
    /// resets, free leaves alone" rule to the other three fields.
    ///
    /// A payload nobody ever reads via `outcome` before this slot is
    /// reused is bounded to leak for at most one recycle cycle, not the
    /// process lifetime — the same bound `free_context`'s doc comment
    /// already accepts for the stale `state` value.
    #[inline(always)]
    fn claim_context(&self, index: u32) {
        let ctx = self.get_context_ptr(index);
        unsafe {
            (*ctx).generation.fetch_add(1, Ordering::AcqRel);
            (*ctx).cancel_requested.store(false, Ordering::Relaxed);
            let payload_ptr =
                core::mem::replace(&mut (*ctx).panic_payload_ptr, core::ptr::null_mut());
            if !payload_ptr.is_null() {
                // Cold: almost every claimed slot's previous occupant
                // finished normally or was never queried via
                // `crate::api::outcome` in the first place — this branch
                // is the rare "an un-retrieved panic payload is still
                // sitting here" case.
                core::hint::cold_path();
                drop(Box::from_raw(
                    payload_ptr.cast::<Box<dyn core::any::Any + Send>>(),
                ));
            }
        }
    }

    /// Returns a context to this worker's local batch cache (no atomics),
    /// donating a batch back to the shared free list with one CAS per
    /// `batch_size` slots when the cache fills up. Callers with no cache
    /// slot of their own fall back to [`Self::free_context_global`]
    /// directly.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn free_context(&self, index: u32) {
        // Deliberately do NOT reset `state`, `generation`, `cancel_requested`,
        // or drop `panic_payload_ptr` here. All four are finalized instead
        // by `claim_context`, at the point this slot is next claimed by
        // `alloc_context` — see that function's doc comment for why: a
        // joiner's `crate::api::outcome` reads a just-terminated fiber's
        // `state`/`panic_payload_ptr`, gated on `generation` matching its
        // handle, strictly after its `dtact_await` on the same handle has
        // already returned — necessarily after whatever this function does,
        // on a different thread. Finalizing eagerly here raced that read
        // and lost it almost every time under real concurrent load
        // (reproduced empirically). Leaving all four alone until the slot
        // is actually claimed by a new fiber is what makes that read safe:
        // nothing overwrites them in between.

        let worker_id = crate::future_bridge::CURRENT_WORKER_ID.with(core::cell::Cell::get);
        if worker_id >= self.local_caches.len() {
            // Cold: see the matching branch in `alloc_context`.
            core::hint::cold_path();
            self.free_context_global(index);
            return;
        }
        // SAFETY: see `LocalFreeCache`'s doc comment.
        let cache = unsafe { &mut *self.local_caches[worker_id].get() };
        if cache.len as usize == 2 * self.batch_size as usize {
            self.donate_batch(cache);
        }
        // SAFETY: see the matching hint in `alloc_context` — `cache.len` is
        // always in bounds of `slots` (donate above resets it to
        // `self.batch_size` whenever it would otherwise reach capacity).
        unsafe { core::hint::assert_unchecked((cache.len as usize) < cache.slots.len()) };
        cache.slots[cache.len as usize] = index;
        cache.len += 1;
    }

    /// Uncached O(1) pop directly from the shared free list, one CAS per
    /// call — today's original `alloc_context` algorithm, unchanged.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn alloc_context_global(&self) -> Option<u32> {
        let mut head = self.free_head.load(Ordering::Acquire);
        loop {
            let index = head as u32;
            let r#gen = (head >> 32) as u32;
            if index == u32::MAX {
                core::hint::cold_path();
                return None;
            }

            let ctx = self.get_context_ptr(index);
            let next = unsafe { (*ctx).next_free.load(Ordering::Relaxed) };

            let new_head = (u64::from(r#gen.wrapping_add(1)) << 32) | u64::from(next);

            // Under loom use strong CAS to avoid spurious-failure branch explosion.
            #[cfg(not(loom))]
            let cas = self.free_head.compare_exchange_weak(
                head,
                new_head,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            #[cfg(loom)]
            let cas = self.free_head.compare_exchange(
                head,
                new_head,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            match cas {
                Ok(_) => {
                    self.claim_context(index);
                    return Some(index);
                }
                Err(latest) => {
                    // Cold: contention on `free_head` is rare by design —
                    // the whole point of `LocalFreeCache` is to keep most
                    // callers off this shared, single-CAS path entirely.
                    core::hint::cold_path();
                    head = latest;
                }
            }
        }
    }

    /// Uncached O(1) push directly onto the shared free list, one CAS per
    /// call — today's original `free_context` free-list step, unchanged
    /// (the state-reset/generation-bump prefix now lives in `free_context`
    /// itself, since it must run regardless of which path handles linking).
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    fn free_context_global(&self, index: u32) {
        let ctx = self.get_context_ptr(index);
        let mut head = self.free_head.load(Ordering::Relaxed);
        loop {
            let current_idx = head as u32;
            let r#gen = (head >> 32) as u32;
            unsafe { (*ctx).next_free.store(current_idx, Ordering::Relaxed) };
            let new_head = (u64::from(r#gen.wrapping_add(1)) << 32) | u64::from(index);
            // Strong CAS under loom eliminates spurious-failure branches.
            #[cfg(not(loom))]
            let cas = self.free_head.compare_exchange_weak(
                head,
                new_head,
                Ordering::Release,
                Ordering::Relaxed,
            );
            #[cfg(loom)]
            let cas = self.free_head.compare_exchange(
                head,
                new_head,
                Ordering::Release,
                Ordering::Relaxed,
            );
            match cas {
                Ok(_) => break,
                Err(h) => {
                    core::hint::cold_path();
                    head = h;
                }
            }
        }
    }

    /// Refills `cache` from the shared free list: walks up to
    /// `self.batch_size` `next_free` links (relaxed reads — safe to read
    /// speculatively before ownership is confirmed, exactly like
    /// `alloc_context_global` already does for one node), then swings
    /// `free_head` past the whole run with **one** CAS. Retries the whole
    /// walk on CAS failure, same shape as the single-node path's retry
    /// loop, just over a bigger unit of work per attempt. ABA safety is
    /// unaffected: `free_head`'s generation still increments by exactly 1
    /// per head mutation, whether that mutation moves 1 node or
    /// `batch_size` nodes. Leaves `cache.len == 0` if the shared list is
    /// already exhausted.
    #[allow(clippy::cast_possible_truncation)]
    fn refill_batch(&self, cache: &mut LocalFreeCache) {
        let want = self.batch_size as usize;
        // SAFETY: `batch_size` is clamped to `MAX_LOCAL_BATCH` at
        // construction (`new_local_caches`), always strictly less than
        // `slots.len() == 2 * MAX_LOCAL_BATCH` — the compiler can't relate
        // a runtime field to the fixed-size array without this hint.
        unsafe { core::hint::assert_unchecked(want < cache.slots.len()) };
        loop {
            let head = self.free_head.load(Ordering::Acquire);
            let mut idx = head as u32;
            if idx == u32::MAX {
                core::hint::cold_path();
                return; // shared list exhausted
            }
            let r#gen = (head >> 32) as u32;

            let mut collected = 0usize;
            while collected < want && idx != u32::MAX {
                cache.slots[collected] = idx;
                collected += 1;
                let ctx = self.get_context_ptr(idx);
                idx = unsafe { (*ctx).next_free.load(Ordering::Relaxed) };
            }
            // `idx` is now the first node NOT claimed (possibly u32::MAX).
            let new_head = (u64::from(r#gen.wrapping_add(1)) << 32) | u64::from(idx);
            #[cfg(not(loom))]
            let cas = self.free_head.compare_exchange_weak(
                head,
                new_head,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            #[cfg(loom)]
            let cas = self.free_head.compare_exchange(
                head,
                new_head,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            if cas.is_ok() {
                #[allow(clippy::cast_possible_truncation)]
                let collected_u32 = collected as u32;
                cache.len = collected_u32;
                return;
            }
            // Cold: list changed under us — fall through to retry the walk
            // from the fresh head. Rare by the same reasoning as above.
            core::hint::cold_path();
        }
    }

    /// Donates `self.batch_size` slots from `cache` back to the shared free
    /// list: links them into a private chain (non-atomic — exclusively
    /// owned until spliced in), then swings `free_head` to the chain's head
    /// with **one** CAS, splicing the previous head onto the chain's tail.
    /// Mirrors `free_context_global`'s single-node push, just for a
    /// pre-built chain. Leaves `self.batch_size` slots resident in `cache`.
    #[allow(clippy::cast_possible_truncation)]
    fn donate_batch(&self, cache: &mut LocalFreeCache) {
        let n = self.batch_size as usize;
        debug_assert!(cache.len as usize >= n);
        let start = cache.len as usize - n;
        // SAFETY: `donate_batch` is only ever called with `cache.len ==
        // 2 * self.batch_size` (the "full" trigger in `free_context`), and
        // `2 * self.batch_size <= slots.len()` by construction
        // (`new_local_caches` clamps `batch_size` to `MAX_LOCAL_BATCH ==
        // slots.len() / 2`) — so `start + n == cache.len <= slots.len()`,
        // and every index touched below is in bounds.
        unsafe { core::hint::assert_unchecked(start + n <= cache.slots.len()) };

        // Link the donated slots into a private chain: nothing else can see
        // these indices yet, so plain (non-atomic) stores are enough.
        for i in start..start + n - 1 {
            let ctx = self.get_context_ptr(cache.slots[i]);
            unsafe {
                (*ctx)
                    .next_free
                    .store(cache.slots[i + 1], Ordering::Relaxed);
            }
        }
        let chain_head = cache.slots[start];
        let tail_ctx = self.get_context_ptr(cache.slots[start + n - 1]);

        let mut head = self.free_head.load(Ordering::Relaxed);
        loop {
            let old_idx = head as u32;
            let r#gen = (head >> 32) as u32;
            unsafe { (*tail_ctx).next_free.store(old_idx, Ordering::Relaxed) };
            let new_head = (u64::from(r#gen.wrapping_add(1)) << 32) | u64::from(chain_head);
            #[cfg(not(loom))]
            let cas = self.free_head.compare_exchange_weak(
                head,
                new_head,
                Ordering::Release,
                Ordering::Relaxed,
            );
            #[cfg(loom)]
            let cas = self.free_head.compare_exchange(
                head,
                new_head,
                Ordering::Release,
                Ordering::Relaxed,
            );
            match cas {
                Ok(_) => break,
                Err(h) => {
                    core::hint::cold_path();
                    head = h;
                }
            }
        }
        #[allow(clippy::cast_possible_truncation)]
        let n_u32 = n as u32;
        cache.len -= n_u32;
    }

    /// Returns the base pointer and layout metadata for direct dispatcher access.
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn get_dispatch_layout(&self) -> (*mut u8, usize, usize, usize) {
        // Use the page size captured at construction so the layout the
        // dispatcher sees always matches the layout the arena was built with.
        let page_size = self.page_size;
        let align = 64;
        let context_sz = (core::mem::size_of::<FiberContext>() + align - 1) & !(align - 1);
        let guard_size = if self.safety == SafetyLevel::Safety0 {
            0
        } else {
            page_size
        };
        // context_offset: byte offset within each slot where FiberContext begins
        let context_offset = self.slot_size - context_sz;
        (self.base_ptr, self.slot_size, guard_size, context_offset)
    }
}

impl Drop for ContextPool {
    #[inline(always)]
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.base_ptr.cast(), self.total_size);
        }
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            VirtualFree(self.base_ptr.cast(), 0, MEM_RELEASE);
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::FiberContext;

    /// Locks in `FiberContext`'s cache-line isolation (see the struct's
    /// doc comment): `state`/`cancel_requested` must stay on their own
    /// line, separate from the thread-local `adaptive_spin_count`/
    /// `spin_failure_count` pair, so a regression here is caught at test
    /// time rather than rediscovered by profiling a false-sharing
    /// regression later. Mirrors `dta_scheduler::layout_tests`'s
    /// `Worker` check.
    #[test]
    fn fiber_context_hot_fields_stay_cache_line_isolated() {
        assert_eq!(core::mem::align_of::<FiberContext>(), 64);

        let state_line = core::mem::offset_of!(FiberContext, state) / 64;
        let cancel_line = core::mem::offset_of!(FiberContext, cancel_requested) / 64;
        assert_eq!(
            state_line, cancel_line,
            "state and cancel_requested must share one cache line"
        );

        let spin_count_line = core::mem::offset_of!(FiberContext, adaptive_spin_count) / 64;
        let spin_failure_line = core::mem::offset_of!(FiberContext, spin_failure_count) / 64;
        assert_eq!(
            spin_count_line, spin_failure_line,
            "adaptive_spin_count and spin_failure_count must share one cache line"
        );
        assert_ne!(
            state_line, spin_count_line,
            "the state/cancel_requested line must not overlap the thread-local spin-budget line"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Sets `CURRENT_WORKER_ID` for the duration of the guard, restoring it
    /// to the "not a worker" sentinel on drop (including on panic/early
    /// return) — required because `cargo test`'s thread pool reuses OS
    /// threads across tests, and a leaked worker-id assignment on a shared
    /// thread-local could make an unrelated later test on that same OS
    /// thread unexpectedly take the cached path against a *different*
    /// `ContextPool` instance.
    struct WorkerIdGuard;

    impl Drop for WorkerIdGuard {
        fn drop(&mut self) {
            crate::future_bridge::CURRENT_WORKER_ID.with(|c| c.set(usize::MAX));
        }
    }

    fn as_worker(id: usize) -> WorkerIdGuard {
        crate::future_bridge::CURRENT_WORKER_ID.with(|c| c.set(id));
        WorkerIdGuard
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn cached_path_never_duplicates_or_loses_slots() {
        let pool = ContextPool::new(64, 8192, SafetyLevel::Safety0, 0, 1).expect("pool init");
        let _guard = as_worker(0);

        let mut seen = HashSet::new();
        for _ in 0..64 {
            let idx = pool.alloc_context().expect("pool has 64 slots");
            assert!(seen.insert(idx), "duplicate index {idx} handed out");
        }
        assert!(
            pool.alloc_context().is_none(),
            "65th alloc on a 64-slot pool must fail"
        );

        for idx in seen {
            pool.free_context(idx);
        }

        // Every slot must be allocable again after freeing them all.
        let mut recovered = HashSet::new();
        for _ in 0..64 {
            let idx = pool
                .alloc_context()
                .expect("all 64 slots should be free again");
            assert!(recovered.insert(idx));
        }
        assert!(pool.alloc_context().is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn cached_path_forces_refill_and_donate_cycles() {
        // capacity=64 -> batch_size = (64/8).clamp(1,32) = 8, so a cache
        // holds up to 2*8=16 before donating and refills 8 at a time —
        // allocating 30 in a row forces multiple refills, and freeing them
        // all back forces at least one donate.
        let pool = ContextPool::new(64, 8192, SafetyLevel::Safety0, 0, 1).expect("pool init");
        let _guard = as_worker(0);

        let mut allocated = Vec::new();
        for _ in 0..30 {
            allocated.push(pool.alloc_context().expect("pool has capacity"));
        }
        let unique: HashSet<_> = allocated.iter().copied().collect();
        assert_eq!(unique.len(), 30, "refill must not hand out duplicates");

        for idx in allocated {
            pool.free_context(idx);
        }

        // No capacity may have been stranded in the cache: the pool must be
        // fully drainable again from scratch.
        let mut recovered = HashSet::new();
        for _ in 0..64 {
            let idx = pool
                .alloc_context()
                .expect("no capacity should be lost across a donate cycle");
            assert!(recovered.insert(idx));
        }
        assert!(pool.alloc_context().is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn distinct_worker_ids_never_share_a_slot_concurrently() {
        let pool = ContextPool::new(64, 8192, SafetyLevel::Safety0, 0, 2).expect("pool init");

        let mut worker0_slots = HashSet::new();
        {
            let _guard = as_worker(0);
            for _ in 0..8 {
                worker0_slots.insert(pool.alloc_context().expect("pool has capacity"));
            }
        }

        let mut worker1_slots = HashSet::new();
        {
            let _guard = as_worker(1);
            for _ in 0..8 {
                worker1_slots.insert(pool.alloc_context().expect("pool has capacity"));
            }
        }

        assert!(
            worker0_slots.is_disjoint(&worker1_slots),
            "two distinct worker-id caches on the same pool handed out overlapping indices"
        );

        // Clean up so the pool's Drop doesn't matter for this test's intent.
        for idx in worker0_slots.into_iter().chain(worker1_slots) {
            pool.free_context(idx);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn uncached_sentinel_path_matches_existing_global_behavior() {
        // No `as_worker` guard: CURRENT_WORKER_ID stays at its default
        // sentinel (usize::MAX), exactly like every pre-existing
        // ContextPool test (none of which go through a real worker
        // dispatch loop) — this must behave exactly as it did before this
        // change, since it takes the untouched `_global` fallback path.
        let pool = ContextPool::new(2, 8192, SafetyLevel::Safety0, 0, 1).expect("pool init");
        let a = pool.alloc_context().expect("first slot");
        let b = pool.alloc_context().expect("second slot");
        assert_ne!(a, b);
        assert!(pool.alloc_context().is_none());
        pool.free_context(a);
        assert!(pool.alloc_context().is_some());
    }
}

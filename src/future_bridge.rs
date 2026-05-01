
// use the context switch module instead of writing it all over.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::pin::Pin;
use std::arch::naked_asm;
use std::cell::RefCell;
use std::future::Future;
use std::task::Context;
use std::task::Poll;

/// A synchronus `bincode::de::read::Reader` implementation that runs entirely inside a Fiber stack.
/// Implicitly yields execution back to the root event loop context if not enough bytes are available.
pub(crate) struct FiberReader<'a, R: futures_io::AsyncRead + Unpin> {
    /// Phanton marker tracking the inner lifetime.
    pub(crate) inner: std::marker::PhantomData<&'a mut R>,
    /// Context pointer to read bounds or swap threads.
    pub(crate) ctx: *mut FiberContext,
}

impl<R: futures_io::AsyncRead + Unpin> crate::de::read::Reader for FiberReader<'_, R> {
    #[inline]
    fn read(
        &mut self,
        bytes: &mut [u8],
    ) -> Result<(), crate::error::DecodeError> {
        let n = bytes.len();
        let mut written = 0;
        let ctx = unsafe { &mut *self.ctx };
        while written < n {
            let buf = unsafe { &mut *ctx.buf_ptr };

            if buf.is_empty() {
                unsafe {
                    ctx.status = FiberStatus::Yielded;
                    switch_context(&raw mut ctx.regs, &raw const ctx.executor_regs);

                    if ctx.status == FiberStatus::Finished {
                        return crate::error::cold_decode_error_unexpected_end(n - written);
                    }
                }
            }

            let buf = unsafe { &mut *ctx.buf_ptr };

            if buf.is_empty() {
                return crate::error::cold_decode_error_unexpected_end(n - written);
            }
            let to_copy = core::cmp::min(n - written, buf.len());
            bytes[written..written + to_copy].copy_from_slice(&buf[0..to_copy]);

            unsafe {
                ctx.buf_ptr = core::ptr::slice_from_raw_parts_mut(
                    buf.as_mut_ptr().add(to_copy),
                    buf.len() - to_copy,
                );
            }

            written += to_copy;
        }
        Ok(())
    }
}

/// Standard entry point that allows asynchronously decoding structs by transparently spawning an executor-integrated Fiber state machine.
///
/// # CRITICAL SAFETY WARNING: THREAD MIGRATION & TLS
///
/// Under standard usage, `BridgeFuture` implements `Send` allowing execution on
/// multi-threaded runtimes (like Tokio's worker pool).
///
/// **DO NOT use Thread-Local Storage (TLS) or `!Send` types (e.g. `Rc`, `RefCell`, `MutexGuard`) inside your `Decode` trait implementations!**
///
/// When the underlying reader returns `Poll::Pending`, the fiber execution stack
/// is suspended. A work-stealing executor may then migrate the suspended `BridgeFuture`
/// to a completely different OS thread.
///
/// Normally, the Rust compiler analyzes `.await` points and prevents futures holding
/// `!Send` data from implementing `Send`. However, because the fiber's yield point is
/// hidden behind the synchronous `bincode::de::read::Reader::read` invocation, the
/// compiler **cannot** analyze the variables held on the fiber's stack.
///
/// If a fiber migrates threads while holding a reference to Thread **A**'s TLS,
/// upon resuming, it will execute on Thread **B** while still illegally referencing
/// Thread **A**'s memory buffer, resulting in **Undefined Behavior**, panicking, or
/// silent memory corruption.
pub(crate) struct AsyncFiberBridge<R: futures_io::AsyncRead + Unpin> {
    /// Underlying futures_io-based `AsyncRead` source.
    pub(crate) reader: R,
}

impl<R: futures_io::AsyncRead + Unpin> AsyncFiberBridge<R> {
    /// Constructs a new asynchronous bridge mapping `futures_io`'s `AsyncRead`.
    #[inline(always)]
    pub(crate) const fn new(reader: R) -> Self {
        Self { reader }
    }

    /// Spawns the parsing process, converting the synchronous Decode traits to a Future.
    #[inline(always)]
    pub(crate) fn run<F, T>(
        self,
        f: F,
    ) -> impl Future<Output = Result<T, crate::error::DecodeError>>
    where
        F: FnOnce(&mut FiberReader<'_, R>) -> Result<T, crate::error::DecodeError>,
    {
        BridgeFuture {
            reader: self.reader,
            f: Some(f),
            ctx: None,
            result: None,
            _marker: core::marker::PhantomData,
        }
    }
}

#[inline(always)]
const unsafe fn dummy_invoke(_: *mut ()) {}

#[inline]
unsafe extern "C" fn fiber_trampoline() {
    unsafe {
        let ctx_ptr = CURRENT_FIBER.with(core::cell::Cell::get);
        let ctx = &mut *ctx_ptr;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (ctx.invoke_closure)(ctx.closure_ptr);
        }));

        ctx.status = if let Err(e) = result {
            ctx.panic_payload = Some(e);
            FiberStatus::Panicked
        } else {
            FiberStatus::Finished
        };

        // Clear the thread-local before switching back — the fiber is done.
        CURRENT_FIBER.with(|c| c.set(core::ptr::null_mut()));
        switch_context(&raw mut ctx.regs, &raw const ctx.executor_regs);
        unreachable!("fiber finished and should not be resumed");
    }
}

/// Helper: set the thread-local and assert thread affinity, then switch.
///
/// # Safety
/// `ctx` must be a valid, pinned `FiberContext` whose stack is live.
#[inline]
unsafe fn resume_fiber(ctx: &mut FiberContext) {
    unsafe {
        // Thread affinity check is completely omitted; fibers can transparently
        // migrate across executor threads since `mmap` and heap regions share
        // the generic virtual address space and executor generic regs snapshot
        // the active execution context per poll dynamically.

        CURRENT_FIBER.with(|c| c.set(core::ptr::from_mut(ctx)));
        switch_context(&raw mut ctx.executor_regs, &raw const ctx.regs);
        // After returning here the fiber has yielded or finished.
        // Clear the thread-local to prevent stale pointer access.
        CURRENT_FIBER.with(|c| c.set(core::ptr::null_mut()));
    }
}

struct BridgeFuture<R, F, T> {
    reader: R,
    f: Option<F>,
    ctx: Option<Box<FiberContext>>,
    result: Option<Result<T, crate::error::DecodeError>>,
    _marker: core::marker::PhantomData<T>,
}

// SAFETY: BridgeFuture is Send+Sync when its components are.
//
// WARNING: As stated on `AsyncFiberBridge`, this circumvents the compiler's
// ability to analyze variables held across yield points. The fiber natively
// guarantees pointer safety for generic hardware execution bounds via the `mmap`
// generic process address space and dynamic `executor_regs` restoring. However,
// `!Send` structures (like `Rc` and `thread_local!`) instantiated inside the
// user's `Decode` stack will blindly be migrated across threads, which is **UB**.
//
// By using this abstraction, the caller is trusted that their `Decode` derivations
// strictly instantiate thread-safe, `Send`-equivalent variables locally.
#[allow(clippy::non_send_fields_in_send_ty)]
unsafe impl<R: Send, F: Send, T: Send> Send for BridgeFuture<R, F, T> {}
unsafe impl<R: Sync, F: Sync, T: Sync> Sync for BridgeFuture<R, F, T> {}

impl<R, F, T> Future for BridgeFuture<R, F, T>
where
    R: futures_io::AsyncRead + Unpin,
    F: FnOnce(&mut FiberReader<'_, R>) -> Result<T, crate::error::DecodeError>,
{
    type Output = Result<T, crate::error::DecodeError>;

    #[allow(clippy::too_many_lines)]
    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Self::Output> {
        // -- Initialise fiber on first poll ----------------------------------
        if self.ctx.is_none() {
            let mut ctx = if let Some(ctx) = CONTEXT_POOL.with(|pool| pool.borrow_mut().pop()) {
                ctx
            } else {
                // Default 2 MiB usable stack via mmap with guard region.
                match GuardedStack::new(2 * 1024 * 1024) {
                    | Ok(stack) => {
                        Box::new(FiberContext {
                            stack,
                            regs: Registers::new(),
                            executor_regs: Registers::new(),
                            status: FiberStatus::Initial,
                            panic_payload: None,
                            trampoline: fiber_trampoline,
                            invoke_closure: dummy_invoke,
                            closure_ptr: core::ptr::null_mut(),
                            result_ptr: core::ptr::null_mut(),
                            reader_ptr: core::ptr::null_mut(),
                            buf_ptr: core::ptr::slice_from_raw_parts_mut(core::ptr::null_mut(), 0),
                            read_buffer: alloc::vec![0; 8192].into_boxed_slice(),
                        })
                    },
                    | Err(e) => return Poll::Ready(Err(e)),
                }
            };

            ctx.status = FiberStatus::Initial;
            ctx.panic_payload = None;
            ctx.result_ptr = core::ptr::null_mut();
            ctx.reader_ptr = core::ptr::null_mut();
            ctx.buf_ptr = core::ptr::slice_from_raw_parts_mut(core::ptr::null_mut(), 0);

            let sp = ctx.stack.top();

            #[cfg(all(target_arch = "x86_64", unix))]
            {
                // x86_64 SysV: RSP must be 16-aligned on entry + 8 (for return address).
                ctx.regs.gprs[0] = sp - 8;
                ctx.regs.gprs[1] = 0; // RBP (frame pointer) = NULL → end of frame chain
                ctx.regs.gprs[7] = fiber_trampoline as *const () as u64; // return address
                // MXCSR default: 0x1F80 = all FP exceptions masked.
                ctx.regs.extended_state[24..28].copy_from_slice(&0x1F80u32.to_ne_bytes());
            }
            #[cfg(all(target_arch = "x86_64", windows))]
            {
                // x86_64 Win64: MUST have 32 bytes of shadow space above the return address.
                // sp - 8 (return addr) - 32 (shadow) = sp - 40.
                ctx.regs.gprs[0] = sp - 40;
                ctx.regs.gprs[1] = 0; // RBP (frame pointer) = NULL
                ctx.regs.gprs[7] = fiber_trampoline as *const () as u64; // return addr
                ctx.regs.gprs[10] = ctx.stack.top(); // TIB StackBase
                ctx.regs.gprs[11] = ctx.stack.bottom(); // TIB StackLimit
                ctx.regs.gprs[12] = ctx.stack.allocation_base(); // TIB DeallocationStack
                ctx.regs.gprs[13] = 0xFFFFFFFFFFFFFFFFu64; // ExceptionList terminator
                ctx.regs.extended_state[24..28].copy_from_slice(&0x1F80u32.to_ne_bytes());
            }
            #[cfg(all(target_arch = "aarch64", unix))]
            {
                ctx.regs.gprs[10] = 0; // x29 (frame pointer) = NULL
                ctx.regs.gprs[11] = fiber_trampoline as u64; // x30 (LR)
                ctx.regs.gprs[12] = sp; // SP
            }
            #[cfg(all(target_arch = "aarch64", windows))]
            {
                ctx.regs.gprs[10] = 0; // x29 (frame pointer) = NULL
                ctx.regs.gprs[11] = fiber_trampoline as u64; // x30 (LR)
                ctx.regs.gprs[12] = sp; // SP
                ctx.regs.gprs[13] = ctx.stack.top(); // TEB StackBase
                ctx.regs.gprs[14] = ctx.stack.bottom(); // TEB StackLimit
                ctx.regs.gprs[15] = ctx.stack.bottom(); // TEB DeallocationStack
            }
            #[cfg(target_arch = "riscv64")]
            {
                ctx.regs.gprs[0] = sp; // SP
                ctx.regs.gprs[1] = 0; // s0/fp (frame pointer) = NULL
                ctx.regs.gprs[13] = fiber_trampoline as u64; // RA
            }

            let this = unsafe { self.as_mut().get_unchecked_mut() };
            this.ctx = Some(ctx);
        }

        let this = unsafe { self.get_unchecked_mut() };
        let this_ptr = core::ptr::from_mut::<Self>(this).cast::<()>();
        let ctx = this.ctx.as_mut().unwrap();

        // Refresh the result pointer on every poll — the BridgeFuture may
        // have been moved by the executor (it implements Unpin implicitly
        // via DerefMut on the Pin, but we use get_unchecked_mut above).
        ctx.result_ptr = (&raw mut this.result).cast::<()>();

        // -- First poll: set up the closure and do the initial switch --------
        if this.f.is_some() && ctx.status == FiberStatus::Initial {
            unsafe fn invoke<R: futures_io::AsyncRead + Unpin, F, T>(data: *mut ())
            where
                F: FnOnce(&mut FiberReader<'_, R>) -> Result<T, crate::error::DecodeError>,
            {
                unsafe {
                    let this = &mut *data.cast::<BridgeFuture<R, F, T>>();
                    let f = this.f.take().unwrap();
                    let ctx_ptr = CURRENT_FIBER.with(core::cell::Cell::get);
                    let mut real_reader: FiberReader<'_, R> = FiberReader {
                        inner: core::marker::PhantomData,
                        ctx: ctx_ptr,
                    };
                    let res = f(&mut real_reader);
                    let rp = (*ctx_ptr)
                        .result_ptr
                        .cast::<Option<Result<T, crate::error::DecodeError>>>();
                    *rp = Some(res);
                }
            }

            ctx.closure_ptr = this_ptr;
            ctx.invoke_closure = invoke::<R, F, T>;

            ctx.status = FiberStatus::Running;

            unsafe {
                resume_fiber(ctx);
            }
        }

        loop {
            let ctx = this.ctx.as_mut().unwrap();

            match ctx.status {
                | FiberStatus::Finished => {
                    // Return context to pool.
                    if let Some(ctx) = this.ctx.take() {
                        CONTEXT_POOL.with(|pool| {
                            let mut p = pool.borrow_mut();
                            if p.len() < MAX_POOLED_CONTEXTS {
                                p.push(ctx);
                            }
                        });
                    }
                    return Poll::Ready(this.result.take().unwrap());
                },
                | FiberStatus::Panicked => {
                    let payload = ctx.panic_payload.take().unwrap();
                    if let Some(ctx) = this.ctx.take() {
                        CONTEXT_POOL.with(|pool| {
                            let mut p = pool.borrow_mut();
                            if p.len() < MAX_POOLED_CONTEXTS {
                                p.push(ctx);
                            }
                        });
                    }
                    std::panic::resume_unwind(payload);
                },
                | FiberStatus::Yielded => {
                    // The fiber needs more data — try to read from the
                    // async reader.

                    // We must borrow ctx and this.reader disjointly.
                    // Since this.reader and this.ctx are distinct fields, we can do:
                    let ctx_read_buf = &mut ctx.read_buffer[..];
                    let poll_res = Pin::new(&mut this.reader).poll_read(cx, ctx_read_buf);
                    match poll_res {
                        | Poll::Ready(Ok(filled)) => {
                            if filled == 0 {
                                // EOF — tell the fiber so it can return an error.
                                ctx.status = FiberStatus::Finished;
                                ctx.buf_ptr = core::ptr::slice_from_raw_parts_mut(
                                    ctx.read_buffer.as_mut_ptr(),
                                    0,
                                );
                                unsafe {
                                    resume_fiber(ctx);
                                }
                                continue;
                            }
                            ctx.status = FiberStatus::Running;
                            ctx.buf_ptr = core::ptr::slice_from_raw_parts_mut(
                                ctx.read_buffer.as_mut_ptr(),
                                filled,
                            );
                            unsafe {
                                resume_fiber(ctx);
                            }
                        },
                        | Poll::Ready(Err(e)) => {
                            if let Some(ctx) = this.ctx.take() {
                                CONTEXT_POOL.with(|pool| {
                                    let mut p = pool.borrow_mut();
                                    if p.len() < MAX_POOLED_CONTEXTS {
                                        p.push(ctx);
                                    }
                                });
                            }
                            return Poll::Ready(crate::error::cold_decode_error_io(e, 1));
                        },
                        | Poll::Pending => return Poll::Pending,
                    }
                },
                | _ => {
                    unreachable!("invalid fiber status in poll loop");
                },
            }
        }
    }
}

// Clean up fiber resources if the future is dropped while the fiber is
// suspended (e.g. the task is cancelled). The GuardedStack and Context
// memories will be successfully unmapped / dropped natively.
impl<R, F, T> Drop for BridgeFuture<R, F, T> {
    #[inline(always)]
    fn drop(&mut self) {
        if let Some(ctx) = self.ctx.take() {
            // Context is dropped instead of pooled to discard dirty internal state.
            drop(ctx);
        }
    }
}

/// A lightweight adapter to use `tokio::io::AsyncRead` with `AsyncFiberBridge`.
#[cfg(all(feature = "tokio", feature = "async-fiber"))]
pub(crate) struct TokioReader<R>(pub(crate) R);

#[cfg(all(feature = "tokio", feature = "async-fiber"))]
impl<R: tokio::io::AsyncRead + Unpin> futures_io::AsyncRead for TokioReader<R> {
    #[inline]
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        let mut read_buf = tokio::io::ReadBuf::new(buf);
        match Pin::new(&mut self.0).poll_read(cx, &mut read_buf) {
            | Poll::Ready(Ok(())) => Poll::Ready(Ok(read_buf.filled().len())),
            | Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            | Poll::Pending => Poll::Pending,
        }
    }
}

//! Multi-producer, single-consumer channel — bounded ([`channel`], with
//! backpressure) and unbounded ([`unbounded_channel`]).
//!
//! The message buffer itself is lock-free (no `std::sync::Mutex`), not
//! just the waiter bookkeeping — see [`Queue`]'s doc for why bounded and
//! unbounded need genuinely different underlying structures to achieve
//! that.

use super::wait_queue::WaitQueue;
use crate::lockfree::{BoundedMpmcQueue, MpmcStack};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll};

/// The two message-buffer shapes a `Shared<T>` can hold.
///
/// - **Bounded**: a fixed-capacity [`BoundedMpmcQueue`] (Vyukov's
///   lock-free ring-buffer algorithm) — genuinely lock-free push/pop, no
///   mutex anywhere on the hot path. This replaced an earlier
///   `Mutex<VecDeque<T>>` that measured multiple times slower than
///   `tokio::sync::mpsc` under contention (`benches/sync_performance.rs`'s
///   `mpsc_multi_producer_throughput`) — the mutex itself, not just the
///   waiter bookkeeping, was the bottleneck.
/// - **Unbounded**: a [`MpmcStack`] (the same lock-free Treiber stack
///   used elsewhere in this crate). A stack alone is LIFO, which would
///   silently reorder messages — wrong for a channel. `try_pop` restores
///   FIFO order the same way [`MpmcStack::drain_all`] already does
///   internally (reverse the LIFO pop order): the *single* receiver (this
///   type isn't `Clone`) keeps a small local `VecDeque` refilled by
///   draining+reversing the whole stack whenever it runs dry, so ordering
///   is exactly as if every message had gone through one shared FIFO,
///   amortized to O(1) per message with zero contention on the drain
///   itself (one atomic pointer swap, not a per-message lock). A bounded
///   ring can't be reused here because "unbounded" means capacity isn't
///   fixed at construction time.
#[repr(align(64))]
enum Queue<T> {
    Bounded(Box<BoundedMpmcQueue<T>>),
    Unbounded(MpmcStack<T>),
}

impl<T> Queue<T> {
    /// Push `value`. Always succeeds for the unbounded variant; for the
    /// bounded variant, hands `value` back if the queue is currently at
    /// capacity.
    #[inline(always)]
    fn try_push(&self, value: T) -> Result<(), T> {
        match self {
            Self::Bounded(q) => q.try_push(value),
            Self::Unbounded(q) => {
                q.push(value);
                Ok(())
            }
        }
    }
}

#[repr(align(64))]
struct Shared<T> {
    queue: Queue<T>,
    sender_count: AtomicUsize,
    receiver_dropped: AtomicBool,
    /// Woken when the queue transitions empty -> nonempty (or the last
    /// sender drops) — always exactly zero or one waiter (`Receiver`
    /// isn't `Clone`), but reuses the same shared queue type as
    /// everything else in this module for consistency.
    recv_wait: WaitQueue,
    /// Woken when the queue transitions full -> not-full (or the
    /// receiver drops) — potentially many waiting senders. Only ever
    /// populated for the bounded variant (the unbounded variant never
    /// rejects a push, so nothing ever registers here), but kept
    /// unconditionally rather than behind an `Option` for the same
    /// "reuse the same shared type" reasoning as `recv_wait`.
    send_wait: WaitQueue,
}

impl<T> Shared<T> {
    #[inline(always)]
    fn is_closed_for_send(&self) -> bool {
        self.receiver_dropped.load(Ordering::Acquire)
    }

    #[inline(always)]
    fn is_closed_for_recv(&self) -> bool {
        self.sender_count.load(Ordering::Acquire) == 0
    }
}

/// Create a bounded channel: [`Sender::send`] waits (without blocking the
/// OS thread) once `capacity` unreceived messages are buffered.
#[must_use]
#[inline]
pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    let shared = Arc::new(Shared {
        queue: Queue::Bounded(Box::new(BoundedMpmcQueue::new(capacity.max(1)))),
        sender_count: AtomicUsize::new(1),
        receiver_dropped: AtomicBool::new(false),
        recv_wait: WaitQueue::new(),
        send_wait: WaitQueue::new(),
    });
    (
        Sender {
            shared: shared.clone(),
        },
        Receiver {
            shared,
            local: RefCell::new(VecDeque::new()),
        },
    )
}

/// Create an unbounded channel: [`UnboundedSender::send`] never waits,
/// buffering as many messages as memory allows.
#[must_use]
#[inline]
pub fn unbounded_channel<T>() -> (UnboundedSender<T>, UnboundedReceiver<T>) {
    let shared = Arc::new(Shared {
        queue: Queue::Unbounded(MpmcStack::new()),
        sender_count: AtomicUsize::new(1),
        receiver_dropped: AtomicBool::new(false),
        recv_wait: WaitQueue::new(),
        send_wait: WaitQueue::new(),
    });
    (
        UnboundedSender {
            inner: Sender {
                shared: shared.clone(),
            },
        },
        UnboundedReceiver {
            inner: Receiver {
                shared,
                local: RefCell::new(VecDeque::new()),
            },
        },
    )
}

/// The sending half of a bounded [`channel`]. Cheaply [`Clone`]-able —
/// every clone counts toward the channel staying open.
#[repr(align(64))]
pub struct Sender<T> {
    shared: Arc<Shared<T>>,
}

impl<T> Clone for Sender<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        self.shared.sender_count.fetch_add(1, Ordering::AcqRel);
        Self {
            shared: self.shared.clone(),
        }
    }
}

impl<T> Drop for Sender<T> {
    #[inline(always)]
    fn drop(&mut self) {
        if self.shared.sender_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Last sender gone — wake the receiver so a pending `.recv()`
            // observes channel closure instead of hanging forever.
            self.shared.recv_wait.wake_all();
        }
    }
}

impl<T> Sender<T> {
    /// Send `value`, waiting if the channel is currently at `capacity`.
    ///
    /// # Errors
    /// Returns `value` back in [`SendError`] if the receiver has been
    /// dropped.
    #[inline(always)]
    pub async fn send(&self, value: T) -> Result<(), SendError<T>> {
        let mut value = Some(value);
        std::future::poll_fn(|cx| self.poll_send(cx, &mut value)).await
    }

    fn poll_send(&self, cx: &Context<'_>, value: &mut Option<T>) -> Poll<Result<(), SendError<T>>> {
        if self.shared.is_closed_for_send() {
            return Poll::Ready(Err(SendError(value.take().expect("value present"))));
        }

        if !self.shared.send_wait.has_waiters() {
            match self
                .shared
                .queue
                .try_push(value.take().expect("value present"))
            {
                Ok(()) => {
                    // Only wake the receiver if it is actually parked waiting for data
                    if self.shared.recv_wait.has_waiters() {
                        self.shared.recv_wait.wake_one();
                    }
                    return Poll::Ready(Ok(()));
                }
                Err(v) => *value = Some(v),
            }
        }

        let token = self.shared.send_wait.register(cx.waker());
        if self.shared.is_closed_for_send() {
            self.shared.send_wait.cancel(token);
            return Poll::Ready(Err(SendError(value.take().expect("value present"))));
        }

        match self
            .shared
            .queue
            .try_push(value.take().expect("value present"))
        {
            Ok(()) => {
                self.shared.send_wait.cancel(token);
                // Slow-path success: check if receiver needs a wake
                if self.shared.recv_wait.has_waiters() {
                    self.shared.recv_wait.wake_one();
                }
                Poll::Ready(Ok(()))
            }
            Err(v) => {
                *value = Some(v);
                Poll::Pending
            }
        }
    }

    /// `true` once the receiver has been dropped — a subsequent
    /// [`send`](Self::send) is guaranteed to fail.
    #[must_use]
    #[inline(always)]
    pub fn is_closed(&self) -> bool {
        self.shared.is_closed_for_send()
    }
}

/// The receiving half of a bounded [`channel`].
#[repr(align(64))]
pub struct Receiver<T> {
    shared: Arc<Shared<T>>,
    /// Local FIFO-order refill buffer for the unbounded variant's
    /// stack-backed queue — see [`Queue`]'s doc. Always empty and unused
    /// for the bounded variant. Exclusive to this `Receiver` (never
    /// touched by any other thread — `Receiver` isn't `Clone`), so a
    /// plain `RefCell` (no atomics needed) is sound even though `recv`
    /// only takes `&mut self` at the outer async-fn level, not down
    /// through `poll_recv`/`try_pop`.
    local: RefCell<VecDeque<T>>,
}

// SAFETY: `local` is only ever touched from inside `try_pop`, itself only
// ever called (transitively) from `recv(&mut self)` — the `&mut`
// exclusive borrow that async fn holds for its whole duration is the
// actual guarantee that no two calls into `local` overlap, even though
// `poll_recv`'s `&self` signature (required by `std::future::poll_fn`)
// means the auto-trait deriver can't see that exclusivity itself; `Sync`
// just needs to be true, and it is, by construction. `RefCell` still
// panics (rather than silently racing) if this invariant is ever somehow
// violated, rather than relying solely on this unsafe impl.
unsafe impl<T: Send> Sync for Receiver<T> {}

impl<T> Drop for Receiver<T> {
    #[inline(always)]
    fn drop(&mut self) {
        self.shared.receiver_dropped.store(true, Ordering::Release);
        self.shared.send_wait.wake_all();
    }
}

impl<T> Receiver<T> {
    /// Receive the next message, waiting if the channel is currently
    /// empty. Returns `None` once every [`Sender`] has been dropped and
    /// the buffer is drained.
    #[inline(always)]
    pub async fn recv(&mut self) -> Option<T> {
        std::future::poll_fn(|cx| self.poll_recv(cx)).await
    }

    #[inline]
    fn poll_recv(&self, cx: &Context<'_>) -> Poll<Option<T>> {
        if let Some(v) = self.try_pop() {
            return Poll::Ready(Some(v));
        }
        if self.shared.is_closed_for_recv() {
            // The last sender could have pushed its final item and then
            // dropped (closing the channel) concurrently with the
            // `try_pop` above finding it momentarily empty — one more
            // drain attempt here catches that item before reporting the
            // channel exhausted, rather than silently losing it.
            return Poll::Ready(self.try_pop());
        }
        let token = self.shared.recv_wait.register(cx.waker());
        if let Some(v) = self.try_pop() {
            self.shared.recv_wait.cancel(token);
            return Poll::Ready(Some(v));
        }
        if self.shared.is_closed_for_recv() {
            let v = self.try_pop();
            self.shared.recv_wait.cancel(token);
            return Poll::Ready(v);
        }
        Poll::Pending
    }

    #[inline]
    fn try_pop(&self) -> Option<T> {
        let v = match &self.shared.queue {
            Queue::Bounded(q) => q.try_pop(),
            Queue::Unbounded(q) => {
                let mut local = self.local.borrow_mut();
                local.pop_front().map_or_else(
                    || {
                        // Zero-allocation refill optimization
                        q.drain_into_vec_deque(&mut local);
                        local.pop_front()
                    },
                    Some,
                )
            }
        };

        // Only wake a blocked sender if an item was actually removed
        // AND there are senders currently parked waiting for a slot.
        if v.is_some() && self.shared.send_wait.has_waiters() {
            self.shared.send_wait.wake_one();
        }
        v
    }
}

/// The sending half of an [`unbounded_channel`]. Cheaply [`Clone`]-able.
#[repr(align(64))]
pub struct UnboundedSender<T> {
    inner: Sender<T>,
}

impl<T> Clone for UnboundedSender<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> UnboundedSender<T> {
    /// Send `value`. Never waits — the unbounded channel has no capacity
    /// limit, so this is a plain (non-`async`) method, matching
    /// `tokio::sync::mpsc::UnboundedSender::send`.
    ///
    /// # Errors
    /// Returns `value` back in [`SendError`] if the receiver has been
    /// dropped.
    #[inline(always)]
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        if self.inner.shared.is_closed_for_send() {
            return Err(SendError(value));
        }
        // Infallible for the unbounded (stack-backed) queue variant.
        self.inner
            .shared
            .queue
            .try_push(value)
            .unwrap_or_else(|_| unreachable!("unbounded queue variant never rejects a push"));
        self.inner.shared.recv_wait.wake_one();
        Ok(())
    }

    /// `true` once the receiver has been dropped.
    #[must_use]
    #[inline(always)]
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
}

/// The receiving half of an [`unbounded_channel`].
#[repr(align(64))]
pub struct UnboundedReceiver<T> {
    inner: Receiver<T>,
}

impl<T> UnboundedReceiver<T> {
    /// Receive the next message, waiting if the channel is currently
    /// empty. Returns `None` once every sender has been dropped and the
    /// buffer is drained.
    #[inline(always)]
    pub async fn recv(&mut self) -> Option<T> {
        self.inner.recv().await
    }
}

/// Error returned when sending on a channel whose receiver has been
/// dropped. Carries the value that couldn't be delivered back to the
/// caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(align(64))]
pub struct SendError<T>(pub T);

impl<T> std::fmt::Display for SendError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("channel closed: receiver dropped")
    }
}

impl<T: std::fmt::Debug> std::error::Error for SendError<T> {}

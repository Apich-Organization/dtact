/// Raw Hardware Timestamp.
///
/// Returns a monotonically increasing cycle count from the CPU.
/// This is a non-serializing instruction designed for maximum performance
/// and minimum pipeline disturbance.
#[inline(always)]
#[must_use]
pub fn rdtsc() -> u64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut cnt: u64;
        core::arch::asm!("mrs {0}, cntvct_el0", out(reg) cnt);
        cnt
    }
    #[cfg(target_arch = "riscv64")]
    unsafe {
        let mut cycles: u64;
        core::arch::asm!("rdcycle {0}", out(reg) cycles);
        cycles
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )))]
    {
        0
    }
}

/// Fast Core ID hint.
///
/// Attempts to retrieve the current Core ID using the fastest available
/// non-serializing hardware instruction (e.g. RDPID on `x86_64`).
#[inline(always)]
#[must_use]
pub fn get_cpu_fast() -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut aux: u32 = 0;
        #[cfg(feature = "hw-acceleration")]
        {
            // RDPID is non-serializing and unprivileged (fast-path).
            // Requires relatively new CPU (Haswell/Broadwell+ or newer).
            core::arch::asm!(
                "rdpid {0:e}",
                out(reg) aux,
                options(nostack, preserves_flags),
            );
        }
        #[cfg(not(feature = "hw-acceleration"))]
        {
            // Fallback to RDTSCP for legacy compatibility.
            core::arch::x86_64::__rdtscp(&mut aux);
        }
        aux
    }
    #[cfg(all(not(target_arch = "x86_64"), target_os = "linux"))]
    unsafe {
        let mut cpu: u32 = 0;
        libc::getcpu(&mut cpu, core::ptr::null_mut());
        cpu
    }
    #[cfg(not(any(target_arch = "x86_64", target_os = "linux")))]
    {
        0
    }
}

/// Ultra-fast tick for local execution.
///
/// A thin wrapper around `rdtsc` used for microsecond-level latency
/// measurements within the scheduler dispatch loop.
#[inline(always)]
#[must_use]
pub fn get_tick() -> u64 {
    rdtsc()
}

/// Atomic Timestamp + Core ID.
///
/// Returns a tuple of (Timestamp, Core ID). If the `hypervisor` feature is
/// enabled, this function uses serializing instructions (LFENCE) to ensure
/// timestamp monotonicity even across VM migrations.
#[inline(always)]
#[must_use]
pub fn get_tick_with_cpu() -> (u64, u32) {
    #[cfg(feature = "hypervisor")]
    {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // Fully serializing version for cloud stability (LFENCE + RDTSC + Core ID)
            core::arch::asm!("lfence", options(nostack, preserves_flags));
            let tsc = core::arch::x86_64::_rdtsc();
            let cpu = get_cpu_fast();
            (tsc, cpu)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            (rdtsc(), get_cpu_fast())
        }
    }
    #[cfg(not(feature = "hypervisor"))]
    {
        // High-performance monotonic path
        (rdtsc(), get_cpu_fast())
    }
}

/// Blocks the current OS thread until notified via `futex_wake`.
///
/// Utilizes the Linux `FUTEX_WAIT` system call for efficient, zero-CPU
/// blocking of host threads awaiting fiber completion.
///
/// # Safety
/// * `addr` must point to a valid `AtomicU8`.
#[cfg(target_os = "linux")]
#[inline(always)]
pub unsafe fn futex_wait(addr: *const core::sync::atomic::AtomicU8, val: u8) {
    unsafe {
        loop {
            let ret = libc::syscall(
                libc::SYS_futex,
                addr,
                libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
                libc::c_int::from(val),
                core::ptr::null::<libc::timespec>(),
            );
            if ret == 0 {
                break;
            }
            let err = *libc::__errno_location();
            if err == libc::EAGAIN || err == libc::EINTR {
                continue;
            }
            break; // Other errors
        }
    }
}

/// Wakes all OS threads currently blocked on the specified address.
///
/// # Safety
/// * `addr` must point to a valid `AtomicU8`.
#[cfg(target_os = "linux")]
#[inline(always)]
pub unsafe fn futex_wake(addr: *const core::sync::atomic::AtomicU8) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            addr,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            i32::MAX,
        );
    }
}

/// Cross-platform fallback for `futex_wait`.
#[cfg(not(target_os = "linux"))]
#[inline(always)]
pub fn futex_wait(_addr: *const core::sync::atomic::AtomicU8, _val: u8) {
    std::thread::yield_now();
}

/// Cross-platform fallback for `futex_wake`.
#[cfg(not(target_os = "linux"))]
#[inline(always)]
pub fn futex_wake(_addr: *const core::sync::atomic::AtomicU8) {}

std::thread_local! {
    static CACHED_TID: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

/// Returns a unique identifier for the current OS thread.
///
/// Caches the thread ID in thread-local storage after the first lookup
/// to avoid repeated system call overhead.
#[inline(always)]
#[must_use]
pub fn get_thread_id() -> u64 {
    CACHED_TID.with(|c| {
        let mut tid = c.get();
        if tid == 0 {
            #[cfg(target_os = "linux")]
            unsafe {
                tid = libc::syscall(libc::SYS_gettid).cast_unsigned();
            }
            #[cfg(target_os = "windows")]
            unsafe {
                tid = windows_sys::Win32::System::Threading::GetCurrentThreadId().cast_unsigned();
            }
            c.set(tid);
        }
        tid
    })
}

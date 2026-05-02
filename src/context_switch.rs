//! # Low-Level Context Switching Assembly
//! 
//! This module contains the architecture-specific assembly trampolines for 
//! saving and restoring fiber execution contexts.
//! 
//! ## Performance Strategy
//! 1. **Hardware Prefetching**: Every switch proactively warms the cache with the
//!    target fiber's stack and register metadata.
//! 2. **Windows ABI Compliance**: Preserves the Thread Information Block (TIB)
//!    stack limits and SEH pointers across switches.
//! 3. **Non-Serializing State**: Minimizes pipeline stalls by using 
//!    non-serializing instructions where possible.

use crate::memory_management::Registers;
use core::arch::naked_asm;

// ============================================================================
// CROSS-THREAD WITH FLOAT
// ============================================================================

/// Switches execution context while preserving floating-point state.
/// 
/// Supports cross-thread migration by saving/restoring the full callee-saved
/// register set and extended SIMD state (FXSAVE/FXRSTOR).
#[cfg(all(target_arch = "x86_64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetcht0 [rsi]", // Prefetch restore context
        "mov rax, [rsi]",   // Get target stack top
        "prefetcht0 [rax]", // Prefetch target stack
        "mov [rdi + 0], rsp",
        "mov [rdi + 8], rbp",
        "mov [rdi + 16], rbx",
        "mov [rdi + 24], r12",
        "mov [rdi + 32], r13",
        "mov [rdi + 40], r14",
        "mov [rdi + 48], r15",
        "fxsave [rdi + 128]",
        "lea rax, [rip + 1f]",
        "mov [rdi + 56], rax",
        "fxrstor [rsi + 128]",
        "mov rsp, [rsi + 0]",
        "mov rbp, [rsi + 8]",
        "mov rbx, [rsi + 16]",
        "mov r12, [rsi + 24]",
        "mov r13, [rsi + 32]",
        "mov r14, [rsi + 40]",
        "mov r15, [rsi + 48]",
        "jmp [rsi + 56]",
        "1: ret"
    );
}

/// Windows-compatible context switch with TIB preservation.
#[cfg(all(target_arch = "x86_64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetcht0 [rdx]", 
        "mov rax, [rdx]",
        "prefetcht0 [rax]",
        "mov [rcx + 0], rsp",
        "mov [rcx + 8], rbp",
        "mov [rcx + 16], rbx",
        "mov [rcx + 24], r12",
        "mov [rcx + 32], r13",
        "mov [rcx + 40], r14",
        "mov [rcx + 48], r15",
        "mov [rcx + 64], rdi",
        "mov [rcx + 72], rsi",
        // Save Windows TIB Stack Metadata
        "mov rax, gs:[0x08]",
        "mov [rcx + 80], rax",
        "mov rax, gs:[0x10]",
        "mov [rcx + 88], rax",
        "mov rax, gs:[0x1478]",
        "mov [rcx + 96], rax",
        "mov rax, gs:[0x00]",
        "mov [rcx + 104], rax",
        "fxsave [rcx + 128]",
        "lea rax, [rip + 1f]",
        "mov [rcx + 56], rax",
        "fxrstor [rdx + 128]",
        // Restore Windows TIB Stack Metadata
        "mov rax, [rdx + 80]",
        "mov gs:[0x08], rax",
        "mov rax, [rdx + 88]",
        "mov gs:[0x10], rax",
        "mov rax, [rdx + 96]",
        "mov gs:[0x1478], rax",
        "mov rax, [rdx + 104]",
        "mov gs:[0x00], rax",
        "mov rsp, [rdx + 0]",
        "mov rbp, [rdx + 8]",
        "mov rbx, [rdx + 16]",
        "mov r12, [rdx + 24]",
        "mov r13, [rdx + 32]",
        "mov r14, [rdx + 40]",
        "mov r15, [rdx + 48]",
        "mov rdi, [rdx + 64]",
        "mov rsi, [rdx + 72]",
        "jmp [rdx + 56]",
        "1: ret"
    );
}

/// AArch64 Unix-compatible context switch with PRFM hints.
#[cfg(all(target_arch = "aarch64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prfm pldl1keep, [x1]", 
        "ldr x9, [x1, 96]",     
        "prfm pldl1keep, [x9]", 
        "stp x19, x20, [x0, 0]",
        "stp x21, x22, [x0, 16]",
        "stp x23, x24, [x0, 32]",
        "stp x25, x26, [x0, 48]",
        "stp x27, x28, [x0, 64]",
        "stp x29, x30, [x0, 80]",
        "mov x9, sp",
        "str x9, [x0, 96]",
        "stp q8, q9, [x0, 128]",
        "stp q10, q11, [x0, 160]",
        "stp q12, q13, [x0, 192]",
        "stp q14, q15, [x0, 224]",
        "ldp x19, x20, [x1, 0]",
        "ldp x21, x22, [x1, 16]",
        "ldp x23, x24, [x1, 32]",
        "ldp x25, x26, [x1, 48]",
        "ldp x27, x28, [x1, 64]",
        "ldp x29, x30, [x1, 80]",
        "ldr x9, [x1, 96]",
        "mov sp, x9",
        "ldp q8, q9, [x1, 128]",
        "ldp q10, q11, [x1, 160]",
        "ldp q12, q13, [x1, 192]",
        "ldp q14, q15, [x1, 224]",
        "ret"
    );
}

/// Windows-compatible AArch64 switch (x18 TEB preservation).
#[cfg(all(target_arch = "aarch64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prfm pldl1keep, [x1]", 
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "stp x19, x20, [x0, 0]",
        "stp x21, x22, [x0, 16]",
        "stp x23, x24, [x0, 32]",
        "stp x25, x26, [x0, 48]",
        "stp x27, x28, [x0, 64]",
        "stp x29, x30, [x0, 80]",
        "mov x9, sp",
        "str x9, [x0, 96]",
        "ldr x9, [x18, #0x08]",
        "str x9, [x0, #104]",
        "ldr x9, [x18, #0x10]",
        "str x9, [x0, #112]",
        "ldr x9, [x18, #0x1478]",
        "str x9, [x0, #120]",
        "stp q8, q9, [x0, 128]",
        "stp q10, q11, [x0, 160]",
        "stp q12, q13, [x0, 192]",
        "stp q14, q15, [x0, 224]",
        "ldp q8, q9, [x1, 128]",
        "ldp q10, q11, [x1, 160]",
        "ldp q12, q13, [x1, 192]",
        "ldp q14, q15, [x1, 224]",
        "ldr x9, [x1, #104]",
        "str x9, [x18, #0x08]",
        "ldr x9, [x1, #112]",
        "str x9, [x18, #0x10]",
        "ldr x9, [x1, #120]",
        "str x9, [x18, #0x1478]",
        "ldp x19, x20, [x1, 0]",
        "ldp x21, x22, [x1, 16]",
        "ldp x23, x24, [x1, 32]",
        "ldp x25, x26, [x1, 48]",
        "ldp x27, x28, [x1, 64]",
        "ldp x29, x30, [x1, 80]",
        "ldr x9, [x1, 96]",
        "mov sp, x9",
        "ret"
    );
}

/// RISC-V 64-bit switch with hardware-level prefetching.
#[cfg(all(target_arch = "riscv64", unix, feature = "hw-acceleration"))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetch.r 0(a1)", 
        "ld a2, 0(a1)",     
        "prefetch.r 0(a2)", 
        "sd sp, 0(a0)",
        "sd s0, 8(a0)",
        "sd s1, 16(a0)",
        "sd s2, 24(a0)",
        "sd s3, 32(a0)",
        "sd s4, 40(a0)",
        "sd s5, 48(a0)",
        "sd s6, 56(a0)",
        "sd s7, 64(a0)",
        "sd s8, 72(a0)",
        "sd s9, 80(a0)",
        "sd s10, 88(a0)",
        "sd s11, 96(a0)",
        "sd ra, 104(a0)",
        "fsd fs0, 128(a0)",
        "fsd fs1, 136(a0)",
        "fsd fs2, 144(a0)",
        "fsd fs3, 152(a0)",
        "fsd fs4, 160(a0)",
        "fsd fs5, 168(a0)",
        "fsd fs6, 176(a0)",
        "fsd fs7, 184(a0)",
        "fsd fs8, 192(a0)",
        "fsd fs9, 200(a0)",
        "fsd fs10, 208(a0)",
        "fsd fs11, 216(a0)",
        "ld sp, 0(a1)",
        "ld s0, 8(a1)",
        "ld s1, 16(a1)",
        "ld s2, 24(a1)",
        "ld s3, 32(a1)",
        "ld s4, 40(a1)",
        "ld s5, 48(a1)",
        "ld s6, 56(a1)",
        "ld s7, 64(a1)",
        "ld s8, 72(a1)",
        "ld s9, 80(a1)",
        "ld s10, 88(a1)",
        "ld s11, 96(a1)",
        "ld ra, 104(a1)",
        "fld fs0, 128(a1)",
        "fld fs1, 136(a1)",
        "fld fs2, 144(a1)",
        "fld fs3, 152(a1)",
        "fld fs4, 160(a1)",
        "fld fs5, 168(a1)",
        "fld fs6, 176(a1)",
        "fld fs7, 184(a1)",
        "fld fs8, 192(a1)",
        "fld fs9, 200(a1)",
        "fld fs10, 208(a1)",
        "fld fs11, 216(a1)",
        "ret"
    );
}

#[cfg(all(target_arch = "riscv64", unix, not(feature = "hw-acceleration")))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "sd sp, 0(a0)",
        "sd s0, 8(a0)",
        "sd s1, 16(a0)",
        "sd s2, 24(a0)",
        "sd s3, 32(a0)",
        "sd s4, 40(a0)",
        "sd s5, 48(a0)",
        "sd s6, 56(a0)",
        "sd s7, 64(a0)",
        "sd s8, 72(a0)",
        "sd s9, 80(a0)",
        "sd s10, 88(a0)",
        "sd s11, 96(a0)",
        "sd ra, 104(a0)",
        "fsd fs0, 128(a0)",
        "fsd fs1, 136(a0)",
        "fsd fs2, 144(a0)",
        "fsd fs3, 152(a0)",
        "fsd fs4, 160(a0)",
        "fsd fs5, 168(a0)",
        "fsd fs6, 176(a0)",
        "fsd fs7, 184(a0)",
        "fsd fs8, 192(a0)",
        "fsd fs9, 200(a0)",
        "fsd fs10, 208(a0)",
        "fsd fs11, 216(a0)",
        "ld sp, 0(a1)",
        "ld s0, 8(a1)",
        "ld s1, 16(a1)",
        "ld s2, 24(a1)",
        "ld s3, 32(a1)",
        "ld s4, 40(a1)",
        "ld s5, 48(a1)",
        "ld s6, 56(a1)",
        "ld s7, 64(a1)",
        "ld s8, 72(a1)",
        "ld s9, 80(a1)",
        "ld s10, 88(a1)",
        "ld s11, 96(a1)",
        "ld ra, 104(a1)",
        "fld fs0, 128(a1)",
        "fld fs1, 136(a1)",
        "fld fs2, 144(a1)",
        "fld fs3, 152(a1)",
        "fld fs4, 160(a1)",
        "fld fs5, 168(a1)",
        "fld fs6, 176(a1)",
        "fld fs7, 184(a1)",
        "fld fs8, 192(a1)",
        "fld fs9, 200(a1)",
        "fld fs10, 208(a1)",
        "fld fs11, 216(a1)",
        "ret"
    );
}

// ============================================================================
// CROSS-THREAD NO FLOAT
// ============================================================================

/// Switches execution context without preserving floating-point state.
/// 
/// Optimized for non-numerical tasks, significantly reducing the memory
/// footprint of each switch by ignoring the extended SIMD context.
#[cfg(all(target_arch = "x86_64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "mov [rdi + 0], rsp",
        "mov [rdi + 8], rbp",
        "mov [rdi + 16], rbx",
        "mov [rdi + 24], r12",
        "mov [rdi + 32], r13",
        "mov [rdi + 40], r14",
        "mov [rdi + 48], r15",
        "lea rax, [rip + 1f]",
        "mov [rdi + 56], rax",
        "mov rsp, [rsi + 0]",
        "mov rbp, [rsi + 8]",
        "mov rbx, [rsi + 16]",
        "mov r12, [rsi + 24]",
        "mov r13, [rsi + 32]",
        "mov r14, [rsi + 40]",
        "mov r15, [rsi + 48]",
        "jmp [rsi + 56]",
        "1: ret"
    );
}

#[cfg(all(target_arch = "x86_64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "mov [rcx + 0], rsp",
        "mov [rcx + 8], rbp",
        "mov [rcx + 16], rbx",
        "mov [rcx + 24], r12",
        "mov [rcx + 32], r13",
        "mov [rcx + 40], r14",
        "mov [rcx + 48], r15",
        "mov [rcx + 64], rdi",
        "mov [rcx + 72], rsi",
        "mov rax, gs:[0x08]",
        "mov [rcx + 80], rax",
        "mov rax, gs:[0x10]",
        "mov [rcx + 88], rax",
        "mov rax, gs:[0x1478]",
        "mov [rcx + 96], rax",
        "mov rax, gs:[0x00]",
        "mov [rcx + 104], rax",
        "lea rax, [rip + 1f]",
        "mov [rcx + 56], rax",
        "mov rax, [rdx + 80]",
        "mov gs:[0x08], rax",
        "mov rax, [rdx + 88]",
        "mov gs:[0x10], rax",
        "mov rax, [rdx + 96]",
        "mov gs:[0x1478], rax",
        "mov rax, [rdx + 104]",
        "mov gs:[0x00], rax",
        "mov rsp, [rdx + 0]",
        "mov rbp, [rdx + 8]",
        "mov rbx, [rdx + 16]",
        "mov r12, [rdx + 24]",
        "mov r13, [rdx + 32]",
        "mov r14, [rdx + 40]",
        "mov r15, [rdx + 48]",
        "mov rdi, [rdx + 64]",
        "mov rsi, [rdx + 72]",
        "jmp [rdx + 56]",
        "1: ret"
    );
}

#[cfg(all(target_arch = "aarch64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "stp x19, x20, [x0, 0]",
        "stp x21, x22, [x0, 16]",
        "stp x23, x24, [x0, 32]",
        "stp x25, x26, [x0, 48]",
        "stp x27, x28, [x0, 64]",
        "stp x29, x30, [x0, 80]",
        "mov x9, sp",
        "str x9, [x0, 96]",
        "ldp x19, x20, [x1, 0]",
        "ldp x21, x22, [x1, 16]",
        "ldp x23, x24, [x1, 32]",
        "ldp x25, x26, [x1, 48]",
        "ldp x27, x28, [x1, 64]",
        "ldp x29, x30, [x1, 80]",
        "ldr x9, [x1, 96]",
        "mov sp, x9",
        "ret"
    );
}

#[cfg(all(target_arch = "aarch64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "stp x19, x20, [x0, 0]",
        "stp x21, x22, [x0, 16]",
        "stp x23, x24, [x0, 32]",
        "stp x25, x26, [x0, 48]",
        "stp x27, x28, [x0, 64]",
        "stp x29, x30, [x0, 80]",
        "mov x9, sp",
        "str x9, [x0, 96]",
        "ldr x9, [x18, #0x08]",
        "str x9, [x0, #104]",
        "ldr x9, [x18, #0x10]",
        "str x9, [x0, #112]",
        "ldr x9, [x18, #0x1478]",
        "str x9, [x0, #120]",
        "ldr x9, [x1, #104]",
        "str x9, [x18, #0x08]",
        "ldr x9, [x1, #112]",
        "str x9, [x18, #0x10]",
        "ldr x9, [x1, #120]",
        "str x9, [x18, #0x1478]",
        "ldp x19, x20, [x1, 0]",
        "ldp x21, x22, [x1, 16]",
        "ldp x23, x24, [x1, 32]",
        "ldp x25, x26, [x1, 48]",
        "ldp x27, x28, [x1, 64]",
        "ldp x29, x30, [x1, 80]",
        "ldr x9, [x1, 96]",
        "mov sp, x9",
        "ret"
    );
}

#[cfg(all(target_arch = "riscv64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "sd sp, 0(a0)",
        "sd s0, 8(a0)",
        "sd s1, 16(a0)",
        "sd s2, 24(a0)",
        "sd s3, 32(a0)",
        "sd s4, 40(a0)",
        "sd s5, 48(a0)",
        "sd s6, 56(a0)",
        "sd s7, 64(a0)",
        "sd s8, 72(a0)",
        "sd s9, 80(a0)",
        "sd s10, 88(a0)",
        "sd s11, 96(a0)",
        "sd ra, 104(a0)",
        "ld sp, 0(a1)",
        "ld s0, 8(a1)",
        "ld s1, 16(a1)",
        "ld s2, 24(a1)",
        "ld s3, 32(a1)",
        "ld s4, 40(a1)",
        "ld s5, 48(a1)",
        "ld s6, 56(a1)",
        "ld s7, 64(a1)",
        "ld s8, 72(a1)",
        "ld s9, 80(a1)",
        "ld s10, 88(a1)",
        "ld s11, 96(a1)",
        "ld ra, 104(a1)",
        "ret"
    );
}

// ============================================================================
// SAME-THREAD WITH FLOAT
// ============================================================================

/// Lightweight context switch for fibers pinned to the current thread.
/// 
/// Skips the preservation of OS-specific TIB/TEB metadata, assuming the
/// target fiber will always execute on the same physical host thread.
#[cfg(all(target_arch = "x86_64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "mov [rdi + 0], rsp",
        "mov [rdi + 8], rbp",
        "mov [rdi + 16], rbx",
        "mov [rdi + 24], r12",
        "mov [rdi + 32], r13",
        "mov [rdi + 40], r14",
        "mov [rdi + 48], r15",
        "fxsave [rdi + 128]",
        "lea rax, [rip + 1f]",
        "mov [rdi + 56], rax",
        "fxrstor [rsi + 128]",
        "mov rsp, [rsi + 0]",
        "mov rbp, [rsi + 8]",
        "mov rbx, [rsi + 16]",
        "mov r12, [rsi + 24]",
        "mov r13, [rsi + 32]",
        "mov r14, [rsi + 40]",
        "mov r15, [rsi + 48]",
        "jmp [rsi + 56]",
        "1: ret"
    );
}

#[cfg(all(target_arch = "x86_64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "mov [rcx + 0], rsp",
        "mov [rcx + 8], rbp",
        "mov [rcx + 16], rbx",
        "mov [rcx + 24], r12",
        "mov [rcx + 32], r13",
        "mov [rcx + 40], r14",
        "mov [rcx + 48], r15",
        "mov [rcx + 64], rdi",
        "mov [rcx + 72], rsi",
        "fxsave [rcx + 128]",
        "lea rax, [rip + 1f]",
        "mov [rcx + 56], rax",
        "fxrstor [rdx + 128]",
        "mov rsp, [rdx + 0]",
        "mov rbp, [rdx + 8]",
        "mov rbx, [rdx + 16]",
        "mov r12, [rdx + 24]",
        "mov r13, [rdx + 32]",
        "mov r14, [rdx + 40]",
        "mov r15, [rdx + 48]",
        "mov rdi, [rdx + 64]",
        "mov rsi, [rdx + 72]",
        "jmp [rdx + 56]",
        "1: ret"
    );
}

#[cfg(all(target_arch = "aarch64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "stp x19, x20, [x0, 0]",
        "stp x21, x22, [x0, 16]",
        "stp x23, x24, [x0, 32]",
        "stp x25, x26, [x0, 48]",
        "stp x27, x28, [x0, 64]",
        "stp x29, x30, [x0, 80]",
        "mov x9, sp",
        "str x9, [x0, 96]",
        "stp q8, q9, [x0, 128]",
        "stp q10, q11, [x0, 160]",
        "stp q12, q13, [x0, 192]",
        "stp q14, q15, [x0, 224]",
        "ldp x19, x20, [x1, 0]",
        "ldp x21, x22, [x1, 16]",
        "ldp x23, x24, [x1, 32]",
        "ldp x25, x26, [x1, 48]",
        "ldp x27, x28, [x1, 64]",
        "ldp x29, x30, [x1, 80]",
        "ldr x9, [x1, 96]",
        "mov sp, x9",
        "ldp q8, q9, [x1, 128]",
        "ldp q10, q11, [x1, 160]",
        "ldp q12, q13, [x1, 192]",
        "ldp q14, q15, [x1, 224]",
        "ret"
    );
}

#[cfg(all(target_arch = "aarch64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "stp x19, x20, [x0, 0]",
        "stp x21, x22, [x0, 16]",
        "stp x23, x24, [x0, 32]",
        "stp x25, x26, [x0, 48]",
        "stp x27, x28, [x0, 64]",
        "stp x29, x30, [x0, 80]",
        "mov x9, sp",
        "str x9, [x0, 96]",
        "stp q8, q9, [x0, 128]",
        "stp q10, q11, [x0, 160]",
        "stp q12, q13, [x0, 192]",
        "stp q14, q15, [x0, 224]",
        "ldp q8, q9, [x1, 128]",
        "ldp q10, q11, [x1, 160]",
        "ldp q12, q13, [x1, 192]",
        "ldp q14, q15, [x1, 224]",
        "ldp x19, x20, [x1, 0]",
        "ldp x21, x22, [x1, 16]",
        "ldp x23, x24, [x1, 32]",
        "ldp x25, x26, [x1, 48]",
        "ldp x27, x28, [x1, 64]",
        "ldp x29, x30, [x1, 80]",
        "ldr x9, [x1, 96]",
        "mov sp, x9",
        "ret"
    );
}

#[cfg(all(target_arch = "riscv64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "sd sp, 0(a0)",
        "sd s0, 8(a0)",
        "sd s1, 16(a0)",
        "sd s2, 24(a0)",
        "sd s3, 32(a0)",
        "sd s4, 40(a0)",
        "sd s5, 48(a0)",
        "sd s6, 56(a0)",
        "sd s7, 64(a0)",
        "sd s8, 72(a0)",
        "sd s9, 80(a0)",
        "sd s10, 88(a0)",
        "sd s11, 96(a0)",
        "sd ra, 104(a0)",
        "fsd fs0, 128(a0)",
        "fsd fs1, 136(a0)",
        "fsd fs2, 144(a0)",
        "fsd fs3, 152(a0)",
        "fsd fs4, 160(a0)",
        "fsd fs5, 168(a0)",
        "fsd fs6, 176(a0)",
        "fsd fs7, 184(a0)",
        "fsd fs8, 192(a0)",
        "fsd fs9, 200(a0)",
        "fsd fs10, 208(a0)",
        "fsd fs11, 216(a0)",
        "ld sp, 0(a1)",
        "ld s0, 8(a1)",
        "ld s1, 16(a1)",
        "ld s2, 24(a1)",
        "ld s3, 32(a1)",
        "ld s4, 40(a1)",
        "ld s5, 48(a1)",
        "ld s6, 56(a1)",
        "ld s7, 64(a1)",
        "ld s8, 72(a1)",
        "ld s9, 80(a1)",
        "ld s10, 88(a1)",
        "ld s11, 96(a1)",
        "ld ra, 104(a1)",
        "fld fs0, 128(a1)",
        "fld fs1, 136(a1)",
        "fld fs2, 144(a1)",
        "fld fs3, 152(a1)",
        "fld fs4, 160(a1)",
        "fld fs5, 168(a1)",
        "fld fs6, 176(a1)",
        "fld fs7, 184(a1)",
        "fld fs8, 192(a1)",
        "fld fs9, 200(a1)",
        "fld fs10, 208(a1)",
        "fld fs11, 216(a1)",
        "ret"
    );
}

// ============================================================================
// SAME-THREAD NO FLOAT
// ============================================================================

/// The fastest possible context switch: same-thread and no floating-point.
/// 
/// Utilizes aggressive hardware prefetching (`prefetcht0` / `prfm`) to 
/// eliminate memory stalls during local fiber handoffs.
#[cfg(all(target_arch = "x86_64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetcht0 [rsi]", 
        "mov rax, [rsi]",
        "prefetcht0 [rax]",
        "mov [rdi + 0], rsp",
        "mov [rdi + 8], rbp",
        "mov [rdi + 16], rbx",
        "mov [rdi + 24], r12",
        "mov [rdi + 32], r13",
        "mov [rdi + 40], r14",
        "mov [rdi + 48], r15",
        "lea rax, [rip + 1f]",
        "mov [rdi + 56], rax",
        "mov rsp, [rsi + 0]",
        "mov rbp, [rsi + 8]",
        "mov rbx, [rsi + 16]",
        "mov r12, [rsi + 24]",
        "mov r13, [rsi + 32]",
        "mov r14, [rsi + 40]",
        "mov r15, [rsi + 48]",
        "jmp [rsi + 56]",
        "1: ret"
    );
}

#[cfg(all(target_arch = "x86_64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetcht0 [rdx]", 
        "mov rax, [rdx]",
        "prefetcht0 [rax]",
        "mov [rcx + 0], rsp",
        "mov [rcx + 8], rbp",
        "mov [rcx + 16], rbx",
        "mov [rcx + 24], r12",
        "mov [rcx + 32], r13",
        "mov [rcx + 40], r14",
        "mov [rcx + 48], r15",
        "mov [rcx + 64], rdi",
        "mov [rcx + 72], rsi",
        "lea rax, [rip + 1f]",
        "mov [rcx + 56], rax",
        "mov rsp, [rdx + 0]",
        "mov rbp, [rdx + 8]",
        "mov rbx, [rdx + 16]",
        "mov r12, [rdx + 24]",
        "mov r13, [rdx + 32]",
        "mov r14, [rdx + 40]",
        "mov r15, [rdx + 48]",
        "mov rdi, [rdx + 64]",
        "mov rsi, [rdx + 72]",
        "jmp [rdx + 56]",
        "1: ret"
    );
}

#[cfg(all(target_arch = "aarch64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prfm pldl1keep, [x1]", 
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "stp x19, x20, [x0, 0]",
        "stp x21, x22, [x0, 16]",
        "stp x23, x24, [x0, 32]",
        "stp x25, x26, [x0, 48]",
        "stp x27, x28, [x0, 64]",
        "stp x29, x30, [x0, 80]",
        "mov x9, sp",
        "str x9, [x0, 96]",
        "ldp x19, x20, [x1, 0]",
        "ldp x21, x22, [x1, 16]",
        "ldp x23, x24, [x1, 32]",
        "ldp x25, x26, [x1, 48]",
        "ldp x27, x28, [x1, 64]",
        "ldp x29, x30, [x1, 80]",
        "ldr x9, [x1, 96]",
        "mov sp, x9",
        "ret"
    );
}

#[cfg(all(target_arch = "aarch64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prfm pldl1keep, [x1]", 
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "stp x19, x20, [x0, 0]",
        "stp x21, x22, [x0, 16]",
        "stp x23, x24, [x0, 32]",
        "stp x25, x26, [x0, 48]",
        "stp x27, x28, [x0, 64]",
        "stp x29, x30, [x0, 80]",
        "mov x9, sp",
        "str x9, [x0, 96]",
        "ldp x19, x20, [x1, 0]",
        "ldp x21, x22, [x1, 16]",
        "ldp x23, x24, [x1, 32]",
        "ldp x25, x26, [x1, 48]",
        "ldp x27, x28, [x1, 64]",
        "ldp x29, x30, [x1, 80]",
        "ldr x9, [x1, 96]",
        "mov sp, x9",
        "ret"
    );
}

#[cfg(all(target_arch = "riscv64", unix, feature = "hw-acceleration"))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetch.r 0(a1)", 
        "ld a2, 0(a1)",     
        "prefetch.r 0(a2)", 
        "sd sp, 0(a0)",
        "sd s0, 8(a0)",
        "sd s1, 16(a0)",
        "sd s2, 24(a0)",
        "sd s3, 32(a0)",
        "sd s4, 40(a0)",
        "sd s5, 48(a0)",
        "sd s6, 56(a0)",
        "sd s7, 64(a0)",
        "sd s8, 72(a0)",
        "sd s9, 80(a0)",
        "sd s10, 88(a0)",
        "sd s11, 96(a0)",
        "sd ra, 104(a0)",
        "ld sp, 0(a1)",
        "ld s0, 8(a1)",
        "ld s1, 16(a1)",
        "ld s2, 24(a1)",
        "ld s3, 32(a1)",
        "ld s4, 40(a1)",
        "ld s5, 48(a1)",
        "ld s6, 56(a1)",
        "ld s7, 64(a1)",
        "ld s8, 72(a1)",
        "ld s9, 80(a1)",
        "ld s10, 88(a1)",
        "ld s11, 96(a1)",
        "ld ra, 104(a1)",
        "ret"
    );
}

#[cfg(all(target_arch = "riscv64", unix, not(feature = "hw-acceleration")))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "sd sp, 0(a0)",
        "sd s0, 8(a0)",
        "sd s1, 16(a0)",
        "sd s2, 24(a0)",
        "sd s3, 32(a0)",
        "sd s4, 40(a0)",
        "sd s5, 48(a0)",
        "sd s6, 56(a0)",
        "sd s7, 64(a0)",
        "sd s8, 72(a0)",
        "sd s9, 80(a0)",
        "sd s10, 88(a0)",
        "sd s11, 96(a0)",
        "sd ra, 104(a0)",
        "ld sp, 0(a1)",
        "ld s0, 8(a1)",
        "ld s1, 16(a1)",
        "ld s2, 24(a1)",
        "ld s3, 32(a1)",
        "ld s4, 40(a1)",
        "ld s5, 48(a1)",
        "ld s6, 56(a1)",
        "ld s7, 64(a1)",
        "ld s8, 72(a1)",
        "ld s9, 80(a1)",
        "ld s10, 88(a1)",
        "ld s11, 96(a1)",
        "ld ra, 104(a1)",
        "ret"
    );
}

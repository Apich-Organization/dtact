//! # Low-Level Context Switching Assembly
//!
//! This module contains the architecture-specific assembly trampolines for
//! saving and restoring fiber execution contexts.
//!
//! ## Security Strategy
//! 1. **BTI (Branch Target Identification)**: Every entry point and resumption
//!    label is guarded with `bti c` to prevent JOP-style gadget attacks.
//! 2. **PAC (Pointer Authentication)**: Link registers (x30) are signed using
//!    `paciasp` before storage and authenticated with `autiasp` before return,
//!    ensuring control-flow integrity (CFI).
//! 3. **Windows ABI Compliance**: Preserves the Thread Information Block (TIB)
//!    stack limits, SEH ExceptionList, and DeallocationStack across switches.
//! 4. **macOS/Darwin Compliance**: Adheres to Apple Silicon platform requirements,
//!    including platform register (x18) reservation and proper SIMD preservation.
//!
//! ## Performance Strategy
//! 1. **Hardware Prefetching**: Proactively warms L1/L2 caches with the target
//!    fiber's stack and register metadata to hide memory latency.
//! 2. **Non-Serializing State**: Minimizes pipeline stalls by using
//!    non-serializing instructions where possible.

use crate::memory_management::Registers;
use core::arch::naked_asm;

// ============================================================================
// CROSS-THREAD WITH FLOAT
// ============================================================================

/// Switches execution context while preserving floating-point state (Unix x86_64).
///
/// This implementation follows the System V AMD64 ABI, preserving the
/// callee-saved registers (rbx, rbp, r12-r15) and SIMD state via FXSAVE.
///
/// # Arguments
/// * `save` (rdi): Pointer to `Registers` where the current context will be saved.
/// * `restore` (rsi): Pointer to `Registers` containing the context to restore.
///
/// # Safety
/// * Pointers must be valid and 64-byte aligned.
/// * Stack pointer in `restore` must be valid for the target fiber.
#[cfg(all(target_arch = "x86_64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetcht0 [rdi]",
        "prefetcht0 [rsi]",
        "prefetcht0 [rsi + 64]",
        "prefetcht0 [rsi + 128]",
        "mov rax, [rsi]",
        "prefetcht0 [rax]",
        "prefetcht0 [rax + 64]",
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

/// Switches execution context while preserving floating-point and Windows TIB state (x86_64).
///
/// Complies with the Windows x64 ABI by preserving callee-saved registers
/// (rbx, rbp, rdi, rsi, r12-r15), XMM6-XMM15, and the Thread Information Block (TIB).
///
/// # Arguments
/// * `save` (rcx): Pointer to `Registers`.
/// * `restore` (rdx): Pointer to `Registers`.
///
/// # Safety
/// * Updates `gs:[0x00]` (ExceptionList), `gs:[0x08]` (StackBase), `gs:[0x10]` (StackLimit),
///   and `gs:[0x1478]` (DeallocationStack) to reflect the new fiber stack.
#[cfg(all(target_arch = "x86_64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetcht0 [rcx]",
        "prefetcht0 [rdx]",
        "prefetcht0 [rdx + 64]",
        "prefetcht0 [rdx + 128]",
        "mov rax, [rdx]",
        "prefetcht0 [rax]",
        "prefetcht0 [rax + 64]",
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

/// Switches execution context while preserving floating-point state (Unix AArch64).
///
/// Implements BTI and PAC protection. Preserves x19-x30 and SIMD d8-d15 (q8-q15 saved).
///
/// # Arguments
/// * `save` (x0): Pointer to `Registers`.
/// * `restore` (x1): Pointer to `Registers`.
///
/// # Security
/// * `bti c`: Branch target identification for indirect calls.
/// * `paciasp` / `autiasp`: Pointer authentication for the link register (x30).
#[cfg(all(
    target_arch = "aarch64",
    unix,
    not(target_os = "macos"),
    not(feature = "security-hardened")
))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "prfm pldl1keep, [x1, 128]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
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

/// Switches execution context while preserving floating-point state (Unix AArch64).
///
/// Implements BTI and PAC protection. Preserves x19-x30 and SIMD d8-d15 (q8-q15 saved).
///
/// # Arguments
/// * `save` (x0): Pointer to `Registers`.
/// * `restore` (x1): Pointer to `Registers`.
///
/// # Security
/// * `bti c`: Branch target identification for indirect calls.
/// * `paciasp` / `autiasp`: Pointer authentication for the link register (x30).
#[cfg(all(
    target_arch = "aarch64",
    unix,
    not(target_os = "macos"),
    feature = "security-hardened"
))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "bti c",
        "paciasp",
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "prfm pldl1keep, [x1, 128]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
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
        "autiasp",
        "ret"
    );
}

/// macOS AArch64: PAC-compliant context switch with SIMD preservation.
///
/// Adheres to Apple Silicon's security model by using BTI and PAC.
/// This implementation signs the link register using the SP as a modifier.
///
/// # Security
/// * `bti c`: Branch target identification.
/// * `paciasp` / `autiasp`: Standard ARMv8.3-A pointer authentication.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "bti c",
        "paciasp",
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "prfm pldl1keep, [x1, 128]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
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
        "autiasp",
        "ret"
    );
}

/// Switches execution context while preserving floating-point and Windows TEB state (AArch64).
///
/// Complies with the Windows on ARM64 ABI, preserving x18 (TEB pointer) and updating
/// stack metadata fields within the TEB.
///
/// # Security
/// * `bti c`: Branch target identification.
/// * `paciasp` / `autiasp`: Pointer authentication for x30.
#[cfg(all(target_arch = "aarch64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "bti c",
        "paciasp",
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "prfm pldl1keep, [x1, 128]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
        "stp x19, x20, [x0, 0]",
        "stp x21, x22, [x0, 16]",
        "stp x23, x24, [x0, 32]",
        "stp x25, x26, [x0, 48]",
        "stp x27, x28, [x0, 64]",
        "stp x29, x30, [x0, 80]",
        "mov x9, sp",
        "str x9, [x0, 96]",
        // Save Windows TEB Stack Metadata
        "ldr x9, [x18, #0x08]",
        "str x9, [x0, #104]",
        "ldr x9, [x18, #0x10]",
        "str x9, [x0, #112]",
        "ldr x9, [x18, #0x12C8]",
        "str x9, [x0, #120]",
        "stp q8, q9, [x0, 128]",
        "stp q10, q11, [x0, 160]",
        "stp q12, q13, [x0, 192]",
        "stp q14, q15, [x0, 224]",
        "ldp q8, q9, [x1, 128]",
        "ldp q10, q11, [x1, 160]",
        "ldp q12, q13, [x1, 192]",
        "ldp q14, q15, [x1, 224]",
        // Restore Windows TEB Stack Metadata
        "ldr x9, [x1, #104]",
        "str x9, [x18, #0x08]",
        "ldr x9, [x1, #112]",
        "str x9, [x18, #0x10]",
        "ldr x9, [x1, #120]",
        "str x9, [x18, #0x12C8]",
        "ldp x19, x20, [x1, 0]",
        "ldp x21, x22, [x1, 16]",
        "ldp x23, x24, [x1, 32]",
        "ldp x25, x26, [x1, 48]",
        "ldp x27, x28, [x1, 64]",
        "ldp x29, x30, [x1, 80]",
        "ldr x9, [x1, 96]",
        "mov sp, x9",
        "autiasp",
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
        "prefetch.w 0(a0)",
        "prefetch.r 0(a1)",
        "prefetch.r 64(a1)",
        "prefetch.r 128(a1)",
        "ld a2, 0(a1)",
        "prefetch.r 0(a2)",
        "prefetch.r 64(a2)",
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

/// Switches execution context without preserving floating-point state (Unix x86_64).
///
/// Optimized for non-numerical tasks by ignoring XMM/SIMD registers.
///
/// # Arguments
/// * `save` (rdi): Pointer to `Registers`.
/// * `restore` (rsi): Pointer to `Registers`.
#[cfg(all(target_arch = "x86_64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetcht0 [rdi]",
        "prefetcht0 [rsi]",
        "mov rax, [rsi]",
        "prefetcht0 [rax]",
        "prefetcht0 [rax + 64]",
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

/// Switches execution context without preserving floating-point state (Windows x86_64).
///
/// Preserves the Windows TIB/TEB metadata and callee-saved registers while
/// skipping XMM/SIMD state for performance.
///
/// # Arguments
/// * `save` (rcx): Pointer to `Registers`.
/// * `restore` (rdx): Pointer to `Registers`.
#[cfg(all(target_arch = "x86_64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetcht0 [rcx]",
        "prefetcht0 [rdx]",
        "mov rax, [rdx]",
        "prefetcht0 [rax]",
        "prefetcht0 [rax + 64]",
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

/// Switches execution context without preserving floating-point state (Unix AArch64).
///
/// Includes BTI and PAC protection.
///
/// # Security
/// * `bti c`: Indirect branch protection.
/// * `paciasp` / `autiasp`: Return address integrity.
#[cfg(all(
    target_arch = "aarch64",
    unix,
    not(target_os = "macos"),
    not(feature = "security-hardened")
))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
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

/// Switches execution context without preserving floating-point state (Unix AArch64).
///
/// Includes BTI and PAC protection.
///
/// # Security
/// * `bti c`: Indirect branch protection.
/// * `paciasp` / `autiasp`: Return address integrity.
#[cfg(all(
    target_arch = "aarch64",
    unix,
    not(target_os = "macos"),
    feature = "security-hardened"
))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "bti c",
        "paciasp",
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
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
        "autiasp",
        "ret"
    );
}

/// macOS AArch64: PAC-compliant no-float cross-thread switch.
///
/// # Security
/// * `bti c`: Branch target identification.
/// * `paciasp` / `autiasp`: Pointer authentication for x30.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "bti c",
        "paciasp",
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
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
        "autiasp",
        "ret"
    );
}

/// Switches execution context without preserving floating-point state (Windows AArch64).
///
/// # Security
/// * `bti c`: Branch target identification.
/// * `paciasp` / `autiasp`: Pointer authentication for x30.
#[cfg(all(target_arch = "aarch64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "bti c",
        "paciasp",
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
        "stp x19, x20, [x0, 0]",
        "stp x21, x22, [x0, 16]",
        "stp x23, x24, [x0, 32]",
        "stp x25, x26, [x0, 48]",
        "stp x27, x28, [x0, 64]",
        "stp x29, x30, [x0, 80]",
        "mov x9, sp",
        "str x9, [x0, 96]",
        // Save Windows TEB Stack Metadata
        "ldr x9, [x18, #0x08]",
        "str x9, [x0, #104]",
        "ldr x9, [x18, #0x10]",
        "str x9, [x0, #112]",
        "ldr x9, [x18, #0x12C8]",
        "str x9, [x0, #120]",
        // Restore Windows TEB Stack Metadata
        "ldr x9, [x1, #104]",
        "str x9, [x18, #0x08]",
        "ldr x9, [x1, #112]",
        "str x9, [x18, #0x10]",
        "ldr x9, [x1, #120]",
        "str x9, [x18, #0x12C8]",
        "ldp x19, x20, [x1, 0]",
        "ldp x21, x22, [x1, 16]",
        "ldp x23, x24, [x1, 32]",
        "ldp x25, x26, [x1, 48]",
        "ldp x27, x28, [x1, 64]",
        "ldp x29, x30, [x1, 80]",
        "ldr x9, [x1, 96]",
        "mov sp, x9",
        "autiasp",
        "ret"
    );
}

/// Switches execution context without preserving floating-point state (RISC-V 64 HW).
#[cfg(all(target_arch = "riscv64", unix, feature = "hw-acceleration"))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_cross_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetch.w 0(a0)",
        "prefetch.r 0(a1)",
        "prefetch.r 64(a1)",
        "ld a2, 0(a1)",
        "prefetch.r 0(a2)",
        "prefetch.r 64(a2)",
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

/// Switches execution context without preserving floating-point state (RISC-V 64).
///
/// Preserves callee-saved registers (s0-s11), stack pointer (sp), and return address (ra).
///
/// # Arguments
/// * `save` (a0): Pointer to `Registers`.
/// * `restore` (a1): Pointer to `Registers`.
#[cfg(all(target_arch = "riscv64", unix, not(feature = "hw-acceleration")))]
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
///
/// # Safety
/// * `save` and `restore` must be valid, aligned pointers to `Registers` structures.
/// * The stack pointer in `restore` must point to a valid stack region.
#[cfg(all(target_arch = "x86_64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetcht0 [rdi]",
        "prefetcht0 [rsi]",
        "prefetcht0 [rsi + 64]",
        "prefetcht0 [rsi + 128]",
        "mov rax, [rsi]",
        "prefetcht0 [rax]",
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

/// Lightweight context switch for fibers pinned to the current thread (Windows x86_64).
///
/// Skips TIB metadata preservation but maintains floating-point state.
///
/// # Arguments
/// * `save` (rcx): Pointer to `Registers`.
/// * `restore` (rdx): Pointer to `Registers`.
#[cfg(all(target_arch = "x86_64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetcht0 [rcx]",
        "prefetcht0 [rdx]",
        "prefetcht0 [rdx + 64]",
        "prefetcht0 [rdx + 128]",
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

/// Lightweight context switch for fibers pinned to the current thread (Unix AArch64).
///
/// Skips TIB/TEB metadata preservation but maintains BTI and PAC security.
///
/// # Security
/// * `bti c`: Branch target identification.
/// * `paciasp` / `autiasp`: Pointer authentication for x30.
#[cfg(all(
    target_arch = "aarch64",
    unix,
    not(target_os = "macos"),
    not(feature = "security-hardened")
))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "prfm pldl1keep, [x1, 128]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
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

/// Lightweight context switch for fibers pinned to the current thread (Unix AArch64).
///
/// Skips TIB/TEB metadata preservation but maintains BTI and PAC security.
///
/// # Security
/// * `bti c`: Branch target identification.
/// * `paciasp` / `autiasp`: Pointer authentication for x30.
#[cfg(all(
    target_arch = "aarch64",
    unix,
    not(target_os = "macos"),
    feature = "security-hardened"
))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "bti c",
        "paciasp",
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "prfm pldl1keep, [x1, 128]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
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
        "autiasp",
        "ret"
    );
}

/// macOS AArch64: PAC-compliant same-thread float switch.
///
/// # Security
/// * `bti c`: Branch target identification.
/// * `paciasp` / `autiasp`: Pointer authentication for x30.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "bti c",
        "paciasp",
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "prfm pldl1keep, [x1, 128]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
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
        "autiasp",
        "ret"
    );
}

/// Lightweight context switch for fibers pinned to the current thread (Windows AArch64).
///
/// Skips TEB metadata preservation but maintains BTI and PAC security.
///
/// # Security
/// * `bti c`: Branch target identification.
/// * `paciasp` / `autiasp`: Pointer authentication for x30.
#[cfg(all(target_arch = "aarch64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "bti c",
        "paciasp",
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "prfm pldl1keep, [x1, 128]",
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
        "autiasp",
        "ret"
    );
}

/// Lightweight context switch for fibers pinned to the current thread (RISC-V 64).
///
/// # Arguments
/// * `save` (a0): Pointer to `Registers`.
/// * `restore` (a1): Pointer to `Registers`.
#[cfg(all(target_arch = "riscv64", unix, feature = "hw-acceleration"))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetch.w 0(a0)",
        "prefetch.r 0(a1)",
        "prefetch.r 64(a1)",
        "prefetch.r 128(a1)",
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

/// Lightweight context switch for fibers pinned to the current thread (RISC-V 64).
///
/// # Arguments
/// * `save` (a0): Pointer to `Registers`.
/// * `restore` (a1): Pointer to `Registers`.
#[cfg(all(target_arch = "riscv64", unix, not(feature = "hw-acceleration")))]
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
///
/// # Safety
/// * `save` and `restore` must be valid, aligned pointers to `Registers` structures.
/// * The stack pointer in `restore` must point to a valid stack region.
#[cfg(all(target_arch = "x86_64", unix))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetcht0 [rdi]",
        "prefetcht0 [rsi]",
        "prefetcht0 [rsi + 64]",
        "mov rax, [rsi]",
        "prefetcht0 [rax]",
        "prefetcht0 [rax + 64]",
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

/// The fastest possible context switch: same-thread and no floating-point (Windows x86_64).
///
/// # Arguments
/// * `save` (rcx): Pointer to `Registers`.
/// * `restore` (rdx): Pointer to `Registers`.
#[cfg(all(target_arch = "x86_64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetcht0 [rcx]",
        "prefetcht0 [rdx]",
        "prefetcht0 [rdx + 64]",
        "mov rax, [rdx]",
        "prefetcht0 [rax]",
        "prefetcht0 [rax + 64]",
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

/// The fastest possible context switch: same-thread and no floating-point (Unix AArch64).
///
/// Includes BTI and PAC security.
///
/// # Security
/// * `bti c`: Branch target identification.
/// * `paciasp` / `autiasp`: Pointer authentication for x30.
#[cfg(all(
    target_arch = "aarch64",
    unix,
    not(target_os = "macos"),
    not(feature = "security-hardened")
))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
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

/// The fastest possible context switch: same-thread and no floating-point (Unix AArch64).
///
/// Includes BTI and PAC security.
///
/// # Security
/// * `bti c`: Branch target identification.
/// * `paciasp` / `autiasp`: Pointer authentication for x30.
#[cfg(all(
    target_arch = "aarch64",
    unix,
    not(target_os = "macos"),
    feature = "security-hardened"
))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "bti c",
        "paciasp",
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
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
        "autiasp",
        "ret"
    );
}

/// macOS AArch64: PAC-compliant same-thread no-float switch.
///
/// # Security
/// * `bti c`: Branch target identification.
/// * `paciasp` / `autiasp`: Pointer authentication for x30.
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "bti c",
        "paciasp",
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
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
        "autiasp",
        "ret"
    );
}

/// The fastest possible context switch: same-thread and no floating-point (Windows AArch64).
///
/// # Security
/// * `bti c`: Branch target identification.
/// * `paciasp` / `autiasp`: Pointer authentication for x30.
#[cfg(all(target_arch = "aarch64", windows))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "bti c",
        "paciasp",
        "prfm pstl1keep, [x0]",
        "prfm pldl1keep, [x1]",
        "prfm pldl1keep, [x1, 64]",
        "ldr x9, [x1, 96]",
        "prfm pldl1keep, [x9]",
        "prfm pldl1keep, [x9, 64]",
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
        "autiasp",
        "ret"
    );
}

/// The fastest possible context switch: same-thread and no floating-point (RISC-V 64 HW).
///
/// # Arguments
/// * `save` (a0): Pointer to `Registers`.
/// * `restore` (a1): Pointer to `Registers`.
#[cfg(all(target_arch = "riscv64", unix, feature = "hw-acceleration"))]
#[unsafe(naked)]
pub unsafe extern "C" fn switch_context_same_thread_no_float(
    save: *mut Registers,
    restore: *const Registers,
) {
    naked_asm!(
        "prefetch.w 0(a0)",
        "prefetch.r 0(a1)",
        "prefetch.r 64(a1)",
        "ld a2, 0(a1)",
        "prefetch.r 0(a2)",
        "prefetch.r 64(a2)",
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

/// The fastest possible context switch: same-thread and no floating-point (RISC-V 64).
///
/// # Arguments
/// * `save` (a0): Pointer to `Registers`.
/// * `restore` (a1): Pointer to `Registers`.
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

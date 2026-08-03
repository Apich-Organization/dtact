#[cfg(all(feature = "hw-acceleration", target_arch = "x86_64"))]
fn test_hw() {
    unsafe {
        let mut x: u32 = 0;
        core::arch::asm!(
                "mov rax, {0}",
                "umonitor rax",
                "mov rcx, 1000",
                "mov rdx, 0",
                "mov rax, 0",
                "umwait ecx",
                in(reg) &mut x,
                out("rax") _,
                out("rcx") _,
                out("rdx") _,
                options(nostack, preserves_flags),
            );
    }
}
pub fn main() {
    #[cfg(all(feature = "hw-acceleration", target_arch = "x86_64"))]
    test_hw();
}

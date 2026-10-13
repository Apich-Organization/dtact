import re

with open('src/benchmark/dta_harness.rs', 'r') as f:
    content = f.read()

# Fix needless_range_loop in dta_harness.rs
content = re.sub(
    r'for core in 0\.\.self\.worker_count\(\) \{\n(\s*)let worker = unsafe \{ &\*self\.scheduler\.workers\[core\]\.get\(\) \};',
    r'for worker_cell in self.scheduler.workers.iter().take(self.worker_count()) {\n\1let worker = unsafe { &*worker_cell.get() };',
    content
)

with open('src/benchmark/dta_harness.rs', 'w') as f:
    f.write(content)

with open('src/dta_scheduler.rs', 'r') as f:
    content = f.read()

# Fix manual_bit_width in dta_scheduler.rs by adding #[allow(clippy::manual_bit_width)]
# (as instructed in memory: "When addressing clippy::manual_bit_width warnings ... use #[allow(clippy::manual_bit_width)] rather than changing to .bit_width()")
content = re.sub(
    r'const fn ceil_log2\(n: usize\) -> u8 \{',
    r'#[allow(clippy::manual_bit_width)]\n    const fn ceil_log2(n: usize) -> u8 {',
    content
)

with open('src/dta_scheduler.rs', 'w') as f:
    f.write(content)

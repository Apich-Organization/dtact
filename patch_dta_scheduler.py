import re

with open('src/dta_scheduler.rs', 'r') as f:
    content = f.read()

# 1. Update push_batch signature and body
content = re.sub(
    r'pub fn push_batch\(&mut self, chunk: &TaskChunk\) \{(\s*)let count = chunk\.count as usize;(\s*)let tail = self\.local_tail\.load\(Ordering::Relaxed\);',
    r'pub fn push_batch(&mut self, chunk: &TaskChunk, cur_len: usize, fixed_head: usize) {\1let count = chunk.count as usize;\2// Compute exact tail offset from cached invariants to avoid redundant atomic load\2let tail = fixed_head.wrapping_add(cur_len) & LOCAL_QUEUE_MASK;',
    content
)

# 2. Update route_chunk signature and body
content = re.sub(
    r'fn route_chunk\(\s*&self,\s*worker: &mut Worker,\s*current_core: usize,\s*chunk: TaskChunk,\s*cur_len: usize,\s*\) -> usize \{',
    r'fn route_chunk(\n        &self,\n        worker: &mut Worker,\n        current_core: usize,\n        chunk: TaskChunk,\n        cur_len: usize,\n        fixed_head: usize,\n    ) -> usize {',
    content
)

content = re.sub(
    r'self\.route_local\(worker, current_core, chunk\);',
    r'self.route_local(worker, current_core, chunk, cur_len, fixed_head);',
    content
)

# 3. Update route_local signature and body
content = re.sub(
    r'fn route_local\(&self, worker: &mut Worker, _core: usize, chunk: TaskChunk\) \{(\s*)worker\.push_batch\(&chunk\);',
    r'fn route_local(&self, worker: &mut Worker, _core: usize, chunk: TaskChunk, cur_len: usize, fixed_head: usize) {\1worker.push_batch(&chunk, cur_len, fixed_head);',
    content
)

# 4. Update poll_mailboxes calls
content = re.sub(
    r'let added = self\.route_chunk\(worker, current_core, chunk, cur_len\);',
    r'let added = self.route_chunk(worker, current_core, chunk, cur_len, fixed_head);',
    content
)

# 5. Update drain_warehouse calls
content = re.sub(
    r'worker\.push_batch\(&chunk\);(\s*)cur_len \+= chunk\.count as usize;',
    r'worker.push_batch(&chunk, cur_len, fixed_head);\1cur_len += chunk.count as usize;',
    content
)

# 6. Update the outdated doc comment on poll_mailboxes
content = re.sub(
    r'not change here \(we are only pushing work, not consuming\)\. Only\s+/// `local_tail` is re-loaded per iteration, halving the atomic reads in',
    r'not change here (we are only pushing work, not consuming). `local_tail`\n    /// offsets are computed mathematically from invariants passed down to helper functions,\n    /// eliminating redundant atomic reads in',
    content
)

with open('src/dta_scheduler.rs', 'w') as f:
    f.write(content)

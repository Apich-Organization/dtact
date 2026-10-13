import re

with open('tests/scheduler.rs', 'r') as f:
    content = f.read()

# Fix needless_range_loop in scheduler.rs tests
content = re.sub(
    r'for j in 0\.\.64 \{\n(\s*)let mailbox = &scheduler\.mailboxes\[i\]\[j\];',
    r'for mailbox in scheduler.mailboxes[i].iter().take(64) {\n\1',
    content
)

with open('tests/scheduler.rs', 'w') as f:
    f.write(content)

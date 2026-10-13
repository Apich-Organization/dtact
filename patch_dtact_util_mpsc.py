import re

with open('dtact-util/src/sync/mpsc.rs', 'r') as f:
    content = f.read()

# Fix future_not_send warnings by adding #[allow(clippy::future_not_send)]
# to the three methods identified by clippy.
content = re.sub(
    r'pub async fn send\(&self, value: T\) -> Result<\(\), SendError<T>> \{',
    r'#[allow(clippy::future_not_send)]\n    pub async fn send(&self, value: T) -> Result<(), SendError<T>> {',
    content
)

content = re.sub(
    r'pub async fn recv\(&mut self\) -> Option<T> \{',
    r'#[allow(clippy::future_not_send)]\n    pub async fn recv(&mut self) -> Option<T> {',
    content
)

with open('dtact-util/src/sync/mpsc.rs', 'w') as f:
    f.write(content)

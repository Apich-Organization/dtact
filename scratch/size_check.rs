extern crate dtact;
use dtact::dta_scheduler::Worker;
use dtact::memory_management::ContextPool;
use dtact::Runtime;

fn main() {
    println!("Size of Worker: {}", core::mem::size_of::<Worker>());
    println!("Align of Worker: {}", core::mem::align_of::<Worker>());
    println!("Size of Runtime: {}", core::mem::size_of::<Runtime>());
    println!("Align of Runtime: {}", core::mem::align_of::<Runtime>());
}

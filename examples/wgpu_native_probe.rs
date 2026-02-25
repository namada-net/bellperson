//! Native probe for the wgpu backend.
//!
//! Usage:
//!   cargo run --example wgpu_native_probe --features groth16,wgpu

#[cfg(feature = "wgpu")]
fn main() {
    use nam_bellperson::gpu::{WgpuDevice, WgpuEngine};

    let devices = WgpuDevice::enumerate();
    println!("Discovered {} adapters", devices.len());
    for d in devices {
        println!(
            "- {} (vendor={}, device={})",
            d.name(),
            d.vendor_id(),
            d.device_id()
        );
    }

    match WgpuEngine::new() {
        Ok(_) => println!("wgpu engine init: ok"),
        Err(e) => println!("wgpu engine init: failed: {e}"),
    }
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    println!("Enable feature `wgpu` to run this example.");
}

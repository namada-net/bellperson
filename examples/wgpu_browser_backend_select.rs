//! Browser/backend selection helper for wgpu bellperson backend.
//!
//! This example demonstrates configuring backend selection without env vars,
//! which is useful in wasm/browser contexts.

#[cfg(feature = "wgpu")]
fn main() {
    // For browser builds, this can be called from JS/wasm init logic.
    // Valid values: metal,vulkan,dx12,gl,browser_webgpu
    nam_bellperson::gpu::set_wgpu_backends_from_str("browser_webgpu,gl");

    // Optional adapter substring filter (native use-cases mainly).
    nam_bellperson::gpu::set_wgpu_adapter_name_filter(None);

    println!("Configured wgpu backends for browser-friendly selection.");
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    println!("Enable feature `wgpu` to use this example.");
}

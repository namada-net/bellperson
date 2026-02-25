//! wasm-friendly runtime backend configuration demo.
//!
//! This example is intended for browser/WASM integration points where environment
//! variables are unavailable. It shows calling exported setters directly.

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
fn main() {
    nam_bellperson::gpu::wasm_set_wgpu_backends("browser_webgpu,gl");
    nam_bellperson::gpu::wasm_set_wgpu_adapter_name_filter(None);
}

#[cfg(not(all(feature = "wgpu", target_arch = "wasm32")))]
fn main() {
    // No-op outside wasm+wgpu.
}

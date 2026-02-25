//! WebGPU device management and initialization.
//!
//! This module handles WebGPU device initialization for both native platforms
//! (Vulkan, Metal, DirectX12) and web browsers via WebGPU.
//!
//! Browser compatibility:
//! - Chrome 113+ (WebGPU stable)
//! - Firefox 126+ (WebGPU enabled by default)
//! - Safari 18+ (WebGPU support)

use std::sync::Arc;
use std::sync::Mutex;

use wgpu::{Adapter, Device, DeviceDescriptor, Instance, Queue};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use super::error::{WgpuError, WgpuResult};

static BACKENDS_OVERRIDE: Mutex<Option<wgpu::Backends>> = Mutex::new(None);
static ADAPTER_NAME_OVERRIDE: Mutex<Option<String>> = Mutex::new(None);

/// WebGPU engine wrapper with thread-safe reference counting
pub struct WgpuEngine {
    device: Arc<Device>,
    queue: Arc<Queue>,
    adapter: Adapter,
}

impl WgpuEngine {
    /// Create a new WebGPU engine
    pub fn new() -> WgpuResult<Self> {
        if let Ok(val) = std::env::var("BELLMAN_NO_GPU") {
            if val != "0" {
                return Err(WgpuError::Disabled);
            }
        }

        Self::init_engine()
    }

    fn init_engine() -> WgpuResult<WgpuEngine> {
        let backends = configured_backends();
        let adapter_name_filter = configured_adapter_name();

        let power_preference = match std::env::var("WGPU_POWER_PREF").ok().as_deref() {
            Some("low") => wgpu::PowerPreference::LowPower,
            _ => wgpu::PowerPreference::HighPerformance,
        };

        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends,
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::default(),
            flags: wgpu::InstanceFlags::empty(),
        });

        #[cfg(not(target_arch = "wasm32"))]
        {
            let infos = instance
                .enumerate_adapters(backends)
                .into_iter()
                .map(|a| format!("{:?}", a.get_info()))
                .collect::<Vec<_>>();
            log::info!(
                "WGPU adapter probe backends={:?} filter={:?} adapters={:?}",
                backends,
                adapter_name_filter,
                infos
            );
        }

        let adapter = pick_adapter(
            &instance,
            backends,
            power_preference,
            adapter_name_filter.as_deref(),
        )
        .ok_or_else(|| WgpuError::DeviceNotAvailable("No GPU adapter found".to_string()))?;

        let (device, queue) = block_on_compat(adapter.request_device(
            &DeviceDescriptor {
                label: Some("bellperson-wgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .map_err(|e| WgpuError::DeviceNotAvailable(e.to_string()))?;

        if crate::gpu::require_gpu() {
            let info = adapter.get_info();
            log::info!("WGPU strict adapter selected: {:?}", info);
        }

        log::info!("WebGPU device initialized: {:?}", adapter.get_info());

        Ok(WgpuEngine {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter,
        })
    }

    /// Get a reference to the WebGPU device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get an Arc reference to the WebGPU device for thread safety
    pub fn device_arc(&self) -> Arc<Device> {
        Arc::clone(&self.device)
    }

    /// Get a reference to the WebGPU queue
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Get an Arc reference to the WebGPU queue for thread safety
    pub fn queue_arc(&self) -> Arc<Queue> {
        Arc::clone(&self.queue)
    }

    /// Get a reference to the WebGPU adapter
    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }
}

pub fn set_wgpu_backends_from_str(backends_csv: &str) {
    let backends = parse_backends_env(Some(backends_csv));
    if let Ok(mut guard) = BACKENDS_OVERRIDE.lock() {
        *guard = Some(backends);
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wasm_set_wgpu_backends(backends_csv: &str) {
    set_wgpu_backends_from_str(backends_csv);
}

pub fn clear_wgpu_backends_override() {
    if let Ok(mut guard) = BACKENDS_OVERRIDE.lock() {
        *guard = None;
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wasm_clear_wgpu_backends_override() {
    clear_wgpu_backends_override();
}

pub fn set_wgpu_adapter_name_filter(name_substring: Option<&str>) {
    if let Ok(mut guard) = ADAPTER_NAME_OVERRIDE.lock() {
        *guard = name_substring.map(ToOwned::to_owned);
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn wasm_set_wgpu_adapter_name_filter(name_substring: Option<String>) {
    set_wgpu_adapter_name_filter(name_substring.as_deref());
}

fn configured_backends() -> wgpu::Backends {
    if let Ok(guard) = BACKENDS_OVERRIDE.lock() {
        if let Some(backends) = *guard {
            return backends;
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        // Browser default: prefer WebGPU, allow GL fallback where supported.
        return parse_backends_env(Some("browser_webgpu,gl"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        parse_backends_env(std::env::var("WGPU_BACKEND").ok().as_deref())
    }
}

fn configured_adapter_name() -> Option<String> {
    if let Ok(guard) = ADAPTER_NAME_OVERRIDE.lock() {
        if let Some(name) = guard.clone() {
            return Some(name);
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        None
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var("WGPU_ADAPTER_NAME").ok()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn pick_adapter(
    instance: &Instance,
    backends: wgpu::Backends,
    power_preference: wgpu::PowerPreference,
    adapter_name_filter: Option<&str>,
) -> Option<Adapter> {
    let needle = adapter_name_filter.map(|s| s.to_ascii_lowercase());

    // Prefer explicit adapter selection from enumerated adapters.
    let mut adapters = instance.enumerate_adapters(backends).into_iter();
    if let Some(needle) = needle.as_deref() {
        if let Some(a) = adapters.find(|a| a.get_info().name.to_ascii_lowercase().contains(needle))
        {
            return Some(a);
        }
    } else if let Some(a) = adapters.next() {
        return Some(a);
    }

    // Fallback to request_adapter when enumeration didn't produce a match.
    block_on_compat(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
}

#[cfg(target_arch = "wasm32")]
fn pick_adapter(
    instance: &Instance,
    _backends: wgpu::Backends,
    power_preference: wgpu::PowerPreference,
    _adapter_name_filter: Option<&str>,
) -> Option<Adapter> {
    block_on_compat(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
}

fn parse_backends_env(value: Option<&str>) -> wgpu::Backends {
    let Some(value) = value else {
        return wgpu::Backends::all();
    };

    let mut out = wgpu::Backends::empty();
    for tok in value.split(',').map(|s| s.trim().to_ascii_lowercase()) {
        match tok.as_str() {
            "metal" => out |= wgpu::Backends::METAL,
            "vulkan" => out |= wgpu::Backends::VULKAN,
            "dx12" => out |= wgpu::Backends::DX12,
            "gl" => out |= wgpu::Backends::GL,
            "browser_webgpu" => out |= wgpu::Backends::BROWSER_WEBGPU,
            "" => {}
            _ => {}
        }
    }

    if out.is_empty() {
        wgpu::Backends::all()
    } else {
        out
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn block_on_compat<F: std::future::Future>(fut: F) -> F::Output {
    pollster::block_on(fut)
}

#[cfg(target_arch = "wasm32")]
fn block_on_compat<F: std::future::Future>(_fut: F) -> F::Output {
    panic!("wgpu synchronous initialization is not available on wasm32; use browser async setup")
}

/// Represents a discovered WebGPU device
pub struct WgpuDevice {
    name: String,
    vendor: u32,
    device_id: u32,
}

impl WgpuDevice {
    /// Enumerate all available WebGPU devices
    pub fn enumerate() -> Vec<WgpuDevice> {
        #[cfg(target_arch = "wasm32")]
        {
            Vec::new()
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let instance = Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                dx12_shader_compiler: wgpu::Dx12Compiler::default(),
                gles_minor_version: wgpu::Gles3MinorVersion::default(),
                flags: wgpu::InstanceFlags::empty(),
            });

            let mut devices = Vec::new();

            for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
                let info = adapter.get_info();
                devices.push(WgpuDevice {
                    name: info.name,
                    vendor: info.vendor,
                    device_id: info.device,
                });
            }

            devices
        }
    }

    /// Get the device name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the vendor ID
    pub fn vendor_id(&self) -> u32 {
        self.vendor
    }

    /// Get the device ID
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

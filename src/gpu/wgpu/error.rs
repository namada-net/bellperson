//! Error types for the WebGPU backend.

use bellpepper_core::SynthesisError;

#[derive(thiserror::Error, Debug)]
pub enum WgpuError {
    #[error("WebGPU error: {0}")]
    Simple(&'static str),

    #[error("No WebGPU device available: {0}")]
    DeviceNotAvailable(String),

    #[error("Shader compilation error: {0}")]
    ShaderCompilation(String),

    #[error("Pipeline creation error: {0}")]
    PipelineCreation(String),

    #[error("Buffer error: {0}")]
    BufferError(String),

    #[error("Command encoder error: {0}")]
    CommandEncoder(String),

    #[error("GPU compute error: {0}")]
    Compute(String),

    #[error("Feature not supported: {0}")]
    NotSupported(String),

    #[error("WebGPU is disabled via environment variable")]
    Disabled,

    #[error("WASM initialization error: {0}")]
    WasmError(String),
}

pub type WgpuResult<T> = std::result::Result<T, WgpuError>;

impl From<WgpuError> for SynthesisError {
    fn from(e: WgpuError) -> Self {
        SynthesisError::MalformedProofs(format!("Encountered a WebGPU Error: {}", e))
    }
}

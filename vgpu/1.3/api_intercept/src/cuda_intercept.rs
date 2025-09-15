//! CUDA API Interception

use std::sync::Arc;
use std::ffi::c_void;
use libc::{c_int, size_t};
use anyhow::Result;
use tracing::{info, debug};
use vgpu_core::VirtualGPUCore;

pub struct CUDAInterceptor {
    vgpu: Arc<VirtualGPUCore>,
    device_count: i32,
    memory_allocations: std::collections::HashMap<*mut c_void, usize>,
}

impl CUDAInterceptor {
    pub async fn new(vgpu: Arc<VirtualGPUCore>) -> Result<Self> {
        Ok(Self {
            vgpu,
            device_count: 1, // Single vGPU device
            memory_allocations: std::collections::HashMap::new(),
        })
    }
}

// CUDA API implementations
pub extern "C" fn cu_init(flags: c_int) -> c_int {
    info!("CUDA cuInit called with flags: {}", flags);
    0 // CUDA_SUCCESS
}

pub extern "C" fn cu_device_get(device: *mut c_int, ordinal: c_int) -> c_int {
    debug!("CUDA cuDeviceGet called for device {}", ordinal);
    if ordinal >= 1 {
        return 101; // CUDA_ERROR_INVALID_DEVICE
    }
    unsafe {
        *device = 0; // Return virtual device 0
    }
    0 // CUDA_SUCCESS
}

pub extern "C" fn cu_mem_alloc(dptr: *mut *mut c_void, bytesize: size_t) -> c_int {
    debug!("CUDA cuMemAlloc called for {} bytes", bytesize);
    
    // Allocate using system malloc and redirect to vGPU memory management
    let ptr = unsafe { libc::malloc(bytesize) };
    if ptr.is_null() {
        return 2; // CUDA_ERROR_OUT_OF_MEMORY
    }
    
    unsafe {
        *dptr = ptr;
    }
    
    0 // CUDA_SUCCESS
}
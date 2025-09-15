//! OpenCL API Interception

use std::sync::Arc;
use std::ffi::c_void;
use libc::c_int;
use anyhow::Result;
use tracing::{info, debug};
use vgpu_core::VirtualGPUCore;

pub struct OpenCLInterceptor {
    vgpu: Arc<VirtualGPUCore>,
    platform_count: i32,
}

impl OpenCLInterceptor {
    pub async fn new(vgpu: Arc<VirtualGPUCore>) -> Result<Self> {
        Ok(Self {
            vgpu,
            platform_count: 1, // Single vGPU platform
        })
    }
}

pub extern "C" fn cl_get_platform_ids(
    num_entries: c_int,
    platforms: *mut *mut c_void,
    num_platforms: *mut c_int,
) -> c_int {
    info!("OpenCL clGetPlatformIDs called");
    
    if !num_platforms.is_null() {
        unsafe {
            *num_platforms = 1; // One virtual platform
        }
    }
    
    if !platforms.is_null() && num_entries >= 1 {
        unsafe {
            *platforms = 0x1000 as *mut c_void; // Fake platform handle
        }
    }
    
    0 // CL_SUCCESS
}
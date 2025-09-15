//! API Interception Layer for vGPU v1.3
//! 
//! Comprehensive CUDA, OpenCL, and Vulkan API interception with
//! transparent redirection to infinite mathematical engines.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::ffi::{CStr, CString};

use anyhow::{Result, Error};
use libc::{c_char, c_int, c_void, size_t};
use libloading::{Library, Symbol};
use tracing::{info, debug, warn};

use vgpu_core::VirtualGPUCore;
use infinite_engines::ComputeInput;

pub mod cuda_intercept;
pub mod opencl_intercept;
pub mod vulkan_intercept;

/// Main API interception manager
pub struct APIInterceptionManager {
    /// vGPU core instance
    vgpu_core: Arc<VirtualGPUCore>,
    
    /// CUDA interception
    cuda_intercept: cuda_intercept::CUDAInterceptor,
    
    /// OpenCL interception  
    opencl_intercept: opencl_intercept::OpenCLInterceptor,
    
    /// Vulkan interception
    vulkan_intercept: vulkan_intercept::VulkanInterceptor,
    
    /// Interception statistics
    stats: Arc<Mutex<InterceptionStats>>,
}

#[derive(Debug, Default)]
pub struct InterceptionStats {
    pub cuda_calls: u64,
    pub opencl_calls: u64,
    pub vulkan_calls: u64,
    pub total_redirected: u64,
    pub total_fallback: u64,
}

impl APIInterceptionManager {
    pub async fn new() -> Result<Self> {
        info!("Initializing API Interception Manager");
        
        let config = vgpu_core::VGPUConfig::default();
        let vgpu_core = Arc::new(VirtualGPUCore::new(config).await?);
        
        Ok(Self {
            vgpu_core: vgpu_core.clone(),
            cuda_intercept: cuda_intercept::CUDAInterceptor::new(vgpu_core.clone()).await?,
            opencl_intercept: opencl_intercept::OpenCLInterceptor::new(vgpu_core.clone()).await?,
            vulkan_intercept: vulkan_intercept::VulkanInterceptor::new(vgpu_core.clone()).await?,
            stats: Arc::new(Mutex::new(InterceptionStats::default())),
        })
    }
    
    pub fn get_stats(&self) -> InterceptionStats {
        self.stats.lock().unwrap().clone()
    }
}

// Export C-compatible functions for dynamic library interposition
#[no_mangle]
pub extern "C" fn cuInit(flags: c_int) -> c_int {
    cuda_intercept::cu_init(flags)
}

#[no_mangle]
pub extern "C" fn cuDeviceGet(device: *mut c_int, ordinal: c_int) -> c_int {
    cuda_intercept::cu_device_get(device, ordinal)
}

#[no_mangle]
pub extern "C" fn cuMemAlloc_v2(dptr: *mut *mut c_void, bytesize: size_t) -> c_int {
    cuda_intercept::cu_mem_alloc(dptr, bytesize)
}

#[no_mangle]
pub extern "C" fn clGetPlatformIDs(
    num_entries: c_int,
    platforms: *mut *mut c_void,
    num_platforms: *mut c_int,
) -> c_int {
    opencl_intercept::cl_get_platform_ids(num_entries, platforms, num_platforms)
}

#[no_mangle]
pub extern "C" fn vkCreateInstance(
    create_info: *const c_void,
    allocator: *const c_void,
    instance: *mut *mut c_void,
) -> c_int {
    vulkan_intercept::vk_create_instance(create_info, allocator, instance)
}

// Global API manager instance
static mut API_MANAGER: Option<APIInterceptionManager> = None;
static INIT: std::sync::Once = std::sync::Once::new();

pub fn get_api_manager() -> &'static APIInterceptionManager {
    unsafe {
        INIT.call_once(|| {
            let manager = futures::executor::block_on(APIInterceptionManager::new())
                .expect("Failed to initialize API manager");
            API_MANAGER = Some(manager);
        });
        API_MANAGER.as_ref().unwrap()
    }
}
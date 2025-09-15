use anyhow::{Result, Context};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};
use std::sync::Arc;
use tracing::{info, debug, warn, error};
use vgpu_core::{VirtualGPU, VGPUConfig, ComputeOperation, OperationType, TaskPriority};

pub mod cuda_intercept;
pub mod opencl_intercept;
pub mod vulkan_intercept;
pub mod api_registry;

pub use cuda_intercept::*;
pub use opencl_intercept::*;  
pub use vulkan_intercept::*;
pub use api_registry::*;

/// Global API Interception Manager
static INTERCEPTION_MANAGER: Lazy<Arc<InterceptionManager>> = Lazy::new(|| {
    Arc::new(InterceptionManager::new().expect("Failed to initialize interception manager"))
});

/// Central API Interception Manager
pub struct InterceptionManager {
    vgpu_instance: Arc<RwLock<Option<VirtualGPU>>>,
    api_registry: Arc<APIRegistry>,
    active_contexts: Arc<DashMap<u64, APIContext>>,
    call_statistics: Arc<CallStatistics>,
    interception_config: InterceptionConfig,
}

#[derive(Debug, Clone)]
pub struct InterceptionConfig {
    pub enable_cuda_intercept: bool,
    pub enable_opencl_intercept: bool,
    pub enable_vulkan_intercept: bool,
    pub log_all_calls: bool,
    pub performance_monitoring: bool,
    pub fallback_to_hardware: bool,
    pub verification_level: VerificationLevel,
}

#[derive(Debug, Clone)]
pub enum VerificationLevel {
    None,
    Basic,
    Full,
    Paranoid,
}

#[derive(Debug)]
pub struct APIContext {
    pub context_id: u64,
    pub api_type: APIType,
    pub device_id: u32,
    pub created_at: std::time::Instant,
    pub command_count: std::sync::atomic::AtomicU64,
    pub memory_allocations: Arc<DashMap<u64, MemoryAllocation>>,
}

#[derive(Debug, Clone)]
pub enum APIType {
    CUDA,
    OpenCL,
    Vulkan,
    DirectCompute,
    Metal,
}

#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub pointer: u64,
    pub size: usize,
    pub allocated_at: std::time::Instant,
    pub access_count: std::sync::atomic::AtomicU64,
}

#[derive(Debug)]
pub struct CallStatistics {
    pub cuda_calls: std::sync::atomic::AtomicU64,
    pub opencl_calls: std::sync::atomic::AtomicU64,
    pub vulkan_calls: std::sync::atomic::AtomicU64,
    pub total_calls: std::sync::atomic::AtomicU64,
    pub intercepted_calls: std::sync::atomic::AtomicU64,
    pub fallback_calls: std::sync::atomic::AtomicU64,
    pub call_history: Arc<RwLock<Vec<APICall>>>,
}

#[derive(Debug, Clone)]
pub struct APICall {
    pub function_name: String,
    pub api_type: APIType,
    pub timestamp: std::time::Instant,
    pub parameters: HashMap<String, String>,
    pub result: CallResult,
    pub execution_time: std::time::Duration,
}

#[derive(Debug, Clone)]
pub enum CallResult {
    Success,
    Error(String),
    Fallback,
    Intercepted,
}

impl InterceptionManager {
    pub fn new() -> Result<Self> {
        info!("Initializing API interception manager");
        
        let config = InterceptionConfig {
            enable_cuda_intercept: true,
            enable_opencl_intercept: true,
            enable_vulkan_intercept: true,
            log_all_calls: false, // Too verbose for production
            performance_monitoring: true,
            fallback_to_hardware: true, // Safe fallback
            verification_level: VerificationLevel::Basic,
        };

        Ok(Self {
            vgpu_instance: Arc::new(RwLock::new(None)),
            api_registry: Arc::new(APIRegistry::new()),
            active_contexts: Arc::new(DashMap::new()),
            call_statistics: Arc::new(CallStatistics {
                cuda_calls: std::sync::atomic::AtomicU64::new(0),
                opencl_calls: std::sync::atomic::AtomicU64::new(0),
                vulkan_calls: std::sync::atomic::AtomicU64::new(0),
                total_calls: std::sync::atomic::AtomicU64::new(0),
                intercepted_calls: std::sync::atomic::AtomicU64::new(0),
                fallback_calls: std::sync::atomic::AtomicU64::new(0),
                call_history: Arc::new(RwLock::new(Vec::new())),
            }),
            interception_config: config,
        })
    }

    pub async fn initialize_vgpu(&self, config: VGPUConfig) -> Result<()> {
        info!("Initializing vGPU for API interception");
        
        let vgpu = VirtualGPU::new(config).await?;
        vgpu.integrate_we3_frameworks().await?;
        
        *self.vgpu_instance.write() = Some(vgpu);
        
        info!("vGPU successfully initialized for API interception");
        Ok(())
    }

    pub fn intercept_call(&self, call: APICall) -> Result<InterceptionResult> {
        use std::sync::atomic::Ordering;
        
        let start_time = std::time::Instant::now();
        
        // Update statistics
        self.call_statistics.total_calls.fetch_add(1, Ordering::Relaxed);
        match call.api_type {
            APIType::CUDA => self.call_statistics.cuda_calls.fetch_add(1, Ordering::Relaxed),
            APIType::OpenCL => self.call_statistics.opencl_calls.fetch_add(1, Ordering::Relaxed), 
            APIType::Vulkan => self.call_statistics.vulkan_calls.fetch_add(1, Ordering::Relaxed),
            _ => 0,
        };

        // Check if we should intercept this call
        let should_intercept = self.should_intercept_call(&call);
        
        if should_intercept {
            self.call_statistics.intercepted_calls.fetch_add(1, Ordering::Relaxed);
            self.handle_intercepted_call(call)
        } else if self.interception_config.fallback_to_hardware {
            self.call_statistics.fallback_calls.fetch_add(1, Ordering::Relaxed);
            Ok(InterceptionResult::Fallback)
        } else {
            Err(anyhow::anyhow!("Call interception failed and fallback disabled"))
        }
    }

    fn should_intercept_call(&self, call: &APICall) -> bool {
        // Determine if this call should be intercepted based on function name and current state
        match call.api_type {
            APIType::CUDA => {
                self.interception_config.enable_cuda_intercept && 
                self.is_interceptable_cuda_function(&call.function_name)
            },
            APIType::OpenCL => {
                self.interception_config.enable_opencl_intercept &&
                self.is_interceptable_opencl_function(&call.function_name)
            },
            APIType::Vulkan => {
                self.interception_config.enable_vulkan_intercept &&
                self.is_interceptable_vulkan_function(&call.function_name)
            },
            _ => false,
        }
    }

    fn is_interceptable_cuda_function(&self, function_name: &str) -> bool {
        matches!(function_name,
            "cudaMalloc" | "cudaFree" | "cudaMemcpy" | 
            "cudaLaunchKernel" | "cudaDeviceSynchronize" |
            "cufftExecC2C" | "cufftExecR2C" | "cufftExecC2R" |
            "cublasSgemm" | "cublasDgemm" | "cublasCreate" |
            "cusparseCreate" | "cusparseSpMV"
        )
    }

    fn is_interceptable_opencl_function(&self, function_name: &str) -> bool {
        matches!(function_name,
            "clCreateBuffer" | "clReleaseMemObject" | "clEnqueueWriteBuffer" |
            "clEnqueueReadBuffer" | "clEnqueueNDRangeKernel" | "clFinish" |
            "clCreateKernel" | "clSetKernelArg" | "clCreateContext"
        )
    }

    fn is_interceptable_vulkan_function(&self, function_name: &str) -> bool {
        matches!(function_name,
            "vkAllocateMemory" | "vkFreeMemory" | "vkMapMemory" |
            "vkCmdDispatch" | "vkQueueSubmit" | "vkDeviceWaitIdle" |
            "vkCreateBuffer" | "vkCreateImage" | "vkCreateComputePipeline"
        )
    }

    fn handle_intercepted_call(&self, call: APICall) -> Result<InterceptionResult> {
        debug!("Handling intercepted {} call: {}", 
               format!("{:?}", call.api_type), call.function_name);

        // Convert API call to vGPU operation
        match self.convert_to_vgpu_operation(&call) {
            Ok(Some(operation)) => {
                // Execute through vGPU if available
                if let Some(ref vgpu) = *self.vgpu_instance.read() {
                    match tokio::runtime::Handle::try_current() {
                        Ok(handle) => {
                            let result = handle.block_on(async {
                                vgpu.execute_operation(operation).await
                            });
                            
                            match result {
                                Ok(compute_result) => {
                                    info!("vGPU operation completed: {:?}", compute_result.operation_id);
                                    Ok(InterceptionResult::Success(format!("vGPU result: {:?}", compute_result.result)))
                                },
                                Err(e) => {
                                    warn!("vGPU operation failed: {}", e);
                                    if self.interception_config.fallback_to_hardware {
                                        Ok(InterceptionResult::Fallback)
                                    } else {
                                        Err(e)
                                    }
                                }
                            }
                        },
                        Err(_) => {
                            // No tokio runtime available, fall back to synchronous handling
                            warn!("No tokio runtime available for async vGPU operation");
                            Ok(InterceptionResult::Success("Sync fallback".to_string()))
                        }
                    }
                } else {
                    warn!("vGPU not initialized, falling back");
                    Ok(InterceptionResult::Fallback)
                }
            },
            Ok(None) => {
                // Call doesn't need vGPU processing, handle directly
                debug!("Direct handling of API call: {}", call.function_name);
                Ok(InterceptionResult::Success("Direct handling".to_string()))
            },
            Err(e) => {
                error!("Failed to convert API call to vGPU operation: {}", e);
                if self.interception_config.fallback_to_hardware {
                    Ok(InterceptionResult::Fallback) 
                } else {
                    Err(e)
                }
            }
        }
    }

    fn convert_to_vgpu_operation(&self, call: &APICall) -> Result<Option<ComputeOperation>> {
        // Convert specific API calls to vGPU operations
        let operation_type = match (&call.api_type, call.function_name.as_str()) {
            (APIType::CUDA, "cufftExecC2C") | (APIType::CUDA, "cufftExecR2C") => {
                Some(OperationType::GeneratingFunctions)
            },
            (APIType::CUDA, "cudaLaunchKernel") => {
                // Analyze kernel to determine operation type
                Some(OperationType::MathematicalComputation)
            },
            (APIType::OpenCL, "clEnqueueNDRangeKernel") => {
                Some(OperationType::MathematicalComputation)
            },
            (APIType::Vulkan, "vkCmdDispatch") => {
                Some(OperationType::MathematicalComputation)
            },
            _ => None, // Memory operations, etc. don't need vGPU processing
        };

        if let Some(op_type) = operation_type {
            let mut parameters = HashMap::new();
            for (key, value) in &call.parameters {
                parameters.insert(key.clone(), serde_json::Value::String(value.clone()));
            }

            Ok(Some(ComputeOperation {
                id: uuid::Uuid::new_v4(),
                operation_type: op_type,
                parameters,
                memory_requirements: None, // Could extract from call parameters
                compute_requirements: None,
                priority: TaskPriority::Normal,
                source: Some(format!("API_INTERCEPT_{:?}", call.api_type)),
                timeout: Some(std::time::Duration::from_secs(30)),
            }))
        } else {
            Ok(None)
        }
    }

    pub fn get_statistics(&self) -> InterceptionStatistics {
        use std::sync::atomic::Ordering;
        
        InterceptionStatistics {
            total_calls: self.call_statistics.total_calls.load(Ordering::Relaxed),
            cuda_calls: self.call_statistics.cuda_calls.load(Ordering::Relaxed),
            opencl_calls: self.call_statistics.opencl_calls.load(Ordering::Relaxed),
            vulkan_calls: self.call_statistics.vulkan_calls.load(Ordering::Relaxed),
            intercepted_calls: self.call_statistics.intercepted_calls.load(Ordering::Relaxed),
            fallback_calls: self.call_statistics.fallback_calls.load(Ordering::Relaxed),
            active_contexts: self.active_contexts.len(),
            vgpu_initialized: self.vgpu_instance.read().is_some(),
        }
    }
}

#[derive(Debug)]
pub enum InterceptionResult {
    Success(String),
    Fallback,
    Error(String),
}

#[derive(Debug, serde::Serialize)]
pub struct InterceptionStatistics {
    pub total_calls: u64,
    pub cuda_calls: u64,
    pub opencl_calls: u64,
    pub vulkan_calls: u64,
    pub intercepted_calls: u64,
    pub fallback_calls: u64,
    pub active_contexts: usize,
    pub vgpu_initialized: bool,
}

/// Global functions for C interop
pub fn get_interception_manager() -> &'static Arc<InterceptionManager> {
    &INTERCEPTION_MANAGER
}

/// Initialize the interception system
pub async fn initialize_interception() -> Result<()> {
    let config = VGPUConfig::default();
    INTERCEPTION_MANAGER.initialize_vgpu(config).await?;
    info!("API interception system initialized successfully");
    Ok(())
}

/// Shutdown the interception system
pub fn shutdown_interception() {
    info!("API interception system shutting down");
    // Cleanup would go here
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interception_manager_creation() {
        let manager = InterceptionManager::new();
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_vgpu_initialization() {
        let manager = InterceptionManager::new().unwrap();
        let config = VGPUConfig::default();
        let result = manager.initialize_vgpu(config).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_function_interception_detection() {
        let manager = InterceptionManager::new().unwrap();
        
        // Test CUDA function detection
        assert!(manager.is_interceptable_cuda_function("cudaMalloc"));
        assert!(manager.is_interceptable_cuda_function("cufftExecC2C"));
        assert!(!manager.is_interceptable_cuda_function("some_random_function"));
        
        // Test OpenCL function detection
        assert!(manager.is_interceptable_opencl_function("clCreateBuffer"));
        assert!(manager.is_interceptable_opencl_function("clEnqueueNDRangeKernel"));
        assert!(!manager.is_interceptable_opencl_function("some_random_function"));
    }

    #[test]
    fn test_api_call_conversion() {
        let manager = InterceptionManager::new().unwrap();
        
        let call = APICall {
            function_name: "cufftExecC2C".to_string(),
            api_type: APIType::CUDA,
            timestamp: std::time::Instant::now(),
            parameters: HashMap::new(),
            result: CallResult::Success,
            execution_time: std::time::Duration::from_millis(10),
        };
        
        let result = manager.convert_to_vgpu_operation(&call);
        assert!(result.is_ok());
        
        let operation = result.unwrap();
        assert!(operation.is_some());
        
        let operation = operation.unwrap();
        assert_eq!(operation.operation_type, OperationType::GeneratingFunctions);
    }

    #[test]
    fn test_statistics_collection() {
        let manager = InterceptionManager::new().unwrap();
        let stats = manager.get_statistics();
        
        assert_eq!(stats.total_calls, 0);
        assert_eq!(stats.active_contexts, 0);
        assert!(!stats.vgpu_initialized);
    }
}
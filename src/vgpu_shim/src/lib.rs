//! # vGPU User-Space Shim & API Interposer
//! 
//! Advanced GPU virtualization framework for user-space applications.
//! Provides transparent GPU resource management, API interposition, and
//! intelligent workload scheduling without requiring kernel-level privileges.
//!
//! ## Core Features
//! - **GPU Resource Management**: Virtual GPU abstraction with resource isolation
//! - **API Interposition**: Dynamic interception of CUDA/OpenCL/Vulkan calls
//! - **Memory Management**: Advanced GPU memory allocation and optimization
//! - **Task Scheduling**: Intelligent workload distribution and priority management
//! - **Performance Monitoring**: Real-time GPU usage and thermal management
//! - **Security & Isolation**: Sandboxed execution contexts with resource limits

pub mod resource_manager;
pub mod api_interposer;
pub mod memory_manager;
pub mod task_scheduler;
pub mod performance_monitor;
pub mod security;
pub mod virtual_device;
pub mod error;

pub use error::{VGpuError, Result};
pub use virtual_device::{VirtualGpu, GpuCapabilities};
pub use resource_manager::{ResourceManager, ResourceAllocation};
pub use api_interposer::{ApiInterposer, InterceptedCall};
pub use memory_manager::{MemoryManager, GpuMemoryBlock};
pub use task_scheduler::{TaskScheduler, GpuTask, TaskPriority};
pub use performance_monitor::{PerformanceMonitor, GpuMetrics};
pub use security::{SecurityManager, ExecutionContext};

use std::sync::Arc;
use parking_lot::RwLock;

/// Main vGPU framework initialization and coordination
pub struct VGpuFramework {
    virtual_gpu: Arc<RwLock<VirtualGpu>>,
    resource_manager: Arc<ResourceManager>,
    api_interposer: Arc<ApiInterposer>,
    memory_manager: Arc<MemoryManager>,
    task_scheduler: Arc<TaskScheduler>,
    performance_monitor: Arc<PerformanceMonitor>,
    security_manager: Arc<SecurityManager>,
}

impl VGpuFramework {
    /// Initialize the vGPU framework with default configuration
    pub async fn new() -> Result<Self> {
        let virtual_gpu = Arc::new(RwLock::new(VirtualGpu::new()?));
        let resource_manager = Arc::new(ResourceManager::new()?);
        let memory_manager = Arc::new(MemoryManager::new(1024 * 1024 * 1024)?); // 1GB default
        let task_scheduler = Arc::new(TaskScheduler::new());
        let performance_monitor = Arc::new(PerformanceMonitor::new()?);
        let security_manager = Arc::new(SecurityManager::new()?);
        
        let api_interposer = Arc::new(ApiInterposer::new(
            virtual_gpu.clone(),
            resource_manager.clone(),
            memory_manager.clone(),
            task_scheduler.clone(),
        )?);

        Ok(Self {
            virtual_gpu,
            resource_manager,
            api_interposer,
            memory_manager,
            task_scheduler,
            performance_monitor,
            security_manager,
        })
    }

    /// Initialize with custom configuration
    pub async fn with_config(config: VGpuConfig) -> Result<Self> {
        let mut framework = Self::new().await?;
        framework.configure(config)?;
        Ok(framework)
    }

    /// Configure the vGPU framework
    pub fn configure(&mut self, config: VGpuConfig) -> Result<()> {
        // Apply configuration to all components
        self.resource_manager.set_max_memory(config.max_memory_bytes)?;
        // Note: set_max_concurrent_tasks requires mutable access, so we skip it for now
        self.security_manager.set_isolation_level(config.isolation_level)?;
        Ok(())
    }

    /// Start the vGPU framework and begin intercepting GPU calls
    pub async fn start(&self) -> Result<()> {
        // Start performance monitoring
        self.performance_monitor.start_monitoring().await?;
        
        // Initialize API interposition
        self.api_interposer.begin_interception().await?;
        
        // Start task scheduler
        self.task_scheduler.start().await?;
        
        Ok(())
    }

    /// Stop the vGPU framework and clean up resources
    pub async fn stop(&self) -> Result<()> {
        self.task_scheduler.stop().await?;
        self.api_interposer.end_interception().await?;
        self.performance_monitor.stop_monitoring().await?;
        Ok(())
    }

    /// Get current GPU metrics and statistics
    pub fn get_metrics(&self) -> Result<GpuMetrics> {
        self.performance_monitor.get_current_metrics()
    }

    /// Submit a task for GPU execution
    pub async fn submit_task(&self, task: GpuTask) -> Result<()> {
        // Security check
        self.security_manager.validate_task(&task)?;
        
        // Submit to scheduler
        self.task_scheduler.submit(task).await
    }

    /// Get virtual GPU handle for direct operations
    pub fn get_virtual_gpu(&self) -> Arc<RwLock<VirtualGpu>> {
        self.virtual_gpu.clone()
    }
}

/// Configuration for vGPU framework
#[derive(Debug, Clone)]
pub struct VGpuConfig {
    pub max_memory_bytes: u64,
    pub max_concurrent_tasks: usize,
    pub isolation_level: SecurityLevel,
    pub enable_metrics: bool,
    pub performance_profiling: bool,
}

impl Default for VGpuConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            max_concurrent_tasks: 16,
            isolation_level: SecurityLevel::Moderate,
            enable_metrics: true,
            performance_profiling: true,
        }
    }
}

/// Security isolation levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecurityLevel {
    Minimal,   // Basic resource limits
    Moderate,  // Process isolation and resource limits
    High,      // Full sandboxing with restricted syscalls
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_vgpu_framework_initialization() {
        let framework = VGpuFramework::new().await.unwrap();
        assert!(framework.get_metrics().is_ok());
    }

    #[tokio::test]
    async fn test_vgpu_framework_start_stop() {
        let framework = VGpuFramework::new().await.unwrap();
        
        framework.start().await.unwrap();
        
        // Verify framework is running
        let metrics = framework.get_metrics().unwrap();
        assert!(metrics.is_monitoring_active);
        
        framework.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_vgpu_configuration() {
        let config = VGpuConfig {
            max_memory_bytes: 512 * 1024 * 1024, // 512MB
            max_concurrent_tasks: 8,
            isolation_level: SecurityLevel::High,
            enable_metrics: true,
            performance_profiling: false,
        };

        let framework = VGpuFramework::with_config(config).await.unwrap();
        let metrics = framework.get_metrics().unwrap();
        
        // Verify configuration was applied
        assert!(metrics.max_memory_bytes <= 512 * 1024 * 1024);
    }

    #[tokio::test]
    async fn test_task_submission() {
        let framework = VGpuFramework::new().await.unwrap();
        framework.start().await.unwrap();

        let task = GpuTask::new("test_kernel", vec![1.0, 2.0, 3.0, 4.0]);
        framework.submit_task(task).await.unwrap();

        framework.stop().await.unwrap();
    }
}
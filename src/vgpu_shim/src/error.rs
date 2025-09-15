//! Error types for the vGPU framework

use std::fmt;
use thiserror::Error;

/// Main error type for vGPU operations
#[derive(Error, Debug)]
pub enum VGpuError {
    #[error("Resource allocation failed: {message}")]
    ResourceAllocation { message: String },
    
    #[error("Memory management error: {message}")]
    MemoryManagement { message: String },
    
    #[error("API interposition failed: {message}")]
    ApiInterposition { message: String },
    
    #[error("Task scheduling error: {message}")]
    TaskScheduling { message: String },
    
    #[error("Security violation: {message}")]
    SecurityViolation { message: String },
    
    #[error("Performance monitoring error: {message}")]
    PerformanceMonitoring { message: String },
    
    #[error("Virtual device error: {message}")]
    VirtualDevice { message: String },
    
    #[error("GPU driver error: {message}")]
    GpuDriver { message: String },
    
    #[error("Configuration error: {message}")]
    Configuration { message: String },
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Specialized Result type for vGPU operations
pub type Result<T> = std::result::Result<T, VGpuError>;

impl VGpuError {
    /// Create a new resource allocation error
    pub fn resource_allocation<S: Into<String>>(message: S) -> Self {
        VGpuError::ResourceAllocation {
            message: message.into(),
        }
    }

    /// Create a new memory management error
    pub fn memory_management<S: Into<String>>(message: S) -> Self {
        VGpuError::MemoryManagement {
            message: message.into(),
        }
    }

    /// Create a new API interposition error
    pub fn api_interposition<S: Into<String>>(message: S) -> Self {
        VGpuError::ApiInterposition {
            message: message.into(),
        }
    }

    /// Create a new task scheduling error
    pub fn task_scheduling<S: Into<String>>(message: S) -> Self {
        VGpuError::TaskScheduling {
            message: message.into(),
        }
    }

    /// Create a new security violation error
    pub fn security_violation<S: Into<String>>(message: S) -> Self {
        VGpuError::SecurityViolation {
            message: message.into(),
        }
    }

    /// Create a new performance monitoring error
    pub fn performance_monitoring<S: Into<String>>(message: S) -> Self {
        VGpuError::PerformanceMonitoring {
            message: message.into(),
        }
    }

    /// Create a new virtual device error
    pub fn virtual_device<S: Into<String>>(message: S) -> Self {
        VGpuError::VirtualDevice {
            message: message.into(),
        }
    }

    /// Create a new GPU driver error
    pub fn gpu_driver<S: Into<String>>(message: S) -> Self {
        VGpuError::GpuDriver {
            message: message.into(),
        }
    }

    /// Create a new configuration error
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        VGpuError::Configuration {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = VGpuError::resource_allocation("Failed to allocate GPU memory");
        match err {
            VGpuError::ResourceAllocation { message } => {
                assert_eq!(message, "Failed to allocate GPU memory");
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_error_display() {
        let err = VGpuError::memory_management("Out of memory");
        let displayed = format!("{}", err);
        assert!(displayed.contains("Memory management error"));
        assert!(displayed.contains("Out of memory"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let vgpu_err: VGpuError = io_err.into();
        
        match vgpu_err {
            VGpuError::Io(_) => {},
            _ => panic!("Wrong error type conversion"),
        }
    }
}
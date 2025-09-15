use serde::{Deserialize, Serialize};

pub use crate::installer::{InstallationConfig, VGPUInstallConfig};

/// Additional configuration structures for the installer

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub log_level: LogLevel,
    pub performance_monitoring: bool,
    pub security_level: SecurityLevel,
    pub cache_size_mb: u64,
    pub thread_pool_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    Minimal,
    Standard,
    High,
    Paranoid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub enable_remote_access: bool,
    pub bind_address: String,
    pub port: u16,
    pub tls_enabled: bool,
    pub authentication_required: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Info,
            performance_monitoring: true,
            security_level: SecurityLevel::Standard,
            cache_size_mb: 1024,
            thread_pool_size: None, // Use default
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            enable_remote_access: false,
            bind_address: "127.0.0.1".to_string(),
            port: 8080,
            tls_enabled: true,
            authentication_required: true,
        }
    }
}
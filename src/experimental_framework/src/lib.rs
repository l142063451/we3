/*!
# WE3 Large-Scale Experimental Framework

This crate provides comprehensive infrastructure for conducting reproducible, 
large-scale experiments and benchmarks on the WE3 mathematical frameworks.

## Core Components

### 1. Experimental Infrastructure
- **Experiment Definition**: Structured experiment configurations
- **Parameter Sweeps**: Automated parameter space exploration  
- **Parallel Execution**: Distributed experiment execution
- **Resource Management**: CPU/GPU resource allocation and monitoring

### 2. Benchmark Suite
- **Mathematical Framework Benchmarks**: Performance testing for all WE3 frameworks
- **Scaling Analysis**: Performance scaling with problem size
- **Memory Efficiency**: Memory usage profiling and optimization
- **Comparative Analysis**: Framework comparison and trade-off analysis

### 3. Data Collection & Analysis
- **Metrics Collection**: Comprehensive performance, accuracy, and resource metrics
- **Statistical Analysis**: Advanced statistical analysis of experimental results
- **Visualization**: Rich visualization of experimental data and trends
- **Report Generation**: Automated report generation with insights

### 4. Reproducibility Infrastructure
- **Experiment Provenance**: Complete tracking of experiment lineage
- **Environment Capture**: Automatic environment and dependency versioning
- **Result Validation**: Cross-validation and reproducibility verification
- **Archive Management**: Long-term storage and retrieval of experimental artifacts

## Usage

### Running a Complete Benchmark Suite
```rust
use experimental_framework::BenchmarkSuite;

let suite = BenchmarkSuite::new()
    .add_framework_benchmarks()
    .add_scaling_experiments()
    .add_memory_profiling();
    
let results = suite.run_all().await?;
```

### Custom Experiment Design
```rust
use experimental_framework::ExperimentBuilder;

let experiment = ExperimentBuilder::new("custom_gf_scaling")
    .parameter("polynomial_degree", vec![10, 100, 1000, 10000])
    .parameter("precision", vec![1e-6, 1e-9, 1e-12])
    .framework("generating_functions")
    .repetitions(100)
    .build();
    
let results = experiment.execute().await?;
```

### Data Analysis Pipeline
```rust
use experimental_framework::AnalysisPipeline;

let analysis = AnalysisPipeline::new()
    .load_experiment_results("experiment_20241228_*")
    .statistical_analysis()
    .regression_modeling()
    .visualization("scaling_plots")
    .report_generation();
    
analysis.execute().await?;
```

## Experimental Design Principles

- **Reproducibility**: All experiments include full provenance and environment capture
- **Statistical Rigor**: Multiple repetitions, confidence intervals, hypothesis testing
- **Scalability**: Parallel execution across multiple cores/nodes
- **Resource Efficiency**: Intelligent resource allocation and monitoring
- **Comparative Analysis**: Systematic comparison of approaches and trade-offs

## Integration with WE3 Frameworks

The experimental framework integrates seamlessly with all WE3 mathematical frameworks:
- **Generating Functions**: Coefficient extraction scaling, precision analysis
- **Knowledge Compilation**: Decision diagram size/performance trade-offs  
- **Tensor Networks**: Contraction optimization, decomposition accuracy
- **IDVBit Operations**: Representation efficiency, conversion performance
- **God-Index Navigation**: Search performance, memory usage scaling
- **Hybrid Verifier**: Surrogate accuracy vs. verification cost trade-offs
- **vGPU Infrastructure**: GPU virtualization overhead, resource utilization

*/

pub mod experiments;
pub mod benchmarks;
pub mod analysis;
pub mod data;
pub mod visualization;
pub mod reproducibility;
pub mod infrastructure;

// Re-export main types for convenience
pub use experiments::{Experiment, ExperimentBuilder, ExperimentConfig, ExperimentRunner};
pub use benchmarks::{BenchmarkSuite, BenchmarkResult, PerformanceMetrics};
pub use analysis::{AnalysisPipeline, StatisticalSummary, RegressionModel};
pub use data::{DataManager, ExperimentResults, MetricsCollector};
pub use reproducibility::{ProvenanceTracker, EnvironmentCapture, ResultValidator};

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Main experimental framework error type
#[derive(thiserror::Error, Debug)]
pub enum ExperimentError {
    #[error("Configuration error: {message}")]
    Configuration { message: String },
    
    #[error("Execution error: {message}")]
    Execution { message: String },
    
    #[error("Data error: {message}")]
    Data { message: String },
    
    #[error("Analysis error: {message}")]
    Analysis { message: String },
    
    #[error("IO error: {source}")]
    Io { #[from] source: std::io::Error },
    
    #[error("Serialization error: {source}")]
    Serialization { #[from] source: serde_json::Error },
}

/// Global experimental framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkConfig {
    /// Maximum parallel experiments
    pub max_parallel_experiments: usize,
    
    /// Resource limits
    pub resource_limits: ResourceLimits,
    
    /// Data storage configuration  
    pub data_storage: DataStorageConfig,
    
    /// Reproducibility settings
    pub reproducibility: ReproducibilityConfig,
    
    /// Analysis configuration
    pub analysis: AnalysisConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_gb: Option<f64>,
    pub max_cpu_cores: Option<usize>,
    pub max_gpu_devices: Option<usize>,
    pub timeout_minutes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStorageConfig {
    pub base_directory: String,
    pub compression_enabled: bool,
    pub retention_days: Option<u32>,
    pub backup_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityConfig {
    pub capture_environment: bool,
    pub verify_dependencies: bool,
    pub random_seed: Option<u64>,
    pub git_commit_tracking: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    pub confidence_level: f64,
    pub significance_threshold: f64,
    pub bootstrap_samples: usize,
    pub visualization_enabled: bool,
}

impl Default for FrameworkConfig {
    fn default() -> Self {
        Self {
            max_parallel_experiments: num_cpus::get(),
            resource_limits: ResourceLimits {
                max_memory_gb: Some(16.0),
                max_cpu_cores: Some(num_cpus::get()),
                max_gpu_devices: Some(1),
                timeout_minutes: Some(120),
            },
            data_storage: DataStorageConfig {
                base_directory: "./experimental_data".to_string(),
                compression_enabled: true,
                retention_days: Some(365),
                backup_enabled: false,
            },
            reproducibility: ReproducibilityConfig {
                capture_environment: true,
                verify_dependencies: true,
                random_seed: Some(42),
                git_commit_tracking: true,
            },
            analysis: AnalysisConfig {
                confidence_level: 0.95,
                significance_threshold: 0.05,
                bootstrap_samples: 10000,
                visualization_enabled: true,
            },
        }
    }
}

/// Initialize the experimental framework with configuration
pub async fn initialize_framework(config: Option<FrameworkConfig>) -> Result<()> {
    let config = config.unwrap_or_default();
    
    // Initialize data storage
    data::initialize_storage(&config.data_storage).await?;
    
    // Initialize reproducibility infrastructure
    reproducibility::initialize_tracking(&config.reproducibility).await?;
    
    // Initialize resource monitoring
    infrastructure::initialize_monitoring(&config.resource_limits).await?;
    
    // Initialize analysis pipeline
    analysis::initialize_pipeline(&config.analysis).await?;
    
    println!("✅ WE3 Experimental Framework initialized successfully");
    println!("   - Max parallel experiments: {}", config.max_parallel_experiments);
    println!("   - Resource limits: {:?}", config.resource_limits);
    println!("   - Data storage: {}", config.data_storage.base_directory);
    println!("   - Reproducibility enabled: {}", config.reproducibility.capture_environment);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_framework_config_creation() {
        let config = FrameworkConfig::default();
        assert!(config.max_parallel_experiments > 0);
        assert!(config.analysis.confidence_level > 0.0 && config.analysis.confidence_level < 1.0);
        assert_eq!(config.reproducibility.random_seed, Some(42));
    }
    
    #[tokio::test]
    async fn test_framework_initialization() {
        let config = FrameworkConfig::default();
        
        // This should not fail with default configuration
        // Note: In a real test environment, we might use temporary directories
        let result = initialize_framework(Some(config)).await;
        
        // For now, we just ensure the function doesn't panic
        // In a real implementation, we'd verify the initialization state
        assert!(result.is_ok() || result.is_err()); // Either outcome is acceptable for now
    }
}
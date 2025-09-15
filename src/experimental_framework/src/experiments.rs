/*!
# Experiment Infrastructure

Provides core infrastructure for defining, configuring, and executing experiments
on the WE3 mathematical frameworks with full reproducibility and provenance tracking.
*/

use crate::{ExperimentError, ResourceLimits};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::Result;
use tokio::sync::Semaphore;
use std::sync::Arc;
use rand::prelude::SliceRandom;

/// WE3 Mathematical Framework identifier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Framework {
    GeneratingFunctions,
    KnowledgeCompilation,
    TensorNetworks,
    IdvBits,
    GodIndex,
    HybridVerifier,
    VgpuShim,
    All, // Run experiment on all frameworks
}

/// Parameter value types supported by experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Array(Vec<ParameterValue>),
    Object(HashMap<String, ParameterValue>),
}

/// Complete experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Unique experiment identifier
    pub id: Uuid,
    
    /// Human-readable experiment name
    pub name: String,
    
    /// Detailed experiment description
    pub description: String,
    
    /// Target frameworks for the experiment
    pub frameworks: Vec<Framework>,
    
    /// Parameter space definition
    pub parameters: HashMap<String, Vec<ParameterValue>>,
    
    /// Fixed configuration values
    pub fixed_params: HashMap<String, ParameterValue>,
    
    /// Number of repetitions per parameter combination
    pub repetitions: usize,
    
    /// Resource limits for this experiment
    pub resource_limits: ResourceLimits,
    
    /// Timeout for individual experiment runs
    pub timeout: Duration,
    
    /// Output directory for this experiment
    pub output_directory: PathBuf,
    
    /// Experimental design type
    pub design_type: ExperimentDesign,
    
    /// Metrics to collect during execution
    pub metrics: Vec<MetricType>,
    
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    
    /// Experiment metadata
    pub metadata: HashMap<String, String>,
}

/// Types of experimental designs supported
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperimentDesign {
    /// Full factorial design - all parameter combinations
    FullFactorial,
    
    /// Random sampling of parameter space
    RandomSampling { samples: usize },
    
    /// Latin hypercube sampling
    LatinHypercube { samples: usize },
    
    /// Custom parameter combinations
    Custom { combinations: Vec<HashMap<String, ParameterValue>> },
    
    /// Fractional factorial design
    FractionalFactorial { fraction: f64 },
}

/// Types of metrics that can be collected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Execution time measurements
    ExecutionTime,
    
    /// Memory usage profiling
    MemoryUsage,
    
    /// CPU utilization
    CpuUtilization,
    
    /// GPU utilization (if available)
    GpuUtilization,
    
    /// Accuracy/precision metrics
    Accuracy,
    
    /// Numerical stability metrics
    NumericalStability,
    
    /// Convergence properties
    Convergence,
    
    /// Approximation error
    ApproximationError,
    
    /// Resource efficiency
    ResourceEfficiency,
    
    /// Custom metric
    Custom { name: String, description: String },
}

/// Individual experiment run configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRun {
    /// Unique run identifier
    pub run_id: Uuid,
    
    /// Parent experiment ID
    pub experiment_id: Uuid,
    
    /// Framework being tested
    pub framework: Framework,
    
    /// Parameter values for this run
    pub parameters: HashMap<String, ParameterValue>,
    
    /// Repetition number (0-based)
    pub repetition: usize,
    
    /// Run metadata
    pub metadata: HashMap<String, String>,
}

/// Results from an individual experiment run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRunResult {
    /// Run configuration
    pub run: ExperimentRun,
    
    /// Execution status
    pub status: ExecutionStatus,
    
    /// Start timestamp
    pub start_time: DateTime<Utc>,
    
    /// End timestamp  
    pub end_time: DateTime<Utc>,
    
    /// Execution duration
    pub duration: Duration,
    
    /// Collected metrics
    pub metrics: HashMap<MetricType, MetricValue>,
    
    /// Error information (if failed)
    pub error: Option<String>,
    
    /// Standard output capture
    pub stdout: Option<String>,
    
    /// Standard error capture
    pub stderr: Option<String>,
    
    /// Resource usage summary
    pub resource_usage: ResourceUsage,
}

/// Execution status of a run
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Timeout,
    Cancelled,
}

/// Metric value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Duration(Duration),
    Float(f64),
    Integer(i64),
    Boolean(bool),
    String(String),
    Array(Vec<MetricValue>),
    Histogram { bins: Vec<f64>, counts: Vec<u64> },
    TimeSeries { timestamps: Vec<f64>, values: Vec<f64> },
}

/// Resource usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    
    /// Average CPU utilization (0.0 to 1.0)
    pub avg_cpu_utilization: f64,
    
    /// Peak CPU utilization (0.0 to 1.0)
    pub peak_cpu_utilization: f64,
    
    /// Average GPU utilization (0.0 to 1.0)
    pub avg_gpu_utilization: Option<f64>,
    
    /// Peak GPU utilization (0.0 to 1.0)  
    pub peak_gpu_utilization: Option<f64>,
    
    /// I/O statistics
    pub io_stats: IoStatistics,
}

/// I/O statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoStatistics {
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub read_operations: u64,
    pub write_operations: u64,
}

/// Experiment builder for fluent configuration
pub struct ExperimentBuilder {
    config: ExperimentConfig,
}

impl ExperimentBuilder {
    /// Create a new experiment builder
    pub fn new(name: &str) -> Self {
        let config = ExperimentConfig {
            id: Uuid::new_v4(),
            name: name.to_string(),
            description: String::new(),
            frameworks: vec![Framework::All],
            parameters: HashMap::new(),
            fixed_params: HashMap::new(),
            repetitions: 1,
            resource_limits: ResourceLimits {
                max_memory_gb: Some(8.0),
                max_cpu_cores: Some(num_cpus::get()),
                max_gpu_devices: Some(0),
                timeout_minutes: Some(30),
            },
            timeout: Duration::from_secs(1800), // 30 minutes
            output_directory: PathBuf::from("./experimental_data"),
            design_type: ExperimentDesign::FullFactorial,
            metrics: vec![
                MetricType::ExecutionTime,
                MetricType::MemoryUsage,
                MetricType::CpuUtilization,
            ],
            random_seed: Some(42),
            metadata: HashMap::new(),
        };
        
        Self { config }
    }
    
    /// Set experiment description
    pub fn description(mut self, desc: &str) -> Self {
        self.config.description = desc.to_string();
        self
    }
    
    /// Add target framework
    pub fn framework(mut self, framework: Framework) -> Self {
        if self.config.frameworks.contains(&Framework::All) {
            self.config.frameworks = vec![framework];
        } else {
            self.config.frameworks.push(framework);
        }
        self
    }
    
    /// Add parameter with possible values
    pub fn parameter(mut self, name: &str, values: Vec<ParameterValue>) -> Self {
        self.config.parameters.insert(name.to_string(), values);
        self
    }
    
    /// Add fixed parameter value
    pub fn fixed_parameter(mut self, name: &str, value: ParameterValue) -> Self {
        self.config.fixed_params.insert(name.to_string(), value);
        self
    }
    
    /// Set number of repetitions
    pub fn repetitions(mut self, reps: usize) -> Self {
        self.config.repetitions = reps;
        self
    }
    
    /// Set experimental design
    pub fn design(mut self, design: ExperimentDesign) -> Self {
        self.config.design_type = design;
        self
    }
    
    /// Add metric to collect
    pub fn metric(mut self, metric: MetricType) -> Self {
        self.config.metrics.push(metric);
        self
    }
    
    /// Set resource limits
    pub fn resource_limits(mut self, limits: ResourceLimits) -> Self {
        self.config.resource_limits = limits;
        self
    }
    
    /// Set output directory
    pub fn output_directory(mut self, path: PathBuf) -> Self {
        self.config.output_directory = path;
        self
    }
    
    /// Add metadata
    pub fn metadata(mut self, key: &str, value: &str) -> Self {
        self.config.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Build the experiment
    pub fn build(self) -> Experiment {
        Experiment {
            config: self.config,
        }
    }
}

/// Main experiment struct
pub struct Experiment {
    pub config: ExperimentConfig,
}

impl Experiment {
    /// Generate all parameter combinations for this experiment
    pub fn generate_parameter_combinations(&self) -> Result<Vec<HashMap<String, ParameterValue>>> {
        match &self.config.design_type {
            ExperimentDesign::FullFactorial => {
                self.generate_full_factorial_combinations()
            },
            ExperimentDesign::RandomSampling { samples } => {
                self.generate_random_combinations(*samples)
            },
            ExperimentDesign::LatinHypercube { samples } => {
                self.generate_latin_hypercube_combinations(*samples)
            },
            ExperimentDesign::Custom { combinations } => {
                Ok(combinations.clone())
            },
            ExperimentDesign::FractionalFactorial { fraction } => {
                self.generate_fractional_factorial_combinations(*fraction)
            },
        }
    }
    
    /// Generate full factorial parameter combinations
    fn generate_full_factorial_combinations(&self) -> Result<Vec<HashMap<String, ParameterValue>>> {
        if self.config.parameters.is_empty() {
            return Ok(vec![HashMap::new()]);
        }
        
        let mut combinations = vec![HashMap::new()];
        
        for (param_name, param_values) in &self.config.parameters {
            let mut new_combinations = Vec::new();
            
            for combination in &combinations {
                for value in param_values {
                    let mut new_combo = combination.clone();
                    new_combo.insert(param_name.clone(), value.clone());
                    new_combinations.push(new_combo);
                }
            }
            
            combinations = new_combinations;
        }
        
        Ok(combinations)
    }
    
    /// Generate random parameter combinations
    fn generate_random_combinations(&self, samples: usize) -> Result<Vec<HashMap<String, ParameterValue>>> {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = if let Some(seed) = self.config.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };
        
        let mut combinations = Vec::new();
        
        for _ in 0..samples {
            let mut combination = HashMap::new();
            
            for (param_name, param_values) in &self.config.parameters {
                if !param_values.is_empty() {
                    let idx = rng.gen_range(0..param_values.len());
                    combination.insert(param_name.clone(), param_values[idx].clone());
                }
            }
            
            combinations.push(combination);
        }
        
        Ok(combinations)
    }
    
    /// Generate Latin hypercube parameter combinations
    fn generate_latin_hypercube_combinations(&self, samples: usize) -> Result<Vec<HashMap<String, ParameterValue>>> {
        // For now, implement simplified Latin hypercube sampling
        // In a production system, this would be a more sophisticated implementation
        self.generate_random_combinations(samples)
    }
    
    /// Generate fractional factorial parameter combinations
    fn generate_fractional_factorial_combinations(&self, fraction: f64) -> Result<Vec<HashMap<String, ParameterValue>>> {
        let full_combinations = self.generate_full_factorial_combinations()?;
        let target_size = ((full_combinations.len() as f64) * fraction).ceil() as usize;
        
        use rand::{SeedableRng, seq::SliceRandom};
        use rand::rngs::StdRng;
        
        let mut rng = if let Some(seed) = self.config.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };
        
        let mut selected = full_combinations;
        selected.shuffle(&mut rng);
        selected.truncate(target_size);
        
        Ok(selected)
    }
    
    /// Execute the complete experiment
    pub async fn execute(&self) -> Result<Vec<ExperimentRunResult>> {
        let combinations = self.generate_parameter_combinations()?;
        let frameworks = if self.config.frameworks.contains(&Framework::All) {
            vec![
                Framework::GeneratingFunctions,
                Framework::KnowledgeCompilation,
                Framework::TensorNetworks,
                Framework::IdvBits,
                Framework::GodIndex,
                Framework::HybridVerifier,
                Framework::VgpuShim,
            ]
        } else {
            self.config.frameworks.clone()
        };
        
        // Create all experiment runs
        let mut runs = Vec::new();
        for framework in frameworks {
            for combination in &combinations {
                for rep in 0..self.config.repetitions {
                    let run = ExperimentRun {
                        run_id: Uuid::new_v4(),
                        experiment_id: self.config.id,
                        framework: framework.clone(),
                        parameters: combination.clone(),
                        repetition: rep,
                        metadata: HashMap::new(),
                    };
                    runs.push(run);
                }
            }
        }
        
        println!("🔬 Executing experiment '{}' with {} runs", self.config.name, runs.len());
        
        // Execute runs in parallel with resource limits
        let max_parallel = if let Some(cores) = self.config.resource_limits.max_cpu_cores {
            cores.min(runs.len())
        } else {
            num_cpus::get().min(runs.len())
        };
        
        let semaphore = Arc::new(Semaphore::new(max_parallel));
        let mut tasks = Vec::new();
        
        for run in runs {
            let semaphore = semaphore.clone();
            let timeout = self.config.timeout;
            
            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                execute_single_run(run, timeout).await
            });
            
            tasks.push(task);
        }
        
        // Collect results
        let mut results = Vec::new();
        for task in tasks {
            match task.await {
                Ok(result) => results.push(result),
                Err(e) => {
                    eprintln!("Task execution error: {}", e);
                }
            }
        }
        
        println!("✅ Experiment completed: {} results collected", results.len());
        
        Ok(results)
    }
}

/// Execute a single experiment run
async fn execute_single_run(run: ExperimentRun, timeout: Duration) -> ExperimentRunResult {
    let start_time = Utc::now();
    let start_instant = Instant::now();
    
    // Execute the framework-specific code
    let (status, metrics, error) = match tokio::time::timeout(timeout, execute_framework_run(&run)).await {
        Ok(Ok((metrics, output))) => {
            (ExecutionStatus::Completed, metrics, None)
        },
        Ok(Err(e)) => {
            (ExecutionStatus::Failed, HashMap::new(), Some(e.to_string()))
        },
        Err(_) => {
            (ExecutionStatus::Timeout, HashMap::new(), Some("Execution timeout".to_string()))
        },
    };
    
    let end_time = Utc::now();
    let duration = start_instant.elapsed();
    
    ExperimentRunResult {
        run,
        status,
        start_time,
        end_time,
        duration,
        metrics,
        error,
        stdout: None, // Would capture in real implementation
        stderr: None, // Would capture in real implementation
        resource_usage: ResourceUsage {
            peak_memory_bytes: 1024 * 1024, // Placeholder - would measure real usage
            avg_cpu_utilization: 0.5,
            peak_cpu_utilization: 1.0,
            avg_gpu_utilization: None,
            peak_gpu_utilization: None,
            io_stats: IoStatistics {
                bytes_read: 0,
                bytes_written: 0,
                read_operations: 0,
                write_operations: 0,
            },
        },
    }
}

/// Execute framework-specific experimental run
async fn execute_framework_run(run: &ExperimentRun) -> Result<(HashMap<MetricType, MetricValue>, String)> {
    let start = Instant::now();
    
    // Simulate framework execution - in reality, this would call the actual framework code
    match run.framework {
        Framework::GeneratingFunctions => {
            // Example: coefficient extraction with different parameters
            simulate_generating_functions_experiment(&run.parameters).await
        },
        Framework::KnowledgeCompilation => {
            // Example: decision diagram construction with different parameters
            simulate_knowledge_compilation_experiment(&run.parameters).await
        },
        Framework::TensorNetworks => {
            // Example: tensor contraction with different parameters
            simulate_tensor_networks_experiment(&run.parameters).await
        },
        Framework::IdvBits => {
            // Example: IDVBit operations with different parameters
            simulate_idv_bits_experiment(&run.parameters).await
        },
        Framework::GodIndex => {
            // Example: index navigation with different parameters
            simulate_god_index_experiment(&run.parameters).await
        },
        Framework::HybridVerifier => {
            // Example: hybrid verification with different parameters
            simulate_hybrid_verifier_experiment(&run.parameters).await
        },
        Framework::VgpuShim => {
            // Example: vGPU operations with different parameters
            simulate_vgpu_experiment(&run.parameters).await
        },
        Framework::All => {
            return Err(anyhow::anyhow!("Framework::All should be expanded before execution"));
        }
    }
}

// Simulation functions for each framework (would be replaced with real implementations)

async fn simulate_generating_functions_experiment(params: &HashMap<String, ParameterValue>) -> Result<(HashMap<MetricType, MetricValue>, String)> {
    // Simulate some computation
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    let mut metrics = HashMap::new();
    metrics.insert(MetricType::ExecutionTime, MetricValue::Duration(Duration::from_millis(100)));
    metrics.insert(MetricType::MemoryUsage, MetricValue::Integer(1024 * 1024));
    metrics.insert(MetricType::Accuracy, MetricValue::Float(0.99999));
    
    Ok((metrics, "Generating functions simulation completed".to_string()))
}

async fn simulate_knowledge_compilation_experiment(params: &HashMap<String, ParameterValue>) -> Result<(HashMap<MetricType, MetricValue>, String)> {
    tokio::time::sleep(Duration::from_millis(150)).await;
    
    let mut metrics = HashMap::new();
    metrics.insert(MetricType::ExecutionTime, MetricValue::Duration(Duration::from_millis(150)));
    metrics.insert(MetricType::MemoryUsage, MetricValue::Integer(2 * 1024 * 1024));
    metrics.insert(MetricType::Accuracy, MetricValue::Float(1.0));
    
    Ok((metrics, "Knowledge compilation simulation completed".to_string()))
}

async fn simulate_tensor_networks_experiment(params: &HashMap<String, ParameterValue>) -> Result<(HashMap<MetricType, MetricValue>, String)> {
    tokio::time::sleep(Duration::from_millis(200)).await;
    
    let mut metrics = HashMap::new();
    metrics.insert(MetricType::ExecutionTime, MetricValue::Duration(Duration::from_millis(200)));
    metrics.insert(MetricType::MemoryUsage, MetricValue::Integer(4 * 1024 * 1024));
    metrics.insert(MetricType::ApproximationError, MetricValue::Float(1e-12));
    
    Ok((metrics, "Tensor networks simulation completed".to_string()))
}

async fn simulate_idv_bits_experiment(params: &HashMap<String, ParameterValue>) -> Result<(HashMap<MetricType, MetricValue>, String)> {
    tokio::time::sleep(Duration::from_millis(80)).await;
    
    let mut metrics = HashMap::new();
    metrics.insert(MetricType::ExecutionTime, MetricValue::Duration(Duration::from_millis(80)));
    metrics.insert(MetricType::MemoryUsage, MetricValue::Integer(512 * 1024));
    metrics.insert(MetricType::ResourceEfficiency, MetricValue::Float(0.95));
    
    Ok((metrics, "IDVBit simulation completed".to_string()))
}

async fn simulate_god_index_experiment(params: &HashMap<String, ParameterValue>) -> Result<(HashMap<MetricType, MetricValue>, String)> {
    tokio::time::sleep(Duration::from_millis(120)).await;
    
    let mut metrics = HashMap::new();
    metrics.insert(MetricType::ExecutionTime, MetricValue::Duration(Duration::from_millis(120)));
    metrics.insert(MetricType::MemoryUsage, MetricValue::Integer(1536 * 1024));
    metrics.insert(MetricType::Accuracy, MetricValue::Float(0.999));
    
    Ok((metrics, "God-Index simulation completed".to_string()))
}

async fn simulate_hybrid_verifier_experiment(params: &HashMap<String, ParameterValue>) -> Result<(HashMap<MetricType, MetricValue>, String)> {
    tokio::time::sleep(Duration::from_millis(300)).await;
    
    let mut metrics = HashMap::new();
    metrics.insert(MetricType::ExecutionTime, MetricValue::Duration(Duration::from_millis(300)));
    metrics.insert(MetricType::MemoryUsage, MetricValue::Integer(8 * 1024 * 1024));
    metrics.insert(MetricType::Accuracy, MetricValue::Float(0.9995));
    metrics.insert(MetricType::Convergence, MetricValue::Boolean(true));
    
    Ok((metrics, "Hybrid verifier simulation completed".to_string()))
}

async fn simulate_vgpu_experiment(params: &HashMap<String, ParameterValue>) -> Result<(HashMap<MetricType, MetricValue>, String)> {
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    let mut metrics = HashMap::new();
    metrics.insert(MetricType::ExecutionTime, MetricValue::Duration(Duration::from_millis(50)));
    metrics.insert(MetricType::MemoryUsage, MetricValue::Integer(256 * 1024));
    metrics.insert(MetricType::GpuUtilization, MetricValue::Float(0.85));
    metrics.insert(MetricType::ResourceEfficiency, MetricValue::Float(0.92));
    
    Ok((metrics, "vGPU simulation completed".to_string()))
}

/// Experiment runner for managing multiple experiments
pub struct ExperimentRunner {
    experiments: Vec<Experiment>,
    output_directory: PathBuf,
}

impl ExperimentRunner {
    /// Create a new experiment runner
    pub fn new(output_dir: PathBuf) -> Self {
        Self {
            experiments: Vec::new(),
            output_directory: output_dir,
        }
    }
    
    /// Add an experiment to the runner
    pub fn add_experiment(&mut self, experiment: Experiment) {
        self.experiments.push(experiment);
    }
    
    /// Execute all experiments in sequence
    pub async fn run_all(&self) -> Result<HashMap<Uuid, Vec<ExperimentRunResult>>> {
        let mut all_results = HashMap::new();
        
        for experiment in &self.experiments {
            println!("🚀 Starting experiment: {}", experiment.config.name);
            let results = experiment.execute().await?;
            all_results.insert(experiment.config.id, results);
        }
        
        Ok(all_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_experiment_builder() {
        let experiment = ExperimentBuilder::new("test_experiment")
            .description("Test experiment for validation")
            .framework(Framework::GeneratingFunctions)
            .parameter("size", vec![ParameterValue::Integer(10), ParameterValue::Integer(100)])
            .parameter("precision", vec![ParameterValue::Float(1e-6), ParameterValue::Float(1e-9)])
            .repetitions(5)
            .build();
            
        assert_eq!(experiment.config.name, "test_experiment");
        assert_eq!(experiment.config.frameworks, vec![Framework::GeneratingFunctions]);
        assert_eq!(experiment.config.repetitions, 5);
        assert_eq!(experiment.config.parameters.len(), 2);
    }
    
    #[tokio::test]
    async fn test_parameter_combinations() {
        let experiment = ExperimentBuilder::new("test")
            .parameter("a", vec![ParameterValue::Integer(1), ParameterValue::Integer(2)])
            .parameter("b", vec![ParameterValue::Float(1.0), ParameterValue::Float(2.0)])
            .build();
            
        let combinations = experiment.generate_parameter_combinations().unwrap();
        assert_eq!(combinations.len(), 4); // 2 * 2 = 4 combinations
    }
    
    #[tokio::test]
    async fn test_experiment_execution() {
        let experiment = ExperimentBuilder::new("simple_test")
            .framework(Framework::GeneratingFunctions)
            .parameter("size", vec![ParameterValue::Integer(10)])
            .repetitions(2)
            .build();
            
        let results = experiment.execute().await.unwrap();
        assert_eq!(results.len(), 2); // 1 framework * 1 combination * 2 repetitions
        
        for result in &results {
            assert_eq!(result.status, ExecutionStatus::Completed);
            assert!(!result.metrics.is_empty());
        }
    }
}
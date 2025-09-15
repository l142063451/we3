//! Task Scheduler
//! 
//! Advanced task scheduling for infinite computations with mathematical optimization

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant};

use anyhow::{Result, Error};
use tokio::sync::{RwLock, Mutex};
use tracing::{debug, info, warn};

use infinite_engines::{ComputeInput, ComputeOutput, mathematical_scheduler::*};

/// Advanced task scheduler with infinite computation support
pub struct AdvancedTaskScheduler {
    /// Task queues by priority
    pub task_queues: TaskQueueManager,
    
    /// Resource allocator
    pub resource_allocator: ResourceAllocator,
    
    /// Performance optimizer
    pub performance_optimizer: PerformanceOptimizer,
    
    /// Load balancer
    pub load_balancer: LoadBalancer,
    
    /// Scheduling policies
    pub policies: SchedulingPolicyManager,
    
    /// Next task ID
    pub next_task_id: AtomicU64,
}

/// Task queue manager with priority-based queuing
pub struct TaskQueueManager {
    /// High priority queue for urgent computations
    pub high_priority: Arc<Mutex<VecDeque<ScheduledComputeTask>>>,
    
    /// Normal priority queue for standard computations  
    pub normal_priority: Arc<Mutex<VecDeque<ScheduledComputeTask>>>,
    
    /// Background priority queue for low-priority tasks
    pub background_priority: Arc<Mutex<VecDeque<ScheduledComputeTask>>>,
    
    /// Infinite computation queue for unbounded tasks
    pub infinite_queue: Arc<Mutex<VecDeque<InfiniteComputeTask>>>,
    
    /// Queue statistics
    pub queue_stats: Arc<RwLock<QueueStatistics>>,
}

/// Scheduled compute task with resource requirements
#[derive(Debug, Clone)]
pub struct ScheduledComputeTask {
    /// Task ID
    pub id: u64,
    
    /// Compute input
    pub input: ComputeInput,
    
    /// Task priority
    pub priority: TaskPriority,
    
    /// Resource requirements
    pub resources: TaskResourceRequirements,
    
    /// Scheduling metadata
    pub metadata: TaskSchedulingMetadata,
    
    /// Performance requirements
    pub performance: TaskPerformanceRequirements,
}

/// Infinite compute task with specialized handling
#[derive(Debug, Clone)]
pub struct InfiniteComputeTask {
    /// Base task
    pub base_task: ScheduledComputeTask,
    
    /// Infinite computation parameters
    pub infinite_params: InfiniteComputationParameters,
    
    /// Convergence monitoring
    pub convergence: ConvergenceMonitoring,
    
    /// Resource scaling
    pub scaling: ResourceScaling,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Background = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
    Infinite = 4,
}

#[derive(Debug, Clone)]
pub struct TaskResourceRequirements {
    /// CPU cores needed
    pub cpu_cores: usize,
    
    /// Memory required (GB)
    pub memory_gb: f64,
    
    /// Mathematical engines needed
    pub engines: Vec<String>,
    
    /// Estimated FLOPS requirement
    pub estimated_flops: u64,
    
    /// Bandwidth requirement (GB/s)
    pub bandwidth_gbps: f64,
    
    /// Energy budget (J)
    pub energy_budget: f64,
}

#[derive(Debug, Clone)]
pub struct TaskSchedulingMetadata {
    /// Task submission time
    pub submitted_at: Instant,
    
    /// Task deadline (optional)
    pub deadline: Option<Instant>,
    
    /// Task dependencies
    pub dependencies: Vec<u64>,
    
    /// Task category
    pub category: TaskCategory,
    
    /// Scheduling hints
    pub hints: TaskSchedulingHints,
}

#[derive(Debug, Clone)]
pub enum TaskCategory {
    NumericalComputation,
    SymbolicComputation,
    TensorOperations,
    GeneratingFunctions,
    MemoryOperations,
    BandwidthOptimization,
    InfiniteComputation,
}

#[derive(Debug, Clone)]
pub struct TaskSchedulingHints {
    /// Preferred execution engine
    pub preferred_engine: Option<String>,
    
    /// Can be preempted
    pub preemptible: bool,
    
    /// Requires continuous execution
    pub continuous: bool,
    
    /// Memory access pattern
    pub memory_pattern: String,
    
    /// Computational pattern
    pub compute_pattern: String,
}

#[derive(Debug, Clone)]
pub struct TaskPerformanceRequirements {
    /// Target latency (microseconds)
    pub target_latency_us: u64,
    
    /// Target throughput (operations/sec)
    pub target_throughput: u64,
    
    /// Required accuracy (decimal places)
    pub required_accuracy: u32,
    
    /// Energy efficiency target (FLOPS/W)
    pub efficiency_target: f64,
}

#[derive(Debug, Clone)]
pub struct InfiniteComputationParameters {
    /// Computation strategy
    pub strategy: InfiniteStrategyType,
    
    /// Resource scaling strategy
    pub scaling_strategy: ResourceScalingStrategy,
    
    /// Termination criteria
    pub termination: TerminationCriteria,
    
    /// Progress monitoring interval
    pub monitoring_interval: Duration,
}

#[derive(Debug, Clone)]
pub struct ConvergenceMonitoring {
    /// Target convergence criteria
    pub criteria: Vec<ConvergenceCriterion>,
    
    /// Current progress
    pub progress: f64,
    
    /// Convergence history
    pub history: Vec<ConvergencePoint>,
    
    /// Stability assessment
    pub stability: StabilityAssessment,
}

#[derive(Debug, Clone)]
pub struct ConvergencePoint {
    /// Time point
    pub timestamp: Instant,
    
    /// Progress value
    pub progress: f64,
    
    /// Error estimate
    pub error_estimate: f64,
    
    /// Convergence rate
    pub rate: f64,
}

#[derive(Debug, Clone)]
pub struct StabilityAssessment {
    /// Stability score (0-1)
    pub score: f64,
    
    /// Oscillation amplitude
    pub oscillation: f64,
    
    /// Trend direction
    pub trend: TrendDirection,
    
    /// Confidence level
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum TrendDirection {
    Converging,
    Diverging,
    Oscillating,
    Stable,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ResourceScaling {
    /// Current resource allocation
    pub current_resources: TaskResourceRequirements,
    
    /// Scaling triggers
    pub triggers: Vec<ScalingTrigger>,
    
    /// Scaling history
    pub history: Vec<ScalingEvent>,
    
    /// Maximum resources
    pub max_resources: TaskResourceRequirements,
}

/// Resource allocator with mathematical optimization
pub struct ResourceAllocator {
    /// Available resources
    pub available_resources: Arc<RwLock<AvailableResources>>,
    
    /// Resource reservations
    pub reservations: Arc<RwLock<HashMap<u64, ResourceReservation>>>,
    
    /// Allocation strategies
    pub strategies: Vec<AllocationStrategy>,
    
    /// Performance metrics
    pub metrics: AllocationMetrics,
}

#[derive(Debug, Clone)]
pub struct AvailableResources {
    /// CPU cores available
    pub cpu_cores: usize,
    
    /// Memory available (GB)
    pub memory_gb: f64,
    
    /// Engine availability
    pub engines: HashMap<String, EngineAvailability>,
    
    /// FLOPS capacity
    pub flops_capacity: u64,
    
    /// Bandwidth capacity (GB/s)
    pub bandwidth_capacity_gbps: f64,
}

#[derive(Debug, Clone)]
pub struct EngineAvailability {
    /// Engine name
    pub name: String,
    
    /// Current utilization (0-1)
    pub utilization: f64,
    
    /// Available capacity
    pub available_capacity: f64,
    
    /// Performance rating
    pub performance_rating: f64,
}

#[derive(Debug, Clone)]
pub struct ResourceReservation {
    /// Task ID
    pub task_id: u64,
    
    /// Reserved resources
    pub resources: TaskResourceRequirements,
    
    /// Reservation time
    pub reserved_at: Instant,
    
    /// Reservation duration
    pub duration: Duration,
    
    /// Reservation priority
    pub priority: TaskPriority,
}

#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    WorstFit,
    MathematicalOptimal,
    PerformanceOptimized,
    EnergyOptimized,
}

#[derive(Debug, Clone)]
pub struct AllocationMetrics {
    /// Allocation success rate
    pub success_rate: f64,
    
    /// Average allocation time
    pub avg_allocation_time: Duration,
    
    /// Resource utilization efficiency
    pub utilization_efficiency: f64,
    
    /// Fragmentation level
    pub fragmentation: f64,
}

/// Performance optimizer for scheduling decisions
pub struct PerformanceOptimizer {
    /// Optimization algorithms
    pub algorithms: Vec<OptimizationAlgorithm>,
    
    /// Performance models
    pub models: Vec<PerformanceModel>,
    
    /// Optimization history
    pub history: Vec<OptimizationResult>,
    
    /// Current optimization state
    pub state: OptimizationState,
}

#[derive(Debug, Clone)]
pub struct OptimizationAlgorithm {
    /// Algorithm name
    pub name: String,
    
    /// Algorithm type
    pub algorithm_type: OptimizationType,
    
    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,
    
    /// Performance characteristics
    pub performance: AlgorithmPerformance,
}

#[derive(Debug, Clone)]
pub enum OptimizationType {
    Greedy,
    DynamicProgramming,
    GeneticAlgorithm,
    SimulatedAnnealing,
    ReinforcementLearning,
    MathematicalOptimization,
}

#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    MinimizeLatency,
    MaximizeThroughput,
    MinimizeEnergyConsumption,
    MaximizeResourceUtilization,
    MinimizeCost,
    MaximizeAccuracy,
}

#[derive(Debug, Clone)]
pub struct AlgorithmPerformance {
    /// Solution quality
    pub quality_score: f64,
    
    /// Execution time
    pub execution_time: Duration,
    
    /// Convergence rate
    pub convergence_rate: f64,
    
    /// Success rate
    pub success_rate: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceModel {
    /// Model name
    pub name: String,
    
    /// Model type
    pub model_type: ModelType,
    
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    
    /// Accuracy metrics
    pub accuracy: ModelAccuracy,
}

#[derive(Debug, Clone)]
pub enum ModelType {
    Linear,
    Polynomial,
    Exponential,
    Logarithmic,
    NeuralNetwork,
    StatisticalModel,
}

#[derive(Debug, Clone)]
pub struct ModelAccuracy {
    /// Mean absolute error
    pub mae: f64,
    
    /// Root mean square error
    pub rmse: f64,
    
    /// R-squared coefficient
    pub r_squared: f64,
    
    /// Confidence interval
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimization timestamp
    pub timestamp: Instant,
    
    /// Algorithm used
    pub algorithm: String,
    
    /// Objectives achieved
    pub objectives_achieved: HashMap<String, f64>,
    
    /// Performance improvement
    pub improvement: f64,
    
    /// Optimization cost
    pub cost: f64,
}

#[derive(Debug, Clone)]
pub struct OptimizationState {
    /// Current objective values
    pub current_objectives: HashMap<String, f64>,
    
    /// Optimization in progress
    pub optimizing: bool,
    
    /// Last optimization time
    pub last_optimization: Option<Instant>,
    
    /// Next scheduled optimization
    pub next_optimization: Option<Instant>,
}

/// Load balancer for distributing tasks across engines
pub struct LoadBalancer {
    /// Load balancing strategies
    pub strategies: Vec<LoadBalancingStrategy>,
    
    /// Engine load monitoring
    pub engine_monitor: EngineLoadMonitor,
    
    /// Load distribution history
    pub distribution_history: Vec<LoadDistributionEvent>,
    
    /// Performance metrics
    pub metrics: LoadBalancingMetrics,
}

#[derive(Debug, Clone)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    WeightedRoundRobin { weights: HashMap<String, f64> },
    CapabilityBased,
    PerformanceBased,
    MathematicalOptimal,
}

#[derive(Debug)]
pub struct EngineLoadMonitor {
    /// Engine load tracking
    pub engine_loads: Arc<RwLock<HashMap<String, EngineLoad>>>,
    
    /// Monitoring interval
    pub monitoring_interval: Duration,
    
    /// Load history
    pub load_history: Arc<RwLock<HashMap<String, Vec<LoadSnapshot>>>>,
}

#[derive(Debug, Clone)]
pub struct EngineLoad {
    /// Engine name
    pub engine: String,
    
    /// Current utilization (0-1)
    pub utilization: f64,
    
    /// Active tasks count
    pub active_tasks: usize,
    
    /// Queue length
    pub queue_length: usize,
    
    /// Average response time
    pub avg_response_time: Duration,
    
    /// FLOPS utilization
    pub flops_utilization: f64,
}

#[derive(Debug, Clone)]
pub struct LoadSnapshot {
    /// Snapshot timestamp
    pub timestamp: Instant,
    
    /// Load at snapshot time
    pub load: EngineLoad,
}

#[derive(Debug, Clone)]
pub struct LoadDistributionEvent {
    /// Event timestamp
    pub timestamp: Instant,
    
    /// Task ID distributed
    pub task_id: u64,
    
    /// Selected engine
    pub selected_engine: String,
    
    /// Load balancing strategy used
    pub strategy: String,
    
    /// Engine loads at decision time
    pub engine_loads: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct LoadBalancingMetrics {
    /// Load distribution efficiency
    pub distribution_efficiency: f64,
    
    /// Average load variance across engines
    pub load_variance: f64,
    
    /// Engine utilization balance
    pub utilization_balance: f64,
    
    /// Response time improvement
    pub response_time_improvement: f64,
}

/// Scheduling policy manager
pub struct SchedulingPolicyManager {
    /// Active policies
    pub policies: Vec<SchedulingPolicy>,
    
    /// Policy evaluation metrics
    pub evaluation_metrics: HashMap<String, PolicyMetrics>,
    
    /// Policy adaptation settings
    pub adaptation: PolicyAdaptation,
}

#[derive(Debug, Clone)]
pub struct SchedulingPolicy {
    /// Policy name
    pub name: String,
    
    /// Policy rules
    pub rules: Vec<PolicyRule>,
    
    /// Policy priority
    pub priority: u32,
    
    /// Policy effectiveness
    pub effectiveness: f64,
}

#[derive(Debug, Clone)]
pub struct PolicyRule {
    /// Rule condition
    pub condition: String,
    
    /// Rule action
    pub action: PolicyAction,
    
    /// Rule weight
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub enum PolicyAction {
    SetPriority(TaskPriority),
    AllocateResources(TaskResourceRequirements),
    SelectEngine(String),
    DelayExecution(Duration),
    RejectTask(String),
    ModifyRequirements(TaskResourceRequirements),
}

#[derive(Debug, Clone)]
pub struct PolicyMetrics {
    /// Policy application count
    pub applications: u64,
    
    /// Success rate
    pub success_rate: f64,
    
    /// Average impact on performance
    pub performance_impact: f64,
    
    /// Resource efficiency impact
    pub resource_impact: f64,
}

#[derive(Debug, Clone)]
pub struct PolicyAdaptation {
    /// Enable adaptive policies
    pub enabled: bool,
    
    /// Adaptation interval
    pub interval: Duration,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Adaptation history
    pub history: Vec<AdaptationEvent>,
}

#[derive(Debug, Clone)]
pub struct AdaptationEvent {
    /// Adaptation timestamp
    pub timestamp: Instant,
    
    /// Policy adapted
    pub policy: String,
    
    /// Adaptation type
    pub adaptation_type: AdaptationType,
    
    /// Performance before
    pub performance_before: f64,
    
    /// Performance after
    pub performance_after: f64,
}

#[derive(Debug, Clone)]
pub enum AdaptationType {
    RuleWeightAdjustment,
    RuleAddition,
    RuleRemoval,
    PolicyPriorityChange,
    ParameterTuning,
}

impl AdvancedTaskScheduler {
    /// Create new advanced task scheduler
    pub async fn new() -> Self {
        Self {
            task_queues: TaskQueueManager::new(),
            resource_allocator: ResourceAllocator::new().await,
            performance_optimizer: PerformanceOptimizer::new(),
            load_balancer: LoadBalancer::new(),
            policies: SchedulingPolicyManager::new(),
            next_task_id: AtomicU64::new(1),
        }
    }
    
    /// Submit task for scheduling
    pub async fn submit_task(&self, input: ComputeInput, priority: TaskPriority) -> Result<u64> {
        let task_id = self.next_task_id.fetch_add(1, Ordering::Relaxed);
        
        debug!("Submitting task {} with priority {:?}", task_id, priority);
        
        let task = ScheduledComputeTask {
            id: task_id,
            input: input.clone(),
            priority: priority.clone(),
            resources: self.estimate_resource_requirements(&input).await?,
            metadata: TaskSchedulingMetadata {
                submitted_at: Instant::now(),
                deadline: None,
                dependencies: Vec::new(),
                category: self.categorize_task(&input),
                hints: TaskSchedulingHints {
                    preferred_engine: None,
                    preemptible: true,
                    continuous: false,
                    memory_pattern: "sequential".to_string(),
                    compute_pattern: "batch".to_string(),
                },
            },
            performance: TaskPerformanceRequirements {
                target_latency_us: 1000,
                target_throughput: 1_000_000,
                required_accuracy: 128,
                efficiency_target: 1_000_000.0,
            },
        };
        
        // Add to appropriate queue
        match priority {
            TaskPriority::High | TaskPriority::Critical => {
                let mut queue = self.task_queues.high_priority.lock().await;
                queue.push_back(task);
            }
            TaskPriority::Normal => {
                let mut queue = self.task_queues.normal_priority.lock().await;
                queue.push_back(task);
            }
            TaskPriority::Background => {
                let mut queue = self.task_queues.background_priority.lock().await;
                queue.push_back(task);
            }
            TaskPriority::Infinite => {
                let infinite_task = InfiniteComputeTask {
                    base_task: task,
                    infinite_params: InfiniteComputationParameters {
                        strategy: InfiniteStrategyType::IterativeRefinement,
                        scaling_strategy: ResourceScalingStrategy::Adaptive,
                        termination: TerminationCriteria {
                            max_time: None,
                            convergence: ConvergenceCriteria::default(),
                            resource_limits: ResourceLimits {
                                max_cpu_time: Duration::from_secs(86400), // 24 hours
                                max_memory_mb: 1024.0 * 1024.0, // 1TB
                                max_energy_j: 1e6, // 1MJ
                            },
                            quality_criteria: QualityCriteria {
                                min_quality: 0.999,
                                metrics: Vec::new(),
                                assessment_method: QualityAssessment::WeightedAverage,
                            },
                        },
                        monitoring_interval: Duration::from_secs(10),
                    },
                    convergence: ConvergenceMonitoring {
                        criteria: vec![ConvergenceCriterion {
                            criterion_type: CriterionType::RelativeError,
                            tolerance: 1e-12,
                            max_iterations: None,
                        }],
                        progress: 0.0,
                        history: Vec::new(),
                        stability: StabilityAssessment {
                            score: 0.0,
                            oscillation: 0.0,
                            trend: TrendDirection::Unknown,
                            confidence: 0.0,
                        },
                    },
                    scaling: ResourceScaling {
                        current_resources: task.resources.clone(),
                        triggers: Vec::new(),
                        history: Vec::new(),
                        max_resources: TaskResourceRequirements {
                            cpu_cores: num_cpus::get(),
                            memory_gb: 1024.0,
                            engines: vec!["all".to_string()],
                            estimated_flops: u64::MAX,
                            bandwidth_gbps: f64::INFINITY,
                            energy_budget: 1e6,
                        },
                    },
                };
                
                let mut queue = self.task_queues.infinite_queue.lock().await;
                queue.push_back(infinite_task);
            }
        }
        
        // Update queue statistics
        self.update_queue_statistics().await;
        
        info!("Task {} submitted successfully", task_id);
        Ok(task_id)
    }
    
    /// Get next task for execution
    pub async fn get_next_task(&self) -> Result<Option<(ScheduledComputeTask, String)>> {
        // Check queues in priority order
        
        // High priority first
        {
            let mut queue = self.task_queues.high_priority.lock().await;
            if let Some(task) = queue.pop_front() {
                let engine = self.select_optimal_engine(&task).await?;
                return Ok(Some((task, engine)));
            }
        }
        
        // Normal priority
        {
            let mut queue = self.task_queues.normal_priority.lock().await;
            if let Some(task) = queue.pop_front() {
                let engine = self.select_optimal_engine(&task).await?;
                return Ok(Some((task, engine)));
            }
        }
        
        // Background priority
        {
            let mut queue = self.task_queues.background_priority.lock().await;
            if let Some(task) = queue.pop_front() {
                let engine = self.select_optimal_engine(&task).await?;
                return Ok(Some((task, engine)));
            }
        }
        
        // Infinite computation (special handling)
        {
            let mut queue = self.task_queues.infinite_queue.lock().await;
            if let Some(infinite_task) = queue.pop_front() {
                let engine = self.select_optimal_engine(&infinite_task.base_task).await?;
                return Ok(Some((infinite_task.base_task, engine)));
            }
        }
        
        Ok(None)
    }
    
    async fn estimate_resource_requirements(&self, input: &ComputeInput) -> Result<TaskResourceRequirements> {
        // Estimate resources based on input characteristics
        let estimated_flops = match &input.data {
            infinite_engines::ComputeData::Numerical { array, .. } => {
                (array.len() as u64) * 1000 // Estimate 1000 FLOPS per element
            }
            infinite_engines::ComputeData::TensorNetwork { nodes, .. } => {
                nodes.iter().map(|n| n.data.len() as u64).sum::<u64>() * 10000
            }
            _ => 1_000_000, // Default estimate
        };
        
        Ok(TaskResourceRequirements {
            cpu_cores: (estimated_flops / 1_000_000_000).max(1) as usize, // 1 core per GFLOPS
            memory_gb: (estimated_flops as f64 / 1e9) * 0.1, // 0.1 GB per GFLOPS
            engines: vec!["auto".to_string()],
            estimated_flops,
            bandwidth_gbps: (estimated_flops as f64 / 1e9) * 10.0, // 10 GB/s per GFLOPS
            energy_budget: estimated_flops as f64 * 1e-9, // 1nJ per FLOP
        })
    }
    
    fn categorize_task(&self, input: &ComputeInput) -> TaskCategory {
        match &input.data {
            infinite_engines::ComputeData::Numerical { .. } => TaskCategory::NumericalComputation,
            infinite_engines::ComputeData::Symbolic { .. } => TaskCategory::SymbolicComputation,
            infinite_engines::ComputeData::TensorNetwork { .. } => TaskCategory::TensorOperations,
            infinite_engines::ComputeData::GeneratingFunction { .. } => TaskCategory::GeneratingFunctions,
            _ => TaskCategory::NumericalComputation,
        }
    }
    
    async fn select_optimal_engine(&self, task: &ScheduledComputeTask) -> Result<String> {
        // Use load balancer to select optimal engine
        let engine_loads = self.load_balancer.engine_monitor.engine_loads.read().await;
        
        let optimal_engine = match task.metadata.category {
            TaskCategory::NumericalComputation => {
                if task.resources.estimated_flops == u64::MAX {
                    "idvbit".to_string() // Best for infinite computations
                } else {
                    "tensor".to_string()
                }
            }
            TaskCategory::SymbolicComputation => "storage".to_string(),
            TaskCategory::TensorOperations => "tensor".to_string(),
            TaskCategory::GeneratingFunctions => "gf".to_string(),
            TaskCategory::MemoryOperations => "storage".to_string(),
            TaskCategory::BandwidthOptimization => "bandwidth".to_string(),
            TaskCategory::InfiniteComputation => "idvbit".to_string(),
        };
        
        // Verify engine availability
        if let Some(load) = engine_loads.get(&optimal_engine) {
            if load.utilization < 0.9 {
                return Ok(optimal_engine);
            }
        }
        
        // Fallback to least loaded engine
        let least_loaded = engine_loads.iter()
            .min_by(|(_, a), (_, b)| a.utilization.partial_cmp(&b.utilization).unwrap())
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "idvbit".to_string());
            
        Ok(least_loaded)
    }
    
    async fn update_queue_statistics(&self) {
        let mut stats = self.task_queues.queue_stats.write().await;
        
        let high_count = self.task_queues.high_priority.lock().await.len();
        let normal_count = self.task_queues.normal_priority.lock().await.len();
        let background_count = self.task_queues.background_priority.lock().await.len();
        let infinite_count = self.task_queues.infinite_queue.lock().await.len();
        
        stats.tasks_per_priority.insert(TaskPriority::High, high_count);
        stats.tasks_per_priority.insert(TaskPriority::Normal, normal_count);
        stats.tasks_per_priority.insert(TaskPriority::Background, background_count);
        stats.tasks_per_priority.insert(TaskPriority::Infinite, infinite_count);
        
        stats.throughput_tasks_per_sec = 100.0; // Estimate
        stats.utilization_percentage = 75.0; // Estimate
    }
}

// Helper implementations
impl TaskQueueManager {
    pub fn new() -> Self {
        Self {
            high_priority: Arc::new(Mutex::new(VecDeque::new())),
            normal_priority: Arc::new(Mutex::new(VecDeque::new())),
            background_priority: Arc::new(Mutex::new(VecDeque::new())),
            infinite_queue: Arc::new(Mutex::new(VecDeque::new())),
            queue_stats: Arc::new(RwLock::new(QueueStatistics {
                tasks_per_priority: HashMap::new(),
                avg_wait_times: HashMap::new(),
                throughput_tasks_per_sec: 0.0,
                utilization_percentage: 0.0,
            })),
        }
    }
}

impl ResourceAllocator {
    pub async fn new() -> Self {
        let mut engines = HashMap::new();
        engines.insert("idvbit".to_string(), EngineAvailability {
            name: "idvbit".to_string(),
            utilization: 0.0,
            available_capacity: 1.0,
            performance_rating: 1.0,
        });
        engines.insert("tensor".to_string(), EngineAvailability {
            name: "tensor".to_string(),
            utilization: 0.0,
            available_capacity: 1.0,
            performance_rating: 0.9,
        });
        
        Self {
            available_resources: Arc::new(RwLock::new(AvailableResources {
                cpu_cores: num_cpus::get(),
                memory_gb: 1024.0,
                engines,
                flops_capacity: u64::MAX,
                bandwidth_capacity_gbps: f64::INFINITY,
            })),
            reservations: Arc::new(RwLock::new(HashMap::new())),
            strategies: vec![AllocationStrategy::MathematicalOptimal],
            metrics: AllocationMetrics {
                success_rate: 0.99,
                avg_allocation_time: Duration::from_millis(10),
                utilization_efficiency: 0.85,
                fragmentation: 0.05,
            },
        }
    }
}

impl PerformanceOptimizer {
    pub fn new() -> Self {
        Self {
            algorithms: vec![
                OptimizationAlgorithm {
                    name: "MathematicalOptimal".to_string(),
                    algorithm_type: OptimizationType::MathematicalOptimization,
                    objectives: vec![
                        OptimizationObjective::MinimizeLatency,
                        OptimizationObjective::MaximizeThroughput,
                        OptimizationObjective::MaximizeResourceUtilization,
                    ],
                    performance: AlgorithmPerformance {
                        quality_score: 0.95,
                        execution_time: Duration::from_millis(100),
                        convergence_rate: 0.99,
                        success_rate: 0.98,
                    },
                }
            ],
            models: Vec::new(),
            history: Vec::new(),
            state: OptimizationState {
                current_objectives: HashMap::new(),
                optimizing: false,
                last_optimization: None,
                next_optimization: Some(Instant::now() + Duration::from_secs(60)),
            },
        }
    }
}

impl LoadBalancer {
    pub fn new() -> Self {
        let mut engine_loads = HashMap::new();
        engine_loads.insert("idvbit".to_string(), EngineLoad {
            engine: "idvbit".to_string(),
            utilization: 0.0,
            active_tasks: 0,
            queue_length: 0,
            avg_response_time: Duration::from_millis(100),
            flops_utilization: 0.0,
        });
        
        Self {
            strategies: vec![LoadBalancingStrategy::MathematicalOptimal],
            engine_monitor: EngineLoadMonitor {
                engine_loads: Arc::new(RwLock::new(engine_loads)),
                monitoring_interval: Duration::from_secs(1),
                load_history: Arc::new(RwLock::new(HashMap::new())),
            },
            distribution_history: Vec::new(),
            metrics: LoadBalancingMetrics {
                distribution_efficiency: 0.9,
                load_variance: 0.1,
                utilization_balance: 0.85,
                response_time_improvement: 0.2,
            },
        }
    }
}

impl SchedulingPolicyManager {
    pub fn new() -> Self {
        Self {
            policies: vec![
                SchedulingPolicy {
                    name: "InfinitePriority".to_string(),
                    rules: vec![
                        PolicyRule {
                            condition: "task.estimated_flops == MAX".to_string(),
                            action: PolicyAction::SetPriority(TaskPriority::Infinite),
                            weight: 1.0,
                        }
                    ],
                    priority: 1,
                    effectiveness: 0.9,
                }
            ],
            evaluation_metrics: HashMap::new(),
            adaptation: PolicyAdaptation {
                enabled: true,
                interval: Duration::from_secs(300), // 5 minutes
                learning_rate: 0.1,
                history: Vec::new(),
            },
        }
    }
}
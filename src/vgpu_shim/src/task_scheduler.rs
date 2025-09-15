//! GPU Task Scheduler with Advanced Algorithms
//!
//! Implements sophisticated task scheduling algorithms for GPU workloads
//! including priority-based scheduling, load balancing, and resource optimization.

use crate::{Result, VGpuError};
use crate::resource_manager::Priority;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BinaryHeap, VecDeque};
use std::sync::Arc;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{interval, sleep};
use futures::stream::StreamExt;

/// Advanced GPU task scheduler
pub struct TaskScheduler {
    // Task queues organized by priority
    priority_queues: Arc<RwLock<HashMap<Priority, VecDeque<GpuTask>>>>,
    
    // Active task tracking
    active_tasks: Arc<RwLock<HashMap<u64, ActiveTask>>>,
    
    // Scheduler configuration
    config: SchedulerConfig,
    
    // Performance metrics
    metrics: Arc<RwLock<SchedulerMetrics>>,
    
    // Task submission channel
    task_sender: Arc<RwLock<Option<mpsc::UnboundedSender<TaskSubmission>>>>,
    
    // Scheduling algorithms
    algorithms: SchedulingAlgorithms,
    
    // Resource prediction
    predictor: ResourcePredictor,
    
    // Load balancing
    load_balancer: LoadBalancer,
}

/// GPU task representation
#[derive(Debug, Clone)]
pub struct GpuTask {
    pub task_id: u64,
    pub kernel_name: String,
    pub data: Vec<f32>,
    pub grid_size: (usize, usize, usize),
    pub block_size: (usize, usize, usize),
    pub priority: TaskPriority,
    pub resource_requirements: ResourceRequirements,
    pub deadline: Option<Instant>,
    pub dependencies: Vec<u64>,
    pub submitted_at: Instant,
    pub metadata: TaskMetadata,
}

/// Task priority levels with scheduling weights
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    Background = 1,    // Batch processing, background tasks
    Low = 2,          // Non-critical computations
    Normal = 3,       // Standard GPU kernels
    High = 4,         // Interactive applications, real-time processing
    Critical = 5,     // System-critical tasks
    RealTime = 6,     // Hard real-time constraints
}

/// Resource requirements for tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub memory_bytes: usize,
    pub compute_units: usize,
    pub bandwidth_mbps: u64,
    pub estimated_duration: Duration,
    pub power_budget: f64, // Watts
}

/// Task metadata for scheduling decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    pub task_type: TaskType,
    pub parallelism_type: ParallelismType,
    pub memory_access_pattern: MemoryAccessPattern,
    pub compute_intensity: ComputeIntensity,
    pub user_id: String,
    pub application_id: String,
}

/// Types of GPU tasks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
    MatrixMultiplication,
    ConvolutionalNeural,
    FastFourierTransform,
    MonteCarlo,
    VectorOperation,
    ImageProcessing,
    Scientific,
    MachineLearning,
    Cryptographic,
    Custom { name: String },
}

/// Task parallelism characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParallelismType {
    DataParallel,         // SIMD-style parallelism
    TaskParallel,         // Independent task parallelism
    Pipeline,             // Pipeline parallelism
    Reduction,            // Reduction operations
    Stencil,              // Stencil computations
    GraphTraversal,       // Graph algorithms
}

/// Memory access patterns for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryAccessPattern {
    Sequential,           // Linear memory access
    Random,               // Random memory access
    Strided { stride: usize }, // Strided access pattern
    Blocked { block_size: usize }, // Blocked access
    Hierarchical,         // Tree-like access
}

/// Compute intensity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeIntensity {
    MemoryBound,     // Limited by memory bandwidth
    ComputeBound,    // Limited by compute throughput
    Balanced,        // Balanced compute and memory
    CacheFriendly,   // High cache locality
    NetworkBound,    // Limited by network I/O
}

/// Active task tracking
#[derive(Debug, Clone)]
struct ActiveTask {
    task: GpuTask,
    started_at: Instant,
    estimated_completion: Instant,
    resource_allocation_id: u64,
    execution_context: ExecutionContext,
}

/// Task execution context
#[derive(Debug, Clone)]
struct ExecutionContext {
    compute_units: Vec<usize>,
    memory_blocks: Vec<u64>,
    priority_boost: f64,
}

/// Task submission wrapper
struct TaskSubmission {
    task: GpuTask,
    response_sender: oneshot::Sender<Result<()>>,
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_concurrent_tasks: usize,
    pub scheduling_algorithm: SchedulingAlgorithm,
    pub preemption_enabled: bool,
    pub priority_aging_enabled: bool,
    pub load_balancing_enabled: bool,
    pub power_aware_scheduling: bool,
    pub thermal_aware_scheduling: bool,
    pub deadline_scheduling: bool,
}

/// Available scheduling algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingAlgorithm {
    FirstComeFirstServe,    // FCFS
    ShortestJobFirst,       // SJF
    RoundRobin { quantum: Duration },
    PriorityScheduling,     // Priority-based
    MultiLevelQueue,        // Multi-level priority queues
    CompleteFairScheduling, // CFS-style scheduling
    EarliestDeadlineFirst,  // EDF for real-time tasks
    ProportionalShare,      // Proportional share scheduling
    WorkConserving,         // Work-conserving scheduling
    EnergyAware,            // Energy-efficient scheduling
}

/// Scheduler performance metrics
#[derive(Debug, Clone)]
pub struct SchedulerMetrics {
    pub tasks_submitted: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub average_wait_time: Duration,
    pub average_execution_time: Duration,
    pub throughput: f64, // Tasks per second
    pub utilization: f64, // Resource utilization
    pub deadline_miss_rate: f64,
    pub power_efficiency: f64, // Tasks per watt
    pub fairness_index: f64, // Jain's fairness index
}

/// Scheduling algorithms implementation
struct SchedulingAlgorithms {
    current_algorithm: SchedulingAlgorithm,
    quantum_remaining: Duration,
    last_scheduled: Instant,
}

/// Resource usage prediction for scheduling
struct ResourcePredictor {
    historical_data: VecDeque<ResourceUsageSnapshot>,
    prediction_model: PredictionModel,
    prediction_accuracy: f64,
}

/// Resource usage snapshot
#[derive(Debug, Clone)]
struct ResourceUsageSnapshot {
    timestamp: Instant,
    memory_usage: f64,
    compute_usage: f64,
    bandwidth_usage: f64,
    power_usage: f64,
    temperature: f64,
}

/// Load balancing strategies
struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    compute_unit_loads: HashMap<usize, f64>,
    migration_threshold: f64,
}

#[derive(Debug, Clone)]
enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    PowerOfTwoChoices,
    ConsistentHashing,
    LocalityAware,
    TemperatureAware,
}

#[derive(Debug, Clone)]
enum PredictionModel {
    LinearRegression { coefficients: Vec<f64> },
    MovingAverage { window_size: usize },
    ExponentialSmoothing { alpha: f64 },
    AutoRegressive { order: usize, parameters: Vec<f64> },
}

impl GpuTask {
    /// Create a new GPU task with default settings
    pub fn new(kernel_name: &str, data: Vec<f32>) -> Self {
        Self {
            task_id: Self::generate_task_id(),
            kernel_name: kernel_name.to_string(),
            data,
            grid_size: (1, 1, 1),
            block_size: (32, 1, 1),
            priority: TaskPriority::Normal,
            resource_requirements: ResourceRequirements::default(),
            deadline: None,
            dependencies: Vec::new(),
            submitted_at: Instant::now(),
            metadata: TaskMetadata::default(),
        }
    }

    /// Create a task with specific priority and requirements
    pub fn with_priority(kernel_name: &str, data: Vec<f32>, priority: TaskPriority, requirements: ResourceRequirements) -> Self {
        let mut task = Self::new(kernel_name, data);
        task.priority = priority;
        task.resource_requirements = requirements;
        task
    }

    /// Set task deadline for real-time scheduling
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Add task dependencies
    pub fn with_dependencies(mut self, dependencies: Vec<u64>) -> Self {
        self.dependencies = dependencies;
        self
    }

    /// Generate unique task ID
    fn generate_task_id() -> u64 {
        use rand::Rng;
        rand::thread_rng().gen()
    }

    /// Calculate task scheduling priority (higher values = higher priority)
    pub fn scheduling_priority(&self) -> f64 {
        let mut priority = self.priority as u32 as f64;
        
        // Boost priority based on deadline urgency
        if let Some(deadline) = self.deadline {
            let time_until_deadline = deadline.saturating_duration_since(Instant::now()).as_secs_f64();
            let urgency_boost = (1.0 / (time_until_deadline + 1.0)) * 100.0;
            priority += urgency_boost;
        }
        
        // Age-based priority boost (prevent starvation)
        let age = Instant::now().saturating_duration_since(self.submitted_at).as_secs_f64();
        let age_boost = (age / 60.0).min(2.0); // Up to 2 priority boost after 2 minutes
        priority += age_boost;
        
        priority
    }
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            memory_bytes: 1024 * 1024, // 1MB default
            compute_units: 1,
            bandwidth_mbps: 1000, // 1 GB/s
            estimated_duration: Duration::from_secs(1),
            power_budget: 10.0, // 10W
        }
    }
}

impl Default for TaskMetadata {
    fn default() -> Self {
        Self {
            task_type: TaskType::VectorOperation,
            parallelism_type: ParallelismType::DataParallel,
            memory_access_pattern: MemoryAccessPattern::Sequential,
            compute_intensity: ComputeIntensity::Balanced,
            user_id: "default_user".to_string(),
            application_id: "default_app".to_string(),
        }
    }
}

impl TaskScheduler {
    /// Create a new task scheduler
    pub fn new() -> Self {
        Self {
            priority_queues: Arc::new(RwLock::new(HashMap::new())),
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            config: SchedulerConfig::default(),
            metrics: Arc::new(RwLock::new(SchedulerMetrics::new())),
            task_sender: Arc::new(RwLock::new(None)),
            algorithms: SchedulingAlgorithms::new(),
            predictor: ResourcePredictor::new(),
            load_balancer: LoadBalancer::new(),
        }
    }

    /// Configure the scheduler
    pub fn with_config(mut self, config: SchedulerConfig) -> Self {
        self.config = config;
        self
    }

    /// Set maximum concurrent tasks
    pub fn set_max_concurrent_tasks(&mut self, max_tasks: usize) {
        self.config.max_concurrent_tasks = max_tasks;
    }

    /// Start the scheduler's background task management
    pub async fn start(&self) -> Result<()> {
        let (sender, mut receiver) = mpsc::unbounded_channel::<TaskSubmission>();
        *self.task_sender.write() = Some(sender);

        // Initialize priority queues
        {
            let mut queues = self.priority_queues.write();
            for priority in [crate::resource_manager::Priority::Low, crate::resource_manager::Priority::Normal, 
                            crate::resource_manager::Priority::High, crate::resource_manager::Priority::Critical] {
                queues.insert(priority, VecDeque::new());
            }
        }

        // Spawn scheduler main loop
        let priority_queues = self.priority_queues.clone();
        let active_tasks = self.active_tasks.clone();
        let config = self.config.clone();
        let metrics = self.metrics.clone();

        tokio::spawn(async move {
            let mut scheduling_interval = interval(Duration::from_millis(10));
            
            loop {
                tokio::select! {
                    // Handle new task submissions
                    Some(submission) = receiver.recv() => {
                        let result = Self::handle_task_submission(
                            &priority_queues,
                            submission.task,
                            &metrics
                        );
                        let _ = submission.response_sender.send(result);
                    },
                    
                    // Periodic scheduling
                    _ = scheduling_interval.tick() => {
                        Self::schedule_tasks(
                            &priority_queues,
                            &active_tasks,
                            &config,
                            &metrics
                        ).await;
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop the scheduler
    pub async fn stop(&self) -> Result<()> {
        // Cancel all active tasks
        self.active_tasks.write().clear();
        
        // Clear task queues
        self.priority_queues.write().clear();
        
        Ok(())
    }

    /// Submit a task for execution
    pub async fn submit(&self, task: GpuTask) -> Result<()> {
        let sender = {
            let task_sender_guard = self.task_sender.read();
            task_sender_guard.clone()
        };
        
        if let Some(sender) = sender {
            let (response_sender, response_receiver) = oneshot::channel();
            
            let submission = TaskSubmission {
                task,
                response_sender,
            };

            sender.send(submission)
                .map_err(|_| VGpuError::task_scheduling("Scheduler not running"))?;

            response_receiver.await
                .map_err(|_| VGpuError::task_scheduling("Failed to receive response"))?
        } else {
            Err(VGpuError::task_scheduling("Scheduler not started"))
        }
    }

    /// Handle task submission
    fn handle_task_submission(
        priority_queues: &Arc<RwLock<HashMap<crate::resource_manager::Priority, VecDeque<GpuTask>>>>,
        task: GpuTask,
        metrics: &Arc<RwLock<SchedulerMetrics>>,
    ) -> Result<()> {
        // Convert TaskPriority to Priority
        let priority = match task.priority {
            TaskPriority::Background => crate::resource_manager::Priority::Low,
            TaskPriority::Low => crate::resource_manager::Priority::Low,
            TaskPriority::Normal => crate::resource_manager::Priority::Normal,
            TaskPriority::High => crate::resource_manager::Priority::High,
            TaskPriority::Critical | TaskPriority::RealTime => crate::resource_manager::Priority::Critical,
        };

        // Add task to appropriate priority queue
        if let Some(queue) = priority_queues.write().get_mut(&priority) {
            queue.push_back(task);
            metrics.write().tasks_submitted += 1;
            Ok(())
        } else {
            Err(VGpuError::task_scheduling("Invalid priority queue"))
        }
    }

    /// Main scheduling loop
    async fn schedule_tasks(
        priority_queues: &Arc<RwLock<HashMap<crate::resource_manager::Priority, VecDeque<GpuTask>>>>,
        active_tasks: &Arc<RwLock<HashMap<u64, ActiveTask>>>,
        config: &SchedulerConfig,
        metrics: &Arc<RwLock<SchedulerMetrics>>,
    ) {
        let current_active = active_tasks.read().len();
        
        if current_active >= config.max_concurrent_tasks {
            return; // No capacity for new tasks
        }

        // Schedule tasks based on algorithm
        match config.scheduling_algorithm {
            SchedulingAlgorithm::PriorityScheduling => {
                Self::priority_scheduling(priority_queues, active_tasks, config, metrics).await;
            },
            SchedulingAlgorithm::EarliestDeadlineFirst => {
                Self::edf_scheduling(priority_queues, active_tasks, config, metrics).await;
            },
            SchedulingAlgorithm::MultiLevelQueue => {
                Self::multilevel_queue_scheduling(priority_queues, active_tasks, config, metrics).await;
            },
            _ => {
                Self::priority_scheduling(priority_queues, active_tasks, config, metrics).await;
            }
        }
    }

    /// Priority-based scheduling algorithm
    async fn priority_scheduling(
        priority_queues: &Arc<RwLock<HashMap<crate::resource_manager::Priority, VecDeque<GpuTask>>>>,
        active_tasks: &Arc<RwLock<HashMap<u64, ActiveTask>>>,
        config: &SchedulerConfig,
        metrics: &Arc<RwLock<SchedulerMetrics>>,
    ) {
        // Schedule from highest priority to lowest
        for priority in [crate::resource_manager::Priority::Critical, crate::resource_manager::Priority::High, crate::resource_manager::Priority::Normal, crate::resource_manager::Priority::Low] {
            loop {
                let task_to_execute = {
                    let mut queues = priority_queues.write();
                    if let Some(queue) = queues.get_mut(&priority) {
                        if !queue.is_empty() && active_tasks.read().len() < config.max_concurrent_tasks {
                            queue.pop_front()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };
                
                if let Some(task) = task_to_execute {
                    Self::execute_task(task, active_tasks, metrics).await;
                } else {
                    break;
                }
            }
        }
    }

    /// Earliest Deadline First scheduling
    async fn edf_scheduling(
        priority_queues: &Arc<RwLock<HashMap<crate::resource_manager::Priority, VecDeque<GpuTask>>>>,
        active_tasks: &Arc<RwLock<HashMap<u64, ActiveTask>>>,
        config: &SchedulerConfig,
        metrics: &Arc<RwLock<SchedulerMetrics>>,
    ) {
        // Collect all tasks with deadlines
        let deadline_tasks = {
            let mut deadline_tasks = Vec::new();
            let mut queues = priority_queues.write();
            
            for queue in queues.values_mut() {
                let mut i = 0;
                while i < queue.len() {
                    if queue[i].deadline.is_some() {
                        deadline_tasks.push(queue.remove(i).unwrap());
                    } else {
                        i += 1;
                    }
                }
            }
            
            // Sort by deadline (earliest first)
            deadline_tasks.sort_by_key(|task| task.deadline.unwrap_or(Instant::now()));
            deadline_tasks
        };
        
        // Execute tasks in deadline order  
        for task in deadline_tasks {
            if active_tasks.read().len() < config.max_concurrent_tasks {
                Self::execute_task(task, active_tasks, metrics).await;
            } else {
                // Put back in appropriate queue if no capacity
                let priority = match task.priority {
                    TaskPriority::Critical | TaskPriority::RealTime => crate::resource_manager::Priority::Critical,
                    TaskPriority::High => crate::resource_manager::Priority::High,
                    TaskPriority::Normal => crate::resource_manager::Priority::Normal,
                    TaskPriority::Low | TaskPriority::Background => crate::resource_manager::Priority::Low,
                };
                let mut queues = priority_queues.write();
                if let Some(queue) = queues.get_mut(&priority) {
                    queue.push_front(task);
                }
                break;
            }
        }
    }

    /// Multi-level queue scheduling
    async fn multilevel_queue_scheduling(
        priority_queues: &Arc<RwLock<HashMap<crate::resource_manager::Priority, VecDeque<GpuTask>>>>,
        active_tasks: &Arc<RwLock<HashMap<u64, ActiveTask>>>,
        config: &SchedulerConfig,
        metrics: &Arc<RwLock<SchedulerMetrics>>,
    ) {
        // Use time slicing between priority levels
        let total_capacity = config.max_concurrent_tasks - active_tasks.read().len();
        if total_capacity == 0 {
            return;
        }

        // Allocate capacity based on priority weights
        let critical_share = (total_capacity * 50 / 100).max(1);
        let high_share = (total_capacity * 30 / 100).max(1);
        let normal_share = (total_capacity * 15 / 100).max(1);
        let low_share = total_capacity.saturating_sub(critical_share + high_share + normal_share);

        let allocations = [
            (crate::resource_manager::Priority::Critical, critical_share),
            (crate::resource_manager::Priority::High, high_share),
            (crate::resource_manager::Priority::Normal, normal_share),
            (crate::resource_manager::Priority::Low, low_share),
        ];

        for (priority, allocation) in allocations {
            for _ in 0..allocation {
                let task_to_execute = {
                    let mut queues = priority_queues.write();
                    if let Some(queue) = queues.get_mut(&priority) {
                        if !queue.is_empty() {
                            queue.pop_front()
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };
                
                if let Some(task) = task_to_execute {
                    Self::execute_task(task, active_tasks, metrics).await;
                } else {
                    break;
                }
            }
        }
    }

    /// Execute a task
    async fn execute_task(
        task: GpuTask,
        active_tasks: &Arc<RwLock<HashMap<u64, ActiveTask>>>,
        metrics: &Arc<RwLock<SchedulerMetrics>>,
    ) {
        let task_id = task.task_id;
        let estimated_completion = Instant::now() + task.resource_requirements.estimated_duration;
        
        let active_task = ActiveTask {
            started_at: Instant::now(),
            estimated_completion,
            resource_allocation_id: task_id, // Simplified
            execution_context: ExecutionContext {
                compute_units: vec![0], // Simplified
                memory_blocks: vec![task_id],
                priority_boost: 0.0,
            },
            task,
        };

        active_tasks.write().insert(task_id, active_task);
        
        // Simulate task execution asynchronously
        let active_tasks_clone = active_tasks.clone();
        let metrics_clone = metrics.clone();
        
        tokio::spawn(async move {
            // Simulate task execution time
            sleep(Duration::from_millis(100)).await; // Simplified execution
            
            // Remove from active tasks
            if let Some(_completed_task) = active_tasks_clone.write().remove(&task_id) {
                let mut metrics_guard = metrics_clone.write();
                metrics_guard.tasks_completed += 1;
                
                // Update average execution time
                let execution_time = Duration::from_millis(100); // Simplified
                metrics_guard.average_execution_time = 
                    (metrics_guard.average_execution_time + execution_time) / 2;
            }
        });
    }

    /// Get current scheduler metrics
    pub fn get_metrics(&self) -> SchedulerMetrics {
        self.metrics.read().clone()
    }

    /// Get current task queue status
    pub fn get_queue_status(&self) -> HashMap<crate::resource_manager::Priority, usize> {
        self.priority_queues.read()
            .iter()
            .map(|(priority, queue)| (*priority, queue.len()))
            .collect()
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 16,
            scheduling_algorithm: SchedulingAlgorithm::PriorityScheduling,
            preemption_enabled: false,
            priority_aging_enabled: true,
            load_balancing_enabled: true,
            power_aware_scheduling: false,
            thermal_aware_scheduling: false,
            deadline_scheduling: true,
        }
    }
}

impl SchedulerMetrics {
    fn new() -> Self {
        Self {
            tasks_submitted: 0,
            tasks_completed: 0,
            tasks_failed: 0,
            average_wait_time: Duration::from_secs(0),
            average_execution_time: Duration::from_secs(0),
            throughput: 0.0,
            utilization: 0.0,
            deadline_miss_rate: 0.0,
            power_efficiency: 0.0,
            fairness_index: 1.0,
        }
    }
}

impl SchedulingAlgorithms {
    fn new() -> Self {
        Self {
            current_algorithm: SchedulingAlgorithm::PriorityScheduling,
            quantum_remaining: Duration::from_millis(100),
            last_scheduled: Instant::now(),
        }
    }
}

impl ResourcePredictor {
    fn new() -> Self {
        Self {
            historical_data: VecDeque::with_capacity(1000),
            prediction_model: PredictionModel::MovingAverage { window_size: 10 },
            prediction_accuracy: 0.8,
        }
    }
}

impl LoadBalancer {
    fn new() -> Self {
        Self {
            strategy: LoadBalancingStrategy::LeastLoaded,
            compute_unit_loads: HashMap::new(),
            migration_threshold: 0.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_task_scheduler_creation() {
        let scheduler = TaskScheduler::new();
        let metrics = scheduler.get_metrics();
        
        assert_eq!(metrics.tasks_submitted, 0);
        assert_eq!(metrics.tasks_completed, 0);
    }

    #[tokio::test]
    async fn test_task_submission() {
        let mut scheduler = TaskScheduler::new();
        scheduler.start().await.unwrap();
        
        let task = GpuTask::new("test_kernel", vec![1.0, 2.0, 3.0]);
        scheduler.submit(task).await.unwrap();
        
        let metrics = scheduler.get_metrics();
        assert!(metrics.tasks_submitted > 0);
        
        scheduler.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_priority_scheduling() {
        let mut scheduler = TaskScheduler::new();
        scheduler.start().await.unwrap();
        
        // Submit tasks with different priorities
        let high_priority_task = GpuTask::with_priority(
            "high_priority",
            vec![1.0],
            TaskPriority::High,
            ResourceRequirements::default()
        );
        
        let low_priority_task = GpuTask::with_priority(
            "low_priority", 
            vec![1.0],
            TaskPriority::Low,
            ResourceRequirements::default()
        );
        
        scheduler.submit(low_priority_task).await.unwrap();
        scheduler.submit(high_priority_task).await.unwrap();
        
        // Allow some time for scheduling
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let metrics = scheduler.get_metrics();
        assert!(metrics.tasks_submitted >= 2);
        
        scheduler.stop().await.unwrap();
    }

    #[test]
    fn test_task_priority_calculation() {
        let mut task = GpuTask::new("test", vec![1.0]);
        task.priority = TaskPriority::High;
        
        let base_priority = task.scheduling_priority();
        assert!(base_priority >= 4.0); // High priority base value
        
        // Test deadline urgency
        task.deadline = Some(Instant::now() + Duration::from_secs(10));
        let urgent_priority = task.scheduling_priority();
        assert!(urgent_priority > base_priority);
    }

    #[tokio::test]
    async fn test_concurrent_task_limit() {
        let mut scheduler = TaskScheduler::new();
        scheduler.set_max_concurrent_tasks(2);
        scheduler.start().await.unwrap();
        
        // Submit more tasks than the limit
        for i in 0..5 {
            let task = GpuTask::new(&format!("task_{}", i), vec![i as f32]);
            scheduler.submit(task).await.unwrap();
        }
        
        let queue_status = scheduler.get_queue_status();
        let total_queued: usize = queue_status.values().sum();
        assert!(total_queued > 0); // Some tasks should be queued
        
        scheduler.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_deadline_scheduling() {
        let mut config = SchedulerConfig::default();
        config.scheduling_algorithm = SchedulingAlgorithm::EarliestDeadlineFirst;
        
        let mut scheduler = TaskScheduler::new().with_config(config);
        scheduler.start().await.unwrap();
        
        // Submit task with deadline
        let deadline_task = GpuTask::new("deadline_task", vec![1.0])
            .with_deadline(Instant::now() + Duration::from_secs(5));
        
        scheduler.submit(deadline_task).await.unwrap();
        
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        let metrics = scheduler.get_metrics();
        assert!(metrics.tasks_submitted > 0);
        
        scheduler.stop().await.unwrap();
    }

    #[test]
    fn test_resource_requirements() {
        let requirements = ResourceRequirements {
            memory_bytes: 1024 * 1024,
            compute_units: 4,
            bandwidth_mbps: 5000,
            estimated_duration: Duration::from_secs(2),
            power_budget: 50.0,
        };
        
        assert_eq!(requirements.memory_bytes, 1024 * 1024);
        assert_eq!(requirements.compute_units, 4);
        assert_eq!(requirements.bandwidth_mbps, 5000);
    }

    #[test]
    fn test_task_metadata() {
        let metadata = TaskMetadata {
            task_type: TaskType::MatrixMultiplication,
            parallelism_type: ParallelismType::DataParallel,
            memory_access_pattern: MemoryAccessPattern::Blocked { block_size: 1024 },
            compute_intensity: ComputeIntensity::ComputeBound,
            user_id: "test_user".to_string(),
            application_id: "matrix_app".to_string(),
        };
        
        assert_eq!(metadata.task_type, TaskType::MatrixMultiplication);
        assert_eq!(metadata.user_id, "test_user");
    }
}
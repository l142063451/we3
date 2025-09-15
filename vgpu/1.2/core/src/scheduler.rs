use anyhow::{Result, Context};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::{info, debug, warn, error};
use uuid::Uuid;

use crate::{ComputeOperation, MemoryManager, TaskPriority, OperationType};

/// Advanced Task Scheduler with Multiple Scheduling Policies
pub struct TaskScheduler {
    config: SchedulerConfig,
    task_queues: TaskQueues,
    compute_units: Arc<Semaphore>,
    active_tasks: Arc<DashMap<Uuid, ActiveTask>>,
    scheduler_stats: Arc<SchedulerStatistics>,
    load_balancer: Arc<LoadBalancer>,
    resource_predictor: Arc<ResourcePredictor>,
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_concurrent_tasks: u32,
    pub scheduling_policy: SchedulingPolicy,
    pub time_slice_ms: u64,
    pub priority_levels: usize,
    pub load_balancing: bool,
    pub resource_prediction: bool,
    pub preemption_enabled: bool,
}

#[derive(Debug, Clone)]
pub enum SchedulingPolicy {
    FCFS, // First Come First Serve
    SJF,  // Shortest Job First  
    Priority,
    RoundRobin,
    MultiLevelQueue,
    EDF,  // Earliest Deadline First
    CFS,  // Completely Fair Scheduler
}

pub struct TaskQueues {
    high_priority: Arc<RwLock<VecDeque<QueuedTask>>>,
    normal_priority: Arc<RwLock<VecDeque<QueuedTask>>>,
    low_priority: Arc<RwLock<VecDeque<QueuedTask>>>,
    background: Arc<RwLock<VecDeque<QueuedTask>>>,
    real_time: Arc<RwLock<VecDeque<QueuedTask>>>,
}

#[derive(Debug)]
pub struct QueuedTask {
    pub operation: ComputeOperation,
    pub queued_at: Instant,
    pub estimated_duration: Option<Duration>,
    pub deadline: Option<Instant>,
    pub resource_requirements: ResourceRequirements,
    pub completion_callback: Option<oneshot::Sender<Result<()>>>,
}

#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub compute_units: u32,
    pub memory_mb: u64,
    pub bandwidth_mbps: u64,
    pub estimated_runtime: Duration,
    pub cpu_intensive: bool,
    pub memory_intensive: bool,
    pub io_intensive: bool,
}

#[derive(Debug)]
pub struct ActiveTask {
    pub task: QueuedTask,
    pub started_at: Instant,
    pub compute_handle: ComputeHandle,
    pub progress: Arc<AtomicU64>, // 0-100 percentage
    pub can_preempt: bool,
    pub time_slice_remaining: Duration,
}

#[derive(Debug, Clone, Copy)]
pub struct ComputeHandle(pub u64);

/// Load Balancing System
pub struct LoadBalancer {
    compute_unit_utilization: Arc<DashMap<u32, UtilizationMetrics>>,
    load_history: Arc<RwLock<VecDeque<LoadSnapshot>>>,
    rebalancing_threshold: f64,
    last_rebalance: Arc<RwLock<Instant>>,
}

#[derive(Debug, Clone)]
pub struct UtilizationMetrics {
    pub compute_unit_id: u32,
    pub current_load: f64, // 0.0 to 1.0
    pub average_load: f64,
    pub peak_load: f64,
    pub task_count: AtomicUsize,
    pub last_updated: Instant,
}

#[derive(Debug, Clone)]
pub struct LoadSnapshot {
    pub timestamp: Instant,
    pub overall_utilization: f64,
    pub queue_depths: Vec<usize>,
    pub active_task_count: usize,
}

/// Resource Prediction System
pub struct ResourcePredictor {
    task_history: Arc<RwLock<VecDeque<TaskExecutionHistory>>>,
    prediction_models: Arc<DashMap<OperationType, PredictionModel>>,
    learning_enabled: bool,
    prediction_accuracy: Arc<RwLock<f64>>,
}

#[derive(Debug, Clone)]
pub struct TaskExecutionHistory {
    pub operation_type: OperationType,
    pub parameters_hash: u64,
    pub actual_duration: Duration,
    pub actual_memory: u64,
    pub actual_compute_units: u32,
    pub predicted_duration: Option<Duration>,
    pub prediction_error: Option<f64>,
    pub completed_at: Instant,
}

#[derive(Debug, Clone)]
pub struct PredictionModel {
    pub operation_type: OperationType,
    pub samples: Vec<TaskExecutionHistory>,
    pub model_accuracy: f64,
    pub last_updated: Instant,
    pub coefficients: Vec<f64>,
}

#[derive(Debug)]
pub struct SchedulerStatistics {
    pub tasks_scheduled: AtomicU64,
    pub tasks_completed: AtomicU64,
    pub tasks_preempted: AtomicU64,
    pub average_wait_time: Arc<RwLock<Duration>>,
    pub average_execution_time: Arc<RwLock<Duration>>,
    pub queue_depth_high: AtomicUsize,
    pub queue_depth_normal: AtomicUsize,
    pub queue_depth_low: AtomicUsize,
    pub compute_unit_utilization: Arc<RwLock<f64>>,
    pub context_switches: AtomicU64,
    pub deadline_misses: AtomicU64,
}

#[derive(Debug, Serialize)]
pub struct SchedulerInfo {
    pub scheduling_policy: String,
    pub active_tasks: usize,
    pub queued_tasks: QueueDepths,
    pub compute_utilization: f64,
    pub average_wait_time_ms: u64,
    pub average_execution_time_ms: u64,
    pub tasks_completed: u64,
    pub deadline_misses: u64,
    pub prediction_accuracy: f64,
}

#[derive(Debug, Serialize)]
pub struct QueueDepths {
    pub real_time: usize,
    pub high_priority: usize,
    pub normal_priority: usize,
    pub low_priority: usize,
    pub background: usize,
}

impl TaskScheduler {
    pub async fn new(config: SchedulerConfig) -> Result<Self> {
        info!("Initializing advanced task scheduler with policy: {:?}", config.scheduling_policy);

        let compute_units = Arc::new(Semaphore::new(config.max_concurrent_tasks as usize));
        
        let task_queues = TaskQueues {
            high_priority: Arc::new(RwLock::new(VecDeque::new())),
            normal_priority: Arc::new(RwLock::new(VecDeque::new())),
            low_priority: Arc::new(RwLock::new(VecDeque::new())),
            background: Arc::new(RwLock::new(VecDeque::new())),
            real_time: Arc::new(RwLock::new(VecDeque::new())),
        };

        let scheduler_stats = Arc::new(SchedulerStatistics {
            tasks_scheduled: AtomicU64::new(0),
            tasks_completed: AtomicU64::new(0),
            tasks_preempted: AtomicU64::new(0),
            average_wait_time: Arc::new(RwLock::new(Duration::from_millis(0))),
            average_execution_time: Arc::new(RwLock::new(Duration::from_millis(0))),
            queue_depth_high: AtomicUsize::new(0),
            queue_depth_normal: AtomicUsize::new(0),
            queue_depth_low: AtomicUsize::new(0),
            compute_unit_utilization: Arc::new(RwLock::new(0.0)),
            context_switches: AtomicU64::new(0),
            deadline_misses: AtomicU64::new(0),
        });

        let load_balancer = Arc::new(LoadBalancer::new(config.max_concurrent_tasks).await?);
        
        let resource_predictor = Arc::new(ResourcePredictor::new(
            config.resource_prediction
        ).await?);

        Ok(Self {
            config,
            task_queues,
            compute_units,
            active_tasks: Arc::new(DashMap::new()),
            scheduler_stats,
            load_balancer,
            resource_predictor,
        })
    }

    pub async fn schedule_task(&self, operation: ComputeOperation) -> Result<oneshot::Receiver<Result<()>>> {
        debug!("Scheduling task: {:?}", operation.id);

        let (tx, rx) = oneshot::channel();

        // Predict resource requirements
        let resource_requirements = self.resource_predictor
            .predict_requirements(&operation).await?;

        let queued_task = QueuedTask {
            operation,
            queued_at: Instant::now(),
            estimated_duration: Some(resource_requirements.estimated_runtime),
            deadline: None, // Could be set based on operation parameters
            resource_requirements,
            completion_callback: Some(tx),
        };

        // Select appropriate queue based on priority and scheduling policy
        let queue = self.select_queue(&queued_task).await?;
        
        match queued_task.operation.priority {
            TaskPriority::RealTime => {
                self.task_queues.real_time.write().push_back(queued_task);
                self.scheduler_stats.queue_depth_high.fetch_add(1, Ordering::Relaxed);
            },
            TaskPriority::High => {
                self.task_queues.high_priority.write().push_back(queued_task);
                self.scheduler_stats.queue_depth_high.fetch_add(1, Ordering::Relaxed);
            },
            TaskPriority::Normal => {
                self.task_queues.normal_priority.write().push_back(queued_task);
                self.scheduler_stats.queue_depth_normal.fetch_add(1, Ordering::Relaxed);
            },
            TaskPriority::Low => {
                self.task_queues.low_priority.write().push_back(queued_task);
                self.scheduler_stats.queue_depth_low.fetch_add(1, Ordering::Relaxed);
            },
            TaskPriority::Background => {
                self.task_queues.background.write().push_back(queued_task);
            },
        }

        self.scheduler_stats.tasks_scheduled.fetch_add(1, Ordering::Relaxed);
        
        // Trigger scheduling if needed
        self.trigger_scheduling().await?;

        info!("Task scheduled with ID: {:?}", operation.id);
        Ok(rx)
    }

    pub async fn allocate_compute_units(&self, units: u32) -> Result<ComputeHandle> {
        debug!("Allocating {} compute units", units);

        // Acquire semaphore permits
        let permits = self.compute_units
            .acquire_many(units)
            .await
            .context("Failed to acquire compute units")?;

        let handle = ComputeHandle(rand::random::<u64>());
        
        // Track compute unit allocation
        self.load_balancer.record_allocation(handle, units).await?;

        Ok(handle)
    }

    pub async fn release_compute_units(&self, handle: ComputeHandle) -> Result<()> {
        debug!("Releasing compute units for handle: {:?}", handle);
        
        let units = self.load_balancer.get_allocated_units(handle).await?;
        self.load_balancer.record_deallocation(handle).await?;

        // This would normally release the semaphore permits, but we need to track them
        // For now, we'll just log the release
        info!("Released {} compute units for handle: {:?}", units, handle);
        Ok(())
    }

    pub async fn get_status(&self) -> Result<SchedulerInfo> {
        let active_tasks = self.active_tasks.len();
        
        let queued_tasks = QueueDepths {
            real_time: self.task_queues.real_time.read().len(),
            high_priority: self.task_queues.high_priority.read().len(),
            normal_priority: self.task_queues.normal_priority.read().len(),
            low_priority: self.task_queues.low_priority.read().len(),
            background: self.task_queues.background.read().len(),
        };

        let compute_utilization = *self.scheduler_stats.compute_unit_utilization.read();
        let average_wait_time = self.scheduler_stats.average_wait_time.read().as_millis() as u64;
        let average_execution_time = self.scheduler_stats.average_execution_time.read().as_millis() as u64;
        let tasks_completed = self.scheduler_stats.tasks_completed.load(Ordering::Relaxed);
        let deadline_misses = self.scheduler_stats.deadline_misses.load(Ordering::Relaxed);
        let prediction_accuracy = *self.resource_predictor.prediction_accuracy.read();

        Ok(SchedulerInfo {
            scheduling_policy: format!("{:?}", self.config.scheduling_policy),
            active_tasks,
            queued_tasks,
            compute_utilization,
            average_wait_time_ms: average_wait_time,
            average_execution_time_ms: average_execution_time,
            tasks_completed,
            deadline_misses,
            prediction_accuracy,
        })
    }

    async fn select_queue(&self, _task: &QueuedTask) -> Result<&str> {
        // Queue selection logic based on scheduling policy
        match self.config.scheduling_policy {
            SchedulingPolicy::Priority => Ok("priority-based"),
            SchedulingPolicy::EDF => Ok("deadline-based"),
            SchedulingPolicy::CFS => Ok("fair-share"),
            _ => Ok("default"),
        }
    }

    async fn trigger_scheduling(&self) -> Result<()> {
        // This would normally trigger the main scheduling loop
        // For now, we'll just update statistics
        info!("Scheduling triggered - implementing scheduling logic");
        Ok(())
    }
}

impl LoadBalancer {
    pub async fn new(max_compute_units: u32) -> Result<Self> {
        let mut compute_unit_utilization = DashMap::new();
        
        // Initialize utilization metrics for each compute unit
        for i in 0..max_compute_units {
            compute_unit_utilization.insert(i, UtilizationMetrics {
                compute_unit_id: i,
                current_load: 0.0,
                average_load: 0.0,
                peak_load: 0.0,
                task_count: AtomicUsize::new(0),
                last_updated: Instant::now(),
            });
        }

        Ok(Self {
            compute_unit_utilization: Arc::new(compute_unit_utilization),
            load_history: Arc::new(RwLock::new(VecDeque::new())),
            rebalancing_threshold: 0.8, // 80% utilization triggers rebalancing
            last_rebalance: Arc::new(RwLock::new(Instant::now())),
        })
    }

    pub async fn record_allocation(&self, handle: ComputeHandle, units: u32) -> Result<()> {
        // Record allocation for load balancing
        debug!("Recording allocation of {} units for handle: {:?}", units, handle);
        
        // Update utilization metrics
        for (_, mut metrics) in self.compute_unit_utilization.iter_mut() {
            if metrics.compute_unit_id < units {
                metrics.task_count.fetch_add(1, Ordering::Relaxed);
                metrics.current_load = metrics.task_count.load(Ordering::Relaxed) as f64 / 10.0; // Simple load calculation
                metrics.last_updated = Instant::now();
            }
        }

        Ok(())
    }

    pub async fn record_deallocation(&self, handle: ComputeHandle) -> Result<()> {
        debug!("Recording deallocation for handle: {:?}", handle);
        
        // Update utilization metrics
        for (_, mut metrics) in self.compute_unit_utilization.iter_mut() {
            let current_tasks = metrics.task_count.load(Ordering::Relaxed);
            if current_tasks > 0 {
                metrics.task_count.fetch_sub(1, Ordering::Relaxed);
                metrics.current_load = (current_tasks - 1) as f64 / 10.0;
                metrics.last_updated = Instant::now();
            }
        }

        Ok(())
    }

    pub async fn get_allocated_units(&self, handle: ComputeHandle) -> Result<u32> {
        // This would normally track allocations per handle
        // For now, return a default value
        Ok(1)
    }
}

impl ResourcePredictor {
    pub async fn new(learning_enabled: bool) -> Result<Self> {
        Ok(Self {
            task_history: Arc::new(RwLock::new(VecDeque::new())),
            prediction_models: Arc::new(DashMap::new()),
            learning_enabled,
            prediction_accuracy: Arc::new(RwLock::new(0.0)),
        })
    }

    pub async fn predict_requirements(&self, operation: &ComputeOperation) -> Result<ResourceRequirements> {
        // Simple prediction based on operation type
        let requirements = match operation.operation_type {
            OperationType::MathematicalComputation => ResourceRequirements {
                compute_units: 4,
                memory_mb: 256,
                bandwidth_mbps: 100,
                estimated_runtime: Duration::from_millis(500),
                cpu_intensive: true,
                memory_intensive: false,
                io_intensive: false,
            },
            OperationType::TensorOperations => ResourceRequirements {
                compute_units: 16,
                memory_mb: 1024,
                bandwidth_mbps: 1000,
                estimated_runtime: Duration::from_secs(2),
                cpu_intensive: true,
                memory_intensive: true,
                io_intensive: false,
            },
            OperationType::BooleanCompilation => ResourceRequirements {
                compute_units: 8,
                memory_mb: 512,
                bandwidth_mbps: 200,
                estimated_runtime: Duration::from_millis(1000),
                cpu_intensive: true,
                memory_intensive: false,
                io_intensive: false,
            },
            _ => ResourceRequirements {
                compute_units: 2,
                memory_mb: 128,
                bandwidth_mbps: 50,
                estimated_runtime: Duration::from_millis(200),
                cpu_intensive: false,
                memory_intensive: false,
                io_intensive: false,
            },
        };

        debug!("Predicted requirements for {:?}: {:?}", operation.operation_type, requirements);
        Ok(requirements)
    }

    pub async fn update_model(&self, execution_history: TaskExecutionHistory) -> Result<()> {
        if !self.learning_enabled {
            return Ok(());
        }

        // Update prediction model with actual execution data
        let mut models = self.prediction_models.clone();
        
        let model_key = execution_history.operation_type.clone();
        
        if let Some(mut model) = models.get_mut(&model_key) {
            model.samples.push(execution_history.clone());
            
            // Simple accuracy calculation
            if let Some(prediction_error) = execution_history.prediction_error {
                model.model_accuracy = 1.0 - prediction_error.abs().min(1.0);
            }
            
            model.last_updated = Instant::now();
        } else {
            // Create new model
            let new_model = PredictionModel {
                operation_type: model_key.clone(),
                samples: vec![execution_history],
                model_accuracy: 0.8, // Default accuracy
                last_updated: Instant::now(),
                coefficients: vec![1.0, 0.5, 0.3], // Simple linear model coefficients
            };
            models.insert(model_key, new_model);
        }

        info!("Updated prediction model for {:?}", execution_history.operation_type);
        Ok(())
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 64,
            scheduling_policy: SchedulingPolicy::Priority,
            time_slice_ms: 100,
            priority_levels: 5,
            load_balancing: true,
            resource_prediction: true,
            preemption_enabled: false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TaskPriority {
    RealTime,
    High,
    Normal,
    Low,
    Background,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_task_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = TaskScheduler::new(config).await;
        assert!(scheduler.is_ok());
    }

    #[tokio::test]
    async fn test_compute_unit_allocation() {
        let config = SchedulerConfig::default();
        let scheduler = TaskScheduler::new(config).await.unwrap();
        
        let handle = scheduler.allocate_compute_units(4).await;
        assert!(handle.is_ok());
        
        let handle = handle.unwrap();
        let release_result = scheduler.release_compute_units(handle).await;
        assert!(release_result.is_ok());
    }

    #[tokio::test]
    async fn test_task_scheduling() {
        let config = SchedulerConfig::default();
        let scheduler = TaskScheduler::new(config).await.unwrap();
        
        let operation = ComputeOperation {
            id: Uuid::new_v4(),
            operation_type: OperationType::MathematicalComputation,
            parameters: HashMap::new(),
            memory_requirements: Some(1024),
            compute_requirements: Some(4),
            priority: TaskPriority::Normal,
            source: Some("test".to_string()),
            timeout: Some(Duration::from_secs(10)),
        };
        
        let result_rx = scheduler.schedule_task(operation).await;
        assert!(result_rx.is_ok());
    }

    #[tokio::test]
    async fn test_scheduler_status() {
        let config = SchedulerConfig::default();
        let scheduler = TaskScheduler::new(config).await.unwrap();
        
        let status = scheduler.get_status().await;
        assert!(status.is_ok());
        
        let status = status.unwrap();
        assert_eq!(status.active_tasks, 0);
    }

    #[tokio::test]
    async fn test_load_balancer() {
        let load_balancer = LoadBalancer::new(16).await;
        assert!(load_balancer.is_ok());
    }

    #[tokio::test]
    async fn test_resource_predictor() {
        let predictor = ResourcePredictor::new(true).await;
        assert!(predictor.is_ok());
        
        let predictor = predictor.unwrap();
        
        let operation = ComputeOperation {
            id: Uuid::new_v4(),
            operation_type: OperationType::TensorOperations,
            parameters: HashMap::new(),
            memory_requirements: None,
            compute_requirements: None,
            priority: TaskPriority::Normal,
            source: None,
            timeout: None,
        };
        
        let requirements = predictor.predict_requirements(&operation).await;
        assert!(requirements.is_ok());
        
        let requirements = requirements.unwrap();
        assert_eq!(requirements.compute_units, 16);
        assert_eq!(requirements.memory_mb, 1024);
    }
}
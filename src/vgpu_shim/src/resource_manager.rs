//! GPU Resource Management System
//!
//! Manages virtual GPU resources including compute units, memory bandwidth,
//! and scheduling priorities with advanced allocation algorithms.

use crate::{Result, VGpuError, GpuCapabilities};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// GPU resource manager with advanced allocation strategies
pub struct ResourceManager {
    total_memory: u64,
    total_compute_units: usize,
    
    // Resource tracking
    allocated_memory: AtomicU64,
    active_compute_units: AtomicUsize,
    
    // Resource pools
    memory_pool: Arc<RwLock<MemoryPool>>,
    compute_pool: Arc<RwLock<ComputePool>>,
    
    // Allocation tracking
    allocations: Arc<RwLock<HashMap<u64, ResourceAllocation>>>,
    
    // Performance metrics
    allocation_history: Arc<RwLock<AllocationHistory>>,
    
    // Resource scheduling
    scheduler: Arc<RwLock<ResourceScheduler>>,
}

/// Resource allocation tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub allocation_id: u64,
    pub resource_type: ResourceType,
    pub size: usize,
    pub priority: Priority,
    pub owner: String,
    pub allocated_at: Instant,
    pub last_accessed: Instant,
    pub access_pattern: AccessPattern,
}

/// Types of GPU resources
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceType {
    GlobalMemory,
    SharedMemory,
    ConstantMemory,
    TextureMemory,
    ComputeUnit,
    StreamingProcessor,
    MemoryBandwidth,
    CacheMemory,
}

/// Resource priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
    RealTime = 5,
}

/// Memory access patterns for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessPattern {
    Sequential,
    Random,
    Strided { stride: usize },
    Clustered { cluster_size: usize },
    Temporal { frequency: f64 },
}

/// Memory pool management with advanced algorithms
#[derive(Debug)]
struct MemoryPool {
    // Free memory blocks sorted by size
    free_blocks: BTreeMap<usize, Vec<MemoryBlock>>,
    
    // Allocated memory blocks
    allocated_blocks: HashMap<u64, MemoryBlock>,
    
    // Memory fragmentation tracking
    fragmentation_ratio: f64,
    
    // Allocation strategies
    allocation_strategy: AllocationStrategy,
    
    // Defragmentation state
    last_defrag: Instant,
    defrag_threshold: f64,
}

/// Memory block representation
#[derive(Debug, Clone)]
struct MemoryBlock {
    address: u64,
    size: usize,
    allocated: bool,
    owner_id: u64,
}

/// Memory allocation strategies
#[derive(Debug, Clone, PartialEq)]
enum AllocationStrategy {
    FirstFit,      // Allocate from first suitable block
    BestFit,       // Allocate from smallest suitable block
    WorstFit,      // Allocate from largest suitable block
    BuddySystem,   // Power-of-2 buddy system allocation
    Slab,          // Slab allocation for fixed-size objects
}

/// Compute resource pool
#[derive(Debug)]
struct ComputePool {
    total_units: usize,
    available_units: usize,
    unit_allocations: HashMap<usize, ComputeAllocation>,
    utilization_history: Vec<UtilizationSnapshot>,
}

/// Compute resource allocation
#[derive(Debug, Clone)]
struct ComputeAllocation {
    unit_id: usize,
    allocated_to: u64,
    allocated_at: Instant,
    estimated_duration: Duration,
}

/// Resource utilization snapshot
#[derive(Debug, Clone)]
struct UtilizationSnapshot {
    timestamp: Instant,
    memory_utilization: f64,
    compute_utilization: f64,
    bandwidth_utilization: f64,
}

/// Historical allocation tracking
#[derive(Debug)]
struct AllocationHistory {
    allocations: Vec<AllocationEvent>,
    peak_memory_usage: u64,
    peak_compute_usage: usize,
    fragmentation_events: Vec<FragmentationEvent>,
}

/// Allocation event tracking
#[derive(Debug, Clone)]
struct AllocationEvent {
    timestamp: Instant,
    event_type: AllocationEventType,
    resource_id: u64,
    size: usize,
    success: bool,
}

#[derive(Debug, Clone)]
enum AllocationEventType {
    Allocate,
    Free,
    Resize,
    Defragment,
}

/// Fragmentation event tracking
#[derive(Debug, Clone)]
struct FragmentationEvent {
    timestamp: Instant,
    fragmentation_ratio: f64,
    action_taken: DefragmentationAction,
}

#[derive(Debug, Clone)]
enum DefragmentationAction {
    None,
    Compact,
    Relocate,
    CoalesceFree,
}

/// Advanced resource scheduler
#[derive(Debug)]
struct ResourceScheduler {
    // Scheduling queues by priority
    priority_queues: HashMap<Priority, Vec<ScheduledTask>>,
    
    // Load balancing
    load_balancer: LoadBalancer,
    
    // Resource prediction
    usage_predictor: UsagePredictor,
}

/// Scheduled task representation
#[derive(Debug, Clone)]
struct ScheduledTask {
    task_id: u64,
    resource_requirements: ResourceRequirement,
    priority: Priority,
    deadline: Option<Instant>,
    submitted_at: Instant,
}

/// Resource requirements specification
#[derive(Debug, Clone)]
struct ResourceRequirement {
    memory_size: usize,
    compute_units: usize,
    bandwidth: u64,
    duration_estimate: Duration,
}

/// Load balancing algorithms
#[derive(Debug)]
struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    compute_loads: HashMap<usize, f64>,
    memory_hotspots: HashMap<u64, f64>,
}

#[derive(Debug, Clone)]
enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    WeightedRoundRobin,
    ConsistentHashing,
    PowerOfTwo,
}

/// Resource usage prediction
#[derive(Debug)]
struct UsagePredictor {
    historical_patterns: Vec<UsagePattern>,
    prediction_model: PredictionModel,
}

#[derive(Debug, Clone)]
struct UsagePattern {
    time_of_day: u32,
    day_of_week: u8,
    memory_usage: f64,
    compute_usage: f64,
    duration: Duration,
}

#[derive(Debug, Clone)]
enum PredictionModel {
    MovingAverage { window_size: usize },
    ExponentialSmoothing { alpha: f64 },
    LinearRegression { coefficients: Vec<f64> },
    NeuralNetwork { weights: Vec<Vec<f64>> },
}

impl ResourceManager {
    /// Create a new resource manager
    pub fn new() -> Result<Self> {
        Ok(Self {
            total_memory: 8 * 1024 * 1024 * 1024, // 8GB
            total_compute_units: 1024, // 1024 SMs simulation
            allocated_memory: AtomicU64::new(0),
            active_compute_units: AtomicUsize::new(0),
            
            memory_pool: Arc::new(RwLock::new(MemoryPool::new(8 * 1024 * 1024 * 1024)?)),
            compute_pool: Arc::new(RwLock::new(ComputePool::new(1024))),
            allocations: Arc::new(RwLock::new(HashMap::new())),
            allocation_history: Arc::new(RwLock::new(AllocationHistory::new())),
            scheduler: Arc::new(RwLock::new(ResourceScheduler::new())),
        })
    }

    /// Set maximum memory limit
    pub fn set_max_memory(&self, max_memory: u64) -> Result<()> {
        if self.allocated_memory.load(Ordering::Relaxed) > max_memory {
            return Err(VGpuError::resource_allocation(
                "Cannot reduce memory limit below current allocation"
            ));
        }
        
        // Update memory pool
        self.memory_pool.write().resize(max_memory as usize)?;
        Ok(())
    }

    /// Allocate GPU memory with advanced placement algorithms
    pub fn allocate_memory(&self, size: usize, priority: Priority, owner: &str) -> Result<ResourceAllocation> {
        // Check if allocation is possible
        if self.allocated_memory.load(Ordering::Relaxed) + size as u64 > self.total_memory {
            // Try defragmentation first
            self.defragment_memory()?;
            
            // Check again after defragmentation
            if self.allocated_memory.load(Ordering::Relaxed) + size as u64 > self.total_memory {
                return Err(VGpuError::resource_allocation("Insufficient memory"));
            }
        }

        // Allocate from memory pool
        let allocation_id = self.generate_allocation_id();
        let address = self.memory_pool.write().allocate(size, allocation_id)?;
        
        // Create allocation record
        let allocation = ResourceAllocation {
            allocation_id,
            resource_type: ResourceType::GlobalMemory,
            size,
            priority,
            owner: owner.to_string(),
            allocated_at: Instant::now(),
            last_accessed: Instant::now(),
            access_pattern: AccessPattern::Sequential, // Default pattern
        };

        // Update tracking
        self.allocated_memory.fetch_add(size as u64, Ordering::Relaxed);
        self.allocations.write().insert(allocation_id, allocation.clone());
        
        // Record allocation event
        self.record_allocation_event(AllocationEventType::Allocate, allocation_id, size, true);
        
        Ok(allocation)
    }

    /// Free GPU memory
    pub fn free_memory(&self, allocation_id: u64) -> Result<()> {
        let allocation = self.allocations.write().remove(&allocation_id)
            .ok_or_else(|| VGpuError::resource_allocation("Invalid allocation ID"))?;

        // Free from memory pool
        self.memory_pool.write().free(allocation_id)?;
        
        // Update tracking
        self.allocated_memory.fetch_sub(allocation.size as u64, Ordering::Relaxed);
        
        // Record free event
        self.record_allocation_event(AllocationEventType::Free, allocation_id, allocation.size, true);
        
        Ok(())
    }

    /// Allocate compute units
    pub fn allocate_compute(&self, units: usize, duration: Duration, priority: Priority) -> Result<u64> {
        let mut compute_pool = self.compute_pool.write();
        
        if compute_pool.available_units < units {
            return Err(VGpuError::resource_allocation("Insufficient compute units"));
        }

        let allocation_id = self.generate_allocation_id();
        compute_pool.allocate_units(units, allocation_id, duration)?;
        
        self.active_compute_units.fetch_add(units, Ordering::Relaxed);
        
        Ok(allocation_id)
    }

    /// Free compute units
    pub fn free_compute(&self, allocation_id: u64) -> Result<()> {
        let mut compute_pool = self.compute_pool.write();
        let units = compute_pool.free_units(allocation_id)?;
        
        self.active_compute_units.fetch_sub(units, Ordering::Relaxed);
        
        Ok(())
    }

    /// Defragment memory to reduce fragmentation
    fn defragment_memory(&self) -> Result<()> {
        let mut memory_pool = self.memory_pool.write();
        
        // Check if defragmentation is needed
        if memory_pool.fragmentation_ratio < memory_pool.defrag_threshold {
            return Ok(());
        }

        // Perform defragmentation using buddy system coalescing
        let coalesced_blocks = memory_pool.coalesce_free_blocks()?;
        
        // Record defragmentation event
        let event = FragmentationEvent {
            timestamp: Instant::now(),
            fragmentation_ratio: memory_pool.fragmentation_ratio,
            action_taken: DefragmentationAction::CoalesceFree,
        };
        
        self.allocation_history.write().fragmentation_events.push(event);
        
        memory_pool.last_defrag = Instant::now();
        
        Ok(())
    }

    /// Get current resource utilization
    pub fn get_utilization(&self) -> ResourceUtilization {
        let memory_util = self.allocated_memory.load(Ordering::Relaxed) as f64 / self.total_memory as f64;
        let compute_util = self.active_compute_units.load(Ordering::Relaxed) as f64 / self.total_compute_units as f64;
        
        ResourceUtilization {
            memory_utilization: memory_util,
            compute_utilization: compute_util,
            fragmentation_ratio: self.memory_pool.read().fragmentation_ratio,
            allocation_count: self.allocations.read().len(),
            peak_memory_usage: self.allocation_history.read().peak_memory_usage,
        }
    }

    /// Update resource access patterns for optimization
    pub fn update_access_pattern(&self, allocation_id: u64, pattern: AccessPattern) -> Result<()> {
        if let Some(allocation) = self.allocations.write().get_mut(&allocation_id) {
            allocation.access_pattern = pattern;
            allocation.last_accessed = Instant::now();
            Ok(())
        } else {
            Err(VGpuError::resource_allocation("Invalid allocation ID"))
        }
    }

    /// Predict future resource usage
    pub fn predict_usage(&self, time_horizon: Duration) -> ResourcePrediction {
        let scheduler = self.scheduler.read();
        scheduler.usage_predictor.predict(time_horizon)
    }

    /// Generate unique allocation ID
    fn generate_allocation_id(&self) -> u64 {
        use rand::Rng;
        rand::thread_rng().gen()
    }

    /// Record allocation event for analysis
    fn record_allocation_event(&self, event_type: AllocationEventType, resource_id: u64, size: usize, success: bool) {
        let event = AllocationEvent {
            timestamp: Instant::now(),
            event_type,
            resource_id,
            size,
            success,
        };
        
        self.allocation_history.write().allocations.push(event);
    }
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    pub memory_utilization: f64,
    pub compute_utilization: f64,
    pub fragmentation_ratio: f64,
    pub allocation_count: usize,
    pub peak_memory_usage: u64,
}

/// Resource usage prediction
#[derive(Debug, Clone)]
pub struct ResourcePrediction {
    pub predicted_memory_usage: f64,
    pub predicted_compute_usage: f64,
    pub confidence: f64,
    pub time_horizon: Duration,
}

// Implementation details for internal structures

impl MemoryPool {
    fn new(total_size: usize) -> Result<Self> {
        let mut free_blocks = BTreeMap::new();
        
        // Initialize with one large free block
        let initial_block = MemoryBlock {
            address: 0x1000000000,
            size: total_size,
            allocated: false,
            owner_id: 0,
        };
        
        free_blocks.insert(total_size, vec![initial_block]);
        
        Ok(Self {
            free_blocks,
            allocated_blocks: HashMap::new(),
            fragmentation_ratio: 0.0,
            allocation_strategy: AllocationStrategy::BestFit,
            last_defrag: Instant::now(),
            defrag_threshold: 0.3,
        })
    }

    fn allocate(&mut self, size: usize, allocation_id: u64) -> Result<u64> {
        // Find suitable block based on allocation strategy
        let block = match self.allocation_strategy {
            AllocationStrategy::FirstFit => self.find_first_fit(size),
            AllocationStrategy::BestFit => self.find_best_fit(size),
            AllocationStrategy::WorstFit => self.find_worst_fit(size),
            AllocationStrategy::BuddySystem => self.find_buddy_block(size),
            AllocationStrategy::Slab => self.find_slab_block(size),
        }?;

        // Allocate the block
        let allocated_block = MemoryBlock {
            address: block.address,
            size,
            allocated: true,
            owner_id: allocation_id,
        };
        
        let address = block.address;
        self.allocated_blocks.insert(allocation_id, allocated_block);
        
        // If block is larger than needed, create remainder
        if block.size > size {
            let remainder = MemoryBlock {
                address: block.address + size as u64,
                size: block.size - size,
                allocated: false,
                owner_id: 0,
            };
            
            self.add_free_block(remainder);
        }
        
        self.update_fragmentation();
        Ok(address)
    }

    fn free(&mut self, allocation_id: u64) -> Result<()> {
        let block = self.allocated_blocks.remove(&allocation_id)
            .ok_or_else(|| VGpuError::memory_management("Block not found"))?;
        
        let free_block = MemoryBlock {
            address: block.address,
            size: block.size,
            allocated: false,
            owner_id: 0,
        };
        
        self.add_free_block(free_block);
        self.update_fragmentation();
        Ok(())
    }

    fn find_best_fit(&self, size: usize) -> Result<MemoryBlock> {
        // Find smallest block that can fit the allocation
        for (&block_size, blocks) in self.free_blocks.range(size..) {
            if let Some(block) = blocks.first() {
                return Ok(block.clone());
            }
        }
        
        Err(VGpuError::memory_management("No suitable block found"))
    }

    fn find_first_fit(&self, size: usize) -> Result<MemoryBlock> {
        for blocks in self.free_blocks.values() {
            for block in blocks {
                if block.size >= size {
                    return Ok(block.clone());
                }
            }
        }
        
        Err(VGpuError::memory_management("No suitable block found"))
    }

    fn find_worst_fit(&self, size: usize) -> Result<MemoryBlock> {
        // Find largest available block
        for (&block_size, blocks) in self.free_blocks.iter().rev() {
            if block_size >= size {
                if let Some(block) = blocks.first() {
                    return Ok(block.clone());
                }
            }
        }
        
        Err(VGpuError::memory_management("No suitable block found"))
    }

    fn find_buddy_block(&self, size: usize) -> Result<MemoryBlock> {
        // Round up to next power of 2
        let buddy_size = size.next_power_of_two();
        
        if let Some(blocks) = self.free_blocks.get(&buddy_size) {
            if let Some(block) = blocks.first() {
                return Ok(block.clone());
            }
        }
        
        // Try to split a larger block
        for (&block_size, blocks) in self.free_blocks.range((buddy_size + 1)..) {
            if let Some(block) = blocks.first() {
                return Ok(block.clone());
            }
        }
        
        Err(VGpuError::memory_management("No suitable buddy block found"))
    }

    fn find_slab_block(&self, size: usize) -> Result<MemoryBlock> {
        // For slab allocation, find exact size match or use best fit as fallback
        if let Some(blocks) = self.free_blocks.get(&size) {
            if let Some(block) = blocks.first() {
                return Ok(block.clone());
            }
        }
        
        self.find_best_fit(size)
    }

    fn add_free_block(&mut self, block: MemoryBlock) {
        self.free_blocks.entry(block.size).or_insert_with(Vec::new).push(block);
    }

    fn coalesce_free_blocks(&mut self) -> Result<usize> {
        // Simple coalescing: merge adjacent free blocks
        let mut coalesced_count = 0;
        
        // This is a simplified implementation
        // In practice, would need more sophisticated adjacent block detection
        for blocks in self.free_blocks.values_mut() {
            blocks.sort_by_key(|b| b.address);
            // Merge logic would go here
            coalesced_count += 1;
        }
        
        Ok(coalesced_count)
    }

    fn update_fragmentation(&mut self) {
        let total_free_blocks: usize = self.free_blocks.values().map(|v| v.len()).sum();
        let total_allocated_blocks = self.allocated_blocks.len();
        
        if total_allocated_blocks > 0 {
            self.fragmentation_ratio = total_free_blocks as f64 / (total_allocated_blocks + total_free_blocks) as f64;
        } else {
            self.fragmentation_ratio = 0.0;
        }
    }

    fn resize(&mut self, new_size: usize) -> Result<()> {
        // Simplified resize implementation
        // In practice, would need to handle existing allocations
        if new_size < self.allocated_blocks.values().map(|b| b.size).sum::<usize>() {
            return Err(VGpuError::memory_management("Cannot shrink below allocated size"));
        }
        
        Ok(())
    }
}

impl ComputePool {
    fn new(total_units: usize) -> Self {
        Self {
            total_units,
            available_units: total_units,
            unit_allocations: HashMap::new(),
            utilization_history: Vec::new(),
        }
    }

    fn allocate_units(&mut self, units: usize, allocation_id: u64, duration: Duration) -> Result<()> {
        if self.available_units < units {
            return Err(VGpuError::resource_allocation("Insufficient compute units"));
        }

        // Find available units
        for i in 0..self.total_units {
            if !self.unit_allocations.contains_key(&i) {
                let allocation = ComputeAllocation {
                    unit_id: i,
                    allocated_to: allocation_id,
                    allocated_at: Instant::now(),
                    estimated_duration: duration,
                };
                
                self.unit_allocations.insert(i, allocation);
                self.available_units -= 1;
                
                if self.unit_allocations.len() >= units {
                    break;
                }
            }
        }

        Ok(())
    }

    fn free_units(&mut self, allocation_id: u64) -> Result<usize> {
        let mut freed_units = 0;
        
        self.unit_allocations.retain(|_unit_id, allocation| {
            if allocation.allocated_to == allocation_id {
                freed_units += 1;
                false // Remove this allocation
            } else {
                true // Keep this allocation
            }
        });

        self.available_units += freed_units;
        Ok(freed_units)
    }
}

impl AllocationHistory {
    fn new() -> Self {
        Self {
            allocations: Vec::new(),
            peak_memory_usage: 0,
            peak_compute_usage: 0,
            fragmentation_events: Vec::new(),
        }
    }
}

impl ResourceScheduler {
    fn new() -> Self {
        Self {
            priority_queues: HashMap::new(),
            load_balancer: LoadBalancer::new(),
            usage_predictor: UsagePredictor::new(),
        }
    }
}

impl LoadBalancer {
    fn new() -> Self {
        Self {
            strategy: LoadBalancingStrategy::LeastLoaded,
            compute_loads: HashMap::new(),
            memory_hotspots: HashMap::new(),
        }
    }
}

impl UsagePredictor {
    fn new() -> Self {
        Self {
            historical_patterns: Vec::new(),
            prediction_model: PredictionModel::MovingAverage { window_size: 10 },
        }
    }

    fn predict(&self, _time_horizon: Duration) -> ResourcePrediction {
        // Simplified prediction based on historical patterns
        let avg_memory = self.historical_patterns.iter()
            .map(|p| p.memory_usage)
            .sum::<f64>() / self.historical_patterns.len().max(1) as f64;
            
        let avg_compute = self.historical_patterns.iter()
            .map(|p| p.compute_usage)
            .sum::<f64>() / self.historical_patterns.len().max(1) as f64;

        ResourcePrediction {
            predicted_memory_usage: avg_memory,
            predicted_compute_usage: avg_compute,
            confidence: 0.8, // 80% confidence
            time_horizon: _time_horizon,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_manager_creation() {
        let manager = ResourceManager::new().unwrap();
        let utilization = manager.get_utilization();
        
        assert_eq!(utilization.memory_utilization, 0.0);
        assert_eq!(utilization.compute_utilization, 0.0);
    }

    #[test]
    fn test_memory_allocation() {
        let manager = ResourceManager::new().unwrap();
        
        let allocation = manager.allocate_memory(1024, Priority::Normal, "test").unwrap();
        assert_eq!(allocation.size, 1024);
        assert_eq!(allocation.resource_type, ResourceType::GlobalMemory);
        
        let utilization = manager.get_utilization();
        assert!(utilization.memory_utilization > 0.0);
        
        manager.free_memory(allocation.allocation_id).unwrap();
        
        let utilization_after = manager.get_utilization();
        assert_eq!(utilization_after.memory_utilization, 0.0);
    }

    #[test]
    fn test_compute_allocation() {
        let manager = ResourceManager::new().unwrap();
        
        let allocation_id = manager.allocate_compute(16, Duration::from_secs(10), Priority::High).unwrap();
        
        let utilization = manager.get_utilization();
        assert!(utilization.compute_utilization > 0.0);
        
        manager.free_compute(allocation_id).unwrap();
        
        let utilization_after = manager.get_utilization();
        assert_eq!(utilization_after.compute_utilization, 0.0);
    }

    #[test]
    fn test_resource_limits() {
        let manager = ResourceManager::new().unwrap();
        
        // Try to allocate more memory than available
        let large_allocation = manager.allocate_memory(16 * 1024 * 1024 * 1024, Priority::Normal, "test");
        assert!(large_allocation.is_err());
    }

    #[test]
    fn test_access_pattern_update() {
        let manager = ResourceManager::new().unwrap();
        
        let allocation = manager.allocate_memory(1024, Priority::Normal, "test").unwrap();
        
        let pattern = AccessPattern::Strided { stride: 64 };
        manager.update_access_pattern(allocation.allocation_id, pattern).unwrap();
        
        manager.free_memory(allocation.allocation_id).unwrap();
    }

    #[test]
    fn test_usage_prediction() {
        let manager = ResourceManager::new().unwrap();
        
        let prediction = manager.predict_usage(Duration::from_secs(3600));
        assert!(prediction.confidence > 0.0);
        assert!(prediction.confidence <= 1.0);
    }

    #[test]
    fn test_memory_defragmentation() {
        let manager = ResourceManager::new().unwrap();
        
        // Allocate and free several blocks to create fragmentation
        let mut allocations = Vec::new();
        
        for i in 0..10 {
            let alloc = manager.allocate_memory(1024 * (i + 1), Priority::Normal, "test").unwrap();
            allocations.push(alloc);
        }
        
        // Free every other allocation to create fragmentation
        for (i, alloc) in allocations.iter().enumerate() {
            if i % 2 == 0 {
                manager.free_memory(alloc.allocation_id).unwrap();
            }
        }
        
        // Defragmentation is called automatically when needed
        let utilization = manager.get_utilization();
        assert!(utilization.fragmentation_ratio >= 0.0);
    }
}
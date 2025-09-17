//! Performance optimization and metrics

use crate::*;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};

/// Performance metrics for the God-Index system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Family-specific metrics
    pub family_metrics: HashMap<FamilyId, FamilyMetrics>,
    
    /// Query performance metrics
    pub query_metrics: QueryMetrics,
    
    /// Navigation metrics
    pub navigation_metrics: NavigationMetrics,
    
    /// Memory usage metrics
    pub memory_metrics: MemoryMetrics,
    
    /// System-wide performance statistics
    pub system_metrics: SystemMetrics,
}

/// Query performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetrics {
    pub total_queries: u64,
    pub average_query_time: Duration,
    pub query_type_distribution: HashMap<String, u64>,
    pub cache_hit_rate: f64,
    pub frequent_queries: Vec<(NodeId, NodeId)>, // For path queries
}

/// Navigation performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationMetrics {
    pub total_navigations: u64,
    pub average_navigation_time: Duration,
    pub path_cache_size: usize,
    pub spatial_query_count: u64,
    pub graph_traversal_count: u64,
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub total_memory_usage: u64, // bytes
    pub family_cache_usage: u64,
    pub navigation_cache_usage: u64,
    pub query_cache_usage: u64,
    pub index_memory_usage: u64,
}

/// System-wide performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub uptime: Duration,
    pub cpu_usage: f64, // percentage
    pub memory_usage: f64, // percentage
    pub disk_usage: u64, // bytes
    pub network_io: u64, // bytes
}

/// Optimization report for various system components
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// Navigation graph optimizations
    pub navigation_optimizations: Vec<NavigationOptimization>,
    
    /// Query processing optimizations
    pub query_optimizations: Vec<QueryOptimization>,
    
    /// Family structure optimizations
    pub family_optimizations: Vec<(FamilyId, FamilyOptimization)>,
    
    /// Memory optimizations
    pub memory_optimizations: Vec<MemoryOptimization>,
}

/// Memory optimization results
#[derive(Debug, Clone)]
pub struct MemoryOptimization {
    pub optimization_type: MemoryOptimizationType,
    pub memory_saved: u64,
    pub performance_impact: f64,
}

/// Types of memory optimizations
#[derive(Debug, Clone)]
pub enum MemoryOptimizationType {
    CacheCompaction,
    IndexCompression,
    GarbageCollection,
    DataStructureOptimization,
}

impl PerformanceMetrics {
    /// Create new performance metrics instance
    pub fn new() -> Self {
        Self {
            family_metrics: HashMap::new(),
            query_metrics: QueryMetrics::default(),
            navigation_metrics: NavigationMetrics::default(),
            memory_metrics: MemoryMetrics::default(),
            system_metrics: SystemMetrics::default(),
        }
    }

    /// Initialize metrics for a new family
    pub fn initialize_family(&mut self, family_id: FamilyId) {
        self.family_metrics.insert(family_id, FamilyMetrics::default());
    }

    /// Record a query execution
    pub fn record_query(&mut self, query_type: &QueryType, duration: Duration) {
        self.query_metrics.total_queries += 1;
        
        // Update average query time using incremental mean
        let n = self.query_metrics.total_queries as f64;
        let old_avg = self.query_metrics.average_query_time.as_secs_f64();
        let new_duration = duration.as_secs_f64();
        let new_avg = (old_avg * (n - 1.0) + new_duration) / n;
        self.query_metrics.average_query_time = Duration::from_secs_f64(new_avg);
        
        // Update query type distribution
        let query_type_str = format!("{:?}", query_type);
        *self.query_metrics.query_type_distribution.entry(query_type_str).or_insert(0) += 1;
    }

    /// Record a navigation operation
    pub fn record_navigation(&mut self, duration: Duration) {
        self.navigation_metrics.total_navigations += 1;
        
        // Update average navigation time
        let n = self.navigation_metrics.total_navigations as f64;
        let old_avg = self.navigation_metrics.average_navigation_time.as_secs_f64();
        let new_duration = duration.as_secs_f64();
        let new_avg = (old_avg * (n - 1.0) + new_duration) / n;
        self.navigation_metrics.average_navigation_time = Duration::from_secs_f64(new_avg);
    }

    /// Get metrics for a specific family
    pub fn get_family_metrics(&self, family_id: FamilyId) -> FamilyMetrics {
        self.family_metrics.get(&family_id).cloned().unwrap_or_default()
    }

    /// Get frequently executed queries
    pub fn get_frequent_queries(&self) -> Vec<(NodeId, NodeId)> {
        self.query_metrics.frequent_queries.clone()
    }

    /// Update memory usage statistics
    pub fn update_memory_usage(&mut self, component: MemoryComponent, usage: u64) {
        match component {
            MemoryComponent::FamilyCache => self.memory_metrics.family_cache_usage = usage,
            MemoryComponent::NavigationCache => self.memory_metrics.navigation_cache_usage = usage,
            MemoryComponent::QueryCache => self.memory_metrics.query_cache_usage = usage,
            MemoryComponent::IndexStructures => self.memory_metrics.index_memory_usage = usage,
        }
        
        // Update total memory usage
        self.memory_metrics.total_memory_usage = 
            self.memory_metrics.family_cache_usage +
            self.memory_metrics.navigation_cache_usage +
            self.memory_metrics.query_cache_usage +
            self.memory_metrics.index_memory_usage;
    }

    /// Generate optimization recommendations
    pub fn generate_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        // Recommend query caching if cache hit rate is low
        if self.query_metrics.cache_hit_rate < 0.5 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationType::QueryCaching,
                priority: Priority::High,
                description: "Enable more aggressive query caching to improve performance".to_string(),
                expected_improvement: 2.0,
            });
        }
        
        // Recommend memory optimization if usage is high
        if self.memory_metrics.total_memory_usage > 1_000_000_000 { // 1GB
            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationType::MemoryOptimization,
                priority: Priority::Medium,
                description: "Optimize memory usage through cache compaction".to_string(),
                expected_improvement: 1.5,
            });
        }
        
        // Recommend index optimization if queries are slow
        if self.query_metrics.average_query_time > Duration::from_millis(100) {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationType::IndexOptimization,
                priority: Priority::High,
                description: "Optimize index structures to improve query performance".to_string(),
                expected_improvement: 3.0,
            });
        }
        
        recommendations
    }
}

/// Memory component types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryComponent {
    FamilyCache,
    NavigationCache,
    QueryCache,
    IndexStructures,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub recommendation_type: OptimizationType,
    pub priority: Priority,
    pub description: String,
    pub expected_improvement: f64, // Multiplier (e.g., 2.0 = 2x faster)
}

/// Types of optimizations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OptimizationType {
    QueryCaching,
    MemoryOptimization,
    IndexOptimization,
    NavigationOptimization,
    ParallelProcessing,
}

/// Priority levels for optimizations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

impl OptimizationReport {
    /// Create a new empty optimization report
    pub fn new() -> Self {
        Self {
            navigation_optimizations: Vec::new(),
            query_optimizations: Vec::new(),
            family_optimizations: Vec::new(),
            memory_optimizations: Vec::new(),
        }
    }

    /// Calculate total memory saved by optimizations
    pub fn total_memory_saved(&self) -> u64 {
        self.memory_optimizations.iter()
            .map(|opt| opt.memory_saved)
            .sum()
    }

    /// Calculate average performance improvement
    pub fn average_performance_improvement(&self) -> f64 {
        let improvements: Vec<f64> = self.query_optimizations.iter()
            .map(|opt| opt.estimated_speedup)
            .collect();
        
        if improvements.is_empty() {
            1.0
        } else {
            improvements.iter().sum::<f64>() / improvements.len() as f64
        }
    }
}

// Default implementations
impl Default for FamilyMetrics {
    fn default() -> Self {
        Self {
            average_query_time: Duration::from_nanos(0),
            hotspot_analysis: None,
            cache_efficiency: 1.0,
        }
    }
}

impl Default for QueryMetrics {
    fn default() -> Self {
        Self {
            total_queries: 0,
            average_query_time: Duration::from_nanos(0),
            query_type_distribution: HashMap::new(),
            cache_hit_rate: 0.0,
            frequent_queries: Vec::new(),
        }
    }
}

impl Default for NavigationMetrics {
    fn default() -> Self {
        Self {
            total_navigations: 0,
            average_navigation_time: Duration::from_nanos(0),
            path_cache_size: 0,
            spatial_query_count: 0,
            graph_traversal_count: 0,
        }
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            total_memory_usage: 0,
            family_cache_usage: 0,
            navigation_cache_usage: 0,
            query_cache_usage: 0,
            index_memory_usage: 0,
        }
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            uptime: Duration::from_secs(0),
            cpu_usage: 0.0,
            memory_usage: 0.0,
            disk_usage: 0,
            network_io: 0,
        }
    }
}
//! Performance Monitoring System for vGPU
//!
//! Real-time performance monitoring and metrics collection for the virtual GPU
//! including utilization tracking, thermal management, and power monitoring.

use crate::{Result, VGpuError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::RwLock;
use std::time::{Duration, Instant};
use tokio::time::interval;

/// Performance monitoring system
pub struct PerformanceMonitor {
    // Current metrics
    current_metrics: Arc<RwLock<GpuMetrics>>,
    
    // Historical data
    metrics_history: Arc<RwLock<VecDeque<GpuMetrics>>>,
    
    // Monitoring configuration
    config: MonitoringConfig,
    
    // Performance counters
    counters: Arc<RwLock<PerformanceCounters>>,
    
    // Thermal monitoring
    thermal_monitor: Arc<RwLock<ThermalMonitor>>,
    
    // Power monitoring
    power_monitor: Arc<RwLock<PowerMonitor>>,
    
    // Monitoring state
    is_monitoring: Arc<RwLock<bool>>,
}

/// GPU performance metrics
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    pub timestamp: Instant,
    pub compute_utilization: f64,        // 0.0 to 1.0
    pub memory_utilization: f64,         // 0.0 to 1.0  
    pub memory_bandwidth_usage: f64,     // GB/s
    pub temperature: f64,                // Celsius
    pub power_usage: f64,                // Watts
    pub fan_speed: f64,                  // RPM
    pub clock_speed: f64,                // MHz
    pub memory_clock_speed: f64,         // MHz
    pub voltage: f64,                    // Volts
    pub operations_per_second: f64,      // Operations/sec
    pub memory_transfers_per_second: f64, // Transfers/sec
    pub cache_hit_rate: f64,             // 0.0 to 1.0
    pub active_tasks: usize,
    pub queued_tasks: usize,
    pub is_monitoring_active: bool,
    pub max_memory_bytes: u64,
}

/// Performance monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub sampling_interval: Duration,
    pub history_size: usize,
    pub enable_thermal_monitoring: bool,
    pub enable_power_monitoring: bool,
    pub enable_detailed_counters: bool,
    pub alert_thresholds: AlertThresholds,
}

/// Alert thresholds for performance monitoring
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub max_temperature: f64,        // °C
    pub max_power: f64,              // Watts
    pub max_memory_utilization: f64, // 0.0 to 1.0
    pub min_cache_hit_rate: f64,     // 0.0 to 1.0
}

/// Performance counters
#[derive(Debug, Clone)]
pub struct PerformanceCounters {
    // Compute counters
    pub instructions_executed: u64,
    pub cycles_elapsed: u64,
    pub warps_executed: u64,
    pub threads_executed: u64,
    
    // Memory counters
    pub memory_reads: u64,
    pub memory_writes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub memory_bank_conflicts: u64,
    
    // Execution counters
    pub kernel_launches: u64,
    pub context_switches: u64,
    pub interrupts: u64,
    pub dma_transfers: u64,
    
    // Error counters
    pub computation_errors: u64,
    pub memory_errors: u64,
    pub thermal_throttling_events: u64,
    pub power_throttling_events: u64,
}

/// Thermal monitoring system
#[derive(Debug)]
pub struct ThermalMonitor {
    current_temperature: f64,
    temperature_history: VecDeque<ThermalSample>,
    thermal_zones: HashMap<ThermalZone, f64>,
    cooling_strategy: CoolingStrategy,
    throttling_active: bool,
}

/// Thermal zones on the GPU
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ThermalZone {
    Core,           // GPU core
    Memory,         // Memory modules
    PowerDelivery,  // Power delivery circuits
    VRM,           // Voltage regulation modules
    Ambient,       // Ambient sensor
}

/// Thermal sample
#[derive(Debug, Clone)]
pub struct ThermalSample {
    pub timestamp: Instant,
    pub temperature: f64,
    pub zone: ThermalZone,
}

/// Cooling strategies
#[derive(Debug, Clone)]
pub enum CoolingStrategy {
    Passive,                    // No active cooling
    FanControl { target_temp: f64 }, // Fan-based cooling
    ThermalThrottling { threshold: f64 }, // Frequency reduction
    Adaptive,                   // Adaptive cooling based on workload
}

/// Power monitoring system
#[derive(Debug)]
pub struct PowerMonitor {
    current_power: f64,
    power_history: VecDeque<PowerSample>,
    power_rails: HashMap<PowerRail, f64>,
    power_budget: f64,
    power_efficiency: f64, // Operations per watt
}

/// Power rails
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PowerRail {
    Core,      // Core voltage rail
    Memory,    // Memory voltage rail
    IO,        // I/O voltage rail
    PLL,       // PLL voltage rail
    Auxiliary, // Auxiliary circuits
}

/// Power sample
#[derive(Debug, Clone)]
pub struct PowerSample {
    pub timestamp: Instant,
    pub power: f64,
    pub voltage: f64,
    pub current: f64,
    pub rail: PowerRail,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Result<Self> {
        Ok(Self {
            current_metrics: Arc::new(RwLock::new(GpuMetrics::default())),
            metrics_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            config: MonitoringConfig::default(),
            counters: Arc::new(RwLock::new(PerformanceCounters::new())),
            thermal_monitor: Arc::new(RwLock::new(ThermalMonitor::new())),
            power_monitor: Arc::new(RwLock::new(PowerMonitor::new())),
            is_monitoring: Arc::new(RwLock::new(false)),
        })
    }

    /// Start performance monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        *self.is_monitoring.write() = true;
        
        // Start monitoring tasks
        self.start_metrics_collection().await?;
        
        if self.config.enable_thermal_monitoring {
            self.start_thermal_monitoring().await?;
        }
        
        if self.config.enable_power_monitoring {
            self.start_power_monitoring().await?;
        }
        
        Ok(())
    }

    /// Stop performance monitoring
    pub async fn stop_monitoring(&self) -> Result<()> {
        *self.is_monitoring.write() = false;
        Ok(())
    }

    /// Start metrics collection task
    async fn start_metrics_collection(&self) -> Result<()> {
        let current_metrics = self.current_metrics.clone();
        let metrics_history = self.metrics_history.clone();
        let is_monitoring = self.is_monitoring.clone();
        let sampling_interval = self.config.sampling_interval;
        let history_size = self.config.history_size;
        
        tokio::spawn(async move {
            let mut interval_timer = interval(sampling_interval);
            
            while *is_monitoring.read() {
                interval_timer.tick().await;
                
                // Collect current metrics
                let metrics = Self::collect_current_metrics().await;
                
                // Update current metrics
                *current_metrics.write() = metrics.clone();
                
                // Add to history
                let mut history = metrics_history.write();
                history.push_back(metrics);
                
                // Trim history if too large
                while history.len() > history_size {
                    history.pop_front();
                }
            }
        });
        
        Ok(())
    }

    /// Start thermal monitoring task
    async fn start_thermal_monitoring(&self) -> Result<()> {
        let thermal_monitor = self.thermal_monitor.clone();
        let is_monitoring = self.is_monitoring.clone();
        
        tokio::spawn(async move {
            let mut interval_timer = interval(Duration::from_millis(500));
            
            while *is_monitoring.read() {
                interval_timer.tick().await;
                
                {
                    let mut monitor = thermal_monitor.write();
                    monitor.update_temperatures();
                    monitor.apply_thermal_management();
                } // Drop the lock before continuing the loop
            }
        });
        
        Ok(())
    }

    /// Start power monitoring task
    async fn start_power_monitoring(&self) -> Result<()> {
        let power_monitor = self.power_monitor.clone();
        let is_monitoring = self.is_monitoring.clone();
        
        tokio::spawn(async move {
            let mut interval_timer = interval(Duration::from_millis(100));
            
            while *is_monitoring.read() {
                interval_timer.tick().await;
                
                {
                    let mut monitor = power_monitor.write();
                    monitor.update_power_measurements();
                    monitor.optimize_power_efficiency();
                } // Drop the lock before continuing the loop
            }
        });
        
        Ok(())
    }

    /// Collect current performance metrics
    async fn collect_current_metrics() -> GpuMetrics {
        // Simulate realistic GPU metrics
        let base_utilization = Self::simulate_workload_utilization();
        let memory_util = Self::simulate_memory_utilization(base_utilization);
        let temperature = Self::simulate_temperature(base_utilization);
        let power = Self::simulate_power_usage(base_utilization, temperature);
        
        GpuMetrics {
            timestamp: Instant::now(),
            compute_utilization: base_utilization,
            memory_utilization: memory_util,
            memory_bandwidth_usage: memory_util * 900.0, // Max 900 GB/s
            temperature,
            power_usage: power,
            fan_speed: Self::calculate_fan_speed(temperature),
            clock_speed: Self::calculate_clock_speed(base_utilization, temperature),
            memory_clock_speed: Self::calculate_memory_clock(memory_util),
            voltage: Self::calculate_voltage(base_utilization),
            operations_per_second: base_utilization * 10_000_000.0, // Up to 10M ops/sec
            memory_transfers_per_second: memory_util * 1_000_000.0,
            cache_hit_rate: Self::simulate_cache_hit_rate(),
            active_tasks: Self::simulate_active_tasks(base_utilization),
            queued_tasks: Self::simulate_queued_tasks(base_utilization),
            is_monitoring_active: true,
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
        }
    }

    /// Simulate workload utilization
    fn simulate_workload_utilization() -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Base utilization with some randomness
        let base = 0.3; // 30% base utilization
        let variance = rng.gen::<f64>() * 0.4; // Up to 40% additional
        (base + variance).min(1.0)
    }

    /// Simulate memory utilization
    fn simulate_memory_utilization(compute_util: f64) -> f64 {
        // Memory utilization often correlates with compute utilization
        let correlation = 0.7;
        let base_memory = compute_util * correlation;
        let noise = (rand::random::<f64>() - 0.5) * 0.1; // ±5% noise
        (base_memory + noise).max(0.0).min(1.0)
    }

    /// Simulate GPU temperature
    fn simulate_temperature(utilization: f64) -> f64 {
        let idle_temp = 35.0; // °C
        let max_temp_rise = 50.0; // Up to 85°C under full load
        idle_temp + (utilization * max_temp_rise)
    }

    /// Simulate power usage
    fn simulate_power_usage(utilization: f64, temperature: f64) -> f64 {
        let idle_power = 50.0; // Watts
        let max_power = 350.0; // Watts under full load
        let thermal_factor = 1.0 + (temperature - 35.0) / 100.0; // Slight increase with temp
        
        let base_power = idle_power + (utilization * (max_power - idle_power));
        base_power * thermal_factor
    }

    /// Calculate fan speed based on temperature
    fn calculate_fan_speed(temperature: f64) -> f64 {
        let min_rpm = 800.0;
        let max_rpm = 3000.0;
        let temp_threshold = 60.0; // Start ramping up fan at 60°C
        
        if temperature < temp_threshold {
            min_rpm
        } else {
            let temp_factor = (temperature - temp_threshold) / (85.0 - temp_threshold);
            min_rpm + temp_factor * (max_rpm - min_rpm)
        }
    }

    /// Calculate GPU clock speed
    fn calculate_clock_speed(utilization: f64, temperature: f64) -> f64 {
        let base_clock = 1200.0; // MHz
        let boost_clock = 1800.0; // MHz
        let thermal_limit = 83.0; // °C
        
        // Boost frequency under load, but throttle if too hot
        let utilization_boost = utilization * (boost_clock - base_clock);
        let thermal_throttle = if temperature > thermal_limit {
            (temperature - thermal_limit) * -20.0 // Reduce by 20MHz per degree above limit
        } else {
            0.0
        };
        
        (base_clock + utilization_boost + thermal_throttle).max(base_clock * 0.5)
    }

    /// Calculate memory clock speed
    fn calculate_memory_clock(memory_util: f64) -> f64 {
        let base_memory_clock = 1000.0; // MHz
        let max_memory_clock = 1750.0; // MHz
        
        base_memory_clock + memory_util * (max_memory_clock - base_memory_clock)
    }

    /// Calculate supply voltage
    fn calculate_voltage(utilization: f64) -> f64 {
        let idle_voltage = 0.85; // Volts
        let max_voltage = 1.10; // Volts
        
        idle_voltage + utilization * (max_voltage - idle_voltage)
    }

    /// Simulate cache hit rate
    fn simulate_cache_hit_rate() -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Good cache hit rate with some variance
        let base_hit_rate = 0.85; // 85%
        let variance = (rng.gen::<f64>() - 0.5) * 0.1; // ±5%
        (base_hit_rate + variance).max(0.0).min(1.0)
    }

    /// Simulate active tasks
    fn simulate_active_tasks(utilization: f64) -> usize {
        (utilization * 16.0) as usize // Up to 16 concurrent tasks
    }

    /// Simulate queued tasks
    fn simulate_queued_tasks(utilization: f64) -> usize {
        if utilization > 0.8 {
            ((utilization - 0.8) * 50.0) as usize // Queue builds up when highly utilized
        } else {
            0
        }
    }

    /// Get current GPU metrics
    pub fn get_current_metrics(&self) -> Result<GpuMetrics> {
        Ok(self.current_metrics.read().clone())
    }

    /// Get historical metrics
    pub fn get_metrics_history(&self, duration: Duration) -> Vec<GpuMetrics> {
        let cutoff_time = Instant::now() - duration;
        self.metrics_history.read()
            .iter()
            .filter(|m| m.timestamp >= cutoff_time)
            .cloned()
            .collect()
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStatistics {
        let history = self.metrics_history.read();
        
        if history.is_empty() {
            return PerformanceStatistics::default();
        }

        let compute_utils: Vec<f64> = history.iter().map(|m| m.compute_utilization).collect();
        let memory_utils: Vec<f64> = history.iter().map(|m| m.memory_utilization).collect();
        let temperatures: Vec<f64> = history.iter().map(|m| m.temperature).collect();
        let power_usage: Vec<f64> = history.iter().map(|m| m.power_usage).collect();

        PerformanceStatistics {
            average_compute_utilization: Self::calculate_average(&compute_utils),
            max_compute_utilization: compute_utils.iter().fold(0.0, |a, &b| a.max(b)),
            average_memory_utilization: Self::calculate_average(&memory_utils),
            max_memory_utilization: memory_utils.iter().fold(0.0, |a, &b| a.max(b)),
            average_temperature: Self::calculate_average(&temperatures),
            max_temperature: temperatures.iter().fold(0.0, |a, &b| a.max(b)),
            average_power: Self::calculate_average(&power_usage),
            max_power: power_usage.iter().fold(0.0, |a, &b| a.max(b)),
            thermal_throttling_events: self.counters.read().thermal_throttling_events,
            power_throttling_events: self.counters.read().power_throttling_events,
            uptime: if let (Some(last), Some(first)) = (history.back(), history.front()) {
                last.timestamp.duration_since(first.timestamp)
            } else {
                Duration::from_secs(0)
            },
        }
    }

    /// Calculate average of a slice of values
    fn calculate_average(values: &[f64]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    /// Update performance counter
    pub fn increment_counter(&self, counter_type: CounterType) {
        let mut counters = self.counters.write();
        
        match counter_type {
            CounterType::InstructionsExecuted => counters.instructions_executed += 1,
            CounterType::CyclesElapsed => counters.cycles_elapsed += 1,
            CounterType::MemoryRead => counters.memory_reads += 1,
            CounterType::MemoryWrite => counters.memory_writes += 1,
            CounterType::CacheHit => counters.cache_hits += 1,
            CounterType::CacheMiss => counters.cache_misses += 1,
            CounterType::KernelLaunch => counters.kernel_launches += 1,
            CounterType::ContextSwitch => counters.context_switches += 1,
            CounterType::ComputationError => counters.computation_errors += 1,
            CounterType::MemoryError => counters.memory_errors += 1,
            CounterType::ThermalThrottling => counters.thermal_throttling_events += 1,
            CounterType::PowerThrottling => counters.power_throttling_events += 1,
        }
    }

    /// Check if alerts should be triggered
    pub fn check_alerts(&self) -> Vec<PerformanceAlert> {
        let metrics = self.current_metrics.read();
        let thresholds = &self.config.alert_thresholds;
        let mut alerts = Vec::new();
        
        if metrics.temperature > thresholds.max_temperature {
            alerts.push(PerformanceAlert {
                alert_type: AlertType::OverTemperature,
                severity: AlertSeverity::Critical,
                message: format!("Temperature {}°C exceeds threshold {}°C", 
                    metrics.temperature, thresholds.max_temperature),
                timestamp: Instant::now(),
            });
        }
        
        if metrics.power_usage > thresholds.max_power {
            alerts.push(PerformanceAlert {
                alert_type: AlertType::OverPower,
                severity: AlertSeverity::Warning,
                message: format!("Power usage {}W exceeds threshold {}W", 
                    metrics.power_usage, thresholds.max_power),
                timestamp: Instant::now(),
            });
        }
        
        if metrics.memory_utilization > thresholds.max_memory_utilization {
            alerts.push(PerformanceAlert {
                alert_type: AlertType::HighMemoryUsage,
                severity: AlertSeverity::Warning,
                message: format!("Memory utilization {:.1}% exceeds threshold {:.1}%", 
                    metrics.memory_utilization * 100.0, thresholds.max_memory_utilization * 100.0),
                timestamp: Instant::now(),
            });
        }
        
        if metrics.cache_hit_rate < thresholds.min_cache_hit_rate {
            alerts.push(PerformanceAlert {
                alert_type: AlertType::LowCacheHitRate,
                severity: AlertSeverity::Info,
                message: format!("Cache hit rate {:.1}% below threshold {:.1}%", 
                    metrics.cache_hit_rate * 100.0, thresholds.min_cache_hit_rate * 100.0),
                timestamp: Instant::now(),
            });
        }
        
        alerts
    }
}

/// Performance counter types
#[derive(Debug, Clone, Copy)]
pub enum CounterType {
    InstructionsExecuted,
    CyclesElapsed,
    MemoryRead,
    MemoryWrite,
    CacheHit,
    CacheMiss,
    KernelLaunch,
    ContextSwitch,
    ComputationError,
    MemoryError,
    ThermalThrottling,
    PowerThrottling,
}

/// Performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    pub average_compute_utilization: f64,
    pub max_compute_utilization: f64,
    pub average_memory_utilization: f64,
    pub max_memory_utilization: f64,
    pub average_temperature: f64,
    pub max_temperature: f64,
    pub average_power: f64,
    pub max_power: f64,
    pub thermal_throttling_events: u64,
    pub power_throttling_events: u64,
    pub uptime: Duration,
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: Instant,
}

/// Alert types
#[derive(Debug, Clone, PartialEq)]
pub enum AlertType {
    OverTemperature,
    OverPower,
    HighMemoryUsage,
    LowCacheHitRate,
    ComputationError,
    MemoryError,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

// Implementation details for internal structures

impl Default for GpuMetrics {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            compute_utilization: 0.0,
            memory_utilization: 0.0,
            memory_bandwidth_usage: 0.0,
            temperature: 35.0, // Idle temperature
            power_usage: 50.0,  // Idle power
            fan_speed: 800.0,   // Idle fan speed
            clock_speed: 1200.0, // Base clock
            memory_clock_speed: 1000.0, // Base memory clock
            voltage: 0.85,      // Idle voltage
            operations_per_second: 0.0,
            memory_transfers_per_second: 0.0,
            cache_hit_rate: 0.85, // Good default hit rate
            active_tasks: 0,
            queued_tasks: 0,
            is_monitoring_active: false,
            max_memory_bytes: 8 * 1024 * 1024 * 1024,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            sampling_interval: Duration::from_millis(100),
            history_size: 1000,
            enable_thermal_monitoring: true,
            enable_power_monitoring: true,
            enable_detailed_counters: true,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_temperature: 85.0,  // °C
            max_power: 300.0,       // Watts
            max_memory_utilization: 0.9, // 90%
            min_cache_hit_rate: 0.7, // 70%
        }
    }
}

impl PerformanceCounters {
    fn new() -> Self {
        Self {
            instructions_executed: 0,
            cycles_elapsed: 0,
            warps_executed: 0,
            threads_executed: 0,
            memory_reads: 0,
            memory_writes: 0,
            cache_hits: 0,
            cache_misses: 0,
            memory_bank_conflicts: 0,
            kernel_launches: 0,
            context_switches: 0,
            interrupts: 0,
            dma_transfers: 0,
            computation_errors: 0,
            memory_errors: 0,
            thermal_throttling_events: 0,
            power_throttling_events: 0,
        }
    }
}

impl ThermalMonitor {
    fn new() -> Self {
        let mut thermal_zones = HashMap::new();
        thermal_zones.insert(ThermalZone::Core, 35.0);
        thermal_zones.insert(ThermalZone::Memory, 30.0);
        thermal_zones.insert(ThermalZone::PowerDelivery, 40.0);
        thermal_zones.insert(ThermalZone::VRM, 45.0);
        thermal_zones.insert(ThermalZone::Ambient, 25.0);

        Self {
            current_temperature: 35.0,
            temperature_history: VecDeque::with_capacity(100),
            thermal_zones,
            cooling_strategy: CoolingStrategy::Adaptive,
            throttling_active: false,
        }
    }

    fn update_temperatures(&mut self) {
        // Simulate temperature readings from different zones
        for (zone, temp) in &mut self.thermal_zones {
            let noise = (rand::random::<f64>() - 0.5) * 2.0; // ±1°C noise
            *temp = (*temp + noise).max(20.0).min(100.0);
            
            let sample = ThermalSample {
                timestamp: Instant::now(),
                temperature: *temp,
                zone: zone.clone(),
            };
            
            self.temperature_history.push_back(sample);
        }

        // Update current temperature (use core temperature)
        if let Some(&core_temp) = self.thermal_zones.get(&ThermalZone::Core) {
            self.current_temperature = core_temp;
        }

        // Trim history
        while self.temperature_history.len() > 100 {
            self.temperature_history.pop_front();
        }
    }

    fn apply_thermal_management(&mut self) {
        let critical_temp = 85.0;
        
        if self.current_temperature > critical_temp && !self.throttling_active {
            self.throttling_active = true;
            // In practice, would reduce clock speeds or take other cooling actions
        } else if self.current_temperature < critical_temp - 5.0 && self.throttling_active {
            self.throttling_active = false;
            // Restore normal operation
        }
    }
}

impl PowerMonitor {
    fn new() -> Self {
        let mut power_rails = HashMap::new();
        power_rails.insert(PowerRail::Core, 30.0);      // 30W core
        power_rails.insert(PowerRail::Memory, 15.0);    // 15W memory
        power_rails.insert(PowerRail::IO, 3.0);         // 3W I/O
        power_rails.insert(PowerRail::PLL, 1.0);        // 1W PLL
        power_rails.insert(PowerRail::Auxiliary, 1.0);  // 1W auxiliary

        Self {
            current_power: 50.0,
            power_history: VecDeque::with_capacity(1000),
            power_rails,
            power_budget: 350.0, // 350W total budget
            power_efficiency: 0.0, // Operations per watt
        }
    }

    fn update_power_measurements(&mut self) {
        // Update power measurements for each rail
        let mut total_power = 0.0;
        
        for (rail, power) in &mut self.power_rails {
            let utilization = rand::random::<f64>(); // Simulate varying utilization
            let base_power = match rail {
                PowerRail::Core => 30.0 + utilization * 200.0,      // 30-230W
                PowerRail::Memory => 15.0 + utilization * 50.0,     // 15-65W  
                PowerRail::IO => 3.0 + utilization * 7.0,           // 3-10W
                PowerRail::PLL => 1.0 + utilization * 2.0,          // 1-3W
                PowerRail::Auxiliary => 1.0 + utilization * 4.0,    // 1-5W
            };
            
            *power = base_power;
            total_power += base_power;
            
            let sample = PowerSample {
                timestamp: Instant::now(),
                power: base_power,
                voltage: 1.0, // Simplified
                current: base_power, // Simplified (assuming 1V)
                rail: rail.clone(),
            };
            
            self.power_history.push_back(sample);
        }

        self.current_power = total_power;

        // Trim history
        while self.power_history.len() > 1000 {
            self.power_history.pop_front();
        }
    }

    fn optimize_power_efficiency(&mut self) {
        // Simple power efficiency calculation
        let operations = 1000000.0; // Simulate 1M operations/sec
        self.power_efficiency = operations / self.current_power;

        // In practice, would implement dynamic voltage/frequency scaling
        // or other power optimization techniques
    }
}

impl Default for PerformanceStatistics {
    fn default() -> Self {
        Self {
            average_compute_utilization: 0.0,
            max_compute_utilization: 0.0,
            average_memory_utilization: 0.0,
            max_memory_utilization: 0.0,
            average_temperature: 0.0,
            max_temperature: 0.0,
            average_power: 0.0,
            max_power: 0.0,
            thermal_throttling_events: 0,
            power_throttling_events: 0,
            uptime: Duration::from_secs(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new().unwrap();
        
        let metrics = monitor.get_current_metrics().unwrap();
        assert_eq!(metrics.compute_utilization, 0.0);
        assert!(!metrics.is_monitoring_active);
    }

    #[tokio::test]
    async fn test_start_stop_monitoring() {
        let monitor = PerformanceMonitor::new().unwrap();
        
        monitor.start_monitoring().await.unwrap();
        assert!(*monitor.is_monitoring.read());
        
        // Allow some time for metrics collection
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        let metrics = monitor.get_current_metrics().unwrap();
        assert!(metrics.is_monitoring_active);
        
        monitor.stop_monitoring().await.unwrap();
        assert!(!*monitor.is_monitoring.read());
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let monitor = PerformanceMonitor::new().unwrap();
        monitor.start_monitoring().await.unwrap();
        
        // Wait for some metrics to be collected
        tokio::time::sleep(Duration::from_millis(300)).await;
        
        let history = monitor.get_metrics_history(Duration::from_secs(1));
        assert!(!history.is_empty());
        
        monitor.stop_monitoring().await.unwrap();
    }

    #[test]
    fn test_performance_counters() {
        let monitor = PerformanceMonitor::new().unwrap();
        
        monitor.increment_counter(CounterType::InstructionsExecuted);
        monitor.increment_counter(CounterType::CacheHit);
        monitor.increment_counter(CounterType::KernelLaunch);
        
        let counters = monitor.counters.read();
        assert_eq!(counters.instructions_executed, 1);
        assert_eq!(counters.cache_hits, 1);
        assert_eq!(counters.kernel_launches, 1);
    }

    #[test]
    fn test_alert_checking() {
        let mut config = MonitoringConfig::default();
        config.alert_thresholds.max_temperature = 50.0; // Low threshold for testing
        
        let monitor = PerformanceMonitor::new().unwrap();
        
        // Simulate high temperature
        {
            let mut metrics = monitor.current_metrics.write();
            metrics.temperature = 60.0; // Above threshold
        }
        
        let alerts = monitor.check_alerts();
        assert!(!alerts.is_empty());
        
        let temp_alert = alerts.iter().find(|a| a.alert_type == AlertType::OverTemperature);
        assert!(temp_alert.is_some());
        assert_eq!(temp_alert.unwrap().severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_performance_statistics() {
        let monitor = PerformanceMonitor::new().unwrap();
        
        // Add some mock history
        {
            let mut history = monitor.metrics_history.write();
            for i in 0..10 {
                let mut metrics = GpuMetrics::default();
                metrics.compute_utilization = (i as f64) / 10.0;
                metrics.temperature = 40.0 + (i as f64);
                history.push_back(metrics);
            }
        }
        
        let stats = monitor.get_performance_stats();
        assert!(stats.average_compute_utilization > 0.0);
        assert!(stats.max_compute_utilization > 0.0);
        assert!(stats.average_temperature >= 40.0);
    }

    #[test]
    fn test_thermal_monitor() {
        let mut thermal_monitor = ThermalMonitor::new();
        
        assert_eq!(thermal_monitor.current_temperature, 35.0);
        assert!(!thermal_monitor.throttling_active);
        assert_eq!(thermal_monitor.thermal_zones.len(), 5);
    }

    #[test]
    fn test_power_monitor() {
        let power_monitor = PowerMonitor::new();
        
        assert_eq!(power_monitor.current_power, 50.0);
        assert_eq!(power_monitor.power_budget, 350.0);
        assert_eq!(power_monitor.power_rails.len(), 5);
    }

    #[test]
    fn test_metric_simulation_functions() {
        let utilization = PerformanceMonitor::simulate_workload_utilization();
        assert!(utilization >= 0.0 && utilization <= 1.0);
        
        let memory_util = PerformanceMonitor::simulate_memory_utilization(0.5);
        assert!(memory_util >= 0.0 && memory_util <= 1.0);
        
        let temperature = PerformanceMonitor::simulate_temperature(0.8);
        assert!(temperature > 35.0); // Should be above idle
        
        let power = PerformanceMonitor::simulate_power_usage(0.5, 60.0);
        assert!(power > 50.0); // Should be above idle
        
        let fan_speed = PerformanceMonitor::calculate_fan_speed(70.0);
        assert!(fan_speed > 800.0); // Should be above minimum
    }
}
//! Security and Isolation Manager for vGPU
//!
//! Provides security controls, resource isolation, and access management
//! for the virtual GPU environment with sandboxed execution contexts.

use crate::{Result, VGpuError, SecurityLevel, GpuTask};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use parking_lot::RwLock;
use std::time::{Duration, Instant};

/// Security and isolation management system
pub struct SecurityManager {
    // Security configuration
    isolation_level: SecurityLevel,
    
    // Execution contexts
    contexts: Arc<RwLock<HashMap<u64, ExecutionContext>>>,
    
    // Access control
    access_control: Arc<RwLock<AccessControlMatrix>>,
    
    // Resource limits
    resource_limits: Arc<RwLock<HashMap<String, ResourceLimits>>>,
    
    // Security policies
    policies: Arc<RwLock<SecurityPolicies>>,
    
    // Audit logging
    audit_log: Arc<RwLock<Vec<AuditEvent>>>,
    
    // Threat detection
    threat_detector: Arc<RwLock<ThreatDetector>>,
}

/// Sandboxed execution context
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub context_id: u64,
    pub user_id: String,
    pub application_id: String,
    pub isolation_level: SecurityLevel,
    pub allowed_resources: HashSet<ResourceType>,
    pub resource_quotas: ResourceQuotas,
    pub permissions: Permissions,
    pub created_at: Instant,
    pub last_activity: Instant,
    pub is_active: bool,
}

/// Resource types that can be controlled
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    Memory,
    ComputeUnits,
    NetworkAccess,
    FileSystem,
    SystemCalls,
    InterProcessComm,
    TimerAccess,
    DeviceAccess,
}

/// Resource quotas for execution contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuotas {
    pub max_memory_bytes: u64,
    pub max_compute_units: usize,
    pub max_execution_time: Duration,
    pub max_api_calls_per_second: u32,
    pub max_file_operations: u32,
    pub max_network_connections: u32,
}

/// Permission levels for different operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permissions {
    pub can_allocate_memory: bool,
    pub can_launch_kernels: bool,
    pub can_access_files: bool,
    pub can_create_contexts: bool,
    pub can_modify_resources: bool,
    pub can_view_other_contexts: bool,
    pub sandbox_escape_protection: bool,
}

/// Resource limits per user/application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub user_id: String,
    pub max_concurrent_contexts: usize,
    pub max_total_memory: u64,
    pub max_total_compute_units: usize,
    pub rate_limits: RateLimits,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    pub api_calls_per_minute: u32,
    pub memory_allocations_per_minute: u32,
    pub kernel_launches_per_minute: u32,
}

/// Access control matrix
#[derive(Debug)]
pub struct AccessControlMatrix {
    // Subject -> Resource -> Permissions
    permissions: HashMap<String, HashMap<ResourceType, HashSet<Operation>>>,
    
    // Role-based access control
    roles: HashMap<String, Role>,
    user_roles: HashMap<String, HashSet<String>>,
}

/// Security operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operation {
    Read,
    Write,
    Execute,
    Create,
    Delete,
    Modify,
    Admin,
}

/// Security roles
#[derive(Debug, Clone)]
pub struct Role {
    pub name: String,
    pub permissions: HashMap<ResourceType, HashSet<Operation>>,
    pub inherits_from: Vec<String>,
}

/// Security policies
#[derive(Debug, Clone)]
pub struct SecurityPolicies {
    pub enforce_isolation: bool,
    pub require_authentication: bool,
    pub audit_all_operations: bool,
    pub prevent_sandbox_escape: bool,
    pub enable_threat_detection: bool,
    pub max_context_lifetime: Duration,
    pub automatic_cleanup: bool,
}

/// Audit event for security logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: u64,
    pub timestamp: Instant,
    pub event_type: AuditEventType,
    pub user_id: String,
    pub context_id: u64,
    pub resource_type: ResourceType,
    pub operation: Operation,
    pub success: bool,
    pub details: String,
    pub risk_level: RiskLevel,
}

/// Types of auditable events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    ContextCreation,
    ContextDestruction,
    ResourceAllocation,
    ResourceDeallocation,
    PermissionCheck,
    PolicyViolation,
    ThreatDetected,
    SecurityBreach,
    RateLimitExceeded,
    UnauthorizedAccess,
}

/// Risk levels for security events
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Threat detection system
#[derive(Debug)]
pub struct ThreatDetector {
    // Behavioral analysis
    behavior_patterns: HashMap<String, BehaviorPattern>,
    
    // Anomaly detection
    baseline_metrics: BaselineMetrics,
    
    // Known threat signatures
    threat_signatures: Vec<ThreatSignature>,
    
    // Active monitoring
    monitoring_rules: Vec<MonitoringRule>,
}

/// User behavior pattern analysis
#[derive(Debug, Clone)]
pub struct BehaviorPattern {
    pub user_id: String,
    pub typical_memory_usage: f64,
    pub typical_compute_usage: f64,
    pub typical_api_call_rate: f64,
    pub typical_session_duration: Duration,
    pub anomaly_threshold: f64,
}

/// Baseline metrics for anomaly detection
#[derive(Debug, Clone)]
pub struct BaselineMetrics {
    pub average_memory_allocation_size: usize,
    pub average_kernel_execution_time: Duration,
    pub average_api_call_frequency: f64,
    pub normal_access_patterns: Vec<AccessPattern>,
}

#[derive(Debug, Clone)]
pub struct AccessPattern {
    pub resource_type: ResourceType,
    pub access_frequency: f64,
    pub time_of_day_distribution: Vec<f64>,
}

/// Threat signature for pattern matching
#[derive(Debug, Clone)]
pub struct ThreatSignature {
    pub signature_id: String,
    pub threat_type: ThreatType,
    pub pattern: String,
    pub severity: RiskLevel,
}

/// Types of security threats
#[derive(Debug, Clone, PartialEq)]
pub enum ThreatType {
    BufferOverflow,
    PrivilegeEscalation,
    ResourceExhaustion,
    SandboxEscape,
    DataExfiltration,
    CodeInjection,
    TimingAttack,
    SideChannelAttack,
}

/// Monitoring rules for threat detection
#[derive(Debug, Clone)]
pub struct MonitoringRule {
    pub rule_id: String,
    pub condition: MonitoringCondition,
    pub action: SecurityAction,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub enum MonitoringCondition {
    MemoryUsageThreshold { threshold: f64 },
    ApiCallRateThreshold { threshold: f64 },
    SuspiciousPattern { pattern: String },
    ResourceAccessAnomaly,
    UnusualBehavior { deviation: f64 },
}

#[derive(Debug, Clone)]
pub enum SecurityAction {
    LogEvent,
    ThrottleUser,
    SuspendContext,
    BlockUser,
    AlertAdministrator,
    TerminateSession,
}

impl SecurityManager {
    /// Create a new security manager
    pub fn new() -> Result<Self> {
        Ok(Self {
            isolation_level: SecurityLevel::Moderate,
            contexts: Arc::new(RwLock::new(HashMap::new())),
            access_control: Arc::new(RwLock::new(AccessControlMatrix::new())),
            resource_limits: Arc::new(RwLock::new(HashMap::new())),
            policies: Arc::new(RwLock::new(SecurityPolicies::default())),
            audit_log: Arc::new(RwLock::new(Vec::new())),
            threat_detector: Arc::new(RwLock::new(ThreatDetector::new())),
        })
    }

    /// Set isolation level
    pub fn set_isolation_level(&self, level: SecurityLevel) -> Result<()> {
        self.isolation_level = level;
        
        // Update existing contexts
        let mut contexts = self.contexts.write();
        for context in contexts.values_mut() {
            if level > context.isolation_level {
                context.isolation_level = level;
                // Apply stricter isolation
                self.apply_isolation_level(context, level)?;
            }
        }
        
        Ok(())
    }

    /// Create a new execution context
    pub fn create_context(&self, user_id: &str, application_id: &str) -> Result<u64> {
        // Check resource limits
        self.check_user_limits(user_id)?;
        
        // Create context
        let context_id = self.generate_context_id();
        let context = ExecutionContext {
            context_id,
            user_id: user_id.to_string(),
            application_id: application_id.to_string(),
            isolation_level: self.isolation_level,
            allowed_resources: self.determine_allowed_resources(user_id)?,
            resource_quotas: self.get_resource_quotas(user_id)?,
            permissions: self.get_user_permissions(user_id)?,
            created_at: Instant::now(),
            last_activity: Instant::now(),
            is_active: true,
        };

        // Apply isolation
        self.apply_isolation_level(&context, self.isolation_level)?;
        
        // Store context
        self.contexts.write().insert(context_id, context);
        
        // Audit event
        self.log_audit_event(AuditEventType::ContextCreation, user_id, context_id, 
            ResourceType::SystemCalls, Operation::Create, true, 
            "Context created successfully", RiskLevel::Low);
        
        Ok(context_id)
    }

    /// Destroy an execution context
    pub fn destroy_context(&self, context_id: u64) -> Result<()> {
        let context = self.contexts.write().remove(&context_id)
            .ok_or_else(|| VGpuError::security_violation("Context not found"))?;
        
        // Clean up resources associated with this context
        self.cleanup_context_resources(&context)?;
        
        // Audit event
        self.log_audit_event(AuditEventType::ContextDestruction, &context.user_id, context_id,
            ResourceType::SystemCalls, Operation::Delete, true,
            "Context destroyed successfully", RiskLevel::Low);
        
        Ok(())
    }

    /// Validate a task against security policies
    pub fn validate_task(&self, task: &GpuTask) -> Result<()> {
        // Check if task metadata contains security-relevant information
        let user_id = &task.metadata.user_id;
        
        // Check user permissions
        self.check_permissions(user_id, ResourceType::ComputeUnits, Operation::Execute)?;
        
        // Check resource quotas
        self.check_resource_quotas(user_id, task)?;
        
        // Threat detection
        self.detect_threats(task)?;
        
        Ok(())
    }

    /// Check if user has permission for operation
    pub fn check_permissions(&self, user_id: &str, resource: ResourceType, operation: Operation) -> Result<()> {
        let access_control = self.access_control.read();
        
        if let Some(user_perms) = access_control.permissions.get(user_id) {
            if let Some(resource_ops) = user_perms.get(&resource) {
                if resource_ops.contains(&operation) {
                    return Ok(());
                }
            }
        }
        
        // Check role-based permissions
        if let Some(user_roles) = access_control.user_roles.get(user_id) {
            for role_name in user_roles {
                if let Some(role) = access_control.roles.get(role_name) {
                    if let Some(role_ops) = role.permissions.get(&resource) {
                        if role_ops.contains(&operation) {
                            return Ok(());
                        }
                    }
                }
            }
        }
        
        // Log unauthorized access attempt
        self.log_audit_event(AuditEventType::UnauthorizedAccess, user_id, 0,
            resource, operation, false, 
            "Permission denied", RiskLevel::Medium);
        
        Err(VGpuError::security_violation("Permission denied"))
    }

    /// Apply isolation level to context
    fn apply_isolation_level(&self, context: &ExecutionContext, level: SecurityLevel) -> Result<()> {
        match level {
            SecurityLevel::Minimal => {
                // Basic resource limits only
                self.apply_basic_resource_limits(context)?;
            },
            SecurityLevel::Moderate => {
                // Process isolation and resource limits
                self.apply_process_isolation(context)?;
                self.apply_resource_limits(context)?;
            },
            SecurityLevel::High => {
                // Full sandboxing with restricted syscalls
                self.apply_full_sandboxing(context)?;
                self.apply_syscall_filtering(context)?;
            },
        }
        Ok(())
    }

    /// Apply basic resource limits
    fn apply_basic_resource_limits(&self, _context: &ExecutionContext) -> Result<()> {
        // Implement basic memory and compute limits
        // This is a simplified implementation
        Ok(())
    }

    /// Apply process isolation
    fn apply_process_isolation(&self, _context: &ExecutionContext) -> Result<()> {
        // Implement process-level isolation
        // In a real implementation, this would use OS features like namespaces, cgroups
        Ok(())
    }

    /// Apply resource limits
    fn apply_resource_limits(&self, _context: &ExecutionContext) -> Result<()> {
        // Implement comprehensive resource limiting
        Ok(())
    }

    /// Apply full sandboxing
    fn apply_full_sandboxing(&self, _context: &ExecutionContext) -> Result<()> {
        // Implement full sandbox with restricted capabilities
        // This would use features like seccomp, capabilities, chroot
        Ok(())
    }

    /// Apply syscall filtering
    fn apply_syscall_filtering(&self, _context: &ExecutionContext) -> Result<()> {
        // Implement syscall filtering using seccomp or similar
        Ok(())
    }

    /// Check user resource limits
    fn check_user_limits(&self, user_id: &str) -> Result<()> {
        let resource_limits = self.resource_limits.read();
        
        if let Some(limits) = resource_limits.get(user_id) {
            let contexts = self.contexts.read();
            let user_contexts: Vec<_> = contexts.values()
                .filter(|c| c.user_id == user_id && c.is_active)
                .collect();
            
            if user_contexts.len() >= limits.max_concurrent_contexts {
                return Err(VGpuError::security_violation("Too many concurrent contexts"));
            }
        }
        
        Ok(())
    }

    /// Determine allowed resources for user
    fn determine_allowed_resources(&self, user_id: &str) -> Result<HashSet<ResourceType>> {
        let access_control = self.access_control.read();
        let mut allowed_resources = HashSet::new();
        
        // Get user permissions
        if let Some(user_perms) = access_control.permissions.get(user_id) {
            for resource in user_perms.keys() {
                allowed_resources.insert(resource.clone());
            }
        }
        
        // Add role-based permissions
        if let Some(user_roles) = access_control.user_roles.get(user_id) {
            for role_name in user_roles {
                if let Some(role) = access_control.roles.get(role_name) {
                    for resource in role.permissions.keys() {
                        allowed_resources.insert(resource.clone());
                    }
                }
            }
        }
        
        Ok(allowed_resources)
    }

    /// Get resource quotas for user
    fn get_resource_quotas(&self, _user_id: &str) -> Result<ResourceQuotas> {
        // Return default quotas for now
        Ok(ResourceQuotas::default())
    }

    /// Get user permissions
    fn get_user_permissions(&self, _user_id: &str) -> Result<Permissions> {
        // Return default permissions for now
        Ok(Permissions::default())
    }

    /// Check resource quotas for task
    fn check_resource_quotas(&self, user_id: &str, task: &GpuTask) -> Result<()> {
        let contexts = self.contexts.read();
        
        // Find user's contexts
        for context in contexts.values() {
            if context.user_id == user_id {
                let memory_required = task.resource_requirements.memory_bytes as u64;
                if memory_required > context.resource_quotas.max_memory_bytes {
                    return Err(VGpuError::security_violation("Memory quota exceeded"));
                }
                
                let compute_required = task.resource_requirements.compute_units;
                if compute_required > context.resource_quotas.max_compute_units {
                    return Err(VGpuError::security_violation("Compute quota exceeded"));
                }
            }
        }
        
        Ok(())
    }

    /// Detect security threats in task
    fn detect_threats(&self, task: &GpuTask) -> Result<()> {
        let threat_detector = self.threat_detector.read();
        
        // Check for suspicious patterns
        for signature in &threat_detector.threat_signatures {
            if self.matches_threat_signature(task, signature) {
                self.log_audit_event(AuditEventType::ThreatDetected, &task.metadata.user_id, task.task_id,
                    ResourceType::ComputeUnits, Operation::Execute, false,
                    &format!("Threat signature matched: {}", signature.signature_id),
                    signature.severity.clone());
                
                if signature.severity == RiskLevel::Critical {
                    return Err(VGpuError::security_violation("Critical threat detected"));
                }
            }
        }
        
        // Behavioral analysis
        if let Some(pattern) = threat_detector.behavior_patterns.get(&task.metadata.user_id) {
            if self.is_anomalous_behavior(task, pattern) {
                self.log_audit_event(AuditEventType::ThreatDetected, &task.metadata.user_id, task.task_id,
                    ResourceType::ComputeUnits, Operation::Execute, false,
                    "Anomalous behavior detected", RiskLevel::Medium);
            }
        }
        
        Ok(())
    }

    /// Check if task matches threat signature
    fn matches_threat_signature(&self, task: &GpuTask, signature: &ThreatSignature) -> bool {
        match signature.threat_type {
            ThreatType::BufferOverflow => {
                // Check for excessive memory allocation
                task.resource_requirements.memory_bytes > 1024 * 1024 * 1024 // 1GB
            },
            ThreatType::ResourceExhaustion => {
                // Check for resource exhaustion patterns
                task.resource_requirements.compute_units > 100
            },
            _ => false, // Simplified - real implementation would be more sophisticated
        }
    }

    /// Check for anomalous user behavior
    fn is_anomalous_behavior(&self, task: &GpuTask, pattern: &BehaviorPattern) -> bool {
        let memory_deviation = (task.resource_requirements.memory_bytes as f64 - pattern.typical_memory_usage).abs() / pattern.typical_memory_usage;
        let compute_deviation = (task.resource_requirements.compute_units as f64 - pattern.typical_compute_usage).abs() / pattern.typical_compute_usage;
        
        memory_deviation > pattern.anomaly_threshold || compute_deviation > pattern.anomaly_threshold
    }

    /// Clean up resources for destroyed context
    fn cleanup_context_resources(&self, _context: &ExecutionContext) -> Result<()> {
        // Clean up any resources allocated to this context
        // This is a simplified implementation
        Ok(())
    }

    /// Log security audit event
    fn log_audit_event(&self, event_type: AuditEventType, user_id: &str, context_id: u64, 
                      resource_type: ResourceType, operation: Operation, success: bool,
                      details: &str, risk_level: RiskLevel) {
        
        let event = AuditEvent {
            event_id: self.generate_event_id(),
            timestamp: Instant::now(),
            event_type,
            user_id: user_id.to_string(),
            context_id,
            resource_type,
            operation,
            success,
            details: details.to_string(),
            risk_level,
        };
        
        self.audit_log.write().push(event);
    }

    /// Generate unique context ID
    fn generate_context_id(&self) -> u64 {
        use rand::Rng;
        rand::thread_rng().gen()
    }

    /// Generate unique event ID
    fn generate_event_id(&self) -> u64 {
        use rand::Rng;
        rand::thread_rng().gen()
    }

    /// Get security audit log
    pub fn get_audit_log(&self, filter: Option<AuditFilter>) -> Vec<AuditEvent> {
        let audit_log = self.audit_log.read();
        
        match filter {
            Some(filter) => audit_log.iter().filter(|event| filter.matches(event)).cloned().collect(),
            None => audit_log.clone(),
        }
    }

    /// Get active contexts
    pub fn get_active_contexts(&self) -> Vec<ExecutionContext> {
        self.contexts.read().values()
            .filter(|c| c.is_active)
            .cloned()
            .collect()
    }
}

/// Audit log filter
#[derive(Debug, Clone)]
pub struct AuditFilter {
    pub user_id: Option<String>,
    pub event_type: Option<AuditEventType>,
    pub risk_level: Option<RiskLevel>,
    pub time_range: Option<(Instant, Instant)>,
}

impl AuditFilter {
    pub fn matches(&self, event: &AuditEvent) -> bool {
        if let Some(ref user_id) = self.user_id {
            if event.user_id != *user_id {
                return false;
            }
        }
        
        if let Some(ref event_type) = self.event_type {
            if std::mem::discriminant(&event.event_type) != std::mem::discriminant(event_type) {
                return false;
            }
        }
        
        if let Some(ref risk_level) = self.risk_level {
            if event.risk_level != *risk_level {
                return false;
            }
        }
        
        if let Some((start, end)) = self.time_range {
            if event.timestamp < start || event.timestamp > end {
                return false;
            }
        }
        
        true
    }
}

// Default implementations

impl Default for ResourceQuotas {
    fn default() -> Self {
        Self {
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            max_compute_units: 16,
            max_execution_time: Duration::from_secs(3600), // 1 hour
            max_api_calls_per_second: 1000,
            max_file_operations: 10000,
            max_network_connections: 100,
        }
    }
}

impl Default for Permissions {
    fn default() -> Self {
        Self {
            can_allocate_memory: true,
            can_launch_kernels: true,
            can_access_files: false, // Disabled by default for security
            can_create_contexts: false, // Disabled by default
            can_modify_resources: false, // Disabled by default
            can_view_other_contexts: false, // Disabled by default
            sandbox_escape_protection: true,
        }
    }
}

impl Default for SecurityPolicies {
    fn default() -> Self {
        Self {
            enforce_isolation: true,
            require_authentication: true,
            audit_all_operations: true,
            prevent_sandbox_escape: true,
            enable_threat_detection: true,
            max_context_lifetime: Duration::from_hours(24),
            automatic_cleanup: true,
        }
    }
}

impl AccessControlMatrix {
    fn new() -> Self {
        Self {
            permissions: HashMap::new(),
            roles: HashMap::new(),
            user_roles: HashMap::new(),
        }
    }
}

impl ThreatDetector {
    fn new() -> Self {
        Self {
            behavior_patterns: HashMap::new(),
            baseline_metrics: BaselineMetrics::new(),
            threat_signatures: Self::create_default_signatures(),
            monitoring_rules: Vec::new(),
        }
    }

    fn create_default_signatures() -> Vec<ThreatSignature> {
        vec![
            ThreatSignature {
                signature_id: "BUFFER_OVERFLOW_1".to_string(),
                threat_type: ThreatType::BufferOverflow,
                pattern: "excessive_memory_allocation".to_string(),
                severity: RiskLevel::High,
            },
            ThreatSignature {
                signature_id: "RESOURCE_EXHAUSTION_1".to_string(),
                threat_type: ThreatType::ResourceExhaustion,
                pattern: "excessive_compute_units".to_string(),
                severity: RiskLevel::Critical,
            },
        ]
    }
}

impl BaselineMetrics {
    fn new() -> Self {
        Self {
            average_memory_allocation_size: 1024 * 1024, // 1MB
            average_kernel_execution_time: Duration::from_millis(100),
            average_api_call_frequency: 10.0, // 10 calls/second
            normal_access_patterns: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuTask};
    use crate::task_scheduler::{TaskMetadata, ResourceRequirements};

    #[test]
    fn test_security_manager_creation() {
        let security_manager = SecurityManager::new().unwrap();
        let contexts = security_manager.get_active_contexts();
        assert_eq!(contexts.len(), 0);
    }

    #[test]
    fn test_context_creation() {
        let security_manager = SecurityManager::new().unwrap();
        
        let context_id = security_manager.create_context("test_user", "test_app").unwrap();
        assert!(context_id > 0);
        
        let contexts = security_manager.get_active_contexts();
        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].user_id, "test_user");
        assert_eq!(contexts[0].application_id, "test_app");
    }

    #[test]
    fn test_context_destruction() {
        let security_manager = SecurityManager::new().unwrap();
        
        let context_id = security_manager.create_context("test_user", "test_app").unwrap();
        security_manager.destroy_context(context_id).unwrap();
        
        let contexts = security_manager.get_active_contexts();
        assert_eq!(contexts.len(), 0);
    }

    #[test]
    fn test_isolation_level_setting() {
        let security_manager = SecurityManager::new().unwrap();
        
        security_manager.set_isolation_level(SecurityLevel::High).unwrap();
        assert_eq!(security_manager.isolation_level, SecurityLevel::High);
    }

    #[test]
    fn test_task_validation() {
        let security_manager = SecurityManager::new().unwrap();
        
        let mut task = GpuTask::new("test_kernel", vec![1.0, 2.0, 3.0]);
        task.metadata = TaskMetadata {
            user_id: "test_user".to_string(),
            ..TaskMetadata::default()
        };
        
        // This should fail due to lack of permissions setup
        assert!(security_manager.validate_task(&task).is_err());
    }

    #[test]
    fn test_resource_quotas() {
        let quotas = ResourceQuotas::default();
        
        assert_eq!(quotas.max_memory_bytes, 1024 * 1024 * 1024);
        assert_eq!(quotas.max_compute_units, 16);
        assert_eq!(quotas.max_api_calls_per_second, 1000);
    }

    #[test]
    fn test_permissions() {
        let permissions = Permissions::default();
        
        assert!(permissions.can_allocate_memory);
        assert!(permissions.can_launch_kernels);
        assert!(!permissions.can_access_files);
        assert!(!permissions.can_create_contexts);
        assert!(permissions.sandbox_escape_protection);
    }

    #[test]
    fn test_audit_event_logging() {
        let security_manager = SecurityManager::new().unwrap();
        
        security_manager.log_audit_event(
            AuditEventType::ContextCreation,
            "test_user",
            123,
            ResourceType::Memory,
            Operation::Create,
            true,
            "Test event",
            RiskLevel::Low
        );
        
        let audit_log = security_manager.get_audit_log(None);
        assert_eq!(audit_log.len(), 1);
        assert_eq!(audit_log[0].user_id, "test_user");
        assert_eq!(audit_log[0].context_id, 123);
        assert!(audit_log[0].success);
    }

    #[test]
    fn test_audit_log_filtering() {
        let security_manager = SecurityManager::new().unwrap();
        
        // Log multiple events
        security_manager.log_audit_event(AuditEventType::ContextCreation, "user1", 1, 
            ResourceType::Memory, Operation::Create, true, "Event 1", RiskLevel::Low);
        security_manager.log_audit_event(AuditEventType::ThreatDetected, "user2", 2,
            ResourceType::ComputeUnits, Operation::Execute, false, "Event 2", RiskLevel::High);
        
        // Test filtering by user
        let filter = AuditFilter {
            user_id: Some("user1".to_string()),
            event_type: None,
            risk_level: None,
            time_range: None,
        };
        
        let filtered_log = security_manager.get_audit_log(Some(filter));
        assert_eq!(filtered_log.len(), 1);
        assert_eq!(filtered_log[0].user_id, "user1");
        
        // Test filtering by risk level
        let filter = AuditFilter {
            user_id: None,
            event_type: None,
            risk_level: Some(RiskLevel::High),
            time_range: None,
        };
        
        let filtered_log = security_manager.get_audit_log(Some(filter));
        assert_eq!(filtered_log.len(), 1);
        assert_eq!(filtered_log[0].risk_level, RiskLevel::High);
    }

    #[test]
    fn test_threat_detector() {
        let threat_detector = ThreatDetector::new();
        
        assert!(!threat_detector.threat_signatures.is_empty());
        assert!(threat_detector.behavior_patterns.is_empty());
        assert_eq!(threat_detector.baseline_metrics.average_memory_allocation_size, 1024 * 1024);
    }

    #[test]
    fn test_security_policies() {
        let policies = SecurityPolicies::default();
        
        assert!(policies.enforce_isolation);
        assert!(policies.require_authentication);
        assert!(policies.audit_all_operations);
        assert!(policies.prevent_sandbox_escape);
        assert!(policies.enable_threat_detection);
        assert!(policies.automatic_cleanup);
    }

    #[test]
    fn test_threat_signature_matching() {
        let security_manager = SecurityManager::new().unwrap();
        
        // Create a task that should trigger buffer overflow detection
        let mut task = GpuTask::new("test_kernel", vec![1.0; 1000]);
        task.resource_requirements = ResourceRequirements {
            memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB - should trigger threshold
            compute_units: 1,
            bandwidth_mbps: 1000,
            estimated_duration: Duration::from_secs(1),
            power_budget: 10.0,
        };
        
        let signature = ThreatSignature {
            signature_id: "TEST_OVERFLOW".to_string(),
            threat_type: ThreatType::BufferOverflow,
            pattern: "test_pattern".to_string(),
            severity: RiskLevel::High,
        };
        
        assert!(security_manager.matches_threat_signature(&task, &signature));
    }
}
//! # Hybrid Surrogate+Verifier Framework
//!
//! Advanced hybrid system combining machine learning surrogates with formal verification
//! for fast, reliable computation in infinite-solution-space problems. This framework
//! provides mathematically rigorous verification of heuristic solutions with intelligent
//! precomputation and caching strategies.
//!
//! ## Core Components
//!
//! - **Surrogate Models**: Neural networks, polynomial approximators, statistical models
//! - **Verification Engine**: Formal mathematical proof systems and bounds checking
//! - **Precompute Manager**: Intelligent caching, prediction, and optimization
//! - **Hybrid Orchestrator**: Seamless integration and decision routing
//!
//! This library enables rapid exploration of mathematical solution spaces while
//! maintaining mathematical rigor through formal verification backends.

use num_complex::Complex;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

pub mod surrogate;
pub mod verifier;
pub mod precompute;
pub mod orchestrator;
pub mod error;

pub use error::*;

/// Complex number type alias for consistency across the framework
pub type ComplexF64 = Complex<f64>;

/// Result type for hybrid verifier operations
pub type HVResult<T> = Result<T, HVError>;

/// Unique identifier for computational problems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProblemId(pub u64);

/// Unique identifier for verification certificates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CertificateId(pub u64);

/// Solution confidence levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// Heuristic solution, not verified
    Heuristic,
    /// Statistical bounds, high confidence
    Statistical,
    /// Mathematical bounds, verified
    Bounded,
    /// Formally proven, complete certainty
    Proven,
}

/// Computational complexity classes for problem classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexityClass {
    /// Polynomial time solvable
    P,
    /// Nondeterministic polynomial time
    NP,
    /// Polynomial space
    PSpace,
    /// Exponential time
    ExpTime,
    /// Undecidable problems
    Undecidable,
    /// Custom complexity class
    Custom(String),
}

/// Mathematical problem specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemSpec {
    /// Problem identifier
    pub id: ProblemId,
    
    /// Human-readable description
    pub description: String,
    
    /// Problem dimensions and parameters
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Expected complexity class
    pub complexity_class: ComplexityClass,
    
    /// Verification requirements
    pub verification_level: ConfidenceLevel,
    
    /// Timeout constraints
    pub timeout: Option<std::time::Duration>,
}

/// Solution to a mathematical problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    /// Problem this solves
    pub problem_id: ProblemId,
    
    /// Solution data
    pub data: serde_json::Value,
    
    /// Confidence level achieved
    pub confidence: ConfidenceLevel,
    
    /// Verification certificate (if applicable)
    pub certificate: Option<CertificateId>,
    
    /// Computational metrics
    pub metrics: ComputationMetrics,
    
    /// Error bounds (if applicable)
    pub error_bounds: Option<ErrorBounds>,
}

/// Computational performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationMetrics {
    /// Wall clock time taken
    pub wall_time: std::time::Duration,
    
    /// CPU time consumed
    pub cpu_time: std::time::Duration,
    
    /// Memory usage peak
    pub peak_memory: u64,
    
    /// Number of operations performed
    pub operation_count: u64,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Mathematical error bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBounds {
    /// Absolute error bound
    pub absolute: f64,
    
    /// Relative error bound
    pub relative: f64,
    
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    
    /// Statistical significance level
    pub significance: f64,
}

/// Main hybrid verifier system
#[derive(Debug)]
pub struct HybridVerifier {
    /// Surrogate model subsystem
    surrogate_engine: surrogate::SurrogateEngine,
    
    /// Verification subsystem
    verification_engine: verifier::VerificationEngine,
    
    /// Precomputation manager
    precompute_manager: precompute::PrecomputeManager,
    
    /// System orchestrator
    orchestrator: orchestrator::HybridOrchestrator,
    
    /// Problem registry
    problem_registry: HashMap<ProblemId, ProblemSpec>,
    
    /// Solution cache
    solution_cache: HashMap<ProblemId, Solution>,
}

impl HybridVerifier {
    /// Create a new hybrid verifier system
    pub fn new() -> Self {
        Self {
            surrogate_engine: surrogate::SurrogateEngine::new(),
            verification_engine: verifier::VerificationEngine::new(),
            precompute_manager: precompute::PrecomputeManager::new(),
            orchestrator: orchestrator::HybridOrchestrator::new(),
            problem_registry: HashMap::new(),
            solution_cache: HashMap::new(),
        }
    }
    
    /// Register a new mathematical problem
    pub fn register_problem(&mut self, spec: ProblemSpec) -> HVResult<()> {
        let problem_id = spec.id;
        self.problem_registry.insert(problem_id, spec);
        self.precompute_manager.register_problem(problem_id)?;
        Ok(())
    }
    
    /// Solve a problem using the hybrid approach
    pub fn solve(&mut self, problem_id: ProblemId) -> HVResult<Solution> {
        // Check if solution already exists in cache
        if let Some(cached_solution) = self.solution_cache.get(&problem_id) {
            return Ok(cached_solution.clone());
        }
        
        let spec = self.problem_registry.get(&problem_id)
            .ok_or(HVError::ProblemNotFound(problem_id))?;
        
        // Use orchestrator to determine solution strategy
        let solution = self.orchestrator.solve(
            spec,
            &mut self.surrogate_engine,
            &mut self.verification_engine,
            &mut self.precompute_manager,
        )?;
        
        // Cache the solution
        self.solution_cache.insert(problem_id, solution.clone());
        
        Ok(solution)
    }
    
    /// Verify an existing solution
    pub fn verify_solution(&mut self, problem_id: ProblemId, solution: &Solution) -> HVResult<ConfidenceLevel> {
        let spec = self.problem_registry.get(&problem_id)
            .ok_or(HVError::ProblemNotFound(problem_id))?;
        
        self.verification_engine.verify_solution(spec, solution)
    }
    
    /// Get performance statistics
    pub fn get_statistics(&self) -> HVResult<SystemStatistics> {
        Ok(SystemStatistics {
            problems_registered: self.problem_registry.len(),
            solutions_cached: self.solution_cache.len(),
            surrogate_stats: self.surrogate_engine.get_statistics()?,
            verification_stats: self.verification_engine.get_statistics()?,
            precompute_stats: self.precompute_manager.get_statistics()?,
        })
    }
}

impl Default for HybridVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// System-wide performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatistics {
    pub problems_registered: usize,
    pub solutions_cached: usize,
    pub surrogate_stats: surrogate::SurrogateStatistics,
    pub verification_stats: verifier::VerificationStatistics,
    pub precompute_stats: precompute::PrecomputeStatistics,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_problem_registration() {
        let mut hv = HybridVerifier::new();
        
        let spec = ProblemSpec {
            id: ProblemId(1),
            description: "Test optimization problem".to_string(),
            parameters: HashMap::new(),
            complexity_class: ComplexityClass::NP,
            verification_level: ConfidenceLevel::Statistical,
            timeout: Some(std::time::Duration::from_secs(60)),
        };
        
        assert!(hv.register_problem(spec).is_ok());
        assert_eq!(hv.problem_registry.len(), 1);
    }
    
    #[test]
    fn test_confidence_level_ordering() {
        assert!(ConfidenceLevel::Heuristic < ConfidenceLevel::Statistical);
        assert!(ConfidenceLevel::Statistical < ConfidenceLevel::Bounded);
        assert!(ConfidenceLevel::Bounded < ConfidenceLevel::Proven);
    }
    
    #[test]
    fn test_system_creation() {
        let hv = HybridVerifier::new();
        let stats = hv.get_statistics().unwrap();
        
        assert_eq!(stats.problems_registered, 0);
        assert_eq!(stats.solutions_cached, 0);
    }
}
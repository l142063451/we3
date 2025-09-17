//! Error types for the Hybrid Surrogate+Verifier system

use thiserror::Error;
use serde::{Serialize, Deserialize};
use crate::{ProblemId, CertificateId};

/// Result type for hybrid verifier operations
pub type HVResult<T> = Result<T, HVError>;

/// Comprehensive error types for hybrid verifier operations
#[derive(Debug, Clone, Error)]
pub enum HVError {
    #[error("Problem not found: {0:?}")]
    ProblemNotFound(ProblemId),
    
    #[error("Solution verification failed")]
    VerificationFailed,
    
    #[error("Surrogate model error: {0}")]
    SurrogateModelError(String),
    
    #[error("Verification timeout")]
    VerificationTimeout,
    
    #[error("Certificate not found: {0:?}")]
    CertificateNotFound(CertificateId),
    
    #[error("Invalid problem specification: {0}")]
    InvalidProblemSpec(String),
    
    #[error("Precomputation failed: {0}")]
    PrecomputationFailed(String),
    
    #[error("Mathematical computation error: {0}")]
    ComputationError(String),
    
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    
    #[error("Numerical instability detected: {0}")]
    NumericalInstability(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Neural network training failed: {0}")]
    TrainingFailed(String),
    
    #[error("Statistical significance too low: {got} < {required}")]
    InsufficientSignificance { got: f64, required: f64 },
    
    #[error("Error bounds exceeded: {actual} > {maximum}")]
    BoundsExceeded { actual: f64, maximum: f64 },
    
    #[error("Memory allocation failed: {0}")]
    MemoryError(String),
    
    #[error("Threading error: {0}")]
    ThreadingError(String),
}

impl From<serde_json::Error> for HVError {
    fn from(err: serde_json::Error) -> Self {
        HVError::SerializationError(err.to_string())
    }
}
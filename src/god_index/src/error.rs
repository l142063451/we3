//! Error types for the God-Index system

use thiserror::Error;

/// Result type for God-Index operations
pub type GIResult<T> = Result<T, GIError>;

/// Comprehensive error types for God-Index operations
#[derive(Debug, Clone, Error)]
pub enum GIError {
    #[error("Mathematical family not found")]
    FamilyNotFound,
    
    #[error("Index mapping not found: {0}")]
    MappingNotFound(String),
    
    #[error("Invalid family structure: {0}")]
    InvalidFamilyStructure(String),
    
    #[error("Mapping creation failed: {0}")]
    MappingCreationFailed(String),
    
    #[error("Navigation path not found: {0}")]
    NavigationPathNotFound(String),
    
    #[error("Query resolution failed: {0}")]
    QueryResolutionFailed(String),
    
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
    
    #[error("Index out of bounds: {index} >= {bound}")]
    IndexOutOfBounds { index: u64, bound: u64 },
    
    #[error("Incompatible family types: {0} and {1}")]
    IncompatibleFamilyTypes(String, String),
    
    #[error("Mathematical computation error: {0}")]
    ComputationError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Memory allocation failed: {0}")]
    MemoryError(String),
    
    #[error("Threading error: {0}")]
    ThreadingError(String),
    
    #[error("Invalid query pattern: {0}")]
    InvalidQueryPattern(String),
    
    #[error("Cross-family relation error: {0}")]
    CrossFamilyRelationError(String),
}
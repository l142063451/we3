//! Index mappings between mathematical families

use crate::*;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Types of mappings between mathematical families
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MappingType {
    /// One-to-one correspondence
    Bijective,
    /// Many-to-one mapping
    Surjective,
    /// One-to-many mapping
    Injective,
    /// General function mapping
    General,
    /// Isomorphism preserving structure
    Isomorphism,
    /// Homomorphism preserving operations
    Homomorphism,
    /// Custom mapping with specified properties
    Custom { properties: Vec<String> },
}

/// Index mapping between two mathematical families
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMapping {
    /// Source family identifier
    pub source_family: FamilyId,
    
    /// Target family identifier
    pub target_family: FamilyId,
    
    /// Type of mapping relationship
    pub mapping_type: MappingType,
    
    /// Forward mapping function
    pub forward_mapping: MappingFunction,
    
    /// Inverse mapping function (if exists)
    pub inverse_mapping: Option<MappingFunction>,
    
    /// Mapping properties and invariants
    pub properties: MappingProperties,
    
    /// Performance statistics
    pub performance_stats: MappingStats,
}

/// Mathematical function defining the mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MappingFunction {
    /// Explicit lookup table for finite mappings
    LookupTable(HashMap<u64, u64>),
    
    /// Polynomial transformation f(x) = sum(a_i * x^i)
    Polynomial { coefficients: Vec<f64> },
    
    /// Matrix transformation Ax for linear mappings
    Matrix { matrix: Vec<Vec<f64>> },
    
    /// Modular arithmetic transformation (ax + b) mod m
    ModularArithmetic { a: u64, b: u64, modulus: u64 },
    
    /// Combinatorial transformation based on generating functions
    GeneratingFunction { numerator: Vec<ComplexF64>, denominator: Vec<ComplexF64> },
    
    /// Bit manipulation transformation
    BitTransformation { operations: Vec<BitOperation> },
    
    /// Recursive function with base cases
    Recursive { base_cases: Vec<(u64, u64)>, recursion_rule: String },
    
    /// Custom function defined by algorithm
    Custom { algorithm_name: String, parameters: HashMap<String, serde_json::Value> },
}

/// Bit manipulation operations for transformations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BitOperation {
    /// Shift bits left by n positions
    LeftShift(u32),
    /// Shift bits right by n positions
    RightShift(u32),
    /// XOR with constant mask
    XorMask(u64),
    /// AND with constant mask
    AndMask(u64),
    /// OR with constant mask
    OrMask(u64),
    /// Bit reversal
    Reverse,
    /// Gray code transformation
    GrayCode,
    /// Custom bit operation
    Custom(String),
}

/// Properties and invariants of the mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingProperties {
    /// Whether mapping preserves order
    pub order_preserving: bool,
    
    /// Whether mapping preserves structure
    pub structure_preserving: bool,
    
    /// Domain size (None for infinite)
    pub domain_size: Option<u64>,
    
    /// Codomain size (None for infinite)
    pub codomain_size: Option<u64>,
    
    /// Computational complexity
    pub complexity: AlgorithmComplexity,
    
    /// Mathematical invariants preserved
    pub invariants: Vec<MathematicalInvariant>,
    
    /// Error bounds for approximate mappings
    pub error_bounds: Option<ErrorBounds>,
}

/// Mathematical invariant preserved by mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalInvariant {
    pub invariant_name: String,
    pub invariant_type: InvariantType,
    pub tolerance: Option<f64>,
}

/// Types of mathematical invariants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvariantType {
    /// Numerical invariant (sum, product, norm)
    Numerical,
    /// Structural invariant (connectivity, symmetry)
    Structural,
    /// Topological invariant (genus, Euler characteristic)
    Topological,
    /// Algebraic invariant (rank, determinant)
    Algebraic,
    /// Custom invariant
    Custom(String),
}

/// Error bounds for approximate mappings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBounds {
    pub absolute_error: f64,
    pub relative_error: f64,
    pub confidence_level: f64,
}

/// Performance statistics for mappings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingStats {
    pub total_evaluations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_evaluation_time: std::time::Duration,
    pub memory_usage: u64, // bytes
}

/// Cross-family mapping relating different mathematical families
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossFamilyMapping {
    /// Relationship type between families
    pub relationship_type: RelationshipType,
    
    /// Forward transformation
    pub forward_transform: IndexMapping,
    
    /// Backward transformation (if exists)
    pub backward_transform: Option<IndexMapping>,
    
    /// Compatibility conditions
    pub compatibility_conditions: Vec<CompatibilityCondition>,
}

/// Types of relationships between families
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Direct embedding of one family in another
    Embedding,
    /// Quotient relationship
    Quotient,
    /// Dual relationship
    Dual,
    /// Tensor product relationship
    TensorProduct,
    /// Restriction to subfamily
    Restriction,
    /// Extension from subfamily
    Extension,
    /// Custom relationship
    Custom(String),
}

/// Compatibility condition for cross-family mappings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityCondition {
    pub condition_name: String,
    pub condition_type: ConditionType,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Types of compatibility conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    /// Structural compatibility
    Structural,
    /// Numerical compatibility
    Numerical,
    /// Algebraic compatibility
    Algebraic,
    /// Custom condition
    Custom(String),
}

impl IndexMapping {
    /// Create a new index mapping between families
    pub fn new(
        source: FamilyId,
        target: FamilyId,
        mapping_type: MappingType,
        source_family: &StructuredFamily,
        target_family: &StructuredFamily,
    ) -> GIResult<Self> {
        // Verify compatibility between family types
        let compatible = Self::verify_compatibility(&source_family.family_type, &target_family.family_type)?;
        if !compatible {
            return Err(GIError::IncompatibleFamilyTypes(
                format!("{:?}", source_family.family_type),
                format!("{:?}", target_family.family_type),
            ));
        }

        // Create appropriate mapping function based on family types and mapping type
        let forward_mapping = Self::create_mapping_function(
            &source_family.family_type,
            &target_family.family_type,
            &mapping_type,
        )?;

        // Create inverse mapping if bijective
        let inverse_mapping = match mapping_type {
            MappingType::Bijective | MappingType::Isomorphism => {
                Some(Self::create_inverse_mapping(&forward_mapping)?)
            },
            _ => None,
        };

        // Determine mapping properties
        let properties = Self::analyze_mapping_properties(
            &forward_mapping,
            &mapping_type,
            source_family,
            target_family,
        );

        Ok(Self {
            source_family: source,
            target_family: target,
            mapping_type,
            forward_mapping,
            inverse_mapping,
            properties,
            performance_stats: MappingStats::default(),
        })
    }

    /// Verify compatibility between two family types
    fn verify_compatibility(source_type: &FamilyType, target_type: &FamilyType) -> GIResult<bool> {
        match (source_type, target_type) {
            // Compatible within same type
            (FamilyType::Combinatorial, FamilyType::Combinatorial) => Ok(true),
            (FamilyType::Algebraic, FamilyType::Algebraic) => Ok(true),
            (FamilyType::Geometric, FamilyType::Geometric) => Ok(true),
            
            // Cross-type compatibility
            (FamilyType::Combinatorial, FamilyType::Algebraic) => Ok(true), // Graph -> Group
            (FamilyType::Algebraic, FamilyType::Geometric) => Ok(true), // Ring -> Variety
            (FamilyType::Geometric, FamilyType::Analytic) => Ok(true), // Manifold -> Function Space
            
            // Default to incompatible for safety
            _ => Ok(false),
        }
    }

    /// Create mapping function based on family types
    fn create_mapping_function(
        source_type: &FamilyType,
        target_type: &FamilyType,
        mapping_type: &MappingType,
    ) -> GIResult<MappingFunction> {
        match (source_type, target_type, mapping_type) {
            // Simple bijective mappings use identity-like transforms
            (FamilyType::Combinatorial, FamilyType::Combinatorial, MappingType::Bijective) => {
                Ok(MappingFunction::Polynomial { coefficients: vec![0.0, 1.0] }) // f(x) = x
            },
            
            // Algebraic mappings often use matrix transformations
            (FamilyType::Algebraic, FamilyType::Algebraic, MappingType::Homomorphism) => {
                Ok(MappingFunction::Matrix { 
                    matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]] // Identity matrix
                })
            },
            
            // Combinatorial to algebraic might use modular arithmetic
            (FamilyType::Combinatorial, FamilyType::Algebraic, _) => {
                Ok(MappingFunction::ModularArithmetic { a: 1, b: 0, modulus: 1000 })
            },
            
            // Default polynomial mapping
            _ => Ok(MappingFunction::Polynomial { coefficients: vec![0.0, 1.0] }),
        }
    }

    /// Create inverse mapping function
    fn create_inverse_mapping(forward: &MappingFunction) -> GIResult<MappingFunction> {
        match forward {
            MappingFunction::Polynomial { coefficients } => {
                if coefficients.len() == 2 && coefficients[1] != 0.0 {
                    // For linear f(x) = ax + b, inverse is (x - b) / a
                    let a = coefficients[1];
                    let b = coefficients[0];
                    Ok(MappingFunction::Polynomial { 
                        coefficients: vec![-b / a, 1.0 / a] 
                    })
                } else {
                    Err(GIError::MappingCreationFailed("Cannot invert non-linear polynomial".to_string()))
                }
            },
            MappingFunction::ModularArithmetic { a, b, modulus } => {
                // Inverse of (ax + b) mod m is (x - b) * a^(-1) mod m
                if let Some(a_inv) = Self::mod_inverse(*a, *modulus) {
                    Ok(MappingFunction::ModularArithmetic { 
                        a: a_inv, 
                        b: (modulus - (b * a_inv) % modulus) % modulus, 
                        modulus: *modulus 
                    })
                } else {
                    Err(GIError::MappingCreationFailed("Cannot compute modular inverse".to_string()))
                }
            },
            _ => Err(GIError::MappingCreationFailed("Inverse mapping not implemented for this function type".to_string())),
        }
    }

    /// Compute modular inverse using extended Euclidean algorithm
    fn mod_inverse(a: u64, m: u64) -> Option<u64> {
        fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
            if a == 0 {
                (b, 0, 1)
            } else {
                let (g, y1, x1) = extended_gcd(b % a, a);
                let x = x1 - (b / a) * y1;
                let y = y1;
                (g, x, y)
            }
        }

        let (g, x, _) = extended_gcd(a as i64, m as i64);
        if g != 1 {
            None
        } else {
            Some(((x % m as i64 + m as i64) % m as i64) as u64)
        }
    }

    /// Analyze properties of the mapping
    fn analyze_mapping_properties(
        mapping_func: &MappingFunction,
        mapping_type: &MappingType,
        source_family: &StructuredFamily,
        target_family: &StructuredFamily,
    ) -> MappingProperties {
        let order_preserving = match mapping_func {
            MappingFunction::Polynomial { coefficients } => {
                coefficients.len() == 2 && coefficients[1] > 0.0
            },
            MappingFunction::Matrix { matrix } => {
                // Check if matrix has positive diagonal entries
                matrix.iter().enumerate().all(|(i, row)| 
                    i < row.len() && row[i] > 0.0
                )
            },
            _ => false,
        };

        let structure_preserving = matches!(mapping_type, 
            MappingType::Isomorphism | MappingType::Homomorphism
        );

        let complexity = match mapping_func {
            MappingFunction::LookupTable(_) => AlgorithmComplexity::Constant,
            MappingFunction::Polynomial { coefficients } => {
                if coefficients.len() <= 2 {
                    AlgorithmComplexity::Constant
                } else {
                    AlgorithmComplexity::Linear
                }
            },
            MappingFunction::Matrix { .. } => AlgorithmComplexity::Linear,
            _ => AlgorithmComplexity::Linear,
        };

        MappingProperties {
            order_preserving,
            structure_preserving,
            domain_size: None, // Would need analysis of source family
            codomain_size: None, // Would need analysis of target family
            complexity,
            invariants: Vec::new(), // Would be populated based on specific mapping
            error_bounds: None,
        }
    }

    /// Evaluate the forward mapping
    pub fn evaluate_forward(&mut self, input: u64) -> GIResult<u64> {
        let start_time = std::time::Instant::now();
        let result = self.evaluate_mapping_function(&self.forward_mapping, input);
        
        // Update performance statistics
        let duration = start_time.elapsed();
        self.performance_stats.total_evaluations += 1;
        self.performance_stats.average_evaluation_time = 
            (self.performance_stats.average_evaluation_time * (self.performance_stats.total_evaluations - 1) as u32 + duration) 
            / self.performance_stats.total_evaluations as u32;
        
        result
    }

    /// Evaluate the inverse mapping
    pub fn evaluate_inverse(&mut self, input: u64) -> GIResult<u64> {
        if let Some(inverse) = &self.inverse_mapping {
            let start_time = std::time::Instant::now();
            let result = self.evaluate_mapping_function(inverse, input);
            
            // Update performance statistics
            let duration = start_time.elapsed();
            self.performance_stats.total_evaluations += 1;
            self.performance_stats.average_evaluation_time = 
                (self.performance_stats.average_evaluation_time * (self.performance_stats.total_evaluations - 1) as u32 + duration) 
                / self.performance_stats.total_evaluations as u32;
            
            result
        } else {
            Err(GIError::MappingNotFound("Inverse mapping not available".to_string()))
        }
    }

    /// Evaluate a specific mapping function
    fn evaluate_mapping_function(&self, func: &MappingFunction, input: u64) -> GIResult<u64> {
        match func {
            MappingFunction::LookupTable(table) => {
                table.get(&input).copied()
                    .ok_or_else(|| GIError::IndexOutOfBounds { index: input, bound: table.len() as u64 })
            },
            MappingFunction::Polynomial { coefficients } => {
                let x = input as f64;
                let result: f64 = coefficients.iter().enumerate()
                    .map(|(i, &coeff)| coeff * x.powi(i as i32))
                    .sum();
                Ok(result.round() as u64)
            },
            MappingFunction::ModularArithmetic { a, b, modulus } => {
                Ok((a * input + b) % modulus)
            },
            MappingFunction::BitTransformation { operations } => {
                let mut result = input;
                for op in operations {
                    result = self.apply_bit_operation(result, op)?;
                }
                Ok(result)
            },
            _ => Err(GIError::ComputationError("Mapping function evaluation not implemented".to_string())),
        }
    }

    /// Apply a bit operation
    fn apply_bit_operation(&self, input: u64, operation: &BitOperation) -> GIResult<u64> {
        match operation {
            BitOperation::LeftShift(n) => Ok(input << n),
            BitOperation::RightShift(n) => Ok(input >> n),
            BitOperation::XorMask(mask) => Ok(input ^ mask),
            BitOperation::AndMask(mask) => Ok(input & mask),
            BitOperation::OrMask(mask) => Ok(input | mask),
            BitOperation::Reverse => Ok(input.reverse_bits()),
            BitOperation::GrayCode => {
                // Convert to Gray code: G(n) = n XOR (n >> 1)
                Ok(input ^ (input >> 1))
            },
            BitOperation::Custom(name) => {
                Err(GIError::ComputationError(format!("Custom bit operation '{}' not implemented", name)))
            },
        }
    }
}

impl Default for MappingStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            cache_hits: 0,
            cache_misses: 0,
            average_evaluation_time: std::time::Duration::from_nanos(0),
            memory_usage: 0,
        }
    }
}
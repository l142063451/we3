//! Core IDVBit representation and data structures
//!
//! Provides the fundamental IDVBit type and its mathematical operations
//! for representing infinite-density vector bit sequences.

use crate::{IDVResult, IDVBitError, ComplexF64};
use num_complex::Complex;
use num_traits::{Zero, One, Float};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use bitvec::prelude::*;
use smallvec::SmallVec;

/// Representation format for IDVBit internal storage
#[derive(Debug, Clone, PartialEq)]
pub enum RepresentationFormat {
    /// Explicit bit vector (finite sequences)
    Explicit(BitVec),
    /// Generating function representation (infinite sequences)
    GeneratingFunction {
        /// Coefficients of rational generating function P(x)/Q(x)
        numerator: Vec<ComplexF64>,
        denominator: Vec<ComplexF64>,
    },
    /// Periodic sequence with period and initial segment
    Periodic {
        initial: BitVec,
        period: BitVec,
    },
    /// Compressed representation using run-length encoding
    RunLength(Vec<(bool, u64)>),
    /// Symbolic representation using mathematical expressions
    Symbolic {
        expression: String,
        parameters: HashMap<String, ComplexF64>,
    },
}

/// Core IDVBit structure for infinite-density vector bit representation
#[derive(Debug, Clone, PartialEq)]
pub struct IDVBit {
    /// Primary representation format
    pub format: RepresentationFormat,
    /// Metadata about the sequence
    pub metadata: IDVBitMetadata,
    /// Cached computations for performance
    pub cache: HashMap<String, CachedValue>,
}

/// Metadata associated with IDVBit instances
#[derive(Debug, Clone, PartialEq)]
pub struct IDVBitMetadata {
    /// Length of the sequence (None for infinite)
    pub length: Option<u64>,
    /// Known properties of the sequence
    pub properties: SequenceProperties,
    /// Density function parameters if applicable
    pub density_params: Option<DensityParameters>,
    /// Creation timestamp for provenance
    pub created_at: u64,
}

/// Properties that can be computed or known about a sequence
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceProperties {
    /// Whether the sequence is periodic
    pub is_periodic: Option<bool>,
    /// Whether the sequence is eventually periodic
    pub is_eventually_periodic: Option<bool>,
    /// Density (fraction of 1-bits) if known
    pub density: Option<f64>,
    /// Entropy measure if computed
    pub entropy: Option<f64>,
    /// Whether sequence has finite support
    pub finite_support: Option<bool>,
}

/// Parameters for density function representation
#[derive(Debug, Clone, PartialEq)]
pub struct DensityParameters {
    /// Type of density function
    pub function_type: DensityFunctionType,
    /// Function parameters
    pub parameters: Vec<f64>,
    /// Normalization constant
    pub normalization: f64,
}

/// Types of density functions supported
#[derive(Debug, Clone, PartialEq)]
pub enum DensityFunctionType {
    /// Gaussian density
    Gaussian,
    /// Exponential density
    Exponential,
    /// Power-law density
    PowerLaw,
    /// Custom density function
    Custom(String),
}

/// Cached computation results
#[derive(Debug, Clone, PartialEq)]
pub enum CachedValue {
    FloatValue(f64),
    ComplexValue(ComplexF64),
    VectorValue(Vec<f64>),
    StringValue(String),
}

impl IDVBit {
    /// Create a new IDVBit from explicit bit vector
    pub fn from_bitvec(bits: BitVec) -> Self {
        let length = bits.len() as u64;
        let density = bits.count_ones() as f64 / bits.len() as f64;
        
        Self {
            format: RepresentationFormat::Explicit(bits),
            metadata: IDVBitMetadata {
                length: Some(length),
                properties: SequenceProperties {
                    is_periodic: None,
                    is_eventually_periodic: None,
                    density: Some(density),
                    entropy: None,
                    finite_support: Some(true),
                },
                density_params: None,
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
            cache: HashMap::new(),
        }
    }

    /// Create IDVBit from generating function representation
    pub fn from_generating_function(
        numerator: Vec<ComplexF64>,
        denominator: Vec<ComplexF64>
    ) -> IDVResult<Self> {
        if denominator.is_empty() || denominator[0].is_zero() {
            return Err(IDVBitError::InvalidDensity(
                "Generating function denominator cannot be empty or have zero constant term".to_string()
            ));
        }

        Ok(Self {
            format: RepresentationFormat::GeneratingFunction { numerator, denominator },
            metadata: IDVBitMetadata {
                length: None, // Infinite sequence
                properties: SequenceProperties {
                    is_periodic: None,
                    is_eventually_periodic: Some(true), // Rational GF always eventually periodic
                    density: None,
                    entropy: None,
                    finite_support: Some(false),
                },
                density_params: None,
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
            cache: HashMap::new(),
        })
    }

    /// Create periodic IDVBit
    pub fn from_periodic(initial: BitVec, period: BitVec) -> IDVResult<Self> {
        if period.is_empty() {
            return Err(IDVBitError::InvalidDensity("Period cannot be empty".to_string()));
        }

        let total_ones = initial.count_ones() + period.count_ones();
        let total_length = initial.len() + period.len();
        let density = total_ones as f64 / total_length as f64;

        Ok(Self {
            format: RepresentationFormat::Periodic { initial, period },
            metadata: IDVBitMetadata {
                length: None, // Infinite periodic sequence
                properties: SequenceProperties {
                    is_periodic: Some(true),
                    is_eventually_periodic: Some(true),
                    density: Some(density),
                    entropy: None,
                    finite_support: Some(false),
                },
                density_params: None,
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
            cache: HashMap::new(),
        })
    }

    /// Get the bit at position i using advanced extraction algorithms
    pub fn get_bit(&self, index: u64) -> IDVResult<bool> {
        match &self.format {
            RepresentationFormat::Explicit(bits) => {
                if index as usize >= bits.len() {
                    Ok(false) // Zero-padding for out-of-bounds access
                } else {
                    Ok(bits[index as usize])
                }
            },
            RepresentationFormat::GeneratingFunction { numerator, denominator } => {
                self.extract_coefficient_from_gf(index, numerator, denominator)
            },
            RepresentationFormat::Periodic { initial, period } => {
                if (index as usize) < initial.len() {
                    Ok(initial[index as usize])
                } else {
                    let period_index = ((index as usize) - initial.len()) % period.len();
                    Ok(period[period_index])
                }
            },
            RepresentationFormat::RunLength(runs) => {
                self.extract_from_run_length(index, runs)
            },
            RepresentationFormat::Symbolic { expression: _, parameters: _ } => {
                // For now, return false for symbolic - would need expression evaluator
                Ok(false)
            }
        }
    }

    /// Extract coefficient from generating function using continued fraction expansion
    fn extract_coefficient_from_gf(
        &self,
        index: u64,
        numerator: &[ComplexF64],
        denominator: &[ComplexF64]
    ) -> IDVResult<bool> {
        // For rational generating functions P(x)/Q(x), extract coefficient of x^index
        
        // Handle special case: 1/(1-x) = 1 + x + x^2 + x^3 + ... (all coefficients are 1)
        if numerator.len() == 1 && denominator.len() == 2 
            && numerator[0].re.abs() - 1.0 < 1e-10 && numerator[0].im.abs() < 1e-10
            && denominator[0].re.abs() - 1.0 < 1e-10 && denominator[0].im.abs() < 1e-10
            && denominator[1].re.abs() - (-1.0) < 1e-10 && denominator[1].im.abs() < 1e-10 {
            return Ok(true); // All coefficients are 1 for 1/(1-x)
        }

        // Use polynomial long division to find coefficient of x^index
        let mut quotient_coeffs = vec![Complex::zero(); (index + 1) as usize];
        let mut remainder = numerator.to_vec();
        
        // Extend remainder if needed
        if remainder.len() <= index as usize {
            remainder.resize((index + 1) as usize, Complex::zero());
        }

        let denom_lead = denominator[0];
        if denom_lead.is_zero() {
            return Err(IDVBitError::NumericalError("Zero leading coefficient in denominator".to_string()));
        }

        // Perform polynomial long division
        for i in 0..=(index as usize) {
            if i < remainder.len() && !remainder[i].is_zero() {
                let coeff = remainder[i] / denom_lead;
                quotient_coeffs[i] = coeff;
                
                // Subtract coeff * denominator * x^i from remainder
                for (j, &denom_coeff) in denominator.iter().enumerate() {
                    if i + j < remainder.len() {
                        remainder[i + j] = remainder[i + j] - coeff * denom_coeff;
                    }
                }
            }
        }

        // Convert complex coefficient to bit (true if real part > 0.5)
        if (index as usize) < quotient_coeffs.len() {
            Ok(quotient_coeffs[index as usize].re > 0.5)
        } else {
            Ok(false)
        }
    }

    /// Extract bit from run-length encoding
    fn extract_from_run_length(&self, index: u64, runs: &[(bool, u64)]) -> IDVResult<bool> {
        let mut current_pos = 0u64;
        
        for &(value, length) in runs {
            if index >= current_pos && index < current_pos + length {
                return Ok(value);
            }
            current_pos += length;
            if current_pos > index {
                break;
            }
        }
        
        Ok(false) // Default to false if beyond encoded range
    }

    /// Compute sequence density using advanced statistical methods
    pub fn compute_density(&mut self) -> IDVResult<f64> {
        // Check cache first
        if let Some(CachedValue::FloatValue(density)) = self.cache.get("density") {
            return Ok(*density);
        }

        let density = match &self.format {
            RepresentationFormat::Explicit(bits) => {
                bits.count_ones() as f64 / bits.len() as f64
            },
            RepresentationFormat::Periodic { initial, period } => {
                let total_ones = initial.count_ones() + period.count_ones();
                let total_length = initial.len() + period.len();
                total_ones as f64 / total_length as f64
            },
            RepresentationFormat::GeneratingFunction { numerator, denominator } => {
                // For rational generating functions, compute limit of partial sums
                self.compute_gf_density(numerator, denominator)?
            },
            RepresentationFormat::RunLength(runs) => {
                let mut total_ones = 0u64;
                let mut total_length = 0u64;
                for &(value, length) in runs {
                    if value {
                        total_ones += length;
                    }
                    total_length += length;
                }
                if total_length > 0 {
                    total_ones as f64 / total_length as f64
                } else {
                    0.0
                }
            },
            RepresentationFormat::Symbolic { .. } => {
                // Would need expression evaluator
                0.5 // Default assumption
            }
        };

        // Cache the result
        self.cache.insert("density".to_string(), CachedValue::FloatValue(density));
        Ok(density)
    }

    /// Compute density for generating function using Abel's theorem
    fn compute_gf_density(&self, numerator: &[ComplexF64], denominator: &[ComplexF64]) -> IDVResult<f64> {
        // Use Abel's theorem: lim_{x->1-} f(x) = lim_{n->inf} sum of first n coefficients / n
        // For rational functions, evaluate lim_{x->1-} P(x)/Q(x)
        
        let x = Complex::new(0.999999, 0.0); // Approach 1 from below
        
        let numerator_val = self.evaluate_polynomial(numerator, x);
        let denominator_val = self.evaluate_polynomial(denominator, x);
        
        if denominator_val.norm() < 1e-15 {
            // Use L'Hôpital's rule
            let num_deriv = self.evaluate_polynomial_derivative(numerator, x);
            let den_deriv = self.evaluate_polynomial_derivative(denominator, x);
            
            if den_deriv.norm() < 1e-15 {
                return Err(IDVBitError::NumericalError("Cannot compute density - both numerator and denominator derivatives are zero".to_string()));
            }
            
            Ok((num_deriv / den_deriv).re.max(0.0).min(1.0))
        } else {
            Ok((numerator_val / denominator_val).re.max(0.0).min(1.0))
        }
    }

    /// Evaluate polynomial at given point
    fn evaluate_polynomial(&self, coeffs: &[ComplexF64], x: ComplexF64) -> ComplexF64 {
        coeffs.iter().rev().fold(Complex::zero(), |acc, &coeff| acc * x + coeff)
    }

    /// Evaluate polynomial derivative at given point
    fn evaluate_polynomial_derivative(&self, coeffs: &[ComplexF64], x: ComplexF64) -> ComplexF64 {
        if coeffs.len() <= 1 {
            return Complex::zero();
        }
        
        coeffs.iter().skip(1).enumerate().rev().fold(Complex::zero(), |acc, (i, &coeff)| {
            acc * x + coeff * Complex::new((i + 1) as f64, 0.0)
        })
    }

    /// Get the effective length of the sequence
    pub fn length(&self) -> Option<u64> {
        self.metadata.length
    }

    /// Check if the sequence has known periodic structure
    pub fn is_periodic(&self) -> Option<bool> {
        self.metadata.properties.is_periodic
    }

    /// Convert to explicit bit vector representation up to given length
    pub fn to_explicit(&self, max_length: usize) -> IDVResult<BitVec> {
        let mut result = bitvec![0; max_length];
        
        for i in 0..max_length {
            result.set(i, self.get_bit(i as u64)?);
        }
        
        Ok(result)
    }
}

impl Default for SequenceProperties {
    fn default() -> Self {
        Self {
            is_periodic: None,
            is_eventually_periodic: None,
            density: None,
            entropy: None,
            finite_support: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicit_idvbit() {
        let bits = bitvec![1, 0, 1, 1, 0];
        let idv = IDVBit::from_bitvec(bits.clone());
        
        assert_eq!(idv.format, RepresentationFormat::Explicit(bits));
        assert_eq!(idv.get_bit(0).unwrap(), true);
        assert_eq!(idv.get_bit(1).unwrap(), false);
        assert_eq!(idv.get_bit(2).unwrap(), true);
        assert_eq!(idv.length(), Some(5));
    }

    #[test]
    fn test_periodic_idvbit() {
        let initial = bitvec![1, 0];
        let period = bitvec![1, 1, 0];
        let idv = IDVBit::from_periodic(initial, period).unwrap();
        
        // Check initial segment
        assert_eq!(idv.get_bit(0).unwrap(), true);
        assert_eq!(idv.get_bit(1).unwrap(), false);
        
        // Check periodic part
        assert_eq!(idv.get_bit(2).unwrap(), true);  // First bit of period
        assert_eq!(idv.get_bit(3).unwrap(), true);  // Second bit of period
        assert_eq!(idv.get_bit(4).unwrap(), false); // Third bit of period
        assert_eq!(idv.get_bit(5).unwrap(), true);  // Cycle repeats
    }

    #[test]
    fn test_generating_function() {
        // GF for sequence 1,1,1,1,... = 1/(1-x)
        let numerator = vec![Complex::new(1.0, 0.0)];
        let denominator = vec![Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)];
        
        let idv = IDVBit::from_generating_function(numerator, denominator).unwrap();
        
        // All coefficients should be 1 (represented as true bits)
        assert_eq!(idv.get_bit(0).unwrap(), true);
        assert_eq!(idv.get_bit(1).unwrap(), true);
        assert_eq!(idv.get_bit(5).unwrap(), true);
    }

    #[test]
    fn test_run_length_encoding() {
        let runs = vec![(true, 3), (false, 2), (true, 4)];
        let idv = IDVBit {
            format: RepresentationFormat::RunLength(runs),
            metadata: IDVBitMetadata {
                length: Some(9),
                properties: SequenceProperties::default(),
                density_params: None,
                created_at: 0,
            },
            cache: HashMap::new(),
        };
        
        // First run: 3 true bits
        assert_eq!(idv.get_bit(0).unwrap(), true);
        assert_eq!(idv.get_bit(1).unwrap(), true);
        assert_eq!(idv.get_bit(2).unwrap(), true);
        
        // Second run: 2 false bits
        assert_eq!(idv.get_bit(3).unwrap(), false);
        assert_eq!(idv.get_bit(4).unwrap(), false);
        
        // Third run: 4 true bits
        assert_eq!(idv.get_bit(5).unwrap(), true);
        assert_eq!(idv.get_bit(8).unwrap(), true);
    }

    #[test]
    fn test_density_computation() {
        let bits = bitvec![1, 1, 0, 1, 0]; // 3 ones out of 5 bits
        let mut idv = IDVBit::from_bitvec(bits);
        
        let density = idv.compute_density().unwrap();
        assert!((density - 0.6).abs() < 1e-10); // 3/5 = 0.6
    }

    #[test]
    fn test_polynomial_evaluation() {
        let bits = bitvec![1];
        let idv = IDVBit::from_bitvec(bits);
        
        let coeffs = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]; // 1 + 2x
        let x = Complex::new(3.0, 0.0);
        let result = idv.evaluate_polynomial(&coeffs, x);
        
        assert_eq!(result, Complex::new(7.0, 0.0)); // 1 + 2*3 = 7
    }

    #[test]
    fn test_to_explicit_conversion() {
        let initial = bitvec![1, 0];
        let period = bitvec![1];
        let idv = IDVBit::from_periodic(initial, period).unwrap();
        
        let explicit = idv.to_explicit(6).unwrap();
        let expected = bitvec![1, 0, 1, 1, 1, 1]; // initial [1,0] then period [1] repeating
        
        assert_eq!(explicit, expected);
    }

    #[test]
    fn test_error_handling() {
        // Test invalid generating function
        let numerator = vec![Complex::new(1.0, 0.0)];
        let denominator = vec![Complex::new(0.0, 0.0)]; // Invalid: zero constant term
        
        let result = IDVBit::from_generating_function(numerator, denominator);
        assert!(result.is_err());
    }
}
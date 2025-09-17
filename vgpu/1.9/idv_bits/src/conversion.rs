//! Conversion algorithms between different IDVBit representations
//!
//! Provides efficient algorithms for converting between various IDVBit formats,
//! including explicit bits, generating functions, periodic sequences, and
//! superposition states.

use crate::{
    IDVResult, IDVBitError, ComplexF64, IDVBit, SuperpositionState,
    RepresentationFormat, DensityFunction, GaussianDensity
};
use num_complex::Complex;
use num_traits::{Zero, One, Float};
use bitvec::prelude::*;
use std::collections::HashMap;
use rayon::prelude::*;

/// Converter for transforming between IDVBit representations
pub struct IDVBitConverter {
    /// Conversion cache for performance optimization
    cache: HashMap<String, IDVBit>,
    /// Configuration parameters
    config: ConverterConfig,
}

/// Configuration options for IDVBit conversions
#[derive(Debug, Clone)]
pub struct ConverterConfig {
    /// Maximum length for explicit conversions
    pub max_explicit_length: usize,
    /// Tolerance for numerical computations
    pub numerical_tolerance: f64,
    /// Whether to use caching
    pub use_cache: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Precision for generating function coefficients
    pub gf_precision: usize,
}

impl Default for ConverterConfig {
    fn default() -> Self {
        Self {
            max_explicit_length: 10000,
            numerical_tolerance: 1e-12,
            use_cache: true,
            max_cache_size: 1000,
            gf_precision: 50,
        }
    }
}

impl IDVBitConverter {
    /// Create new converter with default configuration
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            config: ConverterConfig::default(),
        }
    }

    /// Create new converter with custom configuration
    pub fn with_config(config: ConverterConfig) -> Self {
        Self {
            cache: HashMap::new(),
            config,
        }
    }

    /// Convert IDVBit to explicit bit vector representation
    pub fn to_explicit(&mut self, idv_bit: &IDVBit, length: usize) -> IDVResult<IDVBit> {
        if length > self.config.max_explicit_length {
            return Err(IDVBitError::ResourceExhaustion(
                format!("Requested length {} exceeds maximum {}", length, self.config.max_explicit_length)
            ));
        }

        let cache_key = format!("explicit_{}_{}", self.hash_idvbit(idv_bit), length);
        
        if self.config.use_cache {
            if let Some(cached) = self.cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }

        let explicit_bits = match &idv_bit.format {
            RepresentationFormat::Explicit(bits) => {
                // Extend or truncate as needed
                let mut result = bits.clone();
                result.resize(length, false);
                result
            },
            RepresentationFormat::GeneratingFunction { numerator, denominator } => {
                self.gf_to_explicit(numerator, denominator, length)?
            },
            RepresentationFormat::Periodic { initial, period } => {
                self.periodic_to_explicit(initial, period, length)?
            },
            RepresentationFormat::RunLength(runs) => {
                self.runlength_to_explicit(runs, length)?
            },
            RepresentationFormat::Symbolic { expression: _, parameters: _ } => {
                // For symbolic, use zero padding (would need expression evaluator)
                bitvec![0; length]
            }
        };

        let result = IDVBit::from_bitvec(explicit_bits);
        
        if self.config.use_cache && self.cache.len() < self.config.max_cache_size {
            self.cache.insert(cache_key, result.clone());
        }

        Ok(result)
    }

    /// Convert IDVBit to generating function representation
    pub fn to_generating_function(&mut self, idv_bit: &IDVBit) -> IDVResult<IDVBit> {
        let cache_key = format!("gf_{}", self.hash_idvbit(idv_bit));
        
        if self.config.use_cache {
            if let Some(cached) = self.cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }

        let (numerator, denominator) = match &idv_bit.format {
            RepresentationFormat::GeneratingFunction { numerator, denominator } => {
                (numerator.clone(), denominator.clone())
            },
            RepresentationFormat::Explicit(bits) => {
                self.explicit_to_gf(bits)?
            },
            RepresentationFormat::Periodic { initial, period } => {
                self.periodic_to_gf(initial, period)?
            },
            RepresentationFormat::RunLength(runs) => {
                // Convert to periodic first if pattern is detected
                let explicit = self.runlength_to_explicit(runs, 1000)?; // Sample length
                self.explicit_to_gf(&explicit)?
            },
            RepresentationFormat::Symbolic { expression: _, parameters: _ } => {
                // Default to zero generating function
                (vec![Complex::zero()], vec![Complex::one()])
            }
        };

        let result = IDVBit::from_generating_function(numerator, denominator)?;
        
        if self.config.use_cache && self.cache.len() < self.config.max_cache_size {
            self.cache.insert(cache_key, result.clone());
        }

        Ok(result)
    }

    /// Convert IDVBit to periodic representation
    pub fn to_periodic(&mut self, idv_bit: &IDVBit) -> IDVResult<Option<IDVBit>> {
        match &idv_bit.format {
            RepresentationFormat::Periodic { initial, period } => {
                Ok(Some(IDVBit::from_periodic(initial.clone(), period.clone())?))
            },
            RepresentationFormat::Explicit(bits) => {
                if let Some((initial, period)) = self.detect_periodicity(bits)? {
                    Ok(Some(IDVBit::from_periodic(initial, period)?))
                } else {
                    Ok(None)
                }
            },
            RepresentationFormat::GeneratingFunction { numerator, denominator } => {
                // Rational generating functions are eventually periodic
                let explicit = self.gf_to_explicit(numerator, denominator, 1000)?;
                if let Some((initial, period)) = self.detect_periodicity(&explicit)? {
                    Ok(Some(IDVBit::from_periodic(initial, period)?))
                } else {
                    Ok(None)
                }
            },
            _ => Ok(None), // Other formats may not have obvious periodic structure
        }
    }

    /// Convert IDVBit to run-length encoding
    pub fn to_run_length(&self, idv_bit: &IDVBit, max_length: usize) -> IDVResult<IDVBit> {
        let explicit_bits = idv_bit.to_explicit(max_length)?;
        let runs = self.explicit_to_runlength(&explicit_bits);
        
        Ok(IDVBit {
            format: RepresentationFormat::RunLength(runs),
            metadata: idv_bit.metadata.clone(),
            cache: HashMap::new(),
        })
    }

    /// Convert IDVBit to superposition state
    pub fn to_superposition(&self, idv_bits: Vec<IDVBit>, amplitudes: Option<Vec<ComplexF64>>) -> IDVResult<SuperpositionState> {
        let amps = if let Some(provided_amplitudes) = amplitudes {
            if provided_amplitudes.len() != idv_bits.len() {
                return Err(IDVBitError::InvalidSuperposition(
                    "Number of amplitudes must match number of IDVBits".to_string()
                ));
            }
            provided_amplitudes
        } else {
            // Create uniform superposition
            let amp = Complex::new(1.0 / (idv_bits.len() as f64).sqrt(), 0.0);
            vec![amp; idv_bits.len()]
        };

        let states = amps.into_iter().zip(idv_bits).collect();
        SuperpositionState::new(states)
    }

    /// Advanced conversion using density function interpolation
    pub fn density_based_conversion(
        &self,
        source: &IDVBit,
        target_density: &dyn DensityFunction,
        length: usize
    ) -> IDVResult<IDVBit> {
        use bitvec::prelude::*;
        let mut result = bitvec![0; length];
        
        // Sample positions using the target density function
        let mut rng = rand::thread_rng();
        let positions: Vec<usize> = (0..length)
            .filter(|&i| {
                let x = i as f64 / length as f64;
                let density_value = target_density.evaluate(x);
                let source_bit = source.get_bit(i as u64).unwrap_or(false);
                
                // Use density to modulate source bits
                if source_bit {
                    let threshold = density_value;
                    use rand::RngCore;
                    let random_val = rng.next_u32() as f64 / u32::MAX as f64;
                    random_val < threshold
                } else {
                    false
                }
            })
            .collect();

        for pos in positions {
            result.set(pos, true);
        }

        Ok(IDVBit::from_bitvec(result))
    }

    /// Batch conversion of multiple IDVBits
    pub fn batch_convert(
        &mut self,
        idv_bits: &[IDVBit],
        target_format: ConversionTarget,
        params: ConversionParams
    ) -> IDVResult<Vec<IDVBit>> {
        let mut results = Vec::with_capacity(idv_bits.len());
        
        for idv_bit in idv_bits {
            let converted = self.convert_single(idv_bit, &target_format, &params)?;
            results.push(converted);
        }

        Ok(results)
    }

    /// Helper methods for specific conversions

    /// Convert explicit bits to generating function
    fn explicit_to_gf(&self, bits: &BitVec) -> IDVResult<(Vec<ComplexF64>, Vec<ComplexF64>)> {
        // Use polynomial interpolation to find rational approximation
        let coefficients: Vec<ComplexF64> = bits
            .iter()
            .map(|bit| Complex::new(if *bit { 1.0 } else { 0.0 }, 0.0))
            .collect();

        // Find rational approximation using Padé approximation
        self.pade_approximation(&coefficients)
    }

    /// Convert generating function to explicit bits
    fn gf_to_explicit(
        &self,
        numerator: &[ComplexF64],
        denominator: &[ComplexF64],
        length: usize
    ) -> IDVResult<BitVec> {
        let mut result = bitvec![0; length];
        
        // Use synthetic division to extract coefficients
        let mut coeffs = self.extract_gf_coefficients(numerator, denominator, length)?;
        
        for (i, coeff) in coeffs.iter().enumerate().take(length) {
            // Convert complex coefficient to bit (threshold at 0.5)
            result.set(i, coeff.re > 0.5);
        }

        Ok(result)
    }

    /// Extract coefficients from generating function using synthetic division
    fn extract_gf_coefficients(
        &self,
        numerator: &[ComplexF64],
        denominator: &[ComplexF64],
        count: usize
    ) -> IDVResult<Vec<ComplexF64>> {
        let mut coefficients = vec![Complex::zero(); count];
        let mut remainder = numerator.to_vec();
        remainder.resize(count + denominator.len(), Complex::zero());

        if denominator.is_empty() || denominator[0].is_zero() {
            return Err(IDVBitError::NumericalError(
                "Invalid denominator in generating function".to_string()
            ));
        }

        let lead_coeff = denominator[0];

        for i in 0..count {
            if i < remainder.len() {
                coefficients[i] = remainder[i] / lead_coeff;
                
                // Update remainder
                for (j, &denom_coeff) in denominator.iter().enumerate() {
                    if i + j < remainder.len() {
                        remainder[i + j] = remainder[i + j] - coefficients[i] * denom_coeff;
                    }
                }
            }
        }

        Ok(coefficients)
    }

    /// Convert periodic to generating function
    fn periodic_to_gf(
        &self,
        initial: &BitVec,
        period: &BitVec
    ) -> IDVResult<(Vec<ComplexF64>, Vec<ComplexF64>)> {
        // For periodic sequence a₀, a₁, ..., aₖ₋₁, b₀, b₁, ..., bₘ₋₁, b₀, b₁, ...
        // GF = (initial_poly + x^k * period_poly) / (1 - x^m)
        
        let initial_coeffs: Vec<ComplexF64> = initial
            .iter()
            .map(|bit| Complex::new(if *bit { 1.0 } else { 0.0 }, 0.0))
            .collect();

        let period_coeffs: Vec<ComplexF64> = period
            .iter()
            .map(|bit| Complex::new(if *bit { 1.0 } else { 0.0 }, 0.0))
            .collect();

        // Numerator: initial + x^k * period * (1 / (1 - x^m))
        // This is a simplified construction - full implementation would be more complex
        let mut numerator = initial_coeffs.clone();
        
        // Extend numerator to accommodate period shift
        let shift = initial.len();
        numerator.resize(shift + period_coeffs.len(), Complex::zero());
        
        for (i, &period_coeff) in period_coeffs.iter().enumerate() {
            numerator[shift + i] = numerator[shift + i] + period_coeff;
        }

        // Denominator: 1 - x^m where m is period length
        let mut denominator = vec![Complex::zero(); period.len() + 1];
        denominator[0] = Complex::one();
        denominator[period.len()] = -Complex::one();

        Ok((numerator, denominator))
    }

    /// Convert periodic to explicit
    fn periodic_to_explicit(
        &self,
        initial: &BitVec,
        period: &BitVec,
        length: usize
    ) -> IDVResult<BitVec> {
        let mut result = bitvec![0; length];
        
        for i in 0..length {
            let bit = if i < initial.len() {
                initial[i]
            } else {
                let period_index = (i - initial.len()) % period.len();
                period[period_index]
            };
            result.set(i, bit);
        }

        Ok(result)
    }

    /// Convert run-length to explicit
    fn runlength_to_explicit(&self, runs: &[(bool, u64)], length: usize) -> IDVResult<BitVec> {
        let mut result = bitvec![0; length];
        let mut pos = 0;

        for &(value, count) in runs {
            for _ in 0..count {
                if pos >= length {
                    break;
                }
                result.set(pos, value);
                pos += 1;
            }
            if pos >= length {
                break;
            }
        }

        Ok(result)
    }

    /// Convert explicit to run-length
    fn explicit_to_runlength(&self, bits: &BitVec) -> Vec<(bool, u64)> {
        let mut runs = Vec::new();
        
        if bits.is_empty() {
            return runs;
        }

        let mut current_bit = bits[0];
        let mut current_count = 1u64;

        for bit in bits.iter().skip(1) {
            if *bit == current_bit {
                current_count += 1;
            } else {
                runs.push((current_bit, current_count));
                current_bit = *bit;
                current_count = 1;
            }
        }

        runs.push((current_bit, current_count));
        runs
    }

    /// Detect periodicity in explicit bit sequence
    fn detect_periodicity(&self, bits: &BitVec) -> IDVResult<Option<(BitVec, BitVec)>> {
        let n = bits.len();
        if n < 4 {
            return Ok(None); // Too short to detect meaningful periods
        }

        // Try different period lengths
        for period_len in 1..=(n / 2) {
            if self.is_periodic_with_length(bits, period_len) {
                // Find the initial non-periodic part
                let mut initial_len = 0;
                
                // Search for where periodicity starts
                for start in 0..(n - period_len) {
                    if self.is_periodic_from_position(bits, start, period_len) {
                        initial_len = start;
                        break;
                    }
                }

                let initial = BitVec::from_iter(bits.iter().take(initial_len));
                let period = BitVec::from_iter(
                    bits.iter()
                        .skip(initial_len)
                        .take(period_len)
                );

                return Ok(Some((initial, period)));
            }
        }

        Ok(None)
    }

    /// Check if sequence is periodic with given period length
    fn is_periodic_with_length(&self, bits: &BitVec, period_len: usize) -> bool {
        if period_len == 0 || bits.len() < 2 * period_len {
            return false;
        }

        for i in 0..(bits.len() - period_len) {
            if bits[i] != bits[i + period_len] {
                return false;
            }
        }

        true
    }

    /// Check if sequence is periodic from given position
    fn is_periodic_from_position(&self, bits: &BitVec, start: usize, period_len: usize) -> bool {
        if start + 2 * period_len > bits.len() {
            return false;
        }

        for i in 0..period_len {
            if bits[start + i] != bits[start + i + period_len] {
                return false;
            }
        }

        true
    }

    /// Padé approximation for rational function fitting
    fn pade_approximation(&self, coefficients: &[ComplexF64]) -> IDVResult<(Vec<ComplexF64>, Vec<ComplexF64>)> {
        let n = coefficients.len();
        if n < 2 {
            return Ok((coefficients.to_vec(), vec![Complex::one()]));
        }

        // Use simple [n/2, n/2] Padé approximant
        let p_degree = n / 2;
        let q_degree = n / 2;

        // This is a simplified implementation - full Padé approximation
        // would involve solving linear systems
        let numerator = coefficients[..p_degree.min(n)].to_vec();
        let mut denominator = vec![Complex::zero(); q_degree + 1];
        denominator[0] = Complex::one();

        Ok((numerator, denominator))
    }

    /// Generate hash for IDVBit for caching
    fn hash_idvbit(&self, idv_bit: &IDVBit) -> u64 {
        // Simple hash based on format type and some content
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        match &idv_bit.format {
            RepresentationFormat::Explicit(bits) => {
                "explicit".hash(&mut hasher);
                bits.len().hash(&mut hasher);
                // Hash first few bits for uniqueness
                for (i, bit) in bits.iter().enumerate().take(32) {
                    (i, *bit).hash(&mut hasher);
                }
            },
            RepresentationFormat::GeneratingFunction { numerator, denominator } => {
                "gf".hash(&mut hasher);
                numerator.len().hash(&mut hasher);
                denominator.len().hash(&mut hasher);
            },
            RepresentationFormat::Periodic { initial, period } => {
                "periodic".hash(&mut hasher);
                initial.len().hash(&mut hasher);
                period.len().hash(&mut hasher);
            },
            RepresentationFormat::RunLength(runs) => {
                "runlength".hash(&mut hasher);
                runs.len().hash(&mut hasher);
            },
            RepresentationFormat::Symbolic { expression, parameters: _ } => {
                "symbolic".hash(&mut hasher);
                expression.hash(&mut hasher);
            },
        }
        
        hasher.finish()
    }

    /// Convert single IDVBit based on target format
    fn convert_single(
        &mut self,
        idv_bit: &IDVBit,
        target: &ConversionTarget,
        params: &ConversionParams
    ) -> IDVResult<IDVBit> {
        match target {
            ConversionTarget::Explicit => {
                self.to_explicit(idv_bit, params.length.unwrap_or(1000))
            },
            ConversionTarget::GeneratingFunction => {
                self.to_generating_function(idv_bit)
            },
            ConversionTarget::Periodic => {
                self.to_periodic(idv_bit)?.ok_or_else(|| {
                    IDVBitError::IncompatibleRepresentation(
                        "No periodic structure detected".to_string()
                    )
                })
            },
            ConversionTarget::RunLength => {
                self.to_run_length(idv_bit, params.length.unwrap_or(1000))
            },
        }
    }
}

/// Target format for conversions
#[derive(Debug, Clone)]
pub enum ConversionTarget {
    Explicit,
    GeneratingFunction,
    Periodic,
    RunLength,
}

/// Parameters for conversion operations
#[derive(Debug, Clone)]
pub struct ConversionParams {
    pub length: Option<usize>,
    pub tolerance: Option<f64>,
    pub preserve_properties: bool,
}

impl Default for ConversionParams {
    fn default() -> Self {
        Self {
            length: None,
            tolerance: None,
            preserve_properties: true,
        }
    }
}

impl Default for IDVBitConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::prelude::*;

    #[test]
    fn test_explicit_to_gf_conversion() {
        let mut converter = IDVBitConverter::new();
        let bits = bitvec![1, 1, 1, 1]; // All ones
        let idv_bit = IDVBit::from_bitvec(bits);
        
        let gf_result = converter.to_generating_function(&idv_bit).unwrap();
        
        match &gf_result.format {
            RepresentationFormat::GeneratingFunction { numerator, denominator } => {
                assert!(!numerator.is_empty());
                assert!(!denominator.is_empty());
            },
            _ => panic!("Expected generating function format"),
        }
    }

    #[test]
    fn test_periodic_conversion() {
        let mut converter = IDVBitConverter::new();
        let bits = bitvec![0, 1, 0, 1, 0, 1]; // Periodic pattern
        let idv_bit = IDVBit::from_bitvec(bits);
        
        let periodic_result = converter.to_periodic(&idv_bit).unwrap();
        assert!(periodic_result.is_some());
        
        if let Some(periodic_idv) = periodic_result {
            match &periodic_idv.format {
                RepresentationFormat::Periodic { initial: _, period } => {
                    assert!(!period.is_empty());
                },
                _ => panic!("Expected periodic format"),
            }
        }
    }

    #[test]
    fn test_run_length_conversion() {
        let converter = IDVBitConverter::new();
        let bits = bitvec![1, 1, 1, 0, 0, 1]; // Mixed runs
        let idv_bit = IDVBit::from_bitvec(bits);
        
        let rl_result = converter.to_run_length(&idv_bit, 6).unwrap();
        
        match &rl_result.format {
            RepresentationFormat::RunLength(runs) => {
                assert_eq!(runs.len(), 3); // Three runs: 3 ones, 2 zeros, 1 one
                assert_eq!(runs[0], (true, 3));
                assert_eq!(runs[1], (false, 2));
                assert_eq!(runs[2], (true, 1));
            },
            _ => panic!("Expected run-length format"),
        }
    }

    #[test]
    fn test_superposition_conversion() {
        let converter = IDVBitConverter::new();
        let bit1 = IDVBit::from_bitvec(bitvec![1, 0]);
        let bit2 = IDVBit::from_bitvec(bitvec![0, 1]);
        
        let superpos = converter.to_superposition(vec![bit1, bit2], None).unwrap();
        assert_eq!(superpos.num_states(), 2);
    }

    #[test]
    fn test_density_based_conversion() {
        let converter = IDVBitConverter::new();
        let source = IDVBit::from_bitvec(bitvec![1, 1, 1, 1]);
        let gaussian = GaussianDensity::standard();
        
        let result = converter.density_based_conversion(&source, &gaussian, 10).unwrap();
        
        // Should produce some variation based on density function
        match &result.format {
            RepresentationFormat::Explicit(bits) => {
                assert_eq!(bits.len(), 10);
            },
            _ => panic!("Expected explicit format"),
        }
    }

    #[test]
    fn test_batch_conversion() {
        let mut converter = IDVBitConverter::new();
        let idv_bits = vec![
            IDVBit::from_bitvec(bitvec![1, 0, 1]),
            IDVBit::from_bitvec(bitvec![0, 1, 0]),
        ];
        
        let results = converter.batch_convert(
            &idv_bits,
            ConversionTarget::RunLength,
            ConversionParams::default()
        ).unwrap();
        
        assert_eq!(results.len(), 2);
        for result in results {
            match &result.format {
                RepresentationFormat::RunLength(_) => {},
                _ => panic!("Expected run-length format"),
            }
        }
    }

    #[test]
    fn test_periodicity_detection() {
        let converter = IDVBitConverter::new();
        let bits = bitvec![0, 1, 1, 0, 1, 1, 0, 1, 1]; // Period: [0,1,1]
        
        let result = converter.detect_periodicity(&bits).unwrap();
        assert!(result.is_some());
        
        if let Some((initial, period)) = result {
            assert_eq!(period, bitvec![0, 1, 1]);
        }
    }

    #[test]
    fn test_conversion_caching() {
        let mut converter = IDVBitConverter::with_config(ConverterConfig {
            use_cache: true,
            max_cache_size: 10,
            ..Default::default()
        });
        
        let idv_bit = IDVBit::from_bitvec(bitvec![1, 0, 1]);
        
        // First conversion
        let result1 = converter.to_explicit(&idv_bit, 5).unwrap();
        
        // Second conversion (should use cache)
        let result2 = converter.to_explicit(&idv_bit, 5).unwrap();
        
        assert_eq!(result1.format, result2.format);
        assert!(converter.cache.len() > 0);
    }

    #[test]
    fn test_error_handling() {
        let mut converter = IDVBitConverter::new();
        let idv_bit = IDVBit::from_bitvec(bitvec![1]);
        
        // Test length limit
        converter.config.max_explicit_length = 5;
        let result = converter.to_explicit(&idv_bit, 10);
        assert!(result.is_err());
    }
}
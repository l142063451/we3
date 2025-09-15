//! Advanced operations on IDVBit sequences
//!
//! Implements mathematical operations including bitwise operations,
//! algebraic operations, and advanced transformations.

use crate::{IDVResult, IDVBitError, ComplexF64, IDVBit, RepresentationFormat};
use num_complex::Complex;
use num_traits::{Zero, One, Float};
use bitvec::prelude::*;
use std::collections::HashMap;
use rayon::prelude::*;

/// Bitwise operations on IDVBit sequences
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BitwiseOperation {
    And,
    Or,
    Xor,
    Not,
    Nand,
    Nor,
}

/// Algebraic operations on IDVBit sequences
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlgebraicOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Root,
}

/// Transformation operations on IDVBit sequences
#[derive(Debug, Clone, PartialEq)]
pub enum TransformOperation {
    /// Fast Fourier Transform
    FFT,
    /// Discrete Wavelet Transform
    DWT,
    /// Hilbert Transform
    Hilbert,
    /// Custom kernel transformation
    CustomKernel(Vec<ComplexF64>),
}

/// IDVBit operation engine
pub struct IDVBitOperations {
    /// Configuration for operations
    config: OperationConfig,
    /// FFT cache for performance
    fft_cache: HashMap<usize, Vec<ComplexF64>>,
}

/// Configuration for IDVBit operations
#[derive(Debug, Clone)]
pub struct OperationConfig {
    /// Maximum length for explicit operations
    pub max_operation_length: usize,
    /// Numerical tolerance
    pub tolerance: f64,
    /// Whether to use parallel processing
    pub use_parallel: bool,
    /// FFT cache size limit
    pub max_fft_cache_size: usize,
}

impl Default for OperationConfig {
    fn default() -> Self {
        Self {
            max_operation_length: 100000,
            tolerance: 1e-12,
            use_parallel: true,
            max_fft_cache_size: 1000,
        }
    }
}

impl IDVBitOperations {
    /// Create new operation engine
    pub fn new() -> Self {
        Self {
            config: OperationConfig::default(),
            fft_cache: HashMap::new(),
        }
    }

    /// Create operation engine with custom configuration
    pub fn with_config(config: OperationConfig) -> Self {
        Self {
            config,
            fft_cache: HashMap::new(),
        }
    }

    /// Perform bitwise operation on two IDVBits
    pub fn bitwise_operation(
        &self,
        left: &IDVBit,
        right: &IDVBit,
        operation: BitwiseOperation
    ) -> IDVResult<IDVBit> {
        // Determine operation length
        let max_length = self.determine_operation_length(left, right)?;
        
        match operation {
            BitwiseOperation::Not => self.bitwise_not(left, max_length),
            _ => self.binary_bitwise_operation(left, right, operation, max_length),
        }
    }

    /// Perform algebraic operation on two IDVBits
    pub fn algebraic_operation(
        &self,
        left: &IDVBit,
        right: &IDVBit,
        operation: AlgebraicOperation
    ) -> IDVResult<IDVBit> {
        match operation {
            AlgebraicOperation::Add => self.algebraic_add(left, right),
            AlgebraicOperation::Subtract => self.algebraic_subtract(left, right),
            AlgebraicOperation::Multiply => self.algebraic_multiply(left, right),
            AlgebraicOperation::Divide => self.algebraic_divide(left, right),
            AlgebraicOperation::Power => self.algebraic_power(left, right),
            AlgebraicOperation::Root => self.algebraic_root(left, right),
        }
    }

    /// Apply transformation to IDVBit sequence
    pub fn transform(&mut self, idv_bit: &IDVBit, transform: TransformOperation) -> IDVResult<IDVBit> {
        match transform {
            TransformOperation::FFT => self.fft_transform(idv_bit),
            TransformOperation::DWT => self.dwt_transform(idv_bit),
            TransformOperation::Hilbert => self.hilbert_transform(idv_bit),
            TransformOperation::CustomKernel(kernel) => self.custom_kernel_transform(idv_bit, &kernel),
        }
    }

    /// Compute convolution of two IDVBit sequences
    pub fn convolve(&mut self, left: &IDVBit, right: &IDVBit) -> IDVResult<IDVBit> {
        // Use FFT-based convolution for efficiency
        let left_fft = self.to_complex_sequence(left, None)?;
        let right_fft = self.to_complex_sequence(right, None)?;
        
        let conv_length = left_fft.len() + right_fft.len() - 1;
        let fft_size = next_power_of_2(conv_length);
        
        // Zero-pad sequences
        let mut left_padded = left_fft;
        let mut right_padded = right_fft;
        left_padded.resize(fft_size, Complex::zero());
        right_padded.resize(fft_size, Complex::zero());
        
        // Forward FFT
        let left_freq = self.fft(&left_padded)?;
        let right_freq = self.fft(&right_padded)?;
        
        // Pointwise multiplication
        let conv_freq: Vec<ComplexF64> = left_freq
            .iter()
            .zip(right_freq.iter())
            .map(|(a, b)| a * b)
            .collect();
        
        // Inverse FFT
        let conv_result = self.ifft(&conv_freq)?;
        
        // Convert back to IDVBit
        self.from_complex_sequence(&conv_result[..conv_length])
    }

    /// Compute correlation between two IDVBit sequences
    pub fn correlate(&mut self, left: &IDVBit, right: &IDVBit) -> IDVResult<IDVBit> {
        // Correlation is convolution with time-reversed second sequence
        let right_reversed = self.reverse_sequence(right)?;
        self.convolve(left, &right_reversed)
    }

    /// Apply sliding window operation
    pub fn sliding_window<F>(
        &self,
        idv_bit: &IDVBit,
        window_size: usize,
        operation: F
    ) -> IDVResult<IDVBit>
    where
        F: Fn(&[bool]) -> bool + Send + Sync,
    {
        use bitvec::prelude::*;
        let length = self.get_effective_length(idv_bit)?;
        if window_size > length {
            return Err(IDVBitError::InvalidSuperposition(
                "Window size exceeds sequence length".to_string()
            ));
        }

        let result_length = length - window_size + 1;
        let mut result = bitvec![0; result_length];

        if self.config.use_parallel {
            let results: IDVResult<Vec<_>> = (0..result_length)
                .into_par_iter()
                .map(|i| {
                    let window: Vec<bool> = (0..window_size)
                        .map(|j| idv_bit.get_bit((i + j) as u64).unwrap_or(false))
                        .collect();
                    Ok(operation(&window))
                })
                .collect();
            
            for (i, bit) in results?.into_iter().enumerate() {
                result.set(i, bit);
            }
        } else {
            for i in 0..result_length {
                let window: Vec<bool> = (0..window_size)
                    .map(|j| idv_bit.get_bit((i + j) as u64).unwrap_or(false))
                    .collect();
                result.set(i, operation(&window));
            }
        }

        Ok(IDVBit::from_bitvec(result))
    }

    /// Compute entropy of IDVBit sequence
    pub fn entropy(&self, idv_bit: &IDVBit, block_size: usize) -> IDVResult<f64> {
        let length = self.get_effective_length(idv_bit)?;
        if block_size == 0 || length == 0 {
            return Ok(0.0);
        }

        let num_blocks = length / block_size;
        let mut pattern_counts: HashMap<Vec<bool>, usize> = HashMap::new();

        for i in 0..num_blocks {
            let block: Vec<bool> = (0..block_size)
                .map(|j| idv_bit.get_bit((i * block_size + j) as u64).unwrap_or(false))
                .collect();
            
            *pattern_counts.entry(block).or_insert(0) += 1;
        }

        // Compute Shannon entropy
        let mut entropy = 0.0;
        let total_blocks = num_blocks as f64;
        
        for &count in pattern_counts.values() {
            let probability = count as f64 / total_blocks;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }

        Ok(entropy)
    }

    /// Compute Lempel-Ziv complexity
    pub fn lempel_ziv_complexity(&self, idv_bit: &IDVBit) -> IDVResult<usize> {
        let length = self.get_effective_length(idv_bit)?;
        let mut complexity = 0;
        let mut i = 0;

        while i < length {
            let mut max_match_len = 0;
            
            // Find longest match in previous sequence
            for start in 0..i {
                let mut match_len = 0;
                
                while i + match_len < length && start + match_len < i {
                    let current_bit = idv_bit.get_bit((i + match_len) as u64).unwrap_or(false);
                    let reference_bit = idv_bit.get_bit((start + match_len) as u64).unwrap_or(false);
                    
                    if current_bit == reference_bit {
                        match_len += 1;
                    } else {
                        break;
                    }
                }
                
                max_match_len = max_match_len.max(match_len);
            }

            // Move forward by at least one position
            i += (max_match_len + 1).max(1);
            complexity += 1;
        }

        Ok(complexity)
    }

    // Private helper methods

    /// Determine appropriate operation length for two sequences
    fn determine_operation_length(&self, left: &IDVBit, right: &IDVBit) -> IDVResult<usize> {
        let left_len = self.get_effective_length(left)?;
        let right_len = self.get_effective_length(right)?;
        let max_len = left_len.max(right_len);
        
        if max_len > self.config.max_operation_length {
            return Err(IDVBitError::ResourceExhaustion(
                format!("Operation length {} exceeds maximum {}", max_len, self.config.max_operation_length)
            ));
        }

        Ok(max_len)
    }

    /// Get effective length of IDVBit sequence
    fn get_effective_length(&self, idv_bit: &IDVBit) -> IDVResult<usize> {
        match &idv_bit.format {
            RepresentationFormat::Explicit(bits) => Ok(bits.len()),
            _ => {
                // For other formats, use a reasonable default
                Ok(1000.min(self.config.max_operation_length))
            }
        }
    }

    /// Perform bitwise NOT operation
    fn bitwise_not(&self, idv_bit: &IDVBit, length: usize) -> IDVResult<IDVBit> {
        use bitvec::prelude::*;
        let mut result = bitvec![0; length];
        
        for i in 0..length {
            let bit = idv_bit.get_bit(i as u64)?;
            result.set(i, !bit);
        }

        Ok(IDVBit::from_bitvec(result))
    }

    /// Perform binary bitwise operation
    fn binary_bitwise_operation(
        &self,
        left: &IDVBit,
        right: &IDVBit,
        operation: BitwiseOperation,
        length: usize
    ) -> IDVResult<IDVBit> {
        use bitvec::prelude::*;
        let mut result = bitvec![0; length];

        if self.config.use_parallel && length > 1000 {
            let results: IDVResult<Vec<_>> = (0..length)
                .into_par_iter()
                .map(|i| {
                    let left_bit = left.get_bit(i as u64)?;
                    let right_bit = right.get_bit(i as u64)?;
                    
                    let result_bit = match operation {
                        BitwiseOperation::And => left_bit & right_bit,
                        BitwiseOperation::Or => left_bit | right_bit,
                        BitwiseOperation::Xor => left_bit ^ right_bit,
                        BitwiseOperation::Nand => !(left_bit & right_bit),
                        BitwiseOperation::Nor => !(left_bit | right_bit),
                        BitwiseOperation::Not => unreachable!("NOT is unary operation"),
                    };
                    
                    Ok((i, result_bit))
                })
                .collect();

            for (i, bit) in results? {
                result.set(i, bit);
            }
        } else {
            for i in 0..length {
                let left_bit = left.get_bit(i as u64)?;
                let right_bit = right.get_bit(i as u64)?;
                
                let result_bit = match operation {
                    BitwiseOperation::And => left_bit & right_bit,
                    BitwiseOperation::Or => left_bit | right_bit,
                    BitwiseOperation::Xor => left_bit ^ right_bit,
                    BitwiseOperation::Nand => !(left_bit & right_bit),
                    BitwiseOperation::Nor => !(left_bit | right_bit),
                    BitwiseOperation::Not => unreachable!("NOT is unary operation"),
                };
                
                result.set(i, result_bit);
            }
        }

        Ok(IDVBit::from_bitvec(result))
    }

    /// Algebraic addition (GF addition for generating functions)
    fn algebraic_add(&self, left: &IDVBit, right: &IDVBit) -> IDVResult<IDVBit> {
        match (&left.format, &right.format) {
            (
                RepresentationFormat::GeneratingFunction { numerator: n1, denominator: d1 },
                RepresentationFormat::GeneratingFunction { numerator: n2, denominator: d2 }
            ) => {
                // Add rational functions: P1/Q1 + P2/Q2 = (P1*Q2 + P2*Q1)/(Q1*Q2)
                let new_numerator = self.polynomial_add(
                    &self.polynomial_multiply(n1, d2)?,
                    &self.polynomial_multiply(n2, d1)?
                );
                let new_denominator = self.polynomial_multiply(d1, d2)?;
                
                IDVBit::from_generating_function(new_numerator, new_denominator)
            },
            _ => {
                // Fall back to bitwise XOR for other formats
                self.bitwise_operation(left, right, BitwiseOperation::Xor)
            }
        }
    }

    /// Algebraic subtraction
    fn algebraic_subtract(&self, left: &IDVBit, right: &IDVBit) -> IDVResult<IDVBit> {
        // For generating functions, this is addition with negated second operand
        self.algebraic_add(left, right) // Simplified - would need proper subtraction
    }

    /// Algebraic multiplication (convolution for sequences)
    fn algebraic_multiply(&self, left: &IDVBit, right: &IDVBit) -> IDVResult<IDVBit> {
        match (&left.format, &right.format) {
            (
                RepresentationFormat::GeneratingFunction { numerator: n1, denominator: d1 },
                RepresentationFormat::GeneratingFunction { numerator: n2, denominator: d2 }
            ) => {
                // Multiply rational functions: (P1/Q1) * (P2/Q2) = (P1*P2)/(Q1*Q2)
                let new_numerator = self.polynomial_multiply(n1, n2)?;
                let new_denominator = self.polynomial_multiply(d1, d2)?;
                
                IDVBit::from_generating_function(new_numerator, new_denominator)
            },
            _ => {
                // For other formats, use convolution
                let mut ops = IDVBitOperations::new();
                ops.convolve(left, right)
            }
        }
    }

    /// Algebraic division
    fn algebraic_divide(&self, left: &IDVBit, right: &IDVBit) -> IDVResult<IDVBit> {
        match (&left.format, &right.format) {
            (
                RepresentationFormat::GeneratingFunction { numerator: n1, denominator: d1 },
                RepresentationFormat::GeneratingFunction { numerator: n2, denominator: d2 }
            ) => {
                // Divide rational functions: (P1/Q1) / (P2/Q2) = (P1*Q2)/(Q1*P2)
                let new_numerator = self.polynomial_multiply(n1, d2)?;
                let new_denominator = self.polynomial_multiply(d1, n2)?;
                
                IDVBit::from_generating_function(new_numerator, new_denominator)
            },
            _ => {
                Err(IDVBitError::IncompatibleRepresentation(
                    "Division only supported for generating function representations".to_string()
                ))
            }
        }
    }

    /// Algebraic power
    fn algebraic_power(&self, base: &IDVBit, exponent: &IDVBit) -> IDVResult<IDVBit> {
        // Simplified implementation - would need more sophisticated power algorithms
        Err(IDVBitError::IncompatibleRepresentation(
            "Power operation not yet implemented".to_string()
        ))
    }

    /// Algebraic root
    fn algebraic_root(&self, base: &IDVBit, index: &IDVBit) -> IDVResult<IDVBit> {
        // Simplified implementation - would need root-finding algorithms
        Err(IDVBitError::IncompatibleRepresentation(
            "Root operation not yet implemented".to_string()
        ))
    }

    /// FFT transform
    fn fft_transform(&mut self, idv_bit: &IDVBit) -> IDVResult<IDVBit> {
        let sequence = self.to_complex_sequence(idv_bit, None)?;
        let fft_result = self.fft(&sequence)?;
        self.from_complex_sequence(&fft_result)
    }

    /// DWT transform (simplified)
    fn dwt_transform(&self, idv_bit: &IDVBit) -> IDVResult<IDVBit> {
        use bitvec::prelude::*;
        // Simplified DWT implementation - would use proper wavelet bases
        let length = self.get_effective_length(idv_bit)?;
        let mut result = bitvec![0; length];
        
        // Simple Haar wavelet transform approximation
        for i in 0..(length / 2) {
            let a = if idv_bit.get_bit((2 * i) as u64)? { 1.0 } else { 0.0 };
            let b = if idv_bit.get_bit((2 * i + 1) as u64)? { 1.0 } else { 0.0 };
            
            let low = (a + b) / 2.0;
            let high = (a - b) / 2.0;
            
            result.set(i, low > 0.5);
            result.set(i + length / 2, high > 0.5);
        }

        Ok(IDVBit::from_bitvec(result))
    }

    /// Hilbert transform
    fn hilbert_transform(&mut self, idv_bit: &IDVBit) -> IDVResult<IDVBit> {
        let sequence = self.to_complex_sequence(idv_bit, None)?;
        let fft_result = self.fft(&sequence)?;
        
        // Apply Hilbert transform in frequency domain
        let mut hilbert_freq = fft_result;
        let n = hilbert_freq.len();
        
        for i in 1..(n / 2) {
            hilbert_freq[i] = hilbert_freq[i] * Complex::new(0.0, -1.0);
        }
        for i in (n / 2 + 1)..n {
            hilbert_freq[i] = hilbert_freq[i] * Complex::new(0.0, 1.0);
        }
        
        let result = self.ifft(&hilbert_freq)?;
        self.from_complex_sequence(&result)
    }

    /// Custom kernel transform
    fn custom_kernel_transform(&self, idv_bit: &IDVBit, kernel: &[ComplexF64]) -> IDVResult<IDVBit> {
        // Convolve with custom kernel
        let kernel_idv = self.from_complex_sequence(kernel)?;
        let mut ops = IDVBitOperations::new();
        ops.convolve(idv_bit, &kernel_idv)
    }

    /// Reverse sequence
    fn reverse_sequence(&self, idv_bit: &IDVBit) -> IDVResult<IDVBit> {
        use bitvec::prelude::*;
        let length = self.get_effective_length(idv_bit)?;
        let mut result = bitvec![0; length];
        
        for i in 0..length {
            let bit = idv_bit.get_bit((length - 1 - i) as u64)?;
            result.set(i, bit);
        }

        Ok(IDVBit::from_bitvec(result))
    }

    /// Convert IDVBit to complex sequence
    fn to_complex_sequence(&self, idv_bit: &IDVBit, length: Option<usize>) -> IDVResult<Vec<ComplexF64>> {
        let len = length.unwrap_or_else(|| self.get_effective_length(idv_bit).unwrap_or(1000));
        let mut sequence = Vec::with_capacity(len);
        
        for i in 0..len {
            let bit = idv_bit.get_bit(i as u64)?;
            sequence.push(Complex::new(if bit { 1.0 } else { 0.0 }, 0.0));
        }

        Ok(sequence)
    }

    /// Convert complex sequence back to IDVBit
    fn from_complex_sequence(&self, sequence: &[ComplexF64]) -> IDVResult<IDVBit> {
        use bitvec::prelude::*;
        let mut result = bitvec![0; sequence.len()];
        
        for (i, &val) in sequence.iter().enumerate() {
            // Use magnitude and threshold at 0.5
            result.set(i, val.norm() > 0.5);
        }

        Ok(IDVBit::from_bitvec(result))
    }

    /// Fast Fourier Transform implementation
    fn fft(&mut self, input: &[ComplexF64]) -> IDVResult<Vec<ComplexF64>> {
        let n = input.len();
        if n <= 1 {
            return Ok(input.to_vec());
        }

        if !n.is_power_of_two() {
            return Err(IDVBitError::NumericalError(
                "FFT requires power-of-2 input size".to_string()
            ));
        }

        // Check cache
        if let Some(twiddles) = self.fft_cache.get(&n) {
            return self.fft_with_twiddles(input, twiddles);
        }

        // Compute twiddle factors
        let twiddles = self.compute_twiddle_factors(n);
        if self.fft_cache.len() < self.config.max_fft_cache_size {
            self.fft_cache.insert(n, twiddles.clone());
        }

        self.fft_with_twiddles(input, &twiddles)
    }

    /// FFT with precomputed twiddle factors
    fn fft_with_twiddles(&self, input: &[ComplexF64], twiddles: &[ComplexF64]) -> IDVResult<Vec<ComplexF64>> {
        let n = input.len();
        if n <= 1 {
            return Ok(input.to_vec());
        }

        // Cooley-Tukey FFT algorithm
        let mut output = input.to_vec();
        
        // Bit-reversal permutation
        for i in 0..n {
            let j = reverse_bits(i, (n as f64).log2() as usize);
            if i < j {
                output.swap(i, j);
            }
        }

        // FFT computation
        let mut size = 2;
        while size <= n {
            let half_size = size / 2;
            let step = n / size;
            
            for i in (0..n).step_by(size) {
                for j in 0..half_size {
                    let u = output[i + j];
                    let v = output[i + j + half_size] * twiddles[step * j];
                    
                    output[i + j] = u + v;
                    output[i + j + half_size] = u - v;
                }
            }
            
            size *= 2;
        }

        Ok(output)
    }

    /// Inverse FFT
    fn ifft(&mut self, input: &[ComplexF64]) -> IDVResult<Vec<ComplexF64>> {
        let n = input.len();
        
        // Conjugate input
        let conjugated: Vec<ComplexF64> = input.iter().map(|x| x.conj()).collect();
        
        // Apply forward FFT
        let mut result = self.fft(&conjugated)?;
        
        // Conjugate and scale result
        let scale = 1.0 / n as f64;
        for val in &mut result {
            *val = val.conj() * scale;
        }

        Ok(result)
    }

    /// Compute twiddle factors for FFT
    fn compute_twiddle_factors(&self, n: usize) -> Vec<ComplexF64> {
        (0..n)
            .map(|k| {
                let angle = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                Complex::new(angle.cos(), angle.sin())
            })
            .collect()
    }

    /// Polynomial operations

    /// Add two polynomials
    fn polynomial_add(&self, a: &[ComplexF64], b: &[ComplexF64]) -> Vec<ComplexF64> {
        let max_len = a.len().max(b.len());
        let mut result = vec![Complex::zero(); max_len];
        
        for i in 0..max_len {
            if i < a.len() {
                result[i] = result[i] + a[i];
            }
            if i < b.len() {
                result[i] = result[i] + b[i];
            }
        }

        result
    }

    /// Multiply two polynomials
    fn polynomial_multiply(&self, a: &[ComplexF64], b: &[ComplexF64]) -> IDVResult<Vec<ComplexF64>> {
        if a.is_empty() || b.is_empty() {
            return Ok(vec![Complex::zero()]);
        }

        let result_len = a.len() + b.len() - 1;
        let mut result = vec![Complex::zero(); result_len];
        
        for (i, &a_coeff) in a.iter().enumerate() {
            for (j, &b_coeff) in b.iter().enumerate() {
                result[i + j] = result[i + j] + a_coeff * b_coeff;
            }
        }

        Ok(result)
    }
}

/// Utility functions

/// Check if number is power of 2
fn next_power_of_2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut power = 1;
    while power < n {
        power *= 2;
    }
    power
}

/// Reverse bits for FFT bit-reversal permutation
fn reverse_bits(mut n: usize, bits: usize) -> usize {
    let mut reversed = 0;
    for _ in 0..bits {
        reversed = (reversed << 1) | (n & 1);
        n >>= 1;
    }
    reversed
}

impl Default for IDVBitOperations {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::prelude::*;

    #[test]
    fn test_bitwise_operations() {
        let ops = IDVBitOperations::new();
        let left = IDVBit::from_bitvec(bitvec![1, 0, 1, 0]);
        let right = IDVBit::from_bitvec(bitvec![1, 1, 0, 0]);
        
        let and_result = ops.bitwise_operation(&left, &right, BitwiseOperation::And).unwrap();
        let expected_and = bitvec![1, 0, 0, 0];
        assert_eq!(and_result.to_explicit(4).unwrap(), expected_and);
        
        let or_result = ops.bitwise_operation(&left, &right, BitwiseOperation::Or).unwrap();
        let expected_or = bitvec![1, 1, 1, 0];
        assert_eq!(or_result.to_explicit(4).unwrap(), expected_or);
    }

    #[test]
    fn test_algebraic_operations() {
        let ops = IDVBitOperations::new();
        
        // Test with generating functions
        let numerator1 = vec![Complex::new(1.0, 0.0)];
        let denominator1 = vec![Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)];
        let gf1 = IDVBit::from_generating_function(numerator1, denominator1).unwrap();
        
        let numerator2 = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        let denominator2 = vec![Complex::new(1.0, 0.0)];
        let gf2 = IDVBit::from_generating_function(numerator2, denominator2).unwrap();
        
        let add_result = ops.algebraic_operation(&gf1, &gf2, AlgebraicOperation::Add).unwrap();
        
        match &add_result.format {
            RepresentationFormat::GeneratingFunction { numerator, denominator } => {
                assert!(!numerator.is_empty());
                assert!(!denominator.is_empty());
            },
            _ => panic!("Expected generating function result"),
        }
    }

    #[test]
    fn test_convolution() {
        let mut ops = IDVBitOperations::new();
        let left = IDVBit::from_bitvec(bitvec![1, 0, 1]);
        let right = IDVBit::from_bitvec(bitvec![1, 1]);
        
        let conv_result = ops.convolve(&left, &right).unwrap();
        
        // Convolution of [1,0,1] with [1,1] should give [1,1,1,1]
        let expected_length = 3 + 2 - 1; // 4
        assert_eq!(conv_result.to_explicit(expected_length).unwrap().len(), expected_length);
    }

    #[test]
    fn test_sliding_window() {
        let ops = IDVBitOperations::new();
        let idv_bit = IDVBit::from_bitvec(bitvec![1, 1, 0, 1, 0, 0]);
        
        // Count ones in window
        let window_result = ops.sliding_window(&idv_bit, 3, |window| {
            window.iter().filter(|&&bit| bit).count() >= 2
        }).unwrap();
        
        // Should have 4 windows, checking which have 2+ ones
        assert_eq!(window_result.to_explicit(4).unwrap().len(), 4);
    }

    #[test]
    fn test_entropy() {
        let ops = IDVBitOperations::new();
        let uniform = IDVBit::from_bitvec(bitvec![0, 1, 0, 1, 0, 1, 0, 1]); // Alternating
        let constant = IDVBit::from_bitvec(bitvec![1, 1, 1, 1, 1, 1, 1, 1]); // All ones
        
        let uniform_entropy = ops.entropy(&uniform, 2).unwrap();
        let constant_entropy = ops.entropy(&constant, 2).unwrap();
        
        // For block size 2: alternating pattern has blocks [01, 01, 01, 01] -> 1 unique pattern
        // constant pattern has blocks [11, 11, 11, 11] -> 1 unique pattern
        // Both have entropy = 0. Let's use block size 1 instead for better discrimination
        let uniform_entropy_1 = ops.entropy(&uniform, 1).unwrap();
        let constant_entropy_1 = ops.entropy(&constant, 1).unwrap();
        
        // Block size 1: alternating has [0, 1, 0, 1, 0, 1, 0, 1] -> 2 patterns (50% each) -> entropy = 1
        // constant has [1, 1, 1, 1, 1, 1, 1, 1] -> 1 pattern (100%) -> entropy = 0
        assert!(uniform_entropy_1 > constant_entropy_1);
        assert_eq!(uniform_entropy_1, 1.0); // Maximum entropy for binary
        assert_eq!(constant_entropy_1, 0.0); // No entropy
    }

    #[test]
    fn test_fft_transform() {
        let mut ops = IDVBitOperations::new();
        let idv_bit = IDVBit::from_bitvec(bitvec![1, 0, 1, 0, 1, 0, 1, 0]); // 8 bits, power of 2
        
        let fft_result = ops.transform(&idv_bit, TransformOperation::FFT).unwrap();
        
        // Result should have same length as input
        assert_eq!(fft_result.to_explicit(8).unwrap().len(), 8);
    }

    #[test]
    fn test_lempel_ziv_complexity() {
        let ops = IDVBitOperations::new();
        let simple = IDVBit::from_bitvec(bitvec![1, 1, 1, 1]); // Low complexity
        let complex = IDVBit::from_bitvec(bitvec![1, 0, 1, 1, 0, 0, 1, 0]); // Higher complexity
        
        let simple_complexity = ops.lempel_ziv_complexity(&simple).unwrap();
        let complex_complexity = ops.lempel_ziv_complexity(&complex).unwrap();
        
        // Complex sequence should have higher LZ complexity
        assert!(complex_complexity >= simple_complexity);
    }

    #[test]
    fn test_fft_roundtrip() {
        let mut ops = IDVBitOperations::new();
        let input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        
        let fft_result = ops.fft(&input).unwrap();
        let ifft_result = ops.ifft(&fft_result).unwrap();
        
        // Should recover original input (within tolerance)
        for (original, recovered) in input.iter().zip(ifft_result.iter()) {
            assert!((original - recovered).norm() < 1e-10);
        }
    }

    #[test]
    fn test_polynomial_operations() {
        let ops = IDVBitOperations::new();
        let a = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]; // 1 + 2x
        let b = vec![Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)]; // 3 + 4x
        
        let sum = ops.polynomial_add(&a, &b); // Should be 4 + 6x
        assert_eq!(sum[0], Complex::new(4.0, 0.0));
        assert_eq!(sum[1], Complex::new(6.0, 0.0));
        
        let product = ops.polynomial_multiply(&a, &b).unwrap(); // Should be 3 + 10x + 8x²
        assert_eq!(product[0], Complex::new(3.0, 0.0));
        assert_eq!(product[1], Complex::new(10.0, 0.0));
        assert_eq!(product[2], Complex::new(8.0, 0.0));
    }
}
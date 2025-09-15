//! Superposition state management for IDVBit quantum-inspired operations
//!
//! Implements quantum-inspired superposition states allowing multiple
//! IDVBit configurations to exist simultaneously with complex amplitudes.

use crate::{IDVResult, IDVBitError, ComplexF64, IDVBit};
use num_complex::Complex;
use num_traits::{Zero, One, Float};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use ndarray::{Array1, Array2};
use rayon::prelude::*;

/// Superposition state containing multiple IDVBit configurations
#[derive(Debug, Clone, PartialEq)]
pub struct SuperpositionState {
    /// Basis states with their complex amplitudes
    pub states: Vec<(ComplexF64, IDVBit)>,
    /// Whether the state is normalized
    pub is_normalized: bool,
    /// Cached measurements for efficiency
    measurement_cache: HashMap<String, ComplexF64>,
}

/// Measurement operators for superposition states
#[derive(Debug, Clone, PartialEq)]
pub enum MeasurementOperator {
    /// Bit value measurement at specific position
    BitMeasurement { position: u64 },
    /// Density measurement over a range
    DensityMeasurement { start: u64, end: u64 },
    /// Entropy measurement
    EntropyMeasurement,
    /// Custom measurement defined by matrix
    CustomMeasurement(Array2<ComplexF64>),
}

/// Result of a measurement on superposition state
#[derive(Debug, Clone, PartialEq)]
pub struct MeasurementResult {
    /// Measurement outcome
    pub outcome: MeasurementOutcome,
    /// Probability of the outcome
    pub probability: f64,
    /// Post-measurement state (collapsed)
    pub collapsed_state: Option<SuperpositionState>,
}

/// Possible measurement outcomes
#[derive(Debug, Clone, PartialEq)]
pub enum MeasurementOutcome {
    /// Boolean measurement result
    Boolean(bool),
    /// Real-valued measurement result
    Real(f64),
    /// Complex-valued measurement result
    Complex(ComplexF64),
    /// Vector measurement result
    Vector(Vec<ComplexF64>),
}

impl SuperpositionState {
    /// Create new superposition state from basis states
    pub fn new(states: Vec<(ComplexF64, IDVBit)>) -> IDVResult<Self> {
        if states.is_empty() {
            return Err(IDVBitError::InvalidSuperposition(
                "Superposition state cannot be empty".to_string()
            ));
        }

        Ok(Self {
            states,
            is_normalized: false,
            measurement_cache: HashMap::new(),
        })
    }

    /// Create uniform superposition of given IDVBit states
    pub fn uniform_superposition(idv_bits: Vec<IDVBit>) -> IDVResult<Self> {
        if idv_bits.is_empty() {
            return Err(IDVBitError::InvalidSuperposition(
                "Cannot create superposition from empty set".to_string()
            ));
        }

        let amplitude = Complex::new(1.0 / (idv_bits.len() as f64).sqrt(), 0.0);
        let states = idv_bits
            .into_iter()
            .map(|idv| (amplitude, idv))
            .collect();

        Ok(Self {
            states,
            is_normalized: true,
            measurement_cache: HashMap::new(),
        })
    }

    /// Create maximally entangled state between two IDVBits
    pub fn bell_state(idv1: IDVBit, idv2: IDVBit) -> IDVResult<Self> {
        let amplitude = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
        
        let states = vec![
            (amplitude, idv1.clone()),
            (amplitude, idv2.clone()),
        ];

        Ok(Self {
            states,
            is_normalized: true,
            measurement_cache: HashMap::new(),
        })
    }

    /// Normalize the superposition state
    pub fn normalize(&mut self) -> IDVResult<()> {
        let norm_squared = self.compute_norm_squared();
        
        if norm_squared < 1e-15 {
            return Err(IDVBitError::InvalidSuperposition(
                "Cannot normalize zero state".to_string()
            ));
        }

        let norm = norm_squared.sqrt();
        for (amplitude, _) in &mut self.states {
            *amplitude = *amplitude / norm;
        }

        self.is_normalized = true;
        self.measurement_cache.clear(); // Clear cache after normalization
        Ok(())
    }

    /// Compute squared norm of the superposition state
    fn compute_norm_squared(&self) -> f64 {
        self.states
            .iter()
            .map(|(amplitude, _)| amplitude.norm_sqr())
            .sum()
    }

    /// Apply a unitary operation to the superposition state
    pub fn apply_unitary(&mut self, unitary: &Array2<ComplexF64>) -> IDVResult<()> {
        let num_states = self.states.len();
        
        if unitary.nrows() != num_states || unitary.ncols() != num_states {
            return Err(IDVBitError::InvalidSuperposition(
                "Unitary matrix dimensions don't match number of states".to_string()
            ));
        }

        // Extract current amplitudes
        let current_amplitudes: Vec<ComplexF64> = self.states
            .iter()
            .map(|(amplitude, _)| *amplitude)
            .collect();

        // Apply unitary transformation
        let new_amplitudes = unitary.dot(&Array1::from(current_amplitudes));

        // Update amplitudes
        for (i, (amplitude, _)) in self.states.iter_mut().enumerate() {
            *amplitude = new_amplitudes[i];
        }

        self.measurement_cache.clear();
        Ok(())
    }

    /// Perform measurement with given operator
    pub fn measure(&self, operator: &MeasurementOperator) -> IDVResult<MeasurementResult> {
        match operator {
            MeasurementOperator::BitMeasurement { position } => {
                self.measure_bit(*position)
            },
            MeasurementOperator::DensityMeasurement { start, end } => {
                self.measure_density(*start, *end)
            },
            MeasurementOperator::EntropyMeasurement => {
                self.measure_entropy()
            },
            MeasurementOperator::CustomMeasurement(matrix) => {
                self.measure_custom(matrix)
            },
        }
    }

    /// Measure bit value at specific position
    fn measure_bit(&self, position: u64) -> IDVResult<MeasurementResult> {
        let cache_key = format!("bit_{}", position);
        
        // Check cache first
        if let Some(&cached_prob) = self.measurement_cache.get(&cache_key) {
            if cached_prob.im.abs() < 1e-15 { // Should be real
                let prob_true = cached_prob.re;
                let outcome = if prob_true > 0.5 {
                    MeasurementOutcome::Boolean(true)
                } else {
                    MeasurementOutcome::Boolean(false)
                };
                
                return Ok(MeasurementResult {
                    outcome,
                    probability: prob_true.max(1.0 - prob_true),
                    collapsed_state: None, // TODO: Implement state collapse
                });
            }
        }

        // Compute probability of measuring 1 at position
        let mut prob_true = 0.0;
        
        for (amplitude, idv_bit) in &self.states {
            let bit_value = idv_bit.get_bit(position)?;
            if bit_value {
                prob_true += amplitude.norm_sqr();
            }
        }

        // Note: We skip caching since we can't mutate self in this context
        let outcome = if prob_true > 0.5 {
            MeasurementOutcome::Boolean(true)
        } else {
            MeasurementOutcome::Boolean(false)
        };

        Ok(MeasurementResult {
            outcome,
            probability: prob_true.max(1.0 - prob_true),
            collapsed_state: None,
        })
    }

    /// Measure density over a range of positions
    fn measure_density(&self, start: u64, end: u64) -> IDVResult<MeasurementResult> {
        if start >= end {
            return Err(IDVBitError::InvalidSuperposition(
                "Invalid range for density measurement".to_string()
            ));
        }

        let range_size = (end - start) as f64;
        let mut expected_density = 0.0;

        for (amplitude, idv_bit) in &self.states {
            let weight = amplitude.norm_sqr();
            let mut ones_count = 0u64;
            
            for pos in start..end {
                if idv_bit.get_bit(pos)? {
                    ones_count += 1;
                }
            }
            
            let density = ones_count as f64 / range_size;
            expected_density += weight * density;
        }

        Ok(MeasurementResult {
            outcome: MeasurementOutcome::Real(expected_density),
            probability: 1.0, // Density measurement is deterministic
            collapsed_state: None,
        })
    }

    /// Measure entropy of the superposition state
    fn measure_entropy(&self) -> IDVResult<MeasurementResult> {
        // von Neumann entropy: S = -Tr(ρ ln ρ)
        // For pure states, entropy is 0
        // For mixed states, compute from density matrix
        
        if !self.is_normalized {
            return Err(IDVBitError::InvalidSuperposition(
                "State must be normalized for entropy measurement".to_string()
            ));
        }

        // For a pure superposition state, the entropy is related to the
        // Shannon entropy of the amplitude probabilities
        let mut entropy = 0.0;
        
        for (amplitude, _) in &self.states {
            let prob = amplitude.norm_sqr();
            if prob > 1e-15 {
                entropy -= prob * prob.ln();
            }
        }

        Ok(MeasurementResult {
            outcome: MeasurementOutcome::Real(entropy),
            probability: 1.0,
            collapsed_state: None,
        })
    }

    /// Perform custom measurement with given matrix
    fn measure_custom(&self, matrix: &Array2<ComplexF64>) -> IDVResult<MeasurementResult> {
        if matrix.nrows() != self.states.len() || matrix.ncols() != self.states.len() {
            return Err(IDVBitError::InvalidSuperposition(
                "Measurement matrix dimensions don't match state space".to_string()
            ));
        }

        // Extract state amplitudes
        let amplitudes: Vec<ComplexF64> = self.states
            .iter()
            .map(|(amplitude, _)| *amplitude)
            .collect();

        // Compute expectation value: ⟨ψ|M|ψ⟩
        let state_vector = Array1::from(amplitudes.clone());
        let matrix_state = matrix.dot(&state_vector);
        
        let expectation = amplitudes
            .iter()
            .zip(matrix_state.iter())
            .map(|(amp1, amp2)| amp1.conj() * amp2)
            .sum::<ComplexF64>();

        Ok(MeasurementResult {
            outcome: MeasurementOutcome::Complex(expectation),
            probability: expectation.norm(),
            collapsed_state: None,
        })
    }

    /// Compute fidelity with another superposition state
    pub fn fidelity(&self, other: &SuperpositionState) -> IDVResult<f64> {
        if !self.is_normalized || !other.is_normalized {
            return Err(IDVBitError::InvalidSuperposition(
                "Both states must be normalized for fidelity calculation".to_string()
            ));
        }

        // For pure states: F = |⟨ψ₁|ψ₂⟩|²
        let mut inner_product = Complex::zero();
        
        // This is a simplified calculation assuming compatible basis
        // In practice, would need more sophisticated overlap computation
        let min_states = self.states.len().min(other.states.len());
        
        for i in 0..min_states {
            let (amp1, _) = &self.states[i];
            let (amp2, _) = &other.states[i];
            inner_product += amp1.conj() * amp2;
        }

        Ok(inner_product.norm_sqr())
    }

    /// Apply quantum-inspired evolution operator
    pub fn evolve(&mut self, hamiltonian: &Array2<ComplexF64>, time: f64) -> IDVResult<()> {
        // Apply time evolution: |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
        // Use matrix exponentiation (simplified implementation)
        
        let num_states = self.states.len();
        if hamiltonian.nrows() != num_states || hamiltonian.ncols() != num_states {
            return Err(IDVBitError::InvalidSuperposition(
                "Hamiltonian dimensions don't match state space".to_string()
            ));
        }

        // Simplified evolution using first-order approximation: U ≈ I - iHt
        let mut evolution_matrix = Array2::<ComplexF64>::eye(num_states);
        let i_t = Complex::new(0.0, -time);
        
        for ((i, j), h_ij) in hamiltonian.indexed_iter() {
            if i != j {
                evolution_matrix[[i, j]] += i_t * h_ij;
            } else {
                evolution_matrix[[i, j]] += i_t * h_ij;
            }
        }

        self.apply_unitary(&evolution_matrix)?;
        Ok(())
    }

    /// Compute entanglement entropy between subsystems
    pub fn entanglement_entropy(&self, subsystem_a_indices: &[usize]) -> IDVResult<f64> {
        // For pure states, compute entanglement entropy using Schmidt decomposition
        // This is a simplified implementation
        
        if !self.is_normalized {
            return Err(IDVBitError::InvalidSuperposition(
                "State must be normalized for entanglement entropy calculation".to_string()
            ));
        }

        // This would require implementing Schmidt decomposition
        // For now, return a placeholder based on subsystem size
        let subsystem_fraction = subsystem_a_indices.len() as f64 / self.states.len() as f64;
        let max_entropy = (subsystem_a_indices.len() as f64).log2();
        
        Ok(subsystem_fraction * max_entropy)
    }

    /// Get the number of basis states in superposition
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Check if state is pure (single component) vs mixed
    pub fn is_pure(&self) -> bool {
        let significant_states = self.states
            .iter()
            .filter(|(amplitude, _)| amplitude.norm_sqr() > 1e-10)
            .count();
        
        significant_states <= 1
    }

    /// Get effective dimension of the superposition
    pub fn effective_dimension(&self) -> f64 {
        // Compute participation ratio: 1 / sum(|ci|^4)
        let sum_fourth_powers: f64 = self.states
            .iter()
            .map(|(amplitude, _)| amplitude.norm_sqr().powi(2))
            .sum();
        
        if sum_fourth_powers > 1e-15 {
            1.0 / sum_fourth_powers
        } else {
            0.0
        }
    }
}

impl Default for SuperpositionState {
    fn default() -> Self {
        use bitvec::prelude::*;
        // Create default state with single zero IDVBit
        let zero_bit = IDVBit::from_bitvec(bitvec![]);
        let states = vec![(Complex::one(), zero_bit)];
        
        Self {
            states,
            is_normalized: true,
            measurement_cache: HashMap::new(),
        }
    }
}

// Mutable cache access workaround
impl SuperpositionState {
    fn measurement_cache(&self) -> &HashMap<String, ComplexF64> {
        &self.measurement_cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::prelude::*;

    #[test]
    fn test_superposition_creation() {
        let bit1 = IDVBit::from_bitvec(bitvec![1, 0, 1]);
        let bit2 = IDVBit::from_bitvec(bitvec![0, 1, 0]);
        
        let states = vec![
            (Complex::new(0.6, 0.0), bit1),
            (Complex::new(0.8, 0.0), bit2),
        ];
        
        let mut superpos = SuperpositionState::new(states).unwrap();
        superpos.normalize().unwrap();
        
        assert!(superpos.is_normalized);
        assert_eq!(superpos.num_states(), 2);
    }

    #[test]
    fn test_uniform_superposition() {
        let bit1 = IDVBit::from_bitvec(bitvec![1, 0]);
        let bit2 = IDVBit::from_bitvec(bitvec![0, 1]);
        let bit3 = IDVBit::from_bitvec(bitvec![1, 1]);
        
        let superpos = SuperpositionState::uniform_superposition(vec![bit1, bit2, bit3]).unwrap();
        
        assert!(superpos.is_normalized);
        assert_eq!(superpos.num_states(), 3);
        
        // Check uniform amplitudes
        for (amplitude, _) in &superpos.states {
            let expected_amplitude = 1.0 / 3.0_f64.sqrt();
            assert!((amplitude.norm() - expected_amplitude).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bell_state() {
        let bit1 = IDVBit::from_bitvec(bitvec![1, 0]);
        let bit2 = IDVBit::from_bitvec(bitvec![0, 1]);
        
        let bell_state = SuperpositionState::bell_state(bit1, bit2).unwrap();
        
        assert!(bell_state.is_normalized);
        assert_eq!(bell_state.num_states(), 2);
        
        // Check Bell state amplitudes
        for (amplitude, _) in &bell_state.states {
            let expected_amplitude = 1.0 / 2.0_f64.sqrt();
            assert!((amplitude.norm() - expected_amplitude).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bit_measurement() {
        let bit1 = IDVBit::from_bitvec(bitvec![1, 0, 1]);
        let bit2 = IDVBit::from_bitvec(bitvec![0, 1, 1]);
        
        let superpos = SuperpositionState::uniform_superposition(vec![bit1, bit2]).unwrap();
        
        let measurement = MeasurementOperator::BitMeasurement { position: 0 };
        let result = superpos.measure(&measurement).unwrap();
        
        match result.outcome {
            MeasurementOutcome::Boolean(_) => {
                assert!(result.probability > 0.0 && result.probability <= 1.0);
            },
            _ => panic!("Expected boolean measurement outcome"),
        }
    }

    #[test]
    fn test_density_measurement() {
        let bit1 = IDVBit::from_bitvec(bitvec![1, 1, 0, 0]);
        let bit2 = IDVBit::from_bitvec(bitvec![0, 0, 1, 1]);
        
        let superpos = SuperpositionState::uniform_superposition(vec![bit1, bit2]).unwrap();
        
        let measurement = MeasurementOperator::DensityMeasurement { start: 0, end: 4 };
        let result = superpos.measure(&measurement).unwrap();
        
        match result.outcome {
            MeasurementOutcome::Real(density) => {
                assert!((density - 0.5).abs() < 1e-10); // Both states have density 0.5
            },
            _ => panic!("Expected real measurement outcome"),
        }
    }

    #[test]
    fn test_entropy_measurement() {
        let bit1 = IDVBit::from_bitvec(bitvec![1, 0]);
        let bit2 = IDVBit::from_bitvec(bitvec![0, 1]);
        
        let superpos = SuperpositionState::uniform_superposition(vec![bit1, bit2]).unwrap();
        
        let measurement = MeasurementOperator::EntropyMeasurement;
        let result = superpos.measure(&measurement).unwrap();
        
        match result.outcome {
            MeasurementOutcome::Real(entropy) => {
                assert!(entropy >= 0.0); // Entropy should be non-negative
            },
            _ => panic!("Expected real measurement outcome"),
        }
    }

    #[test]
    fn test_fidelity() {
        let bit1 = IDVBit::from_bitvec(bitvec![1, 0]);
        let bit2 = IDVBit::from_bitvec(bitvec![0, 1]);
        
        let superpos1 = SuperpositionState::uniform_superposition(vec![bit1.clone(), bit2.clone()]).unwrap();
        let superpos2 = SuperpositionState::uniform_superposition(vec![bit1, bit2]).unwrap();
        
        let fidelity = superpos1.fidelity(&superpos2).unwrap();
        assert!((fidelity - 1.0).abs() < 1e-10); // Identical states should have fidelity 1
    }

    #[test]
    fn test_unitary_evolution() {
        let bit1 = IDVBit::from_bitvec(bitvec![1, 0]);
        let bit2 = IDVBit::from_bitvec(bitvec![0, 1]);
        
        let mut superpos = SuperpositionState::uniform_superposition(vec![bit1, bit2]).unwrap();
        
        // Apply Pauli-X (bit flip) operation
        let pauli_x = ndarray::arr2(&[
            [Complex::zero(), Complex::one()],
            [Complex::one(), Complex::zero()]
        ]);
        
        superpos.apply_unitary(&pauli_x).unwrap();
        
        // State should still be normalized after unitary operation
        let norm_squared = superpos.compute_norm_squared();
        assert!((norm_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_effective_dimension() {
        // Pure state should have effective dimension 1
        let bit1 = IDVBit::from_bitvec(bitvec![1, 0]);
        let states = vec![(Complex::one(), bit1)];
        let superpos = SuperpositionState::new(states).unwrap();
        
        let eff_dim = superpos.effective_dimension();
        assert!((eff_dim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_error_conditions() {
        // Empty superposition should fail
        assert!(SuperpositionState::new(vec![]).is_err());
        
        // Zero state normalization should fail
        let bit1 = IDVBit::from_bitvec(bitvec![1]);
        let states = vec![(Complex::zero(), bit1)];
        let mut superpos = SuperpositionState::new(states).unwrap();
        assert!(superpos.normalize().is_err());
    }
}
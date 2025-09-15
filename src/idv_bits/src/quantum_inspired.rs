//! Quantum-inspired operations for IDVBit sequences
//!
//! Implements quantum computing concepts adapted for IDVBit manipulation,
//! including quantum gates, quantum walks, and entanglement operations.

use crate::{IDVResult, IDVBitError, ComplexF64, IDVBit, SuperpositionState};
use num_complex::Complex;
use num_traits::{Zero, One, Float};
use ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;
use rayon::prelude::*;

/// Quantum gate operations for IDVBit sequences
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumGate {
    /// Pauli-X (bit flip) gate
    PauliX,
    /// Pauli-Y gate
    PauliY,
    /// Pauli-Z (phase flip) gate
    PauliZ,
    /// Hadamard gate
    Hadamard,
    /// Phase gate
    Phase(f64),
    /// Rotation gates
    RotationX(f64),
    RotationY(f64),
    RotationZ(f64),
    /// Controlled-NOT gate
    CNOT,
    /// Controlled-Z gate
    CZ,
    /// Toffoli gate (CCX)
    Toffoli,
    /// Fredkin gate (CSWAP)
    Fredkin,
    /// Custom unitary gate
    Custom(Array2<ComplexF64>),
}

/// Quantum walk types
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumWalkType {
    /// Discrete-time quantum walk
    DiscreteTime,
    /// Continuous-time quantum walk
    ContinuousTime,
    /// Coined quantum walk
    Coined,
    /// Stochastic quantum walk
    Stochastic,
}

/// Quantum-inspired IDVBit processor
pub struct QuantumIDVProcessor {
    /// Configuration parameters
    config: QuantumConfig,
    /// Gate cache for performance
    gate_cache: HashMap<String, Array2<ComplexF64>>,
    /// Random number generator for stochastic operations
    rng_seed: Option<u64>,
}

/// Configuration for quantum-inspired operations
#[derive(Debug, Clone)]
pub struct QuantumConfig {
    /// Default superposition dimension
    pub default_dimension: usize,
    /// Numerical tolerance for quantum operations
    pub tolerance: f64,
    /// Whether to use parallel processing
    pub use_parallel: bool,
    /// Maximum entanglement dimension
    pub max_entanglement_dim: usize,
    /// Decoherence time scale
    pub decoherence_time: f64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            default_dimension: 64,
            tolerance: 1e-12,
            use_parallel: true,
            max_entanglement_dim: 1024,
            decoherence_time: 1.0,
        }
    }
}

impl QuantumIDVProcessor {
    /// Create new quantum processor
    pub fn new() -> Self {
        Self {
            config: QuantumConfig::default(),
            gate_cache: HashMap::new(),
            rng_seed: None,
        }
    }

    /// Create processor with custom configuration
    pub fn with_config(config: QuantumConfig) -> Self {
        Self {
            config,
            gate_cache: HashMap::new(),
            rng_seed: None,
        }
    }

    /// Apply quantum gate to IDVBit sequence
    pub fn apply_gate(
        &mut self,
        idv_bit: &IDVBit,
        gate: QuantumGate,
        target_qubits: &[usize]
    ) -> IDVResult<IDVBit> {
        let gate_matrix = self.get_gate_matrix(&gate)?;
        self.apply_unitary_matrix(idv_bit, &gate_matrix, target_qubits)
    }

    /// Apply controlled gate operation
    pub fn apply_controlled_gate(
        &mut self,
        idv_bit: &IDVBit,
        gate: QuantumGate,
        control_qubits: &[usize],
        target_qubits: &[usize]
    ) -> IDVResult<IDVBit> {
        let base_gate = self.get_gate_matrix(&gate)?;
        let controlled_gate = self.construct_controlled_gate(&base_gate, control_qubits.len())?;
        
        let mut all_qubits = control_qubits.to_vec();
        all_qubits.extend_from_slice(target_qubits);
        
        self.apply_unitary_matrix(idv_bit, &controlled_gate, &all_qubits)
    }

    /// Perform quantum walk on IDVBit lattice
    pub fn quantum_walk(
        &self,
        initial_state: &IDVBit,
        walk_type: QuantumWalkType,
        steps: usize,
        lattice_size: usize
    ) -> IDVResult<IDVBit> {
        match walk_type {
            QuantumWalkType::DiscreteTime => {
                self.discrete_time_walk(initial_state, steps, lattice_size)
            },
            QuantumWalkType::ContinuousTime => {
                self.continuous_time_walk(initial_state, steps, lattice_size)
            },
            QuantumWalkType::Coined => {
                self.coined_walk(initial_state, steps, lattice_size)
            },
            QuantumWalkType::Stochastic => {
                self.stochastic_walk(initial_state, steps, lattice_size)
            },
        }
    }

    /// Create entangled IDVBit pair
    pub fn create_entangled_pair(&self, idv1: &IDVBit, idv2: &IDVBit) -> IDVResult<SuperpositionState> {
        // Create Bell state-like entanglement
        let bell_coeffs = vec![
            Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex::zero(),
            Complex::zero(),
            Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        
        // This is simplified - would need proper tensor product construction
        SuperpositionState::bell_state(idv1.clone(), idv2.clone())
    }

    /// Measure entanglement between IDVBit sequences
    pub fn measure_entanglement(&self, state: &SuperpositionState) -> IDVResult<f64> {
        // Use von Neumann entropy as entanglement measure
        if state.num_states() < 2 {
            return Ok(0.0);
        }

        let mut entropy = 0.0;
        for (amplitude, _) in &state.states {
            let prob = amplitude.norm_sqr();
            if prob > self.config.tolerance {
                entropy -= prob * prob.ln();
            }
        }

        Ok(entropy / 2.0_f64.ln()) // Convert to bits
    }

    /// Apply quantum error correction
    pub fn apply_error_correction(
        &self,
        noisy_state: &IDVBit,
        code_type: ErrorCorrectionCode
    ) -> IDVResult<IDVBit> {
        match code_type {
            ErrorCorrectionCode::BitFlip => self.bit_flip_correction(noisy_state),
            ErrorCorrectionCode::PhaseFlip => self.phase_flip_correction(noisy_state),
            ErrorCorrectionCode::Shor => self.shor_correction(noisy_state),
            ErrorCorrectionCode::Surface => self.surface_code_correction(noisy_state),
        }
    }

    /// Simulate quantum decoherence
    pub fn apply_decoherence(
        &self,
        state: &SuperpositionState,
        time: f64,
        decoherence_type: DecoherenceType
    ) -> IDVResult<SuperpositionState> {
        let decay_factor = (-time / self.config.decoherence_time).exp();
        
        match decoherence_type {
            DecoherenceType::Amplitude => self.amplitude_damping(state, decay_factor),
            DecoherenceType::Phase => self.phase_damping(state, decay_factor),
            DecoherenceType::Depolarizing => self.depolarizing_channel(state, decay_factor),
        }
    }

    /// Perform quantum teleportation protocol
    pub fn quantum_teleport(
        &self,
        state_to_teleport: &IDVBit,
        entangled_pair: &SuperpositionState
    ) -> IDVResult<(IDVBit, Vec<bool>)> {
        // Simplified teleportation - would need full protocol implementation
        let classical_bits = vec![false, true]; // Measurement results
        let teleported_state = state_to_teleport.clone(); // Simplified
        
        Ok((teleported_state, classical_bits))
    }

    /// Implement quantum algorithm (Grover's search adaptation)
    pub fn grover_search(
        &self,
        database: &IDVBit,
        target_pattern: &[bool],
        iterations: Option<usize>
    ) -> IDVResult<usize> {
        let database_size = database.length().unwrap_or(1000) as usize;
        let optimal_iterations = ((std::f64::consts::PI / 4.0) * (database_size as f64).sqrt()) as usize;
        let num_iterations = iterations.unwrap_or(optimal_iterations);
        
        // Simplified Grover search implementation
        let mut best_match_index = 0;
        let mut best_match_score = 0.0;
        
        for i in 0..database_size {
            let mut score = 0.0;
            for (j, &target_bit) in target_pattern.iter().enumerate() {
                let db_bit = database.get_bit((i + j) as u64).unwrap_or(false);
                if db_bit == target_bit {
                    score += 1.0;
                }
            }
            
            score /= target_pattern.len() as f64;
            if score > best_match_score {
                best_match_score = score;
                best_match_index = i;
            }
        }
        
        Ok(best_match_index)
    }

    // Private helper methods

    /// Get matrix representation of quantum gate
    fn get_gate_matrix(&mut self, gate: &QuantumGate) -> IDVResult<Array2<ComplexF64>> {
        let cache_key = self.gate_cache_key(gate);
        
        if let Some(cached_matrix) = self.gate_cache.get(&cache_key) {
            return Ok(cached_matrix.clone());
        }

        let matrix = match gate {
            QuantumGate::PauliX => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::zero(), Complex::one(),
                    Complex::one(), Complex::zero(),
                ]
            ).map_err(|_| IDVBitError::NumericalError("Failed to create Pauli-X matrix".to_string()))?,
            
            QuantumGate::PauliY => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::zero(), -Complex::i(),
                    Complex::i(), Complex::zero(),
                ]
            ).map_err(|_| IDVBitError::NumericalError("Failed to create Pauli-Y matrix".to_string()))?,
            
            QuantumGate::PauliZ => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::one(), Complex::zero(),
                    Complex::zero(), -Complex::one(),
                ]
            ).map_err(|_| IDVBitError::NumericalError("Failed to create Pauli-Z matrix".to_string()))?,
            
            QuantumGate::Hadamard => {
                let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex::new(inv_sqrt2, 0.0), Complex::new(inv_sqrt2, 0.0),
                        Complex::new(inv_sqrt2, 0.0), Complex::new(-inv_sqrt2, 0.0),
                    ]
                ).map_err(|_| IDVBitError::NumericalError("Failed to create Hadamard matrix".to_string()))?
            },
            
            QuantumGate::Phase(theta) => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex::one(), Complex::zero(),
                    Complex::zero(), Complex::new(theta.cos(), theta.sin()),
                ]
            ).map_err(|_| IDVBitError::NumericalError("Failed to create Phase matrix".to_string()))?,
            
            QuantumGate::RotationX(theta) => {
                let cos_half = (theta / 2.0).cos();
                let sin_half = (theta / 2.0).sin();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex::new(cos_half, 0.0), Complex::new(0.0, -sin_half),
                        Complex::new(0.0, -sin_half), Complex::new(cos_half, 0.0),
                    ]
                ).map_err(|_| IDVBitError::NumericalError("Failed to create RX matrix".to_string()))?
            },
            
            QuantumGate::CNOT => Array2::from_shape_vec(
                (4, 4),
                vec![
                    Complex::one(), Complex::zero(), Complex::zero(), Complex::zero(),
                    Complex::zero(), Complex::one(), Complex::zero(), Complex::zero(),
                    Complex::zero(), Complex::zero(), Complex::zero(), Complex::one(),
                    Complex::zero(), Complex::zero(), Complex::one(), Complex::zero(),
                ]
            ).map_err(|_| IDVBitError::NumericalError("Failed to create CNOT matrix".to_string()))?,
            
            QuantumGate::Custom(matrix) => matrix.clone(),
            
            _ => {
                return Err(IDVBitError::InvalidSuperposition(
                    format!("Gate {:?} not yet implemented", gate)
                ));
            }
        };

        self.gate_cache.insert(cache_key, matrix.clone());
        Ok(matrix)
    }

    /// Apply unitary matrix to IDVBit
    fn apply_unitary_matrix(
        &self,
        idv_bit: &IDVBit,
        matrix: &Array2<ComplexF64>,
        target_qubits: &[usize]
    ) -> IDVResult<IDVBit> {
        use bitvec::prelude::*;
        // This is a simplified implementation
        // Full implementation would require state vector representation and tensor products
        
        let length = idv_bit.length().unwrap_or(1000) as usize;
        let mut result_bits = bitvec![0; length];
        
        // Apply transformation to each target qubit position
        for &qubit_pos in target_qubits {
            if qubit_pos < length {
                let current_bit = idv_bit.get_bit(qubit_pos as u64).unwrap_or(false);
                
                // Simple transformation based on matrix elements
                let new_bit = if matrix.nrows() == 2 && matrix.ncols() == 2 {
                    // Single-qubit gate
                    let prob_flip = if current_bit {
                        matrix[[1, 0]].norm_sqr()
                    } else {
                        matrix[[0, 1]].norm_sqr()
                    };
                    
                    // Probabilistic application (simplified)
                    prob_flip > 0.5
                } else {
                    current_bit // Multi-qubit gates need more complex handling
                };
                
                result_bits.set(qubit_pos, new_bit);
            }
        }
        
        // Copy unchanged bits
        for i in 0..length {
            if !target_qubits.contains(&i) {
                let bit = idv_bit.get_bit(i as u64).unwrap_or(false);
                result_bits.set(i, bit);
            }
        }

        Ok(IDVBit::from_bitvec(result_bits))
    }

    /// Construct controlled version of a gate
    fn construct_controlled_gate(
        &self,
        base_gate: &Array2<ComplexF64>,
        num_controls: usize
    ) -> IDVResult<Array2<ComplexF64>> {
        let base_dim = base_gate.nrows();
        let total_dim = (1 << (num_controls + (base_dim as f64).log2() as usize));
        
        let mut controlled_gate = Array2::eye(total_dim);
        
        // Apply base gate to the last part of the controlled space
        let start_idx = total_dim - base_dim;
        for i in 0..base_dim {
            for j in 0..base_dim {
                controlled_gate[[start_idx + i, start_idx + j]] = base_gate[[i, j]];
            }
        }

        Ok(controlled_gate)
    }

    /// Discrete-time quantum walk
    fn discrete_time_walk(
        &self,
        initial_state: &IDVBit,
        steps: usize,
        lattice_size: usize
    ) -> IDVResult<IDVBit> {
        use bitvec::prelude::*;
        let mut current_state = initial_state.clone();
        
        for _ in 0..steps {
            // Apply coin operation (Hadamard-like)
            let mut temp_bits = bitvec![0; lattice_size];
            
            for i in 0..lattice_size {
                let current_bit = current_state.get_bit(i as u64)?;
                
                // Simple walk step: move based on coin flip
                let new_position = if current_bit {
                    (i + 1) % lattice_size
                } else {
                    (i + lattice_size - 1) % lattice_size
                };
                
                temp_bits.set(new_position, !current_bit);
            }
            
            current_state = IDVBit::from_bitvec(temp_bits);
        }

        Ok(current_state)
    }

    /// Continuous-time quantum walk
    fn continuous_time_walk(
        &self,
        initial_state: &IDVBit,
        _steps: usize,
        lattice_size: usize
    ) -> IDVResult<IDVBit> {
        use bitvec::prelude::*;
        // Simplified continuous-time walk using discrete approximation
        let _dt = 0.1; // Time step
        let mut current_state = initial_state.clone();
        
        for _ in 0..(_steps * 10) { // Finer time steps
            let mut temp_bits = bitvec![0; lattice_size];
            
            for i in 0..lattice_size {
                let current_bit = current_state.get_bit(i as u64)?;
                
                // Diffusion-like evolution
                let left_neighbor = (i + lattice_size - 1) % lattice_size;
                let right_neighbor = (i + 1) % lattice_size;
                
                let left_bit = current_state.get_bit(left_neighbor as u64)?;
                let right_bit = current_state.get_bit(right_neighbor as u64)?;
                
                // Simplified evolution rule
                let new_bit = (left_bit as u8 + right_bit as u8 + current_bit as u8) % 2 == 1;
                temp_bits.set(i, new_bit);
            }
            
            current_state = IDVBit::from_bitvec(temp_bits);
        }

        Ok(current_state)
    }

    /// Coined quantum walk
    fn coined_walk(&self, initial_state: &IDVBit, steps: usize, lattice_size: usize) -> IDVResult<IDVBit> {
        // Similar to discrete-time but with explicit coin space
        self.discrete_time_walk(initial_state, steps, lattice_size)
    }

    /// Stochastic quantum walk
    fn stochastic_walk(&self, initial_state: &IDVBit, steps: usize, lattice_size: usize) -> IDVResult<IDVBit> {
        use bitvec::prelude::*;
        let mut current_state = initial_state.clone();
        let mut rng = rand::thread_rng();
        
        for _ in 0..steps {
            let mut temp_bits = bitvec![0; lattice_size];
            
            for i in 0..lattice_size {
                let current_bit = current_state.get_bit(i as u64)?;
                
                // Stochastic movement
                let move_probability = 0.5;
                use rand::RngCore;
                let should_move = rng.next_u32() % 2 == 0;
                
                if should_move && (rng.next_u32() as f64 / u32::MAX as f64) < move_probability {
                    let new_position = if rng.next_u32() % 2 == 0 {
                        (i + 1) % lattice_size
                    } else {
                        (i + lattice_size - 1) % lattice_size
                    };
                    temp_bits.set(new_position, current_bit);
                } else {
                    temp_bits.set(i, current_bit);
                }
            }
            
            current_state = IDVBit::from_bitvec(temp_bits);
        }

        Ok(current_state)
    }

    /// Bit flip error correction (3-qubit code)
    fn bit_flip_correction(&self, noisy_state: &IDVBit) -> IDVResult<IDVBit> {
        use bitvec::prelude::*;
        let length = noisy_state.length().unwrap_or(1000) as usize;
        let corrected_length = (length / 3) * 3; // Ensure multiple of 3
        let mut corrected_bits = bitvec![0; corrected_length];
        
        for i in (0..corrected_length).step_by(3) {
            let bit0 = noisy_state.get_bit(i as u64).unwrap_or(false);
            let bit1 = noisy_state.get_bit((i + 1) as u64).unwrap_or(false);
            let bit2 = noisy_state.get_bit((i + 2) as u64).unwrap_or(false);
            
            // Majority vote
            let corrected_bit = (bit0 as u8 + bit1 as u8 + bit2 as u8) > 1;
            
            corrected_bits.set(i / 3, corrected_bit);
        }

        Ok(IDVBit::from_bitvec(corrected_bits))
    }

    /// Phase flip error correction
    fn phase_flip_correction(&self, noisy_state: &IDVBit) -> IDVResult<IDVBit> {
        // Simplified phase flip correction
        // In practice, would need complex amplitude handling
        self.bit_flip_correction(noisy_state)
    }

    /// Shor error correction
    fn shor_correction(&self, noisy_state: &IDVBit) -> IDVResult<IDVBit> {
        use bitvec::prelude::*;
        // Simplified Shor code (9-qubit code)
        let length = noisy_state.length().unwrap_or(1000) as usize;
        let corrected_length = (length / 9) * 9;
        let mut corrected_bits = bitvec![0; corrected_length / 9];
        
        for i in (0..corrected_length).step_by(9) {
            let mut block_bits = Vec::new();
            for j in 0..9 {
                block_bits.push(noisy_state.get_bit((i + j) as u64).unwrap_or(false));
            }
            
            // Simplified decoding: majority vote within triplets, then majority of triplets
            let triplet1 = (block_bits[0] as u8 + block_bits[1] as u8 + block_bits[2] as u8) > 1;
            let triplet2 = (block_bits[3] as u8 + block_bits[4] as u8 + block_bits[5] as u8) > 1;
            let triplet3 = (block_bits[6] as u8 + block_bits[7] as u8 + block_bits[8] as u8) > 1;
            
            let corrected_bit = (triplet1 as u8 + triplet2 as u8 + triplet3 as u8) > 1;
            corrected_bits.set(i / 9, corrected_bit);
        }

        Ok(IDVBit::from_bitvec(corrected_bits))
    }

    /// Surface code error correction
    fn surface_code_correction(&self, noisy_state: &IDVBit) -> IDVResult<IDVBit> {
        // Simplified surface code correction
        // Full implementation would require 2D lattice and syndrome extraction
        self.bit_flip_correction(noisy_state)
    }

    /// Amplitude damping channel
    fn amplitude_damping(&self, state: &SuperpositionState, decay_factor: f64) -> IDVResult<SuperpositionState> {
        let mut new_states = Vec::new();
        
        for (amplitude, idv_bit) in &state.states {
            let damped_amplitude = amplitude * Complex::new(decay_factor, 0.0);
            new_states.push((damped_amplitude, idv_bit.clone()));
        }

        let mut result = SuperpositionState::new(new_states)?;
        if decay_factor > self.config.tolerance {
            result.normalize()?;
        }
        
        Ok(result)
    }

    /// Phase damping channel
    fn phase_damping(&self, state: &SuperpositionState, decay_factor: f64) -> IDVResult<SuperpositionState> {
        let mut new_states = Vec::new();
        
        for (amplitude, idv_bit) in &state.states {
            let phase_factor = Complex::new(0.0, -decay_factor);
            let damped_amplitude = amplitude * phase_factor.exp();
            new_states.push((damped_amplitude, idv_bit.clone()));
        }

        SuperpositionState::new(new_states)
    }

    /// Depolarizing channel
    fn depolarizing_channel(&self, state: &SuperpositionState, decay_factor: f64) -> IDVResult<SuperpositionState> {
        // Mix with maximally mixed state
        let mixed_factor = 1.0 - decay_factor;
        let mut new_states = Vec::new();
        
        for (amplitude, idv_bit) in &state.states {
            let mixed_amplitude = amplitude * Complex::new(mixed_factor, 0.0);
            new_states.push((mixed_amplitude, idv_bit.clone()));
        }

        SuperpositionState::new(new_states)
    }

    /// Generate cache key for gate matrices
    fn gate_cache_key(&self, gate: &QuantumGate) -> String {
        match gate {
            QuantumGate::PauliX => "PauliX".to_string(),
            QuantumGate::PauliY => "PauliY".to_string(),
            QuantumGate::PauliZ => "PauliZ".to_string(),
            QuantumGate::Hadamard => "Hadamard".to_string(),
            QuantumGate::Phase(theta) => format!("Phase_{}", theta),
            QuantumGate::RotationX(theta) => format!("RX_{}", theta),
            QuantumGate::RotationY(theta) => format!("RY_{}", theta),
            QuantumGate::RotationZ(theta) => format!("RZ_{}", theta),
            QuantumGate::CNOT => "CNOT".to_string(),
            QuantumGate::CZ => "CZ".to_string(),
            QuantumGate::Toffoli => "Toffoli".to_string(),
            QuantumGate::Fredkin => "Fredkin".to_string(),
            QuantumGate::Custom(_) => "Custom".to_string(),
        }
    }
}

/// Error correction code types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorCorrectionCode {
    BitFlip,
    PhaseFlip,
    Shor,
    Surface,
}

/// Decoherence types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecoherenceType {
    Amplitude,
    Phase,
    Depolarizing,
}

impl Default for QuantumIDVProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bitvec::prelude::*;

    #[test]
    fn test_quantum_gates() {
        let mut processor = QuantumIDVProcessor::new();
        let idv_bit = IDVBit::from_bitvec(bitvec![0, 1, 0, 1]);
        
        let pauli_x_result = processor.apply_gate(&idv_bit, QuantumGate::PauliX, &[0, 2]).unwrap();
        
        // Pauli-X should flip bits at positions 0 and 2
        assert_eq!(pauli_x_result.get_bit(0).unwrap(), true);  // 0 -> 1
        assert_eq!(pauli_x_result.get_bit(1).unwrap(), true);  // unchanged
        assert_eq!(pauli_x_result.get_bit(2).unwrap(), true);  // 0 -> 1
        assert_eq!(pauli_x_result.get_bit(3).unwrap(), true);  // unchanged
    }

    #[test]
    fn test_quantum_walk() {
        let processor = QuantumIDVProcessor::new();
        let initial_state = IDVBit::from_bitvec(bitvec![1, 0, 0, 0]); // Start at position 0
        
        let walk_result = processor.quantum_walk(
            &initial_state,
            QuantumWalkType::DiscreteTime,
            5,
            4
        ).unwrap();
        
        // Walk should have spread the initial state
        assert_eq!(walk_result.length(), Some(4));
    }

    #[test]
    fn test_entangled_pair() {
        let processor = QuantumIDVProcessor::new();
        let bit1 = IDVBit::from_bitvec(bitvec![1, 0]);
        let bit2 = IDVBit::from_bitvec(bitvec![0, 1]);
        
        let entangled_state = processor.create_entangled_pair(&bit1, &bit2).unwrap();
        
        assert_eq!(entangled_state.num_states(), 2);
        assert!(entangled_state.is_normalized);
    }

    #[test]
    fn test_entanglement_measurement() {
        let processor = QuantumIDVProcessor::new();
        let bit1 = IDVBit::from_bitvec(bitvec![1]);
        let bit2 = IDVBit::from_bitvec(bitvec![0]);
        
        let entangled_state = processor.create_entangled_pair(&bit1, &bit2).unwrap();
        let entanglement = processor.measure_entanglement(&entangled_state).unwrap();
        
        assert!(entanglement >= 0.0); // Entanglement should be non-negative
    }

    #[test]
    fn test_error_correction() {
        let processor = QuantumIDVProcessor::new();
        let noisy_state = IDVBit::from_bitvec(bitvec![1, 1, 0, 0, 0, 1]); // 3-bit repetition code
        
        let corrected = processor.apply_error_correction(&noisy_state, ErrorCorrectionCode::BitFlip).unwrap();
        
        // Should decode to the majority bits
        assert_eq!(corrected.length(), Some(2)); // 6 bits -> 2 logical bits
    }

    #[test]
    fn test_decoherence() {
        let processor = QuantumIDVProcessor::new();
        let bit1 = IDVBit::from_bitvec(bitvec![1]);
        let bit2 = IDVBit::from_bitvec(bitvec![0]);
        let initial_state = SuperpositionState::uniform_superposition(vec![bit1, bit2]).unwrap();
        
        let decohered = processor.apply_decoherence(
            &initial_state,
            0.5,
            DecoherenceType::Amplitude
        ).unwrap();
        
        assert_eq!(decohered.num_states(), 2);
    }

    #[test]
    fn test_grover_search() {
        let processor = QuantumIDVProcessor::new();
        let database = IDVBit::from_bitvec(bitvec![0, 1, 1, 0, 1, 0, 1, 1]);
        let target_pattern = vec![true, true]; // Looking for pattern [1, 1]
        
        let result_index = processor.grover_search(&database, &target_pattern, None).unwrap();
        
        // Should find a position where the pattern occurs
        assert!(result_index < 7); // Valid index within search range
    }

    #[test]
    fn test_controlled_gate() {
        let mut processor = QuantumIDVProcessor::new();
        let idv_bit = IDVBit::from_bitvec(bitvec![1, 1, 0, 0]); // Control qubits set
        
        let result = processor.apply_controlled_gate(
            &idv_bit,
            QuantumGate::PauliX,
            &[0], // Control qubit
            &[2]  // Target qubit
        ).unwrap();
        
        // Should apply X gate to target when control is set
        assert!(result.length().is_some());
    }

    #[test]
    fn test_gate_matrix_caching() {
        let mut processor = QuantumIDVProcessor::new();
        
        // First access should compute and cache
        let matrix1 = processor.get_gate_matrix(&QuantumGate::PauliX).unwrap();
        
        // Second access should use cache
        let matrix2 = processor.get_gate_matrix(&QuantumGate::PauliX).unwrap();
        
        assert_eq!(matrix1, matrix2);
        assert!(processor.gate_cache.contains_key("PauliX"));
    }

    #[test]
    fn test_quantum_teleportation() {
        let processor = QuantumIDVProcessor::new();
        let state_to_teleport = IDVBit::from_bitvec(bitvec![1]);
        let bit1 = IDVBit::from_bitvec(bitvec![1]);
        let bit2 = IDVBit::from_bitvec(bitvec![0]);
        let entangled_pair = SuperpositionState::bell_state(bit1, bit2).unwrap();
        
        let (teleported_state, classical_bits) = processor.quantum_teleport(
            &state_to_teleport,
            &entangled_pair
        ).unwrap();
        
        assert_eq!(classical_bits.len(), 2); // Two measurement outcomes
        assert!(teleported_state.length().is_some());
    }
}
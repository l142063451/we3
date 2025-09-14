//! # Tensor Networks Library
//!
//! High-performance tensor network implementations with Tensor-Train (TT),
//! Matrix Product State (MPS), and Projected Entangled Pair State (PEPS) decompositions.
//!
//! This library provides mathematically rigorous low-rank tensor decompositions
//! with provable compression bounds for quantum and combinatorial applications.

use ndarray::{Array, IxDyn};
use num::Complex;

pub mod tensor_train;
pub mod matrix_product_state;
pub mod contraction;

/// Represents a high-dimensional tensor with efficient storage
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Tensor data stored as n-dimensional array
    data: Array<Complex<f64>, IxDyn>,
    /// Tensor dimensions
    shape: Vec<usize>,
}

impl Tensor {
    /// Creates a new tensor with given shape
    pub fn new(shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        let data = Array::zeros(IxDyn(&shape));
        
        Self { data, shape }
    }
    
    /// Gets tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Gets tensor element at given indices
    pub fn get(&self, indices: &[usize]) -> Option<Complex<f64>> {
        self.data.get(indices).copied()
    }
    
    /// Sets tensor element at given indices
    pub fn set(&mut self, indices: &[usize], value: Complex<f64>) -> Result<(), &'static str> {
        if let Some(elem) = self.data.get_mut(indices) {
            *elem = value;
            Ok(())
        } else {
            Err("Index out of bounds")
        }
    }
    
    /// Computes Frobenius norm
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(vec![2, 3, 4]);
        assert_eq!(tensor.shape(), &[2, 3, 4]);
    }
    
    #[test]
    fn test_tensor_access() {
        let mut tensor = Tensor::new(vec![2, 2]);
        let val = Complex::new(1.0, 0.5);
        
        assert!(tensor.set(&[0, 1], val).is_ok());
        assert_eq!(tensor.get(&[0, 1]), Some(val));
    }
}
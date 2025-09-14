//! # Generating Functions Engine
//! 
//! High-performance implementation of generating functions for combinatorial analysis,
//! including rational and algebraic generating function extraction, coefficient computation,
//! and Lagrange inversion methods.
//!
//! This module provides mathematically rigorous implementations with provable complexity bounds.

use num::Complex;
use std::collections::HashMap;

pub mod rational;
pub mod algebraic;
pub mod coefficients;

/// Represents a formal power series as a generating function
#[derive(Debug, Clone)]
pub struct GeneratingFunction {
    /// Coefficients of the power series
    coefficients: HashMap<usize, Complex<f64>>,
    /// Maximum computed degree
    max_degree: usize,
}

impl GeneratingFunction {
    /// Creates a new generating function
    pub fn new() -> Self {
        Self {
            coefficients: HashMap::new(),
            max_degree: 0,
        }
    }
    
    /// Sets the coefficient for x^n
    pub fn set_coefficient(&mut self, n: usize, coeff: Complex<f64>) {
        self.coefficients.insert(n, coeff);
        self.max_degree = self.max_degree.max(n);
    }
    
    /// Gets the coefficient for x^n
    pub fn coefficient(&self, n: usize) -> Complex<f64> {
        self.coefficients.get(&n).copied().unwrap_or(Complex::new(0.0, 0.0))
    }
}

impl Default for GeneratingFunction {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_operations() {
        let mut gf = GeneratingFunction::new();
        gf.set_coefficient(0, Complex::new(1.0, 0.0));
        gf.set_coefficient(1, Complex::new(1.0, 0.0));
        
        assert_eq!(gf.coefficient(0), Complex::new(1.0, 0.0));
        assert_eq!(gf.coefficient(1), Complex::new(1.0, 0.0));
        assert_eq!(gf.coefficient(2), Complex::new(0.0, 0.0));
    }
}
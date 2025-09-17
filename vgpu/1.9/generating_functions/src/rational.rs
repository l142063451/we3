//! Rational generating functions module

use crate::GeneratingFunction;
use num::Complex;

/// Represents a rational generating function P(x)/Q(x)
#[derive(Debug, Clone)]
pub struct RationalGF {
    /// Numerator polynomial coefficients
    pub numerator: Vec<Complex<f64>>,
    /// Denominator polynomial coefficients
    pub denominator: Vec<Complex<f64>>,
}

impl RationalGF {
    /// Creates a new rational generating function
    pub fn new(num: Vec<Complex<f64>>, den: Vec<Complex<f64>>) -> Self {
        Self {
            numerator: num,
            denominator: den,
        }
    }
    
    /// Extracts coefficient [x^n] using partial fractions decomposition
    pub fn coefficient(&self, n: usize) -> Complex<f64> {
        if self.denominator.is_empty() || self.denominator[0] == Complex::new(0.0, 0.0) {
            return Complex::new(0.0, 0.0);
        }
        
        // For rational P(x)/Q(x), use recurrence relation from denominator
        // Q(x) = q₀ + q₁x + q₂x² + ... 
        // If Q(x)·F(x) = P(x), then coefficients satisfy recurrence:
        // q₀·fₙ + q₁·fₙ₋₁ + q₂·fₙ₋₂ + ... = pₙ
        
        // Solve for coefficient fₙ = (pₙ - Σ qᵢ·fₙ₋ᵢ) / q₀
        
        // Initialize coefficient cache for dynamic programming
        let mut coeffs = vec![Complex::new(0.0, 0.0); n + 1];
        
        // Base cases from numerator
        for (i, &p_coeff) in self.numerator.iter().enumerate() {
            if i <= n {
                if i == 0 && self.denominator.len() > 0 {
                    coeffs[i] = p_coeff / self.denominator[0];
                } else if i > 0 {
                    let mut sum = Complex::new(0.0, 0.0);
                    for j in 1..self.denominator.len().min(i + 1) {
                        sum += self.denominator[j] * coeffs[i - j];
                    }
                    coeffs[i] = (p_coeff - sum) / self.denominator[0];
                }
            }
        }
        
        // Compute remaining coefficients using recurrence
        for i in self.numerator.len()..=n {
            let mut sum = Complex::new(0.0, 0.0);
            for j in 1..self.denominator.len().min(i + 1) {
                sum += self.denominator[j] * coeffs[i - j];
            }
            coeffs[i] = -sum / self.denominator[0];
        }
        
        coeffs[n]
    }
    
    /// Converts to general generating function representation
    pub fn to_generating_function(&self, max_terms: usize) -> GeneratingFunction {
        let mut gf = GeneratingFunction::new();
        for i in 0..max_terms {
            gf.set_coefficient(i, self.coefficient(i));
        }
        gf
    }
}
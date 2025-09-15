//! Density functions for infinite-density vector bit sequences
//!
//! Provides mathematical functions for modeling and computing density
//! distributions in IDVBit sequences using advanced statistical methods.

use crate::{IDVResult, IDVBitError, ComplexF64};
use num_complex::Complex;
use num_traits::{Zero, One, Float};
use std::f64::consts::PI;
use serde::{Serialize, Deserialize};

/// Density function interface for IDVBit sequences
pub trait DensityFunction: Send + Sync {
    /// Evaluate density at position x
    fn evaluate(&self, x: f64) -> f64;
    
    /// Compute cumulative distribution function
    fn cdf(&self, x: f64) -> f64;
    
    /// Get the normalization constant
    fn normalization(&self) -> f64;
    
    /// Sample from the density function
    fn sample(&self, rng: &mut dyn rand::RngCore) -> f64;
    
    /// Compute moments of the distribution
    fn moment(&self, order: usize) -> Option<f64>;
}

/// Gaussian density function for normally distributed information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaussianDensity {
    /// Mean parameter
    pub mean: f64,
    /// Standard deviation parameter
    pub std_dev: f64,
    /// Precomputed normalization constant
    normalization_const: f64,
}

impl GaussianDensity {
    /// Create new Gaussian density with given parameters
    pub fn new(mean: f64, std_dev: f64) -> IDVResult<Self> {
        if std_dev <= 0.0 {
            return Err(IDVBitError::InvalidDensity(
                "Standard deviation must be positive".to_string()
            ));
        }
        
        let normalization_const = 1.0 / (std_dev * (2.0 * PI).sqrt());
        
        Ok(Self {
            mean,
            std_dev,
            normalization_const,
        })
    }
    
    /// Standard normal distribution (mean=0, std_dev=1)
    pub fn standard() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
            normalization_const: 1.0 / (2.0 * PI).sqrt(),
        }
    }
}

impl DensityFunction for GaussianDensity {
    fn evaluate(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        self.normalization_const * (-0.5 * z * z).exp()
    }
    
    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / (self.std_dev * 2.0_f64.sqrt());
        0.5 * (1.0 + libm::erf(z))
    }
    
    fn normalization(&self) -> f64 {
        self.normalization_const
    }
    
    fn sample(&self, rng: &mut dyn rand::RngCore) -> f64 {
        // Box-Muller transform
        use rand::RngCore;
        
        let u1: f64 = rng.next_u32() as f64 / u32::MAX as f64;
        let u2: f64 = rng.next_u32() as f64 / u32::MAX as f64;
        
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        self.mean + self.std_dev * z0
    }
    
    fn moment(&self, order: usize) -> Option<f64> {
        match order {
            0 => Some(1.0),
            1 => Some(self.mean),
            2 => Some(self.mean * self.mean + self.std_dev * self.std_dev),
            _ => {
                // For higher moments, use the recursive formula for Gaussian moments
                // This is computationally intensive, so we limit to low orders
                if order <= 6 {
                    Some(self.compute_gaussian_moment(order))
                } else {
                    None
                }
            }
        }
    }
}

impl GaussianDensity {
    /// Compute higher-order moments using recursive formula
    fn compute_gaussian_moment(&self, order: usize) -> f64 {
        if order == 0 {
            return 1.0;
        }
        
        let sigma2 = self.std_dev * self.std_dev;
        let mu = self.mean;
        
        // Use the recursive relation for central moments
        // and convert to raw moments
        if order % 2 == 1 {
            // Odd moments
            mu.powi(order as i32)
        } else {
            // Even moments - use recursive formula
            let mut moment = 0.0;
            for k in (0..=order).step_by(2) {
                let binomial_coeff = binomial_coefficient(order, k);
                let central_moment = if k == 0 {
                    1.0
                } else {
                    double_factorial(k - 1) * sigma2.powi((k / 2) as i32)
                };
                moment += binomial_coeff as f64 * mu.powi((order - k) as i32) * central_moment;
            }
            moment
        }
    }
}

/// Exponential density function for exponentially decaying information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExponentialDensity {
    /// Rate parameter (lambda)
    pub rate: f64,
}

impl ExponentialDensity {
    /// Create new exponential density with given rate
    pub fn new(rate: f64) -> IDVResult<Self> {
        if rate <= 0.0 {
            return Err(IDVBitError::InvalidDensity(
                "Rate parameter must be positive".to_string()
            ));
        }
        
        Ok(Self { rate })
    }
}

impl DensityFunction for ExponentialDensity {
    fn evaluate(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.rate * (-self.rate * x).exp()
        }
    }
    
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0 - (-self.rate * x).exp()
        }
    }
    
    fn normalization(&self) -> f64 {
        self.rate
    }
    
    fn sample(&self, rng: &mut dyn rand::RngCore) -> f64 {
        use rand::RngCore;
        let u: f64 = rng.next_u32() as f64 / u32::MAX as f64;
        -u.ln() / self.rate
    }
    
    fn moment(&self, order: usize) -> Option<f64> {
        if order == 0 {
            Some(1.0)
        } else {
            // For exponential distribution, E[X^k] = k! / lambda^k
            Some(factorial(order) / self.rate.powi(order as i32))
        }
    }
}

/// Power-law density function for scale-free information distributions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerLawDensity {
    /// Scaling exponent (alpha)
    pub exponent: f64,
    /// Minimum value (x_min)
    pub x_min: f64,
    /// Precomputed normalization constant
    normalization_const: f64,
}

impl PowerLawDensity {
    /// Create new power-law density with given parameters
    pub fn new(exponent: f64, x_min: f64) -> IDVResult<Self> {
        if exponent <= 1.0 {
            return Err(IDVBitError::InvalidDensity(
                "Power law exponent must be greater than 1".to_string()
            ));
        }
        
        if x_min <= 0.0 {
            return Err(IDVBitError::InvalidDensity(
                "Minimum value must be positive".to_string()
            ));
        }
        
        let normalization_const = (exponent - 1.0) * x_min.powf(exponent - 1.0);
        
        Ok(Self {
            exponent,
            x_min,
            normalization_const,
        })
    }
}

impl DensityFunction for PowerLawDensity {
    fn evaluate(&self, x: f64) -> f64 {
        if x < self.x_min {
            0.0
        } else {
            self.normalization_const / x.powf(self.exponent)
        }
    }
    
    fn cdf(&self, x: f64) -> f64 {
        if x < self.x_min {
            0.0
        } else {
            1.0 - (self.x_min / x).powf(self.exponent - 1.0)
        }
    }
    
    fn normalization(&self) -> f64 {
        self.normalization_const
    }
    
    fn sample(&self, rng: &mut dyn rand::RngCore) -> f64 {
        use rand::RngCore;
        let u: f64 = rng.next_u32() as f64 / u32::MAX as f64;
        self.x_min * (1.0 - u).powf(-1.0 / (self.exponent - 1.0))
    }
    
    fn moment(&self, order: usize) -> Option<f64> {
        let k = order as f64;
        if k >= self.exponent - 1.0 {
            None // Moment doesn't exist
        } else {
            let numerator = (self.exponent - 1.0) * self.x_min.powf(k);
            let denominator = self.exponent - 1.0 - k;
            Some(numerator / denominator)
        }
    }
}

/// Composite density function combining multiple densities
pub struct CompositeDensity {
    /// Component densities with their weights
    components: Vec<(f64, Box<dyn DensityFunction>)>,
}

impl CompositeDensity {
    /// Create new composite density from weighted components
    pub fn new(components: Vec<(f64, Box<dyn DensityFunction>)>) -> IDVResult<Self> {
        if components.is_empty() {
            return Err(IDVBitError::InvalidDensity(
                "Composite density must have at least one component".to_string()
            ));
        }
        
        let total_weight: f64 = components.iter().map(|(w, _)| w).sum();
        if total_weight <= 0.0 {
            return Err(IDVBitError::InvalidDensity(
                "Total weight must be positive".to_string()
            ));
        }
        
        // Normalize weights
        let normalized_components = components
            .into_iter()
            .map(|(w, d)| (w / total_weight, d))
            .collect();
        
        Ok(Self {
            components: normalized_components,
        })
    }
}

impl DensityFunction for CompositeDensity {
    fn evaluate(&self, x: f64) -> f64 {
        self.components
            .iter()
            .map(|(weight, density)| weight * density.evaluate(x))
            .sum()
    }
    
    fn cdf(&self, x: f64) -> f64 {
        self.components
            .iter()
            .map(|(weight, density)| weight * density.cdf(x))
            .sum()
    }
    
    fn normalization(&self) -> f64 {
        1.0 // Already normalized during construction
    }
    
    fn sample(&self, rng: &mut dyn rand::RngCore) -> f64 {
        use rand::RngCore;
        
        // Choose component based on weights
        let u: f64 = rng.next_u32() as f64 / u32::MAX as f64;
        let mut cumulative_weight = 0.0;
        
        for (weight, density) in &self.components {
            cumulative_weight += weight;
            if u <= cumulative_weight {
                return density.sample(rng);
            }
        }
        
        // Fallback (shouldn't happen with proper normalization)
        self.components.last().unwrap().1.sample(rng)
    }
    
    fn moment(&self, order: usize) -> Option<f64> {
        let mut total_moment = 0.0;
        
        for (weight, density) in &self.components {
            if let Some(moment) = density.moment(order) {
                total_moment += weight * moment;
            } else {
                return None; // If any component moment doesn't exist
            }
        }
        
        Some(total_moment)
    }
}

/// Utility functions for mathematical computations

/// Compute factorial of n
fn factorial(n: usize) -> f64 {
    if n == 0 || n == 1 {
        1.0
    } else {
        (2..=n).map(|i| i as f64).product()
    }
}

/// Compute double factorial of n (n!! = n * (n-2) * (n-4) * ...)
fn double_factorial(n: usize) -> f64 {
    if n <= 1 {
        1.0
    } else {
        (1..=n).step_by(2).map(|i| i as f64).product()
    }
}

/// Compute binomial coefficient C(n, k)
fn binomial_coefficient(n: usize, k: usize) -> usize {
    if k > n {
        0
    } else if k == 0 || k == n {
        1
    } else {
        let mut result = 1;
        for i in 0..k.min(n - k) {
            result = result * (n - i) / (i + 1);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_gaussian_density() {
        let gaussian = GaussianDensity::standard();
        
        // Test evaluation at mean (should be maximum)
        let at_mean = gaussian.evaluate(0.0);
        let normalization = 1.0 / (2.0 * PI).sqrt();
        assert!((at_mean - normalization).abs() < 1e-10);
        
        // Test symmetry
        assert!((gaussian.evaluate(1.0) - gaussian.evaluate(-1.0)).abs() < 1e-10);
        
        // Test CDF at mean (should be 0.5)
        assert!((gaussian.cdf(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_density() {
        let exp_density = ExponentialDensity::new(2.0).unwrap();
        
        // Test evaluation at 0 (should be rate parameter)
        assert!((exp_density.evaluate(0.0) - 2.0).abs() < 1e-10);
        
        // Test that density is 0 for negative values
        assert_eq!(exp_density.evaluate(-1.0), 0.0);
        
        // Test CDF
        assert!((exp_density.cdf(0.0) - 0.0).abs() < 1e-10);
        assert!(exp_density.cdf(f64::INFINITY) > 0.999);
    }

    #[test]
    fn test_power_law_density() {
        let power_law = PowerLawDensity::new(2.5, 1.0).unwrap();
        
        // Test that density is 0 for x < x_min
        assert_eq!(power_law.evaluate(0.5), 0.0);
        
        // Test evaluation at x_min
        let at_x_min = power_law.evaluate(1.0);
        let expected = 1.5; // (exponent - 1) * x_min^(exponent - 1) = 1.5 * 1^1.5
        assert!((at_x_min - expected).abs() < 1e-10);
    }

    #[test]
    fn test_moments() {
        let gaussian = GaussianDensity::new(2.0, 3.0).unwrap();
        
        // Test basic moments
        assert!((gaussian.moment(0).unwrap() - 1.0).abs() < 1e-10);
        assert!((gaussian.moment(1).unwrap() - 2.0).abs() < 1e-10); // Mean
        
        let exp_density = ExponentialDensity::new(0.5).unwrap();
        assert!((exp_density.moment(1).unwrap() - 2.0).abs() < 1e-10); // 1/rate = 2
    }

    #[test]
    fn test_sampling() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let gaussian = GaussianDensity::standard();
        
        // Test that sampling produces reasonable values
        let samples: Vec<f64> = (0..1000).map(|_| gaussian.sample(&mut rng)).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        
        // Mean should be approximately 0 for standard normal
        assert!(mean.abs() < 0.2);
    }

    #[test]
    fn test_composite_density() {
        let gaussian = Box::new(GaussianDensity::standard()) as Box<dyn DensityFunction>;
        let exponential = Box::new(ExponentialDensity::new(1.0).unwrap()) as Box<dyn DensityFunction>;
        
        let composite = CompositeDensity::new(vec![
            (0.6, gaussian),
            (0.4, exponential),
        ]).unwrap();
        
        // Test that evaluation combines components
        let value = composite.evaluate(0.0);
        assert!(value > 0.0);
        
        // Test that CDF is valid
        assert!((composite.cdf(-f64::INFINITY) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_error_conditions() {
        // Invalid standard deviation
        assert!(GaussianDensity::new(0.0, -1.0).is_err());
        
        // Invalid rate parameter
        assert!(ExponentialDensity::new(0.0).is_err());
        
        // Invalid power law parameters
        assert!(PowerLawDensity::new(0.5, 1.0).is_err()); // exponent <= 1
        assert!(PowerLawDensity::new(2.0, -1.0).is_err()); // negative x_min
    }

    #[test]
    fn test_mathematical_utilities() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(5), 120.0);
        
        assert_eq!(double_factorial(0), 1.0);
        assert_eq!(double_factorial(5), 15.0); // 5 * 3 * 1
        
        assert_eq!(binomial_coefficient(5, 2), 10);
        assert_eq!(binomial_coefficient(5, 0), 1);
    }
}
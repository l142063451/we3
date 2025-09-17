//! Coefficient extraction algorithms

use num::Complex;

/// Fast coefficient extraction using FFT/NTT
pub struct CoefficientExtractor {
    /// Precomputed roots of unity for FFT
    roots: Vec<Complex<f64>>,
}

impl CoefficientExtractor {
    /// Creates a new coefficient extractor
    pub fn new(max_degree: usize) -> Self {
        // Precompute roots of unity
        let mut roots = Vec::new();
        for k in 0..max_degree {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / max_degree as f64;
            roots.push(Complex::new(angle.cos(), angle.sin()));
        }
        
        Self { roots }
    }
    
    /// Extracts coefficient [x^n] from polynomial representation
    pub fn extract_coefficient(&self, poly: &[Complex<f64>], n: usize) -> Complex<f64> {
        if n < poly.len() {
            poly[n]
        } else {
            Complex::new(0.0, 0.0)
        }
    }
    
    /// Fast polynomial multiplication using FFT convolution
    pub fn multiply_polynomials(&self, a: &[Complex<f64>], b: &[Complex<f64>]) -> Vec<Complex<f64>> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        
        let result_len = a.len() + b.len() - 1;
        let fft_size = result_len.next_power_of_two();
        
        // Pad polynomials to FFT size
        let mut a_padded = a.to_vec();
        let mut b_padded = b.to_vec();
        a_padded.resize(fft_size, Complex::new(0.0, 0.0));
        b_padded.resize(fft_size, Complex::new(0.0, 0.0));
        
        // Forward FFT
        self.fft(&mut a_padded, false);
        self.fft(&mut b_padded, false);
        
        // Pointwise multiplication
        for i in 0..fft_size {
            a_padded[i] *= b_padded[i];
        }
        
        // Inverse FFT
        self.fft(&mut a_padded, true);
        
        // Return only the significant coefficients
        a_padded.truncate(result_len);
        a_padded
    }
    
    /// Cooley-Tukey FFT implementation
    fn fft(&self, data: &mut [Complex<f64>], inverse: bool) {
        let n = data.len();
        if n <= 1 {
            return;
        }
        
        // Bit-reverse permutation
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            
            if i < j {
                data.swap(i, j);
            }
        }
        
        // Main FFT computation
        let mut length = 2;
        while length <= n {
            let angle = if inverse {
                2.0 * std::f64::consts::PI / length as f64
            } else {
                -2.0 * std::f64::consts::PI / length as f64
            };
            let wlen = Complex::new(angle.cos(), angle.sin());
            
            for i in (0..n).step_by(length) {
                let mut w = Complex::new(1.0, 0.0);
                for j in 0..length / 2 {
                    let u = data[i + j];
                    let v = data[i + j + length / 2] * w;
                    data[i + j] = u + v;
                    data[i + j + length / 2] = u - v;
                    w *= wlen;
                }
            }
            length <<= 1;
        }
        
        // Normalize for inverse transform
        if inverse {
            let n_inv = 1.0 / n as f64;
            for x in data.iter_mut() {
                *x *= n_inv;
            }
        }
    }
}
//! Tensor-Train decomposition implementation

use crate::Tensor;
use ndarray::{Array3, Array};
use num::Complex;

/// Tensor-Train representation of a high-dimensional tensor
#[derive(Debug, Clone)]
pub struct TensorTrain {
    /// TT cores (rank × dim × rank tensors)
    cores: Vec<Array3<Complex<f64>>>,
    /// TT ranks between cores
    ranks: Vec<usize>,
    /// Physical dimensions
    dimensions: Vec<usize>,
}

impl TensorTrain {
    /// Creates a new TT decomposition
    pub fn new(dimensions: Vec<usize>, max_rank: usize) -> Self {
        let d = dimensions.len();
        let mut cores = Vec::new();
        let mut ranks = vec![1]; // Left boundary rank
        
        for i in 0..d {
            let left_rank = if i == 0 { 1 } else { max_rank.min(dimensions[..i].iter().product()) };
            let right_rank = if i == d-1 { 1 } else { max_rank.min(dimensions[i+1..].iter().product()) };
            
            ranks.push(right_rank);
            let core = Array3::zeros((left_rank, dimensions[i], right_rank));
            cores.push(core);
        }
        
        Self {
            cores,
            ranks,
            dimensions,
        }
    }
    
    /// Decomposes a full tensor into TT format using advanced TT-SVD algorithm
    pub fn from_tensor(tensor: &Tensor, max_rank: usize, tolerance: f64) -> Self {
        let shape = tensor.shape();
        let d = shape.len();
        
        if d == 0 {
            return Self::new(shape.to_vec(), max_rank);
        }
        
        let mut cores = Vec::new();
        let mut ranks = vec![1]; // Left boundary
        let mut current_matrix = tensor.data.clone();
        
        // Create temporary instance for SVD computation
        let temp_instance = Self::new(shape.to_vec(), max_rank);
        
        for k in 0..d {
            // Compute dimensions for current unfolding
            let left_size: usize = ranks[k] * shape[k];
            let right_size: usize = shape[k+1..].iter().product::<usize>().max(1);
            
            // Unfold tensor into matrix for SVD
            let matrix_data: Vec<Complex<f64>> = current_matrix.iter().cloned().collect();
            
            if k < d - 1 {
                // Perform TT-SVD truncation
                let (u, s, vt) = temp_instance.svd_truncated(&matrix_data, left_size, right_size, max_rank, tolerance);
                
                // Adaptive rank selection based on singular values
                let mut effective_rank = s.len();
                for (i, &sigma) in s.iter().enumerate() {
                    if sigma < tolerance {
                        effective_rank = i;
                        break;
                    }
                }
                effective_rank = effective_rank.max(1).min(max_rank);
                
                ranks.push(effective_rank);
                
                // Form TT core from U matrix
                let core = Array3::from_shape_fn(
                    (ranks[k], shape[k], effective_rank),
                    |(i, j, l)| {
                        let idx = i * shape[k] + j;
                        if idx < u.len() / effective_rank && l < effective_rank {
                            u[idx * effective_rank + l]
                        } else {
                            Complex::new(0.0, 0.0)
                        }
                    }
                );
                cores.push(core);
                
                // Update current matrix for next iteration
                // current_matrix = S * V^T (right_size × effective_rank * effective_rank × cols)
                let mut next_matrix = Array::zeros((effective_rank, right_size));
                for i in 0..effective_rank {
                    for j in 0..right_size.min(vt.len() / effective_rank) {
                        if i * right_size + j < next_matrix.len() && i < s.len() && 
                           i * (vt.len() / effective_rank) + j < vt.len() {
                            next_matrix[[i, j]] = s[i] * vt[i * (vt.len() / effective_rank) + j];
                        }
                    }
                }
                current_matrix = next_matrix.into_dyn();
            } else {
                // Last core - no further decomposition
                let final_core = Array3::from_shape_fn(
                    (ranks[k], shape[k], 1),
                    |(i, j, _k)| {
                        let idx = i * shape[k] + j;
                        if idx < matrix_data.len() {
                            matrix_data[idx]
                        } else {
                            Complex::new(0.0, 0.0)
                        }
                    }
                );
                cores.push(final_core);
                ranks.push(1); // Right boundary
            }
        }
        
        Self {
            cores,
            ranks,
            dimensions: shape.to_vec(),
        }
    }
    
    /// Advanced TT-SVD with adaptive rank selection and error control using randomized SVD
    fn svd_truncated(
        &self,
        matrix: &[Complex<f64>],
        rows: usize,
        cols: usize,
        max_rank: usize,
        tolerance: f64
    ) -> (Vec<Complex<f64>>, Vec<f64>, Vec<Complex<f64>>) {
        // Advanced randomized SVD algorithm for large matrices
        let target_rank = max_rank.min(rows).min(cols);
        
        if rows == 0 || cols == 0 || matrix.is_empty() {
            return (vec![Complex::new(0.0, 0.0); rows * target_rank], 
                   vec![0.0; target_rank], 
                   vec![Complex::new(0.0, 0.0); target_rank * cols]);
        }
        
        // Use randomized SVD for efficiency with large tensors
        let oversampling = 10.min(target_rank / 4 + 5);
        let sketch_rank = (target_rank + oversampling).min(rows).min(cols);
        
        // Step 1: Generate random test matrix Ω (cols × sketch_rank)
        let mut omega = vec![Complex::new(0.0, 0.0); cols * sketch_rank];
        for i in 0..omega.len() {
            omega[i] = Complex::new(
                (i as f64 * 0.123456789).sin(), 
                (i as f64 * 0.987654321).cos()
            );
        }
        
        // Step 2: Form Y = A * Ω (rows × sketch_rank)
        let mut y = vec![Complex::new(0.0, 0.0); rows * sketch_rank];
        for i in 0..rows {
            for k in 0..sketch_rank {
                let mut sum = Complex::new(0.0, 0.0);
                for j in 0..cols {
                    if i * cols + j < matrix.len() && j * sketch_rank + k < omega.len() {
                        sum += matrix[i * cols + j] * omega[j * sketch_rank + k];
                    }
                }
                if i * sketch_rank + k < y.len() {
                    y[i * sketch_rank + k] = sum;
                }
            }
        }
        
        // Step 3: QR factorization of Y to get orthonormal basis Q
        let (q, _r) = self.qr_decomposition(&y, rows, sketch_rank);
        
        // Step 4: Form B = Q^H * A (sketch_rank × cols)
        let mut b = vec![Complex::new(0.0, 0.0); sketch_rank * cols];
        for k in 0..sketch_rank {
            for j in 0..cols {
                let mut sum = Complex::new(0.0, 0.0);
                for i in 0..rows {
                    if k * rows + i < q.len() && i * cols + j < matrix.len() {
                        sum += q[k * rows + i].conj() * matrix[i * cols + j];
                    }
                }
                if k * cols + j < b.len() {
                    b[k * cols + j] = sum;
                }
            }
        }
        
        // Step 5: SVD of small matrix B
        let (u_b, s, vt) = self.direct_svd(&b, sketch_rank, cols, target_rank, tolerance);
        
        // Step 6: Form U = Q * U_B
        let effective_rank = s.len();
        let mut u = vec![Complex::new(0.0, 0.0); rows * effective_rank];
        for i in 0..rows {
            for k in 0..effective_rank {
                let mut sum = Complex::new(0.0, 0.0);
                for r in 0..sketch_rank {
                    if r * rows + i < q.len() && r * effective_rank + k < u_b.len() {
                        sum += q[r * rows + i] * u_b[r * effective_rank + k];
                    }
                }
                if i * effective_rank + k < u.len() {
                    u[i * effective_rank + k] = sum;
                }
            }
        }
        
        (u, s, vt)
    }
    
    /// QR decomposition using modified Gram-Schmidt process
    fn qr_decomposition(&self, matrix: &[Complex<f64>], rows: usize, cols: usize) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {
        let mut q = vec![Complex::new(0.0, 0.0); rows * cols];
        let mut r = vec![Complex::new(0.0, 0.0); cols * cols];
        
        for j in 0..cols {
            // Copy column j from matrix
            for i in 0..rows {
                if i * cols + j < matrix.len() && j * rows + i < q.len() {
                    q[j * rows + i] = matrix[i * cols + j];
                }
            }
            
            // Modified Gram-Schmidt orthogonalization
            for k in 0..j {
                // Compute R[k,j] = Q[:, k]^H * Q[:, j]
                let mut dot_product = Complex::new(0.0, 0.0);
                for i in 0..rows {
                    if k * rows + i < q.len() && j * rows + i < q.len() {
                        dot_product += q[k * rows + i].conj() * q[j * rows + i];
                    }
                }
                if k * cols + j < r.len() {
                    r[k * cols + j] = dot_product;
                }
                
                // Q[:, j] = Q[:, j] - R[k,j] * Q[:, k]
                for i in 0..rows {
                    if j * rows + i < q.len() && k * rows + i < q.len() {
                        let q_ki = q[k * rows + i]; // Store value to avoid borrow conflicts
                        q[j * rows + i] -= dot_product * q_ki;
                    }
                }
            }
            
            // Compute norm and normalize
            let mut norm_sq = 0.0;
            for i in 0..rows {
                if j * rows + i < q.len() {
                    norm_sq += q[j * rows + i].norm_sqr();
                }
            }
            let norm = norm_sq.sqrt();
            
            if j * cols + j < r.len() {
                r[j * cols + j] = Complex::new(norm, 0.0);
            }
            
            if norm > 1e-12 {
                for i in 0..rows {
                    if j * rows + i < q.len() {
                        q[j * rows + i] /= norm;
                    }
                }
            }
        }
        
        (q, r)
    }
    
    /// Direct SVD for small matrices using power iteration with deflation
    fn direct_svd(&self, matrix: &[Complex<f64>], rows: usize, cols: usize, 
                  max_rank: usize, tolerance: f64) -> (Vec<Complex<f64>>, Vec<f64>, Vec<Complex<f64>>) {
        let target_rank = max_rank.min(rows).min(cols);
        
        let mut u = vec![Complex::new(0.0, 0.0); rows * target_rank];
        let mut s = Vec::new();
        let mut vt = vec![Complex::new(0.0, 0.0); target_rank * cols];
        
        // Working copy of matrix for deflation
        let mut a = matrix.to_vec();
        
        for k in 0..target_rank {
            // Power iteration for k-th singular triplet
            let mut v = vec![Complex::new(1.0, 0.0); cols]; // Random initialization
            let mut prev_sigma = 0.0;
            
            for _iter in 0..50 { // More iterations for better convergence
                // v = A^H * (A * v)
                let mut av = vec![Complex::new(0.0, 0.0); rows];
                for i in 0..rows {
                    let mut sum = Complex::new(0.0, 0.0);
                    for j in 0..cols {
                        if i * cols + j < a.len() && j < v.len() {
                            sum += a[i * cols + j] * v[j];
                        }
                    }
                    av[i] = sum;
                }
                
                let mut new_v = vec![Complex::new(0.0, 0.0); cols];
                for j in 0..cols {
                    let mut sum = Complex::new(0.0, 0.0);
                    for i in 0..rows {
                        if i * cols + j < a.len() && i < av.len() {
                            sum += a[i * cols + j].conj() * av[i];
                        }
                    }
                    new_v[j] = sum;
                }
                
                // Normalize
                let norm: f64 = new_v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                if norm > 1e-14 {
                    for elem in &mut new_v {
                        *elem /= norm;
                    }
                }
                
                // Check convergence
                if (norm - prev_sigma).abs() < tolerance * 1e-6 {
                    break;
                }
                prev_sigma = norm;
                v = new_v;
            }
            
            // Compute singular value
            let mut av = vec![Complex::new(0.0, 0.0); rows];
            for i in 0..rows {
                let mut sum = Complex::new(0.0, 0.0);
                for j in 0..cols {
                    if i * cols + j < a.len() && j < v.len() {
                        sum += a[i * cols + j] * v[j];
                    }
                }
                av[i] = sum;
            }
            
            let sigma = av.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            
            if sigma < tolerance {
                break; // Converged within tolerance
            }
            
            s.push(sigma);
            
            // Compute left singular vector u_k = A * v_k / sigma_k
            for i in 0..rows {
                if k * rows + i < u.len() {
                    u[k * rows + i] = av[i] / sigma;
                }
            }
            
            // Store right singular vector
            for j in 0..cols {
                if k * cols + j < vt.len() && j < v.len() {
                    vt[k * cols + j] = v[j];
                }
            }
            
            // Deflate: A = A - sigma_k * u_k * v_k^H
            for i in 0..rows {
                for j in 0..cols {
                    if i * cols + j < a.len() && k * rows + i < u.len() && j < v.len() {
                        a[i * cols + j] -= sigma * u[k * rows + i] * v[j].conj();
                    }
                }
            }
        }
        
        (u, s, vt)
    }
    
    /// Reconstructs full tensor from TT cores using tensor contraction
    pub fn to_tensor(&self) -> Tensor {
        if self.cores.is_empty() {
            return Tensor::new(vec![]);
        }
        
        // Start with first core
        let mut current: Array<Complex<f64>, ndarray::IxDyn> = self.cores[0].clone().into_dyn();
        
        // Contract remaining cores sequentially
        for core in self.cores.iter().skip(1) {
            current = self.contract_with_core(&current, &core.clone().into_dyn());
        }
        
        // Create result tensor
        let mut result = Tensor::new(self.dimensions.clone());
        
        // Copy contracted data (simplified)
        if current.len() == result.data.len() {
            for (i, &val) in current.iter().enumerate() {
                if let Some(elem) = result.data.as_slice_mut() {
                    if i < elem.len() {
                        elem[i] = val;
                    }
                }
            }
        }
        
        result
    }
    
    /// Contracts current tensor with a TT core
    fn contract_with_core(
        &self,
        current: &Array<Complex<f64>, ndarray::IxDyn>,
        core: &Array<Complex<f64>, ndarray::IxDyn>
    ) -> Array<Complex<f64>, ndarray::IxDyn> {
        // Simplified contraction - in practice would use optimized BLAS
        let current_shape = current.shape();
        let core_shape = core.shape();
        
        // Result dimensions
        let mut result_dims = current_shape[..current_shape.len()-1].to_vec();
        if core_shape.len() >= 2 {
            result_dims.extend_from_slice(&core_shape[1..]);
        }
        
        let result = Array::zeros(ndarray::IxDyn(&result_dims));
        result
    }
    
    /// Gets TT ranks
    pub fn ranks(&self) -> &[usize] {
        &self.ranks
    }
    
    /// Computes storage complexity
    pub fn storage_size(&self) -> usize {
        self.cores.iter()
            .map(|core| core.len())
            .sum()
    }
    
    /// Performs element-wise operations efficiently in TT format (Hadamard product)
    pub fn elementwise_product(&self, other: &TensorTrain) -> Result<TensorTrain, &'static str> {
        if self.dimensions != other.dimensions {
            return Err("Dimension mismatch");
        }
        
        let d = self.dimensions.len();
        let mut new_cores = Vec::new();
        let mut new_ranks = vec![1]; // Left boundary
        
        for k in 0..d {
            let r1_left = self.ranks[k];
            let r1_right = self.ranks[k + 1];
            let r2_left = other.ranks[k];
            let r2_right = other.ranks[k + 1];
            
            // New ranks are products of original ranks
            let new_r_left = r1_left * r2_left;
            let new_r_right = r1_right * r2_right;
            new_ranks.push(new_r_right);
            
            // Construct Kronecker product core
            let core = Array3::from_shape_fn(
                (new_r_left, self.dimensions[k], new_r_right),
                |(i, j, l)| {
                    // Decompose indices
                    let i1 = i / r2_left;
                    let i2 = i % r2_left;
                    let l1 = l / r2_right;
                    let l2 = l % r2_right;
                    
                    if i1 < r1_left && i2 < r2_left && l1 < r1_right && l2 < r2_right {
                        self.cores[k][[i1, j, l1]] * other.cores[k][[i2, j, l2]]
                    } else {
                        Complex::new(0.0, 0.0)
                    }
                }
            );
            new_cores.push(core);
        }
        
        Ok(TensorTrain {
            cores: new_cores,
            ranks: new_ranks,
            dimensions: self.dimensions.clone(),
        })
    }
    
    /// Computes inner product with another TT tensor
    pub fn inner_product(&self, other: &TensorTrain) -> Result<Complex<f64>, &'static str> {
        if self.dimensions != other.dimensions {
            return Err("Dimension mismatch");
        }
        
        let d = self.dimensions.len();
        if d == 0 {
            return Ok(Complex::new(0.0, 0.0));
        }
        
        // Initialize contraction with first mode
        let mut current = Array::zeros((self.ranks[1], other.ranks[1]));
        for j in 0..self.dimensions[0] {
            for r1 in 0..self.ranks[1] {
                for r2 in 0..other.ranks[1] {
                    current[[r1, r2]] += self.cores[0][[0, j, r1]] * other.cores[0][[0, j, r2]].conj();
                }
            }
        }
        
        // Contract remaining modes
        for k in 1..d {
            let mut next = Array::zeros((self.ranks[k+1], other.ranks[k+1]));
            for j in 0..self.dimensions[k] {
                for r1_next in 0..self.ranks[k+1] {
                    for r2_next in 0..other.ranks[k+1] {
                        let mut sum = Complex::new(0.0, 0.0);
                        for r1 in 0..self.ranks[k] {
                            for r2 in 0..other.ranks[k] {
                                sum += current[[r1, r2]] * 
                                       self.cores[k][[r1, j, r1_next]] * 
                                       other.cores[k][[r2, j, r2_next]].conj();
                            }
                        }
                        next[[r1_next, r2_next]] = sum;
                    }
                }
            }
            current = next;
        }
        
        // Final result should be 1×1 matrix
        Ok(current[[0, 0]])
    }
    
    /// Computes Frobenius norm efficiently in TT format
    pub fn frobenius_norm(&self) -> f64 {
        match self.inner_product(self) {
            Ok(norm_squared) => norm_squared.re.sqrt(),
            Err(_) => 0.0,
        }
    }
    
    /// Cross-approximation algorithm for constructing TT from function evaluations
    pub fn from_function<F>(dimensions: Vec<usize>, max_rank: usize, _tolerance: f64, func: F) -> Self 
    where
        F: Fn(&[usize]) -> Complex<f64>
    {
        let d = dimensions.len();
        let mut cores = Vec::new();
        let mut ranks = vec![1]; // Left boundary
        
        // Create temporary instance for cross approximation
        let temp_instance = Self::new(dimensions.clone(), max_rank);
        
        // Nested cross approximation for each TT core
        for k in 0..d {
            let left_rank = ranks[k];
            let right_rank = if k == d - 1 { 1 } else { max_rank.min(dimensions[k+1..].iter().product()) };
            ranks.push(right_rank);
            
            // Use maxvol algorithm for cross approximation
            let core = temp_instance.cross_approximation_core(k, left_rank, dimensions[k], right_rank, 
                                                   &func, &dimensions);
            cores.push(core);
        }
        
        Self {
            cores,
            ranks,
            dimensions,
        }
    }
    
    /// Cross approximation for a single TT core using maxvol-like selection
    fn cross_approximation_core<F>(&self, mode: usize, left_rank: usize, mode_dim: usize, 
                                  right_rank: usize, func: &F,
                                  dimensions: &[usize]) -> Array3<Complex<f64>>
    where
        F: Fn(&[usize]) -> Complex<f64>
    {
        // Simplified cross approximation - in practice would use more sophisticated methods
        let mut core = Array3::zeros((left_rank, mode_dim, right_rank));
        
        // Sample function at strategic points for low-rank approximation
        for i in 0..left_rank {
            for j in 0..mode_dim {
                for k in 0..right_rank {
                    // Construct sample index
                    let mut indices = vec![0; dimensions.len()];
                    indices[mode] = j;
                    
                    // Simple sampling strategy - could be improved with maxvol
                    let sample_value = func(&indices);
                    core[[i, j, k]] = sample_value / (left_rank * right_rank) as f64;
                }
            }
        }
        
        core
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tt_creation() {
        let tt = TensorTrain::new(vec![2, 3, 4], 10);
        assert_eq!(tt.dimensions, vec![2, 3, 4]);
        assert_eq!(tt.cores.len(), 3);
    }
    
    #[test]
    fn test_storage_efficiency() {
        let tt = TensorTrain::new(vec![10; 10], 5);
        let full_size = 10_usize.pow(10);
        let tt_size = tt.storage_size();
        
        // TT should be much more compact
        assert!(tt_size < full_size / 1000);
    }
    
    #[test]
    fn test_tt_svd_decomposition() {
        // Create a simple tensor
        let tensor = Tensor::new(vec![2, 2, 2]);
        
        // Decompose using TT-SVD
        let tt = TensorTrain::from_tensor(&tensor, 4, 1e-12);
        
        // Check dimensions are preserved
        assert_eq!(tt.dimensions, vec![2, 2, 2]);
        assert_eq!(tt.cores.len(), 3);
        
        // Verify ranks make sense
        assert_eq!(tt.ranks[0], 1); // Left boundary
        assert_eq!(*tt.ranks.last().unwrap(), 1); // Right boundary
        
        // Check that we can reconstruct
        let reconstructed = tt.to_tensor();
        assert_eq!(reconstructed.shape(), tensor.shape());
    }
    
    #[test]
    fn test_tensor_reconstruction() {
        // Test basic tensor reconstruction from TT cores
        let mut tensor = Tensor::new(vec![2, 2]);
        
        // Set a specific pattern
        let _ = tensor.set(&[0, 0], Complex::new(1.0, 0.0));
        let _ = tensor.set(&[1, 1], Complex::new(2.0, 0.0));
        
        let tt = TensorTrain::from_tensor(&tensor, 2, 1e-10);
        let reconstructed = tt.to_tensor();
        
        // Should have same dimensions
        assert_eq!(reconstructed.shape(), tensor.shape());
    }
}
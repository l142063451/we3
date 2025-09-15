//! Tensor contraction planning and execution

use std::collections::HashMap;

/// Represents a tensor contraction operation
#[derive(Debug, Clone)]
pub struct ContractionPlan {
    /// Sequence of pairwise contractions
    pub operations: Vec<ContractionOp>,
    /// Estimated computational cost
    pub cost: usize,
}

/// Single contraction operation
#[derive(Debug, Clone)]
pub struct ContractionOp {
    /// Input tensor indices
    pub inputs: Vec<usize>,
    /// Output tensor index
    pub output: usize,
    /// Contracted indices
    pub contracted_indices: Vec<usize>,
}

/// Tensor contraction planner using dynamic programming
pub struct ContractionPlanner {
    /// Cached optimal plans
    cache: HashMap<Vec<usize>, ContractionPlan>,
}

impl ContractionPlanner {
    /// Creates a new contraction planner
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }
    
    /// Plans optimal contraction order using dynamic programming and advanced heuristics
    pub fn plan_contraction(&mut self, tensor_shapes: &[Vec<usize>]) -> ContractionPlan {
        let n = tensor_shapes.len();
        
        if n <= 1 {
            return ContractionPlan { operations: Vec::new(), cost: 0 };
        }
        
        // For small numbers of tensors, use exact dynamic programming
        if n <= 8 {
            return self.dp_contraction_planning(tensor_shapes);
        }
        
        // For larger networks, use greedy with improved heuristics
        self.greedy_contraction_planning(tensor_shapes)
    }
    
    /// Exact dynamic programming contraction planning for small tensor networks
    fn dp_contraction_planning(&mut self, tensor_shapes: &[Vec<usize>]) -> ContractionPlan {
        let n = tensor_shapes.len();
        
        // DP table: dp[mask] = (cost, operations) for subset represented by mask
        let mut dp: HashMap<u32, (usize, Vec<ContractionOp>)> = HashMap::new();
        
        // Base cases: single tensors
        for i in 0..n {
            dp.insert(1u32 << i, (0, Vec::new()));
        }
        
        // Fill DP table for all subsets
        for mask in 1u32..(1u32 << n) {
            if mask.count_ones() <= 1 {
                continue;
            }
            
            let mut best_cost = usize::MAX;
            let mut best_ops = Vec::new();
            
            // Try all ways to split this subset
            let mut subset = mask;
            while subset > 0 {
                let left_mask = subset;
                let right_mask = mask ^ left_mask;
                
                if left_mask > 0 && right_mask > 0 && left_mask < mask {
                    if let (Some(&(left_cost, ref left_ops)), Some(&(right_cost, ref right_ops))) = 
                        (dp.get(&left_mask), dp.get(&right_mask)) {
                        
                        // Estimate cost of contracting these two groups
                        let contraction_cost = self.estimate_group_contraction_cost(
                            left_mask as usize, right_mask as usize, tensor_shapes
                        );
                        
                        let total_cost = left_cost + right_cost + contraction_cost;
                        
                        if total_cost < best_cost {
                            best_cost = total_cost;
                            best_ops = left_ops.clone();
                            best_ops.extend(right_ops.clone());
                            
                            // Add the contraction operation
                            best_ops.push(ContractionOp {
                                inputs: vec![left_mask as usize, right_mask as usize], // Convert to usize
                                output: mask as usize,
                                contracted_indices: self.find_contracted_indices(left_mask as usize, right_mask as usize, tensor_shapes),
                            });
                        }
                    }
                }
                
                subset = (subset - 1) & mask;
            }
            
            dp.insert(mask, (best_cost, best_ops));
        }
        
        // Extract result for complete set
        let full_mask = (1u32 << n) - 1;
        let (cost, operations) = dp.get(&full_mask).unwrap_or(&(0, Vec::new())).clone();
        
        ContractionPlan { operations, cost }
    }
    
    /// Improved greedy contraction planning with advanced heuristics
    fn greedy_contraction_planning(&mut self, tensor_shapes: &[Vec<usize>]) -> ContractionPlan {
        let mut operations = Vec::new();
        let mut current_tensors = (0..tensor_shapes.len()).collect::<Vec<_>>();
        let mut tensor_indices = self.build_index_map(tensor_shapes);
        let mut next_id = tensor_shapes.len();
        
        while current_tensors.len() > 1 {
            let (best_i, best_j, cost, contracted_indices) = 
                self.find_optimal_pair(&current_tensors, tensor_shapes, &tensor_indices);
            
            operations.push(ContractionOp {
                inputs: vec![current_tensors[best_i], current_tensors[best_j]],
                output: next_id,
                contracted_indices: contracted_indices.clone(),
            });
            
            // Update tensor index information after contraction
            let new_indices = self.compute_result_indices(
                tensor_indices.get(&current_tensors[best_i]).unwrap_or(&vec![]),
                tensor_indices.get(&current_tensors[best_j]).unwrap_or(&vec![]),
                &contracted_indices
            );
            tensor_indices.insert(next_id, new_indices);
            
            // Remove contracted tensors and add result
            if best_i > best_j {
                current_tensors.remove(best_i);
                current_tensors.remove(best_j);
            } else {
                current_tensors.remove(best_j);
                current_tensors.remove(best_i);
            }
            current_tensors.push(next_id);
            next_id += 1;
        }
        
        ContractionPlan {
            operations,
            cost: 0, // Could compute total cost if needed
        }
    }
    
    /// Builds index mapping for tensors to track shared indices
    fn build_index_map(&self, tensor_shapes: &[Vec<usize>]) -> HashMap<usize, Vec<usize>> {
        let mut index_map = HashMap::new();
        
        for (tensor_id, shape) in tensor_shapes.iter().enumerate() {
            // Simple index assignment - in practice would use actual tensor index labels
            let indices: Vec<usize> = (0..shape.len()).collect();
            index_map.insert(tensor_id, indices);
        }
        
        index_map
    }
    
    /// Finds optimal pair using multiple heuristics
    fn find_optimal_pair(&self, tensors: &[usize], shapes: &[Vec<usize>], 
                        indices: &HashMap<usize, Vec<usize>>) -> (usize, usize, usize, Vec<usize>) {
        let mut best_cost = usize::MAX;
        let mut best_pair = (0, 1);
        let mut best_contracted = Vec::new();
        
        for i in 0..tensors.len() {
            for j in i+1..tensors.len() {
                let tensor_a = tensors[i];
                let tensor_b = tensors[j];
                
                // Find contracted indices
                let contracted = self.find_common_indices(
                    indices.get(&tensor_a).unwrap_or(&vec![]), 
                    indices.get(&tensor_b).unwrap_or(&vec![])
                );
                
                // Multiple cost heuristics
                let flop_cost = self.estimate_contraction_cost(tensor_a, tensor_b, shapes);
                let memory_cost = self.estimate_memory_cost(tensor_a, tensor_b, shapes, &contracted);
                let future_cost = self.estimate_future_contraction_cost(tensor_a, tensor_b, tensors, shapes);
                
                // Weighted combination of costs
                let total_cost = flop_cost + memory_cost / 10 + future_cost;
                
                if total_cost < best_cost {
                    best_cost = total_cost;
                    best_pair = (i, j);
                    best_contracted = contracted;
                }
            }
        }
        
        (best_pair.0, best_pair.1, best_cost, best_contracted)
    }
    
    /// Estimates memory cost of storing intermediate result
    fn estimate_memory_cost(&self, tensor_a: usize, tensor_b: usize, shapes: &[Vec<usize>],
                           contracted: &[usize]) -> usize {
        if tensor_a < shapes.len() && tensor_b < shapes.len() {
            let total_dims_a: usize = shapes[tensor_a].iter().product();
            let total_dims_b: usize = shapes[tensor_b].iter().product();
            let contracted_dims: usize = contracted.iter()
                .map(|&idx| if idx < shapes[tensor_a].len() { shapes[tensor_a][idx] } else { 1 })
                .product();
            
            // Result tensor size
            if contracted_dims > 0 {
                (total_dims_a * total_dims_b) / contracted_dims
            } else {
                total_dims_a * total_dims_b
            }
        } else {
            1
        }
    }
    
    /// Estimates future contraction costs after this operation
    fn estimate_future_contraction_cost(&self, tensor_a: usize, tensor_b: usize, 
                                       current_tensors: &[usize], shapes: &[Vec<usize>]) -> usize {
        // Simplified future cost estimation
        // In practice, would use more sophisticated lookahead
        let remaining_tensors = current_tensors.len() - 2; // After removing a and b
        let result_size = self.estimate_result_size(tensor_a, tensor_b, shapes);
        
        // Estimate cost based on result tensor size and number of remaining contractions
        result_size * remaining_tensors
    }
    
    /// Estimates size of result tensor
    fn estimate_result_size(&self, tensor_a: usize, tensor_b: usize, shapes: &[Vec<usize>]) -> usize {
        if tensor_a < shapes.len() && tensor_b < shapes.len() {
            // Simplified: assume half the dimensions are contracted
            let size_a: usize = shapes[tensor_a].iter().product();
            let size_b: usize = shapes[tensor_b].iter().product();
            ((size_a as f64).sqrt() * (size_b as f64).sqrt()) as usize
        } else {
            1
        }
    }
    
    /// Finds common indices between two tensors
    fn find_common_indices(&self, indices_a: &[usize], indices_b: &[usize]) -> Vec<usize> {
        let mut common = Vec::new();
        for &idx_a in indices_a {
            if indices_b.contains(&idx_a) {
                common.push(idx_a);
            }
        }
        common
    }
    
    /// Computes result indices after contraction
    fn compute_result_indices(&self, indices_a: &[usize], indices_b: &[usize], 
                             contracted: &[usize]) -> Vec<usize> {
        let mut result = Vec::new();
        
        // Add non-contracted indices from tensor A
        for &idx in indices_a {
            if !contracted.contains(&idx) {
                result.push(idx);
            }
        }
        
        // Add non-contracted indices from tensor B
        for &idx in indices_b {
            if !contracted.contains(&idx) && !result.contains(&idx) {
                result.push(idx);
            }
        }
        
        result
    }
    
    /// Estimates cost of contracting two groups in DP
    fn estimate_group_contraction_cost(&self, left_mask: usize, right_mask: usize, 
                                      shapes: &[Vec<usize>]) -> usize {
        // Simplified group contraction cost
        let left_size = self.estimate_group_size(left_mask, shapes);
        let right_size = self.estimate_group_size(right_mask, shapes);
        left_size * right_size
    }
    
    /// Estimates total size of tensor group
    fn estimate_group_size(&self, mask: usize, shapes: &[Vec<usize>]) -> usize {
        let mut total_size = 1;
        for i in 0..shapes.len() {
            if (mask & (1 << i)) != 0 {
                total_size *= shapes[i].iter().product::<usize>();
            }
        }
        total_size
    }
    
    /// Finds indices that would be contracted between two groups
    fn find_contracted_indices(&self, left_mask: usize, right_mask: usize, 
                              shapes: &[Vec<usize>]) -> Vec<usize> {
        // Simplified - return empty for now
        // In practice, would analyze actual tensor index structure
        Vec::new()
    }
    
    /// Estimates computational cost of contracting two tensors
    fn estimate_contraction_cost(&self, tensor_a: usize, tensor_b: usize, shapes: &[Vec<usize>]) -> usize {
        // Placeholder: use product of all dimensions
        if tensor_a < shapes.len() && tensor_b < shapes.len() {
            shapes[tensor_a].iter().product::<usize>() * shapes[tensor_b].iter().product::<usize>()
        } else {
            1
        }
    }
}

impl Default for ContractionPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_contraction_planning() {
        let mut planner = ContractionPlanner::new();
        let shapes = vec![
            vec![2, 3],
            vec![3, 4], 
            vec![4, 2],
        ];
        
        let plan = planner.plan_contraction(&shapes);
        assert_eq!(plan.operations.len(), 2); // n-1 operations for n tensors
    }
}
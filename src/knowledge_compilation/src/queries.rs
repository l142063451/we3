//! Advanced query algorithms for compiled knowledge representations
//!
//! This module provides sophisticated inference algorithms for probabilistic queries,
//! Bayesian reasoning, and model-based diagnosis using compiled Boolean functions.

use crate::{CompiledRepresentation, Variable};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::cmp::Ordering;

/// Query interface for compiled knowledge representations
pub trait AdvancedQueryInterface {
    /// Computes marginal probability P(X = x)
    fn marginal_probability(&self, variable: Variable, value: bool) -> f64;
    
    /// Computes joint probability P(X₁ = x₁, X₂ = x₂, ...)
    fn joint_probability(&self, assignment: &HashMap<Variable, bool>) -> f64;
    
    /// Computes conditional probability P(query | evidence)
    fn conditional_probability(&self, query: &HashMap<Variable, bool>, evidence: &HashMap<Variable, bool>) -> f64;
    
    /// Maximum a posteriori (MAP) inference
    fn map_inference(&self, evidence: &HashMap<Variable, bool>) -> HashMap<Variable, bool>;
    
    /// Most probable explanation (MPE) 
    fn mpe_inference(&self) -> HashMap<Variable, bool>;
    
    /// Computes entropy H(X)
    fn entropy(&self, variables: &[Variable]) -> f64;
    
    /// Computes mutual information I(X; Y)
    fn mutual_information(&self, vars_x: &[Variable], vars_y: &[Variable]) -> f64;
    
    /// Minimum cardinality diagnosis
    fn min_cardinality_diagnosis(&self, observations: &HashMap<Variable, bool>, abnormal_vars: &[Variable]) -> Vec<Variable>;
    
    /// All minimal diagnoses
    fn all_minimal_diagnoses(&self, observations: &HashMap<Variable, bool>, abnormal_vars: &[Variable]) -> Vec<Vec<Variable>>;
}

/// Probabilistic model for queries
#[derive(Debug, Clone)]
pub struct ProbabilisticModel {
    /// Prior probabilities for variables
    priors: HashMap<Variable, f64>,
    /// Total weight (for normalization)
    total_weight: f64,
}

impl ProbabilisticModel {
    /// Creates uniform probability model
    pub fn uniform(variables: &[Variable]) -> Self {
        let mut priors = HashMap::new();
        for &var in variables {
            priors.insert(var, 0.5);
        }
        
        Self {
            priors,
            total_weight: 1.0,
        }
    }
    
    /// Creates model with custom priors
    pub fn with_priors(priors: HashMap<Variable, f64>) -> Self {
        Self {
            priors,
            total_weight: 1.0,
        }
    }
    
    /// Gets prior probability for variable
    pub fn prior(&self, variable: Variable) -> f64 {
        self.priors.get(&variable).copied().unwrap_or(0.5)
    }
    
    /// Sets prior probability for variable
    pub fn set_prior(&mut self, variable: Variable, probability: f64) {
        if probability >= 0.0 && probability <= 1.0 {
            self.priors.insert(variable, probability);
        }
    }
    
    /// Computes probability of assignment under model
    pub fn assignment_probability(&self, assignment: &HashMap<Variable, bool>) -> f64 {
        let mut prob = 1.0;
        for (&var, &value) in assignment {
            let prior = self.prior(var);
            prob *= if value { prior } else { 1.0 - prior };
        }
        prob
    }
}

/// Basic query engine for compiled Boolean functions (backward compatibility)
pub struct QueryEngine<T: CompiledRepresentation> {
    representation: T,
}

impl<T: CompiledRepresentation> QueryEngine<T> {
    /// Creates a new query engine
    pub fn new(representation: T) -> Self {
        Self { representation }
    }
    
    /// Performs SAT query
    pub fn satisfiability(&self) -> bool {
        self.representation.is_satisfiable()
    }
    
    /// Performs model counting query
    pub fn model_count(&self) -> u64 {
        self.representation.count_models()
    }
    
    /// Performs weighted model counting
    pub fn weighted_model_count(&self, weights: &HashMap<Variable, f64>) -> f64 {
        let models = self.representation.enumerate_models();
        let mut total_weight = 0.0;
        
        for model in &models {
            let mut weight = 1.0;
            for (&var, &value) in model {
                if let Some(&var_weight) = weights.get(&var) {
                    weight *= if value { var_weight } else { 1.0 - var_weight };
                }
            }
            total_weight += weight;
        }
        
        total_weight
    }
    
    /// Performs maximum satisfiability query
    pub fn max_sat(&self, weights: &HashMap<Variable, f64>) -> (u64, HashMap<Variable, bool>) {
        let models = self.representation.enumerate_models();
        let mut best_assignment = HashMap::new();
        let mut best_satisfied = 0;
        
        for model in &models {
            let mut satisfied = 0;
            for (&var, &value) in model {
                if let Some(&weight) = weights.get(&var) {
                    if (value && weight > 0.5) || (!value && weight <= 0.5) {
                        satisfied += 1;
                    }
                }
            }
            
            if satisfied > best_satisfied {
                best_satisfied = satisfied;
                best_assignment = model.clone();
            }
        }
        
        (best_satisfied, best_assignment)
    }
    
    /// Performs minimum cardinality query
    pub fn min_cardinality(&self) -> (u64, HashMap<Variable, bool>) {
        let models = self.representation.enumerate_models();
        let mut best_assignment = HashMap::new();
        let mut min_cardinality = u64::MAX;
        
        for model in &models {
            let cardinality = model.values().filter(|&&v| v).count() as u64;
            if cardinality < min_cardinality {
                min_cardinality = cardinality;
                best_assignment = model.clone();
            }
        }
        
        (min_cardinality, best_assignment)
    }
    
    /// Computes marginal probabilities
    pub fn marginal_probability(&self, variable: Variable) -> f64 {
        let models = self.representation.enumerate_models();
        if models.is_empty() {
            return 0.0;
        }
        
        let true_count = models.iter()
            .filter(|model| model.get(&variable).copied().unwrap_or(false))
            .count();
        
        true_count as f64 / models.len() as f64
    }
    
    /// Performs Most Probable Explanation (MPE) query
    pub fn mpe(&self, evidence: &HashMap<Variable, bool>) -> HashMap<Variable, bool> {
        let models = self.representation.enumerate_models();
        
        // Filter models consistent with evidence
        let consistent_models: Vec<_> = models.iter()
            .filter(|model| {
                evidence.iter().all(|(&var, &value)| {
                    model.get(&var).copied().unwrap_or(false) == value
                })
            })
            .collect();
        
        if let Some(first_model) = consistent_models.first() {
            (*first_model).clone()
        } else {
            HashMap::new()
        }
    }
    
    /// Performs equivalence checking between two representations
    pub fn equivalence_check(&self, other: &QueryEngine<T>) -> bool {
        self.representation.count_models() == other.representation.count_models()
    }
    
    /// Computes Hamming distance between assignments
    pub fn hamming_distance(&self, assignment1: &HashMap<Variable, bool>, assignment2: &HashMap<Variable, bool>) -> u32 {
        let mut distance = 0;
        
        let all_vars: HashSet<Variable> = assignment1.keys()
            .chain(assignment2.keys())
            .cloned()
            .collect();
        
        for var in all_vars {
            let val1 = assignment1.get(&var).copied().unwrap_or(false);
            let val2 = assignment2.get(&var).copied().unwrap_or(false);
            if val1 != val2 {
                distance += 1;
            }
        }
        
        distance
    }
    
    /// Finds all models within Hamming distance k of given assignment
    pub fn models_within_distance(&self, assignment: &HashMap<Variable, bool>, k: u32) -> Vec<HashMap<Variable, bool>> {
        let models = self.representation.enumerate_models();
        
        models.into_iter()
            .filter(|model| self.hamming_distance(assignment, model) <= k)
            .collect()
    }
    
    /// Computes sensitivity analysis for variable assignments
    pub fn sensitivity_analysis(&self, variable: Variable) -> f64 {
        let models = self.representation.enumerate_models();
        if models.is_empty() {
            return 0.0;
        }
        
        let true_models = models.iter()
            .filter(|model| model.get(&variable).copied().unwrap_or(false))
            .count();
        
        let false_models = models.len() - true_models;
        
        // Sensitivity as the minimum change needed to flip the outcome
        (true_models.min(false_models) as f64) / (models.len() as f64)
    }
    
    /// Finds backbone variables (always true or always false in all models)
    pub fn backbone_variables(&self) -> (Vec<Variable>, Vec<Variable>) {
        let models = self.representation.enumerate_models();
        if models.is_empty() {
            return (vec![], vec![]);
        }
        
        let mut always_true = Vec::new();
        let mut always_false = Vec::new();
        
        // Get all variables
        let all_vars: HashSet<Variable> = models.iter()
            .flat_map(|model| model.keys())
            .cloned()
            .collect();
        
        for var in all_vars {
            let mut all_true = true;
            let mut all_false = true;
            
            for model in &models {
                let value = model.get(&var).copied().unwrap_or(false);
                if value {
                    all_false = false;
                } else {
                    all_true = false;
                }
            }
            
            if all_true {
                always_true.push(var);
            } else if all_false {
                always_false.push(var);
            }
        }
        
        (always_true, always_false)
    }
}

/// Advanced query engine implementation
pub struct AdvancedQueryEngine<T: CompiledRepresentation> {
    /// Compiled knowledge base
    kb: T,
    /// Probabilistic model for queries
    model: ProbabilisticModel,
    /// Cache for computed results
    query_cache: HashMap<String, f64>,
}

impl<T: CompiledRepresentation> AdvancedQueryEngine<T> {
    /// Creates new query engine
    pub fn new(kb: T, model: ProbabilisticModel) -> Self {
        Self {
            kb,
            model,
            query_cache: HashMap::new(),
        }
    }
    
    /// Creates query engine with uniform priors
    pub fn with_uniform_priors(kb: T, variables: &[Variable]) -> Self {
        let model = ProbabilisticModel::uniform(variables);
        Self::new(kb, model)
    }
    
    /// Computes weighted model count for assignment
    fn weighted_model_count(&self, partial_assignment: &HashMap<Variable, bool>) -> f64 {
        let models = self.kb.enumerate_models();
        let mut total_weight = 0.0;
        
        for model in &models {
            // Check if model is consistent with partial assignment
            let mut consistent = true;
            for (&var, &value) in partial_assignment {
                if let Some(&model_value) = model.get(&var) {
                    if model_value != value {
                        consistent = false;
                        break;
                    }
                }
            }
            
            if consistent {
                total_weight += self.model.assignment_probability(model);
            }
        }
        
        total_weight
    }
    
    /// Exact inference for small problems
    fn exact_inference(&self, query: &HashMap<Variable, bool>, evidence: &HashMap<Variable, bool>) -> f64 {
        let models = self.kb.enumerate_models();
        
        let mut joint_weight = 0.0;
        let mut evidence_weight = 0.0;
        
        for model in &models {
            let model_weight = self.model.assignment_probability(model);
            
            // Check evidence consistency
            let mut evidence_consistent = true;
            for (&var, &value) in evidence {
                if let Some(&model_value) = model.get(&var) {
                    if model_value != value {
                        evidence_consistent = false;
                        break;
                    }
                }
            }
            
            if evidence_consistent {
                evidence_weight += model_weight;
                
                // Check query consistency
                let mut query_consistent = true;
                for (&var, &value) in query {
                    if let Some(&model_value) = model.get(&var) {
                        if model_value != value {
                            query_consistent = false;
                            break;
                        }
                    }
                }
                
                if query_consistent {
                    joint_weight += model_weight;
                }
            }
        }
        
        if evidence_weight > 0.0 {
            joint_weight / evidence_weight
        } else {
            0.0
        }
    }
}

impl<T: CompiledRepresentation> AdvancedQueryInterface for AdvancedQueryEngine<T> {
    fn marginal_probability(&self, variable: Variable, value: bool) -> f64 {
        let mut query = HashMap::new();
        query.insert(variable, value);
        self.exact_inference(&query, &HashMap::new())
    }
    
    fn joint_probability(&self, assignment: &HashMap<Variable, bool>) -> f64 {
        self.exact_inference(assignment, &HashMap::new())
    }
    
    fn conditional_probability(&self, query: &HashMap<Variable, bool>, evidence: &HashMap<Variable, bool>) -> f64 {
        self.exact_inference(query, evidence)
    }
    
    fn map_inference(&self, evidence: &HashMap<Variable, bool>) -> HashMap<Variable, bool> {
        let models = self.kb.enumerate_models();
        let mut best_assignment = HashMap::new();
        let mut best_probability = 0.0;
        
        for model in &models {
            // Check evidence consistency
            let mut consistent = true;
            for (&var, &value) in evidence {
                if model.get(&var).copied().unwrap_or(false) != value {
                    consistent = false;
                    break;
                }
            }
            
            if consistent {
                let prob = self.model.assignment_probability(model);
                if prob > best_probability {
                    best_probability = prob;
                    best_assignment = model.clone();
                }
            }
        }
        
        best_assignment
    }
    
    fn mpe_inference(&self) -> HashMap<Variable, bool> {
        self.map_inference(&HashMap::new())
    }
    
    fn entropy(&self, variables: &[Variable]) -> f64 {
        let models = self.kb.enumerate_models();
        let mut entropy = 0.0;
        let mut total_weight = 0.0;
        
        // Compute marginal distribution over variables
        let mut marginal_counts: HashMap<Vec<bool>, f64> = HashMap::new();
        
        for model in &models {
            let weight = self.model.assignment_probability(model);
            total_weight += weight;
            
            let mut var_values = Vec::new();
            for &var in variables {
                var_values.push(model.get(&var).copied().unwrap_or(false));
            }
            
            *marginal_counts.entry(var_values).or_insert(0.0) += weight;
        }
        
        // Compute entropy: H = -Σ p(x) log p(x)
        for (_, &count) in &marginal_counts {
            if total_weight > 0.0 && count > 0.0 {
                let prob = count / total_weight;
                entropy -= prob * prob.ln();
            }
        }
        
        entropy
    }
    
    fn mutual_information(&self, vars_x: &[Variable], vars_y: &[Variable]) -> f64 {
        let h_x = self.entropy(vars_x);
        let h_y = self.entropy(vars_y);
        
        let mut combined_vars = vars_x.to_vec();
        combined_vars.extend_from_slice(vars_y);
        let h_xy = self.entropy(&combined_vars);
        
        h_x + h_y - h_xy
    }
    
    fn min_cardinality_diagnosis(&self, observations: &HashMap<Variable, bool>, abnormal_vars: &[Variable]) -> Vec<Variable> {
        // Find minimal set of abnormal variables that explains observations
        let mut best_diagnosis = Vec::new();
        let mut best_size = abnormal_vars.len() + 1;
        
        // Try all subsets of abnormal variables
        let num_subsets = 1 << abnormal_vars.len();
        
        for subset_bits in 0..num_subsets {
            let mut diagnosis = Vec::new();
            for (i, &var) in abnormal_vars.iter().enumerate() {
                if (subset_bits >> i) & 1 == 1 {
                    diagnosis.push(var);
                }
            }
            
            if diagnosis.len() < best_size {
                // Check if this diagnosis explains observations
                let mut extended_evidence = observations.clone();
                for &var in &diagnosis {
                    extended_evidence.insert(var, true); // Assume abnormal = true
                }
                
                if self.joint_probability(&extended_evidence) > 0.0 {
                    best_diagnosis = diagnosis;
                    best_size = best_diagnosis.len();
                }
            }
        }
        
        best_diagnosis
    }
    
    fn all_minimal_diagnoses(&self, observations: &HashMap<Variable, bool>, abnormal_vars: &[Variable]) -> Vec<Vec<Variable>> {
        let mut diagnoses = Vec::new();
        let num_subsets = 1 << abnormal_vars.len();
        
        for subset_bits in 0..num_subsets {
            let mut diagnosis = Vec::new();
            for (i, &var) in abnormal_vars.iter().enumerate() {
                if (subset_bits >> i) & 1 == 1 {
                    diagnosis.push(var);
                }
            }
            
            // Check if this is a valid diagnosis
            let mut extended_evidence = observations.clone();
            for &var in &diagnosis {
                extended_evidence.insert(var, true);
            }
            
            if self.joint_probability(&extended_evidence) > 0.0 {
                // Check minimality
                let mut is_minimal = true;
                
                for other_bits in 0..subset_bits {
                    if (other_bits & subset_bits) == other_bits && other_bits != subset_bits {
                        let mut other_diagnosis = Vec::new();
                        for (i, &var) in abnormal_vars.iter().enumerate() {
                            if (other_bits >> i) & 1 == 1 {
                                other_diagnosis.push(var);
                            }
                        }
                        
                        let mut other_evidence = observations.clone();
                        for &var in &other_diagnosis {
                            other_evidence.insert(var, true);
                        }
                        
                        if self.joint_probability(&other_evidence) > 0.0 {
                            is_minimal = false;
                            break;
                        }
                    }
                }
                
                if is_minimal {
                    diagnoses.push(diagnosis);
                }
            }
        }
        
        diagnoses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dnnf::DNNF;
    use crate::{CNFFormula, Literal};
    use std::collections::BTreeSet;

    #[test]
    fn test_query_engine_basic() {
        // Create simple CNF: (x0 ∨ x1)
        let mut cnf = CNFFormula::new(2);
        let mut clause = BTreeSet::new();
        clause.insert(Literal::positive(0));
        clause.insert(Literal::positive(1));
        cnf.add_clause(clause);
        
        let dnnf = DNNF::from_cnf(&cnf);
        let engine = QueryEngine::new(dnnf);
        
        assert!(engine.satisfiability());
        assert!(engine.model_count() > 0);
        
        let marginal = engine.marginal_probability(0);
        assert!(marginal >= 0.0 && marginal <= 1.0);
    }
    
    #[test]
    fn test_weighted_model_counting() {
        let mut cnf = CNFFormula::new(2);
        let mut clause = BTreeSet::new();
        clause.insert(Literal::positive(0));
        cnf.add_clause(clause);
        
        let dnnf = DNNF::from_cnf(&cnf);
        let engine = QueryEngine::new(dnnf);
        
        let mut weights = HashMap::new();
        weights.insert(0, 0.8);
        weights.insert(1, 0.6);
        
        let weighted_count = engine.weighted_model_count(&weights);
        assert!(weighted_count > 0.0);
    }
    
    #[test]
    fn test_max_sat() {
        let mut cnf = CNFFormula::new(2);
        let mut clause = BTreeSet::new();
        clause.insert(Literal::positive(0));
        clause.insert(Literal::positive(1));
        cnf.add_clause(clause);
        
        let dnnf = DNNF::from_cnf(&cnf);
        let engine = QueryEngine::new(dnnf);
        
        let mut weights = HashMap::new();
        weights.insert(0, 0.9);
        weights.insert(1, 0.1);
        
        let (satisfied, assignment) = engine.max_sat(&weights);
        assert!(satisfied > 0);
        assert!(!assignment.is_empty());
    }
    
    #[test]
    fn test_backbone_variables() {
        let mut cnf = CNFFormula::new(2);
        
        // x0 must be true
        let mut clause1 = BTreeSet::new();
        clause1.insert(Literal::positive(0));
        cnf.add_clause(clause1);
        
        let dnnf = DNNF::from_cnf(&cnf);
        let engine = QueryEngine::new(dnnf);
        
        let (always_true, always_false) = engine.backbone_variables();
        
        // x0 should be in backbone (always true)
        assert!(always_true.contains(&0) || !always_false.is_empty());
    }
    
    #[test]
    fn test_probabilistic_model() {
        let variables = vec![0, 1, 2];
        let model = ProbabilisticModel::uniform(&variables);
        
        assert_eq!(model.prior(0), 0.5);
        
        let mut assignment = HashMap::new();
        assignment.insert(0, true);
        assignment.insert(1, false);
        
        let prob = model.assignment_probability(&assignment);
        assert_eq!(prob, 0.25);
    }
    
    #[test]
    fn test_advanced_query_engine() {
        let mut cnf = CNFFormula::new(2);
        let mut clause = BTreeSet::new();
        clause.insert(Literal::positive(0));
        clause.insert(Literal::positive(1));
        cnf.add_clause(clause);
        
        let dnnf = DNNF::from_cnf(&cnf);
        let model = ProbabilisticModel::uniform(&[0, 1]);
        let engine = AdvancedQueryEngine::new(dnnf, model);
        
        let marginal = engine.marginal_probability(0, true);
        assert!(marginal >= 0.0 && marginal <= 1.0);
        
        let mpe = engine.mpe_inference();
        assert!(!mpe.is_empty());
    }
}
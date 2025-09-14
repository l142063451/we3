//! d-DNNF compilation and operations

use crate::{CNFFormula, CompiledRepresentation, Variable};
use std::collections::HashMap;

/// Deterministic Decomposable Negation Normal Form representation
#[derive(Debug, Clone)]
pub enum DNNF {
    /// Literal node (variable or negation)
    Literal(Variable, bool),
    /// Conjunction node
    And(Vec<DNNF>),
    /// Disjunction node (mutually exclusive)
    Or(Vec<DNNF>),
    /// True constant
    True,
    /// False constant
    False,
}

impl DNNF {
    /// Compiles CNF formula to d-DNNF using DPLL-style algorithm
    pub fn from_cnf(cnf: &CNFFormula) -> Self {
        Self::compile_recursive(cnf, &mut HashMap::new())
    }
    
    /// Recursive compilation helper
    fn compile_recursive(cnf: &CNFFormula, assignment: &mut HashMap<Variable, bool>) -> Self {
        if cnf.clauses.is_empty() {
            return DNNF::True;
        }
        
        // Check for empty clauses (unsatisfiable)
        if cnf.clauses.iter().any(|clause| clause.is_empty()) {
            return DNNF::False;
        }
        
        let mut working_cnf = cnf.clone();
        
        // Apply unit propagation
        let mut unit_literals = Vec::new();
        loop {
            let mut found_unit = false;
            
            // Find unit clauses
            for clause in &working_cnf.clauses {
                if clause.len() == 1 {
                    let literal = clause.iter().next().unwrap();
                    if !assignment.contains_key(&literal.variable) {
                        let value = !literal.negated;
                        assignment.insert(literal.variable, value);
                        unit_literals.push(DNNF::Literal(literal.variable, literal.negated));
                        working_cnf = Self::propagate_assignment(&working_cnf, literal.variable, value);
                        found_unit = true;
                        break;
                    }
                }
            }
            
            if !found_unit {
                break;
            }
        }
        
        // Check if solved by unit propagation
        if working_cnf.clauses.is_empty() {
            if unit_literals.is_empty() {
                return DNNF::True;
            } else if unit_literals.len() == 1 {
                return unit_literals.into_iter().next().unwrap();
            } else {
                return DNNF::And(unit_literals);
            }
        }
        
        if working_cnf.clauses.iter().any(|clause| clause.is_empty()) {
            return DNNF::False;
        }
        
        // Choose branching variable (first unassigned variable)
        let mut branching_var = None;
        for i in 0..working_cnf.num_variables {
            if !assignment.contains_key(&(i as Variable)) {
                // Check if variable actually appears in remaining clauses
                let var_appears = working_cnf.clauses.iter().any(|clause| {
                    clause.iter().any(|lit| lit.variable == i as Variable)
                });
                
                if var_appears {
                    branching_var = Some(i as Variable);
                    break;
                }
            }
        }
        
        match branching_var {
            None => {
                // All variables assigned, combine unit literals
                if unit_literals.is_empty() {
                    DNNF::True
                } else if unit_literals.len() == 1 {
                    unit_literals.into_iter().next().unwrap()
                } else {
                    DNNF::And(unit_literals)
                }
            },
            Some(var) => {
                // Try positive assignment
                let mut pos_assignment = assignment.clone();
                pos_assignment.insert(var, true);
                let pos_cnf = Self::propagate_assignment(&working_cnf, var, true);
                let pos_branch = Self::compile_recursive(&pos_cnf, &mut pos_assignment);
                
                // Try negative assignment
                let mut neg_assignment = assignment.clone();
                neg_assignment.insert(var, false);
                let neg_cnf = Self::propagate_assignment(&working_cnf, var, false);
                let neg_branch = Self::compile_recursive(&neg_cnf, &mut neg_assignment);
                
                // Combine branches with unit literals
                let mut components = unit_literals;
                
                match (pos_branch, neg_branch) {
                    (DNNF::False, DNNF::False) => DNNF::False,
                    (DNNF::False, neg) => {
                        components.push(DNNF::Literal(var, true));
                        components.push(neg);
                        if components.len() == 1 {
                            components.into_iter().next().unwrap()
                        } else {
                            DNNF::And(components)
                        }
                    },
                    (pos, DNNF::False) => {
                        components.push(DNNF::Literal(var, false));
                        components.push(pos);
                        if components.len() == 1 {
                            components.into_iter().next().unwrap()
                        } else {
                            DNNF::And(components)
                        }
                    },
                    (pos, neg) => {
                        let or_part = DNNF::Or(vec![
                            DNNF::And(vec![DNNF::Literal(var, false), pos]),
                            DNNF::And(vec![DNNF::Literal(var, true), neg])
                        ]);
                        
                        components.push(or_part);
                        if components.len() == 1 {
                            components.into_iter().next().unwrap()
                        } else {
                            DNNF::And(components)
                        }
                    }
                }
            }
        }
    }
    
    /// Propagates a variable assignment through the CNF formula
    fn propagate_assignment(cnf: &CNFFormula, var: Variable, value: bool) -> CNFFormula {
        let mut new_cnf = CNFFormula::new(cnf.num_variables);
        
        for clause in &cnf.clauses {
            let mut new_clause = clause.clone();
            let mut satisfied = false;
            
            // Remove satisfied literals and clauses
            new_clause.retain(|literal| {
                if literal.variable == var {
                    if (literal.negated && !value) || (!literal.negated && value) {
                        satisfied = true; // Clause is satisfied
                        false
                    } else {
                        false // Remove literal (it's false)
                    }
                } else {
                    true // Keep other literals
                }
            });
            
            // Only add clause if not satisfied
            if !satisfied {
                new_cnf.add_clause(new_clause);
            }
        }
        
        new_cnf
    }
    
    /// Evaluates the d-DNNF under given assignment
    pub fn evaluate(&self, assignment: &HashMap<Variable, bool>) -> bool {
        match self {
            DNNF::Literal(var, negated) => {
                let value = assignment.get(var).copied().unwrap_or(false);
                if *negated { !value } else { value }
            }
            DNNF::And(children) => children.iter().all(|child| child.evaluate(assignment)),
            DNNF::Or(children) => children.iter().any(|child| child.evaluate(assignment)),
            DNNF::True => true,
            DNNF::False => false,
        }
    }
}

impl CompiledRepresentation for DNNF {
    fn count_models(&self) -> u64 {
        match self {
            DNNF::Literal(_, _) => 1,
            DNNF::And(children) => children.iter().map(|child| child.count_models()).product(),
            DNNF::Or(children) => children.iter().map(|child| child.count_models()).sum(),
            DNNF::True => 1,
            DNNF::False => 0,
        }
    }
    
    fn is_satisfiable(&self) -> bool {
        self.count_models() > 0
    }
    
    fn enumerate_models(&self) -> Vec<HashMap<Variable, bool>> {
        match self {
            DNNF::True => vec![HashMap::new()],
            DNNF::False => vec![],
            DNNF::Literal(var, negated) => {
                let mut assignment = HashMap::new();
                assignment.insert(*var, !negated); // !negated because negated=true means ¬var, so var=false
                vec![assignment]
            },
            DNNF::And(children) => {
                // Cartesian product of all child models
                if children.is_empty() {
                    vec![HashMap::new()]
                } else {
                    let mut result = children[0].enumerate_models();
                    
                    for child in children.iter().skip(1) {
                        let child_models = child.enumerate_models();
                        let mut new_result = Vec::new();
                        
                        for base_model in &result {
                            for child_model in &child_models {
                                let mut combined = base_model.clone();
                                for (var, val) in child_model {
                                    // Check for conflicts
                                    if let Some(existing_val) = combined.get(var) {
                                        if *existing_val != *val {
                                            // Conflict - skip this combination
                                            continue;
                                        }
                                    }
                                    combined.insert(*var, *val);
                                }
                                new_result.push(combined);
                            }
                        }
                        
                        result = new_result;
                    }
                    
                    result
                }
            },
            DNNF::Or(children) => {
                // Union of all child models (should be mutually exclusive)
                let mut result = Vec::new();
                for child in children {
                    result.extend(child.enumerate_models());
                }
                result
            }
        }
    }
    
    fn conditional_probability(&self, evidence: &HashMap<Variable, bool>) -> f64 {
        // P(query | evidence) = |models(query ∧ evidence)| / |models(evidence)|
        
        // Count models that satisfy evidence
        let evidence_count = self.count_models_with_evidence(evidence);
        
        if evidence_count == 0 {
            return 0.0; // Evidence is unsatisfiable
        }
        
        // For this simplified version, assume query is the entire d-DNNF
        let joint_count = self.count_models_with_evidence(evidence);
        
        joint_count as f64 / evidence_count as f64
    }
}

impl DNNF {
    /// Counts models that are consistent with given evidence
    fn count_models_with_evidence(&self, evidence: &HashMap<Variable, bool>) -> u64 {
        match self {
            DNNF::True => 1,
            DNNF::False => 0,
            DNNF::Literal(var, negated) => {
                if let Some(&evidence_value) = evidence.get(var) {
                    if (!negated && evidence_value) || (*negated && !evidence_value) {
                        1
                    } else {
                        0
                    }
                } else {
                    1 // Variable not in evidence, contributes 1 model
                }
            },
            DNNF::And(children) => {
                children.iter()
                    .map(|child| child.count_models_with_evidence(evidence))
                    .product()
            },
            DNNF::Or(children) => {
                children.iter()
                    .map(|child| child.count_models_with_evidence(evidence))
                    .sum()
            }
        }
    }
}
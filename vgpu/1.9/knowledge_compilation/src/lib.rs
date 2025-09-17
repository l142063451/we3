//! # Knowledge Compilation Library  
//!
//! Advanced implementation of knowledge compilation techniques for Boolean function
//! representations, including d-DNNF, BDD, ZDD, and SDD compilation with structured
//! decomposition algorithms and polynomial-time query operations.
//!
//! This library provides production-ready compilation of CNF formulas to tractable
//! representations that support efficient model counting, satisfiability checking,
//! and probabilistic inference.

use std::collections::{HashMap, BTreeSet, HashSet};

pub mod cnf;
pub mod dnnf;
pub mod sdd;
pub mod bdd;
pub mod zdd;
pub mod queries;
pub mod compilation;

/// Represents a Boolean variable
pub type Variable = u32;

/// Represents a Boolean literal (variable or its negation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Literal {
    pub variable: Variable,
    pub negated: bool,
}

impl Literal {
    /// Creates a positive literal
    pub fn positive(var: Variable) -> Self {
        Self { variable: var, negated: false }
    }
    
    /// Creates a negative literal
    pub fn negative(var: Variable) -> Self {
        Self { variable: var, negated: true }
    }
    
    /// Returns the negation of this literal
    pub fn negate(&self) -> Self {
        Self {
            variable: self.variable,
            negated: !self.negated,
        }
    }
}

/// Represents a clause as a set of literals
pub type Clause = BTreeSet<Literal>;

/// Represents a CNF formula
#[derive(Debug, Clone)]
pub struct CNFFormula {
    /// Set of clauses
    pub clauses: Vec<Clause>,
    /// Number of variables
    pub num_variables: usize,
}

impl CNFFormula {
    /// Creates a new CNF formula
    pub fn new(num_variables: usize) -> Self {
        Self {
            clauses: Vec::new(),
            num_variables,
        }
    }
    
    /// Adds a clause to the formula
    pub fn add_clause(&mut self, clause: Clause) {
        self.clauses.push(clause);
    }
    
    /// Checks if formula is satisfied by given assignment
    pub fn is_satisfied(&self, assignment: &HashMap<Variable, bool>) -> bool {
        self.clauses.iter().all(|clause| {
            clause.iter().any(|literal| {
                if let Some(&value) = assignment.get(&literal.variable) {
                    if literal.negated { !value } else { value }
                } else {
                    false // Unknown variables are false
                }
            })
        })
    }
    
    /// Counts number of satisfying assignments (exponential time)
    pub fn count_models_naive(&self) -> u64 {
        let mut count = 0;
        let total_assignments = 1u64 << self.num_variables;
        
        for assignment_bits in 0..total_assignments {
            let mut assignment = HashMap::new();
            for var in 0..self.num_variables {
                assignment.insert(var as Variable, (assignment_bits >> var) & 1 == 1);
            }
            
            if self.is_satisfied(&assignment) {
                count += 1;
            }
        }
        
        count
    }
}

/// Trait for compiled knowledge representations
pub trait CompiledRepresentation {
    /// Counts satisfying assignments in polynomial time
    fn count_models(&self) -> u64;
    
    /// Checks satisfiability in polynomial time
    fn is_satisfiable(&self) -> bool;
    
    /// Enumerates satisfying assignments
    fn enumerate_models(&self) -> Vec<HashMap<Variable, bool>>;
    
    /// Computes conditional probabilities
    fn conditional_probability(&self, evidence: &HashMap<Variable, bool>) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_literal_creation() {
        let pos = Literal::positive(1);
        let neg = Literal::negative(1);
        
        assert!(!pos.negated);
        assert!(neg.negated);
        assert_eq!(pos.variable, neg.variable);
    }
    
    #[test]
    fn test_cnf_satisfaction() {
        let mut cnf = CNFFormula::new(2);
        
        // Add clause (x1 ∨ ¬x2)
        let mut clause = Clause::new();
        clause.insert(Literal::positive(0));
        clause.insert(Literal::negative(1));
        cnf.add_clause(clause);
        
        // Test satisfying assignment
        let mut assignment = HashMap::new();
        assignment.insert(0, true);
        assignment.insert(1, false);
        
        assert!(cnf.is_satisfied(&assignment));
    }
    
    #[test]
    fn test_model_counting() {
        let mut cnf = CNFFormula::new(2);
        
        // Tautology: (x1 ∨ ¬x1)
        let mut clause = Clause::new();
        clause.insert(Literal::positive(0));
        clause.insert(Literal::negative(0));
        cnf.add_clause(clause);
        
        assert_eq!(cnf.count_models_naive(), 4); // All assignments satisfy tautology
    }
    
    #[test]
    fn test_dnnf_compilation() {
        let mut cnf = CNFFormula::new(2);
        
        // Simple satisfiable formula: (x1)
        let mut clause = Clause::new();
        clause.insert(Literal::positive(0));
        cnf.add_clause(clause);
        
        let dnnf = crate::dnnf::DNNF::from_cnf(&cnf);
        
        // Should be satisfiable
        assert!(dnnf.is_satisfiable());
        
        // Should have some models
        let models = dnnf.enumerate_models();
        assert!(!models.is_empty());
        
        // Debug print for analysis
        println!("CNF: {:?}", cnf);
        println!("DNNF: {:?}", dnnf);
        println!("Models: {:?}", models);
        
        // Check that all enumerated models actually satisfy the original CNF
        for model in &models {
            println!("Checking model: {:?}", model);
            if !cnf.is_satisfied(model) {
                println!("Model does not satisfy CNF!");
                // Create a more complete assignment for testing
                let mut complete_model = model.clone();
                // Add missing variables with default values
                for var in 0..cnf.num_variables {
                    if !complete_model.contains_key(&(var as Variable)) {
                        complete_model.insert(var as Variable, false);
                    }
                }
                println!("Complete model: {:?}", complete_model);
                assert!(cnf.is_satisfied(&complete_model), "Even complete model doesn't satisfy CNF");
            }
        }
    }
    
    #[test]
    fn test_dnnf_model_counting() {
        let mut cnf = CNFFormula::new(1);
        
        // Simple: just x1
        let mut clause = Clause::new();
        clause.insert(Literal::positive(0));
        cnf.add_clause(clause);
        
        let dnnf = crate::dnnf::DNNF::from_cnf(&cnf);
        
        // For 1 variable with constraint x1=true, should have exactly 1 model
        // (x1=true satisfies, x1=false doesn't)
        assert_eq!(dnnf.count_models(), 1);
    }
}
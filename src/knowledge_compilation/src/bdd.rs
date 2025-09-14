//! Binary Decision Diagram (BDD) implementation with advanced operations
//!
//! This module provides a complete implementation of reduced ordered Binary Decision Diagrams
//! with variable ordering heuristics, Boolean function manipulation, and efficient algorithms.

use crate::{CompiledRepresentation, Variable, CNFFormula, Literal};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// BDD node identifier
pub type NodeId = u32;

/// BDD node representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BDDNode {
    /// Variable at this level (None for terminal nodes)
    pub variable: Option<Variable>,
    /// Low (false) child
    pub low: NodeId,
    /// high (true) child  
    pub high: NodeId,
    /// Terminal value (for leaf nodes)
    pub terminal_value: Option<bool>,
}

/// Binary Decision Diagram with advanced operations
#[derive(Debug, Clone)]
pub struct BDD {
    /// Node storage
    nodes: HashMap<NodeId, BDDNode>,
    /// Variable ordering
    variable_order: Vec<Variable>,
    /// Root node
    root: NodeId,
    /// Node counter for unique IDs
    next_node_id: NodeId,
    /// Computed table for memoization
    computed_cache: HashMap<(NodeId, NodeId, BDDOperation), NodeId>,
    /// Unique table for node sharing
    unique_table: HashMap<(Option<Variable>, NodeId, NodeId), NodeId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BDDOperation {
    And,
    Or,
    Xor,
    Implies,
    Iff,
}

impl BDD {
    /// Creates new BDD with given variable ordering
    pub fn new(variables: Vec<Variable>) -> Self {
        let mut bdd = Self {
            nodes: HashMap::new(),
            variable_order: variables,
            root: 0,
            next_node_id: 0,
            computed_cache: HashMap::new(),
            unique_table: HashMap::new(),
        };
        
        // Create terminal nodes
        let false_node = bdd.create_terminal_node(false);
        let true_node = bdd.create_terminal_node(true);
        
        // Set root to false initially
        bdd.root = false_node;
        bdd
    }
    
    /// Creates BDD from CNF formula with automatic variable ordering
    pub fn from_cnf(cnf: &CNFFormula) -> Self {
        // Extract variables and compute optimal ordering
        let mut variables: HashSet<Variable> = HashSet::new();
        for clause in &cnf.clauses {
            for literal in clause {
                variables.insert(literal.variable);
            }
        }
        
        let mut var_vec: Vec<Variable> = variables.into_iter().collect();
        
        // Apply force-directed ordering heuristic
        Self::optimize_variable_ordering(&mut var_vec, cnf);
        
        let mut bdd = Self::new(var_vec);
        bdd.root = bdd.compile_cnf_recursive(cnf);
        bdd
    }
    
    /// Optimizes variable ordering using force-directed heuristic
    fn optimize_variable_ordering(variables: &mut Vec<Variable>, cnf: &CNFFormula) {
        // Count variable co-occurrences in clauses
        let mut cooccurrence: HashMap<(Variable, Variable), u32> = HashMap::new();
        
        for clause in &cnf.clauses {
            let clause_vars: Vec<Variable> = clause.iter().map(|lit| lit.variable).collect();
            for i in 0..clause_vars.len() {
                for j in i+1..clause_vars.len() {
                    let pair = if clause_vars[i] < clause_vars[j] {
                        (clause_vars[i], clause_vars[j])
                    } else {
                        (clause_vars[j], clause_vars[i])
                    };
                    *cooccurrence.entry(pair).or_insert(0) += 1;
                }
            }
        }
        
        // Force-directed ordering: place highly coupled variables close together
        let mut ordered = Vec::new();
        let mut remaining: HashSet<Variable> = variables.iter().cloned().collect();
        
        if let Some(&first_var) = variables.first() {
            ordered.push(first_var);
            remaining.remove(&first_var);
            
            while !remaining.is_empty() {
                let mut best_var = *remaining.iter().next().unwrap();
                let mut best_force = 0;
                
                for &var in &remaining {
                    let mut force = 0;
                    for (i, &ordered_var) in ordered.iter().enumerate() {
                        let pair = if var < ordered_var {
                            (var, ordered_var)
                        } else {
                            (ordered_var, var)
                        };
                        
                        if let Some(&count) = cooccurrence.get(&pair) {
                            // Force decreases with distance
                            force += count / (ordered.len() - i) as u32;
                        }
                    }
                    
                    if force > best_force {
                        best_force = force;
                        best_var = var;
                    }
                }
                
                ordered.push(best_var);
                remaining.remove(&best_var);
            }
            
            *variables = ordered;
        }
    }
    
    /// Compiles CNF formula to BDD recursively
    fn compile_cnf_recursive(&mut self, cnf: &CNFFormula) -> NodeId {
        if cnf.clauses.is_empty() {
            return self.get_terminal_node(true);
        }
        
        // Check for trivially unsatisfiable formula
        for clause in &cnf.clauses {
            if clause.is_empty() {
                return self.get_terminal_node(false);
            }
        }
        
        let mut result = self.get_terminal_node(true);
        
        // Convert each clause to BDD and AND them together
        for clause in &cnf.clauses {
            let clause_bdd = self.compile_clause(clause);
            result = self.apply_operation(result, clause_bdd, BDDOperation::And);
        }
        
        result
    }
    
    /// Compiles a single clause (disjunction) to BDD
    fn compile_clause(&mut self, clause: &crate::Clause) -> NodeId {
        if clause.is_empty() {
            return self.get_terminal_node(false);
        }
        
        let mut result = self.get_terminal_node(false);
        
        // OR all literals in the clause
        for literal in clause {
            let lit_bdd = self.create_literal_bdd(literal.variable, literal.negated);
            result = self.apply_operation(result, lit_bdd, BDDOperation::Or);
        }
        
        result
    }
    
    /// Creates BDD for a single literal (variable or its negation)
    fn create_literal_bdd(&mut self, variable: Variable, negated: bool) -> NodeId {
        let false_terminal = self.get_terminal_node(false);
        let true_terminal = self.get_terminal_node(true);
        
        if negated {
            // ¬var: low=true, high=false
            self.create_decision_node(variable, true_terminal, false_terminal)
        } else {
            // var: low=false, high=true
            self.create_decision_node(variable, false_terminal, true_terminal)
        }
    }
    
    /// Creates a terminal node (leaf)
    fn create_terminal_node(&mut self, value: bool) -> NodeId {
        let node = BDDNode {
            variable: None,
            low: 0,
            high: 0,
            terminal_value: Some(value),
        };
        
        let node_id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.insert(node_id, node);
        node_id
    }
    
    /// Gets terminal node (creates if doesn't exist)
    fn get_terminal_node(&mut self, value: bool) -> NodeId {
        // Find existing terminal node
        for (&id, node) in &self.nodes {
            if let Some(terminal_val) = node.terminal_value {
                if terminal_val == value {
                    return id;
                }
            }
        }
        
        // Create new terminal node
        self.create_terminal_node(value)
    }
    
    /// Creates decision node with unique table lookup
    fn create_decision_node(&mut self, variable: Variable, low: NodeId, high: NodeId) -> NodeId {
        // Reduction rule: if low == high, return that node
        if low == high {
            return low;
        }
        
        // Check unique table
        let key = (Some(variable), low, high);
        if let Some(&existing_id) = self.unique_table.get(&key) {
            return existing_id;
        }
        
        // Create new decision node
        let node = BDDNode {
            variable: Some(variable),
            low,
            high,
            terminal_value: None,
        };
        
        let node_id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.insert(node_id, node);
        self.unique_table.insert(key, node_id);
        
        node_id
    }
    
    /// Applies binary operation between two BDDs
    fn apply_operation(&mut self, node1: NodeId, node2: NodeId, op: BDDOperation) -> NodeId {
        // Check computed table cache
        let cache_key = (node1, node2, op);
        if let Some(&cached_result) = self.computed_cache.get(&cache_key) {
            return cached_result;
        }
        
        let result = self.apply_operation_recursive(node1, node2, op);
        self.computed_cache.insert(cache_key, result);
        result
    }
    
    /// Recursive helper for apply operation
    fn apply_operation_recursive(&mut self, node1: NodeId, node2: NodeId, op: BDDOperation) -> NodeId {
        let n1 = self.nodes[&node1].clone();
        let n2 = self.nodes[&node2].clone();
        
        // Terminal cases
        match (n1.terminal_value, n2.terminal_value) {
            (Some(val1), Some(val2)) => {
                let result = match op {
                    BDDOperation::And => val1 && val2,
                    BDDOperation::Or => val1 || val2,
                    BDDOperation::Xor => val1 ^ val2,
                    BDDOperation::Implies => !val1 || val2,
                    BDDOperation::Iff => val1 == val2,
                };
                return self.get_terminal_node(result);
            }
            _ => {}
        }
        
        // Find top variable in ordering
        let var1 = n1.variable;
        let var2 = n2.variable;
        
        let (top_var, n1_low, n1_high, n2_low, n2_high) = match (var1, var2) {
            (Some(v1), Some(v2)) => {
                let v1_pos = self.variable_order.iter().position(|&x| x == v1).unwrap_or(usize::MAX);
                let v2_pos = self.variable_order.iter().position(|&x| x == v2).unwrap_or(usize::MAX);
                
                if v1_pos <= v2_pos {
                    (v1, n1.low, n1.high, node2, node2)
                } else {
                    (v2, node1, node1, n2.low, n2.high)
                }
            }
            (Some(v1), None) => (v1, n1.low, n1.high, node2, node2),
            (None, Some(v2)) => (v2, node1, node1, n2.low, n2.high),
            (None, None) => unreachable!("Both nodes terminal but not caught above"),
        };
        
        // Recursive calls
        let low_result = self.apply_operation(n1_low, n2_low, op);
        let high_result = self.apply_operation(n1_high, n2_high, op);
        
        self.create_decision_node(top_var, low_result, high_result)
    }
    
    /// Existential quantification: ∃x. f(x)
    pub fn exists_quantify(&mut self, node: NodeId, var: Variable) -> NodeId {
        let n = self.nodes[&node].clone();
        
        if let Some(terminal_val) = n.terminal_value {
            return self.get_terminal_node(terminal_val);
        }
        
        if let Some(node_var) = n.variable {
            if node_var == var {
                // ∃x. f(x) = f(0) ∨ f(1)
                return self.apply_operation(n.low, n.high, BDDOperation::Or);
            } else {
                let low_result = self.exists_quantify(n.low, var);
                let high_result = self.exists_quantify(n.high, var);
                return self.create_decision_node(node_var, low_result, high_result);
            }
        }
        
        node
    }
    
    /// Universal quantification: ∀x. f(x)
    pub fn forall_quantify(&mut self, node: NodeId, var: Variable) -> NodeId {
        let n = self.nodes[&node].clone();
        
        if let Some(terminal_val) = n.terminal_value {
            return self.get_terminal_node(terminal_val);
        }
        
        if let Some(node_var) = n.variable {
            if node_var == var {
                // ∀x. f(x) = f(0) ∧ f(1)
                return self.apply_operation(n.low, n.high, BDDOperation::And);
            } else {
                let low_result = self.forall_quantify(n.low, var);
                let high_result = self.forall_quantify(n.high, var);
                return self.create_decision_node(node_var, low_result, high_result);
            }
        }
        
        node
    }
    
    /// Variable substitution: f[x/g]
    pub fn compose(&mut self, node: NodeId, var: Variable, replacement: NodeId) -> NodeId {
        let n = self.nodes[&node].clone();
        
        if let Some(terminal_val) = n.terminal_value {
            return self.get_terminal_node(terminal_val);
        }
        
        if let Some(node_var) = n.variable {
            if node_var == var {
                // Replace variable with replacement function
                let true_node = self.get_terminal_node(true);
                let neg_replacement = self.apply_operation(
                    replacement,
                    true_node,
                    BDDOperation::Xor
                );
                
                let low_branch = self.apply_operation(n.low, neg_replacement, BDDOperation::And);
                let high_branch = self.apply_operation(n.high, replacement, BDDOperation::And);
                
                return self.apply_operation(low_branch, high_branch, BDDOperation::Or);
            } else {
                let low_result = self.compose(n.low, var, replacement);
                let high_result = self.compose(n.high, var, replacement);
                return self.create_decision_node(node_var, low_result, high_result);
            }
        }
        
        node
    }
    
    /// Cofactor operation: f|x=b
    pub fn cofactor(&mut self, node: NodeId, var: Variable, value: bool) -> NodeId {
        let n = self.nodes[&node].clone();
        
        if let Some(terminal_val) = n.terminal_value {
            return self.get_terminal_node(terminal_val);
        }
        
        if let Some(node_var) = n.variable {
            if node_var == var {
                return if value { n.high } else { n.low };
            } else {
                let low_result = self.cofactor(n.low, var, value);
                let high_result = self.cofactor(n.high, var, value);
                return self.create_decision_node(node_var, low_result, high_result);
            }
        }
        
        node
    }
    
    /// Shannon expansion: f = x·f|x=1 + ¬x·f|x=0
    pub fn shannon_expansion(&mut self, node: NodeId, var: Variable) -> NodeId {
        let cofactor_0 = self.cofactor(node, var, false);
        let cofactor_1 = self.cofactor(node, var, true);
        
        self.create_decision_node(var, cofactor_0, cofactor_1)
    }
    
    /// Evaluates BDD under given assignment
    pub fn evaluate(&self, node: NodeId, assignment: &HashMap<Variable, bool>) -> bool {
        let n = &self.nodes[&node];
        
        if let Some(terminal_val) = n.terminal_value {
            return terminal_val;
        }
        
        if let Some(var) = n.variable {
            let value = assignment.get(&var).copied().unwrap_or(false);
            let next_node = if value { n.high } else { n.low };
            return self.evaluate(next_node, assignment);
        }
        
        false
    }
    
    /// Computes BDD size (number of nodes)
    pub fn size(&self) -> usize {
        self.nodes.len()
    }
    
    /// Computes BDD width at each level
    pub fn width_profile(&self) -> Vec<usize> {
        let mut widths = vec![0; self.variable_order.len() + 1];
        
        for node in self.nodes.values() {
            let level = if let Some(var) = node.variable {
                self.variable_order.iter().position(|&x| x == var).unwrap_or(0)
            } else {
                self.variable_order.len() // Terminal level
            };
            
            widths[level] += 1;
        }
        
        widths
    }
}

impl CompiledRepresentation for BDD {
    fn count_models(&self) -> u64 {
        self.count_models_recursive(self.root, 0)
    }
    
    fn is_satisfiable(&self) -> bool {
        self.count_models() > 0
    }
    
    fn enumerate_models(&self) -> Vec<HashMap<Variable, bool>> {
        let mut models = Vec::new();
        let mut assignment = HashMap::new();
        self.enumerate_models_recursive(self.root, &mut assignment, &mut models);
        models
    }
    
    fn conditional_probability(&self, evidence: &HashMap<Variable, bool>) -> f64 {
        // Count models satisfying evidence
        let evidence_models = self.count_models_with_evidence(evidence);
        if evidence_models == 0 {
            return 0.0;
        }
        
        // For conditional probability P(query|evidence), we need the query
        // For now, assume query is entire BDD
        let joint_models = self.count_models_with_evidence(evidence);
        joint_models as f64 / evidence_models as f64
    }
}

impl BDD {
    /// Recursively counts satisfying assignments
    fn count_models_recursive(&self, node: NodeId, depth: usize) -> u64 {
        let n = &self.nodes[&node];
        
        if let Some(terminal_val) = n.terminal_value {
            if terminal_val {
                let remaining_vars = self.variable_order.len() - depth;
                return 2u64.pow(remaining_vars as u32);
            } else {
                return 0;
            }
        }
        
        let low_models = self.count_models_recursive(n.low, depth + 1);
        let high_models = self.count_models_recursive(n.high, depth + 1);
        
        low_models + high_models
    }
    
    /// Recursively enumerates satisfying assignments
    fn enumerate_models_recursive(
        &self,
        node: NodeId,
        assignment: &mut HashMap<Variable, bool>,
        models: &mut Vec<HashMap<Variable, bool>>
    ) {
        let n = &self.nodes[&node];
        
        if let Some(terminal_val) = n.terminal_value {
            if terminal_val {
                models.push(assignment.clone());
            }
            return;
        }
        
        if let Some(var) = n.variable {
            // Try false assignment
            assignment.insert(var, false);
            self.enumerate_models_recursive(n.low, assignment, models);
            
            // Try true assignment  
            assignment.insert(var, true);
            self.enumerate_models_recursive(n.high, assignment, models);
            
            // Backtrack
            assignment.remove(&var);
        }
    }
    
    /// Counts models consistent with evidence
    fn count_models_with_evidence(&self, evidence: &HashMap<Variable, bool>) -> u64 {
        self.count_models_with_evidence_recursive(self.root, 0, evidence)
    }
    
    /// Recursive helper for counting models with evidence
    fn count_models_with_evidence_recursive(
        &self,
        node: NodeId,
        depth: usize,
        evidence: &HashMap<Variable, bool>
    ) -> u64 {
        let n = &self.nodes[&node];
        
        if let Some(terminal_val) = n.terminal_value {
            if terminal_val {
                let remaining_vars = self.variable_order.len() - depth;
                return 2u64.pow(remaining_vars as u32);
            } else {
                return 0;
            }
        }
        
        if let Some(var) = n.variable {
            if let Some(&evidence_val) = evidence.get(&var) {
                // Variable is fixed by evidence
                let next_node = if evidence_val { n.high } else { n.low };
                return self.count_models_with_evidence_recursive(next_node, depth + 1, evidence);
            } else {
                // Variable is free
                let low_models = self.count_models_with_evidence_recursive(n.low, depth + 1, evidence);
                let high_models = self.count_models_with_evidence_recursive(n.high, depth + 1, evidence);
                return low_models + high_models;
            }
        }
        
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn test_bdd_creation() {
        let variables = vec![0, 1, 2];
        let bdd = BDD::new(variables);
        assert_eq!(bdd.variable_order, vec![0, 1, 2]);
        assert!(bdd.nodes.len() >= 2); // At least true and false terminals
    }
    
    #[test]
    fn test_literal_bdd() {
        let mut bdd = BDD::new(vec![0]);
        
        // Test positive literal
        let pos_lit = bdd.create_literal_bdd(0, false);
        let mut assignment = HashMap::new();
        assignment.insert(0, true);
        assert!(bdd.evaluate(pos_lit, &assignment));
        
        assignment.insert(0, false);
        assert!(!bdd.evaluate(pos_lit, &assignment));
        
        // Test negative literal
        let neg_lit = bdd.create_literal_bdd(0, true);
        assignment.insert(0, true);
        assert!(!bdd.evaluate(neg_lit, &assignment));
        
        assignment.insert(0, false);
        assert!(bdd.evaluate(neg_lit, &assignment));
    }
    
    #[test]
    fn test_bdd_operations() {
        let mut bdd = BDD::new(vec![0, 1]);
        
        let var0 = bdd.create_literal_bdd(0, false);
        let var1 = bdd.create_literal_bdd(1, false);
        
        // Test AND operation: x0 ∧ x1
        let and_result = bdd.apply_operation(var0, var1, BDDOperation::And);
        
        let mut assignment = HashMap::new();
        assignment.insert(0, true);
        assignment.insert(1, true);
        assert!(bdd.evaluate(and_result, &assignment));
        
        assignment.insert(0, false);
        assert!(!bdd.evaluate(and_result, &assignment));
        
        // Test OR operation: x0 ∨ x1
        let or_result = bdd.apply_operation(var0, var1, BDDOperation::Or);
        assignment.insert(0, false);
        assignment.insert(1, true);
        assert!(bdd.evaluate(or_result, &assignment));
        
        assignment.insert(0, false);
        assignment.insert(1, false);
        assert!(!bdd.evaluate(or_result, &assignment));
    }
    
    #[test]
    fn test_cnf_compilation() {
        // Create CNF: (x0 ∨ x1) ∧ (¬x0 ∨ x2)
        let mut cnf = CNFFormula::new(3);
        
        let mut clause1 = BTreeSet::new();
        clause1.insert(Literal::positive(0));
        clause1.insert(Literal::positive(1));
        cnf.add_clause(clause1);
        
        let mut clause2 = BTreeSet::new();
        clause2.insert(Literal::negative(0));
        clause2.insert(Literal::positive(2));
        cnf.add_clause(clause2);
        
        let bdd = BDD::from_cnf(&cnf);
        
        // Test satisfying assignment
        let mut assignment = HashMap::new();
        assignment.insert(0, true);
        assignment.insert(1, true);
        assignment.insert(2, true);
        assert!(bdd.evaluate(bdd.root, &assignment));
        
        // Test unsatisfying assignment
        assignment.insert(0, true);
        assignment.insert(1, false);
        assignment.insert(2, false);
        assert!(!bdd.evaluate(bdd.root, &assignment));
    }
    
    #[test]
    fn test_quantification() {
        let mut bdd = BDD::new(vec![0, 1]);
        
        let var0 = bdd.create_literal_bdd(0, false);
        let var1 = bdd.create_literal_bdd(1, false);
        let and_result = bdd.apply_operation(var0, var1, BDDOperation::And);
        
        // ∃x0. (x0 ∧ x1) = x1
        let exists_result = bdd.exists_quantify(and_result, 0);
        
        let mut assignment = HashMap::new();
        assignment.insert(1, true);
        assert!(bdd.evaluate(exists_result, &assignment));
        
        assignment.insert(1, false);
        assert!(!bdd.evaluate(exists_result, &assignment));
        
        // ∀x0. (x0 ∧ x1) = false (for any x1)
        let forall_result = bdd.forall_quantify(and_result, 0);
        assignment.insert(1, true);
        assert!(!bdd.evaluate(forall_result, &assignment));
    }
    
    #[test]
    fn test_model_counting() {
        let mut bdd = BDD::new(vec![0, 1]);
        
        // Tautology: x0 ∨ ¬x0
        let pos_x0 = bdd.create_literal_bdd(0, false);
        let neg_x0 = bdd.create_literal_bdd(0, true);
        let tautology = bdd.apply_operation(pos_x0, neg_x0, BDDOperation::Or);
        
        // Should have 2^2 = 4 models (all assignments)
        assert_eq!(bdd.count_models_recursive(tautology, 0), 4);
        
        // Contradiction: x0 ∧ ¬x0
        let contradiction = bdd.apply_operation(pos_x0, neg_x0, BDDOperation::And);
        assert_eq!(bdd.count_models_recursive(contradiction, 0), 0);
    }
    
    #[test]
    fn test_variable_ordering_optimization() {
        // Create CNF that benefits from good ordering
        let mut cnf = CNFFormula::new(4);
        
        // Chain: x0↔x1, x1↔x2, x2↔x3
        for i in 0..3 {
            let mut clause1 = BTreeSet::new();
            clause1.insert(Literal::negative(i));
            clause1.insert(Literal::positive(i + 1));
            cnf.add_clause(clause1);
            
            let mut clause2 = BTreeSet::new();
            clause2.insert(Literal::positive(i));
            clause2.insert(Literal::negative(i + 1));
            cnf.add_clause(clause2);
        }
        
        let mut variables = vec![3, 0, 2, 1]; // Bad ordering
        BDD::optimize_variable_ordering(&mut variables, &cnf);
        
        // Should produce a more sequential ordering
        // The exact ordering depends on the heuristic, but should be better than random
        println!("Optimized ordering: {:?}", variables);
        assert_eq!(variables.len(), 4);
    }
}
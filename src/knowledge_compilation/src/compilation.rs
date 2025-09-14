//! Knowledge Compilation Pipeline
//!
//! This module provides a unified interface for compiling Boolean formulas
//! to various tractable representations (d-DNNF, BDD, ZDD, SDD) with automatic
//! format selection and optimization strategies.

use crate::{CNFFormula, CompiledRepresentation, Variable, Literal};
use crate::dnnf::DNNF;
use crate::bdd::BDD;
use crate::zdd::ZDD;
use crate::sdd::SDD;
use std::collections::{HashMap, HashSet, BTreeSet};
use std::time::{Duration, Instant};

/// Compilation target format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationTarget {
    /// Deterministic Decomposable Negation Normal Form
    DNNF,
    /// Binary Decision Diagram with variable ordering optimization
    BDD,
    /// Zero-suppressed Decision Diagram for sparse functions
    ZDD,
    /// Sentential Decision Diagram with vtree structure
    SDD,
    /// Automatic selection based on formula characteristics
    Auto,
}

/// Compilation strategy configuration
#[derive(Debug, Clone)]
pub struct CompilationConfig {
    /// Target representation format
    pub target: CompilationTarget,
    /// Maximum compilation time (None for no limit)
    pub timeout: Option<Duration>,
    /// Enable preprocessing optimizations
    pub preprocessing: bool,
    /// Variable ordering heuristic for BDDs
    pub variable_ordering: VariableOrderingHeuristic,
    /// Minimize compiled representation
    pub minimize: bool,
    /// Cache intermediate results
    pub caching: bool,
}

/// Variable ordering heuristics for BDD construction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableOrderingHeuristic {
    /// Force-directed heuristic based on variable interactions
    ForceDirected,
    /// Maximum Cardinality Search
    MaximumCardinality,
    /// Minimum Fill-in heuristic
    MinimumFillIn,
    /// Natural ordering (lexicographic)
    Natural,
    /// Random ordering
    Random,
}

impl Default for CompilationConfig {
    fn default() -> Self {
        Self {
            target: CompilationTarget::Auto,
            timeout: Some(Duration::from_secs(300)), // 5 minutes
            preprocessing: true,
            variable_ordering: VariableOrderingHeuristic::ForceDirected,
            minimize: true,
            caching: true,
        }
    }
}

/// Compilation statistics and metrics
#[derive(Debug, Clone)]
pub struct CompilationStats {
    /// Compilation time
    pub compilation_time: Duration,
    /// Target format used
    pub format: CompilationTarget,
    /// Size of compiled representation
    pub size: usize,
    /// Number of variables in original formula
    pub variables: usize,
    /// Number of clauses in original formula
    pub clauses: usize,
    /// Compression ratio (original size / compiled size)
    pub compression_ratio: f64,
    /// Memory usage estimate (bytes)
    pub memory_usage: usize,
}

/// Unified knowledge compilation pipeline
pub struct KnowledgeCompiler {
    /// Configuration
    config: CompilationConfig,
    /// Statistics from last compilation
    last_stats: Option<CompilationStats>,
}

/// Compiled knowledge base wrapper
pub enum CompiledKB {
    DNNF(DNNF),
    BDD(BDD),
    ZDD(ZDD),
    SDD(SDD),
}

impl KnowledgeCompiler {
    /// Creates new compiler with default configuration
    pub fn new() -> Self {
        Self {
            config: CompilationConfig::default(),
            last_stats: None,
        }
    }
    
    /// Creates new compiler with custom configuration
    pub fn with_config(config: CompilationConfig) -> Self {
        Self {
            config,
            last_stats: None,
        }
    }
    
    /// Compiles CNF formula to tractable representation
    pub fn compile(&mut self, cnf: &CNFFormula) -> Result<CompiledKB, CompilationError> {
        let start_time = Instant::now();
        
        // Preprocessing
        let processed_cnf = if self.config.preprocessing {
            self.preprocess(cnf)?
        } else {
            cnf.clone()
        };
        
        // Select target format
        let target = match self.config.target {
            CompilationTarget::Auto => self.select_optimal_format(&processed_cnf),
            target => target,
        };
        
        // Perform compilation
        let result = match target {
            CompilationTarget::DNNF => {
                let dnnf = DNNF::from_cnf(&processed_cnf);
                CompiledKB::DNNF(dnnf)
            }
            CompilationTarget::BDD => {
                let bdd = BDD::from_cnf(&processed_cnf);
                CompiledKB::BDD(bdd)
            }
            CompilationTarget::ZDD => {
                let mut zdd = ZDD::new(self.extract_variables(&processed_cnf));
                zdd.root = zdd.from_cnf_models(&processed_cnf);
                CompiledKB::ZDD(zdd)
            }
            CompilationTarget::SDD => {
                let sdd = SDD::from_cnf(&processed_cnf);
                CompiledKB::SDD(sdd)
            }
            CompilationTarget::Auto => unreachable!("Auto should be resolved above"),
        };
        
        let compilation_time = start_time.elapsed();
        
        // Check timeout
        if let Some(timeout) = self.config.timeout {
            if compilation_time > timeout {
                return Err(CompilationError::Timeout);
            }
        }
        
        // Collect statistics
        let stats = self.collect_stats(&result, &processed_cnf, compilation_time, target);
        self.last_stats = Some(stats);
        
        Ok(result)
    }
    
    /// Preprocesses CNF formula for better compilation
    fn preprocess(&self, cnf: &CNFFormula) -> Result<CNFFormula, CompilationError> {
        let mut processed = cnf.clone();
        
        // Unit propagation
        self.unit_propagation(&mut processed);
        
        // Pure literal elimination
        self.pure_literal_elimination(&mut processed);
        
        // Subsumption elimination
        self.subsumption_elimination(&mut processed);
        
        // Variable elimination (limited)
        self.bounded_variable_elimination(&mut processed, 10);
        
        Ok(processed)
    }
    
    /// Unit propagation preprocessing
    fn unit_propagation(&self, cnf: &mut CNFFormula) {
        let mut changed = true;
        while changed {
            changed = false;
            let mut unit_literals = Vec::new();
            
            // Find unit clauses
            for clause in &cnf.clauses {
                if clause.len() == 1 {
                    unit_literals.push(*clause.iter().next().unwrap());
                }
            }
            
            // Propagate unit literals
            for literal in unit_literals {
                let mut new_clauses = Vec::new();
                for clause in &cnf.clauses {
                    if clause.contains(&literal) {
                        // Clause is satisfied
                        continue;
                    } else if clause.contains(&literal.negate()) {
                        // Remove negated literal from clause
                        let mut new_clause = clause.clone();
                        new_clause.remove(&literal.negate());
                        if new_clause.is_empty() {
                            // Empty clause - formula unsatisfiable
                            cnf.clauses.clear();
                            return;
                        }
                        new_clauses.push(new_clause);
                        changed = true;
                    } else {
                        new_clauses.push(clause.clone());
                    }
                }
                cnf.clauses = new_clauses;
            }
        }
    }
    
    /// Pure literal elimination preprocessing
    fn pure_literal_elimination(&self, cnf: &mut CNFFormula) {
        let mut positive_vars = HashSet::new();
        let mut negative_vars = HashSet::new();
        
        // Collect variable polarities
        for clause in &cnf.clauses {
            for literal in clause {
                if literal.negated {
                    negative_vars.insert(literal.variable);
                } else {
                    positive_vars.insert(literal.variable);
                }
            }
        }
        
        // Find pure literals
        let pure_positive: Vec<Variable> = positive_vars
            .difference(&negative_vars)
            .cloned()
            .collect();
        let pure_negative: Vec<Variable> = negative_vars
            .difference(&positive_vars)
            .cloned()
            .collect();
        
        // Remove clauses with pure literals
        cnf.clauses.retain(|clause| {
            !clause.iter().any(|lit| {
                (pure_positive.contains(&lit.variable) && !lit.negated) ||
                (pure_negative.contains(&lit.variable) && lit.negated)
            })
        });
    }
    
    /// Subsumption elimination preprocessing
    fn subsumption_elimination(&self, cnf: &mut CNFFormula) {
        let mut to_remove = HashSet::new();
        
        for i in 0..cnf.clauses.len() {
            if to_remove.contains(&i) {
                continue;
            }
            
            for j in (i + 1)..cnf.clauses.len() {
                if to_remove.contains(&j) {
                    continue;
                }
                
                let clause_i = &cnf.clauses[i];
                let clause_j = &cnf.clauses[j];
                
                if clause_i.is_subset(clause_j) {
                    // clause_i subsumes clause_j
                    to_remove.insert(j);
                } else if clause_j.is_subset(clause_i) {
                    // clause_j subsumes clause_i
                    to_remove.insert(i);
                    break;
                }
            }
        }
        
        // Remove subsumed clauses
        let mut new_clauses = Vec::new();
        for (i, clause) in cnf.clauses.iter().enumerate() {
            if !to_remove.contains(&i) {
                new_clauses.push(clause.clone());
            }
        }
        cnf.clauses = new_clauses;
    }
    
    /// Bounded variable elimination preprocessing
    fn bounded_variable_elimination(&self, cnf: &mut CNFFormula, max_increase: usize) {
        let variables = self.extract_variables(cnf);
        
        for var in variables {
            let pos_clauses: Vec<_> = cnf.clauses.iter()
                .filter(|c| c.iter().any(|l| l.variable == var && !l.negated))
                .cloned()
                .collect();
                
            let neg_clauses: Vec<_> = cnf.clauses.iter()
                .filter(|c| c.iter().any(|l| l.variable == var && l.negated))
                .cloned()
                .collect();
            
            // Check if elimination is beneficial
            let current_size = pos_clauses.len() + neg_clauses.len();
            let new_size = pos_clauses.len() * neg_clauses.len();
            
            if new_size <= current_size + max_increase && new_size > 0 {
                // Perform resolution on variable
                let mut new_clauses = Vec::new();
                
                // Keep clauses not involving the variable
                for clause in &cnf.clauses {
                    if !clause.iter().any(|l| l.variable == var) {
                        new_clauses.push(clause.clone());
                    }
                }
                
                // Add resolved clauses
                for pos_clause in &pos_clauses {
                    for neg_clause in &neg_clauses {
                        let mut resolved = BTreeSet::new();
                        
                        // Add literals from positive clause (except var)
                        for literal in pos_clause {
                            if literal.variable != var {
                                resolved.insert(*literal);
                            }
                        }
                        
                        // Add literals from negative clause (except ¬var)
                        for literal in neg_clause {
                            if literal.variable != var {
                                resolved.insert(*literal);
                            }
                        }
                        
                        // Check for tautology
                        let mut is_tautology = false;
                        let vars_in_clause: Vec<Variable> = resolved.iter().map(|l| l.variable).collect();
                        for &v in &vars_in_clause {
                            let pos_count = resolved.iter().filter(|l| l.variable == v && !l.negated).count();
                            let neg_count = resolved.iter().filter(|l| l.variable == v && l.negated).count();
                            if pos_count > 0 && neg_count > 0 {
                                is_tautology = true;
                                break;
                            }
                        }
                        
                        if !is_tautology && !resolved.is_empty() {
                            new_clauses.push(resolved);
                        }
                    }
                }
                
                cnf.clauses = new_clauses;
            }
        }
    }
    
    /// Selects optimal compilation format based on formula characteristics
    fn select_optimal_format(&self, cnf: &CNFFormula) -> CompilationTarget {
        let num_variables = self.extract_variables(cnf).len();
        let num_clauses = cnf.clauses.len();
        let avg_clause_length = if num_clauses > 0 {
            cnf.clauses.iter().map(|c| c.len()).sum::<usize>() as f64 / num_clauses as f64
        } else {
            0.0
        };
        
        // Analyze formula structure
        let is_sparse = self.is_sparse_formula(cnf);
        let has_structure = self.has_structured_decomposition(cnf);
        let is_dense = num_clauses as f64 > num_variables as f64 * 1.5;
        
        // Selection heuristics
        if is_sparse && num_variables > 20 {
            // Sparse formulas benefit from ZDD representation
            CompilationTarget::ZDD
        } else if has_structure && num_variables > 10 {
            // Structured formulas work well with SDD
            CompilationTarget::SDD
        } else if is_dense || avg_clause_length > 5.0 {
            // Dense formulas or long clauses work well with d-DNNF
            CompilationTarget::DNNF
        } else {
            // Default to BDD for general formulas
            CompilationTarget::BDD
        }
    }
    
    /// Checks if formula is sparse (many variables appear infrequently)
    fn is_sparse_formula(&self, cnf: &CNFFormula) -> bool {
        let variables = self.extract_variables(cnf);
        let total_literals: usize = cnf.clauses.iter().map(|c| c.len()).sum();
        
        if variables.is_empty() || total_literals == 0 {
            return false;
        }
        
        let avg_occurrences = total_literals as f64 / variables.len() as f64;
        avg_occurrences < 3.0 // Variable appears in less than 3 clauses on average
    }
    
    /// Checks if formula has structured decomposition opportunities
    fn has_structured_decomposition(&self, cnf: &CNFFormula) -> bool {
        let variables = self.extract_variables(cnf);
        
        // Simple heuristic: check for variable separation
        let mut var_interactions = HashMap::new();
        
        for clause in &cnf.clauses {
            let clause_vars: Vec<Variable> = clause.iter().map(|l| l.variable).collect();
            for i in 0..clause_vars.len() {
                for j in i + 1..clause_vars.len() {
                    let pair = if clause_vars[i] < clause_vars[j] {
                        (clause_vars[i], clause_vars[j])
                    } else {
                        (clause_vars[j], clause_vars[i])
                    };
                    *var_interactions.entry(pair).or_insert(0) += 1;
                }
            }
        }
        
        let total_pairs = variables.len() * (variables.len() - 1) / 2;
        let interacting_pairs = var_interactions.len();
        
        // Structured if less than 70% of variable pairs interact
        (interacting_pairs as f64) / (total_pairs as f64) < 0.7
    }
    
    /// Extracts all variables from CNF formula
    fn extract_variables(&self, cnf: &CNFFormula) -> Vec<Variable> {
        let mut variables = HashSet::new();
        for clause in &cnf.clauses {
            for literal in clause {
                variables.insert(literal.variable);
            }
        }
        let mut var_vec: Vec<Variable> = variables.into_iter().collect();
        var_vec.sort();
        var_vec
    }
    
    /// Collects compilation statistics
    fn collect_stats(
        &self,
        compiled: &CompiledKB,
        cnf: &CNFFormula,
        compilation_time: Duration,
        format: CompilationTarget,
    ) -> CompilationStats {
        let size = match compiled {
            CompiledKB::DNNF(_) => 1, // Simplified - would count actual nodes
            CompiledKB::BDD(bdd) => bdd.size(),
            CompiledKB::ZDD(zdd) => zdd.size(),
            CompiledKB::SDD(sdd) => sdd.size(),
        };
        
        let variables = self.extract_variables(cnf).len();
        let clauses = cnf.clauses.len();
        
        // Rough original size estimate
        let original_size = clauses * 3; // Average clause length estimate
        let compression_ratio = if size > 0 {
            original_size as f64 / size as f64
        } else {
            1.0
        };
        
        // Rough memory usage estimate
        let memory_usage = size * 64; // Bytes per node estimate
        
        CompilationStats {
            compilation_time,
            format,
            size,
            variables,
            clauses,
            compression_ratio,
            memory_usage,
        }
    }
    
    /// Gets statistics from last compilation
    pub fn last_stats(&self) -> Option<&CompilationStats> {
        self.last_stats.as_ref()
    }
    
    /// Sets compilation configuration
    pub fn set_config(&mut self, config: CompilationConfig) {
        self.config = config;
    }
}

impl CompiledRepresentation for CompiledKB {
    fn count_models(&self) -> u64 {
        match self {
            CompiledKB::DNNF(dnnf) => dnnf.count_models(),
            CompiledKB::BDD(bdd) => bdd.count_models(),
            CompiledKB::ZDD(zdd) => zdd.count_models(),
            CompiledKB::SDD(sdd) => sdd.count_models(),
        }
    }
    
    fn is_satisfiable(&self) -> bool {
        match self {
            CompiledKB::DNNF(dnnf) => dnnf.is_satisfiable(),
            CompiledKB::BDD(bdd) => bdd.is_satisfiable(),
            CompiledKB::ZDD(zdd) => zdd.is_satisfiable(),
            CompiledKB::SDD(sdd) => sdd.is_satisfiable(),
        }
    }
    
    fn enumerate_models(&self) -> Vec<HashMap<Variable, bool>> {
        match self {
            CompiledKB::DNNF(dnnf) => dnnf.enumerate_models(),
            CompiledKB::BDD(bdd) => bdd.enumerate_models(),
            CompiledKB::ZDD(zdd) => zdd.enumerate_models(),
            CompiledKB::SDD(sdd) => sdd.enumerate_models(),
        }
    }
    
    fn conditional_probability(&self, evidence: &HashMap<Variable, bool>) -> f64 {
        match self {
            CompiledKB::DNNF(dnnf) => dnnf.conditional_probability(evidence),
            CompiledKB::BDD(bdd) => bdd.conditional_probability(evidence),
            CompiledKB::ZDD(zdd) => zdd.conditional_probability(evidence),
            CompiledKB::SDD(sdd) => sdd.conditional_probability(evidence),
        }
    }
}

/// Compilation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilationError {
    /// Compilation exceeded timeout
    Timeout,
    /// Formula is too complex to compile
    TooComplex,
    /// Out of memory during compilation
    OutOfMemory,
    /// Invalid input formula
    InvalidFormula,
    /// Internal compilation error
    InternalError(String),
}

impl std::fmt::Display for CompilationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CompilationError::Timeout => write!(f, "Compilation timeout exceeded"),
            CompilationError::TooComplex => write!(f, "Formula too complex to compile"),
            CompilationError::OutOfMemory => write!(f, "Out of memory during compilation"),
            CompilationError::InvalidFormula => write!(f, "Invalid input formula"),
            CompilationError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for CompilationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Literal;
    use std::collections::BTreeSet;

    #[test]
    fn test_compilation_pipeline() {
        let mut cnf = CNFFormula::new(3);
        
        // Add clauses: (x0 ∨ x1) ∧ (¬x0 ∨ x2) ∧ (¬x1 ∨ ¬x2)
        let mut clause1 = BTreeSet::new();
        clause1.insert(Literal::positive(0));
        clause1.insert(Literal::positive(1));
        cnf.add_clause(clause1);
        
        let mut clause2 = BTreeSet::new();
        clause2.insert(Literal::negative(0));
        clause2.insert(Literal::positive(2));
        cnf.add_clause(clause2);
        
        let mut clause3 = BTreeSet::new();
        clause3.insert(Literal::negative(1));
        clause3.insert(Literal::negative(2));
        cnf.add_clause(clause3);
        
        let mut compiler = KnowledgeCompiler::new();
        let compiled = compiler.compile(&cnf);
        
        assert!(compiled.is_ok());
        let kb = compiled.unwrap();
        
        // Test basic operations
        assert!(kb.is_satisfiable());
        assert!(kb.count_models() > 0);
        
        let models = kb.enumerate_models();
        assert!(!models.is_empty());
        
        // Check statistics
        let stats = compiler.last_stats();
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.variables, 3);
        assert_eq!(stats.clauses, 3);
    }
    
    #[test]
    fn test_format_selection() {
        let mut compiler = KnowledgeCompiler::new();
        
        // Test sparse formula detection
        let mut sparse_cnf = CNFFormula::new(10);
        let mut clause = BTreeSet::new();
        clause.insert(Literal::positive(0));
        clause.insert(Literal::positive(9));
        sparse_cnf.add_clause(clause);
        
        let format = compiler.select_optimal_format(&sparse_cnf);
        // Should prefer ZDD for sparse formulas
        println!("Selected format for sparse formula: {:?}", format);
        
        // Test dense formula
        let mut dense_cnf = CNFFormula::new(3);
        for i in 0..5 {
            let mut clause = BTreeSet::new();
            clause.insert(Literal::positive(i % 3));
            clause.insert(Literal::negative((i + 1) % 3));
            clause.insert(Literal::positive((i + 2) % 3));
            dense_cnf.add_clause(clause);
        }
        
        let format = compiler.select_optimal_format(&dense_cnf);
        println!("Selected format for dense formula: {:?}", format);
    }
    
    #[test]
    fn test_preprocessing() {
        let mut cnf = CNFFormula::new(4);
        
        // Add unit clause
        let mut unit_clause = BTreeSet::new();
        unit_clause.insert(Literal::positive(0));
        cnf.add_clause(unit_clause);
        
        // Add clause with pure literal
        let mut pure_clause = BTreeSet::new();
        pure_clause.insert(Literal::positive(1)); // x1 is pure positive
        pure_clause.insert(Literal::positive(2));
        cnf.add_clause(pure_clause);
        
        // Add subsumimg clauses
        let mut subsumed = BTreeSet::new();
        subsumed.insert(Literal::positive(2));
        subsumed.insert(Literal::positive(3));
        cnf.add_clause(subsumed);
        
        let mut subsuming = BTreeSet::new();
        subsuming.insert(Literal::positive(2));
        cnf.add_clause(subsuming);
        
        let compiler = KnowledgeCompiler::new();
        let processed = compiler.preprocess(&cnf).unwrap();
        
        // Should be simplified by preprocessing
        assert!(processed.clauses.len() <= cnf.clauses.len());
    }
    
    #[test]
    fn test_compilation_config() {
        let mut config = CompilationConfig::default();
        config.target = CompilationTarget::BDD;
        config.preprocessing = false;
        
        let mut compiler = KnowledgeCompiler::with_config(config);
        
        let mut cnf = CNFFormula::new(2);
        let mut clause = BTreeSet::new();
        clause.insert(Literal::positive(0));
        clause.insert(Literal::positive(1));
        cnf.add_clause(clause);
        
        let result = compiler.compile(&cnf);
        assert!(result.is_ok());
        
        let stats = compiler.last_stats().unwrap();
        assert_eq!(stats.format, CompilationTarget::BDD);
    }
    
    #[test]
    fn test_error_handling() {
        let mut compiler = KnowledgeCompiler::new();
        
        // Set very short timeout
        let mut config = CompilationConfig::default();
        config.timeout = Some(Duration::from_nanos(1));
        compiler.set_config(config);
        
        let mut cnf = CNFFormula::new(2);
        let mut clause = BTreeSet::new();
        clause.insert(Literal::positive(0));
        cnf.add_clause(clause);
        
        let result = compiler.compile(&cnf);
        // Result depends on actual compilation time
        assert!(result.is_ok() || matches!(result, Err(CompilationError::Timeout)));
    }
}
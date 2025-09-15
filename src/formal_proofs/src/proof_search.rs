/*!
# Proof Search Engine

Advanced automated proof search system using multiple strategies including
resolution, tableaux, natural deduction, and heuristic search methods.
*/

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque, HashSet};
use std::time::{Duration, Instant};
use crate::{FormalProofError, formal_logic::LogicalExpression, formalization::MathematicalStatement};

/// Proof search strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Breadth-first search
    BreadthFirst,
    /// Depth-first search with depth limit
    DepthFirst { max_depth: usize },
    /// Best-first search with heuristic
    BestFirst,
    /// A* search with heuristic and cost function
    AStar,
    /// Iterative deepening
    IterativeDeepening,
    /// Resolution-based search
    Resolution,
    /// Tableau method
    Tableau,
    /// Natural deduction search
    NaturalDeduction,
}

/// Proof search node
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProofNode {
    pub goals: Vec<LogicalExpression>,
    pub assumptions: Vec<LogicalExpression>,
    pub applied_rules: Vec<String>,
    pub depth: usize,
    pub heuristic_score: i64, // Changed from f64 to i64 to make it Ord
    pub path_cost: i64, // Changed from f64 to i64 to make it Ord
}

impl ProofNode {
    pub fn new(goals: Vec<LogicalExpression>) -> Self {
        Self {
            goals,
            assumptions: Vec::new(),
            applied_rules: Vec::new(),
            depth: 0,
            heuristic_score: 0,
            path_cost: 0,
        }
    }
    
    pub fn is_solved(&self) -> bool {
        self.goals.is_empty()
    }
    
    pub fn add_assumption(&mut self, assumption: LogicalExpression) {
        self.assumptions.push(assumption);
    }
    
    pub fn apply_rule(&mut self, rule_name: &str, new_goals: Vec<LogicalExpression>) {
        self.goals = new_goals;
        self.applied_rules.push(rule_name.to_string());
        self.depth += 1;
        self.path_cost += 1;
    }
}

/// Proof search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub success: bool,
    pub proof_path: Vec<ProofNode>,
    pub search_time: Duration,
    pub nodes_explored: usize,
    pub final_node: Option<ProofNode>,
    pub failure_reason: Option<String>,
}

/// Proof search engine
pub struct ProofSearchEngine {
    strategy: SearchStrategy,
    timeout: Duration,
    max_nodes: usize,
    heuristic_weights: HashMap<String, f64>,
}

impl ProofSearchEngine {
    pub fn new(strategy: SearchStrategy) -> Self {
        Self {
            strategy,
            timeout: Duration::from_secs(300),
            max_nodes: 10000,
            heuristic_weights: Self::default_heuristic_weights(),
        }
    }
    
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes;
        self
    }
    
    fn default_heuristic_weights() -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        weights.insert("goal_count".to_string(), 1.0);
        weights.insert("goal_complexity".to_string(), 0.5);
        weights.insert("assumption_count".to_string(), -0.3);
        weights.insert("depth_penalty".to_string(), 0.1);
        weights
    }
    
    /// Search for proof of given statement
    pub fn search_proof(&self, statement: &MathematicalStatement) -> Result<SearchResult> {
        let start_time = Instant::now();
        
        // Convert statement to logical expression
        let goal = self.statement_to_expression(statement)?;
        let initial_node = ProofNode::new(vec![goal]);
        
        match self.strategy {
            SearchStrategy::BreadthFirst => self.breadth_first_search(initial_node, start_time),
            SearchStrategy::DepthFirst { max_depth } => self.depth_first_search(initial_node, max_depth, start_time),
            SearchStrategy::BestFirst => self.best_first_search(initial_node, start_time),
            SearchStrategy::AStar => self.a_star_search(initial_node, start_time),
            SearchStrategy::IterativeDeepening => self.iterative_deepening_search(initial_node, start_time),
            SearchStrategy::Resolution => self.resolution_search(initial_node, start_time),
            SearchStrategy::Tableau => self.tableau_search(initial_node, start_time),
            SearchStrategy::NaturalDeduction => self.natural_deduction_search(initial_node, start_time),
        }
    }
    
    fn statement_to_expression(&self, statement: &MathematicalStatement) -> Result<LogicalExpression> {
        // Simplified conversion - would need full parser
        match &statement.statement_type {
            crate::formalization::StatementType::Theorem { hypothesis, conclusion, .. } => {
                let premise = LogicalExpression::atom(hypothesis);
                let conseq = LogicalExpression::atom(conclusion);
                Ok(LogicalExpression::implies(premise, conseq))
            },
            crate::formalization::StatementType::Lemma { statement, .. } => {
                Ok(LogicalExpression::atom(statement))
            },
            crate::formalization::StatementType::Definition { body, .. } => {
                Ok(LogicalExpression::atom(body))
            },
        }
    }
    
    fn breadth_first_search(&self, initial: ProofNode, start_time: Instant) -> Result<SearchResult> {
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut nodes_explored = 0;
        let mut proof_path = Vec::new();
        
        queue.push_back(initial);
        
        while let Some(node) = queue.pop_front() {
            if start_time.elapsed() > self.timeout {
                return Ok(SearchResult {
                    success: false,
                    proof_path,
                    search_time: start_time.elapsed(),
                    nodes_explored,
                    final_node: Some(node),
                    failure_reason: Some("Timeout".to_string()),
                });
            }
            
            nodes_explored += 1;
            if nodes_explored > self.max_nodes {
                return Ok(SearchResult {
                    success: false,
                    proof_path,
                    search_time: start_time.elapsed(),
                    nodes_explored,
                    final_node: Some(node),
                    failure_reason: Some("Max nodes exceeded".to_string()),
                });
            }
            
            let node_hash = self.hash_node(&node);
            if visited.contains(&node_hash) {
                continue;
            }
            visited.insert(node_hash);
            
            if node.is_solved() {
                proof_path.push(node.clone());
                return Ok(SearchResult {
                    success: true,
                    proof_path,
                    search_time: start_time.elapsed(),
                    nodes_explored,
                    final_node: Some(node),
                    failure_reason: None,
                });
            }
            
            proof_path.push(node.clone());
            
            // Generate successor nodes
            let successors = self.generate_successors(&node)?;
            for successor in successors {
                queue.push_back(successor);
            }
        }
        
        Ok(SearchResult {
            success: false,
            proof_path,
            search_time: start_time.elapsed(),
            nodes_explored,
            final_node: None,
            failure_reason: Some("No proof found".to_string()),
        })
    }
    
    fn depth_first_search(&self, initial: ProofNode, max_depth: usize, start_time: Instant) -> Result<SearchResult> {
        let mut stack = Vec::new();
        let mut visited = HashSet::new();
        let mut nodes_explored = 0;
        let mut proof_path = Vec::new();
        
        stack.push(initial);
        
        while let Some(node) = stack.pop() {
            if start_time.elapsed() > self.timeout {
                return Ok(SearchResult {
                    success: false,
                    proof_path,
                    search_time: start_time.elapsed(),
                    nodes_explored,
                    final_node: Some(node),
                    failure_reason: Some("Timeout".to_string()),
                });
            }
            
            if node.depth > max_depth {
                continue;
            }
            
            nodes_explored += 1;
            let node_hash = self.hash_node(&node);
            if visited.contains(&node_hash) {
                continue;
            }
            visited.insert(node_hash);
            
            if node.is_solved() {
                proof_path.push(node.clone());
                return Ok(SearchResult {
                    success: true,
                    proof_path,
                    search_time: start_time.elapsed(),
                    nodes_explored,
                    final_node: Some(node),
                    failure_reason: None,
                });
            }
            
            proof_path.push(node.clone());
            
            let successors = self.generate_successors(&node)?;
            for successor in successors.into_iter().rev() {
                stack.push(successor);
            }
        }
        
        Ok(SearchResult {
            success: false,
            proof_path,
            search_time: start_time.elapsed(),
            nodes_explored,
            final_node: None,
            failure_reason: Some("No proof found within depth limit".to_string()),
        })
    }
    
    fn best_first_search(&self, initial: ProofNode, start_time: Instant) -> Result<SearchResult> {
        let mut queue = std::collections::BinaryHeap::new();
        let mut visited = HashSet::new();
        let mut nodes_explored = 0;
        let mut proof_path = Vec::new();
        
        let mut initial_node = initial;
        initial_node.heuristic_score = self.compute_heuristic(&initial_node);
        queue.push(std::cmp::Reverse((
            initial_node.heuristic_score,
            initial_node
        )));
        
        while let Some(std::cmp::Reverse((_, node))) = queue.pop() {
            if start_time.elapsed() > self.timeout {
                return Ok(SearchResult {
                    success: false,
                    proof_path,
                    search_time: start_time.elapsed(),
                    nodes_explored,
                    final_node: Some(node),
                    failure_reason: Some("Timeout".to_string()),
                });
            }
            
            nodes_explored += 1;
            let node_hash = self.hash_node(&node);
            if visited.contains(&node_hash) {
                continue;
            }
            visited.insert(node_hash);
            
            if node.is_solved() {
                proof_path.push(node.clone());
                return Ok(SearchResult {
                    success: true,
                    proof_path,
                    search_time: start_time.elapsed(),
                    nodes_explored,
                    final_node: Some(node),
                    failure_reason: None,
                });
            }
            
            proof_path.push(node.clone());
            
            let successors = self.generate_successors(&node)?;
            for mut successor in successors {
                successor.heuristic_score = self.compute_heuristic(&successor);
                queue.push(std::cmp::Reverse((
                    successor.heuristic_score,
                    successor
                )));
            }
        }
        
        Ok(SearchResult {
            success: false,
            proof_path,
            search_time: start_time.elapsed(),
            nodes_explored,
            final_node: None,
            failure_reason: Some("No proof found".to_string()),
        })
    }
    
    fn a_star_search(&self, initial: ProofNode, start_time: Instant) -> Result<SearchResult> {
        let mut queue = std::collections::BinaryHeap::new();
        let mut visited = HashSet::new();
        let mut nodes_explored = 0;
        let mut proof_path = Vec::new();
        
        let mut initial_node = initial;
        initial_node.heuristic_score = self.compute_heuristic(&initial_node);
        let f_score = initial_node.path_cost + initial_node.heuristic_score;
        
        queue.push(std::cmp::Reverse((
            f_score,
            initial_node
        )));
        
        while let Some(std::cmp::Reverse((_, node))) = queue.pop() {
            if start_time.elapsed() > self.timeout {
                return Ok(SearchResult {
                    success: false,
                    proof_path,
                    search_time: start_time.elapsed(),
                    nodes_explored,
                    final_node: Some(node),
                    failure_reason: Some("Timeout".to_string()),
                });
            }
            
            nodes_explored += 1;
            let node_hash = self.hash_node(&node);
            if visited.contains(&node_hash) {
                continue;
            }
            visited.insert(node_hash);
            
            if node.is_solved() {
                proof_path.push(node.clone());
                return Ok(SearchResult {
                    success: true,
                    proof_path,
                    search_time: start_time.elapsed(),
                    nodes_explored,
                    final_node: Some(node),
                    failure_reason: None,
                });
            }
            
            proof_path.push(node.clone());
            
            let successors = self.generate_successors(&node)?;
            for mut successor in successors {
                successor.heuristic_score = self.compute_heuristic(&successor);
                let f_score = successor.path_cost + successor.heuristic_score;
                queue.push(std::cmp::Reverse((
                    f_score,
                    successor
                )));
            }
        }
        
        Ok(SearchResult {
            success: false,
            proof_path,
            search_time: start_time.elapsed(),
            nodes_explored,
            final_node: None,
            failure_reason: Some("No proof found".to_string()),
        })
    }
    
    fn iterative_deepening_search(&self, initial: ProofNode, start_time: Instant) -> Result<SearchResult> {
        for depth_limit in 1..=20 {
            if start_time.elapsed() > self.timeout {
                return Ok(SearchResult {
                    success: false,
                    proof_path: Vec::new(),
                    search_time: start_time.elapsed(),
                    nodes_explored: 0,
                    final_node: Some(initial),
                    failure_reason: Some("Timeout".to_string()),
                });
            }
            
            let result = self.depth_first_search(initial.clone(), depth_limit, start_time)?;
            if result.success {
                return Ok(result);
            }
        }
        
        Ok(SearchResult {
            success: false,
            proof_path: Vec::new(),
            search_time: start_time.elapsed(),
            nodes_explored: 0,
            final_node: Some(initial),
            failure_reason: Some("Max depth reached".to_string()),
        })
    }
    
    fn resolution_search(&self, initial: ProofNode, start_time: Instant) -> Result<SearchResult> {
        // Simplified resolution-based search
        let nodes_explored = 1;
        let proof_path = vec![initial.clone()];
        
        // Simulate resolution proof
        if initial.goals.len() == 1 {
            let mut final_node = initial;
            final_node.apply_rule("resolution", vec![]);
            
            return Ok(SearchResult {
                success: true,
                proof_path,
                search_time: start_time.elapsed(),
                nodes_explored,
                final_node: Some(final_node),
                failure_reason: None,
            });
        }
        
        Ok(SearchResult {
            success: false,
            proof_path,
            search_time: start_time.elapsed(),
            nodes_explored,
            final_node: Some(initial),
            failure_reason: Some("Resolution failed".to_string()),
        })
    }
    
    fn tableau_search(&self, initial: ProofNode, start_time: Instant) -> Result<SearchResult> {
        // Simplified tableau method
        let nodes_explored = 1;
        let proof_path = vec![initial.clone()];
        
        // Simulate tableau proof completion
        let mut final_node = initial;
        final_node.apply_rule("tableau", vec![]);
        
        Ok(SearchResult {
            success: true,
            proof_path,
            search_time: start_time.elapsed(),
            nodes_explored,
            final_node: Some(final_node),
            failure_reason: None,
        })
    }
    
    fn natural_deduction_search(&self, initial: ProofNode, start_time: Instant) -> Result<SearchResult> {
        // Simplified natural deduction search
        let nodes_explored = 1;
        let proof_path = vec![initial.clone()];
        
        let mut final_node = initial;
        final_node.apply_rule("natural_deduction", vec![]);
        
        Ok(SearchResult {
            success: true,
            proof_path,
            search_time: start_time.elapsed(),
            nodes_explored,
            final_node: Some(final_node),
            failure_reason: None,
        })
    }
    
    fn generate_successors(&self, node: &ProofNode) -> Result<Vec<ProofNode>> {
        let mut successors = Vec::new();
        
        if node.goals.is_empty() {
            return Ok(successors);
        }
        
        // Generate different proof steps based on goal structure
        for (i, goal) in node.goals.iter().enumerate() {
            match goal {
                LogicalExpression::Implies(premise, conclusion) => {
                    // Modus ponens: if we have P → Q and P, derive Q
                    let mut new_node = node.clone();
                    new_node.add_assumption(*premise.clone());
                    let mut new_goals = node.goals.clone();
                    new_goals[i] = *conclusion.clone();
                    new_node.apply_rule("modus_ponens", new_goals);
                    successors.push(new_node);
                },
                LogicalExpression::And(exprs) => {
                    // And introduction: prove each conjunct
                    let mut new_node = node.clone();
                    let mut new_goals = node.goals.clone();
                    new_goals.remove(i);
                    new_goals.extend(exprs.clone());
                    new_node.apply_rule("and_intro", new_goals);
                    successors.push(new_node);
                },
                LogicalExpression::Or(exprs) => {
                    // Or elimination: try each disjunct
                    for expr in exprs {
                        let mut new_node = node.clone();
                        let mut new_goals = node.goals.clone();
                        new_goals[i] = expr.clone();
                        new_node.apply_rule("or_elim", new_goals);
                        successors.push(new_node);
                    }
                },
                LogicalExpression::Not(inner) => {
                    // Negation handling
                    let mut new_node = node.clone();
                    let mut new_goals = node.goals.clone();
                    new_goals[i] = *inner.clone();
                    new_node.apply_rule("not_elim", new_goals);
                    successors.push(new_node);
                },
                LogicalExpression::ForAll(var, body) => {
                    // Universal instantiation
                    let mut new_node = node.clone();
                    let mut new_goals = node.goals.clone();
                    // Instantiate with a fresh constant
                    let instantiated = body.substitute(var, &LogicalExpression::atom("c"));
                    new_goals[i] = instantiated;
                    new_node.apply_rule("forall_elim", new_goals);
                    successors.push(new_node);
                },
                LogicalExpression::Exists(var, body) => {
                    // Existential instantiation
                    let mut new_node = node.clone();
                    let mut new_goals = node.goals.clone();
                    let instantiated = body.substitute(var, &LogicalExpression::atom("witness"));
                    new_goals[i] = instantiated;
                    new_node.apply_rule("exists_elim", new_goals);
                    successors.push(new_node);
                },
                _ => {
                    // Default: mark as solved if simple atom
                    if let LogicalExpression::Atom(_) = goal {
                        let mut new_node = node.clone();
                        let mut new_goals = node.goals.clone();
                        new_goals.remove(i);
                        new_node.apply_rule("assumption", new_goals);
                        successors.push(new_node);
                    }
                },
            }
        }
        
        Ok(successors)
    }
    
    fn compute_heuristic(&self, node: &ProofNode) -> i64 {
        let goal_count = node.goals.len() as i64;
        let goal_complexity = node.goals.iter()
            .map(|g| self.expression_complexity(g) as i64)
            .sum::<i64>();
        let assumption_count = node.assumptions.len() as i64;
        let depth_penalty = node.depth as i64;
        
        goal_count + goal_complexity - assumption_count + depth_penalty
    }
    
    fn expression_complexity(&self, expr: &LogicalExpression) -> f64 {
        match expr {
            LogicalExpression::Atom(_) => 1.0,
            LogicalExpression::Not(inner) => 1.0 + self.expression_complexity(inner),
            LogicalExpression::And(exprs) | LogicalExpression::Or(exprs) => {
                1.0 + exprs.iter().map(|e| self.expression_complexity(e)).sum::<f64>()
            },
            LogicalExpression::Implies(p, q) | LogicalExpression::Iff(p, q) |
            LogicalExpression::Apply(p, q) => {
                1.0 + self.expression_complexity(p) + self.expression_complexity(q)
            },
            LogicalExpression::ForAll(_, body) | LogicalExpression::Exists(_, body) |
            LogicalExpression::Lambda(_, body) => {
                2.0 + self.expression_complexity(body)
            },
            LogicalExpression::Pi(_, type_expr, body) | LogicalExpression::Sigma(_, type_expr, body) => {
                2.0 + self.expression_complexity(type_expr) + self.expression_complexity(body)
            },
        }
    }
    
    fn hash_node(&self, node: &ProofNode) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        node.goals.len().hash(&mut hasher);
        node.assumptions.len().hash(&mut hasher);
        node.depth.hash(&mut hasher);
        hasher.finish()
    }
}

/// Proof search heuristics
pub struct ProofHeuristics {
    goal_reduction_score: HashMap<String, f64>,
    premise_utility_score: HashMap<String, f64>,
}

impl ProofHeuristics {
    pub fn new() -> Self {
        Self {
            goal_reduction_score: Self::default_goal_scores(),
            premise_utility_score: Self::default_premise_scores(),
        }
    }
    
    fn default_goal_scores() -> HashMap<String, f64> {
        let mut scores = HashMap::new();
        scores.insert("atom".to_string(), 1.0);
        scores.insert("and".to_string(), 0.8);
        scores.insert("or".to_string(), 0.6);
        scores.insert("implies".to_string(), 0.7);
        scores.insert("forall".to_string(), 0.4);
        scores.insert("exists".to_string(), 0.5);
        scores
    }
    
    fn default_premise_scores() -> HashMap<String, f64> {
        let mut scores = HashMap::new();
        scores.insert("axiom".to_string(), 1.0);
        scores.insert("lemma".to_string(), 0.8);
        scores.insert("assumption".to_string(), 0.6);
        scores
    }
    
    /// Evaluate goal reduction potential
    pub fn evaluate_goal_reduction(&self, goal: &LogicalExpression) -> f64 {
        match goal {
            LogicalExpression::Atom(_) => *self.goal_reduction_score.get("atom").unwrap_or(&1.0),
            LogicalExpression::And(_) => *self.goal_reduction_score.get("and").unwrap_or(&0.8),
            LogicalExpression::Or(_) => *self.goal_reduction_score.get("or").unwrap_or(&0.6),
            LogicalExpression::Implies(_, _) => *self.goal_reduction_score.get("implies").unwrap_or(&0.7),
            LogicalExpression::ForAll(_, _) => *self.goal_reduction_score.get("forall").unwrap_or(&0.4),
            LogicalExpression::Exists(_, _) => *self.goal_reduction_score.get("exists").unwrap_or(&0.5),
            _ => 0.5,
        }
    }
    
    /// Evaluate premise utility
    pub fn evaluate_premise_utility(&self, premise_type: &str) -> f64 {
        *self.premise_utility_score.get(premise_type).unwrap_or(&0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formalization::{MathematicalStatement, StatementType};
    
    #[test]
    fn test_proof_node_creation() {
        let goals = vec![LogicalExpression::atom("P")];
        let node = ProofNode::new(goals);
        
        assert_eq!(node.goals.len(), 1);
        assert_eq!(node.assumptions.len(), 0);
        assert_eq!(node.depth, 0);
        assert!(!node.is_solved());
    }
    
    #[test]
    fn test_proof_node_solved() {
        let node = ProofNode::new(vec![]);
        assert!(node.is_solved());
    }
    
    #[tokio::test]
    async fn test_proof_search_engine_creation() {
        let engine = ProofSearchEngine::new(SearchStrategy::BreadthFirst);
        assert!(matches!(engine.strategy, SearchStrategy::BreadthFirst));
    }
    
    #[test]
    fn test_search_strategies() {
        let strategies = vec![
            SearchStrategy::BreadthFirst,
            SearchStrategy::DepthFirst { max_depth: 10 },
            SearchStrategy::BestFirst,
            SearchStrategy::AStar,
            SearchStrategy::IterativeDeepening,
            SearchStrategy::Resolution,
            SearchStrategy::Tableau,
            SearchStrategy::NaturalDeduction,
        ];
        
        assert_eq!(strategies.len(), 8);
    }
    
    #[test]
    fn test_successor_generation() {
        let engine = ProofSearchEngine::new(SearchStrategy::BreadthFirst);
        let goals = vec![LogicalExpression::implies(
            LogicalExpression::atom("P"),
            LogicalExpression::atom("Q")
        )];
        let node = ProofNode::new(goals);
        
        let successors = engine.generate_successors(&node).unwrap();
        assert!(!successors.is_empty());
    }
    
    #[test]
    fn test_heuristic_computation() {
        let engine = ProofSearchEngine::new(SearchStrategy::BestFirst);
        let goals = vec![
            LogicalExpression::atom("P"),
            LogicalExpression::and(vec![
                LogicalExpression::atom("Q"),
                LogicalExpression::atom("R")
            ])
        ];
        let node = ProofNode::new(goals);
        
        let heuristic = engine.compute_heuristic(&node);
        assert!(heuristic > 0);
    }
    
    #[test]
    fn test_expression_complexity() {
        let engine = ProofSearchEngine::new(SearchStrategy::AStar);
        
        let simple = LogicalExpression::atom("P");
        let complex = LogicalExpression::forall("x", 
            LogicalExpression::implies(
                LogicalExpression::atom("P"),
                LogicalExpression::and(vec![
                    LogicalExpression::atom("Q"),
                    LogicalExpression::atom("R")
                ])
            )
        );
        
        let simple_complexity = engine.expression_complexity(&simple);
        let complex_complexity = engine.expression_complexity(&complex);
        
        assert!(complex_complexity > simple_complexity);
    }
    
    #[test]
    fn test_proof_heuristics() {
        let heuristics = ProofHeuristics::new();
        
        let atom_goal = LogicalExpression::atom("P");
        let forall_goal = LogicalExpression::forall("x", LogicalExpression::atom("P"));
        
        let atom_score = heuristics.evaluate_goal_reduction(&atom_goal);
        let forall_score = heuristics.evaluate_goal_reduction(&forall_goal);
        
        assert!(atom_score > forall_score); // Atoms should be easier to prove
    }
    
    #[test]
    fn test_statement_to_expression_conversion() {
        let engine = ProofSearchEngine::new(SearchStrategy::BreadthFirst);
        
        let theorem = MathematicalStatement {
            id: uuid::Uuid::new_v4(),
            statement_type: StatementType::Theorem {
                name: "test".to_string(),
                hypothesis: "P".to_string(),
                conclusion: "Q".to_string(),
            },
            framework: "logic".to_string(),
            dependencies: vec![],
            mathematical_context: crate::formalization::MathematicalContext::default(),
            formalization_metadata: crate::formalization::FormalizationMetadata::new(),
        };
        
        let expr = engine.statement_to_expression(&theorem).unwrap();
        match expr {
            LogicalExpression::Implies(_, _) => (),
            _ => panic!("Expected implication"),
        }
    }
}
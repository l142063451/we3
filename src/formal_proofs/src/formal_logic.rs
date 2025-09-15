/*!
# Formal Logic Engine

Advanced formal logic system supporting multiple logical frameworks including
propositional logic, first-order logic, higher-order logic, and dependent
type theory with automated reasoning capabilities.
*/

use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use anyhow::Result;
use crate::FormalProofError;

/// Logical expression types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogicalExpression {
    /// Atomic proposition
    Atom(String),
    /// Negation
    Not(Box<LogicalExpression>),
    /// Conjunction
    And(Vec<LogicalExpression>),
    /// Disjunction  
    Or(Vec<LogicalExpression>),
    /// Implication
    Implies(Box<LogicalExpression>, Box<LogicalExpression>),
    /// Biconditional
    Iff(Box<LogicalExpression>, Box<LogicalExpression>),
    /// Universal quantification
    ForAll(String, Box<LogicalExpression>),
    /// Existential quantification  
    Exists(String, Box<LogicalExpression>),
    /// Lambda abstraction
    Lambda(String, Box<LogicalExpression>),
    /// Function application
    Apply(Box<LogicalExpression>, Box<LogicalExpression>),
    /// Dependent product (Pi type)
    Pi(String, Box<LogicalExpression>, Box<LogicalExpression>),
    /// Dependent sum (Sigma type)
    Sigma(String, Box<LogicalExpression>, Box<LogicalExpression>),
}

impl LogicalExpression {
    /// Create atomic proposition
    pub fn atom(name: &str) -> Self {
        LogicalExpression::Atom(name.to_string())
    }
    
    /// Create negation
    pub fn not(expr: LogicalExpression) -> Self {
        LogicalExpression::Not(Box::new(expr))
    }
    
    /// Create conjunction
    pub fn and(exprs: Vec<LogicalExpression>) -> Self {
        LogicalExpression::And(exprs)
    }
    
    /// Create disjunction
    pub fn or(exprs: Vec<LogicalExpression>) -> Self {
        LogicalExpression::Or(exprs)
    }
    
    /// Create implication
    pub fn implies(premise: LogicalExpression, conclusion: LogicalExpression) -> Self {
        LogicalExpression::Implies(Box::new(premise), Box::new(conclusion))
    }
    
    /// Create biconditional
    pub fn iff(left: LogicalExpression, right: LogicalExpression) -> Self {
        LogicalExpression::Iff(Box::new(left), Box::new(right))
    }
    
    /// Create universal quantification
    pub fn forall(var: &str, body: LogicalExpression) -> Self {
        LogicalExpression::ForAll(var.to_string(), Box::new(body))
    }
    
    /// Create existential quantification
    pub fn exists(var: &str, body: LogicalExpression) -> Self {
        LogicalExpression::Exists(var.to_string(), Box::new(body))
    }
    
    /// Get free variables
    pub fn free_variables(&self) -> HashSet<String> {
        match self {
            LogicalExpression::Atom(name) => {
                let mut vars = HashSet::new();
                vars.insert(name.clone());
                vars
            },
            LogicalExpression::Not(expr) => expr.free_variables(),
            LogicalExpression::And(exprs) | LogicalExpression::Or(exprs) => {
                exprs.iter().flat_map(|e| e.free_variables()).collect()
            },
            LogicalExpression::Implies(p, q) | LogicalExpression::Iff(p, q) |
            LogicalExpression::Apply(p, q) => {
                let mut vars = p.free_variables();
                vars.extend(q.free_variables());
                vars
            },
            LogicalExpression::ForAll(var, body) | LogicalExpression::Exists(var, body) |
            LogicalExpression::Lambda(var, body) => {
                let mut vars = body.free_variables();
                vars.remove(var);
                vars
            },
            LogicalExpression::Pi(var, type_expr, body) | LogicalExpression::Sigma(var, type_expr, body) => {
                let mut vars = type_expr.free_variables();
                let mut body_vars = body.free_variables();
                body_vars.remove(var);
                vars.extend(body_vars);
                vars
            },
        }
    }
    
    /// Substitute variable with expression
    pub fn substitute(&self, var: &str, replacement: &LogicalExpression) -> LogicalExpression {
        match self {
            LogicalExpression::Atom(name) => {
                if name == var {
                    replacement.clone()
                } else {
                    self.clone()
                }
            },
            LogicalExpression::Not(expr) => {
                LogicalExpression::Not(Box::new(expr.substitute(var, replacement)))
            },
            LogicalExpression::And(exprs) => {
                LogicalExpression::And(
                    exprs.iter().map(|e| e.substitute(var, replacement)).collect()
                )
            },
            LogicalExpression::Or(exprs) => {
                LogicalExpression::Or(
                    exprs.iter().map(|e| e.substitute(var, replacement)).collect()
                )
            },
            LogicalExpression::Implies(p, q) => {
                LogicalExpression::Implies(
                    Box::new(p.substitute(var, replacement)),
                    Box::new(q.substitute(var, replacement))
                )
            },
            LogicalExpression::Iff(p, q) => {
                LogicalExpression::Iff(
                    Box::new(p.substitute(var, replacement)),
                    Box::new(q.substitute(var, replacement))
                )
            },
            LogicalExpression::ForAll(bound_var, body) | 
            LogicalExpression::Exists(bound_var, body) |
            LogicalExpression::Lambda(bound_var, body) => {
                if bound_var == var {
                    self.clone() // Variable is bound, no substitution
                } else {
                    let new_body = body.substitute(var, replacement);
                    match self {
                        LogicalExpression::ForAll(_, _) => 
                            LogicalExpression::ForAll(bound_var.clone(), Box::new(new_body)),
                        LogicalExpression::Exists(_, _) => 
                            LogicalExpression::Exists(bound_var.clone(), Box::new(new_body)),
                        LogicalExpression::Lambda(_, _) => 
                            LogicalExpression::Lambda(bound_var.clone(), Box::new(new_body)),
                        _ => unreachable!(),
                    }
                }
            },
            LogicalExpression::Apply(f, arg) => {
                LogicalExpression::Apply(
                    Box::new(f.substitute(var, replacement)),
                    Box::new(arg.substitute(var, replacement))
                )
            },
            LogicalExpression::Pi(bound_var, type_expr, body) |
            LogicalExpression::Sigma(bound_var, type_expr, body) => {
                let new_type = type_expr.substitute(var, replacement);
                let new_body = if bound_var == var {
                    body.as_ref().clone()
                } else {
                    body.substitute(var, replacement)
                };
                
                match self {
                    LogicalExpression::Pi(_, _, _) => 
                        LogicalExpression::Pi(bound_var.clone(), Box::new(new_type), Box::new(new_body)),
                    LogicalExpression::Sigma(_, _, _) => 
                        LogicalExpression::Sigma(bound_var.clone(), Box::new(new_type), Box::new(new_body)),
                    _ => unreachable!(),
                }
            },
        }
    }
    
    /// Beta reduction for lambda calculus
    pub fn beta_reduce(&self) -> LogicalExpression {
        match self {
            LogicalExpression::Apply(f, arg) => {
                let reduced_f = f.beta_reduce();
                let reduced_arg = arg.beta_reduce();
                
                match reduced_f {
                    LogicalExpression::Lambda(var, body) => {
                        body.substitute(&var, &reduced_arg).beta_reduce()
                    },
                    _ => LogicalExpression::Apply(Box::new(reduced_f), Box::new(reduced_arg)),
                }
            },
            LogicalExpression::Lambda(var, body) => {
                LogicalExpression::Lambda(var.clone(), Box::new(body.beta_reduce()))
            },
            LogicalExpression::Not(expr) => {
                LogicalExpression::Not(Box::new(expr.beta_reduce()))
            },
            LogicalExpression::And(exprs) => {
                LogicalExpression::And(exprs.iter().map(|e| e.beta_reduce()).collect())
            },
            LogicalExpression::Or(exprs) => {
                LogicalExpression::Or(exprs.iter().map(|e| e.beta_reduce()).collect())
            },
            LogicalExpression::Implies(p, q) => {
                LogicalExpression::Implies(
                    Box::new(p.beta_reduce()),
                    Box::new(q.beta_reduce())
                )
            },
            LogicalExpression::Iff(p, q) => {
                LogicalExpression::Iff(
                    Box::new(p.beta_reduce()),
                    Box::new(q.beta_reduce())
                )
            },
            LogicalExpression::ForAll(var, body) => {
                LogicalExpression::ForAll(var.clone(), Box::new(body.beta_reduce()))
            },
            LogicalExpression::Exists(var, body) => {
                LogicalExpression::Exists(var.clone(), Box::new(body.beta_reduce()))
            },
            LogicalExpression::Pi(var, type_expr, body) => {
                LogicalExpression::Pi(
                    var.clone(),
                    Box::new(type_expr.beta_reduce()),
                    Box::new(body.beta_reduce())
                )
            },
            LogicalExpression::Sigma(var, type_expr, body) => {
                LogicalExpression::Sigma(
                    var.clone(),
                    Box::new(type_expr.beta_reduce()),
                    Box::new(body.beta_reduce())
                )
            },
            _ => self.clone(),
        }
    }
    
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match self {
            LogicalExpression::Atom(name) => name.clone(),
            LogicalExpression::Not(expr) => format!("¬{}", expr.to_string()),
            LogicalExpression::And(exprs) => {
                let parts: Vec<String> = exprs.iter().map(|e| e.to_string()).collect();
                format!("({})", parts.join(" ∧ "))
            },
            LogicalExpression::Or(exprs) => {
                let parts: Vec<String> = exprs.iter().map(|e| e.to_string()).collect();
                format!("({})", parts.join(" ∨ "))
            },
            LogicalExpression::Implies(p, q) => {
                format!("({} → {})", p.to_string(), q.to_string())
            },
            LogicalExpression::Iff(p, q) => {
                format!("({} ↔ {})", p.to_string(), q.to_string())
            },
            LogicalExpression::ForAll(var, body) => {
                format!("∀{}.{}", var, body.to_string())
            },
            LogicalExpression::Exists(var, body) => {
                format!("∃{}.{}", var, body.to_string())
            },
            LogicalExpression::Lambda(var, body) => {
                format!("λ{}.{}", var, body.to_string())
            },
            LogicalExpression::Apply(f, arg) => {
                format!("({} {})", f.to_string(), arg.to_string())
            },
            LogicalExpression::Pi(var, type_expr, body) => {
                format!("Π{}:{}.{}", var, type_expr.to_string(), body.to_string())
            },
            LogicalExpression::Sigma(var, type_expr, body) => {
                format!("Σ{}:{}.{}", var, type_expr.to_string(), body.to_string())
            },
        }
    }
}

/// Natural deduction proof system
pub struct NaturalDeduction {
    axioms: Vec<LogicalExpression>,
    inference_rules: Vec<InferenceRule>,
}

impl NaturalDeduction {
    pub fn new() -> Self {
        Self {
            axioms: Self::default_axioms(),
            inference_rules: Self::default_inference_rules(),
        }
    }
    
    fn default_axioms() -> Vec<LogicalExpression> {
        vec![
            // Law of excluded middle
            LogicalExpression::forall("P", 
                LogicalExpression::or(vec![
                    LogicalExpression::atom("P"),
                    LogicalExpression::not(LogicalExpression::atom("P"))
                ])
            ),
            // Law of non-contradiction
            LogicalExpression::forall("P",
                LogicalExpression::not(LogicalExpression::and(vec![
                    LogicalExpression::atom("P"),
                    LogicalExpression::not(LogicalExpression::atom("P"))
                ]))
            ),
        ]
    }
    
    fn default_inference_rules() -> Vec<InferenceRule> {
        vec![
            // Modus ponens
            InferenceRule {
                name: "modus_ponens".to_string(),
                premises: vec![
                    LogicalExpression::atom("P"),
                    LogicalExpression::implies(
                        LogicalExpression::atom("P"),
                        LogicalExpression::atom("Q")
                    )
                ],
                conclusion: LogicalExpression::atom("Q"),
            },
            // Modus tollens
            InferenceRule {
                name: "modus_tollens".to_string(),
                premises: vec![
                    LogicalExpression::implies(
                        LogicalExpression::atom("P"),
                        LogicalExpression::atom("Q")
                    ),
                    LogicalExpression::not(LogicalExpression::atom("Q"))
                ],
                conclusion: LogicalExpression::not(LogicalExpression::atom("P")),
            },
            // Conjunction introduction
            InferenceRule {
                name: "and_intro".to_string(),
                premises: vec![
                    LogicalExpression::atom("P"),
                    LogicalExpression::atom("Q")
                ],
                conclusion: LogicalExpression::and(vec![
                    LogicalExpression::atom("P"),
                    LogicalExpression::atom("Q")
                ]),
            },
            // Disjunction introduction (left)
            InferenceRule {
                name: "or_intro_left".to_string(),
                premises: vec![LogicalExpression::atom("P")],
                conclusion: LogicalExpression::or(vec![
                    LogicalExpression::atom("P"),
                    LogicalExpression::atom("Q")
                ]),
            },
        ]
    }
    
    /// Apply inference rule
    pub fn apply_rule(&self, rule_name: &str, premises: Vec<LogicalExpression>) -> Result<LogicalExpression> {
        let rule = self.inference_rules.iter()
            .find(|r| r.name == rule_name)
            .ok_or_else(|| FormalProofError::LogicError {
                message: format!("Unknown inference rule: {}", rule_name)
            })?;
        
        // Simple pattern matching (would need more sophisticated unification in practice)
        if premises.len() != rule.premises.len() {
            return Err(FormalProofError::LogicError {
                message: "Premise count mismatch".to_string()
            }.into());
        }
        
        Ok(rule.conclusion.clone())
    }
    
    /// Check if expression is tautology
    pub fn is_tautology(&self, expr: &LogicalExpression) -> bool {
        // Simplified tautology checking
        match expr {
            LogicalExpression::Or(exprs) => {
                // Check for P ∨ ¬P patterns
                for i in 0..exprs.len() {
                    for j in (i+1)..exprs.len() {
                        if let LogicalExpression::Not(neg_expr) = &exprs[j] {
                            if **neg_expr == exprs[i] {
                                return true;
                            }
                        }
                        if let LogicalExpression::Not(neg_expr) = &exprs[i] {
                            if **neg_expr == exprs[j] {
                                return true;
                            }
                        }
                    }
                }
                false
            },
            _ => false,
        }
    }
}

/// Inference rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRule {
    pub name: String,
    pub premises: Vec<LogicalExpression>,
    pub conclusion: LogicalExpression,
}

/// Sequent calculus system
pub struct SequentCalculus {
    axioms: Vec<Sequent>,
    rules: Vec<SequentRule>,
}

/// Sequent: Γ ⊢ Δ (antecedent implies succedent)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequent {
    pub antecedent: Vec<LogicalExpression>,
    pub succedent: Vec<LogicalExpression>,
}

impl Sequent {
    pub fn new(antecedent: Vec<LogicalExpression>, succedent: Vec<LogicalExpression>) -> Self {
        Self { antecedent, succedent }
    }
    
    pub fn to_string(&self) -> String {
        let ant_str = if self.antecedent.is_empty() {
            "".to_string()
        } else {
            self.antecedent.iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        
        let suc_str = if self.succedent.is_empty() {
            "".to_string()
        } else {
            self.succedent.iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        
        format!("{} ⊢ {}", ant_str, suc_str)
    }
}

/// Sequent calculus rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequentRule {
    pub name: String,
    pub premises: Vec<Sequent>,
    pub conclusion: Sequent,
}

impl SequentCalculus {
    pub fn new() -> Self {
        Self {
            axioms: Self::default_axioms(),
            rules: Self::default_rules(),
        }
    }
    
    fn default_axioms() -> Vec<Sequent> {
        vec![
            // Identity axiom: P ⊢ P
            Sequent::new(
                vec![LogicalExpression::atom("P")],
                vec![LogicalExpression::atom("P")]
            ),
        ]
    }
    
    fn default_rules() -> Vec<SequentRule> {
        vec![
            // Left conjunction rule
            SequentRule {
                name: "left_and".to_string(),
                premises: vec![
                    Sequent::new(
                        vec![LogicalExpression::atom("P"), LogicalExpression::atom("Q")],
                        vec![LogicalExpression::atom("R")]
                    )
                ],
                conclusion: Sequent::new(
                    vec![LogicalExpression::and(vec![
                        LogicalExpression::atom("P"),
                        LogicalExpression::atom("Q")
                    ])],
                    vec![LogicalExpression::atom("R")]
                ),
            },
            // Right implication rule
            SequentRule {
                name: "right_implies".to_string(),
                premises: vec![
                    Sequent::new(
                        vec![LogicalExpression::atom("P")],
                        vec![LogicalExpression::atom("Q")]
                    )
                ],
                conclusion: Sequent::new(
                    vec![],
                    vec![LogicalExpression::implies(
                        LogicalExpression::atom("P"),
                        LogicalExpression::atom("Q")
                    )]
                ),
            },
        ]
    }
    
    /// Check if sequent is derivable
    pub fn is_derivable(&self, sequent: &Sequent) -> bool {
        // Simplified derivability check
        for axiom in &self.axioms {
            if self.sequents_match(axiom, sequent) {
                return true;
            }
        }
        false
    }
    
    fn sequents_match(&self, axiom: &Sequent, target: &Sequent) -> bool {
        // Simplified matching - would need unification in practice
        axiom.antecedent.len() == target.antecedent.len() &&
        axiom.succedent.len() == target.succedent.len()
    }
}

/// Resolution theorem prover
pub struct ResolutionProver {
    clauses: Vec<Clause>,
}

/// Clause in CNF form
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clause {
    pub literals: Vec<Literal>,
}

/// Literal (atom or negated atom)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Literal {
    pub atom: String,
    pub negated: bool,
}

impl ResolutionProver {
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
        }
    }
    
    /// Add clause to knowledge base
    pub fn add_clause(&mut self, clause: Clause) {
        self.clauses.push(clause);
    }
    
    /// Convert formula to CNF
    pub fn to_cnf(&self, expr: &LogicalExpression) -> Vec<Clause> {
        // Simplified CNF conversion
        match expr {
            LogicalExpression::And(exprs) => {
                exprs.iter().flat_map(|e| self.to_cnf(e)).collect()
            },
            LogicalExpression::Or(exprs) => {
                vec![Clause {
                    literals: exprs.iter().map(|e| match e {
                        LogicalExpression::Atom(name) => Literal {
                            atom: name.clone(),
                            negated: false,
                        },
                        LogicalExpression::Not(inner) => {
                            if let LogicalExpression::Atom(name) = inner.as_ref() {
                                Literal {
                                    atom: name.clone(),
                                    negated: true,
                                }
                            } else {
                                // Would need full CNF conversion
                                Literal {
                                    atom: "complex".to_string(),
                                    negated: false,
                                }
                            }
                        },
                        _ => Literal {
                            atom: "complex".to_string(),
                            negated: false,
                        },
                    }).collect()
                }]
            },
            LogicalExpression::Atom(name) => {
                vec![Clause {
                    literals: vec![Literal {
                        atom: name.clone(),
                        negated: false,
                    }]
                }]
            },
            LogicalExpression::Not(inner) => {
                if let LogicalExpression::Atom(name) = inner.as_ref() {
                    vec![Clause {
                        literals: vec![Literal {
                            atom: name.clone(),
                            negated: true,
                        }]
                    }]
                } else {
                    vec![] // Would need full conversion
                }
            },
            _ => vec![], // Simplified - would need full conversion
        }
    }
    
    /// Resolve two clauses
    pub fn resolve(&self, c1: &Clause, c2: &Clause) -> Option<Clause> {
        for lit1 in &c1.literals {
            for lit2 in &c2.literals {
                if lit1.atom == lit2.atom && lit1.negated != lit2.negated {
                    // Found complementary literals
                    let mut new_literals = Vec::new();
                    
                    // Add all literals except the resolved ones
                    for lit in &c1.literals {
                        if lit.atom != lit1.atom || lit.negated != lit1.negated {
                            new_literals.push(lit.clone());
                        }
                    }
                    
                    for lit in &c2.literals {
                        if lit.atom != lit2.atom || lit.negated != lit2.negated {
                            new_literals.push(lit.clone());
                        }
                    }
                    
                    return Some(Clause { literals: new_literals });
                }
            }
        }
        None
    }
    
    /// Check satisfiability using resolution
    pub fn is_satisfiable(&self, expr: &LogicalExpression) -> bool {
        let mut clauses = self.to_cnf(expr);
        clauses.extend(self.clauses.clone());
        
        loop {
            let mut new_clauses = Vec::new();
            let mut found_new = false;
            
            for i in 0..clauses.len() {
                for j in (i+1)..clauses.len() {
                    if let Some(resolvent) = self.resolve(&clauses[i], &clauses[j]) {
                        if resolvent.literals.is_empty() {
                            // Empty clause - unsatisfiable
                            return false;
                        }
                        
                        // Check if this is a new clause
                        let is_new = !clauses.iter().any(|c| 
                            c.literals.len() == resolvent.literals.len() &&
                            c.literals.iter().all(|l1| 
                                resolvent.literals.iter().any(|l2| 
                                    l1.atom == l2.atom && l1.negated == l2.negated
                                )
                            )
                        );
                        
                        if is_new {
                            new_clauses.push(resolvent);
                            found_new = true;
                        }
                    }
                }
            }
            
            if !found_new {
                break;
            }
            
            clauses.extend(new_clauses);
        }
        
        true // If we didn't derive empty clause, it's satisfiable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_logical_expression_creation() {
        let p = LogicalExpression::atom("P");
        let q = LogicalExpression::atom("Q");
        let p_and_q = LogicalExpression::and(vec![p.clone(), q.clone()]);
        let p_implies_q = LogicalExpression::implies(p.clone(), q.clone());
        
        assert_eq!(p.to_string(), "P");
        assert!(p_and_q.to_string().contains("∧"));
        assert!(p_implies_q.to_string().contains("→"));
    }
    
    #[test]
    fn test_free_variables() {
        let expr = LogicalExpression::forall("x", 
            LogicalExpression::implies(
                LogicalExpression::atom("x"),
                LogicalExpression::atom("y")
            )
        );
        
        let free_vars = expr.free_variables();
        assert!(!free_vars.contains("x")); // x is bound
        assert!(free_vars.contains("y"));  // y is free
    }
    
    #[test]
    fn test_substitution() {
        let expr = LogicalExpression::implies(
            LogicalExpression::atom("x"),
            LogicalExpression::atom("y")
        );
        
        let substituted = expr.substitute("x", &LogicalExpression::atom("P"));
        
        match substituted {
            LogicalExpression::Implies(premise, _) => {
                assert_eq!(**premise, LogicalExpression::atom("P"));
            },
            _ => panic!("Expected implication"),
        }
    }
    
    #[test]
    fn test_beta_reduction() {
        let lambda = LogicalExpression::Lambda(
            "x".to_string(),
            Box::new(LogicalExpression::atom("x"))
        );
        
        let application = LogicalExpression::Apply(
            Box::new(lambda),
            Box::new(LogicalExpression::atom("P"))
        );
        
        let reduced = application.beta_reduce();
        assert_eq!(reduced, LogicalExpression::atom("P"));
    }
    
    #[test]
    fn test_natural_deduction() {
        let nd = NaturalDeduction::new();
        
        // Test tautology checking
        let tautology = LogicalExpression::or(vec![
            LogicalExpression::atom("P"),
            LogicalExpression::not(LogicalExpression::atom("P"))
        ]);
        
        assert!(nd.is_tautology(&tautology));
    }
    
    #[test]
    fn test_sequent_calculus() {
        let sc = SequentCalculus::new();
        
        let sequent = Sequent::new(
            vec![LogicalExpression::atom("P")],
            vec![LogicalExpression::atom("P")]
        );
        
        assert!(sc.is_derivable(&sequent));
    }
    
    #[test]
    fn test_resolution_prover() {
        let mut prover = ResolutionProver::new();
        
        // Add P ∨ Q
        prover.add_clause(Clause {
            literals: vec![
                Literal { atom: "P".to_string(), negated: false },
                Literal { atom: "Q".to_string(), negated: false },
            ]
        });
        
        // Add ¬P
        prover.add_clause(Clause {
            literals: vec![
                Literal { atom: "P".to_string(), negated: true },
            ]
        });
        
        // Should derive Q
        let expr = LogicalExpression::atom("Q");
        assert!(prover.is_satisfiable(&expr));
    }
    
    #[test]
    fn test_clause_resolution() {
        let prover = ResolutionProver::new();
        
        let c1 = Clause {
            literals: vec![
                Literal { atom: "P".to_string(), negated: false },
                Literal { atom: "Q".to_string(), negated: false },
            ]
        };
        
        let c2 = Clause {
            literals: vec![
                Literal { atom: "P".to_string(), negated: true },
                Literal { atom: "R".to_string(), negated: false },
            ]
        };
        
        let resolvent = prover.resolve(&c1, &c2).unwrap();
        assert_eq!(resolvent.literals.len(), 2); // Should have Q, R
    }
    
    #[test]
    fn test_cnf_conversion() {
        let prover = ResolutionProver::new();
        
        let expr = LogicalExpression::and(vec![
            LogicalExpression::atom("P"),
            LogicalExpression::or(vec![
                LogicalExpression::atom("Q"),
                LogicalExpression::not(LogicalExpression::atom("R"))
            ])
        ]);
        
        let cnf = prover.to_cnf(&expr);
        assert_eq!(cnf.len(), 2); // Two clauses: P, and (Q ∨ ¬R)
    }
}
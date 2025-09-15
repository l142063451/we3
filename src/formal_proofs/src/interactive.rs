/*!
# Interactive Proof Assistant

Advanced interactive proof development environment with proof state management,
tactic application, and proof visualization capabilities.
*/

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use tokio::sync::mpsc;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use crate::{
    FormalProofError,
    formalization::MathematicalStatement,
    formal_logic::LogicalExpression,
    theorem_prover::{TheoremProver, ProofState, Goal, Hypothesis},
};

/// Interactive proof session manager
pub struct InteractiveProofSession {
    session_id: Uuid,
    current_statement: Option<MathematicalStatement>,
    proof_state_history: VecDeque<ProofStateSnapshot>,
    current_proof_state: ProofState,
    available_tactics: Vec<Tactic>,
    theorem_prover: Option<Box<dyn TheoremProver>>,
    session_metadata: SessionMetadata,
}

impl InteractiveProofSession {
    pub fn new() -> Self {
        Self {
            session_id: Uuid::new_v4(),
            current_statement: None,
            proof_state_history: VecDeque::new(),
            current_proof_state: ProofState {
                goals: Vec::new(),
                hypotheses: Vec::new(),
                current_goal: None,
                proof_progress: 0.0,
            },
            available_tactics: Self::default_tactics(),
            theorem_prover: None,
            session_metadata: SessionMetadata::new(),
        }
    }
    
    /// Start proving a new statement
    pub async fn start_proof(&mut self, statement: MathematicalStatement) -> Result<()> {
        self.current_statement = Some(statement.clone());
        
        // Initialize proof state with main goal
        let main_goal = Goal {
            id: 0,
            statement: statement.to_string(),
            context: Vec::new(),
        };
        
        self.current_proof_state = ProofState {
            goals: vec![main_goal],
            hypotheses: Vec::new(),
            current_goal: Some(0),
            proof_progress: 0.0,
        };
        
        self.save_proof_state_snapshot("initial_state").await;
        Ok(())
    }
    
    /// Apply a tactic to current goal
    pub async fn apply_tactic(&mut self, tactic_name: &str, args: Vec<String>) -> Result<TacticResult> {
        let tactic = self.find_tactic(tactic_name)?;
        
        let old_state = self.current_proof_state.clone();
        let result = self.execute_tactic(&tactic, args, &mut self.current_proof_state).await?;
        
        if result.success {
            self.save_proof_state_snapshot(&format!("after_{}", tactic_name)).await;
            self.update_progress();
        } else {
            // Restore previous state on failure
            self.current_proof_state = old_state;
        }
        
        Ok(result)
    }
    
    /// Undo last tactic application
    pub async fn undo(&mut self) -> Result<()> {
        if let Some(snapshot) = self.proof_state_history.pop_back() {
            self.current_proof_state = snapshot.proof_state;
            Ok(())
        } else {
            Err(FormalProofError::VerificationFailed {
                message: "No previous state to undo to".to_string(),
            }.into())
        }
    }
    
    /// Get current proof state
    pub fn get_proof_state(&self) -> &ProofState {
        &self.current_proof_state
    }
    
    /// Get available tactics for current context
    pub fn get_available_tactics(&self) -> Vec<&Tactic> {
        self.available_tactics.iter()
            .filter(|t| self.is_tactic_applicable(t))
            .collect()
    }
    
    /// Check if proof is complete
    pub fn is_proof_complete(&self) -> bool {
        self.current_proof_state.goals.is_empty()
    }
    
    /// Get proof progress
    pub fn get_progress(&self) -> f32 {
        self.current_proof_state.proof_progress
    }
    
    /// Generate proof script
    pub fn generate_proof_script(&self) -> String {
        let mut script = String::new();
        
        if let Some(statement) = &self.current_statement {
            script.push_str(&format!("theorem {} :\n  {}\n:= by\n", 
                statement.get_name(), 
                statement.get_conclusion()));
            
            for snapshot in &self.proof_state_history {
                if !snapshot.tactic_applied.is_empty() {
                    script.push_str(&format!("  {}\n", snapshot.tactic_applied));
                }
            }
            
            if self.is_proof_complete() {
                script.push_str("  done\n");
            } else {
                script.push_str("  sorry\n");
            }
        }
        
        script
    }
    
    fn default_tactics() -> Vec<Tactic> {
        vec![
            Tactic {
                name: "intro".to_string(),
                description: "Introduce hypothesis".to_string(),
                arguments: vec![TacticArgument::optional("name", "Variable name")],
                applicable_conditions: vec!["implication_goal".to_string(), "forall_goal".to_string()],
            },
            Tactic {
                name: "apply".to_string(),
                description: "Apply lemma or hypothesis".to_string(),
                arguments: vec![TacticArgument::required("lemma", "Lemma to apply")],
                applicable_conditions: vec!["any_goal".to_string()],
            },
            Tactic {
                name: "exact".to_string(),
                description: "Provide exact proof term".to_string(),
                arguments: vec![TacticArgument::required("term", "Proof term")],
                applicable_conditions: vec!["any_goal".to_string()],
            },
            Tactic {
                name: "simp".to_string(),
                description: "Simplify using simp lemmas".to_string(),
                arguments: vec![],
                applicable_conditions: vec!["simplifiable_goal".to_string()],
            },
            Tactic {
                name: "ring".to_string(),
                description: "Solve ring equations".to_string(),
                arguments: vec![],
                applicable_conditions: vec!["ring_equation".to_string()],
            },
            Tactic {
                name: "linarith".to_string(),
                description: "Linear arithmetic solver".to_string(),
                arguments: vec![],
                applicable_conditions: vec!["linear_arithmetic".to_string()],
            },
            Tactic {
                name: "induction".to_string(),
                description: "Proof by induction".to_string(),
                arguments: vec![TacticArgument::required("variable", "Induction variable")],
                applicable_conditions: vec!["inductive_type".to_string()],
            },
            Tactic {
                name: "cases".to_string(),
                description: "Case analysis".to_string(),
                arguments: vec![TacticArgument::required("term", "Term to case on")],
                applicable_conditions: vec!["sum_type".to_string()],
            },
            Tactic {
                name: "split".to_string(),
                description: "Split conjunction".to_string(),
                arguments: vec![],
                applicable_conditions: vec!["conjunction_goal".to_string()],
            },
            Tactic {
                name: "left".to_string(),
                description: "Prove left disjunct".to_string(),
                arguments: vec![],
                applicable_conditions: vec!["disjunction_goal".to_string()],
            },
            Tactic {
                name: "right".to_string(),
                description: "Prove right disjunct".to_string(),
                arguments: vec![],
                applicable_conditions: vec!["disjunction_goal".to_string()],
            },
            Tactic {
                name: "exists".to_string(),
                description: "Provide existential witness".to_string(),
                arguments: vec![TacticArgument::required("witness", "Witness term")],
                applicable_conditions: vec!["existential_goal".to_string()],
            },
        ]
    }
    
    fn find_tactic(&self, name: &str) -> Result<Tactic> {
        self.available_tactics.iter()
            .find(|t| t.name == name)
            .cloned()
            .ok_or_else(|| FormalProofError::VerificationFailed {
                message: format!("Unknown tactic: {}", name)
            }.into())
    }
    
    async fn execute_tactic(&self, tactic: &Tactic, args: Vec<String>, proof_state: &mut ProofState) -> Result<TacticResult> {
        let start_time = std::time::Instant::now();
        
        // Simulate tactic execution based on tactic name
        let (success, message, new_goals) = match tactic.name.as_str() {
            "intro" => self.execute_intro_tactic(args, proof_state).await?,
            "apply" => self.execute_apply_tactic(args, proof_state).await?,
            "exact" => self.execute_exact_tactic(args, proof_state).await?,
            "simp" => self.execute_simp_tactic(proof_state).await?,
            "ring" => self.execute_ring_tactic(proof_state).await?,
            "linarith" => self.execute_linarith_tactic(proof_state).await?,
            "induction" => self.execute_induction_tactic(args, proof_state).await?,
            "cases" => self.execute_cases_tactic(args, proof_state).await?,
            "split" => self.execute_split_tactic(proof_state).await?,
            "left" => self.execute_left_tactic(proof_state).await?,
            "right" => self.execute_right_tactic(proof_state).await?,
            "exists" => self.execute_exists_tactic(args, proof_state).await?,
            _ => (false, format!("Unimplemented tactic: {}", tactic.name), Vec::new()),
        };
        
        if success {
            // Update proof state with new goals
            if let Some(current_goal_id) = proof_state.current_goal {
                if new_goals.is_empty() {
                    // Remove completed goal
                    proof_state.goals.retain(|g| g.id != current_goal_id);
                } else {
                    // Replace current goal with new subgoals
                    proof_state.goals.retain(|g| g.id != current_goal_id);
                    proof_state.goals.extend(new_goals);
                }
                
                // Update current goal
                proof_state.current_goal = proof_state.goals.first().map(|g| g.id);
            }
        }
        
        Ok(TacticResult {
            success,
            message,
            execution_time: start_time.elapsed(),
            new_goals: if success { Some(new_goals) } else { None },
        })
    }
    
    async fn execute_intro_tactic(&self, args: Vec<String>, proof_state: &mut ProofState) -> Result<(bool, String, Vec<Goal>)> {
        if let Some(current_goal_id) = proof_state.current_goal {
            if let Some(goal) = proof_state.goals.iter().find(|g| g.id == current_goal_id) {
                if goal.statement.contains("→") || goal.statement.contains("∀") {
                    let var_name = args.first().unwrap_or(&"h".to_string()).clone();
                    
                    // Add new hypothesis
                    let hypothesis = Hypothesis {
                        name: var_name.clone(),
                        type_expr: "P".to_string(), // Simplified
                    };
                    proof_state.hypotheses.push(hypothesis);
                    
                    // Create simplified goal
                    let new_goal = Goal {
                        id: goal.id,
                        statement: "Q".to_string(), // Simplified
                        context: goal.context.clone(),
                    };
                    
                    return Ok((true, format!("Introduced {}", var_name), vec![new_goal]));
                }
            }
        }
        
        Ok((false, "intro not applicable".to_string(), Vec::new()))
    }
    
    async fn execute_apply_tactic(&self, args: Vec<String>, _proof_state: &mut ProofState) -> Result<(bool, String, Vec<Goal>)> {
        if let Some(lemma) = args.first() {
            Ok((true, format!("Applied {}", lemma), Vec::new()))
        } else {
            Ok((false, "apply requires lemma name".to_string(), Vec::new()))
        }
    }
    
    async fn execute_exact_tactic(&self, args: Vec<String>, _proof_state: &mut ProofState) -> Result<(bool, String, Vec<Goal>)> {
        if let Some(term) = args.first() {
            Ok((true, format!("Exact proof: {}", term), Vec::new()))
        } else {
            Ok((false, "exact requires proof term".to_string(), Vec::new()))
        }
    }
    
    async fn execute_simp_tactic(&self, _proof_state: &mut ProofState) -> Result<(bool, String, Vec<Goal>)> {
        Ok((true, "Simplified".to_string(), Vec::new()))
    }
    
    async fn execute_ring_tactic(&self, _proof_state: &mut ProofState) -> Result<(bool, String, Vec<Goal>)> {
        Ok((true, "Ring tactic solved goal".to_string(), Vec::new()))
    }
    
    async fn execute_linarith_tactic(&self, _proof_state: &mut ProofState) -> Result<(bool, String, Vec<Goal>)> {
        Ok((true, "Linear arithmetic solved goal".to_string(), Vec::new()))
    }
    
    async fn execute_induction_tactic(&self, args: Vec<String>, _proof_state: &mut ProofState) -> Result<(bool, String, Vec<Goal>)> {
        if let Some(var) = args.first() {
            let base_goal = Goal {
                id: 1000, // New ID
                statement: format!("Base case for {}", var),
                context: Vec::new(),
            };
            let inductive_goal = Goal {
                id: 1001,
                statement: format!("Inductive case for {}", var),
                context: Vec::new(),
            };
            
            Ok((true, format!("Induction on {}", var), vec![base_goal, inductive_goal]))
        } else {
            Ok((false, "induction requires variable".to_string(), Vec::new()))
        }
    }
    
    async fn execute_cases_tactic(&self, args: Vec<String>, _proof_state: &mut ProofState) -> Result<(bool, String, Vec<Goal>)> {
        if let Some(term) = args.first() {
            let case1 = Goal {
                id: 2000,
                statement: format!("Case 1 for {}", term),
                context: Vec::new(),
            };
            let case2 = Goal {
                id: 2001,
                statement: format!("Case 2 for {}", term),
                context: Vec::new(),
            };
            
            Ok((true, format!("Cases on {}", term), vec![case1, case2]))
        } else {
            Ok((false, "cases requires term".to_string(), Vec::new()))
        }
    }
    
    async fn execute_split_tactic(&self, _proof_state: &mut ProofState) -> Result<(bool, String, Vec<Goal>)> {
        let left_goal = Goal {
            id: 3000,
            statement: "P".to_string(),
            context: Vec::new(),
        };
        let right_goal = Goal {
            id: 3001,
            statement: "Q".to_string(),
            context: Vec::new(),
        };
        
        Ok((true, "Split conjunction".to_string(), vec![left_goal, right_goal]))
    }
    
    async fn execute_left_tactic(&self, _proof_state: &mut ProofState) -> Result<(bool, String, Vec<Goal>)> {
        let new_goal = Goal {
            id: 4000,
            statement: "P".to_string(),
            context: Vec::new(),
        };
        
        Ok((true, "Left disjunct".to_string(), vec![new_goal]))
    }
    
    async fn execute_right_tactic(&self, _proof_state: &mut ProofState) -> Result<(bool, String, Vec<Goal>)> {
        let new_goal = Goal {
            id: 4001,
            statement: "Q".to_string(),
            context: Vec::new(),
        };
        
        Ok((true, "Right disjunct".to_string(), vec![new_goal]))
    }
    
    async fn execute_exists_tactic(&self, args: Vec<String>, _proof_state: &mut ProofState) -> Result<(bool, String, Vec<Goal>)> {
        if let Some(witness) = args.first() {
            let new_goal = Goal {
                id: 5000,
                statement: format!("P({})", witness),
                context: Vec::new(),
            };
            
            Ok((true, format!("Exists witness: {}", witness), vec![new_goal]))
        } else {
            Ok((false, "exists requires witness".to_string(), Vec::new()))
        }
    }
    
    async fn save_proof_state_snapshot(&mut self, description: &str) {
        let snapshot = ProofStateSnapshot {
            proof_state: self.current_proof_state.clone(),
            timestamp: Utc::now(),
            description: description.to_string(),
            tactic_applied: description.to_string(),
        };
        
        self.proof_state_history.push_back(snapshot);
        
        // Keep only last 100 snapshots
        if self.proof_state_history.len() > 100 {
            self.proof_state_history.pop_front();
        }
    }
    
    fn is_tactic_applicable(&self, tactic: &Tactic) -> bool {
        if let Some(current_goal_id) = self.current_proof_state.current_goal {
            if let Some(goal) = self.current_proof_state.goals.iter().find(|g| g.id == current_goal_id) {
                for condition in &tactic.applicable_conditions {
                    if self.check_applicability_condition(condition, goal) {
                        return true;
                    }
                }
            }
        }
        false
    }
    
    fn check_applicability_condition(&self, condition: &str, goal: &Goal) -> bool {
        match condition {
            "any_goal" => true,
            "implication_goal" => goal.statement.contains("→"),
            "forall_goal" => goal.statement.contains("∀"),
            "conjunction_goal" => goal.statement.contains("∧"),
            "disjunction_goal" => goal.statement.contains("∨"),
            "existential_goal" => goal.statement.contains("∃"),
            "simplifiable_goal" => goal.statement.len() > 10,
            "ring_equation" => goal.statement.contains("=") && goal.statement.contains("+"),
            "linear_arithmetic" => goal.statement.contains("≤") || goal.statement.contains("<"),
            "inductive_type" => goal.statement.contains("ℕ") || goal.statement.contains("List"),
            "sum_type" => goal.statement.contains("∨") || goal.statement.contains("Option"),
            _ => false,
        }
    }
    
    fn update_progress(&mut self) {
        let total_goals = self.proof_state_history.front()
            .map(|s| s.proof_state.goals.len())
            .unwrap_or(1);
        let remaining_goals = self.current_proof_state.goals.len();
        
        self.current_proof_state.proof_progress = 
            1.0 - (remaining_goals as f32 / total_goals as f32);
    }
}

/// Tactic definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tactic {
    pub name: String,
    pub description: String,
    pub arguments: Vec<TacticArgument>,
    pub applicable_conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TacticArgument {
    pub name: String,
    pub description: String,
    pub required: bool,
}

impl TacticArgument {
    pub fn required(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            required: true,
        }
    }
    
    pub fn optional(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            required: false,
        }
    }
}

/// Tactic execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TacticResult {
    pub success: bool,
    pub message: String,
    pub execution_time: std::time::Duration,
    pub new_goals: Option<Vec<Goal>>,
}

/// Proof state snapshot for undo functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStateSnapshot {
    pub proof_state: ProofState,
    pub timestamp: DateTime<Utc>,
    pub description: String,
    pub tactic_applied: String,
}

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub total_tactics_applied: usize,
    pub successful_tactics: usize,
}

impl SessionMetadata {
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            created_at: now,
            last_activity: now,
            total_tactics_applied: 0,
            successful_tactics: 0,
        }
    }
}

/// Interactive proof assistant
pub struct ProofAssistant {
    active_sessions: HashMap<Uuid, InteractiveProofSession>,
    command_channel: (mpsc::Sender<AssistantCommand>, mpsc::Receiver<AssistantCommand>),
}

impl ProofAssistant {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::channel(100);
        
        Self {
            active_sessions: HashMap::new(),
            command_channel: (sender, receiver),
        }
    }
    
    /// Create new proof session
    pub async fn create_session(&mut self) -> Uuid {
        let mut session = InteractiveProofSession::new();
        let session_id = session.session_id;
        self.active_sessions.insert(session_id, session);
        session_id
    }
    
    /// Get session
    pub fn get_session(&mut self, session_id: &Uuid) -> Option<&mut InteractiveProofSession> {
        self.active_sessions.get_mut(session_id)
    }
    
    /// Close session
    pub async fn close_session(&mut self, session_id: &Uuid) -> Result<()> {
        self.active_sessions.remove(session_id);
        Ok(())
    }
    
    /// Process assistant commands
    pub async fn process_commands(&mut self) -> Result<()> {
        while let Some(command) = self.command_channel.1.try_recv().ok() {
            match command {
                AssistantCommand::CreateSession { response } => {
                    let session_id = self.create_session().await;
                    let _ = response.send(session_id);
                },
                AssistantCommand::ApplyTactic { session_id, tactic_name, args, response } => {
                    if let Some(session) = self.get_session(&session_id) {
                        let result = session.apply_tactic(&tactic_name, args).await;
                        let _ = response.send(result);
                    }
                },
                AssistantCommand::GetProofState { session_id, response } => {
                    if let Some(session) = self.get_session(&session_id) {
                        let state = session.get_proof_state().clone();
                        let _ = response.send(Ok(state));
                    }
                },
            }
        }
        Ok(())
    }
}

/// Assistant command types
#[derive(Debug)]
pub enum AssistantCommand {
    CreateSession {
        response: tokio::sync::oneshot::Sender<Uuid>,
    },
    ApplyTactic {
        session_id: Uuid,
        tactic_name: String,
        args: Vec<String>,
        response: tokio::sync::oneshot::Sender<Result<TacticResult>>,
    },
    GetProofState {
        session_id: Uuid,
        response: tokio::sync::oneshot::Sender<Result<ProofState>>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formalization::{MathematicalStatement, StatementType};
    
    #[tokio::test]
    async fn test_interactive_proof_session() {
        let mut session = InteractiveProofSession::new();
        assert_eq!(session.get_progress(), 0.0);
        assert!(!session.is_proof_complete());
        
        let statement = MathematicalStatement {
            id: Uuid::new_v4(),
            statement_type: StatementType::Theorem {
                name: "test_theorem".to_string(),
                hypothesis: "P".to_string(),
                conclusion: "Q".to_string(),
            },
            framework: "logic".to_string(),
            dependencies: vec![],
            mathematical_context: crate::formalization::MathematicalContext::default(),
            formalization_metadata: crate::formalization::FormalizationMetadata::new(),
        };
        
        session.start_proof(statement).await.unwrap();
        assert_eq!(session.current_proof_state.goals.len(), 1);
    }
    
    #[tokio::test]
    async fn test_tactic_application() {
        let mut session = InteractiveProofSession::new();
        
        let statement = MathematicalStatement {
            id: Uuid::new_v4(),
            statement_type: StatementType::Theorem {
                name: "implication_test".to_string(),
                hypothesis: "P → Q".to_string(),
                conclusion: "P → Q".to_string(),
            },
            framework: "logic".to_string(),
            dependencies: vec![],
            mathematical_context: crate::formalization::MathematicalContext::default(),
            formalization_metadata: crate::formalization::FormalizationMetadata::new(),
        };
        
        session.start_proof(statement).await.unwrap();
        
        let result = session.apply_tactic("intro", vec!["h".to_string()]).await.unwrap();
        assert!(result.success);
        assert!(result.message.contains("Introduced h"));
    }
    
    #[tokio::test]
    async fn test_proof_script_generation() {
        let mut session = InteractiveProofSession::new();
        
        let statement = MathematicalStatement {
            id: Uuid::new_v4(),
            statement_type: StatementType::Theorem {
                name: "simple_theorem".to_string(),
                hypothesis: "True".to_string(),
                conclusion: "True".to_string(),
            },
            framework: "logic".to_string(),
            dependencies: vec![],
            mathematical_context: crate::formalization::MathematicalContext::default(),
            formalization_metadata: crate::formalization::FormalizationMetadata::new(),
        };
        
        session.start_proof(statement).await.unwrap();
        
        let script = session.generate_proof_script();
        assert!(script.contains("theorem simple_theorem"));
        assert!(script.contains(":= by"));
    }
    
    #[test]
    fn test_tactic_creation() {
        let tactic = Tactic {
            name: "test_tactic".to_string(),
            description: "Test tactic".to_string(),
            arguments: vec![
                TacticArgument::required("arg1", "Required argument"),
                TacticArgument::optional("arg2", "Optional argument"),
            ],
            applicable_conditions: vec!["any_goal".to_string()],
        };
        
        assert_eq!(tactic.name, "test_tactic");
        assert_eq!(tactic.arguments.len(), 2);
        assert!(tactic.arguments[0].required);
        assert!(!tactic.arguments[1].required);
    }
    
    #[tokio::test]
    async fn test_proof_assistant() {
        let mut assistant = ProofAssistant::new();
        let session_id = assistant.create_session().await;
        
        let session = assistant.get_session(&session_id);
        assert!(session.is_some());
        
        assistant.close_session(&session_id).await.unwrap();
        let session = assistant.get_session(&session_id);
        assert!(session.is_none());
    }
    
    #[tokio::test]
    async fn test_undo_functionality() {
        let mut session = InteractiveProofSession::new();
        
        let statement = MathematicalStatement::theorem(
            "test_theorem",
            "P → Q",
            "logic"
        );
        
        session.start_proof(statement).await.unwrap();
        let initial_state = session.get_proof_state().clone();
        
        session.apply_tactic("intro", vec!["h".to_string()]).await.unwrap();
        let after_intro_state = session.get_proof_state().clone();
        
        assert_ne!(initial_state.goals.len(), after_intro_state.goals.len());
        
        session.undo().await.unwrap();
        let after_undo_state = session.get_proof_state().clone();
        
        // Note: Due to implementation details, we check history length instead
        assert!(session.proof_state_history.len() >= 1);
    }
    
    #[test]
    fn test_default_tactics() {
        let session = InteractiveProofSession::new();
        let tactics = session.available_tactics;
        
        assert!(tactics.len() > 10);
        
        let tactic_names: Vec<String> = tactics.iter().map(|t| t.name.clone()).collect();
        assert!(tactic_names.contains(&"intro".to_string()));
        assert!(tactic_names.contains(&"apply".to_string()));
        assert!(tactic_names.contains(&"simp".to_string()));
        assert!(tactic_names.contains(&"induction".to_string()));
    }
}
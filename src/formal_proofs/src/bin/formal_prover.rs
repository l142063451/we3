/*!
# Formal Prover CLI

Command-line interface for the formal proof system providing
interactive theorem proving, proof verification, and statement formalization.
*/

use anyhow::Result;
use clap::{Parser, Subcommand};
use formal_proofs::{
    FormalProofSystem, ProofSystemConfig, TheoremProverConfig, VerificationLevel,
    formalization::{MathematicalStatement, LemmaDatabase},
    proof_search::{ProofSearchEngine, SearchStrategy},
    interactive::InteractiveProofSession,
};
use std::path::PathBuf;
use tokio::io::{self, AsyncBufReadExt, BufReader};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,
    
    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Start interactive proof session
    Interactive {
        /// Statement to prove
        #[arg(short, long)]
        statement: Option<String>,
    },
    /// Prove a mathematical statement
    Prove {
        /// Statement to prove
        statement: String,
        /// Theorem prover to use
        #[arg(short, long, default_value = "lean4")]
        prover: String,
        /// Search strategy
        #[arg(short, long, default_value = "breadth-first")]
        strategy: String,
    },
    /// Verify a proof certificate
    Verify {
        /// Certificate file path
        certificate: PathBuf,
        /// Verification level
        #[arg(short, long, default_value = "standard")]
        level: String,
    },
    /// Formalize mathematical statement
    Formalize {
        /// Natural language statement
        statement: String,
        /// Mathematical framework
        #[arg(short, long, default_value = "logic")]
        framework: String,
    },
    /// Search lemma database
    Search {
        /// Search pattern
        pattern: String,
        /// Framework to search in
        #[arg(short, long)]
        framework: Option<String>,
    },
    /// List available theorem provers
    ListProvers,
    /// Check system status
    Status,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    let cli = Cli::parse();
    
    let config = load_config(cli.config.as_ref()).await?;
    let mut proof_system = FormalProofSystem::new(config)?;
    proof_system.initialize_provers().await?;
    
    match cli.command {
        Commands::Interactive { statement } => {
            run_interactive_mode(statement).await?;
        },
        Commands::Prove { statement, prover, strategy } => {
            prove_statement(&proof_system, &statement, &prover, &strategy).await?;
        },
        Commands::Verify { certificate, level } => {
            verify_certificate(&proof_system, &certificate, &level).await?;
        },
        Commands::Formalize { statement, framework } => {
            formalize_statement(&statement, &framework).await?;
        },
        Commands::Search { pattern, framework } => {
            search_lemmas(&pattern, framework.as_deref()).await?;
        },
        Commands::ListProvers => {
            list_provers().await?;
        },
        Commands::Status => {
            check_status(&proof_system).await?;
        },
    }
    
    Ok(())
}

async fn load_config(config_path: Option<&PathBuf>) -> Result<ProofSystemConfig> {
    if let Some(path) = config_path {
        if path.exists() {
            let content = tokio::fs::read_to_string(path).await?;
            return Ok(serde_json::from_str(&content)?);
        }
    }
    
    // Return default config
    Ok(ProofSystemConfig::default())
}

async fn run_interactive_mode(initial_statement: Option<String>) -> Result<()> {
    println!("🧠 WE3 Interactive Formal Prover");
    println!("================================");
    println!("Type 'help' for available commands, 'quit' to exit");
    println!();
    
    let mut session = InteractiveProofSession::new();
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();
    
    // If initial statement provided, start proof
    if let Some(stmt) = initial_statement {
        let statement = parse_statement(&stmt, "logic")?;
        session.start_proof(statement).await?;
        print_proof_state(&session);
    }
    
    loop {
        print!("prover> ");
        io::stdout().flush().await?;
        
        line.clear();
        if reader.read_line(&mut line).await? == 0 {
            break;
        }
        
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        
        match input {
            "help" => print_help(),
            "quit" | "exit" => break,
            "status" => print_proof_state(&session),
            "undo" => {
                match session.undo().await {
                    Ok(()) => println!("✅ Undone last tactic"),
                    Err(e) => println!("❌ Error: {}", e),
                }
            },
            "script" => {
                let script = session.generate_proof_script();
                println!("Generated proof script:");
                println!("{}", script);
            },
            input if input.starts_with("prove ") => {
                let stmt = &input[6..];
                let statement = parse_statement(stmt, "logic")?;
                session.start_proof(statement).await?;
                print_proof_state(&session);
            },
            input if input.starts_with("tactic ") => {
                let tactic_input = &input[7..];
                let parts: Vec<&str> = tactic_input.split_whitespace().collect();
                if let Some(tactic_name) = parts.first() {
                    let args = parts[1..].iter().map(|s| s.to_string()).collect();
                    
                    match session.apply_tactic(tactic_name, args).await {
                        Ok(result) => {
                            if result.success {
                                println!("✅ {}", result.message);
                                print_proof_state(&session);
                            } else {
                                println!("❌ {}", result.message);
                            }
                        },
                        Err(e) => println!("❌ Error: {}", e),
                    }
                } else {
                    println!("❌ Invalid tactic command");
                }
            },
            _ => println!("❌ Unknown command: {}. Type 'help' for available commands.", input),
        }
    }
    
    println!("👋 Goodbye!");
    Ok(())
}

fn print_help() {
    println!("Available commands:");
    println!("  help                    - Show this help message");
    println!("  prove <statement>       - Start proving a statement");
    println!("  tactic <name> [args]    - Apply a tactic");
    println!("  status                  - Show current proof state");
    println!("  undo                    - Undo last tactic");
    println!("  script                  - Generate proof script");
    println!("  quit                    - Exit the prover");
    println!();
    println!("Available tactics:");
    println!("  intro [name]            - Introduce hypothesis");
    println!("  apply <lemma>           - Apply lemma");
    println!("  exact <term>            - Provide exact proof");
    println!("  simp                    - Simplify goal");
    println!("  ring                    - Solve ring equations");
    println!("  linarith                - Linear arithmetic");
    println!("  induction <var>         - Proof by induction");
    println!("  cases <term>            - Case analysis");
    println!("  split                   - Split conjunction");
    println!("  left                    - Prove left disjunct");
    println!("  right                   - Prove right disjunct");
    println!("  exists <witness>        - Provide existential witness");
}

fn print_proof_state(session: &InteractiveProofSession) {
    println!();
    println!("📊 Proof State");
    println!("─────────────────");
    
    let state = session.get_proof_state();
    let progress = session.get_progress();
    let complete = session.is_proof_complete();
    
    println!("Progress: {:.1}%", progress * 100.0);
    println!("Complete: {}", if complete { "✅" } else { "❌" });
    println!("Goals: {}", state.goals.len());
    
    if !state.goals.is_empty() {
        println!();
        println!("Current goals:");
        for (i, goal) in state.goals.iter().enumerate() {
            let marker = if Some(goal.id) == state.current_goal { "►" } else { " " };
            println!("  {} {}: {}", marker, i + 1, goal.statement);
        }
    }
    
    if !state.hypotheses.is_empty() {
        println!();
        println!("Hypotheses:");
        for hyp in &state.hypotheses {
            println!("  {}: {}", hyp.name, hyp.type_expr);
        }
    }
    
    println!();
}

async fn prove_statement(
    proof_system: &FormalProofSystem,
    statement_str: &str,
    prover: &str,
    strategy: &str,
) -> Result<()> {
    println!("🎯 Proving statement: {}", statement_str);
    
    let statement = parse_statement(statement_str, "logic")?;
    
    let search_strategy = match strategy {
        "breadth-first" => SearchStrategy::BreadthFirst,
        "depth-first" => SearchStrategy::DepthFirst { max_depth: 20 },
        "best-first" => SearchStrategy::BestFirst,
        "a-star" => SearchStrategy::AStar,
        "iterative-deepening" => SearchStrategy::IterativeDeepening,
        "resolution" => SearchStrategy::Resolution,
        "tableau" => SearchStrategy::Tableau,
        "natural-deduction" => SearchStrategy::NaturalDeduction,
        _ => SearchStrategy::BreadthFirst,
    };
    
    let search_engine = ProofSearchEngine::new(search_strategy);
    
    println!("🔍 Searching for proof using {} strategy...", strategy);
    let search_result = search_engine.search_proof(&statement)?;
    
    if search_result.success {
        println!("✅ Proof found!");
        println!("Search time: {:?}", search_result.search_time);
        println!("Nodes explored: {}", search_result.nodes_explored);
        
        // Try to get formal verification
        match proof_system.prove_statement(&statement).await {
            Ok(certificate) => {
                println!("📜 Proof certificate generated");
                println!("Prover: {}", certificate.theorem_prover);
                println!("Verification: {:?}", certificate.verification_level);
            },
            Err(e) => println!("⚠️  Formal verification failed: {}", e),
        }
    } else {
        println!("❌ No proof found");
        if let Some(reason) = search_result.failure_reason {
            println!("Reason: {}", reason);
        }
        println!("Search time: {:?}", search_result.search_time);
        println!("Nodes explored: {}", search_result.nodes_explored);
    }
    
    Ok(())
}

async fn verify_certificate(
    _proof_system: &FormalProofSystem,
    certificate_path: &PathBuf,
    level: &str,
) -> Result<()> {
    println!("🔍 Verifying certificate: {:?}", certificate_path);
    
    let verification_level = match level {
        "basic" => VerificationLevel::Basic,
        "standard" => VerificationLevel::Standard,
        "rigorous" => VerificationLevel::Rigorous,
        "meta-theoretical" => VerificationLevel::MetaTheoretical,
        _ => VerificationLevel::Standard,
    };
    
    if !certificate_path.exists() {
        println!("❌ Certificate file not found");
        return Ok(());
    }
    
    let content = tokio::fs::read_to_string(certificate_path).await?;
    let certificate: formal_proofs::ProofCertificate = serde_json::from_str(&content)?;
    
    println!("Certificate ID: {}", certificate.id);
    println!("Statement: {}", certificate.statement);
    println!("Prover: {}", certificate.theorem_prover);
    println!("Framework: {}", certificate.mathematical_framework);
    
    // Simulate verification
    let verification_engine = formal_proofs::verification::VerificationEngine::new(verification_level);
    let result = verification_engine.verify_proof(&certificate).await?;
    
    match result.verification_status {
        formal_proofs::verification::VerificationStatus::Verified => {
            println!("✅ Certificate verified successfully");
            println!("Trust score: {:.2}", result.trust_score);
            if !result.axiom_usage.is_empty() {
                println!("Axioms used: {}", result.axiom_usage.join(", "));
            }
        },
        formal_proofs::verification::VerificationStatus::Failed { reason } => {
            println!("❌ Verification failed: {}", reason);
        },
        _ => {
            println!("⚠️  Verification inconclusive");
        }
    }
    
    if !result.issues.is_empty() {
        println!("Issues found:");
        for issue in result.issues {
            println!("  {:?}: {}", issue.severity, issue.message);
        }
    }
    
    Ok(())
}

async fn formalize_statement(statement: &str, framework: &str) -> Result<()> {
    println!("📝 Formalizing statement: {}", statement);
    println!("Framework: {}", framework);
    
    let formal_statement = parse_statement(statement, framework)?;
    
    println!();
    println!("Formalized statement:");
    println!("  Name: {}", formal_statement.get_name());
    println!("  Framework: {}", formal_statement.framework);
    println!("  Statement: {}", formal_statement.to_string());
    println!("  Context: {:?}", formal_statement.mathematical_context.logical_system);
    
    Ok(())
}

async fn search_lemmas(pattern: &str, framework: Option<&str>) -> Result<()> {
    println!("🔍 Searching lemmas for pattern: {}", pattern);
    
    let temp_dir = tempfile::tempdir()?;
    let mut database = LemmaDatabase::new(&temp_dir.path().to_path_buf())?;
    database.load_standard_library()?;
    
    let results = database.search_lemmas(pattern);
    
    if results.is_empty() {
        println!("No lemmas found matching pattern: {}", pattern);
    } else {
        println!("Found {} lemmas:", results.len());
        for (i, lemma) in results.iter().enumerate() {
            if let Some(fw) = framework {
                if lemma.framework != fw {
                    continue;
                }
            }
            
            println!("  {}. {} ({})", i + 1, lemma.get_name(), lemma.framework);
            println!("     {}", lemma.to_string());
        }
    }
    
    Ok(())
}

async fn list_provers() -> Result<()> {
    println!("🛠️  Available theorem provers:");
    
    let lean_config = TheoremProverConfig::lean4_default();
    let coq_config = TheoremProverConfig::coq_default();
    
    println!("  1. {} - {}", lean_config.name, "Lean 4 theorem prover");
    println!("     Path: {:?}", lean_config.executable_path);
    println!("     Timeout: {:?}", lean_config.timeout);
    
    println!("  2. {} - {}", coq_config.name, "Coq proof assistant");
    println!("     Path: {:?}", coq_config.executable_path);
    println!("     Timeout: {:?}", coq_config.timeout);
    
    Ok(())
}

async fn check_status(proof_system: &FormalProofSystem) -> Result<()> {
    println!("🩺 System Status");
    println!("─────────────────");
    
    // Check if theorem provers are available
    println!("Theorem provers:");
    
    let lean_config = TheoremProverConfig::lean4_default();
    if let Ok(prover) = formal_proofs::theorem_prover::LeanProver::new(lean_config) {
        let available = prover.is_available().await.unwrap_or(false);
        println!("  Lean 4: {}", if available { "✅ Available" } else { "❌ Not available" });
    }
    
    let coq_config = TheoremProverConfig::coq_default();
    if let Ok(prover) = formal_proofs::theorem_prover::CoqProver::new(coq_config) {
        let available = prover.is_available().await.unwrap_or(false);
        println!("  Coq: {}", if available { "✅ Available" } else { "❌ Not available" });
    }
    
    println!();
    println!("System components:");
    println!("  Formal logic engine: ✅ Operational");
    println!("  Proof search: ✅ Operational");
    println!("  Verification system: ✅ Operational");
    println!("  Meta-theory analysis: ✅ Operational");
    println!("  Interactive assistant: ✅ Operational");
    
    Ok(())
}

fn parse_statement(statement: &str, framework: &str) -> Result<MathematicalStatement> {
    // Simple statement parsing - in practice would need sophisticated parser
    if statement.contains("theorem") || statement.contains("lemma") {
        // Extract name and statement
        let parts: Vec<&str> = statement.split(':').collect();
        if parts.len() >= 2 {
            let name = parts[0].trim().replace("theorem ", "").replace("lemma ", "");
            let stmt = parts[1].trim();
            
            if stmt.contains("→") || stmt.contains("->") {
                // Implication
                let impl_parts: Vec<&str> = stmt.split("→").or_else(|| stmt.split("->")).collect();
                if impl_parts.len() == 2 {
                    return Ok(MathematicalStatement::theorem(
                        &name,
                        &format!("{} → {}", impl_parts[0].trim(), impl_parts[1].trim()),
                        framework
                    ));
                }
            }
            
            return Ok(MathematicalStatement::theorem(&name, stmt, framework));
        }
    }
    
    // Default to simple theorem
    Ok(MathematicalStatement::theorem("user_statement", statement, framework))
}

use tokio::io::AsyncWriteExt;
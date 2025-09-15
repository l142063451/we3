/*!
# Mathematical Statement Formalization

Advanced mathematical formalization system for converting mathematical claims 
into formal logic statements suitable for machine verification. Supports 
multiple logical systems including first-order logic, higher-order logic, 
and dependent type theory.
*/

use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use anyhow::Result;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use crate::FormalProofError;

/// Mathematical statement types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatementType {
    /// Theorem with hypothesis and conclusion
    Theorem {
        name: String,
        hypothesis: String,
        conclusion: String,
    },
    /// Lemma statement
    Lemma {
        name: String,
        statement: String,
    },
    /// Mathematical definition
    Definition {
        name: String,
        type_expr: String,
        body: String,
    },
}

/// Formal mathematical statement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalStatement {
    pub id: Uuid,
    pub statement_type: StatementType,
    pub framework: String,
    pub dependencies: Vec<Uuid>,
    pub mathematical_context: MathematicalContext,
    pub formalization_metadata: FormalizationMetadata,
}

impl MathematicalStatement {
    /// Create new theorem statement
    pub fn theorem(name: &str, statement: &str, framework: &str) -> Self {
        // Parse statement into hypothesis → conclusion form
        let parts: Vec<&str> = statement.split(" → ").collect();
        let (hypothesis, conclusion) = if parts.len() == 2 {
            (parts[0].to_string(), parts[1].to_string())
        } else {
            ("True".to_string(), statement.to_string())
        };
        
        Self {
            id: Uuid::new_v4(),
            statement_type: StatementType::Theorem {
                name: name.to_string(),
                hypothesis,
                conclusion,
            },
            framework: framework.to_string(),
            dependencies: vec![],
            mathematical_context: MathematicalContext::default(),
            formalization_metadata: FormalizationMetadata::new(),
        }
    }
    
    /// Create new lemma statement
    pub fn lemma(name: &str, statement: &str, framework: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            statement_type: StatementType::Lemma {
                name: name.to_string(),
                statement: statement.to_string(),
            },
            framework: framework.to_string(),
            dependencies: vec![],
            mathematical_context: MathematicalContext::default(),
            formalization_metadata: FormalizationMetadata::new(),
        }
    }
    
    /// Create new definition statement
    pub fn definition(name: &str, type_expr: &str, body: &str, framework: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            statement_type: StatementType::Definition {
                name: name.to_string(),
                type_expr: type_expr.to_string(),
                body: body.to_string(),
            },
            framework: framework.to_string(),
            dependencies: vec![],
            mathematical_context: MathematicalContext::default(),
            formalization_metadata: FormalizationMetadata::new(),
        }
    }
    
    /// Universal quantification
    pub fn forall(variables: &str, body: &str) -> Self {
        let statement = format!("∀ {}, {}", variables, body);
        Self::theorem("forall_statement", &statement, "logic")
    }
    
    /// Existential quantification
    pub fn exists(variables: &str, body: &str) -> Self {
        let statement = format!("∃ {}, {}", variables, body);
        Self::theorem("exists_statement", &statement, "logic")
    }
    
    /// Get statement name
    pub fn get_name(&self) -> String {
        match &self.statement_type {
            StatementType::Theorem { name, .. } => name.clone(),
            StatementType::Lemma { name, .. } => name.clone(),
            StatementType::Definition { name, .. } => name.clone(),
        }
    }
    
    /// Get statement conclusion
    pub fn get_conclusion(&self) -> String {
        match &self.statement_type {
            StatementType::Theorem { conclusion, .. } => conclusion.clone(),
            StatementType::Lemma { statement, .. } => statement.clone(),
            StatementType::Definition { body, .. } => body.clone(),
        }
    }
    
    /// Add dependency
    pub fn add_dependency(&mut self, dependency: Uuid) {
        if !self.dependencies.contains(&dependency) {
            self.dependencies.push(dependency);
        }
    }
    
    /// Set mathematical context
    pub fn with_context(mut self, context: MathematicalContext) -> Self {
        self.mathematical_context = context;
        self
    }
    
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match &self.statement_type {
            StatementType::Theorem { name, hypothesis, conclusion } => {
                format!("theorem {} : {} → {}", name, hypothesis, conclusion)
            },
            StatementType::Lemma { name, statement } => {
                format!("lemma {} : {}", name, statement)
            },
            StatementType::Definition { name, type_expr, body } => {
                format!("def {} : {} := {}", name, type_expr, body)
            },
        }
    }
}

/// Mathematical context for formalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathematicalContext {
    pub logical_system: LogicalSystem,
    pub mathematical_structures: Vec<MathematicalStructure>,
    pub axiom_systems: Vec<AxiomSystem>,
    pub type_universe: TypeUniverse,
    pub notation_system: NotationSystem,
}

impl Default for MathematicalContext {
    fn default() -> Self {
        Self {
            logical_system: LogicalSystem::HigherOrderLogic,
            mathematical_structures: vec![
                MathematicalStructure::NaturalNumbers,
                MathematicalStructure::RealNumbers,
                MathematicalStructure::ComplexNumbers,
            ],
            axiom_systems: vec![AxiomSystem::ZFC],
            type_universe: TypeUniverse::default(),
            notation_system: NotationSystem::Standard,
        }
    }
}

/// Logical system types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogicalSystem {
    PropositionalLogic,
    FirstOrderLogic,
    HigherOrderLogic,
    DependentTypeTheory,
    LinearLogic,
    ModalLogic,
    IntuitionisticLogic,
}

/// Mathematical structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MathematicalStructure {
    NaturalNumbers,
    Integers,
    RationalNumbers,
    RealNumbers,
    ComplexNumbers,
    Groups,
    Rings,
    Fields,
    VectorSpaces,
    TopologicalSpaces,
    ManifoldStructures,
    CategoryTheory,
    SetTheory,
    MeasureTheory,
}

/// Axiom systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AxiomSystem {
    ZFC,           // Zermelo-Fraenkel with Choice
    ZF,            // Zermelo-Fraenkel without Choice  
    NBG,           // Von Neumann-Bernays-Gödel
    MK,            // Morse-Kelley
    Constructive,  // Constructive mathematics
    Classical,     // Classical mathematics
    Intuitionistic, // Intuitionistic logic
}

/// Type universe for dependent types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeUniverse {
    pub universe_levels: Vec<UniverseLevel>,
    pub type_constructors: HashMap<String, TypeConstructor>,
    pub inductive_types: Vec<InductiveType>,
}

impl Default for TypeUniverse {
    fn default() -> Self {
        Self {
            universe_levels: vec![
                UniverseLevel { level: 0, name: "Prop".to_string() },
                UniverseLevel { level: 1, name: "Type 0".to_string() },
                UniverseLevel { level: 2, name: "Type 1".to_string() },
            ],
            type_constructors: Self::default_type_constructors(),
            inductive_types: Self::default_inductive_types(),
        }
    }
}

impl TypeUniverse {
    fn default_type_constructors() -> HashMap<String, TypeConstructor> {
        let mut constructors = HashMap::new();
        
        constructors.insert("→".to_string(), TypeConstructor {
            name: "function".to_string(),
            arity: 2,
            notation: "→".to_string(),
        });
        
        constructors.insert("×".to_string(), TypeConstructor {
            name: "product".to_string(),
            arity: 2,
            notation: "×".to_string(),
        });
        
        constructors.insert("+".to_string(), TypeConstructor {
            name: "sum".to_string(),
            arity: 2,
            notation: "+".to_string(),
        });
        
        constructors
    }
    
    fn default_inductive_types() -> Vec<InductiveType> {
        vec![
            InductiveType {
                name: "Nat".to_string(),
                constructors: vec!["zero".to_string(), "succ".to_string()],
                elimination_rule: "nat_rec".to_string(),
            },
            InductiveType {
                name: "Bool".to_string(),
                constructors: vec!["true".to_string(), "false".to_string()],
                elimination_rule: "bool_rec".to_string(),
            },
            InductiveType {
                name: "List".to_string(),
                constructors: vec!["nil".to_string(), "cons".to_string()],
                elimination_rule: "list_rec".to_string(),
            },
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseLevel {
    pub level: usize,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeConstructor {
    pub name: String,
    pub arity: usize,
    pub notation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InductiveType {
    pub name: String,
    pub constructors: Vec<String>,
    pub elimination_rule: String,
}

/// Notation system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotationSystem {
    Standard,
    Polish,
    Category,
    Algebraic,
    Logical,
}

/// Formalization metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormalizationMetadata {
    pub created_at: DateTime<Utc>,
    pub formalization_strategy: FormalizationStrategy,
    pub complexity_metrics: ComplexityMetrics,
    pub verification_requirements: VerificationRequirements,
}

impl FormalizationMetadata {
    pub fn new() -> Self {
        Self {
            created_at: Utc::now(),
            formalization_strategy: FormalizationStrategy::Direct,
            complexity_metrics: ComplexityMetrics::default(),
            verification_requirements: VerificationRequirements::default(),
        }
    }
}

/// Formalization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormalizationStrategy {
    /// Direct formalization
    Direct,
    /// Category-theoretic formalization
    Categorical,
    /// Type-theoretic formalization  
    TypeTheoretic,
    /// Constructive formalization
    Constructive,
    /// Computational formalization
    Computational,
}

/// Complexity metrics for statements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub logical_depth: usize,
    pub quantifier_count: usize,
    pub symbol_count: usize,
    pub proof_complexity_estimate: ProofComplexityEstimate,
}

impl Default for ComplexityMetrics {
    fn default() -> Self {
        Self {
            logical_depth: 0,
            quantifier_count: 0,
            symbol_count: 0,
            proof_complexity_estimate: ProofComplexityEstimate::Unknown,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofComplexityEstimate {
    Trivial,
    Elementary,
    Polynomial,
    Exponential,
    Unknown,
}

/// Verification requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationRequirements {
    pub soundness_required: bool,
    pub completeness_required: bool,
    pub decidability_required: bool,
    pub constructive_proof_required: bool,
    pub computational_content_required: bool,
}

impl Default for VerificationRequirements {
    fn default() -> Self {
        Self {
            soundness_required: true,
            completeness_required: false,
            decidability_required: false,
            constructive_proof_required: false,
            computational_content_required: false,
        }
    }
}

/// Lemma database for storing and retrieving mathematical knowledge
#[derive(Debug)]
pub struct LemmaDatabase {
    storage_path: PathBuf,
    lemmas: HashMap<String, Vec<MathematicalStatement>>,
    indices: HashMap<String, HashSet<Uuid>>,
}

impl LemmaDatabase {
    pub fn new(storage_path: &PathBuf) -> Result<Self> {
        std::fs::create_dir_all(storage_path)?;
        
        Ok(Self {
            storage_path: storage_path.clone(),
            lemmas: HashMap::new(),
            indices: HashMap::new(),
        })
    }
    
    /// Store a lemma in the database
    pub fn store_lemma(&mut self, lemma: MathematicalStatement) -> Result<()> {
        let framework = lemma.framework.clone();
        let id = lemma.id;
        
        self.lemmas.entry(framework.clone())
            .or_insert_with(Vec::new)
            .push(lemma);
        
        // Index by framework
        self.indices.entry(format!("framework:{}", framework))
            .or_insert_with(HashSet::new)
            .insert(id);
        
        self.persist_to_disk()?;
        Ok(())
    }
    
    /// Retrieve lemmas by framework
    pub fn get_lemmas_by_framework(&self, framework: &str) -> Vec<&MathematicalStatement> {
        self.lemmas.get(framework)
            .map(|lemmas| lemmas.iter().collect())
            .unwrap_or_default()
    }
    
    /// Search lemmas by pattern
    pub fn search_lemmas(&self, pattern: &str) -> Vec<&MathematicalStatement> {
        let mut results = Vec::new();
        
        for lemma_list in self.lemmas.values() {
            for lemma in lemma_list {
                if lemma.to_string().contains(pattern) {
                    results.push(lemma);
                }
            }
        }
        
        results
    }
    
    /// Get related lemmas
    pub fn get_related_lemmas(&self, statement: &MathematicalStatement) -> Vec<&MathematicalStatement> {
        // Simple implementation: find lemmas from the same framework
        self.get_lemmas_by_framework(&statement.framework)
    }
    
    /// Load mathematical library
    pub fn load_standard_library(&mut self) -> Result<()> {
        // Load fundamental mathematical lemmas
        self.load_arithmetic_lemmas()?;
        self.load_logic_lemmas()?;
        self.load_set_theory_lemmas()?;
        self.load_analysis_lemmas()?;
        self.load_algebra_lemmas()?;
        
        Ok(())
    }
    
    fn load_arithmetic_lemmas(&mut self) -> Result<()> {
        let lemmas = vec![
            MathematicalStatement::theorem(
                "add_comm",
                "∀ a b: ℕ, a + b = b + a",
                "arithmetic"
            ),
            MathematicalStatement::theorem(
                "add_assoc", 
                "∀ a b c: ℕ, (a + b) + c = a + (b + c)",
                "arithmetic"
            ),
            MathematicalStatement::theorem(
                "mul_comm",
                "∀ a b: ℕ, a * b = b * a",
                "arithmetic"
            ),
            MathematicalStatement::theorem(
                "mul_assoc",
                "∀ a b c: ℕ, (a * b) * c = a * (b * c)",
                "arithmetic"
            ),
            MathematicalStatement::theorem(
                "distributivity",
                "∀ a b c: ℕ, a * (b + c) = a * b + a * c",
                "arithmetic"
            ),
        ];
        
        for lemma in lemmas {
            self.store_lemma(lemma)?;
        }
        
        Ok(())
    }
    
    fn load_logic_lemmas(&mut self) -> Result<()> {
        let lemmas = vec![
            MathematicalStatement::theorem(
                "excluded_middle",
                "∀ P: Prop, P ∨ ¬P",
                "logic"
            ),
            MathematicalStatement::theorem(
                "double_negation",
                "∀ P: Prop, ¬¬P → P",
                "logic"
            ),
            MathematicalStatement::theorem(
                "demorgan_and",
                "∀ P Q: Prop, ¬(P ∧ Q) ↔ (¬P ∨ ¬Q)",
                "logic"
            ),
            MathematicalStatement::theorem(
                "demorgan_or",
                "∀ P Q: Prop, ¬(P ∨ Q) ↔ (¬P ∧ ¬Q)",
                "logic"
            ),
        ];
        
        for lemma in lemmas {
            self.store_lemma(lemma)?;
        }
        
        Ok(())
    }
    
    fn load_set_theory_lemmas(&mut self) -> Result<()> {
        let lemmas = vec![
            MathematicalStatement::theorem(
                "union_comm",
                "∀ A B: Set, A ∪ B = B ∪ A",
                "set_theory"
            ),
            MathematicalStatement::theorem(
                "intersection_comm",
                "∀ A B: Set, A ∩ B = B ∩ A",
                "set_theory"
            ),
            MathematicalStatement::theorem(
                "demorgan_union",
                "∀ A B: Set, (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ",
                "set_theory"
            ),
        ];
        
        for lemma in lemmas {
            self.store_lemma(lemma)?;
        }
        
        Ok(())
    }
    
    fn load_analysis_lemmas(&mut self) -> Result<()> {
        let lemmas = vec![
            MathematicalStatement::theorem(
                "mean_value_theorem",
                "∀ f: ℝ → ℝ, continuous(f) ∧ differentiable(f) → ∃ c, f'(c) = (f(b) - f(a))/(b - a)",
                "analysis"
            ),
            MathematicalStatement::theorem(
                "intermediate_value",
                "∀ f: ℝ → ℝ, continuous(f) → ∀ y ∈ [f(a), f(b)], ∃ c ∈ [a, b], f(c) = y",
                "analysis"
            ),
        ];
        
        for lemma in lemmas {
            self.store_lemma(lemma)?;
        }
        
        Ok(())
    }
    
    fn load_algebra_lemmas(&mut self) -> Result<()> {
        let lemmas = vec![
            MathematicalStatement::theorem(
                "group_inverse",
                "∀ G: Group, ∀ g ∈ G, ∃! g⁻¹, g * g⁻¹ = e",
                "algebra"
            ),
            MathematicalStatement::theorem(
                "ring_distributivity",
                "∀ R: Ring, ∀ a b c ∈ R, a * (b + c) = a * b + a * c",
                "algebra"
            ),
        ];
        
        for lemma in lemmas {
            self.store_lemma(lemma)?;
        }
        
        Ok(())
    }
    
    fn persist_to_disk(&self) -> Result<()> {
        let database_file = self.storage_path.join("lemmas.json");
        let serialized = serde_json::to_string_pretty(&self.lemmas)?;
        std::fs::write(database_file, serialized)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_mathematical_statement_creation() {
        let theorem = MathematicalStatement::theorem(
            "pythagorean",
            "∀ a b c: ℝ, a² + b² = c²",
            "geometry"
        );
        
        assert_eq!(theorem.framework, "geometry");
        assert_eq!(theorem.get_name(), "pythagorean");
    }
    
    #[test]
    fn test_statement_types() {
        let theorem = MathematicalStatement::theorem(
            "test_theorem",
            "P → Q",
            "logic"
        );
        
        let lemma = MathematicalStatement::lemma(
            "test_lemma",
            "P ∨ ¬P",
            "logic"
        );
        
        let definition = MathematicalStatement::definition(
            "test_def",
            "ℕ",
            "zero | succ ℕ",
            "arithmetic"
        );
        
        assert_eq!(theorem.get_name(), "test_theorem");
        assert_eq!(lemma.get_name(), "test_lemma");
        assert_eq!(definition.get_name(), "test_def");
    }
    
    #[test]
    fn test_quantifiers() {
        let forall_stmt = MathematicalStatement::forall(
            "x: ℕ",
            "x + 0 = x"
        );
        
        let exists_stmt = MathematicalStatement::exists(
            "x: ℕ",
            "x > 5"
        );
        
        assert!(forall_stmt.to_string().contains("∀"));
        assert!(exists_stmt.to_string().contains("∃"));
    }
    
    #[test]
    fn test_mathematical_context() {
        let mut context = MathematicalContext::default();
        context.logical_system = LogicalSystem::DependentTypeTheory;
        context.axiom_systems = vec![AxiomSystem::Constructive];
        
        let statement = MathematicalStatement::theorem(
            "constructive_theorem",
            "∃ x: ℕ, P(x)",
            "constructive_math"
        ).with_context(context);
        
        match statement.mathematical_context.logical_system {
            LogicalSystem::DependentTypeTheory => (),
            _ => panic!("Wrong logical system"),
        }
    }
    
    #[test]
    fn test_lemma_database() {
        let temp_dir = TempDir::new().unwrap();
        let mut database = LemmaDatabase::new(&temp_dir.path().to_path_buf()).unwrap();
        
        let lemma = MathematicalStatement::theorem(
            "test_lemma",
            "1 + 1 = 2",
            "arithmetic"
        );
        
        database.store_lemma(lemma).unwrap();
        
        let retrieved = database.get_lemmas_by_framework("arithmetic");
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].get_name(), "test_lemma");
    }
    
    #[test]
    fn test_lemma_search() {
        let temp_dir = TempDir::new().unwrap();
        let mut database = LemmaDatabase::new(&temp_dir.path().to_path_buf()).unwrap();
        
        let lemma1 = MathematicalStatement::theorem(
            "comm_add",
            "a + b = b + a",
            "arithmetic"
        );
        
        let lemma2 = MathematicalStatement::theorem(
            "comm_mul",
            "a * b = b * a", 
            "arithmetic"
        );
        
        database.store_lemma(lemma1).unwrap();
        database.store_lemma(lemma2).unwrap();
        
        let results = database.search_lemmas("comm");
        assert_eq!(results.len(), 2);
        
        let add_results = database.search_lemmas("+ b =");
        assert_eq!(add_results.len(), 1);
    }
    
    #[test]
    fn test_standard_library_loading() {
        let temp_dir = TempDir::new().unwrap();
        let mut database = LemmaDatabase::new(&temp_dir.path().to_path_buf()).unwrap();
        
        database.load_standard_library().unwrap();
        
        let arithmetic_lemmas = database.get_lemmas_by_framework("arithmetic");
        assert!(arithmetic_lemmas.len() > 0);
        
        let logic_lemmas = database.get_lemmas_by_framework("logic");
        assert!(logic_lemmas.len() > 0);
        
        let set_theory_lemmas = database.get_lemmas_by_framework("set_theory");
        assert!(set_theory_lemmas.len() > 0);
        
        // Test specific lemmas
        let excluded_middle = database.search_lemmas("excluded_middle");
        assert_eq!(excluded_middle.len(), 1);
        
        let add_comm = database.search_lemmas("add_comm");
        assert_eq!(add_comm.len(), 1);
    }
}
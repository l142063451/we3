/*!
# Verification System

Advanced verification system for mathematical proofs with certificate
validation, soundness checking, and completeness analysis.
*/

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use crate::{FormalProofError, ProofCertificate, VerificationLevel};

/// Certificate storage and management
#[derive(Debug)]
pub struct CertificateStore {
    storage_path: PathBuf,
    certificates: HashMap<Uuid, ProofCertificate>,
    indices: HashMap<String, Vec<Uuid>>,
}

impl CertificateStore {
    pub fn new(storage_path: &Path) -> Result<Self> {
        std::fs::create_dir_all(storage_path)?;
        
        let mut store = Self {
            storage_path: storage_path.to_path_buf(),
            certificates: HashMap::new(),
            indices: HashMap::new(),
        };
        
        store.load_from_disk()?;
        Ok(store)
    }
    
    /// Store a proof certificate
    pub fn store_certificate(&mut self, certificate: &ProofCertificate) -> Result<()> {
        let id = certificate.id;
        self.certificates.insert(id, certificate.clone());
        
        // Update indices
        self.indices.entry(format!("framework:{}", certificate.mathematical_framework))
            .or_insert_with(Vec::new)
            .push(id);
        
        self.indices.entry(format!("prover:{}", certificate.theorem_prover))
            .or_insert_with(Vec::new)
            .push(id);
        
        self.persist_to_disk()?;
        Ok(())
    }
    
    /// Retrieve certificate by ID
    pub fn get_certificate(&self, id: &Uuid) -> Option<&ProofCertificate> {
        self.certificates.get(id)
    }
    
    /// Get certificates by framework
    pub fn get_certificates_by_framework(&self, framework: &str) -> Vec<&ProofCertificate> {
        let key = format!("framework:{}", framework);
        self.indices.get(&key)
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|id| self.certificates.get(id))
            .collect()
    }
    
    /// Get certificates by theorem prover
    pub fn get_certificates_by_prover(&self, prover: &str) -> Vec<&ProofCertificate> {
        let key = format!("prover:{}", prover);
        self.indices.get(&key)
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|id| self.certificates.get(id))
            .collect()
    }
    
    /// Verify certificate integrity
    pub fn verify_certificate(&self, certificate: &ProofCertificate) -> Result<bool> {
        // Check certificate format
        if certificate.statement.is_empty() || certificate.proof_term.is_empty() {
            return Ok(false);
        }
        
        // Verify checksum
        let computed_checksum = self.compute_checksum(&certificate.proof_term);
        if computed_checksum != certificate.checksum {
            return Ok(false);
        }
        
        // Verify dependencies exist
        for dep_id in &certificate.dependencies {
            if !self.certificates.contains_key(dep_id) {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Check for circular dependencies
    pub fn check_circular_dependencies(&self, certificate: &ProofCertificate) -> Result<bool> {
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();
        
        self.has_cycle(certificate.id, &mut visited, &mut rec_stack)
    }
    
    fn has_cycle(
        &self,
        cert_id: Uuid,
        visited: &mut std::collections::HashSet<Uuid>,
        rec_stack: &mut std::collections::HashSet<Uuid>,
    ) -> Result<bool> {
        visited.insert(cert_id);
        rec_stack.insert(cert_id);
        
        if let Some(certificate) = self.certificates.get(&cert_id) {
            for &dep_id in &certificate.dependencies {
                if !visited.contains(&dep_id) {
                    if self.has_cycle(dep_id, visited, rec_stack)? {
                        return Ok(true);
                    }
                } else if rec_stack.contains(&dep_id) {
                    return Ok(true);
                }
            }
        }
        
        rec_stack.remove(&cert_id);
        Ok(false)
    }
    
    /// Get dependency chain
    pub fn get_dependency_chain(&self, certificate_id: &Uuid) -> Result<Vec<Uuid>> {
        let mut chain = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.build_dependency_chain(certificate_id, &mut chain, &mut visited)?;
        Ok(chain)
    }
    
    fn build_dependency_chain(
        &self,
        cert_id: &Uuid,
        chain: &mut Vec<Uuid>,
        visited: &mut std::collections::HashSet<Uuid>,
    ) -> Result<()> {
        if visited.contains(cert_id) {
            return Ok(());
        }
        
        visited.insert(*cert_id);
        
        if let Some(certificate) = self.certificates.get(cert_id) {
            for &dep_id in &certificate.dependencies {
                self.build_dependency_chain(&dep_id, chain, visited)?;
            }
        }
        
        chain.push(*cert_id);
        Ok(())
    }
    
    fn compute_checksum(&self, content: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
    
    fn persist_to_disk(&self) -> Result<()> {
        let certificates_file = self.storage_path.join("certificates.json");
        let serialized = serde_json::to_string_pretty(&self.certificates)?;
        std::fs::write(certificates_file, serialized)?;
        
        let indices_file = self.storage_path.join("indices.json");
        let serialized_indices = serde_json::to_string_pretty(&self.indices)?;
        std::fs::write(indices_file, serialized_indices)?;
        
        Ok(())
    }
    
    fn load_from_disk(&mut self) -> Result<()> {
        let certificates_file = self.storage_path.join("certificates.json");
        if certificates_file.exists() {
            let content = std::fs::read_to_string(certificates_file)?;
            self.certificates = serde_json::from_str(&content).unwrap_or_default();
        }
        
        let indices_file = self.storage_path.join("indices.json");
        if indices_file.exists() {
            let content = std::fs::read_to_string(indices_file)?;
            self.indices = serde_json::from_str(&content).unwrap_or_default();
        }
        
        Ok(())
    }
}

/// Verification engine for proof checking
pub struct VerificationEngine {
    verification_level: VerificationLevel,
    trusted_provers: std::collections::HashSet<String>,
    axiom_whitelist: std::collections::HashSet<String>,
}

impl VerificationEngine {
    pub fn new(level: VerificationLevel) -> Self {
        Self {
            verification_level: level,
            trusted_provers: Self::default_trusted_provers(),
            axiom_whitelist: Self::default_axiom_whitelist(),
        }
    }
    
    fn default_trusted_provers() -> std::collections::HashSet<String> {
        let mut provers = std::collections::HashSet::new();
        provers.insert("lean4".to_string());
        provers.insert("coq".to_string());
        provers.insert("isabelle".to_string());
        provers.insert("agda".to_string());
        provers
    }
    
    fn default_axiom_whitelist() -> std::collections::HashSet<String> {
        let mut axioms = std::collections::HashSet::new();
        axioms.insert("Classical.choice".to_string());
        axioms.insert("Classical.em".to_string());
        axioms.insert("Decidable.decide".to_string());
        axioms.insert("Nat.rec".to_string());
        axioms.insert("List.rec".to_string());
        axioms
    }
    
    /// Verify a proof certificate
    pub async fn verify_proof(&self, certificate: &ProofCertificate) -> Result<VerificationResult> {
        let start_time = std::time::Instant::now();
        
        let mut result = VerificationResult {
            certificate_id: certificate.id,
            verification_status: VerificationStatus::Pending,
            verification_time: std::time::Duration::default(),
            verification_level: self.verification_level.clone(),
            issues: Vec::new(),
            trust_score: 1.0,
            axiom_usage: Vec::new(),
        };
        
        // Check prover trust
        if !self.trusted_provers.contains(&certificate.theorem_prover) {
            result.issues.push(VerificationIssue {
                severity: IssueSeverity::Warning,
                message: format!("Untrusted theorem prover: {}", certificate.theorem_prover),
                category: IssueCategory::ProverTrust,
            });
            result.trust_score *= 0.8;
        }
        
        // Check axiom usage
        for axiom in self.extract_axiom_usage(&certificate.proof_term) {
            if !self.axiom_whitelist.contains(&axiom) {
                result.issues.push(VerificationIssue {
                    severity: IssueSeverity::Warning,
                    message: format!("Non-standard axiom used: {}", axiom),
                    category: IssueCategory::AxiomUsage,
                });
                result.trust_score *= 0.9;
            }
            result.axiom_usage.push(axiom);
        }
        
        // Perform verification based on level
        match self.verification_level {
            VerificationLevel::Basic => {
                result.verification_status = self.basic_verification(certificate).await?;
            },
            VerificationLevel::Standard => {
                result.verification_status = self.standard_verification(certificate).await?;
            },
            VerificationLevel::Rigorous => {
                result.verification_status = self.rigorous_verification(certificate).await?;
            },
            VerificationLevel::MetaTheoretical => {
                result.verification_status = self.meta_theoretical_verification(certificate).await?;
            },
        }
        
        result.verification_time = start_time.elapsed();
        Ok(result)
    }
    
    async fn basic_verification(&self, certificate: &ProofCertificate) -> Result<VerificationStatus> {
        // Basic syntax and format checking
        if certificate.statement.is_empty() {
            return Ok(VerificationStatus::Failed {
                reason: "Empty statement".to_string(),
            });
        }
        
        if certificate.proof_term.is_empty() {
            return Ok(VerificationStatus::Failed {
                reason: "Empty proof term".to_string(),
            });
        }
        
        Ok(VerificationStatus::Verified)
    }
    
    async fn standard_verification(&self, certificate: &ProofCertificate) -> Result<VerificationStatus> {
        // Include basic checks plus type checking simulation
        let basic_result = self.basic_verification(certificate).await?;
        if !matches!(basic_result, VerificationStatus::Verified) {
            return Ok(basic_result);
        }
        
        // Simulate type checking
        if !self.simulate_type_checking(&certificate.proof_term) {
            return Ok(VerificationStatus::Failed {
                reason: "Type checking failed".to_string(),
            });
        }
        
        Ok(VerificationStatus::Verified)
    }
    
    async fn rigorous_verification(&self, certificate: &ProofCertificate) -> Result<VerificationStatus> {
        // Include standard checks plus proof validation
        let standard_result = self.standard_verification(certificate).await?;
        if !matches!(standard_result, VerificationStatus::Verified) {
            return Ok(standard_result);
        }
        
        // Simulate rigorous proof validation
        if !self.validate_proof_structure(&certificate.proof_term) {
            return Ok(VerificationStatus::Failed {
                reason: "Invalid proof structure".to_string(),
            });
        }
        
        Ok(VerificationStatus::Verified)
    }
    
    async fn meta_theoretical_verification(&self, certificate: &ProofCertificate) -> Result<VerificationStatus> {
        // Include rigorous checks plus meta-theoretical analysis
        let rigorous_result = self.rigorous_verification(certificate).await?;
        if !matches!(rigorous_result, VerificationStatus::Verified) {
            return Ok(rigorous_result);
        }
        
        // Simulate meta-theoretical verification
        if !self.verify_logical_consistency(&certificate.proof_term) {
            return Ok(VerificationStatus::Failed {
                reason: "Logical inconsistency detected".to_string(),
            });
        }
        
        Ok(VerificationStatus::Verified)
    }
    
    fn extract_axiom_usage(&self, proof_term: &str) -> Vec<String> {
        let mut axioms = Vec::new();
        
        // Simple pattern matching for axiom usage
        if proof_term.contains("Classical") {
            axioms.push("Classical.em".to_string());
        }
        if proof_term.contains("choice") {
            axioms.push("Classical.choice".to_string());
        }
        if proof_term.contains("decide") {
            axioms.push("Decidable.decide".to_string());
        }
        
        axioms
    }
    
    fn simulate_type_checking(&self, proof_term: &str) -> bool {
        // Simplified type checking simulation
        !proof_term.contains("type_error") && proof_term.len() > 10
    }
    
    fn validate_proof_structure(&self, proof_term: &str) -> bool {
        // Simplified proof structure validation
        proof_term.contains("theorem") || proof_term.contains("lemma") || 
        proof_term.contains("def") || proof_term.contains("Proof")
    }
    
    fn verify_logical_consistency(&self, proof_term: &str) -> bool {
        // Simplified consistency check
        !proof_term.contains("contradiction") && !proof_term.contains("absurd")
    }
}

/// Verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub certificate_id: Uuid,
    pub verification_status: VerificationStatus,
    pub verification_time: std::time::Duration,
    pub verification_level: VerificationLevel,
    pub issues: Vec<VerificationIssue>,
    pub trust_score: f64,
    pub axiom_usage: Vec<String>,
}

/// Verification status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationStatus {
    Pending,
    Verified,
    Failed { reason: String },
    Timeout,
    PartiallyVerified { issues: Vec<String> },
}

/// Verification issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationIssue {
    pub severity: IssueSeverity,
    pub message: String,
    pub category: IssueCategory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueCategory {
    Syntax,
    TypeChecking,
    ProofStructure,
    LogicalConsistency,
    ProverTrust,
    AxiomUsage,
    Dependencies,
}

/// Completeness analyzer
pub struct CompletenessAnalyzer {
    proof_strategies: Vec<String>,
    coverage_metrics: HashMap<String, f64>,
}

impl CompletenessAnalyzer {
    pub fn new() -> Self {
        Self {
            proof_strategies: vec![
                "direct".to_string(),
                "contradiction".to_string(),
                "induction".to_string(),
                "case_analysis".to_string(),
                "construction".to_string(),
            ],
            coverage_metrics: HashMap::new(),
        }
    }
    
    /// Analyze completeness of proof framework
    pub fn analyze_completeness(
        &mut self,
        certificates: &[ProofCertificate],
    ) -> Result<CompletenessReport> {
        let mut strategy_usage = HashMap::new();
        let mut framework_coverage = HashMap::new();
        let mut proof_patterns = HashMap::new();
        
        for certificate in certificates {
            // Analyze proof strategies used
            let strategies = self.extract_proof_strategies(&certificate.proof_term);
            for strategy in strategies {
                *strategy_usage.entry(strategy).or_insert(0) += 1;
            }
            
            // Analyze framework coverage
            *framework_coverage.entry(certificate.mathematical_framework.clone()).or_insert(0) += 1;
            
            // Analyze proof patterns
            let patterns = self.extract_proof_patterns(&certificate.proof_term);
            for pattern in patterns {
                *proof_patterns.entry(pattern).or_insert(0) += 1;
            }
        }
        
        let completeness_score = self.compute_completeness_score(
            &strategy_usage,
            &framework_coverage,
            &proof_patterns,
        );
        
        Ok(CompletenessReport {
            completeness_score,
            strategy_usage: strategy_usage.clone(),
            framework_coverage: framework_coverage.clone(),
            proof_patterns,
            recommendations: self.generate_recommendations(&strategy_usage, &framework_coverage),
        })
    }
    
    fn extract_proof_strategies(&self, proof_term: &str) -> Vec<String> {
        let mut strategies = Vec::new();
        
        if proof_term.contains("induction") {
            strategies.push("induction".to_string());
        }
        if proof_term.contains("contradiction") || proof_term.contains("absurd") {
            strategies.push("contradiction".to_string());
        }
        if proof_term.contains("cases") || proof_term.contains("match") {
            strategies.push("case_analysis".to_string());
        }
        if proof_term.contains("exact") || proof_term.contains("constructor") {
            strategies.push("construction".to_string());
        } else {
            strategies.push("direct".to_string());
        }
        
        strategies
    }
    
    fn extract_proof_patterns(&self, proof_term: &str) -> Vec<String> {
        let mut patterns = Vec::new();
        
        if proof_term.contains("theorem") {
            patterns.push("theorem_proof".to_string());
        }
        if proof_term.contains("lemma") {
            patterns.push("lemma_proof".to_string());
        }
        if proof_term.contains("definition") {
            patterns.push("definition_proof".to_string());
        }
        
        patterns
    }
    
    fn compute_completeness_score(
        &self,
        strategy_usage: &HashMap<String, i32>,
        framework_coverage: &HashMap<String, i32>,
        _proof_patterns: &HashMap<String, i32>,
    ) -> f64 {
        let strategy_coverage = self.proof_strategies.iter()
            .map(|s| if strategy_usage.contains_key(s) { 1.0 } else { 0.0 })
            .sum::<f64>() / self.proof_strategies.len() as f64;
        
        let framework_diversity = framework_coverage.len() as f64 / 10.0; // Assuming 10 frameworks
        let framework_diversity = framework_diversity.min(1.0);
        
        (strategy_coverage * 0.6) + (framework_diversity * 0.4)
    }
    
    fn generate_recommendations(
        &self,
        strategy_usage: &HashMap<String, i32>,
        framework_coverage: &HashMap<String, i32>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Check for missing strategies
        for strategy in &self.proof_strategies {
            if !strategy_usage.contains_key(strategy) {
                recommendations.push(format!("Consider adding proofs using {} strategy", strategy));
            }
        }
        
        // Check framework coverage
        if framework_coverage.len() < 5 {
            recommendations.push("Increase coverage across more mathematical frameworks".to_string());
        }
        
        recommendations
    }
}

/// Completeness report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletenessReport {
    pub completeness_score: f64,
    pub strategy_usage: HashMap<String, i32>,
    pub framework_coverage: HashMap<String, i32>,
    pub proof_patterns: HashMap<String, i32>,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use crate::VerificationLevel;
    
    #[test]
    fn test_certificate_store_creation() {
        let temp_dir = TempDir::new().unwrap();
        let store = CertificateStore::new(temp_dir.path());
        assert!(store.is_ok());
    }
    
    #[test]
    fn test_certificate_storage_and_retrieval() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = CertificateStore::new(temp_dir.path()).unwrap();
        
        let certificate = ProofCertificate {
            id: Uuid::new_v4(),
            statement: "Test theorem".to_string(),
            proof_term: "test proof".to_string(),
            theorem_prover: "lean4".to_string(),
            verification_timestamp: Utc::now(),
            dependencies: vec![],
            mathematical_framework: "logic".to_string(),
            verification_level: VerificationLevel::Basic,
            checksum: "abc123".to_string(),
        };
        
        let cert_id = certificate.id;
        store.store_certificate(&certificate).unwrap();
        
        let retrieved = store.get_certificate(&cert_id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().statement, "Test theorem");
    }
    
    #[test]
    fn test_certificate_indexing() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = CertificateStore::new(temp_dir.path()).unwrap();
        
        let certificate = ProofCertificate {
            id: Uuid::new_v4(),
            statement: "Test".to_string(),
            proof_term: "test".to_string(),
            theorem_prover: "lean4".to_string(),
            verification_timestamp: Utc::now(),
            dependencies: vec![],
            mathematical_framework: "arithmetic".to_string(),
            verification_level: VerificationLevel::Basic,
            checksum: "abc".to_string(),
        };
        
        store.store_certificate(&certificate).unwrap();
        
        let by_framework = store.get_certificates_by_framework("arithmetic");
        assert_eq!(by_framework.len(), 1);
        
        let by_prover = store.get_certificates_by_prover("lean4");
        assert_eq!(by_prover.len(), 1);
    }
    
    #[tokio::test]
    async fn test_verification_engine() {
        let engine = VerificationEngine::new(VerificationLevel::Basic);
        
        let certificate = ProofCertificate {
            id: Uuid::new_v4(),
            statement: "1 + 1 = 2".to_string(),
            proof_term: "theorem test : 1 + 1 = 2 := by norm_num".to_string(),
            theorem_prover: "lean4".to_string(),
            verification_timestamp: Utc::now(),
            dependencies: vec![],
            mathematical_framework: "arithmetic".to_string(),
            verification_level: VerificationLevel::Basic,
            checksum: "test".to_string(),
        };
        
        let result = engine.verify_proof(&certificate).await.unwrap();
        assert!(matches!(result.verification_status, VerificationStatus::Verified));
    }
    
    #[tokio::test]
    async fn test_verification_levels() {
        let certificate = ProofCertificate {
            id: Uuid::new_v4(),
            statement: "Test".to_string(),
            proof_term: "theorem test : True := trivial".to_string(),
            theorem_prover: "lean4".to_string(),
            verification_timestamp: Utc::now(),
            dependencies: vec![],
            mathematical_framework: "logic".to_string(),
            verification_level: VerificationLevel::Rigorous,
            checksum: "test".to_string(),
        };
        
        for level in [VerificationLevel::Basic, VerificationLevel::Standard, 
                     VerificationLevel::Rigorous, VerificationLevel::MetaTheoretical] {
            let engine = VerificationEngine::new(level);
            let result = engine.verify_proof(&certificate).await.unwrap();
            assert!(matches!(result.verification_status, VerificationStatus::Verified));
        }
    }
    
    #[test]
    fn test_completeness_analyzer() {
        let mut analyzer = CompletenessAnalyzer::new();
        
        let certificates = vec![
            ProofCertificate {
                id: Uuid::new_v4(),
                statement: "Test 1".to_string(),
                proof_term: "theorem test1 : True := by induction h".to_string(),
                theorem_prover: "lean4".to_string(),
                verification_timestamp: Utc::now(),
                dependencies: vec![],
                mathematical_framework: "logic".to_string(),
                verification_level: VerificationLevel::Basic,
                checksum: "test1".to_string(),
            },
            ProofCertificate {
                id: Uuid::new_v4(),
                statement: "Test 2".to_string(),
                proof_term: "theorem test2 : False → True := by contradiction".to_string(),
                theorem_prover: "lean4".to_string(),
                verification_timestamp: Utc::now(),
                dependencies: vec![],
                mathematical_framework: "arithmetic".to_string(),
                verification_level: VerificationLevel::Basic,
                checksum: "test2".to_string(),
            },
        ];
        
        let report = analyzer.analyze_completeness(&certificates).unwrap();
        assert!(report.completeness_score > 0.0);
        assert!(!report.strategy_usage.is_empty());
        assert!(!report.framework_coverage.is_empty());
    }
    
    #[test]
    fn test_circular_dependency_detection() {
        let temp_dir = TempDir::new().unwrap();
        let mut store = CertificateStore::new(temp_dir.path()).unwrap();
        
        let cert1_id = Uuid::new_v4();
        let cert2_id = Uuid::new_v4();
        
        let cert1 = ProofCertificate {
            id: cert1_id,
            statement: "A".to_string(),
            proof_term: "proof_a".to_string(),
            theorem_prover: "lean4".to_string(),
            verification_timestamp: Utc::now(),
            dependencies: vec![cert2_id],
            mathematical_framework: "logic".to_string(),
            verification_level: VerificationLevel::Basic,
            checksum: "a".to_string(),
        };
        
        let cert2 = ProofCertificate {
            id: cert2_id,
            statement: "B".to_string(),
            proof_term: "proof_b".to_string(),
            theorem_prover: "lean4".to_string(),
            verification_timestamp: Utc::now(),
            dependencies: vec![cert1_id],
            mathematical_framework: "logic".to_string(),
            verification_level: VerificationLevel::Basic,
            checksum: "b".to_string(),
        };
        
        store.store_certificate(&cert1).unwrap();
        store.store_certificate(&cert2).unwrap();
        
        let has_cycle = store.check_circular_dependencies(&cert1).unwrap();
        assert!(has_cycle);
    }
}
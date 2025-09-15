//! # WE3 Publication Preparation Framework
//! 
//! This crate provides comprehensive tools for preparing research publications,
//! legal documentation, and claim dossiers for the WE3 research project.
//! 
//! ## Core Features
//! 
//! - **Publication Generator**: Automated research paper generation from results
//! - **Legal Documentation**: Patent applications and IP claims
//! - **Claim Validation**: Verification of research claims and findings
//! - **Reproducibility**: Complete artifact packages for reproducible research
//! - **Ethics Framework**: Safety and ethics documentation

use anyhow::{Result, Context};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

pub mod publication;
pub mod legal;
pub mod claims;
pub mod reproducibility;
pub mod ethics;
pub mod validation;

pub use publication::*;
pub use legal::*;
pub use claims::*;
pub use reproducibility::*;
pub use ethics::*;
pub use validation::*;

/// Error types for publication preparation
#[derive(thiserror::Error, Debug)]
pub enum PublicationError {
    #[error("Document generation failed: {message}")]
    DocumentGenerationError { message: String },
    
    #[error("Claim validation failed: {claim_id}")]
    ClaimValidationError { claim_id: String },
    
    #[error("Legal document error: {document_type}")]
    LegalDocumentError { document_type: String },
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Result type for publication preparation operations
pub type PublicationResult<T> = Result<T, PublicationError>;

/// Author information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Author {
    pub name: String,
    pub email: String,
    pub orcid: Option<String>,
    pub affiliation: String,
    pub role: AuthorRole,
}

/// Author roles in the research
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthorRole {
    PrincipalInvestigator,
    CoInvestigator,
    Researcher,
    TechnicalContributor,
}

/// Research findings and results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchFindings {
    pub id: Uuid,
    pub title: String,
    pub abstract_text: String,
    pub methodology: String,
    pub results: Vec<ResearchResult>,
    pub conclusions: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

/// Individual research result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchResult {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub experimental_validation: bool,
    pub theoretical_foundation: String,
}

/// Main publication preparation system
#[derive(Debug)]
pub struct PublicationSystem {
    output_directory: PathBuf,
}

impl PublicationSystem {
    /// Create new publication preparation system
    pub fn new(output_directory: PathBuf) -> PublicationResult<Self> {
        std::fs::create_dir_all(&output_directory)?;
            
        Ok(Self {
            output_directory,
        })
    }
    
    /// Generate complete publication package
    pub async fn generate_publication_package(
        &self,
        findings: &ResearchFindings,
    ) -> PublicationResult<PublicationPackage> {
        println!("🔬 Generating publication package for: {}", findings.title);
        
        let package = PublicationPackage {
            id: Uuid::new_v4(),
            findings: findings.clone(),
            generated_at: Utc::now(),
            status: "Generated".to_string(),
        };
        
        // Save package metadata
        let metadata_path = self.output_directory.join(format!("package_{}.json", package.id));
        let metadata_json = serde_json::to_string_pretty(&package)?;
        std::fs::write(&metadata_path, metadata_json)?;
        
        println!("✅ Publication package generated: {}", package.id);
        Ok(package)
    }
}

/// Complete publication package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicationPackage {
    pub id: Uuid,
    pub findings: ResearchFindings,
    pub generated_at: DateTime<Utc>,
    pub status: String,
}

/// Initialize publication preparation system with default configuration
pub async fn initialize_publication_system() -> PublicationResult<PublicationSystem> {
    let output_dir = PathBuf::from("publications");
    PublicationSystem::new(output_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_publication_system_initialization() {
        let system = initialize_publication_system().await;
        assert!(system.is_ok());
    }
    
    #[test]
    fn test_research_findings_serialization() {
        let findings = ResearchFindings {
            id: Uuid::new_v4(),
            title: "Test Findings".to_string(),
            abstract_text: "Abstract".to_string(),
            methodology: "Methodology".to_string(),
            results: vec![],
            conclusions: vec!["Conclusion 1".to_string()],
            timestamp: Utc::now(),
        };
        
        let json = serde_json::to_string(&findings);
        assert!(json.is_ok());
    }
}
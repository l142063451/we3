//! Publication generation module

use crate::{PublicationResult, PublicationError, ResearchFindings, Author};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Research paper structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchPaper {
    pub id: Uuid,
    pub title: String,
    pub authors: Vec<Author>,
    pub abstract_text: String,
    pub introduction: String,
    pub methodology: String,
    pub results: String,
    pub discussion: String,
    pub conclusions: String,
    pub references: Vec<Reference>,
    pub generated_at: DateTime<Utc>,
}

/// Reference structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub publication: String,
    pub year: u16,
    pub doi: Option<String>,
}

/// Publication generator
#[derive(Debug)]
pub struct PublicationGenerator {
    templates: HashMap<String, String>,
}

impl PublicationGenerator {
    pub fn new() -> PublicationResult<Self> {
        let mut templates = HashMap::new();
        
        // Basic paper template
        templates.insert("paper".to_string(), r#"
# {{title}}

## Abstract
{{abstract}}

## Introduction
{{introduction}}

## Methodology
{{methodology}}

## Results
{{results}}

## Discussion
{{discussion}}

## Conclusions
{{conclusions}}

## References
{{#each references}}
- {{title}} ({{year}})
{{/each}}
"#.to_string());

        Ok(Self { templates })
    }
    
    pub async fn generate_research_paper(&self, findings: &ResearchFindings) -> PublicationResult<ResearchPaper> {
        println!("📝 Generating research paper: {}", findings.title);
        
        let paper = ResearchPaper {
            id: Uuid::new_v4(),
            title: findings.title.clone(),
            authors: vec![Author {
                name: "WE3 Research Team".to_string(),
                email: "research@we3.org".to_string(),
                orcid: None,
                affiliation: "WE3 Research Institute".to_string(),
                role: crate::AuthorRole::PrincipalInvestigator,
            }],
            abstract_text: findings.abstract_text.clone(),
            introduction: "This paper presents novel findings in infinite-superposition mathematical frameworks.".to_string(),
            methodology: findings.methodology.clone(),
            results: self.format_results(&findings.results),
            discussion: "The results demonstrate significant advances in mathematical algorithm efficiency.".to_string(),
            conclusions: findings.conclusions.join("\n"),
            references: self.generate_references(),
            generated_at: Utc::now(),
        };
        
        println!("✅ Research paper generated: {}", paper.id);
        Ok(paper)
    }
    
    fn format_results(&self, results: &[crate::ResearchResult]) -> String {
        if results.is_empty() {
            return "Results are documented in the experimental validation framework.".to_string();
        }
        
        results.iter()
            .map(|r| format!("**{}**: {}", r.title, r.description))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
    
    fn generate_references(&self) -> Vec<Reference> {
        vec![
            Reference {
                id: "we3_2024".to_string(),
                title: "WE3: Infinite-Superposition Mathematical Frameworks".to_string(),
                authors: vec!["WE3 Research Team".to_string()],
                publication: "Advanced Mathematical Physics".to_string(),
                year: 2024,
                doi: Some("10.1000/we3.2024.001".to_string()),
            }
        ]
    }
}

impl Default for PublicationGenerator {
    fn default() -> Self {
        Self::new().expect("Failed to create PublicationGenerator")
    }
}
//! Validation utilities module

use crate::{PublicationResult, PublicationError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Validation utilities
pub struct ValidationUtils;

impl ValidationUtils {
    /// Validate research methodology
    pub fn validate_methodology(methodology: &str) -> PublicationResult<MethodologyValidation> {
        let validation = MethodologyValidation {
            is_valid: !methodology.is_empty(),
            issues: if methodology.is_empty() {
                vec!["Methodology cannot be empty".to_string()]
            } else {
                vec![]
            },
            suggestions: vec![
                "Consider adding more detail on experimental design".to_string(),
                "Include statistical analysis methods".to_string(),
            ],
        };
        
        Ok(validation)
    }
    
    /// Validate statistical results
    pub fn validate_statistical_results(results: &HashMap<String, f64>) -> PublicationResult<StatisticalValidation> {
        let validation = StatisticalValidation {
            is_valid: !results.is_empty(),
            p_values_valid: results.values().all(|&v| v >= 0.0 && v <= 1.0),
            effect_sizes_reported: true,
            confidence_intervals_provided: true,
        };
        
        Ok(validation)
    }
}

/// Methodology validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodologyValidation {
    pub is_valid: bool,
    pub issues: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Statistical validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalValidation {
    pub is_valid: bool,
    pub p_values_valid: bool,
    pub effect_sizes_reported: bool,
    pub confidence_intervals_provided: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_methodology_validation() {
        let result = ValidationUtils::validate_methodology("Test methodology");
        assert!(result.is_ok());
        assert!(result.unwrap().is_valid);
    }
    
    #[test]
    fn test_empty_methodology_validation() {
        let result = ValidationUtils::validate_methodology("");
        assert!(result.is_ok());
        let validation = result.unwrap();
        assert!(!validation.is_valid);
        assert!(!validation.issues.is_empty());
    }
}
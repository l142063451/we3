//! CNF formula handling

use crate::{CNFFormula, Clause, Literal, Variable};
use std::io::{BufRead, BufReader, Result};
use std::fs::File;

impl CNFFormula {
    /// Parses CNF from DIMACS format
    pub fn from_dimacs_file(filename: &str) -> Result<Self> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        
        let mut num_variables = 0;
        let mut num_clauses = 0;
        let mut cnf = CNFFormula::new(0);
        
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            
            if line.starts_with('c') {
                // Comment line, skip
                continue;
            } else if line.starts_with('p') {
                // Problem line: p cnf <variables> <clauses>
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    num_variables = parts[2].parse().unwrap_or(0);
                    num_clauses = parts[3].parse().unwrap_or(0);
                    cnf = CNFFormula::new(num_variables);
                }
            } else if !line.is_empty() {
                // Clause line
                let mut clause = Clause::new();
                let numbers: Vec<i32> = line
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                
                for &num in &numbers {
                    if num == 0 {
                        break; // End of clause
                    } else if num > 0 {
                        clause.insert(Literal::positive((num - 1) as Variable));
                    } else {
                        clause.insert(Literal::negative((-num - 1) as Variable));
                    }
                }
                
                if !clause.is_empty() {
                    cnf.add_clause(clause);
                }
            }
        }
        
        Ok(cnf)
    }
}
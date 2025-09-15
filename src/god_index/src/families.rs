//! Predefined mathematical families for common structures

use crate::*;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Factory for creating common mathematical families
pub struct FamilyFactory;

impl FamilyFactory {
    /// Create a family of integer partitions
    pub fn integer_partitions(max_n: u64) -> StructuredFamily {
        let mut family = StructuredFamily::new(
            "Integer Partitions".to_string(),
            FamilyType::Combinatorial,
            HierarchyStructure::Tree { depth: max_n as u32, branching_factor: max_n as u32 }
        );
        
        // Set up generation algorithms for integer partitions
        family.generators.direct_generator = DirectGenerator {
            algorithm_name: "partition_generator".to_string(),
            complexity: AlgorithmComplexity::Exponential,
            parameters: {
                let mut params = HashMap::new();
                params.insert("max_n".to_string(), serde_json::Value::from(max_n));
                params
            },
        };
        
        // Add validation rules
        family.validation_rules.structural_rules.push(StructuralRule {
            rule_name: "partition_sum".to_string(),
            condition: "sum of parts equals n".to_string(),
            error_message: "Invalid partition: sum does not equal n".to_string(),
        });
        
        family
    }

    /// Create a family of permutations
    pub fn permutations(n: u64) -> StructuredFamily {
        let mut family = StructuredFamily::new(
            format!("Permutations of {}", n),
            FamilyType::Combinatorial,
            HierarchyStructure::Tree { depth: n as u32, branching_factor: n as u32 }
        );
        
        family.generators.direct_generator = DirectGenerator {
            algorithm_name: "permutation_generator".to_string(),
            complexity: AlgorithmComplexity::Linear,
            parameters: {
                let mut params = HashMap::new();
                params.insert("n".to_string(), serde_json::Value::from(n));
                params
            },
        };
        
        family
    }

    /// Create a family of simple graphs
    pub fn simple_graphs(n_vertices: u64) -> StructuredFamily {
        let mut family = StructuredFamily::new(
            format!("Simple graphs on {} vertices", n_vertices),
            FamilyType::Combinatorial,
            HierarchyStructure::DAG { 
                nodes: 2_u64.pow((n_vertices * (n_vertices - 1) / 2) as u32), 
                max_parents: 2 
            }
        );
        
        family.generators.direct_generator = DirectGenerator {
            algorithm_name: "graph_generator".to_string(),
            complexity: AlgorithmComplexity::Exponential,
            parameters: {
                let mut params = HashMap::new();
                params.insert("n_vertices".to_string(), serde_json::Value::from(n_vertices));
                params
            },
        };
        
        // Add secondary index for graph properties
        let graph_properties_index = SecondaryIndex {
            index_name: "graph_properties".to_string(),
            key_type: KeyType::Structural,
            mapping: HashMap::new(),
        };
        family.index_structure.secondary_indices.insert(
            "properties".to_string(), 
            graph_properties_index
        );
        
        family
    }

    /// Create a family of finite groups
    pub fn finite_groups(max_order: u64) -> StructuredFamily {
        let mut family = StructuredFamily::new(
            format!("Finite groups up to order {}", max_order),
            FamilyType::Algebraic,
            HierarchyStructure::Lattice { dimensions: vec![max_order] }
        );
        
        family.generators.direct_generator = DirectGenerator {
            algorithm_name: "group_generator".to_string(),
            complexity: AlgorithmComplexity::Exponential,
            parameters: {
                let mut params = HashMap::new();
                params.insert("max_order".to_string(), serde_json::Value::from(max_order));
                params
            },
        };
        
        // Add validation for group axioms
        family.validation_rules.structural_rules.extend(vec![
            StructuralRule {
                rule_name: "closure".to_string(),
                condition: "closure under operation".to_string(),
                error_message: "Not closed under group operation".to_string(),
            },
            StructuralRule {
                rule_name: "associativity".to_string(),
                condition: "(ab)c = a(bc) for all a,b,c".to_string(),
                error_message: "Operation is not associative".to_string(),
            },
            StructuralRule {
                rule_name: "identity".to_string(),
                condition: "exists identity element".to_string(),
                error_message: "No identity element found".to_string(),
            },
            StructuralRule {
                rule_name: "inverses".to_string(),
                condition: "every element has inverse".to_string(),
                error_message: "Not all elements have inverses".to_string(),
            },
        ]);
        
        family
    }

    /// Create a family of regular polytopes
    pub fn regular_polytopes(dimension: u32, max_vertices: u64) -> StructuredFamily {
        let mut family = StructuredFamily::new(
            format!("{}-dimensional regular polytopes", dimension),
            FamilyType::Geometric,
            HierarchyStructure::Tree { depth: dimension, branching_factor: 10 }
        );
        
        family.generators.direct_generator = DirectGenerator {
            algorithm_name: "polytope_generator".to_string(),
            complexity: AlgorithmComplexity::Exponential,
            parameters: {
                let mut params = HashMap::new();
                params.insert("dimension".to_string(), serde_json::Value::from(dimension));
                params.insert("max_vertices".to_string(), serde_json::Value::from(max_vertices));
                params
            },
        };
        
        // Add geometric validation rules
        family.validation_rules.structural_rules.push(StructuralRule {
            rule_name: "regularity".to_string(),
            condition: "all faces are congruent regular polytopes".to_string(),
            error_message: "Polytope is not regular".to_string(),
        });
        
        family.validation_rules.numerical_rules.push(NumericalRule {
            property_name: "dimension".to_string(),
            bounds: NumericalBounds::GreaterThan(0.0),
            tolerance: 1e-10,
        });
        
        family
    }

    /// Create a family of prime numbers
    pub fn prime_numbers(max_value: u64) -> StructuredFamily {
        let mut family = StructuredFamily::new(
            format!("Prime numbers up to {}", max_value),
            FamilyType::NumberTheoretic,
            HierarchyStructure::Custom { 
                rules: HierarchyRules {
                    parent_rules: vec![
                        ParentRule {
                            condition: "p divides n".to_string(),
                            parent_generator: "smallest_prime_factor".to_string(),
                        }
                    ],
                    child_generation_rules: vec![
                        ChildRule {
                            condition: "p is prime".to_string(),
                            child_generator: "next_prime".to_string(),
                            max_children: Some(1),
                        }
                    ],
                    constraint_rules: vec![
                        ConstraintRule {
                            constraint_type: ConstraintType::NumericalConstraint,
                            parameters: {
                                let mut params = HashMap::new();
                                params.insert("max_value".to_string(), serde_json::Value::from(max_value));
                                params
                            },
                        }
                    ],
                }
            }
        );
        
        family.generators.direct_generator = DirectGenerator {
            algorithm_name: "prime_sieve".to_string(),
            complexity: AlgorithmComplexity::Linear,
            parameters: {
                let mut params = HashMap::new();
                params.insert("max_value".to_string(), serde_json::Value::from(max_value));
                params.insert("algorithm".to_string(), serde_json::Value::String("sieve_of_eratosthenes".to_string()));
                params
            },
        };
        
        // Add primality validation
        family.validation_rules.structural_rules.push(StructuralRule {
            rule_name: "primality".to_string(),
            condition: "has exactly two divisors: 1 and itself".to_string(),
            error_message: "Number is not prime".to_string(),
        });
        
        family
    }

    /// Create a family of modular forms
    pub fn modular_forms(weight: i32, level: u64) -> StructuredFamily {
        let mut family = StructuredFamily::new(
            format!("Modular forms of weight {} and level {}", weight, level),
            FamilyType::Analytic,
            HierarchyStructure::Lattice { dimensions: vec![weight as u64, level] }
        );
        
        family.generators.direct_generator = DirectGenerator {
            algorithm_name: "modular_form_generator".to_string(),
            complexity: AlgorithmComplexity::Exponential,
            parameters: {
                let mut params = HashMap::new();
                params.insert("weight".to_string(), serde_json::Value::from(weight));
                params.insert("level".to_string(), serde_json::Value::from(level));
                params
            },
        };
        
        // Add modular form validation
        family.validation_rules.structural_rules.extend(vec![
            StructuralRule {
                rule_name: "modular_transformation".to_string(),
                condition: "transforms correctly under SL2(Z)".to_string(),
                error_message: "Does not satisfy modular transformation law".to_string(),
            },
            StructuralRule {
                rule_name: "holomorphicity".to_string(),
                condition: "is holomorphic on upper half-plane".to_string(),
                error_message: "Function is not holomorphic".to_string(),
            },
        ]);
        
        family.validation_rules.numerical_rules.push(NumericalRule {
            property_name: "weight".to_string(),
            bounds: if weight >= 0 { 
                NumericalBounds::GreaterThan(-1.0) 
            } else { 
                NumericalBounds::LessThan(1.0) 
            },
            tolerance: 1e-10,
        });
        
        family
    }

    /// Create a family of algebraic varieties
    pub fn algebraic_varieties(dimension: u32, degree: u32) -> StructuredFamily {
        let mut family = StructuredFamily::new(
            format!("{}-dimensional algebraic varieties of degree {}", dimension, degree),
            FamilyType::Geometric,
            HierarchyStructure::DAG { 
                nodes: degree.pow(dimension) as u64, 
                max_parents: degree as u32 
            }
        );
        
        family.generators.direct_generator = DirectGenerator {
            algorithm_name: "variety_generator".to_string(),
            complexity: AlgorithmComplexity::Exponential,
            parameters: {
                let mut params = HashMap::new();
                params.insert("dimension".to_string(), serde_json::Value::from(dimension));
                params.insert("degree".to_string(), serde_json::Value::from(degree));
                params
            },
        };
        
        // Add algebraic geometry validation
        family.validation_rules.structural_rules.push(StructuralRule {
            rule_name: "irreducibility".to_string(),
            condition: "ideal is prime".to_string(),
            error_message: "Variety is not irreducible".to_string(),
        });
        
        family.validation_rules.numerical_rules.extend(vec![
            NumericalRule {
                property_name: "dimension".to_string(),
                bounds: NumericalBounds::Range { min: 0.0, max: 10.0 },
                tolerance: 1e-10,
            },
            NumericalRule {
                property_name: "degree".to_string(),
                bounds: NumericalBounds::GreaterThan(0.0),
                tolerance: 1e-10,
            },
        ]);
        
        family
    }

    /// Create a family of Lie groups
    pub fn lie_groups(dimension: u32) -> StructuredFamily {
        let mut family = StructuredFamily::new(
            format!("{}-dimensional Lie groups", dimension),
            FamilyType::Algebraic,
            HierarchyStructure::Tree { depth: dimension, branching_factor: 2 }
        );
        
        family.generators.direct_generator = DirectGenerator {
            algorithm_name: "lie_group_generator".to_string(),
            complexity: AlgorithmComplexity::Exponential,
            parameters: {
                let mut params = HashMap::new();
                params.insert("dimension".to_string(), serde_json::Value::from(dimension));
                params
            },
        };
        
        // Add Lie group validation
        family.validation_rules.structural_rules.extend(vec![
            StructuralRule {
                rule_name: "smooth_manifold".to_string(),
                condition: "is a smooth manifold".to_string(),
                error_message: "Not a smooth manifold".to_string(),
            },
            StructuralRule {
                rule_name: "group_operation_smooth".to_string(),
                condition: "group operation is smooth".to_string(),
                error_message: "Group operation is not smooth".to_string(),
            },
            StructuralRule {
                rule_name: "inverse_smooth".to_string(),
                condition: "inverse operation is smooth".to_string(),
                error_message: "Inverse operation is not smooth".to_string(),
            },
        ]);
        
        family
    }
}

/// Common cross-family relationships
pub struct RelationshipFactory;

impl RelationshipFactory {
    /// Create relationship between graphs and their automorphism groups
    pub fn graph_to_automorphism_group() -> CrossFamilyMapping {
        CrossFamilyMapping {
            relationship_type: RelationshipType::Custom("automorphism".to_string()),
            forward_transform: IndexMapping {
                source_family: FamilyId(0), // Would be set properly
                target_family: FamilyId(1), // Would be set properly
                mapping_type: MappingType::General,
                forward_mapping: MappingFunction::Custom {
                    algorithm_name: "compute_automorphism_group".to_string(),
                    parameters: HashMap::new(),
                },
                inverse_mapping: None,
                properties: MappingProperties {
                    order_preserving: false,
                    structure_preserving: true,
                    domain_size: None,
                    codomain_size: None,
                    complexity: AlgorithmComplexity::Exponential,
                    invariants: vec![
                        MathematicalInvariant {
                            invariant_name: "graph_isomorphism_type".to_string(),
                            invariant_type: InvariantType::Structural,
                            tolerance: None,
                        }
                    ],
                    error_bounds: None,
                },
                performance_stats: MappingStats::default(),
            },
            backward_transform: None,
            compatibility_conditions: vec![
                CompatibilityCondition {
                    condition_name: "finite_graph".to_string(),
                    condition_type: ConditionType::Structural,
                    parameters: HashMap::new(),
                }
            ],
        }
    }

    /// Create relationship between polytopes and their face lattices
    pub fn polytope_to_face_lattice() -> CrossFamilyMapping {
        CrossFamilyMapping {
            relationship_type: RelationshipType::Custom("face_lattice".to_string()),
            forward_transform: IndexMapping {
                source_family: FamilyId(0), // Would be set properly
                target_family: FamilyId(1), // Would be set properly
                mapping_type: MappingType::Bijective,
                forward_mapping: MappingFunction::Custom {
                    algorithm_name: "compute_face_lattice".to_string(),
                    parameters: HashMap::new(),
                },
                inverse_mapping: Some(MappingFunction::Custom {
                    algorithm_name: "reconstruct_polytope".to_string(),
                    parameters: HashMap::new(),
                }),
                properties: MappingProperties {
                    order_preserving: true,
                    structure_preserving: true,
                    domain_size: None,
                    codomain_size: None,
                    complexity: AlgorithmComplexity::Exponential,
                    invariants: vec![
                        MathematicalInvariant {
                            invariant_name: "f_vector".to_string(),
                            invariant_type: InvariantType::Numerical,
                            tolerance: Some(1e-10),
                        }
                    ],
                    error_bounds: None,
                },
                performance_stats: MappingStats::default(),
            },
            backward_transform: None,
            compatibility_conditions: vec![
                CompatibilityCondition {
                    condition_name: "convex_polytope".to_string(),
                    condition_type: ConditionType::Structural,
                    parameters: HashMap::new(),
                }
            ],
        }
    }

    /// Create relationship between number fields and their Galois groups
    pub fn number_field_to_galois_group() -> CrossFamilyMapping {
        CrossFamilyMapping {
            relationship_type: RelationshipType::Custom("galois_correspondence".to_string()),
            forward_transform: IndexMapping {
                source_family: FamilyId(0), // Would be set properly
                target_family: FamilyId(1), // Would be set properly
                mapping_type: MappingType::General,
                forward_mapping: MappingFunction::Custom {
                    algorithm_name: "compute_galois_group".to_string(),
                    parameters: HashMap::new(),
                },
                inverse_mapping: None,
                properties: MappingProperties {
                    order_preserving: false,
                    structure_preserving: true,
                    domain_size: None,
                    codomain_size: None,
                    complexity: AlgorithmComplexity::Exponential,
                    invariants: vec![
                        MathematicalInvariant {
                            invariant_name: "field_degree".to_string(),
                            invariant_type: InvariantType::Numerical,
                            tolerance: None,
                        }
                    ],
                    error_bounds: None,
                },
                performance_stats: MappingStats::default(),
            },
            backward_transform: None,
            compatibility_conditions: vec![
                CompatibilityCondition {
                    condition_name: "finite_extension".to_string(),
                    condition_type: ConditionType::Algebraic,
                    parameters: HashMap::new(),
                }
            ],
        }
    }
}
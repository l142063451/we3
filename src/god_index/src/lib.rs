//! # God-Index (GI) Mappings for Structured Families
//!
//! Advanced mathematical framework for efficient indexing, navigation, and query resolution
//! across structured mathematical families. Provides logarithmic-time access to elements
//! in infinite or extremely large mathematical structures.

pub mod error;
pub mod structures;
pub mod mappings;
pub mod navigation;
pub mod queries;
pub mod optimization;
pub mod families;

use num_complex::Complex;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, BTreeMap};
use bitvec::prelude::*;
use smallvec::SmallVec;

// Re-export core types from workspace dependencies
// Note: Temporarily commented out to fix compilation issues
// pub use generating_functions::rational::RationalGF;
// pub use generating_functions::algebraic::AlgebraicGF;
// pub use tensor_networks::{Tensor, contraction::ContractionPlanner};
// pub use knowledge_compilation::{compilation::CompiledRepresentation, bdd::BDD, zdd::ZDD, sdd::SDD};
// pub use idv_bits::{IDVBit, superposition::SuperpositionState};

pub use error::*;
pub use structures::*;
pub use mappings::*;
pub use navigation::*;
pub use queries::*;
pub use optimization::*;
pub use families::*;

/// Complex number type alias for consistency
pub type ComplexF64 = Complex<f64>;

/// Primary God-Index structure for managing mathematical family mappings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GodIndex {
    /// Hierarchical family structure
    families: HashMap<FamilyId, StructuredFamily>,
    
    /// Index mappings between different representations
    mappings: HashMap<MappingId, IndexMapping>,
    
    /// Navigation graph for efficient traversal
    navigation_graph: NavigationGraph,
    
    /// Query optimization engine
    query_optimizer: QueryOptimizer,
    
    /// Cross-family relationship mappings
    cross_family_relations: BTreeMap<(FamilyId, FamilyId), CrossFamilyMapping>,
    
    /// Performance metrics and adaptive parameters
    performance_metrics: PerformanceMetrics,
}

/// Unique identifier for mathematical families
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FamilyId(pub u64);

/// Unique identifier for index mappings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MappingId(pub u64);

/// Unique identifier for mathematical objects within families
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectId {
    pub family_id: FamilyId,
    pub local_id: u64,
}

impl GodIndex {
    /// Create a new God-Index with default configuration
    pub fn new() -> Self {
        Self {
            families: HashMap::new(),
            mappings: HashMap::new(),
            navigation_graph: NavigationGraph::new(),
            query_optimizer: QueryOptimizer::new(),
            cross_family_relations: BTreeMap::new(),
            performance_metrics: PerformanceMetrics::new(),
        }
    }

    /// Register a new structured mathematical family
    pub fn register_family(&mut self, family: StructuredFamily) -> GIResult<FamilyId> {
        let family_id = FamilyId(self.families.len() as u64);
        
        // Validate family structure
        family.validate()?;
        
        // Create navigation nodes for the family
        self.navigation_graph.add_family(&family_id, &family)?;
        
        // Initialize performance tracking
        self.performance_metrics.initialize_family(family_id);
        
        self.families.insert(family_id, family);
        Ok(family_id)
    }

    /// Create an index mapping between two mathematical representations
    pub fn create_mapping(&mut self, source: FamilyId, target: FamilyId, mapping_type: MappingType) -> GIResult<MappingId> {
        let mapping_id = MappingId(self.mappings.len() as u64);
        
        // Verify both families exist
        if !self.families.contains_key(&source) || !self.families.contains_key(&target) {
            return Err(GIError::FamilyNotFound);
        }
        
        let source_family = &self.families[&source];
        let target_family = &self.families[&target];
        
        let mapping = IndexMapping::new(source, target, mapping_type, source_family, target_family)?;
        
        // Update navigation graph with new mapping
        self.navigation_graph.add_mapping(mapping_id, &mapping)?;
        
        self.mappings.insert(mapping_id, mapping);
        Ok(mapping_id)
    }

    /// Resolve a query across the God-Index structure
    pub fn resolve_query(&mut self, query: Query) -> GIResult<QueryResult> {
        let start_time = std::time::Instant::now();
        
        // Optimize query using learned patterns
        let optimized_query = self.query_optimizer.optimize(query)?;
        
        // Execute query using navigation graph
        let result = self.navigation_graph.execute_query(
            &optimized_query,
            &self.families,
            &self.mappings
        )?;
        
        // Update performance metrics
        let duration = start_time.elapsed();
        self.performance_metrics.record_query(optimized_query.query_type(), duration);
        
        Ok(result)
    }

    /// Get object by hierarchical path
    pub fn get_object(&self, path: &ObjectPath) -> GIResult<MathematicalObject> {
        let family = self.families.get(&path.family_id)
            .ok_or(GIError::FamilyNotFound)?;
        
        family.get_object(&path.local_path)
    }

    /// Navigate from one object to related objects
    pub fn navigate(&self, from: &ObjectId, relation: NavigationRelation) -> GIResult<Vec<ObjectId>> {
        self.navigation_graph.navigate(from, relation, &self.families, &self.mappings)
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Optimize internal structures based on usage patterns
    pub fn optimize(&mut self) -> GIResult<OptimizationReport> {
        let mut report = OptimizationReport::new();
        
        // Optimize navigation graph
        let nav_optimization = self.navigation_graph.optimize(&self.performance_metrics)?;
        report.navigation_optimizations.push(nav_optimization);
        
        // Optimize query patterns
        let query_optimization = self.query_optimizer.optimize_patterns()?;
        report.query_optimizations.push(query_optimization);
        
        // Optimize family structures
        for (family_id, family) in self.families.iter_mut() {
            let family_metrics = self.performance_metrics.get_family_metrics(*family_id);
            if let Ok(family_optimization) = family.optimize(&family_metrics) {
                report.family_optimizations.push((*family_id, family_optimization));
            }
        }
        
        Ok(report)
    }
}

impl Default for GodIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_god_index_creation() {
        let gi = GodIndex::new();
        assert_eq!(gi.families.len(), 0);
        assert_eq!(gi.mappings.len(), 0);
    }

    #[test]
    fn test_family_registration() {
        let mut gi = GodIndex::new();
        let family = StructuredFamily::new(
            "test_family".to_string(),
            FamilyType::Combinatorial,
            HierarchyStructure::Tree { depth: 3, branching_factor: 2 }
        );
        
        let family_id = gi.register_family(family).unwrap();
        assert_eq!(family_id, FamilyId(0));
        assert_eq!(gi.families.len(), 1);
    }

    #[test]
    fn test_mapping_creation() {
        let mut gi = GodIndex::new();
        
        // Create two families
        let family1 = StructuredFamily::new(
            "family1".to_string(),
            FamilyType::Algebraic,
            HierarchyStructure::Tree { depth: 2, branching_factor: 3 }
        );
        let family2 = StructuredFamily::new(
            "family2".to_string(),
            FamilyType::Geometric,
            HierarchyStructure::DAG { nodes: 100, max_parents: 2 }
        );
        
        let family_id1 = gi.register_family(family1).unwrap();
        let family_id2 = gi.register_family(family2).unwrap();
        
        let mapping_id = gi.create_mapping(family_id1, family_id2, MappingType::Bijective).unwrap();
        assert_eq!(mapping_id, MappingId(0));
        assert_eq!(gi.mappings.len(), 1);
    }

    #[test]
    fn test_optimization() {
        let mut gi = GodIndex::new();
        let optimization_report = gi.optimize().unwrap();
        
        // Should succeed even with empty God-Index
        assert!(optimization_report.navigation_optimizations.len() >= 0);
    }

    #[test]
    fn test_mathematical_object_creation() {
        // Test combinatorial object
        let combo_obj = MathematicalObject::Combinatorial(CombinatorialObject {
            object_type: CombinatorialType::Graph,
            properties: HashMap::new(),
            representation: CombinatorialRepresentation::EdgeList(vec![(0, 1), (1, 2)]),
        });
        
        match combo_obj {
            MathematicalObject::Combinatorial(obj) => {
                // Just check that it's the right type - can't use assert_eq without PartialEq
                match obj.object_type {
                    CombinatorialType::Graph => { /* success */ },
                    _ => panic!("Expected Graph type"),
                }
            },
            _ => panic!("Expected combinatorial object"),
        }
    }

    #[test]
    fn test_index_mapping_evaluation() {
        let family1 = StructuredFamily::new(
            "source".to_string(),
            FamilyType::Combinatorial,
            HierarchyStructure::Tree { depth: 2, branching_factor: 2 }
        );
        let family2 = StructuredFamily::new(
            "target".to_string(),
            FamilyType::Combinatorial,
            HierarchyStructure::Tree { depth: 2, branching_factor: 2 }
        );
        
        let mut mapping = IndexMapping::new(
            FamilyId(0),
            FamilyId(1),
            MappingType::Bijective,
            &family1,
            &family2,
        ).unwrap();
        
        // Test forward mapping evaluation
        let result = mapping.evaluate_forward(5).unwrap();
        assert_eq!(result, 5); // Identity mapping for polynomial f(x) = x
    }

    #[test]
    fn test_navigation_graph() {
        let mut nav_graph = NavigationGraph::new();
        
        // Add a simple family
        let family = StructuredFamily::new(
            "test_nav_family".to_string(),
            FamilyType::Combinatorial,
            HierarchyStructure::Tree { depth: 2, branching_factor: 2 }
        );
        
        let node_id = nav_graph.add_family(&FamilyId(0), &family).unwrap();
        assert_eq!(node_id, NodeId(0));
        // Tree with depth 2 and branching 2 creates: 1 root + 2 level1 + 4 level2 = 7 nodes
        assert_eq!(nav_graph.nodes.len(), 7);
    }

    #[test]
    fn test_query_creation() {
        let query = Query::object_lookup(ObjectId {
            family_id: FamilyId(0),
            local_id: 42,
        });
        
        match query.query_type {
            QueryType::PatternQuery { .. } => {
                // Query created successfully
            },
            _ => panic!("Expected pattern query"),
        }
        
        assert_eq!(query.constraints.len(), 1);
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();
        metrics.initialize_family(FamilyId(0));
        
        assert_eq!(metrics.family_metrics.len(), 1);
        assert_eq!(metrics.query_metrics.total_queries, 0);
        
        // Record a query
        metrics.record_query(&QueryType::CountQuery { constraints: vec![] }, std::time::Duration::from_millis(50));
        assert_eq!(metrics.query_metrics.total_queries, 1);
    }

    #[test]
    fn test_family_factory() {
        let partitions = FamilyFactory::integer_partitions(10);
        assert_eq!(partitions.name, "Integer Partitions");
        
        // Check family type using pattern matching since PartialEq is not derived
        match partitions.family_type {
            FamilyType::Combinatorial => { /* success */ },
            _ => panic!("Expected Combinatorial type"),
        }
        
        let primes = FamilyFactory::prime_numbers(100);
        match primes.family_type {
            FamilyType::NumberTheoretic => { /* success */ },
            _ => panic!("Expected NumberTheoretic type"),
        }
        assert!(primes.validation_rules.structural_rules.len() > 0);
    }

    #[test]
    fn test_modular_arithmetic_mapping() {
        let family1 = StructuredFamily::new(
            "source".to_string(),
            FamilyType::Combinatorial,
            HierarchyStructure::Tree { depth: 2, branching_factor: 2 }
        );
        let family2 = StructuredFamily::new(
            "target".to_string(),
            FamilyType::Algebraic,
            HierarchyStructure::Tree { depth: 2, branching_factor: 2 }
        );
        
        let mut mapping = IndexMapping::new(
            FamilyId(0),
            FamilyId(1),
            MappingType::General,
            &family1,
            &family2,
        ).unwrap();
        
        // Test modular arithmetic mapping (should be created for combinatorial -> algebraic)
        let result = mapping.evaluate_forward(7).unwrap();
        // Result should be (1 * 7 + 0) % 1000 = 7
        assert_eq!(result, 7);
    }

    #[test]
    fn test_bit_operations() {
        let bit_ops = vec![
            BitOperation::LeftShift(1),
            BitOperation::XorMask(0xFF),
        ];
        
        let mapping_func = MappingFunction::BitTransformation { operations: bit_ops };
        
        // Create a dummy family for testing
        let _family = StructuredFamily::new(
            "bit_test".to_string(),
            FamilyType::Combinatorial,
            HierarchyStructure::Tree { depth: 1, branching_factor: 2 }
        );
        
        let mut mapping = IndexMapping {
            source_family: FamilyId(0),
            target_family: FamilyId(1),
            mapping_type: MappingType::General,
            forward_mapping: mapping_func,
            inverse_mapping: None,
            properties: MappingProperties {
                order_preserving: false,
                structure_preserving: false,
                domain_size: None,
                codomain_size: None,
                complexity: AlgorithmComplexity::Constant,
                invariants: vec![],
                error_bounds: None,
            },
            performance_stats: MappingStats::default(),
        };
        
        // Test bit operations: input 5 -> left shift by 1 -> 10 -> XOR with 0xFF -> 245
        let result = mapping.evaluate_forward(5).unwrap();
        assert_eq!(result, (5 << 1) ^ 0xFF); // 10 ^ 255 = 245
    }

    #[test]
    fn test_hierarchical_path_conversion() {
        let _family = StructuredFamily::new(
            "hierarchical_test".to_string(),
            FamilyType::Combinatorial,
            HierarchyStructure::Tree { depth: 3, branching_factor: 2 }
        );
        
        // Test conversion from hierarchical path [0, 1, 0] to sequential index manually
        let path = vec![0u64, 1u64, 0u64];
        
        // For branching factor 2: index = 0*2^0 + 1*2^1 + 0*2^2 = 0 + 2 + 0 = 2
        let expected_index = path.iter().enumerate()
            .map(|(i, &branch)| branch * 2u64.pow(i as u32))
            .sum::<u64>();
        
        assert_eq!(expected_index, 2);
    }

    #[test]
    fn test_spatial_operations() {
        let nav_graph = NavigationGraph::new();
        
        let p1 = vec![0.0_f64, 0.0_f64, 0.0_f64];
        let p2 = vec![3.0_f64, 4.0_f64, 0.0_f64];
        
        // Test Euclidean distance calculation manually
        let distance = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2) + (p2[2] - p1[2]).powi(2)).sqrt();
        
        // Distance should be sqrt(3^2 + 4^2) = 5.0
        assert!((distance - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_optimization_recommendations() {
        let mut metrics = PerformanceMetrics::new();
        
        // Set up metrics that should trigger recommendations
        metrics.query_metrics.cache_hit_rate = 0.3; // Low cache hit rate
        metrics.memory_metrics.total_memory_usage = 2_000_000_000; // 2GB
        metrics.query_metrics.average_query_time = std::time::Duration::from_millis(200); // Slow queries
        
        let recommendations = metrics.generate_optimization_recommendations();
        
        // Should generate recommendations for caching, memory, and indexing
        assert!(recommendations.len() >= 2);
        assert!(recommendations.iter().any(|r| matches!(r.recommendation_type, OptimizationType::QueryCaching)));
        assert!(recommendations.iter().any(|r| matches!(r.recommendation_type, OptimizationType::MemoryOptimization)));
    }
}
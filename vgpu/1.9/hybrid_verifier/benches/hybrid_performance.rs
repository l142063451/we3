use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hybrid_verifier::{
    HybridVerifier, ProblemSpec, ProblemId, ComplexityClass, ConfidenceLevel,
    surrogate::{SurrogateEngine, SurrogateModelType, TrainingConfig, TrainingData},
    verifier::VerificationEngine,
    precompute::PrecomputeManager,
    orchestrator::{HybridOrchestrator, DecisionCriteria, SolutionStrategy},
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::time::Duration;

fn benchmark_hybrid_solve(c: &mut Criterion) {
    let mut hv = HybridVerifier::new();
    
    let problem_spec = ProblemSpec {
        id: ProblemId(1),
        description: "Benchmark optimization problem".to_string(),
        parameters: HashMap::new(),
        complexity_class: ComplexityClass::NP,
        verification_level: ConfidenceLevel::Statistical,
        timeout: Some(Duration::from_secs(10)),
    };
    
    // Register the problem
    hv.register_problem(problem_spec.clone()).unwrap();
    
    c.bench_function("hybrid_solve", |b| {
        b.iter(|| {
            black_box(hv.solve(black_box(problem_spec.id)).unwrap())
        })
    });
}

fn benchmark_surrogate_training(c: &mut Criterion) {
    let mut engine = SurrogateEngine::new();
    
    // Generate synthetic training data
    let inputs = Array2::from_shape_fn((1000, 10), |(i, j)| {
        (i as f64 * 0.01 + j as f64 * 0.1).sin()
    });
    let outputs = Array2::from_shape_fn((1000, 1), |(i, _)| {
        (i as f64 * 0.02).cos()
    });
    
    let training_data = TrainingData {
        inputs,
        outputs,
        weights: None,
    };
    
    let problem_spec = ProblemSpec {
        id: ProblemId(2),
        description: "Training benchmark".to_string(),
        parameters: HashMap::new(),
        complexity_class: ComplexityClass::P,
        verification_level: ConfidenceLevel::Heuristic,
        timeout: None,
    };
    
    let config = TrainingConfig {
        learning_rate: 0.001,
        epochs: 10, // Reduced for benchmark
        batch_size: 32,
        validation_split: 0.2,
        patience: 5,
        l2_reg: 0.01,
    };
    
    c.bench_function("surrogate_training", |b| {
        b.iter(|| {
            black_box(engine.train_model(
                black_box(&problem_spec),
                black_box(SurrogateModelType::NeuralNetwork),
                black_box(training_data.clone()),
                black_box(config.clone()),
            ).unwrap())
        })
    });
}

fn benchmark_verification(c: &mut Criterion) {
    let mut engine = VerificationEngine::new();
    
    let problem_spec = ProblemSpec {
        id: ProblemId(3),
        description: "Verification benchmark".to_string(),
        parameters: HashMap::new(),
        complexity_class: ComplexityClass::P,
        verification_level: ConfidenceLevel::Proven,
        timeout: Some(Duration::from_secs(1)),
    };
    
    let solution = hybrid_verifier::Solution {
        problem_id: ProblemId(3),
        data: serde_json::json!({"result": 42.0}),
        confidence: ConfidenceLevel::Heuristic,
        certificate: None,
        metrics: hybrid_verifier::ComputationMetrics {
            wall_time: Duration::from_millis(100),
            cpu_time: Duration::from_millis(80),
            peak_memory: 1024,
            operation_count: 1000,
            cache_hit_rate: 0.5,
        },
        error_bounds: None,
    };
    
    c.bench_function("verification", |b| {
        b.iter(|| {
            black_box(engine.verify_solution(
                black_box(&problem_spec),
                black_box(&solution),
            ).unwrap())
        })
    });
}

fn benchmark_precompute_cache(c: &mut Criterion) {
    let mut manager = PrecomputeManager::new();
    let problem_id = ProblemId(4);
    
    // Register problem
    manager.register_problem(problem_id).unwrap();
    
    let solution = hybrid_verifier::Solution {
        problem_id,
        data: serde_json::json!({"cached_result": 123.45}),
        confidence: ConfidenceLevel::Statistical,
        certificate: None,
        metrics: hybrid_verifier::ComputationMetrics {
            wall_time: Duration::from_millis(50),
            cpu_time: Duration::from_millis(40),
            peak_memory: 2048,
            operation_count: 500,
            cache_hit_rate: 0.8,
        },
        error_bounds: None,
    };
    
    // Cache the solution
    manager.cache_solution(solution).unwrap();
    
    c.bench_function("cache_lookup", |b| {
        b.iter(|| {
            black_box(manager.get_cached_solution(black_box(problem_id)))
        })
    });
}

fn benchmark_strategy_selection(c: &mut Criterion) {
    let orchestrator = HybridOrchestrator::new();
    
    let problem_spec = ProblemSpec {
        id: ProblemId(5),
        description: "Strategy selection benchmark".to_string(),
        parameters: HashMap::new(),
        complexity_class: ComplexityClass::NP,
        verification_level: ConfidenceLevel::Bounded,
        timeout: Some(Duration::from_secs(30)),
    };
    
    let criteria = DecisionCriteria {
        confidence_requirement: ConfidenceLevel::Statistical,
        time_budget: Some(Duration::from_secs(10)),
        accuracy_threshold: 0.95,
        cost_budget: Some(100.0),
        resource_constraints: hybrid_verifier::orchestrator::ResourceConstraints::default(),
    };
    
    c.bench_function("strategy_selection", |b| {
        b.iter(|| {
            black_box(orchestrator.recommend_strategy(
                black_box(&problem_spec),
                black_box(&criteria),
            ).unwrap())
        })
    });
}

fn benchmark_ensemble_execution(c: &mut Criterion) {
    let mut orchestrator = HybridOrchestrator::new();
    let mut surrogate_engine = SurrogateEngine::new();
    let mut verification_engine = VerificationEngine::new();
    let mut precompute_manager = PrecomputeManager::new();
    
    let problem_spec = ProblemSpec {
        id: ProblemId(6),
        description: "Ensemble benchmark".to_string(),
        parameters: HashMap::new(),
        complexity_class: ComplexityClass::P,
        verification_level: ConfidenceLevel::Statistical,
        timeout: Some(Duration::from_secs(5)),
    };
    
    c.bench_function("ensemble_execution", |b| {
        b.iter(|| {
            black_box(orchestrator.solve(
                black_box(&problem_spec),
                black_box(&mut surrogate_engine),
                black_box(&mut verification_engine),
                black_box(&mut precompute_manager),
            ).unwrap())
        })
    });
}

criterion_group!(
    benches,
    benchmark_hybrid_solve,
    benchmark_surrogate_training,
    benchmark_verification,
    benchmark_precompute_cache,
    benchmark_strategy_selection,
    benchmark_ensemble_execution
);
criterion_main!(benches);
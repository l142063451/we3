use criterion::{criterion_group, criterion_main, Criterion};
use knowledge_compilation::*;

fn benchmark_compilation_performance(c: &mut Criterion) {
    c.bench_function("model_counting", |b| {
        let mut cnf = CNFFormula::new(10);
        
        // Add some simple clauses
        for i in 0..5 {
            let mut clause = std::collections::BTreeSet::new();
            clause.insert(Literal::positive(i));
            cnf.add_clause(clause);
        }
        
        b.iter(|| {
            cnf.count_models_naive()
        });
    });
}

criterion_group!(benches, benchmark_compilation_performance);
criterion_main!(benches);
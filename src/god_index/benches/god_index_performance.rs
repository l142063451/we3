use criterion::{black_box, criterion_group, criterion_main, Criterion};
use god_index::*;

fn benchmark_family_registration(c: &mut Criterion) {
    c.bench_function("register_family", |b| {
        b.iter(|| {
            let mut gi = GodIndex::new();
            for i in 0..100 {
                let family = StructuredFamily::new(
                    format!("family_{}", i),
                    FamilyType::Combinatorial,
                    HierarchyStructure::Tree { depth: 3, branching_factor: 2 }
                );
                black_box(gi.register_family(family).unwrap());
            }
        });
    });
}

fn benchmark_mapping_evaluation(c: &mut Criterion) {
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
    
    c.bench_function("mapping_evaluation", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(mapping.evaluate_forward(i).unwrap());
            }
        });
    });
}

fn benchmark_family_factory(c: &mut Criterion) {
    c.bench_function("integer_partitions", |b| {
        b.iter(|| {
            black_box(FamilyFactory::integer_partitions(black_box(20)));
        });
    });
}

criterion_group!(benches, benchmark_family_registration, benchmark_mapping_evaluation, benchmark_family_factory);
criterion_main!(benches);
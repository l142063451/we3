use criterion::{criterion_group, criterion_main, Criterion};
use generating_functions::*;
use num::Complex;

fn benchmark_coefficient_extraction(c: &mut Criterion) {
    c.bench_function("coefficient_extraction", |b| {
        let mut gf = GeneratingFunction::new();
        for i in 0..1000 {
            gf.set_coefficient(i, Complex::new(1.0, 0.0));
        }
        
        b.iter(|| {
            gf.coefficient(500)
        });
    });
}

criterion_group!(benches, benchmark_coefficient_extraction);
criterion_main!(benches);
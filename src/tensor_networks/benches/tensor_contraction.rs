use criterion::{criterion_group, criterion_main, Criterion};
use tensor_networks::*;

fn benchmark_tensor_contraction(c: &mut Criterion) {
    c.bench_function("tensor_contraction", |b| {
        let tensor = Tensor::new(vec![10, 10, 10]);
        
        b.iter(|| {
            tensor.frobenius_norm()
        });
    });
}

criterion_group!(benches, benchmark_tensor_contraction);
criterion_main!(benches);
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use idv_bits::*;
use bitvec::prelude::*;
use num_complex::Complex;

fn benchmark_idvbit_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("IDVBit Creation");
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("from_bitvec", size), size, |b, &size| {
            let bits = bitvec![1; size];
            b.iter(|| {
                black_box(IDVBit::from_bitvec(black_box(bits.clone())))
            });
        });
        
        group.bench_with_input(BenchmarkId::new("from_generating_function", size), size, |b, &size| {
            let numerator = vec![Complex::new(1.0, 0.0); size / 10];
            let denominator = vec![Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)];
            b.iter(|| {
                black_box(IDVBit::from_generating_function(
                    black_box(numerator.clone()),
                    black_box(denominator.clone())
                ).unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_bit_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bit Access");
    
    let explicit_bits = bitvec![1, 0; 5000]; // 10000 alternating bits
    let explicit_idv = IDVBit::from_bitvec(explicit_bits);
    
    let numerator = vec![Complex::new(1.0, 0.0)];
    let denominator = vec![Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)];
    let gf_idv = IDVBit::from_generating_function(numerator, denominator).unwrap();
    
    let initial = bitvec![1, 0];
    let period = bitvec![1, 1, 0];
    let periodic_idv = IDVBit::from_periodic(initial, period).unwrap();
    
    for position in [100, 1000, 5000].iter() {
        group.bench_with_input(BenchmarkId::new("explicit", position), position, |b, &pos| {
            b.iter(|| {
                black_box(explicit_idv.get_bit(black_box(pos as u64)).unwrap())
            });
        });
        
        group.bench_with_input(BenchmarkId::new("generating_function", position), position, |b, &pos| {
            b.iter(|| {
                black_box(gf_idv.get_bit(black_box(pos as u64)).unwrap())
            });
        });
        
        group.bench_with_input(BenchmarkId::new("periodic", position), position, |b, &pos| {
            b.iter(|| {
                black_box(periodic_idv.get_bit(black_box(pos as u64)).unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_density_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Density Computation");
    
    for size in [1000, 5000, 10000].iter() {
        let bits = bitvec![1, 0; size / 2];
        let mut idv = IDVBit::from_bitvec(bits);
        
        group.bench_with_input(BenchmarkId::new("explicit", size), size, |b, _| {
            b.iter(|| {
                black_box(idv.compute_density().unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Conversion Operations");
    
    let mut converter = IDVBitConverter::new();
    
    for size in [100, 500, 1000].iter() {
        let bits = bitvec![1, 0, 1; size / 3];
        let idv = IDVBit::from_bitvec(bits);
        
        group.bench_with_input(BenchmarkId::new("to_explicit", size), size, |b, &length| {
            b.iter(|| {
                black_box(converter.to_explicit(&black_box(idv.clone()), length).unwrap())
            });
        });
        
        group.bench_with_input(BenchmarkId::new("to_generating_function", size), size, |b, _| {
            b.iter(|| {
                black_box(converter.to_generating_function(&black_box(idv.clone())).unwrap())
            });
        });
        
        group.bench_with_input(BenchmarkId::new("to_periodic", size), size, |b, _| {
            b.iter(|| {
                black_box(converter.to_periodic(&black_box(idv.clone())).unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_superposition_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Superposition Operations");
    
    for num_states in [2, 4, 8].iter() {
        let states: Vec<IDVBit> = (0..*num_states)
            .map(|i| IDVBit::from_bitvec(bitvec![i % 2 == 0; 100]))
            .collect();
        let superpos = SuperpositionState::uniform_superposition(states.clone()).unwrap();
        
        group.bench_with_input(BenchmarkId::new("creation", num_states), num_states, |b, _| {
            b.iter(|| {
                black_box(SuperpositionState::uniform_superposition(black_box(states.clone())).unwrap())
            });
        });
        
        group.bench_with_input(BenchmarkId::new("bit_measurement", num_states), num_states, |b, _| {
            let measurement = MeasurementOperator::BitMeasurement { position: 50 };
            b.iter(|| {
                black_box(superpos.measure(&black_box(measurement)).unwrap())
            });
        });
        
        group.bench_with_input(BenchmarkId::new("density_measurement", num_states), num_states, |b, _| {
            let measurement = MeasurementOperator::DensityMeasurement { start: 0, end: 100 };
            b.iter(|| {
                black_box(superpos.measure(&black_box(measurement)).unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_bitwise_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bitwise Operations");
    
    let ops = IDVBitOperations::new();
    
    for size in [1000, 5000, 10000].iter() {
        let left = IDVBit::from_bitvec(bitvec![1, 0; size / 2]);
        let right = IDVBit::from_bitvec(bitvec![0, 1; size / 2]);
        
        group.bench_with_input(BenchmarkId::new("and", size), size, |b, _| {
            b.iter(|| {
                black_box(ops.bitwise_operation(
                    &black_box(left.clone()),
                    &black_box(right.clone()),
                    BitwiseOperation::And
                ).unwrap())
            });
        });
        
        group.bench_with_input(BenchmarkId::new("xor", size), size, |b, _| {
            b.iter(|| {
                black_box(ops.bitwise_operation(
                    &black_box(left.clone()),
                    &black_box(right.clone()),
                    BitwiseOperation::Xor
                ).unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_fft_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Operations");
    
    let mut ops = IDVBitOperations::new();
    
    for size in [256, 512, 1024].iter() {
        let bits = bitvec![1, 0, 1, 0; size / 4];
        let idv = IDVBit::from_bitvec(bits);
        
        group.bench_with_input(BenchmarkId::new("fft_transform", size), size, |b, _| {
            b.iter(|| {
                black_box(ops.transform(&black_box(idv.clone()), TransformOperation::FFT).unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_convolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("Convolution");
    
    let mut ops = IDVBitOperations::new();
    
    for size in [100, 200, 500].iter() {
        let left = IDVBit::from_bitvec(bitvec![1, 0, 1; size / 3]);
        let right = IDVBit::from_bitvec(bitvec![1, 1, 0; size / 3]);
        
        group.bench_with_input(BenchmarkId::new("convolve", size), size, |b, _| {
            b.iter(|| {
                black_box(ops.convolve(&black_box(left.clone()), &black_box(right.clone())).unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_quantum_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Quantum Operations");
    
    let mut processor = QuantumIDVProcessor::new();
    
    for size in [100, 500, 1000].iter() {
        let idv = IDVBit::from_bitvec(bitvec![1, 0; size / 2]);
        
        group.bench_with_input(BenchmarkId::new("pauli_x", size), size, |b, _| {
            b.iter(|| {
                black_box(processor.apply_gate(
                    &black_box(idv.clone()),
                    QuantumGate::PauliX,
                    &[0, 1, 2]
                ).unwrap())
            });
        });
        
        group.bench_with_input(BenchmarkId::new("hadamard", size), size, |b, _| {
            b.iter(|| {
                black_box(processor.apply_gate(
                    &black_box(idv.clone()),
                    QuantumGate::Hadamard,
                    &[0, 1]
                ).unwrap())
            });
        });
        
        group.bench_with_input(BenchmarkId::new("quantum_walk", size), size, |b, _| {
            b.iter(|| {
                black_box(processor.quantum_walk(
                    &black_box(idv.clone()),
                    QuantumWalkType::DiscreteTime,
                    10,
                    size / 10
                ).unwrap())
            });
        });
    }
    
    group.finish();
}

fn benchmark_density_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Density Functions");
    
    let gaussian = GaussianDensity::standard();
    let exponential = ExponentialDensity::new(1.0).unwrap();
    let power_law = PowerLawDensity::new(2.5, 1.0).unwrap();
    
    group.bench_function("gaussian_evaluate", |b| {
        b.iter(|| {
            for x in 0..1000 {
                black_box(gaussian.evaluate(black_box(x as f64 / 1000.0)));
            }
        });
    });
    
    group.bench_function("exponential_evaluate", |b| {
        b.iter(|| {
            for x in 0..1000 {
                black_box(exponential.evaluate(black_box(x as f64 / 1000.0)));
            }
        });
    });
    
    group.bench_function("power_law_evaluate", |b| {
        b.iter(|| {
            for x in 1..1000 {
                black_box(power_law.evaluate(black_box(x as f64 / 1000.0 + 1.0)));
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_idvbit_creation,
    benchmark_bit_access,
    benchmark_density_computation,
    benchmark_conversion,
    benchmark_superposition_operations,
    benchmark_bitwise_operations,
    benchmark_fft_operations,
    benchmark_convolution,
    benchmark_quantum_operations,
    benchmark_density_functions
);

criterion_main!(benches);
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vgpu_shim::{VGpuFramework, VGpuConfig};
use tokio::runtime::Runtime;

fn benchmark_framework_initialization(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("vgpu_framework_init", |b| {
        b.to_async(&rt).iter(|| async {
            let framework = black_box(VGpuFramework::new().await.unwrap());
            framework
        })
    });
}

fn benchmark_task_submission(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("task_submission", |b| {
        b.to_async(&rt).iter(|| async {
            let mut framework = VGpuFramework::new().await.unwrap();
            framework.start().await.unwrap();
            
            let task = vgpu_shim::GpuTask::new("benchmark_kernel", vec![1.0; 1000]);
            black_box(framework.submit_task(task).await.unwrap());
            
            framework.stop().await.unwrap();
        })
    });
}

fn benchmark_memory_allocation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("memory_allocation", |b| {
        b.to_async(&rt).iter(|| async {
            let framework = VGpuFramework::new().await.unwrap();
            let vgpu = framework.get_virtual_gpu();
            
            let address = black_box(
                vgpu.read().allocate_memory(1024 * 1024, vgpu_shim::MemoryType::Global).unwrap()
            );
            
            vgpu.read().free_memory(address).unwrap();
        })
    });
}

criterion_group!(benches, benchmark_framework_initialization, benchmark_task_submission, benchmark_memory_allocation);
criterion_main!(benches);
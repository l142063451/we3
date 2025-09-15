//! vGPU v1.4+ Core Infrastructure - Real Implementation and Testing

use vgpu_infinite_computing::{InfiniteComputingOrchestrator, initialize_infinite_computing_framework};
use vgpu_universal_solver::{UniversalProblemSolver, initialize_universal_solver_framework};

pub mod rendering;
pub mod validation;

use validation::VGPUValidator;

pub struct VGPUv14Core {
    pub infinite_computing: InfiniteComputingOrchestrator,
    pub universal_solver: UniversalProblemSolver,
}

impl VGPUv14Core {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let infinite_computing = initialize_infinite_computing_framework().await?;
        let universal_solver = initialize_universal_solver_framework().await?;
        
        Ok(Self {
            infinite_computing,
            universal_solver,
        })
    }

    /// Run comprehensive validation and performance testing
    pub async fn run_validation(&self) -> Result<String, Box<dyn std::error::Error>> {
        let mut validator = VGPUValidator::new();
        
        println!("Starting comprehensive vGPU v1.4 validation...");
        validator.run_full_validation().await;
        
        let report = validator.generate_report();
        
        // Save report to file
        use std::fs;
        fs::write("/tmp/vgpu_v14_validation_report.md", &report)?;
        println!("Validation report saved to /tmp/vgpu_v14_validation_report.md");
        
        Ok(report)
    }

    /// Quick performance benchmark
    pub async fn benchmark_performance(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Running quick performance benchmark...");
        
        // Test rendering performance
        let benchmark = rendering::run_rendering_benchmark(128, 128, 50);
        println!("Rendering benchmark: {:.1} FPS average", benchmark.average_fps);
        
        Ok(())
    }
}
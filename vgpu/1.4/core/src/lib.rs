//! vGPU v1.4+ Core Infrastructure

use vgpu_infinite_computing::{InfiniteComputingOrchestrator, initialize_infinite_computing_framework};
use vgpu_universal_solver::{UniversalProblemSolver, initialize_universal_solver_framework};

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
}
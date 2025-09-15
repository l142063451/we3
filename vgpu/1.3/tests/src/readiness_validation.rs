use vgpu_core::{VirtualGPUCore, readiness_assessment::{ReadinessAssessment, AssessmentConfig}};
use anyhow::Result;

pub struct ReadinessResult {
    pub gpu_score: f64,
    pub quantum_score: f64,
    pub infinite_flops: bool,
}

pub async fn assess_gpu_replacement_readiness(vgpu: &VirtualGPUCore) -> Result<ReadinessResult> {
    let config = AssessmentConfig::default();
    let mut assessment = ReadinessAssessment::new(config);
    
    let result = assessment.execute_assessment(vgpu).await?;
    
    Ok(ReadinessResult {
        gpu_score: result.readiness_scores.gpu_replacement_score,
        quantum_score: result.readiness_scores.quantum_replacement_score,
        infinite_flops: result.readiness_scores.infinite_score >= 90.0,
    })
}
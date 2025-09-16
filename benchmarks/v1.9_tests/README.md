# vGPU v1.9 Production Testing Framework

This directory contains comprehensive tests for vGPU v1.9 production system.

## Test Categories

1. **Hardware Detection**: Real GPU/accelerator detection and benchmarking
2. **API Interception**: CUDA/OpenCL/Vulkan/Metal/WebGPU API testing
3. **Real Workloads**: AI training, 3D rendering, cryptography, HPC, quantum simulation
4. **Installation**: Cross-platform installer verification
5. **Performance**: Rigorous benchmark testing with full artifacts

## Artifact Requirements

Each test generates complete artifact bundle:
- commit.txt (git SHA)
- hardware.json (detected hardware specs)
- performance.json (measured results)
- verification.log (test execution log)
- container.digest.txt (reproducible environment)
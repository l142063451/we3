#!/usr/bin/env python3
"""
vGPU v1.9 Comprehensive Test Runner - Generates full artifact bundles
"""

import os
import json
import time
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from src.testing.hardware_detector import HardwareDetector
from src.testing.real_workload_tester import RealWorkloadTester


class ComprehensiveTestRunner:
    """Runs comprehensive tests with full artifact generation"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("benchmarks/v1.9_tests")
        self.test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_artifacts = {}
        
    def _get_git_commit(self) -> str:
        """Get current git commit SHA"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return "unknown"
    
    def _get_container_digest(self) -> str:
        """Get container digest if running in container"""
        try:
            # Check if running in Docker
            if Path("/.dockerenv").exists():
                return "docker_container_detected"
            # Check for other container indicators
            elif os.environ.get('CONTAINER'):
                return f"container_{os.environ['CONTAINER']}"
        except:
            pass
        return "bare_metal"
    
    def generate_artifacts(self, test_name: str, test_results: Dict[str, Any]) -> Path:
        """Generate complete artifact bundle for a test"""
        
        # Create test-specific directory
        test_dir = self.output_dir / test_name / self.test_timestamp
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate required artifacts
        artifacts = {}
        
        # 1. commit.txt
        commit_sha = self._get_git_commit()
        commit_file = test_dir / "commit.txt"
        commit_file.write_text(commit_sha)
        artifacts['commit'] = commit_sha
        
        # 2. hardware.json
        hardware_detector = HardwareDetector()
        hardware_info = hardware_detector.generate_hardware_report()
        hardware_file = test_dir / "hardware.json"
        with open(hardware_file, 'w') as f:
            json.dump(hardware_info, f, indent=2)
        artifacts['hardware'] = hardware_file
        
        # 3. container.digest.txt
        container_digest = self._get_container_digest()
        container_file = test_dir / "container.digest.txt"
        container_file.write_text(container_digest)
        artifacts['container'] = container_digest
        
        # 4. performance.json
        performance_data = {
            'test_name': test_name,
            'timestamp': time.time(),
            'test_results': test_results,
            'system_info': {
                'platform': platform.system(),
                'python_version': platform.python_version(),
                'architecture': platform.architecture()[0]
            }
        }
        performance_file = test_dir / "performance.json"
        with open(performance_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
        artifacts['performance'] = performance_file
        
        # 5. verification.log
        log_content = f"""vGPU v1.9 Test Verification Log
===============================

Test Name: {test_name}
Timestamp: {datetime.now().isoformat()}
Git Commit: {commit_sha}
Container: {container_digest}

Test Results Summary:
"""
        if 'speedup_factor' in test_results:
            log_content += f"Speedup Factor: {test_results['speedup_factor']:.2f}x\n"
        if 'execution_time' in test_results:
            log_content += f"Execution Time: {test_results['execution_time']:.4f}s\n"
        if 'status' in test_results:
            log_content += f"Test Status: {test_results['status']}\n"
        
        log_content += f"\nFull test data available in performance.json\n"
        
        log_file = test_dir / "verification.log"
        log_file.write_text(log_content)
        artifacts['verification_log'] = log_file
        
        print(f"✅ Generated artifacts for {test_name} in {test_dir}")
        
        self.test_artifacts[test_name] = {
            'directory': test_dir,
            'artifacts': artifacts,
            'timestamp': self.test_timestamp
        }
        
        return test_dir
    
    def run_ai_training_test(self) -> Dict[str, Any]:
        """Run AI training test with artifacts"""
        print("Running AI Training Test...")
        
        tester = RealWorkloadTester()
        test_result = tester.test_ai_training_workload("medium")
        
        # Generate artifacts
        artifact_dir = self.generate_artifacts("ai_training", test_result)
        
        print(f"AI Training Test: {test_result['speedup_factor']:.1f}x speedup")
        return {
            'test_result': test_result,
            'artifact_directory': artifact_dir
        }
    
    def run_3d_rendering_test(self) -> Dict[str, Any]:
        """Run 3D rendering test with artifacts"""
        print("Running 3D Rendering Test...")
        
        tester = RealWorkloadTester()
        test_result = tester.test_3d_rendering_workload("medium")
        
        # Generate artifacts
        artifact_dir = self.generate_artifacts("3d_rendering", test_result)
        
        print(f"3D Rendering Test: {test_result['fps_speedup']:.1f}x FPS improvement")
        return {
            'test_result': test_result,
            'artifact_directory': artifact_dir
        }
    
    def run_cryptography_test(self) -> Dict[str, Any]:
        """Run cryptography test with artifacts"""
        print("Running Cryptography Test...")
        
        tester = RealWorkloadTester()
        test_result = tester.test_cryptography_workload([512, 1024])
        
        # Generate artifacts
        artifact_dir = self.generate_artifacts("cryptography", test_result)
        
        print(f"Cryptography Test: {test_result['average_speedup']:.1f}x speedup")
        return {
            'test_result': test_result,
            'artifact_directory': artifact_dir
        }
    
    def run_scientific_computing_test(self) -> Dict[str, Any]:
        """Run scientific computing test with artifacts"""
        print("Running Scientific Computing Test...")
        
        tester = RealWorkloadTester()
        test_result = tester.test_scientific_computing_workload()
        
        # Generate artifacts
        artifact_dir = self.generate_artifacts("scientific_computing", test_result)
        
        print(f"Scientific Computing Test: {test_result['gflops']:.1f} GFLOPS")
        return {
            'test_result': test_result,
            'artifact_directory': artifact_dir
        }
    
    def run_quantum_simulation_test(self) -> Dict[str, Any]:
        """Run quantum simulation test with artifacts"""
        print("Running Quantum Simulation Test...")
        
        tester = RealWorkloadTester()
        test_result = tester.test_quantum_simulation_workload(16)
        
        # Generate artifacts
        artifact_dir = self.generate_artifacts("quantum_simulation", test_result)
        
        print(f"Quantum Simulation Test: {test_result['num_qubits']} qubits simulated")
        return {
            'test_result': test_result,
            'artifact_directory': artifact_dir
        }
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite with all artifacts"""
        print("="*50)
        print("vGPU v1.9 Comprehensive Test Suite")
        print("="*50)
        print()
        
        suite_start = time.time()
        test_results = {}
        
        # Run all tests
        test_results['ai_training'] = self.run_ai_training_test()
        test_results['3d_rendering'] = self.run_3d_rendering_test()
        test_results['cryptography'] = self.run_cryptography_test()
        test_results['scientific_computing'] = self.run_scientific_computing_test()
        test_results['quantum_simulation'] = self.run_quantum_simulation_test()
        
        suite_time = time.time() - suite_start
        
        # Generate suite summary
        summary = {
            'test_suite_version': '1.9',
            'total_execution_time': suite_time,
            'tests_completed': len(test_results),
            'all_artifacts': self.test_artifacts,
            'performance_summary': {
                'ai_speedup': test_results['ai_training']['test_result']['speedup_factor'],
                'rendering_fps_improvement': test_results['3d_rendering']['test_result']['fps_speedup'],
                'crypto_speedup': test_results['cryptography']['test_result']['average_speedup'],
                'scientific_gflops': test_results['scientific_computing']['test_result']['gflops'],
                'quantum_qubits': test_results['quantum_simulation']['test_result']['num_qubits']
            }
        }
        
        # Save suite summary
        summary_file = self.output_dir / f"test_suite_summary_{self.test_timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print()
        print("="*50)
        print("Test Suite Completed!")
        print("="*50)
        print(f"Total Time: {suite_time:.2f} seconds")
        print(f"Tests Run: {len(test_results)}")
        print(f"Artifacts Generated: {len(self.test_artifacts)} test directories")
        print(f"Summary: {summary_file}")
        print()
        
        return {
            'summary': summary,
            'test_results': test_results,
            'artifacts': self.test_artifacts
        }


def main():
    """Main entry point for test runner"""
    import sys
    
    print("vGPU v1.9 Comprehensive Test Runner")
    print("===================================")
    
    # Create test runner
    runner = ComprehensiveTestRunner()
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        
        if test_type == 'ai':
            runner.run_ai_training_test()
        elif test_type == 'render':
            runner.run_3d_rendering_test()
        elif test_type == 'crypto':
            runner.run_cryptography_test()
        elif test_type == 'scientific':
            runner.run_scientific_computing_test()
        elif test_type == 'quantum':
            runner.run_quantum_simulation_test()
        elif test_type == 'all':
            runner.run_full_test_suite()
        else:
            print(f"Unknown test type: {test_type}")
            print("Available: ai, render, crypto, scientific, quantum, all")
            return 1
    else:
        # Run full suite by default
        runner.run_full_test_suite()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
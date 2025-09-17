#!/usr/bin/env python3
"""
Hardware Detection v1.9 - Detects real GPU/accelerator hardware and capabilities
"""

import os
import platform
import subprocess
import json
import time
from typing import Dict, List, Any, Optional


class HardwareDetector:
    """Detects and benchmarks real hardware capabilities"""
    
    def __init__(self):
        self.detection_results = {}
        self.benchmark_results = {}
        
    def detect_system_info(self) -> Dict[str, Any]:
        """Detect basic system information"""
        try:
            system_info = {
                'platform': platform.system(),
                'processor': platform.processor(),
                'architecture': platform.architecture()[0],
                'machine': platform.machine(),
                'python_version': platform.python_version(),
                'hostname': platform.node()
            }
            
            # Get CPU information
            if platform.system() == 'Linux':
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        cpuinfo = f.read()
                        # Extract CPU model
                        for line in cpuinfo.split('\n'):
                            if 'model name' in line:
                                system_info['cpu_model'] = line.split(':')[1].strip()
                                break
                except:
                    system_info['cpu_model'] = 'Unknown'
            
            # Get memory information
            try:
                import psutil
                memory = psutil.virtual_memory()
                system_info['memory_total_gb'] = round(memory.total / (1024**3), 1)
                system_info['memory_available_gb'] = round(memory.available / (1024**3), 1)
            except ImportError:
                system_info['memory_info'] = 'psutil not available'
                
            self.detection_results['system'] = system_info
            return system_info
            
        except Exception as e:
            return {'error': str(e)}
    
    def detect_nvidia_gpus(self) -> List[Dict[str, Any]]:
        """Detect NVIDIA GPUs using nvidia-ml-py or nvidia-smi"""
        gpus = []
        
        # Try nvidia-ml-py first
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_info = {
                    'index': i,
                    'name': name,
                    'memory_total_mb': memory_info.total // (1024*1024),
                    'memory_free_mb': memory_info.free // (1024*1024),
                    'driver_version': pynvml.nvmlSystemGetDriverVersion().decode('utf-8'),
                    'detection_method': 'nvidia-ml-py'
                }
                gpus.append(gpu_info)
                
        except ImportError:
            # Fall back to nvidia-smi
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for i, line in enumerate(lines):
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            gpu_info = {
                                'index': i,
                                'name': parts[0],
                                'memory_total_mb': int(parts[1]),
                                'memory_free_mb': int(parts[2]),
                                'driver_version': parts[3],
                                'detection_method': 'nvidia-smi'
                            }
                            gpus.append(gpu_info)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                pass
        except Exception as e:
            print(f"NVIDIA GPU detection error: {e}")
            
        self.detection_results['nvidia_gpus'] = gpus
        return gpus
    
    def detect_opencl_devices(self) -> List[Dict[str, Any]]:
        """Detect OpenCL capable devices"""
        devices = []
        
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            
            for platform in platforms:
                platform_devices = platform.get_devices()
                for device in platform_devices:
                    device_info = {
                        'name': device.name.strip(),
                        'vendor': device.vendor.strip(),
                        'platform': platform.name.strip(),
                        'type': cl.device_type.to_string(device.type),
                        'max_compute_units': device.max_compute_units,
                        'max_work_group_size': device.max_work_group_size,
                        'global_mem_size_mb': device.global_mem_size // (1024*1024),
                        'detection_method': 'pyopencl'
                    }
                    devices.append(device_info)
                    
        except ImportError:
            devices.append({
                'error': 'PyOpenCL not available',
                'suggestion': 'Install pyopencl for OpenCL device detection'
            })
        except Exception as e:
            devices.append({'error': str(e)})
            
        self.detection_results['opencl_devices'] = devices
        return devices
    
    def benchmark_cpu_performance(self) -> Dict[str, Any]:
        """Benchmark CPU performance with mathematical operations"""
        try:
            import numpy as np
            
            # Matrix multiplication benchmark
            start_time = time.time()
            matrix_size = 1000
            a = np.random.random((matrix_size, matrix_size))
            b = np.random.random((matrix_size, matrix_size))
            c = np.dot(a, b)
            matrix_time = time.time() - start_time
            
            # Calculate GFLOPS (2*n^3 operations for matrix multiplication)
            operations = 2 * (matrix_size ** 3)
            gflops = operations / (matrix_time * 1e9)
            
            # FFT benchmark
            start_time = time.time()
            signal_size = 2**20  # 1M samples
            signal = np.random.random(signal_size)
            fft_result = np.fft.fft(signal)
            fft_time = time.time() - start_time
            
            # Prime factorization benchmark (small numbers)
            start_time = time.time()
            test_numbers = list(range(1000, 2000))
            factors = []
            for num in test_numbers:
                f = []
                d = 2
                while d * d <= num:
                    while num % d == 0:
                        f.append(d)
                        num //= d
                    d += 1
                if num > 1:
                    f.append(num)
                factors.append(f)
            factorization_time = time.time() - start_time
            
            benchmark_results = {
                'matrix_multiplication': {
                    'size': f"{matrix_size}x{matrix_size}",
                    'time_seconds': matrix_time,
                    'gflops': gflops
                },
                'fft': {
                    'signal_size': signal_size,
                    'time_seconds': fft_time,
                    'samples_per_second': signal_size / fft_time
                },
                'prime_factorization': {
                    'numbers_tested': len(test_numbers),
                    'time_seconds': factorization_time,
                    'numbers_per_second': len(test_numbers) / factorization_time
                },
                'timestamp': time.time()
            }
            
            self.benchmark_results['cpu'] = benchmark_results
            return benchmark_results
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_hardware_report(self) -> Dict[str, Any]:
        """Generate complete hardware detection and benchmark report"""
        
        # Detect all hardware
        system_info = self.detect_system_info()
        nvidia_gpus = self.detect_nvidia_gpus()
        opencl_devices = self.detect_opencl_devices()
        cpu_benchmarks = self.benchmark_cpu_performance()
        
        report = {
            'detection_timestamp': time.time(),
            'system_info': system_info,
            'nvidia_gpus': nvidia_gpus,
            'opencl_devices': opencl_devices,
            'cpu_benchmarks': cpu_benchmarks,
            'detection_summary': {
                'total_nvidia_gpus': len(nvidia_gpus),
                'total_opencl_devices': len(opencl_devices),
                'has_gpu_acceleration': len(nvidia_gpus) > 0 or len(opencl_devices) > 0
            }
        }
        
        return report
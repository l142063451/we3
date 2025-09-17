#!/usr/bin/env python3
"""
vGPU v1.9 Components Package
Organized components for modular vGPU system architecture
"""

# Import all major components for easy access
from .hardware_detector import HardwareDetector
from .api_interceptor import APIInterceptor, PerformanceMonitor
from .test_framework import ComprehensiveTestSuite, TestResult
from .deployment_manager import DeploymentManager
from .report_generator import ReportGenerator

__all__ = [
    'HardwareDetector',
    'APIInterceptor', 
    'PerformanceMonitor',
    'ComprehensiveTestSuite',
    'TestResult',
    'DeploymentManager',
    'ReportGenerator'
]

# Component metadata
COMPONENTS_INFO = {
    'hardware_detector': {
        'description': 'Hardware detection and profiling with real performance measurements',
        'class': 'HardwareDetector',
        'file': 'hardware_detector.py'
    },
    'api_interceptor': {
        'description': 'GPU API interception and redirection system',
        'class': 'APIInterceptor',
        'file': 'api_interceptor.py'
    },
    'test_framework': {
        'description': 'Comprehensive testing system for all GPU workloads',
        'class': 'ComprehensiveTestSuite', 
        'file': 'test_framework.py'
    },
    'deployment_manager': {
        'description': 'System for generating deployment packages and installation artifacts',
        'class': 'DeploymentManager',
        'file': 'deployment_manager.py'
    },
    'report_generator': {
        'description': 'Comprehensive reporting and analysis system',
        'class': 'ReportGenerator',
        'file': 'report_generator.py'
    }
}

def get_component_info():
    """Get information about available components"""
    return COMPONENTS_INFO

def initialize_all_components(artifacts_dir=None):
    """Initialize all components with shared configuration"""
    components = {
        'hardware_detector': HardwareDetector(artifacts_dir=artifacts_dir),
        'api_interceptor': APIInterceptor(),
        'test_framework': ComprehensiveTestSuite(artifacts_dir=artifacts_dir),
        'deployment_manager': DeploymentManager(artifacts_dir=artifacts_dir),
        'report_generator': ReportGenerator(artifacts_dir=artifacts_dir)
    }
    
    return components
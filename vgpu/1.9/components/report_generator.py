#!/usr/bin/env python3
"""
vGPU v1.9 Reporting Components
Comprehensive reporting and analysis system for performance metrics and results
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive reports for vGPU performance and testing results"""
    
    def __init__(self, artifacts_dir=None):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else Path.cwd() / "reports"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_comprehensive_report(self, 
                                    hardware_info: Dict[str, Any] = None,
                                    api_results: Dict[str, Any] = None,
                                    test_results: Dict[str, Any] = None,
                                    deployment_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        logger.info("Generating comprehensive vGPU system report...")
        
        report = {
            "report_metadata": {
                "generation_time": datetime.now().isoformat(),
                "vgpu_version": "1.9",
                "report_type": "comprehensive_analysis"
            },
            "executive_summary": self._generate_executive_summary(
                hardware_info, api_results, test_results, deployment_info
            ),
            "hardware_analysis": self._analyze_hardware(hardware_info) if hardware_info else {},
            "api_performance": self._analyze_api_performance(api_results) if api_results else {},
            "test_analysis": self._analyze_test_results(test_results) if test_results else {},
            "deployment_status": self._analyze_deployment(deployment_info) if deployment_info else {},
            "recommendations": self._generate_recommendations(hardware_info, test_results),
            "performance_metrics": self._calculate_performance_metrics(test_results) if test_results else {}
        }
        
        # Generate different report formats
        self._generate_json_report(report)
        self._generate_markdown_report(report)
        self._generate_summary_report(report)
        
        logger.info(f"Comprehensive report generated in {self.artifacts_dir}")
        return report
    
    def _generate_executive_summary(self, 
                                  hardware_info: Dict[str, Any],
                                  api_results: Dict[str, Any],
                                  test_results: Dict[str, Any],
                                  deployment_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of system status"""
        summary = {
            "overall_status": "unknown",
            "key_metrics": {},
            "critical_issues": [],
            "success_highlights": []
        }
        
        # Analyze test results for overall status
        if test_results and "summary" in test_results:
            test_summary = test_results["summary"]
            total_tests = test_summary.get("total_tests", 0)
            passed_tests = test_summary.get("passed_tests", 0)
            
            if total_tests > 0:
                success_rate = (passed_tests / total_tests) * 100
                summary["key_metrics"]["test_success_rate"] = f"{success_rate:.1f}%"
                
                if success_rate >= 90:
                    summary["overall_status"] = "excellent"
                    summary["success_highlights"].append(f"Exceptional test performance: {success_rate:.1f}% success rate")
                elif success_rate >= 75:
                    summary["overall_status"] = "good"
                    summary["success_highlights"].append(f"Good test performance: {success_rate:.1f}% success rate")
                elif success_rate >= 50:
                    summary["overall_status"] = "acceptable"
                    summary["critical_issues"].append(f"Moderate test failures: {success_rate:.1f}% success rate")
                else:
                    summary["overall_status"] = "poor"
                    summary["critical_issues"].append(f"High test failure rate: {success_rate:.1f}% success rate")
        
        # Analyze hardware detection
        if hardware_info:
            gpu_count = len(hardware_info.get("gpus", []))
            summary["key_metrics"]["gpus_detected"] = gpu_count
            
            if gpu_count > 0:
                summary["success_highlights"].append(f"Detected {gpu_count} GPU(s)")
            else:
                summary["critical_issues"].append("No GPUs detected")
        
        # Analyze API interception
        if api_results:
            intercepted_count = sum(1 for status in api_results.values() if status)
            total_apis = len(api_results)
            summary["key_metrics"]["apis_intercepted"] = f"{intercepted_count}/{total_apis}"
            
            if intercepted_count >= total_apis * 0.8:
                summary["success_highlights"].append(f"Strong API coverage: {intercepted_count}/{total_apis} APIs")
            elif intercepted_count < total_apis * 0.5:
                summary["critical_issues"].append(f"Limited API coverage: {intercepted_count}/{total_apis} APIs")
        
        return summary
    
    def _analyze_hardware(self, hardware_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze hardware detection and performance results"""
        analysis = {
            "detection_summary": {},
            "performance_analysis": {},
            "compatibility_assessment": {}
        }
        
        # Hardware detection summary
        analysis["detection_summary"] = {
            "gpus_found": len(hardware_info.get("gpus", [])),
            "accelerators_found": len(hardware_info.get("accelerators", [])),
            "quantum_simulators_found": len(hardware_info.get("quantum_simulators", [])),
            "platform": hardware_info.get("platform", {}).get("system", "unknown")
        }
        
        # CPU performance analysis
        cpu_info = hardware_info.get("cpu", {})
        if cpu_info and "estimated_gflops" in cpu_info:
            cpu_gflops = cpu_info["estimated_gflops"]
            analysis["performance_analysis"]["cpu_performance"] = {
                "gflops": cpu_gflops,
                "rating": self._rate_performance(cpu_gflops, "cpu_gflops"),
                "cores": cpu_info.get("physical_cores", "unknown")
            }
        
        # Memory performance analysis
        memory_info = hardware_info.get("memory", {})
        if memory_info and "estimated_bandwidth_gbps" in memory_info:
            memory_bandwidth = memory_info["estimated_bandwidth_gbps"]
            analysis["performance_analysis"]["memory_performance"] = {
                "bandwidth_gbps": memory_bandwidth,
                "rating": self._rate_performance(memory_bandwidth, "memory_bandwidth"),
                "total_gb": round(memory_info.get("total_bytes", 0) / (1024**3), 2)
            }
        
        # GPU compatibility assessment
        gpus = hardware_info.get("gpus", [])
        compatibility_score = 0
        gpu_details = []
        
        for gpu in gpus:
            gpu_rating = self._assess_gpu_compatibility(gpu)
            gpu_details.append({
                "name": gpu.get("name", "Unknown"),
                "type": gpu.get("type", "Unknown"),
                "compatibility_score": gpu_rating
            })
            compatibility_score += gpu_rating
        
        if gpus:
            compatibility_score /= len(gpus)
        
        analysis["compatibility_assessment"] = {
            "overall_score": compatibility_score,
            "rating": self._rate_performance(compatibility_score, "compatibility"),
            "gpu_details": gpu_details
        }
        
        return analysis
    
    def _analyze_api_performance(self, api_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze API interception performance"""
        analysis = {
            "interception_summary": {},
            "api_coverage": {},
            "performance_impact": {}
        }
        
        # Interception summary
        total_apis = len(api_results)
        successful_apis = sum(1 for status in api_results.values() if status)
        
        analysis["interception_summary"] = {
            "total_apis": total_apis,
            "successful_interceptions": successful_apis,
            "success_rate": (successful_apis / total_apis * 100) if total_apis > 0 else 0,
            "failed_apis": [api for api, status in api_results.items() if not status]
        }
        
        # API coverage assessment
        critical_apis = ["CUDA", "OpenGL", "Vulkan"]
        critical_coverage = sum(1 for api in critical_apis if api_results.get(api, False))
        
        analysis["api_coverage"] = {
            "critical_apis_covered": f"{critical_coverage}/{len(critical_apis)}",
            "critical_coverage_rate": (critical_coverage / len(critical_apis) * 100),
            "all_apis_status": api_results
        }
        
        return analysis
    
    def _analyze_test_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze comprehensive test results"""
        analysis = {
            "test_summary": {},
            "performance_breakdown": {},
            "category_analysis": {},
            "performance_trends": {}
        }
        
        # Overall test summary
        summary = test_results.get("summary", {})
        analysis["test_summary"] = {
            "total_tests": summary.get("total_tests", 0),
            "passed_tests": summary.get("passed_tests", 0),
            "failed_tests": summary.get("failed_tests", 0),
            "success_rate": (summary.get("passed_tests", 0) / max(1, summary.get("total_tests", 1))) * 100
        }
        
        # Category-wise analysis
        categories = test_results.get("categories", {})
        for category_name, category_data in categories.items():
            analysis["category_analysis"][category_name] = self._analyze_test_category(category_data)
        
        # Performance metrics extraction
        performance_data = []
        for category_name, category_data in categories.items():
            if "performance_metrics" in category_data:
                for test_name, metrics in category_data["performance_metrics"].items():
                    performance_data.append({
                        "category": category_name,
                        "test": test_name,
                        "metrics": metrics
                    })
        
        analysis["performance_breakdown"] = performance_data
        
        return analysis
    
    def _analyze_test_category(self, category_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual test category results"""
        analysis = {
            "status": category_data.get("status", "unknown"),
            "test_count": len(category_data.get("tests_run", [])),
            "success_rate": 0,
            "avg_execution_time": 0,
            "performance_highlights": []
        }
        
        tests_run = category_data.get("tests_run", [])
        if tests_run:
            passed_tests = [t for t in tests_run if t.get("status") == "PASSED"]
            analysis["success_rate"] = (len(passed_tests) / len(tests_run)) * 100
            
            # Calculate average execution time
            execution_times = [t.get("execution_time", 0) for t in tests_run if "execution_time" in t]
            if execution_times:
                analysis["avg_execution_time"] = statistics.mean(execution_times)
        
        # Extract performance highlights
        performance_metrics = category_data.get("performance_metrics", {})
        for test_name, metrics in performance_metrics.items():
            if "tflops" in metrics:
                analysis["performance_highlights"].append(f"{test_name}: {metrics['tflops']:.2f} TFLOPS")
            elif "pixels_per_second" in metrics:
                analysis["performance_highlights"].append(f"{test_name}: {metrics['pixels_per_second']:.0f} pixels/sec")
        
        return analysis
    
    def _analyze_deployment(self, deployment_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze deployment status and artifacts"""
        analysis = {
            "deployment_summary": {},
            "artifact_status": {},
            "package_info": {}
        }
        
        # Deployment summary
        artifacts = deployment_info.get("artifacts", {})
        total_artifacts = len(artifacts)
        successful_artifacts = sum(1 for a in artifacts.values() if a.get("generated", False))
        
        analysis["deployment_summary"] = {
            "total_artifacts": total_artifacts,
            "successful_generations": successful_artifacts,
            "generation_success_rate": (successful_artifacts / total_artifacts * 100) if total_artifacts > 0 else 0
        }
        
        # Artifact status details
        analysis["artifact_status"] = {
            name: {
                "generated": info.get("generated", False),
                "size_mb": round(info.get("size_bytes", 0) / (1024*1024), 2) if info.get("size_bytes") else 0,
                "path": info.get("path", "")
            }
            for name, info in artifacts.items()
        }
        
        # Package information
        if "package" in deployment_info:
            package = deployment_info["package"]
            analysis["package_info"] = {
                "package_name": package.get("package_name", ""),
                "contents_count": len(package.get("contents", [])),
                "package_path": package.get("package_path", "")
            }
        
        return analysis
    
    def _generate_recommendations(self, 
                                hardware_info: Dict[str, Any] = None,
                                test_results: Dict[str, Any] = None) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Hardware-based recommendations
        if hardware_info:
            gpus = hardware_info.get("gpus", [])
            if not gpus:
                recommendations.append("🔧 Install a dedicated GPU for optimal vGPU performance")
            elif len(gpus) == 1:
                recommendations.append("💡 Consider adding additional GPUs for enhanced parallel processing")
            
            # CPU recommendations
            cpu_info = hardware_info.get("cpu", {})
            if cpu_info.get("estimated_gflops", 0) < 10:
                recommendations.append("⚡ CPU performance is low - consider upgrading for better mathematical acceleration")
            
            # Memory recommendations
            memory_info = hardware_info.get("memory", {})
            total_memory_gb = memory_info.get("total_bytes", 0) / (1024**3)
            if total_memory_gb < 8:
                recommendations.append("🗄️ Increase system memory to at least 8GB for optimal performance")
            elif memory_info.get("estimated_bandwidth_gbps", 0) < 10:
                recommendations.append("🚀 Consider faster memory (higher bandwidth) for improved data transfer")
        
        # Test-based recommendations
        if test_results:
            summary = test_results.get("summary", {})
            success_rate = (summary.get("passed_tests", 0) / max(1, summary.get("total_tests", 1))) * 100
            
            if success_rate < 75:
                recommendations.append("⚠️ Test success rate is below optimal - review failed tests and system configuration")
            
            # Category-specific recommendations
            categories = test_results.get("categories", {})
            
            if "AI Training" in categories:
                ai_category = categories["AI Training"]
                if ai_category.get("tests_failed", 0) > ai_category.get("tests_passed", 0):
                    recommendations.append("🤖 AI training performance needs improvement - optimize mathematical libraries")
            
            if "3D Rendering" in categories:
                render_category = categories["3D Rendering"]
                if render_category.get("tests_failed", 0) > 0:
                    recommendations.append("🎮 3D rendering optimization needed - check graphics drivers and API support")
        
        # General optimization recommendations
        recommendations.extend([
            "📊 Monitor performance metrics regularly for optimization opportunities",
            "🔄 Keep vGPU system updated to latest version for performance improvements",
            "📖 Review documentation for advanced configuration options",
            "🧪 Run comprehensive tests periodically to ensure continued performance"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _calculate_performance_metrics(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        metrics = {
            "computational_performance": {},
            "throughput_metrics": {},
            "efficiency_metrics": {},
            "comparison_baselines": {}
        }
        
        # Extract performance data from test categories
        categories = test_results.get("categories", {})
        
        # AI Training metrics
        if "AI Training" in categories:
            ai_data = categories["AI Training"].get("performance_metrics", {})
            tflops_values = []
            
            for test_name, test_metrics in ai_data.items():
                if "tflops" in test_metrics:
                    tflops_values.append(test_metrics["tflops"])
            
            if tflops_values:
                metrics["computational_performance"]["ai_training"] = {
                    "peak_tflops": max(tflops_values),
                    "avg_tflops": statistics.mean(tflops_values),
                    "min_tflops": min(tflops_values)
                }
        
        # 3D Rendering metrics
        if "3D Rendering" in categories:
            render_data = categories["3D Rendering"].get("performance_metrics", {})
            pixel_rates = []
            
            for test_name, test_metrics in render_data.items():
                if "pixels_per_second" in test_metrics:
                    pixel_rates.append(test_metrics["pixels_per_second"])
            
            if pixel_rates:
                metrics["throughput_metrics"]["rendering"] = {
                    "peak_pixels_per_sec": max(pixel_rates),
                    "avg_pixels_per_sec": statistics.mean(pixel_rates),
                    "min_pixels_per_sec": min(pixel_rates)
                }
        
        # Overall efficiency metrics
        total_execution_time = 0
        total_tests = 0
        
        for category_name, category_data in categories.items():
            tests_run = category_data.get("tests_run", [])
            for test in tests_run:
                if "execution_time" in test:
                    total_execution_time += test["execution_time"]
                    total_tests += 1
        
        if total_tests > 0:
            metrics["efficiency_metrics"] = {
                "avg_test_execution_time": total_execution_time / total_tests,
                "total_testing_time": total_execution_time,
                "tests_per_second": total_tests / total_execution_time if total_execution_time > 0 else 0
            }
        
        # Comparison baselines (theoretical maximums)
        metrics["comparison_baselines"] = {
            "target_ai_tflops": 100.0,  # Target AI performance
            "target_pixel_rate": 1000000000,  # Target pixel processing rate
            "target_test_success_rate": 95.0  # Target test success rate
        }
        
        return metrics
    
    def _rate_performance(self, value: float, metric_type: str) -> str:
        """Rate performance based on metric type and value"""
        if metric_type == "cpu_gflops":
            if value >= 50: return "Excellent"
            elif value >= 20: return "Good"
            elif value >= 10: return "Fair"
            else: return "Poor"
            
        elif metric_type == "memory_bandwidth":
            if value >= 50: return "Excellent"
            elif value >= 25: return "Good" 
            elif value >= 10: return "Fair"
            else: return "Poor"
            
        elif metric_type == "compatibility":
            if value >= 80: return "Excellent"
            elif value >= 60: return "Good"
            elif value >= 40: return "Fair"
            else: return "Poor"
            
        else:
            return "Unknown"
    
    def _assess_gpu_compatibility(self, gpu: Dict[str, Any]) -> float:
        """Assess GPU compatibility score (0-100)"""
        score = 50  # Base score
        
        gpu_type = gpu.get("type", "").upper()
        gpu_name = gpu.get("name", "").upper()
        
        # Type-based scoring
        if gpu_type == "NVIDIA":
            score += 30
        elif gpu_type == "AMD":
            score += 25
        elif gpu_type == "APPLE":
            score += 20
        
        # Name-based scoring (modern cards)
        if any(term in gpu_name for term in ["RTX", "GTX", "RADEON", "VEGA"]):
            score += 15
        
        # Detection method scoring
        detection_method = gpu.get("detected_method", "")
        if detection_method in ["nvidia-smi", "rocm-smi"]:
            score += 5
        
        return min(100, score)
    
    def _generate_json_report(self, report: Dict[str, Any]):
        """Generate JSON format report"""
        json_path = self.artifacts_dir / "comprehensive_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def _generate_markdown_report(self, report: Dict[str, Any]):
        """Generate Markdown format report"""
        md_path = self.artifacts_dir / "comprehensive_report.md"
        
        markdown_content = self._format_markdown_report(report)
        
        with open(md_path, 'w') as f:
            f.write(markdown_content)
    
    def _generate_summary_report(self, report: Dict[str, Any]):
        """Generate executive summary report"""
        summary_path = self.artifacts_dir / "executive_summary.txt"
        
        summary = report.get("executive_summary", {})
        
        content = f"""vGPU v1.9 Executive Summary Report
Generated: {report['report_metadata']['generation_time']}

OVERALL STATUS: {summary.get('overall_status', 'Unknown').upper()}

KEY METRICS:
"""
        
        for metric, value in summary.get("key_metrics", {}).items():
            content += f"• {metric.replace('_', ' ').title()}: {value}\n"
        
        content += "\nSUCCESS HIGHLIGHTS:\n"
        for highlight in summary.get("success_highlights", []):
            content += f"✅ {highlight}\n"
        
        content += "\nCRITICAL ISSUES:\n"
        for issue in summary.get("critical_issues", []):
            content += f"❌ {issue}\n"
        
        content += "\nTOP RECOMMENDATIONS:\n"
        for i, recommendation in enumerate(report.get("recommendations", [])[:5], 1):
            content += f"{i}. {recommendation}\n"
        
        with open(summary_path, 'w') as f:
            f.write(content)
    
    def _format_markdown_report(self, report: Dict[str, Any]) -> str:
        """Format comprehensive report as Markdown"""
        md = f"""# vGPU v1.9 Comprehensive Analysis Report

**Generated:** {report['report_metadata']['generation_time']}  
**Version:** {report['report_metadata']['vgpu_version']}

## Executive Summary

**Overall Status:** {report['executive_summary'].get('overall_status', 'Unknown').upper()}

### Key Metrics
"""
        
        for metric, value in report['executive_summary'].get('key_metrics', {}).items():
            md += f"- **{metric.replace('_', ' ').title()}:** {value}\n"
        
        md += "\n### Success Highlights\n"
        for highlight in report['executive_summary'].get('success_highlights', []):
            md += f"✅ {highlight}\n\n"
        
        md += "### Critical Issues\n"
        for issue in report['executive_summary'].get('critical_issues', []):
            md += f"❌ {issue}\n\n"
        
        # Hardware Analysis
        hw_analysis = report.get('hardware_analysis', {})
        if hw_analysis:
            md += "## Hardware Analysis\n\n"
            
            detection = hw_analysis.get('detection_summary', {})
            md += f"**GPUs Detected:** {detection.get('gpus_found', 0)}\n"
            md += f"**Accelerators Found:** {detection.get('accelerators_found', 0)}\n"
            md += f"**Platform:** {detection.get('platform', 'Unknown')}\n\n"
            
            perf_analysis = hw_analysis.get('performance_analysis', {})
            if 'cpu_performance' in perf_analysis:
                cpu = perf_analysis['cpu_performance']
                md += f"**CPU Performance:** {cpu.get('gflops', 0):.2f} GFLOPS ({cpu.get('rating', 'Unknown')})\n"
            
            if 'memory_performance' in perf_analysis:
                mem = perf_analysis['memory_performance']
                md += f"**Memory Bandwidth:** {mem.get('bandwidth_gbps', 0):.2f} GB/s ({mem.get('rating', 'Unknown')})\n"
        
        # Test Analysis
        test_analysis = report.get('test_analysis', {})
        if test_analysis:
            md += "\n## Test Results Analysis\n\n"
            
            test_summary = test_analysis.get('test_summary', {})
            md += f"**Total Tests:** {test_summary.get('total_tests', 0)}\n"
            md += f"**Passed:** {test_summary.get('passed_tests', 0)}\n"
            md += f"**Failed:** {test_summary.get('failed_tests', 0)}\n"
            md += f"**Success Rate:** {test_summary.get('success_rate', 0):.1f}%\n\n"
            
            # Category breakdown
            category_analysis = test_analysis.get('category_analysis', {})
            for category, analysis in category_analysis.items():
                md += f"### {category}\n"
                md += f"- **Success Rate:** {analysis.get('success_rate', 0):.1f}%\n"
                md += f"- **Avg Execution Time:** {analysis.get('avg_execution_time', 0):.3f}s\n"
                
                highlights = analysis.get('performance_highlights', [])
                if highlights:
                    md += "- **Performance Highlights:**\n"
                    for highlight in highlights:
                        md += f"  - {highlight}\n"
                md += "\n"
        
        # Recommendations
        md += "## Recommendations\n\n"
        for i, recommendation in enumerate(report.get('recommendations', []), 1):
            md += f"{i}. {recommendation}\n"
        
        return md
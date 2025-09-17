#!/usr/bin/env python3
"""
Build Claims Registry - Identify all claimed benchmarks and results
Scans repository for performance claims, benchmark results, and extraordinary claims
"""

import os
import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple
import hashlib

def scan_file_for_claims(filepath: Path) -> List[Dict[str, Any]]:
    """Scan a file for performance claims and benchmark results"""
    
    if not filepath.exists() or filepath.is_dir():
        return []
    
    claims = []
    
    try:
        # Skip binary files and very large files
        if filepath.suffix in ['.exe', '.dll', '.so', '.dylib', '.bin', '.dat']:
            return []
        
        if filepath.stat().st_size > 10_000_000:  # Skip files > 10MB
            return []
            
        content = filepath.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Pattern 1: Performance speedups (e.g., "25.0x speedup", "1,000,000x faster")
            speedup_patterns = [
                r'(\d+(?:,\d+)*(?:\.\d+)?)\s*x\s*(?:speedup|faster|improvement)',
                r'speedup\s*:?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*x',
                r'(\d+(?:,\d+)*(?:\.\d+)?)\s*times\s*faster',
                r'improved?\s*by\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*x'
            ]
            
            for pattern in speedup_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    claims.append({
                        'claim_id': f"{filepath.name}_{line_num}_{len(claims)+1}",
                        'test_file': str(filepath.relative_to(Path.cwd())),
                        'line_number': line_num,
                        'reported_metric': 'speedup',
                        'reported_value': match.group(1).replace(',', ''),
                        'baseline_reference': 'unknown',
                        'claim_text': line,
                        'claim_type': 'performance'
                    })
            
            # Pattern 2: FLOPS claims (e.g., "3.0×10¹² FLOPS", "infinite FLOPS")
            flops_patterns = [
                r'(\d+(?:\.\d+)?)\s*[×x]\s*10[¹²³⁴⁵⁶⁷⁸⁹⁰]+\s*FLOPS',
                r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:G|T|P|E)?FLOPS',
                r'infinite\s+FLOPS',
                r'near-infinite\s+FLOPS'
            ]
            
            for pattern in flops_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    value = match.group(1) if match.groups() else 'infinite'
                    claims.append({
                        'claim_id': f"{filepath.name}_{line_num}_{len(claims)+1}",
                        'test_file': str(filepath.relative_to(Path.cwd())),
                        'line_number': line_num,
                        'reported_metric': 'FLOPS',
                        'reported_value': value,
                        'baseline_reference': 'theoretical',
                        'claim_text': line,
                        'claim_type': 'computational'
                    })
            
            # Pattern 3: Extraordinary claims 
            extraordinary_patterns = [
                r'breakthrough',
                r'revolutionary',
                r'near-infinite',
                r'quadrillion',
                r'transcend(ence|ent)',
                r'universal\s+(?:solver|replacement)',
                r'beyond\s+quantum',
                r'infinite\s+(?:computing|power|speed)',
                r'replace\s+(?:GPU|quantum)',
                r'millions?\s+of\s+times\s+faster'
            ]
            
            for pattern in extraordinary_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    claims.append({
                        'claim_id': f"{filepath.name}_{line_num}_{len(claims)+1}",
                        'test_file': str(filepath.relative_to(Path.cwd())),
                        'line_number': line_num,
                        'reported_metric': 'extraordinary_claim',
                        'reported_value': pattern,
                        'baseline_reference': 'not_applicable',
                        'claim_text': line,
                        'claim_type': 'extraordinary'
                    })
                    break  # Only one extraordinary claim per line
            
            # Pattern 4: Benchmark results with specific values
            benchmark_patterns = [
                r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:ms|seconds?|FPS|ops/sec|MB/s|GB/s)',
                r'success\s+rate:?\s*(\d+(?:\.\d+)?)%',
                r'accuracy:?\s*(\d+(?:\.\d+)?)%',
                r'(\d+)/(\d+)\s*tests?\s*(?:pass|success)'
            ]
            
            for pattern in benchmark_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    claims.append({
                        'claim_id': f"{filepath.name}_{line_num}_{len(claims)+1}",
                        'test_file': str(filepath.relative_to(Path.cwd())),
                        'line_number': line_num,
                        'reported_metric': 'benchmark_result',
                        'reported_value': match.group(1),
                        'baseline_reference': 'measured',
                        'claim_text': line,
                        'claim_type': 'measurement'
                    })
    
    except Exception as e:
        # Log error but continue processing other files
        print(f"Error processing {filepath}: {e}")
        return []
    
    return claims

def find_benchmark_files() -> List[Path]:
    """Find all files that might contain benchmark claims"""
    
    repo_root = Path("/home/runner/work/we3/we3")
    
    # File patterns to search
    patterns = [
        "**/*.py",      # Python files
        "**/*.rs",      # Rust files
        "**/*.md",      # Markdown files
        "**/*.txt",     # Text files
        "**/*.json",    # JSON results
        "**/*.csv",     # CSV results
    ]
    
    # Directories to exclude
    exclude_dirs = {'.git', '__pycache__', 'target', 'node_modules', '.venv', 'venv'}
    
    files = []
    for pattern in patterns:
        for filepath in repo_root.glob(pattern):
            # Skip if any parent directory is in exclude list
            if any(part in exclude_dirs for part in filepath.parts):
                continue
            files.append(filepath)
    
    return sorted(files)

def analyze_benchmark_artifacts() -> List[Dict[str, Any]]:
    """Analyze existing benchmark artifacts for verification status"""
    
    repo_root = Path("/home/runner/work/we3/we3")
    artifacts = []
    
    # Check for benchmark result files
    benchmark_dirs = [
        repo_root / "benchmarks",
        repo_root / "implementations" / "phase-01",
    ]
    
    for bench_dir in benchmark_dirs:
        if not bench_dir.exists():
            continue
            
        for item in bench_dir.rglob("*"):
            if item.is_file():
                artifact_info = {
                    'path': str(item.relative_to(repo_root)),
                    'size': item.stat().st_size,
                    'exists': True,
                    'type': 'unknown'
                }
                
                # Classify artifact type
                if item.suffix == '.json':
                    artifact_info['type'] = 'json_result'
                elif item.suffix in ['.ppm', '.png', '.jpg']:
                    artifact_info['type'] = 'visual_evidence'
                elif item.name == 'verify.sh':
                    artifact_info['type'] = 'verification_script'
                elif item.name == 'run.sh':
                    artifact_info['type'] = 'execution_script'
                elif item.suffix == '.md':
                    artifact_info['type'] = 'documentation'
                
                artifacts.append(artifact_info)
    
    return artifacts

def create_claims_csv(claims: List[Dict[str, Any]], output_path: Path):
    """Create claims.csv file as required"""
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['claim_id', 'test_file', 'reported_metric', 'reported_value', 'baseline_reference', 
                     'line_number', 'claim_text', 'claim_type', 'verification_status']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for claim in claims:
            # Add verification status
            claim['verification_status'] = 'PENDING'
            writer.writerow(claim)

def main():
    """Main claims identification process"""
    
    print("🔍 BUILDING CLAIMS REGISTRY")
    print("=" * 50)
    
    # Find all files to scan
    files_to_scan = find_benchmark_files()
    print(f"Scanning {len(files_to_scan)} files...")
    
    # Collect all claims
    all_claims = []
    files_with_claims = 0
    
    for filepath in files_to_scan:
        claims = scan_file_for_claims(filepath)
        if claims:
            all_claims.extend(claims)
            files_with_claims += 1
            print(f"  {filepath.name}: {len(claims)} claims")
    
    print(f"\n📊 CLAIMS SUMMARY")
    print(f"Files scanned: {len(files_to_scan)}")
    print(f"Files with claims: {files_with_claims}")
    print(f"Total claims found: {len(all_claims)}")
    
    # Analyze claim types
    claim_types = {}
    for claim in all_claims:
        claim_type = claim['claim_type']
        claim_types[claim_type] = claim_types.get(claim_type, 0) + 1
    
    print(f"\nClaim types:")
    for claim_type, count in sorted(claim_types.items()):
        print(f"  {claim_type}: {count}")
    
    # Analyze benchmark artifacts
    artifacts = analyze_benchmark_artifacts()
    print(f"\nBenchmark artifacts found: {len(artifacts)}")
    
    artifact_types = {}
    for artifact in artifacts:
        artifact_type = artifact['type']
        artifact_types[artifact_type] = artifact_types.get(artifact_type, 0) + 1
    
    print("Artifact types:")
    for artifact_type, count in sorted(artifact_types.items()):
        print(f"  {artifact_type}: {count}")
    
    # Create output files
    output_dir = Path("/home/runner/work/we3/we3")
    
    # Create claims.csv as required
    claims_csv = output_dir / "claims.csv"
    create_claims_csv(all_claims, claims_csv)
    print(f"\n✅ Created: {claims_csv}")
    
    # Create detailed claims registry JSON
    claims_json = output_dir / "claims_registry.json"
    registry_data = {
        'scan_date': '2024-12-28',
        'total_claims': len(all_claims),
        'files_scanned': len(files_to_scan),
        'claim_types': claim_types,
        'artifacts': artifacts,
        'claims': all_claims
    }
    
    with open(claims_json, 'w', encoding='utf-8') as f:
        json.dump(registry_data, f, indent=2)
    print(f"✅ Created: {claims_json}")
    
    # Identify high-priority claims for verification
    print(f"\n🎯 HIGH-PRIORITY CLAIMS FOR VERIFICATION:")
    
    extraordinary_claims = [c for c in all_claims if c['claim_type'] == 'extraordinary'][:10]
    performance_claims = [c for c in all_claims if c['claim_type'] == 'performance'][:10]
    
    print("Top 10 Extraordinary Claims:")
    for i, claim in enumerate(extraordinary_claims[:10], 1):
        print(f"  {i}. {claim['test_file']}: {claim['claim_text'][:80]}...")
    
    print("\nTop 10 Performance Claims:")
    for i, claim in enumerate(performance_claims[:10], 1):
        value = claim['reported_value']
        print(f"  {i}. {claim['test_file']}: {value}x {claim['reported_metric']}")
    
    print(f"\n✅ CLAIMS REGISTRY BUILD COMPLETE")
    print(f"Next step: Verify claims using run.sh and verify.sh scripts")
    
    return len(all_claims)

if __name__ == "__main__":
    main()
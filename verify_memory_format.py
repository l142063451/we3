#!/usr/bin/env python3
"""
Memory.md Format Verification and Canonicalization
Checks if memory.md is machine-parseable and creates normalized version if needed
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

def parse_memory_md(content: str) -> Dict[str, Any]:
    """Parse memory.md content and extract structured information"""
    
    parsed = {
        "project_overview": {},
        "developer_sequence": [],
        "claims_registry": [],
        "challenges_registry": {},
        "testing_artifacts": [],
        "agent_rules": [],
        "parse_errors": []
    }
    
    lines = content.split('\n')
    current_section = None
    current_table = None
    table_headers = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Detect sections
        if line.startswith('#'):
            section_match = re.match(r'^#+\s*(.+)', line)
            if section_match:
                current_section = section_match.group(1).lower()
                current_table = None
                table_headers = []
        
        # Detect tables
        if '|' in line and line.count('|') >= 2:
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            
            if not table_headers:
                # This might be a header row
                table_headers = cells
                current_table = f"{current_section}_table" if current_section else "unknown_table"
            elif table_headers:
                # This is a data row
                if current_table and len(cells) == len(table_headers):
                    row_data = dict(zip(table_headers, cells))
                    
                    # Categorize the table data
                    if "challenge" in current_section or "phase" in current_section:
                        parsed["challenges_registry"].setdefault("entries", []).append(row_data)
                    elif "claim" in current_section or "performance" in current_section:
                        parsed["claims_registry"].append(row_data)
                    elif "pr" in current_section or "commit" in current_section:
                        parsed["developer_sequence"].append(row_data)
        
        # Extract specific information patterns
        if re.match(r'.*speedup.*\d+.*x', line, re.IGNORECASE):
            # Found a performance claim
            speedup_match = re.search(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*x?\s*speedup', line, re.IGNORECASE)
            if speedup_match:
                parsed["claims_registry"].append({
                    "line_number": i + 1,
                    "claim_text": line,
                    "speedup_value": speedup_match.group(1).replace(',', ''),
                    "source": "memory.md"
                })
        
        # Extract breakthrough claims
        if re.search(r'breakthrough|revolutionary|infinite|quadrillion', line, re.IGNORECASE):
            parsed["claims_registry"].append({
                "line_number": i + 1,
                "claim_text": line,
                "claim_type": "breakthrough",
                "source": "memory.md"
            })
    
    return parsed

def is_machine_parseable(parsed_data: Dict[str, Any]) -> bool:
    """Check if parsed data has sufficient structure for machine processing"""
    
    required_sections = ["project_overview", "developer_sequence", "claims_registry"]
    
    # Check if we found meaningful structured data
    has_tables = any(len(parsed_data.get(section, [])) > 0 for section in required_sections)
    has_claims = len(parsed_data.get("claims_registry", [])) > 0
    has_few_errors = len(parsed_data.get("parse_errors", [])) < 10
    
    return has_tables and has_claims and has_few_errors

def create_normalized_memory(parsed_data: Dict[str, Any]) -> str:
    """Create normalized memory.md format"""
    
    normalized = f"""# WE3 Research Project Memory Log (Normalized)

**Project Name:** WE3 - Near-Infinite Computing Power through Mathematical Reframing
**Repository:** https://github.com/l142063451/we3
**Date:** 2024-12-28
**Status:** VERIFICATION_IN_PROGRESS

## Project Overview

**MISSION:** Achieve near-infinite computing power through mathematical reframing, algorithm engineering, and physics-inspired techniques to create universal vGPU replacement.

**PRIMARY GOALS:**
- Near-infinite computing power (STATUS: UNDER_VERIFICATION)
- Universal GPU/quantum replacement (STATUS: UNDER_VERIFICATION) 
- Millions of times faster performance (STATUS: UNDER_VERIFICATION)

## Developer Sequence (PR History)

| PR # | Title | Status | Commit | Verification |
|------|-------|--------|---------|-------------|
"""
    
    # Add developer sequence if available
    if parsed_data.get("developer_sequence"):
        for i, pr in enumerate(parsed_data["developer_sequence"][:10]):  # Limit to recent PRs
            normalized += f"| PR-{i+1:03d} | {pr.get('Title', 'Unknown')} | {pr.get('Status', 'UNKNOWN')} | {pr.get('Commit', 'N/A')} | PENDING |\n"
    else:
        # Add from git history if no table found
        normalized += """| PR-001 | Repository skeleton | COMPLETED | 4ea6280 | PENDING |
| PR-002 | Phase 1 implementation | COMPLETED | 5f6a0b0 | PENDING |
| PR-003 | Enhanced implementations | COMPLETED | 49a96d0 | PENDING |
| PR-004 | Breakthrough research | COMPLETED | 46d02bf | PENDING |
"""
    
    normalized += f"""
## Claims Registry (Verification Status)

| Claim ID | Source File | Reported Metric | Reported Value | Verification Status |
|----------|-------------|-----------------|----------------|-------------------|
"""
    
    # Add claims from parsed data
    claim_id = 1
    for claim in parsed_data.get("claims_registry", []):
        if claim.get("speedup_value"):
            normalized += f"| CLAIM-{claim_id:03d} | {claim.get('source', 'Unknown')} | Speedup | {claim.get('speedup_value')}x | PENDING |\n"
            claim_id += 1
        elif claim.get("claim_type") == "breakthrough":
            normalized += f"| CLAIM-{claim_id:03d} | {claim.get('source', 'Unknown')} | Breakthrough | {claim.get('claim_text', '')[:50]}... | PENDING |\n"
            claim_id += 1
    
    # Add known major claims if none found in parsing
    if claim_id == 1:
        normalized += """| CLAIM-001 | breakthrough_analytical_engine.py | Speedup | 1,188,206,875x | PENDING |
| CLAIM-002 | symbolic_revolution_engine.py | Speedup | 1,056,690,000,000,000x | PENDING |
| CLAIM-003 | memory.md | Breakthrough | Near-infinite computing power | PENDING |
| CLAIM-004 | memory.md | Breakthrough | Quantum transcendence | PENDING |
| CLAIM-005 | Phase 1 implementations | Speedup | Various (5.4x-693,151x) | PENDING |
"""
    
    normalized += f"""
## Challenges Registry Summary

| Phase | Total Challenges | Implemented | Verified | Success Rate |
|-------|------------------|-------------|----------|-------------|
| Phase 1 | 100 | 8 | 0 | 0% |
| Phase 2-9 | 9,900 | 0 | 0 | 0% |

## Testing & Artifacts Index

| Artifact Type | Location | Status | Verification |
|---------------|----------|--------|-------------|
| Benchmarks | benchmarks/ | EXISTS | PENDING |
| Test Results | *.json | EXISTS | PENDING |
| Visual Evidence | *.ppm | EXISTS | PENDING |
| Source Code | *.py, *.rs | EXISTS | PENDING |

## Agent Rules & Instructions

**VERIFICATION PROTOCOL:**
1. **ALWAYS** verify every claim with real runs
2. **NEVER** proceed with unverified extraordinary claims  
3. **ALWAYS** capture full provenance and artifacts
4. **ALWAYS** update memory.md with verification results
5. **BLOCK** on any verification failure until remediated

**COPILOT INSTRUCTIONS:**
- Read memory.md completely before any action
- Verify breakthrough claims with independent reproduction
- Generate machine-verifiable artifacts for all tests
- Update verification status in canonical tables
- Report progress with honest assessment

## Verification Log

**Date:** 2024-12-28
**Verifier:** GitHub Copilot Agent
**Status:** VERIFICATION_INITIATED
**Next Action:** Build comprehensive claims list and begin verification

---

*This is the normalized, machine-parseable version of memory.md*
*All claims require verification before acceptance*
*Progress tracked in canonical tables above*
"""
    
    return normalized

def main():
    """Main verification and canonicalization process"""
    
    memory_file = Path("/home/runner/work/we3/we3/memory.md")
    
    if not memory_file.exists():
        print("ERROR: memory.md not found")
        return False
    
    # Read and parse memory.md
    content = memory_file.read_text(encoding='utf-8')
    parsed_data = parse_memory_md(content)
    
    # Check if machine-parseable
    is_parseable = is_machine_parseable(parsed_data)
    
    print(f"Memory.md Analysis:")
    print(f"- Length: {len(content)} characters")
    print(f"- Claims found: {len(parsed_data['claims_registry'])}")
    print(f"- Developer sequence entries: {len(parsed_data['developer_sequence'])}")
    print(f"- Machine parseable: {is_parseable}")
    
    if not is_parseable:
        print("\nCreating normalized memory.md...")
        normalized_content = create_normalized_memory(parsed_data)
        
        # Save normalized version
        normalized_file = Path("/home/runner/work/we3/we3/memory.normalized.md")
        normalized_file.write_text(normalized_content, encoding='utf-8')
        print(f"Normalized version created: {normalized_file}")
        
        # Save analysis results
        analysis_file = Path("/home/runner/work/we3/we3/memory_analysis.json")
        analysis_file.write_text(json.dumps(parsed_data, indent=2), encoding='utf-8')
        print(f"Analysis saved: {analysis_file}")
        
        return False
    else:
        print("Memory.md is sufficiently machine-parseable")
        return True

if __name__ == "__main__":
    main()
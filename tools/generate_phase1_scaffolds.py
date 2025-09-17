#!/usr/bin/env python3
"""
Generate Phase 1 challenge scaffolds

Creates directory structure and placeholder files for the first 100 challenges.
Each challenge gets its own directory with run.sh and verify.sh that fail intentionally.
"""

import json
import os
import sys

def create_challenge_scaffold(challenge_id: str, challenge_data: dict):
    """Create scaffold directory and files for a challenge"""
    
    # Create challenge directory
    challenge_dir = f"implementations/phase-01/{challenge_id}"
    os.makedirs(challenge_dir, exist_ok=True)
    
    # Generate success criteria lines
    success_criteria_lines = '\n'.join(f'echo "- {criterion}"' for criterion in challenge_data.get('success_criteria', []))
    
    # Create run.sh that fails intentionally
    run_sh = f"""#!/bin/bash
# WE3 Challenge Implementation: {challenge_id}
# Title: {challenge_data['title']}
# Category: {challenge_data['category']}
# Difficulty: {challenge_data['difficulty']}
# 
# This is an intentionally failing stub until the challenge is implemented.

set -e

echo "=== WE3 Challenge {challenge_id} ==="
echo "Title: {challenge_data['title']}"
echo "Category: {challenge_data['category']}" 
echo "Difficulty: {challenge_data['difficulty']}"
echo "Description: {challenge_data['description']}"
echo ""
echo "SUCCESS CRITERIA:"
{success_criteria_lines}

echo ""
echo "DEPENDENCIES: {', '.join(challenge_data.get('dependencies', ['None']))}"
echo "TAGS: {', '.join(challenge_data.get('tags', []))}"
echo ""

# Record start time
START_TIME=$(date +%s.%N)

echo "⚠️  CHALLENGE NOT YET IMPLEMENTED ⚠️"
echo "This is a placeholder that fails intentionally until implementation."
echo ""
echo "Expected implementation includes:"
echo "- Actual algorithm implementation"
echo "- Unit tests with verification"
echo "- Performance measurements"
echo "- Formal verification where applicable"
echo ""
echo "Status: STUB - Implementation required"

# Record end time and create basic result.json
END_TIME=$(date +%s.%N)
EXECUTION_TIME=$(echo "$END_TIME - $START_TIME" | bc)

# Create hardware.json
cat > hardware.json <<EOF
{{
  "cpu": "$(lscpu | grep 'Model name' | cut -d':' -f2- | sed 's/^[[:space:]]*//')",
  "memory": "$(free -h | grep '^Mem:' | awk '{{print $2}}')",
  "os": "$(uname -s) $(uname -r)",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
}}
EOF

# Create result.json with failure status
cat > result.json <<EOF
{{
  "challenge_id": "{challenge_id}",
  "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "container": "sha256:local-development-environment",
  "hardware": {{
    "cpu": "$(lscpu | grep 'Model name' | cut -d':' -f2- | sed 's/^[[:space:]]*//' | head -1)",
    "memory": "$(free -h | grep '^Mem:' | awk '{{print $2}}')",
    "os": "$(uname -s) $(uname -r)"
  }},
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)",
  "execution_time": $EXECUTION_TIME,
  "verification": "FAIL",
  "status": "STUB_NOT_IMPLEMENTED",
  "metrics": {{
    "flops": 0,
    "memory_usage": 0,
    "energy_j": 0
  }},
  "implementation_required": true,
  "success_criteria_met": false
}}
EOF

echo "RESULT: FAIL - Challenge stub not yet implemented"
exit 1
"""
    
    with open(f"{challenge_dir}/run.sh", "w") as f:
        f.write(run_sh)
    os.chmod(f"{challenge_dir}/run.sh", 0o755)
    
    # Create verify.sh that checks for implementation
    verify_sh = f"""#!/bin/bash
# Verification script for {challenge_id}
# Checks if the challenge has been properly implemented

set -e

echo "=== Verification for {challenge_id} ==="

# Check if result.json exists and is valid
if [ ! -f "result.json" ]; then
    echo "FAIL: result.json not found"
    exit 1
fi

# Parse result.json to check verification status
VERIFICATION_STATUS=$(python3 -c "import json; data=json.load(open('result.json')); print(data.get('verification', 'UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")

if [ "$VERIFICATION_STATUS" = "PASS" ]; then
    echo "PASS: Challenge verification successful"
    exit 0
else
    echo "FAIL: Challenge verification failed or not implemented"
    echo "Status: $VERIFICATION_STATUS"
    exit 1
fi
"""
    
    with open(f"{challenge_dir}/verify.sh", "w") as f:
        f.write(verify_sh)
    os.chmod(f"{challenge_dir}/verify.sh", 0o755)
    
    # Create README.md with challenge details
    readme = f"""# Challenge {challenge_id}: {challenge_data['title']}

**Category:** {challenge_data['category']}  
**Difficulty:** {challenge_data['difficulty']}  
**Estimated Time:** {challenge_data['estimated_time']}  
**Owner:** {challenge_data['owner']}  

## Description

{challenge_data['description']}

## Success Criteria

{chr(10).join(f"- {criterion}" for criterion in challenge_data.get('success_criteria', []))}

## Verification Method

{challenge_data['verification_method']}

## Dependencies

{', '.join(challenge_data.get('dependencies', ['None']))}

## Tags

{', '.join(challenge_data.get('tags', []))}

## Implementation Status

🚧 **NOT IMPLEMENTED** - This challenge is currently a stub that fails intentionally.

## Usage

```bash
# Run the challenge (will fail until implemented)
./run.sh

# Verify the implementation (will fail until implemented) 
./verify.sh
```

## Implementation Guidelines

1. Replace the stub in `run.sh` with actual algorithm implementation
2. Add comprehensive unit tests with verification
3. Measure and report performance metrics
4. Provide formal verification where applicable
5. Ensure all success criteria are met
6. Update result.json with actual measurements

## Files

- `run.sh` - Main implementation script (currently stub)
- `verify.sh` - Verification script
- `hardware.json` - Hardware specifications (generated)
- `result.json` - Results and metrics (generated)
- `README.md` - This file
"""
    
    with open(f"{challenge_dir}/README.md", "w") as f:
        f.write(readme)

def main():
    # Load challenges registry
    with open('challenges/registry.json', 'r') as f:
        registry = json.load(f)
    
    challenges = registry['challenges']
    
    # Create scaffolds for first 100 challenges (phase-01)
    phase_01_challenges = [c for c in challenges if c['owner'] == 'phase-01']
    
    print(f"Creating scaffolds for {len(phase_01_challenges)} Phase 1 challenges...")
    
    for challenge in phase_01_challenges:
        challenge_id = challenge['challenge_id']
        print(f"Creating scaffold for {challenge_id}: {challenge['title']}")
        create_challenge_scaffold(challenge_id, challenge)
    
    print(f"✅ Created {len(phase_01_challenges)} challenge scaffolds in implementations/phase-01/")
    print("Each challenge has:")
    print("- run.sh (executable stub that fails intentionally)")
    print("- verify.sh (verification script)")
    print("- README.md (challenge documentation)")
    print("")
    print("Next steps:")
    print("1. Implement actual algorithms in run.sh scripts")
    print("2. Add comprehensive testing and verification")
    print("3. Run challenges to generate real performance data")

if __name__ == '__main__':
    main()
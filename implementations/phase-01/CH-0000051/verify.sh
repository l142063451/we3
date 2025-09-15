#!/bin/bash
# Verification script for CH-0000051
# Checks if the challenge has been properly implemented

set -e

echo "=== Verification for CH-0000051 ==="

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

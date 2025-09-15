#!/usr/bin/env python3
"""
WE3 Challenge Registry Generator

Generates 10,000 comprehensive challenges for the WE3 implementation program.
Each challenge follows the strict schema defined in new_implementations.md.
"""

import json
import argparse
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any
import random

# Challenge categories and their distributions
CATEGORIES = {
    "MATHEMATICAL": 2000,
    "ALGORITHMIC": 2000, 
    "CRYPTOGRAPHIC": 1000,
    "RENDERING": 1000,
    "AI_TRAINING": 2000,
    "SYSTEM": 2000
}

DIFFICULTIES = ["TRIVIAL", "EASY", "MEDIUM", "HARD", "EXTREME", "RESEARCH"]
TIME_ESTIMATES = ["1h", "8h", "1d", "1w", "1m"]
VERIFICATION_METHODS = ["UNIT_TEST", "INTEGRATION_TEST", "FORMAL_PROOF", "BENCHMARK"]

# Challenge templates for each category
CHALLENGE_TEMPLATES = {
    "MATHEMATICAL": [
        {
            "title": "Fast Fourier Transform Implementation",
            "description": "Implement high-precision FFT with Cooley-Tukey algorithm",
            "success_criteria": ["FFT correctness within 1e-10 precision", "O(n log n) complexity verified"],
            "tags": ["fft", "signal-processing", "numerical-analysis"]
        },
        {
            "title": "Matrix Decomposition Algorithms", 
            "description": "Implement SVD, QR, LU decomposition with numerical stability",
            "success_criteria": ["Decomposition accuracy within 1e-12", "Condition number handling"],
            "tags": ["linear-algebra", "matrix", "numerical-stability"]
        },
        {
            "title": "Generating Function Coefficient Extraction",
            "description": "Extract coefficients from rational and algebraic generating functions",
            "success_criteria": ["Coefficient accuracy for large indices", "Efficient extraction algorithm"],
            "tags": ["generating-functions", "combinatorics", "algebra"]
        },
        {
            "title": "High-Precision Arithmetic",
            "description": "Arbitrary precision arithmetic with interval bounds",
            "success_criteria": ["Correct rounding modes", "Error bound computation"],
            "tags": ["precision", "arithmetic", "intervals"]
        },
        {
            "title": "Special Function Computation",
            "description": "Compute Bessel, gamma, elliptic functions with high precision",
            "success_criteria": ["Reference accuracy validation", "Efficient series convergence"],
            "tags": ["special-functions", "analysis", "numerical"]
        }
    ],
    "ALGORITHMIC": [
        {
            "title": "Advanced Graph Algorithms",
            "description": "Implement shortest paths, maximum flow, minimum cut algorithms",
            "success_criteria": ["Optimal solution verification", "Complexity bounds met"],
            "tags": ["graphs", "optimization", "algorithms"]
        },
        {
            "title": "Dynamic Programming Solutions",
            "description": "Solve optimization problems using DP with memoization",
            "success_criteria": ["Optimal substructure verification", "Space-time tradeoff analysis"],
            "tags": ["dynamic-programming", "optimization", "memoization"]
        },
        {
            "title": "Advanced Data Structures",
            "description": "Implement B-trees, suffix arrays, persistent data structures",
            "success_criteria": ["Correctness invariants maintained", "Performance characteristics met"],
            "tags": ["data-structures", "trees", "persistence"]
        },
        {
            "title": "String Algorithms",
            "description": "Pattern matching, suffix trees, Burrows-Wheeler transform",
            "success_criteria": ["Pattern matching correctness", "Linear time complexity"],
            "tags": ["strings", "pattern-matching", "text-processing"]
        },
        {
            "title": "Approximation Algorithms",
            "description": "Polynomial-time approximation schemes for NP-hard problems",
            "success_criteria": ["Approximation ratio guaranteed", "Polynomial runtime verified"],
            "tags": ["approximation", "np-hard", "optimization"]
        }
    ],
    "CRYPTOGRAPHIC": [
        {
            "title": "Symmetric Encryption Implementation",
            "description": "Implement AES, ChaCha20 with test vectors validation",
            "success_criteria": ["Test vector compliance", "Side-channel resistance analysis"],
            "tags": ["aes", "chacha20", "encryption", "test-keys-only"]
        },
        {
            "title": "Hash Function Security",
            "description": "Implement SHA-3, BLAKE2 with collision resistance testing",
            "success_criteria": ["Standard compliance", "Performance benchmarks"],
            "tags": ["sha3", "blake2", "hashing", "synthetic-only"]
        },
        {
            "title": "Public Key Cryptography",
            "description": "RSA, ECC implementation with synthetic key testing",
            "success_criteria": ["Signature verification", "Key generation validation"],
            "tags": ["rsa", "ecc", "public-key", "synthetic-keys"]
        },
        {
            "title": "Post-Quantum Cryptography",
            "description": "Implement lattice-based or code-based schemes",
            "success_criteria": ["Security parameter validation", "Performance analysis"],
            "tags": ["post-quantum", "lattice", "security"]
        },
        {
            "title": "Cryptographic Protocols",
            "description": "Zero-knowledge proofs, secure multi-party computation",
            "success_criteria": ["Protocol correctness", "Security proof validation"],
            "tags": ["zero-knowledge", "protocols", "security"]
        }
    ],
    "RENDERING": [
        {
            "title": "3D Rasterization Pipeline",
            "description": "Software 3D rendering with perspective correction",
            "success_criteria": ["Visual output correctness", "Frame rate measurements"],
            "tags": ["3d-rendering", "rasterization", "graphics"]
        },
        {
            "title": "Ray Tracing Implementation", 
            "description": "Monte Carlo ray tracing with global illumination",
            "success_criteria": ["Visual quality metrics", "Convergence analysis"],
            "tags": ["raytracing", "monte-carlo", "lighting"]
        },
        {
            "title": "Image Processing Algorithms",
            "description": "Filtering, enhancement, feature detection algorithms",
            "success_criteria": ["Image quality metrics", "Algorithm correctness"],
            "tags": ["image-processing", "filtering", "computer-vision"]
        },
        {
            "title": "Geometric Processing",
            "description": "Mesh processing, subdivision, simplification algorithms", 
            "success_criteria": ["Mesh quality preservation", "Geometric accuracy"],
            "tags": ["geometry", "meshes", "processing"]
        },
        {
            "title": "Shader Implementation",
            "description": "Vertex/fragment shaders for complex material rendering",
            "success_criteria": ["Shader compilation success", "Visual output validation"],
            "tags": ["shaders", "materials", "gpu-programming"]
        }
    ],
    "AI_TRAINING": [
        {
            "title": "Neural Network Training",
            "description": "Backpropagation implementation with optimization algorithms",
            "success_criteria": ["Training convergence", "Gradient correctness"],
            "tags": ["neural-networks", "backpropagation", "optimization"]
        },
        {
            "title": "Transformer Architecture",
            "description": "Multi-head attention and transformer blocks",
            "success_criteria": ["Architecture correctness", "Training stability"],
            "tags": ["transformers", "attention", "nlp"]
        },
        {
            "title": "Reinforcement Learning",
            "description": "Q-learning, policy gradient, actor-critic methods",
            "success_criteria": ["Policy convergence", "Reward optimization"],
            "tags": ["reinforcement-learning", "q-learning", "policy-gradient"]
        },
        {
            "title": "Convolutional Networks",
            "description": "CNN architectures for image classification",
            "success_criteria": ["Classification accuracy", "Feature learning validation"],
            "tags": ["cnn", "image-classification", "deep-learning"]
        },
        {
            "title": "Optimization Algorithms",
            "description": "Adam, RMSprop, advanced optimizers for deep learning",
            "success_criteria": ["Convergence speed", "Optimization stability"],
            "tags": ["optimization", "adam", "rmsprop", "training"]
        }
    ],
    "SYSTEM": [
        {
            "title": "Memory Management",
            "description": "Advanced memory allocators with fragmentation handling",
            "success_criteria": ["Memory efficiency", "Allocation speed"],
            "tags": ["memory", "allocation", "fragmentation"]
        },
        {
            "title": "Concurrent Programming",
            "description": "Lock-free data structures and parallel algorithms",
            "success_criteria": ["Thread safety", "Performance scaling"],
            "tags": ["concurrency", "lock-free", "parallel"]
        },
        {
            "title": "Performance Monitoring",
            "description": "System metrics collection and analysis tools",
            "success_criteria": ["Accurate measurements", "Low overhead monitoring"],
            "tags": ["performance", "monitoring", "metrics"]
        },
        {
            "title": "Hardware Interfacing",
            "description": "GPU, accelerator, and specialized hardware integration",
            "success_criteria": ["Hardware abstraction correctness", "Performance optimization"],
            "tags": ["hardware", "gpu", "acceleration"]
        },
        {
            "title": "Distributed Systems",
            "description": "Consensus algorithms, distributed storage, replication",
            "success_criteria": ["Consensus correctness", "Fault tolerance"],
            "tags": ["distributed", "consensus", "replication"]
        }
    ]
}

def generate_challenge_id(index: int) -> str:
    """Generate challenge ID in format CH-0000001"""
    return f"CH-{index:07d}"

def select_difficulty() -> str:
    """Select difficulty with weighted distribution"""
    weights = [30, 25, 20, 15, 7, 3]  # Trivial to Research
    return random.choices(DIFFICULTIES, weights=weights)[0]

def select_time_estimate(difficulty: str) -> str:
    """Select time estimate based on difficulty"""
    time_map = {
        "TRIVIAL": ["1h", "1h", "8h"],
        "EASY": ["1h", "8h", "8h"], 
        "MEDIUM": ["8h", "1d", "1d"],
        "HARD": ["1d", "1w", "1w"],
        "EXTREME": ["1w", "1m", "1m"],
        "RESEARCH": ["1m", "1m", "1m"]
    }
    return random.choice(time_map[difficulty])

def select_verification_method(category: str, difficulty: str) -> str:
    """Select verification method based on category and difficulty"""
    if difficulty in ["RESEARCH", "EXTREME"]:
        return random.choice(["FORMAL_PROOF", "BENCHMARK"])
    elif category == "MATHEMATICAL":
        return random.choice(["UNIT_TEST", "FORMAL_PROOF"])
    else:
        return random.choice(["UNIT_TEST", "INTEGRATION_TEST", "BENCHMARK"])

def generate_dependencies(challenge_index: int, total_generated: int) -> List[str]:
    """Generate realistic dependencies for a challenge"""
    if challenge_index <= 100:  # First 100 have no dependencies
        return []
    
    # Generate 0-3 dependencies from earlier challenges
    num_deps = random.choices([0, 1, 2, 3], weights=[50, 30, 15, 5])[0]
    if num_deps == 0:
        return []
    
    max_dep_index = min(challenge_index - 1, total_generated - 1)
    dep_indices = random.sample(range(1, max_dep_index + 1), min(num_deps, max_dep_index))
    return [generate_challenge_id(idx) for idx in dep_indices]

def generate_phase_owner(challenge_index: int) -> str:
    """Assign challenges to phases"""
    if challenge_index <= 100:
        return "phase-01"
    elif challenge_index <= 1000:
        return "phase-02"
    elif challenge_index <= 3000:
        return "phase-03"
    elif challenge_index <= 5000:
        return "phase-04"
    elif challenge_index <= 6500:
        return "phase-05"
    elif challenge_index <= 8000:
        return "phase-06"
    elif challenge_index <= 9000:
        return "phase-07"
    elif challenge_index <= 9500:
        return "phase-08"
    else:
        return "phase-09"

def generate_challenge(index: int, category: str) -> Dict[str, Any]:
    """Generate a single challenge"""
    challenge_id = generate_challenge_id(index)
    difficulty = select_difficulty()
    time_estimate = select_time_estimate(difficulty)
    verification_method = select_verification_method(category, difficulty)
    
    # Select template for this category
    template = random.choice(CHALLENGE_TEMPLATES[category])
    
    # Generate unique title with index
    title = f"{template['title']} #{index}"
    
    # Add category-specific details to description
    description = f"{template['description']} (Challenge {challenge_id})"
    
    # Add difficulty-specific success criteria
    success_criteria = template['success_criteria'].copy()
    if difficulty in ["EXTREME", "RESEARCH"]:
        success_criteria.append("Formal verification of correctness properties")
    if difficulty in ["HARD", "EXTREME", "RESEARCH"]:
        success_criteria.append("Performance analysis with complexity bounds")
    
    # Generate dependencies
    dependencies = generate_dependencies(index, index)
    
    challenge = {
        "challenge_id": challenge_id,
        "title": title,
        "category": category,
        "difficulty": difficulty,
        "estimated_time": time_estimate,
        "owner": generate_phase_owner(index),
        "description": description,
        "success_criteria": success_criteria,
        "verification_method": verification_method,
        "dependencies": dependencies,
        "tags": template["tags"].copy(),
        "created": datetime.now(timezone.utc).isoformat(),
        "status": "PENDING"
    }
    
    return challenge

def generate_challenges(count: int) -> List[Dict[str, Any]]:
    """Generate specified number of challenges"""
    challenges = []
    
    # Create category distribution
    category_list = []
    for category, category_count in CATEGORIES.items():
        category_list.extend([category] * category_count)
    
    # Shuffle to randomize order
    random.shuffle(category_list)
    
    # Ensure we have exactly the requested count
    if len(category_list) > count:
        category_list = category_list[:count]
    elif len(category_list) < count:
        # Fill remaining with random categories
        remaining = count - len(category_list)
        additional = random.choices(list(CATEGORIES.keys()), k=remaining)
        category_list.extend(additional)
    
    # Generate challenges
    for i in range(1, count + 1):
        category = category_list[i - 1]
        challenge = generate_challenge(i, category)
        challenges.append(challenge)
    
    return challenges

def main():
    parser = argparse.ArgumentParser(description='Generate WE3 challenge registry')
    parser.add_argument('--count', type=int, default=10000, help='Number of challenges to generate')
    parser.add_argument('--output', default='challenges/registry.json', help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"Generating {args.count} challenges...")
    
    # Generate challenges
    challenges = generate_challenges(args.count)
    
    # Create output directory if needed
    import os
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save registry
    registry = {
        "schema_version": "1.0",
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_challenges": len(challenges),
        "categories": CATEGORIES,
        "challenges": challenges
    }
    
    with open(args.output, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"Generated {len(challenges)} challenges")
    print(f"Saved to {args.output}")
    
    # Print statistics
    category_stats = {}
    difficulty_stats = {}
    phase_stats = {}
    
    for challenge in challenges:
        cat = challenge['category']
        diff = challenge['difficulty'] 
        owner = challenge['owner']
        
        category_stats[cat] = category_stats.get(cat, 0) + 1
        difficulty_stats[diff] = difficulty_stats.get(diff, 0) + 1
        phase_stats[owner] = phase_stats.get(owner, 0) + 1
    
    print("\nCategory Distribution:")
    for cat, count in sorted(category_stats.items()):
        print(f"  {cat}: {count}")
    
    print("\nDifficulty Distribution:")
    for diff, count in sorted(difficulty_stats.items()):
        print(f"  {diff}: {count}")
    
    print("\nPhase Distribution:")
    for phase, count in sorted(phase_stats.items()):
        print(f"  {phase}: {count}")

if __name__ == '__main__':
    main()
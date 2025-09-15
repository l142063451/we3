# Contributing to WE3 Research Project

## Scientific Integrity and Safety Guidelines

This project follows strict scientific integrity and safety protocols:

### Non-Negotiable Safety Rules
1. **No fabricated experiments or proofs** - Every claim must be supported by machine-checked proofs, reproducible numerical experiments, or published references
2. **Label all outputs** as PROVEN, VERIFIED, HEURISTIC, or SPECULATIVE
3. **Human sign-off required** for any claims replacing fundamental physical laws
4. **No unsafe implementations** - Research is theoretical/simulation/controlled lab-protocol only
5. **Update memory.md** for every change

### Contribution Process

1. **Fork and branch** from the main repository
2. **Follow methodological principles**:
   - Mathematics-first approach with complexity analysis
   - Multi-pronged validation (symbolic, machine-checked, numerical, cross-method)
   - Representation duality demonstration
   - Full tradeoff transparency
   - Complete reproducibility and provenance

3. **Required for each contribution**:
   - Code + tests + benchmarks
   - Jupyter notebooks with mathematical derivations
   - Proof scripts (Lean/Coq) where applicable
   - Memory.md updates with full provenance
   - CI artifacts and performance results

4. **Testing requirements**:
   - Unit tests with property-based testing
   - Cross-method equivalence tests
   - Statistical analysis with confidence intervals
   - Reproducibility across different machines

### Mathematical Framework Requirements

Each mathematical framework implementation must include:
- Formal definitions and key lemmas
- Rust core libraries with Python bindings
- Comprehensive test suites
- Performance metrics (T_compile, S_rep, T_query)
- Error bound analysis

### Code Style
- Rust: Follow `rustfmt` and `clippy` recommendations
- Python: Follow PEP 8, use type hints
- Proofs: Clear, well-documented formal proofs
- Documentation: Clear mathematical notation and explanations

### Pull Request Process
1. Ensure all tests pass and CI is green
2. Update memory.md with provenance information
3. Include performance benchmarks and analysis
4. Provide clear mathematical justification
5. Label claims appropriately (PROVEN/VERIFIED/HEURISTIC/SPECULATIVE)

### Escalation Policy
Contributors must escalate to project maintainers if:
- Finding apparent proofs that contradict accepted physical theory
- Proposing experimental protocols requiring specialized equipment
- Intending to file patents or make extraordinary public claims

## Getting Started

1. Set up development environment (see README.md)
2. Read through existing mathematical frameworks
3. Start with small, well-tested contributions
4. Follow the scientific methodology outlined in memory.md

Thank you for contributing to advancing the frontiers of mathematical and computational physics!
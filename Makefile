# WE3 Research Project Makefile

.PHONY: setup build test clean docs lint format check-format install-poetry install-rust

# Default target
all: build test

# Environment setup
setup: install-poetry install-rust
	@echo "🔧 Setting up development environment..."
	poetry install --no-interaction
	cargo build
	@echo "✅ Environment setup complete"

install-poetry:
	@echo "📦 Installing Poetry..."
	@if ! command -v poetry >/dev/null 2>&1; then \
		curl -sSL https://install.python-poetry.org | python3 -; \
	else \
		echo "Poetry already installed"; \
	fi

install-rust:
	@echo "🦀 Installing Rust..."
	@if ! command -v cargo >/dev/null 2>&1; then \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		source ~/.cargo/env; \
	else \
		echo "Rust already installed"; \
	fi

# Build targets
build: build-rust build-python

build-rust:
	@echo "🦀 Building Rust components..."
	cargo build --release

build-python:
	@echo "🐍 Building Python components..."
	poetry build

# Testing
test: test-rust test-python

test-rust:
	@echo "🧪 Running Rust tests..."
	cargo test --verbose

test-python:
	@echo "🧪 Running Python tests..."
	poetry run pytest python/tests/ -v

# Benchmarking
bench:
	@echo "⚡ Running benchmarks..."
	cargo bench

# Linting and formatting
lint: lint-rust lint-python

lint-rust:
	@echo "🔍 Linting Rust code..."
	cargo clippy --all-targets --all-features -- -D warnings

lint-python:
	@echo "🔍 Linting Python code..."
	poetry run flake8 python/we3
	poetry run mypy python/we3

format: format-rust format-python

format-rust:
	@echo "🎨 Formatting Rust code..."
	cargo fmt

format-python:
	@echo "🎨 Formatting Python code..."
	poetry run black python/we3
	poetry run isort python/we3

check-format: check-format-rust check-format-python

check-format-rust:
	@echo "✅ Checking Rust formatting..."
	cargo fmt --all -- --check

check-format-python:
	@echo "✅ Checking Python formatting..."
	poetry run black --check python/we3
	poetry run isort --check-only python/we3

# Documentation
docs: docs-rust docs-python

docs-rust:
	@echo "📚 Generating Rust documentation..."
	cargo doc --no-deps --document-private-items

docs-python:
	@echo "📚 Generating Python documentation..."
	cd docs && poetry run make html

# Jupyter notebooks
notebooks:
	@echo "📓 Starting Jupyter Lab..."
	poetry run jupyter lab

# Execute notebooks for testing
test-notebooks:
	@echo "🧪 Testing Jupyter notebooks..."
	find python/notebooks -name "*.ipynb" -exec poetry run jupyter nbconvert --to notebook --execute {} \;

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	cargo clean
	rm -rf target/
	rm -rf dist/
	rm -rf python/we3.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Update memory.md with current status
update-memory:
	@echo "📝 Updating memory.md..."
	@echo "- $$(date -u '+%Y-%m-%d %H:%M UTC'): $$(git log -1 --pretty=format:'%s')" >> memory.md

# Run all quality checks
check: check-format lint test
	@echo "✅ All quality checks passed"

# Development workflow
dev: format lint test
	@echo "🚀 Development workflow complete"

# CI simulation
ci: check test-notebooks
	@echo "🎯 CI simulation complete"

# Security audit
audit:
	@echo "🔒 Running security audit..."
	cargo audit
	poetry run pip-audit

# Install pre-commit hooks
install-hooks:
	@echo "🪝 Installing pre-commit hooks..."
	poetry run pre-commit install

# Generate coverage report
coverage:
	@echo "📊 Generating coverage report..."
	poetry run pytest python/tests/ --cov=we3 --cov-report=html --cov-report=term

# Profile performance
profile:
	@echo "⏱️ Running performance profiling..."
	poetry run python -m cProfile -s tottime python/we3/experiments/benchmark.py

# Help target
help:
	@echo "WE3 Research Project - Available targets:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  setup          - Set up complete development environment"
	@echo "  install-poetry - Install Poetry package manager"
	@echo "  install-rust   - Install Rust toolchain"
	@echo ""
	@echo "Building:"
	@echo "  build          - Build all components"
	@echo "  build-rust     - Build Rust components"
	@echo "  build-python   - Build Python components"
	@echo ""
	@echo "Testing:"
	@echo "  test           - Run all tests"
	@echo "  test-rust      - Run Rust tests"
	@echo "  test-python    - Run Python tests"
	@echo "  test-notebooks - Execute and test Jupyter notebooks"
	@echo "  coverage       - Generate test coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           - Run all linters"
	@echo "  format         - Format all code"
	@echo "  check-format   - Check code formatting"
	@echo "  check          - Run all quality checks"
	@echo ""
	@echo "Documentation:"
	@echo "  docs           - Generate all documentation"
	@echo "  notebooks      - Start Jupyter Lab"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          - Clean build artifacts"
	@echo "  audit          - Run security audit"
	@echo "  update-memory  - Update memory.md with current status"
	@echo ""
	@echo "Development:"
	@echo "  dev            - Run development workflow (format + lint + test)"
	@echo "  ci             - Simulate CI pipeline"
	@echo "  bench          - Run performance benchmarks"
	@echo "  profile        - Run performance profiling"
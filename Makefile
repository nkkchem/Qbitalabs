.PHONY: help install install-dev test lint format clean docker-build docker-run docs

PYTHON := python3
PIP := pip
PROJECT_NAME := qbitalabs

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)QBitaLabs$(NC) - Quantum-Bio Swarm Intelligence Platform"
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install the package
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install

install-all: ## Install all dependencies including optional
	$(PIP) install -e ".[dev,neuromorphic,ionq,docs]"

test: ## Run tests
	pytest tests/ -v --cov=src/qbitalabs --cov-report=term-missing

test-fast: ## Run tests without slow markers
	pytest tests/ -v -m "not slow" --cov=src/qbitalabs

test-quantum: ## Run quantum-specific tests
	pytest tests/ -v -m "quantum" --cov=src/qbitalabs

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	pytest tests/e2e/ -v

lint: ## Run linting
	ruff check src/ tests/
	mypy src/

lint-fix: ## Fix linting issues
	ruff check src/ tests/ --fix

format: ## Format code
	black src/ tests/
	ruff check src/ tests/ --fix

format-check: ## Check code formatting
	black --check src/ tests/
	ruff check src/ tests/

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build: ## Build Docker image
	docker build -t $(PROJECT_NAME):latest .

docker-run: ## Run Docker container
	docker run -it --rm -p 8000:8000 $(PROJECT_NAME):latest

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

docs: ## Build documentation
	mkdocs build

docs-serve: ## Serve documentation locally
	mkdocs serve

swarm-demo: ## Run 100-agent swarm demo
	$(PYTHON) examples/100_agent_discovery.py

quantum-demo: ## Run quantum VQE demo
	$(PYTHON) examples/quantum_vqe_molecule.py

neuromorphic-demo: ## Run neuromorphic ECG demo
	$(PYTHON) examples/neuromorphic_ecg.py

api-dev: ## Run API server in development mode
	uvicorn qbitalabs.api.main:app --reload --host 0.0.0.0 --port 8000

api-prod: ## Run API server in production mode
	uvicorn qbitalabs.api.main:app --host 0.0.0.0 --port 8000 --workers 4

notebook: ## Start Jupyter notebook server
	jupyter notebook notebooks/

benchmark: ## Run benchmarks
	$(PYTHON) scripts/benchmark_quantum.sh

setup-env: ## Setup development environment
	./scripts/setup_env.sh

ci: lint test ## Run CI pipeline locally
	@echo "$(GREEN)CI pipeline passed!$(NC)"

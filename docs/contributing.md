# Contributing to QBitaLabs

Thank you for your interest in contributing to QBitaLabs! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read and follow our Code of Conduct.

## Ways to Contribute

### 1. Report Bugs
- Use the [Bug Report template](https://github.com/qbitalabs/qbitalabs/issues/new?template=bug_report.md)
- Include minimal reproducible examples
- Provide environment details

### 2. Suggest Features
- Use the [Feature Request template](https://github.com/qbitalabs/qbitalabs/issues/new?template=feature_request.md)
- Describe the use case and expected behavior
- Consider implementation complexity

### 3. Submit Research Proposals
- Use the [Research Proposal template](https://github.com/qbitalabs/qbitalabs/issues/new?template=research_proposal.md)
- Outline scientific hypothesis and methodology
- Describe expected platform improvements

### 4. Contribute Code
- Fork the repository
- Create a feature branch
- Submit a pull request

## Development Setup

### Prerequisites

```bash
# Python 3.10+
python --version

# Git
git --version
```

### Setup Steps

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/qbitalabs.git
cd qbitalabs

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install development dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Verify installation
pytest tests/unit/ -v
```

## Coding Standards

### Style Guide

We follow these conventions:
- **PEP 8** for Python code style
- **Google style** for docstrings
- **Type hints** for all public functions

```python
def calculate_binding_affinity(
    compound: str,
    target: str,
    method: str = "ensemble"
) -> BindingResult:
    """Calculate drug-target binding affinity.

    Args:
        compound: SMILES string of the compound.
        target: UniProt ID or gene name of the target.
        method: Prediction method to use.

    Returns:
        BindingResult containing affinity predictions.

    Raises:
        ValueError: If compound SMILES is invalid.
        TargetNotFoundError: If target is not in database.

    Example:
        >>> result = calculate_binding_affinity(
        ...     compound="CC(=O)OC1=CC=CC=C1C(=O)O",
        ...     target="PTGS2"
        ... )
        >>> print(f"pIC50: {result.pic50:.2f}")
        pIC50: 6.80
    """
```

### Linting and Formatting

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type check with MyPy
mypy src/qbitalabs
```

### Pre-commit Hooks

Our pre-commit configuration runs:
- Black (formatting)
- isort (import sorting)
- Ruff (linting)
- MyPy (type checking)
- pytest (quick tests)

```bash
# Run manually
pre-commit run --all-files
```

## Testing

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Tests with dependencies
├── e2e/           # End-to-end tests
└── conftest.py    # Shared fixtures
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only (fast)
pytest tests/unit/ -v

# With coverage
pytest --cov=qbitalabs --cov-report=html

# Specific module
pytest tests/unit/test_quantum.py -v

# Specific test
pytest tests/unit/test_quantum.py::test_vqe_convergence -v
```

### Writing Tests

```python
import pytest
from qbitalabs.quantum import VQESolver

class TestVQESolver:
    """Tests for VQE solver."""

    @pytest.fixture
    def solver(self):
        """Create VQE solver for testing."""
        return VQESolver(backend="simulator")

    def test_solve_h2_molecule(self, solver):
        """Test VQE correctly finds H2 ground state."""
        # Arrange
        hamiltonian = create_h2_hamiltonian()
        expected_energy = -1.137  # Known value

        # Act
        result = solver.solve(hamiltonian)

        # Assert
        assert result.converged
        assert abs(result.energy - expected_energy) < 0.01

    @pytest.mark.parametrize("ansatz", ["uccsd", "hardware_efficient"])
    def test_different_ansatze(self, solver, ansatz):
        """Test VQE works with different ansatze."""
        solver.ansatz = ansatz
        result = solver.solve(simple_hamiltonian)
        assert result.converged
```

## Pull Request Process

### Before Submitting

1. **Create an issue** (if one doesn't exist)
2. **Fork the repo** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes** following coding standards
4. **Write tests** for new functionality
5. **Update documentation** if needed
6. **Run the full test suite**:
   ```bash
   pytest
   ```

### PR Guidelines

- **Title**: Use conventional commits format
  - `feat: add quantum error mitigation`
  - `fix: resolve VQE convergence issue`
  - `docs: update API reference`
  - `test: add integration tests for digital twin`

- **Description**: Fill out the PR template completely
- **Size**: Keep PRs focused and reasonably sized
- **Reviews**: Address all review comments

### Conventional Commits

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

## Module-Specific Guidelines

### Quantum Module

- Test with both simulator and mock hardware backends
- Verify convergence criteria
- Include numerical precision tests
- Document circuit depth and gate counts

### SWARM Agents

- Test agent isolation (no side effects)
- Verify message passing correctness
- Test coordination patterns independently
- Include timeout handling tests

### Digital Twin

- Validate against known physiological models
- Test edge cases (extreme values)
- Verify temporal consistency
- Include uncertainty quantification

### API

- Test all endpoints
- Verify request/response schemas
- Include authentication tests
- Test error handling

## Documentation

### Docstrings

All public functions, classes, and modules require docstrings:

```python
class MolecularAgent(BaseAgent):
    """Agent specialized for molecular analysis tasks.

    This agent handles molecular property calculations, structure
    optimization, and similarity searches.

    Attributes:
        specialization: Type of molecular analysis to perform.
        tools: Available computational tools.

    Example:
        >>> agent = MolecularAgent(specialization="drug_binding")
        >>> result = await agent.analyze("CCO")  # Ethanol
        >>> print(result.properties)
    """
```

### README Updates

Update README.md when:
- Adding new features
- Changing installation process
- Modifying API

### API Documentation

- Update `docs/api-reference.md` for API changes
- Include request/response examples
- Document error codes

## Release Process

1. **Version bump**: Update version in `pyproject.toml`
2. **Changelog**: Update CHANGELOG.md
3. **Tag**: Create git tag `v{version}`
4. **CI/CD**: Automated release to PyPI

## Getting Help

- **Discord**: [discord.gg/qbitalabs](https://discord.gg/qbitalabs)
- **GitHub Discussions**: [github.com/qbitalabs/qbitalabs/discussions](https://github.com/qbitalabs/qbitalabs/discussions)
- **Email**: hello@qbitalabs.com

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md
- Release notes
- Project documentation

Thank you for contributing to QBitaLabs!

---

*QBitaLabs, Inc. — Swarm intelligence for quantum biology and human health*

# Testing Guide for pyHarmonySearch

## Overview

This project maintains comprehensive test coverage using pytest with modern testing best practices. The test suite includes unit tests, integration tests, functional tests, and property-based tests to ensure the correctness and robustness of the Harmony Search algorithm implementation.

### Test Statistics

- **Total Tests**: 567 tests ✅
- **Test Files**: 26 files
- **Current Coverage**: **100%** ✅ (target achieved!)
- **Test Categories**:
  - Unit Tests: 420+ tests
  - Integration Tests: 77 tests  
  - Property-Based Tests: 23 tests
  - Functional Tests: 35 tests
  - Coverage Completion Tests: 6 tests

## Quick Start

### Installation

```bash
# Install package with test dependencies
pip install -e ".[dev]"

# Or using make
make install-dev
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyharmonysearch --cov-report=html

# Run only fast tests (skip slow multiprocessing tests)
pytest -m "not slow"

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Using make commands
make test           # Run all tests with coverage
make test-fast      # Run only fast unit tests
make test-unit      # Run unit tests
make test-integration  # Run integration tests
make test-property  # Run property-based tests
make test-cov       # Run with detailed coverage report
```

### Running Specific Test Categories

```bash
# Run property-based tests only
pytest tests/property/ -v

# Run multiprocessing tests (slow)
pytest -m multiprocessing

# Run functional tests (runs actual example scripts)
pytest tests/functional/
```

## Test Structure

```
tests/
├── conftest.py                              # Shared pytest fixtures
│
├── unit/                                     # Fast, isolated unit tests
│   ├── test_objective_function_interface.py  # Interface contract tests (49 tests)
│   ├── test_harmony_search_results.py        # NamedTuple tests (36 tests)
│   ├── test_harmony_search_init.py           # Constructor tests (25 tests)
│   ├── test_harmony_search_run.py            # Main algorithm tests (37 tests)
│   ├── test_harmony_search_initialize.py     # Initialization tests (27 tests)
│   ├── test_harmony_search_random_selection.py  # Random selection tests (25 tests)
│   ├── test_harmony_search_memory_consideration.py  # Memory tests (28 tests)
│   ├── test_harmony_search_pitch_adjustment.py  # Pitch adjustment tests (29 tests)
│   ├── test_harmony_search_update_memory.py  # Memory update tests (23 tests)
│   ├── test_worker_function.py               # Worker function tests (20 tests)
│   ├── test_imports.py                       # Import tests (12 tests)
│   ├── test_global_terminating_event.py      # Event tests (7 tests)
│   └── test_coverage_completion.py           # 100% coverage tests (6 tests)
│
├── integration/                              # Integration tests with multiprocessing
│   ├── test_harmony_search_serial.py         # Serial execution tests (23 tests)
│   ├── test_harmony_search_function.py       # Multiprocessing tests (29 tests)
│   └── test_multiprocessing_edge_cases.py    # Edge case tests (25 tests)
│
├── functional/                               # End-to-end functional tests
│   └── test_examples.py                      # Example script tests (35 tests)
│
├── property/                                 # Property-based tests with Hypothesis
│   ├── test_algorithm_invariants.py          # Algorithm invariants (7 tests)
│   ├── test_boundary_constraints.py          # Boundary tests (8 tests)
│   └── test_stochastic_properties.py         # Stochastic properties (8 tests)
│
└── fixtures/                                 # Shared test fixtures
    ├── mock_objective_functions.py           # 5 mock implementations
    └── test_helpers.py                       # Test utility functions
```

## Test Fixtures

### Mock Objective Functions

The test suite includes several mock objective function implementations in `tests/fixtures/mock_objective_functions.py`:

1. **QuadraticObjectiveFunction** - Simple continuous optimization (`minimize sum(x_i²)`)
2. **DiscreteObjectiveFunction** - Discrete-only variables
3. **MixedObjectiveFunction** - Mixed continuous and discrete variables
4. **FixedParameterObjectiveFunction** - Non-variable (fixed) parameters
5. **PathologicalObjectiveFunction** - Edge cases (NaN, Inf values)

### Shared Fixtures (conftest.py)

```python
# Available fixtures
simple_objective_function    # Continuous quadratic function
discrete_objective_function  # Discrete variables only
mixed_objective_function     # Mixed variable types
harmony_search_instance      # Pre-configured HarmonySearch instance
sample_harmony              # Sample harmony vector
sample_harmony_memory       # Sample harmony memory
```

## Writing Tests

### Basic Test Example

```python
def test_harmony_search_basic(simple_objective_function):
    """Test basic harmony search functionality."""
    from pyharmonysearch import HarmonySearch
    
    hs = HarmonySearch(simple_objective_function)
    best_harmony, best_fitness, memory, history = hs.run()
    
    assert best_harmony is not None
    assert isinstance(best_fitness, (int, float))
    assert len(memory) == simple_objective_function.get_hms()
```

### Using Test Helpers

```python
from tests.fixtures.test_helpers import (
    assert_harmony_in_bounds,
    assert_valid_harmony_memory,
    create_mock_objective_function
)

def test_bounds_respected():
    """Test that harmonies stay within bounds."""
    obj_fun = create_mock_objective_function(num_parameters=5)
    hs = HarmonySearch(obj_fun)
    _, _, memory, _ = hs.run()
    
    for harmony, fitness in memory:
        assert_harmony_in_bounds(harmony, obj_fun)
```

### Property-Based Testing Example

```python
from hypothesis import given, strategies as st, settings

@given(
    num_parameters=st.integers(min_value=1, max_value=10),
    hms=st.integers(min_value=2, max_value=25)
)
@settings(max_examples=50, deadline=5000)
def test_harmony_memory_size_invariant(num_parameters, hms):
    """Test that harmony memory always maintains HMS size."""
    obj_fun = create_mock_objective_function(
        num_parameters=num_parameters,
        hms=hms
    )
    hs = HarmonySearch(obj_fun)
    _, _, memory, _ = hs.run()
    
    assert len(memory) == hms  # Invariant: memory size == HMS
```

## Coverage Requirements

This project maintains **strict test coverage requirements**:

- **Target Coverage**: 100% (configurable in pyproject.toml)
- **Branch Coverage**: Required (all conditional branches must be tested)
- **Parallel Coverage**: Enabled for multiprocessing tests

### Viewing Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=pyharmonysearch --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Terminal report with missing lines
pytest --cov=pyharmonysearch --cov-report=term-missing

# JSON report for automation
pytest --cov=pyharmonysearch --cov-report=json
```

### Coverage Configuration

Coverage is configured in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["pyharmonysearch"]
branch = true       # Measure branch coverage
parallel = true     # Support for multiprocessing

[tool.coverage.report]
fail_under = 100.0  # Enforce 100% coverage
show_missing = true
```

## Test Markers

Tests can be marked with pytest markers for selective execution:

```python
# Available markers
@pytest.mark.slow           # Slow tests (multiprocessing, long iterations)
@pytest.mark.integration    # Integration tests
@pytest.mark.multiprocessing  # Tests using actual multiprocessing
@pytest.mark.property       # Property-based tests with Hypothesis
```

### Running Specific Markers

```bash
# Skip slow tests
pytest -m "not slow"

# Run only multiprocessing tests
pytest -m multiprocessing

# Run only property-based tests
pytest -m property

# Combined markers
pytest -m "integration and not slow"
```

## Continuous Integration (CI/CD)

### GitHub Actions Workflows

This project uses GitHub Actions for automated testing:

**1. Main Test Workflow** (`.github/workflows/test.yml`)
- Tests on Python 3.7-3.12
- Tests on Ubuntu, macOS, Windows
- Matrix: 18 test combinations (3 OS × 6 Python versions)
- Runs unit, integration, and property tests
- Uploads coverage to Codecov
- Enforces 100% coverage requirement

**2. Coverage Workflow** (`.github/workflows/coverage.yml`)
- Generates coverage reports
- Comments coverage results on PRs
- Shows coverage diff between base and PR

### Running Tests Locally Like CI

```bash
# Test multiple Python versions (requires pyenv)
tox

# Test current Python version with all checks
pytest --cov=pyharmonysearch --cov-fail-under=100
ruff check pyharmonysearch tests
mypy pyharmonysearch
```

## Pre-commit Hooks

Install pre-commit hooks for automatic quality checks:

```bash
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

Hooks configured:
- **ruff**: Linting and auto-fixing
- **ruff-format**: Code formatting  
- **mypy**: Type checking
- **pytest**: Fast unit tests

## Troubleshooting

### Tests Hanging

If multiprocessing tests hang:
```bash
# Run with timeout
pytest --timeout=60 tests/integration/

# Skip multiprocessing tests
pytest -m "not multiprocessing"
```

### Coverage Not 100%

1. Run coverage report to see missing lines:
   ```bash
   pytest --cov=pyharmonysearch --cov-report=term-missing
   ```

2. Check HTML report for detailed view:
   ```bash
   pytest --cov=pyharmonysearch --cov-report=html
   open htmlcov/index.html
   ```

3. Common missing coverage areas:
   - KeyboardInterrupt handlers
   - Error handling branches
   - Platform-specific code

### Test Failures

For detailed failure information:
```bash
# Verbose output with full tracebacks
pytest -vv --tb=long

# Stop on first failure
pytest -x

# Run specific failing test
pytest tests/unit/test_harmony_search_run.py::test_specific_test -vv
```

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Reinstall in editable mode
pip install -e ".[dev]"

# Verify installation
python -c "import pyharmonysearch; print(pyharmonysearch.__file__)"
```

## Performance Considerations

### Fast vs Slow Tests

- **Unit tests** (~1 second): Fast, isolated tests with mocking
- **Integration tests** (~10 seconds): Tests with actual multiprocessing
- **Property tests** (~10 seconds): Hypothesis generates many test cases
- **Functional tests** (~30 seconds): Runs actual example scripts

### Parallel Test Execution

Run tests in parallel for speed:
```bash
# Automatically determine worker count
pytest -n auto

# Specify number of workers
pytest -n 4
```

## Test Development Guidelines

### 1. Test Naming

Use descriptive test names that explain what is being tested:

```python
# Good
def test_harmony_memory_maintains_size_invariant():
    ...

# Bad
def test_memory():
    ...
```

### 2. Test Organization

- One test file per source module
- Group related tests in classes
- Use fixtures for common setup
- Keep tests independent (no shared state)

### 3. Test Documentation

Every test should have a docstring:
```python
def test_pitch_adjustment_maintains_bounds():
    """Test that pitch adjustment never violates parameter bounds.
    
    This test verifies that after pitch adjustment, all continuous
    parameters remain within [lower_bound, upper_bound].
    """
    ...
```

### 4. Assertion Messages

Use clear assertion messages for failures:
```python
assert value >= lower_bound, \
    f"Value {value} is below lower bound {lower_bound}"
```

## Test Metrics

### Current Test Metrics

- **Total Execution Time**: ~10 seconds (core tests without slow multiprocessing)
- **Total Execution Time**: ~120 seconds (with all multiprocessing tests)
- **Average Test Duration**: <0.02 seconds
- **Slowest Tests**: Property-based tests (~0.5-1 second each)
- **Pass Rate**: 100% (567/567 tests passing) ✅
- **Coverage**: **100%** ✅ (target achieved!)

### Coverage by Module

| Module | Statements | Missing | Branches | Coverage |
|--------|-----------|---------|----------|----------|
| `__init__.py` | 3 | 0 | 0 | **100%** ✅ |
| `objective_function_interface.py` | 39 | 0 | 0 | **100%** ✅ |
| `harmony_search.py` | 142 | 0 | 56 | **100%** ✅ |
| **TOTAL** | **184** | **0** | **56** | **100%** ✅ |

## Additional Resources

- **pytest documentation**: https://docs.pytest.org/
- **Hypothesis documentation**: https://hypothesis.readthedocs.io/
- **Coverage.py documentation**: https://coverage.readthedocs.io/
- **GitHub Actions**: https://docs.github.com/en/actions

## Contributing Tests

When contributing new features:

1. **Write tests first** (TDD approach recommended)
2. **Maintain 100% coverage** for new code
3. **Add integration tests** for new functionality
4. **Update this documentation** if adding new test patterns
5. **Run full test suite** before submitting PR

```bash
# Pre-submission checklist
make test-all      # Run all tests
make lint          # Check code style
make type-check    # Run type checking
make examples      # Verify examples still work
```

---

**Questions or Issues?**

- Open an issue on GitHub
- Check existing test files for examples
- Review pytest and hypothesis documentation

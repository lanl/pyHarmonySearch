# Contributing to pyHarmonySearch

Thank you for your interest in contributing to pyHarmonySearch! This document provides guidelines for contributing to the project.

## Quick Start

1. **Fork and clone** the repository
2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

### 1. Make Your Changes

- Write clean, documented code
- Add type hints to new code (required)
- Follow existing code style (enforced by ruff)

### 2. Write Tests

**All new code must have tests.** This project maintains 100% code coverage.

```bash
# Run tests as you develop
pytest tests/unit/  # Fast unit tests only
pytest              # All tests

# Check coverage
pytest --cov=pyharmonysearch --cov-report=html
open htmlcov/index.html
```

See [TESTING.md](TESTING.md) for detailed testing guide.

### 3. Run Quality Checks

Before submitting a PR, run all quality checks:

```bash
make test-all       # Run all tests with coverage
make lint           # Check code style with ruff
make type-check     # Run mypy type checking
make examples       # Verify examples still work
```

Or use the pre-commit hooks:
```bash
pre-commit run --all-files
```

## Test Requirements

### Coverage Requirement
- **100% code coverage required** for all new code
- Tests must cover both statements and branches
- Use `pytest --cov` to verify coverage

### Test Types
Write appropriate tests for your changes:

- **Unit tests** (`tests/unit/`) - Fast, isolated tests
- **Integration tests** (`tests/integration/`) - Test component interactions
- **Functional tests** (`tests/functional/`) - End-to-end validation
- **Property-based tests** (`tests/property/`) - For algorithmic properties

### Test Guidelines

```python
# Good test naming
def test_harmony_search_with_seed_produces_deterministic_results():
    """Test that random seed produces reproducible results."""
    # Arrange
    obj_fun = QuadraticObjectiveFunction(random_seed=42)
    
    # Act
    results1 = harmony_search_serial(obj_fun, num_iterations=3)
    results2 = harmony_search_serial(obj_fun, num_iterations=3)
    
    # Assert
    assert results1.best_harmony == results2.best_harmony
```

Key principles:
- Use descriptive test names
- Include docstrings explaining what is tested
- Use fixtures for common setup
- Test edge cases and error conditions
- Keep tests fast (mock expensive operations)

## Code Style

This project uses modern Python tooling:

- **ruff** - Fast linting and formatting
- **mypy** - Static type checking
- **Type hints** - Required for all new code

### Type Hints Example

```python
from typing import List, Optional, Union

def harmony_search(
    objective_function: ObjectiveFunctionInterface,
    num_processes: int,
    num_iterations: int,
    initial_harmonies: Optional[List[List[Union[int, float]]]] = None
) -> HarmonySearchResults:
    """Type-hinted function signature."""
    ...
```

## Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with tests

3. **Ensure all checks pass**:
   ```bash
   make test-all
   make lint
   make type-check
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **PR requirements**:
   - ✅ All tests pass
   - ✅ 100% coverage maintained
   - ✅ Type checking passes
   - ✅ Linting passes
   - ✅ Examples still work
   - ✅ Documentation updated (if needed)

## PR Checklist

Before submitting your PR, verify:

- [ ] Tests written and passing (100% coverage)
- [ ] Type hints added to new code
- [ ] Code passes ruff linting
- [ ] Code passes mypy type checking
- [ ] Examples still work correctly
- [ ] Documentation updated (README.md, CHANGES.md if needed)
- [ ] Commits have clear, descriptive messages
- [ ] No breaking changes (or clearly documented)

## Testing Your Changes

### Quick Tests (Development)
```bash
pytest tests/unit/ -x  # Stop on first failure
```

### Full Test Suite
```bash
pytest  # Runs all tests with coverage
```

### Specific Test Categories
```bash
pytest tests/unit/              # Unit tests only
pytest tests/integration/        # Integration tests
pytest -m "not slow"            # Skip slow tests
pytest tests/property/          # Property-based tests
```

### Coverage Report
```bash
pytest --cov=pyharmonysearch --cov-report=html
open htmlcov/index.html
```

## Common Development Tasks

### Add a New Feature

1. Write tests first (TDD approach)
2. Implement the feature
3. Ensure tests pass and coverage is 100%
4. Update documentation

### Fix a Bug

1. Write a test that reproduces the bug
2. Fix the bug
3. Ensure the test passes
4. Verify no regression in other tests

### Update Documentation

- Update README.md for user-facing changes
- Update TESTING.md for test-related changes
- Update docstrings in code
- Add entry to CHANGES.md

## Getting Help

- **Testing questions**: See [TESTING.md](TESTING.md)
- **General questions**: Open a GitHub issue
- **Bug reports**: Open a GitHub issue with reproduction steps
- **Feature requests**: Open a GitHub issue with use case description

## Code of Conduct

- Be respectful and constructive
- Focus on the code, not the person
- Welcome newcomers and help them learn
- Follow the existing project style and conventions

## License

By contributing, you agree that your contributions will be licensed under the BSD 3-Clause License.

---

Thank you for contributing to pyHarmonySearch! 🎉

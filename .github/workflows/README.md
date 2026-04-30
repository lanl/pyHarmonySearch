# GitHub Actions CI/CD Workflows

This directory contains the automated testing and coverage workflows for pyHarmonySearch.

## Workflows

### 1. test.yml - Main Test Workflow

**Triggers:**
- Push to `main` or `master` branches
- Pull requests to `main` or `master` branches
- Manual workflow dispatch

**Jobs:**

#### Test Job
- **Matrix Testing:** Tests across all supported Python versions (3.7-3.12) and operating systems (Ubuntu, macOS, Windows)
- **Test Execution:** Runs pytest with coverage reporting
- **Coverage Verification:** Ensures 100% test coverage
- **Codecov Upload:** Uploads coverage data to Codecov for tracking
- **Artifacts:** Uploads HTML coverage reports for the latest Python on Ubuntu

#### Quality Job
- **Ruff Linting:** Checks code quality and style with ruff
- **Ruff Formatting:** Verifies code formatting consistency
- **Mypy Type Checking:** Performs static type analysis (continues on error)

#### Examples Job
- **Matrix Testing:** Runs all 5 example scripts individually
- **Timeout Protection:** Each example has a 5-minute timeout
- **Success Verification:** Ensures all examples complete without errors

#### Test Summary Job
- Aggregates results from all jobs
- Fails if any job fails
- Provides a single status check for the entire workflow

### 2. coverage.yml - Coverage Reporting Workflow

**Triggers:**
- Push to `main` or `master` branches
- Pull requests to `main` or `master` branches
- Manual workflow dispatch

**Features:**

#### Coverage Report Generation
- Runs full test suite with multiple coverage report formats (XML, JSON, HTML, terminal)
- Calculates total coverage percentage
- Uploads to Codecov

#### GitHub Integration
- **Summary:** Creates detailed coverage summary in GitHub Actions UI
- **PR Comments:** Posts/updates coverage report comments on pull requests
- **Coverage Diff:** Compares coverage between base branch and PR (PR only)

#### Coverage Artifacts
- Uploads coverage reports (XML, JSON, HTML) as artifacts
- Retains artifacts for 30 days
- Useful for debugging and detailed analysis

#### Threshold Enforcement
- Fails if coverage drops below 100%
- Ensures high code quality standards

## Setup Requirements

### Codecov Integration (Optional)

To enable Codecov uploads:

1. Sign up at [codecov.io](https://codecov.io)
2. Add your repository
3. Add `CODECOV_TOKEN` to your repository secrets:
   - Go to repository Settings → Secrets and variables → Actions
   - Add new secret named `CODECOV_TOKEN`
   - Paste your Codecov token

The workflows will continue to work without Codecov, but uploads will be skipped.

### Required Dependencies

The workflows automatically install these dependencies:

**Test Dependencies:**
- pytest
- pytest-cov
- pytest-xdist
- coverage[toml]

**Quality Dependencies:**
- ruff
- mypy

## Configuration Files

- **pyproject.toml:** Modern Python project configuration (ruff, mypy, pytest, coverage)
- **setup.cfg:** Legacy pytest and coverage configuration (backup)

## Local Testing

Run the same checks locally before pushing:

```bash
# Install dev dependencies
pip install pytest pytest-cov pytest-xdist ruff mypy

# Run tests with coverage
pytest --cov=pyharmonysearch --cov-report=term --cov-report=html -v

# Check coverage threshold
coverage report --fail-under=100

# Run linting
ruff check pyharmonysearch/ examples/

# Check formatting
ruff format --check pyharmonysearch/ examples/

# Run type checking
mypy pyharmonysearch/ --ignore-missing-imports

# Run examples
python examples/2-D_continuous_seed.py
python examples/2-D_discrete_x.py
python examples/2-D_continuous_fixed_x.py
python examples/2-D_continuous_specified_initial_harmonies.py
python examples/5-D_linear_system.py
```

## Workflow Status

Add badges to your README.md:

```markdown
[![Test](https://github.com/lanl/pyHarmonySearch/actions/workflows/test.yml/badge.svg)](https://github.com/lanl/pyHarmonySearch/actions/workflows/test.yml)
[![Coverage](https://github.com/lanl/pyHarmonySearch/actions/workflows/coverage.yml/badge.svg)](https://github.com/lanl/pyHarmonySearch/actions/workflows/coverage.yml)
[![codecov](https://codecov.io/gh/lanl/pyHarmonySearch/branch/main/graph/badge.svg)](https://codecov.io/gh/lanl/pyHarmonySearch)
```

## Troubleshooting

### Workflow Fails

1. Check the workflow run logs in GitHub Actions tab
2. Look for specific job that failed (test, quality, examples)
3. Review error messages and stack traces

### Coverage Below 100%

1. Run tests locally: `pytest --cov=pyharmonysearch --cov-report=term-missing -v`
2. Review missing lines in coverage report
3. Add tests for uncovered code
4. Re-run workflow

### Example Scripts Fail

1. Check if example script runs locally
2. Review timeout settings (currently 5 minutes per example)
3. Check for platform-specific issues in matrix

### Ruff/Mypy Errors

1. Run tools locally to see full output
2. Fix code style issues: `ruff check --fix pyharmonysearch/`
3. Format code: `ruff format pyharmonysearch/`
4. Note: mypy errors don't fail the workflow (continue-on-error: true)

## Modern GitHub Actions Features

These workflows use the latest GitHub Actions features:

- **actions/checkout@v4:** Latest checkout action
- **actions/setup-python@v5:** Latest Python setup with pip caching
- **actions/upload-artifact@v4:** Latest artifact upload
- **actions/github-script@v7:** For PR comment automation
- **codecov/codecov-action@v4:** Latest Codecov integration
- **Matrix strategy:** Parallel testing across multiple configurations
- **Job dependencies:** Proper workflow orchestration
- **Step summaries:** Rich output in GitHub UI
- **Workflow dispatch:** Manual trigger support

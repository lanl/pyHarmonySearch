"""Functional tests for example scripts.

These tests verify that all example scripts in the examples/ directory
run successfully and produce expected output.
"""

import subprocess
import sys
from pathlib import Path

import pytest


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / 'examples'

# All example scripts to test
EXAMPLE_SCRIPTS = [
    '2-D_continuous_seed.py',
    '2-D_continuous_specified_initial_harmonies.py',
    '2-D_continuous_fixed_x.py',
    '2-D_discrete_x.py',
    '5-D_linear_system.py',
]

# Timeout for each example script (in seconds)
# The 5-D linear system example takes significantly longer due to 500k improvisations
EXAMPLE_TIMEOUT = 60
SLOW_EXAMPLE_TIMEOUT = 300  # 5 minutes for slow examples

# Examples that are known to be slow
SLOW_EXAMPLES = {'5-D_linear_system.py'}


def _get_timeout(script_name: str) -> int:
    """Get appropriate timeout for a script.

    Args:
        script_name: Name of the example script.

    Returns:
        Timeout in seconds.
    """
    return SLOW_EXAMPLE_TIMEOUT if script_name in SLOW_EXAMPLES else EXAMPLE_TIMEOUT


class TestExampleScripts:
    """Test suite for verifying example scripts run successfully."""

    @pytest.mark.parametrize('script_name', EXAMPLE_SCRIPTS)
    @pytest.mark.slow
    def test_example_runs_successfully(self, script_name: str) -> None:
        """Test that each example script runs without errors.

        Args:
            script_name: Name of the example script to test.
        """
        script_path = EXAMPLES_DIR / script_name

        # Run the example script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=_get_timeout(script_name),
            cwd=str(EXAMPLES_DIR),
        )

        # Assert the script ran successfully
        assert result.returncode == 0, (
            f'Script {script_name} failed with return code {result.returncode}\n'
            f'STDOUT:\n{result.stdout}\n'
            f'STDERR:\n{result.stderr}'
        )

        # Assert there's some output
        assert result.stdout, f'Script {script_name} produced no output'

    @pytest.mark.parametrize('script_name', EXAMPLE_SCRIPTS)
    @pytest.mark.slow
    def test_example_output_format(self, script_name: str) -> None:
        """Test that each example script produces expected output format.

        Args:
            script_name: Name of the example script to test.
        """
        script_path = EXAMPLES_DIR / script_name

        # Run the example script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=_get_timeout(script_name),
            cwd=str(EXAMPLES_DIR),
        )

        assert result.returncode == 0, f'Script {script_name} failed'

        # Check for expected output format
        output = result.stdout
        assert 'Elapsed time:' in output, f'Script {script_name} missing "Elapsed time" in output'
        assert 'Best harmony:' in output, f'Script {script_name} missing "Best harmony" in output'
        assert 'Best fitness:' in output, f'Script {script_name} missing "Best fitness" in output'

    @pytest.mark.parametrize('script_name', EXAMPLE_SCRIPTS)
    @pytest.mark.slow
    def test_example_completes_within_timeout(self, script_name: str) -> None:
        """Test that each example script completes within reasonable time.

        Args:
            script_name: Name of the example script to test.
        """
        script_path = EXAMPLES_DIR / script_name

        # This test will raise subprocess.TimeoutExpired if it takes too long
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=_get_timeout(script_name),
            cwd=str(EXAMPLES_DIR),
        )

        assert result.returncode == 0, f'Script {script_name} failed'

    def test_seeded_example_produces_deterministic_results(self) -> None:
        """Test that the seeded example produces identical results across multiple runs."""
        script_name = '2-D_continuous_seed.py'
        script_path = EXAMPLES_DIR / script_name

        # Run the example 3 times
        results = []
        for _ in range(3):
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=_get_timeout(script_name),
                cwd=str(EXAMPLES_DIR),
            )

            assert result.returncode == 0, f'Script {script_name} failed on run {_ + 1}'

            # Parse output to extract harmony and fitness (ignore elapsed time)
            output = result.stdout.strip()
            harmony = None
            fitness = None
            for line in output.split('\n'):
                if 'Best harmony:' in line:
                    harmony = line.split('Best harmony:')[1].strip()
                if 'Best fitness:' in line:
                    fitness = line.split('Best fitness:')[1].strip()

            assert harmony is not None, f'Could not find Best harmony in output: {output}'
            assert fitness is not None, f'Could not find Best fitness in output: {output}'

            results.append((harmony, fitness))

        # All results should be identical (deterministic with seed)
        # We compare harmony and fitness, but not elapsed time (which will vary)
        assert results[0] == results[1], (
            'Seeded example produced different results on runs 1 and 2:\n'
            f'Run 1: harmony={results[0][0]}, fitness={results[0][1]}\n'
            f'Run 2: harmony={results[1][0]}, fitness={results[1][1]}'
        )
        assert results[1] == results[2], (
            'Seeded example produced different results on runs 2 and 3:\n'
            f'Run 2: harmony={results[1][0]}, fitness={results[1][1]}\n'
            f'Run 3: harmony={results[2][0]}, fitness={results[2][1]}'
        )

    @pytest.mark.parametrize('script_name', EXAMPLE_SCRIPTS)
    @pytest.mark.slow
    def test_example_no_stderr_errors(self, script_name: str) -> None:
        """Test that examples don't produce errors on stderr.

        Args:
            script_name: Name of the example script to test.
        """
        script_path = EXAMPLES_DIR / script_name

        # Run the example script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=_get_timeout(script_name),
            cwd=str(EXAMPLES_DIR),
        )

        # Check that stderr is empty or only contains warnings (not errors)
        if result.stderr:
            stderr_lower = result.stderr.lower()
            # It's okay to have warnings, but not errors or exceptions
            assert 'error' not in stderr_lower, f'Script {script_name} produced errors:\n{result.stderr}'
            assert 'exception' not in stderr_lower, f'Script {script_name} raised exceptions:\n{result.stderr}'
            assert 'traceback' not in stderr_lower, f'Script {script_name} had traceback:\n{result.stderr}'

    def test_all_example_scripts_exist(self) -> None:
        """Test that all expected example scripts exist in the examples directory."""
        for script_name in EXAMPLE_SCRIPTS:
            script_path = EXAMPLES_DIR / script_name
            assert script_path.exists(), f'Example script {script_name} does not exist at {script_path}'
            assert script_path.is_file(), f'Example script {script_name} is not a file'

    def test_example_scripts_are_executable(self) -> None:
        """Test that example scripts have execute permissions (Unix-like systems)."""
        import stat

        for script_name in EXAMPLE_SCRIPTS:
            script_path = EXAMPLES_DIR / script_name
            file_stat = script_path.stat()
            is_executable = bool(file_stat.st_mode & stat.S_IXUSR)

            # Note: On some systems, Python scripts may not be executable
            # This test checks if they are, but it's informational
            if not is_executable:
                pytest.skip(f'Script {script_name} is not executable (this may be expected on some systems)')

    def test_example_scripts_have_shebang(self) -> None:
        """Test that example scripts have proper shebang line."""
        for script_name in EXAMPLE_SCRIPTS:
            script_path = EXAMPLES_DIR / script_name

            with open(script_path) as f:
                first_line = f.readline().strip()

            assert first_line.startswith('#!'), f'Script {script_name} missing shebang line'
            assert 'python' in first_line.lower(), f'Script {script_name} shebang does not reference Python'

    @pytest.mark.parametrize('script_name', EXAMPLE_SCRIPTS)
    @pytest.mark.slow
    def test_example_output_contains_valid_fitness(self, script_name: str) -> None:
        """Test that example output contains valid numeric fitness value.

        Args:
            script_name: Name of the example script to test.
        """
        script_path = EXAMPLES_DIR / script_name

        # Run the example script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=_get_timeout(script_name),
            cwd=str(EXAMPLES_DIR),
        )

        assert result.returncode == 0, f'Script {script_name} failed'

        # Parse the output to find the fitness value
        output = result.stdout
        for line in output.split('\n'):
            if 'Best fitness:' in line:
                # Extract the fitness value (everything after "Best fitness:")
                fitness_str = line.split('Best fitness:')[1].strip()
                # Try to convert to float
                try:
                    fitness_value = float(fitness_str)
                    # Check that it's a valid number (not NaN or Inf)
                    assert not (fitness_value != fitness_value), f'Fitness is NaN in {script_name}'  # noqa: PLR0124
                    assert fitness_value not in (float('inf'), float('-inf')), (
                        f'Fitness is infinite in {script_name}'
                    )
                except ValueError:
                    pytest.fail(f'Could not parse fitness value "{fitness_str}" as float in {script_name}')
                break
        else:
            pytest.fail(f'Could not find "Best fitness:" line in output of {script_name}')

    @pytest.mark.parametrize('script_name', EXAMPLE_SCRIPTS)
    @pytest.mark.slow
    def test_example_output_contains_valid_harmony(self, script_name: str) -> None:
        """Test that example output contains valid harmony (list of numbers).

        Args:
            script_name: Name of the example script to test.
        """
        script_path = EXAMPLES_DIR / script_name

        # Run the example script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=_get_timeout(script_name),
            cwd=str(EXAMPLES_DIR),
        )

        assert result.returncode == 0, f'Script {script_name} failed'

        # Parse the output to find the harmony value
        output = result.stdout
        for line in output.split('\n'):
            if 'Best harmony:' in line:
                # Extract the harmony value (everything after "Best harmony:")
                harmony_str = line.split('Best harmony:')[1].strip()
                # Try to evaluate as a list
                try:
                    # Safety check: ensure it looks like a list
                    assert harmony_str.startswith('[') and harmony_str.endswith(']'), (
                        f'Harmony does not look like a list in {script_name}'
                    )
                    # Use eval safely (we control the input - it's our own example scripts)
                    harmony_value = eval(harmony_str)  # noqa: S307
                    # Check that it's a list
                    assert isinstance(harmony_value, list), f'Harmony is not a list in {script_name}'
                    # Check that all elements are numbers
                    for val in harmony_value:
                        assert isinstance(val, (int, float)), (
                            f'Harmony contains non-numeric value in {script_name}'
                        )
                except (ValueError, SyntaxError) as e:
                    pytest.fail(f'Could not parse harmony value "{harmony_str}" in {script_name}: {e}')
                break
        else:
            pytest.fail(f'Could not find "Best harmony:" line in output of {script_name}')


class TestExampleScriptCount:
    """Test to ensure we're testing all expected example scripts."""

    def test_example_script_count(self) -> None:
        """Test that we have the expected number of example scripts."""
        # Count actual Python files in examples directory
        actual_scripts = list(EXAMPLES_DIR.glob('*.py'))

        # We expect exactly 5 example scripts
        assert len(EXAMPLE_SCRIPTS) == 5, f'Expected 5 example scripts in test list, found {len(EXAMPLE_SCRIPTS)}'

        # Check that all actual scripts are in our test list
        actual_script_names = {script.name for script in actual_scripts}
        test_script_names = set(EXAMPLE_SCRIPTS)

        missing_from_tests = actual_script_names - test_script_names
        if missing_from_tests:
            pytest.fail(f'Example scripts not covered in tests: {missing_from_tests}')

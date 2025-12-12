import os
import sys
import textwrap
from pathlib import Path

import pytest

# Import the functions to test
from tools.toolkit.builtin.code_tools import run_python_file, run_pytest_tests


def test_run_python_file_success(tmp_path: Path):
    # Create a temporary python script that prints to stdout and stderr
    script_path = tmp_path / "script.py"
    script_content = textwrap.dedent("""
    import sys
    print('Hello stdout')
    print('Hello stderr', file=sys.stderr)
    """)
    script_path.write_text(script_content)

    result = run_python_file(str(script_path))
    assert result["success"] is True
    # The result should contain both stdout and stderr messages
    output = result["result"]
    assert "Hello stdout" in output
    assert "Hello stderr" in output


def test_run_python_file_not_found(tmp_path: Path):
    missing_path = tmp_path / "nonexistent.py"
    result = run_python_file(str(missing_path))
    assert result["success"] is False
    assert "File not found" in result["error"]


def test_run_pytest_tests_success(tmp_path: Path):
    # Create a temporary directory with a simple pytest test file
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_dummy.py"
    test_file.write_text(textwrap.dedent("""
    def test_pass():
        assert 1 + 1 == 2
    """))

    result = run_pytest_tests(str(test_dir))
    assert result["success"] is True
    output = result["result"]
    # Ensure pytest ran and reported the passed test
    assert "1 passed" in output


def test_run_pytest_tests_not_directory(tmp_path: Path):
    # Pass a file path instead of a directory
    test_file = tmp_path / "some_file.txt"
    test_file.write_text("just a file")
    result = run_pytest_tests(str(test_file))
    assert result["success"] is False
    assert "Not a directory" in result["error"]

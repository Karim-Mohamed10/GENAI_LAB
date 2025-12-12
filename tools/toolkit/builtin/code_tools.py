from tools.decorator import tool
from pathlib import Path
import subprocess

@tool()
def run_python_file(file_path: str) -> dict:
    """
    Run a Python file and return its stdout and stderr.
    Returns output as a dictionary with success/error status and result/message.
    """
    try:
        # TODO:
        p = Path(file_path)
        raise NotImplementedError()
        return {"success": True, "result": output.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool()
def run_pytest_tests(directory: str = ".") -> dict:
    """
    Run pytest in the given directory and return its output.
    Returns output as a dictionary with success/error status and result/message.
    """
    try:
        # TODO:
        p = Path(directory)
        raise NotImplementedError()
        return {"success": True, "result": output.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}

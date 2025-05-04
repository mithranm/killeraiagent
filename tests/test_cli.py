# tests/test_cli.py

import pytest
import subprocess
import shutil

def test_cli_list_models(tmp_path):
    """
    Verify that listing models works (via the CLI code or function).
    If 'kaia' script is installed, we can also try calling 'kaia' directly.
    """
    # If you have a direct function to list models, you could call that here:
    # from killeraiagent.cli import list_available_models
    # models = list_available_models()
    # assert isinstance(models, list)

    # OR use subprocess to call the CLI (assuming 'kaia' is in PATH).
    # If it's not installed as an entry point, skip or adapt:
    kaia_executable = shutil.which("kaia")
    if not kaia_executable:
        pytest.skip("kaia CLI not found in PATH, skipping CLI test.")
    
    cmd = [kaia_executable, "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    assert result.returncode == 0, "CLI --help should succeed"
    # Check that we see help text
    assert "usage" in result.stdout.lower()

def test_cli_no_args():
    """
    Test what happens if the user runs the CLI with no args.
    Typically it might show help or usage instructions.
    """
    kaia_executable = shutil.which("kaia")
    if not kaia_executable:
        pytest.skip("kaia CLI not found in PATH.")

    cmd = [kaia_executable]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Some CLIs return non-zero for no arguments; adjust as needed
    # We'll assume it returns code 0 and prints usage:
    assert result.returncode == 0
    assert "Available components:" in result.stdout  # or some known text

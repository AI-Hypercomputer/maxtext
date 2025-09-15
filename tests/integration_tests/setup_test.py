#!/usr/bin/env python3

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Integration test for the setup.sh script's primary installation modes.

This test validates the main installation branches of setup.sh by:
1.  Mocking all external/destructive commands (apt-get, python3, uv, gsutil).
2.  Placing these mocks on a temporary PATH.
3.  Running the real setup.sh script with the four critical mode/device combinations.
4.  Asserting that the script's logic called the correct set of mock commands
    for each specified mode.
"""

import os
import subprocess
import tempfile
import shutil
import shlex
import pytest
from contextlib import contextmanager

# Path to the setup.sh script, assuming test is run from repo root.
SETUP_SCRIPT_PATH = "./setup.sh"

# List of all commands setup.sh might call that we must mock.
COMMANDS_TO_MOCK = [
    "python3",
]


def create_mock_command(bin_path, command_name, log_path):
  """Creates a mock shell script that logs its arguments to a file."""
  script_path = os.path.join(bin_path, command_name)
  safe_log_path = shlex.quote(log_path)

  # The real setup.sh script calls python3 -c 'import sys...'
  # Our mock python3 MUST pass this check, or the script will exit early.
  if command_name == "python3":
    content = f"""#!/bin/bash
    if [[ "$*" == *"-c 'import sys; assert sys.version_info >= (3, 12)'"* ]]; then
    exit 0  # Pass the version check
    else
    # Log all other calls (like uv installs)
    echo "MOCK_CALLED: {command_name} $*" >> {safe_log_path}
    fi
    """
  elif command_name == "lsb_release":
    # Provide fake output for 'lsb_release -c -s'
    content = f"""#!/bin/bash
    echo "jammy"
    """
  elif command_name == "command":
    # Mock 'command -v' to fail, forcing the script to run 'pip install uv'
    content = f"""#!/bin/bash
    exit 1
    """
  else:
    # Default mock for all other commands
    content = f"""#!/bin/bash
    echo "MOCK_CALLED: {command_name} $*" >> {safe_log_path}
    exit 0
    """

  with open(script_path, "w") as f:
    f.write(content)
  os.chmod(script_path, 0o755)  # Make it executable


@contextmanager
def setup_mock_environment():
  """
  Context manager to create a temporary environment with all commands mocked.
  
  Sets up a temp directory, creates a 'bin' subdir, populates it with
  mocks for all external commands, and prepends this 'bin' to the PATH.
  Cleans up everything on exit.
  """
  temp_dir = tempfile.mkdtemp()
  mock_bin_dir = os.path.join(temp_dir, "bin")
  command_log_path = os.path.join(temp_dir, "command.log")
  os.makedirs(mock_bin_dir, exist_ok=True)

  try:
    # Create all mock executables
    for cmd in COMMANDS_TO_MOCK:
      create_mock_command(mock_bin_dir, cmd, command_log_path)

    # Set the new PATH environment variable
    test_env = os.environ.copy()
    test_env["PATH"] = f"{mock_bin_dir}:{test_env.get('PATH', '')}"
    
    # Yield the path to the log file and the env
    yield command_log_path, test_env

  finally:
    # Cleanup: Remove the temporary directory
    if os.path.exists(temp_dir):
      shutil.rmtree(temp_dir)


def run_setup_script(env, mode, device):
  """Helper to run the setup.sh script and return its captured output."""
  # Ensure the real script is executable
  os.chmod(SETUP_SCRIPT_PATH, 0o755)

  # We pipe 'yes' to stdin to auto-accept any potential sudo prompts
  # inside the (sudo bash || bash) block.
  process = subprocess.Popen(
      ["/usr/bin/env", "bash", SETUP_SCRIPT_PATH, f"MODE={mode}", f"DEVICE={device}"],
      env=env,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
  )
  stdout, stderr = process.communicate(input="y\n")
  return stdout, stderr, process.returncode


def read_log_file(log_path):
  """Reads the command log and returns its content as a single string."""
  if not os.path.exists(log_path):
    return ""
  with open(log_path, "r") as f:
    return f.read()


@pytest.mark.integration_test
@pytest.mark.tpu_only
def test_stable_tpu_setup():
  """
  Test setup stable mode on TPU device
  """
  mode, device = "stable", "tpu"
  print(f"Testing MODE={mode} DEVICE={device}...")
  
  with setup_mock_environment() as (log_path, test_env):
    stdout, stderr, code = run_setup_script(test_env, mode=mode, device=device)

    # assert code == 0, f"setup.sh failed with exit code {code}. Stderr:\n{stderr}"
    log_content = read_log_file(log_path)
    
    breakpoint()
    
    # Base assertions: Verify core system and python setup ran
    assert "MOCK_CALLED: sudo apt update" in log_content
    assert "MOCK_CALLED: python3 -m uv pip install -U setuptools wheel uv" in log_content
    assert "MOCK_CALLED: python3 -m uv pip install --no-cache-dir -U -r requirements.txt" in log_content


    assert "Installing stable jax, jaxlib for tpu" in stdout
    stable_cmd = "MOCK_CALLED: python3 -m uv pip install 'jax[tpu]>0.4' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
    assert stable_cmd in log_content
    assert "jax-cuda12-plugin" not in log_content # Should not install GPU pkgs


@pytest.mark.integration_test
@pytest.mark.gpu_only
def test_stable_gpu_setup():
  """
  Test setup stable mode on GPU device
  """
  mode, device = "stable", "gpu"
  print(f"Testing MODE={mode} DEVICE={device}...")
  
  with setup_mock_environment() as (log_path, test_env):
    stdout, stderr, code = run_setup_script(test_env, mode=mode, device=device)

    assert code == 0, f"setup.sh failed with exit code {code}. Stderr:\n{stderr}"
    log_content = read_log_file(log_path)
    
    # Base assertions: Verify core system and python setup ran
    assert "MOCK_CALLED: sudo apt update" in log_content
    assert "MOCK_CALLED: python3 -m uv pip install -U setuptools wheel uv" in log_content
    assert "MOCK_CALLED: python3 -m uv pip install --no-cache-dir -U -r requirements.txt" in log_content

    assert "Installing stable jax, jaxlib for NVIDIA gpu" in stdout
    stable_cmd = "MOCK_CALLED: python3 -m uv pip install jax[cuda12]"
    te_cmd = "MOCK_CALLED: python3 -m uv pip install transformer-engine[jax]"
    assert stable_cmd in log_content
    assert te_cmd in log_content
    assert "jax[tpu]" not in log_content # Should not install TPU pkgs

@pytest.mark.integration_test
@pytest.mark.tpu_only
def test_nightly_tpu_setup():
  """
  Test setup nightly mode on TPU device
  """
  mode, device = "nightly", "tpu"
  print(f"Testing MODE={mode} DEVICE={device}...")
  
  with setup_mock_environment() as (log_path, test_env):
    stdout, stderr, code = run_setup_script(test_env, mode=mode, device=device)

    assert code == 0, f"setup.sh failed with exit code {code}. Stderr:\n{stderr}"
    log_content = read_log_file(log_path)
    
    # Base assertions: Verify core system and python setup ran
    assert "MOCK_CALLED: sudo apt update" in log_content
    assert "MOCK_CALLED: python3 -m uv pip install -U setuptools wheel uv" in log_content
    assert "MOCK_CALLED: python3 -m uv pip install --no-cache-dir -U -r requirements.txt" in log_content

    assert "Installing jax-nightly, jaxlib-nightly" in stdout
    jax_cmd = "MOCK_CALLED: python3 -m uv pip install --pre -U jax -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/"
    jaxlib_cmd = "MOCK_CALLED: python3 -m uv pip install --pre -U jaxlib -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/"
    libtpu_cmd = "MOCK_CALLED: python3 -m uv pip install -U --pre libtpu -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
    assert jax_cmd in log_content
    assert jaxlib_cmd in log_content
    assert libtpu_cmd in log_content # Ran because LIBTPU_GCS_PATH was not set
    assert "jax-cuda12-plugin" not in log_content


@pytest.mark.integration_test
@pytest.mark.gpu_only
def test_nightly_gpu_setup():
  """
  Test setup nightly mode on GPU device
  """
  mode, device = "nightly", "gpu"
  print(f"Testing MODE={mode} DEVICE={device}...")
  
  with setup_mock_environment() as (log_path, test_env):
    stdout, stderr, code = run_setup_script(test_env, mode=mode, device=device)

    assert code == 0, f"setup.sh failed with exit code {code}. Stderr:\n{stderr}"
    log_content = read_log_file(log_path)
    
    # Base assertions: Verify core system and python setup ran
    assert "MOCK_CALLED: sudo apt update" in log_content
    assert "MOCK_CALLED: python3 -m uv pip install -U setuptools wheel uv" in log_content
    assert "MOCK_CALLED: python3 -m uv pip install --no-cache-dir -U -r requirements.txt" in log_content
    assert "Installing latest jax-nightly, jaxlib-nightly" in stdout
    nightly_cmd = "MOCK_CALLED: python3 -m uv pip install -U --pre jax jaxlib jax-cuda12-plugin[with-cuda] jax-cuda12-pjrt -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/"
    te_cmd = "MOCK_CALLED: python3 -m uv pip install https://github.com/NVIDIA/TransformerEngine/archive/9d031f.zip"
    assert nightly_cmd in log_content
    assert te_cmd in log_content
    assert "libtpu-nightly" not in log_content # Should not install TPU pkgs
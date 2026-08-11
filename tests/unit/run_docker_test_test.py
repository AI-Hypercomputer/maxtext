# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for MaxText Docker testing scripts and XPK reuse-image / fast-rebuild options."""

import os
import subprocess
import unittest
from unittest import mock

import pytest

@pytest.mark.cpu_only
class RunDockerTestTest(unittest.TestCase):
  """Tests for run_docker_test.sh / run_docker_test.py CLI options."""

  def test_run_docker_test_help(self):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    script_path = os.path.join(repo_root, "src", "dependencies", "scripts", "run_docker_test.sh")

    result = subprocess.run(
        ["bash", script_path, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    self.assertEqual(result.returncode, 0)
    self.assertIn("build-all", result.stdout)
    self.assertIn("test-only", result.stdout)
    self.assertIn("fast-rebuild", result.stdout)

  def test_run_docker_test_invalid_mode(self):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    script_path = os.path.join(repo_root, "src", "dependencies", "scripts", "run_docker_test.sh")

    result = subprocess.run(
        ["bash", script_path, "--mode=invalid_mode"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("Invalid mode", result.stderr)

  def test_run_docker_test_fast_rebuild_mode(self):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    script_path = os.path.join(repo_root, "src", "dependencies", "scripts", "run_docker_test.sh")

    result = subprocess.run(
        ["bash", script_path, "--mode=fast-rebuild", "--base-image=non_existent_base_img_999"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    self.assertIn("Mode           : fast-rebuild", result.stdout)
    self.assertIn("not found locally", result.stderr)

  def test_run_docker_test_test_only_mode(self):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    script_path = os.path.join(repo_root, "src", "dependencies", "scripts", "run_docker_test.sh")

    result = subprocess.run(
        ["bash", script_path, "--mode=test-only", "--image=non_existent_test_img_456"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    self.assertIn("Mode           : test-only", result.stdout)
    self.assertIn("Skipping image build.", result.stdout)
    self.assertIn("not found locally", result.stderr)


if __name__ == "__main__":
  unittest.main()

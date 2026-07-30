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

from benchmarks import maxtext_xpk_runner
from benchmarks import xpk_configs
import benchmarks.maxtext_trillium_model_configs as model_configs


@pytest.mark.cpu_only
class RunDockerTestTest(unittest.TestCase):
  """Tests for run_docker_test.sh / run_docker_test.py and XPK docker reuse flags."""

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

  def test_xpk_workload_config_default_base_docker_image(self):
    cluster_config = xpk_configs.XpkClusterConfig(
        cluster_name="test-cluster",
        project="test-project",
        zone="us-central2-b",
        device_type="v6e-16",
    )
    wl_config = maxtext_xpk_runner.WorkloadConfig(
        model=model_configs.default_128,
        num_slices=1,
        device_type="v6e-16",
        base_output_directory="gs://test",
        base_docker_image="maxtext_base_image",
        libtpu_type=maxtext_xpk_runner.LibTpuType.MAXTEXT,
        generate_metrics_and_upload_to_big_query=False,
    )
    cmd, _ = maxtext_xpk_runner.generate_xpk_workload_cmd(
        cluster_config=cluster_config,
        wl_config=wl_config,
        workload_name="test-workload",
    )
    self.assertIn('--base-docker-image="maxtext_base_image"', cmd)
    self.assertNotIn('--docker-image="maxtext_base_image"', cmd)

  def test_xpk_workload_config_reuse_image(self):
    cluster_config = xpk_configs.XpkClusterConfig(
        cluster_name="test-cluster",
        project="test-project",
        zone="us-central2-b",
        device_type="v6e-16",
    )
    wl_config = maxtext_xpk_runner.WorkloadConfig(
        model=model_configs.default_128,
        num_slices=1,
        device_type="v6e-16",
        base_output_directory="gs://test",
        base_docker_image="maxtext_base_image",
        libtpu_type=maxtext_xpk_runner.LibTpuType.MAXTEXT,
        generate_metrics_and_upload_to_big_query=False,
        reuse_image=True,
    )
    cmd, _ = maxtext_xpk_runner.generate_xpk_workload_cmd(
        cluster_config=cluster_config,
        wl_config=wl_config,
        workload_name="test-workload",
    )
    self.assertIn('--docker-image="maxtext_base_image"', cmd)
    self.assertNotIn('--base-docker-image="maxtext_base_image"', cmd)

  @mock.patch.object(maxtext_xpk_runner, "run_command_with_updates", return_value=0)
  def test_xpk_workload_config_fast_rebuild(self, mock_run_cmd):
    cluster_config = xpk_configs.XpkClusterConfig(
        cluster_name="test-cluster",
        project="test-project",
        zone="us-central2-b",
        device_type="v6e-16",
    )
    wl_config = maxtext_xpk_runner.WorkloadConfig(
        model=model_configs.default_128,
        num_slices=1,
        device_type="v6e-16",
        base_output_directory="gs://test",
        base_docker_image="maxtext_base_image",
        libtpu_type=maxtext_xpk_runner.LibTpuType.MAXTEXT,
        generate_metrics_and_upload_to_big_query=False,
        fast_rebuild=True,
    )
    cmd, _ = maxtext_xpk_runner.generate_xpk_workload_cmd(
        cluster_config=cluster_config,
        wl_config=wl_config,
        workload_name="test-workload",
    )
    mock_run_cmd.assert_called_once()
    self.assertIn("FAST_REBUILD=true", mock_run_cmd.call_args[0][0])
    self.assertIn('--docker-image="maxtext_base_image__runner"', cmd)
    self.assertNotIn('--base-docker-image=', cmd)


if __name__ == "__main__":
  unittest.main()

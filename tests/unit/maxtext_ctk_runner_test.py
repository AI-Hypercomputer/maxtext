# Copyright 2023–2026 Google LLC
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

"""Unit tests for Cluster Toolkit (gcluster) benchmark runner and configs."""

import argparse
import unittest
from unittest import mock

from benchmarks import benchmark_runner
from benchmarks import maxtext_trillium_model_configs as trillium_configs
from benchmarks import maxtext_v5p_model_configs as v5p_configs
from benchmarks.ctk_configs import ClusterConfig
from benchmarks.maxtext_ctk_runner import (
    LibTpuType,
    PathwaysConfig,
    WorkloadConfig,
    generate_workload_cmd,
    wait_for_workload_completion,
)
from benchmarks.recipes import args_helper


class CtkRunnerTest(unittest.TestCase):
  """Tests for ClusterConfig, WorkloadConfig, and generate_workload_cmd."""

  def test_cluster_config_zone_and_location(self):
    cfg_from_zone = ClusterConfig(
        cluster_name="test-cluster",
        project="test-project",
        zone="us-central2-b",
        device_type="v6e-256",
    )
    self.assertEqual(cfg_from_zone.location, "us-central2-b")
    self.assertEqual(cfg_from_zone.zone, "us-central2-b")

    cfg_from_region = ClusterConfig(
        cluster_name="test-cluster",
        project="test-project",
        location="us-central2",
        device_type="v6e-256",
    )
    self.assertEqual(cfg_from_region.location, "us-central2")
    self.assertEqual(cfg_from_region.zone, "")

    cfg_from_zone_location = ClusterConfig(
        cluster_name="test-cluster",
        project="test-project",
        location="us-central2-b",
        device_type="v6e-256",
    )
    self.assertEqual(cfg_from_zone_location.location, "us-central2-b")
    self.assertEqual(cfg_from_zone_location.zone, "us-central2-b")

  def test_generate_workload_cmd_mcjax(self):
    cluster_config = ClusterConfig(
        cluster_name="v6e-cluster",
        project="my-project",
        location="us-east5",
        device_type="v6e-256",
    )
    wl_config = WorkloadConfig(
        model=trillium_configs.llama2_70b_4096,
        num_slices=2,
        device_type="v6e-256",
        base_output_directory="gs://my-bucket/outputs",
        base_docker_image="maxtext_base_image",
        libtpu_type=LibTpuType.MAXTEXT,
        num_steps=10,
        max_restarts=3,
        priority="high",
        generate_metrics_and_upload_to_big_query=False,
        xpk_storage=["gs://my-bucket;/data;ro"],
    )

    cmd, name = generate_workload_cmd(
        cluster_config=cluster_config,
        wl_config=wl_config,
        user="testuser",
        temp_key="abc",
    )

    self.assertTrue(name.startswith("tes-llama2-70b-2-"))
    self.assertTrue(name.endswith("-abc"))
    self.assertIn("gcluster job submit", cmd)
    self.assertIn("--cluster=v6e-cluster", cmd)
    self.assertIn("--project=my-project", cmd)
    self.assertIn("--location=us-east5", cmd)
    self.assertIn("--compute-type=v6e-4", cmd)
    self.assertIn("--topology=16x16", cmd)
    self.assertIn("--num-slices=2", cmd)
    self.assertIn('--base-image="maxtext_base_image" --build-context=.', cmd)
    self.assertIn('--mount="gs://my-bucket;/data;ro"', cmd)
    self.assertIn("--priority=high", cmd)
    self.assertIn("--restarts=3", cmd)
    self.assertIn("export JAX_PLATFORMS=tpu,cpu", cmd)
    self.assertIn("TPU_WORKER_ID=${TPU_WORKER_ID:-${JOB_COMPLETION_INDEX:--1}}", cmd)
    self.assertNotIn("--pathways", cmd)

  def test_generate_workload_cmd_mcjax_registry_image(self):
    cluster_config = ClusterConfig(
        cluster_name="v6e-cluster",
        project="my-project",
        zone="us-east5-b",
        device_type="v6e-8",
    )
    registry_image = "us-docker.pkg.dev/cloud-tpu-images/maxtext-images/tpu_pre_training:latest"
    wl_config = WorkloadConfig(
        model=trillium_configs.llama2_7b_4096,
        num_slices=1,
        device_type="v6e-8",
        base_output_directory="gs://my-bucket/outputs",
        base_docker_image=registry_image,
        libtpu_type=LibTpuType.MAXTEXT,
        generate_metrics_and_upload_to_big_query=False,
    )

    cmd, _ = generate_workload_cmd(
        cluster_config=cluster_config,
        wl_config=wl_config,
        workload_name="custom-job",
    )

    self.assertIn("--compute-type=v6e-8", cmd)
    self.assertIn("--topology=2x4", cmd)
    self.assertIn(f'--image="{registry_image}"', cmd)
    self.assertNotIn("--base-image", cmd)

  def test_generate_workload_cmd_pathways(self):
    cluster_config = ClusterConfig(
        cluster_name="pw-cluster",
        project="my-project",
        location="europe-west4",
        device_type="v6e-256",
    )
    pw_config = PathwaysConfig(
        server_image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest",
        proxy_server_image="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest",
        runner_image="us-docker.pkg.dev/my-project/maxtext_runner:latest",
        colocated_python_sidecar_image="us-docker.pkg.dev/my-project/sidecar:latest",
    )
    wl_config = WorkloadConfig(
        model=trillium_configs.llama2_70b_4096,
        num_slices=1,
        device_type="v6e-256",
        base_output_directory="gs://my-bucket/pw-outputs",
        base_docker_image=None,
        libtpu_type=LibTpuType.MAXTEXT,
        pathways_config=pw_config,
        generate_metrics_and_upload_to_big_query=False,
    )

    cmd, name = generate_workload_cmd(
        cluster_config=cluster_config,
        wl_config=wl_config,
        user="alice",
        temp_key="xyz",
    )

    self.assertEqual(name, "ali-pw-llama2--1-xyz")
    self.assertIn("--pathways", cmd)
    self.assertIn(
        "--pathways-server-image=us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest",
        cmd,
    )
    self.assertIn(
        "--pathways-proxy-server-image=us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest",
        cmd,
    )
    self.assertIn(
        "--pathways-colocated-python-sidecar-image=us-docker.pkg.dev/my-project/sidecar:latest",
        cmd,
    )
    self.assertIn(
        "--image=us-docker.pkg.dev/my-project/maxtext_runner:latest",
        cmd,
    )
    self.assertIn("--grace-period=300s", cmd)
    self.assertIn("--pathways-gcs-location=gs://my-bucket/pw-outputs", cmd)
    self.assertIn("export JAX_PLATFORMS=proxy", cmd)

  def test_generate_workload_cmd_v5p_shape_and_validation(self):
    cluster_config = ClusterConfig(
        cluster_name="v5p-cluster",
        project="my-project",
        zone="us-central1-a",
        device_type="v5p-128",
    )
    wl_config = WorkloadConfig(
        model=v5p_configs.llama2_70b_v5p_128,
        num_slices=1,
        device_type="v5p-128",
        base_output_directory="gs://my-bucket/v5p",
        base_docker_image="gcr.io/my-project/runner:latest",
        libtpu_type=LibTpuType.MAXTEXT,
        generate_metrics_and_upload_to_big_query=True,
        db_project="test-db-proj",
        db_dataset="test_dataset",
    )

    self.assertEqual(wl_config.num_devices_per_slice, 64)
    self.assertEqual(wl_config.compute_type, "v5p-8")
    self.assertEqual(wl_config.topology, "4x4x4")
    self.assertEqual(wl_config.hardware_id, "v5p")

    cmd, _ = generate_workload_cmd(
        cluster_config=cluster_config,
        wl_config=wl_config,
        workload_name="v5p-job",
    )
    self.assertIn("--compute-type=v5p-8", cmd)
    self.assertIn("--topology=4x4x4", cmd)
    self.assertIn("--number_of_chips=64", cmd)

    # Unknown v5p size without explicit topology should raise ValueError
    with self.assertRaisesRegex(ValueError, "Unsupported v5p size: 24"):
      WorkloadConfig(
          model=v5p_configs.llama2_70b_v5p_128,
          num_slices=1,
          device_type="v5p-24",
          base_output_directory="gs://my-bucket/v5p",
          base_docker_image="gcr.io/my-project/runner:latest",
          libtpu_type=LibTpuType.MAXTEXT,
          generate_metrics_and_upload_to_big_query=False,
      )

    # Unknown v5p size with explicit topology override should succeed
    custom_v5p_wl = WorkloadConfig(
        model=v5p_configs.llama2_70b_v5p_128,
        num_slices=1,
        device_type="v5p-24",
        topology="2x2x3",
        base_output_directory="gs://my-bucket/v5p",
        base_docker_image="gcr.io/my-project/runner:latest",
        libtpu_type=LibTpuType.MAXTEXT,
        generate_metrics_and_upload_to_big_query=False,
    )
    self.assertEqual(custom_v5p_wl.topology, "2x2x3")
    self.assertEqual(custom_v5p_wl.compute_type, "v5p-8")

  def test_invalid_mount_raises_value_error(self):
    cluster_config = ClusterConfig(
        cluster_name="v6e-cluster",
        project="my-project",
        zone="us-east5-b",
        device_type="v6e-256",
    )
    wl_config = WorkloadConfig(
        model=trillium_configs.llama2_70b_4096,
        num_slices=1,
        device_type="v6e-256",
        base_output_directory="gs://my-bucket/outputs",
        base_docker_image="gcr.io/my-project/runner:latest",
        libtpu_type=LibTpuType.MAXTEXT,
        generate_metrics_and_upload_to_big_query=False,
        xpk_storage=["legacy_storage_name"],
    )
    with self.assertRaisesRegex(ValueError, "Invalid mount specification 'legacy_storage_name'"):
      generate_workload_cmd(cluster_config=cluster_config, wl_config=wl_config)

  @mock.patch("benchmarks.maxtext_ctk_runner.run_command_with_updates", return_value=0)
  def test_wait_for_workload_completion(self, mock_run_cmd):
    cluster_config = ClusterConfig(
        cluster_name="v6e-cluster",
        project="my-project",
        location="us-east5",
        device_type="v6e-256",
    )
    rc = wait_for_workload_completion(cluster_config, "test-job", timeout="4h")
    self.assertEqual(rc, 0)
    mock_run_cmd.assert_called_once()
    called_cmd = mock_run_cmd.call_args[0][0]
    self.assertIn("--location=us-east5", called_cmd)
    self.assertIn(
        "kubectl wait --for=jsonpath='{.status.terminalState}' jobset/test-job --timeout=4h",
        called_cmd,
    )
    self.assertIn(
        "kubectl wait --for=condition=Completed jobset/test-job --timeout=0s",
        called_cmd,
    )

  def test_benchmark_runner_subparser_pathways_args(self):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="runner")
    ctk_parser = subparsers.add_parser("ctk")
    benchmark_runner.add_ctk_runner_arguments(ctk_parser)
    benchmark_runner.add_pathways_arguments(ctk_parser)

    options = parser.parse_args(
        [
            "ctk",
            "--use_pathways=true",
            "--pathways_runner_image=us-docker.pkg.dev/path/to/runner",
            "--project=my-project",
            "--location=us-central2",
            "--cluster_name=my-cluster",
            "--device_type=v6e-256",
            "--base_output_directory=gs://my-bucket",
        ]
    )
    self.assertEqual(options.runner, "ctk")
    self.assertTrue(options.use_pathways)
    self.assertEqual(options.location, "us-central2")
    self.assertEqual(options.pathways_runner_image, "us-docker.pkg.dev/path/to/runner")

  @mock.patch("benchmarks.recipes.args_helper.os.system", return_value=0)
  def test_args_helper_delete_uses_gcluster_cancel(self, mock_system):
    cluster_config = ClusterConfig(
        cluster_name="v6e-cluster",
        project="my-project",
        location="us-east5",
        device_type="v6e-256",
    )
    args_helper.handle_delete_specific_workload(cluster_config, "my-job")
    mock_system.assert_called_once_with(
        "gcluster job cancel my-job --project=my-project --cluster=v6e-cluster --location=us-east5"
    )

    mock_system.reset_mock()
    should_continue = args_helper.handle_cmd_args(cluster_config, is_delete=True, user="alice")
    self.assertFalse(should_continue)
    mock_system.assert_called_once()
    self.assertIn("gcluster job cancel", mock_system.call_args[0][0])


if __name__ == "__main__":
  unittest.main()

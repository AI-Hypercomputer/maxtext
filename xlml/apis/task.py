# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base task file for a test job."""

import abc
import dataclasses
import datetime
from typing import Optional, Tuple
import airflow
from airflow.models.taskmixin import DAGNode
from airflow.utils.task_group import TaskGroup
from xlml.apis import gcp_config, metric_config, test_config
from xlml.utils import gpu, metric, ssh, tpu, xpk


class BaseTask(abc.ABC):
  """This is a class to set up base tasks."""

  @abc.abstractmethod
  def run() -> DAGNode:
    """Run a test job.

    Returns:
      A DAG node that executes this test.
    """
    ...


@dataclasses.dataclass
class TpuQueuedResourceTask(BaseTask):
  """This is a class to set up tasks for TPU provisioned by Queued Resource.

  Attributes:
    task_test_config: Test configs to run on this TPU.
    task_gcp_config: Runtime TPU creation parameters.
    task_metric_config: Metric configs to process metrics.
    tpu_name_env_var: The flag to define if set up env variable for tpu name.
    all_workers: The flag to define if run commands on all workers or worker 0
      only.
  """

  # TODO(wcromar): make these attributes less verbose
  task_test_config: test_config.TestConfig[test_config.Tpu]
  task_gcp_config: gcp_config.GCPConfig
  tpu_create_timeout: datetime.timedelta = datetime.timedelta(minutes=60)
  task_metric_config: Optional[metric_config.MetricConfig] = None
  tpu_name_env_var: bool = False
  all_workers: bool = True

  def run(self) -> DAGNode:
    """Run a test job.

    Returns:
      A task group with the following tasks chained: provision, run_model,
      post_process and clean_up.
    """
    with TaskGroup(
        group_id=self.task_test_config.benchmark_id, prefix_group_id=True
    ) as group:
      provision, queued_resource, ssh_keys = self.provision()
      run_model = self.run_model(queued_resource, ssh_keys)
      post_process = self.post_process()
      clean_up = self.clean_up(queued_resource)

      provision >> run_model >> post_process >> clean_up

    return group

  def provision(self) -> Tuple[DAGNode, airflow.XComArg, airflow.XComArg]:
    """Provision a TPU accelerator via a Queued Resource.

    Generates a random TPU name and SSH keys, creates a Queued Resource, and
    runs the test config's setup script on the TPU when it is ready.

    Returns:
      A DAG node that will provision a TPU, an XCom value for the qualified
      queued resource name, and an XCom value for the SSH keys.

    Raises:
      AirflowTaskTimeout: An error occurs when execution_timeout is breached.
    """
    with TaskGroup(group_id="provision") as group:
      with TaskGroup(group_id="initialize"):
        tpu_name = tpu.generate_tpu_name(
            self.task_test_config.benchmark_id, self.tpu_name_env_var
        )
        ssh_keys = ssh.generate_ssh_keys()

      queued_resource_op, queued_resource_name = tpu.create_queued_resource(
          tpu_name,
          self.task_test_config.accelerator,
          self.task_gcp_config,
          ssh_keys,
          self.tpu_create_timeout,
          self.task_test_config.num_slices,
      )
      queued_resource_op >> tpu.ssh_tpu.override(task_id="setup")(
          queued_resource_name,
          # TODO(wcromar): remove split
          self.task_test_config.setup_script,
          ssh_keys,
          self.all_workers,
      )

    return group, queued_resource_name, ssh_keys

  def run_model(
      self,
      # TODO(wcromar): Is there a way to annotate the type of the XCom arg?
      queued_resource: airflow.XComArg,
      ssh_keys: airflow.XComArg,
  ) -> DAGNode:
    """Run the TPU test in `task_test_config`.

    Args:
      queued_resource: XCom value for the queued resource name (string).
      ssh_keys: And XCom value for the TPU's SSH keys (SshKeys).

    Returns:
      A DAG node that executes the model test.
    """
    return tpu.ssh_tpu.override(
        task_id="run_model",
        execution_timeout=datetime.timedelta(
            minutes=self.task_test_config.time_out_in_min
        ),
        owner=self.task_test_config.task_owner,
    )(
        queued_resource,
        # TODO(wcromar): remove split
        self.task_test_config.test_script,
        ssh_keys,
        self.all_workers,
    )

  def post_process(self) -> DAGNode:
    """Process metrics and metadata, and insert them into BigQuery tables.

    Returns:
      A DAG node that executes the post process.
    """
    with TaskGroup(group_id="post_process") as group:
      process_id = metric.generate_process_id.override(retries=0)()
      metric.process_metrics.override(retries=0)(
          process_id,
          self.task_test_config,
          self.task_metric_config,
          self.task_gcp_config,
      )
      return group

  def clean_up(self, queued_resource: airflow.XComArg) -> DAGNode:
    """Clean up TPU resources created by `provision`.

    Args:
      queued_resource: an XCom value for the qualified QR name.

    Returns:
      A DAG node that deletes the queued resource and its owned nodes.

    Raises:
      AirflowTaskTimeout: An error occurs when execution_timeout is breached.
    """
    return tpu.delete_queued_resource.override(group_id="clean_up")(queued_resource)


@dataclasses.dataclass
class TpuXpkTask(BaseTask):
  """This is a class to set up tasks for TPU provisioned by XPK tool.

  Attributes:
    task_test_config: Test configs to run on this TPU.
    task_gcp_config: Runtime TPU creation parameters.
    task_metric_config: Metric configs to process metrics.
  """

  task_test_config: test_config.TestConfig[test_config.Tpu]
  task_gcp_config: gcp_config.GCPConfig
  task_metric_config: Optional[metric_config.MetricConfig] = None

  def run(self) -> DAGNode:
    """Run a test job within a docker image.

    Returns:
      A task group with the following tasks chained: run_model and
      post_process.
    """
    with TaskGroup(group_id=self.task_test_config.benchmark_id) as group:
      self.run_model() >> self.post_process()

    return group

  def run_model(self) -> DAGNode:
    """Run the TPU test in `task_test_config` using xpk.

    Returns:
      A DAG node that executes the model test.
    """
    with TaskGroup(group_id="run_model") as group:
      workload_id = xpk.generate_workload_id(self.task_test_config.benchmark_id)
      run_workload = xpk.run_workload(
          task_id="run_workload",
          cluster_project=self.task_gcp_config.project_name,
          zone=self.task_gcp_config.zone,
          cluster_name=self.task_test_config.cluster_name,
          benchmark_id=self.task_test_config.benchmark_id,
          workload_id=workload_id,
          docker_image=self.task_test_config.docker_image,
          accelerator_type=self.task_test_config.accelerator.name,
          run_cmds=self.task_test_config.test_script,
          task_owner=self.task_test_config.task_owner,
          startup_timeout=self.task_test_config.startup_time_out_in_sec,
          num_slices=self.task_test_config.num_slices,
      )
      wait_for_workload_completion = xpk.wait_for_workload_completion.override(
          timeout=self.task_test_config.time_out_in_min * 60,
      )(
          workload_id=workload_id,
          project_id=self.task_gcp_config.project_name,
          region=self.task_gcp_config.zone[:-2],
          cluster_name=self.task_test_config.cluster_name,
      )

      workload_id >> run_workload >> wait_for_workload_completion
      return group

  def post_process(self) -> DAGNode:
    """Process metrics and metadata, and insert them into BigQuery tables.

    Returns:
      A DAG node that executes the post process.
    """
    with TaskGroup(group_id="post_process") as group:
      process_id = metric.generate_process_id.override(retries=0)()
      metric.process_metrics.override(retries=0)(
          process_id,
          self.task_test_config,
          self.task_metric_config,
          self.task_gcp_config,
      )

      return group


@dataclasses.dataclass
class GpuCreateResourceTask(BaseTask):
  """This is a class to set up tasks for GPU.

  Attributes:
    image_project: the project that an image belongs to.
    image_family: the family group that an image belongs to.
  """

  image_project: str
  image_family: str
  task_test_config: test_config.TestConfig[test_config.Gpu]
  task_gcp_config: gcp_config.GCPConfig
  task_metric_config: Optional[metric_config.MetricConfig] = None

  def run(self) -> DAGNode:
    """Run a test job.

    Returns:
      A task group with the following tasks chained: provision, run_model,
      post_process, clean_up.
    """
    # piz: We skip the queued resource for GPU for now since there is no queued
    # resource command for GPU.
    with TaskGroup(
        group_id=self.task_test_config.benchmark_id, prefix_group_id=True
    ) as group:
      provision, ip_address, instance_name, ssh_keys = self.provision()
      run_model = self.run_model(ip_address, ssh_keys)
      post_process = self.post_process()
      clean_up = self.clean_up(
          instance_name, self.task_gcp_config.project_name, self.task_gcp_config.zone
      )
      provision >> run_model >> post_process >> clean_up
    return group

  def provision(
      self,
  ) -> Tuple[DAGNode, airflow.XComArg, airflow.XComArg, airflow.XComArg]:
    """Provision a GPU accelerator via a resource creation.

    Generates a random GPU name and SSH keys, creates a VM Resource, and
    runs the test config's setup script on the GPU when it is ready.

    Returns:
      A DAG node that will provision a GPU, an XCome value of the ip address
      for the host, an XCom value for the GPU name, and an XCom value for
      the SSH keys.

    Raises:
      AirflowTaskTimeout: An error occurs when execution_timeout is breached.
    """
    with TaskGroup(group_id="provision") as group:
      with TaskGroup(group_id="initialize"):
        gpu_name = gpu.generate_gpu_name()
        ssh_keys = ssh.generate_ssh_keys()

      ip_address = gpu.create_resource.override(task_id="create_resource")(
          gpu_name,
          self.image_project,
          self.image_family,
          self.task_test_config.accelerator,
          self.task_gcp_config,
          ssh_keys,
      )

      gpu.ssh_host.override(task_id="setup")(
          ip_address,
          self.task_test_config.setup_script,
          ssh_keys,
      )
    return group, ip_address, gpu_name, ssh_keys

  def run_model(
      self,
      resource: airflow.XComArg,
      ssh_keys: airflow.XComArg,
  ) -> DAGNode:
    """Run the GPU test in `task_test_config`.

    Args:
      gpu_name: XCom value for the GPU name (string).
      ssh_keys: And XCom value for the GPU's SSH keys (SshKeys).

    Returns:
      A DAG node that executes the model test.
    """
    return gpu.ssh_host.override(
        task_id="run_model",
        execution_timeout=datetime.timedelta(
            minutes=self.task_test_config.time_out_in_min
        ),
        owner=self.task_test_config.task_owner,
    )(
        resource,
        self.task_test_config.test_script,
        ssh_keys,
    )

  def post_process(self) -> DAGNode:
    """Process metrics and metadata, and insert them into BigQuery tables.

    Returns:
      A DAG node that executes the post process.
    """
    with TaskGroup(group_id="post_process") as group:
      process_id = metric.generate_process_id.override(retries=0)()
      metric.process_metrics.override(retries=0)(
          process_id,
          self.task_test_config,
          self.task_metric_config,
          self.task_gcp_config,
      )
      return group

  def clean_up(self, resource: airflow.XComArg, project_id: str, zone: str) -> DAGNode:
    """Clean up GPU resources created by `provision`.

    Args:
      resource: an XCom value for the qualified instance name.
      project_id: project of the instance.
      zone: zone of the instance.
    Returns:
      A DAG node that deletes the resource and its owned nodes.

    Raises:
      AirflowTaskTimeout: An error occurs when execution_timeout is breached.
    """
    return gpu.delete_resource.override(group_id="clean_up")(resource, project_id, zone)

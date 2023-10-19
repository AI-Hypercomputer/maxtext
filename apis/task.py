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
from absl import logging
import airflow
from airflow.models.taskmixin import DAGNode
from airflow.utils.task_group import TaskGroup
from apis import gcp_config, metric_config, test_config
from implementations.utils import metric
from implementations.utils import ssh, tpu


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
class TpuTask(BaseTask):
  """This is a class to set up tasks for TPU.

  Attributes:
    task_test_config: Test configs to run on this TPU.
    task_gcp_config: Runtime TPU creation parameters.
    task_metric_config: Metric configs to process metrics.
  """

  # TODO(wcromar): make these attributes less verbose
  task_test_config: test_config.TestConfig[test_config.Tpu]
  task_gcp_config: gcp_config.GCPConfig
  tpu_create_timeout: datetime.timedelta = datetime.timedelta(minutes=60)
  task_metric_config: Optional[metric_config.MetricConfig] = None

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
        if self.task_test_config.custom_tpu_name.strip():
          base_tpu_name = self.task_test_config.custom_tpu_name
        else:
          base_tpu_name = self.task_test_config.benchmark_id

        tpu_name = tpu.generate_tpu_name(
            base_tpu_name,
            self.task_test_config.tpu_name_with_suffix,
        )
        ssh_keys = ssh.generate_ssh_keys()

      queued_resource_op, queued_resource_name = tpu.create_queued_resource(
          tpu_name,
          self.task_test_config.accelerator,
          self.task_gcp_config,
          ssh_keys,
          self.tpu_create_timeout,
      )
      queued_resource_op >> tpu.ssh_tpu.override(task_id="setup")(
          queued_resource_name,
          # TODO(wcromar): remove split
          self.task_test_config.setup_script,
          ssh_keys,
          self.task_test_config.all_workers,
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
        self.task_test_config.all_workers,
    )

  def post_process(self) -> DAGNode:
    """Process metrics and metadata, and insert them into BigQuery tables.

    Returns:
      A DAG node that executes the post process.
    """
    with TaskGroup(group_id="post_process") as group:
      process_id = metric.generate_process_id.override(retries=1)()
      metric.process_metrics(
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
    return tpu.delete_queued_resource.override(group_id="clean_up")(
        queued_resource
    )


@dataclasses.dataclass
class GpuTask(BaseTask):
  """This is a class to set up tasks for GPU.

  Attributes:
    image: the image version that a GPU runs.
    image_project: the project that an image belongs to.
    image_family: the family group that an image belongs to.
  """

  image: str
  image_project: str
  image_family: str

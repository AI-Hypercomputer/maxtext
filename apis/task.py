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

"""Task file for a test job."""

import abc
import dataclasses
from datetime import timedelta
from airflow.models import baseoperator
from airflow.operators import empty, python_operator
from apis import gcp_config, test_config
from implementations.utils import tpu


class BaseTask:
  """This is a class to set up base tasks."""

  @abc.abstractmethod
  def provision(
      self,
      task_id_suffix: str,
      job_gcp_config: gcp_config.GCPConfig,
      job_test_config: test_config.TestConfig,
  ) -> baseoperator.BaseOperator:
    ...

  @abc.abstractmethod
  def run_model(
      self,
      task_id_suffix: str,
      job_gcp_config: gcp_config.GCPConfig,
      job_test_config: test_config.TestConfig,
  ) -> baseoperator.BaseOperator:
    ...

  @abc.abstractmethod
  def post_process(
      self,
      task_id_suffix: str,
      job_gcp_config: gcp_config.GCPConfig,
      job_test_config: test_config.TestConfig,
  ) -> baseoperator.BaseOperator:
    ...

  @abc.abstractmethod
  def clean_up(
      self, task_id_suffix: str, job_gcp_config: gcp_config.GCPConfig
  ) -> baseoperator.BaseOperator:
    ...


@dataclasses.dataclass
class TPUTask(BaseTask):
  """This is a class to set up tasks for TPU.

  Attributes:
    version: The version of a TPU without 'v', such as 4, 5litepod, etc.
    size: The number of cores of a TPU, such as 8, 16, 32, etc.
    runtime_version: The version of a runtime image that a TPU runs.
    task_owner: The owner of a TPU task.
  """

  version: str
  size: int
  runtime_version: str
  task_owner: str

  def type(self) -> str:
    return f"v{self.version}-{self.size}"

  def provision(
      self,
      task_id_suffix: str,
      job_gcp_config: gcp_config.GCPConfig,
      job_test_config: test_config.TestConfig,
  ) -> baseoperator.BaseOperator:
    """Provision a TPU accelerator (timeout is 60 min).

    Provision steps include 1) a Queued Resource creation, and 2) a TPU setup.

    Args:
      task_id_suffix: An ID suffix of an Airflow task.
      job_gcp_config: Configs of GCP to run a test job.
      job_test_config: Configs for a test job.

    Returns:
      A PythonOperator that execute provision callable.

    Raises:
      AirflowTaskTimeout: An error occurs when execution_timeout is breached.
    """
    return python_operator.PythonOperator(
        task_id=f"provision_tpu_{task_id_suffix}",
        python_callable=tpu.provision,
        op_kwargs={
            "task_id_suffix": task_id_suffix,
            "project_name": job_gcp_config.project_name,
            "project_number": job_gcp_config.project_number,
            "zone": job_gcp_config.zone,
            "type": self.type(),
            "runtime_version": self.runtime_version,
            "set_up_cmd": job_test_config.set_up_cmd,
        },
        execution_timeout=timedelta(minutes=60),
        owner=self.task_owner,
    )

  # TODO(ranran): Implement logic for run_model task
  def run_model(
      self,
      task_id_suffix: str,
      job_gcp_config: gcp_config.GCPConfig,
      job_test_config: test_config.TestConfig,
  ) -> baseoperator.BaseOperator:
    return empty.EmptyOperator(task_id=f"run_model_{task_id_suffix}")

  # TODO(ranran): Implement logic for post_process task
  def post_process(
      self,
      task_id_suffix: str,
      job_gcp_config: gcp_config.GCPConfig,
      job_test_config: test_config.TestConfig,
  ) -> baseoperator.BaseOperator:
    return empty.EmptyOperator(task_id=f"post_process_{task_id_suffix}")

  def clean_up(
      self, task_id_suffix: str, job_gcp_config: gcp_config.GCPConfig
  ) -> baseoperator.BaseOperator:
    """Clean up a TPU accelerator (timeout is 10 min).

    Clean up steps include 1) a TPU deletion, and 2) a Queued Resource deletion.

    Args:
      task_id_suffix: A suffix of an Airflow task.
      job_gcp_config: Configs of GCP to run a test job.

    Returns:
      A PythonOperator that execute clean_up callable.

    Raises:
      AirflowTaskTimeout: An error occurs when execution_timeout is breached.
    """
    return python_operator.PythonOperator(
        task_id=f"clean_up_tpu_{task_id_suffix}",
        python_callable=tpu.clean_up,
        op_kwargs={
            "task_id_suffix": task_id_suffix,
            "project_number": job_gcp_config.project_number,
            "zone": job_gcp_config.zone,
        },
        trigger_rule="all_done",
        execution_timeout=timedelta(minutes=10),
        owner=self.task_owner,
    )


@dataclasses.dataclass
class GPUTask(BaseTask):
  """This is a class to set up tasks for GPU.

  Attributes:
    image: the image version that a GPU runs.
    image_project: the project that an image belongs to.
    image_family: the family group that an image belongs to.
  """

  image: str
  image_project: str
  image_family: str

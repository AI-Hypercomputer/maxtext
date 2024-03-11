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
import shlex
from typing import Optional, Tuple
import airflow
from airflow.models.taskmixin import DAGNode
from airflow.utils.task_group import TaskGroup
from xlml.apis import gcp_config, metric_config, test_config
from xlml.utils import gpu, metric, name_format, ssh, tpu, xpk, gke


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
  task_metric_config: Optional[metric_config.MetricConfig] = None
  tpu_create_timeout: datetime.timedelta = datetime.timedelta(minutes=60)
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
      provision, queued_resource, ssh_keys, gcs_location = self.provision()
      # If you didn't set `MetricConfig.use_runtime_generated_gcs_folder` value in the
      # test config script then `gcs_location` will take no effect.
      if (
          self.task_metric_config
          and self.task_metric_config.use_runtime_generated_gcs_folder
      ):
        env_variable = {
            f"{metric_config.SshEnvVars.GCS_OUTPUT.value}": gcs_location
        }
      else:
        env_variable = None
      run_model = self.run_model(queued_resource, ssh_keys, env_variable)
      post_process = self.post_process(result_location=gcs_location)
      clean_up = self.clean_up(queued_resource)
      provision >> run_model >> post_process >> clean_up

    return group

  def run_with_run_name_generation(self) -> DAGNode:
    """Generate a unique run name and tensorboard file location,
    then run a test job.

    Returns:
      A task group with the following tasks chained: generate_run_name,
      generate_tb_file_location, provision, run_model, post_process,
      and clean_up.
    """
    with TaskGroup(
        group_id=self.task_test_config.benchmark_id, prefix_group_id=True
    ) as group:
      run_name = name_format.generate_run_name(
          self.task_test_config.benchmark_id
      )
      tb_file_location = name_format.generate_tb_file_location(
          run_name, self.task_metric_config.tensorboard_summary.file_location
      )

      # Set run_name in run_model_cmds
      new_run_model_cmds = [f"export M_RUN_NAME={run_name}"]
      for cmd in self.task_test_config.run_model_cmds:
        new_run_model_cmds.append(cmd)
      self.task_test_config.run_model_cmds = new_run_model_cmds

      # Update tensorboard file location
      self.task_metric_config.tensorboard_summary.file_location = (
          tb_file_location
      )

      provision, queued_resource, ssh_keys, gcs_location = self.provision()
      run_model = self.run_model(queued_resource, ssh_keys)
      post_process = self.post_process()
      clean_up = self.clean_up(queued_resource)

      (
          run_name
          >> tb_file_location
          >> provision
          >> run_model
          >> post_process
          >> clean_up
      )

    return group

  def run_with_startup_script(self) -> DAGNode:
    """Run a test job on GCE with startup script.

    Returns:
      A task group with the following tasks chained:
      provision_with_startup_script (create_queued_resource_request +
      wait_for_ready_queued_resource + check_if_startup_script_end),
      post_process and clean_up.
    """

    use_startup_script = True

    with TaskGroup(
        group_id=self.task_test_config.benchmark_id, prefix_group_id=True
    ) as group:
      (
          provision_with_startup_script,
          queued_resource,
          ssh_keys,
      ) = self.provision_with_startup_script()
      post_process = self.post_process(use_startup_script=use_startup_script)
      clean_up = self.clean_up(queued_resource)

      provision_with_startup_script >> post_process >> clean_up

    return group

  def provision(
      self,
  ) -> Tuple[DAGNode, airflow.XComArg, airflow.XComArg, airflow.XComArg]:
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
        output_location = name_format.generate_gcs_folder_location(
            self.task_test_config.benchmark_id
        )

      queued_resource_op, queued_resource_name = tpu.create_queued_resource(
          tpu_name,
          self.task_gcp_config,
          ssh_keys,
          self.tpu_create_timeout,
          self.task_test_config,
      )
      queued_resource_op >> tpu.ssh_tpu.override(task_id="setup")(
          queued_resource_name,
          # TODO(wcromar): remove split
          self.task_test_config.setup_script,
          ssh_keys,
          self.all_workers,
      )
    return group, queued_resource_name, ssh_keys, output_location

  def provision_with_startup_script(
      self,
  ) -> Tuple[DAGNode, airflow.XComArg, airflow.XComArg]:
    """Provision a TPU accelerator via a Queued Resource.

    Generates a random TPU name and SSH keys, creates a Queued Resource, and
    runs the test config's setup script on the TPU when it is ready.

    Returns:
      A DAG node that will provision a TPU, an XCom value for the qualified
      queued resource name, and an XCom value for the SSH keys.

    Raises:
      AirflowTaskTimeout: An error occurs when execution_timeout is breached.
    """
    with TaskGroup(group_id="provision_with_startup_script") as group:
      with TaskGroup(group_id="initialize"):
        tpu_name = tpu.generate_tpu_name(
            self.task_test_config.benchmark_id, self.tpu_name_env_var
        )
        ssh_keys = ssh.generate_ssh_keys()

      queued_resource_op, queued_resource_name = tpu.create_queued_resource(
          tpu_name,
          self.task_gcp_config,
          ssh_keys,
          self.tpu_create_timeout,
          self.task_test_config,
          use_startup_script=True,
      )

    return group, queued_resource_name, ssh_keys

  def run_model(
      self,
      # TODO(wcromar): Is there a way to annotate the type of the XCom arg?
      queued_resource: airflow.XComArg,
      ssh_keys: airflow.XComArg,
      env: Optional[airflow.XComArg] = None,
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
        env,
    )

  def post_process(
      self,
      use_startup_script: bool = False,
      result_location: Optional[str] = None,
  ) -> DAGNode:
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
          use_startup_script=use_startup_script,
          folder_location=result_location,
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

  def run_with_run_name_generation(self) -> DAGNode:
    """Generate a unique run name and tensorboard file location,
    then run a test job within a docker image.

    Returns:
      A task group with the following tasks chained: generate_run_name,
      generate_tb_file_location, run provision, run_model, post_process.
    """
    with TaskGroup(
        group_id=self.task_test_config.benchmark_id, prefix_group_id=True
    ) as group:
      run_name = name_format.generate_run_name(
          self.task_test_config.benchmark_id
      )
      tb_file_location = name_format.generate_tb_file_location(
          run_name, self.task_metric_config.tensorboard_summary.file_location
      )

      # Set run_name in run_model_cmds
      new_run_model_cmds = [f"export M_RUN_NAME={run_name}"]
      for cmd in self.task_test_config.run_model_cmds:
        new_run_model_cmds.append(cmd)
      self.task_test_config.run_model_cmds = new_run_model_cmds

      # Update tensorboard file location
      self.task_metric_config.tensorboard_summary.file_location = (
          tb_file_location
      )

      run_name >> tb_file_location >> self.run_model() >> self.post_process()

    return group

  def run_model(self) -> DAGNode:
    """Run the TPU test in `task_test_config` using xpk.

    Returns:
      A DAG node that executes the model test.
    """
    with TaskGroup(group_id="run_model") as group:
      workload_id = xpk.generate_workload_id(self.task_test_config.benchmark_id)
      run_workload = xpk.run_workload.override(
          owner=self.task_test_config.task_owner
      )(
          task_id="run_workload",
          cluster_project=self.task_gcp_config.project_name,
          zone=self.task_gcp_config.zone,
          cluster_name=self.task_test_config.cluster_name,
          benchmark_id=self.task_test_config.benchmark_id,
          workload_id=workload_id,
          docker_image=self.task_test_config.docker_image,
          accelerator_type=self.task_test_config.accelerator.name,
          run_cmds=self.task_test_config.test_script,
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
    task_test_config: task configutation.
    task_gcp_config: gcp related config (e.g., zone, project) for the task.
    task_metric_config: metric configuration (e.g., result gcs path).
    gpu_create_timeout: timeout when waiting for the GPU vm creation.

  """

  image_project: str
  image_family: str
  task_test_config: test_config.TestConfig[test_config.Gpu]
  task_gcp_config: gcp_config.GCPConfig
  task_metric_config: Optional[metric_config.MetricConfig] = None
  gpu_create_timeout: datetime.timedelta = datetime.timedelta(minutes=60)

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
      (
          provision,
          ip_address,
          instance_name,
          ssh_keys,
          gcs_location,
      ) = self.provision()
      # If you already specify `task_metric_config.json_lines` value in the
      # test config script, then `gcs_location` will take no effect.
      if (
          self.task_metric_config
          and self.task_metric_config.use_runtime_generated_gcs_folder
      ):
        env_variable = {
            f"{metric_config.SshEnvVars.GCS_OUTPUT.value}": gcs_location
        }
      else:
        env_variable = None
      run_model = self.run_model(ip_address, ssh_keys, env_variable)
      post_process = self.post_process(gcs_location)
      clean_up = self.clean_up(
          instance_name,
          self.task_gcp_config.project_name,
          self.task_gcp_config.zone,
      )
      provision >> run_model >> post_process >> clean_up
    return group

  def provision(
      self,
  ) -> Tuple[
      DAGNode,
      airflow.XComArg,
      airflow.XComArg,
      airflow.XComArg,
      airflow.XComArg,
  ]:
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
        gcs_location = name_format.generate_gcs_folder_location(
            self.task_test_config.benchmark_id
        )

      ip_address = gpu.create_resource(
          gpu_name,
          self.image_project,
          self.image_family,
          self.task_test_config.accelerator,
          self.task_gcp_config,
          ssh_keys,
          timeout=self.gpu_create_timeout,
      )

      ip_address >> gpu.ssh_host.override(task_id="setup")(
          ip_address,
          self.task_test_config.setup_script,
          ssh_keys,
      )

    return group, ip_address, gpu_name, ssh_keys, gcs_location

  def run_model(
      self,
      resource: airflow.XComArg,
      ssh_keys: airflow.XComArg,
      env: Optional[airflow.XComArg] = None,
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
        env,
    )

  def post_process(
      self, result_location: Optional[airflow.XComArg] = None
  ) -> DAGNode:
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
          folder_location=result_location,
      )
      return group

  def clean_up(
      self, resource: airflow.XComArg, project_id: str, zone: str
  ) -> DAGNode:
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
    return gpu.delete_resource.override(group_id="clean_up")(
        resource, project_id, zone
    )


# TODO(ranran): This class is big. Let's move it to a new file.
@dataclasses.dataclass
class GpuGkeTask(BaseTask):
  """This is a class to set up tasks for GPU on a GKE cluster.

  Attributes:
    image_project: the project that an image belongs to.
    image_family: the family group that an image belongs to.
    cluster_name: Name of the GCP cluster.
    job_create_timeout: Amount of time to wait for all pods to become active.
  """

  task_test_config: test_config.JSonnetGpuTest
  task_gcp_config: gcp_config.GCPConfig
  cluster_name: str
  job_create_timeout: datetime.timedelta = datetime.timedelta(minutes=10)
  # TODO(wcromar): job history metrics
  # task_metric_config: Optional[metric_config.MetricConfig] = None

  def run(self) -> DAGNode:
    """Run a test job.

    Returns:
      A task group that runs the given test config on a GKE cluster.
    """
    with TaskGroup(
        group_id=self.task_test_config.benchmark_id, prefix_group_id=True
    ) as group:
      job_body = self._get_job_manifest()
      gke.run_job.override(group_id="run_model")(
          job_body,
          self.task_gcp_config,
          self.cluster_name,
          self.job_create_timeout,
      )

    return group

  def _get_job_manifest(self):
    accelerator = self.task_test_config.accelerator
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "generateName": f"{self.task_test_config.benchmark_id}-",
            "labels": {
                "accelerator": accelerator.name,
                "benchmarkId": self.task_test_config.benchmark_id,
            },
        },
        "spec": {
            "activeDeadlineSeconds": int(
                datetime.timedelta(
                    minutes=self.task_test_config.time_out_in_min or 60
                ).total_seconds()
            ),
            "backoffLimit": 0,
            "completionMode": "Indexed",
            "completions": self.task_test_config.num_hosts,
            "parallelism": self.task_test_config.num_hosts,
            "template": {
                "metadata": {
                    # Matches `headless-svc` in GKE cluster.
                    # See deployments directory.
                    "labels": {"headless-svc": "true"},
                },
                "spec": {
                    "subdomain": "headless-svc",
                    "nodeSelector": {
                        "cloud.google.com/gke-accelerator": accelerator.accelerator_type,
                    },
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "main",
                            "image": self.task_test_config.docker_image,
                            "imagePullPolicy": "Always",
                            "command": shlex.split(
                                self.task_test_config.setup_script
                            ),
                            "args": shlex.split(
                                self.task_test_config.test_script
                            ),
                            "resources": {
                                "limits": {
                                    "nvidia.com/gpu": accelerator.count,
                                }
                            },
                            "env": [
                                {
                                    "name": "POD_NAME",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "metadata.name"
                                        }
                                    },
                                },
                                {
                                    "name": "POD_NAMESPACE",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "metadata.namespace"
                                        }
                                    },
                                },
                                {
                                    "name": "JOB_NAME",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "metadata.labels['job-name']"
                                        }
                                    },
                                },
                            ],
                            "volumeMounts": [
                                {
                                    "mountPath": "/dev/shm",
                                    "name": "dshm",
                                    "readOnly": False,
                                },
                            ],
                        },
                    ],
                    "volumes": [
                        {"emptyDir": {"medium": "Memory"}, "name": "dshm"},
                    ],
                },
            },
        },
    }

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

"""Utilities to run workloads with xpk (https://github.com/google/xpk)."""

import uuid
import os
from absl import logging
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from datetime import timedelta
from kubernetes import client as k8s_client
from kubernetes.client import models as k8s_models
from xlml.utils import gke


@task
def generate_workload_id(benchmark_id: str) -> str:
  """Generate a valid workload ID."""
  import re

  short_id = str(uuid.uuid4())[:8]
  # Remove all non-alphanumeric characters, and truncate to ensure the result
  # is less than 40 characters.
  short_benchmark = re.sub(r"[^a-zA-Z0-9-]+", "", benchmark_id)[:32]
  return f"{short_benchmark}{short_id}"


def _run_workload(
    task_id: str,
    cluster_project: str,
    zone: str,
    cluster_name: str,
    benchmark_id: str,
    workload_id: str,
    docker_image: str,
    accelerator_type: str,
    run_cmds: str,
    num_slices: int = 1,
):
  """Run workload through xpk tool."""
  import subprocess

  cmds = (
      "set -xue",
      "git clone https://github.com/google/xpk.git /tmp/xpk",
      "cd /tmp/xpk",
      "gcloud components install kubectl",
      f"gcloud config set project {cluster_project}",
      f"gcloud config set compute/zone {zone}",
      (
          "python3 xpk.py workload create"
          f" --cluster={cluster_name} --workload={workload_id}"
          f" --command='{run_cmds}' --tpu-type={accelerator_type}"
          f" --num-slices={num_slices} --docker-image={docker_image}"
      ),
  )

  subprocess.run(["bash", "-c", ";".join(cmds)], check=True)


# To support local execution using `scripts/local-airflow.sh`, run using
# task.docker in local environments. Otherwise, task.kubernetes should be used.
if "XLMLTEST_LOCAL_AIRFLOW" in os.environ:
  run_workload = task.docker(
      _run_workload, image="google/cloud-sdk:alpine", auto_remove="force"
  )
else:
  run_workload = task.kubernetes(
      _run_workload,
      namespace="composer-user-workloads",
      image="google/cloud-sdk:alpine",
      config_file="/home/airflow/composer_kube_config",
      kubernetes_conn_id="kubernetes_default",
      container_resources=k8s_models.V1ResourceRequirements(
          limits={"ephemeral-storage": "10G"},
      ),
  )


# XPK tests are scheduled by Kueue. Allow up to 20 hours for the workload
# to finish, since it may wait in the queue.
@task.sensor(
    poke_interval=60,
    timeout=timedelta(hours=20).total_seconds(),
    mode="reschedule",
)
def wait_for_workload_completion(
    workload_id: str, project_id: str, region: str, cluster_name: str
) -> bool:
  """Check the workload status."""
  client = gke.get_authenticated_client(project_id, region, cluster_name)

  # Initilize the client
  core_api = k8s_client.CoreV1Api(client)
  logging.info("Successful initilize k8s client from cluster response.")

  # Get pods for the workload
  logging.info(f"Getting pods for workload_id: {workload_id}")
  pods = core_api.list_namespaced_pod(
      label_selector=f"jobset.sigs.k8s.io/jobset-name={workload_id}",
      namespace="default",
  )

  # Check status of pods
  if not pods.items:
    # This could happen when workload is in the queue (not initialized yet)
    logging.info(f"No pod is found for workload selector: {workload_id}.")
    return False

  logging.info(f"pods: {pods}")
  for pod in pods.items:
    if pod.status.phase in ["Pending", "Running"]:
      logging.info(f"One pod phase is: {pod.status.phase}")
      return False
    elif pod.status.phase == "Failed":
      # Don't keep retrying if the pod has failed
      raise AirflowFailException(f"Bad pod phase: {pod.status.phase}")
    elif pod.status.phase in ["Unknown"]:
      raise RuntimeError(f"Bad pod phase: {pod.status.phase}")

  logging.info("All pod(s) phase are succeeded.")
  return True

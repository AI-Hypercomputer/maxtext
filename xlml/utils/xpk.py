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

import base64
from tempfile import NamedTemporaryFile
import uuid
from absl import logging
from airflow.decorators import task
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
import google.auth
import google.auth.transport.requests
from google.cloud import container_v1
from kubernetes import client as k8s_client


@task
def generate_workload_id(benchmark_id: str) -> str:
  """Generate a workload ID."""
  short_id = str(uuid.uuid4())[:8]
  return f"{benchmark_id}-{short_id}"


def run_workload(
    task_id: str,
    cluster_project: str,
    zone: str,
    cluster_name: str,
    benchmark_id: str,
    workload_id: str,
    docker_image: str,
    accelerator_type: str,
    run_cmds: str,
    task_owner: str,
    startup_timeout: int,
    num_slices: int = 1,
) -> KubernetesPodOperator:
  """Run workload through xpk tool.

  The reason to use KubernetesPodOperator instead of BashOperator is that
  xpk must run with Python 3.10 or greater; however, the latest version in
  Composer is Python 3.8, and it's non-trivial to upgrade it as the  Composer
  uses docker images that bundle Airflow releases with Python and other
  libraries.
  """

  cmds = (
      "set -x",
      f"gcloud config set project {cluster_project}",
      f"gcloud config set compute/zone {zone}",
      "git clone https://github.com/google/xpk.git /tmp/xpk",
      "cd /tmp/xpk",
      (
          "python3 xpk.py workload create"
          f" --cluster={cluster_name} --workload={workload_id} --command='{run_cmds}'"
          f" --tpu-type={accelerator_type} --num-slices={num_slices} --docker-image={docker_image}"
      ),
  )

  return KubernetesPodOperator(
      task_id=task_id,
      name=benchmark_id,
      cmds=["/bin/bash", "-c"],
      arguments=[";".join(cmds)],
      namespace="composer-user-workloads",
      image=docker_image,
      config_file="/home/airflow/composer_kube_config",
      kubernetes_conn_id="kubernetes_default",
      startup_timeout_seconds=startup_timeout,
      owner=task_owner,
  )


@task.sensor(poke_interval=60, timeout=600, mode="reschedule")
def wait_for_workload_completion(
    workload_id: str, project_id: str, region: str, cluster_name: str
) -> bool:
  """Check the workload status."""

  # Get cluster configuration
  container_client = container_v1.ClusterManagerClient()
  cluster_path = f"projects/{project_id}/locations/{region}/clusters/{cluster_name}"
  response = container_client.get_cluster(name=cluster_path)
  creds, _ = google.auth.default()
  auth_req = google.auth.transport.requests.Request()
  creds.refresh(auth_req)
  configuration = k8s_client.Configuration()
  configuration.host = f"https://{response.endpoint}"
  with NamedTemporaryFile(delete=False) as ca_cert:
    ca_cert.write(base64.b64decode(response.master_auth.cluster_ca_certificate))
  configuration.ssl_ca_cert = ca_cert.name
  configuration.api_key_prefix["authorization"] = "Bearer"
  configuration.api_key["authorization"] = creds.token

  # Initilize the client
  core_api = k8s_client.CoreV1Api(k8s_client.ApiClient(configuration))
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
    elif pod.status.phase in ["Failed", "Unknown"]:
      raise RuntimeError(f"Bad pod phase: {pod.status.phase}")

  logging.info("All pod(s) phase are succeeded.")
  return True

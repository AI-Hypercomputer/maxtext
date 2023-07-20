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

"""Utilities to integrate with TPU.

TODO(ranran): update REST request to client lib.
"""

import functools
import os
import time
from typing import Any, Iterable, List, Mapping
from absl import logging
from airflow.hooks.subprocess import SubprocessHook
# import billiard as multiprocessing to workaround known issue in Airflow
#  https://github.com/apache/airflow/issues/14896#issuecomment-908516288
import billiard as multiprocessing
from fabric import Connection
import google.auth
import google.auth.transport.requests
import requests


_TPU_BASE_URL = "https://tpu.googleapis.com/v2alpha1/"
_SSH_KEYS_PATH = os.path.expanduser("~/.ssh/google_compute_engine")


def get_headers() -> Mapping[str, str]:
  """Get request headers.

  Returns:
    A dict mapping credentials.
  """
  creds, _ = google.auth.default(
      scopes=["https://www.googleapis.com/auth/cloud-platform"]
  )
  creds.refresh(google.auth.transport.requests.Request())
  return {"Authorization": f"Bearer {creds.token}"}


def create_qr(
    qr_name: str,
    tpu_name: str,
    project_number: str,
    zone: str,
    tpu_type: str,
    runtime_version: str,
    network: str,
    subnetwork: str,
    reserved: bool,
) -> None:
  """Create a Queued Resource.

  Send REST request to create a Queued Resource, and keep
  checking the status every 30 seconds till a TPU is ready.

  Args:
    qr_name: The name of a Queued Resource to be created.
    tpu_name: The name of a TPU to be created.
    project_number: The number of a project to create a TPU.
    zone: The zone to create a TPU.
    tpu_type: The type of a TPU.
    runtime_version: The version of a TPU that runs.
    network: The network that a TPU will be a part of.
    subnetwork: The subnetwork that a TPU will be a part of.
    reserved: The flag to define if a TPU is a Cloud reservation.
  """
  parent = os.path.join("projects", project_number, "locations", zone)
  qr_url = os.path.join(
      _TPU_BASE_URL,
      "projects",
      project_number,
      "locations",
      zone,
      "queuedResources",
  )
  params = {"queued_resource_id": qr_name}
  reqest_json = {
      "tpu": {
          "node_spec": {
              "parent": parent,
              "node_id": tpu_name,
              "node": {
                  "accelerator_type": tpu_type,
                  "runtime_version": runtime_version,
                  "network_config": {
                      "enableExternalIps": True,
                      "network": network,
                      "subnetwork": subnetwork,
                  },
              },
          }
      },
      "guaranteed": {"reserved": reserved},
  }
  print("Request to create Queued Resource:", reqest_json)
  resp = requests.post(
      url=qr_url, params=params, json=reqest_json, headers=get_headers()
  )
  resp.raise_for_status()
  create_op_url = os.path.join(qr_url, qr_name)

  while True:
    resp = requests.get(create_op_url, headers=get_headers())
    if resp.json()["state"]["state"] == "ACTIVE":
      logging.info("Create Queued Resource operation complete.")
      break
    logging.info("Create Queued Resource operation still running...")
    time.sleep(30)


def delete_tpu(tpu_name: str, project_number: str, zone: str) -> None:
  """Delete a TPU.

  Send REST request to delete a TPU, and keep
  checking the status every 30 seconds till a TPU is deleted.

  Args:
    tpu_name: The name of a TPU to be deleted.
    project_number: The number of a project to delete a TPU.
    zone: The zone to delete a TPU.
  """
  tpu_url = os.path.join(
      _TPU_BASE_URL,
      "projects",
      project_number,
      "locations",
      zone,
      "nodes",
      tpu_name,
  )
  resp = requests.delete(tpu_url, headers=get_headers())
  resp.raise_for_status()
  delete_op_url = os.path.join(_TPU_BASE_URL, resp.json()["name"])

  while True:
    resp = requests.get(delete_op_url, headers=get_headers())
    if resp.json()["done"]:
      logging.info("Delete TPU operation complete.")
      break
    logging.info("Delete TPU operation still running...")
    time.sleep(30)


def delete_qr(qr_name: str, project_number: str, zone: str) -> None:
  """Delete a Queued Resource.

  Send REST request to check Queued Resource status, and delete it. Keep
  checking the status every 30 seconds.

  Args:
    qr_name: The name of a Queued Resource to be deleted.
    project_number: The number of a project to delete a Queued Resource.
    zone: The zone to delete a Queued Resource.
  """
  qr_url = os.path.join(
      _TPU_BASE_URL,
      "projects",
      project_number,
      "locations",
      zone,
      "queuedResources",
      qr_name,
  )

  # check if Queued Resource has become SUSPENDED from SUSPENDING
  check_resp = requests.get(qr_url, headers=get_headers())
  check_resp.raise_for_status()

  check_op_url = os.path.join(_TPU_BASE_URL, check_resp.json()["name"])
  while True:
    resp = requests.get(check_op_url, headers=get_headers())
    if resp.json()["state"]["state"] == "SUSPENDED":
      logging.info("Queued Resource is in SUSPENDED status.")
      break
    logging.info("Check Queued Resource operation still running...")
    time.sleep(30)

  # delete Queued Resource
  delete_resp = requests.delete(qr_url, headers=get_headers())
  delete_resp.raise_for_status()

  delete_op_url = os.path.join(_TPU_BASE_URL, delete_resp.json()["name"])
  while True:
    resp = requests.get(delete_op_url, headers=get_headers())
    if resp.json()["done"]:
      logging.info("Delete Queued Resource operation complete.")
      break
    logging.info("Delete Queued Resource operation still running...")
    time.sleep(30)


def get_tpu(tpu_name: str, project_number: str, zone: str) -> Mapping[str, Any]:
  """Get TPU node information.

  Args:
    tpu_name: The name of a TPU.
    project_number: The number of a project that a TPU runs.
    zone: The zone of a project that a TPU runs.

  Returns:
    TPU node information in JSON format.
  """
  tpu_url = os.path.join(
      _TPU_BASE_URL,
      "projects",
      project_number,
      "locations",
      zone,
      "nodes",
      tpu_name,
  )
  resp = requests.get(tpu_url, headers=get_headers())
  return resp.json()


# TODO(wcromar): Update logic to generate a unique SSH key for each test job
def config_ssh_key() -> None:
  """Config the ssh key for connection.

  Raises:
    RuntimeError: An error occurs when configuring a SSH key.
  """
  if os.path.exists(_SSH_KEYS_PATH):
    return

  config_ssh_cmd = [
      "gcloud",
      "compute",
      "config-ssh",
      f"--ssh-key-file={_SSH_KEYS_PATH}",
      "--quiet",
  ]
  hook = SubprocessHook()
  result = hook.run_command(config_ssh_cmd)

  if result.exit_code != 0:
    raise RuntimeError(
        f"The exit code is {result.exit_code} with error: {result.output}"
    )


def get_ip_address(tpu_name: str, project_number: str, zone: str) -> List[str]:
  """Get TPU node information.

  Args:
    tpu_name: The name of a TPU.
    project_number: The number of a project that a TPU runs.
    zone: The zone of a project that a TPU runs.

  Returns:
    A list of IP addresses for all workers.
  """
  tpu_node = get_tpu(tpu_name, project_number, zone)
  ip_addresses = []
  for endpoint in tpu_node["networkEndpoints"]:
    ip_addresses.append(endpoint["ipAddress"])
  return ip_addresses


def create_connect(ip_address: str) -> Connection:
  """Create connection for an IP address."""
  return Connection(ip_address, connect_kwargs={"key_filename": _SSH_KEYS_PATH})


def get_connection(ip_addresses: List[str]) -> Mapping[str, Connection]:
  """Get connection for all IP addresses."""
  config_ssh_key()
  connections = {}
  for ip_address in ip_addresses:
    connections[ip_address] = create_connect(ip_address)
  return connections


def subprocess_helper(tpu_connection, cmds):
  """A helper to run commands in on TPU worker."""
  for cmd in cmds:
    if cmd.startswith("sudo"):
      tpu_connection.sudo(cmd[5:])
    else:
      tpu_connection.run(cmd)


def ssh_tpu(
    tpu_name: str, project_number: str, zone: str, cmds: Iterable[str]
) -> None:
  """SSH TPU and run commands in multi process.

  Args:
   tpu_name: The name of a TPU.
   project_number: The number of a project that a TPU runs.
   zone: The zone of a project that a TPU runs.
   cmds: The commands to run on a TPU.
  """
  tpu_ip_addresses = get_ip_address(tpu_name, project_number, zone)
  tpu_connections = get_connection(tpu_ip_addresses)

  # TODO(ranran): handle disconnection due to maintenance event
  with multiprocessing.Pool(processes=len(tpu_ip_addresses)) as p:
    p.map(
        functools.partial(subprocess_helper, cmds=cmds),
        tpu_connections.values(),
    )


def provision(
    task_id_suffix: str,
    project_number: str,
    zone: str,
    type: str,
    runtime_version: str,
    network: str,
    subnetwork: str,
    reserved: bool,
    set_up_cmd: Iterable[str],
    **kwargs,
) -> None:
  """Create a TPU and run set up commands.

  Args:
    task_id_suffix: The ID suffix for clean up.
    project_number: The number of a project to clean up.
    zone: The zone to clean up.
    type: The type of a TPU, i.e. v4-8.
    runtime_version: The version of a runtime image that a TPU runs.
    network: The network that a TPU will be a part of.
    subnetwork: The subnetwork that a TPU will be a part of.
    reserved: The flag to define if a TPU is a Cloud reservation.
    set_up_cmd: The commands to set up a TPU.
    kwargs: a set of keyword arguments in Airflow context.
  """
  del kwargs
  create_qr(
      f"{task_id_suffix}_qr",
      f"{task_id_suffix}_tpu",
      project_number,
      zone,
      type,
      runtime_version,
      network,
      subnetwork,
      reserved,
  )
  ssh_tpu(f"{task_id_suffix}_tpu", project_number, zone, set_up_cmd)


def run_model(
    task_id_suffix: str,
    project_number: str,
    zone: str,
    run_model_cmd: Iterable[str],
    **kwargs,
) -> None:
  """SSH to TPU to run scripts.

  Args:
    task_id_suffix: The ID suffix for clean up.
    project_number: The number of a project to clean up.
    zone: The zone to clean up.
    run_model_cmd: The commands to run model.
    kwargs: a set of keyword arguments in Airflow context.
  """
  del kwargs
  ssh_tpu(f"{task_id_suffix}_tpu", project_number, zone, run_model_cmd)


def clean_up(
    task_id_suffix: str, project_number: str, zone: str, **kwargs
) -> None:
  """Delete a TPU and a Queued Resource.

  Args:
    task_id_suffix: The ID suffix for clean up.
    project_number: The number of a project to clean up.
    zone: The zone to clean up.
    kwargs: a set of keyword arguments in Airflow context.
  """
  del kwargs
  delete_tpu(f"{task_id_suffix}_tpu", project_number, zone)
  delete_qr(f"{task_id_suffix}_qr", project_number, zone)

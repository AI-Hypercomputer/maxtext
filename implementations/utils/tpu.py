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

import datetime
import functools
import io
import os
import time
from typing import Any, Iterable, List, Mapping
import uuid
from absl import logging
from airflow.decorators import task
import fabric
import google.auth
import google.auth.transport.requests
import paramiko
import requests
from apis import test_config
from implementations.utils import ssh


_TPU_BASE_URL = "https://tpu.googleapis.com/v2alpha1/"

@task
def generate_tpu_name(base_tpu_name: str) -> str:
    return f'{base_tpu_name}-{str(uuid.uuid4())}'

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

@task
def create_qr(
    accelerator: test_config.Tpu,
    tpu_name: str,
    zone: str,
    project_number: str,
    ssh_keys: ssh.SshKeys,
) -> None:
  """Create a Queued Resource.

  Send REST request to create a Queued Resource, and keep
  checking the status every 30 seconds till a TPU is ready.

  Args:
    accelerator: the TPU type to create.
    tpu_name: The name of a TPU to be created.
    zone: The zone to create a TPU.
    project_number: The number of a project to create a TPU.
    ssh_keys: SSH key pair to encode in TPU metadata.
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
  params = {"queued_resource_id": tpu_name}
  reqest_json = {
      "tpu": {
          "node_spec": {
              "parent": parent,
              "node_id": tpu_name,
              "node": {
                  "accelerator_type": accelerator.name,
                  "runtime_version": accelerator.runtime_version,
                  "network_config": {
                      "enableExternalIps": True,
                      "network": accelerator.network,
                      "subnetwork": accelerator.subnetwork,
                  },
                  "metadata": {
                    "ssh-keys": f"xl-ml-test:{ssh_keys.public}"
                  }
              },
          }
      },
      "guaranteed": {"reserved": accelerator.reserved},
  }
  print("Request to create Queued Resource:", reqest_json)
  resp = requests.post(
      url=qr_url, params=params, json=reqest_json, headers=get_headers()
  )
  resp.raise_for_status()
  create_op_url = os.path.join(qr_url, tpu_name)

  while True:
    resp = requests.get(create_op_url, headers=get_headers())
    if resp.json()["state"]["state"] == "ACTIVE":
      logging.info("Create Queued Resource operation complete.")
      break
    logging.info("Create Queued Resource operation still running...")
    time.sleep(30)

@task(trigger_rule="all_done")
def delete_tpu(tpu_name: str, zone: str, project_number: str) -> None:
  """Delete a TPU.

  Send REST request to delete a TPU, and keep
  checking the status every 30 seconds till a TPU is deleted.

  Args:
    tpu_name: The name of a TPU to be deleted.
    zone: The zone to delete a TPU.
    project_number: The number of a project to delete a TPU.
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

@task(trigger_rule="all_done")
def delete_qr(tpu_name: str, zone: str, project: str) -> None:
  """Delete a Queued Resource.

  Send REST request to check Queued Resource status, and delete it. Keep
  checking the status every 30 seconds.

  Args:
    tpu_name: The name of a Queued Resource to be deleted.
    zone: The zone of the Queued Resource to be deleted.
    project: The project of the Queued Resource to be deleted.
  """
  qr_url = os.path.join(
      _TPU_BASE_URL,
      "projects",
      project,
      "locations",
      zone,
      "queuedResources",
      tpu_name
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
  resp.raise_for_status()
  return resp.json()


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


@task
def ssh_tpu(
    tpu_name: str,
    zone: str,
    project_number: str,
    cmds: Iterable[str],
    ssh_keys: ssh.SshKeys
) -> None:
  """SSH TPU and run commands in multi process.

  Args:
   tpu_name: The name of a TPU.
   zone: The zone of a project that a TPU runs.
   project_number: The number of a project that a TPU runs.
   cmds: The commands to run on a TPU.
   ssh_keys: The SSH key pair to use for authentication.
  """
  tpu_ip_addresses = get_ip_address(tpu_name, project_number, zone)

  pkey = paramiko.RSAKey.from_private_key(io.StringIO(ssh_keys.private))
  ssh_group = fabric.ThreadingGroup(
    *tpu_ip_addresses,
    connect_kwargs={
      "auth_strategy":
        paramiko.auth_strategy.InMemoryPrivateKey('xl-ml-test', pkey)
    }
  )
  ssh_group.run('\n'.join(cmds))

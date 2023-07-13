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

import os
import time
from typing import Iterable, Mapping
from absl import logging
import google.auth
import google.auth.transport.requests
import requests


_TPU_BASE_URL = "https://tpu.googleapis.com/v2alpha1/"


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
          "node_spec": [{
              "parent": parent,
              "node_id": tpu_name,
              "node": {
                  "accelerator_type": tpu_type,
                  "runtime_version": runtime_version,
              },
          }]
      }
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


# TODO(ranran): Add implementation of ssh
def ssh() -> None:
  raise NotImplementedError


# TODO(ranran): Add implementation to set up a TPU.
def provision(
    task_id_suffix: str,
    project_name: str,
    project_number: str,
    zone: str,
    type: str,
    runtime_version: str,
    set_up_cmd: Iterable[str],
    **kwargs,
) -> None:
  del kwargs
  create_qr(
      f"{task_id_suffix}_qr",
      f"{task_id_suffix}_tpu",
      project_number,
      zone,
      type,
      runtime_version,
  )


def clean_up(
    task_id_suffix: str, project_number: str, zone: str, **kwargs
) -> None:
  """Delete a TPU and a Queued Resource.

  Args:
    task_id_suffix: The ID suffix for clean up.
    project_number: The number of a project to clean up.
    zone: The zone to clean up.
  """
  del kwargs
  delete_tpu(f"{task_id_suffix}_tpu", project_number, zone)
  delete_qr(f"{task_id_suffix}_qr", project_number, zone)

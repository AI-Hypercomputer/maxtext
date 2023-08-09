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

"""Utilities to create, delete, and SSH with TPUs."""

import datetime
import io
import os
from typing import Iterable, Optional, Tuple
import uuid

from absl import logging
import airflow
from airflow.decorators import task, task_group
from airflow.utils.task_group import TaskGroup
import fabric
import google.api_core.exceptions
import google.auth
import google.cloud.tpu_v2alpha1 as tpu_api
import google.longrunning.operations_pb2 as operations
import paramiko

from apis import gcp_config, test_config
from implementations.utils import ssh


@task
def generate_tpu_name(base_tpu_name: str) -> str:
    return f'{base_tpu_name}-{str(uuid.uuid4())}'

def create_queued_resource(tpu_name: airflow.XComArg, accelerator: test_config.Tpu, gcp: gcp_config.GCPConfig, ssh_keys: airflow.XComArg, timeout: datetime.timedelta) -> Tuple[TaskGroup, airflow.XComArg]:
  """Request a QueuedResource and wait until the nodes are created.

  Args:
    tpu_name: XCom value for unique TPU name
    accelerator: Description of TPU to create.
    gcp: GCP project/zone configuration.
    ssh_keys: XCom value for SSH keys to communicate with these TPUs.
    timeout: Amount of time to wait for TPUs to be created.

  Returns:
    A TaskGroup for the entire create operation and an XCom value for the
    qualified queued_resource name.
  """
  @task
  def create_queued_resource_request(tpu_name: str, ssh_keys: ssh.SshKeys) -> str:
    creds, _ = google.auth.default()
    client = tpu_api.TpuClient(credentials=creds)

    parent = f'projects/{gcp.project_name}/locations/{gcp.zone}'
    queued_resource = tpu_api.QueuedResource(
        # TODO(wcromar): Implement `validUntilDuration` based on `timeout`
        tpu=tpu_api.QueuedResource.Tpu(
            node_spec=[
                tpu_api.QueuedResource.Tpu.NodeSpec(
                    node_id=tpu_name,
                    parent=parent,
                    node=tpu_api.Node(
                        accelerator_type=accelerator.name,
                        description="noteardown",
                        runtime_version=accelerator.runtime_version,
                        network_config=tpu_api.NetworkConfig(
                            network=accelerator.network,
                            subnetwork=accelerator.subnetwork,
                            enable_external_ips=True,
                        ),
                        metadata={
                          'ssh-keys': f'xl-ml-test:{ssh_keys.public}',
                        }
                    )
                )
            ],
        ),
        guaranteed=tpu_api.QueuedResource.Guaranteed(
            reserved=accelerator.reserved,
        ),
    )

    qr_operation = client.create_queued_resource(
      parent=parent,
      queued_resource_id=tpu_name,
      queued_resource=queued_resource)
    response = qr_operation.result()
    logging.info("Create QR response: {}".format(response))
    # TODO(wcromar): do anything about failures

    return response.name

  @task.sensor(poke_interval=60, timeout=timeout.total_seconds(), mode="reschedule")
  def wait_for_ready_queued_resource(qualified_name: str):
    creds, _ = google.auth.default()
    client = tpu_api.TpuClient(credentials=creds)

    qr = client.get_queued_resource(name=qualified_name)
    state = qr.state.state
    logging.info(f"Queued resource state: {state.name}")
    if qr.state.state == tpu_api.QueuedResourceState.State.ACTIVE:
      return True
    elif qr.state.state in [
        tpu_api.QueuedResourceState.State.CREATING,
        tpu_api.QueuedResourceState.State.ACCEPTED,
        tpu_api.QueuedResourceState.State.PROVISIONING]:
      return False
    else:
      raise RuntimeError(f"Bad queued resource state {state.name}")

  with TaskGroup(group_id="create_queued_resource") as tg:
    qualified_name = create_queued_resource_request(tpu_name, ssh_keys)
    wait_for_ready_queued_resource(qualified_name)

  return tg, qualified_name


@task_group
def delete_queued_resource(qualified_name: airflow.XComArg):
  """Implements cascading delete for a Queued Resource.

  Args:
    qualified_name: XCom value holding the qualified name of the queued
      resource.
  """

  @task(trigger_rule="all_done")
  def delete_tpu_nodes_request(qualified_name: str):
    # TODO(wcromar): Find a less repetitive way to manage the TPU client.
    creds, _ = google.auth.default()
    client = tpu_api.TpuClient(credentials=creds)

    try:
      qr = client.get_queued_resource(name=qualified_name)
    except google.api_core.exceptions.NotFound:
      logging.info(f'{qualified_name} not found')
      return

    for node in qr.tpu.node_spec:
      try:
        op = client.delete_node(name=f'{node.parent}/nodes/{node.node_id}')
        logging.info('Delete node state: {}'.format(op))
      except google.api_core.exceptions.NotFound:
        logging.info(f'{node.node_id} is already deleted')

  @task.sensor(poke_interval=60, timeout=3600, mode="reschedule")
  def wait_for_tpu_deletion(qualified_name: str):
    creds, _ = google.auth.default()
    client = tpu_api.TpuClient(credentials=creds)

    qr = client.get_queued_resource(name=qualified_name)
    # Queued Resources can only be deleted once they are SUSPENDED, even if all
    # underlying nodes have already been deleted.
    if qr.state.state in [
        tpu_api.QueuedResourceState.State.SUSPENDED,
        # TPU will be sitting in ACCEPTED if creation timed out.
        tpu_api.QueuedResourceState.State.ACCEPTED]:
      logging.info(f'All TPU nodes deleted for {qualified_name}')
      return True

    logging.info(f"TPU Nodes: {qr.tpu.node_spec}")
    return False

  @task(trigger_rule="all_done")
  def delete_queued_resource_request(qualified_name: str) -> Optional[str]:
    creds, _ = google.auth.default()
    client = tpu_api.TpuClient(credentials=creds)

    try:
      op = client.delete_queued_resource(name=qualified_name)
      logging.info(f"delete op {op}")
    except google.api_core.exceptions.NotFound:
      logging.info(f'{qualified_name} is already deleted')
      return None

    return op.operation.name

  @task.sensor(poke_interval=60, timeout=3600, mode="reschedule")
  def wait_for_queued_resource_deletion(op_name: Optional[str]):
    if not op_name:
      logging.info('No delete operation given')
      return True

    creds, _ = google.auth.default()
    client = tpu_api.TpuClient(credentials=creds)

    op = client.get_operation(
        operations.GetOperationRequest(name=op_name))
    return op.done

  delete_tpu_nodes = delete_tpu_nodes_request(qualified_name) >> wait_for_tpu_deletion(qualified_name)
  qr_op_name = delete_tpu_nodes >> delete_queued_resource_request(qualified_name)
  wait_for_queued_resource_deletion(qr_op_name)

@task
def ssh_tpu(
    tpu_name: str,
    zone: str,
    project: str,
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
  creds, _ = google.auth.default()
  client = tpu_api.TpuClient(credentials=creds)

  qualified_tpu_name = os.path.join(
      "projects",
      project,
      "locations",
      zone,
      "nodes",
      tpu_name,
  )
  node = client.get_node(name=qualified_tpu_name)

  pkey = paramiko.RSAKey.from_private_key(io.StringIO(ssh_keys.private))
  ssh_group = fabric.ThreadingGroup(
    *(endpoint.ip_address for endpoint in node.network_endpoints),
    connect_kwargs={
      "auth_strategy":
        paramiko.auth_strategy.InMemoryPrivateKey('xl-ml-test', pkey)
    }
  )
  ssh_group.run('\n'.join(cmds))

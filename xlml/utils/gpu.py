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

"""Utilities to create, delete, and SSH with GPUs."""

from __future__ import annotations


from absl import logging
import airflow
from airflow.decorators import task, task_group
import fabric
from google.api_core.extended_operation import ExtendedOperation
from google.cloud import compute_v1
import io
import paramiko
import re
from typing import Iterable, Any
import uuid
from xlml.apis import gcp_config, test_config
from xlml.utils import ssh


def get_image_from_family(project: str, family: str) -> compute_v1.Image:
  """
  Retrieve the newest image that is part of a given family in a project.

  Args:
    project: project ID or project number of the Cloud project you want to get image from.
    family: name of the image family you want to get image from.

  Returns:
    An Image object.
  """
  image_client = compute_v1.ImagesClient()
  # List of public operating system (OS) images: https://cloud.google.com/compute/docs/images/os-details
  newest_image = image_client.get_from_family(project=project, family=family)
  return newest_image


def disk_from_image(
    disk_type: str,
    disk_size_gb: int,
    boot: bool,
    source_image: str,
    auto_delete: bool = True,
) -> compute_v1.AttachedDisk:
  """
  Create an AttachedDisk object to be used in VM instance creation. Uses an image as the
  source for the new disk.

  Args:
    disk_type: the type of disk you want to create. This value uses the following format:
        "zones/{zone}/diskTypes/(pd-standard|pd-ssd|pd-balanced|pd-extreme)".
        For example: "zones/us-west3-b/diskTypes/pd-ssd"
    disk_size_gb: size of the new disk in gigabytes
    boot: boolean flag indicating whether this disk should be used as a boot disk of an instance
    source_image: source image to use when creating this disk. You must have read access to this disk. This can be one
        of the publicly available images or an image from one of your projects.
        This value uses the following format: "projects/{project_name}/global/images/{image_name}"
    auto_delete: boolean flag indicating whether this disk should be deleted with the VM that uses it

  Returns:
    AttachedDisk object configured to be created using the specified image.
  """
  boot_disk = compute_v1.AttachedDisk()
  initialize_params = compute_v1.AttachedDiskInitializeParams()
  initialize_params.source_image = source_image
  initialize_params.disk_size_gb = disk_size_gb
  initialize_params.disk_type = disk_type
  boot_disk.initialize_params = initialize_params
  # Remember to set auto_delete to True if you want the disk to be deleted when you delete
  # your VM instance.
  boot_disk.auto_delete = auto_delete
  boot_disk.boot = boot
  return boot_disk


def create_instance(
    project_id: str,
    zone: str,
    instance_name: str,
    disks: list[compute_v1.AttachedDisk],
    machine_type: str,
    network_link: str = "global/networks/default",
    subnetwork_link: str = None,
    internal_ip: str = None,
    external_access: bool = False,
    external_ipv4: str = None,
    accelerators: list[compute_v1.AcceleratorConfig] = None,
    metadata: compute_v1.Metadata = None,
    service_account: compute_v1.ServiceAccount = None,
    spot: bool = False,
    instance_termination_action: str = "STOP",
    custom_hostname: str = None,
    delete_protection: bool = False,
) -> compute_v1.Instance:
  """
  Send an instance creation request to the Compute Engine API and wait for it to complete.

  Args:
      project_id: project ID or project number of the Cloud project you want to use.
      zone: name of the zone to create the instance in. For example: "us-west3-b"
      instance_name: name of the new virtual machine (VM) instance.
      disks: a list of compute_v1.AttachedDisk objects describing the disks
          you want to attach to your new instance.
      machine_type: machine type of the VM being created. This value uses the
          following format: "zones/{zone}/machineTypes/{type_name}".
          For example: "zones/europe-west3-c/machineTypes/f1-micro"
      network_link: name of the network you want the new instance to use.
          For example: "global/networks/default" represents the network
          named "default", which is created automatically for each project.
      subnetwork_link: name of the subnetwork you want the new instance to use.
          This value uses the following format:
          "regions/{region}/subnetworks/{subnetwork_name}"
      internal_ip: internal IP address you want to assign to the new instance.
          By default, a free address from the pool of available internal IP addresses of
          used subnet will be used.
      external_access: boolean flag indicating if the instance should have an external IPv4
          address assigned.
      external_ipv4: external IPv4 address to be assigned to this instance. If you specify
          an external IP address, it must live in the same region as the zone of the instance.
          This setting requires `external_access` to be set to True to work.
      accelerators: a list of AcceleratorConfig objects describing the accelerators that will
          be attached to the new instance.
      metadata: Sets up metadata of the instance.
      service_account: Sets up service account email address and scopes.
      spot: boolean value indicating if the new instance should be a Spot VM or not.
      instance_termination_action: What action should be taken once a Spot VM is terminated.
          Possible values: "STOP", "DELETE"
      custom_hostname: Custom hostname of the new VM instance.
          Custom hostnames must conform to RFC 1035 requirements for valid hostnames.
      delete_protection: boolean value indicating if the new virtual machine should be
          protected against deletion or not.
  Returns:
      Instance object.
  """
  instance_client = compute_v1.InstancesClient()

  # Use the network interface provided in the network_link argument.
  network_interface = compute_v1.NetworkInterface()
  network_interface.network = network_link
  if subnetwork_link:
    network_interface.subnetwork = subnetwork_link

  if internal_ip:
    network_interface.network_i_p = internal_ip

  if external_access:
    access = compute_v1.AccessConfig()
    access.type_ = compute_v1.AccessConfig.Type.ONE_TO_ONE_NAT.name
    access.name = "External NAT"
    access.network_tier = access.NetworkTier.PREMIUM.name
    if external_ipv4:
      access.nat_i_p = external_ipv4
    network_interface.access_configs = [access]

  # Collect information into the Instance object.
  instance = compute_v1.Instance()
  instance.network_interfaces = [network_interface]
  instance.name = instance_name
  instance.disks = disks
  if re.match(r"^zones/[a-z\d\-]+/machineTypes/[a-z\d\-]+$", machine_type):
    instance.machine_type = machine_type
  else:
    instance.machine_type = f"zones/{zone}/machineTypes/{machine_type}"

  instance.scheduling = compute_v1.Scheduling()
  if accelerators:
    instance.guest_accelerators = accelerators
    instance.scheduling.on_host_maintenance = (
        compute_v1.Scheduling.OnHostMaintenance.TERMINATE.name
    )

  if metadata:
    instance.metadata = metadata

  if service_account:
    instance.service_accounts = [service_account]

  if spot:
    # Set the Spot VM setting
    instance.scheduling.provisioning_model = (
        compute_v1.Scheduling.ProvisioningModel.SPOT.name
    )
    instance.scheduling.instance_termination_action = instance_termination_action

  if custom_hostname is not None:
    # Set the custom hostname for the instance
    instance.hostname = custom_hostname

  if delete_protection:
    # Set the delete protection bit
    instance.deletion_protection = True

  # Prepare the request to insert an instance.
  request = compute_v1.InsertInstanceRequest()
  request.zone = zone
  request.project = project_id
  request.instance_resource = instance

  # Wait for the create operation to complete.
  logging.info(f"Creating the {instance_name} instance in {zone}...")

  operation = instance_client.insert(request=request)

  wait_for_extended_operation(operation, "instance creation")

  logging.info(f"Instance {instance_name} created.")
  return instance_client.get(project=project_id, zone=zone, instance=instance_name)


# TODO(piz): Remove wait_for_extended_operation and use dag sensor instead.
def wait_for_extended_operation(
    operation: ExtendedOperation, verbose_name: str = "operation", timeout: int = 300
) -> Any:
  """
  Waits for the extended (long-running) operation to complete.

  If the operation is successful, it will return its result.
  If the operation ends with an error, an exception will be raised.
  If there were any warnings during the execution of the operation
  they will be printed to sys.stderr.

  Args:
      operation: a long-running operation you want to wait on.
      verbose_name: (optional) a more verbose name of the operation,
          used only during error and warning reporting.
      timeout: how long (in seconds) to wait for operation to finish.
          If None, wait indefinitely.

  Returns:
      Whatever the operation.result() returns.

  Raises:
      This method will raise the exception received from `operation.exception()`
      or RuntimeError if there is no exception set, but there is an `error_code`
      set for the `operation`.

      In case of an operation taking longer than `timeout` seconds to complete,
      a `concurrent.futures.TimeoutError` will be raised.
  """
  result = operation.result(timeout=timeout)

  if operation.error_code:
    logging.error(
        f"Error during {verbose_name}: [Code: {operation.error_code}]: {operation.error_message}",
    )
    logging.error(f"Operation ID: {operation.name}")
    raise operation.exception() or RuntimeError(operation.error_message)

  if operation.warnings:
    logging.warning(f"Warnings during {verbose_name}:\n")
    for warning in operation.warnings:
      logging.warning(f" - {warning.code}: {warning.message}")

  return result


def create_metadata(key_val: dict[str, str]) -> compute_v1.Metadata:
  metadata = compute_v1.Metadata()
  metadata.items = [{"key": key, "value": val} for key, val in key_val.items()]
  return metadata


@task
def generate_gpu_name() -> str:
  # note: GPU vm name need to match regex '(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?)', while TPU vm allows '_'.
  # return f'{base_gpu_name}-{str(uuid.uuid4())}'.replace('_', '-')
  # If we use the above base_gpu_name in the return, some potion of the can be longer than 61 as in the regex.
  return f"gpu-{str(uuid.uuid4())}"


@task
def create_resource(
    gpu_name: airflow.XComArg,
    image_project: str,
    image_family: str,
    accelerator: test_config.Gpu,
    gcp: gcp_config.GCPConfig,
    ssh_keys: airflow.XComArg,
) -> str:
  """Request a resource and wait until the nodes are created.

  Args:
    gpu_name: XCom value for unique GPU name.
    image_project: project of the image.
    image_family: family of the image.
    accelerator: Description of GPU to create.
    gcp: GCP project/zone configuration.
    ssh_keys: XCom value for SSH keys to communicate with these GPUs.

  Returns:
    The ip address of the GPU VM.
  """
  image = get_image_from_family(project=image_project, family=image_family)
  disk_type = f"zones/{gcp.zone}/diskTypes/pd-ssd"
  disks = [disk_from_image(disk_type, 100, True, image.self_link)]
  metadata = create_metadata({
      # "install-nvidia-driver": "True",
      "install-nvidia-driver": "False",
      "proxy-mode": "project_editors",
      "ssh-keys": f"cloud-ml-auto-solutions:{ssh_keys.public}",
  })
  acceleratorConfig = compute_v1.AcceleratorConfig(
      accelerator_count=accelerator.count,
      accelerator_type=f"projects/{gcp.project_name}/zones/{gcp.zone}/acceleratorTypes/{accelerator.accelerator_type}",
  )
  service_account = compute_v1.ServiceAccount(
      # email = "cloud-auto-ml-solutions@google.com",
      scopes=["https://www.googleapis.com/auth/cloud-platform"]
  )
  instance = create_instance(
      project_id=gcp.project_name,
      zone=gcp.zone,
      instance_name=gpu_name,
      disks=disks,
      machine_type=accelerator.machine_type,
      service_account=service_account,
      external_access=True,
      accelerators=[acceleratorConfig],
      metadata=metadata,
      instance_termination_action="STOP",
  )
  logging.info("instance info: {instance}")

  ip_pattern = re.compile(r'network_i_p:\s+"([^"]+)"')
  match = ip_pattern.search(str(instance))

  # Extract the matched IP address
  ip_address = "0.0.0.0"
  if match:
    ip_address = match.group(1)
    logging.info(f"Created vm with ip address {ip_address}.")
  else:
    logging.error(f"No IP address found for instance: {gpu_name}.")
    raise (f"Failed to create GPU resource {gpu_name}.")
  return ip_address


@task
def ssh_host(ip_address: str, cmds: Iterable[str], ssh_keys: ssh.SshKeys) -> None:
  """SSH GPU and run commands in multi process.

  Args:
   ip_address: The ip address of the vm resource.
   cmds: The commands to run on a GPU.
   ssh_keys: The SSH key pair to use for authentication.
  """
  pkey = paramiko.RSAKey.from_private_key(io.StringIO(ssh_keys.private))
  logging.info(f"Connecting to IP addresses {ip_address}")

  ssh_group = fabric.ThreadingGroup(
      ip_address,
      user="cloud-ml-auto-solutions",
      connect_kwargs={
          "auth_strategy": paramiko.auth_strategy.InMemoryPrivateKey(
              "cloud-ml-auto-solutions", pkey
          )
      },
  )
  ssh_group.run(cmds)


# TODO(piz): Check why sometime GPU instance doesn't get deleted.
@task_group
def delete_resource(instance_name: airflow.XComArg, project_id: str, zone: str):
  @task(trigger_rule="all_done")
  def delete_resource_request(
      instance_name: str, project_id: str, zone: str
  ) -> airflow.XComArg:
    client = compute_v1.InstancesClient()
    request = compute_v1.DeleteInstanceRequest(
        instance=instance_name,
        project=project_id,
        zone=zone,
    )
    operation = client.delete(request=request)

    return operation.name

  @task.sensor(poke_interval=60, timeout=1800, mode="reschedule")
  def wait_for_resource_deletion(operation_name: airflow.XComArg):
    # Retrives the delete opeartion to check the status.
    client = compute_v1.ZoneOperationsClient()
    request = compute_v1.GetZoneOperationRequest(
        operation=operation_name,
        project=project_id,
        zone=zone,
    )
    operation = client.get(request=request)

    status = operation.status.name

    if status in ("RUNNING", "PENDING"):
      logging.info(f"Resource deletion status: {status}, {operation.status_message}")
      return False
    else:
      if operation.error:
        logging.error(
            f"Error during resource deletion: [Code: {operation.http_error_status_code}]: {operation.http_error_message}",
        )
        logging.error(f"Operation ID: {operation.name}")
        raise operation.exception() or RuntimeError(operation.http_error_message)
      if operation.warnings:
        logging.warning(f"Warnings during resource deletion:\n")
        for warning in operation.warnings:
          logging.warning(f" - {warning.code}: {warning.message}")
      return True

  op = delete_resource_request(instance_name, project_id, zone)
  wait_for_resource_deletion(op)

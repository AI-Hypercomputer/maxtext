"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import abc
import subprocess

from maxtext_xpk_runner import XpkClusterConfig


class DisruptionHandler(abc.ABC):
  """Abstract interface for disruption handlers."""

  @abc.abstractmethod
  def trigger_disruption(
      self, workload_name: str, cluster_config: XpkClusterConfig,
      disruption_config, target_pod_regex: str
  ) -> None:
    """Triggers the workload disruption."""
    pass

  @abc.abstractmethod
  def trigger_recovery(
      self, workload_name: str, cluster_config: XpkClusterConfig,
      disruption_config, target_pod_regex: str
  ) -> None:
    """Triggers workload recovery."""
    pass


class SIGTERMMetadataHandler(DisruptionHandler):
  """Handles SIGTERM disruption via node metadata update."""

  def trigger_disruption(
      self, workload_name: str, cluster_config: XpkClusterConfig,
      disruption_config, target_pod_regex: str
  ) -> None:
    """Triggers SIGTERM disruption by updating node metadata."""
    print(
        "Simulating SIGTERM via node metadata update for pod regex:"
        f" {target_pod_regex} workload: {workload_name}"
    )
    pod_name_command = (
        "kubectl get pods -n default --no-headers -o custom-columns=NAME"
        ":.metadata.name"  # Assuming default namespace, #TODO: Configurable
        f" | grep -E '{target_pod_regex}'"
    )
    try:
      pod_name_process = subprocess.run(
          pod_name_command,
          shell=True,
          check=True,
          capture_output=True,
          text=True,
      )
      pod_name = pod_name_process.stdout.strip()
      if not pod_name:
        print(f"Warning: No pod found matching regex: {target_pod_regex}")
        return

      # Get node name from pod description
      node_name_command = (
          f"kubectl describe pod {pod_name} -n default | grep Node:" #TODO
      )
      node_name_process = subprocess.run(
          node_name_command,
          shell=True,
          check=True,
          capture_output=True,
          text=True,
      )
      node_name_line = node_name_process.stdout.strip()
      node_name = node_name_line.split("/")[0].split(":")[1].strip()
      print(f"Detected node: {node_name} for pod: {pod_name}")

      gcloud_metadata_command = [
          "gcloud",
          "compute",
          "instances",
          "add-metadata",
          node_name,
          "--metadata=maintenance-event-test=TERMINATE_ON_HOST_MAINTENANCE",
          f"--zone={cluster_config.zone}",  # Zone from cluster config
          f"--project={cluster_config.project}",  # Project from config
      ]
      gcloud_metadata_command_str = " ".join(gcloud_metadata_command)
      print(
          f"Executing metadata update command: {gcloud_metadata_command_str}"
      )
      subprocess.run(
          gcloud_metadata_command_str,
          shell=True,
          check=True,
          capture_output=True,
          text=True,
      )
      print(
          "Successfully updated metadata for node:"
          f" {node_name} to simulate SIGTERM"
      )

    except subprocess.CalledProcessError as e:
      print(
          "Error updating metadata for node for pod regex:"
          f" {target_pod_regex}"
      )
      print(f"Return code: {e.returncode}")
      print(f"Stderr: {e.stderr.decode()}")

  def trigger_recovery(
      self, workload_name: str, cluster_config: XpkClusterConfig,
      disruption_config, target_pod_regex: str
  ) -> None:
    """Removes node metadata to recover from SIGTERM simulation."""
    print(
        "Initiating SIGTERM metadata remove recovery for pod regex:"
        f" {target_pod_regex} workload: {workload_name}"
    )
    pod_name_command = (
        "kubectl get pods -n default --no-headers -o custom-columns=NAME"
        ":.metadata.name"  # Assuming default namespace, #TODO: Configurable
        f" | grep -E '{target_pod_regex}'"
    )
    try:
      pod_name_process = subprocess.run(
          pod_name_command,
          shell=True,
          check=True,
          capture_output=True,
          text=True,
      )
      pod_name = pod_name_process.stdout.strip()
      if not pod_name:
        print(f"Warning: No pod found matching regex: {target_pod_regex}")
        return

      # Get node name from pod description
      node_name_command = (
          f"kubectl describe pod {pod_name} -n default | grep Node:" #TODO
      )
      node_name_process = subprocess.run(
          node_name_command,
          shell=True,
          check=True,
          capture_output=True,
          text=True,
      )
      node_name_line = node_name_process.stdout.strip()
      node_name = node_name_line.split("/")[0].split(":")[1].strip()
      print(f"Detected node: {node_name} for pod: {pod_name}")

      gcloud_metadata_command = [
          "gcloud",
          "compute",
          "instances",
          "remove-metadata",
          node_name,
          "--keys=maintenance-event-test",
          f"--zone={cluster_config.zone}",  # Zone from cluster config
          f"--project={cluster_config.project}",  # Project from config
      ]
      gcloud_metadata_command_str = " ".join(gcloud_metadata_command)
      print(
          f"Executing metadata remove command: {gcloud_metadata_command_str}"
      )
      subprocess.run(
          gcloud_metadata_command_str,
          shell=True,
          check=True,
          capture_output=True,
          text=True,
      )
      print(
          "Successfully removed metadata for node:"
          f" {node_name} to recover from SIGTERM simulation"
      )

    except subprocess.CalledProcessError as e:
      print(
          "Error removing metadata for node for pod regex:"
          f" {target_pod_regex}"
      )
      print(f"Return code: {e.returncode}")
      print(f"Stderr: {e.stderr.decode()}")


def create_disruption_handler(disruption_config):
  """Factory function to create the appropriate disruption handler."""
  if disruption_config.method == DisruptionMethod.SIGTERM:
    return SIGTERMMetadataHandler()
  elif disruption_config.method == DisruptionMethod.SIGILL:
    raise NotImplementedError("SIGILL Disruption Handler not implemented yet.")
  else:
    raise ValueError(
        f"Unsupported disruption method: {disruption_config.method}"
    )

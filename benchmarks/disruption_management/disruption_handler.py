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
import dataclasses
import enum
import os
import subprocess
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from xpk_configs import XpkClusterConfig


MCJAX_STANDARD_TARGET_POD_REGEX_SUFFIX = ".*slice-job-0-0.*"
MCJAX_STANDARD_STEP_POD_REGEX_SUFFIX = ".*slice-job-0-0.*"
PATHWAYS_STANDARD_TARGET_POD_REGEX_SUFFIX = ".*worker-0-0.*"
PATHWAYS_STANDARD_STEP_POD_REGEX_SUFFIX = ".*main-0-0.*"

STANDARD_STEP_LOG_REGEX = "completed step: (\\d+)"

PATHWAYS_WORKER_CONTAINER_NAME = "pathways-worker"
MCJAX_WORKER_CONTAINER_NAME = "jax-tpu"


class TriggerType(enum.Enum):
  TIME_SECONDS = "time_seconds"
  STEP = "step"


class DisruptionMethod(enum.Enum):
  SIGILL = "sigill"  # simulate unplanned failure
  SIGTERM = "sigterm"  # planned failure via node metadata update


class RecoverMethod(enum.Enum):
  SIGTERM_METADATA_REMOVE = "sigterm_metadata_remove"  # Remove node metadata


@dataclasses.dataclass
class DisruptionConfig:
  """Configuration for workload disruptions and recovery."""
  name: str  # Name of the disruption config
  trigger_type: TriggerType  # Trigger for disruption (Time_Seconds or Step)
  trigger_value: int  # Value for disruption trigger (seconds or steps)
  disruption_method: DisruptionMethod  # Disruption method (SIGTERM, SIGILL)
  worker_container_name: str  # Name of the container to disrupt

  # TODO(sujinesh): Currently unimplemented.
  # Some disruption types may require a recovery mechanism.
  # For example, Simulating a maintenance event requires adding metadata to the
  # node. To recover from this, a recovery method to remove the node metadata.
  recover_trigger_type: TriggerType = None  # Trigger for recovery
  recover_trigger_value: int = None  # Value for recovery trigger
  recover_method: RecoverMethod = (
      None  # Recovery method (SIGTERM_METADATA_REMOVE)
  )

  # Target pod regex needed for triggering disruption.
  target_pod_regex: str = MCJAX_STANDARD_TARGET_POD_REGEX_SUFFIX


class DisruptionHandler(abc.ABC):
  """Abstract interface for disruption handlers."""

  @abc.abstractmethod
  def trigger_disruption(
      self, workload_name: str, cluster_config: XpkClusterConfig,
      disruption_config, target_pod_regex: str
  ) -> None:
    """Triggers the workload disruption."""
    raise NotImplementedError("Subclasses must implement this method.")

  def trigger_recovery(
      self, workload_name: str, cluster_config: XpkClusterConfig,
      disruption_config, target_pod_regex: str
  ) -> None:
    """Triggers workload recovery. Subclasses may implement this method."""
    pass


class SIGILLHandler(DisruptionHandler):
  """Handles SIGILL disruption by sending a SIGILL signal to the pod."""

  def trigger_disruption(
      self, workload_name: str, cluster_config: XpkClusterConfig,
      disruption_config, target_pod_regex: str
  ) -> None:
    """Triggers SIGILL disruption by executing kill -s SIGILL 1 in the pod."""
    print(
        f"🔥🔥🔥 Beginning SIGILL for workload: {workload_name} with pod regex:"
        f" {target_pod_regex} 🔥🔥🔥"
    )
    pod_name_command = [
        "kubectl",
        "get",
        "pods",
        "-o=name",
        "--no-headers",
        f"| grep -E '{target_pod_regex}'",
    ]
    pod_name_command_str = " ".join(pod_name_command)
    container_name = disruption_config.worker_container_name

    try:
      pod_name_process = subprocess.run(
          pod_name_command_str,
          shell=True,
          check=True,
          capture_output=True,
          text=True,
      )
      pod_name = pod_name_process.stdout.strip()
      if not pod_name:
        print(
            f"Warning: No pod found matching regex: '{target_pod_regex}' for"
            f" workload '{workload_name}'"
        )
        return

      print(f"🔍 Found pod: {pod_name}")
      kill_command = [
          "kubectl",
          "exec",
          "-it",
          pod_name,
          "-c",
          container_name,
          "--",
          "/bin/bash",
          "-c",
          "\"kill -s SIGILL 1\"",
      ]
      kill_command_str = " ".join(kill_command)
      print(f"🔥🔥🔥 Executing command in pod: {kill_command_str} 🔥🔥🔥")
      subprocess.run(
          kill_command_str,
          shell=True,
          check=True,
          capture_output=True,
          text=True,
      )
      print(
          f"✅ Successfully sent SIGILL to pod: {pod_name} in container:"
          f" {container_name}"
      )

    except subprocess.CalledProcessError as e:
      print(
          "❌ Error sending SIGILL to pod(s) matching regex"
          f" '{target_pod_regex}' for workload '{workload_name}'"
      )
      print(f"Return code: {e.returncode}")
      print(f"error: {e}")


def create_disruption_handler(disruption_config):
  """Factory function to create the appropriate disruption handler."""
  if disruption_config.disruption_method == DisruptionMethod.SIGTERM:
    raise NotImplementedError("SIGTERM Disruption Handler not implemented yet.")
  elif disruption_config.disruption_method == DisruptionMethod.SIGILL:
    return SIGILLHandler()
  else:
    raise ValueError(
        f"Unsupported disruption method: {disruption_config.disruption_method}"
    )

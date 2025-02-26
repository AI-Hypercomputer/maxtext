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

import dataclasses
import enum
import threading
from typing import Dict, List

from maxtext_xpk_runner import XpkClusterConfig

from disruption_handler import create_disruption_handler, DisruptionHandler
from monitor import create_monitor, Monitor

MCJAX_STANDARD_TARGET_POD_REGEX_SUFFIX = ".*slice-job-0-0.*"
MCJAX_STANDARD_STEP_POD_REGEX_SUFFIX = ".*slice-job-0-0.*"
PATHWAYS_STANDARD_TARGET_POD_REGEX_SUFFIX = ".*worker-0-0.*"
PATHWAYS_STANDARD_STEP_POD_REGEX_SUFFIX = ".*main-0-0.*"

STANDARD_STEP_LOG_REGEX = "completed step: (\\d+)"
POLLING_INTERVAL_SECONDS = 5


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
  trigger_type: TriggerType  # Trigger for disruption (Time_Seconds or Step)
  trigger_value: int  # Value for disruption trigger (seconds or steps)
  method: DisruptionMethod  # Disruption method (SIGTERM, SIGILL)

  recover_trigger_type: TriggerType  # Trigger for recovery
  recover_trigger_value: int  # Value for recovery trigger
  recover_method: RecoverMethod  # Recovery method (SIGTERM_METADATA_REMOVE)

  target_pod_regex: str = "<workload_name>-.*worker-0-0.*"  # Target pod regex


class DisruptionManager:
  """Manages workload disruptions and recoveries."""

  def __init__(self) -> None:
    """Initializes the DisruptionManager."""
    self.workloads: Dict[str, Dict[str, any]] = {}
    self.threads_to_monitor: List[threading.Thread] = []
    self.lock: threading.Lock = threading.Lock()

  def add_workload(
      self,
      workload_name: str,
      cluster_config: XpkClusterConfig,
      disruption_configs: List[DisruptionConfig],
  ) -> None:
    """Adds a workload and starts monitoring for disruptions & recovery.

    Args:
      workload_name: The name of the workload.
      cluster_config: The cluster configuration for the workload.
      disruption_configs: A list of DisruptionConfig for the workload.
    """
    if not disruption_configs:
      print(f"No disruption configured for workload: {workload_name}")
      return

    # For now only consider first disruption config
    disruption_config = disruption_configs[0]

    thread = threading.Thread(
        target=self._monitor_workload,
        args=(workload_name, cluster_config, disruption_config),
        daemon=True,
    )
    self.threads_to_monitor.append(thread)
    thread.start()
    print(f"Started monitoring thread for workload: {workload_name}")

  def wait_for_disruptions_completed(self) -> None:
    """Waits for all disruption monitoring threads to complete.

    This is a blocking call.
    """
    print("Waiting for disruptions to complete...")
    for thread in self.threads_to_monitor:
      thread.join()
    print("All disruptions completed.")

  def _monitor_workload(
      self,
      workload_name: str,
      cluster_config: XpkClusterConfig,
      disruption_config: DisruptionConfig,
  ) -> None:
    """Monitors workload progress, triggers disruptions, and recoveries."""
    target_pod_regex = f"{workload_name}{disruption_config.target_pod_regex}"

    # Create Monitor based on trigger type
    monitor: Monitor = create_monitor(
        workload_name, disruption_config, target_pod_regex
    )
    disruption_handler: DisruptionHandler = create_disruption_handler(
        disruption_config
    )

    if monitor.monitor_and_detect_trigger():
      print(
          f"Trigger detected for workload: {workload_name}, triggering"
          f" {disruption_config.method}"
      )
      disruption_handler.trigger_disruption(
          workload_name, cluster_config, disruption_config, target_pod_regex
      )
      self._monitor_recovery( # Monitor recovery after disruption
          workload_name, cluster_config, disruption_config,
          disruption_handler, disruption_config, target_pod_regex
      )
    else:
      print(f"Monitoring for workload: {workload_name} exited without trigger.")


  def _monitor_recovery(
      self,
      workload_name: str,
      cluster_config: XpkClusterConfig,
      disruption_config: DisruptionConfig,
      disruption_handler: DisruptionHandler,
      target_pod_regex: str,
  ) -> None:
    """Monitors for recovery trigger and initiates recovery."""
    if disruption_config.recover_trigger_type == TriggerType.TIME_SECONDS:
      time.sleep(disruption_config.recover_trigger_value)
      print(
          f"Time trigger reached for recovery of workload: {workload_name},"
          f" triggering {disruption_config.recover_method}"
      )
      disruption_handler.trigger_recovery(
          workload_name, cluster_config, disruption_config, target_pod_regex
      )
    elif disruption_config.recover_trigger_type == TriggerType.STEP:
      # Step-based recovery trigger (if needed in future) - currently no usecase
      print("STEP-based recovery trigger not yet implemented.")
    else:
      print(
          f"Unknown recover trigger type:"
          f" {disruption_config.recover_trigger_type}"
      )

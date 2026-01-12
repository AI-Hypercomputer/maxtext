# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module defines the DisruptionManager class, which orchestrates the
entire disruption process for one or more workloads.

It manages a collection of monitoring threads, one for each configured
disruption. It provides an interface to add workloads, start the monitoring
process, and wait for all disruptions to complete.
"""

from collections import defaultdict
import threading

from benchmarks.benchmark_utils import Framework
from benchmarks.disruption_management.disruption_handler import create_disruption_handler
from benchmarks.disruption_management.disruption_handler import DisruptionConfig
from benchmarks.disruption_management.disruption_handler import DisruptionHandler
from benchmarks.disruption_management.disruption_handler import DisruptionMethod
from benchmarks.disruption_management.disruption_handler import MCJAX_STANDARD_TARGET_POD_REGEX_SUFFIX
from benchmarks.disruption_management.disruption_handler import MCJAX_STANDARD_STEP_POD_REGEX_SUFFIX
from benchmarks.disruption_management.disruption_handler import MCJAX_WORKER_CONTAINER_NAME
from benchmarks.disruption_management.disruption_handler import PATHWAYS_STANDARD_TARGET_POD_REGEX_SUFFIX
from benchmarks.disruption_management.disruption_handler import PATHWAYS_STANDARD_STEP_POD_REGEX_SUFFIX
from benchmarks.disruption_management.disruption_handler import PATHWAYS_WORKER_CONTAINER_NAME
from benchmarks.disruption_management.disruption_handler import TriggerType
from benchmarks.disruption_management.monitor import create_monitor
from benchmarks.disruption_management.monitor import Monitor
from benchmarks.xpk_configs import XpkClusterConfig


class DisruptionManager:
  """Manages workload disruptions and recoveries."""

  def __init__(self) -> None:
    """Initializes the DisruptionManager."""
    self.threads_to_monitor: defaultdict[str, list[threading.Thread]] = defaultdict(list)

  def add_workload(
    self,
    workload_name: str,
    cluster_config: XpkClusterConfig,
    disruption_configs: list[DisruptionConfig],
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

    # Add each thread to the list of threads to monitor.
    # Note that we do not start the threads here.
    for disruption_config in disruption_configs:
      thread = threading.Thread(
        target=self._monitor_and_disrupt_workload,
        args=(workload_name, cluster_config, disruption_config),
        daemon=True,
      )
      self.threads_to_monitor[workload_name].append(thread)
    print(f"Added {len(disruption_configs)} disruption configs for workload: {workload_name}")

  def remove_workload(self, workload_name: str) -> None:
    """Removes a workload from the disruption manager.

    This will stop new disruptions from being triggered for this workload.
    Running monitoring threads for this workload might continue until their
    current cycle completes.

    Note: this does not stop the workload itself or the threads, just the
    monitoring.

    Args:
      workload_name: The name of the workload to remove.
    """
    if workload_name in self.threads_to_monitor:
      self.threads_to_monitor[workload_name].clear()
      print(f"Stopped monitoring threads for workload '{workload_name}'.")
    else:
      print(f"No monitoring threads found for workload '{workload_name}'.")

  def start_disruptions_and_wait_for_completion(self) -> None:
    """Starts disruption monitoring and waits for all disruptions to complete.

    This is a blocking call.
    """
    print("Starting disruption monitoring! 🔥🩺")
    for workload_name, threads in self.threads_to_monitor.items():
      for thread in threads:
        thread.start()
        print(f"🔥🩺 Started monitoring thread for workload: {workload_name}")

    # Wait for all threads to complete.
    for _, threads in self.threads_to_monitor.items():
      for thread in threads:
        thread.join()

    print("All disruptions completed.")

  def _monitor_and_disrupt_workload(
    self,
    workload_name: str,
    cluster_config: XpkClusterConfig,
    disruption_config: DisruptionConfig,
  ) -> None:
    """Monitors workload progress, triggers disruptions, and recoveries."""
    target_pod_regex = f"{workload_name}{disruption_config.target_pod_regex}"
    step_pod_regex = f"{workload_name}{disruption_config.step_pod_regex}"

    # Create Monitor based on trigger type
    monitor: Monitor = create_monitor(workload_name, disruption_config, step_pod_regex)
    disruption_handler: DisruptionHandler = create_disruption_handler(disruption_config)

    if monitor.monitor_and_detect_trigger():
      print(
        f"🔥🔥🔥 Trigger detected for workload: {workload_name}, triggering {disruption_config.disruption_method} 🔥🔥🔥"
      )
      disruption_handler.trigger_disruption(workload_name, cluster_config, disruption_config, target_pod_regex)
    else:
      print(f"Monitoring for workload: {workload_name} exited without trigger.")

  # TODO(sujinesh): Implement recovery.
  def _monitor_recovery(self) -> None:
    """Monitors for recovery trigger and initiates recovery."""
    raise NotImplementedError("Recovery not implemented yet.")


def construct_disruption_configs(
  framework: str,
  disruption_method: DisruptionMethod,
  disruptions,
) -> list[DisruptionConfig]:
  """Constructs the disruption configs for the benchmark."""

  if Framework(framework) == Framework.PATHWAYS:
    target_pod_regex = PATHWAYS_STANDARD_TARGET_POD_REGEX_SUFFIX
    step_pod_regex = PATHWAYS_STANDARD_STEP_POD_REGEX_SUFFIX
    worker_container_name = PATHWAYS_WORKER_CONTAINER_NAME
  else:
    target_pod_regex = MCJAX_STANDARD_TARGET_POD_REGEX_SUFFIX
    step_pod_regex = MCJAX_STANDARD_STEP_POD_REGEX_SUFFIX
    worker_container_name = MCJAX_WORKER_CONTAINER_NAME

  disruption_config_list = []
  for trigger_type, trigger_values in disruptions.items():
    for trigger_value in trigger_values:
      disruption_config_list.append(
        DisruptionConfig(
          name="_".join([str(trigger_value), trigger_type]),
          trigger_type=TriggerType.TIME_SECONDS if trigger_type == "time_seconds" else TriggerType.STEP,
          trigger_value=trigger_value,
          disruption_method=disruption_method,
          target_pod_regex=target_pod_regex,
          step_pod_regex=step_pod_regex,
          worker_container_name=worker_container_name,
        )
      )
  return disruption_config_list

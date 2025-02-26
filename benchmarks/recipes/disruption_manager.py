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
import re
import subprocess
import threading
import time
from typing import Dict, List

from maxtext_xpk_runner import XpkClusterConfig

MCJAX_STANDARD_TARGET_POD_REGEX_SUFFIX = ".*slice-job-0-0.*"
MCJAX_STANDARD_STEP_POD_REGEX_SUFFIX = ".*slice-job-0-0.*"

PATHWAYS_STANDARD_TARGET_POD_REGEX_SUFFIX = ".*worker-0-0.*"
PATHWAYS_STANDARD_STEP_POD_REGEX_SUFFIX = ".*main-0-0.*"

STANDARD_STEP_LOG_REGEX = "completed step: (\\d+)"
POLLING_INTERVAL_SECONDS = 5

class TriggerType(enum.Enum):
  TIME = "time"
  STEP = "step"


class DisruptionMethod(enum.Enum):
  SIGILL = "sigill"  # simulate unplanned failure
  SIGTERM = "sigterm"  # simulate planned failure


@dataclasses.dataclass
class DisruptionConfig:
  trigger_type: TriggerType  # e.g. Time or Step
  trigger_value: int  # e.g. "200" to represent 200 steps or seconds
  method: DisruptionMethod  # planned or unplanned failure
  target_pod_regex: str = "<workload_name>-.*worker-0-0.*"  # match target
  # step_pod_regex: str = "<workload_name>-.*main-0-0.*"
  # step_log_regex: str = "Completed step: (\\d+)"  # match logs for step


class DisruptionManager:
  """Manages workload disruptions."""

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
    """Adds a workload to the DisruptionManager and starts monitoring it.

    Args:
      workload_name: The name of the workload.
      cluster_config: The cluster configuration for the workload.
      disruption_configs: A list of DisruptionConfig for the workload.
    """
    if not disruption_configs:
      print(f"No disruption configured for workload: {workload_name}")
      return

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
    print("All disruptions completed (placeholder).")

  def _monitor_workload(
      self,
      workload_name: str,
      cluster_config: XpkClusterConfig,
      disruption_config: DisruptionConfig,
  ) -> None:
    """Monitors the workload and triggers disruptions."""
    if (
        disruption_config.trigger_type == TriggerType.TIME
    ):  # TODO: Make this a function
      time.sleep(disruption_config.trigger_value)
      print(
          f"Time trigger reached for workload: {workload_name}, triggering"
          f" {disruption_config.method}"
      )
      print(
          f"Simulating {disruption_config.method} to pod:"
          f" {disruption_config.target_pod_regex}"
      )

    elif (
        disruption_config.trigger_type == TriggerType.STEP
    ):  # TODO: Make this a function
      target_pod_regex = f"{workload_name}{disruption_config.target_pod_regex}"

      print(f"Target pod name: {target_pod_regex}")
      print(f"Step log regex: {STANDARD_STEP_LOG_REGEX}")

      kubectl_logs_command = [
          "kubectl",
          "logs",
          "--follow",
          "$("
          + 'kubectl get pods --no-headers -o custom-columns=":metadata.name"'
          f" | grep -E '{target_pod_regex}')",
      ]
      kubectl_logs_command_str = " ".join(kubectl_logs_command)

      process = None  # Initialize process outside the try block
      try:
        print(f"Tailing logs using command: {kubectl_logs_command_str}")
        process = subprocess.Popen(
            kubectl_logs_command_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # To get text output instead of bytes
        )

        # TODO: If the pod is in the FAILED or COMPLETED state, we should not
        # wait for the trigger.
        while True:
          if process.stdout:
            line = process.stdout.readline()
            if line:
              print(f"Log line: {line.strip()}")
              match = re.search(
                  STANDARD_STEP_LOG_REGEX, line
              )  # Search for step
              if match:
                step_number = int(match.group(1))
                print(f"Detected step: {step_number}")
                if step_number >= disruption_config.trigger_value:
                  print(
                      "STEP trigger reached! Step:"
                      f" {step_number}, Trigger Value:"
                      f" {disruption_config.trigger_value}"
                  )
                  print(
                      "No-op SIGTERM for workload:"
                      f" {workload_name}, regex:"
                      f" {target_pod_regex}"
                  )  # No-op SIGTERM
                  break  # Exit log tailing loop after trigger
            time.sleep(POLLING_INTERVAL_SECONDS)  # Polling interval for logs

      except subprocess.CalledProcessError as e:
        print(f"Error tailing logs for pods matching regex: {target_pod_regex}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output.decode()}")
      except Exception as e:
        print(f"An error occurred during log tailing: {e}")
      finally:
        if process:
          process.terminate()

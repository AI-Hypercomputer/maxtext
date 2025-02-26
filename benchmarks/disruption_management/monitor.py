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

import re
import subprocess
import time

from disruption_manager import POLLING_INTERVAL_SECONDS, STANDARD_STEP_LOG_REGEX
from disruption_manager import TriggerType


class Monitor:
  """Abstract base class for workload monitors."""

  def __init__(self, workload_name: str, trigger_config):
    """Initializes a Monitor."""
    self.workload_name = workload_name
    self.trigger_config = trigger_config

  def monitor_and_detect_trigger(self):
    """Monitors workload and detects trigger condition.

    Returns:
      bool: True if trigger condition is met, False otherwise.
    """
    raise NotImplementedError("Subclasses must implement this method.")


class StepMonitor(Monitor):
  """Monitors workload progress based on steps in logs."""

  def __init__(self, workload_name: str, trigger_config, target_pod_regex: str):
    """Initializes StepMonitor."""
    super().__init__(workload_name, trigger_config)
    self.target_pod_regex = target_pod_regex

  def monitor_and_detect_trigger(self):
    """Monitors logs and detects step trigger."""
    print(f"Using StepMonitor for workload: {self.workload_name}")
    print(f"Target pod regex: {self.target_pod_regex}")
    print(f"Step log regex: {STANDARD_STEP_LOG_REGEX}")

    kubectl_logs_command = [
        "kubectl",
        "logs",
        "--follow",
        "$("
        + 'kubectl get pods --no-headers -o custom-columns=":metadata.name"'
        f" | grep -E '{self.target_pod_regex}')",
    ]
    kubectl_logs_command_str = " ".join(kubectl_logs_command)

    process = None
    try:
      print(f"Tailing logs using command: {kubectl_logs_command_str}")
      process = subprocess.Popen(
          kubectl_logs_command_str,
          shell=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
      )

      while True:
        if process.stdout:
          line = process.stdout.readline()
          if line:
            match = re.search(STANDARD_STEP_LOG_REGEX, line)
            if match:
              step_number = int(match.group(1))
              print(f"Detected step: {step_number}")
              if step_number >= self.trigger_config.trigger_value:
                print(
                    "STEP trigger reached! Step:"
                    f" {step_number}, Trigger Value:"
                    f" {self.trigger_config.trigger_value}"
                  )
                return True  # Trigger condition met
        time.sleep(POLLING_INTERVAL_SECONDS)

    except subprocess.CalledProcessError as e:
      print(f"Error tailing logs for pods matching regex:"
            f" {self.target_pod_regex}")
      print(f"Return code: {e.returncode}")
      print(f"Output: {e.output.decode()}")
    except Exception as e:
      print(f"An error occurred during log tailing: {e}")
    finally:
      if process:
        process.terminate()
    return False # Error or process terminated without trigger


class TimeMonitor(Monitor):
  """Monitors time and triggers after a set duration."""

  def __init__(self, workload_name: str, trigger_config):
    """Initializes TimeMonitor."""
    super().__init__(workload_name, trigger_config)

  def monitor_and_detect_trigger(self):
    """Monitors time and detects time-based trigger."""
    print(f"Using TimeMonitor for workload: {self.workload_name}")
    time.sleep(self.trigger_config.trigger_value)
    print(
        f"Time trigger reached after {self.trigger_config.trigger_value} seconds"
    )
    return True  # Trigger condition met


def create_monitor(workload_name, disruption_config, target_pod_regex):
  """Factory function to create the appropriate monitor."""
  if disruption_config.trigger_type == TriggerType.STEP:
    return StepMonitor(
        workload_name, disruption_config, target_pod_regex=target_pod_regex
    )
  elif disruption_config.trigger_type == TriggerType.TIME_SECONDS:
    return TimeMonitor(workload_name, disruption_config)
  else:
    raise ValueError(
        f"Unsupported trigger type: {disruption_config.trigger_type}"
    )
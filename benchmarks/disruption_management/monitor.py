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

import abc
import os
import re
import subprocess
import sys
import time

from disruption_management.disruption_utils import wait_for_pod_to_start

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from disruption_management.disruption_handler import DisruptionConfig
from disruption_management.disruption_handler import TriggerType

STANDARD_STEP_LOG_REGEX = r".*completed step: (\d+), .*"


class Monitor(abc.ABC):
  """Abstract base class for workload monitors."""

  def __init__(
      self,
      workload_name: str,
      disruption_config: DisruptionConfig,
  ):
    """Initializes a Monitor."""
    self.workload_name = workload_name
    self.disruption_config = disruption_config

  @abc.abstractmethod
  def monitor_and_detect_trigger(self):
    """Monitors workload and detects trigger condition.

    Returns:
      bool: True if trigger condition is met, False otherwise.
    """
    raise NotImplementedError("Subclasses must implement this method.")


class StepMonitor(Monitor):
  """Monitors workload progress based on steps in logs."""

  def __init__(
      self,
      workload_name: str,
      disruption_config: DisruptionConfig,
      step_pod_regex: str,
  ):
    """Initializes StepMonitor."""
    super().__init__(workload_name, disruption_config)
    self.step_pod_regex = step_pod_regex
    if not step_pod_regex:
      raise ValueError("Step pod regex is required for monitoring.")

  def monitor_and_detect_trigger(self):
    """Waits for target pod to start and returns when it detects the step trigger."""
    pod_name = wait_for_pod_to_start(self.workload_name, self.step_pod_regex)
    if not pod_name:
      return False

    kubectl_logs_command = [
        "kubectl",
        "logs",
        "-f",  # Follow the logs for real-time updates
        pod_name,
    ]
    kubectl_logs_command_str = " ".join(kubectl_logs_command)

    process = None
    try:
      print(
          f"Workload '{self.workload_name}', Pod '{pod_name}': Tailing logs for"
          f" real-time step detection (reading in chunks)..."
      )
      process = subprocess.Popen(
          kubectl_logs_command_str,
          shell=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
      )

      last_step = -1
      with process.stdout as pipe:
        for line in iter(pipe.readline, ""):
          print(f"Workload '{self.workload_name}', Pod '{pod_name}': {line}")
          line = line.strip()
          print(
              f"Workload '{self.workload_name}', Pod '{pod_name}': {line}"
          )
          match = re.search(STANDARD_STEP_LOG_REGEX, line)
          if match:
            step_number = int(match.group(1))
            last_step = max(last_step, step_number)  # Update last seen step
            if step_number >= self.disruption_config.trigger_value:
              print(
                  f"Workload '{self.workload_name}', Pod '{pod_name}': STEP"
                  f" trigger reached! Detected step: {step_number}, Trigger"
                  f" Value: {self.disruption_config.trigger_value}."
              )
              return True

    except subprocess.CalledProcessError as e:
      print(
          f"Error getting logs for pod '{pod_name}' of workload"
          f" '{self.workload_name}': {e.stderr}"
      )
      return False
    except Exception as e:
      print(f"An error occurred during log tailing: {e}")
    finally:
      if process:
        process.kill()

    print(
        f"Workload '{self.workload_name}', Pod '{pod_name}': No step trigger"
        " detected."
    )
    return False


class TimeMonitor(Monitor):
  """Monitors time and triggers after a set duration."""

  def monitor_and_detect_trigger(self):
    """Monitors time and detects time-based trigger by sleeping."""
    print(
        f"😴 Using TimeMonitor for workload: {self.workload_name}, sleeping for"
        f" {self.disruption_config.trigger_value} seconds 😴."
    )
    time.sleep(self.disruption_config.trigger_value)
    print(
        "😳 Time trigger reached after"
        f" {self.disruption_config.trigger_value} seconds 😳"
    )
    return True


def create_monitor(workload_name, disruption_config, step_pod_regex):
  """Factory function to create the appropriate monitor."""
  if disruption_config.trigger_type == TriggerType.STEP:
    return StepMonitor(
        workload_name, disruption_config, step_pod_regex=step_pod_regex
    )
  elif disruption_config.trigger_type == TriggerType.TIME_SECONDS:
    return TimeMonitor(workload_name, disruption_config)
  else:
    raise ValueError(
        f"Unsupported trigger type: {disruption_config.trigger_type}"
    )

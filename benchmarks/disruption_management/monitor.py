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
import os
import sys
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from disruption_management.disruption_handler import DisruptionConfig
from disruption_management.disruption_handler import TriggerType

POLLING_INTERVAL_SECONDS = 10


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
      target_pod_regex: str,
  ):
    """Initializes StepMonitor."""
    super().__init__(workload_name, disruption_config)
    self.target_pod_regex = target_pod_regex
    if not target_pod_regex:
      raise ValueError("Target pod regex is required for monitoring.")

  # TODO(sujinesh): Implement step monitor.
  def monitor_and_detect_trigger(self):
    """Monitors logs and detects step trigger."""
    raise NotImplementedError("Step monitor not implemented yet.")


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

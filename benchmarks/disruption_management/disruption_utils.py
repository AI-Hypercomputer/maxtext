# Copyright 2023–2026 Google LLC
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

"""Utility functions for disrupting workloads."""

import subprocess
import time

POLLING_INTERVAL_SECONDS = 10


def execute_command_as_subprocess(
    command: str,
) -> None:
  """Executes the command in the pod."""
  print(f"Executing command: {command}")
  try:
    process = subprocess.run(
        command,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    if process.stdout:
      print(process.stdout)
    if process.stderr:
      print(f"stderr: {process.stderr}")
    print(f"✅ Successfully executed command: {command}")
  except subprocess.CalledProcessError as e:
    print(f"❌ Error executing command: {command}")
    print(f"Return code: {e.returncode}")
    print(f"Error: {e}")


def get_pod_name_from_regex(workload_name: str, pod_regex: str) -> str | None:
  """Returns the name of the first pod matching the regex."""
  print(f"Workload '{workload_name}': Getting pod name matching" f" '{pod_regex}'...")
  pod_name_command = [
      "kubectl",
      "get",
      "pods",
      "-o=custom-columns=NAME:.metadata.name",
      "--no-headers",
      f"| grep -E '{pod_regex}'",
  ]
  pod_name_command_str = " ".join(pod_name_command)
  try:
    process = subprocess.run(
        pod_name_command_str,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    pod_names = process.stdout.strip().splitlines()
    if pod_names:
      # Assuming there's only one step pod.
      pod_name = pod_names[0]
      print(f"Workload '{workload_name}': Found pod: {pod_name}")
      return pod_name
    else:
      print(f"Workload '{workload_name}': No pod found matching" f" regex '{pod_regex}'.")
  except subprocess.CalledProcessError as e:
    print(f"Workload '{workload_name}': Error getting pod information:" f" {e}")
  return None


def get_pod_status(workload_name: str, pod_name: str) -> str | None:
  """Returns the status of the pod."""
  print(f"Workload '{workload_name}': Getting status of pod '{pod_name}'...")
  pod_status_command = [
      "kubectl",
      "get",
      "pod",
      pod_name,
      "-o=jsonpath='{.status.phase}'",
  ]
  pod_status_command_str = " ".join(pod_status_command)
  status_process = subprocess.run(
      pod_status_command_str,
      shell=True,
      check=True,
      capture_output=True,
      text=True,
  )
  pod_status = status_process.stdout.strip()
  print(f"Workload '{workload_name}': Pod '{pod_name}' is in '{pod_status}'" " state.")
  return pod_status


def wait_for_pod_to_start(workload_name: str, pod_regex: str) -> str | None:
  """Waits for the step pod to be in 'Running' state and returns its name."""
  print(f"Workload '{workload_name}': Waiting for pod matching" f" '{pod_regex}' to be in 'Running' state...")
  while True:
    pod_name = get_pod_name_from_regex(workload_name, pod_regex)
    if pod_name:
      pod_status = get_pod_status(workload_name, pod_name)
      if pod_status == "Running":
        print(f"Workload '{workload_name}': Step pod '{pod_name}'" f" is now in 'Running' state.")
        return pod_name
    time.sleep(POLLING_INTERVAL_SECONDS)

  print(
      f"Workload '{workload_name}': Timed out waiting for step pod" f" matching '{pod_regex}' to reach 'Running' state."
  )
  return None

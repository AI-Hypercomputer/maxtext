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

import ctypes.util
import os.path
from platform import processor
from shutil import which
from typing import Literal, cast

import jax
from jaxlib import xla_client

PKG_DIR = os.path.dirname(os.path.abspath(__file__))  # MaxText directory path
EPS: float = 1e-8  # Epsilon to calculate loss
DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE: int = 2 * 1024**3  # Default checkpoint file size

devices: list[xla_client.Device] = []
cpu_present, gpu_present, tpu_present = cast(tuple[None | bool, None | bool, None | bool], (None, None, None))


def device_presence(
    force_refresh=False,
) -> tuple[None | bool, None | bool, None | bool]:
  """Check whether a device is present.
  Args:
      force_refresh (bool, optional): If True, force a refresh of the device list
  Returns:
      tuple[Optional[bool],Optional[bool],Optional[bool]] of (cpu_present, gpu_present, tpu_present)
  """
  global cpu_present, gpu_present, tpu_present
  if cpu_present is None or gpu_present is None or tpu_present is None or force_refresh:
    tpu_present = ctypes.util.find_library("tpu") or ctypes.util.find_library("edgetpu")
    gpu_present = which("nvidia-smi") or os.path.exists("/proc/driver/nvidia")
    cpu_present = os.path.exists("/proc/cpuinfo") or processor()
    # for device in get_devices():
    #   if device.platform == "cpu":
    #     cpu_present = True
    #   elif device.platform == "gpu":
    #     gpu_present = True
    #   elif device.platform == "tpu":
    #     tpu_present = True
  return cpu_present, gpu_present, tpu_present


def has_cpu() -> bool:
  return any(device.platform == "cpu" for device in get_devices())


def has_gpu() -> bool:
  return any(device.platform == "gpu" for device in get_devices())


def has_tpu() -> bool:
  return any(device.platform == "tpu" for device in get_devices())


def is_cpu_only() -> bool:
  return not gpu_present and not tpu_present or os.environ.get("CPU_ONLY_TEST") == 1


def get_devices(backend: None | Literal["cpu", "gpu", "tpu"] = None, force_refresh=False) -> list[xla_client.Device]:
  global devices
  if not devices or force_refresh:
    devices = jax.devices(backend)
  return devices


device_presence.run_before = False
if not device_presence.run_before:
  cpu_present, gpu_present, tpu_present = device_presence()

__all__ = [
    "DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE",
    "EPS",
    "PKG_DIR",
    "device_presence",
    "get_devices",
    "gpu_present",
    "is_cpu_only",
    "tpu_present",
]

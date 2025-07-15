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

import os.path

import jax

PKG_DIR = os.path.dirname(os.path.abspath(__file__))  # MaxText directory path
EPS = 1e-8  # Epsilon to calculate loss
DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3  # Default checkpoint file size

devices = []


def has_cpu():
    return any(device.platform == "cpu" for device in get_devices())


def has_gpu():
    return any(device.platform == "gpu" for device in get_devices())


def has_tpu():
    return any(device.platform == "tpu" for device in get_devices())


def get_devices():
    global devices
    if not devices:
        devices = jax.devices()
    return devices


__all__ = [
    "DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE",
    "EPS",
    "PKG_DIR",
    "has_gpu",
    "has_tpu",
]

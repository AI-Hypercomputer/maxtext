"""
 Copyright 2023 Google LLC

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

""" Static map of TPU names such as v4-8 to properties such as chip layout."""

from dataclasses import dataclass

@dataclass
class SystemCharacteristics:
  platform: str
  topology_name: str
  chip_config_name: str # 'megacore' or 'default'
  chips_per_host_bounds: tuple
  devices_per_slice: int

UserFacingNameToSystemCharacteristics = {
    'v5e-16': SystemCharacteristics(
        'tpu', 'v5e:4x4', 'default', (2, 2, 1), 16
    ),
    'v5e-32': SystemCharacteristics(
        'tpu', 'v5e:4x8', 'default', (2, 2, 1), 32
    ),
    'v5e-64': SystemCharacteristics(
        'tpu', 'v5e:8x8', 'default', (2, 2, 1), 64
    ),
    'v5e-128': SystemCharacteristics(
        'tpu', 'v5e:8x16', 'default', (2, 2, 1), 128
    ),
    'v5e-256': SystemCharacteristics(
        'tpu', 'v5e:16x16', 'default', (2, 2, 1), 256
    ),
    'v4-8': SystemCharacteristics(
      'tpu', 'v4:2x2x1', 'megacore', (2, 2, 1), 4
    ),
    'v4-16': SystemCharacteristics(
      'tpu', 'v4:2x2x2', 'megacore', (2, 2, 1), 8
    ),
    'v4-32': SystemCharacteristics(
      'tpu', 'v4:2x2x4', 'megacore', (2, 2, 1), 16
    ),
    'v4-64': SystemCharacteristics(
      'tpu', 'v4:2x4x4', 'megacore', (2, 2, 1), 32
    ),
    'v4-128': SystemCharacteristics(
      'tpu', 'v4:4x4x4', 'megacore', (2, 2, 1), 64
    ),
    'v4-256': SystemCharacteristics(
      'tpu', 'v4:4x4x8', 'megacore', (2, 2, 1), 128
    ),
    'v4-512': SystemCharacteristics(
      'tpu', 'v4:4x8x8', 'megacore', (2, 2, 1), 256
    ),
    'v4-1024': SystemCharacteristics(
      'tpu', 'v4:8x8x8', 'megacore', (2, 2, 1), 512
    ),
    'v4-1536': SystemCharacteristics(
      'tpu', 'v4:8x8x12','megacore', (2, 2, 1), 768
    ),
    'v4-2048': SystemCharacteristics(
      'tpu', 'v4:8x8x16','megacore', (2, 2, 1), 1024
    ),
    'v4-4096': SystemCharacteristics(
      'tpu', 'v4:8x16x16', 'megacore', (2, 2, 1), 2048
    ),
}

def get_system_characteristics(user_facing_name):
  return UserFacingNameToSystemCharacteristics.get(user_facing_name)

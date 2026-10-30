# Copyright 2026 Google LLC
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

"""Core m3 primitives: sharding, data packing, RoPE, and segment-isolated ops.

Modules:
  * `rope.py` — rotary positional embedding (1D and 3D M-RoPE).
  * `sharding.py` — `Sharding` ABC, `DefaultSharding` rules, 5-axis mesh creation.
  * `data.py` — 5-key `M3Batch`, zero-copy 6-key iterator adapter, and pre-shift document bin-packing.
  * `layers.py` — segment-isolated causal Conv1d, n-gram hashing, and linear recurrence primitives.
"""

from maxtext.m3.core import data
from maxtext.m3.core import layers
from maxtext.m3.core import rope
from maxtext.m3.core import sharding

__all__ = [
    "data",
    "layers",
    "rope",
    "sharding",
]

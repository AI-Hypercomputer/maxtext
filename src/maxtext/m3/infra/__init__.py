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

"""Post-construction transforms applied from training setup, never from models.

Planned modules:
  * `remat.py` — path-transparent remat wrapper + `maybe_apply_remat`.
  * `scan.py` — layer stacking via `jax.lax.scan` + `maybe_apply_scan`.
  * `quantization.py` — qwix int8/fp8 rules + `maybe_quantize`.
  * `offload.py` — `to_host` for optimizer/parameter offloading.

Composition order is fixed:
`construct -> remat -> scan -> quantize -> optimizer -> host offload`.

Model files contain only zero-cost annotations (`checkpoint_name` tags and
sharding specs); they must never import from this package.
"""

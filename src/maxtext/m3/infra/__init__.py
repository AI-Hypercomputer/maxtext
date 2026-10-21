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

Modules:
  * `remat.py` -- `maybe_apply_remat`: sets the `Layers` containers' `Remat`.
  * `scan.py` -- `maybe_apply_scan`: replaces `Layers` with `ScannedLayers`.
  * `quantization.py` -- `maybe_quantize`: qwix int8 / fp8 by interception.
  * `offload.py` -- `HostOffloadedOptimizer`, optimizer state on `pinned_host`.

Composition order is fixed, and `maxtext.m3.train.loop.setup_model` is the
reference for it:
`construct -> remat -> scan -> quantize -> optimizer -> host offload`.

None of these wrap the model in another module. Remat, quantization and
offload leave the parameter tree untouched; scan changes it exactly as
legacy's scanned layout does -- a layer axis on each parameter, no extra path
segment -- and the tests under `tests/unit/m3/` assert both. Model files
contain only zero-cost annotations (`checkpoint_name` tags and sharding
constraints); they must never import from this package.
"""

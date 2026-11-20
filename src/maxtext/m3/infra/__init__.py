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

"""Post-construction transforms applied from training setup.

Modules:
  * `scan.py` — layer stacking via `jax.lax.scan` + `maybe_apply_scan`.
  * `remat.py` — path-transparent remat policies, `maybe_apply_remat`, and `check_tags`.
  * `quantization.py` — `maybe_quantize`: qwix int8 / fp8 by interception, over
    the projection layers of the decoder stack.
  * `named_scopes.py` — `jax.named_scope` per submodule, so profiles and HLO
    dumps read `decoder/layers/3/mlp/wi_0`.
  * `offload.py` — `to_memory`, `HostOffloadedOptimizer`, and
    `maybe_offload_parameters` for pinned-host optimizer/parameter offloading.
  * `aot_estimator.py` — `compile_aot_hbm_estimate`, `trace_abstract_sharding`,
    and `verify_sharding_golden` for virtual-topology CPU AOT HBM estimation.

Composition order is fixed, and `maxtext.m3.train.loop.setup_model` is the
reference for it:
`construct -> scan -> remat -> quantize -> named scopes -> host offload -> optimizer`.

None of these wrap the model in another module, and quantization leaves the
variable tree untouched (asserted). Standard `nnx.Sequential` models rely only
on `checkpoint_name` tags and sharding specs; models with custom internal scan
loops may import policy helpers from `remat.py`, never full transforms.
"""

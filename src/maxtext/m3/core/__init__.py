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

"""Core m3 primitives: the pieces NNX does not already provide.

Modules:
  * `rope.py` -- rotary positional embedding (base variant).
  * `attention.py` -- causal softmax attention as a pure function; takes a
    kernel object that can route it to a fused implementation.
  * `kernels.py` -- fused-kernel adapters (splash / tokamax attention,
    megablox / tokamax grouped matmul): legacy-config translation, the
    `shard_map` each Pallas kernel needs, nothing model-specific.
  * `sharding.py` -- `Sharding` ABC, `DefaultSharding` rules, mesh creation.
  * `layers.py` -- the `Layers` container that `infra` transforms act on.
  * `moe.py` -- routed-expert execution (dense, `ragged_dot`, and the Pallas
    executors from `kernels.py`), aux loss.
  * `interfaces.py` -- `CausalLMOutput`.

Planned modules:
  * `checkpoint.py` -- Orbax wrapper.
  * `data.py` -- data loading helpers on top of the MaxText input pipeline.

Leaf layers (`nnx.Linear`, `nnx.Einsum`, `nnx.Embed`, `nnx.RMSNorm`, ...) are
used directly from NNX and are deliberately not wrapped here.
"""

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

"""m3: Modern MaxText Models.

m3 is the pure-NNX, single-file-per-model rewrite of the MaxText modeling stack.
Models own their own attention, MLP and decoder blocks by composing NNX
primitives; scalability concerns (remat, scan, quantization, host offload) are
applied post-construction as transforms rather than being branched on inside
model code.

Subpackages:
  * `core` — primitives m3 must write itself (RoPE, sharding, checkpoint, data).
  * `infra` — post-construction transforms (remat, scan, quantization, offload).
  * `configs` — slim Pydantic config types plus the YAML defaults.
  * `models` — one self-contained file per model family, plus its sharding rules.
  * `train` — m3's own training loop (one of two supported drivers; the other is
    `maxtext.training_engine.maxtext_engine`).

This package is under active development and is not yet wired into any MaxText
entry point. See `README.md` in this directory for the architecture rules that
contributions must follow.
"""

__all__ = [
    "configs",
    "core",
    "infra",
    "models",
    "train",
]

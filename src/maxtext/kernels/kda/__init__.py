# Copyright 2023-2026 Google LLC
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

"""Kimi Delta Attention (KDA) Pallas TPU kernels: chunked gated delta rule with per-key-channel decay.

Public API:
  chunk_kda: forward + custom VJP of the KDA core on `[B, T, H, d]` inputs (see `README.md`).
  kda_recurrent_reference: the per-token recurrence in pure JAX (reference / tests).
  residual_bytes_per_layer: HBM cost of `chunk_kda(..., save_residuals=True)`.
  segment_resets: `[B, T]` document-start mask from packing segment ids.

Lower-level modules: `fused` (the fused fwd/bwd kernels), `intra` (intra-chunk matrices), `scan` (the
chunk-state recurrence kernels used by the unfused sub-block path).
"""

from maxtext.kernels.kda.api import RESIDUAL_NAME
from maxtext.kernels.kda.api import chunk_kda
from maxtext.kernels.kda.api import residual_bytes_per_layer
from maxtext.kernels.kda.api import segment_resets
from maxtext.kernels.kda.reference import kda_recurrent_reference

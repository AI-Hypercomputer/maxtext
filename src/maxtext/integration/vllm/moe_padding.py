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

"""MoE padding utilities for TPU GMM_v2 kernel alignment."""

from typing import Optional


def next_power_of_two(x: int) -> int:
  """Finds the smallest power of 2 >= x using bit manipulation.

  Args:
    x: The input number (should be an integer > 0).

  Returns:
    The smallest integer power of 2 that is >= x.
  """
  assert x > 0
  if x == 1:
    return 1
  return 1 << (x - 1).bit_length()


def compute_padded_moe_mlp_dim(
    hidden_size: Optional[int],
    moe_mlp_tp_size: int,
    num_lanes: int = 128,
) -> Optional[int]:
  """Computes padded MoE intermediate size for GMM_v2 kernel requirements.

  The GMM_v2 kernel requires the MLP dimension per expert to be at least 2x the
  number of TPU lanes (e.g. 2 * 128 = 256) to ensure efficient execution.

  Args:
    hidden_size: Unpadded MoE intermediate size (e.g. moe_intermediate_size /
      base_moe_mlp_dim).
    moe_mlp_tp_size: TP size across MLP dimensions (e.g. tp * attn_dp).
    num_lanes: Number of TPU lanes (typically 128 for TPU v5p/v6e).

  Returns:
    Padded hidden size, or hidden_size if no padding is required / hidden_size is None.
  """
  if hidden_size is None or moe_mlp_tp_size <= 0 or num_lanes <= 0:
    return hidden_size

  if (hidden_size // moe_mlp_tp_size) % (2 * num_lanes) != 0:
    min_required = 2 * num_lanes * moe_mlp_tp_size
    return next_power_of_two(max(hidden_size, min_required))

  return hidden_size

# Copyright 2023–2026 Google LLC
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

"""Kimi-K3 layer layout (pure Python, no JAX/Flax imports).

Shared by the model (`maxtext.models.kimi_k3`) and checkpoint conversion
(`checkpoint_conversion.utils.param_mapping`), so the scanned parameter layout and the
scanned checkpoint mapping cannot drift apart.
"""

from typing import NamedTuple, Sequence, Tuple

LayerSpec = Tuple[bool, bool]  # (is_linear_attn, is_moe)


def build_kimi_layer_spec(
    num_decoder_layers: int,
    full_attn_layers: Sequence[int],
    first_num_dense_layers: int,
) -> list[LayerSpec]:
  """Returns a list of (is_linear_attn, is_moe) per layer index."""
  full_attn_set = set(full_attn_layers)
  spec = []
  for idx in range(num_decoder_layers):
    is_linear_attn = idx not in full_attn_set
    is_moe = idx >= first_num_dense_layers
    spec.append((is_linear_attn, is_moe))
  return spec


class KimiScanLayout(NamedTuple):
  """Split of the Kimi-K3 stack for `scan_layers=True`.

  Attributes:
    num_prefix: Layers 0..num_prefix-1 are unrolled (`layers_{i}`).
    num_blocks: Number of scanned cycles (`scanned_blocks`), each of `len(pattern)` layers.
    suffix_start: Layers suffix_start..num_layers-1 are unrolled (`layers_{i}`).
    pattern: One (is_linear_attn, is_moe) per sub-layer of a cycle.
  """

  num_prefix: int
  num_blocks: int
  suffix_start: int
  pattern: Tuple[LayerSpec, ...]

  @property
  def cycle(self) -> int:
    return len(self.pattern)

  def scanned_position(self, layer_idx: int):
    """Returns (block, sub_layer) of a scanned layer, or None for an unrolled one."""
    if self.num_blocks == 0 or not self.num_prefix <= layer_idx < self.suffix_start:
      return None
    return divmod(layer_idx - self.num_prefix, self.cycle)

  def scanned_layer_indices(self, sub_layer: int) -> list[int]:
    """Global layer indices stacked (over blocks) into scanned sub-layer `sub_layer`."""
    return [self.num_prefix + b * self.cycle + sub_layer for b in range(self.num_blocks)]


def kimi_scan_layout_from_spec(
    num_decoder_layers: int,
    full_attn_layers: Sequence[int],
    first_num_dense_layers: int,
    cycle: int,
) -> KimiScanLayout:
  """Splits the Kimi-K3 stack into an unrolled prefix, scanned cycles and an unrolled suffix.

  The scanned region starts at the first cycle boundary at or after the dense-MLP layers
  (so every scanned layer is MoE) and spans as many consecutive whole cycles of `cycle`
  layers as repeat the first cycle's (KDA/MLA, dense/MoE) pattern. For the released
  93-layer model: prefix 0..3, 22 blocks of [KDA, KDA, KDA, MLA] over layers 4..91,
  suffix [92].

  Raises:
    ValueError: If `cycle` is not positive, or if the layers after the periodic region
      still contain a whole cycle, i.e. the layer pattern is not periodic.
  """
  num_layers = num_decoder_layers
  if cycle <= 0:
    raise ValueError(f"inhomogeneous_layer_cycle_interval must be positive, got {cycle}.")
  spec = build_kimi_layer_spec(num_layers, full_attn_layers, first_num_dense_layers)
  num_prefix = min(-(-first_num_dense_layers // cycle) * cycle, num_layers)
  pattern = tuple(spec[num_prefix : num_prefix + cycle])
  num_blocks = 0
  if len(pattern) == cycle:
    while (
        num_prefix + (num_blocks + 1) * cycle <= num_layers
        and tuple(spec[num_prefix + num_blocks * cycle : num_prefix + (num_blocks + 1) * cycle]) == pattern
    ):
      num_blocks += 1
  suffix_start = num_prefix + num_blocks * cycle
  if num_layers - suffix_start >= cycle:
    raise ValueError(
        f"Kimi-K3 scan_layers=True needs a periodic layer pattern: layers {num_prefix}..{suffix_start - 1} repeat "
        f"the cycle {pattern} (inhomogeneous_layer_cycle_interval={cycle}), but the {num_layers - suffix_start} "
        f"layers from {suffix_start} on do not (full_attn_layers={sorted(full_attn_layers)}). "
        "Fix full_attn_layers / inhomogeneous_layer_cycle_interval or set scan_layers=False."
    )
  return KimiScanLayout(num_prefix, num_blocks, suffix_start, pattern)

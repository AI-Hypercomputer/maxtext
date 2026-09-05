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

"""Utilities for GLM-5.2 Cross-Layer IndexCache (IndexShare).

References:
  GLM-5.2 / DSA IndexShare: Exploiting cross-layer token selection stability
  to reduce lightning indexer compute by 50% to 75%.
"""

from typing import Sequence


def parse_index_share_pattern(pattern: str | Sequence[str], num_layers: int) -> tuple[str, ...]:
  """Parses and validates the IndexShare pattern string.

  Args:
    pattern: Pattern string (e.g. "FSSS", "F,S,S,S", "FSSSFSSSFSSS...") or list of roles.
    num_layers: Total number of decoder layers in the model.

  Returns:
    A tuple of 'F' (Full layer) and 'S' (Shared layer) strings of length `num_layers`.

  Raises:
    ValueError: If pattern is empty, contains invalid characters, or layer 0 is not 'F'.
  """
  if isinstance(pattern, str):
    # Normalize commas/spaces/case
    clean_pattern = pattern.replace(",", "").replace(" ", "").upper()
  else:
    clean_pattern = "".join(str(x).strip().upper() for x in pattern)

  if not clean_pattern:
    raise ValueError("index_share_pattern cannot be empty.")

  invalid_chars = set(clean_pattern) - {"F", "S"}
  if invalid_chars:
    raise ValueError(
        f"Invalid characters in index_share_pattern: {invalid_chars}. Only 'F' (Full) and 'S' (Shared) are allowed."
    )

  if clean_pattern[0] != "F":
    raise ValueError(f"First layer (Layer 0) must always be 'F' (Full layer), but got '{clean_pattern[0]}'.")

  # If pattern is shorter than num_layers, repeat it periodically to fill num_layers
  if len(clean_pattern) < num_layers:
    repeats = (num_layers + len(clean_pattern) - 1) // len(clean_pattern)
    full_pattern = (clean_pattern * repeats)[:num_layers]
  elif len(clean_pattern) > num_layers:
    full_pattern = clean_pattern[:num_layers]
  else:
    full_pattern = clean_pattern

  return tuple(full_pattern)


def get_donor_layer_indices(pattern_tuple: tuple[str, ...]) -> tuple[int, ...]:
  """For each layer, returns the index of its donor Full (F) layer.

  f(l) = max{ j <= l : pattern[j] == 'F' }
  """
  donor_indices = []
  current_f = 0
  for idx, role in enumerate(pattern_tuple):
    if role == "F":
      current_f = idx
    donor_indices.append(current_f)
  return tuple(donor_indices)


def get_donor_layer_idx(layer_idx: int, pattern_tuple: tuple[str, ...]) -> int:
  """Returns the donor Full (F) layer index for a specific layer."""
  return get_donor_layer_indices(pattern_tuple)[layer_idx]


def get_served_group_sizes(pattern_tuple: tuple[str, ...]) -> tuple[int, ...]:
  """For each layer, returns the group size |Served(f(l))| of its donor F-layer.

  This is used to normalize the multi-layer distillation loss:
    L_multi_I = 1 / |Served(l)| * sum_{j in Served(l)} KL(p^(j) || q^(l))
  """
  donor_indices = get_donor_layer_indices(pattern_tuple)
  # Count how many layers each donor F-layer serves
  counts: dict[int, int] = {}
  for d in donor_indices:
    counts[d] = counts.get(d, 0) + 1

  return tuple(counts[d] for d in donor_indices)


def is_shared_layer(layer_idx: int, pattern_tuple: tuple[str, ...]) -> bool:
  """Returns True if the given layer index is a Shared (S) layer."""
  if layer_idx < 0 or layer_idx >= len(pattern_tuple):
    return False
  return pattern_tuple[layer_idx] == "S"

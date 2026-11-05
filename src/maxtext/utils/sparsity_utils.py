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

"""Helpers for selective-layer N:M weight sparsity.

These helpers are shared by config validation, Qwix rule construction,
checkpoint export and initial weight sparsification so that every code path
selects the same layers and parameters.
"""

import re
from typing import Any

# Decoder layer weights in Hugging Face checkpoints, e.g.
# "model.layers.3.mlp.down_proj.weight" or
# "model.language_model.layers.3.self_attn.q_proj.weight".
_HF_DECODER_LAYER_RE = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.")
# MoE router (gating) projections, which are never sparsified.
_HF_ROUTER_RE = re.compile(r"(\.router\.|\.gate\.weight$)")


def parse_layer_spec(spec: str, num_layers: int | None = None) -> list[int]:
  """Parses a comma-separated layer specification such as "0-13,24-31".

  Each element is either a single index ("5") or an inclusive range ("0-13").
  Elements must be given in ascending order and must not overlap.

  Args:
    spec: The layer specification string.
    num_layers: If given, every index must be smaller than this value.

  Returns:
    The sorted list of selected layer indices.

  Raises:
    ValueError: If the specification is malformed, unordered, overlapping or
      out of bounds.
  """
  layers = []
  for part in spec.split(","):
    part = part.strip()
    match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
    if not match:
      raise ValueError(
          f"Invalid layer specification {spec!r}: expected indices or ranges like '0-13,24-31', got {part!r}."
      )
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) is not None else start
    if end < start:
      raise ValueError(f"Invalid layer specification {spec!r}: range {part!r} is descending.")
    if layers and start <= layers[-1]:
      raise ValueError(f"Invalid layer specification {spec!r}: elements must be ascending and non-overlapping.")
    if num_layers is not None and end >= num_layers:
      raise ValueError(f"Invalid layer specification {spec!r}: layer {end} is out of range for {num_layers} layers.")
    layers.extend(range(start, end + 1))
  return layers


def get_sparse_layer_indices(config: Any) -> list[int] | None:
  """Returns the decoder layer indices selected by `config.sparse_layers`.

  Args:
    config: A MaxText config.

  Returns:
    The sorted layer indices, or None if `sparse_layers` is unset, meaning all
    decoder layers.

  Raises:
    ValueError: If `sparse_layers` is set together with `scan_layers=True`,
      since scanned layers share a single module path and cannot be selected
      individually.
  """
  spec = (config.sparse_layers or "").strip()
  if not spec:
    return None
  if config.scan_layers:
    raise ValueError("sparse_layers requires scan_layers=False.")
  return parse_layer_spec(spec, config.num_decoder_layers)


def is_sparsifiable_hf_param(hf_path: str, target_layers: set[int] | None, experts_only: bool) -> bool:
  """Returns whether a Hugging Face decoder weight should get N:M sparsity.

  Mirrors the Qwix rules used in training: only decoder layer weights are
  selected (never embeddings, the LM head or vision tower weights), MoE router
  projections are always skipped, and with `experts_only` only MoE expert
  weights are selected. Callers must additionally restrict this to matmul
  weights (ndim >= 2).

  Args:
    hf_path: The Hugging Face parameter name.
    target_layers: The selected decoder layer indices, or None for all layers.
    experts_only: Whether only MoE expert weights are sparsified.
  """
  match = _HF_DECODER_LAYER_RE.match(hf_path)
  if not match:
    return False
  if target_layers is not None and int(match.group(1)) not in target_layers:
    return False
  if _HF_ROUTER_RE.search(hf_path):
    return False
  if experts_only:
    return ".experts." in hf_path
  return True

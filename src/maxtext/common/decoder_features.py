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

"""Requests and extraction for optional decoder features used by downstream consumers."""

from dataclasses import dataclass
from typing import Any

import jax


@dataclass(frozen=True)
class DecoderFeatureSpec:
  """Prefill features to retain; layer indices count completed decoder layers, starting at one."""

  input_embeddings: bool = False
  hidden_layers: tuple[int, ...] = ()

  @property
  def enabled(self) -> bool:
    """Whether any features were requested."""
    return self.input_embeddings or bool(self.hidden_layers)

  @property
  def names(self) -> tuple[str, ...]:
    """Keys used in the decoder's intermediate collection and returned feature mapping."""
    embeddings = ("input_embeddings",) if self.input_embeddings else ()
    return embeddings + tuple(f"hidden_states_{layer}" for layer in self.hidden_layers)

  def validate(self, *, num_layers: int, scan_layers: bool) -> None:
    """Rejects requests the current prefill capture implementation cannot satisfy."""
    if any(layer < 1 or layer > num_layers for layer in self.hidden_layers):
      raise ValueError(f"Requested hidden layers must be between 1 and {num_layers}: {self.hidden_layers}.")
    if self.hidden_layers and scan_layers:
      raise ValueError("Hidden-layer feature capture requires scan_layers=false.")


def extract_decoder_features(intermediates: dict, spec: DecoderFeatureSpec) -> dict[str, Any]:
  """Extracts exactly the requested decoder intermediates, ignoring unrelated module outputs."""
  decoder = intermediates.get("decoder", {})
  features = {}
  for name in spec.names:
    leaves = jax.tree.leaves(decoder.get(name, ()))
    if len(leaves) != 1:
      raise ValueError(f"Expected one captured decoder feature {name!r}, got {len(leaves)}.")
    features[name] = leaves[0]
  return features

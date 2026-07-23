# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Target-alignment utilities shared by diffusion training objectives."""

import jax.numpy as jnp


def align_logits_to_targets(logits, alignment, positions=None, validity_mask=None):
  """Aligns model output positions with clean block-diffusion targets."""
  if alignment == "same_position":
    return logits
  if alignment == "shifted":
    if positions is None:
      indices = jnp.maximum(jnp.arange(logits.shape[1], dtype=jnp.int32) - 1, 0)
      return logits[:, indices, :]
    positions = jnp.asarray(positions, dtype=jnp.int32)
    if positions.shape != logits.shape[:2]:
      raise ValueError(f"positions must match logits [batch, length]; got {positions.shape} and {logits.shape[:2]}")
    if validity_mask is None:
      validity_mask = jnp.ones_like(positions, dtype=jnp.bool_)
    sequence_length = logits.shape[1]
    array_indices = jnp.broadcast_to(jnp.arange(sequence_length, dtype=jnp.int32), positions.shape)
    row_indices = jnp.broadcast_to(jnp.arange(logits.shape[0], dtype=jnp.int32)[:, None], positions.shape)
    safe_positions = jnp.clip(positions, 0, sequence_length - 1)
    updates = jnp.where(validity_mask, array_indices + 1, 0)
    position_to_index = jnp.zeros_like(positions).at[row_indices, safe_positions].max(updates)
    previous_positions = jnp.maximum(positions - 1, 0)
    source_indices = jnp.take_along_axis(position_to_index, jnp.clip(previous_positions, 0, sequence_length - 1), axis=1)
    source_indices = jnp.maximum(source_indices - 1, 0)
    return jnp.take_along_axis(logits, source_indices[..., None], axis=1)
  raise ValueError(f"Unsupported block-diffusion logit alignment: {alignment}")

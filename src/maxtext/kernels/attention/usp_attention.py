# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""USP (Ulysses-over-ring) attention layout helpers."""

from __future__ import annotations

from typing import Any

import jax

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.kernels.attention import tokamax_ring_attention
from maxtext.kernels.attention import ulysses_attention
from maxtext.utils import sharding


def is_context_parallel_usp_requested(config: Any) -> bool:
  """Returns True when the config requests USP context parallelism."""
  return config.context_parallel_strategy == "usp"


def validate_usp_runtime(
    *,
    model_mode: str,
    use_ragged_attention: bool = False,
    previous_chunk: Any = None,
    sinks: Any = None,
    indexer_mask: Any = None,
    bidirectional_mask: Any = None,
    record_max_logits: bool = False,
) -> None:
  """Validates runtime-only constraints for the USP path."""
  if model_mode != MODEL_MODE_TRAIN:
    raise ValueError("TPU USP attention is supported only for train mode.")
  if use_ragged_attention:
    raise ValueError("TPU USP attention does not support ragged attention.")
  if previous_chunk is not None:
    raise ValueError("TPU USP attention does not support chunked prefill yet.")
  if sinks is not None:
    raise ValueError("TPU USP attention does not support attention sinks.")
  if indexer_mask is not None:
    raise ValueError("TPU USP attention does not support indexer masks.")
  if bidirectional_mask is not None:
    raise ValueError("TPU USP attention does not support bidirectional masks.")
  if record_max_logits:
    raise NotImplementedError("TPU USP attention does not support record_max_logits yet.")


def call_usp_attention(
    query: Any,
    key: Any,
    value: Any,
    decoder_segment_ids_q: Any,
    ring_kernel: Any,
    ulysses_axis: str,
):
  """Runs ring attention over the Ulysses-exchanged operands and restores the layout."""
  query = ulysses_attention.ulysses_all_to_all(query, ulysses_axis)
  key = ulysses_attention.ulysses_all_to_all(key, ulysses_axis)
  value = ulysses_attention.ulysses_all_to_all(value, ulysses_axis)
  if decoder_segment_ids_q is not None:
    # One gather serves both ring operands; the result stays sequence-sharded
    # over the ring axis, the layout the ring kernel expects.
    ring_segment_ids = jax.lax.all_gather(decoder_segment_ids_q, ulysses_axis, axis=1, tiled=True)
  else:
    ring_segment_ids = None
  attention_output = tokamax_ring_attention.call_ring_attention(
      query,
      key,
      value,
      ring_segment_ids,
      ring_segment_ids,
      ring_kernel,
  )
  return ulysses_attention.inverse_ulysses_all_to_all(attention_output, ulysses_axis)


def with_usp_sequence_axes(axis_names: Any, ring_axis: str, ulysses_axis: str, sequence_dim: int) -> Any:
  """Returns axis names with the sequence dimension set to the ring and Ulysses axes."""
  if axis_names is None:
    return None
  if len(axis_names) <= sequence_dim:
    raise ValueError("TPU USP attention expects a sequence sharding dimension.")
  expected_axes = (ring_axis, ulysses_axis)
  existing_sequence_axes = sharding.mesh_axes_for_dim(axis_names[sequence_dim])
  if existing_sequence_axes and existing_sequence_axes != expected_axes:
    raise ValueError(
        "TPU USP attention expects the existing sequence sharding to be "
        f"unsharded or exactly {expected_axes}, got {existing_sequence_axes}."
    )
  return sharding.with_axis_on_dim(axis_names, expected_axes, sequence_dim)


def _validate_usp_axes_only_on_sequence(
    axis_names: Any,
    *,
    tensor_name: str,
    sequence_dim: int,
    ring_axis: str,
    ulysses_axis: str,
) -> None:
  """Raises if a USP mesh axis appears outside the sequence dimension."""
  for dim, axis_name in enumerate(axis_names):
    if dim == sequence_dim:
      continue
    dim_axes = sharding.mesh_axes_for_dim(axis_name)
    for axis in (ring_axis, ulysses_axis):
      if axis in dim_axes:
        raise ValueError(
            "TPU USP attention requires the context axes to appear only "
            f"on the sequence dimension; got {axis!r} on {tensor_name} dim {dim}."
        )


def validate_usp_mesh_axes(
    *,
    axis_names_q: Any,
    axis_names_kv: Any,
    sequence_dim_q: int,
    sequence_dim_kv: int,
    mesh: Any,
    ring_axis: str,
    ulysses_axis: str,
) -> None:
  """Validates sequence sharding before the USP exchange."""
  if ring_axis == ulysses_axis:
    raise ValueError("TPU USP attention requires context_sharding and ulysses_context_sharding to differ.")
  for axis in (ring_axis, ulysses_axis):
    if axis not in mesh.shape:
      raise ValueError(f"TPU USP attention requires mesh axis {axis!r} to exist.")
  _validate_usp_axes_only_on_sequence(
      axis_names_q,
      tensor_name="Q",
      sequence_dim=sequence_dim_q,
      ring_axis=ring_axis,
      ulysses_axis=ulysses_axis,
  )
  _validate_usp_axes_only_on_sequence(
      axis_names_kv,
      tensor_name="K/V",
      sequence_dim=sequence_dim_kv,
      ring_axis=ring_axis,
      ulysses_axis=ulysses_axis,
  )

  expected_axes = (ring_axis, ulysses_axis)
  q_sequence_axes = sharding.mesh_axes_for_dim(axis_names_q[sequence_dim_q])
  kv_sequence_axes = sharding.mesh_axes_for_dim(axis_names_kv[sequence_dim_kv])
  if q_sequence_axes != expected_axes:
    raise ValueError(
        f"TPU USP attention requires Q sequence sharding to be exactly {expected_axes}, got {q_sequence_axes}."
    )
  if kv_sequence_axes != expected_axes:
    raise ValueError(
        f"TPU USP attention requires K/V sequence sharding to be exactly {expected_axes}, got {kv_sequence_axes}."
    )

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
"""Ulysses attention layout helpers."""

from __future__ import annotations

from typing import Any

import jax

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.utils import sharding


def is_context_parallel_ulysses_requested(config: Any) -> bool:
  """Returns True when the config requests Ulysses context parallelism."""
  return config.context_parallel_strategy == "ulysses"


def validate_ulysses_runtime(
    *,
    model_mode: str,
    use_ragged_attention: bool = False,
    previous_chunk: Any = None,
    sinks: Any = None,
    indexer_mask: Any = None,
    bidirectional_mask: Any = None,
    record_max_logits: bool = False,
) -> None:
  """Validates runtime-only constraints for the Ulysses path."""
  if model_mode != MODEL_MODE_TRAIN:
    raise ValueError("TPU Ulysses attention is supported only for train mode.")
  if use_ragged_attention:
    raise ValueError("TPU Ulysses attention does not support ragged attention.")
  if previous_chunk is not None:
    raise ValueError("TPU Ulysses attention does not support chunked prefill yet.")
  if sinks is not None:
    raise ValueError("TPU Ulysses attention does not support attention sinks.")
  if indexer_mask is not None:
    raise ValueError("TPU Ulysses attention does not support indexer masks.")
  if bidirectional_mask is not None:
    raise ValueError("TPU Ulysses attention does not support bidirectional masks.")
  if record_max_logits:
    raise NotImplementedError("TPU Ulysses attention does not support record_max_logits yet.")


def with_sequence_axis(axis_names: Any, sequence_axis: str, sequence_dim: int) -> Any:
  """Returns axis names with the sequence dimension set to Ulysses."""
  if axis_names is None:
    return None
  if len(axis_names) <= sequence_dim:
    raise ValueError("TPU Ulysses attention expects a sequence sharding dimension.")
  existing_sequence_axes = sharding.mesh_axes_for_dim(axis_names[sequence_dim])
  if existing_sequence_axes and existing_sequence_axes != (sequence_axis,):
    raise ValueError(
        "TPU Ulysses attention expects the existing sequence sharding to be "
        f"unsharded or exactly {(sequence_axis,)}, got {existing_sequence_axes}."
    )
  return sharding.with_axis_on_dim(axis_names, sequence_axis, sequence_dim)


def _validate_ulysses_axis_only_on_sequence(
    axis_names: Any,
    *,
    tensor_name: str,
    sequence_dim: int,
    ulysses_axis: str,
) -> None:
  """Raises if the Ulysses mesh axis appears outside the sequence dimension."""
  for dim, axis_name in enumerate(axis_names):
    if dim == sequence_dim:
      continue
    dim_axes = sharding.mesh_axes_for_dim(axis_name)
    if ulysses_axis in dim_axes:
      raise ValueError(
          "TPU Ulysses attention requires the context axis to appear only "
          f"on the sequence dimension; got {ulysses_axis!r} on {tensor_name} dim {dim}."
      )


def validate_ulysses_mesh_axis(
    *,
    axis_names_q: Any,
    axis_names_kv: Any,
    sequence_dim_q: int,
    sequence_dim_kv: int,
    mesh: Any,
    ulysses_axis: str,
) -> None:
  """Validates sequence sharding before the Ulysses all-to-all."""
  if not ulysses_axis:
    raise ValueError("TPU Ulysses attention requires a non-empty context_sharding axis.")
  if ulysses_axis not in mesh.shape:
    raise ValueError(f"TPU Ulysses attention requires mesh axis {ulysses_axis!r} to exist.")
  _validate_ulysses_axis_only_on_sequence(
      axis_names_q,
      tensor_name="Q",
      sequence_dim=sequence_dim_q,
      ulysses_axis=ulysses_axis,
  )
  _validate_ulysses_axis_only_on_sequence(
      axis_names_kv,
      tensor_name="K/V",
      sequence_dim=sequence_dim_kv,
      ulysses_axis=ulysses_axis,
  )

  expected_axes = (ulysses_axis,)
  q_sequence_axes = sharding.mesh_axes_for_dim(axis_names_q[sequence_dim_q])
  kv_sequence_axes = sharding.mesh_axes_for_dim(axis_names_kv[sequence_dim_kv])
  if q_sequence_axes != expected_axes:
    raise ValueError(
        f"TPU Ulysses attention requires Q sequence sharding to be exactly {expected_axes}, got {q_sequence_axes}."
    )
  if kv_sequence_axes != expected_axes:
    raise ValueError(
        f"TPU Ulysses attention requires K/V sequence sharding to be exactly {expected_axes}, got {kv_sequence_axes}."
    )


def validate_dkv_sharding(
    *,
    axis_names_q: Any,
    axis_names_kv: Any,
    dkv_dim_q: int,
    dkv_dim_kv: int,
) -> None:
  """Validates that the head-dim/D_KV dimension stays local for Ulysses attention."""
  q_dkv_axes = sharding.mesh_axes_for_dim(axis_names_q[dkv_dim_q])
  kv_dkv_axes = sharding.mesh_axes_for_dim(axis_names_kv[dkv_dim_kv])
  if q_dkv_axes or kv_dkv_axes:
    raise ValueError(
        "TPU Ulysses attention does not support sharding the D_KV/head-dim "
        f"dimension; got Q axes {q_dkv_axes} and K/V axes {kv_dkv_axes}."
    )


def validate_head_sharding(
    *,
    axis_names_q: Any,
    axis_names_kv: Any,
    mesh: Any,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim_q: int,
    head_dim_kv: int,
    ulysses_size: int,
) -> None:
  """Validates local head counts before the Ulysses head/sequence exchange."""
  q_head_axes = sharding.mesh_axes_for_dim(axis_names_q[head_dim_q])
  kv_head_axes = sharding.mesh_axes_for_dim(axis_names_kv[head_dim_kv])
  q_head_shards = sharding.mesh_axes_size(mesh, q_head_axes, label="TPU Ulysses attention")
  kv_head_shards = sharding.mesh_axes_size(mesh, kv_head_axes, label="TPU Ulysses attention")
  if num_query_heads % q_head_shards != 0:
    raise ValueError(
        "TPU Ulysses attention requires num_query_heads "
        f"({num_query_heads}) to be divisible by Q head shards ({q_head_shards})."
    )
  if num_kv_heads % kv_head_shards != 0:
    raise ValueError(
        "TPU Ulysses attention requires num_kv_heads "
        f"({num_kv_heads}) to be divisible by KV head shards ({kv_head_shards})."
    )

  if num_kv_heads == 1:
    raise ValueError("TPU Ulysses attention does not support MQA with context_parallel_size > 1.")
  if q_head_axes != kv_head_axes:
    raise ValueError(
        "TPU Ulysses attention requires Q and KV head sharding to match for MHA/GQA, "
        f"got Q head axes {q_head_axes} and KV head axes {kv_head_axes}."
    )

  local_query_heads = num_query_heads // q_head_shards
  local_kv_heads = num_kv_heads // kv_head_shards
  if local_query_heads % local_kv_heads != 0:
    raise ValueError(
        "TPU Ulysses attention requires local query heads "
        f"({local_query_heads}) to be divisible by local KV heads ({local_kv_heads})."
    )
  if local_query_heads % ulysses_size != 0:
    raise ValueError(
        "TPU Ulysses attention requires local query heads "
        f"({local_query_heads}) to be divisible by context_parallel_size ({ulysses_size})."
    )
  if local_kv_heads % ulysses_size != 0:
    raise ValueError(
        "TPU Ulysses attention requires local KV heads "
        f"({local_kv_heads}) to be divisible by context_parallel_size ({ulysses_size})."
    )


def ulysses_all_to_all(tensor: Any, ulysses_axis: str):
  """Moves `[B, H, S/U, D]` to `[B, H/U, S, D]`."""
  return jax.lax.all_to_all(tensor, ulysses_axis, split_axis=1, concat_axis=2, tiled=True)


def inverse_ulysses_all_to_all(tensor: Any, ulysses_axis: str):
  """Moves `[B, H/U, S, D]` back to `[B, H, S/U, D]`."""
  return jax.lax.all_to_all(tensor, ulysses_axis, split_axis=2, concat_axis=1, tiled=True)

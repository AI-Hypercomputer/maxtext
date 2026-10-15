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
import jax.numpy as jnp

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
    attention_label: str,
) -> None:
  """Validates that the head-dim/D_KV dimension stays local for the head exchange."""
  q_dkv_axes = sharding.mesh_axes_for_dim(axis_names_q[dkv_dim_q])
  kv_dkv_axes = sharding.mesh_axes_for_dim(axis_names_kv[dkv_dim_kv])
  if q_dkv_axes or kv_dkv_axes:
    raise ValueError(
        f"{attention_label} does not support sharding the D_KV/head-dim "
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
    attention_label: str,
    allow_kv_head_replication: bool = False,
) -> None:
  """Validates local head counts before the Ulysses head/sequence exchange.

  With `allow_kv_head_replication`, fewer local KV heads than Ulysses ranks are accepted when the ranks are a
  multiple of them: the caller then repeats each KV head with `repeat_kv_heads_for_ulysses` before the exchange.
  """
  q_head_axes = sharding.mesh_axes_for_dim(axis_names_q[head_dim_q])
  kv_head_axes = sharding.mesh_axes_for_dim(axis_names_kv[head_dim_kv])
  q_head_shards = sharding.mesh_axes_size(mesh, q_head_axes, label=attention_label)
  kv_head_shards = sharding.mesh_axes_size(mesh, kv_head_axes, label=attention_label)
  if num_query_heads % q_head_shards != 0:
    raise ValueError(
        f"{attention_label} requires num_query_heads "
        f"({num_query_heads}) to be divisible by Q head shards ({q_head_shards})."
    )
  if num_kv_heads % kv_head_shards != 0:
    raise ValueError(
        f"{attention_label} requires num_kv_heads "
        f"({num_kv_heads}) to be divisible by KV head shards ({kv_head_shards})."
    )

  if num_kv_heads == 1:
    raise ValueError(f"{attention_label} does not support MQA with a Ulysses exchange size > 1.")
  if q_head_axes != kv_head_axes:
    raise ValueError(
        f"{attention_label} requires Q and KV head sharding to match for MHA/GQA, "
        f"got Q head axes {q_head_axes} and KV head axes {kv_head_axes}."
    )

  local_query_heads = num_query_heads // q_head_shards
  local_kv_heads = num_kv_heads // kv_head_shards
  if local_query_heads % local_kv_heads != 0:
    raise ValueError(
        f"{attention_label} requires local query heads "
        f"({local_query_heads}) to be divisible by local KV heads ({local_kv_heads})."
    )
  if local_query_heads % ulysses_size != 0:
    raise ValueError(
        f"{attention_label} requires local query heads "
        f"({local_query_heads}) to be divisible by the Ulysses exchange size ({ulysses_size})."
    )
  if local_kv_heads % ulysses_size != 0 and not (allow_kv_head_replication and ulysses_size % local_kv_heads == 0):
    raise ValueError(
        f"{attention_label} requires local KV heads "
        f"({local_kv_heads}) to be divisible by the Ulysses exchange size ({ulysses_size})"
        + (" or to divide it." if allow_kv_head_replication else ".")
    )


def kv_head_replication(local_kv_heads: int, ulysses_size: int) -> int:
  """How many copies of each local KV head the Ulysses exchange needs.

  The all-to-all hands rank r the r-th contiguous block of heads. When the local KV heads are a multiple of the
  ranks, every rank already receives whole KV heads (1). When the ranks are a multiple of the KV heads, each KV
  head is repeated `ulysses_size // local_kv_heads` times so that every rank receives one.
  """
  if local_kv_heads % ulysses_size == 0:
    return 1
  if ulysses_size % local_kv_heads == 0:
    return ulysses_size // local_kv_heads
  raise ValueError(
      f"TPU Ulysses attention requires local KV heads ({local_kv_heads}) and the Ulysses exchange size "
      f"({ulysses_size}) to divide one another."
  )


def repeat_kv_heads_for_ulysses(tensor: Any, replication: int):
  """Repeats each KV head of `[B, H_kv, S/U, D]` in place, `[kv0, kv0, kv1, kv1]` for 2 heads and 2 copies.

  The copies of a head must be adjacent. Splash pairs query head h with KV head h // (H_q / H_kv), so a query
  head's KV head is the one at the same relative position in the head dimension. The all-to-all keeps that
  position: rank r receives query heads [r * H_q / U, (r + 1) * H_q / U) and KV head r of the repeated tensor,
  which is a copy of KV head r // replication, the one those query heads use. Tiling the heads instead
  (`[kv0, kv1, kv0, kv1]`) would give rank 1 the second KV head for query heads of the first group.

  `jnp.repeat` is linear, so its transpose sums the gradients of the copies back onto each KV head.
  """
  if replication == 1:
    return tensor
  return jnp.repeat(tensor, replication, axis=1)


def ulysses_all_to_all(tensor: Any, ulysses_axis: str):
  """Moves `[B, H, S/U, D]` to `[B, H/U, S, D]`."""
  return jax.lax.all_to_all(tensor, ulysses_axis, split_axis=1, concat_axis=2, tiled=True)


def inverse_ulysses_all_to_all(tensor: Any, ulysses_axis: str):
  """Moves `[B, H/U, S, D]` back to `[B, H, S/U, D]`."""
  return jax.lax.all_to_all(tensor, ulysses_axis, split_axis=2, concat_axis=1, tiled=True)

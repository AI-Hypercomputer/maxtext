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
"""Tokamax ring attention helpers."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
from jax.experimental import pallas as pl

from maxtext.common.common_types import AttentionType, MODEL_MODE_TRAIN
from maxtext.kernels.tokamax_splash_attention import ring_attention_kernel
from maxtext.kernels.tokamax_splash_attention import splash_attention_kernel as tokamax_splash_kernel
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask as tokamax_splash_mask


def is_context_parallel_ring_requested(config: Any) -> bool:
  """Returns True when the config requests ring context parallelism."""
  return config.context_parallel_strategy.lower() == "ring"


def _mesh_axes_for_dim(axis_names: Any) -> tuple[Any, ...]:
  if axis_names is None:
    return ()
  if isinstance(axis_names, str):
    return (axis_names,)
  return tuple(axis for axis in axis_names if axis is not None)


def _mesh_axes_size(mesh: Any, axes: tuple[Any, ...]) -> int:
  size = 1
  for axis in axes:
    if axis not in mesh.shape:
      raise ValueError(f"TPU Tokamax ring attention requires mesh axis {axis!r} to exist.")
    size *= mesh.shape[axis]
  return size


def with_sequence_axis(axis_names: Any, ring_axis: str, sequence_dim: int) -> Any:
  """Returns axis names with the sequence dimension set to the ring axis."""
  if axis_names is None:
    return None
  if len(axis_names) <= sequence_dim:
    raise ValueError("TPU Tokamax ring attention expects a sequence sharding dimension.")
  axes = list(axis_names)
  existing_sequence_axes = _mesh_axes_for_dim(axes[sequence_dim])
  if existing_sequence_axes and existing_sequence_axes != (ring_axis,):
    raise ValueError(
        "TPU Tokamax ring attention expects the existing sequence sharding to be "
        f"unsharded or exactly {(ring_axis,)}, got {existing_sequence_axes}."
    )
  axes[sequence_dim] = ring_axis
  if isinstance(axis_names, jax.sharding.PartitionSpec):
    return jax.sharding.PartitionSpec(*axes, unreduced=axis_names.unreduced, reduced=axis_names.reduced)
  if isinstance(axis_names, tuple):
    return tuple(axes)
  return axes


def _validate_ring_axis_only_on_sequence(
    axis_names: Any,
    *,
    tensor_name: str,
    sequence_dim: int,
    ring_axis: str,
) -> None:
  """Raises if the ring mesh axis appears outside the sequence dimension."""
  for dim, axis_name in enumerate(axis_names):
    if dim == sequence_dim:
      continue
    dim_axes = _mesh_axes_for_dim(axis_name)
    if ring_axis in dim_axes:
      raise ValueError(
          "TPU Tokamax ring attention requires the context axis to appear only "
          f"on the sequence dimension; got {ring_axis!r} on {tensor_name} dim {dim}."
      )


def validate_dkv_sharding(
    *,
    axis_names_q: Any,
    axis_names_kv: Any,
    dkv_dim_q: int,
    dkv_dim_kv: int,
) -> None:
  """Validates that the head-dim/D_KV dimension stays local for ring attention."""
  if len(axis_names_q) <= dkv_dim_q or len(axis_names_kv) <= dkv_dim_kv:
    raise ValueError("TPU Tokamax ring attention expects Q/K/V tensors with a D_KV dimension.")
  q_dkv_axes = _mesh_axes_for_dim(axis_names_q[dkv_dim_q])
  kv_dkv_axes = _mesh_axes_for_dim(axis_names_kv[dkv_dim_kv])
  if q_dkv_axes or kv_dkv_axes:
    raise ValueError(
        "TPU Tokamax ring attention does not support sharding the D_KV/head-dim "
        f"dimension; got Q axes {q_dkv_axes} and K/V axes {kv_dkv_axes}."
    )


def validate_tokamax_ring_runtime(
    *,
    model_mode: str,
    use_ragged_attention: bool = False,
    previous_chunk: Any = None,
    sinks: Any = None,
    indexer_mask: Any = None,
    bidirectional_mask: Any = None,
    record_max_logits: bool = False,
    attention_type: AttentionType = AttentionType.GLOBAL,
) -> None:
  """Validates runtime-only constraints for the MaxText ring path."""
  if model_mode != MODEL_MODE_TRAIN:
    raise ValueError("TPU Tokamax ring attention is supported only for train mode.")
  if use_ragged_attention:
    raise ValueError("TPU Tokamax ring attention does not support ragged attention.")
  if previous_chunk is not None:
    raise ValueError("TPU Tokamax ring attention does not support chunked prefill yet.")
  if sinks is not None:
    raise ValueError("TPU Tokamax ring attention does not support attention sinks.")
  if indexer_mask is not None:
    raise ValueError("TPU Tokamax ring attention does not support indexer masks.")
  if bidirectional_mask is not None:
    raise ValueError("TPU Tokamax ring attention does not support bidirectional masks.")
  if record_max_logits:
    raise NotImplementedError("TPU Tokamax ring attention does not support record_max_logits yet.")
  if attention_type != AttentionType.GLOBAL:
    raise ValueError("TPU Tokamax ring attention is initially supported only for global causal attention.")


def validate_ring_mesh_axis(
    *,
    axis_names_q: Any,
    axis_names_kv: Any,
    sequence_dim_q: int,
    sequence_dim_kv: int,
    mesh: Any,
    ring_axis: str,
) -> None:
  """Validates sequence sharding before ring attention."""
  if not ring_axis:
    raise ValueError("TPU Tokamax ring attention requires a non-empty context_sharding axis.")
  if ring_axis not in mesh.shape:
    raise ValueError(f"TPU Tokamax ring attention requires mesh axis {ring_axis!r} to exist.")

  if sequence_dim_q < 0 or sequence_dim_kv < 0:
    raise ValueError("TPU Tokamax ring attention expects non-negative sequence sharding dimensions.")
  if len(axis_names_q) <= sequence_dim_q or len(axis_names_kv) <= sequence_dim_kv:
    raise ValueError("TPU Tokamax ring attention expects Q/K/V tensors with a sequence sharding dimension.")
  _validate_ring_axis_only_on_sequence(
      axis_names_q,
      tensor_name="Q",
      sequence_dim=sequence_dim_q,
      ring_axis=ring_axis,
  )
  _validate_ring_axis_only_on_sequence(
      axis_names_kv,
      tensor_name="K/V",
      sequence_dim=sequence_dim_kv,
      ring_axis=ring_axis,
  )
  expected_axes = (ring_axis,)
  q_sequence_axes = _mesh_axes_for_dim(axis_names_q[sequence_dim_q])
  key_value_sequence_axes = _mesh_axes_for_dim(axis_names_kv[sequence_dim_kv])
  if q_sequence_axes != expected_axes:
    raise ValueError(
        "TPU Tokamax ring attention requires Q sequence sharding to be exactly "
        f"{expected_axes}, got {q_sequence_axes}."
    )
  if key_value_sequence_axes != expected_axes:
    raise ValueError(
        "TPU Tokamax ring attention requires K/V sequence sharding to be exactly "
        f"{expected_axes}, got {key_value_sequence_axes}."
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
) -> None:
  """Validates that local head layout preserves GQA/MQA head mapping."""
  if len(axis_names_q) <= head_dim_q or len(axis_names_kv) <= head_dim_kv:
    raise ValueError("TPU Tokamax ring attention expects Q/K/V tensors with a head sharding dimension.")
  if num_query_heads <= 0 or num_kv_heads <= 0:
    raise ValueError("TPU Tokamax ring attention requires positive Q and KV head counts.")

  q_head_axes = _mesh_axes_for_dim(axis_names_q[head_dim_q])
  kv_head_axes = _mesh_axes_for_dim(axis_names_kv[head_dim_kv])
  q_head_shards = _mesh_axes_size(mesh, q_head_axes)
  kv_head_shards = _mesh_axes_size(mesh, kv_head_axes)
  if num_query_heads % q_head_shards != 0:
    raise ValueError(
        "TPU Tokamax ring attention requires num_query_heads "
        f"({num_query_heads}) to be divisible by Q head shards ({q_head_shards})."
    )
  if num_kv_heads % kv_head_shards != 0:
    raise ValueError(
        "TPU Tokamax ring attention requires num_kv_heads "
        f"({num_kv_heads}) to be divisible by KV head shards ({kv_head_shards})."
    )

  if num_kv_heads == 1:
    if kv_head_axes:
      raise ValueError("TPU Tokamax ring attention does not support sharding the single MQA KV head.")
    return

  if q_head_axes != kv_head_axes:
    raise ValueError(
        "TPU Tokamax ring attention requires Q and KV head sharding to match for MHA/GQA, "
        f"got Q head axes {q_head_axes} and KV head axes {kv_head_axes}."
    )
  local_query_heads = num_query_heads // q_head_shards
  local_kv_heads = num_kv_heads // kv_head_shards
  if local_query_heads % local_kv_heads != 0:
    raise ValueError(
        "TPU Tokamax ring attention requires local query heads "
        f"({local_query_heads}) to be divisible by local KV heads ({local_kv_heads})."
    )


def build_splash_config(
    config: Any,
    *,
    q_seq_len: int,
    kv_seq_len: int,
    context_parallel_size: int,
    attn_logits_soft_cap: float | None = None,
) -> Any:
  """Converts MaxText Splash config fields into Tokamax `SplashConfig`."""
  if context_parallel_size <= 1:
    raise ValueError("context_parallel_size must be > 1 for ring attention.")
  dq_reduction_steps = config.dq_reduction_steps
  q_seq_len_per_shard = q_seq_len // context_parallel_size
  kv_seq_len_per_shard = kv_seq_len // context_parallel_size
  block_q = min(config.sa_block_q, q_seq_len_per_shard)
  block_kv = min(config.sa_block_kv, kv_seq_len_per_shard)
  block_kv_compute = min(config.sa_block_kv_compute, kv_seq_len_per_shard)
  block_q_dkv = min(config.sa_block_q_dkv, q_seq_len_per_shard)
  block_kv_dkv = min(config.sa_block_kv_dkv, kv_seq_len_per_shard)
  block_kv_dkv_compute = min(config.sa_block_kv_dkv_compute, kv_seq_len_per_shard)
  return tokamax_splash_kernel.SplashConfig(
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      block_q_dkv=block_q_dkv,
      block_kv_dkv=block_kv_dkv,
      block_kv_dkv_compute=block_kv_dkv_compute,
      use_fused_bwd_kernel=True,
      q_layout=tokamax_splash_kernel.QKVLayout[config.sa_q_layout],
      k_layout=tokamax_splash_kernel.QKVLayout[config.sa_k_layout],
      v_layout=tokamax_splash_kernel.QKVLayout[config.sa_v_layout],
      attn_logits_soft_cap=attn_logits_soft_cap,
      residual_checkpoint_name="context",
      use_base2_exp=False,
      fwd_cost_estimate=pl.CostEstimate(flops=config.cost_estimate_flops_fwd, transcendentals=0, bytes_accessed=0)
      if config.cost_estimate_flops_fwd >= 0
      else None,
      bwd_cost_estimate=pl.CostEstimate(flops=config.cost_estimate_flops_bwd, transcendentals=0, bytes_accessed=0)
      if config.cost_estimate_flops_bwd >= 0
      else None,
      dq_reduction_steps=dq_reduction_steps if dq_reduction_steps > 0 else None,
      use_experimental_scheduler=config.use_splash_scheduler,
  )


def _make_causal_mask(shape: tuple[int, int], context_parallel_size: int):
  """Builds a lazy causal mask for ring attention."""
  if context_parallel_size <= 1:
    raise ValueError("context_parallel_size must be > 1 for ring attention.")
  return tokamax_splash_mask.CausalMask(shape=shape, shard_count=context_parallel_size)


def make_sharded_ring_attention_kernel(
    config: Any,
    *,
    query: Any,
    key: Any,
    context_parallel_size: int,
    ring_axis: str,
    attn_logits_soft_cap: float | None,
    maybe_shard_with_pspec: Any,
):
  """Builds and shards the Tokamax ring attention kernel for MaxText."""
  splash_config = build_splash_config(
      config,
      q_seq_len=query.shape[2],
      kv_seq_len=key.shape[2],
      context_parallel_size=context_parallel_size,
      attn_logits_soft_cap=attn_logits_soft_cap,
  )
  if config.use_max_logit_estimate > 0:
    splash_config = dataclasses.replace(splash_config, max_logit_const=config.use_max_logit_estimate)

  mask = _make_causal_mask(
      (query.shape[2], key.shape[2]),
      context_parallel_size,
  )

  @functools.partial(jax.jit, static_argnames=["single_head_mask"])
  def wrap_ring_kernel(single_head_mask):
    return ring_attention_kernel.make_ring_attention(
        single_head_mask,
        config=splash_config,
        is_mqa=False,
        save_residuals=False,
        ring_axis=ring_axis,
        q_seq_shards=context_parallel_size,
        kv_seq_shards=context_parallel_size,
    )

  ring_kernel = wrap_ring_kernel(mask)
  ring_kernel_spec = ring_kernel.manual_sharding_spec()
  ring_kernel = jax.tree.map(
      lambda arr, spec: None if arr is None else maybe_shard_with_pspec(arr, spec),
      ring_kernel,
      ring_kernel_spec,
      is_leaf=lambda x: x is None,
  )
  return splash_config, ring_kernel, ring_kernel_spec


def call_ring_attention(
    query: Any,
    key: Any,
    value: Any,
    decoder_segment_ids_q: Any,
    decoder_segment_ids_kv: Any,
    ring_kernel: Any,
):
  """Calls a Tokamax ring attention kernel over the MaxText batch dimension."""
  if (decoder_segment_ids_q is None) != (decoder_segment_ids_kv is None):
    raise ValueError("decoder_segment_ids_q and decoder_segment_ids_kv must both be set or both be None.")
  if decoder_segment_ids_q is None:
    return jax.vmap(lambda q, k, v: ring_kernel(q, k, v, None), in_axes=(0, 0, 0))(query, key, value)

  if decoder_segment_ids_q.ndim != 2 or decoder_segment_ids_kv.ndim != 2:
    raise ValueError("decoder_segment_ids_q and decoder_segment_ids_kv must have shape [batch, sequence].")
  if decoder_segment_ids_q.shape[0] != query.shape[0] or decoder_segment_ids_kv.shape[0] != key.shape[0]:
    raise ValueError("decoder_segment_ids batch dimension must match query/key batch dimension.")
  if decoder_segment_ids_q.shape[1] != query.shape[2]:
    raise ValueError("decoder_segment_ids_q sequence dimension must match query sequence dimension.")
  if decoder_segment_ids_kv.shape[1] != key.shape[2]:
    raise ValueError("decoder_segment_ids_kv sequence dimension must match key sequence dimension.")

  def call_one(q, k, v, q_segment_ids, kv_segment_ids):
    segment_ids = ring_attention_kernel.SegmentIds(q_segment_ids, kv_segment_ids)
    return ring_kernel(q, k, v, segment_ids)

  return jax.vmap(call_one, in_axes=(0, 0, 0, 0, 0))(query, key, value, decoder_segment_ids_q, decoder_segment_ids_kv)

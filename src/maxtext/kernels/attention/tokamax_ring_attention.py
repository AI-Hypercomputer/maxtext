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
"""MaxText integration for Tokamax ring attention.

Tokamax 0.0.12 source:
https://github.com/openxla/tokamax/tree/4936e75/tokamax/_src/ops/experimental/tpu/splash_attention
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.kernels.tokamax_splash_attention import ring_attention_kernel
from maxtext.kernels.tokamax_splash_attention import splash_attention_kernel as tokamax_splash_kernel
from maxtext.kernels.tokamax_splash_attention import splash_attention_mask as tokamax_splash_mask
from maxtext.utils import max_utils
from maxtext.utils import sharding


def is_context_parallel_ring_requested(config: Any) -> bool:
  """Returns True when the config requests ring context parallelism."""
  return config.context_parallel_strategy.lower() == "ring"


def with_sequence_axis(axis_names: Any, ring_axis: str, sequence_dim: int) -> Any:
  """Returns axis names with the sequence dimension set to the ring axis."""
  if axis_names is None:
    return None
  if len(axis_names) <= sequence_dim:
    raise ValueError("TPU Tokamax ring attention expects a sequence sharding dimension.")
  existing_sequence_axes = sharding.mesh_axes_for_dim(axis_names[sequence_dim])
  if existing_sequence_axes and existing_sequence_axes != (ring_axis,):
    raise ValueError(
        "TPU Tokamax ring attention expects the existing sequence sharding to be "
        f"unsharded or exactly {(ring_axis,)}, got {existing_sequence_axes}."
    )
  return sharding.with_axis_on_dim(axis_names, ring_axis, sequence_dim)


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
    dim_axes = sharding.mesh_axes_for_dim(axis_name)
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
  q_dkv_axes = sharding.mesh_axes_for_dim(axis_names_q[dkv_dim_q])
  kv_dkv_axes = sharding.mesh_axes_for_dim(axis_names_kv[dkv_dim_kv])
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
  if bidirectional_mask is not None:
    raise ValueError("TPU Tokamax ring attention does not support bidirectional masks.")
  if record_max_logits:
    raise NotImplementedError("TPU Tokamax ring attention does not support record_max_logits yet.")


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
  q_sequence_axes = sharding.mesh_axes_for_dim(axis_names_q[sequence_dim_q])
  key_value_sequence_axes = sharding.mesh_axes_for_dim(axis_names_kv[sequence_dim_kv])
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
  q_head_axes = sharding.mesh_axes_for_dim(axis_names_q[head_dim_q])
  kv_head_axes = sharding.mesh_axes_for_dim(axis_names_kv[head_dim_kv])
  q_head_shards = sharding.mesh_axes_size(mesh, q_head_axes, label="TPU Tokamax ring attention")
  kv_head_shards = sharding.mesh_axes_size(mesh, kv_head_axes, label="TPU Tokamax ring attention")
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
    load_balanced: bool = False,
) -> Any:
  """Converts MaxText Splash config fields into Tokamax `SplashConfig`.

  `load_balanced` says whether Q/K/V arrive in DUAL_CHUNK_SWAP order, i.e. whether the ring runs the
  load-balanced causal mask.
  """
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
  if load_balanced and block_q_dkv % tokamax_splash_kernel.NUM_LANES != 0:
    raise ValueError(
        "TPU Tokamax ring attention with a load-balanced causal mask requires "
        f"sa_block_q_dkv ({block_q_dkv}) to be a multiple of {tokamax_splash_kernel.NUM_LANES} after clamping."
    )
  # Ring uses the dynamic-grid dKV path, so mirror Splash's small-kv_steps guard here.
  if dq_reduction_steps == 3 and kv_seq_len_per_shard // block_kv_dkv <= 3:
    dq_reduction_steps = 0
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
      ring_scan_unroll=config.ring_scan_unroll,
      bwd_dkv_megacore=config.sa_bwd_dkv_megacore,
  )


def _make_causal_mask(shape: tuple[int, int], context_parallel_size: int, *, load_balanced: bool = False):
  """Builds a lazy causal mask for ring attention."""
  if context_parallel_size <= 1:
    raise ValueError("context_parallel_size must be > 1 for ring attention.")
  mask = tokamax_splash_mask.CausalMask(shape=shape, shard_count=context_parallel_size)
  if load_balanced:
    sequence_indices = max_utils.reorder_mask_load_balancing(
        np.arange(shape[0], dtype=np.int32), context_parallel_size, 0
    )
    mask.q_sequence = sequence_indices
    mask.kv_sequence = sequence_indices
  return mask


def load_balance_permutations(
    context_parallel_size: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
  """Source-to-destination rank pairs that move a sequence into DUAL_CHUNK_SWAP order.

  With N context ranks the sequence is 2N chunks. In natural order rank d holds chunks 2d and 2d + 1; in
  DUAL_CHUNK_SWAP order (`max_utils.reorder_sequence`) rank r holds chunks r and 2N - 1 - r. Chunk c
  therefore moves to rank c when c < N and to rank 2N - 1 - c otherwise.

  The first tuple moves every rank's first half (chunk 2d), the second tuple every rank's second half
  (chunk 2d + 1). Each is a permutation of the ranks, so each is one `lax.ppermute`. For N = 4 they are
  {0->0, 1->2, 2->3, 3->1} and {0->1, 1->3, 2->2, 3->0}.
  """
  n = context_parallel_size
  if n < 2 or n % 2 != 0:
    raise ValueError(f"DUAL_CHUNK_SWAP load balancing requires an even context_parallel_size >= 2, got {n}.")

  def destination(chunk: int) -> int:
    return chunk if chunk < n else 2 * n - 1 - chunk

  first_halves = tuple((rank, destination(2 * rank)) for rank in range(n))
  second_halves = tuple((rank, destination(2 * rank + 1)) for rank in range(n))
  return first_halves, second_halves


def _to_load_balanced_shard(x: jax.Array, axis_name: str, context_parallel_size: int, seq_dim: int) -> jax.Array:
  """Shard-local body of the natural -> DUAL_CHUNK_SWAP reorder (inside `shard_map`)."""
  first_halves, second_halves = load_balance_permutations(context_parallel_size)
  first_half, second_half = jnp.split(x, 2, axis=seq_dim)
  from_first = lax.ppermute(first_half, axis_name, perm=first_halves)
  from_second = lax.ppermute(second_half, axis_name, perm=second_halves)
  # An even rank r receives chunk r (< N) from the first-half permute and chunk 2N - 1 - r from the
  # second; an odd rank receives them the other way round.
  is_even_rank = lax.axis_index(axis_name) % 2 == 0
  lead = jnp.where(is_even_rank, from_first, from_second)
  tail = jnp.where(is_even_rank, from_second, from_first)
  return jnp.concatenate([lead, tail], axis=seq_dim)


def _to_natural_shard(x: jax.Array, axis_name: str, context_parallel_size: int, seq_dim: int) -> jax.Array:
  """Shard-local body of the DUAL_CHUNK_SWAP -> natural reorder: the inverse of `_to_load_balanced_shard`."""
  first_halves, second_halves = load_balance_permutations(context_parallel_size)
  lead, tail = jnp.split(x, 2, axis=seq_dim)
  is_even_rank = lax.axis_index(axis_name) % 2 == 0
  to_first = jnp.where(is_even_rank, lead, tail)
  to_second = jnp.where(is_even_rank, tail, lead)
  first_half = lax.ppermute(to_first, axis_name, perm=tuple((dst, src) for src, dst in first_halves))
  second_half = lax.ppermute(to_second, axis_name, perm=tuple((dst, src) for src, dst in second_halves))
  return jnp.concatenate([first_half, second_half], axis=seq_dim)


def reorder_for_load_balance(
    x: jax.Array | None,
    *,
    mesh: Any,
    axis_name: str,
    batch_axes: Any,
    to_natural: bool = False,
    seq_dim: int = 1,
) -> jax.Array | None:
  """Moves a sequence-sharded array between natural and DUAL_CHUNK_SWAP order over one mesh axis.

  The result equals `max_utils.reorder_sequence(x, cp_size, seq_dim, to_contiguous=to_natural)`, but it is
  built as two point-to-point `lax.ppermute`s of half a shard each instead of a global reshape/split/stack
  that the SPMD partitioner may lower to an all-gather. Its transpose is the opposite reorder.

  Args:
    x: array whose `seq_dim` is sharded over `axis_name`; dim 0 is the batch. None passes through.
    mesh: the device mesh.
    axis_name: the context mesh axis the sequence is sharded over.
    batch_axes: the PartitionSpec entry for dim 0 (the batch sharding of the surrounding activations). It is
      dropped for arrays whose batch does not divide over it, e.g. broadcast position ids.
    to_natural: False for natural -> DUAL_CHUNK_SWAP, True for the inverse.
    seq_dim: the sequence dimension.
  """
  if x is None:
    return None
  context_parallel_size = mesh.shape[axis_name]
  if x.shape[seq_dim] % (2 * context_parallel_size) != 0:
    raise ValueError(
        f"DUAL_CHUNK_SWAP load balancing needs the sequence length ({x.shape[seq_dim]}) to be divisible by "
        f"2 * context_parallel_size ({2 * context_parallel_size})."
    )
  batch_shards = sharding.mesh_axes_size(mesh, sharding.mesh_axes_for_dim(batch_axes), label="load balance reorder")
  spec = [None] * x.ndim
  spec[0] = batch_axes if x.shape[0] % batch_shards == 0 else None
  spec[seq_dim] = axis_name
  shard_fn = _to_natural_shard if to_natural else _to_load_balanced_shard
  return jax.shard_map(
      functools.partial(
          shard_fn,
          axis_name=axis_name,
          context_parallel_size=context_parallel_size,
          seq_dim=seq_dim,
      ),
      mesh=mesh,
      in_specs=jax.sharding.PartitionSpec(*spec),
      out_specs=jax.sharding.PartitionSpec(*spec),
  )(x)


def make_sharded_ring_attention_kernel(
    config: Any,
    *,
    query: Any,
    key: Any,
    context_parallel_size: int,
    ring_axis: str,
    attn_logits_soft_cap: float | None,
    maybe_shard_with_pspec: Any,
    load_balanced: bool,
    mask: Any = None,
):
  """Builds and shards the Tokamax ring attention kernel for MaxText.

  `load_balanced` says whether Q/K/V arrive in DUAL_CHUNK_SWAP order: either the input pipeline reordered
  the batch (`context_parallel_load_balance`) or the attention layer reordered its own input
  (`context_parallel_attention_load_balance`). The caller decides; the config flag alone cannot tell the two
  apart from a batch in natural order.
  """
  splash_config = build_splash_config(
      config,
      q_seq_len=query.shape[2],
      kv_seq_len=key.shape[2],
      context_parallel_size=context_parallel_size,
      attn_logits_soft_cap=attn_logits_soft_cap,
      load_balanced=load_balanced,
  )
  if config.use_max_logit_estimate > 0:
    splash_config = dataclasses.replace(splash_config, max_logit_const=config.use_max_logit_estimate)

  if mask is None:
    # When using the indexer, causal masking is unified into the dynamic indexer_mask
    # and applied dynamically per block; use FullMask to avoid duplicate static masks.
    if getattr(config, "use_indexer", False):
      mask = tokamax_splash_mask.FullMask((query.shape[2], key.shape[2]))
    else:
      mask = _make_causal_mask(
          (query.shape[2], key.shape[2]),
          context_parallel_size,
          load_balanced=load_balanced,
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
    indexer_mask: Any = None,
):
  """Calls a Tokamax ring attention kernel over the MaxText batch dimension."""
  if (decoder_segment_ids_q is None) != (decoder_segment_ids_kv is None):
    raise ValueError("decoder_segment_ids_q and decoder_segment_ids_kv must both be set or both be None.")
  # Vectorize execution across batch dimension, threading indexer_mask when present.
  # Note: ring_kernel expects positional arguments (q, k, v, segment_ids, sinks, indexer_mask).
  if decoder_segment_ids_q is None:
    if indexer_mask is None:
      return jax.vmap(lambda q, k, v: ring_kernel(q, k, v, None, None, None), in_axes=(0, 0, 0))(query, key, value)
    return jax.vmap(
        lambda q, k, v, im: ring_kernel(q, k, v, None, None, im),
        in_axes=(0, 0, 0, 0),
    )(query, key, value, indexer_mask)

  def call_one(q, k, v, q_segment_ids, kv_segment_ids, im=None):
    segment_ids = ring_attention_kernel.SegmentIds(q_segment_ids, kv_segment_ids)
    return ring_kernel(q, k, v, segment_ids, None, im)

  if indexer_mask is None:
    return jax.vmap(call_one, in_axes=(0, 0, 0, 0, 0))(query, key, value, decoder_segment_ids_q, decoder_segment_ids_kv)
  return jax.vmap(call_one, in_axes=(0, 0, 0, 0, 0, 0))(
      query, key, value, decoder_segment_ids_q, decoder_segment_ids_kv, indexer_mask
  )

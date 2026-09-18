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

"""DSv3 MLA implementation."""

from collections.abc import Mapping
import functools
import math
from typing import Any, Protocol

import jax
import jax.experimental.compute_on
import jax.numpy as jnp
import jaxtyping as jt
from maxtext.models.deepseek_lineage import dsv3_types
from maxtext.models.deepseek_lineage import ops
from maxtext.models.deepseek_lineage import splash_attention_mask as tokamax_splash_mask
from maxtext.models.deepseek_lineage import tokamax_splash_attention_kernel
import typeguard

compute_on = jax.experimental.compute_on.compute_on


class NormFn(Protocol):

  def __call__(
      self,
      x: jt.Num[jax.Array, "*input_dims"],
      scale: jt.Num[jax.Array, "..."],
  ) -> jt.Num[jax.Array, "*input_dims"]:
    ...


class RopeFn(Protocol):

  def __call__(
      self,
      x: jt.Num[jax.Array, "*BT N D"],
      freqs: tuple[
          jt.Num[jax.Array, "*BT 1 D"],
          jt.Num[jax.Array, "*BT 1 D"],
      ],
  ) -> jt.Num[jax.Array, "*BT N D"]:
    ...


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def q_down_projection(
    x: jt.Num[jax.Array, "B T D"],
    wq_down: jt.Num[jax.Array, "D Cq"],
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.Num[jax.Array, "B T Cq"]:
  """Performs query down projection."""
  x_sharding = getattr(jax.typeof(x), "sharding", None)
  assert isinstance(x_sharding, jax.sharding.NamedSharding)
  physical_attention_axis = ops.physical_pspec(jax.sharding.PartitionSpec("attention"), axis_mapping)[0]
  out_spec = jax.sharding.PartitionSpec(x_sharding.spec[0], None, None)

  @compute_on(
      compute_type="device",
      out_memory_spaces=jax.memory.Space.Device,
  )
  def _q_down_all_gather(x: jax.Array) -> jax.Array:
    return jax.lax.all_gather(
        x,
        axis_name=physical_attention_axis,
        axis=1,
        tiled=True,
        to="invarying",
    )

  def _q_down(x: jax.Array, wq_down: jax.Array) -> jax.Array:
    with jax.named_scope("q_down_projection"):
      cq = dot(x, wq_down)
    return _q_down_all_gather(cq)

  return jax.shard_map(
      _q_down,
      mesh=mesh,
      out_specs=out_spec,
      check_vma=True,
  )(x, wq_down)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def q_up_from_q_down(
    q_down: jt.Num[jax.Array, "B T Cq"],
    wq_up: jt.Num[jax.Array, "Cq N QK_plus_R"],
    wq_norm_scale: jt.Num[jax.Array, "Cq"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    *,
    qk_head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: NormFn,
    rope_fn: RopeFn,
) -> jt.Num[jax.Array, "B T N QK_plus_R"]:
  """Computes full query from compressed query."""
  qk_and_rope_head_dim = qk_head_dim + rope_head_dim
  softmax_scale = qk_and_rope_head_dim**-0.5
  if max_position_embeddings > original_max_position_embeddings:
    m = 0.1 * mscale * math.log(rope_factor) + 1.0
    softmax_scale = softmax_scale * m * m
  cq = norm_fn(q_down, wq_norm_scale)
  with jax.named_scope("q_up_projection"):
    q = dot(cq, wq_up)
  q_nope, q_pe = jnp.split(q, [qk_head_dim], axis=-1)
  q_pe = rope_fn(q_pe, yarn_freqs)
  return jnp.concatenate([q_nope, q_pe], axis=-1) * softmax_scale


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def q_projection(
    x: jt.Num[jax.Array, "B T D"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    wq_down: jt.Num[jax.Array, "D Cq"],
    wq_up: jt.Num[jax.Array, "Cq N QK_plus_R"],
    wq_norm_scale: jt.Num[jax.Array, "Cq"],
    *,
    qk_head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.Num[jax.Array, "B T N QK_plus_R"]:
  """Performs query projection."""
  q_down = q_down_projection(
      x,
      wq_down,
      mesh=mesh,
      axis_mapping=axis_mapping,
  )
  return q_up_from_q_down(
      q_down,
      wq_up,
      wq_norm_scale,
      yarn_freqs,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def kv_down_projection(
    x: jt.Num[jax.Array, "B T D"],
    wkv_down: jt.Num[jax.Array, "D Ckv_plus_R"],
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.Num[jax.Array, "B T Ckv_plus_R"]:
  """Performs key/value down projection."""
  x_sharding = getattr(jax.typeof(x), "sharding", None)
  assert isinstance(x_sharding, jax.sharding.NamedSharding)
  physical_attention_axis = ops.physical_pspec(jax.sharding.PartitionSpec("attention"), axis_mapping)[0]
  out_spec = jax.sharding.PartitionSpec(x_sharding.spec[0], None, None)

  @compute_on(
      compute_type="device",
      out_memory_spaces=jax.memory.Space.Device,
  )
  def _kv_down_all_gather(x: jax.Array) -> jax.Array:
    return jax.lax.all_gather(
        x,
        axis_name=physical_attention_axis,
        axis=1,
        tiled=True,
        to="invarying",
    )

  def _kv_down(x: jax.Array, wkv_down: jax.Array) -> jax.Array:
    with jax.named_scope("kv_down_projection"):
      ckv_and_k_pe = dot(x, wkv_down)
    return _kv_down_all_gather(ckv_and_k_pe)

  return jax.shard_map(
      _kv_down,
      mesh=mesh,
      out_specs=out_spec,
      check_vma=True,
  )(x, wkv_down)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def kv_up_from_kv_down(
    kv_down: jt.Num[jax.Array, "B T Ckv_plus_R"],
    wk_up: jt.Num[jax.Array, "Ckv N QK"],
    wv_up: jt.Num[jax.Array, "Ckv N V"],
    wkv_norm_scale: jt.Num[jax.Array, "Ckv"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    *,
    kv_lora_rank: int,
    num_query_heads: int,
    norm_fn: NormFn,
    rope_fn: RopeFn,
) -> tuple[
    jt.Num[jax.Array, "B T N QK_plus_R"],
    jt.Num[jax.Array, "B T N V"],
]:
  """Computes key and value from compressed key/value."""
  ckv, k_pe = jnp.split(kv_down, [kv_lora_rank], axis=-1)
  ckv = norm_fn(ckv, wkv_norm_scale)
  with jax.named_scope("k_up_projection"):
    k_nope = dot(ckv, wk_up)
  with jax.named_scope("v_up_projection"):
    v = dot(ckv, wv_up)

  # Add head dimension of size 1 to k_pe.
  k_pe = jnp.expand_dims(k_pe, axis=2)
  k_pe = rope_fn(k_pe, yarn_freqs)
  sharding = getattr(jax.typeof(k_nope), "sharding", None)
  # Broadcast k_pe to have num_query_heads heads.
  k_pe = jnp.broadcast_to(
      k_pe,
      k_pe.shape[:2] + (num_query_heads, k_pe.shape[-1]),
      out_sharding=sharding,
  )
  return jnp.concatenate([k_nope, k_pe], axis=-1), v


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def kv_projection(
    x: jt.Num[jax.Array, "B T D"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    wkv_down: jt.Num[jax.Array, "D Ckv_plus_R"],
    wk_up: jt.Num[jax.Array, "Ckv N QK"],
    wv_up: jt.Num[jax.Array, "Ckv N V"],
    wkv_norm_scale: jt.Num[jax.Array, "Ckv"],
    *,
    kv_lora_rank: int,
    num_query_heads: int,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> tuple[
    jt.Num[jax.Array, "B T N QK_plus_R"],
    jt.Num[jax.Array, "B T N V"],
]:
  """Performs key/value projection."""
  kv_down = kv_down_projection(
      x,
      wkv_down,
      mesh=mesh,
      axis_mapping=axis_mapping,
  )
  return kv_up_from_kv_down(
      kv_down,
      wk_up,
      wv_up,
      wkv_norm_scale,
      yarn_freqs,
      kv_lora_rank=kv_lora_rank,
      num_query_heads=num_query_heads,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
  )


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def out_projection(
    splash_out: jt.Num[jax.Array, "B T N V"],
    w_out: jt.Num[jax.Array, "N V D"],
) -> jt.Num[jax.Array, "B T D"]:
  """Performs output projection.

  Args:
    splash_out: Splash attention output activations.
    w_out: Output projection weights.

  Returns:
    The output activations.
  """
  attn_sharding = jax.typeof(splash_out).sharding
  assert isinstance(attn_sharding, jax.sharding.NamedSharding)
  head_axis = attn_sharding.spec[2]
  out_spec = jax.sharding.PartitionSpec(attn_sharding.spec[0], head_axis, None)

  @compute_on(
      compute_type="device",
      out_memory_spaces=jax.memory.Space.Device,
  )
  def _out_reduce_scatter(x: jax.Array) -> jax.Array:
    return jax.lax.psum_scatter(
        x,
        axis_name=head_axis,
        scatter_dimension=1,
        tiled=True,
    )

  def _out(splash_out: jax.Array, w_out: jax.Array) -> jax.Array:
    with jax.named_scope("out_projection"):
      out = dot(splash_out, w_out, axes=2)
    return _out_reduce_scatter(out)

  return jax.shard_map(
      _out,
      mesh=attn_sharding.mesh,
      out_specs=out_spec,
      check_vma=True,
  )(splash_out, w_out)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def rotate_half(
    x: jt.Num[jax.Array, "... D"],
) -> jt.Num[jax.Array, "... D"]:
  """Rotates half the hidden dimensions of the input.

  Args:
    x: Input array with shape [..., embedding_dim].

  Returns:
    The array with half dimensions rotated, i.e.
    [x1, x2] -> [-x2, x1].
  """
  x1, x2 = jnp.split(x, 2, axis=-1)
  return jnp.concatenate([-x2, x1], axis=-1)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def yarn(
    x: jt.Num[jax.Array, "B T N D"],
    freqs: tuple[
        jt.Num[jax.Array, "B T 1 D"],
        jt.Num[jax.Array, "B T 1 D"],
    ],
) -> jt.Num[jax.Array, "B T N D"]:
  """Performs YaRN rotary embedding."""
  cos, sin = freqs
  return x * cos + rotate_half(x) * sin


def dot(x, y, axes=1, out_sharding=None):
  """Helper function for jnp.tensordot with default axes=1."""
  return jnp.tensordot(x, y, axes=axes, out_sharding=out_sharding)


@jt.jaxtyped(typechecker=typeguard.typechecked)
def init_splash_kernel(
    sa_block_q: int,
    sa_block_kv: int,
    sa_block_kv_compute: int,
    sa_block_q_dkv: int,
    sa_block_kv_dkv: int,
    sa_block_kv_dkv_compute: int,
    sa_q_layout: Any,
    sa_k_layout: Any,
    sa_v_layout: Any,
    max_target_length: int,
    num_query_heads: int,
    kernel_out_spec: jax.sharding.PartitionSpec,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    qk_diag_skip: bool = True,
    qk_diag_grid: int = 8,
    sv_diag_skip: bool = True,
) -> Any:
  """Initializes the Splash kernel with the given block sizes."""
  del num_query_heads
  physical_kernel_out_spec = ops.physical_pspec(kernel_out_spec, axis_mapping)
  kernel_out_sharding = jax.sharding.NamedSharding(mesh, physical_kernel_out_spec)
  seq_spec_dim = physical_kernel_out_spec[1]

  def _get_num_shards(spec_dim: Any, mesh: jax.sharding.AbstractMesh | jax.sharding.Mesh) -> int:
    if spec_dim is None:
      return 1
    if isinstance(spec_dim, str):
      return mesh.shape[spec_dim]
    return math.prod(mesh.shape[name] for name in spec_dim)

  q_seq_shards = _get_num_shards(seq_spec_dim, mesh)
  if q_seq_shards > 1:
    raise NotImplementedError(
        "Sequence sharding is not supported for" " tokamax_splash_attention_kernel. Only head sharding is supported."
    )

  q_seq_len_per_shard = max_target_length // q_seq_shards

  block_q = min(sa_block_q, q_seq_len_per_shard)
  block_kv = min(sa_block_kv, q_seq_len_per_shard)
  block_kv_compute = min(sa_block_kv_compute, q_seq_len_per_shard)
  effective_qk_diag_skip = qk_diag_skip and (block_q == block_kv == block_kv_compute)
  effective_sv_diag_skip = sv_diag_skip and (block_q == block_kv == block_kv_compute)
  sa_config = tokamax_splash_attention_kernel.SplashConfig(
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      block_q_dkv=min(sa_block_q_dkv, q_seq_len_per_shard),
      block_kv_dkv=min(sa_block_kv_dkv, q_seq_len_per_shard),
      block_kv_dkv_compute=min(sa_block_kv_dkv_compute, q_seq_len_per_shard),
      block_q_dq=None,
      block_kv_dq=None,
      use_fused_bwd_kernel=True,
      q_layout=tokamax_splash_attention_kernel.QKVLayout[sa_q_layout] if isinstance(sa_q_layout, str) else sa_q_layout,
      k_layout=tokamax_splash_attention_kernel.QKVLayout[sa_k_layout] if isinstance(sa_k_layout, str) else sa_k_layout,
      v_layout=tokamax_splash_attention_kernel.QKVLayout[sa_v_layout] if isinstance(sa_v_layout, str) else sa_v_layout,
      qk_diag_skip=effective_qk_diag_skip,
      qk_diag_grid=qk_diag_grid,
      sv_diag_skip=effective_sv_diag_skip,
  )
  mask = tokamax_splash_mask.CausalMask(shape=(max_target_length, max_target_length))
  splash_kernel = tokamax_splash_attention_kernel.make_splash_mha(
      mask=mask,
      q_seq_shards=q_seq_shards,
      config=sa_config,
  )
  kernel_pspec = splash_kernel.manual_sharding_spec(kernel_out_sharding)

  def _reshard_leaf(arr: Any, spec: Any) -> Any:
    if arr is None:
      return None
    sharding = jax.sharding.NamedSharding(mesh, ops.physical_pspec(spec, axis_mapping))
    return jax.reshard(arr, sharding)

  splash_kernel = jax.tree.map(
      _reshard_leaf,
      splash_kernel,
      kernel_pspec,
      is_leaf=lambda x: x is None,
  )
  return splash_kernel


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def splash_attention(
    q: jt.Num[jax.Array, "B T N QK_plus_R"],
    k: jt.Num[jax.Array, "B T N QK_plus_R"],
    v: jt.Num[jax.Array, "B T N V"],
    splash_kernel: Any,
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
) -> tuple[
    jt.Num[jax.Array, "B T N V"],
    jt.Num[jax.Array, "B N T"],
]:
  """Performs splash attention returning output and context."""
  query = jnp.transpose(q, axes=(0, 2, 1, 3))
  key = jnp.transpose(k, axes=(0, 2, 1, 3))
  value = jnp.transpose(v, axes=(0, 2, 1, 3))

  query_spec = jax.typeof(query).sharding.spec
  kv_spec = jax.typeof(value).sharding.spec
  kv_spec = jax.sharding.PartitionSpec(kv_spec[0], kv_spec[1], None, kv_spec[3])
  mesh = jax.typeof(value).sharding.mesh
  # All-gather K and V along the sequence length dimension.
  key = jax.reshard(key, jax.sharding.NamedSharding(mesh, kv_spec))
  value = jax.reshard(value, jax.sharding.NamedSharding(mesh, kv_spec))

  kernel_pspec = jax.tree.map(
      lambda arr: None if arr is None else jax.typeof(arr).sharding.spec,
      splash_kernel,
      is_leaf=lambda x: x is None,
  )

  query_spec = jax.typeof(query).sharding.spec
  context_spec = jax.sharding.PartitionSpec(*query_spec[:3])

  if segment_ids is not None:
    segment_spec = jax.sharding.PartitionSpec(query_spec[0], None)
    segment_ids_tuple = tokamax_splash_attention_kernel.SegmentIds(q=segment_ids, kv=segment_ids)
    segment_ids_in_spec = tokamax_splash_attention_kernel.SegmentIds(
        q=segment_spec,  # pyrefly: ignore[bad-argument-type]
        kv=segment_spec,  # pyrefly: ignore[bad-argument-type]
    )
  else:
    segment_ids_tuple = None
    segment_ids_in_spec = None

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(
          query_spec,
          kv_spec,
          kv_spec,
          kernel_pspec,
          segment_ids_in_spec,
      ),
      out_specs=(
          query_spec,
          context_spec,
      ),
      check_vma=False,
  )
  def wrap_splash_attention(query, key, value, splash_kernel, segment_ids):
    attention_output, context = jax.vmap(
        splash_kernel.manual_fwd,
        in_axes=(0, 0, 0, 0 if segment_ids is not None else None),
        out_axes=(0, 0),
    )(query, key, value, segment_ids)
    return attention_output, context

  attention_output, context = wrap_splash_attention(
      query,
      key,
      value,
      splash_kernel,
      segment_ids_tuple,
  )
  splash_out = jnp.transpose(attention_output, axes=(0, 2, 1, 3))
  return splash_out, context


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def splash_attention_bwd(
    grad_splash_out: jt.Num[jax.Array, "B T N V"],
    splash_out: jt.Num[jax.Array, "B T N V"],
    context: jt.Num[jax.Array, "B N T"],
    q: jt.Num[jax.Array, "B T N QK_plus_R"],
    k: jt.Num[jax.Array, "B T N QK_plus_R"],
    v: jt.Num[jax.Array, "B T N V"],
    splash_kernel: Any,
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
) -> tuple[
    jt.Num[jax.Array, "B T N QK_plus_R"],
    jt.Num[jax.Array, "B T N QK_plus_R"],
    jt.Num[jax.Array, "B T N V"],
]:
  """Performs backward splash attention."""
  do = jnp.transpose(grad_splash_out, axes=(0, 2, 1, 3))
  o = jnp.transpose(splash_out, axes=(0, 2, 1, 3))
  query = jnp.transpose(q, axes=(0, 2, 1, 3))
  key = jnp.transpose(k, axes=(0, 2, 1, 3))
  value = jnp.transpose(v, axes=(0, 2, 1, 3))

  kv_spec = jax.typeof(value).sharding.spec
  kv_spec = jax.sharding.PartitionSpec(kv_spec[0], kv_spec[1], None, kv_spec[3])
  mesh = jax.typeof(value).sharding.mesh

  kernel_pspec = jax.tree.map(
      lambda arr: None if arr is None else jax.typeof(arr).sharding.spec,
      splash_kernel,
      is_leaf=lambda x: x is None,
  )

  query_spec = jax.typeof(query).sharding.spec
  context_spec = jax.sharding.PartitionSpec(*query_spec[:3])

  if segment_ids is not None:
    segment_spec = jax.sharding.PartitionSpec(query_spec[0], None)
    segment_ids_tuple = tokamax_splash_attention_kernel.SegmentIds(q=segment_ids, kv=segment_ids)
    segment_ids_in_spec = tokamax_splash_attention_kernel.SegmentIds(
        q=segment_spec,  # pyrefly: ignore[bad-argument-type]
        kv=segment_spec,  # pyrefly: ignore[bad-argument-type]
    )
  else:
    segment_ids_tuple = None
    segment_ids_in_spec = None

  @functools.partial(
      jax.shard_map,
      mesh=mesh,
      in_specs=(
          query_spec,
          query_spec,
          context_spec,
          query_spec,
          kv_spec,
          kv_spec,
          kernel_pspec,
          segment_ids_in_spec,
      ),
      out_specs=(
          query_spec,
          kv_spec,
          kv_spec,
      ),
      check_vma=False,
  )
  def wrap_splash_attention_bwd(do, o, context, query, key, value, splash_kernel, segment_ids):
    dq, dk, dv = jax.vmap(
        splash_kernel.manual_bwd,
        in_axes=(0, 0, 0, 0, 0, 0, 0 if segment_ids is not None else None),
        out_axes=(0, 0, 0),
    )(query, key, value, o, context, do, segment_ids)
    return dq, dk, dv

  dq, dk, dv = wrap_splash_attention_bwd(
      do,
      o,
      context,
      query,
      key,
      value,
      splash_kernel,
      segment_ids_tuple,
  )

  dq = jnp.transpose(dq, axes=(0, 2, 1, 3))
  dk = jnp.transpose(dk, axes=(0, 2, 1, 3))
  dv = jnp.transpose(dv, axes=(0, 2, 1, 3))
  return dq, dk, dv


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_mla_fwd(
    x: jt.Num[jax.Array, "B T D"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    splash_kernel: Any,
    w: dsv3_types.DSv3MLAWeightsPytree,
    *,
    kv_lora_rank: int,
    qk_head_dim: int,
    rope_head_dim: int,
    num_query_heads: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
) -> tuple[
    jt.Num[jax.Array, "B T D"],
    tuple[
        jt.Num[jax.Array, "B T Cq"],
        jt.Num[jax.Array, "B T Ckv_plus_R"],
        jt.Num[jax.Array, "B N T"],
        jt.Num[jax.Array, "B T N V"],
    ],
]:
  """Forward pass for MLA returning output activations and residuals."""
  q_down = q_down_projection(
      x,
      w.q_down,
      mesh=mesh,
      axis_mapping=axis_mapping,
  )
  q = q_up_from_q_down(
      q_down,
      w.q_up,
      w.q_norm_scale,
      yarn_freqs,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
  )
  kv_down = kv_down_projection(
      x,
      w.kv_down,
      mesh=mesh,
      axis_mapping=axis_mapping,
  )
  k, v = kv_up_from_kv_down(
      kv_down,
      w.k_up,
      w.v_up,
      w.kv_norm_scale,
      yarn_freqs,
      kv_lora_rank=kv_lora_rank,
      num_query_heads=num_query_heads,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
  )
  splash_out, context = splash_attention(
      q,
      k,
      v,
      splash_kernel,
      segment_ids=segment_ids,
  )
  out = out_projection(
      splash_out,
      w.out,
  )
  residuals = (q_down, kv_down, context, splash_out)
  return out, residuals


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_mla_bwd(
    grad_out: jt.Num[jax.Array, "B T D"],
    norm_x: jt.Num[jax.Array, "B T D"],
    q_down: jt.Num[jax.Array, "B T Cq"],
    kv_down: jt.Num[jax.Array, "B T Ckv_plus_R"],
    context: jt.Num[jax.Array, "B N T"],
    splash_out: jt.Num[jax.Array, "B T N V"],
    w: dsv3_types.DSv3MLAWeightsPytree,
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    splash_kernel: Any,
    *,
    kv_lora_rank: int,
    qk_head_dim: int,
    rope_head_dim: int,
    num_query_heads: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
) -> tuple[
    jt.Num[jax.Array, "B T D"],
    dsv3_types.DSv3MLAWeightsPytree,
    tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
]:
  """Backward pass for MLA using saved residuals and rematerialization."""

  # 1. Backprop through out_projection
  def _out_proj_fwd(splash_out, w_out):
    return out_projection(
        splash_out,
        w_out,
    )

  _, out_proj_vjp = jax.vjp(_out_proj_fwd, splash_out, w.out)
  grad_splash_out, grad_w_out = out_proj_vjp(grad_out)

  # 2. Remat q from q_down
  def _q_up_fwd(q_down, wq_up, wq_norm_scale, yarn_freqs):
    return q_up_from_q_down(
        q_down,
        wq_up,
        wq_norm_scale,
        yarn_freqs,
        qk_head_dim=qk_head_dim,
        rope_head_dim=rope_head_dim,
        max_position_embeddings=max_position_embeddings,
        original_max_position_embeddings=original_max_position_embeddings,
        rope_factor=rope_factor,
        mscale=mscale,
        norm_fn=norm_fn,
        rope_fn=rope_fn,
    )

  q, vjp_q = jax.vjp(_q_up_fwd, q_down, w.q_up, w.q_norm_scale, yarn_freqs)

  # 3. Remat k, v from kv_down
  def _kv_up_fwd(kv_down, wk_up, wv_up, wkv_norm_scale, yarn_freqs):
    return kv_up_from_kv_down(
        kv_down,
        wk_up,
        wv_up,
        wkv_norm_scale,
        yarn_freqs,
        kv_lora_rank=kv_lora_rank,
        num_query_heads=num_query_heads,
        norm_fn=norm_fn,
        rope_fn=rope_fn,
    )

  (k, v), vjp_kv = jax.vjp(_kv_up_fwd, kv_down, w.k_up, w.v_up, w.kv_norm_scale, yarn_freqs)

  # 4. Splash attention backward
  dq, dk, dv = splash_attention_bwd(
      grad_splash_out,
      splash_out,
      context,
      q,
      k,
      v,
      splash_kernel,
      segment_ids=segment_ids,
  )

  # 5. Backprop through Q projection
  grad_q_down, grad_w_q_up, grad_w_q_norm_scale, grad_yarn_freqs_q = vjp_q(dq)

  def _q_down_fwd(norm_x, wq_down):
    return q_down_projection(
        norm_x,
        wq_down,
        mesh=mesh,
        axis_mapping=axis_mapping,
    )

  _, vjp_q_down = jax.vjp(_q_down_fwd, norm_x, w.q_down)
  grad_norm_x_q, grad_w_q_down = vjp_q_down(grad_q_down)

  # 6. Backprop through KV projection
  (
      grad_kv_down,
      grad_w_k_up,
      grad_w_v_up,
      grad_w_kv_norm_scale,
      grad_yarn_freqs_kv,
  ) = vjp_kv((dk, dv))

  def _kv_down_fwd(norm_x, wkv_down):
    return kv_down_projection(
        norm_x,
        wkv_down,
        mesh=mesh,
        axis_mapping=axis_mapping,
    )

  _, vjp_kv_down = jax.vjp(_kv_down_fwd, norm_x, w.kv_down)
  grad_norm_x_kv, grad_w_kv_down = vjp_kv_down(grad_kv_down)

  # 7. Accumulate gradients
  grad_norm_x = grad_norm_x_q + grad_norm_x_kv
  grad_yarn_freqs = jax.tree.map(lambda g1, g2: g1 + g2, grad_yarn_freqs_q, grad_yarn_freqs_kv)
  grad_w = dsv3_types.DSv3MLAWeightsPytree(
      q_down=grad_w_q_down,
      q_up=grad_w_q_up,
      q_norm_scale=grad_w_q_norm_scale,
      kv_down=grad_w_kv_down,
      k_up=grad_w_k_up,
      v_up=grad_w_v_up,
      kv_norm_scale=grad_w_kv_norm_scale,
      out=grad_w_out,
  )
  return grad_norm_x, grad_w, grad_yarn_freqs


@functools.partial(jax.custom_vjp, nondiff_argnums=tuple(range(5, 17)))
def _dsv3_mla_vjp(
    x: jt.Num[jax.Array, "B T D"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    splash_kernel: Any,
    w: dsv3_types.DSv3MLAWeightsPytree,
    segment_ids: jt.Num[jax.Array, "B T"] | None,
    kv_lora_rank: int,
    qk_head_dim: int,
    rope_head_dim: int,
    num_query_heads: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
) -> jt.Num[jax.Array, "B T D"]:
  """Custom VJP wrapper for MLA."""
  return _dsv3_mla_custom_fwd(
      x,
      yarn_freqs,
      splash_kernel,
      w,
      segment_ids,
      kv_lora_rank,
      qk_head_dim,
      rope_head_dim,
      num_query_heads,
      max_position_embeddings,
      original_max_position_embeddings,
      rope_factor,
      mscale,
      norm_fn,
      rope_fn,
      mesh,
      axis_mapping,
  )[0]


def _dsv3_mla_custom_fwd(
    x,
    yarn_freqs,
    splash_kernel,
    w,
    segment_ids,
    kv_lora_rank,
    qk_head_dim,
    rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    norm_fn,
    rope_fn,
    mesh,
    axis_mapping,
):
  out, residuals = dsv3_mla_fwd(
      x,
      yarn_freqs,
      splash_kernel,
      w,
      kv_lora_rank=kv_lora_rank,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      num_query_heads=num_query_heads,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      mesh=mesh,
      axis_mapping=axis_mapping,
      segment_ids=segment_ids,
  )
  q_down, kv_down, context, splash_out = residuals
  res = (
      x,
      w,
      yarn_freqs,
      splash_kernel,
      segment_ids,
      q_down,
      kv_down,
      context,
      splash_out,
  )
  return out, res


def _dsv3_mla_custom_bwd(
    kv_lora_rank,
    qk_head_dim,
    rope_head_dim,
    num_query_heads,
    max_position_embeddings,
    original_max_position_embeddings,
    rope_factor,
    mscale,
    norm_fn,
    rope_fn,
    mesh,
    axis_mapping,
    res,
    grad_out,
):
  (
      x,
      w,
      yarn_freqs,
      splash_kernel,
      segment_ids,
      q_down,
      kv_down,
      context,
      splash_out,
  ) = res
  grad_x, grad_w, grad_yarn_freqs = dsv3_mla_bwd(
      grad_out,
      x,
      q_down,
      kv_down,
      context,
      splash_out,
      w,
      yarn_freqs,
      splash_kernel,
      kv_lora_rank=kv_lora_rank,
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      num_query_heads=num_query_heads,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
      norm_fn=norm_fn,
      rope_fn=rope_fn,
      mesh=mesh,
      axis_mapping=axis_mapping,
      segment_ids=segment_ids,
  )
  return grad_x, grad_yarn_freqs, None, grad_w, None


_dsv3_mla_vjp.defvjp(_dsv3_mla_custom_fwd, _dsv3_mla_custom_bwd)


@jax.named_call
@jt.jaxtyped(typechecker=typeguard.typechecked)
def dsv3_mla(
    x: jt.Num[jax.Array, "B T D"],
    yarn_freqs: tuple[
        jt.Num[jax.Array, "B T 1 R"],
        jt.Num[jax.Array, "B T 1 R"],
    ],
    splash_kernel: Any,
    w: dsv3_types.DSv3MLAWeightsPytree,
    *,
    kv_lora_rank: int,
    qk_head_dim: int,
    rope_head_dim: int,
    num_query_heads: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: NormFn,
    rope_fn: RopeFn,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    segment_ids: jt.Num[jax.Array, "B T"] | None = None,
) -> jt.Num[jax.Array, "B T D"]:
  """Performs MLA for DSv3."""
  return _dsv3_mla_vjp(
      x,
      yarn_freqs,
      splash_kernel,
      w,
      segment_ids,
      kv_lora_rank,
      qk_head_dim,
      rope_head_dim,
      num_query_heads,
      max_position_embeddings,
      original_max_position_embeddings,
      rope_factor,
      mscale,
      norm_fn,
      rope_fn,
      mesh,
      axis_mapping,
  )


@jt.jaxtyped(typechecker=typeguard.typechecked)
def get_yarn_freqs(
    positions: jt.Num[jax.Array, "B T"],
    rope_head_dim: int,
    rope_theta: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    beta_fast: int,
    beta_slow: int,
    rope_factor: int,
    out_pspec: jax.sharding.PartitionSpec,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Mapping[str, str | tuple[str, ...]],
    dtype: jt.DTypeLike = jnp.bfloat16,
) -> tuple[jt.Num[jax.Array, "B T 1 R"], jt.Num[jax.Array, "B T 1 R"]]:
  """Initializes YaRN cos and sin frequencies.

  These frequencies are used to rotate pairs of dimensions in the query and key
  vectors. Base frequencies are first computed for each pair of dimensions and
  then piecewise scaled and smoothed across the positions. Up to
  max_position_embeddings frequencies are precomputed and then looked up using
  the input positions, with cos and sin precomputed in the specified dtype.

  Args:
    positions: Position indices corresponding to the positions of the input
      tokens.
    rope_head_dim: RoPE dimension per head.
    rope_theta: RoPE theta, used to determine base frequencies.
    max_position_embeddings: The maximum position indices in the input.
    original_max_position_embeddings: Original maximum position embeddings, used
      for scaling.
    beta_fast: Beta fast, used for scaling.
    beta_slow: Beta slow, used for scaling.
    rope_factor: RoPE factor, used for scaling.
    out_pspec: Logical PartitionSpec for output sharding of frequencies.
    mesh: Physical mesh.
    axis_mapping: Mapping from logical to physical axes.
    dtype: Data type for cos and sin frequencies. Defaults to bfloat16.

  Returns:
    A tuple of (cos, sin) YaRN frequencies each with shape [B, T, 1, R] and the
    specified dtype.
  """
  out_sharding = jax.sharding.NamedSharding(mesh, ops.physical_pspec(out_pspec, axis_mapping))
  half_dim = rope_head_dim // 2
  # Compute base frequencies for each (even-indexed) dimension.
  # (Note: We use jnp.arange with float32 for precision.)
  freqs = 1.0 / (rope_theta ** (2.0 * jnp.arange(0, half_dim, dtype=jnp.float32) / rope_head_dim))

  low = (
      rope_head_dim * math.log(original_max_position_embeddings / (beta_fast * 2 * math.pi)) / (2 * math.log(rope_theta))
  )
  high = (
      rope_head_dim * math.log(original_max_position_embeddings / (beta_slow * 2 * math.pi)) / (2 * math.log(rope_theta))
  )
  low = max(math.floor(low), 0)
  high = min(math.ceil(high), rope_head_dim - 1)
  diff = high - low if high > low else 0.001
  linear_func = (jnp.arange(half_dim, dtype=jnp.float32) - low) / diff
  smooth = 1 - jnp.clip(linear_func, 0, 1)
  # The corrected frequency is a weighted mix of the scaled and base values.
  freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth

  # Precompute frequencies for all positions by taking the outer product.
  t = jnp.arange(max_position_embeddings, dtype=jnp.float32)  # shape [max_position_embeddings]
  # [max_position_embeddings, half_dim] tensor with rows as time steps.
  freqs = jnp.outer(t, freqs)
  # Lookup the precomputed frequencies using the position indices.
  # freqs has shape [max_position_embeddings, half_dim].
  # After indexing, shape becomes [B, T, half_dim]; we then add the head axis.
  freqs = freqs.at[positions].get(out_sharding=out_sharding)
  freqs = freqs[:, :, jnp.newaxis, :]  # shape: [B, T, 1, half_dim]
  freqs = jnp.concatenate([freqs, freqs], axis=-1)  # shape: [B, T, 1, rope_head_dim]
  cos = jnp.cos(freqs).astype(dtype)
  sin = jnp.sin(freqs).astype(dtype)
  return cos, sin

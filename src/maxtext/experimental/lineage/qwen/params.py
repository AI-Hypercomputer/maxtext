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

"""Pure-JAX PyTree parameter structs, shardings, and MaxText Qwen3_5 weight converters."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from maxtext.experimental.lineage.qwen.config import QwenConfig


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MoeParams:
  """Parameters for one Qwen 3.5 / Qwen MAX Sparse MoE + Gated Shared Expert block."""

  router_gate: jax.Array  # [..., D, E] bf16/fp32
  wi: jax.Array  # [..., E, D, 2*M] bf16/fp32 (prefused [gate | up])
  wo: jax.Array  # [..., E, M, D] bf16/fp32
  shared_wi_0: jax.Array  # [..., D, M_s] bf16/fp32
  shared_wi_1: jax.Array  # [..., D, M_s] bf16/fp32
  shared_wo: jax.Array  # [..., M_s, D] bf16/fp32
  shared_gate: jax.Array  # [..., D, 1] bf16/fp32


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GdnSublayerParams:
  """Parameters for one Gated DeltaNet + MoE sub-layer."""

  pre_attn_norm: jax.Array  # [..., D] bf16/fp32 (zero-centered: 1 + scale)
  post_attn_norm: jax.Array  # [..., D] bf16/fp32 (zero-centered: 1 + scale)
  in_proj_qkvz: (
      jax.Array
  )  # [..., D, qkvz_dim] bf16/fp32 (kernel-order [q|k|v|z])
  in_proj_ba: jax.Array  # [..., D, 2*H_v] bf16/fp32 (kernel-order [b|a])
  conv1d: jax.Array  # [..., K_conv, 1, qkv_dim] bf16/fp32
  A_log: jax.Array  # [..., H_v] fp32  # pylint: disable=invalid-name
  dt_bias: jax.Array  # [..., H_v] fp32
  gdn_norm: jax.Array  # [..., D_v] bf16/fp32 (ones-init gated RMSNorm)
  out_proj: jax.Array  # [..., V_dim, D] bf16/fp32
  moe: MoeParams


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class AttnSublayerParams:
  """Parameters for one Global Full Attention + MoE sub-layer."""

  pre_attn_norm: jax.Array  # [..., D] bf16/fp32 (zero-centered: 1 + scale)
  post_attn_norm: jax.Array  # [..., D] bf16/fp32 (zero-centered: 1 + scale)
  q_proj: jax.Array  # [..., D, H_q, 2*D_h] bf16/fp32 (query + output gate)
  k_proj: jax.Array  # [..., D, H_kv, D_h] bf16/fp32
  v_proj: jax.Array  # [..., D, H_kv, D_h] bf16/fp32
  q_norm: jax.Array  # [..., D_h] bf16/fp32 (zero-centered: 1 + scale)
  k_norm: jax.Array  # [..., D_h] bf16/fp32 (zero-centered: 1 + scale)
  out_proj: jax.Array  # [..., H_q*D_h, D] bf16/fp32
  moe: MoeParams


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SuperblockParams:
  """Parameters for `[GDN_0, GDN_1, GDN_2, FullAttn_3]` stacked over superblocks."""

  gdn_0: GdnSublayerParams
  gdn_1: GdnSublayerParams
  gdn_2: GdnSublayerParams
  attn_3: AttnSublayerParams


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ModelParams:
  """Complete Qwen 3.5 / Qwen MAX model parameter PyTree."""

  token_embedder: jax.Array  # [V, D] bf16/fp32
  blocks: SuperblockParams  # leading axis = [num_superblocks, ...]
  decoder_norm: jax.Array  # [D] bf16/fp32 (zero-centered: 1 + scale)
  lm_head: jax.Array  # [D, V] bf16/fp32


def _dense_axis(cfg: QwenConfig) -> str | tuple[str, ...]:
  return (
      cfg.dense_stage2_axis_names
      if cfg.dense_sharding_mode == "fsdp_only"
      else cfg.all_sharded_axes
  )


def _dense_proj_axis(
    cfg: QwenConfig, unsharded_cols: int
) -> str | tuple[str, ...]:
  return (
      cfg.all_sharded_axes
      if cfg.dense_proj_fsdp_expert(unsharded_cols)
      else cfg.dense_stage2_axis_names
  )


def get_param_pspecs(cfg: QwenConfig) -> ModelParams:
  """Returns PartitionSpecs for every parameter leaf (`blocks` stacked on dim 0)."""
  dense_ax = _dense_axis(cfg)
  if cfg.num_experts % cfg.num_devices == 0 or cfg.moe_k_shards > 1:
    # (*ep_axis_tuple, 'fsdp') places local experts (or contiguous K-splits
    # `[E * k_shards, K // k_shards, N]` when `num_devices > num_experts`, e.g.
    # 512c = 1024 TCs with E=512 -> `k_shards=2`) contiguously on major axis 0
    # across FSDP shards, keeping SC1 AllGather/ReduceScatter on major `axis=0`
    # with zero transpose copies and zero non-major `S(9)` scratchpads.
    moe_wi_pspec = P(None, (*cfg.ep_axis_tuple, "fsdp"), None, None)
    moe_wo_pspec = P(None, (*cfg.ep_axis_tuple, "fsdp"), None, None)
  elif cfg.num_experts % cfg.ep_size == 0:
    # Fallback when `num_devices` does not evenly divide `num_experts` either way.
    moe_wi_pspec = P(None, cfg.ep_axis_tuple, "fsdp", None)
    moe_wo_pspec = P(None, cfg.ep_axis_tuple, "fsdp", None)
  else:
    raise ValueError(
        f"num_experts={cfg.num_experts} must be divisible by"
        f" ep_size={cfg.ep_size}."
    )

  router_ax = _dense_proj_axis(cfg, cfg.num_experts)
  shared_ax = _dense_proj_axis(cfg, cfg.shared_mlp_dim)
  shared_gate_pspec = (
      P(None, None, None)
      if cfg.replicate_shared_gate
      else P(None, dense_ax, None)
  )
  gdn_ba_ax = _dense_proj_axis(cfg, 2 * cfg.gdn_num_v_heads)
  attn_kv_ax = _dense_proj_axis(cfg, cfg.num_kv_heads * cfg.head_dim)

  def _moe_pspec() -> MoeParams:
    return MoeParams(
        router_gate=P(None, router_ax, None),
        wi=moe_wi_pspec,
        wo=moe_wo_pspec,
        shared_wi_0=P(None, shared_ax, None),
        shared_wi_1=P(None, shared_ax, None),
        shared_wo=P(None, None, shared_ax),  # shard D (axis 2 = 4096/8192)!
        shared_gate=shared_gate_pspec,
    )

  def _gdn_pspec() -> GdnSublayerParams:
    return GdnSublayerParams(
        pre_attn_norm=P(None, None),
        post_attn_norm=P(None, None),
        in_proj_qkvz=P(None, dense_ax, None),
        in_proj_ba=P(None, gdn_ba_ax, None),
        conv1d=P(None, None, None, None),
        A_log=P(None, None),
        dt_bias=P(None, None),
        gdn_norm=P(None, None),
        out_proj=P(None, dense_ax, None),
        moe=_moe_pspec(),
    )

  attn_pspec = AttnSublayerParams(
      pre_attn_norm=P(None, None),
      post_attn_norm=P(None, None),
      q_proj=P(None, dense_ax, None, None),
      k_proj=P(None, attn_kv_ax, None, None),
      v_proj=P(None, attn_kv_ax, None, None),
      q_norm=P(None, None),
      k_norm=P(None, None),
      out_proj=P(None, dense_ax, None),
      moe=_moe_pspec(),
  )

  return ModelParams(
      token_embedder=P(None, cfg.all_sharded_axes),
      blocks=SuperblockParams(
          gdn_0=_gdn_pspec(),
          gdn_1=_gdn_pspec(),
          gdn_2=_gdn_pspec(),
          attn_3=attn_pspec,
      ),
      decoder_norm=P(None),
      lm_head=P(cfg.all_sharded_axes, None),
  )


def get_abstract_params(
    cfg: QwenConfig,
    mesh: Mesh,
    *,
    param_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> ModelParams:
  """Builds abstract `jax.ShapeDtypeStruct` parameter tree with NamedShardings."""
  pspecs = get_param_pspecs(cfg)
  sb = cfg.num_superblocks
  d = cfg.emb_dim
  e = cfg.num_experts
  m = cfg.moe_mlp_dim
  ms = cfg.shared_mlp_dim
  p_dt = jnp.dtype(param_dtype)

  def _s(shape: tuple[int, ...], dtype, pspec: P) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(
        shape, dtype, sharding=NamedSharding(mesh, pspec)
    )

  def _moe_shapes(ps: MoeParams) -> MoeParams:
    k_shards = max(1, cfg.moe_k_shards)
    if d % k_shards != 0 or m % k_shards != 0:
      raise ValueError(
          f"moe_k_shards={k_shards} must divide emb_dim={d} and"
          f" moe_mlp_dim={m}."
      )
    return MoeParams(
        router_gate=_s((sb, d, e), p_dt, ps.router_gate),
        wi=_s((sb, e * k_shards, d // k_shards, 2 * m), p_dt, ps.wi),
        wo=_s((sb, e * k_shards, m // k_shards, d), p_dt, ps.wo),
        shared_wi_0=_s((sb, d, ms), p_dt, ps.shared_wi_0),
        shared_wi_1=_s((sb, d, ms), p_dt, ps.shared_wi_1),
        shared_wo=_s((sb, ms, d), p_dt, ps.shared_wo),
        shared_gate=_s((sb, d, 1), p_dt, ps.shared_gate),
    )

  def _gdn_shapes(ps: GdnSublayerParams) -> GdnSublayerParams:
    return GdnSublayerParams(
        pre_attn_norm=_s((sb, d), p_dt, ps.pre_attn_norm),
        post_attn_norm=_s((sb, d), p_dt, ps.post_attn_norm),
        in_proj_qkvz=_s((sb, d, cfg.gdn_qkvz_dim), p_dt, ps.in_proj_qkvz),
        in_proj_ba=_s((sb, d, 2 * cfg.gdn_num_v_heads), p_dt, ps.in_proj_ba),
        conv1d=_s(
            (sb, cfg.gdn_conv_kernel_size, 1, cfg.gdn_qkv_dim), p_dt, ps.conv1d
        ),
        A_log=_s((sb, cfg.gdn_num_v_heads), jnp.float32, ps.A_log),
        dt_bias=_s((sb, cfg.gdn_num_v_heads), jnp.float32, ps.dt_bias),
        gdn_norm=_s((sb, cfg.gdn_head_v_dim), p_dt, ps.gdn_norm),
        out_proj=_s((sb, cfg.gdn_value_dim, d), p_dt, ps.out_proj),
        moe=_moe_shapes(ps.moe),
    )

  attn_ps = pspecs.blocks.attn_3
  attn_shapes = AttnSublayerParams(
      pre_attn_norm=_s((sb, d), p_dt, attn_ps.pre_attn_norm),
      post_attn_norm=_s((sb, d), p_dt, attn_ps.post_attn_norm),
      q_proj=_s(
          (sb, d, cfg.num_query_heads, 2 * cfg.head_dim), p_dt, attn_ps.q_proj
      ),
      k_proj=_s((sb, d, cfg.num_kv_heads, cfg.head_dim), p_dt, attn_ps.k_proj),
      v_proj=_s((sb, d, cfg.num_kv_heads, cfg.head_dim), p_dt, attn_ps.v_proj),
      q_norm=_s((sb, cfg.head_dim), p_dt, attn_ps.q_norm),
      k_norm=_s((sb, cfg.head_dim), p_dt, attn_ps.k_norm),
      out_proj=_s((sb, cfg.attn_q_out_dim, d), p_dt, attn_ps.out_proj),
      moe=_moe_shapes(attn_ps.moe),
  )

  return ModelParams(
      token_embedder=_s((cfg.vocab_size, d), p_dt, pspecs.token_embedder),
      blocks=SuperblockParams(
          gdn_0=_gdn_shapes(pspecs.blocks.gdn_0),
          gdn_1=_gdn_shapes(pspecs.blocks.gdn_1),
          gdn_2=_gdn_shapes(pspecs.blocks.gdn_2),
          attn_3=attn_shapes,
      ),
      decoder_norm=_s((d,), p_dt, pspecs.decoder_norm),
      lm_head=_s((d, cfg.vocab_size), p_dt, pspecs.lm_head),
  )


def _local_shape_and_axes(
    global_shape: tuple[int, ...], pspec: P, mesh: Mesh
) -> tuple[tuple[int, ...], tuple[str, ...]]:
  """Computes per-device shard shape and ordered sharded mesh axes for `pspec`."""
  local_dims: list[int] = []
  axes_order: list[str] = []
  for idx, dim in enumerate(global_shape):
    entry = pspec[idx] if idx < len(pspec) else None
    if entry is None:
      local_dims.append(dim)
    elif isinstance(entry, str):
      div = int(mesh.shape[entry])
      if dim % div != 0:
        raise ValueError(f"Dim {dim} not divisible by mesh axis {entry}={div}")
      local_dims.append(dim // div)
      axes_order.append(entry)
    else:
      div = math.prod(int(mesh.shape[a]) for a in entry)
      if dim % div != 0:
        raise ValueError(f"Dim {dim} not divisible by mesh axes {entry}={div}")
      local_dims.append(dim // div)
      axes_order.extend(entry)
  return tuple(local_dims), tuple(axes_order)


def _make_replicated_keys(mesh: Mesh, key: jax.Array, count: int) -> jax.Array:
  """Splits `key` into `count` PRNGKeys replicated across `mesh` (multi-host safe)."""
  rep_sharding = NamedSharding(mesh, P())
  if jax.process_count() == 1:
    return jax.device_put(jax.random.split(key, count), rep_sharding)

  is_typed = jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
  raw_key = jax.random.key_data(key) if is_typed else key
  key_bits = tuple(int(x) for x in raw_key.reshape(-1))
  raw_shape = raw_key.shape

  @functools.partial(jax.jit, out_shardings=rep_sharding)
  def _build_keys():
    arr = jnp.asarray(key_bits, dtype=jnp.uint32).reshape(raw_shape)
    base = jax.random.wrap_key_data(arr) if is_typed else arr
    return jax.random.split(base, count)

  return _build_keys()


def init_random_params(
    cfg: QwenConfig,
    mesh: Mesh,
    key: jax.Array,
    *,
    param_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> ModelParams:
  """Initializes sharded parameters directly on device (1-host & 64-host safe)."""
  abstract = get_abstract_params(cfg, mesh, param_dtype=param_dtype)
  leaves, treedef = jax.tree.flatten_with_path(abstract)
  rep_sharding = NamedSharding(mesh, P())
  keys = _make_replicated_keys(mesh, key, len(leaves))
  compiled_builders: dict[tuple[object, ...], Any] = {}
  out_leaves = []

  for idx, (path, spec) in enumerate(leaves):
    path_str = jax.tree_util.keystr(path)
    pspec = spec.sharding.spec
    local_shape, sharded_axes = _local_shape_and_axes(spec.shape, pspec, mesh)
    target_dtype = jnp.dtype(
        jnp.float32 if spec.dtype == jnp.float32 else param_dtype
    )
    if (
        "pre_attn_norm" in path_str
        or "post_attn_norm" in path_str
        or "decoder_norm" in path_str
        or "q_norm" in path_str
        or "k_norm" in path_str
    ):
      kind = "small_norm"
      scale = 0.02
    elif "gdn_norm" in path_str:
      kind = "ones"
      scale = 1.0
    elif "A_log" in path_str:
      kind = "a_log"
      scale = 1.0
    elif "dt_bias" in path_str:
      kind = "zeros"
      scale = 0.0
    else:
      kind = "normal_scaled"
      if path_str.endswith(".wi"):
        fan_in = cfg.emb_dim
      elif path_str.endswith(".wo"):
        fan_in = cfg.moe_mlp_dim
      else:
        fan_in = spec.shape[-2] if len(spec.shape) >= 2 else spec.shape[-1]
      scale = float(fan_in**-0.5)

    cache_key = (
        local_shape,
        pspec,
        sharded_axes,
        kind,
        scale,
        target_dtype.str,
    )
    builder = compiled_builders.get(cache_key)
    if builder is None:

      def _make_builder(lshape, ps, axes, knd, scl, dt):
        def _local_init(subk):
          if axes:
            shard_id = jnp.int32(0)
            for ax in axes:
              shard_id = shard_id * jnp.int32(
                  mesh.shape[ax]
              ) + jax.lax.axis_index(ax)
            subk = jax.random.fold_in(subk, shard_id)
          if knd == "ones":
            arr = jnp.ones(lshape, dtype=jnp.float32)
          elif knd == "zeros":
            arr = jnp.zeros(lshape, dtype=jnp.float32)
          elif knd == "a_log":
            arr = jnp.log(
                jax.random.uniform(
                    subk, lshape, dtype=jnp.float32, minval=1.0, maxval=8.0
                )
            )
          else:
            arr = jax.random.normal(subk, lshape, dtype=jnp.float32) * scl
          return arr.astype(dt)

        smap_fn = jax.shard_map(
            _local_init,
            mesh=mesh,
            in_specs=(P(),),
            out_specs=ps,
            check_vma=False,
        )
        return jax.jit(
            smap_fn,
            in_shardings=(rep_sharding,),
            out_shardings=NamedSharding(mesh, ps),
        )

      builder = _make_builder(
          local_shape, pspec, sharded_axes, kind, scale, target_dtype
      )
      compiled_builders[cache_key] = builder

    out_leaves.append(builder(keys[idx]))

  return jax.tree.unflatten(treedef, out_leaves)


# Alias matching shared runner conventions
init_params = init_random_params


# ---------------------------------------------------------------------------
# Zero-Flax 1:1 MaxText `Qwen3_5*` Weight Converters (for xparity tests)
# ---------------------------------------------------------------------------


def _unwrap_nnx_value(obj: Any) -> jax.Array:
  """Extracts raw `jax.Array` from a duck-typed NNX Param/Variable without importing Flax."""
  cur = obj
  for _ in range(4):
    if hasattr(cur, "get_value") and callable(cur.get_value):
      cur = cur.get_value()
    elif hasattr(cur, "value") and not isinstance(cur, (jax.Array, np.ndarray)):
      cur = cur.value
    else:
      break
  return jnp.asarray(cur)


def _cast_opt(arr: jax.Array, dtype: jax.typing.DTypeLike | None) -> jax.Array:
  return arr if dtype is None else arr.astype(dtype)


def deinterleave_maxtext_gdn_qkvz(
    mt_qkvz_kernel: jax.Array, cfg: QwenConfig
) -> jax.Array:
  """Converts MaxText grouped `in_proj_qkvz` to contiguous `[q|k|v|z]`."""
  *prefix, d_model, total_cols = mt_qkvz_kernel.shape
  h_k = cfg.gdn_num_k_heads
  h_v = cfg.gdn_num_v_heads
  d_k = cfg.gdn_head_k_dim
  d_v = cfg.gdn_head_v_dim
  v_per_k = h_v // h_k
  expected_cols = 2 * cfg.gdn_key_dim + 2 * cfg.gdn_value_dim
  if total_cols != expected_cols:
    raise ValueError(
        f"Expected in_proj_qkvz trailing cols={expected_cols}, got"
        f" {total_cols}."
    )
  grouped = mt_qkvz_kernel.reshape(
      *prefix, d_model, h_k, 2 * d_k + 2 * v_per_k * d_v
  )
  q_w, k_w, v_w, z_w = jnp.split(
      grouped, [d_k, 2 * d_k, 2 * d_k + v_per_k * d_v], axis=-1
  )
  return jnp.concatenate(
      [
          q_w.reshape(*prefix, d_model, cfg.gdn_key_dim),
          k_w.reshape(*prefix, d_model, cfg.gdn_key_dim),
          v_w.reshape(*prefix, d_model, cfg.gdn_value_dim),
          z_w.reshape(*prefix, d_model, cfg.gdn_value_dim),
      ],
      axis=-1,
  )


def interleave_to_maxtext_gdn_qkvz(
    qkvz_kernel: jax.Array, cfg: QwenConfig
) -> jax.Array:
  """Inverse of `deinterleave_maxtext_gdn_qkvz` (maps `[q | k | v | z]` gradients back to MaxText)."""
  *prefix, d_model, _ = qkvz_kernel.shape
  h_k = cfg.gdn_num_k_heads
  h_v = cfg.gdn_num_v_heads
  d_k = cfg.gdn_head_k_dim
  d_v = cfg.gdn_head_v_dim
  v_per_k = h_v // h_k
  k_dim = cfg.gdn_key_dim
  v_dim = cfg.gdn_value_dim
  q_w, k_w, v_w, z_w = jnp.split(
      qkvz_kernel, [k_dim, 2 * k_dim, 2 * k_dim + v_dim], axis=-1
  )
  grouped = jnp.concatenate(
      [
          q_w.reshape(*prefix, d_model, h_k, d_k),
          k_w.reshape(*prefix, d_model, h_k, d_k),
          v_w.reshape(*prefix, d_model, h_k, v_per_k * d_v),
          z_w.reshape(*prefix, d_model, h_k, v_per_k * d_v),
      ],
      axis=-1,
  )
  return grouped.reshape(*prefix, d_model, cfg.gdn_qkvz_dim)


def deinterleave_maxtext_gdn_ba(
    mt_ba_kernel: jax.Array, cfg: QwenConfig
) -> jax.Array:
  """Converts MaxText `H_k`-grouped `in_proj_ba` (`[..., D, 2*H_v]`) to `[b | a]`."""
  *prefix, d_model, total_cols = mt_ba_kernel.shape
  h_k = cfg.gdn_num_k_heads
  h_v = cfg.gdn_num_v_heads
  v_per_k = h_v // h_k
  if total_cols != 2 * h_v:
    raise ValueError(
        f"Expected in_proj_ba trailing cols={2 * h_v}, got {total_cols}."
    )
  grouped = mt_ba_kernel.reshape(*prefix, d_model, h_k, 2 * v_per_k)
  b_w, a_w = jnp.split(grouped, [v_per_k], axis=-1)
  return jnp.concatenate(
      [b_w.reshape(*prefix, d_model, h_v), a_w.reshape(*prefix, d_model, h_v)],
      axis=-1,
  )


def interleave_to_maxtext_gdn_ba(
    ba_kernel: jax.Array, cfg: QwenConfig
) -> jax.Array:
  """Inverse of `deinterleave_maxtext_gdn_ba` (maps `[b | a]` gradients back to MaxText)."""
  *prefix, d_model, _ = ba_kernel.shape
  h_k = cfg.gdn_num_k_heads
  h_v = cfg.gdn_num_v_heads
  v_per_k = h_v // h_k
  b_w, a_w = jnp.split(ba_kernel, [h_v], axis=-1)
  grouped = jnp.concatenate(
      [
          b_w.reshape(*prefix, d_model, h_k, v_per_k),
          a_w.reshape(*prefix, d_model, h_k, v_per_k),
      ],
      axis=-1,
  )
  return grouped.reshape(*prefix, d_model, 2 * h_v)


def convert_maxtext_moe_params(
    mt_moe: Any,
    cfg: QwenConfig,
    *,
    dtype: jax.typing.DTypeLike | None = None,
) -> MoeParams:
  """Converts MaxText `Qwen3_5SparseMoEBlock` weights 1:1 into `MoeParams`."""
  routed = mt_moe.routed_experts
  router_gate = _cast_opt(_unwrap_nnx_value(routed.gate.kernel), dtype)
  e = cfg.num_experts
  d = cfg.emb_dim
  m = cfg.moe_mlp_dim

  if hasattr(routed, "wi") and getattr(routed, "wi") is not None:
    wi = _unwrap_nnx_value(routed.wi)
    if wi.shape[-2:] != (d, 2 * m):
      wi = wi.reshape(*wi.shape[:-2], e, d, 2 * m)
  else:
    wi_0 = _unwrap_nnx_value(routed.wi_0)
    wi_1 = _unwrap_nnx_value(routed.wi_1)
    if wi_0.shape[-2:] != (d, m):
      wi_0 = wi_0.reshape(*wi_0.shape[:-2], e, d, m)
      wi_1 = wi_1.reshape(*wi_1.shape[:-2], e, d, m)
    wi = jnp.concatenate([wi_0, wi_1], axis=-1)

  wo = _unwrap_nnx_value(routed.wo)
  if wo.shape[-2:] != (m, d):
    wo = wo.reshape(*wo.shape[:-2], e, m, d)
  if cfg.moe_k_shards > 1:
    ks = cfg.moe_k_shards
    wi = wi.reshape(*wi.shape[:-3], e * ks, d // ks, 2 * m)
    wo = wo.reshape(*wo.shape[:-3], e * ks, m // ks, d)

  shared = mt_moe.shared_expert
  shared_wi_0 = _unwrap_nnx_value(shared.wi_0.kernel)
  shared_wi_1 = _unwrap_nnx_value(shared.wi_1.kernel)
  shared_wo = _unwrap_nnx_value(shared.wo.kernel)
  shared_gate = _unwrap_nnx_value(mt_moe.shared_expert_gate.kernel)
  if shared_gate.shape[-1] != 1:
    shared_gate = jnp.expand_dims(shared_gate, axis=-1)

  return MoeParams(
      router_gate=router_gate,
      wi=_cast_opt(wi, dtype),
      wo=_cast_opt(wo, dtype),
      shared_wi_0=_cast_opt(shared_wi_0, dtype),
      shared_wi_1=_cast_opt(shared_wi_1, dtype),
      shared_wo=_cast_opt(shared_wo, dtype),
      shared_gate=_cast_opt(shared_gate, dtype),
  )


def convert_maxtext_gdn_params(
    mt_gdn: Any,
    cfg: QwenConfig,
    *,
    pre_attn_norm: jax.Array | None = None,
    post_attn_norm: jax.Array | None = None,
    moe_params: MoeParams | None = None,
    dtype: jax.typing.DTypeLike | None = None,
) -> GdnSublayerParams:
  """Converts MaxText `Qwen3_5GatedDeltaNet` weights 1:1 into `GdnSublayerParams`."""
  mt_qkvz = _unwrap_nnx_value(mt_gdn.in_proj_qkvz.kernel)
  mt_ba = _unwrap_nnx_value(mt_gdn.in_proj_ba.kernel)
  in_proj_qkvz = deinterleave_maxtext_gdn_qkvz(mt_qkvz, cfg)
  in_proj_ba = deinterleave_maxtext_gdn_ba(mt_ba, cfg)
  conv1d = _unwrap_nnx_value(mt_gdn.conv1d.kernel)
  a_log = _unwrap_nnx_value(mt_gdn.A_log).astype(jnp.float32)
  dt_bias = _unwrap_nnx_value(mt_gdn.dt_bias).astype(jnp.float32)
  gdn_norm = _unwrap_nnx_value(mt_gdn.norm.rms_norm.scale)
  out_proj = _unwrap_nnx_value(mt_gdn.out_proj.kernel)

  lead_shape = in_proj_qkvz.shape[:-2]
  default_dt = in_proj_qkvz.dtype if dtype is None else jnp.dtype(dtype)
  z_norm = jnp.zeros((*lead_shape, cfg.emb_dim), dtype=default_dt)
  if moe_params is None:
    z_moe = MoeParams(
        router_gate=jnp.zeros(
            (*lead_shape, cfg.emb_dim, cfg.num_experts), dtype=default_dt
        ),
        wi=jnp.zeros(
            (*lead_shape, cfg.num_experts, cfg.emb_dim, 2 * cfg.moe_mlp_dim),
            dtype=default_dt,
        ),
        wo=jnp.zeros(
            (*lead_shape, cfg.num_experts, cfg.moe_mlp_dim, cfg.emb_dim),
            dtype=default_dt,
        ),
        shared_wi_0=jnp.zeros(
            (*lead_shape, cfg.emb_dim, cfg.shared_mlp_dim), dtype=default_dt
        ),
        shared_wi_1=jnp.zeros(
            (*lead_shape, cfg.emb_dim, cfg.shared_mlp_dim), dtype=default_dt
        ),
        shared_wo=jnp.zeros(
            (*lead_shape, cfg.shared_mlp_dim, cfg.emb_dim), dtype=default_dt
        ),
        shared_gate=jnp.zeros((*lead_shape, cfg.emb_dim, 1), dtype=default_dt),
    )
  else:
    z_moe = moe_params

  return GdnSublayerParams(
      pre_attn_norm=_cast_opt(
          z_norm if pre_attn_norm is None else pre_attn_norm, dtype
      ),
      post_attn_norm=_cast_opt(
          z_norm if post_attn_norm is None else post_attn_norm, dtype
      ),
      in_proj_qkvz=_cast_opt(in_proj_qkvz, dtype),
      in_proj_ba=_cast_opt(in_proj_ba, dtype),
      conv1d=_cast_opt(conv1d, dtype),
      A_log=a_log,
      dt_bias=dt_bias,
      gdn_norm=_cast_opt(gdn_norm, dtype),
      out_proj=_cast_opt(out_proj, dtype),
      moe=z_moe,
  )


def convert_maxtext_attn_params(
    mt_full_attn: Any,
    cfg: QwenConfig,
    *,
    pre_attn_norm: jax.Array | None = None,
    post_attn_norm: jax.Array | None = None,
    moe_params: MoeParams | None = None,
    dtype: jax.typing.DTypeLike | None = None,
) -> AttnSublayerParams:
  """Converts MaxText `Qwen3_5FullAttention` weights 1:1 into `AttnSublayerParams`."""
  attn = getattr(mt_full_attn, "attention", mt_full_attn)
  q_proj = _unwrap_nnx_value(attn.query.kernel)
  k_proj = _unwrap_nnx_value(attn.key.kernel)
  v_proj = _unwrap_nnx_value(attn.value.kernel)
  q_norm = _unwrap_nnx_value(attn.query_norm.scale)
  k_norm = _unwrap_nnx_value(attn.key_norm.scale)
  out_proj = _unwrap_nnx_value(attn.out.kernel)
  if out_proj.shape[-2:] != (cfg.attn_q_out_dim, cfg.emb_dim):
    out_proj = out_proj.reshape(
        *out_proj.shape[:-3], cfg.attn_q_out_dim, cfg.emb_dim
    )

  lead_shape = q_proj.shape[:-3]
  default_dt = q_proj.dtype if dtype is None else jnp.dtype(dtype)
  z_norm = jnp.zeros((*lead_shape, cfg.emb_dim), dtype=default_dt)
  if moe_params is None:
    z_moe = MoeParams(
        router_gate=jnp.zeros(
            (*lead_shape, cfg.emb_dim, cfg.num_experts), dtype=default_dt
        ),
        wi=jnp.zeros(
            (*lead_shape, cfg.num_experts, cfg.emb_dim, 2 * cfg.moe_mlp_dim),
            dtype=default_dt,
        ),
        wo=jnp.zeros(
            (*lead_shape, cfg.num_experts, cfg.moe_mlp_dim, cfg.emb_dim),
            dtype=default_dt,
        ),
        shared_wi_0=jnp.zeros(
            (*lead_shape, cfg.emb_dim, cfg.shared_mlp_dim), dtype=default_dt
        ),
        shared_wi_1=jnp.zeros(
            (*lead_shape, cfg.emb_dim, cfg.shared_mlp_dim), dtype=default_dt
        ),
        shared_wo=jnp.zeros(
            (*lead_shape, cfg.shared_mlp_dim, cfg.emb_dim), dtype=default_dt
        ),
        shared_gate=jnp.zeros((*lead_shape, cfg.emb_dim, 1), dtype=default_dt),
    )
  else:
    z_moe = moe_params

  return AttnSublayerParams(
      pre_attn_norm=_cast_opt(
          z_norm if pre_attn_norm is None else pre_attn_norm, dtype
      ),
      post_attn_norm=_cast_opt(
          z_norm if post_attn_norm is None else post_attn_norm, dtype
      ),
      q_proj=_cast_opt(q_proj, dtype),
      k_proj=_cast_opt(k_proj, dtype),
      v_proj=_cast_opt(v_proj, dtype),
      q_norm=_cast_opt(q_norm, dtype),
      k_norm=_cast_opt(k_norm, dtype),
      out_proj=_cast_opt(out_proj, dtype),
      moe=z_moe,
  )


def convert_maxtext_decoder_layer_params(
    mt_layer: Any,
    cfg: QwenConfig,
    *,
    dtype: jax.typing.DTypeLike | None = None,
) -> GdnSublayerParams | AttnSublayerParams:
  """Converts a MaxText `Qwen3_5DecoderLayer` into `GdnSublayerParams` or `AttnSublayerParams`."""
  pre_norm = _unwrap_nnx_value(mt_layer.input_layernorm.scale)
  post_norm = _unwrap_nnx_value(mt_layer.post_attention_layernorm.scale)
  moe_p = convert_maxtext_moe_params(mt_layer.mlp, cfg, dtype=dtype)
  if bool(getattr(mt_layer, "is_full_attention_layer", False)):
    return convert_maxtext_attn_params(
        mt_layer.attention,
        cfg,
        pre_attn_norm=pre_norm,
        post_attn_norm=post_norm,
        moe_params=moe_p,
        dtype=dtype,
    )
  return convert_maxtext_gdn_params(
      mt_layer.attention,
      cfg,
      pre_attn_norm=pre_norm,
      post_attn_norm=post_norm,
      moe_params=moe_p,
      dtype=dtype,
  )


def _slice_gdn_sublayer_axis0(
    stacked_gdn: GdnSublayerParams, idx: int
) -> GdnSublayerParams:
  return jax.tree.map(lambda arr: arr[idx], stacked_gdn)


def convert_maxtext_scannable_block_params(
    mt_block: Any,
    cfg: QwenConfig,
    *,
    param_scan_axis: int = 0,
    dtype: jax.typing.DTypeLike | None = None,
) -> SuperblockParams:
  """Converts one MaxText `Qwen3_5ScannableBlock` (`[3 GDN + 1 FullAttn]`) into `SuperblockParams`."""
  local_layers = mt_block.local_layers
  if param_scan_axis != 0:
    raise ValueError(
        "Expected param_scan_axis=0 on Qwen3_5ScannableBlock.local_layers, got"
        f" {param_scan_axis}."
    )
  stacked_gdn = convert_maxtext_decoder_layer_params(
      local_layers, cfg, dtype=dtype
  )
  assert isinstance(stacked_gdn, GdnSublayerParams)
  if stacked_gdn.pre_attn_norm.shape[0] != 3:
    raise ValueError(
        "Expected 3 local GDN layers per superblock, got"
        f" {stacked_gdn.pre_attn_norm.shape[0]}."
    )
  attn_3 = convert_maxtext_decoder_layer_params(
      mt_block.global_layer, cfg, dtype=dtype
  )
  assert isinstance(attn_3, AttnSublayerParams)
  return SuperblockParams(
      gdn_0=_slice_gdn_sublayer_axis0(stacked_gdn, 0),
      gdn_1=_slice_gdn_sublayer_axis0(stacked_gdn, 1),
      gdn_2=_slice_gdn_sublayer_axis0(stacked_gdn, 2),
      attn_3=attn_3,
  )


def stack_and_shard_model_params(
    superblocks: Sequence[SuperblockParams],
    token_embedder: jax.Array,
    decoder_norm: jax.Array,
    lm_head: jax.Array,
    cfg: QwenConfig,
    mesh: Mesh,
    *,
    dtype: jax.typing.DTypeLike | None = None,
) -> ModelParams:
  """Stacks per-superblock `SuperblockParams` onto axis 0 and shards onto `mesh`."""
  if len(superblocks) != cfg.num_superblocks:
    raise ValueError(
        f"Expected {cfg.num_superblocks} superblocks, got {len(superblocks)}."
    )
  blocks_stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *superblocks)
  raw_model = ModelParams(
      token_embedder=_cast_opt(token_embedder, dtype),
      blocks=blocks_stacked,
      decoder_norm=_cast_opt(decoder_norm, dtype),
      lm_head=_cast_opt(lm_head, dtype),
  )
  pspecs = get_param_pspecs(cfg)
  return jax.tree.map(
      lambda arr, ps: jax.device_put(np.asarray(arr), NamedSharding(mesh, ps)),
      raw_model,
      pspecs,
  )

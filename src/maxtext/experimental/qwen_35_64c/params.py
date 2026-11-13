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

"""Pure-JAX PyTree parameter structs, shardings, and abstract initialization."""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.experimental.qwen_35_64c.config import Qwen35Config


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MoeParams:
  """Parameters for one Qwen3Next Sparse MoE + Gated Shared Expert block."""

  router_gate: jax.Array  # [..., D, E] bf16
  wi: jax.Array  # [..., E, D, 2*M] bf16 (prefused gate|up)
  wo: jax.Array  # [..., E, M, D] bf16
  shared_wi_0: jax.Array  # [..., D, M] bf16
  shared_wi_1: jax.Array  # [..., D, M] bf16
  shared_wo: jax.Array  # [..., M, D] bf16
  shared_gate: jax.Array  # [..., D, 1] bf16


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class GdnSublayerParams:
  """Parameters for one Gated DeltaNet + MoE sub-layer."""

  pre_attn_norm: jax.Array  # [..., D] bf16 (zero-centered)
  post_attn_norm: jax.Array  # [..., D] bf16 (zero-centered)
  in_proj_qkvz: jax.Array  # [..., D, qkvz_dim] bf16 (kernel-order [q|k|v|z])
  in_proj_ba: jax.Array  # [..., D, 2*H_v] bf16
  conv1d: jax.Array  # [..., K_conv, 1, qkv_dim] bf16
  A_log: jax.Array  # [..., H_v] fp32
  dt_bias: jax.Array  # [..., H_v] fp32
  gdn_norm: jax.Array  # [..., D_v] bf16 (ones-init gated RMSNorm)
  out_proj: jax.Array  # [..., V_dim, D] bf16
  moe: MoeParams


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class AttnSublayerParams:
  """Parameters for one Global Full Attention + MoE sub-layer."""

  pre_attn_norm: jax.Array  # [..., D] bf16 (zero-centered)
  post_attn_norm: jax.Array  # [..., D] bf16 (zero-centered)
  q_proj: jax.Array  # [..., D, H_q, 2*D_h] bf16 (query + output gate)
  k_proj: jax.Array  # [..., D, H_kv, D_h] bf16
  v_proj: jax.Array  # [..., D, H_kv, D_h] bf16
  q_norm: jax.Array  # [..., D_h] bf16 (zero-centered)
  k_norm: jax.Array  # [..., D_h] bf16 (zero-centered)
  out_proj: jax.Array  # [..., H_q*D_h, D] bf16
  moe: MoeParams


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SuperblockParams:
  """Parameters for [GDN_0, GDN_1, GDN_2, FullAttn_3] stacked over superblocks."""

  gdn_0: GdnSublayerParams
  gdn_1: GdnSublayerParams
  gdn_2: GdnSublayerParams
  attn_3: AttnSublayerParams


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ModelParams:
  """Complete Qwen3.5-397B-A17B model parameter PyTree."""

  token_embedder: jax.Array  # [V, D] bf16
  blocks: SuperblockParams  # leading axis = [num_superblocks, ...]
  decoder_norm: jax.Array  # [D] bf16 (zero-centered)
  lm_head: jax.Array  # [D, V] bf16


def _dense_axis_for_param(cfg: Qwen35Config, param_name: str) -> str | tuple[str, ...] | None:
  kind = cfg.dense_param_sharding_kind(param_name)
  if kind == "replicated":
    return None
  if kind == "fsdp_only":
    return cfg.dense_stage2_axis_names
  return cfg.all_sharded_axes


def get_param_pspecs(cfg: Qwen35Config) -> ModelParams:
  """Returns PartitionSpecs for every parameter leaf (`blocks` stacked on dim 0)."""
  moe_e_ax = (*cfg.ep_axis_tuple, "fsdp")
  rg_ax = _dense_axis_for_param(cfg, "router_gate")
  swi_ax = _dense_axis_for_param(cfg, "shared_wi")
  swo_ax = _dense_axis_for_param(cfg, "shared_wo")
  sg_ax = _dense_axis_for_param(cfg, "shared_gate")
  qkvz_ax = _dense_axis_for_param(cfg, "in_proj_qkvz")
  ba_ax = _dense_axis_for_param(cfg, "in_proj_ba")
  gdn_out_ax = _dense_axis_for_param(cfg, "gdn_out_proj")
  q_ax = _dense_axis_for_param(cfg, "q_proj")
  kv_ax = _dense_axis_for_param(cfg, "kv_proj")
  attn_out_ax = _dense_axis_for_param(cfg, "attn_out_proj")

  def _moe_pspec() -> MoeParams:
    return MoeParams(
        router_gate=P(None, rg_ax, None),
        # (*ep_axis_tuple, 'fsdp') places local experts contiguously across the
        # fsdp shards of each EP rank -> 1 expert per TensorCore!
        wi=P(None, moe_e_ax, None, None),
        wo=P(None, moe_e_ax, None, None),
        shared_wi_0=P(None, swi_ax, None),
        shared_wi_1=P(None, swi_ax, None),
        shared_wo=P(None, None, swo_ax),  # shard D (axis 2) so all_sharded_axes divides cleanly!
        shared_gate=P(None, sg_ax, None),
    )

  def _gdn_pspec() -> GdnSublayerParams:
    return GdnSublayerParams(
        pre_attn_norm=P(None, None),
        post_attn_norm=P(None, None),
        in_proj_qkvz=P(None, qkvz_ax, None),
        in_proj_ba=P(None, ba_ax, None),
        conv1d=P(None, None, None, None),
        A_log=P(None, None),
        dt_bias=P(None, None),
        gdn_norm=P(None, None),
        out_proj=P(None, gdn_out_ax, None),
        moe=_moe_pspec(),
    )

  attn_pspec = AttnSublayerParams(
      pre_attn_norm=P(None, None),
      post_attn_norm=P(None, None),
      q_proj=P(None, q_ax, None, None),
      k_proj=P(None, kv_ax, None, None),
      v_proj=P(None, kv_ax, None, None),
      q_norm=P(None, None),
      k_norm=P(None, None),
      out_proj=P(None, attn_out_ax, None),
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


def get_abstract_params(cfg: Qwen35Config, mesh: Mesh) -> ModelParams:
  """Builds abstract `jax.ShapeDtypeStruct` parameter tree with NamedShardings."""
  pspecs = get_param_pspecs(cfg)
  sb = cfg.num_superblocks
  d = cfg.emb_dim
  e = cfg.num_experts
  m = cfg.moe_mlp_dim
  ms = cfg.shared_mlp_dim

  def _s(shape: tuple[int, ...], dtype, pspec: P) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, dtype, sharding=NamedSharding(mesh, pspec))

  def _moe_shapes(ps: MoeParams) -> MoeParams:
    return MoeParams(
        router_gate=_s((sb, d, e), jnp.bfloat16, ps.router_gate),
        wi=_s((sb, e, d, 2 * m), jnp.bfloat16, ps.wi),
        wo=_s((sb, e, m, d), jnp.bfloat16, ps.wo),
        shared_wi_0=_s((sb, d, ms), jnp.bfloat16, ps.shared_wi_0),
        shared_wi_1=_s((sb, d, ms), jnp.bfloat16, ps.shared_wi_1),
        shared_wo=_s((sb, ms, d), jnp.bfloat16, ps.shared_wo),
        shared_gate=_s((sb, d, 1), jnp.bfloat16, ps.shared_gate),
    )

  def _gdn_shapes(ps: GdnSublayerParams) -> GdnSublayerParams:
    return GdnSublayerParams(
        pre_attn_norm=_s((sb, d), jnp.bfloat16, ps.pre_attn_norm),
        post_attn_norm=_s((sb, d), jnp.bfloat16, ps.post_attn_norm),
        in_proj_qkvz=_s((sb, d, cfg.gdn_qkvz_dim), jnp.bfloat16, ps.in_proj_qkvz),
        in_proj_ba=_s((sb, d, 2 * cfg.gdn_num_v_heads), jnp.bfloat16, ps.in_proj_ba),
        conv1d=_s(
            (sb, cfg.gdn_conv_kernel_size, 1, cfg.gdn_qkv_dim),
            jnp.bfloat16,
            ps.conv1d,
        ),
        A_log=_s((sb, cfg.gdn_num_v_heads), jnp.float32, ps.A_log),
        dt_bias=_s((sb, cfg.gdn_num_v_heads), jnp.float32, ps.dt_bias),
        gdn_norm=_s((sb, cfg.gdn_head_v_dim), jnp.bfloat16, ps.gdn_norm),
        out_proj=_s((sb, cfg.gdn_value_dim, d), jnp.bfloat16, ps.out_proj),
        moe=_moe_shapes(ps.moe),
    )

  attn_ps = pspecs.blocks.attn_3
  attn_shapes = AttnSublayerParams(
      pre_attn_norm=_s((sb, d), jnp.bfloat16, attn_ps.pre_attn_norm),
      post_attn_norm=_s((sb, d), jnp.bfloat16, attn_ps.post_attn_norm),
      q_proj=_s(
          (sb, d, cfg.num_query_heads, 2 * cfg.head_dim),
          jnp.bfloat16,
          attn_ps.q_proj,
      ),
      k_proj=_s((sb, d, cfg.num_kv_heads, cfg.head_dim), jnp.bfloat16, attn_ps.k_proj),
      v_proj=_s((sb, d, cfg.num_kv_heads, cfg.head_dim), jnp.bfloat16, attn_ps.v_proj),
      q_norm=_s((sb, cfg.head_dim), jnp.bfloat16, attn_ps.q_norm),
      k_norm=_s((sb, cfg.head_dim), jnp.bfloat16, attn_ps.k_norm),
      out_proj=_s((sb, cfg.attn_q_out_dim, d), jnp.bfloat16, attn_ps.out_proj),
      moe=_moe_shapes(attn_ps.moe),
  )

  return ModelParams(
      token_embedder=_s((cfg.vocab_size, d), jnp.bfloat16, pspecs.token_embedder),
      blocks=SuperblockParams(
          gdn_0=_gdn_shapes(pspecs.blocks.gdn_0),
          gdn_1=_gdn_shapes(pspecs.blocks.gdn_1),
          gdn_2=_gdn_shapes(pspecs.blocks.gdn_2),
          attn_3=attn_shapes,
      ),
      decoder_norm=_s((d,), jnp.bfloat16, pspecs.decoder_norm),
      lm_head=_s((d, cfg.vocab_size), jnp.bfloat16, pspecs.lm_head),
  )


def _local_shape_and_axes(global_shape: tuple[int, ...], pspec: P, mesh: Mesh) -> tuple[tuple[int, ...], tuple[str, ...]]:
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
    cfg: Qwen35Config,
    mesh: Mesh,
    key: jax.Array,
    *,
    param_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> ModelParams:
  """Initializes sharded parameters directly on device (1-host & 64-host safe)."""
  abstract = get_abstract_params(cfg, mesh)
  leaves, treedef = jax.tree.flatten_with_path(abstract)
  ref_cfg = dataclasses.replace(
      cfg,
      dense_sharding_mode="fsdp_expert",
      replicate_sub_sc_dense=False,
  )
  ref_pspec_leaves = jax.tree.leaves(get_param_pspecs(ref_cfg), is_leaf=lambda x: isinstance(x, P))
  all_axes = cfg.all_sharded_axes
  total_n = math.prod(int(mesh.shape[a]) for a in all_axes)
  s2_axes = all_axes[:-1]
  minor_n = int(mesh.shape[all_axes[-1]])
  rep_sharding = NamedSharding(mesh, P())
  keys = _make_replicated_keys(mesh, key, len(leaves))
  compiled_builders: dict[tuple[object, ...], object] = {}
  out_leaves = []

  for idx, (path, spec) in enumerate(leaves):
    path_str = jax.tree_util.keystr(path)
    pspec = spec.sharding.spec
    ref_pspec = ref_pspec_leaves[idx]
    local_shape, sharded_axes = _local_shape_and_axes(spec.shape, pspec, mesh)
    ref_sh_dim: int | None = None
    if pspec != ref_pspec and all_axes in tuple(ref_pspec):
      cand_dim = tuple(ref_pspec).index(all_axes)
      if cand_dim < len(spec.shape) and spec.shape[cand_dim] % total_n == 0:
        ref_sh_dim = cand_dim
    target_dtype = jnp.dtype(jnp.float32 if spec.dtype == jnp.float32 else param_dtype)
    if (
        "pre_attn_norm" in path_str
        or "post_attn_norm" in path_str
        or "decoder_norm" in path_str
        or "q_norm" in path_str
        or "k_norm" in path_str
    ):
      # Zero-centered RMSNorm (`y * (1 + weight)`): 0.0 -> identity scale 1.0,
      # matching Qwen3.5 (`init.zeros_`) and MaxText.
      kind = "zeros"
      scale = 0.0
    elif "gdn_norm" in path_str:
      kind = "ones"
      scale = 1.0
    elif "A_log" in path_str:
      kind = "a_log"
      scale = 1.0
    elif "dt_bias" in path_str:
      # Official Qwen3.5 & MaxText initialize GDN `dt_bias` to ones.
      kind = "ones"
      scale = 1.0
    elif "token_embedder" in path_str:
      # Shape `(vocab_size, emb_dim)`: variance scaling over `emb_dim` (`out_axis=0`).
      kind = "normal_scaled"
      scale = float(spec.shape[1] ** -0.5)
    elif "conv1d" in path_str:
      # Depthwise conv `(sb, K_conv, 1, qkv_dim)`: `fan_in = K_conv`.
      kind = "normal_scaled"
      scale = float(spec.shape[1] ** -0.5)
    elif "q_proj" in path_str or "k_proj" in path_str or "v_proj" in path_str:
      # Unpacked 4D attention projections `(sb, emb_dim, H, C)` contract over
      # `emb_dim` (`axis=1`), NOT `H = shape[-2]`.
      kind = "normal_scaled"
      scale = float(spec.shape[1] ** -0.5)
    else:
      kind = "normal_scaled"
      fan_in = spec.shape[-2] if len(spec.shape) >= 2 else spec.shape[-1]
      scale = float(fan_in**-0.5)

    cache_key = (
        local_shape,
        pspec,
        sharded_axes,
        ref_sh_dim,
        kind,
        scale,
        target_dtype.str,
    )
    builder = compiled_builders.get(cache_key)
    if builder is None:

      def _make_builder(lshape, ps, axes, rdim, knd, scl, dt):
        def _sample_chunk(subk, shape_c):
          if knd == "ones":
            return jnp.ones(shape_c, dtype=jnp.float32).astype(dt)
          if knd == "zeros":
            return jnp.zeros(shape_c, dtype=jnp.float32).astype(dt)
          if knd == "a_log":
            # Matches official Qwen3.5 `uniform_(0.01, 16).log_()`.
            return jnp.log(jax.random.uniform(subk, shape_c, dtype=jnp.float32, minval=0.01, maxval=16.0)).astype(dt)
          return (jax.random.normal(subk, shape_c, dtype=jnp.float32) * scl).astype(dt)

        def _local_init(subk):
          if rdim is not None and axes in (s2_axes, ()):
            cshape = tuple(
                spec_d // (minor_n if axes == s2_axes else total_n) if i == rdim else spec_d
                for i, spec_d in enumerate(lshape)
            )
            if axes == s2_axes:
              s2_id = jnp.int32(0)
              for ax in s2_axes:
                s2_id = s2_id * jnp.int32(mesh.shape[ax]) + jax.lax.axis_index(ax)
              sids = s2_id * jnp.int32(minor_n) + jnp.arange(minor_n, dtype=jnp.int32)
            else:
              sids = jnp.arange(total_n, dtype=jnp.int32)
            chunks = jax.vmap(lambda sid: _sample_chunk(jax.random.fold_in(subk, sid), cshape))(sids)
            return jnp.moveaxis(chunks, 0, rdim).reshape(lshape)

          if axes:
            shard_id = jnp.int32(0)
            for ax in axes:
              shard_id = shard_id * jnp.int32(mesh.shape[ax]) + jax.lax.axis_index(ax)
            subk = jax.random.fold_in(subk, shard_id)
          return _sample_chunk(subk, lshape)

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
          local_shape,
          pspec,
          sharded_axes,
          ref_sh_dim,
          kind,
          scale,
          target_dtype,
      )
      compiled_builders[cache_key] = builder

    out_leaves.append(builder(keys[idx]))

  return jax.tree.unflatten(treedef, out_leaves)

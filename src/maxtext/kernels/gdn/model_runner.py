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

"""Model-facing execution helper for GatedDeltaNet Pallas kernel and context parallelism."""

import functools
from typing import Any

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from maxtext.common.common_types import (
    Array,
    DType,
    KV_BATCH,
    LENGTH,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_TRAIN,
    ShardMode,
)
from maxtext.utils.sharding import (
    logical_to_mesh_axes,
    remove_incompatible_mesh_axes_from_partition_spec,
)
from .gdn_bwd_pallas import gdn_decoupled_conv1d

try:
  from jax.extend.core.primitives import name_p
except ImportError:
  try:
    from jax._src.ad_checkpoint import name_p
  except ImportError:
    name_p = None

_GDN_SAVED_NAMES = frozenset(
    {
        "gdn",
        "gdn_core_attn_out",
        "gdn_fwd_out",
        "gdn_qkv",
        "gdn_b",
        "gdn_a",
        "gdn_t_inv",
        "gdn_chunk_states",
        "gdn_m_local",
        "gdn_conv_state",
        "gdn_recurrent_state",
        "gdn_conv",
        "gdn_conv_out",
    }
)


def _default_gdn_context_axes(cfg: Any) -> tuple[str, ...]:
  """Mesh axes carrying the GatedDeltaNet sequence, empty when it is replicated."""
  return tuple(
      name
      for name, size in (
          ("context", getattr(cfg, "ici_context_parallelism", 1)),
          ("context_usp_ulysses", getattr(cfg, "ici_context_usp_ulysses_parallelism", 1)),
      )
      if size > 1
  )


def get_gdn_aware_remat_policy(base_policy: Any):
  """Checkpoint policy that preserves GDN forward residuals under any remat policy."""

  def policy(prim, *args, **params):
    is_name = (name_p is not None and prim is name_p) or getattr(prim, "name", None) == "name"
    if is_name and params.get("name") in _GDN_SAVED_NAMES:
      return True
    if base_policy is not None:
      return base_policy(prim, *args, **params)
    return False

  return policy


def get_gdn_kernel_cp_sharding_info(
    config: Any,
    mesh: jax.sharding.Mesh | None,
    get_logical_axis_rules_fn: Any,
    gdn_context_axes_fn: Any = _default_gdn_context_axes,
) -> tuple[Any, tuple[str, ...], str | tuple[str, ...] | None, str | None]:
  """Returns (logical_rules, cp_axes_active, cp_axis_for_pspec, cp_len) for GDN kernel CP sharding."""
  logical_rules = get_logical_axis_rules_fn() or config.logical_axis_rules
  cp_axes_active = tuple(
      ax
      for ax in gdn_context_axes_fn(config)
      if mesh is not None and ax in mesh.axis_names and mesh.shape[ax] > 1
  )
  cp_axis_for_pspec = (
      cp_axes_active[0]
      if len(cp_axes_active) == 1
      else (cp_axes_active if cp_axes_active else None)
  )
  cp_len = LENGTH if cp_axes_active else None
  return logical_rules, cp_axes_active, cp_axis_for_pspec, cp_len


def run_gdn_kernel_layer(
    *,
    layer: Any,
    query: Array,
    key: Array,
    value: Array,
    value_raw: Array,
    b: Array,
    a: Array,
    active_cache: Any,
    decoder_segment_ids: None | Array,
    model_mode: str,
    state_dtype: DType,
    flat_sharding: jax.sharding.NamedSharding | None = None,
    gdn_context_axes_fn: Any = _default_gdn_context_axes,
) -> tuple[Array, Array | None, Array | None]:
  """Executes the fused/decoupled Pallas GDN conv1d + delta-rule kernel with CP support."""
  cfg = layer.config
  batch, seq_len = query.shape[:2]

  cp_axes = tuple(
      ax
      for ax in gdn_context_axes_fn(cfg)
      if layer.mesh and ax in layer.mesh.axis_names and layer.mesh.shape[ax] > 1
  )
  cp_axis_name = cp_axes[0] if len(cp_axes) == 1 else (cp_axes if cp_axes else None)
  cp_size = 1
  if cp_axes:
    for ax in cp_axes:
      cp_size *= layer.mesh.shape[ax]

  gdn_cp_mode = getattr(cfg, "gdn_cp_mode", "auto")
  if gdn_cp_mode == "head":
    if cp_size > 1 and (cp_size > layer.num_k_heads or layer.num_k_heads % cp_size != 0):
      raise ValueError(
          f"GDN head-sharded CP requires num_k_heads ({layer.num_k_heads}) to"
          f" be divisible by cp_size ({cp_size})."
      )
    use_head_sharded_cp = cp_size > 1 and model_mode != MODEL_MODE_AUTOREGRESSIVE
  elif gdn_cp_mode == "auto":
    use_head_sharded_cp = (
        1 < cp_size <= min(2, layer.num_k_heads)
        and (layer.num_k_heads % cp_size == 0)
        and model_mode != MODEL_MODE_AUTOREGRESSIVE
    )
  else:
    use_head_sharded_cp = False

  use_seq_sharded_cp = (
      cp_size > 1
      and model_mode != MODEL_MODE_AUTOREGRESSIVE
      and not use_head_sharded_cp
  )

  if use_head_sharded_cp or use_seq_sharded_cp:
    if getattr(cfg, "context_parallel_load_balance", False):
      raise ValueError("GDN does not support context_parallel_load_balance.")

  if use_head_sharded_cp:
    qkv = jnp.concatenate([query, key, value_raw], axis=3)
  else:
    q = jnp.reshape(query, (batch, seq_len, -1), out_sharding=flat_sharding)
    k = jnp.reshape(key, (batch, seq_len, -1), out_sharding=flat_sharding)
    v = jnp.reshape(value, (batch, seq_len, -1), out_sharding=flat_sharding)
    qkv = jnp.concatenate([q, k, v], axis=-1)

  if decoder_segment_ids is not None:
    mask = decoder_segment_ids != 0
    qkv = jnp.where(mask.reshape(mask.shape + (1,) * (qkv.ndim - mask.ndim)), qkv, 0.0)
    a = jnp.where(mask[..., None], a, jnp.asarray(-1e4, dtype=a.dtype))
    b = jnp.where(mask[..., None], b, jnp.asarray(-1e4, dtype=b.dtype))

  batch, seq_len = qkv.shape[:2]
  conv_kernel_size = cfg.gdn_conv_kernel_dim

  conv_state = None
  recurrent_state = None
  orig_cache_batch = None
  if model_mode != MODEL_MODE_TRAIN and active_cache is not None:
    recurrent_state, conv_state = active_cache.get_gdn_states()
    orig_cache_batch = conv_state.shape[0]

    if conv_state.shape[0] != batch:
      if conv_state.shape[0] == 1:
        conv_state = jnp.broadcast_to(conv_state, (batch,) + conv_state.shape[1:])
      elif conv_state.shape[0] < batch:
        pad_amt = batch - conv_state.shape[0]
        conv_state = jnp.pad(conv_state, ((0, pad_amt), (0, 0), (0, 0)))
      else:
        conv_state = conv_state[:batch]

    if recurrent_state.shape[0] != batch:
      if recurrent_state.shape[0] == 1:
        recurrent_state = jnp.broadcast_to(recurrent_state, (batch,) + recurrent_state.shape[1:])
      elif recurrent_state.shape[0] < batch:
        pad_amt = batch - recurrent_state.shape[0]
        recurrent_state = jnp.pad(recurrent_state, ((0, pad_amt), (0, 0), (0, 0), (0, 0)))
      else:
        recurrent_state = recurrent_state[:batch]

  conv_bias_arg = layer.conv1d.bias.value if getattr(layer.conv1d, "bias", None) is not None else None
  conv_weight_arg = layer.conv1d.kernel.value
  conv_state_arg = (
      conv_state
      if conv_state is not None
      else jnp.zeros(
          (batch, cfg.gdn_conv_kernel_dim - 1, qkv.shape[-1]),
          dtype=cfg.dtype,
      )
  )
  recurrent_state_arg = (
      recurrent_state.astype(state_dtype)
      if recurrent_state is not None
      else jnp.zeros(
          (batch, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim),
          dtype=state_dtype,
      )
  )

  if layer.mesh is not None:
    logical_rules = cfg.logical_axis_rules
    if use_head_sharded_cp:
      num_groups = layer.num_k_heads
      channels_per_group = 2 * layer.head_k_dim + layer.v_heads_per_k_head * layer.head_v_dim

      w = layer.conv1d.kernel.value
      w_q = w[..., : layer.key_dim].reshape(conv_kernel_size, 1, num_groups, layer.head_k_dim)
      w_k = w[..., layer.key_dim : 2 * layer.key_dim].reshape(conv_kernel_size, 1, num_groups, layer.head_k_dim)
      w_v = w[..., 2 * layer.key_dim :].reshape(
          conv_kernel_size, 1, num_groups, layer.v_heads_per_k_head * layer.head_v_dim
      )
      conv_weight_grouped = jnp.concatenate([w_q, w_k, w_v], axis=-1)
      conv_weight_arg = jnp.swapaxes(conv_weight_grouped, 0, 2)

      if conv_bias_arg is not None:
        cb = conv_bias_arg
        cb_q = cb[: layer.key_dim].reshape(num_groups, layer.head_k_dim)
        cb_k = cb[layer.key_dim : 2 * layer.key_dim].reshape(num_groups, layer.head_k_dim)
        cb_v = cb[2 * layer.key_dim :].reshape(num_groups, layer.v_heads_per_k_head * layer.head_v_dim)
        conv_bias_arg = jnp.concatenate([cb_q, cb_k, cb_v], axis=-1)
        conv_bias_pspec = P(cp_axis_name, None)
      else:
        conv_bias_pspec = P()

      if conv_state is not None:
        cs_q = conv_state[..., : layer.key_dim].reshape(
            batch, cfg.gdn_conv_kernel_dim - 1, num_groups, layer.head_k_dim
        )
        cs_k = conv_state[..., layer.key_dim : 2 * layer.key_dim].reshape(
            batch, cfg.gdn_conv_kernel_dim - 1, num_groups, layer.head_k_dim
        )
        cs_v = conv_state[..., 2 * layer.key_dim :].reshape(
            batch, cfg.gdn_conv_kernel_dim - 1, num_groups, layer.v_heads_per_k_head * layer.head_v_dim
        )
        conv_state_arg = jnp.concatenate([cs_q, cs_k, cs_v], axis=-1)
      else:
        conv_state_arg = jnp.zeros(
            (batch, cfg.gdn_conv_kernel_dim - 1, num_groups, channels_per_group),
            dtype=cfg.dtype,
        )
      recurrent_state_arg = (
          recurrent_state.astype(state_dtype)
          if recurrent_state is not None
          else jnp.zeros(
              (batch, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim),
              dtype=state_dtype,
          )
      )
      a_log_arg = layer.A_log[...]
      dt_bias_arg = layer.dt_bias[...]

      mesh_batch = logical_to_mesh_axes((KV_BATCH,), mesh=layer.mesh, rules=logical_rules)[0]
      qkv_pspec = P(mesh_batch, None, cp_axis_name, None)
      b_a_pspec = P(mesh_batch, None, cp_axis_name)
      conv_weight_pspec = P(cp_axis_name, None, None, None)
      a_log_pspec = P(cp_axis_name)
      dt_bias_pspec = P(cp_axis_name)
      conv_state_pspec = P(mesh_batch, None, cp_axis_name, None)
      recurrent_state_pspec = P(mesh_batch, cp_axis_name, None, None)
      out_attn_pspec = P(mesh_batch, cp_axis_name, None, None)
    elif use_seq_sharded_cp:
      conv_weight_arg = layer.conv1d.kernel.value
      conv_state_arg = (
          conv_state
          if conv_state is not None
          else jnp.zeros(
              (batch, cfg.gdn_conv_kernel_dim - 1, qkv.shape[-1]),
              dtype=cfg.dtype,
          )
      )
      recurrent_state_arg = (
          recurrent_state.astype(state_dtype)
          if recurrent_state is not None
          else jnp.zeros(
              (batch, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim),
              dtype=state_dtype,
          )
      )
      a_log_arg = layer.A_log[...]
      dt_bias_arg = layer.dt_bias[...]

      mesh_batch = logical_to_mesh_axes((KV_BATCH,), mesh=layer.mesh, rules=logical_rules)[0]
      qkv_pspec = P(mesh_batch, cp_axis_name, None)
      b_a_pspec = P(mesh_batch, cp_axis_name, None)
      conv_weight_pspec = P()
      conv_bias_pspec = P()
      a_log_pspec = P()
      dt_bias_pspec = P()
      conv_state_pspec = P(mesh_batch, None, None)
      recurrent_state_pspec = P(mesh_batch, None, None, None)
      out_attn_pspec = P(mesh_batch, cp_axis_name, None, None)
    else:
      conv_weight_arg = layer.conv1d.kernel.value
      conv_state_arg = (
          conv_state
          if conv_state is not None
          else jnp.zeros(
              (batch, cfg.gdn_conv_kernel_dim - 1, qkv.shape[-1]),
              dtype=cfg.dtype,
          )
      )
      recurrent_state_arg = (
          recurrent_state.astype(state_dtype)
          if recurrent_state is not None
          else jnp.zeros(
              (batch, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim),
              dtype=state_dtype,
          )
      )
      a_log_arg = layer.A_log[...]
      dt_bias_arg = layer.dt_bias[...]

      qkv_pspec = logical_to_mesh_axes((KV_BATCH, None, None), mesh=layer.mesh, rules=logical_rules)
      b_a_pspec = logical_to_mesh_axes((KV_BATCH, None, None), mesh=layer.mesh, rules=logical_rules)
      conv_weight_pspec = P()
      conv_bias_pspec = P()
      a_log_pspec = P()
      dt_bias_pspec = P()
      conv_state_pspec = logical_to_mesh_axes((KV_BATCH, None, None), mesh=layer.mesh, rules=logical_rules)
      recurrent_state_pspec = logical_to_mesh_axes((KV_BATCH, None, None, None), mesh=layer.mesh, rules=logical_rules)
      out_attn_pspec = qkv_pspec

    qkv_pspec = remove_incompatible_mesh_axes_from_partition_spec(
        qkv_pspec, qkv.shape, layer.mesh, dims=(0,), allow_remove_axes=True
    )
    b_a_pspec = remove_incompatible_mesh_axes_from_partition_spec(
        b_a_pspec, b.shape, layer.mesh, dims=(0,), allow_remove_axes=True
    )
    conv_state_pspec = remove_incompatible_mesh_axes_from_partition_spec(
        conv_state_pspec, conv_state_arg.shape, layer.mesh, dims=(0,), allow_remove_axes=True
    )
    recurrent_state_pspec = remove_incompatible_mesh_axes_from_partition_spec(
        recurrent_state_pspec, recurrent_state_arg.shape, layer.mesh, dims=(0,), allow_remove_axes=True
    )
    out_attn_pspec = remove_incompatible_mesh_axes_from_partition_spec(
        out_attn_pspec,
        (batch, seq_len, layer.num_v_heads, layer.head_v_dim),
        layer.mesh,
        dims=(0,),
        allow_remove_axes=True,
    )

    seg_pspec = P(qkv_pspec[0], qkv_pspec[1]) if decoder_segment_ids is not None else P()
    if cfg.shard_mode == ShardMode.EXPLICIT:
      qkv = jax.sharding.reshard(qkv, jax.sharding.NamedSharding(layer.mesh, qkv_pspec))
      b = jax.sharding.reshard(b, jax.sharding.NamedSharding(layer.mesh, b_a_pspec))
      a = jax.sharding.reshard(a, jax.sharding.NamedSharding(layer.mesh, b_a_pspec))
      conv_state_arg = jax.sharding.reshard(conv_state_arg, jax.sharding.NamedSharding(layer.mesh, conv_state_pspec))
      recurrent_state_arg = jax.sharding.reshard(
          recurrent_state_arg, jax.sharding.NamedSharding(layer.mesh, recurrent_state_pspec)
      )
      if decoder_segment_ids is not None:
        decoder_segment_ids = jax.sharding.reshard(
            decoder_segment_ids, jax.sharding.NamedSharding(layer.mesh, seg_pspec)
        )

    @functools.partial(
        jax.shard_map,
        mesh=layer.mesh,
        in_specs=(
            qkv_pspec,
            b_a_pspec,
            b_a_pspec,
            conv_weight_pspec,
            conv_bias_pspec,
            a_log_pspec,
            dt_bias_pspec,
            conv_state_pspec,
            recurrent_state_pspec,
            seg_pspec,
        ),
        out_specs=(
            out_attn_pspec,
            (conv_state_pspec, recurrent_state_pspec),
        ),
        check_vma=False,
    )
    def shard_mapped_gdn(
        qkv_val,
        b_val,
        a_val,
        cw_val,
        cb_val,
        alog_val,
        dt_val,
        cs_val,
        rs_val,
        seg_val,
    ):
      if use_head_sharded_cp:
        local_num_k_heads = layer.num_k_heads // cp_size
        local_num_v_heads = layer.num_v_heads // cp_size

        b_sz, full_s, _, _ = qkv_val.shape
        q_val = qkv_val[..., : layer.head_k_dim].reshape(b_sz, full_s, -1)
        k_val = qkv_val[..., layer.head_k_dim : 2 * layer.head_k_dim].reshape(b_sz, full_s, -1)
        v_val = qkv_val[..., 2 * layer.head_k_dim :].reshape(b_sz, full_s, -1)
        qkv_val_flat = jnp.concatenate([q_val, k_val, v_val], axis=-1)

        cw_swapped = jnp.swapaxes(cw_val, 0, 2)
        cw_q = cw_swapped[..., : layer.head_k_dim].reshape(cfg.gdn_conv_kernel_dim, 1, -1)
        cw_k = cw_swapped[..., layer.head_k_dim : 2 * layer.head_k_dim].reshape(cfg.gdn_conv_kernel_dim, 1, -1)
        cw_v = cw_swapped[..., 2 * layer.head_k_dim :].reshape(cfg.gdn_conv_kernel_dim, 1, -1)
        cw_val_flat = jnp.concatenate([cw_q, cw_k, cw_v], axis=-1)

        if cb_val is not None:
          cb_q = cb_val[..., : layer.head_k_dim].reshape(-1)
          cb_k = cb_val[..., layer.head_k_dim : 2 * layer.head_k_dim].reshape(-1)
          cb_v = cb_val[..., 2 * layer.head_k_dim :].reshape(-1)
          cb_val_flat = jnp.concatenate([cb_q, cb_k, cb_v], axis=-1)
        else:
          cb_val_flat = None

        if cs_val is not None:
          cs_q = cs_val[..., : layer.head_k_dim].reshape(b_sz, cfg.gdn_conv_kernel_dim - 1, -1)
          cs_k = cs_val[..., layer.head_k_dim : 2 * layer.head_k_dim].reshape(
              b_sz, cfg.gdn_conv_kernel_dim - 1, -1
          )
          cs_v = cs_val[..., 2 * layer.head_k_dim :].reshape(b_sz, cfg.gdn_conv_kernel_dim - 1, -1)
          cs_val_flat = jnp.concatenate([cs_q, cs_k, cs_v], axis=-1)
        else:
          cs_val_flat = None

        out, (next_cs, next_rs) = gdn_decoupled_conv1d(
            qkv=qkv_val_flat,
            b=b_val,
            a=a_val,
            conv_weight=cw_val_flat,
            conv_bias=cb_val_flat,
            a_log=alog_val,
            dt_bias=dt_val,
            conv_state=cs_val_flat,
            recurrent_state=rs_val,
            num_k_heads=local_num_k_heads,
            num_v_heads=local_num_v_heads,
            head_k_dim=layer.head_k_dim,
            head_v_dim=layer.head_v_dim,
            conv_kernel_size=cfg.gdn_conv_kernel_dim,
            chunk_size=cfg.gdn_chunk_size,
            use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn,
            compute_dtype=state_dtype,
            segment_ids=seg_val,
        )

        if next_cs is not None:
          cs_k_dim = local_num_k_heads * layer.head_k_dim
          n_cs_q = next_cs[..., :cs_k_dim].reshape(
              b_sz, cfg.gdn_conv_kernel_dim - 1, local_num_k_heads, layer.head_k_dim
          )
          n_cs_k = next_cs[..., cs_k_dim : 2 * cs_k_dim].reshape(
              b_sz, cfg.gdn_conv_kernel_dim - 1, local_num_k_heads, layer.head_k_dim
          )
          n_cs_v = next_cs[..., 2 * cs_k_dim :].reshape(
              b_sz, cfg.gdn_conv_kernel_dim - 1, local_num_k_heads, layer.v_heads_per_k_head * layer.head_v_dim
          )
          next_cs_out = jnp.concatenate([n_cs_q, n_cs_k, n_cs_v], axis=-1)
        else:
          next_cs_out = None

        out = jax.lax.all_to_all(out, axis_name=cp_axis_name, split_axis=1, concat_axis=2, tiled=True)
        return out, (next_cs_out, next_rs)

      return gdn_decoupled_conv1d(
          qkv=qkv_val,
          b=b_val,
          a=a_val,
          conv_weight=cw_val,
          conv_bias=cb_val,
          a_log=alog_val,
          dt_bias=dt_val,
          conv_state=cs_val,
          recurrent_state=rs_val,
          num_k_heads=layer.num_k_heads,
          num_v_heads=layer.num_v_heads,
          head_k_dim=layer.head_k_dim,
          head_v_dim=layer.head_v_dim,
          conv_kernel_size=cfg.gdn_conv_kernel_dim,
          chunk_size=cfg.gdn_chunk_size,
          use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn,
          compute_dtype=state_dtype,
          cp_axis_name=cp_axis_name if use_seq_sharded_cp else None,
          segment_ids=seg_val,
      )

    core_attn_out, (next_conv_state, next_recurrent_state) = shard_mapped_gdn(
        qkv,
        b,
        a,
        conv_weight_arg,
        conv_bias_arg,
        a_log_arg,
        dt_bias_arg,
        conv_state_arg,
        recurrent_state_arg,
        decoder_segment_ids,
    )
    if use_head_sharded_cp and next_conv_state is not None and next_conv_state.ndim == 4:
      n_cs_q = next_conv_state[..., : layer.head_k_dim].reshape(batch, conv_kernel_size - 1, -1)
      n_cs_k = next_conv_state[..., layer.head_k_dim : 2 * layer.head_k_dim].reshape(
          batch, conv_kernel_size - 1, -1
      )
      n_cs_v = next_conv_state[..., 2 * layer.head_k_dim :].reshape(batch, conv_kernel_size - 1, -1)
      next_conv_state = jnp.concatenate([n_cs_q, n_cs_k, n_cs_v], axis=-1)
  else:
    core_attn_out, (next_conv_state, next_recurrent_state) = gdn_decoupled_conv1d(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=layer.conv1d.kernel.value,
        conv_bias=conv_bias_arg,
        a_log=layer.A_log[...],
        dt_bias=layer.dt_bias[...],
        conv_state=conv_state_arg,
        recurrent_state=recurrent_state_arg,
        num_k_heads=layer.num_k_heads,
        num_v_heads=layer.num_v_heads,
        head_k_dim=layer.head_k_dim,
        head_v_dim=layer.head_v_dim,
        conv_kernel_size=cfg.gdn_conv_kernel_dim,
        chunk_size=cfg.gdn_chunk_size,
        use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn,
        compute_dtype=state_dtype,
        segment_ids=decoder_segment_ids,
    )

  core_attn_out = checkpoint_name(core_attn_out, "gdn_core_attn_out")

  if model_mode != MODEL_MODE_TRAIN and active_cache is not None and orig_cache_batch is not None:
    assert next_conv_state is not None
    assert next_recurrent_state is not None
    if next_conv_state.shape[0] != orig_cache_batch:
      if next_conv_state.shape[0] == 1:
        next_conv_state = jnp.broadcast_to(next_conv_state, (orig_cache_batch,) + next_conv_state.shape[1:])
        next_recurrent_state = jnp.broadcast_to(
            next_recurrent_state, (orig_cache_batch,) + next_recurrent_state.shape[1:]
        )
      elif next_conv_state.shape[0] < orig_cache_batch:
        pad_amt = orig_cache_batch - next_conv_state.shape[0]
        next_conv_state = jnp.pad(next_conv_state, ((0, pad_amt), (0, 0), (0, 0)))
        next_recurrent_state = jnp.pad(next_recurrent_state, ((0, pad_amt), (0, 0), (0, 0), (0, 0)))
      else:
        next_conv_state = next_conv_state[:orig_cache_batch]
        next_recurrent_state = next_recurrent_state[:orig_cache_batch]

  return core_attn_out, next_conv_state, next_recurrent_state

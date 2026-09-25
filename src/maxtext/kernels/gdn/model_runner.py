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
import numpy as np
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
from . import compute_conv1d as gdn_compute_conv1d
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
      ax for ax in gdn_context_axes_fn(config) if mesh is not None and ax in mesh.axis_names and mesh.shape[ax] > 1
  )
  cp_axis_for_pspec = cp_axes_active[0] if len(cp_axes_active) == 1 else (cp_axes_active if cp_axes_active else None)
  cp_len = LENGTH if cp_axes_active else None
  return logical_rules, cp_axes_active, cp_axis_for_pspec, cp_len


def _compute_segmented_conv1d_core(
    qkv: Array,
    conv_weight_3d: Array,
    conv_bias: None | Array,
    seg_2d: Array,
    kernel_size: int,
    conv_state: None | Array,
    conv_halo_seg: None | Array,
    out_dtype: DType,
) -> Array:
  """Core per-tap same-document depthwise Conv1D keeping conv_input in qkv.dtype."""
  batch, seq_len, _ = qkv.shape
  seg_pos = jnp.maximum(seg_2d.astype(jnp.int32), 0)
  valid_mask = (seg_pos > 0)[:, :, None].astype(jnp.float32)

  halo_len = kernel_size - 1
  if conv_state is not None:
    conv_input = jnp.concatenate([conv_state.astype(qkv.dtype), qkv], axis=1)
  else:
    conv_input = jnp.pad(qkv, ((0, 0), (halo_len, 0), (0, 0)))

  if conv_halo_seg is not None:
    halo_pos = jnp.maximum(conv_halo_seg.astype(jnp.int32).reshape(batch, halo_len), 0)
    full_seg = jnp.concatenate([halo_pos, seg_pos], axis=1)
  else:
    full_seg = jnp.pad(seg_pos, ((0, 0), (halo_len, 0)))

  conv_out = sum(
      (
          conv_input[:, k : k + seq_len, :].astype(jnp.float32)
          * ((seg_pos > 0) & (full_seg[:, k : k + seq_len] == seg_pos))[:, :, None].astype(jnp.float32)
      )
      * conv_weight_3d[k, 0, :]
      for k in range(kernel_size)
  )
  if conv_bias is not None:
    conv_out = conv_out + conv_bias.astype(jnp.float32)
  conv_out = conv_out * valid_mask
  return conv_out.astype(out_dtype)


def segmented_causal_depthwise_conv1d(
    qkv: Array,
    conv_weight: Array,
    conv_bias: None | Array = None,
    segment_ids: None | Array | DType = None,
    kernel_size: int | None = None,
    conv_state: None | Array = None,
    conv_halo_seg: None | Array = None,
    dtype: DType | None = None,
) -> Array:
  """Causal depthwise Conv1D with per-tap same-document masking and zero output on padding.

  Supports both the full keyword signature:
    `segmented_causal_depthwise_conv1d(qkv, conv_weight, conv_bias, segment_ids, kernel_size, ...)`
  and the 4-arg positional signature:
    `segmented_causal_depthwise_conv1d(x, kernel, segment_ids, dtype=...)`.
  """
  if conv_bias is not None and (segment_ids is None or not isinstance(segment_ids, (jax.Array, np.ndarray))):
    if segment_ids is not None:
      dtype = segment_ids
    segment_ids = conv_bias
    conv_bias = None
  assert segment_ids is not None, "segment_ids must be provided to segmented_causal_depthwise_conv1d"
  if kernel_size is None:
    kernel_size = int(conv_weight.shape[0])
  out_dtype = dtype if dtype is not None else qkv.dtype

  batch, seq_len, _ = qkv.shape
  seg_2d = jnp.broadcast_to(segment_ids, (batch, seq_len))
  if conv_halo_seg is None:
    # Raw IDs: canonicalize on the full sequence (before any sequence sharding
    # below), and let a caller conv_state continue the first document.
    seg_2d = gdn_compute_conv1d.canonicalize_segment_ids(seg_2d)
    if conv_state is not None:
      conv_halo_seg, _ = gdn_compute_conv1d.initial_state_segment_metadata(seg_2d, kernel_size)
  if conv_weight.ndim == 3:
    conv_weight_3d = conv_weight.astype(jnp.float32)
  else:
    conv_weight_3d = conv_weight[:, None, :].astype(jnp.float32)

  qkv_sharding = getattr(jax.typeof(qkv), "sharding", None)
  if qkv_sharding is not None and getattr(qkv_sharding, "spec", None) is not None:
    qkv_spec = qkv_sharding.spec
    mesh = getattr(qkv_sharding, "mesh", None)
    if len(qkv_spec) >= 3 and any(ax is not None for ax in qkv_spec):
      seg_pspec = P(qkv_spec[0], qkv_spec[1])
      cw_pspec = P(None, None, qkv_spec[2])
      seg_2d = jax.sharding.reshard(seg_2d, seg_pspec)
      conv_weight_3d = jax.sharding.reshard(conv_weight_3d, cw_pspec)
      if conv_bias is not None:
        cb_pspec = P(qkv_spec[2])
        conv_bias = jax.sharding.reshard(conv_bias, cb_pspec)
      else:
        cb_pspec = P()
      if conv_halo_seg is not None:
        conv_halo_seg = jax.sharding.reshard(conv_halo_seg, P(qkv_spec[0], None))

      if qkv_spec[1] is not None and mesh is not None and conv_state is None and conv_halo_seg is None:
        cp_ax = qkv_spec[1]
        halo_len = kernel_size - 1

        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(
                P(qkv_spec[0], qkv_spec[1], qkv_spec[2]),
                cw_pspec,
                cb_pspec,
                seg_pspec,
            ),
            out_specs=P(qkv_spec[0], qkv_spec[1], qkv_spec[2]),
            check_vma=False,
        )
        def _shard_mapped_conv(qkv_loc, cw_loc, cb_loc, seg_loc):
          n_cp = jax.lax.psum(1, axis_name=cp_ax)
          cp_idx = jax.lax.axis_index(cp_ax)
          perm = [(i, (i + 1) % n_cp) for i in range(n_cp)]
          halo_qkv = jax.lax.ppermute(qkv_loc[:, -halo_len:, :], axis_name=cp_ax, perm=perm)
          halo_seg = jax.lax.ppermute(seg_loc[:, -halo_len:], axis_name=cp_ax, perm=perm)
          halo_qkv = jnp.where(cp_idx == 0, jnp.zeros_like(halo_qkv), halo_qkv)
          halo_seg = jnp.where(cp_idx == 0, jnp.zeros_like(halo_seg), halo_seg)
          return _compute_segmented_conv1d_core(
              qkv_loc,
              cw_loc,
              cb_loc,
              seg_loc,
              kernel_size,
              halo_qkv,
              halo_seg,
              out_dtype,
          )

        return _shard_mapped_conv(qkv, conv_weight_3d, conv_bias, seg_2d)

  return _compute_segmented_conv1d_core(
      qkv,
      conv_weight_3d,
      conv_bias,
      seg_2d,
      kernel_size,
      conv_state,
      conv_halo_seg,
      out_dtype,
  )


def segment_next_conv_state(
    conv_state: Array,
    qkv: Array,
    decoder_segment_ids: Array,
    kernel_size: int,
    sequence_packing: bool,
) -> Array:
  """Returns the next Conv1D state of a prefill call: the window ending at the last valid token.

  The caller `conv_state` continues the first document. With
  `sequence_packing`, the window is masked to the document of the last valid
  token. Otherwise every non-padding token is one document. A call without
  valid tokens returns `conv_state` unchanged.

  Args:
    conv_state: [batch, kernel_size - 1, dim] caller conv state.
    qkv: [batch, seq, dim] pre-conv activations (padding already zeroed).
    decoder_segment_ids: [batch, seq] segment IDs (0 = padding).
    kernel_size: Conv1D kernel size.
    sequence_packing: Whether GDN sequence packing is enabled.
  """
  batch, seq_len, _ = qkv.shape
  seg_2d = jnp.broadcast_to(decoder_segment_ids, (batch, seq_len))
  if sequence_packing:
    seg_2d = gdn_compute_conv1d.canonicalize_segment_ids(seg_2d)
  else:
    seg_2d = (seg_2d != 0).astype(jnp.int32)
  conv_halo_seg, _ = gdn_compute_conv1d.initial_state_segment_metadata(seg_2d, kernel_size)
  out_dtype = jnp.result_type(conv_state.dtype, qkv.dtype)
  return gdn_compute_conv1d.extract_segment_conv_state_split(
      conv_state.astype(out_dtype), qkv.astype(out_dtype), seg_2d, kernel_size, conv_halo_seg
  )


def prepare_jax_gdn_segment_masks(
    segment_ids: Array,
    q_c: Array,
    k_c: Array,
    v_c: Array,
    g_c: Array,
    beta_c: Array,
    *,
    batch_size: int,
    seq_len: int,
    num_chunks: int,
    chunk_size: int,
    cp_axis: None | str | tuple[str, ...] = None,
    init_seg: None | Array = None,
    has_initial_state: bool = False,
):
  """Prepares segment-aware masks and segment-local cumulative decay for Pure-JAX GDN.

  Without `init_seg`, raw `segment_ids` are canonicalized (globally under
  `cp_axis`), and `has_initial_state` makes the initial recurrent state continue
  the first document. With `init_seg`, `segment_ids` must already be canonical.
  """
  from . import compute_conv1d as gdn_conv1d  # pylint: disable=import-outside-toplevel,g-import-not-at-top
  from .gdn_bwd import cp_gdn as bwd_cp_gdn  # pylint: disable=import-outside-toplevel,g-import-not-at-top

  seg_2d = jnp.broadcast_to(segment_ids, (batch_size, seq_len))
  if cp_axis is not None and init_seg is None:
    s_enc, _, init_seg = bwd_cp_gdn.gather_cp_segment_metadata(seg_2d, cp_axis, 4, has_initial_state=has_initial_state)
  else:
    if init_seg is None:
      seg_2d = gdn_conv1d.canonicalize_segment_ids(seg_2d)
      if has_initial_state:
        init_seg = jnp.ones((batch_size,), dtype=jnp.float32)
    s_enc = gdn_conv1d.encode_segment_ids(seg_2d, init_seg=init_seg)

  pad_len = num_chunks * chunk_size - seq_len
  if pad_len > 0:
    tail_active = -jnp.abs(s_enc[:, -1:])
    s_enc = jnp.concatenate([s_enc, jnp.broadcast_to(tail_active, (batch_size, pad_len))], axis=1)

  seg_c = s_enc.reshape(batch_size, num_chunks, 1, chunk_size)
  valid_c = seg_c > 0.5
  active_c = jnp.abs(seg_c)
  active_end = active_c[..., -1]
  if init_seg is not None:
    init_active = jnp.abs(init_seg.astype(jnp.float32).reshape(batch_size, 1, 1))
  else:
    init_active = jnp.zeros((batch_size, 1, 1), dtype=jnp.float32)
  seg_prev = jnp.concatenate([init_active, active_end[:, :-1, :]], axis=1)[..., None]

  q_c = jnp.where(valid_c[..., None], q_c, 0.0)
  k_c = jnp.where(valid_c[..., None], k_c, 0.0)
  v_c = jnp.where(valid_c[..., None], v_c, 0.0)
  g_c = jnp.where(valid_c, g_c, 0.0)
  beta_c = jnp.where(valid_c, beta_c, 0.0)

  mask_tril_f32 = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
  same_active = (jnp.abs(active_c[..., :, None] - active_c[..., None, :]) < 0.5) & (active_c[..., :, None] > 0.5)
  mask_cumsum = mask_tril_f32 * same_active.astype(jnp.float32)
  g_cumsum = jnp.einsum(
      "bnhij,bnhj->bnhi",
      mask_cumsum,
      g_c,
      precision=jax.lax.Precision.HIGHEST,
  )

  same_valid = (jnp.abs(seg_c[..., :, None] - seg_c[..., None, :]) < 0.5) & valid_c[..., :, None]
  mask_strict = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1) & same_valid
  mask_causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0) & same_valid
  active_last = active_c[..., -1:]
  m_in = ((jnp.abs(seg_c - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)
  m_out = ((jnp.abs(seg_c - active_last) < 0.5) & (active_last > 0.5)).astype(jnp.float32)
  m_keep = ((jnp.abs(active_last - seg_prev) < 0.5) & (seg_prev > 0.5)).astype(jnp.float32)

  return (
      q_c,
      k_c,
      v_c,
      g_c,
      beta_c,
      g_cumsum,
      mask_strict,
      mask_causal,
      m_in,
      m_out,
      m_keep,
      valid_c,
  )


def run_jax_gdn_delta_rule(
    *,
    layer: Any,
    query: Array,
    key: Array,
    value: Array,
    g: Array,
    beta: Array,
    recurrent_state: None | Array,
    state_sharding: jax.sharding.NamedSharding | None,
    packed_segment_ids: None | Array,
    get_logical_axis_rules_fn: Any,
    gdn_context_axes_fn: Any,
    delta_rule_fn: Any,
) -> tuple[Array, None | Array]:
  """Executes Pure-JAX GDN delta rule inside shard_map with head-CP, seq-CP, and sequence packing."""
  cfg = layer.config
  batch = query.shape[0]
  logical_rules, cp_axes_active, cp_axis_for_pspec, cp_len = get_gdn_kernel_cp_sharding_info(
      cfg, layer.mesh, get_logical_axis_rules_fn, gdn_context_axes_fn
  )
  has_initial_state = recurrent_state is not None
  recurrent_state_arg = (
      recurrent_state
      if recurrent_state is not None
      else jnp.zeros(
          (batch, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim),
          dtype=cfg.dtype,
          out_sharding=state_sharding,
      )
  )
  use_head_cp_jax = bool(cp_axes_active) and getattr(cfg, "gdn_cp_mode", "auto") == "head"
  if use_head_cp_jax:
    mesh_batch = logical_to_mesh_axes((KV_BATCH,), mesh=layer.mesh, rules=logical_rules)[0]
    qkv_pspec = P(mesh_batch, None, cp_axis_for_pspec, None)
    g_beta_pspec = P(mesh_batch, None, cp_axis_for_pspec)
    state_pspec = P(mesh_batch, cp_axis_for_pspec, None, None)
    cp_axes_for_scan = None
  else:
    qkv_pspec = logical_to_mesh_axes((KV_BATCH, cp_len, None, None), mesh=layer.mesh, rules=logical_rules)
    g_beta_pspec = logical_to_mesh_axes((KV_BATCH, cp_len, None), mesh=layer.mesh, rules=logical_rules)
    state_pspec = logical_to_mesh_axes((KV_BATCH, None, None, None), mesh=layer.mesh, rules=logical_rules)
    if not cp_axes_active:
      from maxtext.common.common_types import KV_HEAD  # pylint: disable=import-outside-toplevel,g-import-not-at-top

      qkv_pspec = logical_to_mesh_axes((KV_BATCH, cp_len, KV_HEAD, None), mesh=layer.mesh, rules=logical_rules)
      g_beta_pspec = logical_to_mesh_axes((KV_BATCH, cp_len, KV_HEAD), mesh=layer.mesh, rules=logical_rules)
      state_pspec = logical_to_mesh_axes((KV_BATCH, KV_HEAD, None, None), mesh=layer.mesh, rules=logical_rules)
    else:
      from maxtext.common.common_types import KV_HEAD  # pylint: disable=import-outside-toplevel,g-import-not-at-top

      qkv_pspec = logical_to_mesh_axes((KV_BATCH, cp_len, KV_HEAD, None), mesh=layer.mesh, rules=logical_rules)
      g_beta_pspec = logical_to_mesh_axes((KV_BATCH, cp_len, KV_HEAD), mesh=layer.mesh, rules=logical_rules)
      state_pspec = logical_to_mesh_axes((KV_BATCH, KV_HEAD, None, None), mesh=layer.mesh, rules=logical_rules)
      if qkv_pspec[1] is None:
        qkv_pspec = P(qkv_pspec[0], cp_axis_for_pspec, *qkv_pspec[2:])
      if g_beta_pspec[1] is None:
        g_beta_pspec = P(g_beta_pspec[0], cp_axis_for_pspec, *g_beta_pspec[2:])
    cp_axes_for_scan = cp_axes_active or None

  qkv_pspec = remove_incompatible_mesh_axes_from_partition_spec(
      qkv_pspec,
      query.shape,
      layer.mesh,
      dims=(0,),
      allow_remove_axes=True,
  )
  g_beta_pspec = remove_incompatible_mesh_axes_from_partition_spec(
      g_beta_pspec,
      g.shape,
      layer.mesh,
      dims=(0,),
      allow_remove_axes=True,
  )
  state_pspec = remove_incompatible_mesh_axes_from_partition_spec(
      state_pspec,
      recurrent_state_arg.shape,
      layer.mesh,
      dims=(0,),
      allow_remove_axes=True,
  )
  if packed_segment_ids is not None:
    seg_pspec = remove_incompatible_mesh_axes_from_partition_spec(
        P(qkv_pspec[0], qkv_pspec[1]),
        packed_segment_ids.shape,
        layer.mesh,
        dims=(0,),
        allow_remove_axes=True,
    )
  else:
    seg_pspec = P()

  if cfg.shard_mode == ShardMode.EXPLICIT:
    query = jax.sharding.reshard(query, qkv_pspec)
    key = jax.sharding.reshard(key, qkv_pspec)
    value = jax.sharding.reshard(value, qkv_pspec)
    g = jax.sharding.reshard(g, g_beta_pspec)
    beta = jax.sharding.reshard(beta, g_beta_pspec)
    recurrent_state_arg = jax.sharding.reshard(recurrent_state_arg, state_pspec)
    if packed_segment_ids is not None:
      packed_segment_ids = jax.sharding.reshard(packed_segment_ids, seg_pspec)

  @functools.partial(
      jax.shard_map,
      mesh=layer.mesh,
      in_specs=(
          qkv_pspec,
          qkv_pspec,
          qkv_pspec,
          g_beta_pspec,
          g_beta_pspec,
          state_pspec,
          seg_pspec,
      ),
      out_specs=(
          qkv_pspec,
          state_pspec,
      ),
      check_vma=False,
  )
  def shard_mapped_delta_rule(q, k, v, g_val, beta_val, init_h, seg_val):
    return delta_rule_fn(
        query=q,
        key=k,
        value=v,
        g=g_val,
        beta=beta_val,
        chunk_size=cfg.gdn_chunk_size,
        initial_state=init_h,
        use_qk_norm_in_gdn=cfg.use_qk_norm_in_gdn,
        compute_dtype=cfg.dtype,
        cp_axis=cp_axes_for_scan,
        segment_ids=seg_val,
        has_initial_state=has_initial_state,
    )

  return shard_mapped_delta_rule(query, key, value, g, beta, recurrent_state_arg, packed_segment_ids)


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
      ax for ax in gdn_context_axes_fn(cfg) if layer.mesh and ax in layer.mesh.axis_names and layer.mesh.shape[ax] > 1
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
          f"GDN head-sharded CP requires num_k_heads ({layer.num_k_heads}) to" f" be divisible by cp_size ({cp_size})."
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

  use_seq_sharded_cp = cp_size > 1 and model_mode != MODEL_MODE_AUTOREGRESSIVE and not use_head_sharded_cp

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
    decoder_segment_ids = jnp.broadcast_to(decoder_segment_ids, (batch, seq_len))
    if cfg.shard_mode == ShardMode.EXPLICIT and layer.mesh is not None:
      qkv_sharding = getattr(jax.typeof(qkv), "sharding", None)
      if qkv_sharding is not None and getattr(qkv_sharding, "spec", None) is not None:
        qkv_spec = qkv_sharding.spec
        if len(qkv_spec) >= 2:
          decoder_segment_ids = jax.sharding.reshard(
              decoder_segment_ids,
              jax.sharding.NamedSharding(layer.mesh, P(qkv_spec[0], qkv_spec[1])),
          )
    mask = decoder_segment_ids != 0
    qkv = jnp.where(mask.reshape(mask.shape + (1,) * (qkv.ndim - mask.ndim)), qkv, 0.0)
    a = jnp.where(mask[..., None], a, jnp.asarray(-1e4, dtype=a.dtype))
    b = jnp.where(mask[..., None], b, jnp.asarray(-1e4, dtype=b.dtype))
    if not getattr(cfg, "enable_gdn_sequence_packing", False):
      decoder_segment_ids = None

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
        cs_q = conv_state[..., : layer.key_dim].reshape(batch, cfg.gdn_conv_kernel_dim - 1, num_groups, layer.head_k_dim)
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

    if decoder_segment_ids is not None:
      seg_pspec = remove_incompatible_mesh_axes_from_partition_spec(
          P(qkv_pspec[0], qkv_pspec[1]),
          decoder_segment_ids.shape,
          layer.mesh,
          dims=(0,),
          allow_remove_axes=True,
      )
    else:
      seg_pspec = P()
    if cfg.shard_mode == ShardMode.EXPLICIT:
      qkv = jax.sharding.reshard(qkv, jax.sharding.NamedSharding(layer.mesh, qkv_pspec))
      b = jax.sharding.reshard(b, jax.sharding.NamedSharding(layer.mesh, b_a_pspec))
      a = jax.sharding.reshard(a, jax.sharding.NamedSharding(layer.mesh, b_a_pspec))
      conv_state_arg = jax.sharding.reshard(conv_state_arg, jax.sharding.NamedSharding(layer.mesh, conv_state_pspec))
      recurrent_state_arg = jax.sharding.reshard(
          recurrent_state_arg, jax.sharding.NamedSharding(layer.mesh, recurrent_state_pspec)
      )
      if decoder_segment_ids is not None:
        decoder_segment_ids = jax.sharding.reshard(decoder_segment_ids, jax.sharding.NamedSharding(layer.mesh, seg_pspec))

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
    def shard_mapped_gdn(  # pylint: disable=too-many-positional-arguments
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
          cs_k = cs_val[..., layer.head_k_dim : 2 * layer.head_k_dim].reshape(b_sz, cfg.gdn_conv_kernel_dim - 1, -1)
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
          n_cs_q = next_cs[..., :cs_k_dim].reshape(b_sz, cfg.gdn_conv_kernel_dim - 1, local_num_k_heads, layer.head_k_dim)
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
      n_cs_k = next_conv_state[..., layer.head_k_dim : 2 * layer.head_k_dim].reshape(batch, conv_kernel_size - 1, -1)
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

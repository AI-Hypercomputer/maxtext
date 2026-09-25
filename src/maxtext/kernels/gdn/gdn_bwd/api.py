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

"""Model-facing API and custom VJP boundary for GDN backward kernel."""

import functools
from typing import Any, Optional, Tuple

import jax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

from .. import compute_conv1d as local_compute_conv1d
from .. import wrapper as local_gdn_wrapper
from . import cp_gdn
from .compute_conv1d_bwd import conv1d_silu_bwd
from .compute_conv1d_bwd import conv1d_silu_fwd
from .jax_compute_gdn_states import _compute_forward_conv_and_states
from .jax_compute_gdn_states import pure_jax_decoupled_conv1d_gdn
from .pallas_mosaic_tpu_bwd import pallas_gdn_bwd_kernel


def decoupled_conv1d_gdn_bwd_kernel(
    pre_conv_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    do: jax.Array,
    chunk_states: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array] = None,
    t_inv: Optional[jax.Array] = None,
    qkv: Optional[jax.Array] = None,
    seq_lens: Optional[jax.Array] = None,
    *,
    conv_state: Optional[jax.Array] = None,
    num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    kernel_size: int = 4,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
    vmem_limit_mb: Optional[int] = None,
    head_tile: Optional[int] = None,
    segment_ids: Optional[jax.Array] = None,
    conv_halo_seg: Optional[jax.Array] = None,
    init_seg: Optional[jax.Array] = None,
    interpret: bool | pltpu.InterpretParams | None = None,
) -> Tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    Optional[jax.Array],
    jax.Array,
    jax.Array,
]:
  """Decoupled Conv1D + GDN backward combining Pallas GDN bwd and JAX Conv1D bwd."""
  del seq_lens, qkv
  conv_out, qkv_conv = conv1d_silu_fwd(
      qkv=pre_conv_qkv,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      kernel_size=kernel_size,
      conv_state=conv_state,
      segment_ids=segment_ids,
      conv_halo_seg=conv_halo_seg,
  )

  # pylint: disable=unbalanced-tuple-unpacking
  dy_conv, d_b, d_a, d_a_log, d_dt_bias = pallas_gdn_bwd_kernel(
      qkv_conv=qkv_conv,
      b=b,
      a=a,
      a_log=a_log,
      dt_bias=dt_bias,
      do=do,
      chunk_states=chunk_states,
      t_inv=t_inv,
      num_v_heads=num_v_heads,
      kq_head_dim=kq_head_dim,
      v_head_dim=v_head_dim,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      vmem_limit_mb=vmem_limit_mb,
      head_tile=head_tile,
      segment_ids=segment_ids,
      init_seg=init_seg,
      interpret=interpret,
  )

  d_pre_conv_qkv, d_conv_weight, d_conv_bias = conv1d_silu_bwd(
      qkv=pre_conv_qkv,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      dy=dy_conv,
      kernel_size=kernel_size,
      conv_out=conv_out,
      conv_state=conv_state,
      segment_ids=segment_ids,
      conv_halo_seg=conv_halo_seg,
  )
  # pylint: enable=unbalanced-tuple-unpacking

  return (
      d_pre_conv_qkv,
      d_b,
      d_a,
      d_conv_weight,
      d_conv_bias,
      d_a_log,
      d_dt_bias,
  )


def _run_local_gdn_decoupled_fwd(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    conv_state: Optional[jax.Array],
    recurrent_state: Optional[jax.Array],
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
    segment_ids: Optional[jax.Array] = None,
    conv_halo_seg: Optional[jax.Array] = None,
    init_seg: Optional[jax.Array] = None,
) -> Tuple[
    Tuple[jax.Array, Tuple[jax.Array, jax.Array]],
    Optional[jax.Array],
    Optional[jax.Array],
]:
  """Runs local GDN forward pass on TPU returning (t_inv, chunk_states), or pure JAX on CPU."""
  if (
      jax.extend.backend.get_backend().platform == "cpu"
      or head_k_dim % 128 != 0
      or head_v_dim % 128 != 0
      or chunk_size != 64
      or qkv.shape[1] % chunk_size != 0
  ):
    out, states = pure_jax_decoupled_conv1d_gdn(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        compute_dtype=compute_dtype,
        segment_ids=segment_ids,
        conv_halo_seg=conv_halo_seg,
        init_seg=init_seg,
    )
    _, chunk_states, t_inv = _compute_forward_conv_and_states(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        conv_state=conv_state,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        compute_dtype=compute_dtype,
        segment_ids=segment_ids,
        conv_halo_seg=conv_halo_seg,
        init_seg=init_seg,
    )
    return (out, states), t_inv, chunk_states

  batch_size, seq_len, dim_size = qkv.shape
  num_seqs = batch_size
  num_chunks = seq_len // chunk_size

  qkv_flat = qkv.reshape(-1, dim_size)
  b_flat = b.reshape(-1, b.shape[-1])
  a_flat = a.reshape(-1, a.shape[-1])
  tokamax_conv_weight = jnp.swapaxes(conv_weight, 0, 2)

  has_init = (conv_state is not None) or (recurrent_state is not None)
  query_start_loc = jnp.arange(0, (num_seqs + 1) * seq_len, seq_len, dtype=jnp.int32)
  state_indices = jnp.arange(1, num_seqs + 1, dtype=jnp.int32)
  seq_lens = jnp.full((num_seqs,), seq_len + (1 if has_init else 0), dtype=jnp.int32)
  distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

  if conv_state is None:
    tokamax_conv_state = jnp.zeros((num_seqs + 1, conv_kernel_size - 1, dim_size), dtype=qkv.dtype)
  elif conv_state.shape[0] == num_seqs:
    tokamax_conv_state = jnp.pad(conv_state, ((1, 0), (0, 0), (0, 0)))
  else:
    tokamax_conv_state = conv_state

  if recurrent_state is None:
    tokamax_recurrent_state = jnp.zeros((num_seqs + 1, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32)
  elif recurrent_state.shape[0] == num_seqs:
    tokamax_recurrent_state = jnp.pad(recurrent_state.astype(jnp.float32), ((1, 0), (0, 0), (0, 0), (0, 0)))
  else:
    tokamax_recurrent_state = recurrent_state.astype(jnp.float32)

  (
      core_attn_out_flat,
      (new_conv_state, new_recurrent_state),
      t_inv_raw,
      chunk_states_raw,
  ) = local_gdn_wrapper.fused_conv1d_gdn(
      qkv=qkv_flat,
      b=b_flat,
      a=a_flat,
      conv_state=tokamax_conv_state,
      recurrent_state=tokamax_recurrent_state,
      conv_weight=tokamax_conv_weight,
      conv_bias=conv_bias,
      a_log=a_log,
      dt_bias=dt_bias,
      query_start_loc=query_start_loc,
      state_indices=state_indices,
      distribution=distribution,
      seq_lens=seq_lens,
      n_kq=num_k_heads,
      n_v=num_v_heads,
      d_k=head_k_dim,
      d_v=head_v_dim,
      kernel_size=conv_kernel_size,
      compute_precision=jnp.dtype(compute_dtype),
      mixed_tile_size=chunk_size,
      is_prefill_only=True,
      segment_ids=segment_ids,
      conv_halo_seg=conv_halo_seg,
      init_seg=init_seg,
  )

  core_attn_out = core_attn_out_flat.reshape(batch_size, seq_len, num_v_heads, head_v_dim)
  t_inv = t_inv_raw.astype(jnp.float32).reshape(batch_size, num_chunks, num_v_heads, chunk_size, chunk_size)
  chunk_states = chunk_states_raw.astype(jnp.float32).reshape(batch_size, num_chunks, num_v_heads, head_k_dim, head_v_dim)
  return (
      (
          core_attn_out.astype(qkv.dtype),
          (
              new_conv_state[1:].astype(qkv.dtype),
              new_recurrent_state[1:].astype(jnp.float32),
          ),
      ),
      t_inv,
      chunk_states,
  )


def _is_cp_active(cp_axis_name: str | tuple[str, ...] | None) -> bool:
  if cp_axis_name is None:
    return False
  try:
    return jax.lax.axis_size(cp_axis_name) > 1
  except (NameError, ValueError):
    return False


def _local_segment_metadata(
    segment_ids: Optional[jax.Array],
    has_initial_state: bool,
    conv_kernel_size: int,
) -> Tuple[Optional[jax.Array], Optional[jax.Array], Optional[jax.Array]]:
  """Returns canonical (segment_ids, conv_halo_seg, init_seg) for the non-CP path.

  Canonicalization happens once here, on the full sequence. With caller
  conv/recurrent states, the states continue the first document.
  """
  if segment_ids is None:
    return None, None, None
  segment_ids = local_compute_conv1d.canonicalize_segment_ids(segment_ids)
  if not has_initial_state:
    return segment_ids, None, None
  conv_halo_seg, init_seg = local_compute_conv1d.initial_state_segment_metadata(segment_ids, conv_kernel_size)
  return segment_ids, conv_halo_seg, init_seg


def _run_cp_gdn_decoupled_fwd_impl(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    conv_state: Optional[jax.Array],
    recurrent_state: Optional[jax.Array],
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
    cp_axis_name: str | tuple[str, ...],
    segment_ids: Optional[jax.Array] = None,
    seg_metadata: Optional[Tuple[Optional[jax.Array], Optional[jax.Array], Optional[jax.Array]]] = None,
):
  """Runs 2-pass sequence-sharded CP forward for GDN."""
  batch_size = qkv.shape[0]
  if seg_metadata is not None:
    s_enc_local, conv_halo_seg, init_seg = seg_metadata
  elif segment_ids is not None:
    s_enc_local, conv_halo_seg, init_seg = cp_gdn.gather_cp_segment_metadata(
        segment_ids,
        cp_axis_name,
        conv_kernel_size,
        has_initial_state=(conv_state is not None) or (recurrent_state is not None),
    )
  else:
    s_enc_local, conv_halo_seg, init_seg = None, None, None

  conv_halo = cp_gdn.halo_exchange_for_conv(
      qkv=qkv,
      init_conv_state=conv_state,
      kernel_size=conv_kernel_size,
      cp_axis=cp_axis_name,
      segment_ids=s_enc_local,
  )
  zero_rs = jnp.zeros((batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32)

  # Pass 1: Local GDN with zero initial recurrent state -> yields t_inv and S_ext_local
  (_, states_pass1), t_inv, _ = _run_local_gdn_decoupled_fwd(
      qkv,
      b,
      a,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      conv_halo,
      zero_rs,
      num_k_heads=num_k_heads,
      num_v_heads=num_v_heads,
      head_k_dim=head_k_dim,
      head_v_dim=head_v_dim,
      conv_kernel_size=conv_kernel_size,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      compute_dtype=compute_dtype,
      segment_ids=s_enc_local,
      conv_halo_seg=conv_halo_seg,
      init_seg=init_seg,
  )

  # Convolve only the K channel slice (1/6th of qkv) needed by compose_local_from_t_inv
  q_size = num_k_heads * head_k_dim
  k_size = num_k_heads * head_k_dim
  _, k_conv = conv1d_silu_fwd(
      qkv=qkv[:, :, q_size : q_size + k_size],
      conv_weight=conv_weight[:, :, q_size : q_size + k_size],
      conv_bias=(conv_bias[q_size : q_size + k_size] if conv_bias is not None else None),
      kernel_size=conv_kernel_size,
      conv_state=conv_halo[:, :, q_size : q_size + k_size],
      segment_ids=s_enc_local,
      conv_halo_seg=conv_halo_seg,
  )
  m_local, s_ext_local = cp_gdn.compose_local_from_t_inv(
      qkv_conv=k_conv,
      b=b,
      a=a,
      a_log=a_log,
      dt_bias=dt_bias,
      t_inv=t_inv,
      num_k_heads=num_k_heads,
      num_v_heads=num_v_heads,
      head_k_dim=head_k_dim,
      head_v_dim=head_v_dim,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      s_ext_pass1=states_pass1[1],
      segment_ids=s_enc_local,
      init_seg=init_seg,
  )

  h_init = recurrent_state.astype(jnp.float32) if recurrent_state is not None else zero_rs
  s_in_r, final_rs = cp_gdn.incoming_state(m_local, s_ext_local, h_init, cp_axis_name)

  # Pass 2: Local GDN with true incoming state s_in_r
  (out, (next_cs_local, _)), t_inv_2, chunk_states = _run_local_gdn_decoupled_fwd(
      qkv,
      b,
      a,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      conv_halo,
      s_in_r,
      num_k_heads=num_k_heads,
      num_v_heads=num_v_heads,
      head_k_dim=head_k_dim,
      head_v_dim=head_v_dim,
      conv_kernel_size=conv_kernel_size,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      compute_dtype=compute_dtype,
      segment_ids=s_enc_local,
      conv_halo_seg=conv_halo_seg,
      init_seg=init_seg,
  )
  if s_enc_local is not None:
    # Owner = last rank with a valid local token (rank 0 if none). A rank whose
    # tokens are all padding must not own the state even though its halo is
    # valid: halo_exchange_for_conv zeroes that halo (different document).
    cs_has_valid = jnp.any(s_enc_local > 0, axis=1)
  else:
    cs_has_valid = None
  final_cs = cp_gdn.broadcast_end_conv_state(next_cs_local, cp_axis_name, has_valid=cs_has_valid)
  states = (final_cs.astype(qkv.dtype), final_rs.astype(jnp.float32))
  return (out, states), t_inv_2, chunk_states, conv_halo, s_in_r, m_local


@functools.partial(jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13, 14, 15, 16, 17))
def gdn_decoupled_conv1d(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    conv_state: Optional[jax.Array],
    recurrent_state: Optional[jax.Array],
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
    cp_axis_name: str | tuple[str, ...] | None = None,
    segment_ids: Optional[jax.Array] = None,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
  """Decoupled Conv1D + GDN with Pallas backward pass and optional sequence-sharded CP."""
  if _is_cp_active(cp_axis_name):
    (out, states), _, _, _, _, _ = _run_cp_gdn_decoupled_fwd_impl(
        qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        conv_state,
        recurrent_state,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        compute_dtype=compute_dtype,
        cp_axis_name=cp_axis_name,
        segment_ids=segment_ids,
    )
    return out, states

  segment_ids, conv_halo_seg, init_seg = _local_segment_metadata(
      segment_ids, (conv_state is not None) or (recurrent_state is not None), conv_kernel_size
  )
  (out, states), _, _ = _run_local_gdn_decoupled_fwd(
      qkv,
      b,
      a,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      conv_state,
      recurrent_state,
      num_k_heads=num_k_heads,
      num_v_heads=num_v_heads,
      head_k_dim=head_k_dim,
      head_v_dim=head_v_dim,
      conv_kernel_size=conv_kernel_size,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      compute_dtype=compute_dtype,
      segment_ids=segment_ids,
      conv_halo_seg=conv_halo_seg,
      init_seg=init_seg,
  )
  return out, states


def _unwrap_primal(x):
  return jax.tree.map(lambda v: v.value if hasattr(v, "value") else v, x)


def _unwrap_cotangent(g, like_zero: bool = False):
  if isinstance(g, jax.custom_derivatives.SymbolicZero):
    return jnp.zeros(g.shape, dtype=g.dtype) if like_zero else None
  return g


def _gdn_decoupled_conv1d_fwd(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    conv_state: Optional[jax.Array],
    recurrent_state: Optional[jax.Array],
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
    cp_axis_name: str | tuple[str, ...] | None = None,
    segment_ids: Optional[jax.Array] = None,
):
  """Forward rule for custom_vjp registration of gdn_decoupled_conv1d."""
  (
      qkv,
      b,
      a,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      conv_state,
      recurrent_state,
      segment_ids,
  ) = _unwrap_primal(
      (
          qkv,
          b,
          a,
          conv_weight,
          conv_bias,
          a_log,
          dt_bias,
          conv_state,
          recurrent_state,
          segment_ids,
      )
  )
  qkv = checkpoint_name(qkv, "gdn_fwd_conv")
  qkv = checkpoint_name(qkv, "gdn_conv_out")
  has_initial_state = (conv_state is not None) or (recurrent_state is not None)
  if _is_cp_active(cp_axis_name):
    if segment_ids is not None:
      seg_metadata = cp_gdn.gather_cp_segment_metadata(
          segment_ids, cp_axis_name, conv_kernel_size, has_initial_state=has_initial_state
      )
    else:
      seg_metadata = None
    (out, states), t_inv, chunk_states, conv_halo, s_in_r, m_local = _run_cp_gdn_decoupled_fwd_impl(
        qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        conv_state,
        recurrent_state,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        compute_dtype=compute_dtype,
        cp_axis_name=cp_axis_name,
        segment_ids=segment_ids,
        seg_metadata=seg_metadata,
    )
    out = checkpoint_name(out, "gdn_core_attn_out")
    residuals = (
        checkpoint_name(qkv, "gdn_qkv") if qkv is not None else None,
        checkpoint_name(b, "gdn_b") if b is not None else None,
        checkpoint_name(a, "gdn_a") if a is not None else None,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        checkpoint_name(conv_halo, "gdn_conv_state"),
        checkpoint_name(s_in_r, "gdn_recurrent_state"),
        checkpoint_name(t_inv, "gdn_t_inv") if t_inv is not None else None,
        checkpoint_name(chunk_states, "gdn_chunk_states") if chunk_states is not None else None,
        checkpoint_name(m_local, "gdn_m_local") if m_local is not None else None,
        conv_state is not None,
        recurrent_state is not None,
        seg_metadata if seg_metadata is not None else segment_ids,
    )
    return (out, states), residuals

  segment_ids, conv_halo_seg, init_seg = _local_segment_metadata(segment_ids, has_initial_state, conv_kernel_size)
  (out, states), t_inv, chunk_states = _run_local_gdn_decoupled_fwd(
      qkv,
      b,
      a,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      conv_state,
      recurrent_state,
      num_k_heads=num_k_heads,
      num_v_heads=num_v_heads,
      head_k_dim=head_k_dim,
      head_v_dim=head_v_dim,
      conv_kernel_size=conv_kernel_size,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      compute_dtype=compute_dtype,
      segment_ids=segment_ids,
      conv_halo_seg=conv_halo_seg,
      init_seg=init_seg,
  )
  out = checkpoint_name(out, "gdn_core_attn_out")
  seg_bundle = (segment_ids, conv_halo_seg, init_seg) if segment_ids is not None else None
  residuals = (
      checkpoint_name(qkv, "gdn_qkv") if qkv is not None else None,
      checkpoint_name(b, "gdn_b") if b is not None else None,
      checkpoint_name(a, "gdn_a") if a is not None else None,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      checkpoint_name(conv_state, "gdn_conv_state") if conv_state is not None else None,
      checkpoint_name(recurrent_state, "gdn_recurrent_state") if recurrent_state is not None else None,
      checkpoint_name(t_inv, "gdn_t_inv") if t_inv is not None else None,
      checkpoint_name(chunk_states, "gdn_chunk_states") if chunk_states is not None else None,
      seg_bundle,
  )
  return (out, states), residuals


def _gdn_decoupled_conv1d_bwd(
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
    cp_axis_name: str | tuple[str, ...] | None = None,
    residuals: tuple[Any, ...] | None = None,
    cotangents: tuple[Any, ...] | None = None,
):
  """Backward rule for custom_vjp registration of gdn_decoupled_conv1d."""
  # Support 10-arg positional calls where cp_axis_name was omitted
  if cotangents is None and isinstance(cp_axis_name, tuple) and len(cp_axis_name) >= 10:
    cotangents = residuals
    residuals = cp_axis_name
    cp_axis_name = None

  m_local_fwd = None
  seg_slot = None
  include_segment_ids_grad = False
  if len(residuals) == 15:
    (
        pre_conv_qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        conv_state,
        recurrent_state,
        t_inv_fwd,
        chunk_states,
        m_local_fwd,
        has_user_conv_state,
        has_user_recurrent_state,
        seg_slot,
    ) = residuals
    include_segment_ids_grad = True
  elif len(residuals) == 14:
    (
        pre_conv_qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        conv_state,
        recurrent_state,
        t_inv_fwd,
        chunk_states,
        m_local_fwd,
        has_user_conv_state,
        has_user_recurrent_state,
    ) = residuals
  elif len(residuals) == 12:
    (
        pre_conv_qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        conv_state,
        recurrent_state,
        t_inv_fwd,
        chunk_states,
        seg_slot,
    ) = residuals
    has_user_conv_state = conv_state is not None
    has_user_recurrent_state = recurrent_state is not None
    include_segment_ids_grad = True
  elif len(residuals) == 11:
    (
        pre_conv_qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        conv_state,
        recurrent_state,
        t_inv_fwd,
        chunk_states,
    ) = residuals
    has_user_conv_state = conv_state is not None
    has_user_recurrent_state = recurrent_state is not None
  else:
    (
        pre_conv_qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        conv_state,
        recurrent_state,
        t_inv_fwd,
    ) = residuals
    chunk_states = None
    has_user_conv_state = conv_state is not None
    has_user_recurrent_state = recurrent_state is not None

  use_cp = _is_cp_active(cp_axis_name)
  if isinstance(seg_slot, tuple):
    segment_ids, conv_halo_seg, init_seg = seg_slot
  elif use_cp:
    segment_ids, conv_halo_seg, init_seg = seg_slot, None, None
  else:
    segment_ids, conv_halo_seg, init_seg = _local_segment_metadata(
        seg_slot, bool(has_user_conv_state) or bool(has_user_recurrent_state), conv_kernel_size
    )

  d_out_raw, d_states = cotangents
  d_conv_state_raw, d_recurrent_state_raw = d_states
  d_out = _unwrap_cotangent(d_out_raw, like_zero=True)
  # next_conv_state / next_recurrent_state depend on the inputs even without
  # caller initial states, so their cotangents are always propagated.
  d_conv_state = _unwrap_cotangent(d_conv_state_raw, like_zero=False)
  d_recurrent_state = _unwrap_cotangent(d_recurrent_state_raw, like_zero=False)
  need_dh0 = bool(has_user_recurrent_state)

  if use_cp and segment_ids is not None and conv_halo_seg is None and init_seg is None:
    segment_ids, conv_halo_seg, init_seg = cp_gdn.gather_cp_segment_metadata(
        segment_ids,
        cp_axis_name,
        conv_kernel_size,
        has_initial_state=bool(has_user_conv_state) or bool(has_user_recurrent_state),
    )

  # Recompute forward chunk states and t_inv if not cached in residuals
  conv_out_cached = None
  if chunk_states is None or t_inv_fwd is None:
    qkv_conv, chunk_states_recomputed, t_inv_recomputed = _compute_forward_conv_and_states(
        qkv=pre_conv_qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=recurrent_state,
        conv_state=conv_state,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        compute_dtype=compute_dtype,
        cached_t_inv=t_inv_fwd,
        segment_ids=segment_ids,
        conv_halo_seg=conv_halo_seg,
        init_seg=init_seg,
    )
    if chunk_states is None:
      chunk_states = chunk_states_recomputed
    if t_inv_fwd is None:
      t_inv_fwd = t_inv_recomputed
  else:
    conv_out_cached, qkv_conv = conv1d_silu_fwd(
        qkv=pre_conv_qkv,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        kernel_size=conv_kernel_size,
        conv_state=conv_state,
        segment_ids=segment_ids,
        conv_halo_seg=conv_halo_seg,
    )
  t_inv = t_inv_fwd

  if use_cp:
    # Under jax.shard_map(..., check_vma=False), _shard_map_transpose (shard_map.py:1942)
    # divides cotangents of unmapped out_specs (next_conv_state, next_recurrent_state) by D,
    # and (shard_map.py:1959) emits a single lax.psum across cp_axis_name for cotangents of
    # unmapped in_specs (conv_weight, conv_bias, a_log, dt_bias, conv_state, recurrent_state),
    # avoiding redundant All-Reduce collectives inside the custom_vjp rule.
    dht_final = (
        jax.lax.psum(d_recurrent_state.astype(jnp.float32), cp_axis_name)
        if d_recurrent_state is not None
        else jnp.zeros(
            (pre_conv_qkv.shape[0], num_v_heads, head_k_dim, head_v_dim),
            dtype=jnp.float32,
        )
    )
    d_cs_ext_unscaled = (
        jax.lax.psum(d_conv_state.astype(jnp.float32), cp_axis_name).astype(pre_conv_qkv.dtype)
        if d_conv_state is not None
        else None
    )
    dm_local, ds_ext_local = cp_gdn.compose_bwd_local_from_t_inv(
        qkv_conv=qkv_conv,
        b=b,
        a=a,
        a_log=a_log,
        dt_bias=dt_bias,
        do=d_out,
        t_inv=t_inv,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        m_local_cached=m_local_fwd,
        segment_ids=segment_ids,
        init_seg=init_seg,
    )
    dht_local, _ = cp_gdn.incoming_grad_state(dm_local, ds_ext_local, dht_final, cp_axis_name)
    bwd_out = pallas_gdn_bwd_kernel(
        qkv_conv=qkv_conv,
        b=b,
        a=a,
        a_log=a_log,
        dt_bias=dt_bias,
        do=d_out,
        chunk_states=chunk_states,
        t_inv=t_inv,
        num_v_heads=num_v_heads,
        kq_head_dim=head_k_dim,
        v_head_dim=head_v_dim,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        segment_ids=segment_ids,
        init_seg=init_seg,
        d_recurrent_state=dht_local,
        return_dh0=need_dh0,
    )
    if need_dh0:
      dy_conv, d_b, d_a, d_a_log, d_dt_bias, dh0_local = bwd_out
    else:
      dy_conv, d_b, d_a, d_a_log, d_dt_bias = bwd_out  # pylint: disable=unbalanced-tuple-unpacking
      dh0_local = None

    d_pre_conv_qkv, d_conv_weight, d_conv_bias, d_cs_local = conv1d_silu_bwd(
        qkv=pre_conv_qkv,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        dy=dy_conv,
        kernel_size=conv_kernel_size,
        conv_out=conv_out_cached,
        conv_state=conv_state,
        return_d_conv_state=True,
        segment_ids=segment_ids,
        conv_halo_seg=conv_halo_seg,
    )
    d_cs_ext_for_halo = d_cs_ext_unscaled
    if d_cs_ext_unscaled is not None and segment_ids is not None:
      # With packing, next_conv_state is the window ending at the last valid
      # token of the rank that owns it (see broadcast_end_conv_state). Scatter
      # its cotangent into that rank's tokens / halo instead of the last rank's tail.
      cs_has_valid = jnp.any(segment_ids > 0, axis=1)
      is_owner = cp_gdn.select_end_conv_state_rank(cs_has_valid, cp_axis_name)
      d_cs_owned = jnp.where(is_owner[:, None, None], d_cs_ext_unscaled, jnp.zeros_like(d_cs_ext_unscaled))
      d_pre_conv_qkv, d_cs_local = local_compute_conv1d.extract_segment_conv_state_split_adjoint(
          d_cs_owned,
          d_pre_conv_qkv,
          d_cs_local,
          segment_ids,
          conv_kernel_size,
          conv_halo_seg,
      )
      d_cs_ext_for_halo = None
    d_pre_conv_qkv, _ = cp_gdn.halo_exchange_for_conv_bwd(
        dx=d_pre_conv_qkv,
        d_conv_state=d_cs_local,
        d_conv_state_ext=d_cs_ext_for_halo,
        kernel_size=conv_kernel_size,
        cp_axis=cp_axis_name,
        segment_ids=segment_ids,
    )

    idx = jax.lax.axis_index(cp_axis_name)
    d_init_cs_rank0 = jnp.where(idx == 0, d_cs_local, jnp.zeros_like(d_cs_local))
    d_conv_state_out = d_init_cs_rank0 if has_user_conv_state else None
    if need_dh0 and dh0_local is not None:
      d_recurrent_state_out = jnp.where(idx == 0, dh0_local, jnp.zeros_like(dh0_local))
    else:
      d_recurrent_state_out = None
    out_grads = (
        d_pre_conv_qkv,
        d_b,
        d_a,
        d_conv_weight,
        d_conv_bias,
        d_a_log,
        d_dt_bias,
        d_conv_state_out,
        d_recurrent_state_out,
    )
    if include_segment_ids_grad:
      return out_grads + (None,)
    return out_grads

  bwd_out = pallas_gdn_bwd_kernel(
      qkv_conv=qkv_conv,
      b=b,
      a=a,
      a_log=a_log,
      dt_bias=dt_bias,
      do=d_out,
      chunk_states=chunk_states,
      t_inv=t_inv,
      num_v_heads=num_v_heads,
      kq_head_dim=head_k_dim,
      v_head_dim=head_v_dim,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      segment_ids=segment_ids,
      init_seg=init_seg,
      d_recurrent_state=d_recurrent_state,
      return_dh0=need_dh0,
  )
  # pylint: disable=unbalanced-tuple-unpacking
  if need_dh0:
    dy_conv, d_b, d_a, d_a_log, d_dt_bias, dh0_local = bwd_out
  else:
    dy_conv, d_b, d_a, d_a_log, d_dt_bias = bwd_out
    dh0_local = None

  d_pre_conv_qkv, d_conv_weight, d_conv_bias, d_cs_local = conv1d_silu_bwd(
      qkv=pre_conv_qkv,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      dy=dy_conv,
      kernel_size=conv_kernel_size,
      conv_out=conv_out_cached,
      conv_state=conv_state,
      return_d_conv_state=True,
      segment_ids=segment_ids,
      conv_halo_seg=conv_halo_seg,
  )
  # pylint: enable=unbalanced-tuple-unpacking
  if d_conv_state is not None and segment_ids is not None:
    d_pre_conv_qkv, d_cs_local = local_compute_conv1d.extract_segment_conv_state_split_adjoint(
        d_conv_state,
        d_pre_conv_qkv,
        d_cs_local if has_user_conv_state else None,
        segment_ids,
        conv_kernel_size,
        conv_halo_seg,
    )
  elif d_conv_state is not None and pre_conv_qkv.shape[1] >= conv_kernel_size - 1:
    d_pre_conv_qkv = d_pre_conv_qkv.at[:, -(conv_kernel_size - 1) :, :].add(d_conv_state.astype(d_pre_conv_qkv.dtype))

  d_conv_state_out = d_cs_local if has_user_conv_state else None
  d_recurrent_state_out = dh0_local if need_dh0 else None
  out_grads = (
      d_pre_conv_qkv,
      d_b,
      d_a,
      d_conv_weight,
      d_conv_bias,
      d_a_log,
      d_dt_bias,
      d_conv_state_out,
      d_recurrent_state_out,
  )
  if include_segment_ids_grad:
    return out_grads + (None,)
  return out_grads


gdn_decoupled_conv1d.defvjp(
    _gdn_decoupled_conv1d_fwd,
    _gdn_decoupled_conv1d_bwd,
    symbolic_zeros=False,
)

__all__ = [
    "decoupled_conv1d_gdn_bwd_kernel",
    "gdn_decoupled_conv1d",
    "pallas_gdn_bwd_kernel",
]

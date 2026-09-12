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

"""Pallas emit_pipeline GDN backward kernel on Mosaic TPU."""

import functools
from typing import Any, Optional, Tuple

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

try:
  from maxtext.layers import normalizations
except (ImportError, ModuleNotFoundError):
  try:
    from maxtext.src.maxtext.layers import normalizations
  except (ImportError, ModuleNotFoundError):
    pass

try:
  from maxtext.models.kernels.gdn.gdn_bwd.bwd_memory_ref import make_bwd_block_specs
  from maxtext.models.kernels.gdn.gdn_bwd.runtime_utils import ensure_cpu_interpret_registered
except (ImportError, ModuleNotFoundError):
  try:
    from maxtext.src.maxtext.models.kernels.gdn.gdn_bwd.bwd_memory_ref import make_bwd_block_specs
    from maxtext.src.maxtext.models.kernels.gdn.gdn_bwd.runtime_utils import ensure_cpu_interpret_registered
  except (ImportError, ModuleNotFoundError):
    from .bwd_memory_ref import make_bwd_block_specs
    from .runtime_utils import ensure_cpu_interpret_registered


def _bwd_gdn_pipeline_body(
    qkv_conv_ref: Any,
    b_ref: Any,
    a_ref: Any,
    do_ref: Any,
    chunk_states_ref: Any,
    t_inv_ref: Any,
    a_log_ref: Any,
    dt_bias_ref: Any,
    reset_ref: Any,
    dy_conv_ref: Any,
    d_b_ref: Any,
    d_a_ref: Any,
    d_a_log_ref: Any,
    d_dt_bias_ref: Any,
    d_state_scr: Any,
    *,
    chunk_size: int,
    dim_size: int,
    num_kq_heads: int,
    num_v_heads: int,
    padded_num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    use_qk_norm_in_gdn: bool,
    kernel_size: int = 4,
    pad_len: int = 8,
) -> None:
  """Inner kernel executed per (batch, chunk) by emit_pipeline with manual GDN backward."""
  del kernel_size, pad_len
  c = pl.program_id(1)
  repeats = num_v_heads // num_kq_heads
  q_size = num_kq_heads * kq_head_dim
  k_size = num_kq_heads * kq_head_dim
  v_size = num_v_heads * v_head_dim

  @pl.when(c == 0)
  def _init():
    d_state_scr[...] = jnp.zeros((num_v_heads, kq_head_dim, v_head_dim), dtype=jnp.float32)

  is_reset = reset_ref[...][0, 0] > 0.5
  d_state = jnp.where(is_reset, 0.0, d_state_scr[...])
  y_c = qkv_conv_ref[...]

  # Slice chunk inputs for this head group
  q_orig = y_c[:, :q_size].reshape((chunk_size, num_kq_heads, kq_head_dim)).astype(jnp.float32)
  k_orig = y_c[:, q_size : q_size + k_size].reshape((chunk_size, num_kq_heads, kq_head_dim)).astype(jnp.float32)
  v = (
      y_c[:, q_size + k_size : q_size + k_size + v_size]
      .reshape((chunk_size, num_v_heads, v_head_dim))
      .astype(jnp.float32)
  )

  b_val = b_ref[...][:, :num_v_heads].astype(jnp.float32)
  a_val = a_ref[...][:, :num_v_heads].astype(jnp.float32)
  do_val = do_ref[...].astype(jnp.float32)
  state_prev = chunk_states_ref[...].astype(jnp.float32)
  t_inv_val = t_inv_ref[...].astype(jnp.float32)
  a_log_val = a_log_ref[...][0, :num_v_heads].astype(jnp.float32)
  dt_bias_val = dt_bias_ref[...][0, :num_v_heads].astype(jnp.float32)

  scale = 1.0 / jnp.sqrt(kq_head_dim)
  mask_cumsum = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32))
  mask_strict = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=-1)
  mask_causal = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=jnp.float32), k=0)

  if use_qk_norm_in_gdn:
    norm_q = normalizations.l2norm(q_orig, dim=-1, eps=1e-6)
    norm_k = normalizations.l2norm(k_orig, dim=-1, eps=1e-6)
    q_scaled = norm_q * scale
    k_scaled_val = norm_k
  else:
    q_scaled = q_orig * scale
    k_scaled_val = k_orig

  q_rep = jnp.repeat(q_scaled, repeats, axis=1)
  k_rep = jnp.repeat(k_scaled_val, repeats, axis=1)
  beta = jax.nn.sigmoid(b_val)

  sp_input = a_val + dt_bias_val
  sp_val = jax.nn.softplus(sp_input)
  exp_a_log = jnp.exp(a_log_val)
  log_g = -exp_a_log * sp_val

  cumsum_log_g = jnp.dot(mask_cumsum, log_g)

  q_h = jnp.transpose(q_rep, (1, 0, 2))
  k_h = jnp.transpose(k_rep, (1, 0, 2))
  v_h = jnp.transpose(v, (1, 0, 2))
  beta_h = jnp.transpose(beta, (1, 0))
  cumsum_h = jnp.transpose(cumsum_log_g, (1, 0))
  do_h = jnp.transpose(do_val, (1, 0, 2))

  diff = cumsum_h[:, :, None] - cumsum_h[:, None, :]
  safe_diff_strict = jnp.where(mask_strict[None, :, :] == 1.0, diff, -1e4)
  g_mat_strict = jnp.exp(safe_diff_strict) * mask_strict[None, :, :]

  safe_diff_causal = jnp.where(mask_causal[None, :, :] == 1.0, diff, -1e4)
  g_mat_causal = jnp.exp(safe_diff_causal) * mask_causal[None, :, :]

  gating_forward = jnp.exp(cumsum_h)[:, :, None]
  gating_last = jnp.exp(cumsum_h[:, -1])[:, None, None]
  gating_backward = jnp.exp(cumsum_h[:, -1:] - cumsum_h)[:, :, None]

  k_beta = k_h * beta_h[:, :, None]
  k_h_T = jnp.swapaxes(k_h, -1, -2)
  S_unmasked = jnp.matmul(k_beta, k_h_T)

  A = t_inv_val

  v_beta = v_h * beta_h[:, :, None]
  k_beta_g = k_beta * gating_forward

  u = jnp.matmul(A, v_beta)
  w = jnp.matmul(A, k_beta_g)

  ws = jnp.matmul(w, state_prev)
  v_new = u - ws

  q_g = q_h * gating_forward
  attn_unmasked = jnp.matmul(q_h, k_h_T)
  attn = attn_unmasked * g_mat_causal

  k_scaled_bwd = k_h * gating_backward

  dv_attn = jnp.matmul(jnp.swapaxes(attn, -1, -2), do_h)

  dv_new = dv_attn + jnp.matmul(k_scaled_bwd, d_state)
  d_attn = jnp.matmul(do_h, jnp.swapaxes(v_new, -1, -2))

  du = dv_new
  dw = -jnp.matmul(dv_new, jnp.swapaxes(state_prev, -1, -2))

  d_state_prev = (
      d_state * gating_last + jnp.matmul(jnp.swapaxes(q_g, -1, -2), do_h) - jnp.matmul(jnp.swapaxes(w, -1, -2), dv_new)
  )

  A_T = jnp.swapaxes(A, -1, -2)
  d_v_beta = jnp.matmul(A_T, du)
  d_k_beta_g = jnp.matmul(A_T, dw)

  # NOTE(optimization): The 4-GEMM sequence below (evaluating dA and the sandwich
  # product A_T @ dA @ A_T) can be algebraically collapsed into a single outer-product GEMM.
  #
  # Derivation:
  #   A_T @ dA @ A_T = A_T @ (du @ v_beta.T + dw @ k_beta_g.T) @ A_T
  #                  = (A_T @ du) @ (A @ v_beta).T + (A_T @ dw) @ (A @ k_beta_g).T
  #                  = (A_T @ du) @ u.T + (A_T @ dw) @ w.T
  #   Since dw = -du @ state_prev.T:
  #     A_T @ dw = -(A_T @ du) @ state_prev.T
  #   Substituting:
  #     A_T @ dA @ A_T = (A_T @ du) @ u.T - (A_T @ du) @ state_prev.T @ w.T
  #                    = (A_T @ du) @ (u - w @ state_prev).T
  #                    = d_v_beta @ v_new.T
  #
  # Replacement:
  #   The 2 lines below:
  #     dA = jnp.matmul(du, jnp.swapaxes(v_beta, -1, -2)) + jnp.matmul(dw, jnp.swapaxes(k_beta_g, -1, -2))
  #     dS = jnp.tril(-jnp.matmul(jnp.matmul(A_T, dA), A_T), k=-1)
  #   can be replaced with:
  #     dS = jnp.tril(-jnp.matmul(d_v_beta, jnp.swapaxes(v_new, -1, -2)), k=-1)
  #
  # This eliminates 3 out of 4 matrix multiplications in this block and removes
  # the need to allocate intermediate buffer dA in VMEM.
  dA = jnp.matmul(du, jnp.swapaxes(v_beta, -1, -2)) + jnp.matmul(dw, jnp.swapaxes(k_beta_g, -1, -2))

  dS = jnp.tril(-jnp.matmul(jnp.matmul(A_T, dA), A_T), k=-1)

  d_S_unmasked = dS * g_mat_strict
  d_k_beta_from_S = jnp.matmul(d_S_unmasked, k_h)
  d_k_h_from_S = jnp.matmul(jnp.swapaxes(d_S_unmasked, -1, -2), k_beta)

  d_k_beta = d_k_beta_g * gating_forward + d_k_beta_from_S
  d_beta_h = jnp.sum(d_v_beta * v_h, axis=-1) + jnp.sum(d_k_beta * k_h, axis=-1)
  d_v_h = d_v_beta * beta_h[:, :, None]

  d_attn_unmasked = d_attn * g_mat_causal
  d_q_h_from_attn = jnp.matmul(d_attn_unmasked, k_h)
  d_k_h_from_attn = jnp.matmul(jnp.swapaxes(d_attn_unmasked, -1, -2), q_h)

  d_q_g = jnp.matmul(do_h, jnp.swapaxes(state_prev, -1, -2))
  d_q_h_from_q_g = d_q_g * gating_forward

  d_k_scaled = jnp.matmul(v_new, jnp.swapaxes(d_state, -1, -2))
  d_k_h_from_k_scaled = d_k_scaled * gating_backward

  d_q_h = d_q_h_from_q_g + d_q_h_from_attn
  d_k_h = d_k_h_from_attn + d_k_h_from_S + d_k_beta * beta_h[:, :, None] + d_k_h_from_k_scaled

  # Gating adjoints
  d_gating_forward = jnp.sum(d_q_g * q_h, axis=-1) + jnp.sum(d_k_beta_g * k_beta, axis=-1)
  d_cumsum_from_fwd = d_gating_forward * gating_forward[:, :, 0]

  d_gating_last = jnp.sum(d_state * state_prev, axis=(-1, -2))
  d_cumsum_last = d_gating_last * jnp.exp(cumsum_h[:, -1])

  d_gating_backward = jnp.sum(d_k_scaled * k_h, axis=-1)
  d_diff_bwd = d_gating_backward * gating_backward[:, :, 0]

  d_g_strict = dS * S_unmasked
  d_g_causal = d_attn * attn_unmasked
  d_diff = (d_g_strict * g_mat_strict) + (d_g_causal * g_mat_causal)
  d_cumsum_from_diff = jnp.sum(d_diff, axis=2) - jnp.sum(d_diff, axis=1)

  d_cumsum_h = d_cumsum_from_fwd + d_cumsum_from_diff - d_diff_bwd
  last_col_addition = (d_cumsum_last + jnp.sum(d_diff_bwd, axis=-1))[:, None]
  d_cumsum_h = d_cumsum_h + jnp.pad(last_col_addition, ((0, 0), (chunk_size - 1, 0)))

  d_cumsum_log_g = jnp.transpose(d_cumsum_h, (1, 0))
  d_log_g = jnp.dot(mask_cumsum.T, d_cumsum_log_g)

  sig_sp = jax.nn.sigmoid(sp_input)
  d_a_val = d_log_g * (-exp_a_log * sig_sp)
  d_a_log_val = jnp.sum(d_log_g * (-exp_a_log * sp_val), axis=0)
  d_dt_bias_val = jnp.sum(d_a_val, axis=0)

  d_b_val = (jnp.transpose(d_beta_h, (1, 0))) * beta * (1.0 - beta)

  d_v_val = jnp.transpose(d_v_h, (1, 0, 2))
  d_q_rep = jnp.transpose(d_q_h, (1, 0, 2))
  d_k_rep = jnp.transpose(d_k_h, (1, 0, 2))

  d_q_proj = jnp.sum(
      d_q_rep.reshape(chunk_size, num_kq_heads, repeats, kq_head_dim),
      axis=2,
  )
  d_k_proj = jnp.sum(
      d_k_rep.reshape(chunk_size, num_kq_heads, repeats, kq_head_dim),
      axis=2,
  )

  if use_qk_norm_in_gdn:
    d_q_scaled = d_q_proj * scale
    r_q = jnp.sqrt(jnp.sum(q_orig**2, axis=-1, keepdims=True) + 1e-12)
    q_unit = q_orig / r_q
    d_q = (d_q_scaled - q_unit * jnp.sum(d_q_scaled * q_unit, axis=-1, keepdims=True)) / r_q

    r_k = jnp.sqrt(jnp.sum(k_orig**2, axis=-1, keepdims=True) + 1e-12)
    k_unit = k_orig / r_k
    d_k = (d_k_proj - k_unit * jnp.sum(d_k_proj * k_unit, axis=-1, keepdims=True)) / r_k
  else:
    d_q = d_q_proj * scale
    d_k = d_k_proj

  # Flatten gradients to chunk_size x dim_size and write to refs
  d_q_flat = d_q.reshape(chunk_size, q_size).astype(dy_conv_ref.dtype)
  d_k_flat = d_k.reshape(chunk_size, k_size).astype(dy_conv_ref.dtype)
  d_v_flat = d_v_val.reshape(chunk_size, v_size).astype(dy_conv_ref.dtype)
  dy_conv_ref[...] = jnp.concatenate([d_q_flat, d_k_flat, d_v_flat], axis=-1)

  if padded_num_v_heads > num_v_heads:
    d_b_ref[...] = jnp.pad(d_b_val, ((0, 0), (0, padded_num_v_heads - num_v_heads))).astype(d_b_ref.dtype)
    d_a_ref[...] = jnp.pad(d_a_val, ((0, 0), (0, padded_num_v_heads - num_v_heads))).astype(d_a_ref.dtype)
    d_a_log_ref[...] = jnp.pad(d_a_log_val[None, :], ((0, 0), (0, padded_num_v_heads - num_v_heads))).astype(
        d_a_log_ref.dtype
    )
    d_dt_bias_ref[...] = jnp.pad(d_dt_bias_val[None, :], ((0, 0), (0, padded_num_v_heads - num_v_heads))).astype(
        d_dt_bias_ref.dtype
    )
  else:
    d_b_ref[...] = d_b_val.astype(d_b_ref.dtype)
    d_a_ref[...] = d_a_val.astype(d_a_ref.dtype)
    d_a_log_ref[...] = d_a_log_val[None, :].astype(d_a_log_ref.dtype)
    d_dt_bias_ref[...] = d_dt_bias_val[None, :].astype(d_dt_bias_ref.dtype)

  d_state_scr[...] = d_state_prev.astype(d_state_scr.dtype)


def _pallas_gdn_bwd_kernel_single_group(
    qkv_conv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    do: jax.Array,
    chunk_states: jax.Array,
    t_inv: jax.Array,
    *,
    num_v_heads: int,
    num_kq_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
    vmem_limit_mb: Optional[int] = None,
    segment_ids: Optional[jax.Array] = None,
    interpret: bool | pltpu.InterpretParams | None = None,
    name: str = "gdn_bwd_kernel",
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Executes single head-group Pallas emit_pipeline kernel."""
  batch_size, seq_len, group_dim_size = qkv_conv.shape
  num_chunks = seq_len // chunk_size
  padded_num_v_heads = ((num_v_heads + 127) // 128) * 128

  # Reshape inputs into chunked tensors for emit_pipeline
  qkv_conv_4d = qkv_conv.reshape(batch_size, num_chunks, chunk_size, group_dim_size)

  b_4d = b.reshape(batch_size, num_chunks, chunk_size, num_v_heads)
  if padded_num_v_heads > num_v_heads:
    b_4d = jnp.pad(
        b_4d,
        ((0, 0), (0, 0), (0, 0), (0, padded_num_v_heads - num_v_heads)),
    )

  a_4d = a.reshape(batch_size, num_chunks, chunk_size, num_v_heads)
  if padded_num_v_heads > num_v_heads:
    a_4d = jnp.pad(
        a_4d,
        ((0, 0), (0, 0), (0, 0), (0, padded_num_v_heads - num_v_heads)),
    )

  do_5d = do.reshape(batch_size, num_chunks, chunk_size, num_v_heads, v_head_dim)

  if a_log.ndim == 1:
    a_log_3d = jnp.broadcast_to(a_log[None, None, :], (batch_size, 1, num_v_heads))
  elif a_log.ndim == 2:
    a_log_3d = a_log[:, None, :]
  else:
    a_log_3d = a_log
  if padded_num_v_heads > num_v_heads:
    a_log_3d = jnp.pad(a_log_3d, ((0, 0), (0, 0), (0, padded_num_v_heads - num_v_heads)))

  if dt_bias.ndim == 1:
    dt_bias_3d = jnp.broadcast_to(dt_bias[None, None, :], (batch_size, 1, num_v_heads))
  elif dt_bias.ndim == 2:
    dt_bias_3d = dt_bias[:, None, :]
  else:
    dt_bias_3d = dt_bias
  if padded_num_v_heads > num_v_heads:
    dt_bias_3d = jnp.pad(dt_bias_3d, ((0, 0), (0, 0), (0, padded_num_v_heads - num_v_heads)))

  t_inv_5d = t_inv.astype(jnp.float32)

  # Segment reset tensor for cross-document boundary gradient reset
  if segment_ids is not None and num_chunks > 1:
    end_idx = jnp.arange(1, num_chunks) * chunk_size - 1
    start_next_idx = jnp.arange(1, num_chunks) * chunk_size
    boundaries = segment_ids[:, end_idx] != segment_ids[:, start_next_idx]
    reset_mask = jnp.pad(boundaries, ((0, 0), (0, 1)), constant_values=False)
    reset_hbm = jnp.pad(
        reset_mask[:, :, None, None].astype(jnp.float32),
        ((0, 0), (0, 0), (0, 0), (0, 127)),
    )
  else:
    reset_hbm = jnp.zeros((batch_size, num_chunks, 1, 128), dtype=jnp.float32)

  in_specs, out_specs, num_in, num_out = make_bwd_block_specs(
      batch_size=batch_size,
      num_chunks=num_chunks,
      chunk_size=chunk_size,
      dim_size=group_dim_size,
      num_v_heads=num_v_heads,
      kq_head_dim=kq_head_dim,
      v_head_dim=v_head_dim,
      padded_num_v_heads=padded_num_v_heads,
  )

  out_shapes = (
      jax.ShapeDtypeStruct(
          (batch_size, num_chunks, chunk_size, group_dim_size),
          qkv_conv.dtype,
      ),
      jax.ShapeDtypeStruct(b_4d.shape, b_4d.dtype),
      jax.ShapeDtypeStruct(a_4d.shape, a_4d.dtype),
      jax.ShapeDtypeStruct(
          (batch_size, num_chunks, 1, padded_num_v_heads),
          a_log_3d.dtype,
      ),
      jax.ShapeDtypeStruct(
          (batch_size, num_chunks, 1, padded_num_v_heads),
          dt_bias_3d.dtype,
      ),
  )

  body = functools.partial(
      _bwd_gdn_pipeline_body,
      chunk_size=chunk_size,
      dim_size=group_dim_size,
      num_kq_heads=num_kq_heads,
      num_v_heads=num_v_heads,
      padded_num_v_heads=padded_num_v_heads,
      kq_head_dim=kq_head_dim,
      v_head_dim=v_head_dim,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
  )

  def outer(*refs):
    pltpu.emit_pipeline(
        body,
        grid=(batch_size, num_chunks),
        in_specs=in_specs,
        out_specs=out_specs,
    )(*refs[: num_in + num_out], scratches=tuple(refs[num_in + num_out :]))

  if vmem_limit_mb is not None and vmem_limit_mb <= 64:
    vmem_limit_bytes = int(vmem_limit_mb) * 1024 * 1024
  else:
    tpu_info = pltpu.get_tpu_info()
    vmem_limit_bytes = int(0.85 * tpu_info.vmem_capacity_bytes)

  hbm = pltpu.MemorySpace.HBM
  (
      dy_conv_chunks,
      d_b_chunks,
      d_a_chunks,
      d_a_log_chunks,
      d_dt_bias_chunks,
  ) = pl.pallas_call(
      outer,
      grid=(),
      out_shape=out_shapes,
      in_specs=[pl.BlockSpec(memory_space=hbm)] * num_in,
      out_specs=[pl.BlockSpec(memory_space=hbm)] * num_out,
      scratch_shapes=[
          pltpu.VMEM((num_v_heads, kq_head_dim, v_head_dim), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      interpret=interpret,
      name=name,
  )(
      qkv_conv_4d,
      b_4d,
      a_4d,
      do_5d,
      chunk_states,
      t_inv_5d,
      a_log_3d,
      dt_bias_3d,
      reset_hbm,
  )

  d_a_log_reduced = jnp.sum(d_a_log_chunks[..., 0, :num_v_heads], axis=(0, 1)).astype(a_log.dtype)
  d_dt_bias_reduced = jnp.sum(d_dt_bias_chunks[..., 0, :num_v_heads], axis=(0, 1)).astype(dt_bias.dtype)
  d_b_flat = d_b_chunks[..., :num_v_heads].reshape(batch_size, seq_len, num_v_heads).astype(b.dtype)
  d_a_flat = d_a_chunks[..., :num_v_heads].reshape(batch_size, seq_len, num_v_heads).astype(a.dtype)
  dy_conv_flat = dy_conv_chunks.reshape(batch_size, seq_len, group_dim_size).astype(qkv_conv.dtype)

  return (
      dy_conv_flat,
      d_b_flat,
      d_a_flat,
      d_a_log_reduced,
      d_dt_bias_reduced,
  )


def pallas_gdn_bwd_kernel(
    qkv_conv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    do: jax.Array,
    chunk_states: jax.Array,
    t_inv: jax.Array,
    *,
    num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
    vmem_limit_mb: Optional[int] = None,
    head_tile: Optional[int] = None,
    segment_ids: Optional[jax.Array] = None,
    interpret: bool | pltpu.InterpretParams | None = None,
) -> Tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
  """Executes the Pallas reverse-chunk GDNv3 backward kernel using emit_pipeline.

  Streams natively per group in contiguous layout.
  """
  if interpret is None and jax.default_backend() == "cpu":
    interpret = True
  if interpret:
    ensure_cpu_interpret_registered()

  _, _, dim_size = qkv_conv.shape
  num_kq_heads = (dim_size - num_v_heads * v_head_dim) // (kq_head_dim * 2)
  repeats = num_v_heads // num_kq_heads

  target_tile = 16 if head_tile is None else head_tile
  max_possible = min(num_v_heads, target_tile)
  tile_v_heads = None
  for candidate in range(max_possible, 0, -1):
    if num_v_heads % candidate == 0 and candidate % repeats == 0:
      tile_v_heads = candidate
      break
  if tile_v_heads is None:
    tile_v_heads = repeats if num_v_heads % repeats == 0 else num_v_heads
  num_groups = num_v_heads // tile_v_heads
  tile_kq_heads = tile_v_heads // repeats

  if num_groups == 1:
    return _pallas_gdn_bwd_kernel_single_group(
        qkv_conv=qkv_conv,
        b=b,
        a=a,
        a_log=a_log,
        dt_bias=dt_bias,
        do=do,
        chunk_states=chunk_states,
        t_inv=t_inv,
        num_v_heads=num_v_heads,
        num_kq_heads=num_kq_heads,
        kq_head_dim=kq_head_dim,
        v_head_dim=v_head_dim,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        vmem_limit_mb=vmem_limit_mb,
        segment_ids=segment_ids,
        interpret=interpret,
        name="gdn_bwd_kernel",
    )

  q_size = num_kq_heads * kq_head_dim
  k_size = num_kq_heads * kq_head_dim
  tile_q_size = tile_kq_heads * kq_head_dim
  tile_k_size = tile_kq_heads * kq_head_dim

  dq_list = []
  dk_list = []
  dv_list = []
  db_list = []
  da_list = []
  dal_list = []
  ddt_list = []

  for g in range(num_groups):
    vh_start = g * tile_v_heads
    vh_end = (g + 1) * tile_v_heads
    kqh_start = g * tile_kq_heads
    kqh_end = (g + 1) * tile_kq_heads

    q_g = qkv_conv[:, :, kqh_start * kq_head_dim : kqh_end * kq_head_dim]
    k_g = qkv_conv[:, :, q_size + kqh_start * kq_head_dim : q_size + kqh_end * kq_head_dim]
    v_g = qkv_conv[
        :,
        :,
        q_size + k_size + vh_start * v_head_dim : q_size + k_size + vh_end * v_head_dim,
    ]
    qkv_g = jnp.concatenate([q_g, k_g, v_g], axis=-1)

    b_g = b[:, :, vh_start:vh_end]
    a_g = a[:, :, vh_start:vh_end]
    do_g = do[:, :, vh_start:vh_end, :]
    chunk_states_g = chunk_states[:, :, vh_start:vh_end, :, :]
    t_inv_g = t_inv[:, :, vh_start:vh_end, :, :]

    if a_log.ndim == 1:
      a_log_g = a_log[vh_start:vh_end]
    elif a_log.ndim == 2:
      a_log_g = a_log[:, vh_start:vh_end]
    else:
      a_log_g = a_log[:, :, vh_start:vh_end]

    if dt_bias.ndim == 1:
      dt_bias_g = dt_bias[vh_start:vh_end]
    elif dt_bias.ndim == 2:
      dt_bias_g = dt_bias[:, vh_start:vh_end]
    else:
      dt_bias_g = dt_bias[:, :, vh_start:vh_end]

    dy_g, db_g, da_g, dal_g, ddt_g = _pallas_gdn_bwd_kernel_single_group(
        qkv_conv=qkv_g,
        b=b_g,
        a=a_g,
        a_log=a_log_g,
        dt_bias=dt_bias_g,
        do=do_g,
        chunk_states=chunk_states_g,
        t_inv=t_inv_g,
        num_v_heads=tile_v_heads,
        num_kq_heads=tile_kq_heads,
        kq_head_dim=kq_head_dim,
        v_head_dim=v_head_dim,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        vmem_limit_mb=vmem_limit_mb,
        segment_ids=segment_ids,
        interpret=interpret,
        name=f"gdn_bwd_kernel_group_{g}",
    )

    dq_g = dy_g[:, :, :tile_q_size]
    dk_g = dy_g[:, :, tile_q_size : tile_q_size + tile_k_size]
    dv_g = dy_g[:, :, tile_q_size + tile_k_size :]

    dq_list.append(dq_g)
    dk_list.append(dk_g)
    dv_list.append(dv_g)
    db_list.append(db_g)
    da_list.append(da_g)
    dal_list.append(dal_g)
    ddt_list.append(ddt_g)

  dq_flat = jnp.concatenate(dq_list, axis=-1)
  dk_flat = jnp.concatenate(dk_list, axis=-1)
  dv_flat = jnp.concatenate(dv_list, axis=-1)
  dy_conv_flat = jnp.concatenate([dq_flat, dk_flat, dv_flat], axis=-1).astype(qkv_conv.dtype)
  d_b_flat = jnp.concatenate(db_list, axis=-1)
  d_a_flat = jnp.concatenate(da_list, axis=-1)
  d_a_log_reduced = jnp.concatenate(dal_list, axis=-1)
  d_dt_bias_reduced = jnp.concatenate(ddt_list, axis=-1)

  return (
      dy_conv_flat,
      d_b_flat,
      d_a_flat,
      d_a_log_reduced,
      d_dt_bias_reduced,
  )


pallas_gdn_bwd_computation = pallas_gdn_bwd_kernel

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

"""Hybrid Gated Delta Net (GDN) analytical backward pass using Pallas emit_pipeline.

Fuses Conv1D and analytical Gated Delta Rule backward operations with automatic
double-buffering and pipelined DMA transfers via pltpu.emit_pipeline.
Bypasses jax.vjp(chunk_forward) and internal triangular back-substitutions
by utilizing cached triangular inverse matrices (t_inv) and direct systolic
matrix multiplications.
"""

import functools
from typing import Any, Optional, Tuple

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

try:
  from maxtext.layers import normalizations
except ImportError:
  from maxtext.src.maxtext.layers import normalizations

try:
  import jax.experimental.xla_metadata
  if not hasattr(jax.experimental.xla_metadata, "must_fuse_call"):
    jax.experimental.xla_metadata.must_fuse_call = (
        lambda *args, **kwargs: (lambda fn: fn)
    )
except Exception:
  pass

try:
  from maxtext.models import qwen3
except ImportError:
  from maxtext.src.maxtext.models import qwen3

try:
  from maxtext.models.kernels.gdn import compute_gdn as local_compute_gdn
  from maxtext.models.kernels.gdn import wrapper as local_gdn_wrapper
except ImportError:
  try:
    from maxtext.src.maxtext.models.kernels.gdn import compute_gdn as local_compute_gdn
    from maxtext.src.maxtext.models.kernels.gdn import wrapper as local_gdn_wrapper
  except ImportError:
    from .kernels.gdn import compute_gdn as local_compute_gdn
    from .kernels.gdn import wrapper as local_gdn_wrapper


def ensure_cpu_interpret_registered() -> None:
  """Ensures Pallas CPU interpretation registers TPU hardware info without top-level import side-effects."""
  try:
    from jax._src.pallas.mosaic import tpu_info as _ti  # noqa: E402

    if "cpu" not in _ti.registry:
      _ti.registry["cpu"] = lambda: _ti.get_tpu_info_for_chip(
          _ti.ChipVersion.TPU_V6E, 1
      )
    try:
      _ti.get_tpu_info.cache_clear()
    except Exception:
      pass
  except Exception:  # pragma: no cover
    pass

  try:
    from jax._src.pallas.mosaic import pipeline as _pl_pipeline  # noqa: E402

    if not getattr(_pl_pipeline, "_is_cpu_safe_cbs_patched", False):
      _orig_cbs = getattr(
          _pl_pipeline,
          "_original_create_bounded_slice",
          _pl_pipeline._create_bounded_slice,
      )
      _pl_pipeline._original_create_bounded_slice = _orig_cbs

      def _cpu_safe_create_bounded_slice(
          slice_start,
          slice_size,
          block_size,
          dim_size,
          tiling=None,
          *args,
          **kwargs,
      ):
        if isinstance(slice_size, int) and (
            tiling is None or slice_size % tiling == 0
        ):
          return pl.ds(slice_start, slice_size)
        return _orig_cbs(
            slice_start,
            slice_size,
            block_size,
            dim_size,
            tiling,
            *args,
            **kwargs,
        )

      _pl_pipeline._create_bounded_slice = _cpu_safe_create_bounded_slice
      _pl_pipeline._is_cpu_safe_cbs_patched = True
  except Exception:  # pragma: no cover
    pass


@jax.custom_vjp
def invert_triangular_matrix(t: jax.Array) -> jax.Array:
  """Computes inverse of unit lower-triangular matrix using Tokamax block forward substitution."""
  return local_compute_gdn.invert_triangular_matrix(t, block_size=16)


def _invert_triangular_matrix_fwd(t: jax.Array):
  t_inv = invert_triangular_matrix(t)
  return t_inv, t_inv


def _invert_triangular_matrix_bwd(res, g):
  t_inv = res
  grad_t = jnp.tril(-(t_inv.mT @ g @ t_inv.mT), k=-1)
  return (grad_t,)


invert_triangular_matrix.defvjp(
    _invert_triangular_matrix_fwd,
    _invert_triangular_matrix_bwd,
)


def chunk_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    b_val: jax.Array,
    a_val: jax.Array,
    a_log_val: jax.Array,
    dt_bias_val: jax.Array,
    state_prev: jax.Array,
    *,
    kq_head_dim: int,
    repeats: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool = False,
) -> Tuple[jax.Array, jax.Array]:
  """Computes one chunk forward pass for GDN v3 with WY delta rule."""
  out, state_new, _ = chunk_forward_with_tinv(
      q=q,
      k=k,
      v=v,
      b_val=b_val,
      a_val=a_val,
      a_log_val=a_log_val,
      dt_bias_val=dt_bias_val,
      state_prev=state_prev,
      kq_head_dim=kq_head_dim,
      repeats=repeats,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
  )
  return out, state_new


def chunk_forward_with_tinv(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    b_val: jax.Array,
    a_val: jax.Array,
    a_log_val: jax.Array,
    dt_bias_val: jax.Array,
    state_prev: jax.Array,
    *,
    kq_head_dim: int,
    repeats: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool = False,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Computes one chunk forward pass for GDN v3 and returns (out, state_new, t_inv)."""
  q = q.astype(jnp.float32)
  k = k.astype(jnp.float32)
  v = v.astype(jnp.float32)
  if use_qk_norm_in_gdn:
    q = normalizations.l2norm(q, dim=-1, eps=1e-6)
    k = normalizations.l2norm(k, dim=-1, eps=1e-6)
  scale = 1.0 / jnp.sqrt(kq_head_dim)
  q = q * scale
  b_val = b_val.astype(jnp.float32)
  a_val = a_val.astype(jnp.float32)
  a_log_val = a_log_val.astype(jnp.float32)
  dt_bias_val = dt_bias_val.astype(jnp.float32)
  state_prev = state_prev.astype(jnp.float32)
  q_rep = jnp.repeat(q, repeats, axis=1)
  k_rep = jnp.repeat(k, repeats, axis=1)

  beta = jax.nn.sigmoid(b_val)

  # EXACT GDN v3 gating formula
  log_g = -jnp.exp(a_log_val) * jax.nn.softplus(a_val + dt_bias_val)

  # Fast MXU cumsum replacement
  mask_cumsum = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=log_g.dtype))
  cumsum_log_g = jnp.dot(mask_cumsum, log_g)

  # Transpose to head-first: (H, C, D)
  q_h = jnp.transpose(q_rep, (1, 0, 2))
  k_h = jnp.transpose(k_rep, (1, 0, 2))
  v_h = jnp.transpose(v, (1, 0, 2))
  beta_h = jnp.transpose(beta, (1, 0))
  cumsum_h = jnp.transpose(cumsum_log_g, (1, 0))

  diff = cumsum_h[:, :, None] - cumsum_h[:, None, :]
  mask_strict = jnp.tril(
      jnp.ones((chunk_size, chunk_size), dtype=diff.dtype), k=-1
  )
  safe_diff_strict = jnp.where(mask_strict[None, :, :] == 1.0, diff, -1e4)
  g_mat_strict = jnp.exp(safe_diff_strict) * mask_strict[None, :, :]

  mask_causal = jnp.tril(
      jnp.ones((chunk_size, chunk_size), dtype=diff.dtype), k=0
  )
  safe_diff_causal = jnp.where(mask_causal[None, :, :] == 1.0, diff, -1e4)
  g_mat_causal = jnp.exp(safe_diff_causal) * mask_causal[None, :, :]

  gating_forward = jnp.exp(cumsum_h)[:, :, None]
  gating_last = jnp.exp(cumsum_h[:, -1])[:, None, None]
  gating_backward = jnp.exp(cumsum_h[:, -1:] - cumsum_h)[:, :, None]

  # WY Representation: T = unit lower-triangular Gram matrix
  k_beta = k_h * beta_h[:, :, None]
  S = jnp.matmul(k_beta, jnp.swapaxes(k_h, -1, -2)) * g_mat_strict
  identity_mask = jnp.eye(chunk_size, dtype=S.dtype)[None, :, :]
  t = jnp.where(identity_mask == 1.0, 1.0, S)
  A = invert_triangular_matrix(t)

  v_beta = v_h * beta_h[:, :, None]
  k_beta_g = k_beta * gating_forward
  u = jnp.matmul(A, v_beta)
  w = jnp.matmul(A, k_beta_g)

  # Delta error subtraction against recurrent state
  ws = jnp.matmul(w, state_prev)
  v_new = u - ws

  # Output: cross-chunk state read + intra-chunk attention with v_new
  q_g = q_h * gating_forward
  out_cross = jnp.matmul(q_g, state_prev)

  attn = jnp.matmul(q_h, jnp.swapaxes(k_h, -1, -2)) * g_mat_causal
  out_intra = jnp.matmul(attn, v_new)

  out = out_cross + out_intra
  out = jnp.transpose(out, (1, 0, 2))

  # State update: decayed previous state + rank-1 update from chunk with v_new
  state_prev_decayed = state_prev * gating_last
  k_scaled = k_h * gating_backward
  state_new_intra = jnp.matmul(jnp.swapaxes(k_scaled, -1, -2), v_new)
  state_new = state_prev_decayed + state_new_intra

  return out, state_new, A


def chunk_state_forward_with_cached_tinv(
    k: jax.Array,
    v: jax.Array,
    b_val: jax.Array,
    a_val: jax.Array,
    a_log_val: jax.Array,
    dt_bias_val: jax.Array,
    state_prev: jax.Array,
    t_inv: jax.Array,
    *,
    repeats: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool = False,
) -> jax.Array:
  """Computes next recurrent state for one chunk using cached t_inv without triangular inversion."""
  v = v.astype(jnp.float32)
  k = k.astype(jnp.float32)
  if use_qk_norm_in_gdn:
    k = normalizations.l2norm(k, dim=-1, eps=1e-6)
  k_rep = jnp.repeat(k, repeats, axis=1)

  b_val = b_val.astype(jnp.float32)
  a_val = a_val.astype(jnp.float32)
  a_log_val = a_log_val.astype(jnp.float32)
  dt_bias_val = dt_bias_val.astype(jnp.float32)
  state_prev = state_prev.astype(jnp.float32)

  beta = jax.nn.sigmoid(b_val)
  log_g = -jnp.exp(a_log_val) * jax.nn.softplus(a_val + dt_bias_val)

  mask_cumsum = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=log_g.dtype))
  cumsum_log_g = jnp.dot(mask_cumsum, log_g)

  k_h = jnp.transpose(k_rep, (1, 0, 2))
  v_h = jnp.transpose(v, (1, 0, 2))
  beta_h = jnp.transpose(beta, (1, 0))
  cumsum_h = jnp.transpose(cumsum_log_g, (1, 0))

  gating_forward = jnp.exp(cumsum_h)[:, :, None]
  gating_last = jnp.exp(cumsum_h[:, -1])[:, None, None]
  gating_backward = jnp.exp(cumsum_h[:, -1:] - cumsum_h)[:, :, None]

  A = t_inv.astype(jnp.float32)
  k_beta = k_h * beta_h[:, :, None]
  v_beta = v_h * beta_h[:, :, None]
  k_beta_g = k_beta * gating_forward

  u = jnp.matmul(A, v_beta)
  w = jnp.matmul(A, k_beta_g)

  ws = jnp.matmul(w, state_prev)
  v_new = u - ws

  state_prev_decayed = state_prev * gating_last
  k_scaled = k_h * gating_backward
  state_new_intra = jnp.matmul(jnp.swapaxes(k_scaled, -1, -2), v_new)
  state_new = state_prev_decayed + state_new_intra

  return state_new


def make_bwd_block_specs(
    batch_size: int,
    num_chunks: int,
    chunk_size: int,
    dim_size: int,
    num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    kernel_size: int,
    pad_len: int,
    padded_num_v_heads: int | None = None,
) -> Tuple[list[pl.BlockSpec], list[pl.BlockSpec], int, int]:
  """Constructs reverse-scan Pallas emit_pipeline in_specs and out_specs including t_inv."""
  del batch_size
  if padded_num_v_heads is None:
    padded_num_v_heads = max(num_v_heads, 256)
  rc = lambda c: num_chunks - 1 - c
  in_specs = [
      pl.BlockSpec(
          (None, pl.BoundedSlice(chunk_size + pad_len), dim_size),
          lambda b, c: (b, pl.ds(rc(c) * chunk_size, chunk_size + pad_len), 0),
      ),
      pl.BlockSpec(
          (None, None, chunk_size, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, chunk_size, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, chunk_size, num_v_heads, v_head_dim),
          lambda b, c: (b, rc(c), 0, 0, 0),
      ),
      pl.BlockSpec(
          (None, None, num_v_heads, kq_head_dim, v_head_dim),
          lambda b, c: (b, rc(c), 0, 0, 0),
      ),
      pl.BlockSpec(
          (None, None, num_v_heads, chunk_size, chunk_size),
          lambda b, c: (b, rc(c), 0, 0, 0),
      ),
      pl.BlockSpec(
          (None, 1, padded_num_v_heads),
          lambda b, c: (b, 0, 0),
      ),
      pl.BlockSpec(
          (None, 1, padded_num_v_heads),
          lambda b, c: (b, 0, 0),
      ),
      pl.BlockSpec(
          (kernel_size, dim_size),
          lambda b, c: (0, 0),
      ),
      pl.BlockSpec(
          (dim_size,),
          lambda b, c: (0,),
      ),
  ]
  out_specs = [
      pl.BlockSpec(
          (None, None, chunk_size, dim_size),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, chunk_size, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, chunk_size, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, kernel_size, dim_size),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, 1, dim_size),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, 1, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
      pl.BlockSpec(
          (None, None, 1, padded_num_v_heads),
          lambda b, c: (b, rc(c), 0, 0),
      ),
  ]
  return in_specs, out_specs, len(in_specs), len(out_specs)


def _bwd_analytical_pipeline_body(
    padded_pre_conv_qkv_ref: Any,
    b_ref: Any,
    a_ref: Any,
    do_ref: Any,
    chunk_states_ref: Any,
    t_inv_ref: Any,
    a_log_ref: Any,
    dt_bias_ref: Any,
    conv_weight_ref: Any,
    conv_bias_ref: Any,
    d_pre_conv_qkv_ref: Any,
    d_b_ref: Any,
    d_a_ref: Any,
    d_conv_weight_ref: Any,
    d_conv_bias_ref: Any,
    d_a_log_ref: Any,
    d_dt_bias_ref: Any,
    d_state_scr: Any,
    dz_halo_scratch: Any,
    *,
    chunk_size: int,
    dim_size: int,
    num_kq_heads: int,
    num_v_heads: int,
    padded_num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    kernel_size: int,
    pad_len: int,
    use_qk_norm_in_gdn: bool,
) -> None:
  """Inner kernel executed per (batch, chunk) by emit_pipeline with analytical manual backward."""
  c = pl.program_id(1)
  repeats = num_v_heads // num_kq_heads
  q_size = num_kq_heads * kq_head_dim
  k_size = num_kq_heads * kq_head_dim
  v_size = num_v_heads * v_head_dim

  @pl.when(c == 0)
  def _init():
    d_state_scr[...] = jnp.zeros(
        (num_v_heads, kq_head_dim, v_head_dim), dtype=jnp.float32
    )
    dz_halo_scratch[...] = jnp.zeros((pad_len, dim_size), dtype=jnp.float32)

  d_state = d_state_scr[...]
  dz_halo = dz_halo_scratch[...]

  padded_pre_conv_qkv_val = padded_pre_conv_qkv_ref[...]
  conv_w = conv_weight_ref[...]
  conv_b = conv_bias_ref[...]

  # 1. Recompute z_c = conv1d(x_c) + b and y_c = silu(z_c) directly in VMEM
  z_c = jnp.zeros((chunk_size, dim_size), dtype=jnp.float32)
  for k_idx in range(kernel_size):
    shift = kernel_size - 1 - k_idx
    start = pad_len - shift
    x_s = padded_pre_conv_qkv_val[start : start + chunk_size].astype(
        jnp.float32
    )
    z_c = z_c + x_s * conv_w[k_idx].astype(jnp.float32)
  z_c = z_c + conv_b.astype(jnp.float32)
  y_c = jax.nn.silu(z_c)

  # 2. Slice y_c into q, k, v for GDN reverse pass
  q_orig = (
      y_c[:, :q_size]
      .reshape((chunk_size, num_kq_heads, kq_head_dim))
      .astype(jnp.float32)
  )
  k_orig = (
      y_c[:, q_size : q_size + k_size]
      .reshape((chunk_size, num_kq_heads, kq_head_dim))
      .astype(jnp.float32)
  )
  v = (
      y_c[:, q_size + k_size :]
      .reshape((chunk_size, num_v_heads, v_head_dim))
      .astype(jnp.float32)
  )

  b_val = b_ref[...][:, :num_v_heads].astype(jnp.float32)
  a_val = a_ref[...][:, :num_v_heads].astype(jnp.float32)
  do_val = do_ref[...].astype(jnp.float32)
  state_prev_val = chunk_states_ref[...].astype(jnp.float32)
  t_inv_val = t_inv_ref[...].astype(jnp.float32)
  a_log_val = a_log_ref[...][0, :num_v_heads].astype(jnp.float32)
  dt_bias_val = dt_bias_ref[...][0, :num_v_heads].astype(jnp.float32)

  # 3. Manual Analytical GDN Backward Pass (Bypassing jax.vjp)
  scale = 1.0 / jnp.sqrt(kq_head_dim)
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

  mask_cumsum = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=log_g.dtype))
  cumsum_log_g = jnp.dot(mask_cumsum, log_g)

  q_h = jnp.transpose(q_rep, (1, 0, 2))
  k_h = jnp.transpose(k_rep, (1, 0, 2))
  v_h = jnp.transpose(v, (1, 0, 2))
  beta_h = jnp.transpose(beta, (1, 0))
  cumsum_h = jnp.transpose(cumsum_log_g, (1, 0))

  diff = cumsum_h[:, :, None] - cumsum_h[:, None, :]
  mask_strict = jnp.tril(
      jnp.ones((chunk_size, chunk_size), dtype=diff.dtype), k=-1
  )
  safe_diff_strict = jnp.where(mask_strict[None, :, :] == 1.0, diff, -1e4)
  g_mat_strict = jnp.exp(safe_diff_strict) * mask_strict[None, :, :]

  mask_causal = jnp.tril(
      jnp.ones((chunk_size, chunk_size), dtype=diff.dtype), k=0
  )
  safe_diff_causal = jnp.where(mask_causal[None, :, :] == 1.0, diff, -1e4)
  g_mat_causal = jnp.exp(safe_diff_causal) * mask_causal[None, :, :]

  gating_forward = jnp.exp(cumsum_h)[:, :, None]
  gating_last = jnp.exp(cumsum_h[:, -1])[:, None, None]
  gating_backward = jnp.exp(cumsum_h[:, -1:] - cumsum_h)[:, :, None]

  k_beta = k_h * beta_h[:, :, None]
  k_h_T = jnp.swapaxes(k_h, -1, -2)
  S_unmasked = jnp.matmul(k_beta, k_h_T)

  # Cached t_inv matrix
  A = t_inv_val

  v_beta = v_h * beta_h[:, :, None]
  k_beta_g = k_beta * gating_forward
  u = jnp.matmul(A, v_beta)
  w = jnp.matmul(A, k_beta_g)

  ws = jnp.matmul(w, state_prev_val)
  v_new = u - ws

  q_g = q_h * gating_forward
  attn_unmasked = jnp.matmul(q_h, k_h_T)
  attn = attn_unmasked * g_mat_causal

  k_scaled_bwd = k_h * gating_backward

  # Intermediate Adjoints
  do_h = jnp.transpose(do_val, (1, 0, 2))

  dv_new = jnp.matmul(jnp.swapaxes(attn, -1, -2), do_h) + jnp.matmul(
      k_scaled_bwd, d_state
  )
  d_attn = jnp.matmul(do_h, jnp.swapaxes(v_new, -1, -2))

  du = dv_new
  dw = -jnp.matmul(dv_new, jnp.swapaxes(state_prev_val, -1, -2))

  d_state_prev = (
      d_state * gating_last
      + jnp.matmul(jnp.swapaxes(q_g, -1, -2), do_h)
      - jnp.matmul(jnp.swapaxes(w, -1, -2), dv_new)
  )

  A_T = jnp.swapaxes(A, -1, -2)
  d_v_beta = jnp.matmul(A_T, du)
  d_k_beta_g = jnp.matmul(A_T, dw)
  dA = jnp.matmul(du, jnp.swapaxes(v_beta, -1, -2)) + jnp.matmul(
      dw, jnp.swapaxes(k_beta_g, -1, -2)
  )

  # Closed-form derivative through triangular inverse via systolic matmuls:
  # grad_t = tril(-(A_T @ dA @ A_T), k=-1)
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

  d_q_g = jnp.matmul(do_h, jnp.swapaxes(state_prev_val, -1, -2))
  d_q_h_from_q_g = d_q_g * gating_forward

  d_k_scaled = jnp.matmul(v_new, jnp.swapaxes(d_state, -1, -2))
  d_k_h_from_k_scaled = d_k_scaled * gating_backward

  d_q_h = d_q_h_from_q_g + d_q_h_from_attn
  d_k_h = (
      d_k_h_from_attn
      + d_k_h_from_S
      + d_k_beta * beta_h[:, :, None]
      + d_k_h_from_k_scaled
  )

  # Gating adjoints
  d_gating_forward = jnp.sum(d_q_g * q_h, axis=-1) + jnp.sum(
      d_k_beta_g * k_beta, axis=-1
  )
  d_cumsum_from_fwd = d_gating_forward * gating_forward[:, :, 0]

  d_gating_last = jnp.sum(d_state * state_prev_val, axis=(-1, -2))
  d_cumsum_last = d_gating_last * jnp.exp(cumsum_h[:, -1])

  d_gating_backward = jnp.sum(d_k_scaled * k_h, axis=-1)
  d_diff_bwd = d_gating_backward * gating_backward[:, :, 0]

  d_g_strict = dS * S_unmasked
  d_g_causal = d_attn * attn_unmasked
  d_diff = (d_g_strict * g_mat_strict) + (d_g_causal * g_mat_causal)
  d_cumsum_from_diff = jnp.sum(d_diff, axis=2) - jnp.sum(d_diff, axis=1)

  d_cumsum_h = d_cumsum_from_fwd + d_cumsum_from_diff - d_diff_bwd
  last_col_addition = (d_cumsum_last + jnp.sum(d_diff_bwd, axis=-1))[:, None]
  d_cumsum_h = d_cumsum_h + jnp.pad(
      last_col_addition, ((0, 0), (chunk_size - 1, 0))
  )

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
      d_q_rep.reshape(chunk_size, num_kq_heads, repeats, kq_head_dim), axis=2
  )
  d_k_proj = jnp.sum(
      d_k_rep.reshape(chunk_size, num_kq_heads, repeats, kq_head_dim), axis=2
  )

  if use_qk_norm_in_gdn:
    d_q_scaled = d_q_proj * scale
    r_q = jnp.sqrt(jnp.sum(q_orig**2, axis=-1, keepdims=True) + 1e-12)
    q_unit = q_orig / r_q
    d_q = (
        d_q_scaled
        - q_unit * jnp.sum(d_q_scaled * q_unit, axis=-1, keepdims=True)
    ) / r_q

    r_k = jnp.sqrt(jnp.sum(k_orig**2, axis=-1, keepdims=True) + 1e-12)
    k_unit = k_orig / r_k
    d_k = (
        d_k_proj - k_unit * jnp.sum(d_k_proj * k_unit, axis=-1, keepdims=True)
    ) / r_k
  else:
    d_q = d_q_proj * scale
    d_k = d_k_proj

  # 4. Form dy_c in VMEM (no HBM write)
  dy_c = jnp.concatenate(
      [
          d_q.reshape(chunk_size, q_size),
          d_k.reshape(chunk_size, k_size),
          d_v_val.reshape(chunk_size, v_size),
      ],
      axis=-1,
  )

  # 5. Compute elementwise SiLU' derivative in VMEM: dz_c = dy_c * SiLU'(z_c)
  sig_z = jax.nn.sigmoid(z_c)
  silu_prime = sig_z * (1.0 + z_c * (1.0 - sig_z))
  dz_c = dy_c * silu_prime

  # 6. Emit chunk partials: bias gradient db_c = sum(dz_c)
  d_conv_bias_ref[...] = jnp.sum(dz_c, axis=0, keepdims=True).astype(
      d_conv_bias_ref.dtype
  )

  # 7. Emit chunk partials: weight gradient dw_c[k] = sum(dz_c * x_shifted[k])
  dw_rows = []
  for k_idx in range(kernel_size):
    shift = kernel_size - 1 - k_idx
    start = pad_len - shift
    x_shifted = padded_pre_conv_qkv_val[start : start + chunk_size].astype(
        jnp.float32
    )
    dw_rows.append(jnp.sum(dz_c * x_shifted, axis=0))
  d_cw = jnp.stack(dw_rows, axis=0)
  d_conv_weight_ref[...] = d_cw.astype(d_conv_weight_ref.dtype)

  # 8. Anti-causal transposed convolution in VMEM for dx_c
  dz_extended = jnp.concatenate([dz_c, dz_halo], axis=0)

  dx_c = jnp.zeros((chunk_size, dim_size), dtype=jnp.float32)
  for j_idx in range(kernel_size):
    w_j = conv_w[kernel_size - 1 - j_idx].astype(jnp.float32)
    dx_c = dx_c + dz_extended[j_idx : j_idx + chunk_size] * w_j
  d_pre_conv_qkv_ref[...] = dx_c.astype(d_pre_conv_qkv_ref.dtype)

  # 9. Save boundary cotangents into dz_halo_scratch for next reverse iteration
  padded_halo = jnp.pad(
      dz_c[: kernel_size - 1],
      ((0, pad_len - (kernel_size - 1)), (0, 0)),
  )
  dz_halo_scratch[...] = padded_halo.astype(jnp.float32)

  # 10. Write other outputs & recurrent state carry
  d_b_ref[...] = jnp.pad(
      d_b_val.astype(d_b_ref.dtype),
      ((0, 0), (0, padded_num_v_heads - num_v_heads)),
  )
  d_a_ref[...] = jnp.pad(
      d_a_val.astype(d_a_ref.dtype),
      ((0, 0), (0, padded_num_v_heads - num_v_heads)),
  )
  d_a_log_ref[...] = jnp.pad(
      d_a_log_val.astype(d_a_log_ref.dtype)[None, :],
      ((0, 0), (0, padded_num_v_heads - num_v_heads)),
  )
  d_dt_bias_ref[...] = jnp.pad(
      d_dt_bias_val.astype(d_dt_bias_ref.dtype)[None, :],
      ((0, 0), (0, padded_num_v_heads - num_v_heads)),
  )
  d_state_scr[...] = d_state_prev


def pallas_fused_conv1d_gdn_analytical_bwd_computation(
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
    num_v_heads: int,
    kq_head_dim: int,
    v_head_dim: int,
    kernel_size: int,
    chunk_size: int = 64,
    use_qk_norm_in_gdn: bool = False,
    vmem_limit_mb: Optional[int] = None,
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
  """Executes the Pallas reverse-chunk GDNv3 analytical backward kernel using emit_pipeline."""
  del seq_lens, qkv
  if interpret is None and jax.default_backend() == "cpu":
    interpret = True
  if interpret:
    ensure_cpu_interpret_registered()

  batch_size, seq_len, dim_size = pre_conv_qkv.shape
  num_chunks = seq_len // chunk_size

  num_kq_heads = (dim_size - num_v_heads * v_head_dim) // (kq_head_dim * 2)
  padded_num_v_heads = max(num_v_heads, 256)

  b_4d = b.reshape(batch_size, num_chunks, chunk_size, num_v_heads)
  if padded_num_v_heads > num_v_heads:
    b_4d = jnp.pad(
        b_4d, ((0, 0), (0, 0), (0, 0), (0, padded_num_v_heads - num_v_heads))
    )
  a_4d = a.reshape(batch_size, num_chunks, chunk_size, num_v_heads)
  if padded_num_v_heads > num_v_heads:
    a_4d = jnp.pad(
        a_4d, ((0, 0), (0, 0), (0, 0), (0, padded_num_v_heads - num_v_heads))
    )
  do_4d = do.reshape(
      batch_size, num_chunks, chunk_size, num_v_heads, v_head_dim
  )

  if a_log.ndim == 1:
    a_log_3d = jnp.broadcast_to(
        a_log[None, None, :], (batch_size, 1, num_v_heads)
    )
  elif a_log.ndim == 2:
    a_log_3d = a_log[:, None, :]
  else:
    a_log_3d = a_log
  if padded_num_v_heads > num_v_heads:
    a_log_3d = jnp.pad(
        a_log_3d, ((0, 0), (0, 0), (0, padded_num_v_heads - num_v_heads))
    )

  if dt_bias.ndim == 1:
    dt_bias_3d = jnp.broadcast_to(
        dt_bias[None, None, :], (batch_size, 1, num_v_heads)
    )
  elif dt_bias.ndim == 2:
    dt_bias_3d = dt_bias[:, None, :]
  else:
    dt_bias_3d = dt_bias
  if padded_num_v_heads > num_v_heads:
    dt_bias_3d = jnp.pad(
        dt_bias_3d, ((0, 0), (0, 0), (0, padded_num_v_heads - num_v_heads))
    )

  if conv_weight.ndim == 3:
    conv_weight_2d = conv_weight.squeeze(1)
  else:
    conv_weight_2d = conv_weight

  if conv_bias is None:
    conv_bias_1d = jnp.zeros((dim_size,), dtype=conv_weight_2d.dtype)
  else:
    conv_bias_1d = conv_bias.reshape(-1)

  pad_len = max(((kernel_size - 1 + 7) // 8) * 8, 8)
  pre_conv_pad = jnp.zeros(
      (batch_size, pad_len, dim_size), dtype=pre_conv_qkv.dtype
  )
  padded_pre_conv_qkv = jnp.concatenate([pre_conv_pad, pre_conv_qkv], axis=1)

  t_inv_5d = t_inv.astype(jnp.float32).reshape(
      batch_size, num_chunks, num_v_heads, chunk_size, chunk_size
  )

  in_specs, out_specs, nin, nout = make_bwd_block_specs(
      batch_size=batch_size,
      num_chunks=num_chunks,
      chunk_size=chunk_size,
      dim_size=dim_size,
      num_v_heads=num_v_heads,
      kq_head_dim=kq_head_dim,
      v_head_dim=v_head_dim,
      kernel_size=kernel_size,
      pad_len=pad_len,
      padded_num_v_heads=padded_num_v_heads,
  )

  out_shapes = (
      jax.ShapeDtypeStruct(
          (batch_size, num_chunks, chunk_size, dim_size), pre_conv_qkv.dtype
      ),
      jax.ShapeDtypeStruct(b_4d.shape, b_4d.dtype),
      jax.ShapeDtypeStruct(a_4d.shape, a_4d.dtype),
      jax.ShapeDtypeStruct(
          (batch_size, num_chunks, kernel_size, dim_size), conv_weight_2d.dtype
      ),
      jax.ShapeDtypeStruct(
          (batch_size, num_chunks, 1, dim_size), conv_bias_1d.dtype
      ),
      jax.ShapeDtypeStruct(
          (batch_size, num_chunks, 1, padded_num_v_heads), a_log_3d.dtype
      ),
      jax.ShapeDtypeStruct(
          (batch_size, num_chunks, 1, padded_num_v_heads), dt_bias_3d.dtype
      ),
  )

  body = functools.partial(
      _bwd_analytical_pipeline_body,
      chunk_size=chunk_size,
      dim_size=dim_size,
      num_kq_heads=num_kq_heads,
      num_v_heads=num_v_heads,
      padded_num_v_heads=padded_num_v_heads,
      kq_head_dim=kq_head_dim,
      v_head_dim=v_head_dim,
      kernel_size=kernel_size,
      pad_len=pad_len,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
  )

  def outer(*refs):
    pltpu.emit_pipeline(
        body,
        grid=(batch_size, num_chunks),
        in_specs=in_specs,
        out_specs=out_specs,
    )(*refs[: nin + nout], scratches=tuple(refs[nin + nout :]))

  if vmem_limit_mb is not None and vmem_limit_mb <= 64:
    vmem_limit_bytes = int(vmem_limit_mb) * 1024 * 1024
  else:
    tpu_info = pltpu.get_tpu_info()
    vmem_limit_bytes = int(0.85 * tpu_info.vmem_capacity_bytes)

  hbm = pltpu.MemorySpace.HBM
  (
      d_pre_conv_qkv,
      d_b,
      d_a,
      d_conv_weight_chunks,
      d_conv_bias_chunks,
      d_a_log_chunks,
      d_dt_bias_chunks,
  ) = pl.pallas_call(
      outer,
      grid=(),
      out_shape=out_shapes,
      in_specs=[pl.BlockSpec(memory_space=hbm)] * nin,
      out_specs=[pl.BlockSpec(memory_space=hbm)] * nout,
      scratch_shapes=[
          pltpu.VMEM((num_v_heads, kq_head_dim, v_head_dim), jnp.float32),
          pltpu.VMEM((pad_len, dim_size), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          vmem_limit_bytes=vmem_limit_bytes,
          disable_bounds_checks=True,
      ),
      interpret=interpret,
  )(
      padded_pre_conv_qkv,
      b_4d,
      a_4d,
      do_4d,
      chunk_states,
      t_inv_5d,
      a_log_3d,
      dt_bias_3d,
      conv_weight_2d,
      conv_bias_1d,
  )

  d_conv_weight_reduced = jnp.sum(d_conv_weight_chunks, axis=(0, 1))
  d_conv_bias_reduced = jnp.sum(d_conv_bias_chunks[..., 0, :], axis=(0, 1))
  d_a_log_reduced = jnp.sum(
      d_a_log_chunks[..., 0, :num_v_heads], axis=(0, 1)
  ).astype(a_log.dtype)
  d_dt_bias_reduced = jnp.sum(
      d_dt_bias_chunks[..., 0, :num_v_heads], axis=(0, 1)
  ).astype(dt_bias.dtype)

  d_pre_conv_qkv_flat = d_pre_conv_qkv.reshape(
      batch_size, seq_len, dim_size
  ).astype(pre_conv_qkv.dtype)
  d_b_flat = (
      d_b[..., :num_v_heads]
      .reshape(batch_size, seq_len, num_v_heads)
      .astype(b.dtype)
  )
  d_a_flat = (
      d_a[..., :num_v_heads]
      .reshape(batch_size, seq_len, num_v_heads)
      .astype(a.dtype)
  )

  if conv_weight.ndim == 3:
    d_conv_weight_out = d_conv_weight_reduced[:, None, :].astype(
        conv_weight.dtype
    )
  else:
    d_conv_weight_out = d_conv_weight_reduced.astype(conv_weight.dtype)

  if conv_bias is None:
    d_conv_bias_out = None
  else:
    d_conv_bias_out = d_conv_bias_reduced.reshape(conv_bias.shape).astype(
        conv_bias.dtype
    )

  return (
      d_pre_conv_qkv_flat,
      d_b_flat,
      d_a_flat,
      d_conv_weight_out,
      d_conv_bias_out,
      d_a_log_reduced,
      d_dt_bias_reduced,
  )


def pure_jax_fused_conv1d_gdn(
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
    compute_dtype: jnp.dtype = jnp.float32,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
  """Pure-JAX composite of Conv1D + GDN used during backward pass autodiff."""
  del conv_state
  batch, seq_len, _ = qkv.shape
  key_dim = num_k_heads * head_k_dim

  # Conv1D in FP32
  conv_input = jnp.pad(
      qkv.astype(jnp.float32), ((0, 0), (conv_kernel_size - 1, 0), (0, 0))
  )
  conv_weight_cast = conv_weight.astype(jnp.float32)
  conv_out = jax.lax.conv_general_dilated(
      lhs=conv_input,
      rhs=conv_weight_cast,
      window_strides=(1,),
      padding="VALID",
      dimension_numbers=("NWC", "WIO", "NWC"),
      feature_group_count=qkv.shape[-1],
  )
  if conv_bias is not None:
    conv_out = conv_out + conv_bias.astype(jnp.float32)
  conv_out = conv_out[:, -seq_len:, :]
  qkv_conv = jax.nn.silu(conv_out).astype(jnp.float32)

  q_conv, k_conv, v_conv = jnp.split(qkv_conv, [key_dim, 2 * key_dim], axis=-1)

  # Reshape for GDN
  query = q_conv.reshape(batch, seq_len, num_k_heads, head_k_dim)
  key = k_conv.reshape(batch, seq_len, num_k_heads, head_k_dim)
  value = v_conv.reshape(batch, seq_len, num_v_heads, head_v_dim)

  a_log_cast = jnp.asarray(a_log, dtype=jnp.float32)
  dt_bias_cast = jnp.asarray(dt_bias, dtype=jnp.float32)
  beta = jax.nn.sigmoid(b.astype(jnp.float32))
  g = -jnp.exp(a_log_cast) * jax.nn.softplus(
      a.astype(jnp.float32) + dt_bias_cast
  )

  if num_v_heads > num_k_heads and num_v_heads % num_k_heads == 0:
    repeats = num_v_heads // num_k_heads
    query = jnp.repeat(query, repeats, axis=2)
    key = jnp.repeat(key, repeats, axis=2)

  core_attn_out, next_recurrent_state = qwen3.jax_chunk_gated_delta_rule(
      query=query,
      key=key,
      value=value,
      g=g,
      beta=beta,
      chunk_size=chunk_size,
      initial_state=(
          recurrent_state.astype(jnp.float32)
          if recurrent_state is not None
          else None
      ),
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      compute_dtype=jnp.float32,
  )

  next_conv_state = (
      qkv[:, -(conv_kernel_size - 1) :, :]
      if seq_len >= conv_kernel_size - 1
      else jnp.zeros(
          (batch, conv_kernel_size - 1, qkv.shape[-1]), dtype=qkv.dtype
      )
  )
  if next_recurrent_state is None:
    next_recurrent_state = jnp.zeros(
        (batch, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32
    )

  return core_attn_out.astype(qkv.dtype), (
      next_conv_state.astype(qkv.dtype),
      next_recurrent_state.astype(qkv.dtype),
  )


def _compute_forward_conv_and_states(
    qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    a_log: jax.Array,
    dt_bias: jax.Array,
    recurrent_state: Optional[jax.Array],
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool = False,
    compute_dtype: jnp.dtype = jnp.float32,
    cached_t_inv: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """Computes convolved QKV, inter-chunk states, and t_inv matrices in FP32."""
  del compute_dtype
  batch_size, seq_len, dim_size = qkv.shape
  num_chunks = seq_len // chunk_size

  # Conv1D in FP32
  conv_input = jnp.pad(
      qkv.astype(jnp.float32), ((0, 0), (conv_kernel_size - 1, 0), (0, 0))
  )
  conv_out = jax.lax.conv_general_dilated(
      lhs=conv_input,
      rhs=conv_weight.astype(jnp.float32),
      window_strides=(1,),
      padding="VALID",
      dimension_numbers=("NWC", "WIO", "NWC"),
      feature_group_count=dim_size,
  )
  if conv_bias is not None:
    conv_out = conv_out + conv_bias.astype(jnp.float32)
  conv_out = conv_out[:, -seq_len:, :]
  qkv_conv_f32 = jax.nn.silu(conv_out).astype(jnp.float32)

  # Chunk states progression in FP32
  num_kq_heads = num_k_heads
  q_size = num_kq_heads * head_k_dim
  k_size = num_kq_heads * head_k_dim
  repeats = num_v_heads // num_kq_heads

  q = qkv_conv_f32[:, :, :q_size].reshape(
      batch_size, num_chunks, chunk_size, num_kq_heads, head_k_dim
  )
  k = qkv_conv_f32[:, :, q_size : q_size + k_size].reshape(
      batch_size, num_chunks, chunk_size, num_kq_heads, head_k_dim
  )
  v = qkv_conv_f32[:, :, q_size + k_size :].reshape(
      batch_size, num_chunks, chunk_size, num_v_heads, head_v_dim
  )

  b_4d = b.astype(jnp.float32).reshape(
      batch_size, num_chunks, chunk_size, num_v_heads
  )
  a_4d = a.astype(jnp.float32).reshape(
      batch_size, num_chunks, chunk_size, num_v_heads
  )
  a_log_f32 = a_log.astype(jnp.float32)
  dt_bias_f32 = dt_bias.astype(jnp.float32)

  if recurrent_state is None:
    init_state = jnp.zeros(
        (batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32
    )
  else:
    init_state = recurrent_state.astype(jnp.float32)

  k_chunks = k.swapaxes(0, 1)
  v_chunks = v.swapaxes(0, 1)
  b_chunks = b_4d.swapaxes(0, 1)
  a_chunks = a_4d.swapaxes(0, 1)

  if cached_t_inv is not None:
    t_inv_chunks = cached_t_inv.astype(jnp.float32).swapaxes(0, 1)

    def chunk_cached_step(
        k_single, v_single, b_single, a_single, s_prev, t_inv_single
    ):
      return chunk_state_forward_with_cached_tinv(
          k=k_single,
          v=v_single,
          b_val=b_single,
          a_val=a_single,
          a_log_val=a_log_f32,
          dt_bias_val=dt_bias_f32,
          state_prev=s_prev,
          t_inv=t_inv_single,
          repeats=repeats,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      )

    def scan_fn_cached(carry_state, chunk_inputs):
      k_i, v_i, b_i, a_i, t_inv_i = chunk_inputs
      next_state = jax.vmap(chunk_cached_step)(
          k_i, v_i, b_i, a_i, carry_state, t_inv_i
      )
      return next_state, carry_state

    _, chunk_states = jax.lax.scan(
        scan_fn_cached,
        init_state,
        (k_chunks, v_chunks, b_chunks, a_chunks, t_inv_chunks),
    )
    chunk_states = chunk_states.swapaxes(0, 1)
    t_inv_all = cached_t_inv
  else:

    def chunk_step(q_single, k_single, v_single, b_single, a_single, s_prev):
      return chunk_forward_with_tinv(
          q_single,
          k_single,
          v_single,
          b_single,
          a_single,
          a_log_f32,
          dt_bias_f32,
          s_prev,
          kq_head_dim=head_k_dim,
          repeats=repeats,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=use_qk_norm_in_gdn,
      )

    def scan_fn(carry_state, chunk_inputs):
      q_i, k_i, v_i, b_i, a_i = chunk_inputs
      _, next_state, t_inv_i = jax.vmap(chunk_step)(
          q_i, k_i, v_i, b_i, a_i, carry_state
      )
      return next_state, (carry_state, t_inv_i)

    q_chunks = q.swapaxes(0, 1)
    _, (chunk_states, t_inv_all) = jax.lax.scan(
        scan_fn, init_state, (q_chunks, k_chunks, v_chunks, b_chunks, a_chunks)
    )
    chunk_states = chunk_states.swapaxes(0, 1)
    t_inv_all = t_inv_all.swapaxes(0, 1)

  return qkv_conv_f32.astype(qkv.dtype), chunk_states, t_inv_all


def _run_local_gdn_fused_fwd(
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
) -> Tuple[
    Tuple[jax.Array, Tuple[jax.Array, jax.Array]],
    Optional[jax.Array],
    Optional[jax.Array],
]:
  """Runs local GDN fused forward pass on TPU returning (t_inv, chunk_states), or pure JAX on CPU."""
  if jax.extend.backend.get_backend().platform == "cpu":
    out, states = pure_jax_fused_conv1d_gdn(
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
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        compute_dtype=compute_dtype,
    )
    return (out, states), t_inv, chunk_states

  batch_size, seq_len, dim_size = qkv.shape
  num_seqs = batch_size
  num_chunks = seq_len // chunk_size

  qkv_flat = qkv.reshape(-1, dim_size)
  b_flat = b.reshape(-1, b.shape[-1])
  a_flat = a.reshape(-1, a.shape[-1])
  tokamax_conv_weight = jnp.swapaxes(conv_weight, 0, 2)

  query_start_loc = jnp.arange(
      0, (num_seqs + 1) * seq_len, seq_len, dtype=jnp.int32
  )
  state_indices = jnp.arange(num_seqs, dtype=jnp.int32)
  seq_lens = jnp.full((num_seqs,), seq_len, dtype=jnp.int32)
  distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

  if conv_state is None:
    tokamax_conv_state = jnp.zeros(
        (num_seqs + 1, conv_kernel_size - 1, dim_size), dtype=qkv.dtype
    )
  elif conv_state.shape[0] == num_seqs:
    tokamax_conv_state = jnp.pad(conv_state, ((1, 0), (0, 0), (0, 0)))
  else:
    tokamax_conv_state = conv_state

  if recurrent_state is None:
    tokamax_recurrent_state = jnp.zeros(
        (num_seqs + 1, num_v_heads, head_k_dim, head_v_dim), dtype=qkv.dtype
    )
  elif recurrent_state.shape[0] == num_seqs:
    tokamax_recurrent_state = jnp.pad(
        recurrent_state, ((1, 0), (0, 0), (0, 0), (0, 0))
    )
  else:
    tokamax_recurrent_state = recurrent_state

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
      compute_precision=jnp.dtype(jnp.float32),
      mixed_tile_size=chunk_size,
  )

  core_attn_out = core_attn_out_flat.reshape(
      batch_size, seq_len, num_v_heads, head_v_dim
  )
  t_inv = t_inv_raw.astype(jnp.float32).reshape(
      batch_size, num_chunks, num_v_heads, chunk_size, chunk_size
  )
  chunk_states = chunk_states_raw.astype(jnp.float32).reshape(
      batch_size, num_chunks, num_v_heads, head_k_dim, head_v_dim
  )
  return (
      (
          core_attn_out.astype(qkv.dtype),
          (
              new_conv_state[1:].astype(qkv.dtype),
              new_recurrent_state[1:].astype(qkv.dtype),
          ),
      ),
      t_inv,
      chunk_states,
  )


@functools.partial(
    jax.custom_vjp, nondiff_argnums=(9, 10, 11, 12, 13, 14, 15, 16)
)
def hybrid_fused_conv1d_gdn_analytical(
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
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
  """Hybrid Fused Conv1D + GDN with manual analytical backward pass."""
  (out, states), _, _ = _run_local_gdn_fused_fwd(
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
  )
  return out, states


def _hybrid_fused_conv1d_gdn_analytical_fwd(
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
):
  (out, states), t_inv, chunk_states = _run_local_gdn_fused_fwd(
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
  )
  residuals = (
      qkv,
      b,
      a,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      conv_state,
      recurrent_state,
      t_inv,
      chunk_states,
  )
  return (out, states), residuals


def _hybrid_fused_conv1d_gdn_analytical_bwd(
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
    chunk_size: int,
    use_qk_norm_in_gdn: bool,
    compute_dtype: jnp.dtype,
    residuals: tuple,
    cotangents: tuple,
):
  if len(residuals) == 11:
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

  d_out, d_states = cotangents
  d_conv_state, d_recurrent_state = d_states
  del d_conv_state, d_recurrent_state

  # Recompute forward chunk states and t_inv if not cached in residuals
  if chunk_states is None or t_inv_fwd is None:
    _, chunk_states_recomputed, t_inv_recomputed = (
        _compute_forward_conv_and_states(
            qkv=pre_conv_qkv,
            b=b,
            a=a,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            a_log=a_log,
            dt_bias=dt_bias,
            recurrent_state=recurrent_state,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            chunk_size=chunk_size,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            compute_dtype=compute_dtype,
            cached_t_inv=t_inv_fwd,
        )
    )
    if chunk_states is None:
      chunk_states = chunk_states_recomputed
    if t_inv_fwd is None:
      t_inv_fwd = t_inv_recomputed
  t_inv = t_inv_fwd

  (
      d_pre_conv_qkv,
      d_b,
      d_a,
      d_conv_weight,
      d_conv_bias,
      d_a_log,
      d_dt_bias,
  ) = pallas_fused_conv1d_gdn_analytical_bwd_computation(
      pre_conv_qkv=pre_conv_qkv,
      b=b,
      a=a,
      a_log=a_log,
      dt_bias=dt_bias,
      do=d_out,
      chunk_states=chunk_states,
      t_inv=t_inv,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      num_v_heads=num_v_heads,
      kq_head_dim=head_k_dim,
      v_head_dim=head_v_dim,
      kernel_size=conv_kernel_size,
      chunk_size=chunk_size,
      use_qk_norm_in_gdn=use_qk_norm_in_gdn,
  )
  d_conv_state_out = None if conv_state is None else jnp.zeros_like(conv_state)
  d_recurrent_state_out = (
      None if recurrent_state is None else jnp.zeros_like(recurrent_state)
  )
  return (
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


hybrid_fused_conv1d_gdn_analytical.defvjp(
    _hybrid_fused_conv1d_gdn_analytical_fwd,
    _hybrid_fused_conv1d_gdn_analytical_bwd,
)

__all__ = [
    "chunk_forward",
    "chunk_forward_with_tinv",
    "pallas_fused_conv1d_gdn_analytical_bwd_computation",
    "pure_jax_fused_conv1d_gdn",
    "hybrid_fused_conv1d_gdn_analytical",
    "ensure_cpu_interpret_registered",
]

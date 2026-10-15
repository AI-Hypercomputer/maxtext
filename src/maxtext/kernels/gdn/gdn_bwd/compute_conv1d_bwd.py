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

"""Conv1D + SiLU forward and unrolled shift-multiply backward primitives."""

import contextlib
import contextvars
from typing import Iterator, Optional, Tuple

import jax
import jax.numpy as jnp

# gdn_conv1d_input_in_qkv_dtype (see the comment above _conv1d_silu_fwd_narrow_input at the end of this file).
# conv1d_silu_fwd / conv1d_silu_bwd take it as an argument; left at None they read this trace-time setting, which
# gdn_decoupled_conv1d (gdn_bwd/api.py) scopes around its primal, forward and backward rules from its own static
# argument. A scoped setting rather than an argument on every internal helper because the backward rule is traced
# when the cotangent is pulled back, after the model's forward has returned, so only the custom_vjp's static
# arguments reach it; and the rules reach these functions through several helpers (jax_compute_gdn_states, the CP
# forward) that do not otherwise need it.
_CONV_INPUT_IN_QKV_DTYPE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "gdn_conv1d_input_in_qkv_dtype", default=False
)


@contextlib.contextmanager
def conv_input_in_qkv_dtype_scope(enabled: bool) -> Iterator[None]:
  """Scopes the default of conv1d_silu_fwd / conv1d_silu_bwd's `conv_input_in_qkv_dtype` while tracing."""
  token = _CONV_INPUT_IN_QKV_DTYPE.set(bool(enabled))
  try:
    yield
  finally:
    _CONV_INPUT_IN_QKV_DTYPE.reset(token)


def _resolve_conv_input_in_qkv_dtype(conv_input_in_qkv_dtype: Optional[bool]) -> bool:
  if conv_input_in_qkv_dtype is None:
    return _CONV_INPUT_IN_QKV_DTYPE.get()
  return bool(conv_input_in_qkv_dtype)


def _build_conv_tap_masks(
    segment_ids: jax.Array,
    kernel_size: int,
    conv_halo_seg: Optional[jax.Array] = None,
) -> Tuple[jax.Array, list[jax.Array]]:
  """Builds (valid_mask, tap_masks) for causal Conv1D with sequence packing."""
  _, seq_len = segment_ids.shape
  halo_len = kernel_size - 1
  seg_pos = jnp.maximum(segment_ids.astype(jnp.int32), 0)
  valid_mask = (seg_pos > 0)[:, :, None].astype(jnp.float32)

  if conv_halo_seg is not None:
    halo_pos = jnp.maximum(conv_halo_seg.astype(jnp.int32).reshape(seg_pos.shape[0], halo_len), 0)
    full_seg = jnp.concatenate([halo_pos, seg_pos], axis=1)
  else:
    full_seg = jnp.pad(seg_pos, ((0, 0), (halo_len, 0)))

  tap_masks = [
      ((seg_pos > 0) & (full_seg[:, k : k + seq_len] == seg_pos))[:, :, None].astype(jnp.float32)
      for k in range(kernel_size)
  ]
  return valid_mask, tap_masks


def conv1d_silu_fwd(
    qkv: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    kernel_size: int,
    conv_state: Optional[jax.Array] = None,
    segment_ids: Optional[jax.Array] = None,
    conv_halo_seg: Optional[jax.Array] = None,
    output_dtype: Optional[jnp.dtype] = None,
    conv_input_in_qkv_dtype: Optional[bool] = None,
) -> Tuple[jax.Array, jax.Array]:
  """Forward Conv1D + SiLU returning (conv_out, qkv_conv).

  `conv_input_in_qkv_dtype` (None: the conv_input_in_qkv_dtype_scope() setting, default False) keeps the halo-extended input
  in qkv's dtype and casts each tap to float32 where it is read. Same values; see _conv1d_silu_fwd_narrow_input.
  """
  if _resolve_conv_input_in_qkv_dtype(conv_input_in_qkv_dtype):
    return _conv1d_silu_fwd_narrow_input(
        qkv, conv_weight, conv_bias, kernel_size, conv_state, segment_ids, conv_halo_seg, output_dtype
    )
  _, seq_len, _ = qkv.shape
  if conv_weight.ndim == 3:
    conv_weight_3d = conv_weight.astype(jnp.float32)
  else:
    conv_weight_3d = conv_weight[:, None, :].astype(jnp.float32)

  if conv_state is not None:
    conv_input = jnp.concatenate([conv_state.astype(jnp.float32), qkv.astype(jnp.float32)], axis=1)
  else:
    conv_input = jnp.pad(qkv.astype(jnp.float32), ((0, 0), (kernel_size - 1, 0), (0, 0)))
  if segment_ids is not None:
    valid_mask, tap_masks = _build_conv_tap_masks(segment_ids, kernel_size, conv_halo_seg)
    conv_out = sum(
        (conv_input[:, k : k + seq_len, :] * tap_masks[k]) * conv_weight_3d[k, 0, :] for k in range(kernel_size)
    )
    if conv_bias is not None:
      conv_out = conv_out + conv_bias.astype(jnp.float32)
    conv_out = conv_out * valid_mask
  else:
    conv_out = sum(conv_input[:, k : k + seq_len, :] * conv_weight_3d[k, 0, :] for k in range(kernel_size))
    if conv_bias is not None:
      conv_out = conv_out + conv_bias.astype(jnp.float32)
  qkv_conv = jax.nn.silu(conv_out)
  return conv_out, qkv_conv.astype(output_dtype or qkv.dtype)


def conv1d_silu_bwd(
    qkv: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    dy: jax.Array,
    kernel_size: int,
    conv_out: Optional[jax.Array] = None,
    conv_state: Optional[jax.Array] = None,
    return_d_conv_state: bool = False,
    segment_ids: Optional[jax.Array] = None,
    conv_halo_seg: Optional[jax.Array] = None,
    conv_input_in_qkv_dtype: Optional[bool] = None,
):
  """Dedicated Conv1D + SiLU backward pass using JAX primitives.

  `conv_input_in_qkv_dtype` (None: the conv_input_in_qkv_dtype_scope() setting, default False) keeps the halo-extended input
  in qkv's dtype and casts each tap to float32 where it is read. Same values; see _conv1d_silu_bwd_narrow_input.
  """
  if _resolve_conv_input_in_qkv_dtype(conv_input_in_qkv_dtype):
    return _conv1d_silu_bwd_narrow_input(
        qkv, conv_weight, conv_bias, dy, kernel_size, conv_out, conv_state, return_d_conv_state, segment_ids,
        conv_halo_seg
    )
  _, seq_len, _ = qkv.shape
  if conv_weight.ndim == 3:
    conv_weight_3d = conv_weight.astype(jnp.float32)
  else:
    conv_weight_3d = conv_weight[:, None, :].astype(jnp.float32)

  if conv_state is not None:
    conv_input = jnp.concatenate([conv_state.astype(jnp.float32), qkv.astype(jnp.float32)], axis=1)
  else:
    conv_input = jnp.pad(qkv.astype(jnp.float32), ((0, 0), (kernel_size - 1, 0), (0, 0)))

  valid_mask = None
  tap_masks = None
  if segment_ids is not None:
    valid_mask, tap_masks = _build_conv_tap_masks(segment_ids, kernel_size, conv_halo_seg)

  if conv_out is None:
    # 1. Forward pass: z = conv1d(x) + b
    if segment_ids is not None and tap_masks is not None and valid_mask is not None:
      conv_out = sum(
          (conv_input[:, k : k + seq_len, :] * tap_masks[k]) * conv_weight_3d[k, 0, :] for k in range(kernel_size)
      )
      if conv_bias is not None:
        conv_out = conv_out + conv_bias.astype(jnp.float32)
      conv_out = conv_out * valid_mask
    else:
      conv_out = sum(conv_input[:, k : k + seq_len, :] * conv_weight_3d[k, 0, :] for k in range(kernel_size))
      if conv_bias is not None:
        conv_out = conv_out + conv_bias.astype(jnp.float32)
  z = conv_out.astype(jnp.float32)

  # 2. Adjoint: dz = dy * SiLU'(z)
  sig_z = jax.nn.sigmoid(z)
  silu_prime = sig_z * (1.0 + z * (1.0 - sig_z))
  dz = dy.astype(jnp.float32) * silu_prime
  if valid_mask is not None:
    dz = dz * valid_mask

  # 3. Parameter gradients:
  # Bias gradient: db = sum(dz)
  if conv_bias is not None:
    db = jnp.sum(dz, axis=(0, 1)).astype(conv_bias.dtype)
    if conv_bias.ndim != 1:
      db = db.reshape(conv_bias.shape)
  else:
    db = None

  # Weight gradient: dw[k] = sum_{b,t} dz[b,t] * conv_input[b, t+k]
  dw_rows = []
  for k in range(kernel_size):
    x_k = conv_input[:, k : k + seq_len, :]
    dz_k = dz * tap_masks[k] if tap_masks is not None else dz
    dw_rows.append(jnp.sum(dz_k * x_k, axis=(0, 1)))
  dw = jnp.stack(dw_rows, axis=0)
  if conv_weight.ndim == 3:
    dw = dw[:, None, :].astype(conv_weight.dtype)
  else:
    dw = dw.astype(conv_weight.dtype)

  # 4. Input gradient: dx = transposed convolution of dz with reversed w
  dz_pad = jnp.pad(dz, ((0, 0), (0, kernel_size - 1), (0, 0)))
  w_rev = conv_weight_3d[::-1]
  if tap_masks is not None:
    tap_masks_pad = [jnp.pad(tm, ((0, 0), (0, kernel_size - 1), (0, 0))) for tm in tap_masks]
    dx = sum(
        (dz_pad[:, j : j + seq_len, :] * tap_masks_pad[kernel_size - 1 - j][:, j : j + seq_len, :]) * w_rev[j, 0, :]
        for j in range(kernel_size)
    )
  else:
    dx = sum(dz_pad[:, k : k + seq_len, :] * w_rev[k, 0, :] for k in range(kernel_size))
  dx = dx.astype(qkv.dtype)

  if return_d_conv_state:
    d_cs_list = []
    for m in range(kernel_size - 1):
      val = jnp.zeros((qkv.shape[0], qkv.shape[2]), dtype=jnp.float32)
      for t in range(m + 1):
        k_idx = m - t
        dz_t = dz[:, t, :] * tap_masks[k_idx][:, t, :] if tap_masks is not None else dz[:, t, :]
        val = val + dz_t * conv_weight_3d[k_idx, 0, :]
      d_cs_list.append(val)
    d_conv_state = jnp.stack(d_cs_list, axis=1).astype(qkv.dtype)
    return dx, dw, db, d_conv_state

  return dx, dw, db


# --------------------------------------------------------------------------------------------------------------------
# gdn_conv1d_input_in_qkv_dtype: the halo-extended conv input in qkv's dtype.
#
# The default path above builds conv_input = [conv_state | qkv] (or qkv left-padded by kernel_size - 1 zeros) in
# float32 and slices the kernel_size taps out of it. That materializes a float32 copy of the whole input: on the
# Qwen3.5-397B RL trainer (bf16 qkv, sequence-sharded context parallelism) one f32[1, 16387, 12288] = 805.5 MB buffer
# per GDN layer in the backward pass, written once and read kernel_size times.
#
# The functions below build conv_input in qkv's dtype (bf16 for bf16 qkv and conv_state; the promoted dtype if they
# differ) and cast each tap slice to float32 where it is read, so XLA fuses the cast into the consumer and the
# float32 copy is never written. Every product and sum is formed in the same order, on the same float32 values, as in
# the default path (bf16 -> float32 is exact), so op by op every output is bit-identical; only which tensors are
# materialized in HBM changes. Compiled, the forward outputs and dx / d_conv_state / db stay bit-identical, while the
# weight gradient dw, a float32 reduction over batch x sequence, can come out in a different summation order when XLA
# fuses the tap's cast into the reduction (CPU: a few ulp, only when the pre-activation is recomputed without packing
# or a conv state). No jax.lax.optimization_barrier is used: the narrow buffer alone is what removes the copy.
# Tests: tests/unit/gdn_conv1d_input_in_qkv_dtype_test.py.


def _conv_input_narrow(qkv, conv_state, kernel_size):
  """Halo-extended conv input in the narrowest exact dtype (bf16 for bf16 qkv and conv_state)."""
  if conv_state is not None:
    cdt = jnp.promote_types(qkv.dtype, conv_state.dtype)
    return jnp.concatenate([conv_state.astype(cdt), qkv.astype(cdt)], axis=1)
  return jnp.pad(qkv, ((0, 0), (kernel_size - 1, 0), (0, 0)))


def _conv_weight_3d_f32(conv_weight):
  return conv_weight.astype(jnp.float32) if conv_weight.ndim == 3 else conv_weight[:, None, :].astype(jnp.float32)


def _preact_from_narrow_input(conv_input, w3d, conv_bias, kernel_size, seq_len, tap_masks, valid_mask):
  """f32 pre-activation z with the default path's op order: sum over taps k of (x_k * mask_k) * w_k, + b, * valid."""
  if tap_masks is not None:
    z = sum(
        (conv_input[:, k : k + seq_len, :].astype(jnp.float32) * tap_masks[k]) * w3d[k, 0, :] for k in range(kernel_size)
    )
    if conv_bias is not None:
      z = z + conv_bias.astype(jnp.float32)
    return z * valid_mask
  z = sum(conv_input[:, k : k + seq_len, :].astype(jnp.float32) * w3d[k, 0, :] for k in range(kernel_size))
  if conv_bias is not None:
    z = z + conv_bias.astype(jnp.float32)
  return z


def _conv1d_silu_fwd_narrow_input(
    qkv, conv_weight, conv_bias, kernel_size, conv_state, segment_ids, conv_halo_seg, output_dtype
):
  """conv1d_silu_fwd reading the conv input in qkv's dtype. Returns (f32 conv_out, qkv_conv) like the default path."""
  _, seq_len, _ = qkv.shape
  w3d = _conv_weight_3d_f32(conv_weight)
  conv_input = _conv_input_narrow(qkv, conv_state, kernel_size)
  valid_mask = tap_masks = None
  if segment_ids is not None:
    valid_mask, tap_masks = _build_conv_tap_masks(segment_ids, kernel_size, conv_halo_seg)
  conv_out = _preact_from_narrow_input(conv_input, w3d, conv_bias, kernel_size, seq_len, tap_masks, valid_mask)
  qkv_conv = jax.nn.silu(conv_out)
  return conv_out, qkv_conv.astype(output_dtype or qkv.dtype)


def _conv1d_silu_bwd_narrow_input(
    qkv, conv_weight, conv_bias, dy, kernel_size, conv_out, conv_state, return_d_conv_state, segment_ids, conv_halo_seg
):
  """conv1d_silu_bwd reading the conv input in qkv's dtype; the same steps 1-4 as the default path."""
  _, seq_len, _ = qkv.shape
  w3d = _conv_weight_3d_f32(conv_weight)
  conv_input = _conv_input_narrow(qkv, conv_state, kernel_size)
  valid_mask = tap_masks = None
  if segment_ids is not None:
    valid_mask, tap_masks = _build_conv_tap_masks(segment_ids, kernel_size, conv_halo_seg)
  if conv_out is None:
    conv_out = _preact_from_narrow_input(conv_input, w3d, conv_bias, kernel_size, seq_len, tap_masks, valid_mask)
  z = conv_out.astype(jnp.float32)

  # dz = dy * SiLU'(z), as in the default path.
  sig_z = jax.nn.sigmoid(z)
  silu_prime = sig_z * (1.0 + z * (1.0 - sig_z))
  dz = dy.astype(jnp.float32) * silu_prime
  if valid_mask is not None:
    dz = dz * valid_mask

  if conv_bias is not None:
    db = jnp.sum(dz, axis=(0, 1)).astype(conv_bias.dtype)
    if conv_bias.ndim != 1:
      db = db.reshape(conv_bias.shape)
  else:
    db = None

  # dw[k] = sum_{b,t} dz[b,t] * conv_input[b, t+k], each tap cast to float32 where it is read.
  dw_rows = []
  for k in range(kernel_size):
    x_k = conv_input[:, k : k + seq_len, :].astype(jnp.float32)
    dz_k = dz * tap_masks[k] if tap_masks is not None else dz
    dw_rows.append(jnp.sum(dz_k * x_k, axis=(0, 1)))
  dw = jnp.stack(dw_rows, axis=0)
  dw = dw[:, None, :].astype(conv_weight.dtype) if conv_weight.ndim == 3 else dw.astype(conv_weight.dtype)

  # dx: transposed convolution of dz with the reversed weights (does not read conv_input).
  dz_pad = jnp.pad(dz, ((0, 0), (0, kernel_size - 1), (0, 0)))
  w_rev = w3d[::-1]
  if tap_masks is not None:
    tap_masks_pad = [jnp.pad(tm, ((0, 0), (0, kernel_size - 1), (0, 0))) for tm in tap_masks]
    dx = sum(
        (dz_pad[:, j : j + seq_len, :] * tap_masks_pad[kernel_size - 1 - j][:, j : j + seq_len, :]) * w_rev[j, 0, :]
        for j in range(kernel_size)
    )
  else:
    dx = sum(dz_pad[:, k : k + seq_len, :] * w_rev[k, 0, :] for k in range(kernel_size))
  dx = dx.astype(qkv.dtype)

  if return_d_conv_state:
    d_cs_list = []
    for m in range(kernel_size - 1):
      val = jnp.zeros((qkv.shape[0], qkv.shape[2]), dtype=jnp.float32)
      for t in range(m + 1):
        k_idx = m - t
        dz_t = dz[:, t, :] * tap_masks[k_idx][:, t, :] if tap_masks is not None else dz[:, t, :]
        val = val + dz_t * w3d[k_idx, 0, :]
      d_cs_list.append(val)
    d_conv_state = jnp.stack(d_cs_list, axis=1).astype(qkv.dtype)
    return dx, dw, db, d_conv_state
  return dx, dw, db

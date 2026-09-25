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

from typing import Optional, Tuple

import jax
import jax.numpy as jnp


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
) -> Tuple[jax.Array, jax.Array]:
  """Forward Conv1D + SiLU returning (conv_out, qkv_conv)."""
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
  return conv_out, qkv_conv.astype(qkv.dtype)


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
):
  """Dedicated Conv1D + SiLU backward pass using JAX primitives."""
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

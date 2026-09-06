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


def conv1d_silu_fwd(
    qkv: jax.Array,
    conv_weight: jax.Array,
    conv_bias: Optional[jax.Array],
    kernel_size: int,
) -> Tuple[jax.Array, jax.Array]:
  """Forward Conv1D + SiLU returning (conv_out, qkv_conv)."""
  _, seq_len, _ = qkv.shape
  if conv_weight.ndim == 3:
    conv_weight_3d = conv_weight.astype(jnp.float32)
  else:
    conv_weight_3d = conv_weight[:, None, :].astype(jnp.float32)

  conv_input = jnp.pad(qkv.astype(jnp.float32), ((0, 0), (kernel_size - 1, 0), (0, 0)))
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
) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
  """Dedicated Conv1D + SiLU backward pass using JAX primitives."""
  _, seq_len, _ = qkv.shape
  if conv_weight.ndim == 3:
    conv_weight_3d = conv_weight.astype(jnp.float32)
  else:
    conv_weight_3d = conv_weight[:, None, :].astype(jnp.float32)

  # 1. Forward pass: z = conv1d(x) + b
  conv_input = jnp.pad(qkv.astype(jnp.float32), ((0, 0), (kernel_size - 1, 0), (0, 0)))
  conv_out = sum(conv_input[:, k : k + seq_len, :] * conv_weight_3d[k, 0, :] for k in range(kernel_size))
  if conv_bias is not None:
    conv_out = conv_out + conv_bias.astype(jnp.float32)
  z = conv_out

  # 2. Adjoint: dz = dy * SiLU'(z)
  sig_z = jax.nn.sigmoid(z)
  silu_prime = sig_z * (1.0 + z * (1.0 - sig_z))
  dz = dy.astype(jnp.float32) * silu_prime

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
    dw_rows.append(jnp.sum(dz * x_k, axis=(0, 1)))
  dw = jnp.stack(dw_rows, axis=0)
  if conv_weight.ndim == 3:
    dw = dw[:, None, :].astype(conv_weight.dtype)
  else:
    dw = dw.astype(conv_weight.dtype)

  # 4. Input gradient: dx = transposed convolution of dz with reversed w
  dz_pad = jnp.pad(dz, ((0, 0), (0, kernel_size - 1), (0, 0)))
  w_rev = conv_weight_3d[::-1]
  dx = sum(dz_pad[:, k : k + seq_len, :] * w_rev[k, 0, :] for k in range(kernel_size))
  dx = dx.astype(qkv.dtype)

  return dx, dw, db

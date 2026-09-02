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
# ==============================================================================

"""In-VMEM causal depthwise Conv1D computation."""

import jax
import jax.numpy as jnp

try:
  from maxtext.models.kernels.gdn import config
except (ImportError, ModuleNotFoundError):
  try:
    from maxtext.src.maxtext.models.kernels.gdn import config
  except (ImportError, ModuleNotFoundError):
    from . import config


def causal_conv1d(
    real_sizes: jax.Array,  # [seq]
    lhs: jax.Array,  # [seq, chunk, q, dim_size]
    conv_weight: jax.Array,  # [prev_kernel_size, 1, dim_size]
    conv_bias: jax.Array | None,  # [dim_size]
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
  """Perform causal Conv1D. Returns Conv1D output and convolution states."""

  assert lhs.ndim == 4

  out_list = []

  for c_idx in range(cfg.chunk_size):
    out = jnp.zeros((cfg.seq_tile_size, 1, cfg.dim_size), jnp.float32)

    end_idx = c_idx + cfg.prev_kernel_size
    start_idx = 1 + end_idx - cfg.kernel_size
    for k in range(cfg.kernel_size):
      lhs_curr = lhs[:, start_idx + k]
      out += lhs_curr * conv_weight[k : k + 1]

    if conv_bias is not None:
      out += conv_bias.reshape(1, 1, -1)

    out_list.append(out)

  # Last prev_kernel_size elements needs to be returned as conv_state. However,
  # real_sizes may be smaller than chunk_size. Therefore, slicing last
  # prev_kernel_size elements does not gurantee numeric correctness. Instead,
  # kernel iterate each rows and perform masking to fetch correct values.
  # NOTE: lhs[:, : prev_kernel_size] can be skipped since they were loaded from
  # previous conv states.
  new_conv_state = lhs[:, 1 : cfg.kernel_size]
  real_sizes = real_sizes.reshape(-1, 1, 1, 1)
  # NOTE: Even though for loop is invoked twice, since they are static loops,
  # compiler will perform loop fusion.
  for c_idx in range(2, cfg.chunk_size + 1):
    row_end = c_idx + cfg.prev_kernel_size
    new_conv_state = jnp.where(
        c_idx == real_sizes,
        lhs[:, c_idx:row_end],
        new_conv_state,
    )

  return jnp.stack(out_list, axis=1), new_conv_state

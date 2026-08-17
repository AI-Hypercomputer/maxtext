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

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels.causal_conv1d_gated_delta_rule import config
from maxtext.kernels.causal_conv1d_gated_delta_rule import memory_ref


def load_as_qkv_large(
    qkv_vmem_ref: jax.Ref, cfgs: config.GDNConfig
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Split qkv and transpose by performing 1 load per chunk for large layout.

  Args:
    qkv_vmem_ref: qkv reference in VMEM containing concatenated values of q, k,
      and v of shape [seq_tile_size, chunk_size, 1, num_kq_heads * kq_head_dim *
      2 + num_v_heads * v_head_dim].
    cfgs: GDN configuration object.

  Returns:
    q, k: [seq_tile_size, num_kq_heads, chunk_size, kq_head_dim]
    v: [seq_tile_size, num_v_heads, chunk_size, v_head_dim]
  """

  num_lanes = pltpu.get_tpu_info().num_lanes
  lanes_per_col = qkv_vmem_ref.shape[-1] // num_lanes
  kq_lanes_per_head = cfgs.kq_head_dim // num_lanes
  k_offset = cfgs.num_kq_heads * kq_lanes_per_head

  q_large_list = []
  k_large_list = []
  v_large_list = []

  qkv_slot_flat_ref = qkv_vmem_ref.reshape(-1, num_lanes)  # pyrefly: ignore[missing-attribute]
  for kq_head in range(cfgs.num_kq_heads):
    q_head_list = []
    k_head_list = []
    for lane in range(kq_lanes_per_head):
      q_lane = kq_head * kq_lanes_per_head + lane
      k_lane = k_offset + q_lane

      q_head_list.append(qkv_slot_flat_ref[q_lane::lanes_per_col])
      k_head_list.append(qkv_slot_flat_ref[k_lane::lanes_per_col])
    q_large_list.append(jnp.concat(q_head_list, axis=-1))
    k_large_list.append(jnp.concat(k_head_list, axis=-1))
  v_offset = kq_lanes_per_head * cfgs.num_kq_heads * 2
  v_lanes_per_head = cfgs.v_head_dim // num_lanes
  for v_head in range(cfgs.num_v_heads):
    v_head_list = []
    for lane in range(v_lanes_per_head):
      v_lane = v_offset + v_head * v_lanes_per_head + lane
      v_head_list.append(qkv_slot_flat_ref[v_lane::lanes_per_col])
    v_large_list.append(jnp.concat(v_head_list, axis=-1))

  q_large = jnp.stack(q_large_list, axis=0)
  k_large = jnp.stack(k_large_list, axis=0)
  v_large = jnp.stack(v_large_list, axis=0)

  return q_large, k_large, v_large


def load_as_qkv_compact(
    qkv_vmem_ref: jax.Ref, cfg: config.GDNConfig
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Split qkv and transpose by performing 1 load per head for compact layout.

  Args:
    qkv_vmem_ref: qkv reference in VMEM containing concatenated values of q, k,
      and v of shape [seq_tile_size, chunk_size, 1, num_kq_heads * kq_head_dim *
      2 + num_v_heads * v_head_dim].
    cfg: GDN configuration object.

  Returns:
    q, k: [seq_tile_size, num_kq_heads, chunk_size, 1, kq_head_dim]
    v: [seq_tile_size, num_v_heads, chunk_size, 1, v_head_dim]
  """

  k_offset = cfg.num_kq_heads * cfg.kq_head_dim
  v_offset = cfg.num_kq_heads * 2 * cfg.kq_head_dim

  q_compact_list = []
  k_compact_list = []
  v_compact_list = []

  for kq_head in range(cfg.num_kq_heads):
    q_start = kq_head * cfg.kq_head_dim
    q_end = q_start + cfg.kq_head_dim
    k_start = k_offset + q_start
    k_end = k_start + cfg.kq_head_dim
    q_compact_list.append(qkv_vmem_ref[..., q_start:q_end])
    k_compact_list.append(qkv_vmem_ref[..., k_start:k_end])
  for v_head in range(cfg.num_v_heads):
    v_start = v_offset + v_head * cfg.v_head_dim
    v_end = v_start + cfg.v_head_dim
    v_compact_list.append(qkv_vmem_ref[..., v_start:v_end])

  q_compact = jnp.stack(q_compact_list, axis=1)
  k_compact = jnp.stack(k_compact_list, axis=1)
  v_compact = jnp.stack(v_compact_list, axis=1)

  return q_compact, k_compact, v_compact


def load_compact_to_large(vmem_ref: jax.Ref) -> jax.Array:
  """Use strided load to convert compact to large layout without transpose."""

  # NOTE: Only support 32-bits for now.
  assert vmem_ref.dtype.itemsize == 4
  assert vmem_ref.shape[-2] == 1
  col_size = vmem_ref.shape[-1]
  new_shape = vmem_ref.shape[:-2] + (col_size,)
  tpu_info = pltpu.get_tpu_info()
  num_lanes = tpu_info.num_lanes

  vreg_list = []
  vmem_ref = vmem_ref.reshape(-1, col_size)  # pyrefly: ignore[missing-attribute]
  for col_start in range(0, col_size, num_lanes):
    col_end = min(col_start + num_lanes, col_size)
    vreg = vmem_ref[..., col_start:col_end]
    vreg_list.append(vreg)
  return jnp.concat(vreg_list, axis=-1).reshape(new_shape)


def load_and_select_states(
    metadata_ref: memory_ref.MetadataRef,
    p_id: jax.Array,
    conv_state_slot_ref: jax.Ref,
    recurrent_slot_ref: jax.Ref,
    carry_conv_scratch_ref: jax.Ref | None,
    carry_recurrent_scratch_ref: jax.Ref | None,
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Load correct states from HBM or prior tile, and masks invalid states.

  Reference metadata to select the appropriate prior states. If `is_first_tile`
  is True, it selects states read from HBM. If it is False, it selects
  carry states from previous tile. If `has_initial_state` is False, states are
  zero initialized.

  Args:
    metadata_ref: Metadata reference containing grid and sequence mappings.
    p_id: Current Pallas program ID.
    conv_state_slot_ref: Convolution state read from HBM of shape
      [seq_tile_size, prev_kernel_size, 1, dim_size].
    recurrent_slot_ref: Recurrent state read from HBM of shape [seq_tile_size,
      num_v_heads, kq_head_dim, v_head_dim].
    carry_conv_scratch_ref: Optional inter-tile convolution carry of shape
      [seq_tile_size, prev_kernel_size, 1, dim_size].
    carry_recurrent_scratch_ref: Optional inter-tile recurrent state carry of
      shape [seq_tile_size, num_v_heads, kq_head_dim, v_head_dim].
    cfg: GDN configuration object.

  Returns:
    real_sizes: Valid token count per sequence tile of shape [seq_tile_size].
    prev_conv_state: Selected convolution state of shape [seq_tile_size,
      prev_kernel_size, 1, dim_size] in float32.
    prev_recurrent_state: Selected recurrent state of shape [seq_tile_size,
      num_v_heads, kq_head_dim, v_head_dim].
  """

  real_sizes_list = []
  prev_conv_state_list = []
  prev_recurrent_state_list = []

  for idx in range(cfg.seq_tile_size):
    s_idx = metadata_ref.p_id_to_s_idx[p_id, idx]
    real_sizes = metadata_ref.p_id_to_r_size[p_id, idx]
    is_first_tile = metadata_ref.p_id_is_first_tile[p_id, idx]
    has_initial_state = metadata_ref.s_idx_has_initial_state[s_idx]

    # NOTE: Conv1D mandates fp32 due to its usage of compact layout.
    hbm_conv_state = conv_state_slot_ref[idx].astype(jnp.float32)
    prev_conv_state = jnp.where(has_initial_state, hbm_conv_state, 0)

    if carry_conv_scratch_ref is not None:
      prev_tile_conv = carry_conv_scratch_ref[idx]
      prev_conv_state = jnp.where(
          is_first_tile, prev_conv_state, prev_tile_conv
      )

    hbm_recurrent_state = recurrent_slot_ref[idx]
    prev_recurrent_state = jnp.where(has_initial_state, hbm_recurrent_state, 0)

    if carry_recurrent_scratch_ref is not None:
      prev_tile_recurrent_scratch = carry_recurrent_scratch_ref[idx]
      prev_recurrent_state = jnp.where(
          is_first_tile, prev_recurrent_state, prev_tile_recurrent_scratch
      )

    real_sizes_list.append(real_sizes)
    prev_conv_state_list.append(prev_conv_state)
    prev_recurrent_state_list.append(prev_recurrent_state)

  real_sizes = jnp.stack(real_sizes_list, axis=0)
  prev_conv_state = jnp.stack(prev_conv_state_list, axis=0)
  prev_recurrent_state = jnp.stack(prev_recurrent_state_list, axis=0)

  return real_sizes, prev_conv_state, prev_recurrent_state


def load_activation_as_compact(
    qkv_vreg: jax.Array,
    qkv_vmem_ref: jax.Ref,
    b_vmem_ref: jax.Ref,
    a_vmem_ref: jax.Ref,
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Load activations from VMEM as a compact layout."""

  qkv_vmem_ref[...] = qkv_vreg
  q_compact, k_compact, v_compact = load_as_qkv_compact(qkv_vmem_ref, cfgs)
  b_compact = jnp.expand_dims(b_vmem_ref[...], axis=1)
  a_compact = jnp.expand_dims(a_vmem_ref[...], axis=1)
  return q_compact, k_compact, v_compact, b_compact, a_compact


def load_activation_as_large(
    qkv_vreg: jax.Array,
    qkv_vmem_ref: jax.Ref,
    b_vmem_ref: jax.Ref,
    a_vmem_ref: jax.Ref,
    cfgs: config.GDNConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Load activations from VMEM as a large layout."""

  qkv_vmem_ref[...] = qkv_vreg

  q_large_list = []
  k_large_list = []
  v_large_list = []
  for idx in range(cfgs.seq_tile_size):
    q_large, k_large, v_large = load_as_qkv_large(qkv_vmem_ref.at[idx], cfgs)
    q_large_list.append(q_large)
    k_large_list.append(k_large)
    v_large_list.append(v_large)

  q_large = jnp.stack(q_large_list, axis=0)
  k_large = jnp.stack(k_large_list, axis=0)
  v_large = jnp.stack(v_large_list, axis=0)
  b_large = load_compact_to_large(b_vmem_ref)
  a_large = load_compact_to_large(a_vmem_ref)
  b_large = jnp.expand_dims(b_large, axis=1)
  a_large = jnp.expand_dims(a_large, axis=1)

  return q_large, k_large, v_large, b_large, a_large

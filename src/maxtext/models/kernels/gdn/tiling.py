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

"""Dynamic tiling heuristics and VMEM memory estimation for Fused Conv1D-GDN."""

from jax.experimental import pallas as pl
import jax.numpy as jnp

try:
  from maxtext.models.kernels.gdn import config
except (ImportError, ModuleNotFoundError):
  try:
    from maxtext.src.maxtext.models.kernels.gdn import config
  except (ImportError, ModuleNotFoundError):
    from . import config


def align_to(x: int, alignment: int) -> int:
  """Aligns an integer upward to the nearest multiple of alignment."""
  return pl.cdiv(x, alignment) * alignment


def get_vmem_estimate_bytes(
    tile_b: int,
    chunk_sz: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    act_in_bytes: int,
    act_out_bytes: int,
    conv_state_bytes: int,
    rec_state_bytes: int,
    num_lanes: int,
    conv_state_dim_size: int,
    is_decode: bool = False,
) -> int:
  """Estimates total on-chip VMEM footprint in bytes for a GDN tile."""
  aligned_num_v_heads = align_to(n_v, num_lanes)
  aligned_d_k = align_to(d_k, num_lanes)
  aligned_d_v = align_to(d_v, num_lanes)
  dim_size = align_to(2 * n_kq * d_k + n_v * d_v, num_lanes)
  aligned_out_dim = align_to(n_v * d_v, num_lanes)

  # 1. Double-buffered input activation buffers (QKV, A, B).
  qkv_bytes = 2 * (tile_b * chunk_sz * dim_size * act_in_bytes)
  b_bytes = 2 * (tile_b * chunk_sz * aligned_num_v_heads * act_in_bytes)
  a_bytes = 2 * (tile_b * chunk_sz * aligned_num_v_heads * act_in_bytes)

  # 2. Double-buffered state cache buffers (convolution and recurrent states).
  conv_state_buffer_bytes = 2 * (
      tile_b * max(0, kernel_size - 1) * conv_state_dim_size * conv_state_bytes
  )
  recurrent_state_buffer_bytes = 2 * (
      tile_b * n_v * d_v * d_k * rec_state_bytes
  )

  # 3. Double-buffered output activation buffer.
  out_bytes = 2 * (tile_b * chunk_sz * aligned_out_dim * act_out_bytes)

  # 4. Temporary scratch buffers (allocated in non-batched mode).
  if is_decode:
    scratch_conv_bytes = 0
    scratch_recurrent_bytes = 0
  else:
    scratch_conv_bytes = (
        tile_b * max(0, kernel_size - 1) * dim_size * conv_state_bytes
    )
    scratch_recurrent_bytes = tile_b * n_v * d_v * d_k * rec_state_bytes

  # 5. Static weight cache references in on-chip memory.
  weights_bytes = (
      ((kernel_size - 1) * dim_size * 4)
      + (dim_size * 4)
      + (aligned_num_v_heads * 8)
  )

  # 6. Working memory for intra-chunk recurrence and projections.
  intermediate_bytes = (
      n_v
      * (5 * chunk_sz * chunk_sz + 3 * chunk_sz * (aligned_d_v + aligned_d_k))
      * 4
  )

  return (
      qkv_bytes
      + b_bytes
      + a_bytes
      + conv_state_buffer_bytes
      + recurrent_state_buffer_bytes
      + out_bytes
      + scratch_conv_bytes
      + scratch_recurrent_bytes
      + weights_bytes
      + intermediate_bytes
  )


def calculate_decode_tile_size(
    batch_size: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    conv_state_dim_size: int,
    act_in_dtype: jnp.dtype,
    act_out_dtype: jnp.dtype,
    conv_state_dtype: jnp.dtype,
    recurrent_state_dtype: jnp.dtype,
    num_lanes: int,
    vmem_capacity_limit_bytes: int,
    kernel_size: int = 4,
) -> int:
  """Derives optimal batch tile size for decode execution.

  Searches candidate batch tile sizes within maximum VMEM capacity limits.

  Args:
    batch_size: Total batch size of the active decode sequence.
    n_kq: Number of key/query heads.
    n_v: Number of value heads.
    d_k: Key head dimension.
    d_v: Value head dimension.
    conv_state_dim_size: Feature dimension size for conv state.
    act_in_dtype: Data type for input activations.
    act_out_dtype: Data type for output activations.
    conv_state_dtype: Data type for conv state cache.
    recurrent_state_dtype: Data type for recurrent state matrix.
    num_lanes: Number of lanes for TPU vector layout alignment.
    vmem_capacity_limit_bytes: Maximum allowed VMEM capacity in bytes.
    kernel_size: 1D convolution kernel window size.

  Returns:
    Derived batch tile size fitting within VMEM capacity limits.
  """
  # Return a minimum valid tile size of 1 for empty or zero-length batches.
  if batch_size <= 0:
    return 1

  act_in_bytes = jnp.dtype(act_in_dtype).itemsize
  act_out_bytes = jnp.dtype(act_out_dtype).itemsize
  conv_state_bytes = jnp.dtype(conv_state_dtype).itemsize
  rec_state_bytes = jnp.dtype(recurrent_state_dtype).itemsize

  # Balance vector compute density against on-chip VMEM capacity:
  # - Cap tile size across batch size tiers to maximize vector lane compute
  #   density.
  # - Floor at tile_b = 4 for small batches to ensure compute density.
  # - When value head count is large (n_v >= 64), recurrent state working
  #   memory scales up, so cap max_decode_b to 4 to prevent on-chip memory
  #   overflow.
  if n_v >= 64 or batch_size <= 64:
    max_decode_b = 4
  elif batch_size <= 128:
    max_decode_b = 8
  elif batch_size <= 256:
    max_decode_b = 16
  else:
    max_decode_b = 32

  decode_candidates = [
      c for c in (32, 16, 8, 4, 2, 1) if c <= batch_size and c <= max_decode_b
  ]
  decode_tile_size = decode_candidates[-1]

  for cand in decode_candidates:
    vmem_est = get_vmem_estimate_bytes(
        tile_b=cand,
        chunk_sz=1,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        act_in_bytes=act_in_bytes,
        act_out_bytes=act_out_bytes,
        conv_state_bytes=conv_state_bytes,
        rec_state_bytes=rec_state_bytes,
        num_lanes=num_lanes,
        conv_state_dim_size=conv_state_dim_size,
        is_decode=True,
    )
    if vmem_est <= vmem_capacity_limit_bytes:
      return cand

  return decode_tile_size


def calculate_mixed_tile_size(
    seq_len: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    conv_state_dim_size: int,
    act_in_dtype: jnp.dtype,
    act_out_dtype: jnp.dtype,
    conv_state_dtype: jnp.dtype,
    recurrent_state_dtype: jnp.dtype,
    num_lanes: int,
    vmem_capacity_limit_bytes: int,
    kernel_size: int = 4,
) -> int:
  """Derives optimal chunk tile size for prefill and mixed execution.

  Searches candidate tile sizes within maximum VMEM capacity limits.

  Args:
    seq_len: Sequence length of the active prefill or mixed sequence.
    n_kq: Number of key/query heads.
    n_v: Number of value heads.
    d_k: Key head dimension.
    d_v: Value head dimension.
    conv_state_dim_size: Feature dimension size for conv state.
    act_in_dtype: Data type for input activations.
    act_out_dtype: Data type for output activations.
    conv_state_dtype: Data type for conv state cache.
    recurrent_state_dtype: Data type for recurrent state matrix.
    num_lanes: Number of lanes for TPU vector layout alignment.
    vmem_capacity_limit_bytes: Maximum allowed VMEM capacity in bytes.
    kernel_size: 1D convolution kernel window size.

  Returns:
    Derived chunk tile size fitting within VMEM capacity limits.
  """
  # Return a minimum valid chunk size of 1 for empty or zero-length sequences.
  if seq_len <= 0:
    return 1

  act_in_bytes = jnp.dtype(act_in_dtype).itemsize
  act_out_bytes = jnp.dtype(act_out_dtype).itemsize
  conv_state_bytes = jnp.dtype(conv_state_dtype).itemsize
  rec_state_bytes = jnp.dtype(recurrent_state_dtype).itemsize

  # Limit chunk size to C <= 128: above 128, intra-chunk triangular
  # solve operations and vector register pressure outweigh systolic compute
  # density gains.
  # When value head count is large (n_v >= 64), intra-chunk intermediate
  # memory scales up, so cap chunk search space to C <= 64 to avoid on-chip
  # memory overflow.
  max_chunk_cap = 64 if n_v >= 64 else 128
  prefill_candidates = [
      c
      for c in (128, 64, 32, 16, 8, 4, 2, 1)
      if c <= seq_len and c <= max_chunk_cap
  ]
  mixed_tile_size = prefill_candidates[-1]
  for candidate in prefill_candidates:
    vmem_est = get_vmem_estimate_bytes(
        tile_b=1,
        chunk_sz=candidate,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        kernel_size=kernel_size,
        act_in_bytes=act_in_bytes,
        act_out_bytes=act_out_bytes,
        conv_state_bytes=conv_state_bytes,
        rec_state_bytes=rec_state_bytes,
        num_lanes=num_lanes,
        conv_state_dim_size=conv_state_dim_size,
        is_decode=False,
    )
    if vmem_est <= vmem_capacity_limit_bytes:
      return candidate

  return mixed_tile_size


def get_tile_sizes(
    batch_size: int,
    num_seqs: int,
    padded_batch_size: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    conv_state_dim_size: int,
    act_in_dtype: jnp.dtype,
    act_out_dtype: jnp.dtype,
    conv_state_dtype: jnp.dtype,
    recurrent_state_dtype: jnp.dtype,
    num_lanes: int,
    decode_tile_size: int | None = None,
    mixed_tile_size: int | None = None,
) -> tuple[int, int]:
  """Derives optimal decode and mixed tile sizes fitting within VMEM limits."""
  vmem_capacity_limit_bytes = config.get_vmem_limit_bytes()

  if decode_tile_size is None or decode_tile_size <= 0:
    decode_tile_size = calculate_decode_tile_size(
        batch_size=padded_batch_size,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        conv_state_dim_size=conv_state_dim_size,
        act_in_dtype=act_in_dtype,
        act_out_dtype=act_out_dtype,
        conv_state_dtype=conv_state_dtype,
        recurrent_state_dtype=recurrent_state_dtype,
        num_lanes=num_lanes,
        vmem_capacity_limit_bytes=vmem_capacity_limit_bytes,
        kernel_size=kernel_size,
    )

  if mixed_tile_size is None or mixed_tile_size <= 0:
    if batch_size <= num_seqs:
      # When all sequences have length 1 (decode), size prefill chunks to 1.
      effective_prefill_seq_len = 1
    else:
      # Estimate maximum sequence length across uniform and mixed prefill
      # batches. In an adversarial mixed batch of B tokens with N_seq sequences,
      # the largest prefill sequence length is bounded by
      # B - (N_seq - 1) = B - N_seq + 1.
      effective_prefill_seq_len = max(
          batch_size // max(1, num_seqs),
          batch_size - num_seqs + 1,
      )

    mixed_tile_size = calculate_mixed_tile_size(
        seq_len=effective_prefill_seq_len,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        conv_state_dim_size=conv_state_dim_size,
        act_in_dtype=act_in_dtype,
        act_out_dtype=act_out_dtype,
        conv_state_dtype=conv_state_dtype,
        recurrent_state_dtype=recurrent_state_dtype,
        num_lanes=num_lanes,
        vmem_capacity_limit_bytes=vmem_capacity_limit_bytes,
        kernel_size=kernel_size,
    )

  # Guarantee strictly positive tile sizes (>= 1) for Pallas grid compilation.
  decode_tile_size = max(1, min(decode_tile_size, batch_size))
  mixed_tile_size = max(1, min(mixed_tile_size, batch_size))
  return decode_tile_size, mixed_tile_size

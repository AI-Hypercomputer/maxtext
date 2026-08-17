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

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels.causal_conv1d_gated_delta_rule import compute_conv1d
from maxtext.kernels.causal_conv1d_gated_delta_rule import compute_gdn
from maxtext.kernels.causal_conv1d_gated_delta_rule import config
from maxtext.kernels.causal_conv1d_gated_delta_rule import memory_ref
from maxtext.kernels.causal_conv1d_gated_delta_rule import metadata
from maxtext.kernels.causal_conv1d_gated_delta_rule import vmem_ldst


def inner_kernel(
    # Inputs.
    qkv_slot_ref: jax.Array,  # [seq, chunk, 1, dim_size]
    b_slot_ref: jax.Array,  # [seq, chunk, 1, num_v_heads]
    a_slot_ref: jax.Array,  # [seq, chunk, 1, num_v_heads]
    conv_state_slot_ref: jax.Array,  # [seq, prev_kernel_size, 1, dim_size]
    recurrent_slot_ref: jax.Array,  # [seq, num_v_heads, kq_head, v_head]
    # Outputs.
    out_slot_ref: jax.Array,  # [seq * chunk, num_v_heads, v_head]
    tap_slot_ref: jax.Array,  # [seq, chunk, num_v_heads, chunk]
    # Scratches.
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
) -> None:
  """Orchestrates computation of Conv1D and GDN for a single tile (Training / Prefill Mode)."""
  p_id = pl.program_id(0)

  real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
      metadata_ref=metadata_ref, p_id=p_id, conv_state_slot_ref=conv_state_slot_ref,
      recurrent_slot_ref=recurrent_slot_ref, carry_conv_scratch_ref=carry_conv_scratch_ref,
      carry_recurrent_scratch_ref=carry_recurrent_scratch_ref, cfg=cfg,
  )

  # Step 1: Conv1D
  qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
  qkv_in_compact = jnp.concat([prev_conv, qkv_in_compact], axis=1)

  conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
  conv_bias = weights_ref.conv.bias[...].astype(jnp.float32) if weights_ref.conv.bias is not None else None

  qkv_out_compact, new_conv_state = compute_conv1d.causal_conv1d(
      real_sizes=real_sizes, lhs=qkv_in_compact, conv_weight=conv_weight, conv_bias=conv_bias, cfg=cfg,
  )

  conv_state_slot_ref[...] = new_conv_state
  if carry_conv_scratch_ref is not None:
    carry_conv_scratch_ref[...] = new_conv_state

  qkv_out_compact = jax.nn.silu(qkv_out_compact)

  # Step 2: GDN (Prefill only)
  padding_size = cfg.aligned_num_v_heads - cfg.num_v_heads
  a_log = jnp.pad(weights_ref.gdn.a_log[...], ((0, padding_size)))
  dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], ((0, padding_size)))

  q_large, k_large, v_large, b_large, a_large = vmem_ldst.load_activation_as_large(
      qkv_vreg=qkv_out_compact, qkv_vmem_ref=qkv_slot_ref, b_vmem_ref=b_slot_ref, a_vmem_ref=a_slot_ref, cfgs=cfg,
  )

  out, new_recurrent_state, tap_val = compute_gdn.chunked_gdn(
      q_large=q_large, k_large=k_large, v_large=v_large, b_large=b_large, a_large=a_large,
      state_prev=prev_recurrent, a_log=a_log, dt_bias=dt_bias, cfg=cfg, real_sizes=real_sizes,
  )

  # Store outputs
  out_slot_ref[...] = out.astype(out_slot_ref.dtype)
  recurrent_slot_ref[...] = new_recurrent_state.astype(recurrent_slot_ref.dtype)

  if carry_recurrent_scratch_ref is not None:
    carry_recurrent_scratch_ref[...] = new_recurrent_state

  # Store Tap
  if tap_val.ndim == 3:
      tap_val = jnp.expand_dims(tap_val, 0)
  tap_val_transposed = jnp.swapaxes(tap_val, 1, 2)
  tap_slot_ref[...] = tap_val_transposed.astype(tap_slot_ref.dtype)


def outer_kernel(
    # Inputs.
    metadata_ref: memory_ref.MetadataRef,
    qkv_ref: jax.Array, b_ref: jax.Array, a_ref: jax.Array,
    conv_state_ref: jax.Array, recurrent_state_ref: jax.Array, _: jax.Array, weights_ref: memory_ref.WeightRefs,
    # Outputs.
    out_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    recurrent_state_out_ref: jax.Array,
    tap_ref: jax.Array,
    # Scratches.
    carry_conv_scratch_ref: jax.Array | None, carry_recurrent_scratch_ref: jax.Array | None,
    *, cfg: config.GDNConfig,
) -> None:
  """Setup memory allocations and emit pipeline."""
  del conv_state_out_ref, recurrent_state_out_ref

  qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc = memory_ref.create_allocs(
      metadata_ref=metadata_ref, qkv_ref=qkv_ref, b_ref=b_ref, a_ref=a_ref,
      out_ref=out_ref, conv_state_ref=conv_state_ref, recurrent_state_ref=recurrent_state_ref, cfg=cfg,
  )

  pipeline_mode = pl.Buffered(buffer_count=cfg.num_buffers, use_lookahead=False)
  tap_alloc = memory_ref.OutBufferedRef.output(
      spec=pl.BlockSpec(
          memory_space=pltpu.VMEM, index_map=lambda i: (i,), pipeline_mode=pipeline_mode,
          block_shape=(cfg.seq_tile_size, cfg.chunk_size, cfg.num_v_heads, cfg.chunk_size)
      ),
      dtype_or_type=tap_ref, buffer_count=pipeline_mode.buffer_count,
      use_lookahead=pipeline_mode.use_lookahead, cfg=cfg, metadata_ref=metadata_ref,
  )

  num_tiles = metadata_ref.num_tiles[...]

  pipeline_func = pltpu.emit_pipeline(
      body=functools.partial(inner_kernel, cfg=cfg), grid=(num_tiles,),
      in_specs=(qkv_alloc.spec, b_alloc.spec, a_alloc.spec, conv_alloc.spec, recurrent_alloc.spec),
      out_specs=(out_alloc.spec, tap_alloc.spec),
  )

  @pl.with_scoped(allocations=(qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc, tap_alloc))
  def _run(allocations):
    pipeline_func(
        qkv_ref, b_ref, a_ref, conv_state_ref, recurrent_state_ref, 
        out_ref, tap_ref, 
        scratches=(metadata_ref, weights_ref, carry_conv_scratch_ref, carry_recurrent_scratch_ref),
        allocations=allocations,
    )

  _run()


@jax.jit(
    donate_argnames=("conv_state", "recurrent_state"),
    static_argnames=("n_kq", "n_v", "d_k", "d_v", "kernel_size", "decode_tile_size", "mixed_tile_size", "zero_initialize_out", "compute_precision"),
)
def fused_conv1d_gdn(
    qkv: jax.Array, b: jax.Array, a: jax.Array,
    conv_state: jax.Array, recurrent_state: jax.Array,
    conv_weight: jax.Array, conv_bias: jax.Array | None,
    a_log: jax.Array, dt_bias: jax.Array,
    query_start_loc: jax.Array, state_indices: jax.Array, distribution: jax.Array, seq_lens: jax.Array,
    *, n_kq: int, n_v: int, d_k: int, d_v: int, kernel_size: int,
    zero_initialize_out: bool = True, compute_precision: jnp.dtype = jnp.float32.dtype,
    decode_tile_size: int = 4, mixed_tile_size: int = 64,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array, jax.Array]:
  """Perform training-only conv1d and gdn in a single fused kernel."""
  act_out_dtype = qkv.dtype
  conv_out_dtype = conv_state.dtype
  recurrent_out_dtype = recurrent_state.dtype

  qkv = qkv.astype(jnp.float32)
  b = b.astype(jnp.float32)
  a = a.astype(jnp.float32)
  conv_state = conv_state.astype(jnp.float32)

  num_seqs = state_indices.size
  batch_size, dim = qkv.shape
  act_in_dtype = qkv.dtype

  num_lanes = pltpu.get_tpu_info().num_lanes
  packing = 4 // act_in_dtype.itemsize
  padded_batch_size = pl.cdiv(batch_size, packing) * packing
  mixed_tile_size = min(mixed_tile_size, batch_size)
  aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes

  batch_padding_size = padded_batch_size - batch_size
  num_v_padding_size = aligned_num_v_heads - n_v
  qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0))).reshape(padded_batch_size, 1, -1)
  b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size))).reshape(padded_batch_size, 1, -1)
  a = jnp.pad(a, ((0, batch_padding_size), (0, num_v_padding_size))).reshape(padded_batch_size, 1, -1)

  conv_state_shape = conv_state.shape
  conv_state = conv_state.reshape(-1, kernel_size - 1, 1, dim)
  conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
  conv_bias = conv_bias.astype(jnp.float32) if conv_bias is not None else None

  weights = memory_ref.WeightRefs(
      conv=memory_ref.ConvWeightsRef(weight=conv_weight, bias=conv_bias), 
      gdn=memory_ref.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias)
  )

  smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
  vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

  # STRIPPED CONFIG: Training Mode Only (PER_SEQ)
  cfg = config.GDNConfig(
      mode=config.GDNMode.PER_SEQ,
      batch_size=padded_batch_size,
      kernel_size=kernel_size,
      tile_size=mixed_tile_size,
      dim_size=dim,
      num_kq_heads=n_kq,
      num_v_heads=n_v,
      kq_head_dim=d_k,
      v_head_dim=d_v,
      dtypes=config.Dtypes(
          act_in=act_in_dtype, act_out=act_out_dtype, compute=compute_precision,
          recurrent_state=recurrent_state.dtype, conv_state=conv_state.dtype,
      ),
  )

  metadata_obj = metadata.compute_per_seq_metadata(
      cfg=cfg, seq_lens=seq_lens, query_start_loc=query_start_loc,
      state_indices=state_indices, start_seq=distribution[0], end_seq=distribution[-1],
  )
  metadata_spec = jax.tree.map(lambda _: smem_spec, metadata_obj)

  out_shape = cfg.get_out_shape()
  in_act = jnp.zeros_like(out_shape) # Always initialize for training
  in_out_spec = hbm_spec
  input_output_aliases = {len(metadata_obj) + 3: 1, len(metadata_obj) + 4: 2, len(metadata_obj) + 5: 0}

  tap_shape = jax.ShapeDtypeStruct((padded_batch_size, cfg.num_v_heads, cfg.chunk_size), jnp.float32)

  res = pl.pallas_call(
      functools.partial(outer_kernel, cfg=cfg),
      out_shape=(out_shape, conv_state, recurrent_state, tap_shape),
      in_specs=(metadata_spec, hbm_spec, hbm_spec, hbm_spec, hbm_spec, hbm_spec, in_out_spec, weights_spec),
      out_specs=(hbm_spec, hbm_spec, hbm_spec, hbm_spec),
      scratch_shapes=cfg.get_scratch_shape_dict(),
      compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, vmem_limit_bytes=cfg.get_vmem_limit_bytes()),
      name=cfg.get_kernel_name(),
      metadata=cfg.get_metadata(),
      input_output_aliases=input_output_aliases,
  )(metadata_obj, qkv, b, a, conv_state, recurrent_state, in_act, weights)

  out_act, out_conv_state, out_recurrent_state, out_tap = res

  out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
  out_conv_state = out_conv_state.astype(conv_out_dtype).reshape(conv_state_shape)
  out_recurrent_state = out_recurrent_state.astype(recurrent_out_dtype)

  return (out_conv_state, out_recurrent_state), out_act, out_tap
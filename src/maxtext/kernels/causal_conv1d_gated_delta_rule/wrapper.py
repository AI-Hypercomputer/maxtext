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
    tap_slot_ref: jax.Array,
    # Scratches.
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
) -> None:
  """Orchestrates computation of Conv1D and GDN for a single tile.

  This kernel acts as a facade adhering to strict separation of concerns. It
  operates VMEM reference without knowledge on DMA logic. Furthermore, the
  kernel invokes vmem_ldst to pre-processes data needed for compute and
  invokes compute_conv1d and compute_gdn for actual compute.

  Args:
    qkv_slot_ref: qkv VMEM ref that stores data loaded from HBM.
    b_slot_ref: b VMEM ref that stores data loaded from HBM.
    a_slot_ref: a VMEM ref that stores data loaded from HBM.
    conv_state_slot_ref: Convolution state VMEM ref that stores data loaded from
      HBM. Data written into this VMEM ref will be used for VMEM to HBM write.
    recurrent_slot_ref: Recurrent state VMEM ref that stores data loaded from
      HBM. Data written into this VMEM ref will be used for VMEM to HBM write.
    out_slot_ref: Output VMEM ref that will be used for VMEM to HBM write.
    metadata_ref: Metadata reference containing grid and sequence mappings.
    weights_ref: Weight references for Conv1D and GDN in VMEM.
    carry_conv_scratch_ref: Optional VMEM scratch reference for inter-tile
      convolution carry.
    carry_recurrent_scratch_ref: Optional VMEM scratch reference for inter-tile
      recurrent state carry.
    cfg: GDN configuration object.
  """

  p_id = pl.program_id(0)

  # Prepare states.
  real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
      metadata_ref=metadata_ref,
      p_id=p_id,
      conv_state_slot_ref=conv_state_slot_ref,  # pyrefly: ignore[bad-argument-type]
      recurrent_slot_ref=recurrent_slot_ref,  # pyrefly: ignore[bad-argument-type]
      carry_conv_scratch_ref=carry_conv_scratch_ref,  # pyrefly: ignore[bad-argument-type]
      carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,  # pyrefly: ignore[bad-argument-type]
      cfg=cfg,
  )

  # Step 1: Conv1D.
  # NOTE: Conv1D requires performing sliding window where inputs are slided
  # across rows. If typical 2D layout was used, multiple rows are stored in a
  # single register which necessitate costly shuffling for every sliding.
  # Therefore, it is extremely important to leverage compact layout that
  # ensures 1 register only stores data from 1 row.
  qkv_in_compact = qkv_slot_ref[...].astype(jnp.float32)
  qkv_in_compact = jnp.concat([prev_conv, qkv_in_compact], axis=1)

  # Prepare conv1d weights.
  conv_weight = weights_ref.conv.weight[...].astype(jnp.float32)
  conv_bias = None
  if weights_ref.conv.bias is not None:
    conv_bias = weights_ref.conv.bias[...].astype(jnp.float32)

  qkv_out_compact, new_conv_state = compute_conv1d.causal_conv1d(
      real_sizes=real_sizes,
      lhs=qkv_in_compact,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      cfg=cfg,
  )

  conv_state_slot_ref[...] = new_conv_state
  if carry_conv_scratch_ref is not None:
    carry_conv_scratch_ref[...] = new_conv_state

  # Apply activation function.
  qkv_out_compact = jax.nn.silu(qkv_out_compact)

  # Step 2: GDN.

  # Prepare gdn weights.
  padding_size = cfg.aligned_num_v_heads - cfg.num_v_heads
  a_log = jnp.pad(weights_ref.gdn.a_log[...], ((0, padding_size)))
  dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], ((0, padding_size)))

  # NOTE: Ideally, we want to move this branching logic into gdn.py. However,
  # load_activation_as_compact and load_activation_as_large leverages vmem ldst.
  # Passing refs into gdn.py breaks strict separation of concerns.
  if cfg.chunk_size == 1:
    q_compact, k_compact, v_compact, b_compact, a_compact = (
        vmem_ldst.load_activation_as_compact(
            qkv_vreg=qkv_out_compact,
            qkv_vmem_ref=qkv_slot_ref,  # pyrefly: ignore[bad-argument-type]
            b_vmem_ref=b_slot_ref,  # pyrefly: ignore[bad-argument-type]
            a_vmem_ref=a_slot_ref,  # pyrefly: ignore[bad-argument-type]
            cfgs=cfg,
        )
    )

    out, new_recurrent_state = compute_gdn.recurrent_gdn(
        q_compact=q_compact,
        k_compact=k_compact,
        v_compact=v_compact,
        b_compact=b_compact,
        a_compact=a_compact,
        state_prev=prev_recurrent,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg,
        real_sizes=real_sizes,
    )
    
    # MUST HAVE seq_tile_size DIMENSION (cfg.num_v_heads, not aligned_num_v_heads to match HBM!)
    tap_val = jnp.zeros((cfg.seq_tile_size, cfg.num_v_heads, cfg.chunk_size, cfg.chunk_size), dtype=jnp.float32)

  else:
    q_large, k_large, v_large, b_large, a_large = (
        vmem_ldst.load_activation_as_large(
            qkv_vreg=qkv_out_compact,
            qkv_vmem_ref=qkv_slot_ref,  # pyrefly: ignore[bad-argument-type]
            b_vmem_ref=b_slot_ref,  # pyrefly: ignore[bad-argument-type]
            a_vmem_ref=a_slot_ref,  # pyrefly: ignore[bad-argument-type]
            cfgs=cfg,
        )
    )

    out, new_recurrent_state, tap_val = compute_gdn.chunked_gdn(
        q_large=q_large,
        k_large=k_large,
        v_large=v_large,
        b_large=b_large,
        a_large=a_large,
        state_prev=prev_recurrent,
        a_log=a_log,
        dt_bias=dt_bias,
        cfg=cfg,
        real_sizes=real_sizes,
    )

  # Store output and recurrent to vmem.
  out_slot_ref[...] = out.astype(out_slot_ref.dtype)
  if tap_val.ndim == 3:
      tap_val = jnp.expand_dims(tap_val, 0)
      
  # Transpose from [seq_tile_size, num_v_heads, chunk_size, chunk_size] 
  # to token-first layout [seq_tile_size, chunk_size, num_v_heads, chunk_size]
  tap_val_transposed = jnp.swapaxes(tap_val, 1, 2)
  tap_slot_ref[...] = tap_val_transposed.astype(tap_slot_ref.dtype)
  recurrent_slot_ref[...] = new_recurrent_state.astype(recurrent_slot_ref.dtype)

  if carry_recurrent_scratch_ref is not None:
    carry_recurrent_scratch_ref[...] = new_recurrent_state


def outer_kernel(
    # Inputs.
    metadata_ref: memory_ref.MetadataRef,
    qkv_ref: jax.Array,
    b_ref: jax.Array,
    a_ref: jax.Array,
    conv_state_ref: jax.Array,
    recurrent_state_ref: jax.Array,
    _: jax.Array,
    weights_ref: memory_ref.WeightRefs,
    # Outputs.
    out_ref: jax.Array,
    tap_ref: jax.Array,
    conv_state_out_ref: jax.Array,
    recurrent_state_out_ref: jax.Array,
    # Scratches.
    carry_conv_scratch_ref: jax.Array | None,
    carry_recurrent_scratch_ref: jax.Array | None,
    *,
    cfg: config.GDNConfig,
) -> None:
  """Setup memory allocations and emit pipeline for running inner_kernel."""
  del conv_state_out_ref, recurrent_state_out_ref

  qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc = (
      memory_ref.create_allocs(
          metadata_ref=metadata_ref, qkv_ref=qkv_ref, b_ref=b_ref, a_ref=a_ref,
          out_ref=out_ref, conv_state_ref=conv_state_ref, recurrent_state_ref=recurrent_state_ref, cfg=cfg,
      )
  )

  # --- CREATE DEDICATED TAP ALLOCATOR ---
  pipeline_mode = pl.Buffered(buffer_count=cfg.num_buffers, use_lookahead=False)
  tap_alloc = memory_ref.OutBufferedRef.output(
      spec=pl.BlockSpec(
          memory_space=pltpu.VMEM,
          index_map=lambda i: (i,),
          pipeline_mode=pipeline_mode,
          block_shape=(cfg.seq_tile_size, cfg.chunk_size, cfg.num_v_heads, cfg.chunk_size)
      ),
      dtype_or_type=tap_ref,
      buffer_count=pipeline_mode.buffer_count,
      use_lookahead=pipeline_mode.use_lookahead,
      cfg=cfg,
      metadata_ref=metadata_ref,
  )

  num_tiles = metadata_ref.num_tiles[...]

  pipeline_func = pltpu.emit_pipeline(
      body=functools.partial(inner_kernel, cfg=cfg),
      grid=(num_tiles,),
      in_specs=(qkv_alloc.spec, b_alloc.spec, a_alloc.spec, conv_alloc.spec, recurrent_alloc.spec),
      out_specs=(out_alloc.spec, tap_alloc.spec), # <--- USING TAP_ALLOC.SPEC
  )

  @pl.with_scoped(
      allocations=(qkv_alloc, b_alloc, a_alloc, conv_alloc, recurrent_alloc, out_alloc, tap_alloc), # <--- USING TAP_ALLOC
  )
  def _run(allocations):
    pipeline_func(
        qkv_ref, b_ref, a_ref, conv_state_ref, recurrent_state_ref, 
        out_ref, tap_ref, # <--- PASSING TAP_REF
        scratches=(metadata_ref, weights_ref, carry_conv_scratch_ref, carry_recurrent_scratch_ref),
        allocations=allocations,
    )

  _run()


@jax.jit(
    donate_argnames=("conv_state", "recurrent_state"),
    static_argnames=(
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "kernel_size",
        "decode_tile_size",
        "mixed_tile_size",
        "zero_initialize_out",
        "compute_precision",
    ),
)
def fused_conv1d_gdn(
    qkv: jax.Array,  # [batch_size, n_kq * d_k * 2 + n_v * d_v = dim_size]
    b: jax.Array,  # [batch_size, n_v]
    a: jax.Array,  # [batch_size, n_v]
    conv_state: jax.Array,  # [num_seqs + 1, kernel_size - 1, dim_size]
    recurrent_state: jax.Array,  # [num_seqs + 1, nv, dk, dv]
    conv_weight: jax.Array,  # [kernel_size - 1, dim_size]
    conv_bias: jax.Array | None,  # [dim_size]
    a_log: jax.Array,  # [n_v]
    dt_bias: jax.Array,  # [n_v]
    query_start_loc: jax.Array,  # [num_seqs + 1]
    state_indices: jax.Array,  # [num_seqs]
    distribution: jax.Array,  # [3]
    seq_lens: jax.Array,  # [num_seqs]
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    zero_initialize_out: bool = True,
    compute_precision: jnp.dtype = jnp.float32.dtype,
    # TODO: Calculate tile size based on input dimensions.
    decode_tile_size: int = 4,
    mixed_tile_size: int = 64,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
  """Perform conv1d and gdn in a single fused kernel.

  Args:
    qkv: Mixed query, key, value input tensor of shape [batch_size, dim_size],
      where `dim_size = n_kq * d_k * 2 + n_v * d_v`.
    b: b tensor (for beta) of shape [batch_size, n_v].
    a: a tensor (for g) of shape [batch_size, n_v].
    conv_state: Convolution state cache tensor of shape [num_seqs + 1,
      kernel_size - 1, dim_size] containing the last (kernel_size - 1) tokens
      from the last sequence invocation. The first slot is a null block used for
      padded or invalid tokens. It may contain garbage data if it is a first
      invocation of a sequence.
    recurrent_state: Recurrent state cache tensor of shape [num_seqs + 1, n_v,
      d_k, d_v]. The first slot is a null block used for padded or invalid
      tokens. It may contain garbage data if it is a first invocation of a
      sequence.
    conv_weight: Convolution weight tensor of shape [kernel_size - 1, dim_size].
    conv_bias: Optional convolution bias tensor of shape [dim_size].
    a_log: a_log tensor of shape [n_v].
    dt_bias: dt_bias tensor of shape [n_v].
    query_start_loc: Start locations of sequences of shape [num_seqs + 1].
    state_indices: Indices mapping sequences to state cache slots of shape
      [num_seqs].
    distribution: Tensor of shape [3] int32 — [decode_end, prefill_end,
      mixed_end].
    seq_lens: Sequence lengths for each sequence of shape [num_seqs].
    n_kq: Number of key/query heads.
    n_v: Number of value heads.
    d_k: Key/query dimension.
    d_v: Value dimension.
    kernel_size: Convolution kernel size.
    zero_initialize_out: Whether to zero-initialize the output buffer before
      executing non-batched sequences.
    compute_precision: Computation precision dtype.
    decode_tile_size: Tile size along sequence dimension for decode sequences.
    mixed_tile_size: Tile size along token/chunk dimension for prefill/mixed
      sequences.

  Returns:
    (new_conv_state, new_recurrent_state): Updated convolution state cache and
      recurrent state cache tensors.
    out: Fused output tensor.
  """
  # TODO: Support bf16
  act_out_dtype = qkv.dtype
  conv_out_dtype = conv_state.dtype
  recurrent_out_dtype = recurrent_state.dtype

  qkv = qkv.astype(jnp.float32)
  b = b.astype(jnp.float32)
  a = a.astype(jnp.float32)
  conv_state = conv_state.astype(jnp.float32)

  # Step 1: Validate inputs.
  num_seqs = state_indices.size
  batch_size, dim = qkv.shape
  assert conv_weight.shape == (dim, 1, kernel_size)
  if conv_bias is not None:
    assert conv_bias.shape == (dim,)
  assert query_start_loc.shape == (num_seqs + 1,)
  assert state_indices.shape == (num_seqs,)
  assert distribution.shape == (3,)
  act_in_dtype = qkv.dtype
  assert a.dtype == b.dtype == qkv.dtype == act_in_dtype

  num_lanes = pltpu.get_tpu_info().num_lanes
  packing = 4 // act_in_dtype.itemsize
  padded_batch_size = pl.cdiv(batch_size, packing) * packing
  decode_tile_size = min(decode_tile_size, batch_size)
  mixed_tile_size = min(mixed_tile_size, batch_size)
  aligned_num_v_heads = pl.cdiv(n_v, num_lanes) * num_lanes

  batch_padding_size = padded_batch_size - batch_size
  num_v_padding_size = aligned_num_v_heads - n_v
  qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
  b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
  a = jnp.pad(a, ((0, batch_padding_size), (0, num_v_padding_size)))

  qkv = qkv.reshape(padded_batch_size, 1, -1)
  b = b.reshape(padded_batch_size, 1, -1)
  a = a.reshape(padded_batch_size, 1, -1)

  # Step 3: States and weights pre-processing.
  # To eliminate runtime cost, this logic can be moved into model loading
  conv_state_shape = conv_state.shape
  conv_state = conv_state.reshape(-1, kernel_size - 1, 1, dim)
  conv_weight = conv_weight.swapaxes(0, 2).astype(jnp.float32)
  conv_bias = conv_bias.astype(jnp.float32) if conv_bias is not None else None

  # Step 4: Wrap inputs for the kernel.
  conv_weights = memory_ref.ConvWeightsRef(weight=conv_weight, bias=conv_bias)
  gdn_weights = memory_ref.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias)
  weights = memory_ref.WeightRefs(conv=conv_weights, gdn=gdn_weights)

  # Step 5: Create specs.
  smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)
  vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
  hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
  weights_spec = jax.tree.map(lambda _: vmem_spec, weights)

  def call_kernel(
      in_conv_state: jax.Array,
      in_recurrent_state: jax.Array,
      in_act: jax.Array | None,
      mode: config.GDNMode,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    if mode == config.GDNMode.BATCHED:
      tile_size = decode_tile_size
    else:
      tile_size = mixed_tile_size

    cfg = config.GDNConfig(
        mode=mode,
        batch_size=padded_batch_size,
        kernel_size=kernel_size,
        tile_size=tile_size,
        dim_size=dim,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=d_k,
        v_head_dim=d_v,
        dtypes=config.Dtypes(
            act_in=act_in_dtype,
            act_out=act_out_dtype,
            compute=compute_precision,
            recurrent_state=in_recurrent_state.dtype,
            conv_state=in_conv_state.dtype,
        ),
    )

    # Step 6: Metadata preprocessing. Will be executed multiple times per-layer
    # but will be CSEed by compiler.
    if mode == config.GDNMode.BATCHED:
      metadata_obj = metadata.compute_batched_seq_metadata(
          cfg=cfg,
          seq_lens=seq_lens,
          query_start_loc=query_start_loc,
          state_indices=state_indices,
          end_seq=distribution[0],
      )
    else:
      metadata_obj = metadata.compute_per_seq_metadata(
          cfg=cfg,
          seq_lens=seq_lens,
          query_start_loc=query_start_loc,
          state_indices=state_indices,
          start_seq=distribution[0],
          end_seq=distribution[-1],
      )

    metadata_spec = jax.tree.map(lambda _: smem_spec, metadata_obj)

    # Step 7: Handle case where write needs to be done in existing out.
    in_out_spec = None
    input_output_aliases = {len(metadata_obj) + 3: 1, len(metadata_obj) + 4: 2}
    out_shape = cfg.get_out_shape()

    if in_act is None and zero_initialize_out:
      in_act = jnp.zeros_like(out_shape)
    if in_act is not None:
      out_shape = in_act
      in_out_spec = hbm_spec
      input_output_aliases[len(metadata_obj) + 5] = 0
    # Create the correct HBM shape for the Tap
    tap_shape = jax.ShapeDtypeStruct(
        (padded_batch_size, cfg.num_v_heads, cfg.chunk_size), 
        jnp.float32
    )

    res = pl.pallas_call(
        functools.partial(outer_kernel, cfg=cfg),
        out_shape=(out_shape, in_conv_state, in_recurrent_state, tap_shape), # <--- USING TAP_SHAPE
        in_specs=(
            metadata_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            hbm_spec,
            in_out_spec,
            weights_spec,
        ),
        out_specs=(hbm_spec, hbm_spec, hbm_spec, hbm_spec),
        scratch_shapes=cfg.get_scratch_shape_dict(),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=cfg.get_vmem_limit_bytes(),
        ),
        name=cfg.get_kernel_name(),
        metadata=cfg.get_metadata(),
        input_output_aliases=input_output_aliases,
    )(
        metadata_obj,
        qkv,
        b,
        a,
        in_conv_state,
        in_recurrent_state,
        in_act,
        weights,
    )
    out_act, out_conv, out_rec, out_tap = res
    return (out_conv, out_rec), out_act, out_tap

  out_act, out_conv_state, out_recurrent_state = call_kernel(
      conv_state, recurrent_state, None, config.GDNMode.BATCHED
  )
  out_act, out_conv_state, out_recurrent_state = call_kernel(
      out_conv_state, out_recurrent_state, out_act, config.GDNMode.PER_SEQ
  )

  out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
  out_conv_state = out_conv_state.astype(conv_out_dtype)
  out_conv_state = out_conv_state.reshape(conv_state_shape)
  out_recurrent_state = out_recurrent_state.astype(recurrent_out_dtype)

  return (out_conv_state, out_recurrent_state), out_act

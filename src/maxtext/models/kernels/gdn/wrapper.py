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

"""Top-level Pallas kernel wrapper for fused Conv1D-GDN with triangular inverse caching."""

import functools

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

try:
  from maxtext.models.kernels.gdn import compute_conv1d
  from maxtext.models.kernels.gdn import compute_gdn
  from maxtext.models.kernels.gdn import config
  from maxtext.models.kernels.gdn import memory_ref
  from maxtext.models.kernels.gdn import metadata
  from maxtext.models.kernels.gdn import tiling
  from maxtext.models.kernels.gdn import vmem_ldst
except (ImportError, ModuleNotFoundError):
  try:
    from maxtext.src.maxtext.models.kernels.gdn import compute_conv1d
    from maxtext.src.maxtext.models.kernels.gdn import compute_gdn
    from maxtext.src.maxtext.models.kernels.gdn import config
    from maxtext.src.maxtext.models.kernels.gdn import memory_ref
    from maxtext.src.maxtext.models.kernels.gdn import metadata
    from maxtext.src.maxtext.models.kernels.gdn import tiling
    from maxtext.src.maxtext.models.kernels.gdn import vmem_ldst
  except (ImportError, ModuleNotFoundError):
    from . import compute_conv1d
    from . import compute_gdn
    from . import config
    from . import memory_ref
    from . import metadata
    from . import tiling
    from . import vmem_ldst


def inner_kernel(
    # Inputs.
    qkv_slot_ref: jax.Ref,  # [seq, chunk, 1, dim_size]
    b_slot_ref: jax.Ref,  # [seq, chunk, 1, num_v_heads]
    a_slot_ref: jax.Ref,  # [seq, chunk, 1, num_v_heads]
    conv_state_slot_ref: jax.Ref,  # [seq, prev_kernel_size, 1, dim_size]
    recurrent_slot_ref: jax.Ref,  # [seq, num_v_heads, kq_head, v_head]
    # Outputs.
    out_slot_ref: jax.Array,  # [seq * chunk, num_v_heads, v_head]
    t_inv_slot_ref: jax.Array,  # [seq, num_v_heads, chunk, chunk]
    *args,
    cfg: config.GDNConfig,
    **kwargs,
) -> None:
  """Orchestrates computation of Conv1D and GDN for a single tile.

  This kernel acts as a facade adhering to strict separation of concerns. It
  operates VMEM reference without knowledge on DMA logic. Furthermore, the
  kernel invokes vmem_ldst to pre-processes data needed for compute and
  invokes compute_conv1d and compute_gdn for actual compute.
  """
  if cfg.mode == config.GDNMode.PER_SEQ:
    chunk_states_slot_ref = args[0]
    metadata_ref = args[1]
    weights_ref = args[2]
    carry_conv_scratch_ref = args[3] if len(args) > 3 else None
    carry_recurrent_scratch_ref = args[4] if len(args) > 4 else None
  else:
    chunk_states_slot_ref = None
    metadata_ref = args[0]
    weights_ref = args[1]
    carry_conv_scratch_ref = args[2] if len(args) > 2 else None
    carry_recurrent_scratch_ref = args[3] if len(args) > 3 else None

  p_id = pl.program_id(0)

  # Prepare states.
  real_sizes, prev_conv, prev_recurrent = vmem_ldst.load_and_select_states(
      metadata_ref=metadata_ref,
      p_id=p_id,
      conv_state_slot_ref=conv_state_slot_ref,
      recurrent_slot_ref=recurrent_slot_ref,
      carry_conv_scratch_ref=carry_conv_scratch_ref,
      carry_recurrent_scratch_ref=carry_recurrent_scratch_ref,
      cfg=cfg,
  )

  # Step 1: Conv1D.
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
  padding_size = cfg.aligned_num_v_heads - cfg.num_v_heads
  a_log = jnp.pad(weights_ref.gdn.a_log[...], ((0, padding_size)))
  dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], ((0, padding_size)))

  if cfg.chunk_size == 1:
    q_compact, k_compact, v_compact, b_compact, a_compact = (
        vmem_ldst.load_activation_as_compact(
            qkv_vreg=qkv_out_compact,
            qkv_vmem_ref=qkv_slot_ref,
            b_vmem_ref=b_slot_ref,
            a_vmem_ref=a_slot_ref,
            cfgs=cfg,
        )
    )

    out, new_recurrent_state, t_inv = compute_gdn.recurrent_gdn(
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
  else:
    q_large, k_large, v_large, b_large, a_large = (
        vmem_ldst.load_activation_as_large(
            qkv_vreg=qkv_out_compact,
            qkv_vmem_ref=qkv_slot_ref,
            b_vmem_ref=b_slot_ref,
            a_vmem_ref=a_slot_ref,
            cfgs=cfg,
        )
    )

    out, new_recurrent_state, t_inv = compute_gdn.chunked_gdn(
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

  # Store output, recurrent, and t_inv to vmem.
  out_slot_ref[...] = out.astype(out_slot_ref.dtype)
  recurrent_slot_ref[...] = new_recurrent_state.astype(recurrent_slot_ref.dtype)
  t_inv_slot_ref[...] = t_inv.astype(t_inv_slot_ref.dtype)
  if chunk_states_slot_ref is not None:
    chunk_states_slot_ref[...] = prev_recurrent.astype(
        chunk_states_slot_ref.dtype
    )

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
    conv_state_out_ref: jax.Array,
    recurrent_state_out_ref: jax.Array,
    t_inv_ref: jax.Array,
    *args,
    carry_conv_scratch_ref: jax.Array | None = None,
    carry_recurrent_scratch_ref: jax.Array | None = None,
    cfg: config.GDNConfig,
    **kwargs,
) -> None:
  """Setup memory allocations and emit pipeline for running inner_kernel."""
  del conv_state_out_ref, recurrent_state_out_ref

  chunk_states_ref = (
      args[0]
      if (len(args) > 0 and cfg.mode == config.GDNMode.PER_SEQ)
      else None
  )

  allocs = memory_ref.create_allocs(
      metadata_ref=metadata_ref,
      qkv_ref=qkv_ref,
      b_ref=b_ref,
      a_ref=a_ref,
      out_ref=out_ref,
      conv_state_ref=conv_state_ref,
      recurrent_state_ref=recurrent_state_ref,
      cfg=cfg,
      t_inv_ref=t_inv_ref,
      chunk_states_ref=chunk_states_ref,
  )
  qkv_alloc = allocs[0]
  b_alloc = allocs[1]
  a_alloc = allocs[2]
  conv_alloc = allocs[3]
  recurrent_alloc = allocs[4]
  out_alloc = allocs[5]
  t_inv_alloc = allocs[6]
  chunk_states_alloc = allocs[7] if len(allocs) > 7 else None

  num_tiles = metadata_ref.num_tiles[...]

  out_specs = [out_alloc.spec, t_inv_alloc.spec]
  if chunk_states_alloc is not None:
    out_specs.append(chunk_states_alloc.spec)

  pipeline_func = pltpu.emit_pipeline(
      body=functools.partial(
          inner_kernel,
          cfg=cfg,
      ),
      grid=(num_tiles,),
      in_specs=(
          qkv_alloc.spec,
          b_alloc.spec,
          a_alloc.spec,
          conv_alloc.spec,
          recurrent_alloc.spec,
      ),
      out_specs=tuple(out_specs),
  )

  @pl.with_scoped(allocations=allocs)
  def _run(allocations):
    out_args = [out_ref, t_inv_ref]
    if chunk_states_ref is not None:
      out_args.append(chunk_states_ref)
    pipeline_func(
        qkv_ref,
        b_ref,
        a_ref,
        conv_state_ref,
        recurrent_state_ref,
        *out_args,
        scratches=(
            metadata_ref,
            weights_ref,
            carry_conv_scratch_ref,
            carry_recurrent_scratch_ref,
        ),
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
    decode_tile_size: int | None = None,
    mixed_tile_size: int | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array], jax.Array, jax.Array]:
  """Perform conv1d and gdn in a single fused kernel, returning (out, states, t_inv, chunk_states)."""
  act_in_dtype = qkv.dtype
  act_out_dtype = qkv.dtype
  conv_out_dtype = conv_state.dtype
  recurrent_out_dtype = recurrent_state.dtype
  assert a.dtype == b.dtype == qkv.dtype == act_in_dtype

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

  num_lanes = pltpu.get_tpu_info().num_lanes
  packing = 4 // act_in_dtype.itemsize
  padded_batch_size = pl.cdiv(batch_size, packing) * packing
  conv_state_dim_size = conv_state.shape[-1]

  decode_tile_size, mixed_tile_size = tiling.get_tile_sizes(
      batch_size=batch_size,
      num_seqs=num_seqs,
      padded_batch_size=padded_batch_size,
      n_kq=n_kq,
      n_v=n_v,
      d_k=d_k,
      d_v=d_v,
      kernel_size=kernel_size,
      conv_state_dim_size=conv_state_dim_size,
      act_in_dtype=act_in_dtype,
      act_out_dtype=act_out_dtype,
      conv_state_dtype=conv_state.dtype,
      recurrent_state_dtype=recurrent_state.dtype,
      num_lanes=num_lanes,
      decode_tile_size=decode_tile_size,
      mixed_tile_size=mixed_tile_size,
  )

  batch_padding_size = padded_batch_size - batch_size
  aligned_num_v_heads = tiling.align_to(n_v, num_lanes)
  num_v_padding_size = aligned_num_v_heads - n_v
  qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
  b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
  a = jnp.pad(a, ((0, batch_padding_size), (0, num_v_padding_size)))

  qkv = qkv.reshape(padded_batch_size, 1, -1)
  b = b.reshape(padded_batch_size, 1, -1)
  a = a.reshape(padded_batch_size, 1, -1)

  # Step 3: States and weights pre-processing.
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
  ) -> tuple[jax.Array, ...]:
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

    in_out_spec = None
    input_output_aliases = {len(metadata_obj) + 3: 1, len(metadata_obj) + 4: 2}
    out_shape = cfg.get_out_shape()

    if in_act is None and zero_initialize_out:
      in_act = jnp.zeros_like(out_shape)
    if in_act is not None:
      out_shape = in_act
      in_out_spec = hbm_spec
      input_output_aliases[len(metadata_obj) + 5] = 0

    num_chunks = cfg.batch_size // cfg.chunk_size
    t_inv_shape = jax.ShapeDtypeStruct(
        (num_chunks, cfg.num_v_heads, cfg.chunk_size, cfg.chunk_size),
        cfg.dtypes.compute,
    )

    if mode == config.GDNMode.PER_SEQ:
      chunk_states_shape = jax.ShapeDtypeStruct(
          (num_chunks, cfg.num_v_heads, cfg.kq_head_dim, cfg.v_head_dim),
          cfg.dtypes.compute,
      )
      out_shape_tuple = (
          out_shape,
          in_conv_state,
          in_recurrent_state,
          t_inv_shape,
          chunk_states_shape,
      )
      out_specs_tuple = (hbm_spec, hbm_spec, hbm_spec, hbm_spec, hbm_spec)
    else:
      out_shape_tuple = (
          out_shape,
          in_conv_state,
          in_recurrent_state,
          t_inv_shape,
      )
      out_specs_tuple = (hbm_spec, hbm_spec, hbm_spec, hbm_spec)

    return pl.pallas_call(
        functools.partial(outer_kernel, cfg=cfg),
        out_shape=out_shape_tuple,
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
        out_specs=out_specs_tuple,
        scratch_shapes=cfg.get_scratch_shape_dict(),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=config.get_vmem_limit_bytes(),
        ),
        name=cfg.get_kernel_name(),
        metadata=cfg.get_metadata(),
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

  out_act, out_conv_state, out_recurrent_state, _ = call_kernel(
      conv_state, recurrent_state, None, config.GDNMode.BATCHED
  )
  out_act, out_conv_state, out_recurrent_state, t_inv, chunk_states = (
      call_kernel(
          out_conv_state, out_recurrent_state, out_act, config.GDNMode.PER_SEQ
      )
  )

  out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
  out_conv_state = out_conv_state.astype(conv_out_dtype)
  out_conv_state = out_conv_state.reshape(conv_state_shape)
  out_recurrent_state = out_recurrent_state.astype(recurrent_out_dtype)

  return out_act, (out_conv_state, out_recurrent_state), t_inv, chunk_states

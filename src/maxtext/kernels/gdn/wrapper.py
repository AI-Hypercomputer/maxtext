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

from . import compute_conv1d
from . import compute_gdn
from . import config
from . import memory_ref
from . import metadata
from . import tiling
from . import vmem_ldst


def _pack_fwd_segment_metadata(
    segment_ids: jax.Array,
    conv_halo_seg: jax.Array | None,
    init_seg: jax.Array | None,
    num_seqs: int,
    chunk_size: int,
    kernel_size: int,
) -> tuple[jax.Array, jax.Array]:
  """Packs per-token signed active segment IDs and chunk-boundary auxiliary metadata."""
  prev_kernel_size = kernel_size - 1
  if chunk_size < prev_kernel_size + 1:
    raise ValueError(
        f"GDN sequence packing requires chunk_size >= kernel_size (got chunk_size={chunk_size},"
        f" kernel_size={kernel_size}): each chunk header stores {prev_kernel_size} conv-halo"
        " segment IDs plus the previous chunk's segment ID."
    )
  seg_2d = segment_ids.reshape(num_seqs, -1)
  seq_len = seg_2d.shape[1]
  if seq_len % chunk_size != 0:
    raise ValueError(
        f"GDN sequence packing requires seq_len to be a multiple of chunk_size (got seq_len={seq_len},"
        f" chunk_size={chunk_size})."
    )
  num_chunks = seq_len // chunk_size

  s_enc = compute_conv1d.encode_segment_ids(seg_2d, init_seg=init_seg)
  s_enc_flat = s_enc.reshape(-1)

  s_valid = jnp.maximum(s_enc, 0.0)
  if conv_halo_seg is not None:
    halo_0 = jnp.maximum(
        conv_halo_seg.astype(jnp.float32).reshape(num_seqs, prev_kernel_size),
        0.0,
    )
  else:
    halo_0 = jnp.zeros((num_seqs, prev_kernel_size), dtype=jnp.float32)
  s_valid_3d = s_valid.reshape(num_seqs, num_chunks, chunk_size)
  if prev_kernel_size > 0:
    chunk_halos = jnp.concatenate(
        [halo_0[:, None, :], s_valid_3d[:, :-1, chunk_size - prev_kernel_size :]],
        axis=1,
    )
  else:
    chunk_halos = jnp.zeros((num_seqs, num_chunks, 0), dtype=jnp.float32)

  active_3d = jnp.abs(s_enc.reshape(num_seqs, num_chunks, chunk_size))
  active_end = active_3d[:, :, -1]
  if init_seg is not None:
    init_active = jnp.abs(init_seg.astype(jnp.float32).reshape(num_seqs, 1))
  else:
    init_active = jnp.zeros((num_seqs, 1), dtype=jnp.float32)
  seg_prev_all = jnp.concatenate([init_active, active_end[:, :-1]], axis=1)[..., None]

  seg_aux_header = jnp.concatenate([chunk_halos, seg_prev_all], axis=-1)
  seg_aux = jnp.pad(
      seg_aux_header,
      ((0, 0), (0, 0), (0, chunk_size - (prev_kernel_size + 1))),
  )
  seg_aux_flat = seg_aux.reshape(-1)
  return s_enc_flat, seg_aux_flat


def inner_kernel(
    *refs,
    cfg: config.GDNConfig,
    alloc_names: tuple[str, ...],
    **kwargs,
) -> None:
  """Orchestrates computation of Conv1D and GDN for a single tile.

  This kernel acts as a facade adhering to strict separation of concerns. It
  operates VMEM reference without knowledge on DMA logic. Furthermore, the
  kernel invokes vmem_ldst to pre-processes data needed for compute and
  invokes compute_conv1d and compute_gdn for actual compute.

  `refs` holds the pipeline slots in `alloc_names` order (see
  memory_ref.create_allocs) followed by the scratches:
  (metadata_ref, weights_ref, carry_conv_scratch_ref,
  carry_recurrent_scratch_ref).
  """
  del kwargs
  num_slots = len(alloc_names)
  slots = dict(zip(alloc_names, refs[:num_slots]))
  scratches = refs[num_slots:]
  metadata_ref = scratches[0]
  weights_ref = scratches[1]
  carry_conv_scratch_ref = scratches[2] if len(scratches) > 2 else None
  carry_recurrent_scratch_ref = scratches[3] if len(scratches) > 3 else None

  qkv_slot_ref = slots["qkv"]  # [seq, chunk, 1, dim_size]
  b_slot_ref = slots["b"]  # [seq, chunk, 1, num_v_heads]
  a_slot_ref = slots["a"]  # [seq, chunk, 1, num_v_heads]
  conv_state_slot_ref = slots["conv"]  # [seq, prev_kernel_size, 1, dim_size]
  recurrent_slot_ref = slots["recurrent"]  # [seq, num_v_heads, kq_head, v_head]
  t_inv_in_slot_ref = slots.get("t_inv_in")  # [seq, num_v_heads, chunk, chunk]
  out_slot_ref = slots.get("out")  # [seq * chunk, num_v_heads, v_head]
  t_inv_slot_ref = slots["t_inv"]  # [seq, num_v_heads, chunk, chunk]
  chunk_states_slot_ref = slots.get("chunk_states")

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

  b_vreg = b_slot_ref[...].astype(jnp.float32) if cfg.has_seg_ids else None
  qkv_out_compact, new_conv_state = compute_conv1d.causal_conv1d(
      real_sizes=real_sizes,
      lhs=qkv_in_compact,
      conv_weight=conv_weight,
      conv_bias=conv_bias,
      cfg=cfg,
      b_vreg=b_vreg,
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
    assert not cfg.states_only and t_inv_in_slot_ref is None
    q_compact, k_compact, v_compact, b_compact, a_compact = vmem_ldst.load_activation_as_compact(
        qkv_vreg=qkv_out_compact,
        qkv_vmem_ref=qkv_slot_ref,
        b_vmem_ref=b_slot_ref,
        a_vmem_ref=a_slot_ref,
        cfgs=cfg,
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
    q_large, k_large, v_large, b_large, a_large = vmem_ldst.load_activation_as_large(
        qkv_vreg=qkv_out_compact,
        qkv_vmem_ref=qkv_slot_ref,
        b_vmem_ref=b_slot_ref,
        a_vmem_ref=a_slot_ref,
        cfgs=cfg,
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
        t_inv_in=t_inv_in_slot_ref[...] if t_inv_in_slot_ref is not None else None,
    )

  # Store output, recurrent, and t_inv to vmem.
  if out_slot_ref is not None:
    assert out is not None
    out_slot_ref[...] = out.astype(out_slot_ref.dtype)
  recurrent_slot_ref[...] = new_recurrent_state.astype(recurrent_slot_ref.dtype)
  t_inv_slot_ref[...] = t_inv.astype(t_inv_slot_ref.dtype)
  if chunk_states_slot_ref is not None:
    chunk_states_slot_ref[...] = prev_recurrent.astype(chunk_states_slot_ref.dtype)

  if carry_recurrent_scratch_ref is not None:
    carry_recurrent_scratch_ref[...] = new_recurrent_state


def outer_kernel(
    *args,
    carry_conv_scratch_ref: jax.Array | None = None,
    carry_recurrent_scratch_ref: jax.Array | None = None,
    cfg: config.GDNConfig,
    has_in_act: bool = True,
    **kwargs,
) -> None:
  """Setup memory allocations and emit pipeline for running inner_kernel.

  Positional refs: metadata_ref, qkv_ref, b_ref, a_ref, conv_state_ref,
  recurrent_state_ref, [in_act], [t_inv_in_ref], weights_ref, then outputs
  [out_ref], conv_state_out_ref, recurrent_state_out_ref, t_inv_ref,
  [chunk_states_ref]. Bracketed refs are present depending on `cfg` flags.
  """
  del kwargs
  args = list(args)
  metadata_ref: memory_ref.MetadataRef = args.pop(0)
  qkv_ref = args.pop(0)
  b_ref = args.pop(0)
  a_ref = args.pop(0)
  conv_state_ref = args.pop(0)
  recurrent_state_ref = args.pop(0)
  del has_in_act  # A None operand still occupies a positional slot.
  args.pop(0)  # in_act (aliased to out) or None.
  t_inv_in_ref = args.pop(0) if cfg.t_inv_input else None
  weights_ref: memory_ref.WeightRefs = args.pop(0)
  # Outputs.
  out_ref = None if cfg.states_only else args.pop(0)
  args.pop(0)  # conv_state_out_ref (aliased).
  args.pop(0)  # recurrent_state_out_ref (aliased).
  t_inv_ref = args.pop(0)
  has_chunk_states = cfg.mode == config.GDNMode.PER_SEQ and not cfg.states_only
  chunk_states_ref = args.pop(0) if has_chunk_states else None
  assert not args, len(args)

  allocs, alloc_names = memory_ref.create_allocs(
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
      t_inv_in_ref=t_inv_in_ref,
  )
  hbm_refs = {
      "qkv": qkv_ref,
      "b": b_ref,
      "a": a_ref,
      "conv": conv_state_ref,
      "recurrent": recurrent_state_ref,
      "t_inv_in": t_inv_in_ref,
      "out": out_ref,
      "t_inv": t_inv_ref,
      "chunk_states": chunk_states_ref,
  }
  num_inputs = alloc_names.index("out") if "out" in alloc_names else alloc_names.index("t_inv")
  in_specs = tuple(alloc.spec for alloc in allocs[:num_inputs])
  out_specs = tuple(alloc.spec for alloc in allocs[num_inputs:])

  num_tiles = metadata_ref.num_tiles[...]

  pipeline_func = pltpu.emit_pipeline(
      body=functools.partial(
          inner_kernel,
          cfg=cfg,
          alloc_names=alloc_names,
      ),
      grid=(num_tiles,),
      in_specs=in_specs,
      out_specs=out_specs,
  )

  @pl.with_scoped(allocations=allocs)
  def _run(allocations):
    pipeline_func(
        *(hbm_refs[name] for name in alloc_names),
        scratches=(
            metadata_ref,
            weights_ref,
            carry_conv_scratch_ref,
            carry_recurrent_scratch_ref,
        ),
        allocations=allocations,
    )

  # pylint: disable=no-value-for-parameter
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
        "is_prefill_only",
        "use_qk_norm_in_gdn",
        "states_only",
        "transition_in_state",
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
    is_prefill_only: bool = False,
    use_qk_norm_in_gdn: bool = True,
    segment_ids: jax.Array | None = None,
    conv_halo_seg: jax.Array | None = None,
    init_seg: jax.Array | None = None,
    states_only: bool = False,
    t_inv_in: jax.Array | None = None,  # [num_chunks, n_v, chunk, chunk]
    transition_in_state: bool = False,
) -> tuple[jax.Array | None, tuple[jax.Array, jax.Array], jax.Array, jax.Array | None]:
  """Perform conv1d and gdn in a single fused kernel, returning (out, states, t_inv, chunk_states).

  Experimental options (all default to the original behavior):
    states_only: (F2) skip `out` / `chunk_states` (returned as None) and the
      q-dependent compute; only conv/recurrent states and t_inv are produced.
    t_inv_in: (F3) reuse a previously computed `t_inv` (same layout as the
      returned one) instead of recomputing the Gram matrix and its inverse.
    transition_in_state: (F5) `recurrent_state` is [num_states, n_v, d_k, d_v + d_k];
      the extra [d_k, d_k] block is carried through the same per-chunk update as
      the state, so (starting from the identity) it returns the local state
      transition M_local. Requires `states_only`.
  """
  act_in_dtype = qkv.dtype
  act_out_dtype = qkv.dtype
  conv_out_dtype = conv_state.dtype
  recurrent_out_dtype = recurrent_state.dtype
  assert a.dtype == b.dtype == qkv.dtype == act_in_dtype

  qkv = qkv.astype(jnp.float32)
  if transition_in_state:
    assert states_only and is_prefill_only, "transition_in_state requires states_only, prefill-only"
    assert recurrent_state.shape[-1] == d_v + d_k, (recurrent_state.shape, d_v, d_k)
  b = b.astype(jnp.float32)
  a = a.astype(jnp.float32)
  conv_state = conv_state.astype(jnp.float32)
  # Unpadded inputs, used to extract the packed next conv state below.
  qkv_in = qkv
  conv_state_in = conv_state

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

  if is_prefill_only:
    # Prefill-only calls carry num_seqs equal-length sequences. t_inv/chunk_states hold
    # batch_size // chunk_size chunks, so a partial chunk would be written out of bounds.
    seq_len_per_seq = batch_size // max(1, num_seqs)
    if batch_size % mixed_tile_size != 0 or seq_len_per_seq % mixed_tile_size != 0:
      raise ValueError(
          f"GDN prefill kernel requires batch_size ({batch_size}) and per-sequence length"
          f" ({seq_len_per_seq}) to be multiples of chunk_size ({mixed_tile_size})."
      )

  has_seg_ids = segment_ids is not None
  if has_seg_ids and not is_prefill_only:
    raise ValueError(
        "GDN sequence packing (segment_ids) is only supported for prefill-only calls" " (is_prefill_only=True)."
    )
  if has_seg_ids and conv_halo_seg is None and init_seg is None:
    segment_ids = compute_conv1d.canonicalize_segment_ids(segment_ids.reshape(num_seqs, -1))
    has_init_per_seq = (seq_lens - (query_start_loc[1:] - query_start_loc[:-1])) > 0
    halo_ones, init_ones = compute_conv1d.initial_state_segment_metadata(segment_ids, kernel_size)
    conv_halo_seg = jnp.where(has_init_per_seq[:, None], halo_ones, 0.0)
    init_seg = jnp.where(has_init_per_seq, init_ones, 0.0)

  extra_lanes = 2 if has_seg_ids else 0
  batch_padding_size = padded_batch_size - batch_size
  aligned_num_v_heads = tiling.align_to(n_v + extra_lanes, num_lanes)
  num_v_padding_size = aligned_num_v_heads - n_v
  qkv = jnp.pad(qkv, ((0, batch_padding_size), (0, 0)))
  b = jnp.pad(b, ((0, batch_padding_size), (0, num_v_padding_size)))
  a = jnp.pad(a, ((0, batch_padding_size), (0, num_v_padding_size)))
  if has_seg_ids:
    s_enc_flat, seg_aux_flat = _pack_fwd_segment_metadata(
        segment_ids=segment_ids,
        conv_halo_seg=conv_halo_seg,
        init_seg=init_seg,
        num_seqs=num_seqs,
        chunk_size=mixed_tile_size,
        kernel_size=kernel_size,
    )
    b = b.at[:batch_size, n_v].set(s_enc_flat)
    b = b.at[:batch_size, n_v + 1].set(seg_aux_flat)

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
  ) -> tuple[jax.Array | None, jax.Array, jax.Array, jax.Array, jax.Array | None]:
    if mode == config.GDNMode.BATCHED:
      tile_size = decode_tile_size
    else:
      tile_size = mixed_tile_size
    # The experimental options only apply to the chunked (PER_SEQ) kernel.
    is_per_seq = mode == config.GDNMode.PER_SEQ
    use_t_inv_in = is_per_seq and t_inv_in is not None

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
        has_seg_ids=has_seg_ids,
        use_qk_norm_in_gdn=use_qk_norm_in_gdn,
        dtypes=config.Dtypes(
            act_in=act_in_dtype,
            act_out=act_out_dtype,
            compute=compute_precision,
            recurrent_state=in_recurrent_state.dtype,
            conv_state=in_conv_state.dtype,
        ),
        states_only=is_per_seq and states_only,
        t_inv_input=use_t_inv_in,
        transition_in_state=is_per_seq and transition_in_state,
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
          is_prefill_only=is_prefill_only,
      )

    metadata_spec = jax.tree.map(lambda _: smem_spec, metadata_obj)

    in_out_spec = None
    input_output_aliases = {len(metadata_obj) + 3: 1, len(metadata_obj) + 4: 2}
    out_shape = cfg.get_out_shape()

    if cfg.states_only:
      # F2: no `out` buffer at all.
      in_act = None
      out_shape = None
    elif in_act is None and zero_initialize_out:
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

    out_shape_list = [] if cfg.states_only else [out_shape]
    out_shape_list += [in_conv_state, in_recurrent_state, t_inv_shape]
    if cfg.states_only:
      # Aliased outputs moved up by one (no `out`).
      input_output_aliases = {len(metadata_obj) + 3: 0, len(metadata_obj) + 4: 1}
    if is_per_seq and not cfg.states_only:
      chunk_states_shape = jax.ShapeDtypeStruct(
          (num_chunks, cfg.num_v_heads, cfg.kq_head_dim, cfg.v_head_dim),
          cfg.dtypes.compute,
      )
      out_shape_list.append(chunk_states_shape)
    out_shape_tuple = tuple(out_shape_list)
    out_specs_tuple = tuple(hbm_spec for _ in out_shape_tuple)

    in_specs = [metadata_spec, hbm_spec, hbm_spec, hbm_spec, hbm_spec, hbm_spec, in_out_spec]
    operands = [metadata_obj, qkv, b, a, in_conv_state, in_recurrent_state, in_act]
    if use_t_inv_in:
      assert t_inv_in is not None
      assert t_inv_in.shape == t_inv_shape.shape, (t_inv_in.shape, t_inv_shape.shape)
      in_specs.append(hbm_spec)
      operands.append(t_inv_in)
    in_specs.append(weights_spec)
    operands.append(weights)

    results = pl.pallas_call(
        functools.partial(outer_kernel, cfg=cfg, has_in_act=in_act is not None),
        out_shape=out_shape_tuple,
        in_specs=tuple(in_specs),
        out_specs=out_specs_tuple,
        scratch_shapes=cfg.get_scratch_shape_dict(),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=config.get_vmem_limit_bytes(),
        ),
        name=cfg.get_kernel_name(),
        metadata=cfg.get_metadata(),
    )(*operands)

    results = list(results)
    r_out = None if cfg.states_only else results.pop(0)
    r_conv = results.pop(0)
    r_rec = results.pop(0)
    r_t_inv = results.pop(0)
    r_chunk_states = results.pop(0) if results else None
    return r_out, r_conv, r_rec, r_t_inv, r_chunk_states

  if not is_prefill_only:
    if states_only or t_inv_in is not None or transition_in_state:
      raise ValueError("states_only / t_inv_in / transition_in_state are only supported for prefill-only calls.")
    out_act, out_conv_state, out_recurrent_state, _, _ = call_kernel(
        conv_state, recurrent_state, None, config.GDNMode.BATCHED
    )
  else:
    out_act = None
    out_conv_state = conv_state
    out_recurrent_state = recurrent_state

  out_act, out_conv_state, out_recurrent_state, t_inv, chunk_states = call_kernel(
      out_conv_state, out_recurrent_state, out_act, config.GDNMode.PER_SEQ
  )

  if out_act is not None:
    out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
  out_conv_state = out_conv_state.astype(conv_out_dtype)
  out_conv_state = out_conv_state.reshape(conv_state_shape)
  if has_seg_ids and segment_ids is not None:
    masked_cs = compute_conv1d.extract_segment_conv_state_split(
        conv_state_in[state_indices],
        qkv_in.reshape(num_seqs, -1, dim),
        segment_ids.reshape(num_seqs, -1),
        kernel_size,
        conv_halo_seg,
    ).astype(conv_out_dtype)
    out_conv_state = out_conv_state.at[state_indices].set(masked_cs)
  out_recurrent_state = out_recurrent_state.astype(recurrent_out_dtype)

  return out_act, (out_conv_state, out_recurrent_state), t_inv, chunk_states

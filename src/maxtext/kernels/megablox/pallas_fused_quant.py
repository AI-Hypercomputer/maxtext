# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VMEM-fused Pallas Mosaic TPU kernels for MoE 3D GMM validation and GMM backward quantization."""

import contextlib
import functools
import math
from typing import Any, cast

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2
from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2
from maxtext.kernels.megablox import relayout
import numpy as np
import qwix.pallas as qpl

LANES = 128
_DEFAULT_TILE_M = 512
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024
_HBM_SPEC = pl.BlockSpec(memory_space=pltpu.HBM)
_TINY_SQRT = math.sqrt(float(np.finfo(np.float32).tiny))


def _align_up(val: int, mult: int) -> int:
  return max(mult, ((int(val) + mult - 1) // mult) * mult)


@contextlib.contextmanager
def _tpu_mesh_context():
  """Ensures a valid TPU7x AbstractMesh is active when running in CPU interpret mode or AOT checks."""
  needs_mesh = False
  try:
    if not pltpu.is_tpu_device():
      needs_mesh = True
  except (RuntimeError, ValueError, AttributeError, TypeError):
    needs_mesh = True
  if needs_mesh:
    abstract_mesh = jax.sharding.AbstractMesh(
        (),
        (),
        abstract_device=jax.sharding.AbstractDevice("TPU7x", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      yield
  else:
    yield


def _choose_tile_m(m: int, *, min_tile: int = 128, max_tile: int = _DEFAULT_TILE_M) -> int:
  for cand in (max_tile, 256, min_tile):
    if min_tile <= cand <= m:
      return cand
  return min_tile


def _absmax_scale(amax: jax.Array, qmax: float, scale_dtype: Any) -> jax.Array:
  """Matches `qwix._src.core.qarray.compute_scale_zero_point` symmetric `absmax` scale."""
  raw = amax.astype(jnp.float32) / jnp.float32(qmax)
  safe = jnp.where(raw < _TINY_SQRT, jnp.ones_like(raw), raw)
  return safe.astype(scale_dtype)


def can_use_3d_gmm_tpu(
    *,
    num_rows: int,
    num_local_experts: int,
    d_dim: int,
    m_tp: int,
    moe_fp8: bool,
    act_dtype: Any,
) -> bool:
  """Verifies via `make_gmm_configs` / `make_tgmm_configs` and XIR `memref_reshape` `(d0 // packing) % 8 == 0`."""
  min_d_align = 32 * LANES if moe_fp8 else 16 * LANES
  if d_dim % min_d_align != 0 or m_tp % LANES != 0 or num_local_experts < 1:
    return False
  padded_m = _align_up(num_rows, 256)
  d0 = d_dim // LANES
  out_dt = jnp.dtype(act_dtype)
  lhs_gu_dt = jnp.dtype(jnp.float8_e4m3fn if moe_fp8 else out_dt)
  rhs_dt = jnp.dtype(jnp.float8_e4m3fn if moe_fp8 else out_dt)
  bwd_dt = jnp.dtype(jnp.float8_e5m2 if moe_fp8 else out_dt)
  lhs_dn_bwd_dt = jnp.dtype(jnp.float8_e4m3fn if moe_fp8 else out_dt)

  gs = jax.ShapeDtypeStruct((num_local_experts,), jnp.int32)
  goff = jax.ShapeDtypeStruct((1,), jnp.int32)
  gu_rhs = jax.ShapeDtypeStruct((num_local_experts, d_dim, 2 * m_tp), rhs_dt)
  gu_rscale = jax.ShapeDtypeStruct((num_local_experts, 1, 1, 2 * m_tp), jnp.float32) if moe_fp8 else None
  dn_rhs = jax.ShapeDtypeStruct((num_local_experts, m_tp, d_dim), rhs_dt)
  dn_rscale = jax.ShapeDtypeStruct((num_local_experts, 1, 1, d_dim), jnp.float32) if moe_fp8 else None

  try:
    with _tpu_mesh_context():
      if not pltpu.is_tpu_device():
        return False
      vmem_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
      # 1. gate_up FWD: 3D lhs -> 2D out
      gmm_v2.make_gmm_configs(
          jax.ShapeDtypeStruct((padded_m, d0, LANES), lhs_gu_dt),  # pyrefly: ignore[bad-argument-type]
          gu_rhs,  # pyrefly: ignore[bad-argument-type]
          gu_rscale,  # pyrefly: ignore[bad-argument-type]
          None,
          None,
          gs,  # pyrefly: ignore[bad-argument-type]
          goff,  # pyrefly: ignore[bad-argument-type]
          tile_info=cast(Any, gmm_v2.calculate_tiling),
          vmem_limit_bytes=vmem_limit,
          out_dtype=out_dt,
          acc_dtype=None,
          maybe_quantize_lhs=not moe_fp8,
          zero_initialize=True,
          out_is_3d=False,
      )
      # 2. gate_up DLHS: 2D dout -> 3D dlhs (transpose_rhs=True)
      gmm_v2.make_gmm_configs(
          jax.ShapeDtypeStruct((padded_m, 2 * m_tp), bwd_dt),  # pyrefly: ignore[bad-argument-type]
          gu_rhs,  # pyrefly: ignore[bad-argument-type]
          None,
          None,
          None,
          gs,  # pyrefly: ignore[bad-argument-type]
          goff,  # pyrefly: ignore[bad-argument-type]
          tile_info=cast(Any, gmm_v2.calculate_tiling),
          vmem_limit_bytes=vmem_limit,
          out_dtype=out_dt,
          acc_dtype=None,
          maybe_quantize_lhs=False,
          zero_initialize=True,
          transpose_rhs=True,
          out_is_3d=True,
      )
      # 3. gate_up DRHS: 3D lhs + 2D dout
      tgmm_v2.make_tgmm_configs(
          jax.ShapeDtypeStruct((padded_m, d0, LANES), lhs_gu_dt),  # pyrefly: ignore[bad-argument-type]
          jax.ShapeDtypeStruct((padded_m, 2 * m_tp), bwd_dt),  # pyrefly: ignore[bad-argument-type]
          jax.ShapeDtypeStruct((1, 1, 2 * m_tp), jnp.float32) if moe_fp8 else None,  # pyrefly: ignore[bad-argument-type]
          None,
          gs,  # pyrefly: ignore[bad-argument-type]
          num_local_experts,
          tile_info=tgmm_v2.calculate_tgmm_tiling,
          vmem_limit_bytes=vmem_limit,
          out_dtype=out_dt,
          acc_dtype=None,
          target_zero_ref_bytes=2 * 1024 * 1024,
      )
      # 4. down_proj FWD: 2D lhs -> 3D out
      gmm_v2.make_gmm_configs(
          jax.ShapeDtypeStruct((padded_m, m_tp), out_dt),  # pyrefly: ignore[bad-argument-type]
          dn_rhs,  # pyrefly: ignore[bad-argument-type]
          dn_rscale,  # pyrefly: ignore[bad-argument-type]
          None,
          None,
          gs,  # pyrefly: ignore[bad-argument-type]
          goff,  # pyrefly: ignore[bad-argument-type]
          tile_info=cast(Any, gmm_v2.calculate_tiling),
          vmem_limit_bytes=vmem_limit,
          out_dtype=out_dt,
          acc_dtype=None,
          maybe_quantize_lhs=moe_fp8,
          zero_initialize=True,
          out_is_3d=True,
      )
      # 5. down_proj DLHS: 3D dout -> 2D dlhs (transpose_rhs=True)
      gmm_v2.make_gmm_configs(
          jax.ShapeDtypeStruct((padded_m, d0, LANES), bwd_dt),  # pyrefly: ignore[bad-argument-type]
          dn_rhs,  # pyrefly: ignore[bad-argument-type]
          None,
          None,
          None,
          gs,  # pyrefly: ignore[bad-argument-type]
          goff,  # pyrefly: ignore[bad-argument-type]
          tile_info=cast(Any, gmm_v2.calculate_tiling),
          vmem_limit_bytes=vmem_limit,
          out_dtype=out_dt,
          acc_dtype=None,
          maybe_quantize_lhs=False,
          zero_initialize=True,
          transpose_rhs=True,
          out_is_3d=False,
      )
      # 6. down_proj DRHS: 2D lhs + 3D dout
      tgmm_v2.make_tgmm_configs(
          jax.ShapeDtypeStruct((padded_m, m_tp), lhs_dn_bwd_dt),  # pyrefly: ignore[bad-argument-type]
          jax.ShapeDtypeStruct((padded_m, d0, LANES), bwd_dt),  # pyrefly: ignore[bad-argument-type]
          jax.ShapeDtypeStruct((1, 1, d_dim), jnp.float32) if moe_fp8 else None,  # pyrefly: ignore[bad-argument-type]
          None,
          gs,  # pyrefly: ignore[bad-argument-type]
          num_local_experts,
          tile_info=tgmm_v2.calculate_tgmm_tiling,
          vmem_limit_bytes=vmem_limit,
          out_dtype=out_dt,
          acc_dtype=None,
          target_zero_ref_bytes=2 * 1024 * 1024,
      )
    return True
  except (ValueError, NotImplementedError, AssertionError):
    return False


def can_fuse_bwd_prepare_and_quantize(
    grad: jax.Array,
    lhs: jax.Array | qpl.QArray,
    rhs: jax.Array | qpl.QArray,
    quantization_rule: Any,
    *,
    use_tokamax_backend: bool,
    use_gmm_v2: bool,
    tc_unsort_cfg: Any,
) -> bool:
  """Returns True when `_gmm_bwd` operands match the fused 2-pass VMEM backward quantizer contract."""
  if not (use_tokamax_backend and use_gmm_v2 and tc_unsort_cfg is None and quantization_rule is not None):
    return False
  if not (isinstance(lhs, qpl.QArray) and isinstance(rhs, qpl.QArray)):
    return False
  if quantization_rule.bwd_qtype != jnp.float8_e5m2 or quantization_rule.bwd_calibration_method != "absmax":
    return False
  if quantization_rule.disable_channelwise_axes:
    return False
  if lhs.zero_point is not None or rhs.zero_point is not None:
    return False
  if lhs.scale.size not in (1, grad.shape[0]):
    return False
  r_scale = rhs.scale
  if r_scale.ndim == 3 and r_scale.shape[1] == 1:
    r_scale_2d_cols = r_scale.shape[2]
  elif r_scale.ndim == 2:
    r_scale_2d_cols = r_scale.shape[1]
  elif r_scale.size == 1:
    r_scale_2d_cols = -1
  else:
    return False
  if grad.ndim == 2:
    n_dim = grad.shape[1]
    if n_dim % LANES != 0:
      return False
  elif grad.ndim == 3:
    # `relayout._flat_u32_view` reshapes `[..., d0 // packing, 128] -> [-1, 128]` in VMEM, which XIR requires
    # `(d0 // 4) % 8 == 0` (`d0 % 32 == 0`, i.e. `n_dim % 4096 == 0`) for 8-bit `e5m2` outputs.
    if grad.shape[2] != LANES or grad.shape[1] % 32 != 0:
      return False
    n_dim = grad.shape[1] * LANES
  else:
    return False
  if r_scale_2d_cols not in (-1, n_dim):
    return False
  return n_dim <= 16384


@functools.partial(jax.jit, static_argnames=("tile_m", "is_3d", "interpret"))
def _fused_bwd_quant_2pass(
    grad_pad: jax.Array,
    group_ids_2d: jax.Array,
    rhs_scale_pad: jax.Array,
    lhs_scale_2d: jax.Array,
    *,
    tile_m: int,
    is_3d: bool,
    interpret: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """2-pass VMEM pipeline producing `(dlhs_qval, dlhs_scale, drhs_qval, drhs_scale)` without HBM `bf16` temporaries."""
  m_pad = grad_pad.shape[0]
  n_dim = grad_pad.shape[1] * LANES if is_3d else grad_pad.shape[1]
  d0 = grad_pad.shape[1] if is_3d else 0
  g_pad = rhs_scale_pad.shape[0]
  qmax = float(jnp.finfo(jnp.float8_e5m2).max)
  scale_dt = grad_pad.dtype
  num_tiles = m_pad // tile_m

  def pass1_body(
      step: pltpu.PipelineStep,
      g_ref,
      gid_ref,
      rscale_ref,
      lscale_ref,
      dlhs_q_ref,
      dlhs_s_ref,
      drhs_amax_ref,
  ):
    tid = step.index[0]
    g_2d = relayout.load_3d_as_2d(g_ref, tile_m, d0) if is_3d else g_ref[...]

    one_hot = (gid_ref[...] == lax.broadcasted_iota(jnp.int32, (tile_m, g_pad), 1)).astype(jnp.bfloat16)
    row_rscale = lax.dot_general(
        one_hot,
        rscale_ref[...].astype(scale_dt),
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    dlhs_bf16 = g_2d * row_rscale.astype(scale_dt)
    dlhs_amax = jnp.max(jnp.abs(dlhs_bf16), axis=-1, keepdims=True)
    dlhs_s = _absmax_scale(dlhs_amax, qmax, scale_dt)
    dlhs_q_2d = jnp.clip(dlhs_bf16 / dlhs_s, -qmax, qmax).astype(jnp.float8_e5m2)
    if is_3d:
      relayout.store_2d_as_3d(dlhs_q_ref, dlhs_q_2d, tile_m, d0)
    else:
      dlhs_q_ref[...] = dlhs_q_2d
    dlhs_s_ref[...] = dlhs_s

    drhs_bf16 = g_2d * lscale_ref[...].astype(scale_dt)
    tile_amax_n = jnp.broadcast_to(jnp.max(jnp.abs(drhs_bf16), axis=0, keepdims=True).astype(jnp.float32), (8, n_dim))

    @pl.when(tid == 0)
    def _init_amax():
      drhs_amax_ref[...] = jnp.zeros_like(drhs_amax_ref)

    drhs_amax_ref[...] = jnp.maximum(drhs_amax_ref[...], tile_amax_n)

  spec_grad = (
      pl.BlockSpec((tile_m, d0, LANES), lambda i: (i, 0, 0)) if is_3d else pl.BlockSpec((tile_m, n_dim), lambda i: (i, 0))
  )
  spec_m1 = pl.BlockSpec((tile_m, 1), lambda i: (i, 0))
  spec_rscale = pl.BlockSpec((g_pad, n_dim), lambda i: (0, 0), pipeline_mode=pl.Buffered(buffer_count=1))
  spec_amax = pl.BlockSpec((8, n_dim), lambda i: (0, 0))

  pass1_main = pltpu.emit_pipeline(
      pass1_body,
      grid=(num_tiles,),
      in_specs=(spec_grad, spec_m1, spec_rscale, spec_m1),
      out_specs=(spec_grad, spec_m1, spec_amax),
      dimension_semantics=(pltpu.ARBITRARY,),
      _explicit_indices=True,
  )
  qval_shape = (m_pad, d0, LANES) if is_3d else (m_pad, n_dim)
  flops_p1 = int(2 * m_pad * g_pad * n_dim + 6 * m_pad * n_dim)
  bytes_p1 = int(m_pad * n_dim * (grad_pad.dtype.itemsize + 1) + g_pad * n_dim * 4)
  with _tpu_mesh_context():
    dlhs_q_pad, dlhs_s_pad, drhs_amax_8 = pl.pallas_call(
        pass1_main,
        out_shape=(
            jax.ShapeDtypeStruct(qval_shape, jnp.float8_e5m2),
            jax.ShapeDtypeStruct((m_pad, 1), scale_dt),
            jax.ShapeDtypeStruct((8, n_dim), jnp.float32),
        ),
        in_specs=(_HBM_SPEC,) * 4,
        out_specs=(_HBM_SPEC,) * 3,
        cost_estimate=pl.CostEstimate(flops=flops_p1, transcendentals=0, bytes_accessed=bytes_p1),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
        interpret=interpret,
        name=f"gmm_bwd_fused_dlhs_quant_amax_tm{tile_m}_3d{int(is_3d)}",
    )(grad_pad, group_ids_2d, rhs_scale_pad, lhs_scale_2d)

  drhs_s_2d = _absmax_scale(drhs_amax_8[0:1], qmax, scale_dt)
  drhs_s_8 = jnp.broadcast_to(drhs_s_2d, (8, n_dim))

  def pass2_body(step: pltpu.PipelineStep, g_ref, lscale_ref, drhs_s_ref, drhs_q_ref):
    del step
    g_2d = relayout.load_3d_as_2d(g_ref, tile_m, d0) if is_3d else g_ref[...]
    drhs_bf16 = g_2d * lscale_ref[...].astype(scale_dt)
    drhs_q_2d = jnp.clip(drhs_bf16 / drhs_s_ref[0:1], -qmax, qmax).astype(jnp.float8_e5m2)
    if is_3d:
      relayout.store_2d_as_3d(drhs_q_ref, drhs_q_2d, tile_m, d0)
    else:
      drhs_q_ref[...] = drhs_q_2d

  spec_drhs_s = pl.BlockSpec((8, n_dim), lambda i: (0, 0), pipeline_mode=pl.Buffered(buffer_count=1))
  pass2_main = pltpu.emit_pipeline(
      pass2_body,
      grid=(num_tiles,),
      in_specs=(spec_grad, spec_m1, spec_drhs_s),
      out_specs=spec_grad,
      dimension_semantics=(pltpu.PARALLEL,),
      _explicit_indices=True,
  )
  bytes_p2 = int(m_pad * n_dim * (grad_pad.dtype.itemsize + 1))
  with _tpu_mesh_context():
    drhs_q_pad = pl.pallas_call(
        pass2_main,
        out_shape=jax.ShapeDtypeStruct(qval_shape, jnp.float8_e5m2),
        in_specs=(_HBM_SPEC,) * 3,
        out_specs=_HBM_SPEC,
        cost_estimate=pl.CostEstimate(flops=int(3 * m_pad * n_dim), transcendentals=0, bytes_accessed=bytes_p2),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
        interpret=interpret,
        name=f"gmm_bwd_fused_drhs_quant_tm{tile_m}_3d{int(is_3d)}",
    )(grad_pad, lhs_scale_2d, drhs_s_8)

  return dlhs_q_pad, dlhs_s_pad, drhs_q_pad, drhs_s_2d


def fused_bwd_prepare_and_quantize(
    grad: jax.Array,
    lhs: qpl.QArray,
    rhs: qpl.QArray,
    group_sizes: jax.Array,
    *,
    interpret: bool = False,
) -> tuple[qpl.QArray, qpl.QArray, jax.Array, jax.Array]:
  """Replaces `_bwd_prepare_inputs` + `_bwd_quantize_gradient` with 2 VMEM Pallas passes."""
  is_3d = grad.ndim == 3
  m_orig = grad.shape[0]
  n_dim = grad.shape[1] * LANES if is_3d else grad.shape[1]
  max_tile_m = 128 if n_dim > 8192 else (256 if n_dim > 4096 else _DEFAULT_TILE_M)
  tile_m = _choose_tile_m(m_orig, max_tile=max_tile_m)
  m_pad = _align_up(m_orig, tile_m)
  pad_rows = m_pad - m_orig

  r_scale = rhs.scale
  if r_scale.ndim == 3 and r_scale.shape[1] == 1:
    r_scale = r_scale.squeeze(axis=1)
  if r_scale.size == 1:
    r_scale_2d = jnp.broadcast_to(r_scale.reshape(1, 1).astype(jnp.float32), (1, n_dim))
  elif r_scale.shape[0] == 1:
    r_scale_2d = jnp.broadcast_to(r_scale.reshape(1, n_dim).astype(jnp.float32), (1, n_dim))
  else:
    r_scale_2d = r_scale.astype(jnp.float32)

  num_scale_groups = r_scale_2d.shape[0]
  g_pad = _align_up(max(LANES, num_scale_groups), LANES)
  if g_pad > num_scale_groups:
    r_scale_pad = jnp.pad(r_scale_2d, ((0, g_pad - num_scale_groups), (0, 0)))
  else:
    r_scale_pad = r_scale_2d

  if num_scale_groups == 1:
    gids_orig = jnp.zeros((m_orig,), dtype=jnp.int32)
  else:
    gids_orig = jnp.repeat(
        jnp.arange(num_scale_groups, dtype=jnp.int32),
        group_sizes.astype(jnp.int32),
        total_repeat_length=m_orig,
    )
  gids_pad = (
      jnp.pad(gids_orig, (0, pad_rows), constant_values=max(0, num_scale_groups - 1)) if pad_rows > 0 else gids_orig
  )

  if lhs.scale.size == 1:
    l_scale_orig = jnp.broadcast_to(lhs.scale.reshape(1).astype(jnp.float32), (m_orig,))
  else:
    l_scale_orig = jnp.reshape(lhs.scale, (m_orig,)).astype(jnp.float32)
  l_scale_pad = jnp.pad(l_scale_orig, (0, pad_rows), constant_values=0.0) if pad_rows > 0 else l_scale_orig

  if pad_rows > 0:
    grad_pad = jnp.pad(grad, ((0, pad_rows), *([(0, 0)] * (grad.ndim - 1))))
  else:
    grad_pad = grad

  dlhs_q_p, dlhs_s_p, drhs_q_p, drhs_s_2d = _fused_bwd_quant_2pass(
      grad_pad,
      gids_pad[:, None],
      r_scale_pad,
      l_scale_pad[:, None],
      tile_m=tile_m,
      is_3d=is_3d,
      interpret=interpret,
  )

  dlhs_q = dlhs_q_p[:m_orig] if pad_rows > 0 else dlhs_q_p
  dlhs_s_slice = dlhs_s_p[:m_orig] if pad_rows > 0 else dlhs_s_p
  dlhs_s = dlhs_s_slice.reshape(m_orig, 1, 1) if is_3d else dlhs_s_slice
  drhs_q = drhs_q_p[:m_orig] if pad_rows > 0 else drhs_q_p
  drhs_s = drhs_s_2d.reshape(1, grad.shape[1], LANES) if is_3d else drhs_s_2d

  dlhs_dout = qpl.QArray(qvalue=dlhs_q, scale=dlhs_s, zero_point=None, qtype=jnp.float8_e5m2)
  drhs_dout = qpl.QArray(qvalue=drhs_q, scale=drhs_s, zero_point=None, qtype=jnp.float8_e5m2)
  return dlhs_dout, drhs_dout, lhs.qvalue, rhs.qvalue

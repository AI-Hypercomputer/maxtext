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

"""Split (dgrad + wgrad) Pallas kernels for the fp8 MLA QKV-up backward.

Per device (one TensorCore), for the head shard of N heads, the kernels compute
the backward pass of the fp8 q up, bf16 k up and fp8 v up projections of
`dsv3_mla`:

* `qkv_up_dgrad` applies the transposed YaRN RoPE to dq, quantizes the q and v
  cotangents to e5m2 per (token block, head block) tile with a dynamic absmax
  scale computed in VMEM, and returns the head-unreduced `grad_cq` [B, T, Cq]
  and `grad_kv` = [grad_ckv, sum over heads of dk_pe] [B, T, Ckv + R]. It also
  returns the e5m2 q and v cotangents and their tile scales, so that the wgrad
  kernel reads them instead of redoing the RoPE and quantization of the bf16
  cotangents (and reads 2-2.7x fewer bytes).
* `qkv_up_wgrad` recomputes the normalized and statically quantized cq from
  q_down in VMEM (once per token block) and returns `grad_wq_up`, `grad_wk_up`
  and `grad_wv_up`, accumulated in f32 over the tokens. The normalized ckv is
  taken as an input: caching it in VMEM too does not fit at the production
  per-device batch of 2.

The bf16 cotangents are taken head-major, [B, N, T, D], the layout splash
attention produces them in. The weights and weight gradients are taken
transposed with their head and head-dim axes merged, e.g. wq_up as
[N * (QK + R), Cq]: XLA lays out the [C, N, D] weights C-minor, so this is a
bitcast. Each projection of a (token block, head block) tile is then one
matmul. grad_cq and grad_kv are output token-minor, [B, C, T], like their
consumers want them.
"""

import dataclasses
import functools
from typing import Any

import jax
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

_E5M2 = jnp.float8_e5m2
_VMEM_LIMIT_BYTES = int(0.9 * 64 * 1024 * 1024)
_NN = (((1,), (0,)), ((), ()))  # x @ y
_TN = (((0,), (0,)), ((), ()))  # x.T @ y


@dataclasses.dataclass(frozen=True)
class QkvUpBwdDims:
  """Static dims and constants of the QKV-up backward kernels.

  Attributes:
    qk_head_dim: Non-RoPE q / k head dim (QK).
    rope_head_dim: RoPE head dim (R).
    v_head_dim: v head dim (V).
    softmax_scale: Scale of the q cotangent.
    eps: RMSNorm epsilon of the q and kv norms.
    act_scale: Static fp8 scale of cq and ckv.
    act_qtype: fp8 dtype of cq and ckv.
    weight_scale: Static fp8 scale of wq_up and wv_up.
  """

  qk_head_dim: int
  rope_head_dim: int
  v_head_dim: int
  softmax_scale: float
  eps: float
  act_scale: float
  act_qtype: Any
  weight_scale: float

  @property
  def q_head_dim(self) -> int:
    """Full q / k head dim (QK + R)."""
    return self.qk_head_dim + self.rope_head_dim

  @property
  def act_max(self) -> float:
    """Largest |x| the static act quantization does not clip."""
    return self.act_scale * float(jnp.finfo(self.act_qtype).max)


def _e5m2_scale(amax):
  """Absmax scale of qwix's `absmax` calibration (1 for an all-zero tile)."""
  return jnp.where(amax > 0, amax / float(jnp.finfo(_E5M2).max), 1.0)


def _to_e5m2(x, inv_scale):
  # |x| * inv_scale <= the e5m2 max by construction of the absmax scale, so
  # unlike `quantization._xla_quantize_with_scale` no clip is needed.
  return (x.astype(jnp.float32) * inv_scale).astype(_E5M2)


def _abs_max(x):
  # Reduces over the rows in x's dtype, then over the lanes in f32.
  return jnp.max(jnp.max(jnp.abs(x), axis=0, keepdims=True).astype(jnp.float32))


def _rms_norm(x, scale, eps):
  """`ops.rms_norm` of the bf16 [bt, C] x with the bf16 [1, C] scale."""
  x = x.astype(jnp.float32)
  mean2 = jnp.mean(x * x, axis=-1, keepdims=True)
  y = (x * jax.lax.rsqrt(mean2 + eps)).astype(jnp.bfloat16)
  return (y.astype(jnp.float32) * scale.astype(jnp.float32)).astype(jnp.bfloat16)


def _static_quantize(x, dims: QkvUpBwdDims):
  fmax = float(jnp.finfo(dims.act_qtype).max)
  return jnp.clip(x.astype(jnp.float32) / dims.act_scale, -fmax, fmax).astype(dims.act_qtype)


def _row_chunks(num_rows, chunk, body):
  """Runs body(rows) over [chunk]-row slices of [0, num_rows)."""
  chunk = min(chunk, num_rows)

  def loop_body(i, carry):
    body(pl.ds(pl.multiple_of(i * chunk, chunk), chunk))
    return carry

  jax.lax.fori_loop(0, num_rows // chunk, loop_body, 0)


def _dgrad_kernel(
    dq_ref,
    dk_ref,
    dv_ref,
    kv_down_ref,
    kv_norm_scale_ref,
    cos_sin_ref,
    wq_ref,
    wk_ref,
    wv_ref,
    grad_cq_ref,
    grad_kv_ref,
    gq_ref,
    dv_q_ref,
    scales_ref,
    rope_ref,
    acc_q_ref,
    acc_k_ref,
    acc_v_ref,
    acc_pe_ref,
    *,
    dims: QkvUpBwdDims,
    chunk: int,
):
  """One (batch, token block, head block) step of `qkv_up_dgrad`."""
  n = pl.program_id(2)
  num_heads, block_t = dq_ref.shape[:2]
  qk, r = dims.qk_head_dim, dims.rope_head_dim
  h = r // 2

  # The transposed, unscaled YaRN rotation of the RoPE dims x = [x1, x2] of
  # each head is [x1 * cos1 + x2 * sin2, x2 * cos2 - x1 * sin1], like
  # `dsv3_mla._yarn_tail(..., transpose=True)`; the non-RoPE dims are not
  # rotated. The softmax scale commutes with the absmax quantization and is
  # folded into the dequantization scale.
  cs = cos_sin_ref[...].astype(jnp.float32)
  cos1, cos2, sin1, sin2 = (cs[:, i * h : (i + 1) * h] for i in range(4))
  amax_q = jnp.zeros((), jnp.float32)
  ropes = []
  for i in range(num_heads):
    amax_q = jnp.maximum(amax_q, _abs_max(dq_ref[i, :, :qk]))
    x = dq_ref[i, :, qk:].astype(jnp.float32)
    x1, x2 = x[:, :h], x[:, h:]
    rope = jnp.concatenate([x1 * cos1 + x2 * sin2, x2 * cos2 - x1 * sin1], axis=1).astype(
        jnp.bfloat16
    )  # Rounded like `_yarn_tail`'s output.
    amax_q = jnp.maximum(amax_q, _abs_max(rope))
    ropes.append(rope)
  rope_ref[...] = jnp.concatenate(ropes, axis=1)
  amax_v = jnp.zeros((), jnp.float32)
  for i in range(num_heads):
    amax_v = jnp.maximum(amax_v, _abs_max(dv_ref[i]))
  s_q, s_v = _e5m2_scale(amax_q), _e5m2_scale(amax_v)
  inv_q, inv_v = 1.0 / s_q, 1.0 / s_v
  row = jax.lax.broadcasted_iota(jnp.int32, scales_ref.shape, 0)
  scales_ref[...] = jnp.where(row == 0, s_q, s_v)
  q_dequant = s_q * (dims.softmax_scale * dims.weight_scale)
  v_dequant = s_v * dims.weight_scale

  def accumulate(acc_ref, rows, x, cols=slice(None)):
    # Branch-free: the first head block overwrites the accumulator.
    acc_ref[rows, cols] = jnp.where(n > 0, acc_ref[rows, cols], 0.0) + x

  def rows_body(rows):
    heads = []
    for i in range(num_heads):
      heads.append(dq_ref[i, rows, :qk].astype(jnp.float32))
      heads.append(rope_ref[rows, i * r : (i + 1) * r].astype(jnp.float32))
    gq_ref[rows] = _to_e5m2(jnp.concatenate(heads, axis=1), inv_q)
    accumulate(
        acc_q_ref,
        rows,
        jax.lax.dot_general(gq_ref[rows], wq_ref[...], _NN, preferred_element_type=jnp.float32) * q_dequant,
    )
    accumulate(
        acc_k_ref,
        rows,
        jax.lax.dot_general(
            jnp.concatenate([dk_ref[i, rows, :qk] for i in range(num_heads)], axis=1),
            wk_ref[...],
            _NN,
            preferred_element_type=jnp.float32,
        ),
    )
    dk_pe = dk_ref[0, rows, qk:].astype(jnp.float32)
    for i in range(1, num_heads):
      dk_pe += dk_ref[i, rows, qk:].astype(jnp.float32)
    accumulate(acc_pe_ref, rows, dk_pe, slice(None, r))
    dv_q_ref[rows] = _to_e5m2(
        jnp.concatenate([dv_ref[i, rows] for i in range(num_heads)], axis=1),
        inv_v,
    )
    accumulate(
        acc_v_ref,
        rows,
        jax.lax.dot_general(dv_q_ref[rows], wv_ref[...], _NN, preferred_element_type=jnp.float32) * v_dequant,
    )

  _row_chunks(block_t, chunk, rows_body)

  @pl.when(n == pl.num_programs(2) - 1)
  def _():
    ckv_dim = acc_k_ref.shape[1]

    def write_body(rows):
      ckv = _rms_norm(kv_down_ref[rows, :ckv_dim], kv_norm_scale_ref[...], dims.eps)
      # Straight-through estimator of ckv's static act quantization, which
      # only the fp8 v up projection applies (like `_qdot_bwd(...,
      # ste_x=ckv)`).
      keep = jnp.abs(ckv.astype(jnp.float32)) <= dims.act_max
      grad_ckv = acc_k_ref[rows] + jnp.where(keep, acc_v_ref[rows], 0.0)
      # The outputs are token-minor, like their consumers want them.
      grad_cq_ref[:, rows] = acc_q_ref[rows].T.astype(grad_cq_ref.dtype)
      grad_kv_ref[:ckv_dim, rows] = grad_ckv.T.astype(grad_kv_ref.dtype)
      grad_kv_ref[ckv_dim:, rows] = acc_pe_ref[rows].T[:r].astype(grad_kv_ref.dtype)

    _row_chunks(block_t, chunk, write_body)


def _wgrad_kernel(
    q_down_ref,
    ckv_ref,
    q_norm_scale_ref,
    gq_ref,
    dk_ref,
    dv_q_ref,
    scales_ref,
    grad_wq_ref,
    grad_wk_ref,
    grad_wv_ref,
    cq_q_ref,
    acc_wq_ref,
    acc_wk_ref,
    acc_wv_ref,
    *,
    dims: QkvUpBwdDims,
    chunk: int,
):
  """One (head block, batch, token block) step of `qkv_up_wgrad`."""
  n, b, t = pl.program_id(0), pl.program_id(1), pl.program_id(2)
  num_heads, block_t = dk_ref.shape[:2]
  qk = dims.qk_head_dim
  t_rows = pl.ds(pl.multiple_of(t * block_t, block_t), block_t)
  first = (b == 0) & (t == 0)
  last = (b == pl.num_programs(1) - 1) & (t == pl.num_programs(2) - 1)

  @pl.when(n == 0)
  def stage_cq():
    # cq does not depend on the head block, so it is normalized and quantized
    # for the first head block and kept in VMEM.
    def stage_body(rows):
      cache_rows = pl.ds(pl.multiple_of(t * block_t + rows.start, chunk), chunk)
      cq = _rms_norm(q_down_ref[rows], q_norm_scale_ref[...], dims.eps)
      cq_q_ref[b, cache_rows] = _static_quantize(cq, dims)

    _row_chunks(block_t, chunk, stage_body)

  def tn(x, y):
    return jax.lax.dot_general(x, y, _TN, preferred_element_type=jnp.float32)

  def accumulate(acc_ref, x):
    # Branch-free: the first token block overwrites the accumulator.
    acc_ref[...] = jnp.where(first, 0.0, acc_ref[...]) + x

  s_q, s_v = scales_ref[0:1, 0:1], scales_ref[1:2, 0:1]
  accumulate(
      acc_wq_ref,
      tn(gq_ref[...], cq_q_ref[b, t_rows]) * (s_q * (dims.softmax_scale * dims.act_scale)),
  )
  ckv = ckv_ref[...]
  accumulate(
      acc_wk_ref,
      tn(
          jnp.concatenate([dk_ref[i, :, :qk] for i in range(num_heads)], axis=1),
          ckv,
      ),
  )
  accumulate(
      acc_wv_ref,
      tn(dv_q_ref[...], _static_quantize(ckv, dims)) * (s_v * dims.act_scale),
  )

  @pl.when(last)
  def write_grads():
    grad_wq_ref[...] = acc_wq_ref[...].astype(grad_wq_ref.dtype)
    grad_wk_ref[...] = acc_wk_ref[...].astype(grad_wk_ref.dtype)
    grad_wv_ref[...] = acc_wv_ref[...].astype(grad_wv_ref.dtype)


def _out_struct(shape, dtype, like):
  return jax.ShapeDtypeStruct(shape, dtype, manual_axis_type=jax.typeof(like).manual_axis_type)


def _nbytes(*xs):
  return sum(x.size * x.dtype.itemsize for x in xs)


def qkv_up_dgrad(
    dq: jax.Array,
    dk: jax.Array,
    dv: jax.Array,
    kv_down: jax.Array,
    kv_norm_scale: jax.Array,
    cos_sin: jax.Array,
    wq_up: jax.Array,
    wk_up: jax.Array,
    wv_up: jax.Array,
    *,
    dims: QkvUpBwdDims,
    block_t: int,
    block_n: int,
    chunk: int = 256,
) -> tuple[jax.Array, ...]:
  """Head-unreduced activation gradients of the fp8 QKV-up projections.

  Args:
    dq: q cotangent, [B, N, T, QK + R] bf16.
    dk: k cotangent, [B, N, T, QK + R] bf16.
    dv: v cotangent, [B, N, T, V] bf16.
    kv_down: Compressed kv and k_pe, [B, T, Ckv + R] bf16.
    kv_norm_scale: kv norm scale, [1, Ckv].
    cos_sin: Concatenated YaRN cos and sin frequencies, [B, T, 2 * R].
    wq_up: fp8 q up weights, [N * (QK + R), Cq].
    wk_up: bf16 k up weights, [N * QK, Ckv].
    wv_up: fp8 v up weights, [N * V, Ckv].
    dims: Static dims and constants.
    block_t: Token block size.
    block_n: Head block size (even).
    chunk: Rows of the token block processed at a time.

  Returns:
    Tuple of
    * grad_cq: [B, Cq, T] bf16, pending the head sum and the q norm backward.
    * grad_kv: [B, Ckv + R, T] bf16, grad_ckv (with the v STE applied) and the
      head sum of dk_pe, pending the head sum and the kv norm / RoPE backward.
    * gq: [B, T, N * (QK + R)] e5m2 q cotangent / softmax_scale (with the
      transposed RoPE applied), divided by its tile scale.
    * dv_q: [B, T, N * V] e5m2 v cotangent, divided by its tile scale.
    * scales: [B, T / block_t, N / block_n, 8, 128] f32 tile scales, of gq in
      row 0 and of dv_q in row 1.
  """
  batch, num_heads, num_tokens, d = dq.shape
  if num_tokens % block_t or num_heads % block_n or block_n % 2:
    raise ValueError(f"Bad {block_t=}, {block_n=} for {dq.shape=}.")
  qk, r, v = dims.qk_head_dim, dims.rope_head_dim, dims.v_head_dim
  cq_dim, ckv_dim, kv_width = wq_up.shape[1], wk_up.shape[1], kv_down.shape[2]
  grid = (batch, num_tokens // block_t, num_heads // block_n)

  def heads_spec(head_dim):
    return pl.BlockSpec((None, block_n, block_t, head_dim), lambda b, t, n: (b, n, t, 0))

  def tokens_spec(width):
    return pl.BlockSpec((None, block_t, width), lambda b, t, n: (b, t, 0))

  def head_block_spec(cols, head_dim):
    return pl.BlockSpec((block_n * head_dim, cols), lambda b, t, n: (n, 0))

  def tokens_t_spec(rows):
    return pl.BlockSpec((None, rows, block_t), lambda b, t, n: (b, 0, t))

  def tile_spec(head_dim):
    return pl.BlockSpec((None, block_t, block_n * head_dim), lambda b, t, n: (b, t, n))

  flops = 2 * batch * num_tokens * num_heads * (d * cq_dim + qk * ckv_dim + v * ckv_dim)
  bytes_accessed = (
      _nbytes(dq, dk, dv, kv_down, cos_sin)
      + batch * grid[1] * _nbytes(wq_up, wk_up, wv_up)
      + batch * num_tokens * (num_heads * (d + v) + 2 * (cq_dim + kv_width))
  )
  return pl.pallas_call(
      functools.partial(_dgrad_kernel, dims=dims, chunk=chunk),
      out_shape=(
          _out_struct((batch, cq_dim, num_tokens), jnp.bfloat16, dq),
          _out_struct((batch, kv_width, num_tokens), jnp.bfloat16, dq),
          _out_struct((batch, num_tokens, num_heads * d), _E5M2, dq),
          _out_struct((batch, num_tokens, num_heads * v), _E5M2, dq),
          _out_struct((*grid, 8, 128), jnp.float32, dq),
      ),
      grid=grid,
      in_specs=[
          heads_spec(d),
          heads_spec(d),
          heads_spec(v),
          tokens_spec(kv_width),
          pl.BlockSpec((1, ckv_dim), lambda b, t, n: (0, 0)),
          tokens_spec(cos_sin.shape[2]),
          head_block_spec(cq_dim, d),
          head_block_spec(ckv_dim, qk),
          head_block_spec(ckv_dim, v),
      ],
      out_specs=(
          tokens_t_spec(cq_dim),
          tokens_t_spec(kv_width),
          tile_spec(d),
          tile_spec(v),
          pl.BlockSpec((None, None, None, 8, 128), lambda b, t, n: (b, t, n, 0, 0)),
      ),
      scratch_shapes=[
          pltpu.VMEM((block_t, block_n * r), jnp.bfloat16),
          pltpu.VMEM((block_t, cq_dim), jnp.float32),
          pltpu.VMEM((block_t, ckv_dim), jnp.float32),
          pltpu.VMEM((block_t, ckv_dim), jnp.float32),
          # The head sum of dk_pe, padded to 128 lanes for its transpose.
          pltpu.VMEM((block_t, 128), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(
              pltpu.PARALLEL,
              pltpu.PARALLEL,
              pltpu.ARBITRARY,
          ),
          vmem_limit_bytes=_VMEM_LIMIT_BYTES,
      ),
      cost_estimate=pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed),
      name="qkv_up_dgrad",
  )(dq, dk, dv, kv_down, kv_norm_scale, cos_sin, wq_up, wk_up, wv_up)


def qkv_up_wgrad(
    q_down: jax.Array,
    ckv: jax.Array,
    q_norm_scale: jax.Array,
    gq: jax.Array,
    dk: jax.Array,
    dv_q: jax.Array,
    scales: jax.Array,
    *,
    dims: QkvUpBwdDims,
    block_t: int,
    block_n: int,
    grad_dtype: Any = jnp.bfloat16,
    chunk: int = 256,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Weight gradients of the fp8 QKV-up projections, summed over the tokens.

  Args:
    q_down: Compressed q, [B, T, Cq] bf16.
    ckv: Normalized compressed kv, [B, T, Ckv] bf16.
    q_norm_scale: q norm scale, [1, Cq].
    gq: `qkv_up_dgrad`'s e5m2 q cotangent.
    dk: k cotangent, [B, N, T, QK + R] bf16.
    dv_q: `qkv_up_dgrad`'s e5m2 v cotangent.
    scales: `qkv_up_dgrad`'s tile scales.
    dims: Static dims and constants.
    block_t: Token block size, the same as `qkv_up_dgrad`'s.
    block_n: Head block size, the same as `qkv_up_dgrad`'s.
    grad_dtype: Dtype of the weight gradients.
    chunk: Rows of q_down normalized at a time.

  Returns:
    Tuple of grad_wq_up [N * (QK + R), Cq], grad_wk_up [N * QK, Ckv] and
    grad_wv_up [N * V, Ckv].
  """
  batch, num_heads, num_tokens, d = dk.shape
  qk, v = dims.qk_head_dim, dims.v_head_dim
  cq_dim, ckv_dim = q_down.shape[2], ckv.shape[2]
  grid = (num_heads // block_n, batch, num_tokens // block_t)
  assert scales.shape[:3] == (batch, grid[2], grid[0]), scales.shape

  def tile_spec(head_dim):
    return pl.BlockSpec((None, block_t, block_n * head_dim), lambda n, b, t: (b, t, n))

  def grad_spec(cols, head_dim):
    return pl.BlockSpec((block_n * head_dim, cols), lambda n, b, t: (n, 0))

  # Only the first head block reads q_down; the later ones keep the last block
  # index, so that its block is not fetched again.
  q_down_spec = pl.BlockSpec(
      (None, block_t, cq_dim),
      lambda n, b, t: (
          jnp.where(n == 0, b, batch - 1),
          jnp.where(n == 0, t, grid[2] - 1),
          0,
      ),
      pipeline_mode=pl.Buffered(1),
  )
  flops = 2 * batch * num_tokens * num_heads * (d * cq_dim + qk * ckv_dim + v * ckv_dim)
  bytes_accessed = (
      _nbytes(q_down, gq, dk, dv_q) + grid[0] * _nbytes(ckv) + num_heads * (d * cq_dim + (qk + v) * ckv_dim) * 2
  )
  chunk = min(chunk, block_t)
  return pl.pallas_call(
      functools.partial(_wgrad_kernel, dims=dims, chunk=chunk),
      out_shape=(
          _out_struct((num_heads * d, cq_dim), grad_dtype, dk),
          _out_struct((num_heads * qk, ckv_dim), grad_dtype, dk),
          _out_struct((num_heads * v, ckv_dim), grad_dtype, dk),
      ),
      grid=grid,
      in_specs=[
          q_down_spec,
          pl.BlockSpec((None, block_t, ckv_dim), lambda n, b, t: (b, t, 0)),
          pl.BlockSpec((1, cq_dim), lambda n, b, t: (0, 0)),
          tile_spec(d),
          pl.BlockSpec((None, block_n, block_t, d), lambda n, b, t: (b, n, t, 0)),
          tile_spec(v),
          pl.BlockSpec((None, None, None, 8, 128), lambda n, b, t: (b, t, n, 0, 0)),
      ],
      out_specs=(
          grad_spec(cq_dim, d),
          grad_spec(ckv_dim, qk),
          grad_spec(ckv_dim, v),
      ),
      scratch_shapes=[
          pltpu.VMEM((batch, num_tokens, cq_dim), dims.act_qtype),
          pltpu.VMEM((block_n * d, cq_dim), jnp.float32),
          pltpu.VMEM((block_n * qk, ckv_dim), jnp.float32),
          pltpu.VMEM((block_n * v, ckv_dim), jnp.float32),
      ],
      compiler_params=pltpu.CompilerParams(
          # The head blocks after the first reuse its normalized cq.
          dimension_semantics=(
              pltpu.ARBITRARY,
              pltpu.ARBITRARY,
              pltpu.ARBITRARY,
          ),
          vmem_limit_bytes=_VMEM_LIMIT_BYTES,
      ),
      cost_estimate=pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed),
      name="qkv_up_wgrad",
  )(q_down, ckv, q_norm_scale, gq, dk, dv_q, scales)

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

"""Fused fp8 MLA q / kv up projection forward Pallas kernel.

One `pallas_call` computes, per device, the full query, key and value of MLA
from the compressed query `q_down` and compressed key/value `kv_down`:

  q = yarn_tail(fp8_dot(quant(rms_norm(q_down)), wq_up)) * scale
  k = concat([rms_norm(ckv) @ wk_up, broadcast(yarn(k_pe))])
  v = fp8_dot(quant(rms_norm(ckv)), wv_up) * dequant_scale

with `ckv, k_pe = split(kv_down)`. The norms, static fp8 quantization, YaRN and
scales all happen in VMEM, so neither the normalized / quantized inputs nor the
un-roped projections round-trip HBM.

q and k are written seq-minor (`[B, N, D, T]`) and v head-dim-minor
(`[B, N, T, V]`), the layouts the DSv3 splash attention kernel consumes, so
transposing them back to `[B, T, N, D]` and then to the splash layouts is a
no-op for XLA.
"""

import functools
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _rms_norm(x: jax.Array, scale: jax.Array, eps: float) -> jax.Array:
  """f32 `ops.rms_norm(x, scale, epsilon=eps)` of bf16 x.

  Its intermediate bf16 roundings are skipped, like XLA's (excess precision)
  fusion of the reference norm with its consumers does.

  Args:
    x: Input.
    scale: Scale of the last dimension of x.
    eps: Epsilon.

  Returns:
    The unrounded normalized x.
  """
  x = x.astype(jnp.float32)
  mean2 = jnp.mean(x * x, axis=-1, keepdims=True)
  return x * jax.lax.rsqrt(mean2 + eps) * scale.astype(jnp.float32)


def _static_quantize(x: jax.Array, qtype: Any, scale: float) -> jax.Array:
  """`quantization.static_quantize`."""
  fmax = float(jnp.finfo(qtype).max)
  return jnp.clip(x.astype(jnp.float32) / scale, -fmax, fmax).astype(qtype)


def _nt_dot(x: jax.Array, y: jax.Array) -> jax.Array:
  """Returns the f32 `x @ y.T`."""
  return jax.lax.dot_general(x, y, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def _transpose(x: jax.Array) -> jax.Array:
  """Returns the f32 `x.T` of a bf16 `[M, R]` x, exactly, on the MXU."""
  r = x.shape[1]
  eye = (jax.lax.broadcasted_iota(jnp.int32, (r, r), 0) == jax.lax.broadcasted_iota(jnp.int32, (r, r), 1)).astype(x.dtype)
  return _nt_dot(eye, x)


def _rotate_half_rows(x: jax.Array) -> jax.Array:
  """`dsv3_mla.rotate_half` along the rows of x, i.e. [x1; x2] -> [-x2; x1]."""
  half = x.shape[0] // 2
  return jnp.concatenate([-x[half:], x[:half]], axis=0)


def _qkv_up_kernel(
    q_down_ref,
    kv_down_ref,
    q_norm_scale_ref,
    kv_norm_scale_ref,
    cos_ref,
    sin_ref,
    wq_ref,
    wk_ref,
    wv_ref,
    q_ref,
    k_ref,
    v_ref,
    cq_q_scratch,
    ckv_scratch,
    ckv_q_scratch,
    cos_scratch,
    sin_scratch,
    k_pe_scratch,
    *,
    eps: float,
    act_qtype: Any,
    act_scale: float,
    dequant_scale: float,
    q_scale: float,
    qk_head_dim: int,
    kv_lora_rank: int,
    head_group: int,
):
  """Computes a [block_n heads, block_t tokens] tile of q, k and v.

  The weights come head-major and flattened, i.e. wq_ref is
  `[block_n * (QK + R), Cq]`, wk_ref `[block_n * QK, Ckv]` and wv_ref
  `[Ckv, block_n * V]`, so that each dot covers `head_group` heads.

  The grid is (batch, head tiles, token tiles). The first head tile normalizes
  and quantizes each token tile, transposes its YaRN frequencies and applies
  RoPE to its k_pe into the full-sequence scratch buffers, which the other
  head tiles reuse instead of refetching and recomputing them.

  Args:
    q_down_ref: `[1, block_t, Cq]` compressed query tile.
    kv_down_ref: `[1, block_t, Ckv + R]` compressed key/value and RoPE key tile.
    q_norm_scale_ref: `[1, Cq]` query RMSNorm scale.
    kv_norm_scale_ref: `[1, Ckv]` key/value RMSNorm scale.
    cos_ref: `[1, block_t, R]` YaRN cos tile.
    sin_ref: `[1, block_t, R]` YaRN sin tile.
    wq_ref: `[block_n * (QK + R), Cq]` fp8 q up weight tile.
    wk_ref: `[block_n * QK, Ckv]` bf16 k up weight tile.
    wv_ref: `[Ckv, block_n * V]` fp8 v up weight tile.
    q_ref: `[1, block_n, QK + R, block_t]` output q tile.
    k_ref: `[1, block_n, QK + R, block_t]` output k tile.
    v_ref: `[1, block_n, block_t, V]` output v tile.
    cq_q_scratch: `[T, Cq]` VMEM scratch for quantized cq.
    ckv_scratch: `[T, Ckv]` VMEM scratch for normalized bf16 ckv.
    ckv_q_scratch: `[T, Ckv]` VMEM scratch for quantized ckv.
    cos_scratch: `[R, T]` VMEM scratch for scaled seq-minor cos.
    sin_scratch: `[R, T]` VMEM scratch for scaled seq-minor sin.
    k_pe_scratch: `[R, T]` VMEM scratch for RoPEd seq-minor k_pe.
    eps: RMSNorm epsilon.
    act_qtype: fp8 dtype of quantized activations.
    act_scale: Static activation quantization scale.
    dequant_scale: Product of activation and weight quantization scales.
    q_scale: Product of dequant_scale and softmax_scale.
    qk_head_dim: Non-RoPE query/key head dim QK.
    kv_lora_rank: Compressed key/value dim Ckv.
    head_group: Number of heads per dot inside a head tile.
  """
  block_n, qk_rope_dim, block_t = q_ref.shape[1:]
  v_head_dim = v_ref.shape[3]
  rope_head_dim = cos_ref.shape[2]
  half = rope_head_dim // 2
  rows = pl.ds(pl.multiple_of(pl.program_id(2) * block_t, block_t), block_t)

  @pl.when(pl.program_id(1) == 0)
  def _():
    cq_q_scratch[rows, :] = _static_quantize(
        _rms_norm(q_down_ref[0], q_norm_scale_ref[...], eps),
        act_qtype,
        act_scale,
    )
    kv_down = kv_down_ref[0]
    ckv = _rms_norm(kv_down[:, :kv_lora_rank], kv_norm_scale_ref[...], eps)
    ckv_q_scratch[rows, :] = _static_quantize(ckv, act_qtype, act_scale)
    ckv_scratch[rows, :] = ckv.astype(jnp.bfloat16)
    # Seq-minor [R, block_t] frequencies and k_pe.
    cos = _transpose(cos_ref[0])
    sin = _transpose(sin_ref[0])
    k_pe = _transpose(kv_down[:, kv_lora_rank:])
    # `dsv3_mla.yarn` of the single k_pe head, rounded once like XLA's fusion.
    k_pe_scratch[:, rows] = (k_pe * cos + _rotate_half_rows(k_pe) * sin).astype(k_pe_scratch.dtype)
    # With the q scale folded in like `dsv3_mla._yarn_tail`.
    cos_scratch[:, rows] = cos * q_scale
    sin_scratch[:, rows] = sin * q_scale

  cq_q = cq_q_scratch[rows, :]
  ckv = ckv_scratch[rows, :]
  ckv_q = ckv_q_scratch[rows, :]
  cos = cos_scratch[:, rows]
  sin = sin_scratch[:, rows]
  k_pe = k_pe_scratch[:, rows]

  for g in range(block_n // head_group):
    h0 = g * head_group
    # [head_group * (QK + R), block_t]. The fp8 dot is rounded to bf16 before
    # it is scaled, like the reference.
    q = _nt_dot(wq_ref[h0 * qk_rope_dim : (h0 + head_group) * qk_rope_dim, :], cq_q)
    q = q.astype(jnp.bfloat16).astype(jnp.float32)
    # [head_group * QK, block_t].
    k = _nt_dot(wk_ref[h0 * qk_head_dim : (h0 + head_group) * qk_head_dim, :], ckv)
    # [block_t, head_group * V].
    v = jax.lax.dot_general(
        ckv_q,
        wv_ref[:, h0 * v_head_dim : (h0 + head_group) * v_head_dim],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    for j in range(head_group):
      h = h0 + j
      qh = q[j * qk_rope_dim : (j + 1) * qk_rope_dim]
      x1 = qh[qk_head_dim : qk_head_dim + half]
      x2 = qh[qk_head_dim + half :]
      q_ref[0, h, :qk_head_dim, :] = (qh[:qk_head_dim] * q_scale).astype(q_ref.dtype)
      q_ref[0, h, qk_head_dim : qk_head_dim + half, :] = (x1 * cos[:half] - x2 * sin[:half]).astype(q_ref.dtype)
      q_ref[0, h, qk_head_dim + half :, :] = (x2 * cos[half:] + x1 * sin[half:]).astype(q_ref.dtype)
      k_ref[0, h, :qk_head_dim, :] = k[j * qk_head_dim : (j + 1) * qk_head_dim].astype(k_ref.dtype)
      k_ref[0, h, qk_head_dim:, :] = k_pe
      v_ref[0, h] = (v[:, j * v_head_dim : (j + 1) * v_head_dim] * dequant_scale).astype(v_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "qk_head_dim",
        "kv_lora_rank",
        "softmax_scale",
        "norm_epsilon",
        "act_qtype",
        "act_scale",
        "weight_scale",
        "block_t",
        "block_n",
        "head_group",
    ),
)
def qkv_up_fwd(
    q_down: jax.Array,
    kv_down: jax.Array,
    wq_up: jax.Array,
    wk_up: jax.Array,
    wv_up: jax.Array,
    wq_norm_scale: jax.Array,
    wkv_norm_scale: jax.Array,
    yarn_freqs: tuple[jax.Array, jax.Array],
    *,
    qk_head_dim: int,
    kv_lora_rank: int,
    softmax_scale: float,
    norm_epsilon: float,
    act_qtype: Any,
    act_scale: float,
    weight_scale: float,
    block_t: int = 1024,
    block_n: int = 8,
    head_group: int = 4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Local (per-device) fused fp8 q / kv up projections.

  Args:
    q_down: `[B, T, Cq]` bf16 compressed query.
    kv_down: `[B, T, Ckv + R]` bf16 compressed key/value and RoPE key.
    wq_up: `[Cq, N, QK + R]` fp8 q up weights, quantized with weight_scale.
    wk_up: `[Ckv, N, QK]` unquantized bf16 k up weights.
    wv_up: `[Ckv, N, V]` fp8 v up weights, quantized with weight_scale.
    wq_norm_scale: `[Cq]` query RMSNorm scale.
    wkv_norm_scale: `[Ckv]` key/value RMSNorm scale.
    yarn_freqs: `[B, T, 1, R]` YaRN cos and sin.
    qk_head_dim: Non-RoPE query/key head dim QK.
    kv_lora_rank: Compressed key/value dim Ckv.
    softmax_scale: YaRN-adjusted softmax scale folded into q.
    norm_epsilon: RMSNorm epsilon.
    act_qtype: fp8 dtype of the quantized activations.
    act_scale: Static activation quantization scale.
    weight_scale: Static weight quantization scale.
    block_t: Token tile.
    block_n: Head tile.
    head_group: Number of heads per dot inside a head tile.

  Returns:
    q `[B, N, QK + R, T]`, k `[B, N, QK + R, T]` and v `[B, N, T, V]` in
    bf16, i.e. seq-minor q and k and head-dim-minor v.
  """
  batch, seq_len, q_lora_rank = q_down.shape
  num_heads = wq_up.shape[1]
  qk_rope_dim = wq_up.shape[2]
  v_head_dim = wv_up.shape[2]
  rope_head_dim = qk_rope_dim - qk_head_dim
  kv_dim = kv_down.shape[2]
  block_t = min(block_t, seq_len)
  block_n = min(block_n, num_heads)
  assert seq_len % block_t == 0, (seq_len, block_t)
  assert num_heads % block_n == 0, (num_heads, block_n)
  head_group = min(head_group, block_n)
  assert block_n % head_group == 0, (block_n, head_group)

  # Head-major q / k weights: their heads are computed as w_h @ x.T, i.e.
  # seq-minor.
  wq_t = jnp.transpose(wq_up, (1, 2, 0)).reshape(-1, q_lora_rank)
  wk_t = jnp.transpose(wk_up, (1, 2, 0)).reshape(-1, kv_lora_rank)
  wv_2d = wv_up.reshape(kv_lora_rank, -1)
  cos, sin = (f.reshape(batch, seq_len, rope_head_dim) for f in yarn_freqs)

  dequant_scale = act_scale * weight_scale
  kernel = functools.partial(
      _qkv_up_kernel,
      eps=norm_epsilon,
      act_qtype=act_qtype,
      act_scale=act_scale,
      dequant_scale=dequant_scale,
      q_scale=dequant_scale * softmax_scale,
      qk_head_dim=qk_head_dim,
      kv_lora_rank=kv_lora_rank,
      head_group=head_group,
  )
  num_t = seq_len // block_t
  grid = (batch, num_heads // block_n, num_t)
  # Only the first head tile reads q_down / kv_down; the others keep the last
  # block index, so the pipeline does not refetch them.
  tok = lambda b, n, t: (b, jnp.where(n == 0, t, num_t - 1), 0)
  weight = lambda b, n, t: (n, 0)
  in_specs = [
      pl.BlockSpec((1, block_t, q_lora_rank), tok),
      pl.BlockSpec((1, block_t, kv_dim), tok),
      pl.BlockSpec((1, q_lora_rank), lambda b, n, t: (0, 0)),
      pl.BlockSpec((1, kv_lora_rank), lambda b, n, t: (0, 0)),
      pl.BlockSpec((1, block_t, rope_head_dim), tok),
      pl.BlockSpec((1, block_t, rope_head_dim), tok),
      pl.BlockSpec((block_n * qk_rope_dim, q_lora_rank), weight),
      pl.BlockSpec((block_n * qk_head_dim, kv_lora_rank), weight),
      pl.BlockSpec((kv_lora_rank, block_n * v_head_dim), lambda b, n, t: (0, n)),
  ]
  seq_minor = lambda b, n, t: (b, n, 0, t)
  out_specs = [
      pl.BlockSpec((1, block_n, qk_rope_dim, block_t), seq_minor),
      pl.BlockSpec((1, block_n, qk_rope_dim, block_t), seq_minor),
      pl.BlockSpec((1, block_n, block_t, v_head_dim), lambda b, n, t: (b, n, t, 0)),
  ]
  out_shape = [
      jax.ShapeDtypeStruct((batch, num_heads, qk_rope_dim, seq_len), jnp.bfloat16),
      jax.ShapeDtypeStruct((batch, num_heads, qk_rope_dim, seq_len), jnp.bfloat16),
      jax.ShapeDtypeStruct((batch, num_heads, seq_len, v_head_dim), jnp.bfloat16),
  ]
  flops = 2 * batch * seq_len * num_heads * (q_lora_rank * qk_rope_dim + kv_lora_rank * (qk_head_dim + v_head_dim))
  bytes_accessed = sum(x.size * x.dtype.itemsize for x in (q_down, kv_down, wq_up, wk_up, wv_up)) + sum(
      s.size * 2 for s in out_shape
  )
  q, k, v = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=out_shape,
      scratch_shapes=[
          pltpu.VMEM((seq_len, q_lora_rank), act_qtype),
          pltpu.VMEM((seq_len, kv_lora_rank), jnp.bfloat16),
          pltpu.VMEM((seq_len, kv_lora_rank), act_qtype),
          pltpu.VMEM((rope_head_dim, seq_len), jnp.float32),
          pltpu.VMEM((rope_head_dim, seq_len), jnp.float32),
          pltpu.VMEM((rope_head_dim, seq_len), jnp.bfloat16),
      ],
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("arbitrary", "arbitrary", "arbitrary"),
          vmem_limit_bytes=63 * 1024 * 1024,
      ),
      cost_estimate=pl.CostEstimate(flops=flops, transcendentals=0, bytes_accessed=bytes_accessed),
      name="mla_qkv_up_fwd",
  )(
      q_down,
      kv_down,
      wq_norm_scale.reshape(1, q_lora_rank),
      wkv_norm_scale.reshape(1, kv_lora_rank),
      cos,
      sin,
      wq_t,
      wk_t,
      wv_2d,
  )
  return q, k, v

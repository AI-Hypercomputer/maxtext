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

"""Fused 2-TensorCore fp8 MLA backward Pallas megakernel (M15).

`fused_mla_bwd` has the signature of `dsv3_mla.dsv3_mla_bwd` and runs the
complete fp8 MLA backward pass of one batch element per (device, pallas_call):

  P1  all-gather of grad_out over the core (head) axis via remote DMA
  P2  out projection dgrad (+ STE, di = rowsum(o * do)) and wgrad
  P3  rematerialization of q (RMSNorm, fp8 up projection, YaRN), k and v
  P4  flash attention backward (dq, dk, dv), tokamax dkv formulation
  P5  q / k / v up projection dgrad (+ YaRN transpose, aux freq grads) and wgrad
  P6  reduce-scatter of the compressed cotangents over the core axis via remote
      DMA, STE, RMSNorm VJP, k_pe RoPE VJP
  P7  q / kv down projection dgrad (+ STE) and wgrad

Every phase is a `pltpu.emit_pipeline` over HBM refs; intermediates live in HBM
scratch declared as extra pallas_call outputs. Only the core-axis collectives
are moved inside the kernel; the FSDP reductions of the weight gradients stay
in XLA (`unreduced` cotangent specs), exactly like the reference.

Layout conventions (per TensorCore, N_l local heads, T tokens, T_c = T/2 local
sequence rows, R = 64 rope dims):
  * q / k / dq / dk use a 256 wide head dim: [nope(128) | rope(64) | pad(64)].
    q's pad duplicates its rope part, k's pad is zero, so q . k is unchanged
    and every value stays a full 128 lane tile.
  * The q up weights are extended per head to [W_nope | W_pe | rotate(W_pe)]
    (256 columns), which makes `rotate_half` a matmul by-product in the remat
    and lets the backward contract [g_nope | g_r * cos | g_r * sin] directly.

Quantization matches `dsv3_mla._qdot_bwd` up to the granularity of the dynamic
e5m2 scales, which are taken per (token tile, head group) instead of per
device shard.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np

from maxtext.experimental.lineage import dsv3_types
from maxtext.experimental.lineage import ops
from maxtext.experimental.lineage import quantization

P = jax.sharding.PartitionSpec
F32 = jnp.float32
BF16 = jnp.bfloat16
LOG2E = math.log2(math.e)
MASK_VALUE = -0.7 * float(np.finfo(np.float32).max)

QK_NOPE = 128  # Non-RoPE q / k head dim.
ROPE = 64  # RoPE head dim.
VDIM = 128  # v head dim.
QEXT = 256  # Padded q / k head dim in HBM scratch.
NUM_LANES = 128
NUM_SUBLANES = 8

NN = (((1,), (0,)), ((), ()))
NT = (((1,), (1,)), ((), ()))
TN = (((0,), (0,)), ((), ()))
HIGHEST = jax.lax.Precision.HIGHEST


@dataclasses.dataclass(frozen=True)
class FusedBwdTiling:
  """Tile sizes of the fused backward kernel.

  Attributes:
    bt: Token tile of the projection phases (P2, P3, P5, P6, P7).
    g_out: Heads per step of the out projection dgrad phase (P2).
    g_out_w: Heads per step of the out projection wgrad phase (P2b). Defaults to
      `g_out` when None.
    g_up: Heads per step of the remat / up projection phases.
    bd: Model dim tile of the out projection wgrad and down projection phases.
    bq: Attention q block (== kv block).
    bkv_compute: Attention kv sub-block.
    fuse_transposed_lhs_in_matmul: Mosaic hint to fuse LHS transpose in TN dots.
    vmem_limit_bytes: Scoped VMEM limit of the kernel.
  """

  bt: int = 512
  g_out: int = 8
  g_out_w: int | None = 16
  g_up: int = 8
  bd: int = 1792
  bq: int = 1024
  bkv_compute: int = 512
  fuse_transposed_lhs_in_matmul: bool = False
  vmem_limit_bytes: int = 60 * 1024 * 1024


def _softmax_scale(
    *,
    qk_head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
) -> float:
  softmax_scale = (qk_head_dim + rope_head_dim) ** -0.5
  if max_position_embeddings > original_max_position_embeddings:
    m = 0.1 * mscale * math.log(rope_factor) + 1.0
    softmax_scale = softmax_scale * m * m
  return softmax_scale


def norm_epsilon(norm_fn: Any) -> float | None:
  """Returns the epsilon of a `functools.partial(ops.rms_norm, ...)`, or None."""
  if (
      isinstance(norm_fn, functools.partial)
      and norm_fn.func is ops.rms_norm
      and not norm_fn.args
      and set(norm_fn.keywords) == {"epsilon"}
  ):
    return float(norm_fn.keywords["epsilon"])
  return None


def tiling_fits(
    tiling: FusedBwdTiling,
    *,
    seq_len: int,
    emb_dim: int,
    num_local_heads: int,
    num_cores: int = 2,
) -> bool:
  """Whether `tiling` divides the per-TensorCore problem shape."""
  t_c = seq_len // num_cores
  g_out_w = tiling.g_out_w or tiling.g_out
  return (
      num_cores == 2
      and seq_len % num_cores == 0
      and t_c % tiling.bt == 0
      and tiling.bt % NUM_SUBLANES == 0
      and num_local_heads % tiling.g_out == 0
      and num_local_heads % g_out_w == 0
      and num_local_heads % tiling.g_up == 0
      and emb_dim % tiling.bd == 0
      and tiling.bd % NUM_LANES == 0
      and seq_len % tiling.bq == 0
      and tiling.bq % tiling.bkv_compute == 0
      and tiling.bq % NUM_LANES == 0
      and tiling.bkv_compute % NUM_SUBLANES == 0
  )


def is_supported(
    *,
    norm_fn: Any,
    qk_head_dim: int,
    rope_head_dim: int,
    kv_lora_rank: int,
    v_head_dim: int,
    quant_rule: quantization.GmmQuantRule | None,
    q_lora_rank: int | None = None,
    seq_len: int | None = None,
    emb_dim: int | None = None,
    num_local_heads: int | None = None,
    num_cores: int | None = None,
    tiling: FusedBwdTiling | None = None,
) -> bool:
  """Whether the fused kernel handles this configuration.

  Shape arguments are optional; when all of them are given, the (default)
  tiling is checked against them too.
  """
  ok = (
      quant_rule is not None
      and norm_epsilon(norm_fn) is not None
      and qk_head_dim == QK_NOPE
      and rope_head_dim == ROPE
      and v_head_dim == VDIM
      and kv_lora_rank % NUM_LANES == 0
      and (q_lora_rank is None or q_lora_rank % NUM_LANES == 0)
      and quant_rule.act_qtype == jnp.float8_e4m3fn
      and quant_rule.weight_qtype == jnp.float8_e4m3fn
      and quant_rule.bwd_qtype == jnp.float8_e5m2
      and quant_rule.grad_dtype == BF16
      # `quantization.MlaQuantRule` variants (fp8 splash Q/K, LOG2E-prescaled
      # q, unquantized q projection backward) are not implemented here.
      and getattr(quant_rule, "splash_qk_qtype", None) is None
      and not getattr(quant_rule, "q_prescaled_log2e", False)
      and getattr(quant_rule, "q_proj_bwd_quantized", True)
  )
  if ok and seq_len is not None and emb_dim is not None and num_local_heads is not None and num_cores is not None:
    ok = tiling_fits(
        tiling or FusedBwdTiling(),
        seq_len=seq_len,
        emb_dim=emb_dim,
        num_local_heads=num_local_heads,
        num_cores=num_cores,
    )
  return bool(ok)


# ------------------------------------------------------------------------------
# Sharding helpers (copies of the private `dsv3_mla` ones, to avoid a cycle).
# ------------------------------------------------------------------------------


def _match_spec(x: jax.Array, spec: P) -> jax.Array:
  """Inside a shard_map, makes the local partial x match out spec `spec`."""
  sharded = set(spec.unreduced)
  for part in spec.partitions:
    if part is not None:
      sharded.update((part,) if isinstance(part, str) else part)
  vma = jax.typeof(x).manual_axis_type.varying
  if missing := tuple(sorted(vma - sharded)):
    x = jax.lax.psum(x, missing)
  if invarying := tuple(sorted(spec.unreduced - vma)):
    mesh = jax.sharding.get_abstract_mesh()
    assert all(mesh.shape[a] == 1 for a in invarying), invarying
    x = jax.lax.pcast(x, invarying, to="varying")
  if spec.unreduced:
    x = jax.lax.pcast(x, tuple(sorted(spec.unreduced)), to="unreduced")
  return x


def _cotangent_sharding(x: jax.Array) -> jax.sharding.NamedSharding:
  sharding = jax.typeof(x).sharding
  assert isinstance(sharding, jax.sharding.NamedSharding)
  return sharding.update(spec=sharding.spec.to_ct_spec())


def _axis_names(axis: Any) -> tuple[str, ...]:
  if axis is None:
    return ()
  return (axis,) if isinstance(axis, str) else tuple(axis)


def _spec_entry(spec: P, i: int) -> Any:
  return spec[i] if i < len(spec) else None


def attention_axes(splash_out: jax.Array) -> tuple[str, ...]:
  """Mesh axes the heads of `splash_out` ([B, T, N, V]) are sharded over."""
  return _axis_names(_spec_entry(jax.typeof(splash_out).sharding.spec, 2))


def _quantize_weight(w: jax.Array, rule: quantization.GmmQuantRule):
  if w.dtype == rule.weight_qtype:
    return w
  return quantization.static_quantize(w, rule.weight_qtype, rule.weight_scale)


# ------------------------------------------------------------------------------
# In-kernel helpers.
# ------------------------------------------------------------------------------


def _bf16_round(x: float) -> float:
  """Rounds a Python float to bf16 (host side, safe under tracing)."""
  return float(np.asarray(x, dtype=jnp.bfloat16).astype(np.float32))


def _absmax_scale(x: jax.Array, fmax: float) -> jax.Array:
  """Returns the [1, 1] e5m2 absmax scale of the f32 2-D `x`."""
  amax = jnp.max(jnp.abs(x), axis=0, keepdims=True)
  amax = jnp.max(amax, axis=1, keepdims=True)
  return jnp.where(amax > 2.0**-100, amax / fmax, 1.0)


def _quant(x: jax.Array, inv_scale: Any, fmax: float, dtype: Any) -> jax.Array:
  return jnp.clip(x * inv_scale, -fmax, fmax).astype(dtype)


def _rmsnorm_fwd(x: jax.Array, scale: jax.Array, eps: float, scale_bf16: bool) -> jax.Array:
  """Mirrors `ops.rms_norm` on bf16 x: returns out_as_f32."""
  xf = x.astype(F32)
  mean2 = jnp.mean(xf * xf, axis=1, keepdims=True)
  yf = xf * jax.lax.rsqrt(mean2 + eps)
  if scale_bf16:
    return yf * scale
  return yf.astype(BF16).astype(F32) * scale


def _rmsnorm_vjp(
    x: jax.Array,
    scale: jax.Array,
    g: jax.Array,
    eps: float,
    x_max: float | None,
    scale_bf16: bool,
):
  """VJP of `ops.rms_norm` for bf16 x and f32 cotangent g.

  Args:
    x: bf16 [bt, C] primal input.
    scale: f32 [1, C] norm scale.
    g: f32 [bt, C] cotangent of the norm output.
    eps: RMSNorm epsilon.
    x_max: If given, g is first zeroed where |norm(x)| > x_max (STE).
    scale_bf16: Whether the primal norm scale had dtype bfloat16.

  Returns:
    (f32 grad of x, [8, C] f32 partial grad of scale summed over rows).
  """
  xf = x.astype(F32)
  mean2 = jnp.mean(xf * xf, axis=1, keepdims=True)
  r = jax.lax.rsqrt(mean2 + eps)
  yf = xf * r
  y = yf.astype(BF16).astype(F32)
  if x_max is not None:
    out = (yf * scale).astype(BF16).astype(F32) if scale_bf16 else y * scale
    g = jnp.where(jnp.abs(out) <= x_max, g, 0.0)
  bt, c = g.shape
  g_scale = jnp.sum((g * y).reshape(bt // NUM_SUBLANES, NUM_SUBLANES, c), axis=0)
  gy = g * scale
  gx = r * (gy - yf * jnp.mean(gy * yf, axis=1, keepdims=True))
  return gx, g_scale


def _acc_store(ref, value, first, last=None, out_ref=None, out_dtype=None):
  """ref = value at `first`, else ref += value; copies to out_ref at `last`."""

  @pl.when(first)
  def _():
    ref[...] = value

  @pl.when(jnp.logical_not(first))
  def _():
    ref[...] += value

  if out_ref is not None:
    assert last is not None

    @pl.when(last)
    def _():
      out_ref[...] = ref[...].astype(out_dtype)


class _Refs:
  """Name -> ref namespace for the kernel operands."""

  def __init__(self, names, refs):
    assert len(names) == len(refs), (len(names), len(refs))
    self.__dict__.update(zip(names, refs))

  def __getattr__(self, name: str) -> Any:
    try:
      return self.__dict__[name]
    except KeyError:
      raise AttributeError(name) from None


@dataclasses.dataclass(frozen=True)
class _Static:
  """Static kernel parameters."""

  t: int  # Full sequence length.
  t_c: int  # Local sequence rows.
  d: int
  cq: int
  ckv: int
  n_l: int  # Local heads.
  eps: float
  softmax_scale: float
  base2: bool
  has_seg: bool
  core_axis: str
  mesh_axes: tuple[str, ...]
  tiling: FusedBwdTiling
  # Quantization constants.
  act_inv: float
  act_fmax: float
  act_scale: float
  w_scale: float
  bwd_fmax: float
  x_max: float
  e4m3: Any
  e5m2: Any
  q_scale_bf16: bool = True
  kv_scale_bf16: bool = True
  # Profiling only: subset of phase names to run (see `ALL_PHASES`).
  phases: tuple[str, ...] = ()

  @property
  def kvd_w(self) -> int:
    return self.ckv + ROPE

  @property
  def gkvd_w(self) -> int:
    return self.ckv + NUM_LANES

  @property
  def n_t(self) -> int:
    return self.t // self.tiling.bt

  @property
  def n_tc(self) -> int:
    return self.t_c // self.tiling.bt

  @property
  def g_out_w(self) -> int:
    return self.tiling.g_out_w or self.tiling.g_out

  @property
  def n_g_out(self) -> int:
    return self.n_l // self.tiling.g_out

  @property
  def n_g_out_w(self) -> int:
    return self.n_l // self.g_out_w

  @property
  def n_g_up(self) -> int:
    return self.n_l // self.tiling.g_up

  @property
  def n_d(self) -> int:
    return self.d // self.tiling.bd

  @property
  def n_q(self) -> int:
    return self.t // self.tiling.bq


# ------------------------------------------------------------------------------
# Phases.
# ------------------------------------------------------------------------------


def _core_barrier(r: _Refs, peer_id):
  """Two-phase barrier across the core pair."""
  barrier = pltpu.get_barrier_semaphore()
  pl.semaphore_signal(barrier, device_id=peer_id)
  pl.semaphore_wait(barrier)
  pl.semaphore_signal(r.bsem.at[0], device_id=peer_id)
  pl.semaphore_wait(r.bsem.at[0])


def _p1_allgather(r: _Refs, s: _Static, peer_id):
  """Copies local `go` [t_c, d] to the peer's `go_peer` [t_c, d]."""
  del s
  _core_barrier(r, peer_id)
  cp = pltpu.make_async_remote_copy(
      r.go,
      r.go_peer,
      r.sems.at[0],
      r.sems.at[1],
      device_id=peer_id,
  )
  cp.start()
  cp.wait()


def _p2_out_dgrad(r: _Refs, s: _Static, c, peer):
  """do = STE((gq @ w_out^T) * sg * ws), di = rowsum(o * do), oq, gq."""
  bt, g = s.tiling.bt, s.tiling.g_out
  gw = g * VDIM
  cw = math.gcd(s.d, 1024)
  n_ch = s.d // cw

  def run_half(go_src, base_ti):
    def body(go_ref, w_ref, o_ref, do_ref, di_ref, oq_ref, gq_ref):
      ti, j = pl.program_id(0), pl.program_id(1)
      gti = base_ti + ti

      @pl.when(j == 0)
      def _():
        def chunk_amax(ch):
          sl = slice(ch * cw, (ch + 1) * cw)
          return jnp.max(jnp.abs(go_ref[:, sl].astype(F32)), axis=0, keepdims=True)

        amax = chunk_amax(0)
        for ch in range(1, n_ch):
          amax = jnp.maximum(amax, chunk_amax(ch))
        amax = jnp.max(amax, axis=1, keepdims=True)
        scale = jnp.where(amax > 2.0**-100, amax / s.bwd_fmax, 1.0)
        r.gsc2[gti] = scale
        inv = 1.0 / scale
        for ch in range(n_ch):
          sl = slice(ch * cw, (ch + 1) * cw)
          gq_ref[:, sl] = _quant(go_ref[:, sl].astype(F32), inv, s.bwd_fmax, s.e5m2)

      scale = r.gsc2[gti]
      do = jax.lax.dot_general(gq_ref[...], w_ref[...], NT, preferred_element_type=F32) * (scale * s.w_scale)
      o = o_ref[...]
      of = o.astype(F32)
      do = jnp.where(jnp.abs(of) <= s.x_max, do, 0.0).astype(BF16)
      prod = of * do.astype(F32)
      ones = jnp.ones((NUM_SUBLANES, VDIM), F32)
      for hh in range(g):
        sl = slice(hh * VDIM, (hh + 1) * VDIM)
        do_ref[hh] = do[:, sl]
        di_ref[hh] = jax.lax.dot_general(
            ones,
            prod[:, sl],
            NT,
            precision=HIGHEST,
            preferred_element_type=F32,
        )
      oq_ref[...] = _quant(of, s.act_inv, s.act_fmax, s.e4m3)

    pltpu.emit_pipeline(
        body,
        grid=(s.n_tc, s.n_g_out),
        in_specs=[
            pl.BlockSpec((bt, s.d), lambda ti, j: (ti, 0)),
            pl.BlockSpec((gw, s.d), lambda ti, j: (j, 0)),
            pl.BlockSpec((bt, gw), lambda ti, j: (base_ti + ti, j)),
        ],
        out_specs=[
            pl.BlockSpec((g, bt, VDIM), lambda ti, j: (j, base_ti + ti, 0)),
            pl.BlockSpec((g, NUM_SUBLANES, bt), lambda ti, j: (j, 0, base_ti + ti)),
            pl.BlockSpec((bt, gw), lambda ti, j: (base_ti + ti, j)),
            pl.BlockSpec((bt, s.d), lambda ti, j: (base_ti + ti, 0)),
        ],
    )(go_src, r.w_out, r.o, r.do_hm, r.di8, r.oq_hm, r.gq_hm)

  run_half(r.go, c * s.n_tc)
  run_half(r.go_peer, peer * s.n_tc)


def _p2b_out_wgrad(r: _Refs, s: _Static):
  """gw_out[h, v, d] = sum_t oq[t, h, v] * gq[t, d] * sg_t * act_scale."""
  bt, g, bd = s.tiling.bt, s.g_out_w, s.tiling.bd
  gw = g * VDIM

  def run(acc):
    def body(oq_ref, gq_ref, out_ref):
      ti = pl.program_id(2)
      scale = r.gsc2[ti]
      contrib = jax.lax.dot_general(oq_ref[...], gq_ref[...], TN, preferred_element_type=F32) * (scale * s.act_scale)
      _acc_store(acc, contrib, ti == 0)

      @pl.when(ti == s.n_t - 1)
      def _():
        acc_bf = acc[...].astype(BF16)
        for hh in range(g):
          out_ref[hh] = acc_bf[hh * VDIM : (hh + 1) * VDIM]

    pltpu.emit_pipeline(
        body,
        grid=(s.n_g_out_w, s.n_d, s.n_t),
        in_specs=[
            pl.BlockSpec((bt, gw), lambda j, dj, ti: (ti, j)),
            pl.BlockSpec((bt, bd), lambda j, dj, ti: (ti, dj)),
        ],
        out_specs=[pl.BlockSpec((g, VDIM, bd), lambda j, dj, ti: (j, 0, dj))],
    )(r.oq_hm, r.gq_hm, r.gw_out)

  pl.run_scoped(run, pltpu.VMEM((gw, bd), F32))


def _p3_remat(r: _Refs, s: _Static):
  """Rematerializes cq_q, ckv, q (YaRN'd, scaled), q_pe residual, k and v."""
  bt, g = s.tiling.bt, s.tiling.g_up
  dq_scale = s.act_scale * s.w_scale  # Dequant scale of the fp8 up dots.
  scale_bf = _bf16_round(dq_scale * s.softmax_scale)

  def body(
      qd_ref,
      kvd_ref,
      kpe2_ref,
      cs_s_ref,
      cos2_ref,
      sin2_ref,
      rot_ref,
      wq_ref,
      wk_ref,
      wv_ref,
      qs_ref,
      kvs_ref,
      cqq_ref,
      ckvq_ref,
      ckvb_ref,
      q_ref,
      qpe_ref,
      k_ref,
      v_ref,
  ):  # pyformat: disable
    j = pl.program_id(1)

    @pl.when(j == 0)
    def _():
      cq = _rmsnorm_fwd(qd_ref[...], qs_ref[...], s.eps, s.q_scale_bf16)
      cqq_ref[...] = _quant(cq, s.act_inv, s.act_fmax, s.e4m3)
      ckv = _rmsnorm_fwd(kvd_ref[:, : s.ckv], kvs_ref[...], s.eps, s.kv_scale_bf16)
      ckvb_ref[...] = ckv.astype(BF16)
      ckvq_ref[...] = _quant(ckv, s.act_inv, s.act_fmax, s.e4m3)

    # Roped k_pe, duplicated in both lane halves, then zeroed in the second.
    kpe2 = kpe2_ref[...].astype(F32)
    kpe_rot = jax.lax.dot_general(kpe2, rot_ref[...], NN, precision=HIGHEST, preferred_element_type=F32)
    kpe_r = kpe2 * cos2_ref[...] + kpe_rot * sin2_ref[...]
    lane = jax.lax.broadcasted_iota(jnp.int32, kpe_r.shape, 1)
    kpe_half = jnp.where(lane < ROPE, kpe_r, 0.0).astype(BF16)

    res = jax.lax.dot_general(cqq_ref[...], wq_ref[...], NN, preferred_element_type=F32)
    kn = jax.lax.dot_general(ckvb_ref[...], wk_ref[...], NN, preferred_element_type=F32)
    vv = jax.lax.dot_general(ckvq_ref[...], wv_ref[...], NN, preferred_element_type=F32) * dq_scale
    cs = cs_s_ref[...].astype(F32)
    for hh in range(g):
      base = hh * QEXT
      nope = (res[:, base : base + QK_NOPE].astype(BF16).astype(F32) * scale_bf).astype(BF16)
      v1 = res[:, base + QK_NOPE : base + QEXT].astype(BF16)
      u = v1.astype(F32) * cs
      u_rot = pltpu.roll(u, ROPE, 1).astype(BF16).astype(F32)
      yr = (u + u_rot).astype(BF16)
      q_ref[hh] = jnp.concatenate([nope, yr], axis=1)
      qpe_ref[hh] = (v1.astype(F32) * dq_scale).astype(BF16)
      sl = slice(hh * VDIM, (hh + 1) * VDIM)
      k_ref[hh] = jnp.concatenate([kn[:, sl].astype(BF16), kpe_half], axis=1)
      v_ref[hh] = vv[:, sl].astype(BF16)

  tok = lambda ti, j: (ti, 0)
  pltpu.emit_pipeline(
      body,
      grid=(s.n_t, s.n_g_up),
      in_specs=[
          pl.BlockSpec((bt, s.cq), tok),
          pl.BlockSpec((bt, s.kvd_w), tok),
          pl.BlockSpec((bt, NUM_LANES), tok),
          pl.BlockSpec((bt, NUM_LANES), tok),
          pl.BlockSpec((bt, NUM_LANES), tok),
          pl.BlockSpec((bt, NUM_LANES), tok),
          pl.BlockSpec((NUM_LANES, NUM_LANES), lambda ti, j: (0, 0)),
          pl.BlockSpec((s.cq, g * QEXT), lambda ti, j: (0, j)),
          pl.BlockSpec((s.ckv, g * VDIM), lambda ti, j: (0, j)),
          pl.BlockSpec((s.ckv, g * VDIM), lambda ti, j: (0, j)),
          pl.BlockSpec((1, s.cq), lambda ti, j: (0, 0)),
          pl.BlockSpec((1, s.ckv), lambda ti, j: (0, 0)),
      ],
      out_specs=[
          pl.BlockSpec((bt, s.cq), tok),
          pl.BlockSpec((bt, s.ckv), tok),
          pl.BlockSpec((bt, s.ckv), tok),
          pl.BlockSpec((g, bt, QEXT), lambda ti, j: (j, ti, 0)),
          pl.BlockSpec((g, bt, NUM_LANES), lambda ti, j: (j, ti, 0)),
          pl.BlockSpec((g, bt, QEXT), lambda ti, j: (j, ti, 0)),
          pl.BlockSpec((g, bt, VDIM), lambda ti, j: (j, ti, 0)),
      ],
  )(
      r.qd,
      r.kvd,
      r.kpe2,
      r.cs_s,
      r.cos2,
      r.sin2,
      r.rot,
      r.wq_ext,
      r.wk,
      r.wv,
      r.q_scale,
      r.kv_scale,
      r.cq_q,
      r.ckv_q,
      r.ckv_bf,
      r.q_hm,
      r.qpe_hm,
      r.k_hm,
      r.v_hm,
  )  # pyformat: disable


def _p4_attention_bwd(r: _Refs, s: _Static):
  """Causal flash attention backward, kv-major like tokamax's dkv kernel."""
  bq = s.tiling.bq
  bkv = bq
  bkvc = s.tiling.bkv_compute
  n_sub = bkv // bkvc
  n_q = s.n_q
  exp = jnp.exp2 if s.base2 else jnp.exp

  def run(dq_acc, dk_acc, dv_acc):
    def body(*refs):
      if s.has_seg:
        (q_ref, do_ref, lse_ref, di_ref, k_ref, v_ref, qseg_ref, kvseg_ref, dq_ref, dk_ref, dv_ref) = (
            refs  # pyformat: disable
        )
      else:
        q_ref, do_ref, lse_ref, di_ref, k_ref, v_ref, dq_ref, dk_ref, dv_ref = refs
        qseg_ref = kvseg_ref = None
      j, i = pl.program_id(1), pl.program_id(2)

      @pl.when(jnp.logical_and(j == 0, i == 0))
      def _():
        dq_acc[...] = jnp.zeros_like(dq_acc)

      @pl.when(i == j)
      def _():
        dk_acc[...] = jnp.zeros_like(dk_acc)
        dv_acc[...] = jnp.zeros_like(dv_acc)

      def compute(diag):
        q = q_ref[...]
        sq = q * LOG2E if s.base2 else q
        do = do_ref[...]
        lse = lse_ref[:1, :]
        di = di_ref[:1, :]
        qs = pl.ds(pl.multiple_of(i * bq, bq), bq)

        def sub(si, carry):
          ks = pl.ds(pl.multiple_of(si * bkvc, bkvc), bkvc)
          k = k_ref[ks, :]
          v = v_ref[ks, :]
          qk = jax.lax.dot_general(k, sq, NT, preferred_element_type=F32)
          masks = []
          if diag:
            kv_pos = si * bkvc + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 0)
            q_pos = jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1)
            masks.append(q_pos >= kv_pos)
          if qseg_ref is not None and kvseg_ref is not None:
            kv_ids = jnp.tile(kvseg_ref[ks, :], (1, bq // NUM_LANES))
            masks.append(qseg_ref[:1, :] == kv_ids)
          if masks:
            qk = jnp.where(functools.reduce(jnp.logical_and, masks), qk, MASK_VALUE)  # pyformat: disable
          p = exp(qk - lse)
          dv_acc[ks, :] += jax.lax.dot_general(p.astype(BF16), do, NN, preferred_element_type=F32)
          dp = jax.lax.dot_general(v, do, NT, preferred_element_type=F32)
          ds = (dp - di) * p
          dk_acc[ks, :] += jax.lax.dot_general(ds.astype(BF16), q, NN, preferred_element_type=F32)
          dq_acc[qs, :] += jax.lax.dot_general(ds.T.astype(BF16), k, NN, preferred_element_type=F32)
          return carry

        jax.lax.fori_loop(0, n_sub, sub, 0, unroll=True)

      @pl.when(i == j)
      def _():
        compute(True)

      @pl.when(i > j)
      def _():
        compute(False)

      @pl.when(i == j)
      def _():
        # All kv blocks <= j contributed to q block j by now.
        dq_ref[...] = dq_acc[pl.ds(pl.multiple_of(j * bq, bq), bq), :].astype(BF16)

      @pl.when(i == n_q - 1)
      def _():
        dk_ref[...] = dk_acc[...].astype(BF16)
        dv_ref[...] = dv_acc[...].astype(BF16)

    qidx = lambda h, j, i: (h, jnp.maximum(i, j), 0)
    lidx = lambda h, j, i: (h, 0, jnp.maximum(i, j))
    kidx = lambda h, j, i: (h, j, 0)
    in_specs = [
        pl.BlockSpec((None, bq, QEXT), qidx),
        pl.BlockSpec((None, bq, VDIM), qidx),
        pl.BlockSpec((None, NUM_SUBLANES, bq), lidx),
        pl.BlockSpec((None, NUM_SUBLANES, bq), lidx),
        pl.BlockSpec((None, bkv, QEXT), kidx),
        pl.BlockSpec((None, bkv, VDIM), kidx),
    ]
    ins = [r.q_hm, r.do_hm, r.lse8, r.di8, r.k_hm, r.v_hm]
    if s.has_seg:
      in_specs += [
          pl.BlockSpec((NUM_SUBLANES, bq), lambda h, j, i: (0, jnp.maximum(i, j))),
          pl.BlockSpec((bkv, NUM_LANES), lambda h, j, i: (j, 0)),
      ]  # pyformat: disable
      ins += [r.qseg8, r.kvseg]
    pltpu.emit_pipeline(
        body,
        grid=(s.n_l, n_q, n_q),
        in_specs=in_specs,
        out_specs=[
            pl.BlockSpec((None, bq, QEXT), kidx),
            pl.BlockSpec((None, bkv, QEXT), kidx),
            pl.BlockSpec((None, bkv, VDIM), kidx),
        ],
    )(*ins, r.dq_hm, r.dk_hm, r.dv_hm)

  pl.run_scoped(
      run,
      pltpu.VMEM((s.t, QEXT), F32),
      pltpu.VMEM((bkv, QEXT), F32),
      pltpu.VMEM((bkv, VDIM), F32),
  )


def _p5a_up_dgrad(r: _Refs, s: _Static):
  """gcq, gckv (+ STE of the v part), aux and the quantized up cotangents."""
  bt, g = s.tiling.bt, s.tiling.g_up
  ss = s.softmax_scale
  n_g = s.n_g_up

  def body(
      dq_ref,
      dk_ref,
      dv_ref,
      qpe_ref,
      cs_ref,
      ckvb_ref,
      wq_ref,
      wk_ref,
      wv_ref,
      gcq_ref,
      gckv_ref,
      aux_ref,
      gextq_ref,
      gvq_ref,
  ):  # pyformat: disable
    ti, j = pl.program_id(0), pl.program_id(1)
    cs = cs_ref[...]
    g_parts, gk_parts, gv_parts = [], [], []
    aux_q = jnp.zeros((bt, NUM_LANES), F32)
    aux_k = jnp.zeros((bt, NUM_LANES), F32)
    for hh in range(g):
      gq = dq_ref[hh].astype(F32)
      g_nope = gq[:, :QK_NOPE] * ss
      v = gq[:, QK_NOPE:]
      v = (v + pltpu.roll(v, ROPE, 1)) * ss  # [g_r | g_r] * softmax_scale
      g_parts.append(jnp.concatenate([g_nope, v * cs], axis=1))
      aux_q = aux_q + qpe_ref[hh].astype(F32) * v
      dk = dk_ref[hh]
      gk_parts.append(dk[:, :QK_NOPE])
      aux_k = aux_k + dk[:, QK_NOPE:].astype(F32)
      gv_parts.append(dv_ref[hh].astype(F32))
    g_cat = jnp.concatenate(g_parts, axis=1)
    sq = _absmax_scale(g_cat, s.bwd_fmax)
    r.s5q[ti * n_g + j] = sq
    gq_cat = _quant(g_cat, 1.0 / sq, s.bwd_fmax, s.e5m2)
    gextq_ref[...] = gq_cat
    dcq = jax.lax.dot_general(gq_cat, wq_ref[...], NT, preferred_element_type=F32) * (sq * s.w_scale)
    gk_cat = jnp.concatenate(gk_parts, axis=1)
    dck = jax.lax.dot_general(gk_cat, wk_ref[...], NT, preferred_element_type=F32)
    gv_cat = jnp.concatenate(gv_parts, axis=1)
    sv = _absmax_scale(gv_cat, s.bwd_fmax)
    r.s5v[ti * n_g + j] = sv
    gv_q = _quant(gv_cat, 1.0 / sv, s.bwd_fmax, s.e5m2)
    gvq_ref[...] = gv_q
    dcv = jax.lax.dot_general(gv_q, wv_ref[...], NT, preferred_element_type=F32) * (sv * s.w_scale)
    mask = jnp.abs(ckvb_ref[...].astype(F32)) <= s.x_max
    dckv = dck + jnp.where(mask, dcv, 0.0)
    aux = jnp.concatenate([aux_q, aux_k], axis=1)
    _acc_store(gcq_ref, dcq, j == 0)
    _acc_store(gckv_ref, dckv, j == 0)
    _acc_store(aux_ref, aux, j == 0)

  tok = lambda ti, j: (ti, 0)
  hd = lambda ti, j: (j, ti, 0)
  pltpu.emit_pipeline(
      body,
      grid=(s.n_t, n_g),
      in_specs=[
          pl.BlockSpec((g, bt, QEXT), hd),
          pl.BlockSpec((g, bt, QEXT), hd),
          pl.BlockSpec((g, bt, VDIM), hd),
          pl.BlockSpec((g, bt, NUM_LANES), hd),
          pl.BlockSpec((bt, NUM_LANES), tok),
          pl.BlockSpec((bt, s.ckv), tok),
          pl.BlockSpec((s.cq, g * QEXT), lambda ti, j: (0, j)),
          pl.BlockSpec((s.ckv, g * VDIM), lambda ti, j: (0, j)),
          pl.BlockSpec((s.ckv, g * VDIM), lambda ti, j: (0, j)),
      ],
      out_specs=[
          pl.BlockSpec((bt, s.cq), tok),
          pl.BlockSpec((bt, s.ckv), tok),
          pl.BlockSpec((bt, 2 * NUM_LANES), tok),
          pl.BlockSpec((None, bt, g * QEXT), hd),
          pl.BlockSpec((None, bt, g * VDIM), hd),
      ],
  )(
      r.dq_hm,
      r.dk_hm,
      r.dv_hm,
      r.qpe_hm,
      r.cs,
      r.ckv_bf,
      r.wq_ext,
      r.wk,
      r.wv,
      r.gcq,
      r.gckv,
      r.aux,
      r.gext_q,
      r.gv_q,
  )  # pyformat: disable


def _p5b_up_wgrad(r: _Refs, s: _Static):
  """Weight gradients of the q / k / v up projections."""
  bt, g = s.tiling.bt, s.tiling.g_up
  n_g = s.n_g_up

  def run(acc_q, acc_k, acc_v):
    def body(
        gextq_ref,
        gvq_ref,
        dk_ref,
        cqq_ref,
        ckvq_ref,
        ckvb_ref,
        gwq_ref,
        gwk_ref,
        gwv_ref,
    ):  # pyformat: disable
      j, ti = pl.program_id(0), pl.program_id(1)
      sq = r.s5q[ti * n_g + j] * s.act_scale
      sv = r.s5v[ti * n_g + j] * s.act_scale
      cq_part = jax.lax.dot_general(cqq_ref[...], gextq_ref[...], TN, preferred_element_type=F32) * sq
      dk_cat = jnp.concatenate([dk_ref[hh, :, :QK_NOPE] for hh in range(g)], axis=1)
      k_part = jax.lax.dot_general(ckvb_ref[...], dk_cat, TN, preferred_element_type=F32)
      v_part = jax.lax.dot_general(ckvq_ref[...], gvq_ref[...], TN, preferred_element_type=F32) * sv
      last = ti == s.n_t - 1
      _acc_store(acc_q, cq_part, ti == 0, last, gwq_ref, BF16)
      _acc_store(acc_k, k_part, ti == 0, last, gwk_ref, BF16)
      _acc_store(acc_v, v_part, ti == 0, last, gwv_ref, BF16)

    tok = lambda j, ti: (ti, 0)
    hd = lambda j, ti: (j, ti, 0)
    pltpu.emit_pipeline(
        body,
        grid=(n_g, s.n_t),
        in_specs=[
            pl.BlockSpec((None, bt, g * QEXT), hd),
            pl.BlockSpec((None, bt, g * VDIM), hd),
            pl.BlockSpec((g, bt, QEXT), hd),
            pl.BlockSpec((bt, s.cq), tok),
            pl.BlockSpec((bt, s.ckv), tok),
            pl.BlockSpec((bt, s.ckv), tok),
        ],
        out_specs=[
            pl.BlockSpec((s.cq, g * QEXT), lambda j, ti: (0, j)),
            pl.BlockSpec((s.ckv, g * VDIM), lambda j, ti: (0, j)),
            pl.BlockSpec((s.ckv, g * VDIM), lambda j, ti: (0, j)),
        ],
    )(
        r.gext_q,
        r.gv_q,
        r.dk_hm,
        r.cq_q,
        r.ckv_q,
        r.ckv_bf,
        r.gwq_ext,
        r.gwk,
        r.gwv,
    )  # pyformat: disable

  pl.run_scoped(
      run,
      pltpu.VMEM((s.cq, g * QEXT), F32),
      pltpu.VMEM((s.ckv, g * VDIM), F32),
      pltpu.VMEM((s.ckv, g * VDIM), F32),
  )


def _p6_reduce_scatter_norm_bwd(r: _Refs, s: _Static, c, peer, peer_id):
  """Reduce-scatters gcq / gckv / aux, then STE + RMSNorm / k_pe RoPE VJPs."""
  _core_barrier(r, peer_id)
  bt = s.tiling.bt
  peer_rows = pl.ds(pl.multiple_of(peer * s.t_c, s.t_c), s.t_c)
  own_rows = pl.ds(pl.multiple_of(c * s.t_c, s.t_c), s.t_c)
  copies = []
  for n, (src, dst) in enumerate(((r.gcq, r.rs_cq), (r.gckv, r.rs_ckv), (r.aux, r.rs_aux))):
    copies.append(
        pltpu.make_async_remote_copy(
            src.at[peer_rows],
            dst,
            r.sems.at[2 + 2 * n],
            r.sems.at[3 + 2 * n],
            device_id=peer_id,
        )
    )
  for cp in copies:
    cp.start()
  for cp in copies:
    cp.wait()

  def body(
      gcq_ref,
      rs_cq_ref,
      qd_ref,
      gckv_ref,
      rs_ckv_ref,
      kvd_ref,
      aux_ref,
      rs_aux_ref,
      cos2_ref,
      sin2_ref,
      rot_ref,
      qs_ref,
      kvs_ref,
      gqd_ref,
      gkvd_ref,
      auxr_ref,
      gsq_ref,
      gskv_ref,
  ):  # pyformat: disable
    ti = pl.program_id(0)
    g = gcq_ref[...] + rs_cq_ref[...]
    gx, gs = _rmsnorm_vjp(qd_ref[...], qs_ref[...], g, s.eps, s.x_max, s.q_scale_bf16)
    gqd_ref[...] = gx.astype(BF16)
    g2 = gckv_ref[...] + rs_ckv_ref[...]
    gx2, gs2 = _rmsnorm_vjp(kvd_ref[:, : s.ckv], kvs_ref[...], g2, s.eps, None, s.kv_scale_bf16)
    a = aux_ref[...] + rs_aux_ref[...]
    auxr_ref[...] = a
    gk2 = a[:, NUM_LANES:]
    gk_rot = jax.lax.dot_general(
        gk2 * sin2_ref[...],
        rot_ref[...],
        NN,
        precision=HIGHEST,
        preferred_element_type=F32,
    )
    gkpe = gk2 * cos2_ref[...] - gk_rot
    lane = jax.lax.broadcasted_iota(jnp.int32, gkpe.shape, 1)
    gkpe = jnp.where(lane < ROPE, gkpe, 0.0)
    gkvd_ref[...] = jnp.concatenate([gx2, gkpe], axis=1).astype(BF16)
    _acc_store(gsq_ref, gs, ti == 0)
    _acc_store(gskv_ref, gs2, ti == 0)

  tok = lambda ti: (ti, 0)
  pltpu.emit_pipeline(
      body,
      grid=(s.n_tc,),
      in_specs=[
          pl.BlockSpec((bt, s.cq), tok),
          pl.BlockSpec((bt, s.cq), tok),
          pl.BlockSpec((bt, s.cq), tok),
          pl.BlockSpec((bt, s.ckv), tok),
          pl.BlockSpec((bt, s.ckv), tok),
          pl.BlockSpec((bt, s.kvd_w), tok),
          pl.BlockSpec((bt, 2 * NUM_LANES), tok),
          pl.BlockSpec((bt, 2 * NUM_LANES), tok),
          pl.BlockSpec((bt, NUM_LANES), tok),
          pl.BlockSpec((bt, NUM_LANES), tok),
          pl.BlockSpec((NUM_LANES, NUM_LANES), lambda ti: (0, 0)),
          pl.BlockSpec((1, s.cq), lambda ti: (0, 0)),
          pl.BlockSpec((1, s.ckv), lambda ti: (0, 0)),
      ],
      out_specs=[
          pl.BlockSpec((bt, s.cq), tok),
          pl.BlockSpec((bt, s.gkvd_w), tok),
          pl.BlockSpec((bt, 2 * NUM_LANES), tok),
          pl.BlockSpec((NUM_SUBLANES, s.cq), lambda ti: (0, 0)),
          pl.BlockSpec((NUM_SUBLANES, s.ckv), lambda ti: (0, 0)),
      ],
  )(
      r.gcq.at[own_rows],
      r.rs_cq,
      r.qd.at[own_rows],
      r.gckv.at[own_rows],
      r.rs_ckv,
      r.kvd.at[own_rows],
      r.aux.at[own_rows],
      r.rs_aux,
      r.cos2.at[own_rows],
      r.sin2.at[own_rows],
      r.rot,
      r.q_scale,
      r.kv_scale,
      r.gqd_hm,
      r.gkvd_hm,
      r.auxr,
      r.gscale_q,
      r.gscale_kv,
  )  # pyformat: disable


def _p7a_down_dgrad(r: _Refs, s: _Static):
  """grad_norm_x = STE((gq_q @ wq_down^T) * sq * ws + (gkv_q @ wkv_down^T) * ...)."""
  bt, bd = s.tiling.bt, s.tiling.bd

  def body(gqd_ref, gkvd_ref, x_ref, wq_ref, wkv_ref, gx_ref, gqq_ref, gkvq_ref):
    ti, dj = pl.program_id(0), pl.program_id(1)

    @pl.when(dj == 0)
    def _():
      g = gqd_ref[...].astype(F32)
      sq = _absmax_scale(g, s.bwd_fmax)
      r.s7q[ti] = sq
      gqq_ref[...] = _quant(g, 1.0 / sq, s.bwd_fmax, s.e5m2)
      g2 = gkvd_ref[...].astype(F32)
      skv = _absmax_scale(g2, s.bwd_fmax)
      r.s7kv[ti] = skv
      gkvq_ref[...] = _quant(g2, 1.0 / skv, s.bwd_fmax, s.e5m2)

    sq = r.s7q[ti] * s.w_scale
    skv = r.s7kv[ti] * s.w_scale
    gx = (
        jax.lax.dot_general(gqq_ref[...], wq_ref[...], NT, preferred_element_type=F32) * sq
        + jax.lax.dot_general(gkvq_ref[...], wkv_ref[...], NT, preferred_element_type=F32) * skv
    )
    gx = jnp.where(jnp.abs(x_ref[...].astype(F32)) <= s.x_max, gx, 0.0)
    gx_ref[...] = gx.astype(BF16)

  pltpu.emit_pipeline(
      body,
      grid=(s.n_tc, s.n_d),
      in_specs=[
          pl.BlockSpec((bt, s.cq), lambda ti, dj: (ti, 0)),
          pl.BlockSpec((bt, s.gkvd_w), lambda ti, dj: (ti, 0)),
          pl.BlockSpec((bt, bd), lambda ti, dj: (ti, dj)),
          pl.BlockSpec((bd, s.cq), lambda ti, dj: (dj, 0)),
          pl.BlockSpec((bd, s.gkvd_w), lambda ti, dj: (dj, 0)),
      ],
      out_specs=[
          pl.BlockSpec((bt, bd), lambda ti, dj: (ti, dj)),
          pl.BlockSpec((bt, s.cq), lambda ti, dj: (ti, 0)),
          pl.BlockSpec((bt, s.gkvd_w), lambda ti, dj: (ti, 0)),
      ],
  )(
      r.gqd_hm,
      r.gkvd_hm,
      r.x,
      r.wq_down,
      r.wkv_down,
      r.gx,
      r.gq_q,
      r.gkv_q,
  )  # pyformat: disable


def _p7b_down_wgrad(r: _Refs, s: _Static):
  """gw_down[d, c] = sum_t x_q[t, d] * g_q[t, c] * sg_t * act_scale."""
  bt, bd = s.tiling.bt, s.tiling.bd

  def run(acc_q, acc_kv):
    def body(x_ref, gqq_ref, gkvq_ref, gwq_ref, gwkv_ref):
      dj, ti = pl.program_id(0), pl.program_id(1)
      xq = _quant(x_ref[...].astype(F32), s.act_inv, s.act_fmax, s.e4m3)
      sq = r.s7q[ti] * s.act_scale
      skv = r.s7kv[ti] * s.act_scale
      cq = jax.lax.dot_general(xq, gqq_ref[...], TN, preferred_element_type=F32) * sq
      ckv = jax.lax.dot_general(xq, gkvq_ref[...], TN, preferred_element_type=F32) * skv
      last = ti == s.n_tc - 1
      _acc_store(acc_q, cq, ti == 0, last, gwq_ref, BF16)
      _acc_store(acc_kv, ckv, ti == 0, last, gwkv_ref, BF16)

    pltpu.emit_pipeline(
        body,
        grid=(s.n_d, s.n_tc),
        in_specs=[
            pl.BlockSpec((bt, bd), lambda dj, ti: (ti, dj)),
            pl.BlockSpec((bt, s.cq), lambda dj, ti: (ti, 0)),
            pl.BlockSpec((bt, s.gkvd_w), lambda dj, ti: (ti, 0)),
        ],
        out_specs=[
            pl.BlockSpec((bd, s.cq), lambda dj, ti: (dj, 0)),
            pl.BlockSpec((bd, s.gkvd_w), lambda dj, ti: (dj, 0)),
        ],
    )(r.x, r.gq_q, r.gkv_q, r.gwq_down, r.gwkv_down)

  pl.run_scoped(run, pltpu.VMEM((bd, s.cq), F32), pltpu.VMEM((bd, s.gkvd_w), F32))


# ------------------------------------------------------------------------------
# Kernel assembly.
# ------------------------------------------------------------------------------


ALL_PHASES = ("p1", "p2", "p2b", "p3", "p4", "p5a", "p5b", "p6", "p7a", "p7b")


def _kernel(coords_ref, *refs, names, s: _Static):
  """coords_ref: SMEM int32 [2, n_mesh_axes] = (own, peer) mesh coordinates."""
  r = _Refs(names, refs)
  core_pos = s.mesh_axes.index(s.core_axis)
  c = coords_ref[0, core_pos]
  peer = coords_ref[1, core_pos]
  # Full mesh coordinates of the peer: keeps the Mosaic device id lowering
  # from mixing in `axis_index` of the other mesh axes (which would need
  # `pvary` under `check_vma=True`).
  peer_id = {a: coords_ref[1, i] for i, a in enumerate(s.mesh_axes)}
  phases = {
      "p1": lambda: _p1_allgather(r, s, peer_id),
      "p2": lambda: _p2_out_dgrad(r, s, c, peer),
      "p2b": lambda: _p2b_out_wgrad(r, s),
      "p3": lambda: _p3_remat(r, s),
      "p4": lambda: _p4_attention_bwd(r, s),
      "p5a": lambda: _p5a_up_dgrad(r, s),
      "p5b": lambda: _p5b_up_wgrad(r, s),
      "p6": lambda: _p6_reduce_scatter_norm_bwd(r, s, c, peer, peer_id),
      "p7a": lambda: _p7a_down_dgrad(r, s),
      "p7b": lambda: _p7b_down_wgrad(r, s),
  }
  run_phases = s.phases or ALL_PHASES
  for name in run_phases:
    if name in phases:
      phases[name]()
  if "p1" in run_phases or "p6" in run_phases:
    _core_barrier(r, peer_id)


def _run_kernel(
    coords: jax.Array,
    inputs: dict[str, jax.Array],
    s: _Static,
    varying,
):
  """Runs the megakernel on one batch element; returns the real outputs."""
  t, t_c, d, cq, ckv, n_l = s.t, s.t_c, s.d, s.cq, s.ckv, s.n_l
  g_up = s.tiling.g_up
  e4m3, e5m2 = s.e4m3, s.e5m2
  n_g_up = s.n_g_up
  outputs = {
      # Real outputs.
      "gx": ((t_c, d), BF16),
      "gw_out": ((n_l, VDIM, d), BF16),
      "gwq_ext": ((cq, n_l * QEXT), BF16),
      "gwk": ((ckv, n_l * VDIM), BF16),
      "gwv": ((ckv, n_l * VDIM), BF16),
      "gwq_down": ((d, cq), BF16),
      "gwkv_down": ((d, s.gkvd_w), BF16),
      "gscale_q": ((NUM_SUBLANES, cq), F32),
      "gscale_kv": ((NUM_SUBLANES, ckv), F32),
      "auxr": ((t_c, 2 * NUM_LANES), F32),
      # HBM scratch.
      "go_peer": ((t_c, d), BF16),
      "gq_hm": ((t, d), e5m2),
      "oq_hm": ((t, n_l * VDIM), e4m3),
      "do_hm": ((n_l, t, VDIM), BF16),
      "di8": ((n_l, NUM_SUBLANES, t), F32),
      "cq_q": ((t, cq), e4m3),
      "ckv_q": ((t, ckv), e4m3),
      "ckv_bf": ((t, ckv), BF16),
      "q_hm": ((n_l, t, QEXT), BF16),
      "qpe_hm": ((n_l, t, NUM_LANES), BF16),
      "k_hm": ((n_l, t, QEXT), BF16),
      "v_hm": ((n_l, t, VDIM), BF16),
      "dq_hm": ((n_l, t, QEXT), BF16),
      "dk_hm": ((n_l, t, QEXT), BF16),
      "dv_hm": ((n_l, t, VDIM), BF16),
      "gcq": ((t, cq), F32),
      "gckv": ((t, ckv), F32),
      "aux": ((t, 2 * NUM_LANES), F32),
      "gext_q": ((n_g_up, t, g_up * QEXT), e5m2),
      "gv_q": ((n_g_up, t, g_up * VDIM), e5m2),
      "rs_cq": ((t_c, cq), F32),
      "rs_ckv": ((t_c, ckv), F32),
      "rs_aux": ((t_c, 2 * NUM_LANES), F32),
      "gqd_hm": ((t_c, cq), BF16),
      "gkvd_hm": ((t_c, s.gkvd_w), BF16),
      "gq_q": ((t_c, cq), e5m2),
      "gkv_q": ((t_c, s.gkvd_w), e5m2),
  }
  num_real = 10
  scratch = {
      "sems": pltpu.SemaphoreType.DMA((8,)),
      "bsem": pltpu.SemaphoreType.REGULAR((1,)),
      "gsc2": pltpu.VMEM((s.n_t, 1, 1), F32),
      "s5q": pltpu.VMEM((s.n_t * n_g_up, 1, 1), F32),
      "s5v": pltpu.VMEM((s.n_t * n_g_up, 1, 1), F32),
      "s7q": pltpu.VMEM((s.n_tc, 1, 1), F32),
      "s7kv": pltpu.VMEM((s.n_tc, 1, 1), F32),
  }
  names = list(inputs) + list(outputs) + list(scratch)
  mat = jax.sharding.ManualAxisType(varying=frozenset(varying))
  hbm = pl.BlockSpec(memory_space=pltpu.HBM)
  run_phases = s.phases or ALL_PHASES
  uses_barrier = "p1" in run_phases or "p6" in run_phases
  # BlockSpec(memory_space=pltpu.HBM) marks the Mosaic MLIR arguments as HBM
  # passthroughs, but JAX only populates custom_call_config memory space colors
  # when the input/output avals carry pltpu.HBM. Without explicit aval coloring,
  # XLA's TpuCustomCallMemorySpacePolicy skips buffers that alias with a while
  # loop (e.g. lax.scan over layers), allowing MSA to hoist small constants like
  # `rot` into VMEM while the kernel issues HBM DMAs from their VMEM offset.
  hbm_inputs = [pltpu.with_memory_space_constraint(x, pltpu.HBM) for x in inputs.values()]
  ref_aval = jax.typeof(coords)
  outs = pl.pallas_call(
      functools.partial(_kernel, names=names, s=s),
      out_shape=[
          pltpu.HBM.from_type(ref_aval.update(shape=shape, dtype=dtype, manual_axis_type=mat))
          for shape, dtype in outputs.values()
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[hbm] * len(inputs),
          out_specs=[hbm] * len(outputs),
          scratch_shapes=list(scratch.values()),
      ),
      compiler_params=pltpu.CompilerParams(
          # The barrier semaphore (phases p1 / p6) requires a collective id;
          # Pallas rejects one when no barrier is used (profiling subsets).
          collective_id=0 if uses_barrier else None,
          fuse_transposed_lhs_in_matmul=s.tiling.fuse_transposed_lhs_in_matmul,
          vmem_limit_bytes=s.tiling.vmem_limit_bytes,
      ),
      name="mla_l4_megakernel_bwd_fused",
  )(coords, *hbm_inputs)
  return dict(zip(list(outputs)[:num_real], outs[:num_real]))


def _rotate_half(x: jax.Array) -> jax.Array:
  x1, x2 = jnp.split(x, 2, axis=-1)
  return jnp.concatenate([-x2, x1], axis=-1)


def _rot_matrix() -> np.ndarray:
  """[128, 128] block-diagonal matrix with `y @ R = rotate_half(y)` per half."""
  r = np.zeros((ROPE, ROPE), np.float32)
  h = ROPE // 2
  r[h:, :h] = -np.eye(h, dtype=np.float32)
  r[:h, h:] = np.eye(h, dtype=np.float32)
  out = np.zeros((NUM_LANES, NUM_LANES), np.float32)
  out[:ROPE, :ROPE] = r
  out[ROPE:, ROPE:] = r
  return out


def _negate_fp8(x: jax.Array) -> jax.Array:
  return (-x.astype(BF16)).astype(x.dtype)


def _prep_weights(w: dsv3_types.DSv3MLAWeightsPytree, s: _Static):
  """Relayouts the local weight shards for the kernel (XLA, per device)."""
  cq, ckv, d, n_l = s.cq, s.ckv, s.d, s.n_l
  wq, wk_raw, wv, w_out, w_q_down, w_kv_down = (
      w.q_up,
      w.k_up,
      w.v_up,
      w.out,
      w.q_down,
      w.kv_down,
  )
  assert wq is not None and wk_raw is not None and wv is not None
  assert w_out is not None and w_q_down is not None and w_kv_down is not None
  # wq: [Cq, N_l, 192] e4m3 -> [Cq, N_l * 256] e4m3.
  w_nope, w_pe = wq[..., :QK_NOPE], wq[..., QK_NOPE:]
  w_pe_rot = jnp.concatenate([_negate_fp8(w_pe[..., ROPE // 2 :]), w_pe[..., : ROPE // 2]], axis=-1)
  wq_ext = jnp.concatenate([w_nope, w_pe, w_pe_rot], axis=-1).reshape(cq, n_l * QEXT)
  wk = wk_raw.astype(BF16).reshape(ckv, n_l * VDIM)
  wv = wv.reshape(ckv, n_l * VDIM)
  w_out_2d = w_out.reshape(n_l * VDIM, d)
  wkv_down_pad = jnp.pad(w_kv_down, ((0, 0), (0, s.gkvd_w - s.kvd_w)))
  return dict(
      w_out=w_out_2d,
      wq_ext=wq_ext,
      wk=wk,
      wv=wv,
      wq_down=w_q_down,
      wkv_down=wkv_down_pad,
  )


def _post_weights(o: dict[str, jax.Array], s: _Static):
  """Undoes `_prep_weights`' relayouts on the weight gradients."""
  cq, ckv, n_l = s.cq, s.ckv, s.n_l
  ext = o["gwq_ext"].reshape(cq, n_l, QEXT)
  a = ext[..., QK_NOPE : QK_NOPE + ROPE].astype(F32)
  b = ext[..., QK_NOPE + ROPE :].astype(F32)
  g_pe = a - _rotate_half(b)  # rotate_half^T = -rotate_half.
  gwq_up = jnp.concatenate([ext[..., :QK_NOPE], g_pe.astype(BF16)], axis=-1)
  gwk = o["gwk"].reshape(ckv, n_l, VDIM)
  gwv = o["gwv"].reshape(ckv, n_l, VDIM)
  gw_out = o["gw_out"]
  gwq_down = o["gwq_down"]
  gwkv_down = o["gwkv_down"][:, : s.kvd_w]
  return gwq_up, gwk, gwv, gw_out, gwq_down, gwkv_down


def fused_mla_bwd(
    grad_out: jax.Array,
    norm_x: jax.Array,
    q_down: jax.Array,
    kv_down: jax.Array,
    context: jax.Array,
    splash_out: jax.Array,
    w: dsv3_types.DSv3MLAWeightsPytree,
    yarn_freqs: tuple[jax.Array, jax.Array],
    splash_kernel: Any,
    *,
    kv_lora_rank: int,
    qk_head_dim: int,
    rope_head_dim: int,
    num_query_heads: int,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    rope_factor: int,
    mscale: float,
    norm_fn: Any,
    rope_fn: Any,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    axis_mapping: Any,
    segment_ids: jax.Array | None = None,
    quant_rule: quantization.GmmQuantRule,
    tiling: FusedBwdTiling | None = None,
    phases: tuple[str, ...] | None = None,
):
  """Fused fp8 MLA backward; same contract as `dsv3_mla.dsv3_mla_bwd`.

  `phases` (profiling only) restricts the kernel to a subset of `ALL_PHASES`;
  the outputs are then garbage.
  """
  del rope_fn, num_query_heads  # YaRN is assumed; checked by the caller.
  eps = norm_epsilon(norm_fn)
  assert eps is not None, norm_fn
  assert qk_head_dim == QK_NOPE and rope_head_dim == ROPE, (
      qk_head_dim,
      rope_head_dim,
  )
  assert w.q_down is not None and w.q_up is not None
  assert w.q_norm_scale is not None and w.kv_down is not None
  assert w.k_up is not None and w.v_up is not None
  assert w.kv_norm_scale is not None and w.out is not None
  q_scale_bf16 = w.q_norm_scale.dtype == BF16
  kv_scale_bf16 = w.kv_norm_scale.dtype == BF16

  rule = quant_rule
  w = dataclasses.replace(
      w,
      q_down=_quantize_weight(w.q_down, rule),
      q_up=_quantize_weight(w.q_up, rule),
      kv_down=_quantize_weight(w.kv_down, rule),
      v_up=_quantize_weight(w.v_up, rule),
      out=_quantize_weight(w.out, rule),
  )
  config = splash_kernel.kwargs["config"]
  base2 = bool(getattr(config, "use_base2_exp", True))
  if tiling is None:
    tiling = FusedBwdTiling()

  attn_sharding = jax.typeof(splash_out).sharding
  assert isinstance(attn_sharding, jax.sharding.NamedSharding)
  batch_axis = _spec_entry(attn_sharding.spec, 0)
  head_axis = _spec_entry(attn_sharding.spec, 2)
  core_axes = _axis_names(head_axis)
  assert len(core_axes) == 1, head_axis
  core_axis = core_axes[0]
  assert mesh.shape[core_axis] == 2, mesh.shape
  num_cores = 2
  _, t, d = norm_x.shape
  n = splash_out.shape[2]
  cq = q_down.shape[-1]
  ckv = kv_lora_rank
  assert kv_down.shape[-1] == ckv + ROPE
  assert splash_out.shape[-1] == VDIM
  n_l = n // num_cores
  t_c = t // num_cores
  softmax_scale = _softmax_scale(
      qk_head_dim=qk_head_dim,
      rope_head_dim=rope_head_dim,
      max_position_embeddings=max_position_embeddings,
      original_max_position_embeddings=original_max_position_embeddings,
      rope_factor=rope_factor,
      mscale=mscale,
  )
  s = _Static(
      t=t,
      t_c=t_c,
      d=d,
      cq=cq,
      ckv=ckv,
      n_l=n_l,
      eps=eps,
      softmax_scale=softmax_scale,
      base2=base2,
      has_seg=segment_ids is not None,
      core_axis=core_axis,
      mesh_axes=tuple(mesh.axis_names),
      tiling=tiling,
      act_inv=1.0 / rule.act_scale,
      act_fmax=float(jnp.finfo(rule.act_qtype).max),
      act_scale=rule.act_scale,
      w_scale=rule.weight_scale,
      bwd_fmax=float(jnp.finfo(rule.bwd_qtype).max),
      x_max=rule.act_scale * float(jnp.finfo(rule.act_qtype).max),
      e4m3=rule.act_qtype,
      e5m2=rule.bwd_qtype,
      q_scale_bf16=q_scale_bf16,
      kv_scale_bf16=kv_scale_bf16,
      phases=tuple(phases or ()),
  )
  tl = tiling
  assert t % tl.bt == 0 and t_c % tl.bt == 0, (t, t_c, tl.bt)
  assert n_l % tl.g_out == 0 and n_l % s.g_out_w == 0 and n_l % tl.g_up == 0, (
      n_l,
      tl,
  )
  assert d % tl.bd == 0 and tl.bd % NUM_LANES == 0, (d, tl.bd)
  assert t % tl.bq == 0 and tl.bq % tl.bkv_compute == 0, (t, tl)
  assert tl.bt % NUM_SUBLANES == 0 and tl.bq % NUM_LANES == 0
  assert ckv % NUM_LANES == 0 and cq % NUM_LANES == 0, (ckv, cq)

  # Local sequence rows of grad_out, like `out_projection_bwd`.
  seq_sharding = attn_sharding.update(spec=P(batch_axis, head_axis, None))
  x_sharding = jax.typeof(norm_x).sharding
  grad_out = jax.reshard(grad_out, seq_sharding)
  norm_x = jax.reshard(norm_x, seq_sharding)
  cos, sin = yarn_freqs
  rot = jnp.asarray(_rot_matrix())
  scale_bf = jnp.asarray(softmax_scale * rule.act_scale * rule.weight_scale, BF16)  # pyformat: disable

  gx_spec = _cotangent_sharding(norm_x).spec
  gw_specs = jax.tree.map(lambda a: _cotangent_sharding(a).spec, w)
  freq_spec = P(batch_axis, head_axis, None, None)

  def local(grad_out, norm_x, q_down, kv_down, context, splash_out, w, cos, sin, segment_ids):  # pyformat: disable
    c = jax.lax.axis_index(core_axis)
    bl = grad_out.shape[0]
    wd = _prep_weights(w, s)
    varying = set()
    for a in jax.tree.leaves(
        (grad_out, norm_x, q_down, kv_down, context, splash_out, w, cos, sin, segment_ids)
    ):  # pyformat: disable
      varying |= set(jax.typeof(a).manual_axis_type.varying)
    # (own, peer) mesh coordinates for the in-kernel remote DMAs.
    axis_names = s.mesh_axes

    def coord(a):
      i = jax.lax.axis_index(a).astype(jnp.int32)
      others = tuple(b for b in axis_names if b != a)
      return jax.lax.pcast(i, others, to="varying") if others else i

    own = jnp.stack([coord(a) for a in axis_names])
    core_pos = axis_names.index(core_axis)
    peer_coords = own.at[core_pos].set(1 - own[core_pos])
    coords = jnp.stack([own, peer_coords])
    q_scale = w.q_norm_scale.astype(F32).reshape(1, cq)
    kv_scale = w.kv_norm_scale.astype(F32).reshape(1, ckv)
    per_b = []
    for bi in range(bl):
      cos_b = cos[bi, :, 0, :]  # [T, 64] bf16
      sin_b = sin[bi, :, 0, :]
      cs = jnp.concatenate([cos_b, sin_b], axis=-1)
      kpe = kv_down[bi, :, ckv:]
      inputs = dict(
          go=grad_out[bi],
          x=norm_x[bi],
          qd=q_down[bi],
          kvd=kv_down[bi],
          kpe2=jnp.concatenate([kpe, kpe], axis=-1),
          lse8=jnp.broadcast_to(context[bi][:, None, :].astype(F32), (n_l, NUM_SUBLANES, t)),
          o=splash_out[bi].reshape(t, n_l * VDIM),
          cs_s=(cs * scale_bf).astype(BF16),
          cs=cs.astype(F32),
          cos2=jnp.concatenate([cos_b, cos_b], axis=-1).astype(F32),
          sin2=jnp.concatenate([sin_b, sin_b], axis=-1).astype(F32),
          rot=rot,
          q_scale=q_scale,
          kv_scale=kv_scale,
          **wd,
      )
      if segment_ids is not None:
        seg = segment_ids[bi].astype(jnp.int32)
        inputs["qseg8"] = jnp.broadcast_to(seg[None, :], (NUM_SUBLANES, t))
        inputs["kvseg"] = jnp.broadcast_to(seg[:, None], (t, NUM_LANES))
      per_b.append(_run_kernel(coords, inputs, s, varying))

    def stack(name):
      return jnp.stack([o[name] for o in per_b])

    def wsum(name):
      if bl == 1:
        return per_b[0][name]
      summed = jnp.sum(stack(name).astype(F32), axis=0)
      return summed.astype(per_b[0][name].dtype)

    gx = stack("gx")
    wsummed = {k: wsum(k) for k in ("gwq_ext", "gwk", "gwv", "gw_out", "gwq_down", "gwkv_down")}  # pyformat: disable
    gwq_up, gwk, gwv, gw_out, gwq_down, gwkv_down = _post_weights(wsummed, s)
    gscale_q = jnp.sum(stack("gscale_q"), axis=(0, 1))
    gscale_kv = jnp.sum(stack("gscale_kv"), axis=(0, 1))
    # YaRN frequency gradients on the local rows.
    auxr = stack("auxr")  # [Bl, T_c, 256] f32, reduced over the core axis.
    kpe_loc = jax.lax.dynamic_slice_in_dim(kv_down[:, :, ckv:], c * t_c, t_c, axis=1).astype(F32)
    g_kpe = auxr[..., NUM_LANES : NUM_LANES + ROPE]
    dcos = auxr[..., :ROPE] + kpe_loc * g_kpe
    dsin = auxr[..., ROPE:NUM_LANES] + _rotate_half(kpe_loc) * g_kpe
    dcos = dcos.astype(cos.dtype)[:, :, None, :]
    dsin = dsin.astype(sin.dtype)[:, :, None, :]
    grad_w = dsv3_types.DSv3MLAWeightsPytree(
        q_down=gwq_down,
        q_up=gwq_up,
        q_norm_scale=gscale_q,
        kv_down=gwkv_down,
        k_up=gwk,
        v_up=gwv,
        kv_norm_scale=gscale_kv,
        out=gw_out,
    )
    grad_w = jax.tree.map(_match_spec, grad_w, gw_specs)
    grad_w = dataclasses.replace(
        grad_w,
        q_norm_scale=grad_w.q_norm_scale.astype(w.q_norm_scale.dtype),
        kv_norm_scale=grad_w.kv_norm_scale.astype(w.kv_norm_scale.dtype),
    )
    return (
        _match_spec(gx, gx_spec),
        grad_w,
        (_match_spec(dcos, freq_spec), _match_spec(dsin, freq_spec)),
    )

  gx, grad_w, (dcos, dsin) = jax.shard_map(
      local,
      mesh=mesh,
      out_specs=(gx_spec, gw_specs, (freq_spec, freq_spec)),
      check_vma=True,
  )(
      grad_out,
      norm_x,
      q_down,
      kv_down,
      context,
      splash_out,
      w,
      cos,
      sin,
      segment_ids,
  )  # pyformat: disable
  freq_sharding = jax.typeof(cos).sharding
  grad_freqs = (
      jax.reshard(dcos, freq_sharding),
      jax.reshard(dsin, freq_sharding),
  )
  return jax.reshard(gx, x_sharding), grad_w, grad_freqs

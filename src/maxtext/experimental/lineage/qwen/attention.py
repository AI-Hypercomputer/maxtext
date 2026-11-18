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

"""Pure-JAX SEQ_MINOR Full GQA Splash Attention mixer for Qwen MAX & Qwen 3.5."""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol

import jax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.pallas.ops.tpu import splash_attention
import jax.numpy as jnp

from maxtext.experimental.lineage.qwen import gdn
from maxtext.experimental.lineage.qwen.config import QwenConfig

# Zero-host-offload HBM activation banking names (`S(0)` HBM only; ZERO `S(5)`
# `pinned_host` offload).
ATTN_SAVED_NAMES: tuple[str, ...] = (
    "context",
    "query_proj",
    "key_proj",
    "value_proj",
    "out_proj",
)


class AttnWeightCarrier(Protocol):
  """Structural protocol matching `GatheredAttnWeights` and `AttnWeights`."""

  pre_attn_norm: jax.Array
  q_proj: Any
  k_proj: Any
  v_proj: Any
  q_norm: jax.Array
  k_norm: jax.Array
  out_proj: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class AttnWeights:
  """Standalone pure-JAX Full GQA weight PyTree (for isolated tests & benchmarks)."""

  pre_attn_norm: jax.Array
  q_proj: jax.Array | gdn.QuantizedDenseWeightFp8
  k_proj: jax.Array | gdn.QuantizedDenseWeightFp8
  v_proj: jax.Array | gdn.QuantizedDenseWeightFp8
  q_norm: jax.Array
  k_norm: jax.Array
  out_proj: jax.Array | gdn.QuantizedDenseWeightFp8


def compute_rope_cos_sin(
    positions: jax.Array,
    rotary_dim: int,
    max_timescale: float = 10_000_000.0,
) -> tuple[jax.Array, jax.Array]:
  """Precomputes `(cos, sin)` (`[B, S, 1, rotary_dim // 2]` fp32) outside `scan`."""
  half = rotary_dim // 2
  freq_seq = jnp.arange(0, half, dtype=jnp.float32)
  inv_freq = 1.0 / (max_timescale ** (freq_seq / float(half)))
  angles = (
      positions.astype(jnp.float32)[:, :, None, None]
      * inv_freq[None, None, None, :]
  )
  return jnp.cos(angles), jnp.sin(angles)


def apply_partial_rope(
    x: jax.Array,
    positions: jax.Array,
    rotary_dim: int,
    max_timescale: float = 10_000_000.0,
    *,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
  """Applies NeoX-style half-rotated RoPE to the first `rotary_dim` channels.

  Args:
    x: `[B, S, H, D_h]` query or key tensor.
    positions: `[B, S]` token position indices.
    rotary_dim: Number of rotated channels (`64` = `0.25 * 256` for Qwen).
    max_timescale: RoPE base frequency (`10_000_000.0`).
    rope_cos_sin: Optional precomputed `(cos, sin)` tables from
      `compute_rope_cos_sin`.

  Returns:
    Rotated tensor with shape `x.shape` and dtype `x.dtype`.
  """
  orig_dtype = x.dtype
  x_rot = x[..., :rotary_dim].astype(jnp.float32)
  x_pass = x[..., rotary_dim:]

  half = rotary_dim // 2
  if rope_cos_sin is not None:
    cos, sin = rope_cos_sin
  else:
    cos, sin = compute_rope_cos_sin(positions, rotary_dim, max_timescale)

  x1, x2 = x_rot[..., :half], x_rot[..., half:]
  y1 = x1 * cos - x2 * sin
  y2 = x2 * cos + x1 * sin
  y_rot = jnp.concatenate([y1, y2], axis=-1).astype(orig_dtype)
  return jnp.concatenate([y_rot, x_pass], axis=-1)


@jax.custom_vjp
def _attn_sigmoid_gate(attn_out: jax.Array, gate: jax.Array) -> jax.Array:
  """Computes `(attn_out * sigmoid(gate))` in FP32 while saving only BF16 inputs."""
  s = jax.nn.sigmoid(gate.astype(jnp.float32))
  return (attn_out.astype(jnp.float32) * s).astype(gate.dtype)


def _attn_sigmoid_gate_fwd(attn_out: jax.Array, gate: jax.Array):
  return _attn_sigmoid_gate(attn_out, gate), (attn_out, gate)


def _attn_sigmoid_gate_bwd(res, g_out: jax.Array):
  attn_out, gate = res
  a_f32 = attn_out.astype(jnp.float32)
  g_f32 = g_out.astype(jnp.float32)
  s = jax.nn.sigmoid(gate.astype(jnp.float32))
  g_attn = (g_f32 * s).astype(attn_out.dtype)
  g_gate = (g_f32 * a_f32 * s * (1.0 - s)).astype(gate.dtype)
  return g_attn, g_gate


_attn_sigmoid_gate.defvjp(_attn_sigmoid_gate_fwd, _attn_sigmoid_gate_bwd)


def _project_qkv(
    normed: jax.Array,
    proj_w: Any,
    *,
    num_heads: int,
    out_head_dim: int,
    use_2d_matmul: bool,
    use_fp8: bool,
) -> jax.Array:
  """Projects `[B, S, D]` -> `[B, S, H, C]` via 3D einsum, 2D matmul, or native FP8."""
  bsz, seq_len, _ = normed.shape
  if use_fp8 or gdn.is_quantized_weight_fp8(proj_w):
    flat = gdn.dense_linear(normed, proj_w, use_fp8=True)
    return flat.reshape(bsz, seq_len, num_heads, out_head_dim)
  if use_2d_matmul or proj_w.ndim == 2:
    flat = jnp.matmul(normed, proj_w.reshape(normed.shape[-1], -1))
    return flat.reshape(bsz, seq_len, num_heads, out_head_dim)
  return jnp.einsum("bsd,dhc->bshc", normed, proj_w)


def _splash_core(
    q_scaled: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    sa_v_seq_minor: bool | None = None,
) -> jax.Array:
  """Executes only the Splash Attention (or pure-JAX CPU) core."""
  _, seq_len, _, _ = q_scaled.shape
  if cfg.use_pallas_kernels:
    q_bhsd = jnp.swapaxes(q_scaled, 1, 2)
    k_bhsd = jnp.swapaxes(k, 1, 2)
    v_bhsd = jnp.swapaxes(v, 1, 2)

    qk_layout = (
        splash_attention.QKVLayout.SEQ_MINOR
        if cfg.sa_qk_seq_minor
        else splash_attention.QKVLayout.HEAD_DIM_MINOR
    )
    if sa_v_seq_minor is None:
      v_layout = qk_layout
    else:
      v_layout = (
          splash_attention.QKVLayout.SEQ_MINOR
          if sa_v_seq_minor
          else splash_attention.QKVLayout.HEAD_DIM_MINOR
      )
    block_sizes = splash_attention.BlockSizes(
        block_q=min(cfg.sa_block_q, seq_len),
        block_kv=min(cfg.sa_block_kv, seq_len),
        block_kv_compute=min(cfg.sa_block_kv_compute, seq_len),
        block_q_dkv=min(cfg.sa_block_q_dkv, seq_len),
        block_kv_dkv=min(cfg.sa_block_kv_dkv, seq_len),
        block_kv_dkv_compute=min(cfg.sa_block_kv_dkv_compute, seq_len),
        use_fused_bwd_kernel=True,
        q_layout=qk_layout,
        k_layout=qk_layout,
        v_layout=v_layout,
    )
    causal = splash_attention.CausalMask(shape=(seq_len, seq_len))
    mh_mask = splash_attention.MultiHeadMask(
        masks=[causal for _ in range(cfg.num_query_heads)]
    )
    splash_fn = splash_attention.make_splash_mha(
        mh_mask,
        block_sizes=block_sizes,
        head_shards=1,
        q_seq_shards=1,
        residual_checkpoint_name="context",
    )

    def _single_batch_splash(q_i, k_i, v_i, seg_i):
      seg = splash_attention.SegmentIds(q=seg_i, kv=seg_i)
      return splash_fn(q_i, k_i, v_i, segment_ids=seg)

    attn_bhsd = jax.vmap(_single_batch_splash)(
        q_bhsd, k_bhsd, v_bhsd, segment_ids
    )
    if cfg.attn_bf16_out_fused_gate:
      attn_bhsd = attn_bhsd.astype(q_scaled.dtype)
    return jnp.swapaxes(attn_bhsd, 1, 2)

  repeats = cfg.num_query_heads // cfg.num_kv_heads
  k_rep = jnp.repeat(k, repeats, axis=2)
  v_rep = jnp.repeat(v, repeats, axis=2)
  logits = jnp.einsum(
      "bshd,bthd->bhst",
      q_scaled.astype(jnp.float32),
      k_rep.astype(jnp.float32),
      precision=jax.lax.Precision.HIGHEST,
  )
  idx = jnp.arange(seq_len)
  causal_mask = idx[:, None] >= idx[None, :]
  same_seg = (segment_ids[:, :, None] == segment_ids[:, None, :]) & (
      segment_ids[:, :, None] > 0
  )
  mask = causal_mask[None, None, :, :] & same_seg[:, None, :, :]
  logits = jnp.where(mask, logits, -1e30)
  probs = jax.nn.softmax(logits, axis=-1)
  attn_out = jnp.einsum(
      "bhst,bthd->bshd",
      probs,
      v_rep.astype(jnp.float32),
      precision=jax.lax.Precision.HIGHEST,
  ).astype(q_scaled.dtype)
  return checkpoint_name(attn_out, "context")


def full_attn_mixer(
    x: jax.Array,
    w: AttnWeightCarrier,
    positions: jax.Array,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
    sa_v_seq_minor: bool | None = None,
) -> jax.Array:
  """Executes Qwen Full GQA Attention (`x`: `[B_mb, S, D]`) and returns `x + attn_out`.

  Supports both Qwen MAX (`D=8192, H_q=64, H_kv=4, D_h=256, rotary_dim=64`) and
  Qwen 3.5 (`D=4096, H_q=32, H_kv=2, D_h=256, rotary_dim=64`) with `SEQ_MINOR`
  Splash Attention (`use_fused_bwd_kernel=True`), zero-centered QK norms,
  sigmoid output gate, HBM activation banking (`context`, `query_proj`,
  `key_proj`, `value_proj`, `out_proj` — zero `S(5)` host offload), and optional
  P1 native MXU FP8 QKV/Out projections.

  Args:
    x: Input activations of shape `[B_mb, S, emb_dim]`.
    w: Full attention weights PyTree matching `AttnWeightCarrier`.
    positions: Token positions of shape `[B_mb, S]`.
    segment_ids: Packed sequence segment IDs of shape `[B_mb, S]`.
    cfg: Frozen `QwenConfig` specifying dimensions and Splash block sizes.
    rope_cos_sin: Optional precomputed `(cos, sin)` tables from
      `compute_rope_cos_sin`.
    sa_v_seq_minor: Optional override for Splash V tensor `SEQ_MINOR` layout.

  Returns:
    Output activations `x + attn_out` of shape `[B_mb, S, emb_dim]`.
  """
  bsz, seq_len, _ = x.shape
  use_fp8 = bool(cfg.dense_fp8_matmul or cfg.dense_fp8_attn)
  normed = gdn.rms_norm_zero_centered(x, w.pre_attn_norm, cfg.norm_eps)

  qg = _project_qkv(
      normed,
      w.q_proj,
      num_heads=cfg.num_query_heads,
      out_head_dim=2 * cfg.head_dim,
      use_2d_matmul=cfg.attn_qkv_2d_matmul,
      use_fp8=use_fp8,
  )
  q_raw, gate = qg[..., : cfg.head_dim], qg[..., cfg.head_dim :]
  q_raw = checkpoint_name(q_raw, "query_proj")

  k_raw = _project_qkv(
      normed,
      w.k_proj,
      num_heads=cfg.num_kv_heads,
      out_head_dim=cfg.head_dim,
      use_2d_matmul=cfg.attn_qkv_2d_matmul,
      use_fp8=use_fp8,
  )
  k_raw = checkpoint_name(k_raw, "key_proj")

  v = _project_qkv(
      normed,
      w.v_proj,
      num_heads=cfg.num_kv_heads,
      out_head_dim=cfg.head_dim,
      use_2d_matmul=cfg.attn_qkv_2d_matmul,
      use_fp8=use_fp8,
  )
  v = checkpoint_name(v, "value_proj")

  q = gdn.rms_norm_zero_centered(q_raw, w.q_norm, cfg.norm_eps)
  k = gdn.rms_norm_zero_centered(k_raw, w.k_norm, cfg.norm_eps)

  q = apply_partial_rope(
      q,
      positions,
      cfg.rotary_dim,
      cfg.rope_max_timescale,
      rope_cos_sin=rope_cos_sin,
  )
  k = apply_partial_rope(
      k,
      positions,
      cfg.rotary_dim,
      cfg.rope_max_timescale,
      rope_cos_sin=rope_cos_sin,
  )

  scale = cfg.head_dim**-0.5
  q_scaled = (q.astype(jnp.float32) * scale).astype(x.dtype)
  attn_out = _splash_core(
      q_scaled, k, v, segment_ids, cfg, sa_v_seq_minor=sa_v_seq_minor
  )

  gate_3d = gate.reshape(bsz, seq_len, cfg.attn_q_out_dim)
  if cfg.use_pallas_kernels and cfg.attn_gate_bf16_align:
    gate_3d = gdn.forward_layout_constraint(gate_3d, (2, 0, 1))
  if cfg.attn_bf16_out_fused_gate:
    gated = _attn_sigmoid_gate(
        attn_out.reshape(bsz, seq_len, cfg.attn_q_out_dim), gate_3d
    )
  else:
    gated = (
        attn_out.reshape(bsz, seq_len, cfg.attn_q_out_dim).astype(jnp.float32)
        * jax.nn.sigmoid(gate_3d.astype(jnp.float32))
    ).astype(x.dtype)
  proj = gdn.dense_linear(gated, w.out_proj, use_fp8=use_fp8)
  proj = checkpoint_name(proj, "out_proj")
  return x + proj


def full_attn_mixer_fwd_save_res(
    x: jax.Array,
    w: AttnWeightCarrier,
    positions: jax.Array,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
    sa_v_seq_minor: bool | None = None,
) -> tuple[jax.Array, tuple[Any, jax.Array]]:
  """Forward `full_attn_mixer` returning `(y, (vjp_splash, attn_out))` with zero weight capture."""
  bsz, seq_len, _ = x.shape
  use_fp8 = bool(cfg.dense_fp8_matmul or cfg.dense_fp8_attn)
  normed = gdn.rms_norm_zero_centered(x, w.pre_attn_norm, cfg.norm_eps)
  qg = _project_qkv(
      normed,
      w.q_proj,
      num_heads=cfg.num_query_heads,
      out_head_dim=2 * cfg.head_dim,
      use_2d_matmul=cfg.attn_qkv_2d_matmul,
      use_fp8=use_fp8,
  )
  q_raw, gate = qg[..., : cfg.head_dim], qg[..., cfg.head_dim :]
  k_raw = _project_qkv(
      normed,
      w.k_proj,
      num_heads=cfg.num_kv_heads,
      out_head_dim=cfg.head_dim,
      use_2d_matmul=cfg.attn_qkv_2d_matmul,
      use_fp8=use_fp8,
  )
  v = _project_qkv(
      normed,
      w.v_proj,
      num_heads=cfg.num_kv_heads,
      out_head_dim=cfg.head_dim,
      use_2d_matmul=cfg.attn_qkv_2d_matmul,
      use_fp8=use_fp8,
  )
  q = gdn.rms_norm_zero_centered(q_raw, w.q_norm, cfg.norm_eps)
  k = gdn.rms_norm_zero_centered(k_raw, w.k_norm, cfg.norm_eps)
  q = apply_partial_rope(
      q,
      positions,
      cfg.rotary_dim,
      cfg.rope_max_timescale,
      rope_cos_sin=rope_cos_sin,
  )
  k = apply_partial_rope(
      k,
      positions,
      cfg.rotary_dim,
      cfg.rope_max_timescale,
      rope_cos_sin=rope_cos_sin,
  )
  scale = cfg.head_dim**-0.5
  q_scaled = (q.astype(jnp.float32) * scale).astype(x.dtype)

  attn_out, vjp_splash = jax.vjp(
      lambda q_in, k_in, v_in: _splash_core(
          q_in, k_in, v_in, segment_ids, cfg, sa_v_seq_minor=sa_v_seq_minor
      ),
      q_scaled,
      k,
      v,
  )

  gate_3d = gate.reshape(bsz, seq_len, cfg.attn_q_out_dim)
  if cfg.use_pallas_kernels and cfg.attn_gate_bf16_align:
    gate_3d = gdn.forward_layout_constraint(gate_3d, (2, 0, 1))
  if cfg.attn_bf16_out_fused_gate:
    gated = _attn_sigmoid_gate(
        attn_out.reshape(bsz, seq_len, cfg.attn_q_out_dim), gate_3d
    )
  else:
    gated = (
        attn_out.reshape(bsz, seq_len, cfg.attn_q_out_dim).astype(jnp.float32)
        * jax.nn.sigmoid(gate_3d.astype(jnp.float32))
    ).astype(x.dtype)
  proj = gdn.dense_linear(gated, w.out_proj, use_fp8=use_fp8)
  return x + proj, (vjp_splash, attn_out)


def full_attn_mixer_bwd_from_res(
    g_y: jax.Array,
    x: jax.Array,
    saved_res: tuple[Any, jax.Array],
    w: AttnWeightCarrier,
    positions: jax.Array,
    segment_ids: jax.Array,
    cfg: QwenConfig,
    *,
    rope_cos_sin: tuple[jax.Array, jax.Array] | None = None,
) -> tuple[jax.Array, AttnWeights]:
  """Backward `full_attn_mixer` reusing saved Splash residuals (zero Splash FWD remat)."""
  del segment_ids
  vjp_splash, attn_out = saved_res
  bsz, seq_len, _ = x.shape
  use_fp8 = bool(cfg.dense_fp8_matmul or cfg.dense_fp8_attn)

  def _pre_splash_and_gate(
      x_in, pre_norm_w, q_proj_w, k_proj_w, v_proj_w, q_norm_w, k_norm_w
  ):
    normed = gdn.rms_norm_zero_centered(x_in, pre_norm_w, cfg.norm_eps)
    qg = _project_qkv(
        normed,
        q_proj_w,
        num_heads=cfg.num_query_heads,
        out_head_dim=2 * cfg.head_dim,
        use_2d_matmul=cfg.attn_qkv_2d_matmul,
        use_fp8=use_fp8,
    )
    q_raw, gate = qg[..., : cfg.head_dim], qg[..., cfg.head_dim :]
    k_raw = _project_qkv(
        normed,
        k_proj_w,
        num_heads=cfg.num_kv_heads,
        out_head_dim=cfg.head_dim,
        use_2d_matmul=cfg.attn_qkv_2d_matmul,
        use_fp8=use_fp8,
    )
    v = _project_qkv(
        normed,
        v_proj_w,
        num_heads=cfg.num_kv_heads,
        out_head_dim=cfg.head_dim,
        use_2d_matmul=cfg.attn_qkv_2d_matmul,
        use_fp8=use_fp8,
    )
    q = gdn.rms_norm_zero_centered(q_raw, q_norm_w, cfg.norm_eps)
    k = gdn.rms_norm_zero_centered(k_raw, k_norm_w, cfg.norm_eps)
    q = apply_partial_rope(
        q,
        positions,
        cfg.rotary_dim,
        cfg.rope_max_timescale,
        rope_cos_sin=rope_cos_sin,
    )
    k = apply_partial_rope(
        k,
        positions,
        cfg.rotary_dim,
        cfg.rope_max_timescale,
        rope_cos_sin=rope_cos_sin,
    )
    scale = cfg.head_dim**-0.5
    q_scaled = (q.astype(jnp.float32) * scale).astype(x_in.dtype)
    gate_3d = gate.reshape(bsz, seq_len, cfg.attn_q_out_dim)
    if cfg.use_pallas_kernels and cfg.attn_gate_bf16_align:
      gate_3d = gdn.forward_layout_constraint(gate_3d, (2, 0, 1))
    return q_scaled, k, v, gate_3d

  def _post_splash(attn_o, gate_3d, out_proj_w):
    if cfg.attn_bf16_out_fused_gate:
      gated = _attn_sigmoid_gate(
          attn_o.reshape(bsz, seq_len, cfg.attn_q_out_dim), gate_3d
      )
    else:
      gated = (
          attn_o.reshape(bsz, seq_len, cfg.attn_q_out_dim).astype(jnp.float32)
          * jax.nn.sigmoid(gate_3d.astype(jnp.float32))
      ).astype(x.dtype)
    return gdn.dense_linear(gated, out_proj_w, use_fp8=use_fp8)

  (_, _, _, gate_3d), vjp_pre = jax.vjp(
      _pre_splash_and_gate,
      x,
      w.pre_attn_norm,
      w.q_proj,
      w.k_proj,
      w.v_proj,
      w.q_norm,
      w.k_norm,
  )
  _, vjp_post = jax.vjp(_post_splash, attn_out, gate_3d, w.out_proj)
  g_attn_out, g_gate_3d, g_out_proj = vjp_post(g_y)
  g_q_scaled, g_k, g_v = vjp_splash(g_attn_out)
  (
      g_x_pre,
      g_pre_norm,
      g_q_proj,
      g_k_proj,
      g_v_proj,
      g_q_norm,
      g_k_norm,
  ) = vjp_pre((g_q_scaled, g_k, g_v, g_gate_3d))
  return (
      g_y + g_x_pre,
      AttnWeights(
          pre_attn_norm=g_pre_norm,
          q_proj=g_q_proj,
          k_proj=g_k_proj,
          v_proj=g_v_proj,
          q_norm=g_q_norm,
          k_norm=g_k_norm,
          out_proj=g_out_proj,
      ),
  )


__all__ = [
    "ATTN_SAVED_NAMES",
    "AttnWeightCarrier",
    "AttnWeights",
    "apply_partial_rope",
    "compute_rope_cos_sin",
    "full_attn_mixer",
    "full_attn_mixer_bwd_from_res",
    "full_attn_mixer_fwd_save_res",
]

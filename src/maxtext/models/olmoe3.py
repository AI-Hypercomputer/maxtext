# Copyright 2023-2026 Google LLC
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

"""OLMoE3 decoder layer (AI2's hybrid KDA + latent-MoE architecture).

Reference: ``allenai/OLMo-core@akshitab/standalone``,
``src/scripts/standalone/standalone_model.py``.

Layer structure, matching the reference block:

* Mixer alternates Kimi Delta Attention (KDA, linear attention) with full
  attention. Full-attention layers are those where
  ``(layer_idx + 1) % inhomogeneous_layer_cycle_interval == 0``, which at
  interval 5 reproduces the reference's ``range(4, n_layers, 5)``.
* Four RMSNorms per block (pre and post, around both the mixer and the FFN).
  The post-norm is applied to the sublayer output *before* the residual add.
* FFN is a shared SwiGLU plus, on all layers past ``first_num_dense_layers``,
  routed experts that run entirely in a compressed latent space: the block
  projects ``emb_dim -> moe_expert_input_dim``, runs the experts there, and
  projects back. The router still scores the full-width residual.

Routing matches the reference: top-k weights renormalized to sum to ``top_k``
(``restore_weight_scale``), a batch-level globally-balanced load-balancing
loss, a 1e-5 router z-loss, and the EMo document-pool router (``emo_enabled``,
with the canonical pools min=top_k, max=eval=num_experts).
"""

import functools
from typing import Any

from jax.sharding import Mesh, PartitionSpec
import jax
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx
from jax.ad_checkpoint import checkpoint_name

from maxtext.common.common_types import Config, BATCH, LENGTH, EMBED
from maxtext.layers import attentions
from maxtext.layers import initializers as max_initializers
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.linears import DenseGeneral, MlpBlock
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils

# The reference uses a separate, looser epsilon for the KDA output norm than for
# the block RMSNorms (kda_norm_eps=1e-5 vs rms_norm_eps=1e-6). Kept as a module
# constant rather than a config key until a model needs to vary it.
_KDA_NORM_EPS = 1e-5

# Reference init std and router z-loss weight (fixed across the ladder).
_INIT_STD = 0.02
_ROUTER_Z_LOSS_WEIGHT = 1e-5


def olmoe3_init(key, shape, dtype=jnp.float32, in_axis=None, out_axis=None):
  """The reference init: truncated normal, std ``_INIT_STD``, bounds +/-3 std.

  Every embedding, projection, router, and expert weight in the reference uses
  this fixed-std init (fan-in plays no part). The trailing axis arguments are
  accepted and ignored so the same function satisfies both MaxText initializer
  signatures (``NdInitializer`` and plain ``Initializer``).
  """
  del in_axis, out_axis
  return jax.nn.initializers.truncated_normal(stddev=_INIT_STD, lower=-3.0, upper=3.0)(key, shape, dtype)


def _dense(config, in_features, out_features, kernel_axes, quant, rngs, use_bias=False):
  """DenseGeneral with the config-derived arguments this model always passes."""
  return DenseGeneral(
      in_features_shape=in_features,
      out_features_shape=out_features,
      axis=-1,
      kernel_init=olmoe3_init,
      kernel_axes=kernel_axes,
      dtype=config.dtype,
      weight_dtype=config.weight_dtype,
      quant=quant,
      shard_mode=config.shard_mode,
      matmul_precision=config.matmul_precision,
      use_bias=use_bias,
      rngs=rngs,
  )


def causal_depthwise_conv(x: jnp.ndarray, weight: jnp.ndarray, segment_ids: None | jnp.ndarray) -> jnp.ndarray:
  """Short causal depthwise convolution over time, followed by SiLU.

  Implemented as a sum of lagged copies rather than a conv op so that packed
  documents can be masked per lag: a token never convolves with tokens from a
  preceding document. This is the unfused equivalent of passing ``cu_seqlens``
  to OLMo-core's packed-document convolution kernel.

  Args:
    x: Input of shape ``[batch, length, width]``.
    weight: Depthwise kernel of shape ``[width, kernel_size]``.
    segment_ids: Optional ``[batch, length]`` packing segment ids.

  Returns:
    ``[batch, length, width]``, SiLU applied.
  """
  seq_len = x.shape[1]
  kernel_size = weight.shape[-1]
  acc = x * weight[:, kernel_size - 1]
  if kernel_size > 1:
    x_pad = jnp.pad(x, ((0, 0), (kernel_size - 1, 0), (0, 0)))
    ids_pad = (
        jnp.pad(segment_ids, ((0, 0), (kernel_size - 1, 0)), constant_values=-1)
        if segment_ids is not None
        else None
    )
    for tap in range(kernel_size - 1):
      shifted = x_pad[:, tap : tap + seq_len, :]
      if ids_pad is not None:
        shifted_ids = ids_pad[:, tap : tap + seq_len]
        shifted = jnp.where((shifted_ids == segment_ids)[..., None], shifted, 0.0)
      acc = acc + shifted * weight[:, tap]
  return jax.nn.silu(acc)


class OLMoE3KimiDeltaAttention(nnx.Module):
  """Kimi Delta Attention: gated delta rule with per-channel (vector) decay.

  This differs from MaxText's existing GatedDeltaNet in the decay term. GDN
  learns one decay scalar per head (``A_log`` shaped ``[num_heads]``); KDA
  produces a decay per key channel from a low-rank projection, shaped
  ``[batch, length, heads, key_head_dim]``. With scalar decay the intra-chunk
  term collapses to a mask on ``K K^T``; per channel it does not, but it still
  factors once the cumulative decay is folded into the operands, which is what
  ``_delta_rule_chunked`` does.

  ``_delta_rule_scan`` is the unfused reference. It is kept because the chunked
  rule is checked against it, and because it handles sequence lengths that are
  not a multiple of ``gdn_chunk_size``.
  """

  def __init__(self, config: Config, mesh: Mesh, quant: None | Quant = None, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    cfg = config

    self.num_heads = cfg.gdn_num_value_heads
    self.head_k_dim = cfg.gdn_key_head_dim
    self.head_v_dim = cfg.gdn_value_head_dim
    key_width = self.num_heads * self.head_k_dim
    value_width = self.num_heads * self.head_v_dim
    conv_size = cfg.gdn_conv_kernel_dim

    self.w_q = _dense(cfg, cfg.emb_dim, key_width, ("embed", "mlp"), quant, rngs)
    self.w_k = _dense(cfg, cfg.emb_dim, key_width, ("embed", "mlp"), quant, rngs)
    self.w_v = _dense(cfg, cfg.emb_dim, value_width, ("embed", "mlp"), quant, rngs)

    conv_init = jax.nn.initializers.truncated_normal(stddev=_INIT_STD, lower=-3.0, upper=3.0)
    self.q_conv = nnx.Param(conv_init(rngs.params(), (key_width, conv_size), cfg.weight_dtype))
    self.k_conv = nnx.Param(conv_init(rngs.params(), (key_width, conv_size), cfg.weight_dtype))
    self.v_conv = nnx.Param(conv_init(rngs.params(), (value_width, conv_size), cfg.weight_dtype))

    # Low-rank decay projection: emb -> head_v_dim -> key_width.
    self.f_proj_1 = _dense(cfg, cfg.emb_dim, self.head_v_dim, ("embed", "mlp"), quant, rngs)
    self.f_proj_2 = _dense(cfg, self.head_v_dim, key_width, ("mlp", "embed"), quant, rngs)
    self.w_b = _dense(cfg, cfg.emb_dim, self.num_heads, ("embed", None), quant, rngs)

    a_log_init = nnx.initializers.uniform(scale=15.0)  # U[1, 16) after the +1 below
    self.A_log = nnx.Param(jnp.log(1.0 + a_log_init(rngs.params(), (self.num_heads,), jnp.float32)))
    self.dt_bias = nnx.Param(jnp.zeros((key_width,), jnp.float32))

    # Low-rank output gate: emb -> head_v_dim -> value_width (this one has a bias).
    self.g_proj_1 = _dense(cfg, cfg.emb_dim, self.head_v_dim, ("embed", "mlp"), quant, rngs)
    self.g_proj_2 = _dense(cfg, self.head_v_dim, value_width, ("mlp", "embed"), quant, rngs, use_bias=True)

    self.o_norm = RMSNorm(
        num_features=self.head_v_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("norm",),
        epsilon=_KDA_NORM_EPS,
        rngs=rngs,
    )
    self.w_out = _dense(cfg, value_width, cfg.emb_dim, ("mlp", "embed"), quant, rngs)

  def __call__(self, x: jnp.ndarray, decoder_segment_ids: None | jnp.ndarray = None) -> jnp.ndarray:
    batch, seq_len, _ = x.shape
    heads, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
    key_width = heads * dk
    value_width = heads * dv

    # An fp32 conv weight otherwise promotes q/k/v (and their kernel residuals) to fp32.
    cast = self.config.kda_conv_in_compute_dtype
    q_conv, k_conv, v_conv = (w[...].astype(x.dtype) if cast else w[...] for w in (self.q_conv, self.k_conv, self.v_conv))
    q_raw, k_raw, v_raw = self.w_q(x), self.w_k(x), self.w_v(x)
    f_1, g_1, b_raw = self.f_proj_1(x), self.g_proj_1(x), self.w_b(x)
    qkv_raw = checkpoint_name(jnp.concatenate([q_raw, k_raw, v_raw], axis=-1), "qkv_proj")
    f_1 = checkpoint_name(f_1, "qkv_proj")
    g_1 = checkpoint_name(g_1, "qkv_proj")
    b_raw = checkpoint_name(b_raw, "qkv_proj")
    qkv_conv = jnp.concatenate([q_conv, k_conv, v_conv], axis=0)
    qkv = causal_depthwise_conv(qkv_raw, qkv_conv, decoder_segment_ids)
    q = qkv[..., :key_width].reshape(batch, seq_len, heads, dk)
    k = qkv[..., key_width : 2 * key_width].reshape(batch, seq_len, heads, dk)
    v = qkv[..., 2 * key_width :].reshape(batch, seq_len, heads, dv)

    raw_g = checkpoint_name(self.f_proj_2(f_1).reshape(batch, seq_len, heads, dk), "qkv_proj")
    # Reference uses allow_neg_eigval (beta in [0, 2)); False clamps to [0, 1].
    beta_scale = 2.0 if self.config.kda_allow_neg_eigval else 1.0
    beta = beta_scale * jax.nn.sigmoid(b_raw.astype(jnp.float32))

    if self.config.use_tokamax_kda:
      out = self._tokamax_kda(q, k, v, raw_g, beta, decoder_segment_ids)
    else:
      # Unfused path: L2-norm and query scale explicit (the kernel does them internally).
      q = _l2_normalize(q.astype(jnp.float32), scale=dk**-0.5)
      k = _l2_normalize(k.astype(jnp.float32))

      dt = self.dt_bias[...].reshape(1, 1, heads, dk)
      log_decay = -jnp.exp(self.A_log[...]).reshape(1, 1, heads, 1) * jax.nn.softplus(raw_g.astype(jnp.float32) + dt)

      if decoder_segment_ids is None:
        resets = jnp.zeros((batch, seq_len), dtype=bool)
      else:
        prev_ids = jnp.pad(decoder_segment_ids, ((0, 0), (1, 0)), constant_values=-1)[:, :seq_len]
        resets = decoder_segment_ids != prev_ids

      chunk = self.config.gdn_chunk_size
      if chunk > 0 and seq_len % chunk == 0:
        state_dt = jnp.bfloat16 if self.config.gdn_state_dtype == "bfloat16" else jnp.float32
        out = _delta_rule_chunked(
            q.astype(state_dt),
            k.astype(state_dt),
            v.astype(state_dt),
            log_decay,
            beta,
            resets,
            chunk,
            self.config.gdn_state_dtype,
        )
      else:
        out = _delta_rule_scan(q, k, v.astype(jnp.float32), jnp.exp(log_decay), beta, resets)

    gate_logits = checkpoint_name(self.g_proj_2(g_1).reshape(batch, seq_len, heads, dv), "qkv_proj")
    gate = jax.nn.sigmoid(gate_logits)
    out = checkpoint_name((self.o_norm(out.astype(x.dtype)) * gate).reshape(batch, seq_len, heads * dv), "context")
    return checkpoint_name(self.w_out(out), "out_proj")

  def _tokamax_kda(self, q, k, v, raw_g, beta, decoder_segment_ids) -> jnp.ndarray:
    """Fused KDA via tokamax (PR #1103, experimental). Raw q/k/v/gate in: the

    kernel does the q/k L2-norm, K**-0.5 scale, and decay activation. beta in
    [0, 2) matches the reference; parity checked against _delta_rule_scan.
    """
    # pylint: disable=g-import-not-at-top,import-outside-toplevel
    # Lazy import: releases before the KDA PR lack this module.
    from tokamax._src.ops.experimental.kda import api as kda_api

    # Pallas kernels can't be auto-partitioned, so run under shard_map with only
    # the batch axis split (sequence and heads stay whole; packing preserved).
    batch_axes = nn.logical_to_mesh_axes(("activation_batch",), self.config.logical_axis_rules)[0]
    # Segments only for packed data; the varlen path pads to seq + 63*max_segments.
    has_segments = decoder_segment_ids is not None and bool(self.config.packing)
    in_specs = [PartitionSpec(batch_axes, None, None, None)] * 4 + [PartitionSpec(batch_axes, None, None)]
    if has_segments:
      in_specs.append(PartitionSpec(batch_axes, None))
    in_specs += [PartitionSpec(), PartitionSpec()]
    max_num_segments = int(self.config.tokamax_kda_max_num_segments)
    decay_floor = float(self.config.tokamax_kda_log_decay_floor)
    l2norm_outside = bool(self.config.tokamax_kda_l2norm_outside)

    @functools.partial(
        jax.shard_map,
        mesh=self.mesh,
        in_specs=tuple(in_specs),
        out_specs=PartitionSpec(batch_axes, None, None, None),
        check_vma=False,
    )
    def call_kernel(q_, k_, v_, g_, beta_, *rest):
      def head_first(t):
        return t.transpose(2, 0, 1, 3)

      seg_kwargs = {}
      if has_segments:
        # tokamax expects 1-indexed segment ids with 0 reserved for padding,
        # which is MaxText's packed-data convention already.
        seg_kwargs = {"segment_ids": rest[0], "max_num_segments": max_num_segments}
      a_log_, dt_bias_ = rest[-2], rest[-1]
      g_hf = head_first(g_)
      if decay_floor > 0:
        # With the gate activated in-kernel the kernel skips its safe-gate centering and
        # overflows fp32 exp once a step decays past ~30 nats (NaN). Activate here and floor.
        heads, dk = g_hf.shape[0], g_hf.shape[-1]
        log_decay = -jnp.exp(a_log_).reshape(heads, 1, 1, 1) * jax.nn.softplus(
            g_hf.astype(jnp.float32) + dt_bias_.reshape(heads, 1, 1, dk)
        )
        gate_kwargs = {"use_gate_in_kernel": False}
        g_hf = jnp.maximum(log_decay, -decay_floor)
      else:
        gate_kwargs = {"use_gate_in_kernel": True, "a_log": a_log_, "delta_time_bias": dt_bias_}
      if l2norm_outside:
        q_, k_ = _kernel_l2norm(q_), _kernel_l2norm(k_)
      out, _ = kda_api.kimi_delta_attention(
          head_first(q_),
          head_first(k_),
          head_first(v_),
          g_hf,
          beta_.transpose(2, 0, 1),
          use_qk_l2norm=not l2norm_outside,
          **gate_kwargs,
          **seg_kwargs,
      )
      return out.transpose(1, 2, 0, 3)

    args = [q, k, v, raw_g, beta]
    if has_segments:
      args.append(decoder_segment_ids.astype(jnp.int32))
    args += [self.A_log[...], self.dt_bias[...]]
    return call_kernel(*args)


def _kernel_l2norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
  """The tokamax KDA kernel's q/k L2-norm (fp32 math, eps inside the rsqrt, input dtype out)."""
  x_f = x.astype(jnp.float32)
  return (x_f * jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def _l2_normalize(x: jnp.ndarray, eps: float = 1e-12, scale: float = 1.0) -> jnp.ndarray:
  inv_norm = jax.lax.rsqrt(jnp.maximum(jnp.sum(x * x, axis=-1, keepdims=True), eps))
  if scale != 1.0:
    inv_norm = inv_norm * scale
  return x * inv_norm


def _invert_block_unit_lower_impl(
    m_diag: jnp.ndarray,
    m_full: None | jnp.ndarray = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
) -> jnp.ndarray:
  """Inverts unit lower-triangular (I + M) via 16x16 MXU power series + 2x2 block doubling."""
  b_sub = m_diag.shape[-1]
  eye = jnp.eye(b_sub, dtype=m_diag.dtype)
  m2 = jnp.matmul(m_diag, m_diag, precision=precision, preferred_element_type=jnp.float32).astype(m_diag.dtype)
  p2 = jnp.matmul(eye - m_diag, eye + m2, precision=precision, preferred_element_type=jnp.float32).astype(m_diag.dtype)
  if b_sub <= 4:
    a_blk = p2
  else:
    m4 = jnp.matmul(m2, m2, precision=precision, preferred_element_type=jnp.float32).astype(m_diag.dtype)
    p4 = jnp.matmul(p2, eye + m4, precision=precision, preferred_element_type=jnp.float32).astype(m_diag.dtype)
    if b_sub <= 8:
      a_blk = p4
    else:
      m8 = jnp.matmul(m4, m4, precision=precision, preferred_element_type=jnp.float32).astype(m_diag.dtype)
      a_blk = jnp.matmul(p4, eye + m8, precision=precision, preferred_element_type=jnp.float32).astype(m_diag.dtype)

  while a_blk.shape[-3] > 1:
    assert m_full is not None
    k_curr = a_blk.shape[-3]
    b_curr = a_blk.shape[-1]
    a_00 = a_blk[..., 0::2, :, :]
    a_11 = a_blk[..., 1::2, :, :]
    m_10 = jnp.stack(
        [
            m_full[
                ...,
                (2 * i + 1) * b_curr : (2 * i + 2) * b_curr,
                (2 * i) * b_curr : (2 * i + 1) * b_curr,
            ]
            for i in range(k_curr // 2)
        ],
        axis=-3,
    )
    tmp = jnp.matmul(a_11, m_10, precision=precision, preferred_element_type=jnp.float32).astype(m_diag.dtype)
    a_10 = -jnp.matmul(tmp, a_00, precision=precision, preferred_element_type=jnp.float32).astype(m_diag.dtype)
    top = jnp.concatenate([a_00, jnp.zeros_like(a_00)], axis=-1)
    bot = jnp.concatenate([a_10, a_11], axis=-1)
    a_blk = jnp.concatenate([top, bot], axis=-2)

  return a_blk[..., 0, :, :]


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _invert_block_unit_lower(
    m_diag: jnp.ndarray,
    m_full: None | jnp.ndarray = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
) -> jnp.ndarray:
  """Inverts unit lower-triangular (I + M) with custom VJP dM = tril(-A^T dA A^T, -1)."""
  return _invert_block_unit_lower_impl(m_diag, m_full, precision)


def _invert_block_unit_lower_fwd(
    m_diag: jnp.ndarray,
    m_full: None | jnp.ndarray,
    precision: jax.lax.Precision,
):
  a = checkpoint_name(_invert_block_unit_lower_impl(m_diag, m_full, precision), "context")
  return a, (a, m_diag.shape[-3], m_full is not None)


def _invert_block_unit_lower_bwd(
    precision: jax.lax.Precision,
    res,
    da: jnp.ndarray,
):
  a, num_sub, has_m_full = res
  c = a.shape[-1]
  b_sub = c // num_sub
  a_t = a.swapaxes(-1, -2)
  da_a_t = jnp.matmul(
      da.astype(a.dtype), a_t, precision=precision, preferred_element_type=jnp.float32
  ).astype(a.dtype)
  dm = -jnp.tril(
      jnp.matmul(a_t, da_a_t, precision=precision, preferred_element_type=jnp.float32).astype(a.dtype),
      k=-1,
  )
  if not has_m_full:
    return (dm[..., None, :, :], None)
  dm_diag = jnp.stack(
      [dm[..., i * b_sub : (i + 1) * b_sub, i * b_sub : (i + 1) * b_sub] for i in range(num_sub)],
      axis=-3,
  )
  strict_blk = jnp.tril(jnp.ones((num_sub, num_sub), dtype=bool), k=-1)[..., :, None, :, None]
  dm_full = jnp.where(
      strict_blk,
      dm.reshape(*dm.shape[:-2], num_sub, b_sub, num_sub, b_sub),
      0.0,
  ).reshape(dm.shape)
  return (dm_diag, dm_full)


_invert_block_unit_lower.defvjp(_invert_block_unit_lower_fwd, _invert_block_unit_lower_bwd)


def _kda_subblock_bilinear_core(
    q_i: jnp.ndarray,
    k_row_i: jnp.ndarray,
    k_col_i: jnp.ndarray,
    rel: jnp.ndarray,
    seg: jnp.ndarray,
    beta_i: jnp.ndarray,
    c: int,
    compute_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, None | jnp.ndarray, jnp.ndarray]:
  """Evaluates M_diag, M_full, and scores as a bilinear function of (q_i, k_row_i) and k_col_i."""
  lead_bh = q_i.shape[:-2]
  lead_seg = seg.shape[:-1]
  dk = q_i.shape[-1]
  b_sub = 16 if (c >= 32 and c % 16 == 0 and (c & (c - 1)) == 0) else c
  num_sub = c // b_sub
  dot_prec = jax.lax.Precision.DEFAULT if compute_dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST

  q_blk = q_i.reshape(*lead_bh, num_sub, b_sub, dk)
  k_row_blk = k_row_i.reshape(*lead_bh, num_sub, b_sub, dk)
  k_col_blk = k_col_i.reshape(*lead_bh, num_sub, b_sub, dk)
  rel_blk = jax.lax.stop_gradient(rel).astype(compute_dtype).reshape(*lead_bh, num_sub, b_sub, dk)
  seg_blk = jax.lax.stop_gradient(seg).reshape(*lead_seg, num_sub, b_sub)
  beta_blk = beta_i.astype(compute_dtype).reshape(*lead_bh, num_sub, b_sub)

  causal_sub = jnp.tril(jnp.ones((b_sub, b_sub), dtype=bool))
  strict_sub = jnp.tril(jnp.ones((b_sub, b_sub), dtype=bool), k=-1)
  same_doc_diag = (seg_blk[..., :, None] == seg_blk[..., None, :])[..., None, :, :, :]
  pair_rel_diag = rel_blk[..., :, None, :] - rel_blk[..., None, :, :]
  diag_mask = (causal_sub & same_doc_diag)[..., None]
  pair_decay_diag = jnp.exp(jnp.where(diag_mask, pair_rel_diag, -jnp.inf)).astype(compute_dtype)

  qk_blk = jnp.stack([k_row_blk, q_blk], axis=-3)
  k_dec = k_col_blk[..., None, :, :] * pair_decay_diag
  ms_raw = jnp.einsum(
      "...rsid,...rijd->...srij",
      qk_blk,
      k_dec,
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  ).astype(compute_dtype)
  m_diag = jnp.where(strict_sub & same_doc_diag, ms_raw[..., 0, :, :, :], 0.0) * beta_blk[..., None]
  scores_diag = jnp.where(causal_sub & same_doc_diag, ms_raw[..., 1, :, :, :], 0.0).astype(compute_dtype)

  if num_sub == 1:
    return m_diag, None, scores_diag[..., 0, :, :]

  rel_piv_r = rel_blk[..., :1, :]
  seg_piv_r = seg_blk[..., :1]
  same_left = (seg_blk == seg_piv_r)[..., None, :, :, None]
  alpha_left = jnp.exp(jnp.where(same_left, rel_blk - rel_piv_r, -jnp.inf)).astype(compute_dtype)

  strict_blk = jnp.tril(jnp.ones((num_sub, num_sub), dtype=bool), k=-1)
  same_right = (seg_piv_r[..., :, None, :] == seg_blk[..., None, :, :])[..., None, :, :, :, None]
  right_mask = strict_blk[..., :, :, None, None] & same_right
  rel_right = rel_piv_r[..., :, None, :, :] - rel_blk[..., None, :, :, :]
  alpha_right = jnp.exp(jnp.where(right_mask, rel_right, -jnp.inf)).astype(compute_dtype)

  k_hat = (k_col_blk[..., None, :, :, :] * alpha_right).reshape(*lead_bh, num_sub, c, dk)
  qk_tilde = jnp.concatenate([k_row_blk * alpha_left, q_blk * alpha_left], axis=-2)
  ms_off = (
      jnp.matmul(qk_tilde, k_hat.swapaxes(-1, -2), precision=dot_prec, preferred_element_type=jnp.float32)
      .astype(compute_dtype)
      .reshape(*lead_bh, num_sub, 2 * b_sub, num_sub, b_sub)
      .swapaxes(-2, -3)
  )
  ms_off = jnp.where(strict_blk[..., :, :, None, None], ms_off, 0.0)
  m_off = ms_off[..., :b_sub, :] * beta_blk[..., :, None, :, None]
  scores_off = ms_off[..., b_sub:, :].astype(compute_dtype)

  m_full = m_off.swapaxes(-2, -3).reshape(*lead_bh, c, c)
  eye_k = jnp.eye(num_sub, dtype=compute_dtype)[..., :, :, None, None]
  scores = (scores_off + scores_diag[..., :, None, :, :] * eye_k).swapaxes(-2, -3).reshape(*lead_bh, c, c)
  return m_diag, m_full, scores


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _compute_kda_subblock_matrices(
    q_i: jnp.ndarray,
    k_i: jnp.ndarray,
    rel: jnp.ndarray,
    seg: jnp.ndarray,
    beta_i: jnp.ndarray,
    c: int,
    compute_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, None | jnp.ndarray, jnp.ndarray]:
  """Builds strictly lower-triangular M blocks and intra-chunk scores using 16-token sub-block pivot factorization."""
  return _kda_subblock_bilinear_core(q_i, k_i, k_i, rel, seg, beta_i, c, compute_dtype)


def _compute_kda_subblock_matrices_fwd(
    q_i: jnp.ndarray,
    k_i: jnp.ndarray,
    rel: jnp.ndarray,
    seg: jnp.ndarray,
    beta_i: jnp.ndarray,
    c: int,
    compute_dtype: jnp.dtype,
):
  out = _kda_subblock_bilinear_core(q_i, k_i, k_i, rel, seg, beta_i, c, compute_dtype)
  return out, (q_i, k_i, rel, seg, beta_i)


def _compute_kda_subblock_matrices_bwd(
    c: int,
    compute_dtype: jnp.dtype,
    res,
    cts,
):
  q_i, k_i, rel, seg, beta_i = res
  _, vjp_fn = jax.vjp(
      lambda q_, kr_, kc_, b_: _kda_subblock_bilinear_core(
          q_, kr_, kc_, rel, seg, b_, c, compute_dtype
      ),
      q_i,
      k_i,
      k_i,
      beta_i,
  )
  dq_i, dk_row_i, dk_col_i, dbeta_i = vjp_fn(cts)
  dk_i = (dk_row_i + dk_col_i).astype(k_i.dtype)
  # Exact closed-form gauge-invariance adjoint for per-channel relative log-decay `rel`:
  # Every term in M and S scales q_i and k_row_i by exp(+rel_i) and k_col_j by exp(-rel_j),
  # and the off-diagonal sub-block pivot rel_{r,0} cancels identically.
  drel = (
      q_i.astype(jnp.float32) * dq_i.astype(jnp.float32)
      + k_i.astype(jnp.float32) * (dk_row_i.astype(jnp.float32) - dk_col_i.astype(jnp.float32))
  ).astype(rel.dtype)
  return (dq_i.astype(q_i.dtype), dk_i, drel, None, dbeta_i.astype(beta_i.dtype))


_compute_kda_subblock_matrices.defvjp(
    _compute_kda_subblock_matrices_fwd,
    _compute_kda_subblock_matrices_bwd,
)


def _compute_kda_subblock_inv_and_scores(
    q_i: jnp.ndarray,
    k_i: jnp.ndarray,
    rel: jnp.ndarray,
    seg: jnp.ndarray,
    beta_i: jnp.ndarray,
    c: int,
    compute_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes (I + M)^-1 and intra-chunk scores using 16-token sub-block pivot factorization on the MXU."""
  dot_prec = jax.lax.Precision.DEFAULT if compute_dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST
  m_diag, m_full, scores = _compute_kda_subblock_matrices(q_i, k_i, rel, seg, beta_i, c, compute_dtype)
  m_inv = _invert_block_unit_lower(m_diag, m_full, dot_prec)
  return m_inv, checkpoint_name(scores, "context")


def _delta_rule_chunked_baseline(
    q, k, v, log_decay, beta, resets, chunk_size: int, state_dtype: str = "float32"
) -> jnp.ndarray:
  """Baseline chunked delta rule (full 5D pair_decay + 5-step 64x64 Newton-Schulz inside lax.scan)."""
  batch, seq_len, heads, dk = q.shape
  dv = v.shape[-1]
  num_chunks = seq_len // chunk_size
  c = chunk_size
  compute_dtype = jnp.bfloat16 if state_dtype == "bfloat16" else jnp.float32

  def to_chunks(x):
    return x.reshape(batch, num_chunks, c, heads, -1).transpose(1, 0, 3, 2, 4)

  q_c = to_chunks(q)
  k_c = to_chunks(k)
  v_c = to_chunks(v)
  log_a_c = to_chunks(log_decay)
  beta_c = beta.reshape(batch, num_chunks, c, heads).transpose(1, 0, 3, 2)
  reset_c = resets.reshape(batch, num_chunks, c).transpose(1, 0, 2)

  eye = jnp.eye(c, dtype=q.dtype)
  causal = jnp.tril(jnp.ones((c, c), dtype=bool))
  strict = jnp.tril(jnp.ones((c, c), dtype=bool), k=-1)
  positions = jnp.arange(c)

  def body(state, xs):
    q_i, k_i, v_i, log_a, beta_i, reset_i = xs
    seg = jnp.cumsum(reset_i.astype(jnp.int32), axis=-1)
    seg_start = jax.lax.cummax(jnp.where(reset_i, positions, 0), axis=1)

    log_cum = jnp.cumsum(log_a, axis=-2)
    gather = seg_start[:, None, :, None]
    base = jnp.take_along_axis(log_cum, gather, axis=-2) - jnp.take_along_axis(log_a, gather, axis=-2)
    rel = log_cum - base
    cum = jnp.exp(rel)
    q_i = q_i.astype(compute_dtype)
    k_i = k_i.astype(compute_dtype)
    v_i = v_i.astype(compute_dtype)

    same_doc = seg[:, None, :, None] == seg[:, None, None, :]
    from_prev = (seg[:, None, :, None] == 0).astype(v_i.dtype)

    pair_rel = rel[..., :, None, :] - rel[..., None, :, :]
    pair_mask = (causal[None, None] & same_doc)[..., None]
    pair_decay = jnp.exp(jnp.where(pair_mask, pair_rel, -jnp.inf)).astype(compute_dtype)

    m = jnp.einsum("bhid,bhjd,bhijd->bhij", k_i, k_i, pair_decay)
    m = jnp.where(strict & same_doc, m, 0.0) * beta_i[..., None]
    a_mat = eye + m
    m_inv = eye - m
    for _ in range(max(0, (c - 1).bit_length() - 1)):
      m_inv = m_inv @ (2.0 * eye - a_mat @ m_inv)
    carried = jnp.einsum("bhid,bhdv->bhiv", k_i * cum.astype(compute_dtype), state) * from_prev
    delta = jnp.einsum(
        "bhij,bhjv->bhiv", m_inv.astype(compute_dtype), (v_i - carried) * beta_i[..., None].astype(compute_dtype)
    )

    scores = jnp.einsum("bhid,bhjd,bhijd->bhij", q_i, k_i, pair_decay)
    scores = jnp.where(causal & same_doc, scores, 0.0).astype(compute_dtype)
    out = jnp.einsum("bhid,bhdv->bhiv", q_i * cum.astype(compute_dtype), state) * from_prev
    out = out + jnp.einsum("bhij,bhjv->bhiv", scores, delta)

    any_reset = (seg[:, -1] > 0)[:, None, None, None]
    decayed = state * cum[..., -1, :][..., None].astype(compute_dtype)
    state = jnp.where(any_reset, jnp.zeros_like(decayed), decayed)
    last_seg = (seg == seg[:, -1:])[:, None, :, None]
    k_carry = k_i * jnp.exp(jnp.where(last_seg, rel[..., -1:, :] - rel, -jnp.inf)).astype(compute_dtype)
    state = (state + jnp.einsum("bhid,bhiv->bhdv", k_carry, delta)).astype(compute_dtype)
    return state, out.astype(jnp.float32)

  init_state = jnp.zeros((batch, heads, dk, dv), compute_dtype)
  body_remat = jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable)
  _, outputs = jax.lax.scan(body_remat, init_state, (q_c, k_c, v_c, log_a_c, beta_c, reset_c))
  return outputs.transpose(1, 0, 3, 2, 4).reshape(batch, seq_len, heads, dv)


def _delta_rule_chunked_inscan_opt(
    q, k, v, log_decay, beta, resets, chunk_size: int, state_dtype: str = "float32"
) -> jnp.ndarray:
  """In-scan optimized chunked delta rule (16-token sub-block pivot MXU factorization + 16->32->64 block inversion)."""
  batch, seq_len, heads, dk = q.shape
  dv = v.shape[-1]
  num_chunks = seq_len // chunk_size
  c = chunk_size
  compute_dtype = jnp.bfloat16 if state_dtype == "bfloat16" else jnp.float32

  def to_chunks(x):
    return x.reshape(batch, num_chunks, c, heads, -1).transpose(1, 0, 3, 2, 4)

  q_c = to_chunks(q.astype(compute_dtype))
  k_c = to_chunks(k.astype(compute_dtype))
  v_c = to_chunks(v.astype(compute_dtype))
  log_a_c = to_chunks(log_decay)
  beta_c = beta.reshape(batch, num_chunks, c, heads).transpose(1, 0, 3, 2)
  reset_c = resets.reshape(batch, num_chunks, c).transpose(1, 0, 2)

  positions = jnp.arange(c)

  def body(state, xs):
    q_i, k_i, v_i, log_a, beta_i, reset_i = xs
    seg = jnp.cumsum(reset_i.astype(jnp.int32), axis=-1)
    seg_start = jax.lax.cummax(jnp.where(reset_i, positions, 0), axis=1)

    log_cum = jnp.cumsum(log_a, axis=-2)
    gather = seg_start[:, None, :, None]
    base = jnp.take_along_axis(log_cum, gather, axis=-2) - jnp.take_along_axis(log_a, gather, axis=-2)
    rel = log_cum - base
    from_prev_mask = seg[:, None, :, None] == 0
    cum_prev = jnp.exp(jnp.where(from_prev_mask, rel, -jnp.inf)).astype(compute_dtype)
    q_i = q_i.astype(compute_dtype)
    k_i = k_i.astype(compute_dtype)
    v_i = v_i.astype(compute_dtype)

    m_inv, scores = _compute_kda_subblock_inv_and_scores(q_i, k_i, rel, seg, beta_i, c, compute_dtype)

    carried = jnp.matmul(k_i * cum_prev, state)
    delta = jnp.matmul(m_inv.astype(compute_dtype), (v_i - carried) * beta_i[..., None].astype(compute_dtype))
    out = jnp.matmul(q_i * cum_prev, state) + jnp.matmul(scores, delta)

    any_reset = (seg[:, -1] > 0)[:, None, None, None]
    decayed = state * jnp.exp(rel[..., -1, :])[..., None].astype(compute_dtype)
    state = jnp.where(any_reset, jnp.zeros_like(decayed), decayed)
    last_seg = (seg == seg[:, -1:])[:, None, :, None]
    k_carry = k_i * jnp.exp(jnp.where(last_seg, rel[..., -1:, :] - rel, -jnp.inf)).astype(compute_dtype)
    state = (state + jnp.matmul(k_carry.swapaxes(-1, -2), delta)).astype(compute_dtype)
    return state, out.astype(jnp.float32)

  init_state = jnp.zeros((batch, heads, dk, dv), compute_dtype)
  body_remat = jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable)
  _, outputs = jax.lax.scan(body_remat, init_state, (q_c, k_c, v_c, log_a_c, beta_c, reset_c))
  return outputs.transpose(1, 0, 3, 2, 4).reshape(batch, seq_len, heads, dv)


@functools.partial(
    jax.checkpoint,
    policy=jax.checkpoint_policies.nothing_saveable,
    static_argnums=(8,),
)
def _run_kda_wy_scan_and_readout(
    q_c: jnp.ndarray,
    k_c: jnp.ndarray,
    v_c: jnp.ndarray,
    rel: jnp.ndarray,
    seg: jnp.ndarray,
    beta_c: jnp.ndarray,
    m_inv: jnp.ndarray,
    scores_c: jnp.ndarray,
    compute_dtype: jnp.dtype,
) -> jnp.ndarray:
  """Computes WY factors, inter-chunk scan, and readout with nothing_saveable so large states are not cached."""
  dv = v_c.shape[-1]
  batch, heads, _, dk = q_c.shape[1], q_c.shape[2], q_c.shape[3], q_c.shape[4]
  dot_prec = jax.lax.Precision.DEFAULT if compute_dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST
  q_dt = q_c.astype(compute_dtype)
  k_dt = k_c.astype(compute_dtype)
  v_dt = v_c.astype(compute_dtype)
  m_inv_dt = m_inv.astype(compute_dtype)
  beta_dt = beta_c[..., None].astype(compute_dtype)
  from_prev_mask = seg[..., None, :, None] == 0
  cum_prev = jnp.exp(jnp.where(from_prev_mask, rel, -jnp.inf)).astype(compute_dtype)

  k_cum_prev = k_dt * cum_prev
  q_cum_prev = q_dt * cum_prev
  uw_all = jnp.matmul(
      m_inv_dt,
      jnp.concatenate([v_dt * beta_dt, k_cum_prev * beta_dt], axis=-1),
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  ).astype(compute_dtype)
  u_c, w_c = jnp.split(uw_all, [dv], axis=-1)

  any_reset_c = (seg[..., -1] > 0)[..., None, None, None]
  decay_end_c = jnp.exp(rel[..., -1:, :]).astype(compute_dtype).swapaxes(-1, -2)
  last_seg = (seg == seg[..., -1:])[..., None, :, None]
  k_carry_t_c = (
      k_dt * jnp.exp(jnp.where(last_seg, rel[..., -1:, :] - rel, -jnp.inf)).astype(compute_dtype)
  ).swapaxes(-1, -2)

  def scan_step(state, xs):
    w_i, u_i, k_carry_t_i, decay_end_i, any_reset_i = xs
    state_in = state
    delta_i = (
        u_i - jnp.matmul(w_i, state_in, precision=dot_prec, preferred_element_type=jnp.float32).astype(compute_dtype)
    ).astype(compute_dtype)
    decayed = state_in * decay_end_i
    state_next = jnp.where(any_reset_i, jnp.zeros_like(decayed), decayed)
    state_next = (
        state_next
        + jnp.matmul(k_carry_t_i, delta_i, precision=dot_prec, preferred_element_type=jnp.float32).astype(compute_dtype)
    ).astype(compute_dtype)
    return state_next, (state_in, delta_i)

  init_state = jnp.zeros((batch, heads, dk, dv), compute_dtype)
  _, (states_in, deltas) = jax.lax.scan(
      scan_step, init_state, (w_c, u_c, k_carry_t_c, decay_end_c, any_reset_c)
  )

  outputs = (
      jnp.matmul(q_cum_prev, states_in, precision=dot_prec, preferred_element_type=jnp.float32)
      + jnp.matmul(scores_c.astype(compute_dtype), deltas, precision=dot_prec, preferred_element_type=jnp.float32)
  ).astype(compute_dtype)
  return outputs


def _kda_wy_scan_and_readout_fwd_impl(
    q_c: jnp.ndarray,
    k_c: jnp.ndarray,
    v_c: jnp.ndarray,
    rel: jnp.ndarray,
    seg: jnp.ndarray,
    beta_c: jnp.ndarray,
    m_diag: jnp.ndarray,
    m_full: None | jnp.ndarray,
    scores_c: jnp.ndarray,
    compute_dtype: jnp.dtype,
):
  """Forward pass of KDA block inversion + WY scan + readout, returning compact `deltas` and `m_inv`."""
  dv = v_c.shape[-1]
  batch, heads, _, dk = q_c.shape[1], q_c.shape[2], q_c.shape[3], q_c.shape[4]
  dot_prec = jax.lax.Precision.DEFAULT if compute_dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST
  m_inv_dt = checkpoint_name(
      _invert_block_unit_lower_impl(m_diag, m_full, dot_prec).astype(compute_dtype),
      "context",
  )
  q_dt = q_c.astype(compute_dtype)
  k_dt = k_c.astype(compute_dtype)
  v_dt = v_c.astype(compute_dtype)
  scores_dt = scores_c.astype(compute_dtype)
  beta_dt = beta_c[..., None].astype(compute_dtype)
  from_prev_mask = seg[..., None, :, None] == 0
  cum_prev = jnp.exp(jnp.where(from_prev_mask, rel, -jnp.inf)).astype(compute_dtype)

  k_cum_prev = k_dt * cum_prev
  q_cum_prev = q_dt * cum_prev
  uw_all = jnp.matmul(
      m_inv_dt,
      jnp.concatenate([v_dt * beta_dt, k_cum_prev * beta_dt], axis=-1),
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  ).astype(compute_dtype)
  u_c, w_c = jnp.split(uw_all, [dv], axis=-1)

  any_reset_c = (seg[..., -1] > 0)[..., None, None, None]
  decay_end_c = jnp.exp(rel[..., -1:, :]).astype(compute_dtype).swapaxes(-1, -2)
  last_seg = (seg == seg[..., -1:])[..., None, :, None]
  k_carry_t_c = (
      k_dt * jnp.exp(jnp.where(last_seg, rel[..., -1:, :] - rel, -jnp.inf)).astype(compute_dtype)
  ).swapaxes(-1, -2)

  def scan_step(state, xs):
    w_i, u_i, k_carry_t_i, decay_end_i, any_reset_i = xs
    state_in = state
    delta_i = (
        u_i - jnp.matmul(w_i, state_in, precision=dot_prec, preferred_element_type=jnp.float32).astype(compute_dtype)
    ).astype(compute_dtype)
    decayed = state_in * decay_end_i
    state_next = jnp.where(any_reset_i, jnp.zeros_like(decayed), decayed)
    state_next = (
        state_next
        + jnp.matmul(k_carry_t_i, delta_i, precision=dot_prec, preferred_element_type=jnp.float32).astype(compute_dtype)
    ).astype(compute_dtype)
    return state_next, (state_in, delta_i)

  init_state = jnp.zeros((batch, heads, dk, dv), compute_dtype)
  _, (states_in, deltas) = jax.lax.scan(
      scan_step, init_state, (w_c, u_c, k_carry_t_c, decay_end_c, any_reset_c)
  )
  deltas = checkpoint_name(deltas, "context")

  outputs = (
      jnp.matmul(q_cum_prev, states_in, precision=dot_prec, preferred_element_type=jnp.float32)
      + jnp.matmul(scores_dt, deltas, precision=dot_prec, preferred_element_type=jnp.float32)
  ).astype(compute_dtype)
  return outputs, m_inv_dt, deltas


@functools.partial(jax.custom_vjp, nondiff_argnums=(9,))
def _run_kda_wy_scan_and_readout_custom(
    q_c: jnp.ndarray,
    k_c: jnp.ndarray,
    v_c: jnp.ndarray,
    rel: jnp.ndarray,
    seg: jnp.ndarray,
    beta_c: jnp.ndarray,
    m_diag: jnp.ndarray,
    m_full: None | jnp.ndarray,
    scores_c: jnp.ndarray,
    compute_dtype: jnp.dtype,
) -> jnp.ndarray:
  """Custom-VJP KDA block inversion + WY scan + readout with 1-GEMM Gram-Adjoint Collapse."""
  outputs, _, _ = _kda_wy_scan_and_readout_fwd_impl(
      q_c, k_c, v_c, rel, seg, beta_c, m_diag, m_full, scores_c, compute_dtype
  )
  return outputs


def _run_kda_wy_scan_and_readout_custom_fwd(
    q_c: jnp.ndarray,
    k_c: jnp.ndarray,
    v_c: jnp.ndarray,
    rel: jnp.ndarray,
    seg: jnp.ndarray,
    beta_c: jnp.ndarray,
    m_diag: jnp.ndarray,
    m_full: None | jnp.ndarray,
    scores_c: jnp.ndarray,
    compute_dtype: jnp.dtype,
):
  outputs, m_inv_dt, deltas = _kda_wy_scan_and_readout_fwd_impl(
      q_c, k_c, v_c, rel, seg, beta_c, m_diag, m_full, scores_c, compute_dtype
  )
  res = (
      q_c,
      k_c,
      v_c,
      rel,
      seg,
      beta_c,
      m_inv_dt,
      scores_c,
      deltas,
      m_diag.shape[-3],
      m_full is not None,
  )
  return outputs, res


def _run_kda_wy_scan_and_readout_custom_bwd(
    compute_dtype: jnp.dtype,
    res,
    do: jnp.ndarray,
):
  (
      q_c,
      k_c,
      v_c,
      rel,
      seg,
      beta_c,
      m_inv_dt,
      scores_c,
      deltas,
      num_sub,
      has_m_full,
  ) = res
  batch, heads, c, dk = q_c.shape[1], q_c.shape[2], q_c.shape[3], q_c.shape[4]
  dv = v_c.shape[-1]
  b_sub = c // num_sub
  dot_prec = jax.lax.Precision.DEFAULT if compute_dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST

  q_dt = q_c.astype(compute_dtype)
  k_dt = k_c.astype(compute_dtype)
  v_dt = v_c.astype(compute_dtype)
  scores_dt = scores_c.astype(compute_dtype)
  do_dt = do.astype(compute_dtype)
  beta_dt = beta_c[..., None].astype(compute_dtype)

  from_prev_mask = seg[..., None, :, None] == 0
  cum_prev = jnp.exp(jnp.where(from_prev_mask, rel, -jnp.inf)).astype(compute_dtype)
  k_cum_prev = k_dt * cum_prev
  q_cum_prev = q_dt * cum_prev

  any_reset_c = (seg[..., -1] > 0)[..., None, None, None]
  decay_end_c = jnp.exp(rel[..., -1:, :]).astype(compute_dtype).swapaxes(-1, -2)
  last_seg = (seg == seg[..., -1:])[..., None, :, None]
  decay_to_end = jnp.exp(jnp.where(last_seg, rel[..., -1:, :] - rel, -jnp.inf)).astype(compute_dtype)
  k_carry_c = k_dt * decay_to_end

  # 1. Reconstruct `states_in` from compact saved `deltas` via 1 parallel MXU matmul + 0-matmul scan
  state_updates = jnp.matmul(
      k_carry_c.swapaxes(-1, -2),
      deltas,
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  ).astype(compute_dtype)

  def reconstruct_state_step(state, xs):
    upd_i, decay_end_i, any_reset_i = xs
    state_in = state
    decayed = state_in * decay_end_i
    state_next = (jnp.where(any_reset_i, jnp.zeros_like(decayed), decayed) + upd_i).astype(compute_dtype)
    return state_next, state_in

  init_state = jnp.zeros((batch, heads, dk, dv), compute_dtype)
  _, states_in = jax.lax.scan(
      reconstruct_state_step, init_state, (state_updates, decay_end_c, any_reset_c)
  )

  # 2. Hoist pre-scan backward matmuls in parallel across all chunks
  w_t_c = jnp.matmul(
      m_inv_dt,
      k_cum_prev * beta_dt,
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  ).astype(compute_dtype).swapaxes(-1, -2)
  qs_t = jnp.concatenate([q_cum_prev, scores_dt], axis=-1).swapaxes(-1, -2)
  do_qs_t = jnp.matmul(
      qs_t,
      do_dt,
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  ).astype(compute_dtype)
  do_q_t = do_qs_t[..., :dk, :]
  do_scores_t = do_qs_t[..., dk:, :]

  # 3. 2-matmul reverse scan over chunks for d_deltas and dstates_next
  def bwd_scan_step(dstate_next, xs):
    w_t_i, k_carry_i, do_q_t_i, do_scores_t_i, decay_end_i, any_reset_i = xs
    d_delta_i = (
        do_scores_t_i
        + jnp.matmul(k_carry_i, dstate_next, precision=dot_prec, preferred_element_type=jnp.float32).astype(
            compute_dtype
        )
    ).astype(compute_dtype)
    dstate_decayed = jnp.where(any_reset_i, jnp.zeros_like(dstate_next), dstate_next) * decay_end_i
    dstate_in = (
        do_q_t_i
        + dstate_decayed
        - jnp.matmul(w_t_i, d_delta_i, precision=dot_prec, preferred_element_type=jnp.float32).astype(compute_dtype)
    ).astype(compute_dtype)
    return dstate_in, (d_delta_i, dstate_next)

  init_dstate = jnp.zeros((batch, heads, dk, dv), compute_dtype)
  _, (d_deltas, dstates_next) = jax.lax.scan(
      bwd_scan_step,
      init_dstate,
      (w_t_c, k_carry_c, do_q_t, do_scores_t, decay_end_c, any_reset_c),
      reverse=True,
  )

  # 4. 1-GEMM Gram-Adjoint Collapse + Fused 128-row MXU tile outside scan:
  #    G_c = A_c^T @ d_deltas; P_c = [dO_c; -G_c] ([128, 512]), R_c = [S_{c-1}^T, \Delta_c^T] ([512, 320])
  g_c = jnp.matmul(
      m_inv_dt.swapaxes(-1, -2),
      d_deltas,
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  ).astype(compute_dtype)
  p_c = jnp.concatenate([do_dt, -g_c], axis=-2)
  r_c = jnp.concatenate([states_in.swapaxes(-1, -2), deltas.swapaxes(-1, -2)], axis=-1)
  pr_out = jnp.matmul(
      p_c,
      r_c,
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  ).astype(compute_dtype)

  qk_part = pr_out[..., :dk]
  sm_part = pr_out[..., dk:]
  dq_cum_prev = qk_part[..., :c, :]
  dk_cum_beta = qk_part[..., c:, :]
  dscores_c = sm_part[..., :c, :].astype(scores_c.dtype)
  dm = jnp.tril(sm_part[..., c:, :], k=-1)

  if not has_m_full:
    dm_diag = dm[..., None, :, :]
    dm_full = None
  else:
    dm_diag = jnp.stack(
        [dm[..., i * b_sub : (i + 1) * b_sub, i * b_sub : (i + 1) * b_sub] for i in range(num_sub)],
        axis=-3,
    )
    strict_blk = jnp.tril(jnp.ones((num_sub, num_sub), dtype=bool), k=-1)[..., :, None, :, None]
    dm_full = jnp.where(
        strict_blk,
        dm.reshape(*dm.shape[:-2], num_sub, b_sub, num_sub, b_sub),
        0.0,
    ).reshape(dm.shape)

  dk_carry = jnp.matmul(
      deltas,
      dstates_next.swapaxes(-1, -2),
      precision=dot_prec,
      preferred_element_type=jnp.float32,
  ).astype(compute_dtype)

  # 5. Elementwise cotangents for q_c, k_c, v_c, beta_c, rel
  dv_c = (g_c * beta_dt).astype(v_c.dtype)
  dk_cum_prev = (dk_cum_beta * beta_dt).astype(compute_dtype)
  dbeta_c = (
      jnp.sum(g_c.astype(jnp.float32) * v_dt.astype(jnp.float32), axis=-1)
      + jnp.sum(dk_cum_beta.astype(jnp.float32) * k_cum_prev.astype(jnp.float32), axis=-1)
  ).astype(beta_c.dtype)

  dq_c = (dq_cum_prev * cum_prev).astype(q_c.dtype)
  dk_c = (dk_cum_prev * cum_prev + dk_carry * decay_to_end).astype(k_c.dtype)

  dcum_prev = (
      dq_cum_prev.astype(jnp.float32) * q_dt.astype(jnp.float32)
      + dk_cum_prev.astype(jnp.float32) * k_dt.astype(jnp.float32)
  )
  drel_1 = jnp.where(from_prev_mask, dcum_prev * cum_prev.astype(jnp.float32), 0.0)

  d_decay_to_end = dk_carry.astype(jnp.float32) * k_dt.astype(jnp.float32) * decay_to_end.astype(jnp.float32)
  d_diff = jnp.where(last_seg, d_decay_to_end, 0.0)

  d_decay_end = jnp.sum(
      jnp.where(any_reset_c, 0.0, dstates_next.astype(jnp.float32)) * states_in.astype(jnp.float32),
      axis=-1,
      keepdims=True,
  ).swapaxes(-1, -2)
  drel_end_2 = d_decay_end * decay_end_c.swapaxes(-1, -2).astype(jnp.float32)

  drel = drel_1 - d_diff
  drel_last_extra = drel_end_2 + jnp.sum(d_diff, axis=-2, keepdims=True)
  drel = drel.at[..., -1:, :].add(drel_last_extra).astype(rel.dtype)

  return (dq_c, dk_c, dv_c, drel, None, dbeta_c, dm_diag, dm_full, dscores_c)


_run_kda_wy_scan_and_readout_custom.defvjp(
    _run_kda_wy_scan_and_readout_custom_fwd,
    _run_kda_wy_scan_and_readout_custom_bwd,
)


def _delta_rule_chunked_wy_autodiff(
    q, k, v, log_decay, beta, resets, chunk_size: int, state_dtype: str = "float32"
) -> jnp.ndarray:
  """WY-hoisted chunked delta rule with JAX autodiff over `_run_kda_wy_scan_and_readout` (reference)."""
  batch, seq_len, heads, dk = q.shape
  dv = v.shape[-1]
  num_chunks = seq_len // chunk_size
  c = chunk_size
  compute_dtype = jnp.bfloat16 if state_dtype == "bfloat16" else jnp.float32

  def to_chunks(x):
    return x.reshape(batch, num_chunks, c, heads, -1).transpose(1, 0, 3, 2, 4)

  q_c = checkpoint_name(to_chunks(q.astype(compute_dtype)), "context")
  k_c = checkpoint_name(to_chunks(k.astype(compute_dtype)), "context")
  v_c = checkpoint_name(to_chunks(v.astype(compute_dtype)), "context")
  log_a_c = to_chunks(log_decay)
  beta_c = checkpoint_name(beta.reshape(batch, num_chunks, c, heads).transpose(1, 0, 3, 2), "context")
  reset_c = resets.reshape(batch, num_chunks, c).transpose(1, 0, 2)

  positions = jnp.arange(c)
  seg = checkpoint_name(jnp.cumsum(reset_c.astype(jnp.int32), axis=-1), "context")
  seg_start = jax.lax.cummax(jnp.where(reset_c, positions, 0), axis=2)

  log_cum = jnp.cumsum(log_a_c, axis=-2)
  gather = seg_start[:, :, None, :, None]
  base = jnp.take_along_axis(log_cum, gather, axis=-2) - jnp.take_along_axis(log_a_c, gather, axis=-2)
  rel = checkpoint_name(log_cum - base, "context")

  m_inv, scores_c = _compute_kda_subblock_inv_and_scores(q_c, k_c, rel, seg, beta_c, c, compute_dtype)

  outputs = _run_kda_wy_scan_and_readout(q_c, k_c, v_c, rel, seg, beta_c, m_inv, scores_c, compute_dtype)
  outputs = checkpoint_name(outputs, "context")
  return outputs.transpose(1, 0, 3, 2, 4).reshape(batch, seq_len, heads, dv)


@jax.custom_vjp
def _chunk_segmented_cumsum(log_a_c: jnp.ndarray, reset_c: jnp.ndarray, seg: jnp.ndarray) -> jnp.ndarray:
  """Segmented intra-chunk cumulative sum of `log_a_c` with MXU-accelerated backward pass.

  Forward pass computes `rel[i] = sum_{j=seg_start[i]}^{i} log_a_c[j]` using prefix sums
  and `take_along_axis`. Standard JAX autodiff lowers the transpose of the two
  `take_along_axis` calls into two 5D dynamic `scatter-add` fusions (`scatter_fusion`
  and `scatter_fusion.1`, costing 8.12 ms/layer at L=8192 on TPU v4). Because the linear
  operator is `L_{i,j} = 1(j <= i and seg[i] == seg[j])`, its exact transpose is the
  segmented upper-triangular matrix `L^T_{j,i} = 1(j <= i and seg[j] == seg[i])`, which
  executes as a single `[C, C] @ [C, dk]` MXU matmul in 0.02 ms.
  """
  c = log_a_c.shape[-2]
  positions = jnp.arange(c)
  seg_start = jax.lax.cummax(jnp.where(reset_c, positions, 0), axis=2)
  log_cum = jnp.cumsum(log_a_c, axis=-2)
  gather = seg_start[:, :, None, :, None]
  base = jnp.take_along_axis(log_cum, gather, axis=-2) - jnp.take_along_axis(log_a_c, gather, axis=-2)
  return log_cum - base


def _chunk_segmented_cumsum_fwd(log_a_c, reset_c, seg):
  rel = _chunk_segmented_cumsum(log_a_c, reset_c, seg)
  return rel, seg


def _chunk_segmented_cumsum_bwd(seg, drel):
  same_seg_upper = jnp.triu(seg[:, :, None, :, None] == seg[:, :, None, None, :]).astype(drel.dtype)
  dlog_a_c = jnp.matmul(same_seg_upper, drel, precision=jax.lax.Precision.HIGHEST)
  return dlog_a_c, None, None


_chunk_segmented_cumsum.defvjp(_chunk_segmented_cumsum_fwd, _chunk_segmented_cumsum_bwd)


def _delta_rule_chunked(
    q, k, v, log_decay, beta, resets, chunk_size: int, state_dtype: str = "float32"
) -> jnp.ndarray:
  """Optimized chunked delta rule. Mathematically identical to ``_delta_rule_scan``.

  Combines six TPU MXU/VMEM co-design optimizations:
  1. 16-token sub-block pivot factorization in compute_dtype: factors strictly
     off-diagonal 16x16 sub-blocks (75% of the chunk matrix) into overflow-free
     2D MXU matmuls while evaluating the 16x16 diagonal blocks in compute_dtype
     (halving 6D VMEM footprint and backward reduction work).
  2. Closed-form gauge-invariance sub-block custom VJP (`_compute_kda_subblock_matrices`):
     eliminates differentiating through the 6D `[128, 1, 8, 4, 16, 16, 256]` diagonal
     decay tensor and off-diagonal pivot factors (`fusion.83`, `fusion.84`, `fusion.401`),
     computing `drel = q * dq + k * (dk_row - dk_col)` in closed form.
  3. Segmented cumsum custom VJP (`_chunk_segmented_cumsum`): replaces the two 5D
     backward `scatter-add` fusions from `take_along_axis` (`scatter_fusion` and
     `scatter_fusion.1`) with an exact segmented upper-triangular MXU matmul.
  4. Hierarchical 16 -> 32 -> 64 block-triangular MXU inversion: replaces 5-step
     64x64 Newton-Schulz in forward.
  5. Three-stage WY-hoisted recurrence: vectorizes intra-chunk WY factors (w, u,
     scores, q_cum, k_carry) and final readout across chunks in parallel outside
     lax.scan, reducing the sequential scan body to 2 MXU matmuls per chunk.
  6. Analytical Custom VJP with 1-GEMM Gram-Adjoint Collapse & 0-matmul state
     reconstruction: caches compact `deltas` (4 MiB at 4k, 8 MiB at 8k) and `m_inv`
     instead of `states_in` (268 MiB), reconstructs `states_in` in backward via 1
     parallel MXU matmul + elementwise scan, runs a 2-matmul reverse scan, and
     collapses `dq_cum_prev`, `dk_cum_beta`, `dscores_c`, and `dM = tril(-A^T dA A^T, -1)`
     into a single fused `[128, 512] @ [512, 320]` MXU matmul.
  """
  batch, seq_len, heads, dk = q.shape
  dv = v.shape[-1]
  num_chunks = seq_len // chunk_size
  c = chunk_size
  compute_dtype = jnp.bfloat16 if state_dtype == "bfloat16" else jnp.float32

  def to_chunks(x):
    return x.reshape(batch, num_chunks, c, heads, -1).transpose(1, 0, 3, 2, 4)

  q_c = checkpoint_name(to_chunks(q.astype(compute_dtype)), "context")
  k_c = checkpoint_name(to_chunks(k.astype(compute_dtype)), "context")
  v_c = checkpoint_name(to_chunks(v.astype(compute_dtype)), "context")
  log_a_c = to_chunks(log_decay)
  beta_c = checkpoint_name(beta.reshape(batch, num_chunks, c, heads).transpose(1, 0, 3, 2), "context")
  reset_c = resets.reshape(batch, num_chunks, c).transpose(1, 0, 2)

  seg = checkpoint_name(jnp.cumsum(reset_c.astype(jnp.int32), axis=-1), "context")
  rel = checkpoint_name(_chunk_segmented_cumsum(log_a_c, reset_c, seg), "context")

  m_diag, m_full, scores_c = _compute_kda_subblock_matrices(q_c, k_c, rel, seg, beta_c, c, compute_dtype)
  scores_c = checkpoint_name(scores_c, "context")

  outputs = _run_kda_wy_scan_and_readout_custom(
      q_c, k_c, v_c, rel, seg, beta_c, m_diag, m_full, scores_c, compute_dtype
  )
  outputs = checkpoint_name(outputs, "context")
  return outputs.transpose(1, 0, 3, 2, 4).reshape(batch, seq_len, heads, dv)


_delta_rule_chunked_wy_opt = _delta_rule_chunked


def _delta_rule_scan(q, k, v, decay, beta, resets) -> jnp.ndarray:
  """Unfused delta-rule recurrence, scanned over time.

  State is ``[batch, heads, key_head_dim, value_head_dim]`` and is zeroed at
  document boundaries so a packed batch behaves like independent sequences.
  """

  def step(state, xs):
    q_t, k_t, v_t, decay_t, beta_t, reset_t = xs
    state = jnp.where(reset_t[:, None, None, None], 0.0, state)
    state = state * decay_t[..., None]
    prediction = jnp.einsum("bhkv,bhk->bhv", state, k_t)
    delta = (v_t - prediction) * beta_t[..., None]
    state = state + jnp.einsum("bhk,bhv->bhkv", k_t, delta)
    return state, jnp.einsum("bhkv,bhk->bhv", state, q_t)

  batch, _, heads, dk = q.shape
  dv = v.shape[-1]
  init_state = jnp.zeros((batch, heads, dk, dv), jnp.float32)
  # Scan over time, so move the length axis to the front.
  xs = (
      jnp.swapaxes(q, 0, 1),
      jnp.swapaxes(k, 0, 1),
      jnp.swapaxes(v, 0, 1),
      jnp.swapaxes(decay, 0, 1),
      jnp.swapaxes(beta, 0, 1),
      jnp.swapaxes(resets, 0, 1),
  )
  _, outputs = jax.lax.scan(step, init_state, xs)
  return jnp.swapaxes(outputs, 0, 1)


class OLMoE3Attention(nnx.Module):
  """Full attention for OLMoE3: GQA, NoPE, per-head QK-norm, sigmoid output gate.

  The gate is produced by widening the query projection to ``2 * head_dim`` and
  splitting, which is how MaxText's shared attention implements gated hybrids.
  Parameter count matches the reference's separate ``w_g`` projection; only the
  storage layout differs, which the checkpoint converter must account for.
  """

  def __init__(self, config: Config, mesh: Mesh, model_mode: str, quant: None | Quant = None, *, rngs: nnx.Rngs):
    self.config = config
    cfg = config
    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, cfg.emb_dim)

    self.attention = attentions.Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        out_axis_names=(BATCH, LENGTH, EMBED),
        mesh=mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        use_qk_norm=cfg.use_qk_norm,
        query_pre_attn_scalar=cfg.head_dim**-0.5,
        model_mode=model_mode,
        is_nope_layer=True,  # OLMoE3 uses no positional embedding at all.
        rngs=rngs,
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      kv_cache=None,
      attention_metadata=None,
  ):
    return self.attention(
        inputs_q=inputs,
        inputs_kv=inputs,
        inputs_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )


class OLMoE3LatentRoutedMoE(moe.RoutedMoE):
  """RoutedMoE whose router scores the full-width residual.

  The experts run at ``moe_expert_input_dim`` (the latent), but OLMoE3 routes on
  the uncompressed ``emb_dim`` residual, so the gate has to be sized separately
  from the expert input. ``RoutedMoE`` already accepts ``gate_inputs``; only the
  gate's input width needs overriding.
  """

  def get_topk(self, gate_logits, pre_bias_logits, rngs=None, input_ids=None, forced_routed_experts=None):
    """Top-k routing weights, rescaled to sum to ``top_k`` as in the reference.

    ``input_ids`` carries the packed segment ids (EMo routes per document); it
    is consumed here and never forwarded, so the base class's hash-routing
    interpretation of the argument can't engage. ``forced_routed_experts`` is
    passed straight through to the base class.

    The base class's softmax over the top-k logits already equals the
    reference's L1-normalized gather of full-softmax scores
    (``normalize_expert_weights=1.0``); ``restore_weight_scale=True`` then
    multiplies by ``top_k``, which is the only piece missing here. The masked
    logits leave that unchanged: every selected expert is inside the pool, so
    the gathered scores are the unmasked ones.
    """
    unmasked_logits = pre_bias_logits if pre_bias_logits is not None else gate_logits
    if self.config.emo_enabled:
      gate_logits = self._emo_mask_logits(gate_logits, input_ids, rngs)
    top_k_weights, top_k_indices = super().get_topk(
        gate_logits,
        unmasked_logits if (self.config.emo_enabled and self.config.moe_lean_routing) else pre_bias_logits,
        rngs,
        input_ids=None,
        forced_routed_experts=forced_routed_experts,
    )
    return top_k_weights * self.num_experts_per_tok, top_k_indices

  def _emo_mask_logits(self, gate_logits, segment_ids, rngs):
    """Masks router logits to each document's expert pool (the EMo router).

    Reference: ``EMoRouter`` in the standalone model. A document aggregates its
    tokens' full-softmax scores, keeps its top ``pool`` experts by that total,
    and every token of the document then routes within the pool. ``pool`` is
    sampled uniformly from [emo_min, emo_max] per document during training and
    fixed at ``emo_eval_document_expert_pool`` otherwise.

    MaxText packs documents contiguously, so the per-document score totals are
    cumulative-sum differences at document boundaries rather than a scatter
    over a static document count — this also makes the masking independent of
    the segment-id numbering (only boundaries matter). The auxiliary losses
    intentionally see the unmasked logits, as in the reference.
    """
    cfg = self.config
    batch, seq_len, num_experts = gate_logits.shape
    logits_for_pool = jax.lax.stop_gradient(gate_logits) if cfg.moe_lean_routing else gate_logits
    scores = jax.nn.softmax(logits_for_pool.astype(jnp.float32), axis=-1)

    if segment_ids is None:
      document_scores = jnp.sum(scores, axis=1, keepdims=True)
      if rngs is not None and cfg.model_call_mode != "inference":
        rng = rngs.params() if hasattr(rngs, "params") and callable(getattr(rngs, "params")) else rngs
        draws = jax.random.randint(
            rng, (batch, seq_len), cfg.emo_min_document_expert_pool, cfg.emo_max_document_expert_pool + 1
        )
        pool = draws[:, :1]
      else:
        pool = jnp.full((batch, 1), cfg.emo_eval_document_expert_pool, dtype=jnp.int32)
    elif seq_len >= 256:
      max_docs = 64
      prev_ids = jnp.pad(segment_ids, ((0, 0), (1, 0)), constant_values=-1)[:, :seq_len]
      is_first = segment_ids != prev_ids
      doc_id = jnp.minimum(jnp.cumsum(is_first.astype(jnp.int32), axis=1) - 1, max_docs - 1)
      doc_mask = doc_id[:, :, None] == jnp.arange(max_docs, dtype=jnp.int32)[None, None, :]
      doc_scores = jnp.matmul(
          doc_mask.swapaxes(-1, -2).astype(jnp.bfloat16),
          scores.astype(jnp.bfloat16),
          preferred_element_type=jnp.float32,
      )
      if rngs is not None and cfg.model_call_mode != "inference":
        rng = rngs.params() if hasattr(rngs, "params") and callable(getattr(rngs, "params")) else rngs
        draws = jax.random.randint(
            rng, (batch, seq_len), cfg.emo_min_document_expert_pool, cfg.emo_max_document_expert_pool + 1
        )
        doc_start = jnp.argmax(doc_mask, axis=1)
        doc_pool = jnp.take_along_axis(draws, doc_start, axis=1)
      else:
        doc_pool = jnp.full((batch, max_docs), cfg.emo_eval_document_expert_pool, dtype=jnp.int32)
      if cfg.moe_lean_routing:
        doc_keep = self._emo_keep_by_threshold(
            jax.lax.stop_gradient(doc_scores), doc_pool, bisect=cfg.emo_threshold_by_bisection
        )
      else:
        order = jnp.argsort(-doc_scores, axis=-1)
        rank = jnp.argsort(order, axis=-1)
        doc_keep = rank < doc_pool[..., None]
      keep = jnp.take_along_axis(doc_keep, doc_id[..., None], axis=1)
      if cfg.moe_lean_routing:
        return jnp.where(keep, jax.lax.stop_gradient(gate_logits), -jnp.inf)
      return jnp.where(keep, gate_logits, -jnp.inf)
    else:
      positions = jnp.arange(seq_len)
      prev_ids = jnp.pad(segment_ids, ((0, 0), (1, 0)), constant_values=-1)[:, :seq_len]
      is_first = segment_ids != prev_ids  # position 0 always starts a document
      next_ids = jnp.pad(segment_ids, ((0, 0), (0, 1)), constant_values=-1)[:, 1:]
      is_last = segment_ids != next_ids  # the final position always ends one

      start = jax.lax.cummax(jnp.where(is_first, positions, 0), axis=1)
      end = jnp.flip(jax.lax.cummin(jnp.flip(jnp.where(is_last, positions, seq_len), axis=1), axis=1), axis=1)

      csum = jnp.cumsum(scores, axis=1)
      csum_end = jnp.take_along_axis(csum, end[..., None], axis=1)
      before = jnp.take_along_axis(csum, jnp.maximum(start - 1, 0)[..., None], axis=1)
      csum_before = jnp.where((start == 0)[..., None], 0.0, before)
      document_scores = csum_end - csum_before  # [b, s, E]: each token sees its document's totals

      if rngs is not None and cfg.model_call_mode != "inference":
        rng = rngs.params() if hasattr(rngs, "params") and callable(getattr(rngs, "params")) else rngs
        draws = jax.random.randint(
            rng, (batch, seq_len), cfg.emo_min_document_expert_pool, cfg.emo_max_document_expert_pool + 1
        )
        # One draw per document: every token reads the draw at its document start.
        pool = jnp.take_along_axis(draws, start, axis=1)
      else:
        pool = jnp.full((batch, seq_len), cfg.emo_eval_document_expert_pool, dtype=jnp.int32)

    if cfg.moe_lean_routing:
      keep = self._emo_keep_by_threshold(
          jax.lax.stop_gradient(document_scores), pool, bisect=cfg.emo_threshold_by_bisection
      )
      return jnp.where(keep, jax.lax.stop_gradient(gate_logits), -jnp.inf)
    else:
      order = jnp.argsort(-document_scores, axis=-1)
      rank = jnp.argsort(order, axis=-1)
      keep = rank < pool[..., None]
    return jnp.where(keep, gate_logits, -jnp.inf)

  @staticmethod
  def _emo_keep_by_threshold(scores, pool, bisect=False):
    """``argsort(argsort(-scores)) < pool`` with one value sort instead of two argsorts.

    The pool-th largest score is the threshold. Everything above it is kept, and
    ties at the threshold fill the remaining slots lowest index first, which is
    the order the stable argsort gives. ``bisect`` finds the threshold with 32
    counting passes over the scores' order-preserving uint32 keys instead of a sort.
    """
    if bisect:
      return moe.keep_top_by_bisection(scores, pool[..., None])
    num_experts = scores.shape[-1]
    ascending = jnp.sort(scores, axis=-1)
    thr = jnp.take_along_axis(ascending, (num_experts - pool)[..., None], axis=-1)
    above = scores > thr
    at = scores == thr
    slots = pool[..., None] - jnp.sum(above, axis=-1, keepdims=True)
    return above | (at & (jnp.cumsum(at, axis=-1) <= slots))

  def get_ragged_buffer_factor(self):
    """Returns ragged_buffer_factor, defaulting to 1.125 for 512-expert top-16 EP>1 lean routing."""
    factor = super().get_ragged_buffer_factor()
    if factor > 0.0:
      return factor
    if factor < -1.5:
      return -1.0
    if self.config.moe_lean_routing and self.get_expert_parallelism_size() > 1 and not self.config.use_ring_of_experts:
      return 1.125
    return factor

  def load_balance_loss(self, top_k_indices, logits) -> jax.Array:
    """OLMo-core's batch-level load-balancing loss (``global_load_balancing``).

    ``lb = (E / K) * sum_e mean_{b,s}(probs_e) * counts_e / (B * S)``, scaled by
    ``load_balance_loss_weight``. Counts are summed over the whole batch rather
    than per sequence; under jit the batch axis here is the global batch, so
    the reduction spans all data-parallel replicas, which is exactly the
    reference's all-reduced global balancing.
    """
    counts = jnp.bincount(jnp.ravel(top_k_indices), length=self.num_experts).astype(jnp.float32)
    tokens = top_k_indices.shape[0] * top_k_indices.shape[1]
    mean_probs = logits.astype(jnp.float32).mean(axis=(0, 1))
    lb = (mean_probs * counts).sum() * self.num_experts / (self.num_experts_per_tok * tokens)
    return lb * self.config.load_balance_loss_weight

  def _add_router_z_loss(self, lb_loss, gate_logits, pre_bias_logits):
    if self.config.load_balance_loss_weight > 0.0 and lb_loss is not None:
      logits = pre_bias_logits if pre_bias_logits is not None else gate_logits
      z_loss = jnp.mean(jax.scipy.special.logsumexp(logits.astype(jnp.float32), axis=-1) ** 2)
      return lb_loss + _ROUTER_Z_LOSS_WEIGHT * z_loss
    return lb_loss

  def sparse_matmul(self, inputs, gate_logits, pre_bias_logits, *args, **kwargs):
    output, lb_loss, bias_updates = super().sparse_matmul(inputs, gate_logits, pre_bias_logits, *args, **kwargs)
    return output, self._add_router_z_loss(lb_loss, gate_logits, pre_bias_logits), bias_updates

  def dense_matmul(self, inputs, gate_logits, pre_bias_logits, *args, **kwargs):
    output, lb_loss, bias_updates = super().dense_matmul(inputs, gate_logits, pre_bias_logits, *args, **kwargs)
    return output, self._add_router_z_loss(lb_loss, gate_logits, pre_bias_logits), bias_updates

  def __init__(self, *args, gate_in_features: int, **kwargs):
    super().__init__(*args, **kwargs)
    cfg = self.config
    if cfg.emo_enabled:
      if cfg.emo_min_document_expert_pool < self.num_experts_per_tok:
        raise ValueError("emo_min_document_expert_pool must be >= num_experts_per_tok.")
      pool_ok = cfg.emo_min_document_expert_pool <= cfg.emo_max_document_expert_pool <= self.num_experts
      eval_ok = self.num_experts_per_tok <= cfg.emo_eval_document_expert_pool <= self.num_experts
      if not pool_ok or not eval_ok:
        raise ValueError("EMo pool sizes must satisfy top_k <= min <= max/eval <= num_experts.")
    if gate_in_features != self.moe_expert_input_dim:
      self.gate = moe.GateLogit(
          in_features_shape=gate_in_features,
          out_features_shape=self.num_experts,
          mesh=self.mesh,
          model_name=self.config.model_name,
          dtype=jnp.float32 if self.config.float32_gate_logits else self.dtype,
          weight_dtype=self.weight_dtype,
          quant=self.quant,
          kernel_init=self.kernel_init,
          kernel_axes=self.kernel_axes,
          use_bias=self.config.routed_bias,
          score_func=self.config.routed_score_func,
          matmul_precision=self.config.matmul_precision,
          shard_mode=self.config.shard_mode,
          rngs=self.rngs,
      )


class OLMoE3DecoderLayer(nnx.Module):
  """One OLMoE3 block: KDA or full attention, then shared FFN plus latent MoE."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      layer_idx: int,
      quant: None | Quant = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = config
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

    def block_norm():
      return RMSNorm(
          num_features=cfg.emb_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=("norm",),
          epsilon=cfg.normalization_layer_epsilon,
          rngs=rngs,
      )

    # Four norms per block: pre and post around each sublayer.
    self.attn_in_norm = block_norm()
    self.attn_out_norm = block_norm()
    self.ffn_in_norm = block_norm()
    self.ffn_out_norm = block_norm()

    self.is_full_attention_layer = (layer_idx + 1) % cfg.inhomogeneous_layer_cycle_interval == 0
    if self.is_full_attention_layer:
      self.mixer = OLMoE3Attention(cfg, mesh, model_mode, quant, rngs=rngs)
    else:
      self.mixer = OLMoE3KimiDeltaAttention(cfg, mesh, quant, rngs=rngs)

    # Layer 0 is dense with a wide SwiGLU; MoE layers keep a narrow shared expert,
    # whose width is `shared_expert_mlp_dim` (defaults to the routed-expert width).
    self.is_dense_layer = layer_idx < cfg.first_num_dense_layers
    shared_dim = cfg.mlp_dim if self.is_dense_layer else cfg.shared_expert_mlp_dim
    self.shared_ffn = MlpBlock(
        config=cfg,
        mesh=mesh,
        in_features=cfg.emb_dim,
        intermediate_dim=shared_dim,
        activations=cfg.mlp_activations,
        kernel_init=olmoe3_init,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=quant,
        rngs=rngs,
    )

    if self.is_dense_layer:
      self.latent_down = self.latent_up = self.moe_block = None
    else:
      latent_dim = cfg.moe_expert_input_dim
      if latent_dim <= 0:
        raise ValueError("olmoe3 requires moe_expert_input_dim > 0 (the routed-expert latent width).")
      self.latent_down = _dense(cfg, cfg.emb_dim, latent_dim, ("embed", "mlp"), quant, rngs)
      self.latent_up = _dense(cfg, latent_dim, cfg.emb_dim, ("mlp", "embed"), quant, rngs)
      self.moe_block = OLMoE3LatentRoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=olmoe3_init,
          kernel_axes=("embed", None),
          intermediate_dim=cfg.moe_mlp_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          quant=quant,
          rngs=rngs,
          gate_in_features=cfg.emb_dim,
      )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | dict[str, Any] = None,
      attention_metadata: None | dict[str, Any] = None,
  ):
    del previous_chunk, slot
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    normed = self.attn_in_norm(inputs)
    if self.is_full_attention_layer:
      mixer_out, kv_cache = self.mixer(
          normed,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
    else:
      mixer_out = self.mixer(normed, decoder_segment_ids)

    hidden = inputs + self.attn_out_norm(mixer_out)
    hidden = nn.with_logical_constraint(hidden, self.activation_axis_names)
    hidden = checkpoint_name(hidden, "decoder_layer_input")

    ffn_in = self.ffn_in_norm(hidden)
    ffn_out = self.shared_ffn(ffn_in, deterministic=deterministic)
    if self.moe_block is not None:
      # input_ids carries the packed segment ids for the EMo document pools.
      latent_in = checkpoint_name(self.latent_down(ffn_in), "out_proj")
      routed, load_balance_loss, _ = self.moe_block(
          latent_in, gate_inputs=ffn_in, input_ids=decoder_segment_ids
      )
      if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
        self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)
      ffn_out = ffn_out + self.latent_up(routed)

    layer_output = hidden + self.ffn_out_norm(ffn_out)
    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    if self.config.scan_layers:
      return layer_output, None
    return layer_output, kv_cache


class OLMoE3ScannableBlock(nnx.Module):
  """One full mixer cycle (``inhomogeneous_layer_cycle_interval`` layers).

  OLMoE3 layers are not homogeneous (the mixer alternates and layer 0 is dense),
  so scanning happens over whole cycles rather than single layers.

  ``first_layer_idx`` is the global index of this block's first layer. Scanned
  copies share one parameter set, so the dense prefix cycle must be built and
  applied separately with ``first_layer_idx=0`` while the scanned cycles start
  past ``first_num_dense_layers``. Passing the within-cycle index instead would
  make layer 0 of *every* cycle dense.

  With ``olmoe3_per_layer_remat`` each layer is rematerialized on its own under
  ``remat_policy_fn``, so the backward holds one layer's recomputed activations
  instead of the whole cycle's.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant=None,
      *,
      first_layer_idx: int = 0,
      remat_policy_fn: Any | None = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.remat_policy_fn = remat_policy_fn
    self.remat_layers = config.olmoe3_per_layer_remat and config.remat_policy != "none"
    for i in range(config.inhomogeneous_layer_cycle_interval):
      setattr(
          self,
          f"layer_{i}",
          OLMoE3DecoderLayer(
              config=config,
              mesh=mesh,
              model_mode=model_mode,
              layer_idx=first_layer_idx + i,
              quant=quant,
              rngs=rngs.fork(),
          ),
      )

  def __call__(
      self,
      carry: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
  ):
    x = carry
    for i in range(self.config.inhomogeneous_layer_cycle_interval):
      layer = getattr(self, f"layer_{i}")
      args = (decoder_segment_ids, decoder_positions, deterministic, model_mode, previous_chunk, slot)
      if not self.remat_layers:
        x, _ = layer(x, *args)
        continue
      graphdef, state = nnx.split(layer)

      def run_layer(state_in, x_in, graphdef=graphdef, args=args):
        merged = nnx.merge(graphdef, state_in)
        out, _ = merged(x_in, *args)
        return out, nnx.state(merged)

      # The layers are unrolled here, so CSE must be blocked or XLA merges the
      # recompute back into the forward and saves everything.
      x, new_state = jax.checkpoint(run_layer, policy=self.remat_policy_fn, prevent_cse=True)(state, x)
      nnx.update(layer, new_state)
    return x, None


OLMoE3DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    OLMoE3DecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

OLMoE3ScannableBlockToLinen = nnx_wrappers.to_linen_class(
    OLMoE3ScannableBlock,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

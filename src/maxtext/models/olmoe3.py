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

Not yet implemented here (see the bring-up doc): the EMo document-pool router
and globally-balanced auxiliary loss. Phase 0 uses the standard top-k router.
"""

from typing import Any

from jax.sharding import Mesh
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


def _dense(config, in_features, out_features, kernel_axes, quant, rngs, use_bias=False):
  """DenseGeneral with the config-derived arguments this model always passes."""
  return DenseGeneral(
      in_features_shape=in_features,
      out_features_shape=out_features,
      axis=-1,
      kernel_init=max_initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
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
  acc = jnp.zeros_like(x)
  for tap in range(kernel_size):
    lag = kernel_size - 1 - tap
    shifted = jnp.pad(x, ((0, 0), (lag, 0), (0, 0)))[:, :seq_len, :]
    if segment_ids is not None and lag > 0:
      # -1 marks the left pad so it never matches a real segment id.
      shifted_ids = jnp.pad(segment_ids, ((0, 0), (lag, 0)), constant_values=-1)[:, :seq_len]
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

    conv_init = nnx.initializers.normal(stddev=0.02)
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

    q = causal_depthwise_conv(self.w_q(x), self.q_conv[...], decoder_segment_ids).reshape(batch, seq_len, heads, dk)
    k = causal_depthwise_conv(self.w_k(x), self.k_conv[...], decoder_segment_ids).reshape(batch, seq_len, heads, dk)
    v = causal_depthwise_conv(self.w_v(x), self.v_conv[...], decoder_segment_ids).reshape(batch, seq_len, heads, dv)

    # The fused KDA kernel L2-normalizes q/k and applies the query scale
    # internally; both are explicit here.
    q = _l2_normalize(q.astype(jnp.float32)) * (dk**-0.5)
    k = _l2_normalize(k.astype(jnp.float32))

    raw_g = self.f_proj_2(self.f_proj_1(x)).reshape(batch, seq_len, heads, dk).astype(jnp.float32)
    dt = self.dt_bias[...].reshape(1, 1, heads, dk)
    log_decay = -jnp.exp(self.A_log[...]).reshape(1, 1, heads, 1) * jax.nn.softplus(raw_g + dt)
    # beta in [0, 2): the reference sets allow_neg_eigval=True.
    beta = 2.0 * jax.nn.sigmoid(self.w_b(x).astype(jnp.float32))

    if decoder_segment_ids is None:
      resets = jnp.zeros((batch, seq_len), dtype=bool)
    else:
      prev_ids = jnp.pad(decoder_segment_ids, ((0, 0), (1, 0)), constant_values=-1)[:, :seq_len]
      resets = decoder_segment_ids != prev_ids

    chunk = self.config.gdn_chunk_size
    if chunk > 0 and seq_len % chunk == 0:
      out = _delta_rule_chunked(q, k, v.astype(jnp.float32), log_decay, beta, resets, chunk, self.config.gdn_state_dtype)
    else:
      out = _delta_rule_scan(q, k, v.astype(jnp.float32), jnp.exp(log_decay), beta, resets)

    gate = jax.nn.sigmoid(self.g_proj_2(self.g_proj_1(x)).reshape(batch, seq_len, heads, dv))
    out = self.o_norm(out.astype(x.dtype)) * gate
    return self.w_out(out.reshape(batch, seq_len, heads * dv))


def _l2_normalize(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
  return x * jax.lax.rsqrt(jnp.maximum(jnp.sum(x * x, axis=-1, keepdims=True), eps))


def _delta_rule_chunked(q, k, v, log_decay, beta, resets, chunk_size: int, state_dtype: str = "float32") -> jnp.ndarray:
  """Chunked delta rule. Mathematically identical to ``_delta_rule_scan``.

  The per-token scan is bound by HBM bandwidth, not compute: it reloads the whole
  ``[batch, heads, dk, dv]`` state every timestep to do a couple of FLOPs per
  element, which measures out around 0.5 FLOP/byte against a machine that needs
  ~348 to saturate the MXU. Chunking loads the state once per chunk and turns
  the intra-chunk work into ``C x C`` matmuls, so intensity rises roughly with
  ``C`` and the sequential chain shortens from ``T`` to ``T/C``.

  The algebra works because KDA's decay, though per key channel rather than per
  head, still factors. With chunk-local ``A_t = prod_{i<=t} a_i``::

      S_t[d,v] = sum_{j<=t} (A_t[d] / A_j[d]) k_j[d] delta_j[v]
      o_t[v]   = sum_{j<=t} ( (q_t * A_t) . (k_j / A_j) ) delta_j[v]

  so folding ``A`` into q/k and ``1/A`` into its partner restores a plain matmul.
  ``A`` is chunk-local, which is what bounds the growth of ``1/A``.
  """
  batch, seq_len, heads, dk = q.shape
  dv = v.shape[-1]
  num_chunks = seq_len // chunk_size
  c = chunk_size
  # Decay/cumulative math stays float32; the operands and the carried state use
  # the compute dtype, which halves the traffic on the bandwidth-bound path.
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

    # A document boundary inside a chunk restarts the recurrence, so the
    # cumulative decay restarts with it. Zeroing the decay at the boundary
    # instead would poison every later cumprod and silently drive the tail of
    # the chunk to zero.
    seg = jnp.cumsum(reset_i.astype(jnp.int32), axis=-1)
    seg_start = jax.lax.cummax(jnp.where(reset_i, positions, 0), axis=1)

    log_cum = jnp.cumsum(log_a, axis=-2)
    gather = seg_start[:, None, :, None]
    base = jnp.take_along_axis(log_cum, gather, axis=-2) - jnp.take_along_axis(log_a, gather, axis=-2)
    rel = log_cum - base  # <= 0, so exp(rel) is always safe
    cum = jnp.exp(rel)
    q_i = q_i.astype(compute_dtype)
    k_i = k_i.astype(compute_dtype)
    v_i = v_i.astype(compute_dtype)

    # The pairwise terms only ever need exp(rel_i - rel_j) for i >= j, which is
    # bounded by 1. Per-channel decay stops that collapsing to a mask, so the
    # cumulative decay has to be folded into the two operands separately, and
    # *that* is what can overflow: exp(-rel_j) grows like decay**-C. Splitting
    # the fold symmetrically about the chunk midpoint halves the worst-case
    # exponent in each operand, which doubles the usable chunk size.
    shift = 0.5 * rel[..., -1:, :]
    q_fold = q_i * jnp.exp(rel - shift).astype(compute_dtype)
    k_fold = k_i * jnp.exp(rel - shift).astype(compute_dtype)
    k_inv = k_i * jnp.exp(shift - rel).astype(compute_dtype)

    same_doc = seg[:, None, :, None] == seg[:, None, None, :]
    from_prev = (seg[:, None, :, None] == 0).astype(v_i.dtype)

    # (I + diag(beta) M) delta = diag(beta) (v - carried prediction)
    m = jnp.einsum("bhid,bhjd->bhij", k_fold, k_inv)
    m = jnp.where(strict & same_doc, m, 0.0) * beta_i[..., None]
    m_inv = jax.scipy.linalg.solve_triangular(eye + m, jnp.broadcast_to(eye, m.shape), lower=True, unit_diagonal=True)
    carried = jnp.einsum("bhid,bhdv->bhiv", k_i * cum.astype(compute_dtype), state) * from_prev
    delta = jnp.einsum(
        "bhij,bhjv->bhiv", m_inv.astype(compute_dtype), (v_i - carried) * beta_i[..., None].astype(compute_dtype)
    )

    scores = jnp.einsum("bhid,bhjd->bhij", q_fold, k_inv)
    scores = jnp.where(causal & same_doc, scores, 0.0).astype(compute_dtype)
    out = jnp.einsum("bhid,bhdv->bhiv", q_i * cum.astype(compute_dtype), state) * from_prev
    out = out + jnp.einsum("bhij,bhjv->bhiv", scores, delta)

    # Carry: the incoming state survives only if this chunk held no reset, and
    # only the final segment's updates propagate past the chunk boundary.
    any_reset = (seg[:, -1] > 0)[:, None, None, None]
    decayed = state * cum[..., -1, :][..., None].astype(compute_dtype)
    state = jnp.where(any_reset, jnp.zeros_like(decayed), decayed)
    last_seg = (seg == seg[:, -1:])[:, None, :, None]
    k_carry = k_i * (jnp.exp(rel[..., -1:, :] - rel) * last_seg).astype(compute_dtype)
    # The carry dtype has to match the scan's init exactly or lax.scan rejects it.
    state = (state + jnp.einsum("bhid,bhiv->bhdv", k_carry, delta)).astype(compute_dtype)
    return state, out.astype(jnp.float32)

  init_state = jnp.zeros((batch, heads, dk, dv), compute_dtype)
  _, outputs = jax.lax.scan(body, init_state, (q_c, k_c, v_c, log_a_c, beta_c, reset_c))
  return outputs.transpose(1, 0, 3, 2, 4).reshape(batch, seq_len, heads, dv)


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

  def __init__(self, *args, gate_in_features: int, **kwargs):
    super().__init__(*args, **kwargs)
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

    # Layer 0 is dense with a wide SwiGLU; MoE layers keep a narrow shared expert.
    self.is_dense_layer = layer_idx < cfg.first_num_dense_layers
    shared_dim = cfg.mlp_dim if self.is_dense_layer else cfg.moe_mlp_dim
    self.shared_ffn = MlpBlock(
        config=cfg,
        mesh=mesh,
        in_features=cfg.emb_dim,
        intermediate_dim=shared_dim,
        activations=cfg.mlp_activations,
        kernel_init=max_initializers.nd_dense_init(cfg.dense_init_scale, "fan_in", "truncated_normal"),
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
          kernel_init=max_initializers.nd_dense_init(cfg.dense_init_scale, "fan_in", "truncated_normal"),
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

    ffn_in = self.ffn_in_norm(hidden)
    ffn_out = self.shared_ffn(ffn_in, deterministic=deterministic)
    if self.moe_block is not None:
      routed, load_balance_loss, _ = self.moe_block(self.latent_down(ffn_in), gate_inputs=ffn_in)
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
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant=None,
      *,
      first_layer_idx: int = 0,
      rngs: nnx.Rngs,
  ):
    self.config = config
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
      x, _ = getattr(self, f"layer_{i}")(
          x,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk,
          slot,
      )
    return x, None


OLMoE3DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    OLMoE3DecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

OLMoE3ScannableBlockToLinen = nnx_wrappers.to_linen_class(
    OLMoE3ScannableBlock,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

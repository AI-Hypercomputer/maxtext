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

"""pi0.5 (Physical Intelligence) vision-language-action model on top of the Gemma decoder.

Reference: openpi (https://github.com/Physical-Intelligence/openpi) and its LeRobot port
(``lerobot.policies.pi05``), checkpoint ``lerobot/pi05_base``.

The policy is a PaliGemma VLM (SigLIP So400m/14 at 224 px + Gemma-2B) plus a Gemma-300M "action expert":

* Prefix = ``[image_0 (256), image_1 (256), image_2 (256), text (200 incl. discretized state)]``. All valid prefix
  tokens attend to each other bidirectionally (prefix-LM with an empty causal part).
* Suffix = ``action_horizon`` (50) noisy action tokens, embedded by ``action_in_proj``. They attend to every valid
  prefix token and bidirectionally to each other; prefix tokens never attend to the suffix.
* Every layer is a *joint* layer: each stream keeps its own norms / q,k,v,o / MLP weights (VLM: width 2048, expert:
  width 1024, both 8 query heads x 256, 1 KV head) and the two streams share one softmax.
* The expert's RMSNorms are adaptive (adaRMS): ``dense(time_cond) -> (scale, shift, gate)``,
  ``out = norm(x) * (1 + scale) + shift`` and the residual is gated: ``x + gate * f(out)``.
* Actions are sampled by flow matching with noise at t = 1: ``x <- x - v(x, t) / n`` for ``t = 1, 1 - 1/n, ...``.

``Pi05DecoderLayer`` behaves exactly like ``GemmaDecoderLayer`` unless the pi0.5 helpers below drive it, so the
standard text path of the Gemma backbone keeps working.
"""

# pylint: disable=arguments-differ,too-many-positional-arguments

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from maxtext.common.common_types import MODEL_MODE_TRAIN, Array, Config
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import DenseGeneral, MlpBlock
from maxtext.models.gemma import GemmaDecoderLayer
from maxtext.utils import max_utils

# openpi / LeRobot additive mask value (``OPENPI_ATTENTION_MASK_VALUE``); kept identical so that fully masked rows
# (padding queries) produce the same uniform softmax as the reference.
ATTENTION_MASK_VALUE = -2.3819763e38


def _cfg(config, name, default):
  return getattr(config, name, default)


class Pi05AdaRMSNorm(nnx.Module):
  """Adaptive RMSNorm (``PiGemmaRMSNorm`` with ``cond_dim``): returns ``(normed, gate)``.

  ``normed = x * rsqrt(mean(x^2) + eps) * (1 + scale) + shift`` with ``(scale, shift, gate) = dense(cond)``.
  The variance and the modulation are computed in float32 like the reference.
  """

  def __init__(self, dim: int, cond_dim: int, epsilon: float, dtype, weight_dtype, matmul_precision, *, rngs):
    self.dim = dim
    self.epsilon = epsilon
    self.dense = DenseGeneral(
        in_features_shape=cond_dim,
        out_features_shape=3 * dim,
        axis=-1,
        dtype=dtype,
        weight_dtype=weight_dtype,
        kernel_axes=("embed", "mlp"),
        use_bias=True,
        matmul_precision=matmul_precision,
        rngs=rngs,
    )

  def __call__(self, x: Array, cond: Array) -> tuple[Array, Array]:
    """x: [B, S, dim]; cond: [B, cond_dim] (shared over tokens) or [B, S, cond_dim] (per token)."""
    modulation = self.dense(cond.astype(x.dtype))
    if modulation.ndim == 2:
      modulation = modulation[:, None, :]
    scale, shift, gate = jnp.split(modulation, 3, axis=-1)
    var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normed = x.astype(jnp.float32) * jax.lax.rsqrt(var + self.epsilon)
    out = normed * (1.0 + scale.astype(jnp.float32)) + shift.astype(jnp.float32)
    return out.astype(x.dtype), gate.astype(x.dtype)


def apply_rope(x: Array, positions: Array, base: float) -> Array:
  """Gemma / HF rotary embedding (rotate_half), with the HF angle arithmetic.

  ``angle = position * inv_freq`` with ``inv_freq = 1 / base**(2i/d)`` in float32, exactly like
  ``GemmaRotaryEmbedding``. MaxText's ``RotaryEmbedding`` computes ``position / timescale`` instead, which differs by
  ~1 ulp of the angle (~6e-5 rad at position ~1000) and is visible in fp32 parity over the 968-token pi0.5 prefix.
  """
  head_dim = x.shape[-1]
  inv_freq = 1.0 / (np.float32(base) ** (np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32) / head_dim))
  freqs = positions.astype(jnp.float32)[:, :, None] * jnp.asarray(inv_freq, jnp.float32)[None, None, :]  # [B,S,D/2]
  emb = jnp.concatenate([freqs, freqs], axis=-1)[:, :, None, :]
  cos, sin = jnp.cos(emb), jnp.sin(emb)
  x32 = x.astype(jnp.float32)
  x1, x2 = jnp.split(x32, 2, axis=-1)
  rotated = jnp.concatenate([-x2, x1], axis=-1)
  return (x32 * cos + rotated * sin).astype(x.dtype)


def _qkv(attn: Attention, hidden: Array, positions: Array, rope_base: float) -> tuple[Array, Array, Array]:
  """Projections + RoPE of one stream. The 1/sqrt(head_dim) scale is folded into the query kernel."""
  query = attn.query_projection(hidden)
  key = attn.kv_projection(hidden, proj_name="key")
  value = attn.kv_projection(hidden, proj_name="value")
  query = apply_rope(query, positions, rope_base)
  key = apply_rope(key, positions, rope_base)
  return query, key, value


def _attend(query: Array, key: Array, value: Array, bias: Array) -> Array:
  """Grouped-query softmax attention. query [B,Q,H,D]; key/value [B,K,G,D]; bias [B,Q,K] additive (float32)."""
  b, q_len, num_heads, head_dim = query.shape
  num_kv = key.shape[2]
  rep = num_heads // num_kv
  q = query.reshape(b, q_len, num_kv, rep, head_dim).astype(jnp.float32)
  k = key.astype(jnp.float32)
  v = value.astype(jnp.float32)
  logits = jnp.einsum("bqgrd,bkgd->bgrqk", q, k, precision=jax.lax.Precision.HIGHEST)
  logits = logits + bias[:, None, None, :, :]
  probs = jax.nn.softmax(logits, axis=-1)
  out = jnp.einsum("bgrqk,bkgd->bqgrd", probs, v, precision=jax.lax.Precision.HIGHEST)
  return out.reshape(b, q_len, num_heads, head_dim).astype(query.dtype)


class Pi05DecoderLayer(GemmaDecoderLayer):
  """Gemma-2B decoder layer (the VLM stream) + the matching Gemma-300M action-expert layer.

  Called with the standard decoder arguments it is exactly a ``GemmaDecoderLayer``. The pi0.5 forward uses
  ``prefix_forward`` (VLM stream only, returns the rotated K/V for the suffix) and ``suffix_forward`` (action stream
  attending to the cached prefix K/V and to itself), which together equal the reference joint layer.
  """

  def __init__(self, config: Config, mesh, model_mode: str, quant=None, *, rngs: nnx.Rngs):
    super().__init__(config, mesh, model_mode, quant, rngs=rngs)
    cfg = config
    width = cfg.action_hidden_size
    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(cfg, model_mode)
    action_shape = (batch_size, seq_len, width)
    norm_kwargs = {
        "cond_dim": width,
        "epsilon": cfg.normalization_layer_epsilon,
        "dtype": cfg.dtype,
        "weight_dtype": cfg.weight_dtype,
        "matmul_precision": cfg.matmul_precision,
        "rngs": rngs,
    }
    self.pre_self_attention_norm_action = Pi05AdaRMSNorm(width, **norm_kwargs)
    self.self_attention_action = Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        mesh=mesh,
        attention_kernel="dot_product",
        inputs_q_shape=action_shape,
        inputs_kv_shape=action_shape,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        quant=quant,
        model_mode=MODEL_MODE_TRAIN,  # never owns a KV cache
        base_kv_cache=False,
        rngs=rngs,
    )
    self.pre_ffw_norm_action = Pi05AdaRMSNorm(width, **norm_kwargs)
    self.mlp_action = MlpBlock(
        config=cfg,
        mesh=mesh,
        in_features=width,
        intermediate_dim=cfg.action_mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=quant,
        model_mode=model_mode,
        rngs=rngs,
    )

  # ---------------------------------------------------------------------------------------------
  def prefix_forward(self, x: Array, positions: Array, bias: Array) -> tuple[Array, tuple[Array, Array]]:
    """VLM stream with an explicit additive mask. Returns (x_out [B,P,E], (key, value) after RoPE)."""
    residual = x
    hidden = self.pre_self_attention_norm(x)
    query, key, value = _qkv(self.self_attention, hidden, positions, self.config.rope_max_timescale)
    out = _attend(query, key, value, bias)
    x = residual + self.self_attention.out_projection(out)
    hidden = self.pre_ffw_norm(x)
    x = x + self.mlp(hidden, deterministic=True)
    return x, (key, value)

  def suffix_forward(self, a: Array, cond: Array, positions: Array, prefix_kv: tuple[Array, Array], bias: Array) -> Array:
    """Action stream: queries of the suffix attend to [cached prefix K/V, suffix K/V]. bias [B, A, P + A]."""
    residual = a
    hidden, gate = self.pre_self_attention_norm_action(a, cond)
    query, key, value = _qkv(self.self_attention_action, hidden, positions, self.config.rope_max_timescale)
    key = jnp.concatenate([prefix_kv[0].astype(key.dtype), key], axis=1)
    value = jnp.concatenate([prefix_kv[1].astype(value.dtype), value], axis=1)
    out = _attend(query, key, value, bias)
    a = residual + self.self_attention_action.out_projection(out) * gate
    residual = a
    hidden, gate = self.pre_ffw_norm_action(a, cond)
    a = residual + self.mlp_action(hidden, deterministic=True) * gate
    return a


# -------------------------------------------------------------------------------------------------
# Model-level action head (flow-matching projections, time MLP, expert final norm)
# -------------------------------------------------------------------------------------------------
class Pi05ActionHead(nnx.Module):
  """``action_in_proj``, ``time_mlp_in/out``, ``action_out_proj`` and the expert's final adaptive norm."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    cfg = config
    self.config = cfg
    width = cfg.action_hidden_size
    dense = {"dtype": cfg.dtype, "weight_dtype": cfg.weight_dtype, "matmul_precision": cfg.matmul_precision, "rngs": rngs}
    # The action-space projections are tiny (32 features); keep them replicated so any mesh works.
    self.action_in_proj = DenseGeneral(cfg.action_dim, width, kernel_axes=(None, None), use_bias=True, **dense)
    self.time_mlp_in = DenseGeneral(width, width, kernel_axes=("embed", "mlp"), use_bias=True, **dense)
    self.time_mlp_out = DenseGeneral(width, width, kernel_axes=("embed", "mlp"), use_bias=True, **dense)
    self.action_out_proj = DenseGeneral(width, cfg.action_dim, kernel_axes=(None, None), use_bias=True, **dense)
    self.norm_action = Pi05AdaRMSNorm(
        width,
        cond_dim=width,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

  def time_cond(self, time_emb: Array) -> Array:
    x = jax.nn.silu(self.time_mlp_in(time_emb.astype(jnp.float32)))
    return jax.nn.silu(self.time_mlp_out(x))


def sinusoidal_time_embedding(t, dim: int, min_period: float, max_period: float) -> Array:
  """openpi ``create_sinusoidal_pos_embedding``: [sin(2*pi*t/period), cos(...)], period log-spaced.

  With a concrete (numpy / python) ``t`` the embedding is computed in float64 on the host, exactly like the
  reference (which runs it in float64); with a traced ``t`` it is computed in float32.
  """
  fraction = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
  period = min_period * (max_period / min_period) ** fraction
  scale = 1.0 / period * 2.0 * math.pi
  if isinstance(t, (float, int, np.ndarray, np.floating)):
    arg = np.asarray(t, np.float64)[..., None] * scale
    return jnp.asarray(np.concatenate([np.sin(arg), np.cos(arg)], axis=-1).astype(np.float32))
  arg = t.astype(jnp.float32)[..., None] * jnp.asarray(scale, jnp.float32)
  return jnp.concatenate([jnp.sin(arg), jnp.cos(arg)], axis=-1)


# -------------------------------------------------------------------------------------------------
# Prefix / suffix construction
# -------------------------------------------------------------------------------------------------
def embed_images(model, images: Array) -> Array:
  """images [num_cam, B, H, W, 3] in [-1, 1] -> [num_cam, B, 256, emb_dim] (SigLIP + linear projector)."""
  num_cam, batch = images.shape[:2]
  flat = images.reshape((num_cam * batch,) + images.shape[2:])
  emb, _ = model.vision_encoder(flat, deterministic=True)  # [num_cam*B, 1, 256, E]
  emb = emb.reshape(num_cam, batch, emb.shape[-2], emb.shape[-1])
  return emb


def embed_prefix(model, images: Array, img_masks: Array, tokens: Array, token_masks: Array):
  """Returns (prefix_embs [B,P,E], prefix_pad [B,P] bool). ``img_masks`` [num_cam, B]."""
  cfg = model.config
  img_emb = embed_images(model, images).astype(cfg.dtype)
  num_cam, batch, n_img, emb_dim = img_emb.shape
  img_emb = jnp.transpose(img_emb, (1, 0, 2, 3)).reshape(batch, num_cam * n_img, emb_dim)
  img_pad = jnp.repeat(jnp.transpose(img_masks.astype(bool), (1, 0)), n_img, axis=1)  # [B, num_cam*n_img]
  # Gemma token embeddings are stored pre-multiplied by sqrt(emb_dim) (MaxText convention), matching
  # ``GemmaTextScaledWordEmbedding`` in the reference.
  txt_emb = model.token_embedder(tokens, model_mode=MODEL_MODE_TRAIN).astype(cfg.dtype)
  embs = jnp.concatenate([img_emb, txt_emb], axis=1)
  pad = jnp.concatenate([img_pad, token_masks.astype(bool)], axis=1)
  return embs, pad


def prefix_attention_bias(prefix_pad: Array) -> Array:
  """All valid prefix tokens attend to all valid prefix tokens (``make_att_2d_masks`` with att_masks = 0)."""
  allowed = prefix_pad[:, None, :] & prefix_pad[:, :, None]
  return jnp.where(allowed, 0.0, ATTENTION_MASK_VALUE).astype(jnp.float32)


def suffix_attention_bias(prefix_pad: Array, horizon: int) -> Array:
  """Suffix queries attend to valid prefix keys and to every suffix key: [B, A, P + A]."""
  batch, plen = prefix_pad.shape
  prefix_part = jnp.broadcast_to(prefix_pad[:, None, :], (batch, horizon, plen))
  suffix_part = jnp.ones((batch, horizon, horizon), dtype=bool)
  allowed = jnp.concatenate([prefix_part, suffix_part], axis=-1)
  return jnp.where(allowed, 0.0, ATTENTION_MASK_VALUE).astype(jnp.float32)


def _layers(model):
  decoder = model.decoder
  return [getattr(decoder, f"layers_{i}") for i in range(model.config.num_decoder_layers)]


def prefix_forward(model, prefix_embs: Array, prefix_pad: Array, return_hidden: bool = False):
  """Runs the VLM stream over the prefix. Returns (normed_out [B,P,E], kv_cache list[(k, v)][, hidden list])."""
  positions = (jnp.cumsum(prefix_pad.astype(jnp.int32), axis=1) - 1).astype(jnp.int32)
  bias = prefix_attention_bias(prefix_pad)
  x = prefix_embs.astype(model.config.dtype)
  kv_cache = []
  hidden = []
  for layer in _layers(model):
    x, kv = layer.prefix_forward(x, positions, bias)
    kv_cache.append(kv)
    if return_hidden:
      hidden.append(x)
  out = model.decoder.apply_decoder_norm(x, reduce_mhc=False)
  if return_hidden:
    return out, kv_cache, hidden
  return out, kv_cache


def prefix_logits(model, normed_prefix: Array, full_precision: bool = True) -> Array:
  """Tied-embedding logits (PaliGemma ``lm_head``) from already-normed prefix hidden states.

  MaxText's shared-embedding head (``attend_on_embedding``) always casts the table to bfloat16, which costs ~1e-3
  relative logit error. With ``full_precision`` the product is done in the table's dtype (float32 for parity runs).
  The stored table is the HF embedding times sqrt(emb_dim), hence the division.
  """
  if not full_precision:
    return model.decoder.apply_output_head(
        shared_embedding=model.token_embedder,
        y=normed_prefix,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        normalize_y=False,
    )
  table = model.token_embedder.embedding[...]
  logits = jnp.einsum(
      "...e,ve->...v",
      normed_prefix.astype(table.dtype),
      table,
      precision=jax.lax.Precision.HIGHEST,
      preferred_element_type=jnp.float32,
  )
  return logits / jnp.sqrt(jnp.float32(normed_prefix.shape[-1]))


def denoise_step(model, kv_cache, prefix_pad: Array, x_t: Array, time_emb: Array) -> Array:
  """One velocity evaluation v(x_t, t). x_t [B, A, action_dim]; time_emb [B, width] (sinusoidal). Returns [B,A,D]."""
  cfg = model.config
  head = model.action_head
  horizon = x_t.shape[1]
  a = head.action_in_proj(x_t.astype(jnp.float32)).astype(cfg.dtype)
  cond = head.time_cond(time_emb).astype(cfg.dtype)
  offset = jnp.sum(prefix_pad.astype(jnp.int32), axis=-1, keepdims=True)
  positions = (offset + jnp.arange(horizon, dtype=jnp.int32)[None, :]).astype(jnp.int32)
  bias = suffix_attention_bias(prefix_pad, horizon)
  for layer, kv in zip(_layers(model), kv_cache):
    a = layer.suffix_forward(a, cond, positions, kv, bias)
  a, _ = head.norm_action(a, cond)
  return head.action_out_proj(a.astype(jnp.float32)).astype(jnp.float32)


def sample_actions(
    model,
    kv_cache,
    prefix_pad: Array,
    noise: Array,
    num_steps: int | None = None,
    denoise_fn=None,
    return_trajectory: bool = False,
):
  """Euler flow integration from noise (t = 1) to actions (t = 0), mirroring ``euler_integrate``."""
  cfg = model.config
  num_steps = num_steps or cfg.flow_num_inference_steps
  batch = noise.shape[0]
  dt = -1.0 / num_steps
  denoise_fn = denoise_fn or (lambda kv, pad, x, temb: denoise_step(model, kv, pad, x, temb))
  x = noise.astype(jnp.float32)
  traj = [x]
  for step in range(num_steps):
    t = 1.0 + step * dt
    temb = sinusoidal_time_embedding(
        np.full((batch,), t, np.float32), cfg.action_hidden_size, cfg.flow_min_period, cfg.flow_max_period
    )
    v = denoise_fn(kv_cache, prefix_pad, x, temb)
    x = x + dt * v
    traj.append(x)
  if return_trajectory:
    return x, jnp.stack(traj, axis=0)
  return x


def flow_matching_loss(model, images, img_masks, tokens, token_masks, actions, noise, time) -> Array:
  """Training loss (MSE on the velocity u = noise - actions, NOISE_AT_ONE convention). Returns [B, A, D]."""
  cfg = model.config
  t = time[:, None, None]
  x_t = t * noise + (1.0 - t) * actions
  u_t = noise - actions
  embs, pad = embed_prefix(model, images, img_masks, tokens, token_masks)
  kv_cache = prefix_forward(model, embs, pad)[1]
  temb = sinusoidal_time_embedding(time, cfg.action_hidden_size, cfg.flow_min_period, cfg.flow_max_period)
  v_t = denoise_step(model, kv_cache, pad, x_t, temb)
  return jnp.square(u_t - v_t)


def discretize_state(state: np.ndarray, bins: int = 256) -> np.ndarray:
  """openpi ``PaligemmaTokenizer``: normalized state in [-1, 1] -> bin indices 0..bins-1."""
  return np.digitize(state, bins=np.linspace(-1, 1, bins + 1)[:-1]) - 1


def build_prompt(task: str, state: np.ndarray | None) -> str:
  """pi0.5 prompt with the discretized state (``Pi05PrepareStateTokenizerProcessorStep``)."""
  cleaned = task.strip().replace("_", " ").replace("\n", " ")
  if state is None:
    return f"Task: {cleaned};\nAction: "
  state_str = " ".join(map(str, discretize_state(np.asarray(state))))
  return f"Task: {cleaned}, State: {state_str};\nAction: "


__all__: list[Any] = [
    "Pi05AdaRMSNorm",
    "Pi05DecoderLayer",
    "Pi05ActionHead",
    "sinusoidal_time_embedding",
    "embed_images",
    "embed_prefix",
    "prefix_forward",
    "prefix_logits",
    "denoise_step",
    "sample_actions",
    "flow_matching_loss",
    "build_prompt",
    "discretize_state",
]

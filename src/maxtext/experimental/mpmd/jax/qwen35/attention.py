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

"""Qwen3.5's gated full attention (every 4th layer) and its rotary embeddings.

Mirrors `Attention.__call__` with `is_qwen3_hybrid` (MaxText's layers/attentions.py,
wrapped by `Qwen3NextFullAttention` in models/qwen3.py):

    q2 = x @ query  [B,S,H,2D]  -> per head: first D = query, second D = gate
    k, v = x @ key, x @ value   [B,S,Hkv,D]                       (GQA)
    q, k = zero-centred RMSNorm over D, BEFORE the rotary embedding
    q, k = rotary embedding on the first rotary_dim features (text mRoPE or
           partial RoPE, by use_mrope)
    q = q * D^-1/2                (query_pre_attn_scalar; the kernel scale is 1)
    o = attention(q, k, v)        causal + segment mask
    o = o.reshape(B,S,H*D) * sigmoid(gate)
    out = o @ out                 kernel [H*D, E]

The rotary embedding is one of MaxText's two for the qwen3_5 decoder, by its
`use_mrope` (`Qwen35Config.use_mrope`):

  * use_mrope=true (Qwen3.5's ymls; Qwen3OmniMoe text mRoPE, embeddings.py's
    Qwen3OmniMoeThinkerTextRotaryEmbedding): text positions are broadcast to
    the three (t, h, w) sections, so the section interleave is the identity and
    what remains is rotate-half RoPE on the first `rotary_dim` features with
    fp32 cos/sin of `position * (1 / timescale)`, an fp32 rotation and one
    final rounding to `dtype` (`mrope_text`).
  * use_mrope=false (Qwen3.8-2.4T's yml, which leaves it at base.yml's false;
    PartialRotaryEmbedding, Qwen3-Next's): sin/cos of `position / timescale`,
    each rounded to the activation dtype, and the rotation in that dtype
    (`partial_rope`).
The same rotation; in fp32 they part in the last bits (a division against a
multiply by the reciprocal), in bf16 by the roundings.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from maxtext.experimental.mpmd.jax.qwen35 import _ops
from maxtext.experimental.mpmd.jax.qwen35 import layers


def rope_timescale(config):
  """`[rotary_dim // 2]` fp32 timescales, as MaxText's RotaryEmbedding.timescale
  computes them (`rope_inv_freq` is their reciprocal)."""
  rd = config.rotary_dim
  fraction = 2 * jnp.arange(0, rd // 2) / rd
  return config.rope_min_timescale * (config.rope_max_timescale / config.rope_min_timescale) ** fraction


def rope_inv_freq(config):
  """`[rotary_dim // 2]` fp32 inverse frequencies, computed as MaxText does.

  Traced (not numpy), so inside a jit XLA constant-folds it exactly as it folds
  MaxText's in-graph copy; called eagerly it is the same folded constant (a
  test pins the two bitwise).  A caller that computes it once
  (`rope_inv_freq_jit`) passes it as `model.forward`'s `inv_freq`.
  """
  return 1.0 / rope_timescale(config)


rope_inv_freq_jit = jax.jit(rope_inv_freq, static_argnums=0)


def mrope_text(inputs, positions, inv_freq, config):
  """Rotate-half RoPE on `inputs[..., :rotary_dim]`, fp32 cos/sin, rounded once.

  inputs `[B, S, N, D]` (`dtype`), positions `[B, S]` int, inv_freq
  `[rotary_dim // 2]` fp32.

  The cos/sin tables are written per call -- for q and for k, in every
  attention layer -- as MaxText writes them, and XLA folds that: in the
  optimized program of a scanned 4-block forward (CPU, jax 0.10.2) there is
  one cosine and one sine, shared by q and k and hoisted out of the block
  scan, plus one more of each in a remat backward's replay.  Building the
  tables once per forward by hand would buy nothing and change the program
  (a checkpointed scan would then keep them as residuals), so it is not done
  (review finding 10).
  """
  rd = config.rotary_dim
  freqs = positions[..., jnp.newaxis] * inv_freq[jnp.newaxis, jnp.newaxis, :]  # [B, S, rd/2] fp32
  emb = jnp.concatenate([freqs, freqs], axis=-1)
  cos = jnp.cos(emb)[:, :, jnp.newaxis, :]
  sin = jnp.sin(emb)[:, :, jnp.newaxis, :]
  x_rot, x_pass = jnp.split(inputs, [rd], axis=-1)
  x1, x2 = jnp.split(x_rot, 2, axis=-1)
  rotated = x_rot * cos + jnp.concatenate((-x2, x1), axis=-1) * sin
  return jnp.concatenate([rotated, x_pass], axis=-1).astype(config.dtype)


def partial_rope(inputs, positions, config):
  """MaxText's PartialRotaryEmbedding (use_mrope=false: Qwen3.8-2.4T, Qwen3-Next)
  on `inputs[..., :rotary_dim]`, op for op: sin / cos of `position / timescale`
  in fp32, each rounded to the inputs' dtype, the rotate-half rotation in that
  dtype, the pass-through features appended, the result in `dtype`.

  inputs `[B, S, N, D]`, positions `[B, S]` int."""
  rd = config.rotary_dim
  x_rot, x_pass = jnp.split(inputs, [rd], axis=-1)
  sinusoid = positions[:, :, jnp.newaxis, jnp.newaxis] / rope_timescale(config)
  sin_half = jnp.sin(sinusoid).astype(x_rot.dtype)
  cos_half = jnp.cos(sinusoid).astype(x_rot.dtype)
  sin = jnp.concatenate([sin_half, sin_half], axis=-1)
  cos = jnp.concatenate([cos_half, cos_half], axis=-1)
  x1, x2 = jnp.split(x_rot, 2, axis=-1)
  rotated = (x_rot * cos) + (jnp.concatenate((-x2, x1), axis=-1) * sin)
  return jnp.concatenate([rotated, x_pass], axis=-1).astype(config.dtype)


def rope(inputs, positions, inv_freq, config):
  """The rotary embedding `config.use_mrope` names (the module docstring):
  `mrope_text` on `inv_freq` (None: computed in-graph), or `partial_rope`."""
  if not config.use_mrope:
    return partial_rope(inputs, positions, config)
  return mrope_text(inputs, positions, rope_inv_freq(config) if inv_freq is None else inv_freq, config)


def gated_attention(params, x, positions, segment_ids, config, inv_freq=None):
  """`[B, S, E]` -> `[B, S, E]`.  `inv_freq=None` computes it in-graph (the
  mRoPE path's; the partial RoPE computes its timescales in-graph always)."""
  b, s, _ = x.shape
  h, d = config.num_query_heads, config.head_dim
  q = _ops.dense(params["query"], x, config)  # [B, S, H, 2D]
  k = _ops.dense(params["key"], x, config)  # [B, S, Hkv, D]
  v = _ops.dense(params["value"], x, config)
  q, gate = jnp.split(q, 2, axis=-1)
  gate = gate.reshape(b, s, h * d)
  q = layers.rms_norm(params["query_norm"], q, config)
  k = layers.rms_norm(params["key_norm"], k, config)
  q = rope(q, positions, inv_freq, config)
  k = rope(k, positions, inv_freq, config)
  q = q * (config.head_dim**-0.5)
  q = jax.ad_checkpoint.checkpoint_name(q, "query_proj")
  k = jax.ad_checkpoint.checkpoint_name(k, "key_proj")
  v = jax.ad_checkpoint.checkpoint_name(v, "value_proj")
  out = _ops.attention(q, k, v, segment_ids, config)  # [B, S, H, D]
  out = jax.ad_checkpoint.checkpoint_name(out, "attention_out")
  out = out.reshape(b, s, h * d) * jax.nn.sigmoid(gate)
  out = _ops.dense(params["out"], out, config)
  return jax.ad_checkpoint.checkpoint_name(out, "out_proj")

# Copyright 2023–2026 Google LLC
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

"""inclusionAI Ming-3.0-audio-tiny talker (text -> speech) heads.

Reference: https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny, subfolder `talker/`
(`inference.py`, `talker_module/{cfm,dit,aggregator,modules}.py`, `AudioVAE/*`).

The talker is a continuous-latent autoregressive TTS model:

  * LLM: a Qwen2 decoder (24 layers, hidden 896). In MaxText this is the regular `qwen2` decoder of
    `models.Transformer` fed with `decoder_input_embeddings`; its final-normed last hidden state conditions
    the flow-matching head below.
  * CFM (`MingTalkerDiT` + `cfm_sample`): a DiT velocity network integrated with 10 Euler steps and
    classifier-free guidance. It turns the LLM hidden state + the previous latent into one 64-dim latent
    (`patch_size` = 1 latent per step).
  * Aggregator (`MingTalkerAggregator`): embeds the sampled latent back into an LLM input embedding for
    the next autoregressive step.
  * Stop head: `Linear(896, 2)`; generation stops when `softmax[1] > 0.5` (after the first few steps).
  * AudioVAE decoder (`MingAudioVAEDecoder`): latents -> linear x4 upsampling -> sliding-window causal Qwen2
    backbone -> ISTFT head -> 44.1 kHz waveform (882 samples per upsampled frame, 3528 per latent).

These modules are owned by `models.Transformer` as `ming_talker` when `config.ming_talker` is true. The
talker's AudioVAE *encoder* and the CAM++ speaker encoder are only used for voice cloning and are not ported.
"""

# pylint: disable=arguments-differ

import math

from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from maxtext.common.common_types import Array, Config
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.normalizations import RMSNorm

# Timesteps of the publisher's `get_epss_timesteps` (in units of 1/32) for the supported step counts.
_EPSS_TIMESTEPS = {
    5: [0, 2, 4, 8, 16, 32],
    6: [0, 2, 4, 6, 8, 16, 32],
    7: [0, 2, 4, 6, 8, 16, 24, 32],
    10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
    12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
}
# `CFM.sway_sampling_coef`.
_SWAY_SAMPLING_COEF = -1.0


def bf16_round(x) -> np.ndarray:
  """Rounds float32 values to bfloat16 precision (the publisher's checkpoint stores these buffers in bf16)."""
  # Pure numpy (ml_dtypes) so it also works on trace-time constants inside jit.
  return np.asarray(x, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def epss_timesteps(steps: int) -> np.ndarray:
  """`get_epss_timesteps` followed by CFM sway sampling, as float32 numpy array of length `steps + 1`."""
  if steps in _EPSS_TIMESTEPS:
    t = np.asarray(_EPSS_TIMESTEPS[steps], np.float32) * np.float32(1 / 32)
  else:
    t = np.linspace(0, 1, steps + 1, dtype=np.float32)
  return (t + _SWAY_SAMPLING_COEF * (np.cos(np.pi / 2 * t) - 1 + t)).astype(np.float32)


def dit_rope_freqs(seq_len: int, dim: int, base: float = 10000.0) -> np.ndarray:
  """x-transformers `RotaryEmbedding(dim).forward_from_seq_len`: interleaved frequencies `[seq_len, dim]`.

  The checkpoint persists `inv_freq` in bfloat16, so the frequencies are rounded the same way.
  """
  inv_freq = bf16_round(1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim))))
  freqs = np.arange(seq_len, dtype=np.float32)[:, None] * inv_freq[None, :]
  return np.repeat(freqs, 2, axis=-1)


def apply_interleaved_rope(x: Array, freqs: Array) -> Array:
  """x-transformers `apply_rotary_pos_emb` on `[B, T, H, D]` with interleaved pairs `(x[2i], x[2i+1])`."""
  x32 = x.astype(jnp.float32)
  pairs = x32.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
  rotated = jnp.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)
  cos = jnp.cos(freqs)[None, :, None, :]
  sin = jnp.sin(freqs)[None, :, None, :]
  return (x32 * cos + rotated * sin).astype(x.dtype)


def qwen2_rope(x: Array, positions: Array, theta: float) -> Array:
  """Qwen2 (non-interleaved, rotate-half) rotary embedding on `[B, T, H, D]` with positions `[T]`."""
  dim = x.shape[-1]
  inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
  freqs = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
  emb = jnp.concatenate([freqs, freqs], axis=-1)
  cos = jnp.cos(emb)[None, :, None, :]
  sin = jnp.sin(emb)[None, :, None, :]
  x32 = x.astype(jnp.float32)
  x1, x2 = x32[..., : dim // 2], x32[..., dim // 2 :]
  rotated = jnp.concatenate([-x2, x1], axis=-1)
  return (x32 * cos + rotated * sin).astype(x.dtype)


def _dense(config: Config, in_features: int, out_features: int, kernel_axes, use_bias: bool, rngs: nnx.Rngs):
  return DenseGeneral(
      in_features_shape=in_features,
      out_features_shape=out_features,
      use_bias=use_bias,
      dtype=config.dtype,
      weight_dtype=config.weight_dtype,
      matmul_precision=config.matmul_precision,
      kernel_axes=kernel_axes,
      rngs=rngs,
  )


def _rms_norm(config: Config, num_features: int, rngs: nnx.Rngs, eps: float = 1e-6):
  return RMSNorm(
      num_features=num_features,
      epsilon=eps,
      dtype=config.dtype,
      weight_dtype=config.weight_dtype,
      kernel_axes=("norm",),
      rngs=rngs,
  )


def _attention(q: Array, k: Array, v: Array, mask: Array | None = None) -> Array:
  """Softmax attention in float32 on `[B, T, H, D]` (k/v may have fewer heads: grouped-query attention)."""
  num_heads, num_kv_heads, head_dim = q.shape[2], k.shape[2], q.shape[3]
  if num_kv_heads != num_heads:
    k = jnp.repeat(k, num_heads // num_kv_heads, axis=2)
    v = jnp.repeat(v, num_heads // num_kv_heads, axis=2)
  hp = lax.Precision.HIGHEST
  logits = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32), precision=hp)
  logits = logits * (head_dim**-0.5)
  if mask is not None:
    logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)
  probs = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum("bhqk,bkhd->bqhd", probs, v.astype(jnp.float32), precision=hp)


# -----------------------------------------
# DiT blocks shared by the CFM head and the aggregator
# -----------------------------------------


class MingTalkerDiTAttention(nnx.Module):
  """`talker_module.modules.Attention`: bidirectional MHA with biases and interleaved RoPE on all heads."""

  def __init__(self, config: Config, hidden_size: int, num_heads: int, *, rngs: nnx.Rngs):
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    self.to_q = _dense(config, hidden_size, hidden_size, ("embed", "heads"), True, rngs)
    self.to_k = _dense(config, hidden_size, hidden_size, ("embed", "heads"), True, rngs)
    self.to_v = _dense(config, hidden_size, hidden_size, ("embed", "heads"), True, rngs)
    self.to_out = _dense(config, hidden_size, hidden_size, ("heads", "embed"), True, rngs)

  def __call__(self, x: Array, rope_freqs: Array) -> Array:
    b, t, d = x.shape
    q = self.to_q(x).reshape(b, t, self.num_heads, self.head_dim)
    k = self.to_k(x).reshape(b, t, self.num_heads, self.head_dim)
    v = self.to_v(x).reshape(b, t, self.num_heads, self.head_dim)
    q = apply_interleaved_rope(q, rope_freqs)
    k = apply_interleaved_rope(k, rope_freqs)
    o = _attention(q, k, v)
    return self.to_out(o.reshape(b, t, d).astype(x.dtype))


class MingTalkerDiTBlock(nnx.Module):
  """`talker_module.modules.DiTBlock`: pre-RMSNorm attention + pre-RMSNorm GELU(tanh) MLP."""

  def __init__(self, config: Config, hidden_size: int, num_heads: int, mlp_ratio: int, *, rngs: nnx.Rngs):
    inner = int(hidden_size * mlp_ratio)
    self.norm1 = _rms_norm(config, hidden_size, rngs)
    self.attn = MingTalkerDiTAttention(config, hidden_size, num_heads, rngs=rngs)
    self.norm2 = _rms_norm(config, hidden_size, rngs)
    self.mlp_in = _dense(config, hidden_size, inner, ("embed", "mlp"), True, rngs)
    self.mlp_out = _dense(config, inner, hidden_size, ("mlp", "embed"), True, rngs)

  def __call__(self, x: Array, rope_freqs: Array) -> Array:
    x = x + self.attn(self.norm1(x), rope_freqs)
    x = x + self.mlp_out(jax.nn.gelu(self.mlp_in(self.norm2(x)), approximate=True))
    return x


class MingTalkerDiT(nnx.Module):
  """`talker_module.dit.DiT`: flow-matching velocity network conditioned on the LLM hidden state.

  Input sequence: `[t_embed + cond_embed, x_embed(history latents), x_embed(noisy latents)]`.
  """

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    hidden = config.ming_talker_dit_hidden_size
    latent = config.ming_talker_latent_dim
    self.num_heads = config.ming_talker_dit_num_heads
    self.freq_embed_dim = 256
    self.x_embedder = _dense(config, latent, hidden, ("mlp", "embed"), True, rngs)
    self.c_embedder = _dense(config, config.emb_dim, hidden, ("mlp", "embed"), True, rngs)
    self.time_fc1 = _dense(config, self.freq_embed_dim, hidden, ("mlp", "embed"), True, rngs)
    self.time_fc2 = _dense(config, hidden, hidden, ("mlp", "embed"), True, rngs)
    for i in range(config.ming_talker_dit_num_layers):
      setattr(
          self,
          f"blocks_{i}",
          MingTalkerDiTBlock(config, hidden, self.num_heads, config.ming_talker_dit_mlp_ratio, rngs=rngs),
      )
    self.final_norm = _rms_norm(config, hidden, rngs)
    self.final_linear = _dense(config, hidden, latent, ("embed", "mlp"), True, rngs)

  def timestep_embedding(self, t: Array) -> Array:
    """`SinusPositionEmbedding(256)` (scale 1000) followed by `time_mlp` (Linear, SiLU, Linear)."""
    half = self.freq_embed_dim // 2
    freqs = jnp.exp(jnp.arange(half, dtype=jnp.float32) * -(math.log(10000) / (half - 1)))
    args = 1000.0 * t.astype(jnp.float32)[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1).astype(self.config.dtype)
    return self.time_fc2(jax.nn.silu(self.time_fc1(emb)))

  def __call__(self, x: Array, t: Array, cond: Array, history: Array) -> Array:
    """Velocity for the noisy latents.

    Args:
      x: `[B, P, latent]` noisy latents.
      t: `[B]` flow time in [0, 1].
      cond: `[B, 1, emb]` LLM hidden state (zeros for the unconditional CFG branch).
      history: `[B, H, latent]` previous latents.

    Returns:
      `[B, P, latent]` predicted velocity.
    """
    num_x = x.shape[1]
    h = self.x_embedder(jnp.concatenate([history, x], axis=1).astype(self.config.dtype))
    y = self.timestep_embedding(t)[:, None, :] + self.c_embedder(cond.astype(self.config.dtype))
    h = jnp.concatenate([y.astype(h.dtype), h], axis=1)
    freqs = jnp.asarray(dit_rope_freqs(h.shape[1], h.shape[-1] // self.num_heads))
    for i in range(self.config.ming_talker_dit_num_layers):
      h = getattr(self, f"blocks_{i}")(h, freqs)
    out = self.final_linear(self.final_norm(h))
    return out[:, -num_x:]

  def forward_with_cfg(self, x: Array, t: Array, cond: Array, history: Array, cfg_strength) -> Array:
    """Classifier-free guidance: `pred + (pred - pred_uncond) * cfg_strength` (one batched DiT call)."""
    batch = x.shape[0]
    out = self(
        jnp.concatenate([x, x], axis=0),
        jnp.broadcast_to(t, (2 * batch,)),
        jnp.concatenate([cond, jnp.zeros_like(cond)], axis=0),
        jnp.concatenate([history, history], axis=0),
    )
    pred, null_pred = out[:batch], out[batch:]
    return pred + (pred - null_pred) * cfg_strength


def cfm_sample(dit: MingTalkerDiT, cond, history, noise, steps: int, cfg_strength, sigma, temperature, sde_noise=None):
  """`CFM.sample`: Euler integration of the guided velocity field from `noise` (t=0) to a latent (t=1).

  Args:
    dit: velocity network.
    cond: `[B, 1, emb]` LLM hidden state.
    history: `[B, H, latent]` previous latents.
    noise: `[B, P, latent]` initial Gaussian noise.
    steps: number of Euler steps.
    cfg_strength, sigma, temperature: CFG scale and SDE noise parameters (`sde_args`).
    sde_noise: optional `[steps, B, P, latent]` SDE noise; only used when `sigma * sqrt(temperature) != 0`.

  Returns:
    `[B, P, latent]` sampled latent (float32).
  """
  t = jnp.asarray(epss_timesteps(steps))
  y = noise.astype(jnp.float32)
  for step in range(steps):
    dt = t[step + 1] - t[step]
    velocity = dit.forward_with_cfg(y, t[step], cond, history, cfg_strength).astype(jnp.float32)
    y = y + velocity * dt
    if sde_noise is not None:
      y = y + sigma * jnp.sqrt(temperature) * jnp.sqrt(jnp.abs(dt)) * sde_noise[step].astype(jnp.float32)
  return y


class MingTalkerAggregator(nnx.Module):
  """`talker_module.aggregator.Aggregator`: latents `[B, P, latent]` -> next LLM input embedding `[B, 1, emb]`."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    hidden = config.ming_talker_aggregator_hidden_size
    self.num_heads = config.ming_talker_aggregator_num_heads
    self.word_embedder = nnx.Param(
        jax.nn.initializers.normal(stddev=0.02)(rngs.params(), (1, hidden), config.weight_dtype),
        out_sharding=(None, "embed"),
    )
    self.x_embedder = _dense(config, config.ming_talker_latent_dim, hidden, ("mlp", "embed"), True, rngs)
    for i in range(config.ming_talker_aggregator_num_layers):
      setattr(
          self,
          f"blocks_{i}",
          MingTalkerDiTBlock(config, hidden, self.num_heads, config.ming_talker_aggregator_mlp_ratio, rngs=rngs),
      )
    self.final_norm = _rms_norm(config, hidden, rngs)
    self.final_linear = _dense(config, hidden, config.emb_dim, ("embed", "mlp"), True, rngs)

  def __call__(self, latents: Array) -> Array:
    h = self.x_embedder(latents.astype(self.config.dtype))
    cls = jnp.broadcast_to(self.word_embedder[...].astype(h.dtype)[None], (h.shape[0], 1, h.shape[-1]))
    h = jnp.concatenate([cls, h], axis=1)
    freqs = jnp.asarray(dit_rope_freqs(h.shape[1], h.shape[-1] // self.num_heads))
    for i in range(self.config.ming_talker_aggregator_num_layers):
      h = getattr(self, f"blocks_{i}")(h, freqs)
    return self.final_linear(self.final_norm(h))[:, :1]


# -----------------------------------------
# AudioVAE decoder
# -----------------------------------------


class MingVAEDecoderLayer(nnx.Module):
  """Qwen2 decoder layer (q/k/v biases, GQA, SwiGLU) with a causal sliding-window mask."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    d = config.ming_vae_hidden_size
    self.num_heads = config.ming_vae_num_heads
    self.num_kv_heads = config.ming_vae_num_kv_heads
    self.head_dim = d // self.num_heads
    kv_dim = self.num_kv_heads * self.head_dim
    eps = config.ming_vae_rms_norm_eps
    self.input_layernorm = _rms_norm(config, d, rngs, eps)
    self.q_proj = _dense(config, d, d, ("embed", "heads"), True, rngs)
    self.k_proj = _dense(config, d, kv_dim, ("embed", "kv_heads"), True, rngs)
    self.v_proj = _dense(config, d, kv_dim, ("embed", "kv_heads"), True, rngs)
    self.o_proj = _dense(config, d, d, ("heads", "embed"), False, rngs)
    self.post_attention_layernorm = _rms_norm(config, d, rngs, eps)
    self.gate_proj = _dense(config, d, config.ming_vae_mlp_dim, ("embed", "mlp"), False, rngs)
    self.up_proj = _dense(config, d, config.ming_vae_mlp_dim, ("embed", "mlp"), False, rngs)
    self.down_proj = _dense(config, config.ming_vae_mlp_dim, d, ("mlp", "embed"), False, rngs)

  def __call__(self, x: Array, positions: Array, mask: Array) -> Array:
    b, t, d = x.shape
    h = self.input_layernorm(x)
    q = self.q_proj(h).reshape(b, t, self.num_heads, self.head_dim)
    k = self.k_proj(h).reshape(b, t, self.num_kv_heads, self.head_dim)
    v = self.v_proj(h).reshape(b, t, self.num_kv_heads, self.head_dim)
    theta = self.config.ming_vae_rope_theta
    q = qwen2_rope(q, positions, theta)
    k = qwen2_rope(k, positions, theta)
    o = _attention(q, k, v, mask[None, None])
    x = x + self.o_proj(o.reshape(b, t, d).astype(x.dtype))
    h = self.post_attention_layernorm(x)
    return x + self.down_proj(jax.nn.silu(self.gate_proj(h)) * self.up_proj(h))


def linear_upsample(x: Array, scale: int) -> Array:
  """`nn.Upsample(scale_factor=scale, mode="linear", align_corners=False)` along axis 1 of `[B, T, C]`."""
  length = x.shape[1]
  dst = np.arange(length * scale, dtype=np.float32)
  src = np.maximum((dst + 0.5) / scale - 0.5, 0.0)
  idx0 = np.floor(src).astype(np.int32)
  idx1 = np.minimum(idx0 + 1, length - 1)
  lam = jnp.asarray(src - idx0, x.dtype)[None, :, None]
  return x[:, idx0] * (1 - lam) + x[:, idx1] * lam


def periodic_hann_window(n: int) -> np.ndarray:
  """`torch.hann_window(n)` (periodic), computed in float32 like PyTorch."""
  k = np.arange(n, dtype=np.float32)
  return (np.float32(0.5) - np.float32(0.5) * np.cos(k * np.float32(2 * np.pi / n))).astype(np.float32)


def istft_same(spec_mag: Array, spec_phase: Array, n_fft: int, hop: int, num_frames: Array | None = None) -> Array:
  """`AudioVAE.istft.ISTFT(padding="same")` (non-streaming) on `[B, T, n_fft // 2 + 1]` magnitude/phase.

  The window is the bf16-rounded periodic Hann window stored in the checkpoint. Returns `[B, T * hop]`.
  With `num_frames` (possibly traced), frames `>= num_frames` are padding: they are excluded from both the
  overlap-add and the window envelope, so the first `num_frames * hop` samples equal the unpadded result.
  """
  if n_fft % hop:
    raise ValueError(f"ISTFT expects n_fft ({n_fft}) to be a multiple of hop ({hop}).")
  window = jnp.asarray(bf16_round(periodic_hann_window(n_fft)))
  spec = spec_mag * (jnp.cos(spec_phase) + 1j * jnp.sin(spec_phase))
  frames = jnp.fft.irfft(spec, n=n_fft, axis=-1).astype(jnp.float32) * window  # [B, T, n_fft]
  t = frames.shape[1]
  valid = jnp.ones((t,), jnp.float32) if num_frames is None else (jnp.arange(t) < num_frames).astype(jnp.float32)
  frames = frames * valid[None, :, None]
  overlap = n_fft // hop

  def overlap_add(f):
    out = jnp.zeros((f.shape[0], (t + overlap - 1) * hop), jnp.float32)
    pieces = f.reshape(f.shape[0], t, overlap, hop)
    for k in range(overlap):
      out = out.at[:, k * hop : k * hop + t * hop].add(pieces[:, :, k].reshape(f.shape[0], t * hop))
    return out

  y = overlap_add(frames)
  envelope = overlap_add((window**2)[None, None, :] * valid[None, :, None])
  pad = (n_fft - hop) // 2
  y, envelope = y[:, pad:-pad], envelope[:, pad:-pad]
  # The envelope is positive over the real frames; it can only vanish in the padded tail.
  return jnp.where(envelope > 1e-11, y / jnp.where(envelope > 1e-11, envelope, 1.0), 0.0)


class MingAudioVAEDecoder(nnx.Module):
  """`AudioVAE.vae_modules.Decoder.low_level_reconstruct` (non-streaming): latents `[B, N, 64]` -> `[B, N * 3528]`."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    d = config.ming_vae_hidden_size
    self.hop = config.ming_vae_hop_length
    self.n_fft = 4 * self.hop
    self.fc1 = _dense(config, config.ming_talker_latent_dim, d, ("mlp", "embed"), True, rngs)
    for i in range(config.ming_vae_num_layers):
      setattr(self, f"layers_{i}", MingVAEDecoderLayer(config, rngs=rngs))
    self.norm = _rms_norm(config, d, rngs, config.ming_vae_rms_norm_eps)
    self.head_out = _dense(config, d, self.n_fft + 2, ("embed", "mlp"), True, rngs)

  def spectrogram(self, latents: Array) -> Array:
    """Latents `[B, N, latent]` -> ISTFT-head outputs `[B, N * patch, n_fft + 2]` (log-magnitude | phase).

    The backbone is causal, so frames only depend on earlier latents: callers may right-pad the latents (with
    copies of the last latent, which keeps the linear upsampling of the last real latent unchanged) to bucket
    shapes and slice the first `N * patch` frames.
    """
    cfg = self.config
    x = self.fc1(latents.astype(cfg.dtype))
    x = linear_upsample(x, cfg.ming_vae_patch_size)
    t = x.shape[1]
    positions = jnp.arange(t)
    q_idx, k_idx = positions[:, None], positions[None, :]
    # transformers sliding-window causal mask: kv_idx <= q_idx and q_idx - kv_idx < sliding_window.
    mask = (k_idx <= q_idx) & (q_idx - k_idx < cfg.ming_vae_sliding_window)
    for i in range(cfg.ming_vae_num_layers):
      x = getattr(self, f"layers_{i}")(x, positions, mask)
    x = self.norm(x)
    return self.head_out(x).astype(jnp.float32)

  def spectrogram_to_waveform(self, pred: Array, num_frames: Array | None = None) -> Array:
    """ISTFT head: `[B, T, n_fft + 2]` -> `[B, T * hop]` waveform (frames `>= num_frames` are padding)."""
    half = pred.shape[-1] // 2
    mag = jnp.minimum(jnp.exp(pred[..., :half]), 1e2)
    return istft_same(mag, pred[..., half:], self.n_fft, self.hop, num_frames)

  def __call__(self, latents: Array, num_latents: Array | None = None) -> Array:
    """Latents `[B, N, latent]` -> waveform `[B, N * patch * hop]`.

    `num_latents` marks the real latents of an edge-padded input (shape bucketing); samples past
    `num_latents * patch * hop` are not part of the output and must be sliced off by the caller.
    """
    num_frames = None if num_latents is None else num_latents * self.config.ming_vae_patch_size
    return self.spectrogram_to_waveform(self.spectrogram(latents), num_frames)


# -----------------------------------------
# Container owned by models.Transformer
# -----------------------------------------


class MingTalkerHeads(nnx.Module):
  """Non-LLM talker modules: CFM DiT, aggregator, stop/speaker heads and AudioVAE decoder."""

  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    self.cfm = MingTalkerDiT(config, rngs=rngs)
    self.aggregator = MingTalkerAggregator(config, rngs=rngs)
    self.stop_head = _dense(config, config.emb_dim, 2, ("embed", None), True, rngs)
    self.spk_head = _dense(config, config.ming_talker_speaker_dim, config.emb_dim, (None, "embed"), True, rngs)
    self.audio_vae = MingAudioVAEDecoder(config, rngs=rngs)

  def sample_latent(self, hidden, history, noise, cfg_strength, sigma, temperature, sde_noise=None):
    """One talker step after the LLM: (latent, next input embedding, stop probability).

    Args:
      hidden: `[B, 1, emb]` final-normed LLM hidden state of the last position.
      history: `[B, H, latent]` previous latents (zeros at the first step).
      noise: `[B, P, latent]` initial CFM noise.
    """
    latent = cfm_sample(
        self.cfm, hidden, history, noise, self.config.ming_talker_cfm_steps, cfg_strength, sigma, temperature, sde_noise
    )
    next_embed = self.aggregator(latent)
    stop_prob = jax.nn.softmax(self.stop_head(hidden[:, -1]).astype(jnp.float32), axis=-1)[:, 1]
    return latent, next_embed, stop_prob

  def decode_audio(self, latents: Array, num_latents: Array | None = None) -> Array:
    """AudioVAE decoder: `[B, N, latent]` -> `[B, N * patch * hop]` waveform."""
    return self.audio_vae(latents, num_latents)

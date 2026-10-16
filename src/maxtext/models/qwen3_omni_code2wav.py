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

"""Qwen3-Omni waveform decoder and its convolutional/transformer components."""

import math

from flax import nnx
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Array, Config


def _param(rngs: nnx.Rngs, shape, dtype, init=None, scale: float = 0.02) -> nnx.Param:
  if init is None:
    value = jax.random.normal(rngs.params(), shape, dtype) * scale
  else:
    value = jnp.full(shape, init, dtype)
  return nnx.Param(value)


class ParamGroup(nnx.Module):
  """A named bag of ``nnx.Param`` leaves (e.g. ``conv.weight`` / ``conv.bias``) mirroring an HF submodule."""

  def __init__(self, **params: nnx.Param):
    for name, value in params.items():
      setattr(self, name, value)

  def __getattr__(self, name: str) -> nnx.Param:
    # Attributes are set dynamically in __init__; this only runs on misses and keeps linters from flagging no-member.
    raise AttributeError(f"{type(self).__name__} has no parameter {name!r}")


class Qwen3OmniCausalConv1d(nnx.Module):
  """HF ``Qwen3OmniMoeCausalConvNet``: left-padded (causal) 1-D convolution. Layout ``[batch, channels, time]``.

  Kernel is stored in PyTorch layout ``[out_channels, in_channels // groups, kernel_size]``.
  """

  def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1, *, dtype, rngs):
    self.stride = stride
    self.dilation = dilation
    self.groups = groups
    self.effective_kernel_size = (kernel_size - 1) * dilation + 1
    self.padding = self.effective_kernel_size - stride
    self.conv = ParamGroup(
        weight=_param(rngs, (out_channels, in_channels // groups, kernel_size), dtype),
        bias=_param(rngs, (out_channels,), dtype, init=0.0),
    )

  def __call__(self, x: Array) -> Array:
    length = x.shape[-1]
    n_frames = (length - self.effective_kernel_size + self.padding) / self.stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.effective_kernel_size - self.padding)
    extra_padding = ideal_length - length
    x = jnp.pad(x, ((0, 0), (0, 0), (self.padding, extra_padding)))
    weight = self.conv.weight.get_value().astype(x.dtype)
    y = lax.conv_general_dilated(
        x,
        weight,
        window_strides=(self.stride,),
        padding=[(0, 0)],
        rhs_dilation=(self.dilation,),
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=self.groups,
        precision=lax.Precision.HIGHEST,
    )
    return y + self.conv.bias.get_value().astype(y.dtype)[None, :, None]


class Qwen3OmniCausalTransConv1d(nnx.Module):
  """HF ``Qwen3OmniMoeCausalTransConvNet``: transposed conv with symmetric trimming of ``kernel - stride``.

  Kernel is stored in PyTorch layout ``[in_channels, out_channels, kernel_size]``.
  """

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, *, dtype, rngs):
    self.stride = stride
    self.kernel_size = kernel_size
    pad = kernel_size - stride
    self.left_pad = math.ceil(pad)
    self.right_pad = self.left_pad
    self.conv = ParamGroup(
        weight=_param(rngs, (in_channels, out_channels, kernel_size), dtype),
        bias=_param(rngs, (out_channels,), dtype, init=0.0),
    )

  def __call__(self, x: Array) -> Array:
    weight = self.conv.weight.get_value().astype(x.dtype)
    # PyTorch ConvTranspose1d(in, out) is the input-gradient of Conv1d(out, in) whose weight is the same
    # `[in, out, k]` tensor read in OIH layout; `transpose_kernel=True` flips/swaps it accordingly.
    y = lax.conv_transpose(
        x,
        weight,
        strides=(self.stride,),
        padding="VALID",
        dimension_numbers=("NCH", "OIH", "NCH"),
        transpose_kernel=True,
        precision=lax.Precision.HIGHEST,
    )
    y = y + self.conv.bias.get_value().astype(y.dtype)[None, :, None]
    return y[..., self.left_pad : y.shape[-1] - self.right_pad]


class Qwen3OmniSnakeBeta(nnx.Module):
  """SnakeBeta activation: ``x + 1/(exp(beta)+eps) * sin(x * exp(alpha))^2`` with per-channel ``alpha``/``beta``."""

  def __init__(self, channels, *, dtype, rngs):
    self.alpha = _param(rngs, (channels,), dtype, init=0.0)
    self.beta = _param(rngs, (channels,), dtype, init=0.0)
    self.no_div_by_zero = 1e-9

  def __call__(self, x: Array) -> Array:
    alpha = jnp.exp(self.alpha.get_value().astype(x.dtype))[None, :, None]
    beta = jnp.exp(self.beta.get_value().astype(x.dtype))[None, :, None]
    return x + (1.0 / (beta + self.no_div_by_zero)) * jnp.square(jnp.sin(x * alpha))


class Qwen3OmniConvNeXtBlock(nnx.Module):
  """HF ``Qwen3OmniMoeConvNeXtBlock``: depthwise causal conv -> LayerNorm -> MLP(GELU) -> layer scale -> residual."""

  def __init__(self, dim, *, dtype, rngs):
    self.dwconv = Qwen3OmniCausalConv1d(dim, dim, kernel_size=7, groups=dim, dilation=1, dtype=dtype, rngs=rngs)
    self.norm = ParamGroup(weight=_param(rngs, (dim,), dtype, init=1.0), bias=_param(rngs, (dim,), dtype, init=0.0))
    self.pwconv1 = ParamGroup(weight=_param(rngs, (4 * dim, dim), dtype), bias=_param(rngs, (4 * dim,), dtype, init=0.0))
    self.pwconv2 = ParamGroup(weight=_param(rngs, (dim, 4 * dim), dtype), bias=_param(rngs, (dim,), dtype, init=0.0))
    self.gamma = _param(rngs, (dim,), dtype, init=1e-6)
    self.eps = 1e-6

  def __call__(self, x: Array) -> Array:
    residual = x
    h = self.dwconv(x)
    h = jnp.swapaxes(h, 1, 2)  # [B, T, C]
    mean = jnp.mean(h, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(h - mean), axis=-1, keepdims=True)
    h = (h - mean) * lax.rsqrt(var + self.eps)
    h = h * self.norm.weight.get_value().astype(h.dtype) + self.norm.bias.get_value().astype(h.dtype)
    h = h @ self.pwconv1.weight.get_value().astype(h.dtype).T + self.pwconv1.bias.get_value().astype(h.dtype)
    h = jax.nn.gelu(h, approximate=False)
    h = h @ self.pwconv2.weight.get_value().astype(h.dtype).T + self.pwconv2.bias.get_value().astype(h.dtype)
    h = self.gamma.get_value().astype(h.dtype) * h
    return residual + jnp.swapaxes(h, 1, 2)


class Qwen3OmniCode2WavTransformerLayer(nnx.Module):
  """HF ``Qwen3OmniMoeCode2WavTransformerLayer``: sliding-window causal attention + SwiGLU MLP with layer scale."""

  def __init__(self, config: Config, *, dtype, rngs):
    hidden = config.code2wav_hidden_size
    self.num_heads = config.code2wav_num_attention_heads
    self.num_kv_heads = config.code2wav_num_key_value_heads
    self.head_dim = hidden // self.num_heads
    self.sliding_window = config.code2wav_sliding_window
    self.rope_theta = config.code2wav_rope_theta
    self.eps = config.code2wav_rms_norm_eps
    inter = config.code2wav_intermediate_size
    self.self_attn = ParamGroup(
        q_proj=_param(rngs, (self.num_heads * self.head_dim, hidden), dtype),
        k_proj=_param(rngs, (self.num_kv_heads * self.head_dim, hidden), dtype),
        v_proj=_param(rngs, (self.num_kv_heads * self.head_dim, hidden), dtype),
        o_proj=_param(rngs, (hidden, self.num_heads * self.head_dim), dtype),
    )
    self.mlp = ParamGroup(
        gate_proj=_param(rngs, (inter, hidden), dtype),
        up_proj=_param(rngs, (inter, hidden), dtype),
        down_proj=_param(rngs, (hidden, inter), dtype),
    )
    self.input_layernorm = _param(rngs, (hidden,), dtype, init=1.0)
    self.post_attention_layernorm = _param(rngs, (hidden,), dtype, init=1.0)
    self.self_attn_layer_scale = _param(rngs, (hidden,), dtype, init=config.code2wav_layer_scale_initial_scale)
    self.mlp_layer_scale = _param(rngs, (hidden,), dtype, init=config.code2wav_layer_scale_initial_scale)

  def _rms_norm(self, x: Array, weight: nnx.Param) -> Array:
    x32 = x.astype(jnp.float32)
    x32 = x32 * lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + self.eps)
    return weight.get_value().astype(x.dtype) * x32.astype(x.dtype)

  def _rope(self, x: Array) -> Array:
    """HF rotate-half RoPE over ``[B, T, H, D]`` with positions ``arange(T)``."""
    seq = x.shape[1]
    dim = x.shape[-1]
    inv_freq = 1.0 / (self.rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    freqs = jnp.arange(seq, dtype=jnp.float32)[:, None] * inv_freq[None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos = jnp.cos(emb)[None, :, None, :]
    sin = jnp.sin(emb)[None, :, None, :]
    x1, x2 = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return (x.astype(jnp.float32) * cos + rotated * sin).astype(x.dtype)

  def __call__(self, x: Array) -> Array:
    batch, seq, _ = x.shape
    residual = x
    h = self._rms_norm(x, self.input_layernorm)
    dt = h.dtype
    q = (h @ self.self_attn.q_proj.get_value().astype(dt).T).reshape(batch, seq, self.num_heads, self.head_dim)
    k = (h @ self.self_attn.k_proj.get_value().astype(dt).T).reshape(batch, seq, self.num_kv_heads, self.head_dim)
    v = (h @ self.self_attn.v_proj.get_value().astype(dt).T).reshape(batch, seq, self.num_kv_heads, self.head_dim)
    q = self._rope(q)
    k = self._rope(k)
    rep = self.num_heads // self.num_kv_heads
    if rep > 1:
      k = jnp.repeat(k, rep, axis=2)
      v = jnp.repeat(v, rep, axis=2)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * (self.head_dim**-0.5)
    q_idx = jnp.arange(seq)[:, None]
    k_idx = jnp.arange(seq)[None, :]
    allowed = (k_idx <= q_idx) & (k_idx > q_idx - self.sliding_window)
    scores = jnp.where(allowed[None, None], scores, jnp.finfo(jnp.float32).min)
    probs = jax.nn.softmax(scores, axis=-1).astype(dt)
    attn = jnp.einsum("bhqk,bkhd->bqhd", probs, v).reshape(batch, seq, self.num_heads * self.head_dim)
    attn = attn @ self.self_attn.o_proj.get_value().astype(dt).T
    x = residual + self.self_attn_layer_scale.get_value().astype(dt) * attn

    residual = x
    h = self._rms_norm(x, self.post_attention_layernorm)
    mlp = self.mlp
    gate = h @ mlp.gate_proj.get_value().astype(dt).T
    up = h @ mlp.up_proj.get_value().astype(dt).T
    m = (jax.nn.silu(gate) * up) @ mlp.down_proj.get_value().astype(dt).T
    return residual + self.mlp_layer_scale.get_value().astype(dt) * m


class Qwen3OmniCode2WavTransformer(nnx.Module):
  """HF ``Qwen3OmniMoeCode2WavTransformerModel``: stacked sliding-window layers + final RMSNorm."""

  def __init__(self, config: Config, *, dtype, rngs):
    self.num_layers = config.code2wav_num_hidden_layers
    self.eps = config.code2wav_rms_norm_eps
    for i in range(self.num_layers):
      setattr(self, f"layers_{i}", Qwen3OmniCode2WavTransformerLayer(config, dtype=dtype, rngs=rngs))
    self.norm = _param(rngs, (config.code2wav_hidden_size,), dtype, init=1.0)

  def __call__(self, hidden: Array) -> Array:
    for i in range(self.num_layers):
      hidden = getattr(self, f"layers_{i}")(hidden)
    h32 = hidden.astype(jnp.float32)
    h32 = h32 * lax.rsqrt(jnp.mean(jnp.square(h32), axis=-1, keepdims=True) + self.eps)
    return self.norm.get_value().astype(hidden.dtype) * h32.astype(hidden.dtype)


class Qwen3OmniCode2WavDecoderResidualUnit(nnx.Module):
  """HF ``Qwen3OmniMoeCode2WavDecoderResidualUnit``."""

  def __init__(self, dim, dilation, *, dtype, rngs):
    self.act1 = Qwen3OmniSnakeBeta(dim, dtype=dtype, rngs=rngs)
    self.conv1 = Qwen3OmniCausalConv1d(dim, dim, kernel_size=7, dilation=dilation, dtype=dtype, rngs=rngs)
    self.act2 = Qwen3OmniSnakeBeta(dim, dtype=dtype, rngs=rngs)
    self.conv2 = Qwen3OmniCausalConv1d(dim, dim, kernel_size=1, dtype=dtype, rngs=rngs)

  def __call__(self, x: Array) -> Array:
    return x + self.conv2(self.act2(self.conv1(self.act1(x))))


class Qwen3OmniCode2WavDecoderBlock(nnx.Module):
  """HF ``Qwen3OmniMoeCode2WavDecoderBlock``: SnakeBeta -> CausalTransConv(2r, r) -> 3 residual units."""

  def __init__(self, config: Config, layer_idx: int, *, dtype, rngs):
    in_dim = config.code2wav_decoder_dim // 2**layer_idx
    out_dim = config.code2wav_decoder_dim // 2 ** (layer_idx + 1)
    rate = config.code2wav_upsample_rates[layer_idx]
    blocks = [
        Qwen3OmniSnakeBeta(in_dim, dtype=dtype, rngs=rngs),
        Qwen3OmniCausalTransConv1d(in_dim, out_dim, 2 * rate, rate, dtype=dtype, rngs=rngs),
    ] + [Qwen3OmniCode2WavDecoderResidualUnit(out_dim, d, dtype=dtype, rngs=rngs) for d in (1, 3, 9)]
    self.num_blocks = len(blocks)
    for i, block in enumerate(blocks):
      setattr(self, f"block_{i}", block)

  def __call__(self, x: Array) -> Array:
    for i in range(self.num_blocks):
      x = getattr(self, f"block_{i}")(x)
    return x


class Qwen3OmniCode2Wav(nnx.Module):
  """HF ``Qwen3OmniMoeCode2Wav`` vocoder: ``[batch, num_quantizers, T]`` codes -> ``[batch, 1, T * total_upsample]``."""

  def __init__(self, config: Config, mesh: Mesh, *, rngs: nnx.Rngs):
    self.config = config
    dtype = jnp.dtype(config.code2wav_dtype)
    self.dtype = dtype
    hidden = config.code2wav_hidden_size
    self.codebook_size = config.code2wav_codebook_size
    self.num_quantizers = config.code2wav_num_quantizers
    self.total_upsample = int(math.prod(config.code2wav_upsample_rates) * math.prod(config.code2wav_upsampling_ratios))
    self.code_embedding = _param(rngs, (self.codebook_size * self.num_quantizers, hidden), dtype)
    self.num_transformer_layers = config.code2wav_num_hidden_layers
    self.pre_transformer = Qwen3OmniCode2WavTransformer(config, dtype=dtype, rngs=rngs)
    self.num_upsample = len(config.code2wav_upsampling_ratios)
    for i, factor in enumerate(config.code2wav_upsampling_ratios):
      setattr(self, f"upsample_{i}_0", Qwen3OmniCausalTransConv1d(hidden, hidden, factor, factor, dtype=dtype, rngs=rngs))
      setattr(self, f"upsample_{i}_1", Qwen3OmniConvNeXtBlock(hidden, dtype=dtype, rngs=rngs))
    decoder = [Qwen3OmniCausalConv1d(hidden, config.code2wav_decoder_dim, 7, dtype=dtype, rngs=rngs)]
    for i in range(len(config.code2wav_upsample_rates)):
      decoder.append(Qwen3OmniCode2WavDecoderBlock(config, i, dtype=dtype, rngs=rngs))
    output_dim = config.code2wav_decoder_dim // 2 ** len(config.code2wav_upsample_rates)
    decoder += [
        Qwen3OmniSnakeBeta(output_dim, dtype=dtype, rngs=rngs),
        Qwen3OmniCausalConv1d(output_dim, 1, 7, dtype=dtype, rngs=rngs),
    ]
    self.num_decoder_blocks = len(decoder)
    for i, block in enumerate(decoder):
      setattr(self, f"decoder_{i}", block)

  def __call__(self, codes: Array) -> Array:
    if codes.shape[1] != self.num_quantizers:
      raise ValueError(f"Expected {self.num_quantizers} layers of codes, got {codes.shape[1]}")
    offset = (jnp.arange(self.num_quantizers, dtype=codes.dtype) * self.codebook_size)[None, :, None]
    table = self.code_embedding.get_value()
    hidden = jnp.take(table, codes + offset, axis=0).astype(self.dtype).mean(axis=1)  # [B, T, D]
    hidden = self.pre_transformer(hidden)
    hidden = jnp.swapaxes(hidden, 1, 2)  # [B, D, T]
    for i in range(self.num_upsample):
      hidden = getattr(self, f"upsample_{i}_0")(hidden)
      hidden = getattr(self, f"upsample_{i}_1")(hidden)
    wav = hidden
    for i in range(self.num_decoder_blocks):
      wav = getattr(self, f"decoder_{i}")(wav)
    return jnp.clip(wav, -1.0, 1.0)

  def chunked_decode(self, codes: Array, chunk_size: int = 300, left_context_size: int = 25) -> Array:
    """Decodes long code sequences in chunks with left context (mirrors HF ``chunked_decode``)."""
    wavs = []
    start_index = 0
    total = codes.shape[-1]
    while start_index < total:
      end_index = min(start_index + chunk_size, total)
      context_size = left_context_size if start_index - left_context_size > 0 else start_index
      codes_chunk = codes[..., start_index - context_size : end_index]
      wav_chunk = self(codes_chunk)
      wavs.append(wav_chunk[..., context_size * self.total_upsample :])
      start_index = end_index
    return jnp.concatenate(wavs, axis=-1)

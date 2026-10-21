# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SplitMix64 n-gram hashing and Layer-2 PLE module for Qwen3.8-Flash-Next."""

import math
from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
# pylint: disable=no-name-in-module,too-many-positional-arguments
from maxtext.m3.models.qwen38_flash_next import hyper_connection

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
  """Exact 64-bit integer SplitMix64 mixing."""
  value = (value + _SPLITMIX_GAMMA) & _MASK64
  value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
  value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
  return (value ^ (value >> 31)) & _MASK64


splitmix64 = _splitmix64


def _is_prime(value: int) -> bool:
  """Returns True if value is prime."""
  if value < 2 or value % 2 == 0:
    return value == 2
  return all(value % d != 0 for d in range(3, math.isqrt(value) + 1, 2))


def _find_nth_prime_after(start: int, count: int) -> int:
  """Finds the count-th prime strictly greater than start."""
  prime = start
  for _ in range(count):
    prime += 1
    while not _is_prime(prime):
      prime += 1
  return prime


def compute_ple_buffers(
    ngram_size: int = 3,
    heads_per_ngram: int = 8,
    ngram_vocab_size_base: int = 20_000_000,
    make_ngram_vocab_size_divisible_by: int = 128,
    vocab_size: int = 248320,
    ple_layer_index: int = 0,
    seed: int = 1234,
) -> tuple[list[int], list[int], list[int], int]:
  """Computes non-trainable PLE buffer arrays (multipliers, moduli, offsets, padded_vocab_size)."""
  half_bound = max(1, (((1 << 63) - 1) // max(vocab_size, 1)) // 2)
  base_seed = seed + _PRIME_1 * ple_layer_index
  multipliers = [
      2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1 for i in range(ngram_size)
  ]
  ngram_heads = (ngram_size - 1) * heads_per_ngram
  moduli, table_offsets, tot = [], [], 0
  for h in range(ngram_heads):
    sz = _find_nth_prime_after(ngram_vocab_size_base - 1, ple_layer_index * ngram_heads + h + 1)
    moduli.append(sz)
    table_offsets.append(tot)
    tot += sz
  padded_vocab_size = math.ceil(tot / make_ngram_vocab_size_divisible_by) * make_ngram_vocab_size_divisible_by
  return multipliers, moduli, table_offsets, padded_vocab_size


def _shift_right_ignore_eos(token_history: jax.Array, shift: int, eos_token_id: int) -> jax.Array:
  """Shifts tokens right by shift positions, resetting context on EOS."""
  if shift == 0:
    return token_history
  b, s = token_history.shape
  pos = jnp.arange(s, dtype=jnp.int32)
  eos_pos = jnp.where(token_history == eos_token_id, pos[None, :], -1)
  prev_eos = jnp.concatenate([jnp.full((b, 1), -1, dtype=jnp.int32), jax.lax.cummax(eos_pos, axis=1)[:, :-1]], axis=1)
  src = pos - shift
  shifted = jnp.take_along_axis(token_history, jnp.maximum(src, 0)[None, :], axis=1)
  valid = (pos[None, :] - (prev_eos + 1) >= shift) & (src[None, :] >= 0)
  return jnp.where(valid, shifted, jnp.full_like(shifted, eos_token_id))


def _mul_u64_bytes(tokens: jax.Array, mult_bytes: jax.Array) -> jax.Array:
  """Multiplies tokens by 8-byte multiplier in uint32 with carry propagation."""
  t, carry, out = tokens.astype(jnp.uint32), jnp.zeros_like(tokens, dtype=jnp.uint32), []
  for b in range(8):
    prod = t * mult_bytes[b] + carry
    out.append(prod & jnp.uint32(0xFF))
    carry = prod >> jnp.uint32(8)
  return jnp.stack(out, axis=-1)


class Qwen38NGramEmbedding(nnx.Module):
  """Hashed n-gram embedding table with SplitMix64 multipliers."""

  def __init__(
      self,
      eos_token_id: int,
      vocab_size: int = 248320,
      ple_embed_dim: int = 2560,
      ngram_size: int = 3,
      heads_per_ngram: int = 8,
      ngram_vocab_size_base: int = 20_000_000,
      make_ngram_vocab_size_divisible_by: int = 128,
      seed: int = 1234,
      ple_layer_index: int = 0,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      *,
      rngs: nnx.Rngs,
  ):
    super().__init__()
    self.ple_embed_dim = ple_embed_dim
    self.ngram_size = ngram_size
    self.heads_per_ngram = heads_per_ngram
    self.ngram_heads = (ngram_size - 1) * heads_per_ngram
    self.eos_token_id = eos_token_id

    if ple_embed_dim % self.ngram_heads != 0:
      raise ValueError(f"ple_embed_dim ({ple_embed_dim}) not divisible by ngram_heads ({self.ngram_heads}).")
    head_dim_per_ngram = ple_embed_dim // self.ngram_heads

    (
        self.layer_multipliers,
        self.head_vocab_sizes,
        self.head_offsets,
        self.padded_vocab_size,
    ) = compute_ple_buffers(
        ngram_size,
        heads_per_ngram,
        ngram_vocab_size_base,
        make_ngram_vocab_size_divisible_by,
        vocab_size,
        ple_layer_index,
        seed,
    )

    if (
        max(self.head_vocab_sizes) > (1 << 28)
        or self.padded_vocab_size >= (1 << 31)
        or vocab_size > (1 << 24)
        or eos_token_id >= (1 << 24)
    ):
      raise ValueError(
          "PLE hashing uint32 bounds exceeded: head vocab must be <= 2**28, padded vocab < 2**31, "
          f"vocab_size/eos_token_id < 2**24; got max head {max(self.head_vocab_sizes)}, "
          f"padded {self.padded_vocab_size}, vocab_size {vocab_size}, eos_token_id {eos_token_id}."
      )

    self._mult_bytes_tuple = tuple(tuple((m >> (8 * b)) & 0xFF for b in range(8)) for m in self.layer_multipliers)

    self.ngram_embedding = nnx.Embed(
        self.padded_vocab_size, head_dim_per_ngram, dtype=dtype, param_dtype=weight_dtype, rngs=rngs
    )

  @property
  def head_vocab_sizes_arr(self) -> jax.Array:
    """Returns head vocabulary sizes as a uint32 JAX array."""
    return jnp.array(self.head_vocab_sizes, dtype=jnp.uint32)

  @property
  def head_offsets_arr(self) -> jax.Array:
    """Returns head vocabulary offsets as a uint32 JAX array."""
    return jnp.array(self.head_offsets, dtype=jnp.uint32)

  @property
  def mult_bytes(self) -> jax.Array:
    """Returns SplitMix64 multiplier bytes as a uint32 JAX array."""
    return jnp.array(self._mult_bytes_tuple, dtype=jnp.uint32)

  def compute_ngram_ids(self, input_ids: jax.Array) -> jax.Array:
    """Computes exact 64-bit n-gram hash IDs for each token position."""
    b, s = input_ids.shape
    mult_bytes = self.mult_bytes
    head_vocab_sizes_arr = self.head_vocab_sizes_arr
    head_offsets_arr = self.head_offsets_arr
    context = jnp.full((b, self.ngram_size - 1), self.eos_token_id, dtype=input_ids.dtype)
    history = jnp.concatenate([context, input_ids], axis=-1)
    shifted_tokens = [_shift_right_ignore_eos(history, sh, self.eos_token_id) for sh in range(self.ngram_size)]
    shifted_bytes = [_mul_u64_bytes(shifted_tokens[p], mult_bytes[p]) for p in range(self.ngram_size)]

    blocks = []
    for ngram in range(2, self.ngram_size + 1):
      s_idx = (ngram - 2) * self.heads_per_ngram
      e_idx = s_idx + self.heads_per_ngram
      mixed = shifted_bytes[0]
      for pos in range(1, ngram):
        mixed = jnp.bitwise_xor(mixed, shifted_bytes[pos])

      v = head_vocab_sizes_arr[s_idx:e_idx][None, None, :]
      rem = jnp.zeros((b, history.shape[1], v.shape[-1]), dtype=jnp.uint32)
      for b_idx in range(7, -1, -1):
        byte = mixed[:, :, b_idx : b_idx + 1]
        rem = (rem * jnp.uint32(16) + ((byte >> jnp.uint32(4)) & jnp.uint32(0xF))) % v
        rem = (rem * jnp.uint32(16) + (byte & jnp.uint32(0xF))) % v

      blocks.append((rem + head_offsets_arr[s_idx:e_idx][None, None, :]).astype(jnp.int32))

    return jnp.concatenate(blocks, axis=-1)[:, -s:, :]

  def __call__(self, input_ids: jax.Array) -> jax.Array:
    return self.ngram_embedding(self.compute_ngram_ids(input_ids)).reshape(input_ids.shape[0], -1, self.ple_embed_dim)


class Qwen38PLELayer(nnx.Module):
  """PLE layer injecting hashed n-gram features into hyper-connection streams."""

  def __init__(
      self,
      eos_token_id: int,
      emb_dim: int = 2560,
      hc_count: int = 4,
      vocab_size: int = 248320,
      ple_embed_dim: int = 2560,
      ngram_size: int = 3,
      heads_per_ngram: int = 8,
      ngram_vocab_size_base: int = 20_000_000,
      make_ngram_vocab_size_divisible_by: int = 128,
      ple_conv_kernel_size: int = 4,
      seed: int = 1234,
      ple_layer_index: int = 0,
      eps: float = 1e-6,
      dtype: Any = jnp.bfloat16,
      weight_dtype: Any = jnp.bfloat16,
      sharding: Any | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    del sharding
    super().__init__()
    self.emb_dim, self.hc_count = emb_dim, hc_count
    self.hc_hidden_size = emb_dim * hc_count
    self.short_conv_state_len = (ple_conv_kernel_size - 1) * ngram_size
    self.gate_scale = 1.0 / math.sqrt(emb_dim)

    self.ple_embedding = Qwen38NGramEmbedding(
        eos_token_id,
        vocab_size,
        ple_embed_dim,
        ngram_size,
        heads_per_ngram,
        ngram_vocab_size_base,
        make_ngram_vocab_size_divisible_by,
        seed,
        ple_layer_index,
        dtype,
        weight_dtype,
        rngs=rngs,
    )

    kw = {"use_bias": False, "dtype": dtype, "param_dtype": weight_dtype, "rngs": rngs}
    self.key_proj = nnx.Linear(ple_embed_dim, self.hc_hidden_size, **kw)
    self.value_proj = nnx.Linear(ple_embed_dim, emb_dim, **kw)

    norm_kw = {"group_size": emb_dim, "eps": eps, "rngs": rngs}
    self.norm_key = hyper_connection.Qwen38RMSNorm(self.hc_hidden_size, **norm_kw)
    self.norm_query = hyper_connection.Qwen38RMSNorm(self.hc_hidden_size, **norm_kw)
    self.norm_conv = hyper_connection.Qwen38RMSNorm(self.hc_hidden_size, **norm_kw)

    self.conv1d = nnx.Conv(
        in_features=self.hc_hidden_size,
        out_features=self.hc_hidden_size,
        kernel_size=(ple_conv_kernel_size,),
        feature_group_count=self.hc_hidden_size,
        kernel_dilation=(ngram_size,),
        padding="VALID",
        use_bias=False,
        kernel_init=nnx.initializers.zeros_init(),
        dtype=dtype,
        param_dtype=weight_dtype,
        rngs=rngs,
    )

  def _short_conv(self, hidden_states: jax.Array) -> jax.Array:
    padded = jnp.pad(hidden_states, ((0, 0), (self.short_conv_state_len, 0), (0, 0)))
    return jax.nn.silu(self.conv1d(padded))

  def __call__(self, hidden_states: jax.Array, input_ids: jax.Array, conv_mask: jax.Array | None = None) -> jax.Array:
    """Forward pass injecting hashed n-gram features into hidden streams."""
    is_4d = hidden_states.ndim == 4
    b, s = hidden_states.shape[:2]
    hidden_flat = hidden_states.reshape(b, s, self.hc_hidden_size) if is_4d else hidden_states

    embeds = self.ple_embedding(input_ids)
    key_normed = self.norm_key(self.key_proj(embeds)).reshape(b, s, self.hc_count, self.emb_dim)
    value = self.value_proj(embeds)
    query_normed = self.norm_query(hidden_flat).reshape(b, s, self.hc_count, self.emb_dim)

    gate = jnp.sum(key_normed.astype(jnp.float32) * query_normed.astype(jnp.float32), axis=-1, keepdims=True)
    gate = jnp.sqrt(jnp.maximum(jnp.abs(gate * self.gate_scale), 1e-6)) * jnp.sign(gate)
    gated_val = (jax.nn.sigmoid(gate) * value[:, :, None, :].astype(jnp.float32)).astype(hidden_states.dtype)

    gated_flat = gated_val.reshape(b, s, self.hc_hidden_size)
    gated_norm = self.norm_conv(gated_flat)

    if conv_mask is not None:
      m = conv_mask[:, :, None].astype(gated_flat.dtype)
      gated_flat, gated_norm = gated_flat * m, gated_norm * m

    out_flat = gated_flat + self._short_conv(gated_norm)
    return out_flat.reshape(b, s, self.hc_count, self.emb_dim) if is_4d else out_flat

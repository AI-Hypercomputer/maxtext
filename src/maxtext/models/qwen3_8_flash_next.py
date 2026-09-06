# Copyright 2025 Google LLC
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

"""Qwen3.8-Flash-Next (Qwen4Exp) model layers and modules."""

import ast
import math
from typing import Any, cast
from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import Array, Config, DType
from maxtext.layers import embeddings
from maxtext.layers import initializers as max_initializers
from maxtext.layers import linears
from maxtext.layers import nnx_wrappers

from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.models.qwen3 import (
    Qwen3NextGatedDeltaNet,
    Qwen3NextFullAttention,
    Qwen3NextSparseMoeBlock,
)
from maxtext.utils import max_utils


class Qwen3_8FlashNextRMSNorm(nnx.Module):
  """RMSNorm with optional grouping and unit offset scaling (1.0 + weight)."""

  def __init__(
      self,
      num_features: int,
      group_size: int | None = None,
      epsilon: float = 1e-6,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.group_size = group_size
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype

    self.weight = nnx.Param(
        jnp.zeros((num_features,), dtype=weight_dtype),
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)

    if self.group_size is not None:
      orig_shape = x_f32.shape
      x_reshaped = x_f32.reshape(*orig_shape[:-1], -1, self.group_size)
      variance = jnp.mean(jnp.square(x_reshaped), axis=-1, keepdims=True)
      normed = x_reshaped * jax.lax.rsqrt(variance + self.epsilon)
      normed = normed.reshape(orig_shape)
    else:
      variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
      normed = x_f32 * jax.lax.rsqrt(variance + self.epsilon)

    scale = 1.0 + self.weight[...].astype(jnp.float32)
    output = normed * scale
    return output.astype(orig_dtype)


class Qwen3_8FlashNextHyperConnection(nnx.Module):
  """HyperConnection block for multi-stream residual routing and mixing."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh | None = None,
      use_combine: bool = True,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.use_combine = use_combine
    self.hc_count = config.hc_count
    self.hidden_size = config.emb_dim
    hc_hidden_size = self.hc_count * self.hidden_size

    self.hc_norm = Qwen3_8FlashNextRMSNorm(
        num_features=hc_hidden_size,
        group_size=self.hidden_size,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        rngs=rngs,
    )

    self.input_mix_weight_down = linears.DenseGeneral(
        in_features_shape=hc_hidden_size,
        out_features_shape=config.hc_lowrank,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("embed", None),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

    self.input_mix_weight_up = linears.DenseGeneral(
        in_features_shape=config.hc_lowrank,
        out_features_shape=hc_hidden_size,
        use_bias=False,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=(None, "embed"),
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

    if use_combine:
      self.block_inject_weight = linears.DenseGeneral(
          in_features_shape=hc_hidden_size,
          out_features_shape=self.hc_count,
          use_bias=False,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("embed", None),
          matmul_precision=config.matmul_precision,
          rngs=rngs,
      )
    else:
      self.block_inject_weight = None

  def __call__(self, hyper_input: jnp.ndarray) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    hyper_input_normed = self.hc_norm(hyper_input)

    down_out = self.input_mix_weight_down(hyper_input_normed) / self.hc_count
    down_act = jax.nn.silu(down_out)
    up_out = self.input_mix_weight_up(down_act)
    input_mix_weight = jax.nn.sigmoid(up_out)

    orig_shape = hyper_input.shape[:-1]
    input_mix_weight = input_mix_weight.reshape(*orig_shape, self.hc_count, self.hidden_size)
    normed_reshaped = hyper_input_normed.reshape(*orig_shape, self.hc_count, self.hidden_size)

    mixed_input = jnp.mean(input_mix_weight * normed_reshaped, axis=-2)

    if not self.use_combine or self.block_inject_weight is None:
      return mixed_input

    inject_out = self.block_inject_weight(hyper_input_normed) / self.hc_count
    injection_weights = 2.0 * jax.nn.sigmoid(inject_out)

    return mixed_input, hyper_input, injection_weights


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
  value = (value + _SPLITMIX_GAMMA) & _MASK64
  value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
  value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
  return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> list[int]:
  max_long = (1 << 63) - 1
  multiplier_max = max_long // max(unigram_vocab_size, 1)
  half_bound = max(1, multiplier_max // 2)
  base_seed = seed + _PRIME_1 * ple_layer_index
  multipliers = []
  for index in range(ngram_size):
    value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
    multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
  return multipliers


def _is_prime(value: int) -> bool:
  if value < 2:
    return False
  if value % 2 == 0:
    return value == 2
  for divisor in range(3, math.isqrt(value) + 1, 2):
    if value % divisor == 0:
      return False
  return True


def _find_nth_prime_after(start: int, count: int) -> int:
  prime = start
  for _ in range(count):
    prime += 1
    while not _is_prime(prime):
      prime += 1
  return prime


class Qwen3_8FlashNextPLELayer(nnx.Module):
  """PLE (Per-Layer Embedding) module."""

  def __init__(
      self,
      config: Config,
      layer_idx: int,
      ple_layer_index: int = 0,
      mesh: Mesh | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.layer_idx = layer_idx
    self.ple_layer_index = ple_layer_index
    self.mesh = mesh
    cfg = config

    self.hidden_size = cfg.emb_dim
    self.hc_count = cfg.hc_count
    self.ple_embed_dim = getattr(cfg, "ple_embed_dim", 2560)
    hc_hidden_size = self.hidden_size * self.hc_count
    self.ngram_size = getattr(cfg, "ngram_size", 3)
    self.context_len = self.ngram_size - 1
    self.heads_per_ngram = getattr(cfg, "heads_per_ngram", 8)
    self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
    self.unigram_vocab_size = cfg.vocab_size
    self.ngram_vocab_size_base = getattr(cfg, "ngram_vocab_size_base", 20000000)
    self.head_dim_per_ngram = self.ple_embed_dim // self.ngram_heads
    self.seed = getattr(cfg, "seed", 0)
    self.eos_token_id = getattr(cfg, "eos_token_id", 248044)

    head_vocab_sizes = []
    head_offsets = []
    total_vocab_size = 0
    for head_idx in range(self.ngram_heads):
      global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
      size = _find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
      head_vocab_sizes.append(size)
      head_offsets.append(total_vocab_size)
      total_vocab_size += size

    self.head_vocab_sizes_list = head_vocab_sizes
    self.head_offsets_list = head_offsets
    self.layer_multipliers_list = _build_layer_multipliers(
        self.unigram_vocab_size, self.ngram_size, self.ple_layer_index, self.seed
    )

    padded_vocab_size = math.ceil(total_vocab_size / 128) * 128
    self.ngram_embedding = embeddings.Embed(
        num_embeddings=padded_vocab_size,
        num_features=self.head_dim_per_ngram,
        config=cfg,
        mesh=mesh,
        rngs=rngs,
    )

    self.key_proj = linears.DenseGeneral(
        in_features_shape=self.ple_embed_dim,
        out_features_shape=hc_hidden_size,
        use_bias=False,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", None),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    self.value_proj = linears.DenseGeneral(
        in_features_shape=self.ple_embed_dim,
        out_features_shape=self.hidden_size,
        use_bias=False,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        kernel_axes=("embed", None),
        matmul_precision=cfg.matmul_precision,
        rngs=rngs,
    )

    self.norm_key = Qwen3_8FlashNextRMSNorm(
        num_features=hc_hidden_size,
        group_size=self.hidden_size,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    self.norm_query = Qwen3_8FlashNextRMSNorm(
        num_features=hc_hidden_size,
        group_size=self.hidden_size,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    self.norm_conv = Qwen3_8FlashNextRMSNorm(
        num_features=hc_hidden_size,
        group_size=self.hidden_size,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

    conv_kernel_size = getattr(cfg, "ple_conv_kernel_size", 4)
    self.conv_dilation = self.ngram_size
    self.short_conv_state_len = (conv_kernel_size - 1) * self.conv_dilation

    self.conv1d = nnx.Conv(
        in_features=hc_hidden_size,
        out_features=hc_hidden_size,
        kernel_size=(conv_kernel_size,),
        kernel_dilation=(self.conv_dilation,),
        padding="VALID",
        feature_group_count=hc_hidden_size,
        use_bias=False,
        dtype=cfg.dtype,
        param_dtype=cfg.weight_dtype,
        rngs=rngs,
    )

  def _shift_right_ignore_eos(self, token_ids: jnp.ndarray, shift: int) -> jnp.ndarray:
    """Shift tokens right within EOS-delimited segments, padding with EOS."""
    if shift == 0:
      return token_ids
    batch_size, seq_len = token_ids.shape
    positions = jnp.arange(seq_len, dtype=jnp.int32)
    eos_positions = jnp.where(token_ids == self.eos_token_id, positions, -1)
    previous_eos_inclusive = jax.lax.cummax(eos_positions, axis=1)
    previous_eos = jnp.concatenate(
        [jnp.full((batch_size, 1), -1, dtype=jnp.int32), previous_eos_inclusive[:, :-1]], axis=1
    )
    segment_start = previous_eos + 1
    position_in_segment = jnp.expand_dims(positions, 0) - segment_start
    source_positions = positions - shift
    gather_positions = jnp.broadcast_to(jnp.maximum(source_positions, 0)[None, :], (batch_size, seq_len))
    shifted = jnp.take_along_axis(token_ids, gather_positions, axis=1)
    valid = (position_in_segment >= shift) & (jnp.expand_dims(source_positions, 0) >= 0)
    return jnp.where(valid, shifted, self.eos_token_id)

  @staticmethod
  def _mul64(a: jnp.ndarray, b: int | jnp.uint32) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the low and high uint32 words of the product."""
    a = a.astype(jnp.uint32)
    b = jnp.uint32(b)
    a0 = a & jnp.uint32(0xFFFF)
    a1 = a >> 16
    b0 = b & jnp.uint32(0xFFFF)
    b1 = b >> 16

    p0 = a0 * b0
    p1 = a0 * b1
    p2 = a1 * b0
    p3 = a1 * b1

    mid = p1 + p2 + (p0 >> 16)
    lo = ((mid & jnp.uint32(0xFFFF)) << 16) | (p0 & jnp.uint32(0xFFFF))
    hi = p3 + (mid >> 16)
    return lo, hi

  @classmethod
  def _compute_term_64(cls, t: jnp.ndarray, m: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Multiply token IDs by a 64-bit multiplier and return two uint32 words."""
    m_lo = m & 0xFFFFFFFF
    m_hi = (m >> 32) & 0xFFFFFFFF

    t_lo_prod_lo, t_lo_prod_hi = cls._mul64(t, m_lo)
    t_hi_prod_lo, _ = cls._mul64(t, m_hi)

    lo = t_lo_prod_lo
    hi = t_lo_prod_hi + t_hi_prod_lo
    return lo, hi

  @classmethod
  def _mul_mod_p(cls, a: jnp.ndarray, b: int | jnp.uint32 | jnp.ndarray, p_val: int) -> jnp.ndarray:
    p = jnp.uint32(p_val)
    base_mod_val = (1 << 32) % int(p_val)
    base_mod = jnp.uint32(base_mod_val)
    lo, hi = cls._mul64(a, b)
    hi_lo, hi_hi = cls._mul64(jnp.remainder(hi, p), base_mod)
    t1 = jnp.remainder(hi_lo, p) + jnp.remainder(jnp.remainder(hi_hi, p) * base_mod, p)
    return jnp.remainder(jnp.remainder(lo, p) + t1, p)

  @classmethod
  def _mod_p_from_lo_hi(cls, lo: jnp.ndarray, hi: jnp.ndarray, p_val: int) -> jnp.ndarray:
    """Reduce signed 64-bit values represented by two uint32 words modulo p."""
    p = jnp.uint32(p_val)
    base_mod_val = (1 << 32) % int(p_val)
    two_64_mod_val = (base_mod_val * base_mod_val) % int(p_val)
    base_mod = jnp.uint32(base_mod_val)
    two_64_mod = jnp.uint32(two_64_mod_val)

    hi_m = jnp.remainder(hi, p)
    lo_m = jnp.remainder(lo, p)

    term_hi = cls._mul_mod_p(hi_m, base_mod, p_val)
    u64_m = jnp.remainder(term_hi + lo_m, p)

    is_neg = hi >= jnp.uint32(0x80000000)
    res = jnp.where(is_neg, jnp.remainder(u64_m + p - two_64_mod, p), u64_m)
    return res

  def _ngram_embedding_lookup(self, input_ids: jnp.ndarray) -> jnp.ndarray:
    """Hash token n-grams and concatenate their per-head embeddings."""
    input_ids = input_ids.astype(jnp.uint32)
    batch_size, seq_len = input_ids.shape
    previous_context = jnp.full((batch_size, self.context_len), self.eos_token_id, dtype=jnp.uint32)
    token_history = jnp.concatenate([previous_context, input_ids], axis=-1)

    shifted_tokens = [self._shift_right_ignore_eos(token_history, shift) for shift in range(self.ngram_size)]

    blocks = []
    for ngram in range(2, self.ngram_size + 1):
      start_idx = (ngram - 2) * self.heads_per_ngram
      end_idx = start_idx + self.heads_per_ngram

      lo_acc, hi_acc = self._compute_term_64(shifted_tokens[0], self.layer_multipliers_list[0])
      for position in range(1, ngram):
        lo, hi = self._compute_term_64(shifted_tokens[position], self.layer_multipliers_list[position])
        lo_acc = lo_acc ^ lo
        hi_acc = hi_acc ^ hi

      cur_sizes = self.head_vocab_sizes_list[start_idx:end_idx]
      cur_offsets = self.head_offsets_list[start_idx:end_idx]

      head_blocks = []
      for p_val, off_val in zip(cur_sizes, cur_offsets):
        off = jnp.uint32(off_val)
        res = self._mod_p_from_lo_hi(lo_acc, hi_acc, p_val) + off
        head_blocks.append(jnp.expand_dims(res, -1))

      blocks.append(jnp.concatenate(head_blocks, axis=-1))

    ngram_ids = jnp.concatenate(blocks, axis=-1)[:, -seq_len:]
    embeddings_out = self.ngram_embedding(ngram_ids.astype(jnp.int32))
    return embeddings_out.reshape(batch_size, seq_len, self.ple_embed_dim)

  def _short_conv(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
    seq_len = hidden_states.shape[1]
    padded = jnp.pad(hidden_states, ((0, 0), (self.short_conv_state_len, 0), (0, 0)))
    conv_out = jax.nn.silu(self.conv1d(padded))

    return conv_out[:, -seq_len:, :]

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      decoder_input_tokens: jnp.ndarray | None = None,
  ) -> jnp.ndarray:
    if decoder_input_tokens is None:
      return jnp.zeros_like(hidden_states)

    ngram_embeddings = self._ngram_embedding_lookup(decoder_input_tokens)
    orig_shape = hidden_states.shape[:-1]

    key_normed = self.norm_key(self.key_proj(ngram_embeddings)).reshape(*orig_shape, self.hc_count, self.hidden_size)
    value = self.value_proj(ngram_embeddings)
    query_normed = self.norm_query(hidden_states).reshape(*orig_shape, self.hc_count, self.hidden_size)

    gate = jnp.sum(key_normed * query_normed, axis=-1, keepdims=True) / math.sqrt(self.hidden_size)
    gate = jnp.sqrt(jnp.maximum(jnp.abs(gate), 1e-6)) * jnp.sign(gate)
    gated_value = jax.nn.sigmoid(gate) * jnp.expand_dims(value, -2)

    gated_value_flat = gated_value.reshape(*orig_shape, self.hc_count * self.hidden_size)
    gated_value_normed = self.norm_conv(gated_value_flat)

    output = gated_value_flat + self._short_conv(gated_value_normed)
    return output


class Qwen3_8FlashNextDecoderLayer(nnx.Module):
  """Decoder Layer for Qwen3.8-Flash-Next with HyperConnection, GDN/Attention, and SparseMoE."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      layer_idx: int,
      quant: Quant | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.quant = quant
    cfg = self.config

    ple_layer_ids = getattr(cfg, "ple_layer_ids", (2,))
    if isinstance(ple_layer_ids, str):
      ple_layer_ids = (
          ast.literal_eval(ple_layer_ids) if ("(" in ple_layer_ids or "[" in ple_layer_ids) else (int(ple_layer_ids),)
      )
    ple_layer_index = ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in ple_layer_ids else None

    if ple_layer_index is not None:
      self.ple = Qwen3_8FlashNextPLELayer(
          config=cfg,
          layer_idx=layer_idx,
          ple_layer_index=ple_layer_index,
          mesh=mesh,
          rngs=rngs,
      )
    else:
      self.ple = None

    self.attn_hyper_connection = Qwen3_8FlashNextHyperConnection(config=cfg, mesh=mesh, use_combine=True, rngs=rngs)

    is_full_attention_layer = (self.layer_idx + 1) % cfg.inhomogeneous_layer_cycle_interval == 0
    if is_full_attention_layer:
      self.attention = Qwen3NextFullAttention(
          config=cfg,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=model_mode,
          layer_idx=self.layer_idx,
          rngs=rngs,
      )
    else:
      batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
      dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
      self.attention = Qwen3NextGatedDeltaNet(
          config=cfg,
          inputs_shape=dummy_inputs_shape,
          mesh=self.mesh,
          dtype=cfg.dtype,
          model_mode=model_mode,
          rngs=rngs,
      )

    self.mlp_hyper_connection = Qwen3_8FlashNextHyperConnection(config=cfg, mesh=mesh, use_combine=True, rngs=rngs)

    self.mlp = Qwen3NextSparseMoeBlock(config=cfg, mesh=self.mesh, quant=self.quant, rngs=rngs)

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: jnp.ndarray | None = None,
      decoder_positions: jnp.ndarray | None = None,
      deterministic: bool = True,
      model_mode: str = "train",
      previous_chunk=None,
      slot: int | None = None,
      kv_cache: dict[str, Array] | None = None,
      attention_metadata: dict[str, Any] | None = None,
      forced_routed_experts: jnp.ndarray | None = None,
      decoder_input_tokens: jnp.ndarray | None = None,
  ):
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    hidden_states = inputs

    if self.ple is not None:
      hidden_states = hidden_states + self.ple(hidden_states, decoder_input_tokens=decoder_input_tokens)

    mixed_input, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)

    if isinstance(self.attention, Qwen3NextFullAttention):
      attn_out, new_kv_cache = cast(Qwen3NextFullAttention, self.attention)(
          mixed_input,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
      )
    else:
      attn_out, new_kv_cache = cast(Qwen3NextGatedDeltaNet, self.attention)(
          mixed_input,
          model_mode=model_mode,
          kv_cache=kv_cache,
          decoder_segment_ids=decoder_segment_ids,
          attention_metadata=attention_metadata,
      )

    injection = jnp.expand_dims(attn_out, -2) * jnp.expand_dims(injection_weights, -1)
    orig_shape = hidden_states.shape[:-1]
    hidden_states = hyper_input + injection.reshape(*orig_shape, self.config.hc_count * self.config.emb_dim)

    mixed_input, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
    mlp_out, load_balance_loss = self.mlp(
        mixed_input,
        deterministic=deterministic,
        forced_routed_experts=forced_routed_experts,
    )

    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)

    injection = jnp.expand_dims(mlp_out, -2) * jnp.expand_dims(injection_weights, -1)
    hidden_states = hyper_input + injection.reshape(*orig_shape, self.config.hc_count * self.config.emb_dim)

    return hidden_states, new_kv_cache


class Qwen3_8FlashNextScannableBlock(nnx.Module):
  """Scannable block for Qwen3.8-Flash-Next covering an inhomogeneous cycle."""

  def __init__(self, config: Config, mesh: Mesh, model_mode: str, quant=None, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    cfg = self.config

    for i in range(cfg.inhomogeneous_layer_cycle_interval):
      layer_rngs = self.rngs.fork()
      layer_name = f"layer_{i}"
      layer = Qwen3_8FlashNextDecoderLayer(
          config=self.config,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=self.model_mode,
          layer_idx=i,
          rngs=layer_rngs,
      )
      setattr(self, layer_name, layer)

  def __call__(
      self,
      carry: jnp.ndarray,
      decoder_segment_ids: jnp.ndarray | None,
      decoder_positions: jnp.ndarray | None,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: int | None = None,
      forced_routed_experts: jnp.ndarray | None = None,
      decoder_input_tokens: jnp.ndarray | None = None,
  ) -> tuple[Array, None]:
    cfg = self.config
    x = carry

    for i in range(cfg.inhomogeneous_layer_cycle_interval):
      layer = getattr(self, f"layer_{i}")
      layer_forced_routed_experts = forced_routed_experts[i] if forced_routed_experts is not None else None
      x, _ = layer(
          x,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk,
          slot,
          forced_routed_experts=layer_forced_routed_experts,
          decoder_input_tokens=decoder_input_tokens,
      )

    return x, None


Qwen3_8FlashNextDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Qwen3_8FlashNextDecoderLayer,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

Qwen3_8FlashNextScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Qwen3_8FlashNextScannableBlock,
    base_metadata_fn=max_initializers.variable_to_logically_partitioned,
)

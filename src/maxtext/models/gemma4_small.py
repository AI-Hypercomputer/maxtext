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

"""Specialized layers for Gemma 4 small (E2B and E4B)."""

import jax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx

from maxtext.common.common_types import Config, AttentionType, MODEL_MODE_PREFILL
from maxtext.layers import initializers
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import DenseGeneral, MlpBlock
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils


# E2B repeats 4 sliding + 1 global (period 5); E4B repeats 5 sliding + 1 global
# (period 6). E4B is also used as the default for unrecognized model names.
GEMMA4_E2B_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)
GEMMA4_E4B_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def get_attention_pattern(model_name):
  """Returns the repeating sliding/global attention pattern for a small variant."""
  if model_name == "gemma4-e2b":
    return GEMMA4_E2B_ATTENTION_PATTERN
  return GEMMA4_E4B_ATTENTION_PATTERN


def get_attention_type(layer_id, model_name=None):
  """Returns the attention type for ``layer_id`` under the variant's pattern."""
  pattern = get_attention_pattern(model_name)
  return pattern[layer_id % len(pattern)]


def build_layer_types(num_layers, model_name):
  """Returns the per-layer attention-type tuple for the full decoder stack."""
  return tuple(get_attention_type(i, model_name) for i in range(num_layers))


def first_kv_shared_layer_idx(num_layers: int, num_kv_shared_layers: int) -> int:
  """Index of the first KV-shared layer, or ``num_layers`` if none."""
  if num_kv_shared_layers <= 0:
    return num_layers
  return max(0, num_layers - num_kv_shared_layers)


def is_kv_shared_layer(layer_idx: int, num_layers: int, num_kv_shared_layers: int) -> bool:
  """Returns True iff layer ``layer_idx`` reuses K/V from an earlier layer."""
  if num_kv_shared_layers <= 0:
    return False
  first = first_kv_shared_layer_idx(num_layers, num_kv_shared_layers)
  return layer_idx >= first > 0


def kv_donor_layer_idx(
    layer_idx: int,
    layer_types: tuple[AttentionType, ...],
    num_kv_shared_layers: int,
) -> int | None:
  """Index of the layer that owns the K/V used by ``layer_idx``.

  A shared layer reuses K/V from the last non-shared layer of the same
  attention type.
  """
  num_layers = len(layer_types)
  if not is_kv_shared_layer(layer_idx, num_layers, num_kv_shared_layers):
    return None
  first = first_kv_shared_layer_idx(num_layers, num_kv_shared_layers)
  layer_type = layer_types[layer_idx]
  for j in range(first - 1, -1, -1):
    if layer_types[j] == layer_type:
      return j
  return None


def is_kv_donor_layer(
    layer_idx: int,
    layer_types: tuple[AttentionType, ...],
    num_kv_shared_layers: int,
) -> bool:
  """Returns True iff this layer's K/V are reused by some shared layer."""
  num_layers = len(layer_types)
  if layer_idx < 0 or layer_idx >= num_layers:
    return False
  if num_kv_shared_layers <= 0:
    return False
  for j in range(num_layers):
    if kv_donor_layer_idx(j, layer_types, num_kv_shared_layers) == layer_idx:
      return True
  return False


class Gemma4SmallPLE(nnx.Module):
  """Builds the ``[B, S, num_layers, D_ple]`` per-layer-input tensor."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh

    num_layers = config.num_decoder_layers
    ple_dim = config.hidden_size_per_layer_input
    vocab_ple = config.vocab_size_per_layer_input

    self.embed_tokens_per_layer = nnx.Param(
        nn.initializers.normal(stddev=ple_dim**-0.5)(
            rngs.params(),
            (vocab_ple, num_layers * ple_dim),
            config.weight_dtype,
        ),
        sharding=("vocab", "embed_vocab"),
    )

    self.per_layer_model_projection = DenseGeneral(
        in_features_shape=config.emb_dim,
        out_features_shape=num_layers * ple_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("embed", "mlp"),
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        rngs=rngs,
    )

    self.per_layer_projection_norm = RMSNorm(
        num_features=ple_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    # Plain Python floats — storing them as jnp arrays would let nnx promote
    # them to Variables, which then collide with the param tree on restore.
    self._ple_dim = ple_dim
    self._num_layers = num_layers
    self._embed_scale_value = float(ple_dim) ** 0.5
    self._proj_scale_value = float(config.emb_dim) ** -0.5

  def __call__(self, input_ids: jax.Array, inputs_embeds: jax.Array) -> jax.Array:
    """Returns ``per_layer_inputs`` of shape ``[B, S, L, D_ple]``."""
    cfg = self.config

    embed_scale = jnp.asarray(self._embed_scale_value, cfg.dtype)
    proj_scale = jnp.asarray(self._proj_scale_value, cfg.dtype)
    inv_sqrt2 = jnp.asarray(2.0**-0.5, cfg.dtype)

    embedding = jnp.asarray(self.embed_tokens_per_layer.value, cfg.dtype)
    identity = embedding[input_ids.astype(jnp.int32)] * embed_scale
    identity = identity.reshape(*input_ids.shape, self._num_layers, self._ple_dim)

    context = self.per_layer_model_projection(inputs_embeds.astype(cfg.dtype))
    context = context * proj_scale
    context = context.reshape(*inputs_embeds.shape[:-1], self._num_layers, self._ple_dim)
    context = self.per_layer_projection_norm(context)

    out = (identity + context) * inv_sqrt2
    return out.astype(cfg.dtype)


PLEToLinen = nnx_wrappers.to_linen_class(
    Gemma4SmallPLE,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class Gemma4SmallDecoderLayer(nnx.Module):
  """Transformer decoder layer for Gemma 4 small (E2B / E4B)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      attention_type: AttentionType = AttentionType.LOCAL_SLIDING,
      layer_idx: int = 0,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.attention_type = attention_type
    self.layer_idx = layer_idx

    num_layers = config.num_decoder_layers
    self.is_shared = is_kv_shared_layer(layer_idx, num_layers, config.num_kv_shared_layers)

    # Global layers may use larger head_dim and a different RoPE base.
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    if attention_type == AttentionType.GLOBAL:
      if config.global_num_kv_heads:
        num_kv_heads = config.global_num_kv_heads
      if config.global_head_dim:
        head_dim = config.global_head_dim
      partial_rotary_factor = config.global_rope_proportion
      max_timescale = (
          config.global_rope_max_timescale if config.global_rope_max_timescale > 0 else config.rope_max_timescale
      )
    else:
      partial_rotary_factor = config.local_rope_proportion
      max_timescale = (
          config.local_rope_max_timescale if config.local_rope_max_timescale > 0 else config.rope_max_timescale
      )

    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim

    self.pre_self_attention_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.post_self_attention_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.pre_ffw_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.post_ffw_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    self.self_attention = Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        mesh=mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        dropout_rate=config.dropout_rate,
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        attention_type=attention_type,
        sliding_window_size=config.sliding_window_size,
        attn_logits_soft_cap=config.attn_logits_soft_cap,
        use_qk_norm=True,
        use_v_norm=True,
        # HF Gemma 4 attention is unscaled (scaling=1.0); weights absorb 1/sqrt(d).
        query_pre_attn_scalar=1.0,
        rope_max_timescale=max_timescale,
        partial_rotary_factor=partial_rotary_factor,
        share_kv_layer=self.is_shared,
        model_mode=model_mode,
        rngs=rngs,
    )

    # E2B widens the MLP on KV-shared layers to compensate for the missing
    # K/V parameters; E4B leaves it alone.
    mlp_dim = config.mlp_dim
    if self.is_shared and config.use_double_wide_mlp:
      mlp_dim = mlp_dim * 2
    self.mlp = MlpBlock(
        in_features=config.emb_dim,
        intermediate_dim=mlp_dim,
        activations=config.mlp_activations,
        intermediate_dropout_rate=config.dropout_rate,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        config=config,
        quant=quant,
        model_mode=model_mode,
        mesh=mesh,
        rngs=rngs,
    )

    ple_dim = config.hidden_size_per_layer_input
    if ple_dim > 0:
      self.per_layer_input_gate = DenseGeneral(
          in_features_shape=config.emb_dim,
          out_features_shape=ple_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("embed", "mlp"),
          shard_mode=config.shard_mode,
          matmul_precision=config.matmul_precision,
          rngs=rngs,
      )
      self.per_layer_projection = DenseGeneral(
          in_features_shape=ple_dim,
          out_features_shape=config.emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("mlp", "embed"),
          shard_mode=config.shard_mode,
          matmul_precision=config.matmul_precision,
          rngs=rngs,
      )
      self.post_per_layer_input_norm = RMSNorm(
          num_features=config.emb_dim,
          epsilon=config.normalization_layer_epsilon,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          rngs=rngs,
      )
    else:
      self.per_layer_input_gate = None
      self.per_layer_projection = None
      self.post_per_layer_input_norm = None

    self.layer_scalar = nnx.Param(jnp.ones((1,), dtype=config.weight_dtype), sharding=(None,))

    if model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

  def compute_shared_kv(self, inputs: jax.Array, decoder_positions: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns the rotated, normed K / V for this (non-shared) layer.

    Used by the decoder loop on donor layers, so the K / V can be threaded into
    downstream KV-shared layers via ``shared_key`` / ``shared_value``.
    """
    if self.is_shared:
      raise ValueError("compute_shared_kv must not be called on a KV-shared layer.")
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    h = self.pre_self_attention_norm(inputs)
    return self.self_attention.compute_shared_kv(h, inputs_positions=decoder_positions)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state=None,
      slot=None,
      bidirectional_mask=None,
      kv_cache=None,
      attention_metadata=None,
      per_layer_input: jax.Array | None = None,
      shared_key: jax.Array | None = None,
      shared_value: jax.Array | None = None,
  ):
    cfg = self.config

    if isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    # Bidirectional image-token mask is only meaningful in local-sliding
    # layers (matches gemma4.py).
    if self.attention_type != AttentionType.LOCAL_SLIDING:
      bidirectional_mask = None

    residual = inputs
    h = self.pre_self_attention_norm(inputs)
    h = nn.with_logical_constraint(h, self.activation_axis_names)

    attn_out, kv_cache = self.self_attention(
        h,
        h,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        bidirectional_mask=bidirectional_mask,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
        shared_key=shared_key,
        shared_value=shared_value,
    )
    attn_out = self.post_self_attention_norm(attn_out)
    attn_out = nn.with_logical_constraint(attn_out, self.activation_axis_names)
    h = residual + attn_out

    residual = h
    mlp_in = self.pre_ffw_norm(h)
    mlp_out = self.mlp(mlp_in, deterministic=deterministic)
    mlp_out = self.post_ffw_norm(mlp_out)
    mlp_out = nn.with_logical_constraint(mlp_out, self.activation_axis_names)
    h = residual + mlp_out

    if self.per_layer_input_gate is not None and per_layer_input is not None:
      residual = h
      gate = self.per_layer_input_gate(h)
      gate = jax.nn.gelu(gate.astype(jnp.float32), approximate=True).astype(cfg.dtype)
      gated = gate * per_layer_input.astype(cfg.dtype)
      proj = self.per_layer_projection(gated)
      proj = self.post_per_layer_input_norm(proj)
      proj = nn.with_logical_constraint(proj, self.activation_axis_names)
      h = residual + proj

    h = h * jnp.asarray(self.layer_scalar.value, cfg.dtype)
    h = nn.with_logical_constraint(h, self.activation_axis_names)

    return h


Gemma4SmallDecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Gemma4SmallDecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)

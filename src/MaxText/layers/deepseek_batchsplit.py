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

# fmt: off

"""Alternative DeepSeek model definition with batch-split schedule."""

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from MaxText import common_types
from MaxText import max_utils
from MaxText.common_types import Config
from MaxText.inference import page_manager
from MaxText.layers import attention_mla
from MaxText.layers import initializers
from MaxText.layers import linears
from MaxText.layers import moe
from MaxText.layers import normalizations
from MaxText.layers import nnx_wrappers
from MaxText.layers import quantizations
from MaxText.sharding import maybe_shard_with_logical, create_sharding

class DeepSeekBatchSplitGenericLayer(nnx.Module):
  """Generic DeepSeek layer with Multi-Head Latent Attention.

  This is to be used as a base class for DeepSeek layers with dense/sparse MLPs.
  This class follows a pattern of separating module creation from execution.
  """
  def __init__(
      self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: quantizations.AqtQuantization|None = None,
  ) -> None:

    self.config = config
    self.model_mode = model_mode
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs

    batch_size, sequence_length = max_utils.get_batch_seq_len_for_mode(self.config, model_mode)
    self.dummy_inputs_shape = (batch_size, sequence_length, self.config.emb_dim)

    self.out_sharding = create_sharding(self.mesh, self.logical_axis_names)
    self.mlp_intermediate_sharding = create_sharding(self.mesh, self.mlp_logical_axis_names)

    self.pre_attention_layer_norm = normalizations.RMSNorm(
        num_features=self.dummy_inputs_shape[-1],
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

    self.post_attention_layer_norm = normalizations.RMSNorm(
        num_features=self.dummy_inputs_shape[-1],
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        epsilon=config.normalization_layer_epsilon,
        rngs=self.rngs,
    )

    self.self_attention = attention_mla.MLA(
        config=self.config,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        attention_kernel=self.config.attention,
        attention_type=self.config.attention_type,
        inputs_q_shape=self.dummy_inputs_shape,
        inputs_kv_shape=self.dummy_inputs_shape,
        mesh=self.mesh,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(self.config),
        q_lora_rank=self.config.q_lora_rank,
        kv_lora_rank=self.config.kv_lora_rank,
        qk_nope_head_dim=self.config.qk_nope_head_dim,
        qk_rope_head_dim=self.config.qk_rope_head_dim,
        v_head_dim=self.config.v_head_dim,
        max_position_embeddings=self.config.max_position_embeddings,
        original_max_position_embeddings=self.config.original_max_position_embeddings,
        mscale=self.config.mscale,
        rope_factor=self.config.rope_factor,
        model_mode=self.model_mode,
        attn_logits_soft_cap=self.config.attn_logits_soft_cap,
        rngs=self.rngs,
    )

    self.dropout = linears.Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    x = self.with_logical_constraint(inputs)
    x = jax.ad_checkpoint.checkpoint_name(x, "decoder_layer_input")

    x += self.attention_op(
        self.pre_attention_norm_op(x),
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        previous_chunk,
        page_state,
        slot,
    )

    mlp_output = self.mlp_op(self.post_attention_norm_op(x), deterministic)
    if isinstance(mlp_output, tuple):
      x += mlp_output[0]
    else:
      x += mlp_output
    x = self.dropout_op(x, deterministic)
    return self.post_process(x, kv_cache=kv_cache)

  @property
  def logical_axis_names(self):
    if self.model_mode == common_types.MODEL_MODE_PREFILL:
      return (
          "activation_batch",
          "prefill_activation_norm_length",
          "activation_embed",
      )
    return (
        "activation_batch",
        "activation_norm_length",
        "activation_embed",
    )

  @property
  def mlp_logical_axis_names(self):
    if self.model_mode == common_types.MODEL_MODE_PREFILL:
      return (
          "activation_batch",
          "prefill_activation_norm_length",
          "activation_mlp",
      )
    return (
        "activation_batch",
        "activation_norm_length",
        "activation_mlp",
    )

  def with_logical_constraint(self, x):
    return maybe_shard_with_logical(
      x, logical_axes=self.logical_axis_names,
      mesh=self.mesh, shard_mode=self.config.shard_mode
    )

  def pre_attention_norm_op(self, x):
    return self.with_logical_constraint(self.pre_attention_layer_norm(x))

  def post_attention_norm_op(self, x):
    return self.with_logical_constraint(self.post_attention_layer_norm(x))

  def attention_op(
      self,
      x,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
  ):
    """Executes the attention layer."""
    attention_result, _ = self.self_attention(
      x,
      x,
      decoder_positions,
      decoder_segment_ids=decoder_segment_ids,
      deterministic=deterministic,
      model_mode=self.model_mode,
      previous_chunk=previous_chunk,
      page_state=page_state,
      slot=slot,
    )
    return self.with_logical_constraint(attention_result)

  def mlp_op(self, x, deterministic, *args, **kwargs):
    """Executes the MLP operation. To be implemented by subclasses."""
    raise NotImplementedError()

  def dropout_op(self, x, deterministic):
    return self.with_logical_constraint(
        self.dropout(x, deterministic=deterministic)
    )

  def post_process(self, x, kv_cache=None):
    """Collect statistics about the output of the layer."""
    if self.config.record_internal_nn_metrics:
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(x))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(x))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(x == 0) / jnp.size(x),
      )

    if self.config.scan_layers:
      return x, None
    return x, kv_cache


class DeepSeekDenseLayer(DeepSeekBatchSplitGenericLayer):
  """DeepSeek layer with dense MLP."""

  def __init__(self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: quantizations.AqtQuantization|None = None,):

    super().__init__(config, model_mode, mesh, rngs, quant)

    self.mlp = linears.MlpBlock(
        config=self.config,
        mesh=self.mesh,
        in_features=self.dummy_inputs_shape[-1],
        intermediate_dim=self.config.mlp_dim,
        activations=self.config.mlp_activations,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=self.quant,
        model_mode=model_mode,
        rngs=self.rngs,
    )

  def mlp_op(self, x, deterministic, *args, **kwargs):
    return self.with_logical_constraint(
      self.mlp(
        x,
        deterministic,
        intermediate_sharding=self.mlp_intermediate_sharding,
        out_sharding=self.out_sharding
      )
    )


DeepSeekDenseLayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekDenseLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)

class DeepSeekMoELayer(DeepSeekBatchSplitGenericLayer):
  """DeepSeek MoE layer that uses a batch-split schedule."""
  def __init__(self,
      config: Config,
      model_mode: str,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: quantizations.AqtQuantization|None = None,):

    super().__init__(config, model_mode, mesh, rngs, quant)

    self.DeepSeekMoeBlock_0 = moe.RoutedAndSharedMoE(
        config=self.config,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=quant,
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state: None | page_manager.PageState = None,
      slot: None | int = None,
      kv_cache=None,
      attention_metadata=None,
      split_factor: int = 2,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    x = self.with_logical_constraint(inputs)
    x = jax.ad_checkpoint.checkpoint_name(x, "decoder_layer_input")

    # Helper functions.
    def _split(x):
      if x is None:
        return [None] * split_factor
      else:
        return jnp.split(x, split_factor, axis=0)

    def _merge(x):
      return jnp.concatenate(x, axis=0)

    def _attn(x, decoder_segment_ids, decoder_positions):
      return self.attention_op(
          self.pre_attention_norm_op(x),
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          previous_chunk,
          page_state,
          slot,
      )

    def _moe(x):
      output, _, _ = self.mlp_op(self.post_attention_norm_op(x), deterministic)
      return output

    # Split the inputs into micro-batches.
    x = _split(x)
    dpos = _split(decoder_positions)
    dseg = _split(decoder_segment_ids)

    # Attention.
    x = [xi + _attn(xi, yi, zi) for xi, yi, zi in zip(x, dseg, dpos)]

    # Mixture-of-experts.
    x = [xi + _moe(xi) for xi in x]

    # Merge the micro-batches back into a single batch.
    x = _merge(x)

    x = self.dropout_op(x, deterministic)
    return self.post_process(x, kv_cache=kv_cache)

  def mlp_op(self, x, deterministic, *args, **kwargs):
    return self.with_logical_constraint(
      self.DeepSeekMoeBlock_0(
        x,intermediate_sharding=self.mlp_intermediate_sharding,
        out_sharding=self.out_sharding
      )
    )

DeepSeekMoELayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekMoELayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)

# Copyright 2023–2025 Google LLC
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

from flax import linen as nn
import jax
import jax.numpy as jnp
from MaxText import common_types
from MaxText.inference import page_manager
from MaxText.layers import attention_mla
from MaxText.layers import initializers
from MaxText.layers import linears
from MaxText.layers import moe
from MaxText.layers import normalizations
from MaxText.layers import quantizations


class DeepSeekGenericLayer(nn.Module):
  """Generic DeepSeek layer with Multi-Head Latent Attention.

  This is to be used as a base class for DeepSeek layers with dense/sparse MLPs.

  This class follows a pattern of separating module creation from execution.
  `*_layer()` methods (e.g., `attention_layer`) are factories for `nn.Module`s,
  called in `setup()` to initialize sub-layers. The module instances are stored
  in `*_op` attributes (e.g., `self.attention_op`). The corresponding methods
  (e.g., `attention`) are called during execution in `__call__` and wrap the
  `*_op` modules with logic like logical constraints. This keeps `__call__`
  clean and readable.
  """

  config: common_types.Config
  mesh: jax.sharding.Mesh
  model_mode: str
  quant: None | quantizations.AqtQuantization = None

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
  ):
    x = self.with_logical_constraint(inputs)
    x = jax.ad_checkpoint.checkpoint_name(x, "decoder_layer_input")

    x += self.attention(
        self.pre_attention_norm(x),
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        previous_chunk,
        page_state,
        slot,
    )

    x += self.mlp(self.post_attention_norm(x), deterministic)
    x = self.dropout(x, deterministic)
    return self.post_process(x)

  def setup(self):
    self.pre_attention_norm_op = self.rms_norm_layer("pre_attention_layer_norm")
    self.post_attention_norm_op = self.rms_norm_layer(
        "post_attention_layer_norm"
    )
    self.attention_op = self.attention_layer()
    self.mlp_op = self.mlp_layer()
    self.dropout_op = self.dropout_layer()

  @property
  def logical_axis_names(self):
    if self.model_mode == common_types.MODEL_MODE_PREFILL:
      return (
          "activation_batch",
          "prefill_activation_norm_length",
          "activation_embed",
      )
    else:
      return (
          "activation_batch",
          "activation_norm_length",
          "activation_embed",
      )

  def with_logical_constraint(self, x):
    return nn.with_logical_constraint(x, self.logical_axis_names)

  def rms_norm_layer(self, name):
    return normalizations.rms_norm(
        num_features=self.config.base_emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        name=name,
        kernel_axes=("norm",),
        epsilon=self.config.normalization_layer_epsilon,
    )

  def pre_attention_norm(self, x):
    return self.with_logical_constraint(self.pre_attention_norm_op(x))

  def post_attention_norm(self, x):
    return self.with_logical_constraint(self.post_attention_norm_op(x))

  def attention_layer(self):
    inputs_shape = (
        self.config.per_device_batch_size,
        self.config.max_target_length,
        self.config.base_emb_dim,
    )
    return attention_mla.mla_as_linen(
        config=self.config,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        attention_kernel=self.config.attention,
        attention_type=self.config.attention_type,
        inputs_q_shape=inputs_shape,
        inputs_kv_shape=inputs_shape,
        mesh=self.mesh,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        name="self_attention",
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
    )

  def attention(
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
    return self.with_logical_constraint(
        self.attention_op(
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
    )

  def mlp_layer(self):
    raise NotImplementedError()

  def mlp(self, x, deterministic):
    raise NotImplementedError()

  def dropout_layer(self):
    return nn.Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,))

  def dropout(self, x, deterministic):
    return self.with_logical_constraint(
        self.dropout_op(x, deterministic=deterministic)
    )

  def post_process(self, x):
    """Collect statistics about the output of the layer."""
    if self.config.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(x))
      self.sow("intermediates", "activation_stdev", jnp.std(x))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(x == 0) / jnp.size(x),
      )

    if self.config.scan_layers:
      return x, None
    else:
      return x


class DeepSeekDenseLayer(DeepSeekGenericLayer):
  """DeepSeek layer with dense MLP."""

  def mlp_layer(self):
    return linears.mlp_block(
        in_features=self.config.base_emb_dim,
        intermediate_dim=self.config.mlp_dim,
        activations=self.config.mlp_activations,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        name="mlp",
        config=self.config,
        quant=self.quant,
        mesh=self.mesh,
    )

  def mlp(self, x, deterministic):
    return self.with_logical_constraint(self.mlp_op(x, deterministic))


class DeepSeekMoELayer(DeepSeekGenericLayer):
  """DeepSeek MoE layer that uses a batch-split schedule."""

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
      split_factor: int = 2,
  ):
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
      return self.attention(
          self.pre_attention_norm(x),
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          previous_chunk,
          page_state,
          slot,
      )

    def _moe(x):
      return self.mlp(self.post_attention_norm(x), deterministic)

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

    x = self.dropout(x, deterministic)
    return self.post_process(x)

  def init(self, *args, **kwargs):
    # Calls the parent init method for testing parity.
    return super().init(*args, **kwargs, method=super().__call__)

  def mlp_layer(self):
    # NOTE: the naming mismatch here is to ensure reverse compatibility with
    # existing checkpoints. The `name` represents the weight name in
    # JAX/checkpoints and so the class name is just for readability.
    return moe.get_routed_and_shared_moe(
        name="DeepSeekMoeBlock_0",
        config=self.config,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(
            1.0, "fan_in", "truncated_normal"
        ),
        kernel_axes=("embed", None),
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        quant=self.quant,
    )

  def mlp(self, x, _):
    return self.with_logical_constraint(self.mlp_op(x))

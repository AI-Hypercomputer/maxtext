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

"""Module for decoder layers"""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
import inspect
import warnings
from typing import Any

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax import nnx
from flax.nnx import wrappers as nnx_wrappers
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from maxtext.common.common_types import (
    EP_AS_CONTEXT,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    Config,
    DecoderBlockType,
    ShardMode,
)
from maxtext.inference import page_manager
from maxtext.layers import initializers, linears, mhc, normalizations, quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.embeddings import Embed, PositionalEmbedding, attend_on_embedding
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.models import (
    deepseek,
    deepseek_batchsplit,
    gemma,
    gemma2,
    gemma3,
    gpt3,
    gpt_oss,
    llama2,
    llama4,
    mistral,
    mixtral,
    olmo3,
    qwen3,
    simple_layer,
)
from maxtext.multimodal import utils as mm_utils
from maxtext.utils import max_logging, max_utils, maxtext_utils, sharding
from maxtext.utils.sharding import create_sharding
from maxtext.layers.pipeline import create_nnx_pipeline


# ------------------------------------------------------------------------------
# The network: Decoder Definitions
# ------------------------------------------------------------------------------


class NNXDecoderLayer(nnx.Module):
  """
  Transformer decoder layer converted to NNX.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant = None,
      name: str = "decoder_layer",
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant

    cfg = self.config

    self.pre_self_attention_norm = RMSNorm(
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    self.self_attention = Attention(
        config=self.config,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=(1, 1, cfg.emb_dim),
        inputs_kv_shape=(1, 1, cfg.emb_dim),
        mesh=mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        float32_qk_product=cfg.float32_qk_product,
        float32_logits=cfg.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        prefill_cache_axis_order=tuple(map(int, cfg.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, cfg.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, cfg.compute_axis_order.split(","))),
        reshape_q=cfg.reshape_q,
        use_mrope=cfg.use_mrope,
        mrope_section=cfg.mrope_section,
        share_kv_projections=cfg.share_kv_projections,
        model_mode=model_mode,
        rngs=rngs,
    )

    self.mlp = linears.MlpBlock(
        in_features=cfg.emb_dim,
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        model_mode=model_mode,
        config=cfg,
        quant=self.quant,
        mesh=self.mesh,
        rngs=rngs,
    )

    self.dropout = linears.Dropout(rate=cfg.dropout_rate, rngs=rngs, broadcast_dims=(-2,))

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
      kv_cache: jax.Array | None = None,
      attention_metadata: dict[str, Any] | None = None,
  ):
    cfg = self.config
    mesh = self.mesh
    _maybe_shard_with_logical = functools.partial(
        sharding.maybe_shard_with_logical,
        mesh=mesh,
        shard_mode=cfg.shard_mode,
        debug_sharding=cfg.debug_sharding,
    )

    if self.model_mode == MODEL_MODE_PREFILL:
      logical_axis_names = ("activation_batch", "prefill_activation_length", "activation_embed")
    elif self.config.expert_shard_attention_option == EP_AS_CONTEXT and self.model_mode == MODEL_MODE_TRAIN:
      logical_axis_names = ("activation_batch_no_exp", "activation_length", "activation_embed")
    else:
      logical_axis_names = ("activation_batch", "activation_length_no_exp", "activation_embed")

    inputs = _maybe_shard_with_logical(inputs, logical_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    lnx = self.pre_self_attention_norm(inputs)
    lnx = _maybe_shard_with_logical(lnx, logical_axis_names)

    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    attention_lnx = _maybe_shard_with_logical(attention_lnx, logical_axis_names)

    mlp_lnx = self.mlp(lnx, deterministic=deterministic)
    mlp_lnx = _maybe_shard_with_logical(mlp_lnx, logical_axis_names)

    next_layer_addition = mlp_lnx + attention_lnx
    next_layer_addition_dropped_out = self.dropout(next_layer_addition, deterministic=deterministic)

    layer_output = next_layer_addition_dropped_out + inputs
    layer_output = _maybe_shard_with_logical(layer_output, logical_axis_names)

    if cfg.record_internal_nn_metrics:
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output, kv_cache


def deepstack_process(hidden_states, bidirectional_mask, visual_embeds):
  """Process deepstack visual embeddings by adding them to hidden states at visual token positions.

  Args:
    hidden_states:[batch, seq_len, hidden_dim] decoder hidden states
    bidirectional_mask: [batch, seq_len] boolean mask marking visual token positions
    visual_embeds: [batch, num_visual_tokens, hidden_dim] visual features from encoder layer

  Returns:
    Updated hidden_states with visual features added at visual positions
  """
  # Expand mask to [batch, seq_len, 1] for broadcasting
  mask_expanded = bidirectional_mask[:, :, jnp.newaxis]
  # Use cumsum to map each True position in mask to its index in visual_embeds
  visual_token_idx = jnp.cumsum(bidirectional_mask, axis=1) - 1  # [batch, seq_len], 0-indexed

  # Gather visual tokens: for each position, get the corresponding visual token
  batch_idx = jnp.arange(hidden_states.shape[0])[:, jnp.newaxis]  # [batch, 1]
  visual_embeds_scattered = visual_embeds[batch_idx, visual_token_idx, :]  # [batch, seq_len, hidden]
  # Only add where mask is True: hidden_states += visual_embeds * mask
  hidden_states = hidden_states + visual_embeds_scattered * mask_expanded
  return hidden_states


class NNXSequentialPipelineStage(nnx.Module):
  """Sequential unscanned series of decoder layers formatted for a single pipeline stage."""

  def __init__(
      self, layer_cls, num_layers: int, config: Config, mesh: Mesh, quant: Quant, model_mode: str, *, rngs: nnx.Rngs
  ):
    self.config = config
    self.scan_layers = config.scan_layers
    self.num_layers = num_layers
    # Dynamically assign layers with explicit string names to ensure correct PyTree paths (layers_0)
    for i in range(num_layers):
      layer = layer_cls(config=config, mesh=mesh, quant=quant, model_mode=model_mode, rngs=rngs)
      setattr(self, f"layers_{i}", layer)

  def __call__(self, inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, **kwargs):
    for i in range(self.num_layers):
      layer = getattr(self, f"layers_{i}")
      out = layer(inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, **kwargs)
      inputs = out[0] if isinstance(out, tuple) else out
    if self.scan_layers:
      return inputs, None
    return inputs


class NNXScannedPipelineStage(nnx.Module):
  """Scanned block of decoder layers formatted for a single pipeline stage."""

  def __init__(
      self, layer_cls, num_layers: int, config: Config, mesh: Mesh, quant: Quant, model_mode: str, *, rngs: nnx.Rngs
  ):
    self.config = config

    def create_layer_fn(rng):
      return layer_cls(config=config, mesh=mesh, quant=quant, model_mode=model_mode, rngs=rng)

    # Workaround for Deepseek MTP test failure.
    # TODO: Handle this properly.
    try:
      forked_rngs = rngs.fork(split=num_layers)
    except:  # pylint: disable=bare-except
      forked_rngs = rngs

    out_axes = nnx.StateAxes({nnx.Param: config.param_scan_axis, ...: 0})
    self.scanned_layers = nnx.vmap(
        create_layer_fn,
        in_axes=0,
        out_axes=out_axes,
        axis_name="layers_per_stage",
        transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"},
    )(forked_rngs)

  def __call__(self, inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, **kwargs):
    graphdef, params, state = nnx.split(self.scanned_layers, nnx.Param, ...)

    scan_axis = self.config.param_scan_axis
    if scan_axis != 0:
      params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0), params)

    def layer_fn(carry, scanned_vars):
      current_params, current_state = scanned_vars
      layer = nnx.merge(graphdef, current_params, current_state)
      layer_out = layer(carry, decoder_segment_ids, decoder_positions, deterministic, model_mode, **kwargs)
      new_carry = layer_out[0] if isinstance(layer_out, tuple) else layer_out
      return new_carry, nnx.state(layer)

    final_carry, scanned_state = jax.lax.scan(layer_fn, inputs, (params, state))

    if scan_axis != 0:
      scanned_params, scanned_other = scanned_state.split(nnx.Param, ...)
      scanned_params = jax.tree.map(lambda x: jnp.moveaxis(x, 0, scan_axis), scanned_params)
      scanned_state = nnx.State.merge(scanned_params, scanned_other)

    self.scanned_layers = nnx.merge(graphdef, scanned_state)

    if self.config.scan_layers:
      return final_carry, None
    return final_carry


class NNXDecoder(nnx.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture, using NNX."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: None | Quant = None,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    self.rngs = rngs

    decoder_block_classes = self.get_decoder_layers()

    self.decoder_norm = self.get_norm_layer(num_features=config.emb_dim, rngs=rngs)(
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        parameter_memory_host_offload=config.parameter_memory_host_offload,
    )

    if config.trainable_position_size > 0:
      self.position_embedder = Embed(
          num_embeddings=config.trainable_position_size,
          num_features=config.emb_dim,
          dtype=config.dtype,
          embedding_init=nn.initializers.normal(stddev=1.0),
          config=config,
          mesh=self.mesh,
          rngs=rngs,
      )

    self.dropout = linears.Dropout(rate=config.dropout_rate, rngs=rngs, broadcast_dims=(-2,))

    self.positional_embedding = PositionalEmbedding(embedding_dims=config.base_emb_dim)

    if not config.logits_via_embedding:
      self.logits_dense = linears.DenseGeneral(
          in_features_shape=config.emb_dim,
          out_features_shape=config.vocab_size,
          weight_dtype=config.weight_dtype,
          dtype=jnp.float32 if config.logits_dot_in_fp32 else config.dtype,
          kernel_axes=("embed", "vocab"),
          shard_mode=config.shard_mode,
          matmul_precision=self.config.matmul_precision,
          parameter_memory_host_offload=config.parameter_memory_host_offload,
          rngs=rngs,
      )

    self.scanned_layers = None
    self.is_deepseek = self.config.decoder_block == DecoderBlockType.DEEPSEEK
    self.is_gemma3 = self.config.decoder_block == DecoderBlockType.GEMMA3

    if config.using_pipeline_parallelism:

      def stage_factory(rngs):
        return self._get_pipeline_stage_module(decoder_block_classes, rngs)

      self.pipeline_module = create_nnx_pipeline(
          config=config,
          stage_factory=stage_factory,
          mesh=mesh,
          remat_policy=self.get_remat_policy(),
          rngs=rngs,
      )

      if self.is_deepseek:
        assert len(decoder_block_classes) == 2
        dense_cls, moe_cls = decoder_block_classes
        if config.scan_layers:
          self.dense_layers = self._create_scanned_layers(
              dense_cls, length=config.first_num_dense_layers, metadata_axis_name="dense_layers", rngs=rngs
          )
          num_moe_outside = (config.num_decoder_layers - config.first_num_dense_layers) - config.pipeline_parallel_layers
          if num_moe_outside > 0:
            self.moe_layers_outside_pipeline = self._create_scanned_layers(
                moe_cls, length=num_moe_outside, metadata_axis_name="moe_layers", rngs=rngs
            )
        else:
          self.num_dense_layers = config.first_num_dense_layers
          for i in range(self.num_dense_layers):
            self._create_and_register_layer(dense_cls, rngs, "dense_layers", i)

          self.num_moe_outside_pipeline = (
              config.num_decoder_layers - config.first_num_dense_layers
          ) - config.pipeline_parallel_layers
          if self.num_moe_outside_pipeline > 0:
            for i in range(self.num_moe_outside_pipeline):
              self._create_and_register_layer(moe_cls, rngs, "moe_layers_outside_pipeline", i)

      else:
        remaining_layers = config.num_decoder_layers - config.pipeline_parallel_layers
        if remaining_layers > 0:
          base_cls = decoder_block_classes[0]
          if config.scan_layers:
            self.layers_outside_pipeline = self._create_scanned_layers(
                base_cls, length=remaining_layers, metadata_axis_name="layers", rngs=rngs
            )
          else:
            self.num_layers_outside_pipeline = remaining_layers
            for i in range(self.num_layers_outside_pipeline):
              self._create_and_register_layer(base_cls, rngs, "layers_outside_pipeline", i)

    else:
      # Setup for Standard Non-Pipeline Execution
      if self.config.scan_layers:
        if self.is_deepseek:
          assert len(decoder_block_classes) == 2
          dense_cls, moe_cls = decoder_block_classes
          self.dense_layers = self._create_scanned_layers(
              dense_cls, length=config.first_num_dense_layers, metadata_axis_name="dense_layers", rngs=rngs
          )
          num_moe = config.num_decoder_layers - config.first_num_dense_layers
          self.moe_layers = self._create_scanned_layers(
              moe_cls, length=num_moe, metadata_axis_name="moe_layers", rngs=rngs
          )
        elif self.is_gemma3:
          attention_pattern_length = len(gemma3.GEMMA3_ATTENTION_PATTERN)
          scan_length = config.num_decoder_layers // attention_pattern_length
          num_remaining_layers = config.num_decoder_layers % attention_pattern_length
          layer_kwargs = {"num_of_layers": attention_pattern_length}
          rem_layer_kwargs = {"num_of_layers": num_remaining_layers}
          RemattedGemma3Block = gemma3.Gemma3ScannableBlock
          if scan_length > 0:
            self.layers = self._create_scanned_layers(
                RemattedGemma3Block, length=scan_length, metadata_axis_name="layers", rngs=rngs, **layer_kwargs
            )
          self.layers_remainder = RemattedGemma3Block(
              config=self.config, mesh=mesh, quant=self.quant, model_mode=self.model_mode, **rem_layer_kwargs, rngs=rngs
          )
        else:
          layer_cls = decoder_block_classes[0]
          num_layers = int(config.num_decoder_layers / config.inhomogeneous_layer_cycle_interval)
          layer_kwargs = {}
          if config.decoder_block == DecoderBlockType.LLAMA4:
            layer_kwargs = {
                "nope_layer_interval": self.config.nope_layer_interval,
                "interleave_moe_layer_step": self.config.interleave_moe_layer_step,
            }
          self.layers = self._create_scanned_layers(
              layer_cls, length=num_layers, metadata_axis_name="layers", rngs=rngs, **layer_kwargs
          )
      else:
        if self.is_deepseek:
          dense_cls, moe_cls = decoder_block_classes
          self.num_dense_layers = config.first_num_dense_layers
          for i in range(self.num_dense_layers):
            self._create_and_register_layer(dense_cls, rngs, "dense_layers", i)
          self.num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers
          for i in range(self.num_moe_layers):
            self._create_and_register_layer(moe_cls, rngs, "moe_layers", i)
        else:
          layer_cls = decoder_block_classes[0]
          self.num_layers = config.num_decoder_layers
          for lyr in range(self.num_layers):
            layer_kwargs = {}
            if config.decoder_block == DecoderBlockType.GEMMA3:
              layer_kwargs = {"attention_type": gemma3.get_attention_type(layer_id=lyr)}
            elif config.decoder_block == DecoderBlockType.LLAMA4:
              layer_kwargs = {
                  "is_nope_layer": llama4.determine_is_nope_layer(lyr, self.config.nope_layer_interval),
                  "is_moe_layer": llama4.determine_is_moe_layer(lyr, self.config.interleave_moe_layer_step),
              }
            elif config.decoder_block == DecoderBlockType.QWEN3_NEXT:
              layer_kwargs = {"layer_idx": lyr}
            elif config.decoder_block == DecoderBlockType.GPT_OSS:
              layer_kwargs = {"attention_type": gpt_oss.get_attention_type(layer_id=lyr)}
            elif config.decoder_block == DecoderBlockType.OLMO3:
              layer_kwargs = {"attention_type": olmo3.get_attention_type(layer_id=lyr)}
            self._create_and_register_layer(layer_cls, rngs, "layers", lyr, **layer_kwargs)

  def _get_pipeline_stage_module(self, decoder_blocks, rngs):
    """Retrieves the wrapper module formatted for single pipeline stage execution."""
    cfg = self.config
    base_stage_cls = decoder_blocks[1] if self.is_deepseek else decoder_blocks[0]

    if cfg.num_layers_per_pipeline_stage == 1:
      return self._create_single_layer(base_stage_cls, rngs)
    elif cfg.scan_layers_per_stage or cfg.scan_layers:
      return NNXScannedPipelineStage(
          base_stage_cls, cfg.num_layers_per_pipeline_stage, cfg, self.mesh, self.quant, self.model_mode, rngs=rngs
      )
    return NNXSequentialPipelineStage(
        base_stage_cls, cfg.num_layers_per_pipeline_stage, cfg, self.mesh, self.quant, self.model_mode, rngs=rngs
    )

  def _create_and_register_layer(self, layer_cls, rngs, base_name, i, **layer_kwargs):
    attr_name = f"{base_name}_{i}"
    layer = self._create_single_layer(layer_cls, rngs, **layer_kwargs)
    setattr(self, attr_name, layer)

  def _create_single_layer(self, decoder_layer_class, rngs, **kwargs):
    """Helper to create a single layer (Linen or NNX)."""
    if issubclass(decoder_layer_class, nnx.Module):
      return decoder_layer_class(
          config=self.config, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, rngs=rngs, **kwargs
      )
    else:
      layer_linen = decoder_layer_class(
          config=self.config, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, **kwargs
      )
      return nnx_wrappers.ToNNX(layer_linen, rngs=rngs)

  def _create_scanned_layers(
      self, decoder_layer_class, length: int, metadata_axis_name: str, rngs: nnx.Rngs, **layer_kwargs
  ):
    """Creates a VMapped stack of layers, forcing parameter init for Compact modules."""

    def create_layer_fn(rng):
      return decoder_layer_class(
          config=self.config, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, rngs=rng, **layer_kwargs
      )

    # Workaround for Deepseek MTP test failure.
    # TODO: Handle this properly.
    try:
      forked_rngs = rngs.fork(split=length)
    except:  # pylint: disable=bare-except
      forked_rngs = rngs

    out_axes = nnx.StateAxes({nnx.Param: self.config.param_scan_axis, ...: 0})
    layers_vmapped = nnx.vmap(
        create_layer_fn,
        in_axes=0,
        out_axes=out_axes,
        axis_name=metadata_axis_name,
        transform_metadata={nnx.PARTITION_NAME: metadata_axis_name},
    )(forked_rngs)
    return layers_vmapped

  def _apply_layer_with_remat(self, layer: nnx.Module, y: jax.Array, policy: Any, prevent_cse: bool, **kwargs):
    """Helper to cleanly apply jax.checkpoint to a single unscanned layer or block."""
    graphdef, state = nnx.split(layer)

    def pure_layer_fn(state_in, y_in):
      merged_layer = nnx.merge(graphdef, state_in)
      out = merged_layer(y_in, **kwargs)
      return out, nnx.state(merged_layer)

    checkpointed_fn = jax.checkpoint(pure_layer_fn, policy=policy, prevent_cse=prevent_cse)
    out, new_state = checkpointed_fn(state, y)
    nnx.update(layer, new_state)
    return out

  def _apply_layers_sequentially(self, layers, x_in, *args, length: int, **kwargs):
    """Runs the layer stack using nnx.scan."""
    policy = self.get_remat_policy()
    prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
    graphdef, params, state = nnx.split(
        layers, nnx.Param, ...
    )  # state: the mutable state we carry (KV cache, RNGs, etc.)

    scan_axis = self.config.param_scan_axis
    if scan_axis != 0:
      # Move scan_axis to 0 so scan can iterate over it
      def move_axis_abstract_safe(x):
        if isinstance(x, jax.ShapeDtypeStruct):
          # Manually calculate new shape for abstract tracers
          new_shape = list(x.shape)
          ax = scan_axis if scan_axis >= 0 else len(new_shape) + scan_axis
          val = new_shape.pop(ax)
          new_shape.insert(0, val)
          return jax.ShapeDtypeStruct(tuple(new_shape), x.dtype)
        return jnp.moveaxis(x, scan_axis, 0)

      params = jax.tree.map(move_axis_abstract_safe, params)

    layer_cls = layers.__class__
    sig = inspect.signature(layer_cls.__call__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or "kwargs" in sig.parameters}

    def layer_fn(carry, scanned_vars):
      current_params, current_state = scanned_vars
      if self.config.parameter_memory_host_offload:
        current_params = jax.tree.map(lambda x: jax.device_put(x, max_utils.device_space()), current_params)
      # Merge using the SLICED state
      layer = nnx.merge(graphdef, current_params, current_state)

      # Run the layer (Filter kwargs if using the solution from previous turn)
      layer_out = layer(carry, *args, **valid_kwargs)
      new_carry = layer_out[0] if isinstance(layer_out, tuple) else layer_out
      return new_carry, nnx.state(layer)

    layer_fn = jax.checkpoint(layer_fn, policy=policy, prevent_cse=prevent_cse)
    final_carry, scanned_state = jax.lax.scan(layer_fn, x_in, (params, state))

    if scan_axis != 0:
      # Only move the axis back on the params, NOT the mutables!
      def move_axis_back_abstract_safe(x):
        if isinstance(x, jax.ShapeDtypeStruct):
          new_shape = list(x.shape)
          val = new_shape.pop(0)
          ax = scan_axis if scan_axis >= 0 else len(new_shape) + 1 + scan_axis
          new_shape.insert(ax, val)
          return jax.ShapeDtypeStruct(tuple(new_shape), x.dtype)
        return jnp.moveaxis(x, 0, scan_axis)

      params = jax.tree.map(move_axis_back_abstract_safe, params)

    final_state = nnx.State.merge(params, scanned_state)
    # Skip direct mutation of 'layers' during compilation to avoid TraceContextError.
    # The caller will handle the final state update.
    return final_carry, final_state

  def _apply_interleaved_scanned_layers(
      self, layers, y, layer_args, layer_kwargs, start_idx, end_idx, engram_indices, decoder_input_tokens
  ):
    """Applies a mix of scanned standard layers and unscanned Engram layers efficiently using native NNX state slicing."""
    policy = self.get_remat_policy()
    prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
    graphdef, params, mutables = nnx.split(layers, nnx.Param, ...)

    scan_axis = self.config.param_scan_axis
    if scan_axis != 0:
      max_logging.log(f"nnx_decoders: Moving param scan_axis from {scan_axis} to 0 for interleaved scan.")
      params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0), params)

    def get_chunk(pytree, start, end):
      return jax.tree.map(lambda x: x[start:end], pytree)

    updated_mutables_chunks = []
    current_idx = start_idx

    while current_idx < end_idx:
      if current_idx in engram_indices:
        # Single engram layer execution
        eng_params = get_chunk(params, current_idx, current_idx + 1)
        eng_mutables = get_chunk(mutables, current_idx, current_idx + 1)

        # Squeeze the vmapped 'layers' dimension out for isolated execution
        eng_params = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), eng_params)
        eng_mutables = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), eng_mutables)

        if self.config.parameter_memory_host_offload:
          eng_params = jax.tree.map(lambda x: jax.device_put(x, max_utils.device_space()), eng_params)

        layer = nnx.merge(graphdef, eng_params, eng_mutables)
        kwargs_with_tokens = {**layer_kwargs, "decoder_input_tokens": decoder_input_tokens, "layer_idx": current_idx}

        sig = inspect.signature(layer.__call__)
        valid_kwargs = {k: v for k, v in kwargs_with_tokens.items() if k in sig.parameters or "kwargs" in sig.parameters}

        layer_out = layer(y, *layer_args, **valid_kwargs)
        y = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        _, new_eng_mutables = nnx.split(layer, nnx.Param, ...)
        new_eng_mutables = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), new_eng_mutables)
        updated_mutables_chunks.append(new_eng_mutables)
        current_idx += 1
      else:
        # Scan a continuous chunk of non-engram layers
        next_engrams = [l for l in engram_indices if l > current_idx]
        if next_engrams:
          min_next_engram = min(next_engrams)
          next_boundary = min(end_idx, min_next_engram)
        else:
          next_boundary = end_idx

        chunk_params = get_chunk(params, current_idx, next_boundary)
        chunk_mutables = get_chunk(mutables, current_idx, next_boundary)

        sig = inspect.signature(layers.__call__)
        valid_kwargs = {k: v for k, v in layer_kwargs.items() if k in sig.parameters or "kwargs" in sig.parameters}

        def layer_fn(carry, scanned_vars):
          curr_p, curr_m = scanned_vars
          if self.config.parameter_memory_host_offload:
            curr_p = jax.tree.map(lambda x: jax.device_put(x, max_utils.device_space()), curr_p)
          l = nnx.merge(graphdef, curr_p, curr_m)
          l_out = l(carry, *layer_args, **valid_kwargs)
          n_carry = l_out[0] if isinstance(l_out, tuple) else l_out
          _, n_mut = nnx.split(l, nnx.Param, ...)
          return n_carry, n_mut

        layer_fn = jax.checkpoint(layer_fn, policy=policy, prevent_cse=prevent_cse)
        y, new_chunk_mutables = jax.lax.scan(layer_fn, y, (chunk_params, chunk_mutables))
        updated_mutables_chunks.append(new_chunk_mutables)
        current_idx = next_boundary

    if updated_mutables_chunks:
      final_mutables = jax.tree.map(lambda *chunks: jnp.concatenate(chunks, axis=0), *updated_mutables_chunks)
    else:
      final_mutables = mutables

    if scan_axis != 0:
      max_logging.log(f"nnx_decoders: Moving param scan_axis 0 back to {scan_axis} for interleaved scan.")
      # Only move the axis back on params!
      params = jax.tree.map(lambda x: jnp.moveaxis(x, 0, scan_axis), params)

    final_state = nnx.State.merge(params, final_mutables)
    nnx.update(layers, final_state)
    return y, layers

  def _run_unscanned_layers_loop(
      self,
      base_name,
      num_layers,
      y,
      layer_args,
      layer_kwargs,
      kv_caches=None,
      deepstack_visual_embeds=None,
      bidirectional_mask=None,
      layer_idx_offset=0,
      decoder_input_tokens=None,
  ):
    """DRY Helper for looping unscanned lists of layers while correctly handling remat, state, engrams, and KV cache."""
    policy = self.get_remat_policy()
    prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)

    def pure_layer_fn(graphdef, state_in, y_in, kv_in, dynamic_kwargs):
      merged_layer = nnx.merge(graphdef, state_in)
      out_y, out_kv = merged_layer(y_in, *layer_args, kv_cache=kv_in, **dynamic_kwargs)
      return out_y, out_kv, nnx.state(merged_layer)

    checkpointed_fn = jax.checkpoint(pure_layer_fn, policy=policy, prevent_cse=prevent_cse)

    for lyr in range(num_layers):
      attr_name = f"{base_name}_{lyr}"
      layer = getattr(self, attr_name)
      graphdef, state = nnx.split(layer)
      global_lyr = layer_idx_offset + lyr

      # Prepare dynamic KV Cache unwrapping
      kv_cache = None
      if kv_caches is not None and self.config.decoder_block != DecoderBlockType.QWEN3_NEXT:
        kv_cache = kv_caches[global_lyr]
      elif kv_caches is not None and self.config.decoder_block == DecoderBlockType.QWEN3_NEXT:
        if (global_lyr + 1) % self.config.inhomogeneous_layer_cycle_interval == 0:
          kv_cache = (kv_caches["key_cache"][global_lyr], kv_caches["value_cache"][global_lyr])

      # Prepare dynamic Kwargs (Engrams, Layer ID)
      current_kwargs = dict(layer_kwargs)
      if self.config.engram_layers:
        current_kwargs["decoder_input_tokens"] = decoder_input_tokens
      if self.config.decoder_block == DecoderBlockType.DEEPSEEK:
        current_kwargs["layer_idx"] = global_lyr

      y, returned_cache, new_state = checkpointed_fn(graphdef, state, y, kv_cache, current_kwargs)
      # Re-merge the state back to the explicit attribute to prevent cross-boundary TraceContextErrors
      setattr(self, attr_name, nnx.merge(graphdef, new_state))

      # Write updated KV Cache back properly
      if kv_caches is not None and returned_cache is not None:
        if self.config.decoder_block != DecoderBlockType.QWEN3_NEXT:
          kv_caches[global_lyr] = returned_cache
        elif (global_lyr + 1) % self.config.inhomogeneous_layer_cycle_interval == 0:
          kv_caches["key_cache"][global_lyr] = returned_cache[0]
          kv_caches["value_cache"][global_lyr] = returned_cache[1]

      if deepstack_visual_embeds is not None and global_lyr < len(deepstack_visual_embeds):
        visual_embeds = deepstack_visual_embeds[global_lyr]
        if bidirectional_mask is not None and visual_embeds is not None:
          y = deepstack_process(y, bidirectional_mask, visual_embeds)

    return y

  def get_decoder_layers(self):
    """Retrieves decoder layer classes based on config using a dictionary lookup."""
    cfg = self.config

    def get_scannable(normal_cls, scannable_cls):
      return [scannable_cls] if cfg.scan_layers else [normal_cls]

    def get_deepseek():
      if cfg.use_batch_split_schedule:
        return [deepseek_batchsplit.DeepSeekDenseLayer, deepseek_batchsplit.DeepSeekMoELayer]
      return [deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer]

    layer_map = {
        DecoderBlockType.DEFAULT: [NNXDecoderLayer],
        DecoderBlockType.LLAMA2: [llama2.LlamaDecoderLayer],
        DecoderBlockType.MISTRAL: [mistral.MistralDecoderLayer],
        DecoderBlockType.MIXTRAL: [mixtral.MixtralDecoderLayer],
        DecoderBlockType.GEMMA: [gemma.GemmaDecoderLayer],
        DecoderBlockType.GEMMA2: [gemma2.Gemma2DecoderLayer],
        DecoderBlockType.GEMMA3: [gemma3.Gemma3DecoderLayer],
        DecoderBlockType.GPT3: [gpt3.Gpt3DecoderLayer],
        DecoderBlockType.QWEN3: [qwen3.Qwen3DecoderLayer],
        DecoderBlockType.QWEN3_MOE: [qwen3.Qwen3MoeDecoderLayer],
        DecoderBlockType.SIMPLE: [simple_layer.SimpleDecoderLayer],
        DecoderBlockType.SIMPLE_MLP: [simple_layer.SimpleMlpDecoderLayer],
        DecoderBlockType.DEEPSEEK: get_deepseek(),
        DecoderBlockType.GPT_OSS: get_scannable(gpt_oss.GptOssDecoderLayer, gpt_oss.GptOssScannableBlock),
        DecoderBlockType.QWEN3_NEXT: get_scannable(qwen3.Qwen3NextDecoderLayer, qwen3.Qwen3NextScannableBlock),
        DecoderBlockType.LLAMA4: get_scannable(llama4.Llama4DecoderLayer, llama4.Llama4ScannableBlock),
        DecoderBlockType.OLMO3: get_scannable(olmo3.Olmo3DecoderLayer, olmo3.Olmo3ScannableBlock),
    }

    if cfg.decoder_block not in layer_map:
      raise ValueError(f"Incorrect decoder_block name {cfg.decoder_block.value=}")
    return layer_map[cfg.decoder_block]

  def minimal_policy(self, with_context=False, with_quantization=False):
    """Helper for creating minimal checkpoint policies."""
    names = [
        "query_proj",
        "value_proj",
        "key_proj",
        "qkv_proj",
        "out_proj",
        "mlpwi_0",
        "mlpwi_1",
        "mlpwi",
        "mlpwo",
    ]
    if with_context:
      names.append("context")
    if with_quantization:
      names.append("quantization")
    return jax.checkpoint_policies.save_only_these_names(*names)

  def get_remat_policy(self):
    """Get remat policy for jax.checkpoint."""
    policy = None
    cfg = self.config
    if cfg.remat_policy != "none":
      if cfg.remat_policy in ("minimal_with_context", "minimal_flash"):
        # save all
        if cfg.remat_policy == "minimal_flash":
          max_logging.log("WARNING: 'minimal_flash' will be deprecated soon, please use 'minimal_with_context' instead.")
        policy = self.minimal_policy(with_context=True)
      elif cfg.remat_policy == "minimal":
        # save all except context
        policy = self.minimal_policy()
      elif cfg.remat_policy == "minimal_with_quantization":
        if cfg.scan_layers:
          warnings.warn(
              "Scan layers can introduce overhead to checkpointed values that in some configurations is slower"
              "than not checkpointing at all. If you are using scan layers, benchmark with and without quantization "
              "checkpointing in your workflow to see which is faster. Without scan layers, checkpointing quantizations is "
              "beneficial for performance."
          )
        policy = self.minimal_policy(with_context=False, with_quantization=True)
      elif cfg.remat_policy == "minimal_with_context_and_quantization":
        if cfg.scan_layers:
          warnings.warn(
              "Scan layers can introduce overhead to checkpointed values that in some configurations is slower"
              "than not checkpointing at all. If you are using scan layers, benchmark with and without quantization "
              "checkpointing in your workflow to see which is faster. Without scan layers, checkpointing quantizations is "
              "beneficial for performance."
          )
        policy = self.minimal_policy(with_context=True, with_quantization=True)
      elif cfg.remat_policy == "save_dot_with_context_except_mlp":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj", "context", "out_proj"
        )
      elif cfg.remat_policy == "save_dot_except_mlpwi":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj", "mlpwo"
        )
      elif cfg.remat_policy == "save_dot_except_mlp":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj"
        )
      elif cfg.remat_policy == "save_qkv_proj":
        policy = jax.checkpoint_policies.save_only_these_names("query_proj", "value_proj", "key_proj", "qkv_proj")
      elif cfg.remat_policy == "qkv_proj_offloaded":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=["query_proj", "value_proj", "key_proj"],
            offload_src="device",
            offload_dst="pinned_host",
        )
      elif cfg.remat_policy == "minimal_offloaded":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=[
                "query_proj",
                "value_proj",
                "key_proj",
                "qkv_proj",
                "out_proj",
                "mlpwi_0",
                "mlpwi_1",
                "mlpwi",
                "mlpwo",
            ],
            offload_src="device",
            offload_dst="pinned_host",
        )
      elif cfg.remat_policy == "custom":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=cfg.tensors_on_device,
            names_which_can_be_offloaded=cfg.tensors_to_offload,
            offload_src="device",
            offload_dst="pinned_host",
        )
      elif cfg.remat_policy == "save_out_proj":
        policy = jax.checkpoint_policies.save_only_these_names("out_proj")
      else:
        assert cfg.remat_policy == "full", "Remat policy needs to be on list of remat policies"
    return policy

  def get_norm_layer(self, num_features: int, rngs: nnx.Rngs):
    """Helper to retrieve the correct normalization layer class based on config, partially applied with common arguments."""
    if self.config.decoder_block in (
        DecoderBlockType.DEFAULT,
        DecoderBlockType.LLAMA2,
        DecoderBlockType.MISTRAL,
        DecoderBlockType.MIXTRAL,
        DecoderBlockType.DEEPSEEK,
        DecoderBlockType.GEMMA,
        DecoderBlockType.GEMMA2,
        DecoderBlockType.GEMMA3,
        DecoderBlockType.QWEN3,
        DecoderBlockType.QWEN3_MOE,
        DecoderBlockType.GPT_OSS,
        DecoderBlockType.SIMPLE,
        DecoderBlockType.SIMPLE_MLP,
        DecoderBlockType.LLAMA4,
        DecoderBlockType.OLMO3,
    ):
      return functools.partial(RMSNorm, num_features=num_features, shard_mode=self.config.shard_mode, rngs=rngs)
    elif self.config.decoder_block == DecoderBlockType.GPT3:
      return functools.partial(
          gpt3.Gpt3LayerNorm, num_features=num_features, reductions_in_fp32=False, use_bias=True, rngs=rngs
      )
    elif self.config.decoder_block == DecoderBlockType.QWEN3_NEXT:
      return functools.partial(
          normalizations.Qwen3NextRMSNorm, num_features=num_features, shard_mode=self.config.shard_mode, rngs=rngs
      )
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  def _apply_embedding(
      self,
      shared_embedding: nnx.Module,
      decoder_input_tokens,
      decoder_positions,
      deterministic,
      model_mode,
      image_embeddings=None,
      bidirectional_mask=None,
      image_masks=None,
      audio_embeddings=None,
      audio_masks=None,
  ):
    """Applies token embedding, adds positional embedding, and merges multimodal embeddings if provided."""
    cfg = self.config
    y = shared_embedding(decoder_input_tokens.astype("int32"), model_mode=model_mode)

    if image_embeddings is not None and cfg.use_multimodal:
      if cfg.model_name in {
          "gemma3-4b",
          "gemma3-12b",
          "gemma3-27b",
          "llama4-17b-16e",
          "llama4-17b-128e",
          "qwen3-omni-30b-a3b",
      }:
        y = mm_utils.merge_mm_embeddings(
            text_embeddings=y,
            multimodal_embeddings=image_embeddings,
            mask=bidirectional_mask,
            token_masks=image_masks,
        )
      else:
        raise ValueError(f"Unsupported model_name for multimodal: {cfg.model_name}")

    if audio_embeddings is not None and cfg.use_audio:
      if cfg.model_name in ["qwen3-omni-30b-a3b"]:
        y = mm_utils.merge_mm_embeddings(
            text_embeddings=y,
            multimodal_embeddings=audio_embeddings,
            mask=audio_masks,
            token_masks=None,
        )
      else:
        raise ValueError(f"Unsupported model_name for audio: {cfg.model_name}")

    y = self.dropout(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    if cfg.use_untrainable_positional_embedding:
      y += self.positional_embedding(y, decoder_positions)

    if cfg.trainable_position_size > 0 and self.position_embedder:
      y += self.position_embedder(decoder_positions.astype("int32"), model_mode=model_mode)

    return y

  def apply_output_head(self, shared_embedding, y, deterministic, model_mode):
    """Applies final normalization and projects hidden states to logits."""
    cfg = self.config
    if cfg.shard_mode == ShardMode.EXPLICIT:
      norm_out_sharding = create_sharding(self.mesh, ("activation_batch", "activation_length_no_exp", "activation_embed"))
    else:
      norm_out_sharding = None

    y = self.decoder_norm(y, out_sharding=norm_out_sharding)
    y = self.dropout(y, deterministic=deterministic)  # NNX call

    if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
      out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
    else:
      out_sharding = create_sharding(
          self.mesh, ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_vocab")
      )

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      if isinstance(shared_embedding, nnx.Module):
        embedding_table = shared_embedding.embedding.value
      else:
        embedding_table = shared_embedding.variables["params"]["embedding"]
      if isinstance(embedding_table, nn.spmd.LogicallyPartitioned):
        embedding_table = embedding_table.unbox()
      attend_dtype = jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype
      logits = attend_on_embedding(y, embedding_table, attend_dtype, self.config, out_sharding)

      if self.config.normalize_embedding_logits:
        # Correctly normalize pre-softmax logits for this shared case.
        logits = logits / jnp.sqrt(y.shape[-1])
      if cfg.final_logits_soft_cap:
        logits = logits / cfg.final_logits_soft_cap
        logits = jnp.tanh(logits) * cfg.final_logits_soft_cap
    else:
      logits = self.logits_dense(y, out_sharding=out_sharding)

    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)

    return logits

  def __call__(
      self,
      shared_embedding: Any,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      deterministic=False,
      model_mode=MODEL_MODE_TRAIN,
      previous_chunk=None,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
      bidirectional_mask: None | Any = None,
      image_embeddings: None | jnp.ndarray = None,
      image_masks: None | jnp.ndarray = None,
      kv_caches: list[jax.Array] | None = None,
      attention_metadata=None,
      audio_embeddings: None | jnp.ndarray = None,
      audio_masks: None | jnp.ndarray = None,
      deepstack_visual_embeds: None | list[jnp.ndarray] = None,
  ):
    cfg = self.config
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    # [batch, length] -> [batch, length, emb_dim]
    y = self._apply_embedding(
        shared_embedding,
        decoder_input_tokens,
        decoder_positions,
        deterministic,
        model_mode,
        image_embeddings,
        bidirectional_mask,
        image_masks,
        audio_embeddings,
        audio_masks,
    )

    mhc_expand, mhc_reduce = mhc.get_functions(cfg.mhc_expansion_rate)
    if cfg.mhc_expansion_rate > 1:
      # (batch, length, emb_dim) --> (batch, length, mhc_expansion_rate, emb_dim)
      y = mhc_expand(y)

    layer_args = (decoder_segment_ids, decoder_positions, deterministic, model_mode)

    layer_kwargs = {}
    if cfg.decoder_block == DecoderBlockType.GEMMA3:
      layer_kwargs["bidirectional_mask"] = bidirectional_mask

    if attention_metadata is not None:
      layer_kwargs["attention_metadata"] = attention_metadata
    elif cfg.decoder_block == DecoderBlockType.DEEPSEEK and cfg.scan_layers:
      layer_kwargs = {"previous_chunk": previous_chunk, "page_state": page_state, "slot": slot}

    # -------------------------------------------------------------------------
    # Execution Routing (Pipeline vs Direct)
    # -------------------------------------------------------------------------
    if cfg.using_pipeline_parallelism:
      logical_partition_spec = self.pipeline_module.get_weight_sharding() if cfg.pipeline_fsdp_ag_once else None

      if self.is_deepseek:
        logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(cfg.logical_axis_rules)
        with self.mesh, nn.partitioning.axis_rules(logical_axis_rules_pp_as_dp):
          if cfg.scan_layers:
            if cfg.engram_layers:
              y, self.dense_layers = self._apply_interleaved_scanned_layers(
                  self.dense_layers,
                  y,
                  layer_args,
                  layer_kwargs,
                  start_idx=0,
                  end_idx=cfg.first_num_dense_layers,
                  engram_indices=cfg.engram_layers,
                  decoder_input_tokens=decoder_input_tokens,
              )
              if hasattr(self, "moe_layers_outside_pipeline"):
                num_moe_outside = (cfg.num_decoder_layers - cfg.first_num_dense_layers) - cfg.pipeline_parallel_layers
                y, self.moe_layers_outside_pipeline = self._apply_interleaved_scanned_layers(
                    self.moe_layers_outside_pipeline,
                    y,
                    layer_args,
                    layer_kwargs,
                    start_idx=cfg.first_num_dense_layers,
                    end_idx=cfg.first_num_dense_layers + num_moe_outside,
                    engram_indices=cfg.engram_layers,
                    decoder_input_tokens=decoder_input_tokens,
                )
            else:
              y, new_state = self._apply_layers_sequentially(
                  self.dense_layers, y, *layer_args, length=cfg.first_num_dense_layers, **layer_kwargs
              )
              self._trace_safe_update(self.dense_layers, new_state)
              if hasattr(self, "moe_layers_outside_pipeline"):
                num_moe_outside = (cfg.num_decoder_layers - cfg.first_num_dense_layers) - cfg.pipeline_parallel_layers
                y, new_state = self._apply_layers_sequentially(
                    self.moe_layers_outside_pipeline, y, *layer_args, length=num_moe_outside, **layer_kwargs
                )
                self._trace_safe_update(self.moe_layers_outside_pipeline, new_state)
          else:
            y = self._run_unscanned_layers_loop(
                base_name="dense_layers",
                num_layers=self.num_dense_layers,
                y=y,
                layer_args=layer_args,
                layer_kwargs=layer_kwargs,
                kv_caches=kv_caches,
                deepstack_visual_embeds=deepstack_visual_embeds,
                bidirectional_mask=bidirectional_mask,
                layer_idx_offset=0,
                decoder_input_tokens=decoder_input_tokens,
            )
            if hasattr(self, "num_moe_outside_pipeline") and self.num_moe_outside_pipeline > 0:
              y = self._run_unscanned_layers_loop(
                  base_name="moe_layers_outside_pipeline",
                  num_layers=self.num_moe_outside_pipeline,
                  y=y,
                  layer_args=layer_args,
                  layer_kwargs=layer_kwargs,
                  kv_caches=kv_caches,
                  deepstack_visual_embeds=deepstack_visual_embeds,
                  bidirectional_mask=bidirectional_mask,
                  layer_idx_offset=cfg.first_num_dense_layers,
                  decoder_input_tokens=decoder_input_tokens,
              )

        y = self.pipeline_module(
            y,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            logical_partition_spec=logical_partition_spec,
        )

      else:
        # Standard Pipeline Run
        y = self.pipeline_module(
            y,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            logical_partition_spec=logical_partition_spec,
        )

        # Remaining standard layers
        if hasattr(self, "num_layers_outside_pipeline") or hasattr(self, "layers_outside_pipeline"):
          logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(cfg.logical_axis_rules)
          with self.mesh, nn.partitioning.axis_rules(logical_axis_rules_pp_as_dp):
            if cfg.scan_layers:
              y, new_state = self._apply_layers_sequentially(
                  self.layers_outside_pipeline,
                  y,
                  *layer_args,
                  length=len(self.layers_outside_pipeline.scanned_layers),
                  **layer_kwargs,
              )
              self._trace_safe_update(self.layers_outside_pipeline, new_state)
            else:
              y = self._run_unscanned_layers_loop(
                  base_name="layers_outside_pipeline",
                  num_layers=self.num_layers_outside_pipeline,
                  y=y,
                  layer_args=layer_args,
                  layer_kwargs=layer_kwargs,
                  kv_caches=kv_caches,
                  deepstack_visual_embeds=deepstack_visual_embeds,
                  bidirectional_mask=bidirectional_mask,
                  layer_idx_offset=cfg.pipeline_parallel_layers,
                  decoder_input_tokens=decoder_input_tokens,
              )

    else:
      # Non-Pipeline Run
      if cfg.scan_layers:
        if self.is_deepseek:
          if cfg.engram_layers:
            y, self.dense_layers = self._apply_interleaved_scanned_layers(
                self.dense_layers,
                y,
                layer_args,
                layer_kwargs,
                start_idx=0,
                end_idx=cfg.first_num_dense_layers,
                engram_indices=cfg.engram_layers,
                decoder_input_tokens=decoder_input_tokens,
            )
            num_moe = cfg.num_decoder_layers - cfg.first_num_dense_layers
            y, self.moe_layers = self._apply_interleaved_scanned_layers(
                self.moe_layers,
                y,
                layer_args,
                layer_kwargs,
                start_idx=cfg.first_num_dense_layers,
                end_idx=cfg.num_decoder_layers,
                engram_indices=cfg.engram_layers,
                decoder_input_tokens=decoder_input_tokens,
            )
          else:
            y, new_state = self._apply_layers_sequentially(
                self.dense_layers, y, *layer_args, length=cfg.first_num_dense_layers, **layer_kwargs
            )
            self._trace_safe_update(self.dense_layers, new_state)
            num_moe = cfg.num_decoder_layers - cfg.first_num_dense_layers

            # Use raw deepseek_batchsplit logic for MoE scanned layers to minimize VRAM overhead
            layer_is_initializing = self.quant is not None and len(nnx.state(self.moe_layers, "aqt")) == 0
            if cfg.use_batch_split_schedule and not layer_is_initializing:
              raw_weights = nnx.to_pure_dict(nnx.state(self.moe_layers, nnx.Param))
              y = deepseek_batchsplit.scan_batch_split_layers(
                  y,
                  raw_weights,
                  decoder_positions,
                  decoder_segment_ids,
                  model_mode=model_mode,
                  mesh=self.mesh,
                  quant=self.quant,
                  cfg=cfg,
                  policy=self.get_remat_policy(),
              )
            else:
              y, new_state = self._apply_layers_sequentially(
                  self.moe_layers, y, *layer_args, length=num_moe, **layer_kwargs
              )
              self._trace_safe_update(self.moe_layers, new_state)

        elif self.is_gemma3:
          y = self._apply_gemma3_scanned_blocks(
              y,
              decoder_segment_ids,
              decoder_positions,
              deterministic,
              model_mode,
              bidirectional_mask,
              previous_chunk,
              page_state,
              slot,
          )
        else:
          y, new_state = self._apply_layers_sequentially(
              self.layers, y, *layer_args, length=cfg.num_decoder_layers, **layer_kwargs
          )
          self._trace_safe_update(self.layers, new_state)
      else:
        if self.is_deepseek:
          y = self._run_unscanned_layers_loop(
              base_name="dense_layers",
              num_layers=self.num_dense_layers,
              y=y,
              layer_args=layer_args,
              layer_kwargs=layer_kwargs,
              kv_caches=kv_caches,
              deepstack_visual_embeds=deepstack_visual_embeds,
              bidirectional_mask=bidirectional_mask,
              layer_idx_offset=0,
              decoder_input_tokens=decoder_input_tokens,
          )
          y = self._run_unscanned_layers_loop(
              base_name="moe_layers",
              num_layers=self.num_moe_layers,
              y=y,
              layer_args=layer_args,
              layer_kwargs=layer_kwargs,
              kv_caches=kv_caches,
              deepstack_visual_embeds=deepstack_visual_embeds,
              bidirectional_mask=bidirectional_mask,
              layer_idx_offset=cfg.first_num_dense_layers,
              decoder_input_tokens=decoder_input_tokens,
          )
        else:
          y = self._run_unscanned_layers_loop(
              base_name="layers",
              num_layers=self.num_layers,
              y=y,
              layer_args=layer_args,
              layer_kwargs=layer_kwargs,
              kv_caches=kv_caches,
              deepstack_visual_embeds=deepstack_visual_embeds,
              bidirectional_mask=bidirectional_mask,
              layer_idx_offset=0,
              decoder_input_tokens=decoder_input_tokens,
          )

    assert isinstance(y, jax.Array)
    # After the final transformer layer, `y` holds the raw, un-normalized hidden state.
    if cfg.mhc_expansion_rate > 1:
      # (batch, length, mhc_expansion_rate, emb_dim) --> (batch, length, emb_dim)

      hidden_state = mhc_reduce(y)
    else:
      hidden_state = y

    # When invoking from vLLM with RPA attention, logit computation is deferred to a later stage.
    if cfg.attention == "vllm_rpa":
      if not cfg.logits_via_embedding and hasattr(self, "logits_dense"):
        if self.quant is not None and len(nnx.state(self.logits_dense, "aqt")) == 0:
          _ = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)
      logits = None
    # When vocab tiling is enabled in training mode, full logits won't generate to reduce memory
    # Instead, we keep track on the hidden states, which has smaller size compared to full logits
    elif cfg.num_vocab_tiling > 1 and self.model_mode == MODEL_MODE_TRAIN:
      logits = None
      self.sow(nnx.Intermediate, "hidden_states", hidden_state)
    else:
      logits = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)

    return logits, hidden_state, kv_caches

  def _trace_safe_update(self, layer, new_state):
    """Updates the layer state only if not currently in a tracing context where mutation is forbidden."""
    # Check if we are in an abstract tracing context (like jax.eval_shape)
    # where mutation of variables from outer scopes is disallowed.
    is_tracing = any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(new_state))
    if not is_tracing:
      nnx.update(layer, new_state)

  def _apply_gemma3_scanned_blocks(
      self,
      y,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      bidirectional_mask,
      previous_chunk,
      page_state,
      slot,
  ):
    """Applies Gemma3 scanned decoder blocks, handling main scan and remainders."""

    cfg = self.config

    # Define the repeating pattern length and calculate how many full blocks to scan
    attention_pattern_length = len(gemma3.GEMMA3_ATTENTION_PATTERN)
    scan_length = cfg.num_decoder_layers // attention_pattern_length

    layer_args = (decoder_segment_ids, decoder_positions, deterministic, model_mode)
    layer_kwargs = {"bidirectional_mask": bidirectional_mask}

    # Apply the main scan over the full blocks
    if scan_length > 0:
      y, new_state = self._apply_layers_sequentially(self.layers, y, *layer_args, length=scan_length, **layer_kwargs)
      self._trace_safe_update(self.layers, new_state)

    # Apply any remaining layers that did not fit into a full scanned block
    num_remaining_layers = cfg.num_decoder_layers % attention_pattern_length
    if num_remaining_layers > 0:
      policy = self.get_remat_policy()
      prevent_cse = maxtext_utils.should_prevent_cse_in_remat(cfg)

      def pure_gemma_fn(graphdef, state_in, y_in):
        merged_layer = nnx.merge(graphdef, state_in)
        out_y, _ = merged_layer(
            y_in, *layer_args, previous_chunk=previous_chunk, page_state=page_state, slot=slot, **layer_kwargs
        )
        return out_y, nnx.state(merged_layer)

      checkpointed_gemma_fn = jax.checkpoint(pure_gemma_fn, policy=policy, prevent_cse=prevent_cse)
      graphdef, state = nnx.split(self.layers_remainder)
      y, new_state = checkpointed_gemma_fn(graphdef, state, y)
      self.layers_remainder = nnx.merge(graphdef, new_state)

    return y


def decoder_as_linen(
    config: Config,
    mesh: Mesh,
    rngs: nnx.Rngs,
    model_mode: str,
    quant: None | Quant = None,
):
  """Creates a Decoder module."""
  module = nnx_wrappers.to_linen(
      NNXDecoder,
      config=config,
      mesh=mesh,
      model_mode=model_mode,
      rngs=rngs,
      quant=quant,
      name="decoder",
      abstract_init=False,
      metadata_fn=initializers.variable_to_logically_partitioned,
  )
  return module

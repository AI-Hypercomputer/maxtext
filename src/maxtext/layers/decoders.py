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

"""Module for decoder layers"""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
from typing import Any

from flax import linen as nn
from flax import nnx
from flax.linen.partitioning import ScanIn
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh
from MaxText import sharding
from MaxText.common_types import Config, DecoderBlockType, EP_AS_CONTEXT, ShardMode
from MaxText.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from maxtext.inference import page_manager
from maxtext.layers import linears
from maxtext.layers import mhc
from maxtext.layers import normalizations
from maxtext.layers import pipeline
from maxtext.layers import quantizations
from maxtext.layers.attentions import attention_as_linen
from maxtext.layers.embeddings import attend_on_embedding, embed_as_linen, positional_embedding_as_linen
from maxtext.layers.normalizations import rms_norm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.models import (
    deepseek,
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
from MaxText.sharding import create_sharding
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils

# ------------------------------------------------------------------------------
# The network: Decoder Definitions
# ------------------------------------------------------------------------------


class DecoderLayer(nn.Module):
  """
  Transformer decoder layer that attends to the encoder.
  This is the core, reusable building block for both the main model's
  decoder stack and the auxiliary MTP layers.
  """

  config: Config
  mesh: Mesh
  model_mode: str
  quant: None | Quant = None

  @nn.compact
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

    if model_mode == MODEL_MODE_PREFILL:
      inputs = _maybe_shard_with_logical(inputs, logical_axis_names)
    else:
      inputs = _maybe_shard_with_logical(inputs, logical_axis_names)

    inputs = checkpoint_name(inputs, "decoder_layer_input")
    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = rms_norm(
        num_features=inputs.shape[-1],
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_self_attention_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )(inputs)
    if model_mode == MODEL_MODE_PREFILL:
      lnx = _maybe_shard_with_logical(lnx, logical_axis_names)
    else:
      lnx = _maybe_shard_with_logical(lnx, logical_axis_names)

    attention_layer = attention_as_linen(
        config=self.config,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        inputs_q_shape=lnx.shape,
        inputs_kv_shape=lnx.shape,
        mesh=mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
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
        model_mode=model_mode,
    )

    attention_lnx, kv_cache = attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )

    if model_mode == MODEL_MODE_PREFILL:
      attention_lnx = _maybe_shard_with_logical(attention_lnx, logical_axis_names)
    else:
      attention_lnx = _maybe_shard_with_logical(attention_lnx, logical_axis_names)

    # MLP block.
    mlp_lnx = linears.mlp_block(
        in_features=lnx.shape[-1],
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        model_mode=model_mode,
        config=cfg,
        quant=self.quant,
        mesh=self.mesh,
    )(lnx, deterministic=deterministic)
    if model_mode == MODEL_MODE_PREFILL:
      mlp_lnx = _maybe_shard_with_logical(mlp_lnx, logical_axis_names)
    else:
      mlp_lnx = _maybe_shard_with_logical(mlp_lnx, logical_axis_names)

    next_layer_addition = mlp_lnx + attention_lnx

    next_layer_addition_dropped_out = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        next_layer_addition, deterministic=deterministic
    )

    layer_output = next_layer_addition_dropped_out + inputs
    if model_mode == MODEL_MODE_PREFILL:
      layer_output = _maybe_shard_with_logical(
          layer_output,
          logical_axis_names,
      )
    else:
      layer_output = _maybe_shard_with_logical(
          layer_output,
          logical_axis_names,
      )

    if cfg.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output, kv_cache


class SequentialBlockDecoderLayers(nn.Module):
  """Sequential unscanned series of decoder layers."""

  decoder_layer: Any
  num_decoder_layers: int
  config: Config
  mesh: Mesh
  quant: Quant
  model_mode: str

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids,
      decoder_positions,
      deterministic: bool,
      model_mode,
      slot: None | int = None,
      page_state: None | page_manager.PageState = None,
  ) -> jnp.ndarray:
    for lyr in range(self.num_decoder_layers):
      inputs = self.decoder_layer(
          config=self.config, mesh=self.mesh, name=f"layers_{lyr}", quant=self.quant, model_mode=model_mode
      )(
          inputs,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          slot=slot,
          page_state=page_state,
      )
      if self.config.scan_layers:
        inputs = inputs[0]  #  When scan_layers is True the decoder layers return (outputs, None).
    if self.config.scan_layers:
      return inputs, None  # pytype: disable=bad-return-type
    else:
      return inputs


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""

  config: Config
  mesh: Mesh
  quant: None | Quant = None
  model_mode: str = MODEL_MODE_TRAIN

  def setup(self):
    """Initialize decoder layer."""
    self.decoder_layer = self.get_decoder_layers()
    self.norm_layer = self.get_norm_layer(num_features=self.config.emb_dim)
    if self.config.using_pipeline_parallelism:
      pipeline_stage_module = self.get_pipeline_stage_module(self.decoder_layer)
      remat_policy = self.get_remat_policy()
      self.pipeline_module = pipeline.Pipeline(
          config=self.config, mesh=self.mesh, layers=pipeline_stage_module, remat_policy=remat_policy
      )

  def minimal_policy(self, with_context=False):
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
    return jax.checkpoint_policies.save_only_these_names(*names)

  def get_remat_policy(self):
    """Get remat policy"""
    policy = None
    cfg = self.config
    if cfg.remat_policy != "none":
      if cfg.remat_policy in ("minimal_with_context", "minimal_flash"):
        # save all
        if cfg.remat_policy == "minimal_flash":
          max_logging.log("WARNING: 'minimal_flash' will be deprecated soon, please use 'minimal_with_context' instead.")
          max_logging.log("WARNING: 'minimal_flash' will be deprecated soon, please use 'minimal_with_context' instead.")
        policy = self.minimal_policy(with_context=True)
      elif cfg.remat_policy == "minimal":
        # save all except context
        policy = self.minimal_policy()
      elif cfg.remat_policy == "save_dot_with_context_except_mlp":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
            "context",
            "out_proj",
        )
      elif cfg.remat_policy == "save_dot_except_mlpwi":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
            "out_proj",
            "mlpwo",
        )
      elif cfg.remat_policy == "save_dot_except_mlp":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
            "out_proj",
        )
      elif cfg.remat_policy == "save_qkv_proj":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
        )
      elif cfg.remat_policy == "qkv_proj_offloaded":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=["query_proj", "value_proj", "key_proj"],
            offload_src="device",
            offload_dst="pinned_host",
        )
      elif cfg.remat_policy == "minimal_offloaded":
        # offload all except context
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
        policy = jax.checkpoint_policies.save_only_these_names(
            "out_proj",
        )
      else:
        assert cfg.remat_policy == "full", "Remat policy needs to be on list of remat policies"
        policy = None
    return policy

  def get_decoder_layers(self):
    """Retrieves a list of decoder layer classes based on the `decoder_block` config.

    Returns:
        A list containing one or more `nn.Module` classes for the decoder.
    """
    match self.config.decoder_block:
      case DecoderBlockType.DEFAULT:
        return [DecoderLayer]
      case DecoderBlockType.LLAMA2:
        return [llama2.LlamaDecoderLayerToLinen]
      case DecoderBlockType.MISTRAL:
        # TODO(ranran): update to Mistral with sliding window attention
        return [mistral.MistralDecoderLayerToLinen]
      case DecoderBlockType.MIXTRAL:
        return [mixtral.MixtralDecoderLayerToLinen]
      case DecoderBlockType.DEEPSEEK:
        return [
            deepseek.DeepSeekDenseLayerToLinen,
            deepseek.DeepSeekMoELayerToLinen,
        ]
      case DecoderBlockType.GEMMA:
        return [gemma.GemmaDecoderLayerToLinen]
      case DecoderBlockType.GEMMA2:
        return [gemma2.Gemma2DecoderLayerToLinen]
      case DecoderBlockType.GEMMA3:
        return [gemma3.Gemma3DecoderLayerToLinen]
      case DecoderBlockType.GPT3:
        return [gpt3.Gpt3DecoderLayerToLinen]
      case DecoderBlockType.GPT_OSS:
        return [gpt_oss.GptOssScannableBlockToLinen] if self.config.scan_layers else [gpt_oss.GptOssDecoderLayerToLinen]
      case DecoderBlockType.QWEN3:
        return [qwen3.Qwen3DecoderLayerToLinen]
      case DecoderBlockType.QWEN3_MOE:
        return [qwen3.Qwen3MoeDecoderLayerToLinen]
      case DecoderBlockType.QWEN3_NEXT:
        return [qwen3.Qwen3NextScannableBlockToLinen] if self.config.scan_layers else [qwen3.Qwen3NextDecoderLayerToLinen]
      case DecoderBlockType.SIMPLE:
        return [simple_layer.SimpleDecoderLayerToLinen]
      case DecoderBlockType.SIMPLE_MLP:
        return [simple_layer.SimpleMlpDecoderLayerToLinen]
      case DecoderBlockType.LLAMA4:
        return [llama4.Llama4ScannableBlockToLinen] if self.config.scan_layers else [llama4.Llama4DecoderLayerToLinen]
      case DecoderBlockType.OLMO3:
        return [olmo3.Olmo3ScannableBlockToLinen] if self.config.scan_layers else [olmo3.Olmo3DecoderLayerToLinen]

      case _:
        # Default case to handle any unknown decoder block types.
        raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  def set_remat_policy(self, block_layers, policy):
    """Set remat policy"""
    RemattedBlockLayers = []
    for block_layer in block_layers:
      if self.config.parameter_memory_host_offload:
        # Define parameter movement with mesh-based sharding
        def move_to_device(variables):
          """Move parameters to device with proper sharding."""

          def map_fn(path, value):
            max_logging.log(f"models.py: Moving parameter {path} to device")
            return jax.device_put(value, max_utils.device_space())

          return jax.tree_util.tree_map_with_path(map_fn, variables)

        # Transform layer class before remat
        block_layer = nn.map_variables(block_layer, ["params"], move_to_device, mutable=True)

      # Apply remat policy to layer
      layer = nn.remat(
          block_layer,
          prevent_cse=maxtext_utils.should_prevent_cse_in_remat(self.config),
          policy=policy,
          static_argnums=(4, 5),  # Deterministic and model mode are static arguments.
      )
      RemattedBlockLayers.append(layer)
    return RemattedBlockLayers

  def get_norm_layer(self, num_features: int):
    """get normalization layer (return type inherits from nn.Module)"""
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
      return functools.partial(rms_norm, num_features=num_features, shard_mode=self.config.shard_mode)
    elif self.config.decoder_block == DecoderBlockType.GPT3:
      return functools.partial(gpt3.gpt3_layer_norm, num_features=num_features, reductions_in_fp32=False, use_bias=True)
    elif self.config.decoder_block == DecoderBlockType.QWEN3_NEXT:
      return functools.partial(
          normalizations.Qwen3NextRMSNormLinen, num_features=num_features, shard_mode=self.config.shard_mode
      )
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  def scan_decoder_layers(self, cfg, decoder_layer, length, metadata_axis_name, mesh, in_axes_tuple, **kwargs):
    """scan decoder layers, calls `flax.linen.transforms.scan`"""
    initializing = self.is_mutable_collection("params")
    params_spec = cfg.param_scan_axis if initializing else ScanIn(cfg.param_scan_axis)
    cache_spec = 0
    scan_fn = nn.scan(
        decoder_layer,
        variable_axes={
            "params": params_spec,
            "cache": cache_spec,
            "intermediates": 0,
            "aqt": 0,
            "_overwrite_with_gradient": 0,
        },
        split_rngs={
            "params": True,
            "dropout": cfg.enable_dropout,
        },
        in_axes=in_axes_tuple,
        length=length,
        metadata_params={nn.PARTITION_NAME: metadata_axis_name},
    )
    return scan_fn(
        config=cfg, mesh=mesh, name=metadata_axis_name, quant=self.quant, **kwargs  # pytype: disable=wrong-keyword-args
    )

  def get_pipeline_stage_module(self, decoder_blocks):
    """get pipeline stage module"""

    def get_layer_to_pipeline(blocks, cfg):
      if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
        return blocks[1]  # return the sparse block
      else:
        return blocks[0]

    cfg = self.config
    base_stage = get_layer_to_pipeline(decoder_blocks, cfg)
    if cfg.set_remat_policy_on_layers_per_stage:
      policy = self.get_remat_policy()
      base_stage = self.set_remat_policy([base_stage], policy)[0]
    if cfg.num_layers_per_pipeline_stage == 1:
      stage_module = base_stage(config=cfg, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode)
    elif cfg.scan_layers_per_stage:
      stage_module = self.scan_decoder_layers(
          cfg,
          base_stage,
          cfg.num_layers_per_pipeline_stage,
          "layers_per_stage",
          self.mesh,
          in_axes_tuple=(nn.broadcast,) * 4,
      )
    else:
      stage_module = SequentialBlockDecoderLayers(
          decoder_layer=base_stage,
          num_decoder_layers=cfg.num_layers_per_pipeline_stage,
          config=cfg,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=self.model_mode,
      )
    return stage_module

  @nn.compact
  def _apply_embedding(
      self,
      shared_embedding: nn.Module | nnx.Module,
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
    """Applies token and positional embeddings to the input tokens."""
    cfg = self.config

    y = shared_embedding(decoder_input_tokens.astype("int32"), model_mode=model_mode)

    # Merge the image embeddings with the text embeddings for multimodal models
    if image_embeddings is not None and cfg.use_multimodal:
      if cfg.model_name in [
          "gemma3-4b",
          "gemma3-12b",
          "gemma3-27b",
          "llama4-17b-16e",
          "llama4-17b-128e",
          "qwen3-omni-30b-a3b",
      ]:
        y = mm_utils.merge_mm_embeddings(
            text_embeddings=y,
            multimodal_embeddings=image_embeddings,
            mask=bidirectional_mask,
            token_masks=image_masks,
        )
      # TODO(hengtaoguo): Add support for other multimodal models such as Llama4, refactor if needed
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

    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    if cfg.use_untrainable_positional_embedding:
      y += positional_embedding_as_linen(embedding_dims=cfg.base_emb_dim)(y.shape[1], decoder_positions)

    if cfg.trainable_position_size > 0:
      y += embed_as_linen(
          num_embeddings=cfg.trainable_position_size,
          num_features=cfg.emb_dim,
          dtype=cfg.dtype,
          embedding_init=nn.initializers.normal(stddev=1.0),
          name="position_embedder",
          config=cfg,
          mesh=self.mesh,
      )(decoder_positions.astype("int32"), model_mode=model_mode)
    return y

  @nn.compact
  def apply_output_head(self, shared_embedding: nn.Module | nnx.Module, y, deterministic, model_mode):
    """Applies final normalization and projects hidden states to logits."""

    cfg = self.config
    if cfg.shard_mode == ShardMode.EXPLICIT:
      norm_out_sharding = create_sharding(self.mesh, ("activation_batch", "activation_length_no_exp", "activation_embed"))
    else:
      norm_out_sharding = None

    y = self.get_norm_layer(num_features=y.shape[-1])(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="decoder_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
        parameter_memory_host_offload=cfg.parameter_memory_host_offload,
    )(y, out_sharding=norm_out_sharding)
    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)

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
      logits = linears.dense_general(
          inputs_shape=y.shape,
          out_features_shape=cfg.vocab_size,
          weight_dtype=cfg.weight_dtype,
          dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
          kernel_axes=("embed", "vocab"),
          shard_mode=cfg.shard_mode,
          name="logits_dense",
          matmul_precision=self.config.matmul_precision,
          parameter_memory_host_offload=cfg.parameter_memory_host_offload,
      )(
          y,
          out_sharding=out_sharding,
      )  # We do not quantize the logits matmul.

    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)

    return logits

  # TODO(aireenmei, Hengtaoguo): consolidate all multimodal inputs into a class as input to the encoder
  @nn.compact
  def __call__(
      self,
      shared_embedding: nn.Module | nnx.Module,
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
  ):
    cfg = self.config
    mesh = self.mesh
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

    policy = self.get_remat_policy()
    RemattedBlockLayers = self.set_remat_policy(self.decoder_layer, policy)
    # scan does not support kwargs in layer call, passing broadcast_args as positional arg
    broadcast_args = (
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
    )
    if cfg.using_pipeline_parallelism:
      if cfg.pipeline_fsdp_ag_once:
        logical_partition_spec = self.pipeline_module.get_weight_sharding(
            y, decoder_segment_ids, decoder_positions, deterministic, model_mode
        )
      else:
        logical_partition_spec = None  # This partition spec is only used for the fsdp_ag_once feature.
      if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
        assert len(RemattedBlockLayers) == 2, "Scanned layers must have a length of 2 using deepseek."
        dense_layer = RemattedBlockLayers[0]
        moe_layer = RemattedBlockLayers[1]
        num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
        num_moe_layers_outside_pp = num_moe_layers - self.config.pipeline_parallel_layers
        logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(self.config.logical_axis_rules)
        # We chose not to pipeline the dense layers, only sparse for SPMD.
        with self.mesh, nn.partitioning.axis_rules(logical_axis_rules_pp_as_dp):
          y, _ = self.scan_decoder_layers(
              cfg,
              dense_layer,
              cfg.first_num_dense_layers,
              "dense_layers",
              mesh,
              in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
              model_mode=model_mode,
          )(y, *broadcast_args)
          if num_moe_layers_outside_pp > 0:
            y, _ = self.scan_decoder_layers(
                cfg,
                moe_layer,
                num_moe_layers_outside_pp,
                "moe_layers",
                mesh,
                in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
                model_mode=model_mode,
            )(y, *broadcast_args)
        y = self.pipeline_module(y, *broadcast_args, logical_partition_spec=logical_partition_spec)
      else:  # Not DeepSeek
        y = self.pipeline_module(y, *broadcast_args, logical_partition_spec=logical_partition_spec)
        remaining_layers = self.config.num_decoder_layers - self.config.pipeline_parallel_layers
        if remaining_layers > 0:
          logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(self.config.logical_axis_rules)
          with self.mesh, nn.partitioning.axis_rules(logical_axis_rules_pp_as_dp):
            y, _ = self.scan_decoder_layers(
                cfg,
                RemattedBlockLayers[0],
                remaining_layers,
                "layers_outside_pipeline",
                mesh,
                in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
                model_mode=model_mode,
            )(y, *broadcast_args)
    else:
      if cfg.scan_layers:
        if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
          assert len(RemattedBlockLayers) == 2, "Scanned layers must have a length of 2 using deepseek."
          layer_call_kwargs = {
              "page_state": page_state,
              "previous_chunk": previous_chunk,
              "slot": slot,
          }
          dense_layer = RemattedBlockLayers[0]
          dense_layer.__call__ = functools.partial(dense_layer.__call__, **layer_call_kwargs)
          y, _ = self.scan_decoder_layers(
              cfg,
              dense_layer,
              cfg.first_num_dense_layers,
              "dense_layers",
              mesh,
              in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
              model_mode=model_mode,
          )(y, *broadcast_args)
          moe_layer = RemattedBlockLayers[1]
          moe_layer.__call__ = functools.partial(moe_layer.__call__, **layer_call_kwargs)
          num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
          y, _ = self.scan_decoder_layers(
              cfg,
              moe_layer,
              num_moe_layers,
              "moe_layers",
              mesh,
              in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
              model_mode=model_mode,
          )(y, *broadcast_args)
        elif cfg.decoder_block == DecoderBlockType.GEMMA3:
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
          RemattedBlockLayer = RemattedBlockLayers[0]
          scan_length = int(cfg.num_decoder_layers / cfg.inhomogeneous_layer_cycle_interval)
          layer_kwargs = {}
          if cfg.decoder_block == DecoderBlockType.LLAMA4:
            layer_kwargs = {
                "nope_layer_interval": self.config.nope_layer_interval,
                "interleave_moe_layer_step": self.config.interleave_moe_layer_step,
            }
          y, _ = self.scan_decoder_layers(
              cfg,
              RemattedBlockLayer,
              scan_length,
              "layers",
              mesh,
              in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
              model_mode=model_mode,
              **layer_kwargs,
          )(y, *broadcast_args)
      else:
        if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
          assert len(RemattedBlockLayers) == 2, "Unscanned layers must have a length of 2 using deepseek."
          dense_layer = RemattedBlockLayers[0]
          moe_layer = RemattedBlockLayers[1]

          layers = [dense_layer, moe_layer]
          layer_prefixes = ["dense_layers", "moe_layers"]
          num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
          num_layers_list = [cfg.first_num_dense_layers, num_moe_layers]
          # Iterate over the two layer groups (dense and MoE) and apply layer transformation
          for layer, num_layers, layer_prefix in zip(layers, num_layers_list, layer_prefixes):
            for index in range(num_layers):
              kv_cache = kv_caches[index] if kv_caches is not None else None
              y, kv_cache = layer(
                  config=cfg, mesh=mesh, name=f"{layer_prefix}_{index}", quant=self.quant, model_mode=self.model_mode
              )(
                  y,
                  decoder_segment_ids,
                  decoder_positions,
                  deterministic,
                  model_mode,
                  previous_chunk=previous_chunk,
                  page_state=page_state,
                  slot=slot,
                  kv_cache=kv_cache,
                  attention_metadata=attention_metadata,
              )
              if kv_caches is not None and kv_cache is not None:
                kv_caches[index] = kv_cache
        else:
          for lyr in range(cfg.num_decoder_layers):
            RemattedBlockLayer = RemattedBlockLayers[0]
            layer_kwargs = {}
            layer_call_kwargs = {}
            if cfg.decoder_block == DecoderBlockType.GEMMA3:
              # Gemma3 uses both global and sliding window attention depending on the layer index.
              layer_kwargs = {"attention_type": gemma3.get_attention_type(layer_id=lyr)}
              layer_call_kwargs = {"bidirectional_mask": bidirectional_mask}
            if cfg.decoder_block == DecoderBlockType.LLAMA4:
              layer_kwargs = {
                  "is_nope_layer": llama4.determine_is_nope_layer(lyr, self.config.nope_layer_interval),
                  "is_moe_layer": llama4.determine_is_moe_layer(lyr, self.config.interleave_moe_layer_step),
              }
            if cfg.decoder_block == DecoderBlockType.QWEN3_NEXT:
              layer_kwargs = {"layer_idx": lyr}
            kv_cache = None
            if kv_caches is not None and cfg.decoder_block != DecoderBlockType.QWEN3_NEXT:
              kv_cache = kv_caches[lyr]
            elif kv_caches is not None and cfg.decoder_block == DecoderBlockType.QWEN3_NEXT:
              # For Qwen3Next, kv_caches is a dictionary of lists of caches.
              if (lyr + 1) % cfg.inhomogeneous_layer_cycle_interval == 0:
                kv_cache = (kv_caches["key_cache"][lyr], kv_caches["value_cache"][lyr])

            if cfg.decoder_block == DecoderBlockType.GPT_OSS:
              layer_kwargs = {"attention_type": gpt_oss.get_attention_type(layer_id=lyr)}
            if cfg.decoder_block == DecoderBlockType.OLMO3:
              layer_kwargs = {"attention_type": olmo3.get_attention_type(layer_id=lyr)}
            layer = RemattedBlockLayer(
                config=cfg, mesh=mesh, name=f"layers_{lyr}", quant=self.quant, model_mode=self.model_mode, **layer_kwargs
            )
            y, returned_cache = layer(
                y,
                decoder_segment_ids,
                decoder_positions,
                deterministic,
                model_mode,
                previous_chunk=previous_chunk,
                page_state=page_state,
                slot=slot,
                kv_cache=kv_cache,
                attention_metadata=attention_metadata,
                **layer_call_kwargs,
            )
            if kv_caches is not None and returned_cache is not None:
              if cfg.decoder_block != DecoderBlockType.QWEN3_NEXT:
                kv_caches[lyr] = returned_cache
              elif (lyr + 1) % cfg.inhomogeneous_layer_cycle_interval == 0:
                kv_caches["key_cache"][lyr] = returned_cache[0]
                kv_caches["value_cache"][lyr] = returned_cache[1]

    assert isinstance(y, jax.Array)

    # After the final transformer layer, `y` holds the raw, un-normalized hidden state.
    if cfg.mhc_expansion_rate > 1:
      # (batch, length, mhc_expansion_rate, emb_dim) --> (batch, length, emb_dim)
      hidden_state = mhc_reduce(y)
    else:
      hidden_state = y

    # When initializing with vLLM RPA attention, we need to run the output head to
    # initialize any parameters associated with it.
    if self.is_initializing() and cfg.attention == "vllm_rpa":
      _ = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)

    # When invoking from vLLM with RPA attention, logit computation is deferred to a later stage.
    if cfg.attention == "vllm_rpa":
      logits = None

    # When vocab tiling is enabled in training mode, full logits won't generate to reduce memory
    # Instead, we keep track on the hidden states, which has smaller size compared to full logits
    elif cfg.num_vocab_tiling > 1 and self.model_mode == MODEL_MODE_TRAIN:
      logits = None
      self.sow("intermediates", "hidden_states", hidden_state)

    else:
      logits = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)

    # The API of the Decoder is now a tuple, providing both the main output
    # and the raw hidden state needed for auxiliary tasks.
    return logits, hidden_state, kv_caches

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
    mesh = self.mesh

    # Define the repeating pattern length and calculate how many full blocks to scan
    attention_pattern_length = len(gemma3.GEMMA3_ATTENTION_PATTERN)
    scan_length = cfg.num_decoder_layers // attention_pattern_length

    policy = self.get_remat_policy()
    RemattedGemma3Block = self.set_remat_policy([gemma3.Gemma3ScannableBlockToLinen], policy)[0]

    layer_call_kwargs = {"bidirectional_mask": bidirectional_mask}
    layer_kwargs = {"num_of_layers": attention_pattern_length}

    # Apply the main scan over the full blocks
    if scan_length > 0:
      broadcast_args = (
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
      )
      y, _ = self.scan_decoder_layers(
          cfg,
          RemattedGemma3Block,
          scan_length,
          "layers",
          mesh,
          in_axes_tuple=(nn.broadcast,) * len(broadcast_args),
          model_mode=self.model_mode,
          **layer_kwargs,
      )(y, *broadcast_args, **layer_call_kwargs)

    # Apply any remaining layers that did not fit into a full scanned block
    num_remaining_layers = cfg.num_decoder_layers % attention_pattern_length
    if num_remaining_layers > 0:
      # We name the remainder block with a 'remainder' suffix to avoid parameter name collisions
      rem_layer_kwargs = {"num_of_layers": num_remaining_layers}
      layer = RemattedGemma3Block(
          config=cfg, mesh=mesh, quant=self.quant, model_mode=self.model_mode, name="layers_remainder", **rem_layer_kwargs
      )  # pytype: disable=wrong-keyword-args
      y, _ = layer(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          page_state=page_state,
          slot=slot,
          **layer_call_kwargs,
      )
    return y

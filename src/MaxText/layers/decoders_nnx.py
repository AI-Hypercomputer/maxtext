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

""""NNX version of decoder layers using jax.lax.scan and jax.checkpoint."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any
import functools

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from flax import nnx
from flax import linen as nn

from MaxText.common_types import DecoderBlockType, ShardMode, Config, EP_AS_CONTEXT
from MaxText.common_types import MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from MaxText import max_logging
from MaxText import max_utils
from MaxText.sharding import create_sharding
from MaxText.inference import page_manager
from MaxText.layers import linears
from MaxText.layers import quantizations
from MaxText.layers import pipeline
from MaxText import maxtext_utils
from MaxText import multimodal_utils
from MaxText import sharding
from MaxText.layers.attentions import attention_as_linen, Attention
from MaxText.layers.normalizations import rms_norm, RMSNorm
from MaxText.layers.embeddings import Embed, attend_on_embedding, embed_as_linen, positional_embedding_as_linen
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.layers.linears import MlpBlock
from MaxText.layers import (
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
    qwen3,
    simple_layer,
)
from MaxText.layers import nnx_wrappers

# ------------------------------------------------------------------------------
# NNX Decoder Definitions
# ------------------------------------------------------------------------------


class DecoderLayerNNX(nnx.Module):
  """NNX version of DecoderLayer."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: None | Quant = None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: nnx.Rngs = None,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode

    cfg = self.config

    self.pre_self_attention_norm = RMSNorm(
        num_features=cfg.emb_dim,
        shard_mode=cfg.shard_mode,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
        parameter_memory_host_offload=cfg.parameter_memory_host_offload,
        rngs=rngs,
    )

    # Pure NNX Attention
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
        model_mode=model_mode,
        rngs=rngs,
    )

    # Pure NNX MLP
    self.mlp = MlpBlock(
        config=cfg,
        mesh=self.mesh,
        in_features=cfg.emb_dim,
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        quant=self.quant,
        model_mode=model_mode,
        rngs=rngs,
    )

    self.dropout = nnx.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,), rngs=rngs)

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

    return layer_output, kv_cache


class DecoderNNX(nnx.Module):
  """A stack of decoder layers as an NNX module using jax.lax.scan."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: None | Quant = None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs: nnx.Rngs = None,
  ):
    """Initialize NNX decoder with layers."""
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    self.rngs = rngs

    cfg = self.config

    # Initialize norm layer as an NNX module
    self.decoder_norm = self._create_norm_layer(cfg)

    # Initialize decoder layers based on config
    self.decoder_layer_classes = self.get_decoder_layers()

    # Instantiate layers
    if cfg.scan_layers:
      if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
        dense_layer_cls = self.decoder_layer_classes[0]
        moe_layer_cls = self.decoder_layer_classes[1]
        self.scanned_dense_layers = self._create_scanned_layers(
            dense_layer_cls, cfg.first_num_dense_layers, self.rngs.fork()
        )
        num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
        self.scanned_moe_layers = self._create_scanned_layers(moe_layer_cls, num_moe_layers, self.rngs.fork())
        self.dense_graph_def, _ = nnx.split(self.scanned_dense_layers)
        self.moe_graph_def, _ = nnx.split(self.scanned_moe_layers)
      elif cfg.decoder_block == DecoderBlockType.GEMMA3:
        from MaxText.layers import gemma3

        scannable_block_cls = gemma3.Gemma3ScannableBlock
        attention_pattern_length = len(gemma3.GEMMA3_ATTENTION_PATTERN)
        scan_length = cfg.num_decoder_layers // attention_pattern_length
        layer_kwargs = {"num_of_layers": attention_pattern_length}

        if scan_length > 0:
          self.scanned_layers = self._create_scanned_layers(
              scannable_block_cls, scan_length, self.rngs.fork(), **layer_kwargs
          )
          self.layers_graph_def, _ = nnx.split(self.scanned_layers)

        num_remaining_layers = cfg.num_decoder_layers % attention_pattern_length
        if num_remaining_layers > 0:
          rem_layer_kwargs = {"num_of_layers": num_remaining_layers}
          self.remaining_layer = scannable_block_cls(
              config=cfg,
              mesh=self.mesh,
              quant=self.quant,
              model_mode=self.model_mode,
              rngs=self.rngs.fork(),
              **rem_layer_kwargs,
          )
      else:
        layer_kwargs = {}
        if cfg.decoder_block == DecoderBlockType.LLAMA4:
          layer_kwargs = {
              "nope_layer_interval": self.config.nope_layer_interval,
              "interleave_moe_layer_step": self.config.interleave_moe_layer_step,
          }
        scan_length = int(cfg.num_decoder_layers / cfg.inhomogeneous_layer_cycle_interval)
        self.scanned_layers = self._create_scanned_layers(
            self.decoder_layer_classes[0], scan_length, self.rngs.fork(), **layer_kwargs
        )
        self.layers_graph_def, _ = nnx.split(self.scanned_layers)
    else:
      self.layers = []
      if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
        dense_layer_cls = self.decoder_layer_classes[0]
        moe_layer_cls = self.decoder_layer_classes[1]
        layers_cls = [dense_layer_cls, moe_layer_cls]
        num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
        num_layers_list = [cfg.first_num_dense_layers, num_moe_layers]

        for layer_cls, num_layers in zip(layers_cls, num_layers_list):
          for _ in range(num_layers):
            if issubclass(layer_cls, nnx.Module):
              layer = layer_cls(config=cfg, mesh=self.mesh, quant=self.quant, model_mode=model_mode, rngs=self.rngs.fork())
            else:
              layer_linen = layer_cls(
                  config=cfg, mesh=self.mesh, quant=self.quant, model_mode=model_mode, name=f"layers_{len(self.layers)}"
              )
              layer = nnx_wrappers.ToNNX(layer_linen, rngs=self.rngs.fork())
            self.layers.append(layer)
      else:
        for lyr in range(cfg.num_decoder_layers):
          layer_cls = self.decoder_layer_classes[0]
          layer_kwargs = {}
          if cfg.decoder_block == DecoderBlockType.GEMMA3:
            layer_kwargs = {"attention_type": gemma3.get_attention_type(layer_id=lyr)}
          if cfg.decoder_block == DecoderBlockType.LLAMA4:
            layer_kwargs = {
                "is_nope_layer": llama4.determine_is_nope_layer(lyr, self.config.nope_layer_interval),
                "is_moe_layer": llama4.determine_is_moe_layer(lyr, self.config.interleave_moe_layer_step),
            }
          if cfg.decoder_block == DecoderBlockType.QWEN3_NEXT:
            layer_kwargs = {"layer_idx": lyr}
          if cfg.decoder_block == DecoderBlockType.GPT_OSS:
            layer_kwargs = {"attention_type": gpt_oss.get_attention_type(layer_id=lyr)}

          if issubclass(layer_cls, nnx.Module):
            layer = layer_cls(
                config=cfg, mesh=self.mesh, quant=self.quant, model_mode=model_mode, rngs=self.rngs.fork(), **layer_kwargs
            )
          else:
            layer_linen = layer_cls(
                config=cfg, mesh=self.mesh, quant=self.quant, model_mode=model_mode, name=f"layers_{lyr}", **layer_kwargs
            )
            layer = nnx_wrappers.ToNNX(layer_linen, rngs=self.rngs.fork())
          self.layers.append(layer)

    # Setup pipeline if needed
    if self.config.using_pipeline_parallelism:
      pipeline_stage_module = self.get_pipeline_stage_module(self.decoder_layer_classes)
      remat_policy = self.get_remat_policy()
      self.pipeline_module = pipeline.Pipeline(
          config=self.config, mesh=self.mesh, layers=pipeline_stage_module, remat_policy=remat_policy
      )

  def _create_norm_layer(self, cfg: Config):
    """Create normalization layer as NNX module."""
    if cfg.decoder_block in (
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
        DecoderBlockType.QWEN3_NEXT,
        DecoderBlockType.GPT_OSS,
        DecoderBlockType.SIMPLE,
        DecoderBlockType.SIMPLE_MLP,
        DecoderBlockType.LLAMA4,
    ):
      # Create RMS norm layer as NNX
      return RMSNorm(
          num_features=cfg.emb_dim,
          shard_mode=cfg.shard_mode,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          epsilon=cfg.normalization_layer_epsilon,
          kernel_axes=("norm",),
          parameter_memory_host_offload=cfg.parameter_memory_host_offload,
          rngs=self.rngs,
      )
    elif cfg.decoder_block == DecoderBlockType.GPT3:
      return gpt3.Gpt3LayerNorm(
          num_features=cfg.emb_dim,
          reductions_in_fp32=False,
          use_bias=True,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          epsilon=cfg.normalization_layer_epsilon,
          kernel_axes=("norm",),
          parameter_memory_host_offload=cfg.parameter_memory_host_offload,
          rngs=self.rngs,
      )
    else:
      raise ValueError(f"Incorrect decoder_block name {cfg.decoder_block.value=}")

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
    """Get remat policy for jax.checkpoint."""
    policy = None
    cfg = self.config
    if cfg.remat_policy != "none":
      if cfg.remat_policy in ("minimal_with_context", "minimal_flash"):
        if cfg.remat_policy == "minimal_flash":
          max_logging.log("WARNING: 'minimal_flash' will be deprecated soon, please use 'minimal_with_context' instead.")
        policy = self.minimal_policy(with_context=True)
      elif cfg.remat_policy == "minimal":
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
        policy = None
    return policy

  def get_decoder_layers(self):
    """Retrieves decoder layer classes based on config."""
    match self.config.decoder_block:
      case DecoderBlockType.DEFAULT:
        return [DecoderLayerNNX]
      case DecoderBlockType.LLAMA2:
        return [llama2.LlamaDecoderLayer]
      case DecoderBlockType.MISTRAL:
        return [mistral.MistralDecoderLayer]
      case DecoderBlockType.MIXTRAL:
        return [mixtral.MixtralDecoderLayer]
      case DecoderBlockType.DEEPSEEK:
        if self.config.use_batch_split_schedule:
          return [deepseek_batchsplit.DeepSeekDenseLayer, deepseek_batchsplit.DeepSeekMoELayer]
        else:
          return [deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer]
      case DecoderBlockType.GEMMA:
        return [gemma.GemmaDecoderLayer]
      case DecoderBlockType.GEMMA2:
        return [gemma2.Gemma2DecoderLayer]
      case DecoderBlockType.GEMMA3:
        return [gemma3.Gemma3DecoderLayer]
      case DecoderBlockType.GPT3:
        return [gpt3.Gpt3DecoderLayer]
      case DecoderBlockType.GPT_OSS:
        return [gpt_oss.GptOssScannableBlock] if self.config.scan_layers else [gpt_oss.GptOssDecoderLayer]
      case DecoderBlockType.QWEN3:
        return [qwen3.Qwen3DecoderLayer]
      case DecoderBlockType.QWEN3_MOE:
        return [qwen3.Qwen3MoeDecoderLayer]
      case DecoderBlockType.QWEN3_NEXT:
        return [qwen3.Qwen3NextScannableBlock] if self.config.scan_layers else [qwen3.Qwen3NextDecoderLayer]
      case DecoderBlockType.SIMPLE:
        return [simple_layer.SimpleDecoderLayer]
      case DecoderBlockType.SIMPLE_MLP:
        return [simple_layer.SimpleMlpDecoderLayer]
      case DecoderBlockType.LLAMA4:
        return [llama4.Llama4ScannableBlock] if self.config.scan_layers else [llama4.Llama4DecoderLayer]
      case _:
        raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  def get_pipeline_stage_module(self, decoder_blocks):
    """Get pipeline stage module."""

    def get_layer_to_pipeline(blocks, cfg):
      if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
        return blocks[1]
      else:
        return blocks[0]

    cfg = self.config
    base_stage = get_layer_to_pipeline(decoder_blocks, cfg)
    if cfg.set_remat_policy_on_layers_per_stage:
      policy = self.get_remat_policy()
      # Apply remat policy via jax.checkpoint in the layer

    if issubclass(base_stage, nnx.Module):
      stage_module = nnx_wrappers.ToLinen(
          nnx_class=base_stage,
          kwargs={
              "config": cfg,
              "mesh": self.mesh,
              "quant": self.quant,
              "model_mode": self.model_mode,
          },
      )
    else:
      if cfg.num_layers_per_pipeline_stage == 1:
        stage_module = base_stage(config=cfg, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode)
      else:
        # For pipeline stages, wrap the layer
        stage_module = base_stage(config=cfg, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode)
    return stage_module

  def _create_scanned_layers(self, decoder_layer_class, length: int, rngs: nnx.Rngs, **layer_kwargs):
    """Create scanned layers using NNX vmap. Checkpointing is applied at scan level."""

    def create_layer_fn(rng):
      if issubclass(decoder_layer_class, nnx.Module):
        layer = decoder_layer_class(
            config=self.config, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, rngs=rng, **layer_kwargs
        )
      else:
        layer_linen = decoder_layer_class(
            config=self.config, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, **layer_kwargs
        )
        layer = nnx_wrappers.ToNNX(layer_linen, rngs=rng)
      return layer

    nnx.split_rngs(rngs, splits=length)
    layers_vmapped = nnx.vmap(create_layer_fn, in_axes=0, out_axes=0)(rngs)

    return layers_vmapped

  def _remat_layer(self, layer, policy, prevent_cse):
    """Apply jax.checkpoint to layer's __call__ method."""
    original_call = layer.__call__

    def checkpointed_call(*args, **kwargs):
      return jax.checkpoint(original_call, policy=policy, prevent_cse=prevent_cse)(*args, **kwargs)

    object.__setattr__(layer, "__call__", checkpointed_call)
    return layer

  def _apply_layers_sequentially(self, layers, x_in, *args, length: int, **kwargs):
    """Apply layers using jax.lax.scan, mimicking Linen's nn.scan approach.

    Linen's nn.scan works directly with parameter PyTrees and uses vmap to apply
    the same computation with different parameters. We do the same here:

    1. Extract layer's __call__ as a pure function
    2. Use scan to iterate over parameter PyTrees (not module objects)
    3. Avoid nnx.merge inside the loop - work with raw PyTrees like Linen does

    This mimics Linen's approach of scanning over variable collections.
    """

    # Split once - separate static graph structure from parameters
    graphdef, params, other_state = nnx.split(layers, nnx.Param, ...)

    # Get checkpoint policy
    policy = self.get_remat_policy()
    prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)

    # Create a pure function that applies the layer computation
    # This captures graphdef and other_state in closure (like Linen captures the module class)
    def pure_layer_fn(carry, layer_params):
      """Pure function that operates on parameter PyTrees, mimicking Linen's approach.

      Linen's nn.scan does essentially this:
      - Takes parameters as a PyTree (not a module object)
      - Applies the layer computation with those parameters
      - Returns new carry

      We do the same by merging params into the layer structure only once per iteration.
      The key is that merge happens inside scan (unavoidable with NNX), but we minimize
      what we pass through scan by only scanning over params.
      """
      # Reconstruct layer - this is the overhead compared to Linen
      # Linen doesn't need this because it works directly with FrozenDict variables
      layer = nnx.merge(graphdef, layer_params, other_state)

      # Forward pass - pure computation on activations
      layer_out = layer(carry, *args, **kwargs)

      # Extract carry for next iteration
      new_carry = layer_out[0] if isinstance(layer_out, tuple) else layer_out

      return new_carry, None

    # Apply checkpoint to the entire scan body (like Linen's nn.scan + nn.remat)
    if policy is not None:
      pure_layer_fn = jax.checkpoint(pure_layer_fn, policy=policy, prevent_cse=prevent_cse)

    # Scan over parameter PyTrees (like Linen scans over variable_axes)
    # This creates a single computation graph that's reused with different parameters
    final_carry, _ = jax.lax.scan(pure_layer_fn, x_in, params)

    return final_carry, None

  def _apply_embedding(
      self,
      shared_embedding,
      decoder_input_tokens,
      decoder_positions,
      deterministic,
      model_mode,
      image_embeddings=None,
      bidirectional_mask=None,
      image_masks=None,
  ):
    """Applies token and positional embeddings to input tokens."""
    cfg = self.config

    if isinstance(shared_embedding, nnx.Module):
      y = shared_embedding(decoder_input_tokens.astype("int32"), model_mode=model_mode)
    else:
      y = shared_embedding(decoder_input_tokens.astype("int32"), model_mode=model_mode)

    # Merge multimodal embeddings
    if image_embeddings is not None and cfg.use_multimodal:
      if cfg.model_name in [
          "gemma3-4b",
          "gemma3-12b",
          "gemma3-27b",
          "llama4-17b-16e",
          "llama4-17b-128e",
          "qwen3-omni-30b-a3b",
      ]:
        y = multimodal_utils.merge_mm_embeddings(
            text_embeddings=y,
            vision_embeddings=image_embeddings,
            mask=bidirectional_mask,
            image_masks=image_masks,
        )
      else:
        raise ValueError(f"Unsupported model_name for multimodal: {cfg.model_name}")

    # Apply dropout
    y = nnx.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    # Apply positional embeddings
    if cfg.use_untrainable_positional_embedding:
      positional_emb = positional_embedding_as_linen(embedding_dims=cfg.base_emb_dim)
      y = positional_emb(y, decoder_positions)

    if cfg.trainable_position_size > 0:
      if not hasattr(self, "position_embedder"):
        self.position_embedder = Embed(
            mesh=self.mesh,
            num_embeddings=cfg.trainable_position_size,
            num_features=cfg.emb_dim,
            dtype=cfg.dtype,
            embedding_init=nn.initializers.normal(stddev=1.0),
            config=cfg,
            rngs=self.rngs,
        )
      y += self.position_embedder(decoder_positions.astype("int32"), model_mode=model_mode)

    return y

  def apply_output_head(self, shared_embedding, y, deterministic, model_mode):
    """Applies final normalization and projects hidden states to logits."""
    cfg = self.config

    # Apply normalization
    if cfg.shard_mode == ShardMode.EXPLICIT:
      norm_out_sharding = create_sharding(self.mesh, ("activation_batch", "activation_length_no_exp", "activation_embed"))
    else:
      norm_out_sharding = None

    # Create norm layer and apply
    y = self.decoder_norm(y, out_sharding=norm_out_sharding)

    # Dropout
    y = nnx.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)(y, deterministic=deterministic)

    # Compute sharding for logits
    if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
      out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
    else:
      out_sharding = create_sharding(
          self.mesh, ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_vocab")
      )

    # Compute logits
    if cfg.logits_via_embedding:
      if isinstance(shared_embedding, nnx.Module):
        embedding_table = shared_embedding.embedding.value
      else:
        embedding_table = shared_embedding.variables["params"]["embedding"]
      if isinstance(embedding_table, nn.spmd.LogicallyPartitioned):
        embedding_table = embedding_table.unbox()
      attend_dtype = jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype
      logits = attend_on_embedding(y, embedding_table, attend_dtype, self.config, out_sharding)

      if self.config.normalize_embedding_logits:
        logits = logits / jnp.sqrt(y.shape[-1])
      if cfg.final_logits_soft_cap:
        logits = logits / cfg.final_logits_soft_cap
        logits = jnp.tanh(logits) * cfg.final_logits_soft_cap
    else:
      logits = linears.dense_general(
          inputs_shape=y.shape,
          out_features_shape=cfg.vocab_size,
          weight_dtype=cfg.weight_dtype,
          dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,
          kernel_axes=("embed", "vocab"),
          shard_mode=cfg.shard_mode,
          name="logits_dense",
          matmul_precision=self.config.matmul_precision,
          parameter_memory_host_offload=cfg.parameter_memory_host_offload,
      )(y, out_sharding=out_sharding)

    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)

    return logits

  def __call__(
      self,
      shared_embedding,
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
  ):
    """Main forward pass of the decoder."""
    cfg = self.config
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    # Apply embeddings
    y = self._apply_embedding(
        shared_embedding,
        decoder_input_tokens,
        decoder_positions,
        deterministic,
        model_mode,
        image_embeddings,
        bidirectional_mask,
        image_masks,
    )

    # Apply decoder layers using scan if enabled
    if cfg.scan_layers:
      if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
        # DeepSeek specific scan logic

        # Apply dense layers sequentially
        y, _ = self._apply_layers_sequentially(
            self.scanned_dense_layers,
            y,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            length=cfg.first_num_dense_layers,
            previous_chunk=previous_chunk,
            slot=slot,
            page_state=page_state,
            kv_cache=None,
            attention_metadata=attention_metadata,
        )

        # Apply MoE layers sequentially
        num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
        y, _ = self._apply_layers_sequentially(
            self.scanned_moe_layers,
            y,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            length=num_moe_layers,
            previous_chunk=previous_chunk,
            slot=slot,
            page_state=page_manager.PageState,
            kv_cache=None,
            attention_metadata=attention_metadata,
        )
      elif cfg.decoder_block == DecoderBlockType.GEMMA3:
        # Gemma3 specific scan logic
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
        # Standard scan logic
        layer_kwargs = {}
        if cfg.decoder_block == DecoderBlockType.LLAMA4:
          layer_kwargs = {
              "nope_layer_interval": self.config.nope_layer_interval,
              "interleave_moe_layer_step": self.config.interleave_moe_layer_step,
          }

        scan_length = int(cfg.num_decoder_layers / cfg.inhomogeneous_layer_cycle_interval)
        y, _ = self._apply_layers_sequentially(
            self.scanned_layers,
            y,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            length=scan_length,
            previous_chunk=previous_chunk,
            slot=slot,
            page_state=page_state,
            kv_cache=None,
            attention_metadata=attention_metadata,
            **layer_kwargs,
        )
    else:
      # Apply layers sequentially
      for lyr, layer_nnx in enumerate(self.layers):
        layer_call_kwargs = {}

        if cfg.decoder_block == DecoderBlockType.GEMMA3:
          layer_call_kwargs = {"bidirectional_mask": bidirectional_mask}

        kv_cache = kv_caches[lyr] if kv_caches is not None else None

        # Apply remat policy
        policy = self.get_remat_policy()
        prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)

        def layer_call(
            y,
            kv_cache,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            previous_chunk,
            slot,
            page_state,
            attention_metadata,
            **extra_kwargs,
        ):
          return layer_nnx(
              y,
              decoder_segment_ids,
              decoder_positions,
              deterministic,
              model_mode,
              previous_chunk=previous_chunk,
              slot=slot,
              page_state=page_state,
              kv_cache=kv_cache,
              attention_metadata=attention_metadata,
              **extra_kwargs,
          )

        rematted_layer_call = jax.checkpoint(layer_call, policy=policy, prevent_cse=prevent_cse)

        layer_out = rematted_layer_call(
            y,
            kv_cache,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            previous_chunk,
            slot,
            page_state,
            attention_metadata,
            **layer_call_kwargs,
        )

        if isinstance(layer_out, tuple):
          y, kv_cache = layer_out
        else:
          y = layer_out

        if kv_caches is not None and kv_cache is not None:
          kv_caches[lyr] = kv_cache

    hidden_state = y

    # Generate logits
    if cfg.num_vocab_tiling > 1 and self.model_mode == MODEL_MODE_TRAIN:
      logits = None
    else:
      logits = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)

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

    # Define the repeating pattern length and calculate how many full blocks to scan
    attention_pattern_length = len(gemma3.GEMMA3_ATTENTION_PATTERN)
    scan_length = cfg.num_decoder_layers // attention_pattern_length

    layer_call_kwargs = {"bidirectional_mask": bidirectional_mask}

    # Apply the main blocks sequentially
    if scan_length > 0:
      y, _ = self._apply_layers_sequentially(
          self.scanned_layers,
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          length=scan_length,
          previous_chunk=previous_chunk,
          slot=slot,
          page_state=page_state,
          kv_cache=None,
          attention_metadata=None,  # Gemma3 might not use this in scan?
          **layer_call_kwargs,
      )

    # Apply any remaining layers that did not fit into a full scanned block
    num_remaining_layers = cfg.num_decoder_layers % attention_pattern_length
    if num_remaining_layers > 0:
      layer = self.remaining_layer

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

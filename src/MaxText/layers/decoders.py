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

""""Module for decoder layers."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any, Callable, Union, Type
import functools

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from flax import linen as nn
from flax import nnx
from flax.linen.partitioning import ScanIn
from flax.core.spmd import logical_axis_rules as nn_logical_axis_rules
from flax import nnx
import numpy as np

from MaxText.common_types import Array, DecoderBlockType, Config, MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from MaxText import max_logging
from MaxText import max_utils
from MaxText.inference import page_manager
from MaxText.layers import initializers
from MaxText.layers import linears
from MaxText.layers import quantizations
from MaxText.layers import pipeline
from MaxText import maxtext_utils
from MaxText import multimodal_utils
from MaxText.layers.attentions import Attention
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.embeddings import attend_on_embedding, Embed, PositionalEmbedding,_MAX_WAVELENGTH
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.layers import (
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
    qwen3,
    simple_layer,
)

# ------------------------------------------------------------------------------
# The network: Decoder Definitions
# ------------------------------------------------------------------------------


class DecoderLayer(nnx.Module):
  """
  Transformer decoder layer that attends to the encoder.
  This is the core, reusable building block for both the main model's
  decoder stack and the auxiliary MTP layers.
  """

  def __init__(self, config: Config, mesh: Mesh, quant: Quant | None = None, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    self.rngs = rngs if rngs is not None else nnx.Rngs(0)
    inputs_shape = (
      int(self.config.per_device_batch_size),
      int(self.config.max_target_length),
      int(self.config.emb_dim),
    )

    self.mlp = linears.MlpBlock(
      in_features=inputs_shape[-1],
      intermediate_dim=self.config.mlp_dim,
      activations=self.config.mlp_activations,
      intermediate_dropout_rate=self.config.dropout_rate,
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      config=self.config,
      quant=quant,
      model_mode=model_mode,
      rngs=self.rngs
    )

    self.drop_out = nnx.Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,),rngs=self.rngs)

    self.pre_self_attention_norm = RMSNorm(
      num_features=inputs_shape[-1],
      dtype=self.config.dtype,
      weight_dtype=self.config.weight_dtype,
      kernel_axes=("norm", ),
      epsilon=self.config.normalization_layer_epsilon,
      rngs=self.rngs
    )

    self.self_attention = Attention(
        config=self.config,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        attention_kernel=self.config.attention,
        inputs_q_shape=inputs_shape,
        inputs_kv_shape=inputs_shape,
        mesh=mesh,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        dropout_rate=self.config.dropout_rate,
        name="self_attention",
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(self.config),
        prefill_cache_axis_order=tuple(map(int, self.config.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, self.config.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, self.config.compute_axis_order.split(","))),
        reshape_q=self.config.reshape_q,
        model_mode=model_mode,
        rngs=self.rngs
    )

  def __call__(
      self,
      inputs : Array,
      decoder_segment_ids : Array,
      decoder_positions : Array,
      deterministic : bool,
      model_mode : str ,
      previous_chunk: Array | None = None,
      slot: int | None = None,
      page_state: page_manager.PageState | None = None,
  ):
    cfg = self.config
    logical_axis_names = (
      ("activation_batch", "prefill_activation_length", "activation_embed")
      if model_mode == MODEL_MODE_PREFILL
      else ("activation_batch", "activation_length", "activation_embed")
    )

    inputs = nn.with_logical_constraint(inputs, logical_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = self.pre_self_attention_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, logical_axis_names)

    attention_lnx = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )
    attention_lnx = nn.with_logical_constraint(attention_lnx, logical_axis_names)

    # MLP block.
    mlp_lnx = self.mlp(lnx, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, logical_axis_names)

    next_layer_addition = mlp_lnx + attention_lnx

    next_layer_addition_dropped_out = self.drop_out(
        next_layer_addition, deterministic=deterministic
    )

    layer_output = next_layer_addition_dropped_out + inputs
    layer_output = nn.with_logical_constraint(
        layer_output,
        logical_axis_names,
    )

    if cfg.record_internal_nn_metrics:
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    return layer_output, None if cfg.scan_layers else layer_output


class SequentialBlockDecoderLayers(nnx.Module):
  """Sequential unscanned series of decoder layers."""

  def __init__(self,decoder_layer:Any,num_decoder_layers:int, config: Config, mesh: Mesh, quant: Quant|None=None, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    self.rngs = rngs if rngs is not None else nnx.Rngs(0)
    self.num_decoder_layers = num_decoder_layers
    self.decoder_layer = decoder_layer

  def __call__(
      self,
      inputs: Array,
      decoder_segment_ids: Array,
      decoder_positions : Array,
      deterministic: bool,
      model_mode : str,
      slot: int | None = None,
      page_state: page_manager.PageState | None = None,
  ) -> Union[Array, tuple[Array, None]]:
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
    return inputs

class Decoder(nnx.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture.
  """

  BROADCAST_ARGS_LENGTH : int = 4

  def __init__(
      self,
      config: Config,
      shared_embedding: nn.Module,
      mesh: Mesh,
      quant: Quant | None=None,
      model_mode: str = MODEL_MODE_TRAIN,
      rngs : nnx.Rngs | None = None,
  ):
    self.config = config
    self.shared_embedding = shared_embedding
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    self.rngs = rngs if rngs is not None else nnx.Rngs(0)

    layer_classes = self.get_decoder_layers()
    policy = self.get_remat_policy()
    self.rematted_layer_classes = self.set_remat_policy(layer_classes, policy)
    
    self.embedding_dropout = nnx.Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,))
    self.output_head_dropout = nnx.Dropout(rate=self.config.dropout_rate, broadcast_dims=(-2,))
    
    self.decoder_norm = self.get_norm_layer(num_features=self.config.emb_dim)(
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        rngs=self.rngs,
    )

    self._pipeline_module: pipeline.Pipeline | None = None
    if config.using_pipeline_parallelism:
      pipeline_stage_module = self.get_pipeline_stage_module(layer_classes)
      remat_policy = self.get_remat_policy()
      self._pipeline_module = pipeline.Pipeline(
          config=config,
          mesh=self.mesh,
          layers=pipeline_stage_module,
          remat_policy=remat_policy,
      )
      if config.decoder_block == DecoderBlockType.DEEPSEEK:
        self._build_exec_deepseek_pipeline()
      else:
        self._build_exec_standard_pipeline()
    else:
      if config.scan_layers:
        self._build_exec_scanned()
      else:
        self._build_exec_unscanned()

    sequence_length = int(config.max_target_length)
    if model_mode == MODEL_MODE_PREFILL:
      sequence_length = int(config.max_prefill_predict_length)
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
      sequence_length = 1

    inputs_shape = (
      int(config.micro_batch_size_to_train_on),
      sequence_length,
      int(self.config.emb_dim),
    )

    self.logits_dense = linears.dense_general(
          inputs_shape=inputs_shape,
          out_features_shape=self.config.vocab_size,
          weight_dtype=self.config.weight_dtype,
          dtype=jnp.float32 if self.config.logits_dot_in_fp32 else self.config.dtype,
          kernel_axes=("embed", "vocab"),
          name="logits_dense",
          matmul_precision=self.config.matmul_precision,
          parameter_memory_host_offload=self.config.parameter_memory_host_offload,
      )

    # Untrainable (static) positional embedding
    self._static_pos_embedding = None
    if self.config.use_untrainable_positional_embedding:
        self._static_pos_embedding = PositionalEmbedding(
            embedding_dims=self.config.base_emb_dim,
            max_wavelength=_MAX_WAVELENGTH,
        )

    # Trainable position embedding
    self.position_embedder = None
    if self.config.trainable_position_size > 0:
        self.position_embedder = Embed(
            num_embeddings=self.config.trainable_position_size,
            num_features=self.config.emb_dim,
            dtype=self.config.dtype,
            embedding_init=nn.initializers.normal(stddev=1.0),
            config=self.config,
            rngs=self.rngs,
        )

  def get_remat_policy(self)-> Callable[..., bool]|None:
    cfg = self.config
    policy_name = cfg.remat_policy

    if policy_name == "none" or policy_name == "full":
      return None
    
    static_policies : dict[str,Any] = {
        "minimal": jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
        "save_dot_with_context_except_mlp": jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj", "context", "out_proj"
        ),
        "save_dot_except_mlpwi": jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj", "mlpwo"
        ),
        "save_dot_except_mlp": jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj", "out_proj"
        ),
        "save_qkv_proj": jax.checkpoint_policies.save_only_these_names(
            "query_proj", "value_proj", "key_proj", "qkv_proj"
        ),
        "save_out_proj": jax.checkpoint_policies.save_only_these_names("out_proj"),
        "minimal_flash": jax.checkpoint_policies.save_from_both_policies(
            jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
            jax.checkpoint_policies.save_only_these_names("context")
        ),
    }

    dynamic_policies : dict[str,Callable] = {
        "qkv_proj_offloaded": lambda: jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=["query_proj", "value_proj", "key_proj"],
            offload_src="device",
            offload_dst="pinned_host"
        ),
        "minimal_offloaded": lambda: jax.checkpoint_policies.offload_dot_with_no_batch_dims(
            offload_src="device",
            offload_dst="pinned_host"
        ),
        "custom": lambda: jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=cfg.tensors_on_device,
            names_which_can_be_offloaded=cfg.tensors_to_offload,
            offload_src="device",
            offload_dst="pinned_host"
        ),
    }

    if policy_name in static_policies:
      return static_policies[policy_name]
    elif policy_name in dynamic_policies:
      return dynamic_policies[policy_name]()
    raise ValueError(f"Remat policy needs to be on list of remat policies, get : '{policy_name}'")

  def get_decoder_layers(self)->list[Type[nnx.Module]]:
    # TODO(ranran): update to Mistral with sliding window attention
    decoder_layer_map = {
        DecoderBlockType.DEFAULT: [DecoderLayer],
        DecoderBlockType.LLAMA2: [llama2.LlamaDecoderLayer],
        DecoderBlockType.MISTRAL: [mistral.MistralDecoderLayer],
        DecoderBlockType.MIXTRAL: [mixtral.MixtralDecoderLayer],
        DecoderBlockType.DEEPSEEK: [deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer],
        DecoderBlockType.GEMMA: [gemma.GemmaDecoderLayer],
        DecoderBlockType.GEMMA2: [gemma2.Gemma2DecoderLayer],
        DecoderBlockType.GEMMA3: [gemma3.Gemma3DecoderLayer],
        DecoderBlockType.GPT3: [gpt3.Gpt3DecoderLayer],
        DecoderBlockType.QWEN3: [qwen3.Qwen3DecoderLayer],
        DecoderBlockType.SIMPLE: [simple_layer.SimpleDecoderLayer],
        DecoderBlockType.SIMPLE_MLP: [simple_layer.SimpleMlpDecoderLayer],
        DecoderBlockType.GPT_OSS:[gpt_oss.GptOssScannableBlock] if self.config.scan_layers else [gpt_oss.GptOssDecoderLayer],
        DecoderBlockType.LLAMA4: (
           [llama4.Llama4ScannableBlock] 
           if self.config.scan_layers 
           else [llama4.Llama4DecoderLayer]
        ),
    }

    decoder_type = self.config.decoder_block
    if decoder_type in decoder_layer_map:
      return decoder_layer_map[decoder_type]
    raise ValueError(f"Incorrect decoder_block name: {decoder_type.value}")

  def set_remat_policy(self, block_layers, policy : Callable[..., bool]|None = None)->list[Type[nnx.Module]]:
    RemattedBlockLayers = []
    for block_layer in block_layers:
      if self.config.parameter_memory_host_offload:
        def move_to_device(variables):
          """Move parameters to device with proper sharding."""
          def map_fn(path, value):
            max_logging.log(f"models.py: Moving parameter {path} to device")
            return jax.device_put(value, max_utils.device_space())

          return jax.tree_util.tree_map_with_path(map_fn, variables)
        
        # Transform layer class before remat
        graphdef, params = nnx.split(block_layer, nnx.Param)
        params = move_to_device(params)
        block_layer = nnx.merge(graphdef, params)

      # Apply remat policy to layer
      layer = nnx.remat(
          block_layer,
          prevent_cse=not self.config.scan_layers,
          policy=policy,
          static_argnums=(4, 5),  # Deterministic and model mode are static arguments.
      )
      RemattedBlockLayers.append(layer)
    return RemattedBlockLayers

  def get_norm_layer(self, num_features: int)-> Callable[...,Any]:
    if self.config.decoder_block in {
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
    }:
      return functools.partial(RMSNorm, num_features=num_features,rngs=self.rngs)
    elif self.config.decoder_block == DecoderBlockType.GPT3:
      return functools.partial(
          gpt3.Gpt3LayerNorm,
          num_features=num_features,
          reductions_in_fp32=False,
          use_bias=True,
          rngs=self.rngs,
      )
    raise ValueError(f"Incorrect config decoder_block name : {self.config.decoder_block.value}")

  def _make_scan_runner(self, layer_ctor_or_fn, length:int, name:str, **layer_kwargs):
    """Return a callable: run(y, *broadcast_args) -> y that executes a prebuilt scan."""
    cfg = self.config
    mesh = self.mesh

    def build_scan_fn():
      # we bake in_axes based on the 4-tuple broadcast args we use everywhere:
      in_axes_tuple = (nn.broadcast,) * self.BROADCAST_ARGS_LENGTH
      return self.scan_decoder_layers(
          cfg, layer_ctor_or_fn, length, name, mesh, in_axes_tuple,
          model_mode=self.model_mode, **layer_kwargs
      )

    scan_fn = build_scan_fn()

    def run(y, *broadcast_args, **kwargs):
      y, _ = scan_fn(y, *broadcast_args, **kwargs)
      return y

    return run
  
  def _calculate_partition_spec(self, y, decoder_segment_ids, decoder_positions, deterministic, model_mode):
    return (
      None if not self.config.pipeline_fsdp_ag_once 
      else 
      self.pipeline_module.get_weight_sharding(
        y, decoder_segment_ids, decoder_positions, deterministic, model_mode
      )
    )

  def _build_exec_deepseek_pipeline(self):
    cfg = self.config
    mesh = self.mesh
    if len(self.rematted_layer_classes) != 2:
      raise ValueError(
          f"Scanned layers must have a length of 2 when using DeepSeek, "
          f"but got {len(self.rematted_layer_classes)}."
      )
    dense_cls, moe_cls = self.rematted_layer_classes

    num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
    num_moe_layers_outside_pp = num_moe_layers - cfg.pipeline_parallel_layers

    # Prebuild scan runners (outside pipeline)
    self.dense_layers = self._make_scan_runner(
        dense_cls, cfg.first_num_dense_layers, "dense_layers"
    )
    self.moe_layers = None
    if num_moe_layers_outside_pp > 0:
      self.moe_layers = self._make_scan_runner(
          moe_cls, num_moe_layers_outside_pp, "moe_layers"
      )

    logical_axis_rules_pp_as_dp = maxtext_utils.logical_axis_rules_pp_act_as_dp(cfg.logical_axis_rules)

    def exec_run(y, decoder_segment_ids, decoder_positions, deterministic, model_mode,
                 previous_chunk=None, slot=None, page_state=None, bidirectional_mask=None):
      # Execute scans outside pipeline under adjusted logical axis rules
      with mesh, nn_logical_axis_rules(logical_axis_rules_pp_as_dp):
        y = self.dense_layers(y, decoder_segment_ids, decoder_positions, deterministic, model_mode)
        if self.moe_layers is not None:
          y = self.moe_layers(y, decoder_segment_ids, decoder_positions, deterministic, model_mode)

      # Optionally compute weight sharding once (shape-dependent)
      partition_spec = self._calculate_partition_spec(
        y, decoder_segment_ids, decoder_positions, deterministic, model_mode)

      # Pipeline proper (stage module was built in __init__)
      y = self.pipeline_module(y, decoder_segment_ids, decoder_positions, deterministic, model_mode,
                               partition_spec=partition_spec)
      return y

    self._exec = exec_run

  def _build_exec_standard_pipeline(self):
    """Pipeline (non-DeepSeek). After pipeline, possibly run remaining scanned layers."""
    cfg = self.config
    mesh = self.mesh
    # Remaining layers after pipeline, if any, are scanned with the single base layer
    remaining_layers = cfg.num_decoder_layers - cfg.pipeline_parallel_layers
    self.layers_outside_pipeline = None
    if remaining_layers > 0:
      self.layers_outside_pipeline = self._make_scan_runner(
          self.rematted_layer_classes[0], remaining_layers, "layers_outside_pipeline"
      )
    logical_axis_rules_pp_as_dp = maxtext_utils.logical_axis_rules_pp_act_as_dp(cfg.logical_axis_rules)

    def exec_run(y, decoder_segment_ids, decoder_positions, deterministic, model_mode,
                 previous_chunk=None, slot=None, page_state=None, bidirectional_mask=None):
      
      # Optionally compute weight sharding once (shape-dependent)
      partition_spec = self._calculate_partition_spec(
        y, decoder_segment_ids, decoder_positions, deterministic, model_mode)

      y = self.pipeline_module(y, decoder_segment_ids, decoder_positions, deterministic, model_mode,
                               partition_spec=partition_spec)

      if self.layers_outside_pipeline is not None:
        with mesh, nn_logical_axis_rules(logical_axis_rules_pp_as_dp):
          y = self.layers_outside_pipeline(y, decoder_segment_ids, decoder_positions, deterministic, model_mode)
      return y

    self._exec = exec_run

  def _build_exec_scanned(self):
    """No pipeline, scanned execution. Handle DeepSeek / Gemma3 / Llama4 / others."""
    cfg = self.config
    mesh = self.mesh

    if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
      if len(self.rematted_layer_classes) != 2:
        raise ValueError(
          f"Scanned layers must have a length of 2 when using DeepSeek, "
          f"but got {len(self.rematted_layer_classes)}."
        )
      dense_cls, moe_cls = self.rematted_layer_classes
      self.dense_layers = self._make_scan_runner(dense_cls, cfg.first_num_dense_layers, "dense_layers")
      num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
      self.moe_layers = self._make_scan_runner(moe_cls, num_moe_layers, "moe_layers")

      def exec_run(y, decoder_segment_ids, decoder_positions, deterministic, model_mode,
                   previous_chunk=None, slot=None, page_state=None, bidirectional_mask=None):
        y = self.dense_layers(
          y, decoder_segment_ids, decoder_positions, deterministic, model_mode,
          previous_chunk=previous_chunk,page_state=page_state,slot=slot
        )
        if self.moe_layers is not None:
          y = self.moe_layers(y, decoder_segment_ids, decoder_positions, deterministic, model_mode,
            previous_chunk=previous_chunk,page_state=page_state,slot=slot
          )
        return y

      self._exec = exec_run
      return

    if cfg.decoder_block == DecoderBlockType.GEMMA3:
      attention_pattern_length = len(gemma3.GEMMA3_ATTENTION_PATTERN)
      scan_length = cfg.num_decoder_layers // attention_pattern_length
      RemattedGemma3Block = self.set_remat_policy([gemma3.Gemma3ScannableBlock], self.get_remat_policy())[0]

      # main scan (full patterns)
      self.layers = None
      if scan_length > 0:
        self.layers = self._make_scan_runner(
            RemattedGemma3Block, scan_length, "layers", num_of_layers=attention_pattern_length
        )

      # remainder block (module instance)
      self.layers_remainder = None
      num_remaining_layers = cfg.num_decoder_layers % attention_pattern_length
      if num_remaining_layers > 0:
        self.layers_remainder = RemattedGemma3Block(
            config=cfg, mesh=mesh, quant=self.quant, model_mode=self.model_mode,
            name="layers_remainder", num_of_layers=num_remaining_layers
        )

      def exec_run(y, decoder_segment_ids, decoder_positions, deterministic, model_mode,
                   previous_chunk=None, slot=None, page_state=None, bidirectional_mask=None):
        if self.layers is not None:
          y = self.layers(y, decoder_segment_ids, decoder_positions, deterministic, model_mode)
        if self.layers_remainder is not None:
          # Call remainder with extra kwargs
          y, _ = self.layers_remainder(
              y,
              decoder_segment_ids,
              decoder_positions,
              deterministic,
              model_mode,
              previous_chunk=previous_chunk,
              page_state=page_state,
              slot=slot,
              bidirectional_mask=bidirectional_mask,
          )
        return y

      self._exec = exec_run
      return

    # All other scanned (including LLAMA4 scanned scannable block)
    RemattedBlockLayer = self.rematted_layer_classes[0]
    scan_length = int(cfg.num_decoder_layers / cfg.inhomogeneous_layer_cycle_interval)
    # For LLAMA4 scanned, the layer itself needs kwargs — bake them into the scan builder
    llama4_kwargs = {}
    if cfg.decoder_block == DecoderBlockType.LLAMA4:
      llama4_kwargs = {
        "nope_layer_interval": cfg.nope_layer_interval,
        "interleave_moe_layer_step": cfg.interleave_moe_layer_step,
      }
    self.layers = self._make_scan_runner(RemattedBlockLayer, scan_length, "layers", **llama4_kwargs)

    def exec_run(y, decoder_segment_ids, decoder_positions, deterministic, model_mode,
                 previous_chunk=None, slot=None, page_state=None, bidirectional_mask=None):
      if self.layers is not None:
        y = self.layers(y, 
            decoder_segment_ids, 
            decoder_positions, 
            deterministic, 
            model_mode
          )
      return y

    self._exec = exec_run

  def _build_exec_unscanned(self):
    """No pipeline, unscanned (instantiate all per-layer modules now)."""
    cfg = self.config
    mesh = self.mesh
    self._unscanned_layers: list[nnx.Module] = []

    if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
      if len(self.rematted_layer_classes) != 2:
        raise ValueError(
          f"Unscanned layers must have a length of 2 when using DeepSeek, but got {len(self.rematted_layer_classes)}."
        )
      dense_cls, moe_cls = self.rematted_layer_classes
      # Instantiate all layers now with unique names.
      for idx in range(cfg.first_num_dense_layers):
        self._unscanned_layers.append(
          dense_cls(config=cfg, mesh=mesh, name=f"dense_layers_{idx}", quant=self.quant, model_mode=self.model_mode)
        )
      for idx in range(cfg.num_decoder_layers - cfg.first_num_dense_layers):
        self._unscanned_layers.append(
          moe_cls(config=cfg, mesh=mesh, name=f"moe_layers_{idx}", quant=self.quant, model_mode=self.model_mode)
        )

    else:
      base_cls = self.rematted_layer_classes[0]
      for idx in range(cfg.num_decoder_layers):
        layer_kwargs = {}
        if cfg.decoder_block == DecoderBlockType.GEMMA3:
          layer_kwargs = {"attention_type": gemma3.get_attention_type(layer_id=idx)}
        elif cfg.decoder_block == DecoderBlockType.LLAMA4:
          layer_kwargs = {
              "is_nope_layer": llama4.determine_is_nope_layer(idx, cfg.nope_layer_interval),
              "is_moe_layer": llama4.determine_is_moe_layer(idx, cfg.interleave_moe_layer_step),
          }
        self._unscanned_layers.append(
          base_cls(config=cfg, mesh=mesh, name=f"layers_{idx}", quant=self.quant, model_mode=self.model_mode, **layer_kwargs)
        )

    def exec_run(y, decoder_segment_ids, decoder_positions, deterministic, model_mode,
                 previous_chunk=None, slot=None, page_state=None, bidirectional_mask=None):
      for layer in self._unscanned_layers:
        y = layer(
            y, decoder_segment_ids, decoder_positions, deterministic, model_mode,
            previous_chunk=previous_chunk, page_state=page_state, slot=slot, bidirectional_mask=bidirectional_mask
        )
      return y

    self._exec = exec_run

  def scan_decoder_layers(self, cfg:Config, decoder_layer: Callable, length:int, metadata_axis_name:str, mesh:Mesh, in_axes_tuple:Any, model_mode:str, **kwargs):
    params_spec = cfg.param_scan_axis
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
        config=cfg,
        mesh=mesh,
        name=metadata_axis_name,
        quant=self.quant,
        model_mode=model_mode,
        **kwargs
    )


  def get_pipeline_stage_module(self, decoder_blocks:list[Type[nnx.Module]]) -> nnx.Module:
    """get pipeline stage module"""

    def get_layer_to_pipeline(blocks: list[Type[nnx.Module]], cfg:Config)->Callable[..., nnx.Module]:
      if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
        return blocks[1]  # return the sparse block
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
          in_axes_tuple=(nn.broadcast,) * self.BROADCAST_ARGS_LENGTH,
          model_mode=self.model_mode,
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
  
  def _apply_embedding(
      self,
      decoder_input_tokens:Array,
      decoder_positions:Array,
      deterministic:bool,
      model_mode:str,
      image_embeddings: np.ndarray | Array | None=None,
      bidirectional_mask=None,
  )->Array:
    cfg = self.config

    y = self.shared_embedding(decoder_input_tokens.astype("int32"), model_mode=model_mode)
    
    # Merge the image embeddings with the text embeddings for multimodal models
    if image_embeddings is not None and cfg.use_multimodal:
      # TODO(hengtaoguo): Add support for other multimodal models such as Llama4, refactor if needed
      if cfg.model_name not in {"gemma3-4b", "gemma3-12b", "gemma3-27b", "llama4-17b-16e", "llama4-17b-128e"}:
        raise ValueError(f"Unsupported model_name for multimodal: {cfg.model_name}")
      
      y = multimodal_utils.merge_mm_embeddings(
          text_embeddings=y,
          vision_embeddings=image_embeddings,
          mask=bidirectional_mask,
      )
      
    y = self.embedding_dropout(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    if cfg.use_untrainable_positional_embedding:
      y = self.static_pos_embedding(y, decoder_positions)

    if cfg.trainable_position_size > 0 and self.position_embedder:
      y += self.position_embedder(decoder_positions, model_mode=model_mode)
    return y

  def _apply_output_head(self, y:Array, deterministic:bool, model_mode:str)->Array:
    cfg = self.config

    # Use the pre-instantiated norm layer
    y = self.decoder_norm(y)
    y = self.output_head_dropout(y, deterministic=deterministic)

    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      if isinstance(self.shared_embedding, nnx.Module):
        embedding_table = self.shared_embedding.embedding.value
      else:
        embedding_table = self.shared_embedding.variables["params"]["embedding"]
      if isinstance(embedding_table, nn.spmd.LogicallyPartitioned):
        embedding_table = embedding_table.unbox()
      attend_dtype = jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype
      logits = attend_on_embedding(y, embedding_table, attend_dtype, self.config)
      if self.config.normalize_embedding_logits:
        logits = logits / jnp.sqrt(y.shape[-1])
      if cfg.final_logits_soft_cap:
        logits = logits / cfg.final_logits_soft_cap
        logits = jnp.tanh(logits) * cfg.final_logits_soft_cap
    else:
      logits = self.logits_dense(y)

    logical_axis_resource = (
       (None, None, "activation_vocab") 
       if model_mode in {MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE} 
       else ("activation_embed_and_logits_batch", "activation_length", "activation_vocab")
    )
    logits = nn.with_logical_constraint(logits, logical_axis_resource)

    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)

    return logits

  def __call__(
      self,
      decoder_input_tokens: Array,
      decoder_positions: Array,
      decoder_segment_ids: Array|None=None,
      deterministic:bool=False,
      model_mode:str=MODEL_MODE_TRAIN,
      previous_chunk: Array | None =None,
      slot: int | None = None,
      page_state: page_manager.PageState | None = None,
      bidirectional_mask: Array | None = None,
      image_embeddings: Array | None = None,
  )->tuple[Array,Array]:

    if decoder_input_tokens.ndim != 2: 
      raise ValueError(
          f"`decoder_input_tokens` must have shape [batch, length], "
          f"but got array with shape {decoder_input_tokens.shape}."
      )

    y = self._apply_embedding(
        decoder_input_tokens,
        decoder_positions,
        deterministic,
        model_mode,
        image_embeddings,
        bidirectional_mask,
    )

    y = self._exec(
        y,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        previous_chunk=previous_chunk,
        slot=slot,
        page_state=page_state,
        bidirectional_mask=bidirectional_mask,
    )

    if not isinstance(y, jax.Array):
      raise TypeError(f"Expected `y` to be a jax.Array, but got {type(y).__name__}.")

    hidden_state = y
    logits = self._apply_output_head(hidden_state, deterministic, model_mode)
    return logits, hidden_state

  @property
  def pipeline_module(self) -> pipeline.Pipeline:
    if self._pipeline_module is None:
      raise RuntimeError(
        "Pipeline module is not initialized. Set 'ici_pipeline_parallelism' or `dcn_pipeline_parallelism` value "
        +"larger than 1 in config to enable pipeline parallelism.")
    return self._pipeline_module

  @property
  def static_pos_embedding(self)->nn.Module:
    if self._static_pos_embedding is None:
      raise RuntimeError("Set 'use_untrainable_positional_embedding' in config to enable positional embedding")
    return self._static_pos_embedding

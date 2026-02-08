import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from MaxText import max_logging, maxtext_utils, multimodal_utils, sharding
from MaxText.common_types import (
    EP_AS_CONTEXT,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    Config,
    DecoderBlockType,
    ShardMode,
)
from MaxText.inference import page_manager
from MaxText.layers import (
    deepseek,
    deepseek_batchsplit,
    gemma,
    gemma2,
    gemma3,
    gpt3,
    gpt_oss,
    initializers,
    linears,
    llama2,
    llama4,
    mistral,
    mixtral,
    nnx_wrappers,
    quantizations,
    qwen3,
    simple_layer,
    olmo3,
)

from MaxText.layers import nnx_pipeline as pipeline

# Assumes these modules are adapted for NNX
from MaxText.layers.attentions import Attention
from MaxText.layers.embeddings import Embed, PositionalEmbedding, attend_on_embedding
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.quantizations import AqtQuantization as Quant
from MaxText.sharding import create_sharding


class NNXDecoderLayer(nnx.Module):
  """
  Transformer decoder layer that attends to the encoder.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: Quant | None = None,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
      **kwargs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode

    # Initialize Pre-Attention Norm
    self.pre_self_attention_norm = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    # Initialize Attention
    self.self_attention = Attention(
        config=self.config,
        num_query_heads=self.config.num_query_heads,
        num_kv_heads=self.config.num_kv_heads,
        head_dim=self.config.head_dim,
        max_target_length=self.config.max_target_length,
        mesh=mesh,
        attention_kernel=self.config.attention,
        inputs_q_shape=(1, 1, self.config.emb_dim),
        inputs_kv_shape=(1, 1, self.config.emb_dim),
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        max_prefill_predict_length=self.config.max_prefill_predict_length,
        dropout_rate=self.config.dropout_rate,
        float32_qk_product=self.config.float32_qk_product,
        float32_logits=self.config.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(config),
        prefill_cache_axis_order=tuple(map(int, self.config.prefill_cache_axis_order.split(","))),
        ar_cache_axis_order=tuple(map(int, self.config.ar_cache_axis_order.split(","))),
        compute_axis_order=tuple(map(int, self.config.compute_axis_order.split(","))),
        reshape_q=self.config.reshape_q,
        model_mode=model_mode,
        rngs=rngs,
        **kwargs,
    )

    # Initialize MLP
    self.mlp = linears.MlpBlock(
        in_features=self.config.emb_dim,
        intermediate_dim=self.config.mlp_dim,
        activations=self.config.mlp_activations,
        intermediate_dropout_rate=self.config.dropout_rate,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        model_mode=model_mode,
        config=self.config,
        quant=self.quant,
        mesh=self.mesh,
        rngs=rngs,
    )

    # Initialize Dropout
    self.dropout = linears.Dropout(rate=config.dropout_rate, rngs=rngs, broadcast_dims=(-2,))

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
      **kwargs,
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
        **kwargs,
    )
    attention_lnx = _maybe_shard_with_logical(attention_lnx, logical_axis_names)

    mlp_lnx = self.mlp(lnx, deterministic=deterministic)
    mlp_lnx = _maybe_shard_with_logical(mlp_lnx, logical_axis_names)

    next_layer_addition = mlp_lnx + attention_lnx
    next_layer_addition_dropped_out = self.dropout(next_layer_addition, deterministic=deterministic)

    layer_output = next_layer_addition_dropped_out + inputs
    layer_output = _maybe_shard_with_logical(layer_output, logical_axis_names)

    if self.config.record_internal_nn_metrics:
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if self.config.scan_layers:
      return layer_output, None

    return layer_output, kv_cache


class NNXSequentialBlockDecoderLayers(nnx.Module):
  """Sequential unscanned series of decoder layers."""

  def __init__(
      self,
      decoder_layer: Any,
      num_decoder_layers: int,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: Quant,
      remat_policy: Any = None,
      layer_idx_offset: int = 0,
      **kwargs,
  ):
    self.config = config
    self.num_decoder_layers = num_decoder_layers
    self.remat_policy = remat_policy
    max_logging.info(f"NNXSequential - Init for {num_decoder_layers} layers with offset {layer_idx_offset}")

    for i in range(num_decoder_layers):
      # Calculate layer-specific arguments based on global index
      global_idx = layer_idx_offset + i
      layer_init_kwargs = kwargs.copy()
      
      if config.decoder_block == DecoderBlockType.GEMMA3:
          layer_init_kwargs["attention_type"] = gemma3.get_attention_type(layer_id=global_idx)
      elif config.decoder_block == DecoderBlockType.LLAMA4:
          layer_init_kwargs["is_nope_layer"] = llama4.determine_is_nope_layer(global_idx, config.nope_layer_interval)
          layer_init_kwargs["is_moe_layer"] = llama4.determine_is_moe_layer(global_idx, config.interleave_moe_layer_step)
      elif config.decoder_block == DecoderBlockType.QWEN3_NEXT:
          layer_init_kwargs["layer_idx"] = global_idx
      elif config.decoder_block == DecoderBlockType.GPT_OSS:
          layer_init_kwargs["attention_type"] = gpt_oss.get_attention_type(layer_id=global_idx)

      setattr(
          self,
          f"layers_{i}",
          decoder_layer(config=config, mesh=mesh, model_mode=model_mode, rngs=rngs, quant=quant, **layer_init_kwargs),
      )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids,
      decoder_positions,
      deterministic: bool,
      model_mode,
      slot: int | None = None,
      page_state: Any | None = None,
      **kwargs,
  ) -> jnp.ndarray:
    
    prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)

    for number in range(self.num_decoder_layers):
      layer = getattr(self, f"layers_{number}")
      
      if self.remat_policy is not None:
         # OOM Fix: Split Params (Static) from Mutable State.
         graphdef, params, mutable_state = nnx.split(layer, nnx.Param, ...)
         
         def rematted_forward(g, p, s, x, *args, **kw):
             m = nnx.merge(g, p, s)
             out = m(x, *args, **kw)
             _, new_s = nnx.split(m, nnx.Param, ...)
             return out, new_s
             
         forward_with_ckpt = jax.checkpoint(rematted_forward, policy=self.remat_policy, prevent_cse=prevent_cse)
         
         inputs_out, new_mutable_state = forward_with_ckpt(
             graphdef, 
             params, 
             mutable_state, 
             inputs, 
             decoder_segment_ids,
             decoder_positions,
             deterministic,
             model_mode,
             None, # previous_chunk
             slot,
             page_state,
             None, # kv_cache
             None, # attention_metadata
             **kwargs # pass remaining kwargs
         )
         nnx.update(layer, new_mutable_state)
         inputs = inputs_out
      else:
         inputs = layer(
            inputs,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            slot=slot,
            page_state=page_state,
            **kwargs,
         )

      if self.config.scan_layers:
        inputs = inputs[0]
        
    if self.config.scan_layers:
      return inputs, None
    return inputs


class NNXScannedBlockDecoderLayers(nnx.Module):
  """Scanned series of decoder layers.
  
  Used primarily by Pipeline to wrap vmapped layers in a callable module.
  """
  def __init__(self, vmapped_layers, num_layers, config, remat_policy=None):
      self.layers = vmapped_layers
      self.num_layers = num_layers
      self.config = config
      self.remat_policy = remat_policy

  def __call__(self, x, *args, **kwargs):
      return _apply_scanned_layers(
          self.layers, 
          x, 
          *args, 
          length=self.num_layers, 
          remat_policy=self.remat_policy, 
          config=self.config,
          **kwargs
      )


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
    max_logging.log(f"DEBUG: Initializing NNXDecoder for model {config.model_name}")
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
          embedding_init=nnx.initializers.normal(stddev=1.0),
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

    self.using_pipeline = config.using_pipeline_parallelism
    self.is_deepseek = self.config.decoder_block == DecoderBlockType.DEEPSEEK
    self.is_gemma3 = self.config.decoder_block == DecoderBlockType.GEMMA3
    self.remat_policy = self.get_remat_policy()

    if self.using_pipeline:
      max_logging.log(f"DEBUG: Initializing Pipeline Module with {config.pipeline_parallel_layers} layers.")
      self.pipeline_module = self.get_pipeline_stage_module(decoder_block_classes)
      self.num_pipeline_layers = config.pipeline_parallel_layers
      
      num_remaining_layers = config.num_decoder_layers - self.num_pipeline_layers
      if self.is_deepseek:
          # DeepSeek Dense Layers (Pre-Pipeline)
          dense_cls = decoder_block_classes[0]
          num_dense = config.first_num_dense_layers
          
          if config.scan_layers:
              # Directly store vmapped module to avoid extra nesting
              self.dense_layers = self._create_scanned_layers_vmap(dense_cls, length=num_dense, rngs=rngs)
          else:
              self.dense_layers = NNXSequentialBlockDecoderLayers(
                  dense_cls, num_dense, config, mesh, model_mode, rngs, quant, 
                  remat_policy=self.remat_policy
              )

      if num_remaining_layers > 0:
        max_logging.log(f"DEBUG: Initializing Remainder Layers: {num_remaining_layers} layers.")
        layer_cls = decoder_block_classes[1] if self.is_deepseek else decoder_block_classes[0]
        
        if config.scan_layers:
           # Directly store vmapped module to avoid extra nesting
           self.remainder_layers = self._create_scanned_layers_vmap(layer_cls, length=num_remaining_layers, rngs=rngs)
        else:
          self.remainder_layers = NNXSequentialBlockDecoderLayers(
              layer_cls, num_remaining_layers, config, mesh, model_mode, rngs, quant,
              remat_policy=self.remat_policy,
              layer_idx_offset=self.num_pipeline_layers
          )

    elif self.config.scan_layers:
      if self.is_deepseek:
        assert len(decoder_block_classes) == 2
        dense_cls, moe_cls = decoder_block_classes

        num_dense = config.first_num_dense_layers
        self.dense_layers = self._create_scanned_layers_vmap(dense_cls, length=num_dense, rngs=rngs)

        num_moe = config.num_decoder_layers - config.first_num_dense_layers
        self.moe_stack = self._create_scanned_layers_vmap(moe_cls, length=num_moe, rngs=rngs)
        
      elif self.is_gemma3:
        # Gemma 3 scanning logic
        attention_pattern_length = len(gemma3.GEMMA3_ATTENTION_PATTERN)
        scan_length = config.num_decoder_layers // attention_pattern_length
        num_remaining_layers = config.num_decoder_layers % attention_pattern_length
        rem_layer_kwargs = {"num_of_layers": num_remaining_layers}

        RemattedGemma3Block = gemma3.Gemma3ScannableBlock

        if scan_length > 0:
          self.layers = self._create_scanned_layers_vmap(RemattedGemma3Block, length=scan_length, rngs=rngs, num_of_layers=attention_pattern_length)
          
        self.layers_remainder = RemattedGemma3Block(
            config=self.config, mesh=mesh, quant=self.quant, model_mode=self.model_mode, **rem_layer_kwargs, rngs=rngs
        )
      else:
        # STANDARD SCANNED PATH
        layer_cls = decoder_block_classes[0]
        num_layers = config.num_decoder_layers
        # Directly store vmapped module to result in 'decoder/layers/...'
        self.layers = self._create_scanned_layers_vmap(layer_cls, length=num_layers, rngs=rngs)
    else:
      # Sequential Unscanned
      # Standard Sequential (No Pipeline, No Scan)
      # FIX: Flatten layers to match Linen's decoder/layers_0 structure
      if self.is_deepseek:
          dense_cls, moe_cls = decoder_block_classes
          num_dense = config.first_num_dense_layers
          num_moe = config.num_decoder_layers - config.first_num_dense_layers
          
          # Flatten Dense
          for i in range(num_dense):
             layer = self._create_single_layer(dense_cls, rngs)
             setattr(self, f"dense_layers_{i}", layer)
             
          # Flatten MoE
          for i in range(num_moe):
             layer = self._create_single_layer(moe_cls, rngs)
             setattr(self, f"moe_layers_{i}", layer)
             
      else:
        layer_cls = decoder_block_classes[0]
        for i in range(config.num_decoder_layers):
             # Replicate model-specific kwarg logic
             layer_kwargs = {}
             if config.decoder_block == DecoderBlockType.GEMMA3:
                 layer_kwargs["attention_type"] = gemma3.get_attention_type(layer_id=i)
             elif config.decoder_block == DecoderBlockType.LLAMA4:
                 layer_kwargs["is_nope_layer"] = llama4.determine_is_nope_layer(i, config.nope_layer_interval)
                 layer_kwargs["is_moe_layer"] = llama4.determine_is_moe_layer(i, config.interleave_moe_layer_step)
             elif config.decoder_block == DecoderBlockType.QWEN3_NEXT:
                 layer_kwargs["layer_idx"] = i
             elif config.decoder_block == DecoderBlockType.GPT_OSS:
                 layer_kwargs["attention_type"] = gpt_oss.get_attention_type(layer_id=i)

             layer = self._create_single_layer(layer_cls, rngs, **layer_kwargs)
             setattr(self, f"layers_{i}", layer)


  def get_pipeline_stage_module(self, decoder_blocks):
    """Creates the Pipeline module with the correct stage configuration."""
    max_logging.log(f"DEBUG: get_pipeline_stage_module called.")
    cfg = self.config

    def get_layer_to_pipeline(blocks, cfg):
      if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
        return blocks[1]
      else:
        return blocks[0]

    base_stage_cls = get_layer_to_pipeline(decoder_blocks, cfg)
    remat_policy = self.get_remat_policy() if cfg.set_remat_policy_on_layers_per_stage else None
    
    if cfg.num_layers_per_pipeline_stage == 1:
      stage_module = NNXSequentialBlockDecoderLayers(
          decoder_layer=base_stage_cls,
          num_decoder_layers=1,
          config=cfg,
          mesh=self.mesh,
          model_mode=self.model_mode,
          rngs=self.rngs,
          quant=self.quant,
          remat_policy=remat_policy
      )
    elif cfg.scan_layers_per_stage:
      vmapped_stage = self._create_scanned_layers_vmap(
          base_stage_cls,
          length=cfg.num_layers_per_pipeline_stage,
          rngs=self.rngs,
      )
      # For Pipeline, we MUST return a callable Module, so we wrap it.
      stage_module = NNXScannedBlockDecoderLayers(
          vmapped_stage, 
          cfg.num_layers_per_pipeline_stage, 
          cfg, 
          remat_policy=remat_policy
      )
    else:
      stage_module = NNXSequentialBlockDecoderLayers(
          decoder_layer=base_stage_cls,
          num_decoder_layers=cfg.num_layers_per_pipeline_stage,
          config=cfg,
          mesh=self.mesh,
          model_mode=self.model_mode,
          rngs=self.rngs,
          quant=self.quant,
          remat_policy=remat_policy
      )

    return pipeline.Pipeline(
        config=cfg,
        layers=stage_module,
        mesh=self.mesh,
        remat_policy=self.get_remat_policy(),
        rngs=self.rngs,
    )

  def _create_single_layer(self, decoder_layer_class, rngs, **kwargs):
    if issubclass(decoder_layer_class, nnx.Module):
      return decoder_layer_class(
          config=self.config, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, rngs=rngs, **kwargs
      )
    else:
      layer_linen = decoder_layer_class(
          config=self.config, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, **kwargs
      )
      return nnx_wrappers.ToNNX(layer_linen, rngs=rngs)

  def _create_scanned_layers_vmap(self, decoder_layer_class, length: int, rngs: nnx.Rngs, **layer_kwargs):
    """Creates a VMapped stack of layers, forcing parameter init."""
    def create_layer_fn(layer_rngs):
      layer = decoder_layer_class(
          config=self.config, 
          mesh=self.mesh, 
          quant=self.quant, 
          model_mode=self.model_mode, 
          rngs=layer_rngs, 
          **layer_kwargs
      )
      return layer

    try:
      forked_rngs = rngs.fork(split=length)
    except: 
      pass

    layers_vmapped = nnx.vmap(
        create_layer_fn,
        in_axes=0,
        out_axes=0,
        axis_name="layers",
        transform_metadata={nnx.PARTITION_NAME: "layers"},
    )(forked_rngs)

    max_logging.log("DEBUG: Patching sharding specs for scanned layers...")
    # Patch sharding specs for scanned layers to include 'layers' axis
    
    return layers_vmapped

  def _get_pipeline_logical_spec(self):
      """Extracts weight sharding spec from the pipeline module to support FSDP AG."""
      if not hasattr(self, 'pipeline_module'):
          return None
      
      try:
          state = nnx.state(self.pipeline_module.layers)
          def _get_spec(leaf):
             if hasattr(leaf, 'sharding') and hasattr(leaf.sharding, 'spec'):
                 return leaf.sharding.spec
             if hasattr(leaf, 'value') and isinstance(leaf.value, nn.spmd.LogicallyPartitioned):
                 return leaf.value.partitions
             return None
          
          return jax.tree.map(_get_spec, state)
      except:
          return None

  def _apply_layers_sequentially(self, layers, x_in, *args, length: int, **kwargs):
    """Runs the layer stack using nnx.scan."""
    policy = self.get_remat_policy()
    prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
    graphdef, params, state = nnx.split(layers, nnx.Param, ...)

    layer_cls = layers.__class__
    sig = inspect.signature(layer_cls.__call__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or "kwargs" in sig.parameters}

    def layer_fn(carry, scanned_vars):
      current_params, current_state = scanned_vars
      layer = nnx.merge(graphdef, current_params, current_state)
      layer_out = layer(carry, *args, **valid_kwargs)
      new_carry = layer_out[0] if isinstance(layer_out, tuple) else layer_out
      new_current_state = nnx.state(layer)
      return new_carry, new_current_state

    layer_fn = jax.checkpoint(layer_fn, policy=policy, prevent_cse=prevent_cse)
    final_carry, scanned_state = jax.lax.scan(layer_fn, x_in, (params, state))
    nnx.update(layers, scanned_state)

    return final_carry, None


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

  def get_norm_layer(self, num_features: int, rngs: nnx.Rngs):
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
        DecoderBlockType.QWEN3_NEXT,
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
    """Applies token and positional embeddings to the input tokens."""
    cfg = self.config

    y = shared_embedding(decoder_input_tokens.astype("int32"), model_mode=model_mode)

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
            multimodal_embeddings=image_embeddings,
            mask=bidirectional_mask,
            token_masks=image_masks,
        )
      # TODO(hengtaoguo): Add support for other multimodal models such as Llama4, refactor if needed
      else:
        raise ValueError(f"Unsupported model_name for multimodal: {cfg.model_name}")

    if audio_embeddings is not None and cfg.use_audio:
      if cfg.model_name in ["qwen3-omni-30b-a3b"]:
        y = multimodal_utils.merge_mm_embeddings(
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
      y = self.positional_embedding(y, decoder_positions)

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

    y = self.decoder_norm(y, norm_out_sharding)
    y = self.dropout(y, deterministic=deterministic)  # NNX call

    if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
      out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
    else:
      out_sharding = create_sharding(
          self.mesh, ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_vocab")
      )

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
  ):
    cfg = self.config
    assert decoder_input_tokens.ndim == 2  # [batch, len]

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
    layer_args = (decoder_segment_ids, decoder_positions, deterministic, model_mode)

    layer_kwargs = {
        "previous_chunk": previous_chunk,
        "page_state": page_state,
        "slot": slot,
        "attention_metadata": attention_metadata,
    }
    if cfg.decoder_block == DecoderBlockType.GEMMA3:
      layer_kwargs["bidirectional_mask"] = bidirectional_mask


    if self.using_pipeline:
      if cfg.pipeline_fsdp_ag_once:
        logical_partition_spec = self._get_pipeline_logical_spec()
      else:
        logical_partition_spec = None
      
      if self.is_deepseek:
          # DeepSeek Dense - check if scanned or sequential
          if cfg.scan_layers:
             y, _ = self._apply_layers_sequentially(self.dense_layers, y, *layer_args, length=cfg.first_num_dense_layers, **layer_kwargs)
          else:
             y = self.dense_layers(y, *layer_args, **layer_kwargs)

      y = self.pipeline_module(y, *layer_args, logical_partition_spec=logical_partition_spec)
      
      if hasattr(self, 'remainder_layers'):
        kv_start_idx = self.num_pipeline_layers 
        
        if isinstance(self.remainder_layers, NNXSequentialBlockDecoderLayers):
             idx_counter = 0
             for i in range(self.remainder_layers.num_decoder_layers):
                layer = getattr(self.remainder_layers, f"layers_{i}")
                global_idx = kv_start_idx + idx_counter
                kv_cache = kv_caches[global_idx] if kv_caches is not None else None
                out = layer(y, *layer_args, kv_cache=kv_cache, **layer_kwargs)
                
                if isinstance(out, tuple):
                   y, kv_cache_out = out
                else:
                   y = out
                   kv_cache_out = None
                
                if kv_caches is not None:
                   kv_caches[global_idx] = kv_cache_out
                idx_counter += 1
        else:
            # Scanned remainder - direct helper call
            num_remaining = cfg.num_decoder_layers - self.num_pipeline_layers
            y, _ = self._apply_layers_sequentially(self.remainder_layers, y, *layer_args, length=num_remaining, **layer_kwargs)

    elif cfg.scan_layers:
      if self.is_deepseek:
        y, _ = self._apply_layers_sequentially(self.dense_layers, y, *layer_args, length=cfg.first_num_dense_layers, **layer_kwargs)
        num_moe = cfg.num_decoder_layers - cfg.first_num_dense_layers
        y, _ = self._apply_layers_sequentially(self.moe_stack, y, *layer_args, length=num_moe, **layer_kwargs)
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
        # Standard Scan: Directly call helper on vmapped module
        y, _ = self._apply_layers_sequentially(self.layers, y, *layer_args, length=cfg.num_decoder_layers, **layer_kwargs)

    else:
      # Sequential Unscanned
      # Standard Sequential (No Pipeline, No Scan)
      idx_counter = 0
      prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)

      if self.is_deepseek:
          # Iterate over flattened attributes
          layer_names = [f"dense_layers_{i}" for i in range(cfg.first_num_dense_layers)]
          layer_names += [f"moe_layers_{i}" for i in range(cfg.num_decoder_layers - cfg.first_num_dense_layers)]
      else:
          layer_names = [f"layers_{i}" for i in range(cfg.num_decoder_layers)]

      for layer_name in layer_names:
          layer = getattr(self, layer_name)
          kv_cache = kv_caches[idx_counter] if kv_caches is not None else None

          # Apply Remat + OOM Fix
          if self.remat_policy is not None:
             graphdef, params, mutable_state = nnx.split(layer, nnx.Param, ...)
             
             def rematted_forward(g, p, s, x, kv, *args, **kw):
                 m = nnx.merge(g, p, s)
                 out = m(x, *args, kv_cache=kv, **kw)
                 _, new_s = nnx.split(m, nnx.Param, ...)
                 return out, new_s
             
             forward_ckpt = jax.checkpoint(rematted_forward, policy=self.remat_policy, prevent_cse=prevent_cse)
             y, new_mutable_state = forward_ckpt(graphdef, params, mutable_state, y, kv_cache, *layer_args, **layer_kwargs)
             nnx.update(layer, new_mutable_state)
          else:
             y = layer(y, *layer_args, kv_cache=kv_cache, **layer_kwargs)

          if isinstance(y, tuple):
             y, kv_cache_out = y
          else:
             kv_cache_out = None
          
          if kv_caches is not None:
            kv_caches[idx_counter] = kv_cache_out
          idx_counter += 1

    assert isinstance(y, jax.Array)
    hidden_state = y

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
    scan_length = cfg.num_decoder_layers // len(gemma3.GEMMA3_ATTENTION_PATTERN)
    layer_call_kwargs = {"bidirectional_mask": bidirectional_mask}

    if scan_length > 0:
      broadcast_args = (
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
      )
      # Layers is the vmapped module
      y, _ = self._apply_layers_sequentially(self.layers, y, *broadcast_args, length=scan_length, **layer_call_kwargs)

    num_remaining_layers = cfg.num_decoder_layers % len(gemma3.GEMMA3_ATTENTION_PATTERN)
    if num_remaining_layers > 0:
      y, _ = self.layers_remainder(
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


def _apply_scanned_layers(layers, x_in, *args, length: int, remat_policy=None, config=None, **kwargs):
    """Helper for NNXScannedBlockDecoderLayers.__call__ (Pipeline use case)."""
    # This is identical to _apply_layers_sequentially but externalized
    prevent_cse = maxtext_utils.should_prevent_cse_in_remat(config) if config else False
    
    graphdef, params, state = nnx.split(layers, nnx.Param, ...) 

    def layer_fn(carry, scanned_vars):
      current_params, current_state = scanned_vars
      layer = nnx.merge(graphdef, current_params, current_state)
      layer_out = layer(carry, *args, **kwargs)
      new_carry = layer_out[0] if isinstance(layer_out, tuple) else layer_out
      new_current_state = nnx.state(layer)
      return new_carry, new_current_state

    if remat_policy:
        layer_fn = jax.checkpoint(layer_fn, policy=remat_policy, prevent_cse=prevent_cse)

    final_carry, scanned_state = jax.lax.scan(layer_fn, x_in, (params, state), length=length)
    nnx.update(layers, scanned_state)

    return final_carry, None


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
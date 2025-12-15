from typing import Any, Callable, Sequence, Optional, Tuple, List, Union
import functools
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from flax import linen as nn
from flax import nnx
from flax.nnx import Rngs

from MaxText.common_types import (
    DecoderBlockType,
    ShardMode,
    Config,
    EP_AS_CONTEXT,
    MODEL_MODE_TRAIN,
    MODEL_MODE_PREFILL,
    MODEL_MODE_AUTOREGRESSIVE,
)
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
from MaxText.layers.attentions import attention_as_linen
from MaxText.layers.normalizations import rms_norm
from MaxText.layers.embeddings import Embed, attend_on_embedding, embed_as_linen, positional_embedding_as_linen
from MaxText.layers.embeddings import Embed, attend_on_embedding, embed_as_linen, positional_embedding_as_linen
from MaxText.layers.quantizations import AqtQuantization as Quant
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


# ------------------------------------------------------------------------------
# Decoder Layer (NNX Implementation)
# ------------------------------------------------------------------------------

class DecoderLayer(nnx.Module):
    """Transformer decoder layer that attends to the encoder."""

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

class ScannedBlock(nnx.Module):
  """Wraps a vmapped layer stack to execute it via jax.lax.scan.
     This replaces the closure 'scan_runner' to make NNX happy.
  """
  def __init__(self, layers_vmapped, length, config, remat_policy):
    self.layers = layers_vmapped
    self.length = length
    self.config = config
    self.remat_policy = remat_policy

  def __call__(self, x_in, *args, **kwargs):
    # Split the vmapped module into Graph and Params
    graph_def, params_stack = nnx.split(self.layers)
    
    # Prepare kwargs (filter out model_mode if needed, or pass through)
    run_kwargs = kwargs.copy()
    # Ensure model_mode isn't passed twice if it's in *args (broadcast_args)
    run_kwargs.pop('model_mode', None)

    def forward_single_step(carry, params_slice):
      # Merge params back into a functional instance for this step
      layer_i = nnx.merge(graph_def, params_slice)
      
      # Run the layer
      # Note: *args captures [segment_ids, positions, deterministic, model_mode] 
      layer_out = layer_i(carry, *args, **run_kwargs)

      # Handle potential tuple return (e.g. (output, None)) from DecoderLayer
      if isinstance(layer_out, tuple):
        new_carry = layer_out[0]
        extra_out = layer_out[1]
      else:
        new_carry = layer_out
        extra_out = None

      # Split again to capture any state updates (if mutable)
      _, new_params_slice = nnx.split(layer_i)

      return new_carry, (new_params_slice, extra_out)

    # Apply Checkpointing (Remat)
    # Using jax.checkpoint instead of nnx.remat to keep explicit control over policy
    prevent_cse = not self.config.scan_pipeline_iterations
    rematted_step = jax.checkpoint(forward_single_step, policy=self.remat_policy, prevent_cse=prevent_cse)
    
    # Run Scan
    final_carry, (new_params_stack, stacked_outs) = jax.lax.scan(
      rematted_step,
      init=x_in,
      xs=params_stack,
      length=self.length,
    )

    # Update the stored parameters with the result (if they changed)
    nnx.update(self.layers, new_params_stack)

    # Return structure matching original code: (output, extra)
    return final_carry, stacked_outs


class SequentialBlockDecoderLayers(nnx.Module):
  """Sequential unscanned series of decoder layers."""


  def __init__(self,decoder_layer:Any, num_decoder_layers:int, config:Config, mesh:Mesh, quant:Quant, model_mode:str, rngs:nnx.Rngs):
    self.decoder_layer = decoder_layer
    self.num_decoder_layers = num_decoder_layers
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    self.rngs = rngs
    for lyr in range(num_decoder_layers):
      new_layer = self.decoder_layer(
          config=self.config, mesh=self.mesh, quant=self.quant, model_mode=model_mode,
          rngs=self.rngs
      )
      setattr(self, f"layer_{lyr}", new_layer)

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
      inputs = getattr(self,f"layer_{lyr}")(
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
class Decoder(nnx.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""
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
    self.rngs = rngs
  
    super().__init__()

    """Initialize decoder layer."""
    self.decoder_layer = self.get_decoder_layers()
    self.norm_layer = self.get_norm_layer(num_features=config.emb_dim)(
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        # name="decoder_norm",
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        parameter_memory_host_offload=config.parameter_memory_host_offload,
        rngs=self.rngs
    )
    if self.config.using_pipeline_parallelism:
      pipeline_stage_module = self.get_pipeline_stage_module(self.decoder_layer)
      remat_policy = self.get_remat_policy()
      self.pipeline_module = pipeline.Pipeline(
          config=self.config, mesh=self.mesh, layers=pipeline_stage_module, remat_policy=remat_policy,
          rngs=self.rngs
      )


    self.position_embedder = Embed(
        num_embeddings=config.trainable_position_size,
        num_features=config.emb_dim,
        dtype=config.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        config=config,
        mesh=self.mesh,
        rngs=rngs,
    )

    self.dropout = linears.Dropout(rate=config.dropout_rate, broadcast_dims=(-2,), rngs=self.rngs)

    self.positional_embedding = PositionalEmbedding(embedding_dims=config.base_emb_dim)

    policy = self.get_remat_policy()
    self.RemattedBlockLayers = self.set_remat_policy(self.decoder_layer, policy)

    broadcast_args_len = 4
    self.moe_layer = None

    if config.using_pipeline_parallelism:
      self.dense_layer = self.RemattedBlockLayers[0]
      if config.decoder_block == DecoderBlockType.DEEPSEEK:
        self.dense_layers = self.scan_decoder_layers(
            config,
            self.dense_layer,
            config.first_num_dense_layers,
            "dense_layers",
            mesh,
            in_axes_tuple=(nn.broadcast,) * broadcast_args_len,
            model_mode=model_mode,
        )

        self.moe_layers = self.scan_decoder_layers(
            config,
            self.moe_layer,
            config.num_moe_layers_outside_pp,
            "moe_layers",
            mesh,
            in_axes_tuple=(nn.broadcast,) * broadcast_args_len,
            model_mode=model_mode,
        )
      else:
        remaining_layers = self.config.num_decoder_layers - self.config.pipeline_parallel_layers
        breakpoint()
        self.layers_outside_pipeline = self.scan_decoder_layers(
                    config,
                    self.RemattedBlockLayers[0],
                    remaining_layers,
                    "layers_outside_pipeline",
                    mesh,
                    in_axes_tuple=(nn.broadcast,) * broadcast_args_len,
                    model_mode=model_mode,
                )
    else:
      if config.scan_layers:
        if config.decoder_block == DecoderBlockType.DEEPSEEK:

          dense_layer = self.RemattedBlockLayers[0]
          dense_layer.__call__ = functools.partial(dense_layer.__call__, **layer_call_kwargs)
          moe_layer = self.RemattedBlockLayers[1]
          moe_layer.__call__ = functools.partial(moe_layer.__call__, **layer_call_kwargs)
          y, _ = self.scan_decoder_layers(
              config,
              dense_layer,
              config.first_num_dense_layers,
              "dense_layers",
              mesh,
              in_axes_tuple=(nn.broadcast,) * len(broadcast_args_len),
              model_mode=model_mode,
          )
        elif config.decoder_block == DecoderBlockType.GEMMA3:
          pass
        else:
          RemattedBlockLayer = self.RemattedBlockLayers[0]
          scan_length = int(config.num_decoder_layers / config.inhomogeneous_layer_cycle_interval)
          layer_kwargs = {}
          if config.decoder_block == DecoderBlockType.LLAMA4:
            layer_kwargs = {
                "nope_layer_interval": self.config.nope_layer_interval,
                "interleave_moe_layer_step": self.config.interleave_moe_layer_step,
            }
          self.weights, self.layers = self.scan_decoder_layers(
              config,
              RemattedBlockLayer,
              scan_length,
              "layers",
              mesh,
              in_axes_tuple=(nn.broadcast,) * broadcast_args_len,
              model_mode=model_mode,
              **layer_kwargs,
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
        if self.config.use_batch_split_schedule:
          return [deepseek_batchsplit.DeepSeekDenseLayer, deepseek_batchsplit.DeepSeekMoELayer]
        else:
          return [deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer]
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

      # rematted_step = jax.checkpoint(block_layer, prevent_cse=True)

      # Apply remat policy to layer
      # layer = nn.remat(
      #     block_layer,
      #     prevent_cse=maxtext_utils.should_prevent_cse_in_remat(self.config),
      #     policy=policy,
      #     static_argnums=(4, 5),  # Deterministic and model mode are static arguments.
      # )
      RemattedBlockLayers.append(block_layer)
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
        DecoderBlockType.QWEN3_NEXT,
        DecoderBlockType.GPT_OSS,
        DecoderBlockType.SIMPLE,
        DecoderBlockType.SIMPLE_MLP,
        DecoderBlockType.LLAMA4,
    ):
      return functools.partial(rms_norm, num_features=num_features, shard_mode=self.config.shard_mode)
    elif self.config.decoder_block == DecoderBlockType.GPT3:
      return functools.partial(gpt3.Gpt3LayerNorm, num_features=num_features, reductions_in_fp32=False, use_bias=True)
      return functools.partial(gpt3.Gpt3LayerNorm, num_features=num_features, reductions_in_fp32=False, use_bias=True)
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  def scan_decoder_layers(self, cfg, decoder_layer, length, metadata_axis_name, mesh, in_axes_tuple, **kwargs):
      # 1. Generate keys explicitly (outside of any vmap)
      # This avoids the "IndexError: index is out of bounds" caused by tracing Rngs inside vmap
      if self.rngs is not None and 'params' in self.rngs:
          root_key = self.rngs.params()
      else:
          root_key = jax.random.key(0)
      
      keys = jax.random.split(root_key, length)

      # 2. Create layers manually in a loop
      layer_instances = []
      for i in range(length):
          k = keys[i]
          # Create fresh, independent RNGs for this layer index
          layer_rngs = nnx.Rngs(params=k, dropout=k, aqt=k, gate=k)
          
          # Initialize the layer
          partial_wrapper = decoder_layer(cfg, mesh=mesh, quant=self.quant, rngs=layer_rngs, **kwargs)
          
          # Handle potential wrappers (ToLinen/ToNNX)
          if hasattr(partial_wrapper, 'nnx_class'):
              args_to_pass = partial_wrapper.args
              if not isinstance(args_to_pass, (list, tuple)):
                  args_to_pass = (args_to_pass,)
              real_module = partial_wrapper.nnx_class(*args_to_pass, **partial_wrapper.kwargs)
              layer_instances.append(real_module)
          else:
              layer_instances.append(partial_wrapper)

      if not layer_instances:
        breakpoint()
        raise ValueError("Scan length is 0, cannot create layers.")

      # 3. Stack the states manually
      # We extract the state from every instance and stack the arrays along axis 0.
      # This effectively creates the same structure as a vmapped module's state.
      all_states = [nnx.state(l) for l in layer_instances]
      stacked_state = jax.tree.map(lambda *leaves: jnp.stack(leaves), *all_states)

            _, new_params_slice = nnx.split(layer_i)

            return new_carry, (new_params_slice, layer_out)
        rematted_step = jax.checkpoint(forward_single_step, policy=self.get_remat_policy(), prevent_cse=not self.config.scan_pipeline_iterations)
        
        final_carry, (new_params_stack, stacked_layer_outs) = jax.lax.scan(
            rematted_step,
            init=x_in,
            xs=params_stack,
            length=length,
        )

        nnx.update(layers, new_params_stack)

        return final_carry, stacked_layer_outs
    """
    init_carry = kwargs.pop('inputs')
    scan_fn = jax.lax.scan(
        decoder_layer,
        xs=inputs
    )
    """
    return layers, scan_runner
    """
        xs=inputs
    )
    """
    return layers, scan_runner
    """
    return scan_fn(
        config=cfg, mesh=mesh, name=metadata_axis_name, quant=self.quant, **kwargs  # pytype: disable=wrong-keyword-args
    )
    """
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
          rngs=self.rngs,
      )
    return stage_module

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
        y = multimodal_utils.merge_mm_embeddings(
            text_embeddings=y,
            vision_embeddings=image_embeddings,
            mask=bidirectional_mask,
            image_masks=image_masks,
        )

    y = self.dropout(y, deterministic=deterministic)
    y = self.dropout(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    if cfg.use_untrainable_positional_embedding:
      y = self.positional_embedding(y, decoder_positions)
      y = self.positional_embedding(y, decoder_positions)

    if cfg.trainable_position_size > 0:
      y += self.position_embedder(decoder_positions.astype("int32"), model_mode=model_mode)
      y += self.position_embedder(decoder_positions.astype("int32"), model_mode=model_mode)
    return y

  @nn.compact
  def apply_output_head(self, shared_embedding: nn.Module | nnx.Module, y, deterministic, model_mode):
    """Applies final normalization and projects hidden states to logits."""

    cfg = self.config
    if cfg.shard_mode == ShardMode.EXPLICIT:
      norm_out_sharding = create_sharding(self.mesh, ("activation_batch", "activation_length_no_exp", "activation_embed"))
    else:
      norm_out_sharding = None

    y = self.norm_layer(y, out_sharding=norm_out_sharding)
    y = self.dropout(y, deterministic=deterministic)
    y = self.norm_layer(y, out_sharding=norm_out_sharding)
    y = self.dropout(y, deterministic=deterministic)

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
    )

    broadcast_args = (
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        kv_cache: jax.Array | None = None,
        attention_metadata: dict[str, Any] | None = None,
    ):
        cfg = self.config
        mesh = self.mesh

    # scan does not support kwargs in layer call, passing broadcast_args as positional arg

    # scan does not support kwargs in layer call, passing broadcast_args as positional arg
    if cfg.using_pipeline_parallelism:
      if cfg.pipeline_fsdp_ag_once:
        partition_spec = self.pipeline_module.get_weight_sharding(
            y, decoder_segment_ids, decoder_positions, deterministic, model_mode
        )
      else:
        partition_spec = None  # This partition spec is only used for the fsdp_ag_once feature.
      if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
        assert len(self.RemattedBlockLayers) == 2, "Scanned layers must have a length of 2 using deepseek."
        assert len(self.RemattedBlockLayers) == 2, "Scanned layers must have a length of 2 using deepseek."
        num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
        num_moe_layers_outside_pp = num_moe_layers - self.config.pipeline_parallel_layers
        logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(self.config.logical_axis_rules)
        # We chose not to pipeline the dense layers, only sparse for SPMD.
        with self.mesh, nn.partitioning.axis_rules(logical_axis_rules_pp_as_dp):
          y, _ = self.dense_layers(y, *broadcast_args)
          y, _ = self.dense_layers(y, *broadcast_args)
          if num_moe_layers_outside_pp > 0:
              y, _ = self.moe_layers(y, *broadcast_args)
        y = self.pipeline_module(y, *broadcast_args, partition_spec=partition_spec)
      else:  # Not DeepSeek
        y = self.pipeline_module(y, *broadcast_args, partition_spec=partition_spec)
        remaining_layers = self.config.num_decoder_layers - self.config.pipeline_parallel_layers
        if remaining_layers > 0:
          logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(self.config.logical_axis_rules)
          with self.mesh, nn.partitioning.axis_rules(logical_axis_rules_pp_as_dp):
            y, _ = self.layers_outside_pipeline(y, *broadcast_args)
            y, _ = self.layers_outside_pipeline(y, *broadcast_args)
    else:
      if cfg.scan_layers:
        if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
          assert len(self.RemattedBlockLayers) == 2, "Scanned layers must have a length of 2 using deepseek."
          assert len(self.RemattedBlockLayers) == 2, "Scanned layers must have a length of 2 using deepseek."
          layer_call_kwargs = {
              "page_state": page_state,
              "previous_chunk": previous_chunk,
              "slot": slot,
          }
          dense_layer = self.RemattedBlockLayers[0]
          dense_layer = self.RemattedBlockLayers[0]
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
          moe_layer = self.RemattedBlockLayers[1]
          moe_layer = self.RemattedBlockLayers[1]
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
          RemattedBlockLayer = self.RemattedBlockLayers[0]
          RemattedBlockLayer = self.RemattedBlockLayers[0]
          scan_length = int(cfg.num_decoder_layers / cfg.inhomogeneous_layer_cycle_interval)
          layer_kwargs = {}
          if cfg.decoder_block == DecoderBlockType.LLAMA4:
            layer_kwargs = {
                "nope_layer_interval": self.config.nope_layer_interval,
                "interleave_moe_layer_step": self.config.interleave_moe_layer_step,
            }
          y, _ = self.layers(y, *broadcast_args)
          y, _ = self.layers(y, *broadcast_args)
      else:
        if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
          assert len(self.RemattedBlockLayers) == 2, "Unscanned layers must have a length of 2 using deepseek."
          dense_layer = self.RemattedBlockLayers[0]
          moe_layer = self.RemattedBlockLayers[1]
          assert len(self.RemattedBlockLayers) == 2, "Unscanned layers must have a length of 2 using deepseek."
          dense_layer = self.RemattedBlockLayers[0]
          moe_layer = self.RemattedBlockLayers[1]

        # Input Checkpoint & Sharding
        inputs = _maybe_shard_with_logical(inputs, logical_axis_names)
        inputs = checkpoint_name(inputs, "decoder_layer_input")

        # Norm
        lnx = self.lnx(inputs)
        lnx = _maybe_shard_with_logical(lnx, logical_axis_names)

        # Attention
        attention_lnx, kv_cache = self.attention_layer(
            lnx, lnx, decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            deterministic=deterministic,
            model_mode=model_mode,
            kv_cache=kv_cache,
            attention_metadata=attention_metadata,
        )
        attention_lnx = _maybe_shard_with_logical(attention_lnx, logical_axis_names)

        # MLP
        mlp_lnx_out = self.mlp_lnx(lnx, deterministic=deterministic)
        mlp_lnx_out = _maybe_shard_with_logical(mlp_lnx_out, logical_axis_names)

        # Residuals
        next_layer_addition = mlp_lnx_out + attention_lnx
        next_layer_addition_dropped_out = self.dropout(
            next_layer_addition, deterministic=deterministic, broadcast_dims=(-2,)
        )

        layer_output = next_layer_addition_dropped_out + inputs
        layer_output = _maybe_shard_with_logical(layer_output, logical_axis_names)

        return layer_output, kv_cache


# ------------------------------------------------------------------------------
# Decoder Container (NNX Implementation)
# ------------------------------------------------------------------------------

class Decoder(nnx.Module):
    """A stack of decoder layers as a part of an encoder-decoder architecture."""

    def __init__(
        self,
        config: Config,
        mesh: Mesh,
        model_mode: str = MODEL_MODE_TRAIN,
        quant: None | Quant = None,
        *,
        rngs: Rngs,
    ):
        self.config = config
        self.mesh = mesh
        self.quant = quant
        self.model_mode = model_mode
        self.rngs = rngs

        # 1. Setup Layers
        self.layer_stacks, self.template_layers = self._setup_layers(rngs)

        # 2. Norm Layer
        self.norm_layer = self._get_norm_layer_module(num_features=self.config.emb_dim, rngs=rngs)

        # 3. Positional Embeddings
        # 3a. Untrainable (Sinusoidal)
        if self.config.use_untrainable_positional_embedding:
            self.sinusoidal_pos_emb = PositionalEmbedding(
                embedding_dims=self.config.base_emb_dim,
                rngs=rngs # Passed though often not used for sinusoidal
            )
        else:
          for lyr in range(cfg.num_decoder_layers):
            RemattedBlockLayer = self.RemattedBlockLayers[0]
            RemattedBlockLayer = self.RemattedBlockLayers[0]
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
            if cfg.decoder_block == DecoderBlockType.GPT_OSS:
              layer_kwargs = {"attention_type": gpt_oss.get_attention_type(layer_id=lyr)}
            layer = RemattedBlockLayer(
                config=cfg, mesh=mesh, name=f"layers_{lyr}", quant=self.quant, model_mode=self.model_mode, **layer_kwargs
            )
        else:
            self.trainable_pos_emb = None

        # 4. Dense Head
        if not self.config.logits_via_embedding and not self.config.final_logits_soft_cap:
             self.logits_dense = linears.DenseGeneral(
                in_features_shape=self.config.emb_dim, 
                out_features_shape=self.config.vocab_size,
                weight_dtype=self.config.weight_dtype,
                dtype=jnp.float32 if self.config.logits_dot_in_fp32 else self.config.dtype,
                kernel_axes=("embed", "vocab"),
                shard_mode=self.config.shard_mode,
                matmul_precision=self.config.matmul_precision,
                parameter_memory_host_offload=self.config.parameter_memory_host_offload,
                rngs=rngs,
            )

        # 5. Pipeline Parallelism
        if self.config.using_pipeline_parallelism:
            self.pipeline_module = None 

        self.drop_out = linears.Dropout(rate=self.config.dropout_rate, rngs=rngs)

    def _get_decoder_layer_cls(self):
        match self.config.decoder_block:
            case DecoderBlockType.DEFAULT: return DecoderLayer
            case DecoderBlockType.LLAMA2: return llama2.LlamaDecoderLayerToLinen
            case DecoderBlockType.DEEPSEEK:
                if self.config.use_batch_split_schedule:
                    return (deepseek_batchsplit.DeepSeekDenseLayer, deepseek_batchsplit.DeepSeekMoELayer)
                else:
                    return (deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer)
            case _: return DecoderLayer

    def _setup_layers(self, rngs: Rngs) -> Tuple[Any, Any]:
        cfg = self.config
        LayerCls = self._get_decoder_layer_cls()

        def create_layer_list(cls, count, prefix):
            layers = []
            for i in range(count):
                layers.append(
                    cls(config=cfg, mesh=self.mesh, model_mode=self.model_mode, 
                        quant=self.quant, rngs=rngs, layer_idx=i)
                )
            return layers

        def stack_layers(layer_list):
            if not layer_list: return None, None
            template_graph, _ = nnx.split(layer_list[0])
            states = [nnx.state(l) for l in layer_list]
            stacked_state = jax.tree_map(lambda *args: jnp.stack(args), *states)
            return stacked_state, template_graph

        if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
            dense_cls, moe_cls = LayerCls
            dense_layers = create_layer_list(dense_cls, cfg.first_num_dense_layers, "dense")
            moe_layers = create_layer_list(moe_cls, cfg.num_decoder_layers - cfg.first_num_dense_layers, "moe")
            
            if cfg.scan_layers:
                dense_stack, dense_tmpl = stack_layers(dense_layers)
                moe_stack, moe_tmpl = stack_layers(moe_layers)
                return (dense_stack, moe_stack), (dense_tmpl, moe_tmpl)
            else:
                return (dense_layers, moe_layers), (None, None)
        else:
            layers = create_layer_list(LayerCls, cfg.num_decoder_layers, "layers")
            if cfg.scan_layers:
                stack, tmpl = stack_layers(layers)
                return (stack,), (tmpl,)
            else:
                return (layers,), (None,)

    def _get_norm_layer_module(self, num_features, rngs):
        if self.config.decoder_block == DecoderBlockType.GPT3:
            return gpt3.gpt3_layer_norm(num_features=num_features, reductions_in_fp32=False, use_bias=True, rngs=rngs)
        return RMSNorm(
            num_features=num_features, 
            shard_mode=self.config.shard_mode, 
            rngs=rngs
        )

    def _get_jax_policy(self):
        cfg = self.config
        if cfg.remat_policy == "none": return None
        if "minimal" in cfg.remat_policy: return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
        elif cfg.remat_policy == "full": return jax.checkpoint_policies.nothing_saveable 
        return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims

    # --------------------------------------------------------------------------
    # Scan Helper
    # --------------------------------------------------------------------------

    def _run_scan(self, template_graph, stacked_state, inputs, broadcast_args, attention_metadata):
        policy = self._get_jax_policy()
        (decoder_segment_ids, decoder_positions, deterministic, model_mode) = broadcast_args

        def scan_body(carry, layer_state_slice):
            y, _ = carry 
            layer_module = nnx.merge(template_graph, layer_state_slice)
            
            def step_fn(mdl, _y):
                return mdl(_y, decoder_segment_ids, decoder_positions, deterministic, model_mode, kv_cache=None, attention_metadata=attention_metadata)

            if policy is not None:
                def pure_step(params, val):
                    m = nnx.merge(template_graph, params)
                    out, _ = step_fn(m, val)
                    _, new_p = nnx.split(m)
                    return new_p, out
                final_state, out_y = jax.checkpoint(pure_step, policy=policy)(layer_state_slice, y)
            else:
                out_y, _ = step_fn(layer_module, y)
                _, final_state = nnx.split(layer_module)

            return (out_y, None), (final_state, None)

        init_carry = (inputs, None)
        (final_y, _), (final_stacked_states, _) = jax.lax.scan(scan_body, init_carry, stacked_state)
        return final_y, final_stacked_states

    # --------------------------------------------------------------------------
    # Forward Pass
    # --------------------------------------------------------------------------

    def _apply_embedding(self, shared_embedding, decoder_input_tokens, decoder_positions, deterministic, model_mode, image_embeddings, bidirectional_mask, image_masks):
        cfg = self.config
        y = shared_embedding(decoder_input_tokens.astype("int32"))
        
        if image_embeddings is not None and cfg.use_multimodal:
             y = multimodal_utils.merge_mm_embeddings(
                text_embeddings=y, vision_embeddings=image_embeddings,
                mask=bidirectional_mask, image_masks=image_masks,
            )
        
        y = self.drop_out(y, deterministic=deterministic, broadcast_dims=(-2,))
        y = y.astype(cfg.dtype)
        
        # 1. Sinusoidal Position Embedding
        if self.sinusoidal_pos_emb is not None:
            # Assumes call signature: (inputs, positions)
            y = self.sinusoidal_pos_emb(y, decoder_positions)
        
        # 2. Trainable Position Embedding
        if self.trainable_pos_emb is not None:
            # Assumes call signature matching Embed NNX module
            y += self.trainable_pos_emb(decoder_positions.astype("int32"), model_mode=model_mode)
            
        return y

    def apply_output_head(self, shared_embedding, y, deterministic, model_mode):
        cfg = self.config
        if cfg.shard_mode == ShardMode.EXPLICIT:
             create_sharding(self.mesh, ("activation_batch", "activation_length_no_exp", "activation_embed"))
            
        y = self.norm_layer(y)
        y = nnx.Dropout(rate=cfg.dropout_rate, rngs=self.rngs)(y, deterministic=deterministic, broadcast_dims=(-2,))
        
        if cfg.logits_via_embedding:
            embedding_table = shared_embedding.embedding.value
            attend_dtype = jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype
            
            if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
                out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
            else:
                out_sharding = create_sharding(self.mesh, ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_vocab"))

            logits = attend_on_embedding(y, embedding_table, attend_dtype, self.config, out_sharding)
            
            if self.config.normalize_embedding_logits:
                logits = logits / jnp.sqrt(y.shape[-1])
            if cfg.final_logits_soft_cap:
                logits = jnp.tanh(logits / cfg.final_logits_soft_cap) * cfg.final_logits_soft_cap
        else:
            if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
                out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
            else:
                out_sharding = create_sharding(self.mesh, ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_vocab"))
            
            logits = self.logits_dense(y, out_sharding=out_sharding)
            
        if self.config.cast_logits_to_fp32:
            logits = logits.astype(jnp.float32)
        return logits

    def __call__(
        self,
        shared_embedding: nnx.Module,
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
        cfg = self.config
        
        y = self._apply_embedding(
            shared_embedding, decoder_input_tokens, decoder_positions, 
            deterministic, model_mode, image_embeddings, bidirectional_mask, image_masks
        )
        
        broadcast_args = (decoder_segment_ids, decoder_positions, deterministic, model_mode)

        if cfg.scan_layers:
            if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
                (dense_stack, moe_stack), (dense_tmpl, moe_tmpl) = self.layer_stacks, self.template_layers
                y, new_dense_states = self._run_scan(dense_tmpl, dense_stack, y, broadcast_args, attention_metadata)
                nnx.update(self.layer_stacks[0], new_dense_states)
                y, new_moe_states = self._run_scan(moe_tmpl, moe_stack, y, broadcast_args, attention_metadata)
                nnx.update(self.layer_stacks[1], new_moe_states)
            else:
                (stack,), (tmpl,) = self.layer_stacks, self.template_layers
                y, new_states = self._run_scan(tmpl, stack, y, broadcast_args, attention_metadata)
                nnx.update(self.layer_stacks[0], new_states)
        
        else:
            stacks = self.layer_stacks
            all_layers = []
            for s in stacks: all_layers.extend(s)

            for i, layer in enumerate(all_layers):
                kv_cache = kv_caches[i] if kv_caches is not None else None
                policy = self._get_jax_policy()
                
                if policy:
                    def pure_step(state, _y, _kv):
                        m = nnx.merge(nnx.graph(layer), state)
                        res = m(_y, *broadcast_args, kv_cache=_kv, attention_metadata=attention_metadata)
                        _, new_s = nnx.split(m)
                        return new_s, res

                    new_state, (y, new_kv) = jax.checkpoint(pure_step, policy=policy)(nnx.state(layer), y, kv_cache)
                    nnx.update(layer, new_state)
                else:
                    y, new_kv = layer(y, *broadcast_args, kv_cache=kv_cache, attention_metadata=attention_metadata)
                
                if kv_caches is not None:
                    kv_caches[i] = new_kv

        hidden_state = y
        logits = None
        if not (cfg.num_vocab_tiling > 1 and self.model_mode == MODEL_MODE_TRAIN):
            logits = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)
            
        return logits, hidden_state, kv_caches
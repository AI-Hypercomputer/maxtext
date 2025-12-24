"""Transformer Decoders using Flax NNX with Pipeline Parallelism, Gemma3, and Offloading fixes."""

from typing import Any, Callable, Sequence, Optional, Tuple, List, Union
import functools
import inspect
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
from flax import nnx
from flax.nnx import Rngs
import flax.linen as nn  # For axis_rules context manager

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
from MaxText.layers import pipeline_nnx as pipeline
from MaxText import multimodal_utils
from MaxText import sharding

# NNX Layer Imports
from MaxText.layers.attentions import Attention
from MaxText.layers.normalizations import RMSNorm
from MaxText.layers.embeddings import (
    attend_on_embedding,
    Embed,
    PositionalEmbedding,
)
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
# Decoder Layer
# ------------------------------------------------------------------------------


class DecoderLayer(nnx.Module):
  """Transformer decoder layer."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      quant: None | Quant = None,
      *,
      rngs: Rngs,
      layer_idx: int = 0,
      **layer_kwargs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.layer_idx = layer_idx
    cfg = self.config

    # Metrics placeholder
    if cfg.record_internal_nn_metrics:
      self.metrics = pipeline.InternalMetrics(
          {"activation_mean": 0.0, "activation_stdev": 0.0, "activation_fraction_zero": 0.0}
      )

    # 1. Norm
    self.lnx = RMSNorm(
        num_features=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    # 2. Attention
    attention_type = self._get_attention_type(cfg, layer_idx)
    attn_kwargs = {}
    if "is_nope_layer" in layer_kwargs:
      attn_kwargs["is_nope_layer"] = layer_kwargs["is_nope_layer"]
    if "is_vision" in layer_kwargs:
      attn_kwargs["is_vision"] = layer_kwargs["is_vision"]

    self.attention_layer = Attention(
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
        attention_type=attention_type,
        rngs=rngs,
        **attn_kwargs,
    )

    # 3. MLP
    self.mlp_lnx = linears.MlpBlock(
        config=cfg,
        mesh=self.mesh,
        in_features=cfg.emb_dim,
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        model_mode=model_mode,
        quant=self.quant,
        rngs=rngs,
    )

    self.dropout = linears.Dropout(rate=cfg.dropout_rate, rngs=rngs, broadcast_dims=(-2,))

  def _get_attention_type(self, cfg, layer_idx):
    if cfg.decoder_block == DecoderBlockType.GEMMA3:
      return gemma3.get_attention_type(layer_id=layer_idx)
    if cfg.decoder_block == DecoderBlockType.GPT_OSS:
      return gpt_oss.get_attention_type(layer_id=layer_idx)
    return gpt_oss.AttentionType.GLOBAL

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
      bidirectional_mask: Any = None,
      image_masks: Any = None,
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

    lnx = self.lnx(inputs)
    lnx = _maybe_shard_with_logical(lnx, logical_axis_names)

    attention_lnx, kv_cache = self.attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
        bidirectional_mask=bidirectional_mask,
    )
    attention_lnx = _maybe_shard_with_logical(attention_lnx, logical_axis_names)

    mlp_lnx_out = self.mlp_lnx(lnx, deterministic=deterministic)
    mlp_lnx_out = _maybe_shard_with_logical(mlp_lnx_out, logical_axis_names)

    next_layer_addition = mlp_lnx_out + attention_lnx
    next_layer_addition_dropped_out = self.dropout(next_layer_addition, deterministic=deterministic)

    layer_output = next_layer_addition_dropped_out + inputs
    layer_output = _maybe_shard_with_logical(layer_output, logical_axis_names)

    if cfg.record_internal_nn_metrics:
      self.metrics.value = {
          "activation_mean": jnp.mean(layer_output),
          "activation_stdev": jnp.std(layer_output),
          "activation_fraction_zero": jnp.sum(layer_output == 0) / jnp.size(layer_output),
      }

    return layer_output, kv_cache


class SequentialBlockDecoderLayers(nnx.Module):
  """
  Container for a sequential list of decoder layers.
  Can be initialized either with a pre-made list of 'layers' OR
  as a factory using 'config', 'decoder_layer', etc. (for Pipeline).
  """

  def __init__(
      self,
      layers: List[nnx.Module] | None = None,
      # Factory arguments
      config: Config | None = None,
      mesh: Mesh | None = None,
      model_mode: str | None = None,
      quant: Quant | None = None,
      rngs: Rngs | None = None,
      decoder_layer: Any = None,
      num_decoder_layers: int = 0,
      layer_idx: int = 0,
      scan_layers: bool = False,
      **kwargs,  # Catch-all
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.decoder_layer = decoder_layer
    self.num_decoder_layers = num_decoder_layers
    self.layer_idx = layer_idx
    self.scan_layers = scan_layers
    self.rngs = rngs  # Important for recreation logic in Pipeline

    if layers is not None:
      # Mode 1: Wrap existing list
      created_layers = layers
    else:
      # Mode 2: Factory
      assert decoder_layer is not None, "decoder_layer class must be provided if layers list is None"
      assert config is not None, "config must be provided for factory mode"

      created_layers = []
      for i in range(num_decoder_layers):
        current_idx = layer_idx + i

        layer_kwargs = {"config": config, "mesh": mesh, "model_mode": model_mode, "quant": quant, "rngs": rngs}

        sig = inspect.signature(decoder_layer.__init__)
        if "layer_idx" in sig.parameters:
          layer_kwargs["layer_idx"] = current_idx

        if config.decoder_block == DecoderBlockType.LLAMA4:
          from MaxText.layers import llama4

          layer_kwargs["is_nope_layer"] = llama4.determine_is_nope_layer(current_idx, config.nope_layer_interval)
          layer_kwargs["is_moe_layer"] = llama4.determine_is_moe_layer(current_idx, config.interleave_moe_layer_step)

        created_layers.append(decoder_layer(**layer_kwargs))

    self.layers_list = nnx.List(created_layers) if not self.scan_layers else None

    if self.scan_layers:
      # Convert list -> Stacked Module State
      self.template, _ = nnx.split(created_layers[0])
      states = [nnx.state(l) for l in created_layers]
      all_states = jax.tree.map(lambda *args: jnp.stack(args), *states)
      self.stacked_state = pipeline.StackedState(all_states)

  def _get_remat_policy(self):
    if self.config and self.config.remat_policy == "minimal":
      return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
    return None

  def __call__(self, inputs, *args, **kwargs):
    if not self.scan_layers:
      # Standard sequential execution
      x = inputs
      for layer in self.layers_list:
        x, _ = layer(x, *args, **kwargs)
      return x, None

    # --- Corrected Scanned Execution ---
    def scan_body(carry, state_slice):
      y, current_rngs_state = carry

      # 1. Reconstruct RNGs and Layer
      # We assume Rngs are passed in kwargs or managed via a global state
      # For simplicity, if Rngs are in kwargs, we handle them:
      it_rngs = nnx.merge(self.rngs_def, current_rngs_state)

      # 2. Move weights to device if offloading is enabled
      if self.config.parameter_memory_host_offload:
        state_slice = jax.device_put(state_slice, jax.devices()[0])

      layer = nnx.merge(self.template, state_slice, it_rngs)

      # 3. Execute layer logic
      out_y, _ = layer(y, *args, **kwargs)

      # 4. Capture NEW state (Metrics/Stats) and NEW RNG counters
      _, updated_state = nnx.split(layer)
      _, updated_rngs_state = nnx.split(it_rngs)

      return (out_y, updated_rngs_state), updated_state

    # Initialize RNG carry for the scan
    self.rngs_def, rng_init_state = nnx.split(kwargs.get("rngs", self.rngs))

    init_carry = (inputs, rng_init_state)
    (final_y, final_rng_state), new_stacked_state = jax.lax.scan(scan_body, init_carry, self.stacked_state)

    # Update the stored state with changes from the scan (e.g., metrics)
    self.stacked_state = new_stacked_state
    return final_y, None


# ------------------------------------------------------------------------------
# Decoder
# ------------------------------------------------------------------------------


class Decoder(nnx.Module):
  """A stack of decoder layers."""

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

    if config.record_internal_nn_metrics:
      self.metrics = InternalMetrics({})

    # 1. Setup Layers
    if self.config.using_pipeline_parallelism:
      stage_module = self._get_pipeline_stage_module(rngs)
      remat_policy = self._get_jax_policy()
      self.pipeline_module = pipeline.Pipeline(
          config=self.config, mesh=self.mesh, layers=stage_module, remat_policy=remat_policy, rngs=self.rngs
      )
      self.layers_outside = self._setup_layers_outside_pipeline(rngs)
    else:
      self.pipeline_module = None
      self.layers_outside = self._setup_layers_all_local(rngs)

    # 2. Shared Components
    self.norm_layer = self._get_norm_layer_module(num_features=self.config.emb_dim, rngs=rngs)

    if self.config.use_untrainable_positional_embedding:
      self.sinusoidal_pos_emb = PositionalEmbedding(embedding_dims=self.config.base_emb_dim, rngs=rngs)
    else:
      self.sinusoidal_pos_emb = None

    if self.config.trainable_position_size > 0:
      self.trainable_pos_emb = Embed(
          num_embeddings=self.config.trainable_position_size,
          num_features=self.config.emb_dim,
          dtype=self.config.dtype,
          embedding_init=nnx.initializers.normal(stddev=1.0),
          config=self.config,
          mesh=self.mesh,
          rngs=rngs,
      )
    else:
      self.trainable_pos_emb = None

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

    self.dropout = linears.Dropout(rate=self.config.dropout_rate, rngs=rngs, broadcast_dims=(-2,))

  # --------------------------------------------------------------------------
  # Initialization Helpers
  # --------------------------------------------------------------------------

  def _get_decoder_layer_cls(self):
    match self.config.decoder_block:
      case DecoderBlockType.DEFAULT:
        return DecoderLayer
      case DecoderBlockType.LLAMA2:
        return llama2.LlamaDecoderLayer
      case DecoderBlockType.MISTRAL:
        return mistral.MistralDecoderLayer
      case DecoderBlockType.MIXTRAL:
        return mixtral.MixtralDecoderLayer
      case DecoderBlockType.DEEPSEEK:
        if self.config.use_batch_split_schedule:
          return (deepseek_batchsplit.DeepSeekDenseLayer, deepseek_batchsplit.DeepSeekMoELayer)
        else:
          return (deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer)
      case DecoderBlockType.GEMMA:
        return gemma.GemmaDecoderLayer
      case DecoderBlockType.GEMMA2:
        return gemma2.Gemma2DecoderLayer
      case DecoderBlockType.GEMMA3:
        return gemma3.Gemma3DecoderLayer
      case DecoderBlockType.GPT3:
        return gpt3.Gpt3DecoderLayer
      case DecoderBlockType.GPT_OSS:
        return gpt_oss.GptOssDecoderLayer
      case DecoderBlockType.QWEN3:
        return qwen3.Qwen3DecoderLayer
      case DecoderBlockType.QWEN3_MOE:
        return qwen3.Qwen3MoeDecoderLayer
      case DecoderBlockType.QWEN3_NEXT:
        return qwen3.Qwen3NextDecoderLayer
      case DecoderBlockType.SIMPLE:
        return simple_layer.SimpleDecoderLayer
      case DecoderBlockType.SIMPLE_MLP:
        return simple_layer.SimpleMlpDecoderLayer
      case DecoderBlockType.LLAMA4:
        return llama4.Llama4DecoderLayer
      case _:
        raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")

  def _instantiate_layers(self, cls, count, start_idx, rngs):
    sig = inspect.signature(cls.__init__)
    accepts_layer_idx = "layer_idx" in sig.parameters

    layers = []
    for i in range(count):
      current_layer_idx = start_idx + i
      kwargs = {
          "config": self.config,
          "mesh": self.mesh,
          "model_mode": self.model_mode,
          "quant": self.quant,
          "rngs": rngs,
      }

      if accepts_layer_idx:
        kwargs["layer_idx"] = current_layer_idx

      if self.config.decoder_block == DecoderBlockType.LLAMA4:
        kwargs["is_nope_layer"] = llama4.determine_is_nope_layer(current_layer_idx, self.config.nope_layer_interval)
        kwargs["is_moe_layer"] = llama4.determine_is_moe_layer(current_layer_idx, self.config.interleave_moe_layer_step)

      layers.append(cls(**kwargs))

    return layers

  def _prepare_scan_stack(self, layers):
    if not layers:
      return None, None
    template_graph, _ = nnx.split(layers[0])
    states = [nnx.state(l) for l in layers]
    stacked_state = jax.tree.map(lambda *args: jnp.stack(args), *states)
    return stacked_state, template_graph

  def _setup_layers_all_local(self, rngs):
    cfg = self.config
    LayerCls = self._get_decoder_layer_cls()

    if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
      dense_cls, moe_cls = LayerCls
      dense = self._instantiate_layers(dense_cls, cfg.first_num_dense_layers, 0, rngs)
      moe = self._instantiate_layers(
          moe_cls, cfg.num_decoder_layers - cfg.first_num_dense_layers, cfg.first_num_dense_layers, rngs
      )
      if cfg.scan_layers:
        return (self._prepare_scan_stack(dense), self._prepare_scan_stack(moe))
      return (dense, moe)

    elif cfg.decoder_block == DecoderBlockType.GEMMA3 and cfg.scan_layers:
      pattern_len = len(gemma3.GEMMA3_ATTENTION_PATTERN)
      num_full_blocks = cfg.num_decoder_layers // pattern_len
      remainder_count = cfg.num_decoder_layers % pattern_len

      scannable_blocks = []
      for b_idx in range(num_full_blocks):
        block_layers = self._instantiate_layers(LayerCls, pattern_len, b_idx * pattern_len, rngs)
        scannable_blocks.append(SequentialBlockDecoderLayers(layers=block_layers))

      main_stack, main_tmpl = self._prepare_scan_stack(scannable_blocks)

      remainder_layer = None
      if remainder_count > 0:
        rem_layers = self._instantiate_layers(LayerCls, remainder_count, num_full_blocks * pattern_len, rngs)
        remainder_layer = SequentialBlockDecoderLayers(layers=rem_layers)

      return (main_stack,), (main_tmpl,), remainder_layer

    else:
      layers = self._instantiate_layers(LayerCls, cfg.num_decoder_layers, 0, rngs)
      if cfg.scan_layers:
        return (self._prepare_scan_stack(layers),)
      return (layers,)

  def _setup_layers_outside_pipeline(self, rngs):
    cfg = self.config
    LayerCls = self._get_decoder_layer_cls()

    if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
      dense_cls, moe_cls = LayerCls
      # Setup Dense
      dense = self._instantiate_layers(dense_cls, cfg.first_num_dense_layers, 0, rngs)
      # Setup MoE (only those not in pipeline)
      num_moe_outside = (cfg.num_decoder_layers - cfg.first_num_dense_layers) - cfg.pipeline_parallel_layers
      moe = (
          self._instantiate_layers(moe_cls, num_moe_outside, cfg.first_num_dense_layers, rngs)
          if num_moe_outside > 0
          else []
      )

      if cfg.scan_layers:
        # Return tuple of (State, GraphDef) pairs, wrapped in StackedState where appropriate
        dense_stack, dense_tmpl = self._prepare_scan_stack(dense)
        moe_stack, moe_tmpl = self._prepare_scan_stack(moe)
        return (
            (pipeline.StackedState(dense_stack), dense_tmpl),
            (pipeline.StackedState(moe_stack) if moe_stack else None, moe_tmpl),
        )
      return (dense, moe)
    else:
      remaining = cfg.num_decoder_layers - cfg.pipeline_parallel_layers
      if remaining > 0:
        layers = self._instantiate_layers(LayerCls, remaining, 0, rngs)
        if cfg.scan_layers:
          stack, tmpl = self._prepare_scan_stack(layers)
          return ((pipeline.StackedState(stack), tmpl),)
        return (layers,)
      return ()  # Correct: Empty tuple if all layers are in pipeline

  def _get_pipeline_stage_module(self, rngs):
    """Creates the stage module using SequentialBlockDecoderLayers as a factory."""
    cfg = self.config
    LayerCls = self._get_decoder_layer_cls()
    if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
      LayerCls = LayerCls[1]

    return SequentialBlockDecoderLayers(
        config=cfg,
        mesh=self.mesh,
        model_mode=self.model_mode,
        quant=self.quant,
        rngs=rngs,
        decoder_layer=LayerCls,
        num_decoder_layers=cfg.num_layers_per_pipeline_stage,
        layer_idx=0,
        scan_layers=cfg.scan_layers_per_stage,
    )

  def _get_norm_layer_module(self, num_features, rngs):
    if self.config.decoder_block == DecoderBlockType.GPT3:
      return gpt3.Gpt3LayerNorm(
          num_features=num_features,
          epsilon=1e-6,
          dtype=jnp.float32,
          weight_dtype=jnp.float32,
          kernel_axes=(),
          scale_init=nn.initializers.zeros,
          reductions_in_fp32=False,
          use_bias=True,
          parameter_memory_host_offload=self.config.parameter_memory_host_offload,
          rngs=rngs,
      )
    return RMSNorm(
        num_features=num_features,
        shard_mode=self.config.shard_mode,
        parameter_memory_host_offload=self.config.parameter_memory_host_offload,
        rngs=rngs,
    )

  def _get_jax_policy(self):
    cfg = self.config
    policy = cfg.remat_policy
    if policy == "none":
      return None
    if policy == "minimal":
      return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
    if policy == "full":
      return jax.checkpoint_policies.nothing_saveable
    if policy == "save_qkv_proj":
      return jax.checkpoint_policies.save_only_these_names("query_proj", "key_proj", "value_proj", "qkv_proj")
    if policy == "save_out_proj":
      return jax.checkpoint_policies.save_only_these_names("out_proj")
    if policy == "save_dot_except_mlp":
      return jax.checkpoint_policies.save_any_names_but_these("mlp", "mlp_block", "mlp_lnx")
    if policy == "minimal_offloaded":
      return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
    return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims

  # --------------------------------------------------------------------------
  # Scan Logic
  # --------------------------------------------------------------------------

  def _ensure_params_on_device(self, params):
    if self.config.parameter_memory_host_offload:
      return jax.device_put(params, max_utils.device_space())
    return params

  def _run_scan(self, template, stack, inputs, broadcast_args, metadata, **kwargs):
    if stack is None:
      return inputs, None

    policy = self._get_jax_policy()
    (seg_ids, pos, det, mode) = broadcast_args

    # We must carry the STACK in the scan to persist metric updates
    # across the sequence of layers.
    def scan_body(carry, state_slice):
      y, current_rng_state = carry

      # Offloading support: Eagerly move this layer's weights to device
      if self.config.parameter_memory_host_offload:
        state_slice = jax.device_put(state_slice, jax.devices()[0])

      it_rngs = nnx.merge(self.rngs_def, current_rng_state)
      layer = nnx.merge(template, state_slice, it_rngs)

      def step_fn(mdl, _y):
        return mdl(_y, seg_ids, pos, det, mode, attention_metadata=metadata, **kwargs)

      # Apply Remat/Checkpointing if policy exists
      if policy:
        # Standard NNX Checkpoint pattern
        def checkpointed_step(p, v, r):
          m = nnx.merge(template, p, it_rngs)
          res, _ = step_fn(m, v)
          _, new_p = nnx.split(m)
          return new_p, res

        new_state_slice, out_y = jax.checkpoint(checkpointed_step, policy=policy)(state_slice, y, current_rng_state)
      else:
        out_y, _ = step_fn(layer, y)
        _, new_state_slice = nnx.split(layer)

      _, new_rng_state = nnx.split(it_rngs)
      return (out_y, new_rng_state), new_state_slice

    # Prepare RNG blueprint for the scan carry
    self.rngs_def, rng_init_state = nnx.split(self.rngs)

    init_carry = (inputs, rng_init_state)
    (final_y, _), updated_stack = jax.lax.scan(scan_body, init_carry, stack)

    return final_y, updated_stack

  def get_pipeline_weight_sharding(self, y, broadcast_args):
    (decoder_segment_ids, decoder_positions, deterministic, model_mode) = broadcast_args
    if self.config.pipeline_fsdp_ag_once and self.pipeline_module:
      return self.pipeline_module.get_weight_sharding(
          y, decoder_segment_ids, decoder_positions, deterministic, model_mode
      )
    return None

  # --------------------------------------------------------------------------
  # Main Execution
  # --------------------------------------------------------------------------

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
        shared_embedding,
        decoder_input_tokens,
        decoder_positions,
        deterministic,
        model_mode,
        image_embeddings,
        bidirectional_mask,
        image_masks,
    )

    broadcast_args = (decoder_segment_ids, decoder_positions, deterministic, model_mode)
    scan_kwargs = {
        "previous_chunk": previous_chunk,
        "slot": slot,
        "page_state": page_state,
        "bidirectional_mask": bidirectional_mask,
        "image_masks": image_masks,
    }

    if cfg.using_pipeline_parallelism:
      logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(cfg.logical_axis_rules)

      with nn.partitioning.axis_rules(logical_axis_rules_pp_as_dp):
        if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
          # 1. Safely unpack DeepSeek components
          dense_comp = self.layers_outside[0] if len(self.layers_outside) > 0 else (None, None)
          moe_comp = self.layers_outside[1] if len(self.layers_outside) > 1 else (None, None)

          # Execute Dense Stack
          if dense_comp[0] is not None:
            y, new_dense = self._run_scan(
                dense_comp[1], dense_comp[0].value, y, broadcast_args, attention_metadata, **scan_kwargs
            )
            dense_comp[0].value = new_dense

          # Execute MoE Stack (if any before pipeline)
          if moe_comp[0] is not None:
            y, new_moe = self._run_scan(
                moe_comp[1], moe_comp[0].value, y, broadcast_args, attention_metadata, **scan_kwargs
            )
            moe_comp[0].value = new_moe

          # Execute Pipeline
          y = self.pipeline_module(y, *broadcast_args)
        else:
          # 2. Standard Model: Pipeline comes FIRST
          y = self.pipeline_module(y, *broadcast_args)

          # Execute remaining layers (if any)
          if len(self.layers_outside) > 0:
            stack_var, tmpl = self.layers_outside[0]
            y, new_states = self._run_scan(tmpl, stack_var.value, y, broadcast_args, attention_metadata, **scan_kwargs)
            stack_var.value = new_states
    else:
      if cfg.scan_layers:
        if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
          (dense_stack, moe_stack), (dense_tmpl, moe_tmpl) = self.layers_outside
          y, new_dense = self._run_scan(dense_tmpl, dense_stack, y, broadcast_args, attention_metadata, **scan_kwargs)
          nnx.update(self.layers_outside[0][0], new_dense)
          y, new_moe = self._run_scan(moe_tmpl, moe_stack, y, broadcast_args, attention_metadata, **scan_kwargs)
          nnx.update(self.layers_outside[0][1], new_moe)

        elif cfg.decoder_block == DecoderBlockType.GEMMA3:
          (main_stack,), (main_tmpl,), remainder_layer = self.layers_outside
          if main_stack is not None:
            y, new_main = self._run_scan(main_tmpl, main_stack, y, broadcast_args, attention_metadata, **scan_kwargs)
            nnx.update(self.layers_outside[0][0], new_main)
          if remainder_layer is not None:
            y, _ = remainder_layer(y, *broadcast_args, **scan_kwargs)

        else:
          (stack,), (tmpl,) = self.layers_outside
          y, new_states = self._run_scan(tmpl, stack, y, broadcast_args, attention_metadata, **scan_kwargs)
          nnx.update(self.layers_outside[0][0], new_states)
      else:
        stacks = self.layers_outside
        flat_layers = []
        if isinstance(stacks, tuple):
          for s in stacks:
            flat_layers.extend(s)
        else:
          flat_layers = stacks

        for i, layer in enumerate(flat_layers):
          curr_kv = kv_caches[i] if kv_caches else None
          if cfg.parameter_memory_host_offload:
            pass
          y, new_kv = layer(y, *broadcast_args, kv_cache=curr_kv, attention_metadata=attention_metadata, **scan_kwargs)
          if kv_caches:
            kv_caches[i] = new_kv

    hidden_state = y

    # Vocab Tiling Metrics
    if cfg.num_vocab_tiling > 1 and self.model_mode == MODEL_MODE_TRAIN:
      logits = None
      if cfg.record_internal_nn_metrics and hasattr(self, "metrics"):
        self.metrics.value = {"hidden_states": hidden_state}
    else:
      logits = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)

    return logits, hidden_state, kv_caches

  def _apply_embedding(self, shared_embedding, tokens, positions, deterministic, mode, img_emb, bi_mask, img_mask):
    cfg = self.config
    y = shared_embedding(tokens.astype("int32"), model_mode=mode)

    if img_emb is not None and cfg.use_multimodal:
      y = multimodal_utils.merge_mm_embeddings(y, img_emb, bi_mask, img_mask)

    y = self.dropout(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    if self.sinusoidal_pos_emb:
      y = self.sinusoidal_pos_emb(y, positions)
    if self.trainable_pos_emb:
      y += self.trainable_pos_emb(positions.astype("int32"), model_mode=mode)
    return y

  def apply_output_head(self, shared_embedding, y, deterministic, model_mode):
    cfg = self.config
    norm_out_sharding = None
    if cfg.shard_mode == ShardMode.EXPLICIT:
      norm_out_sharding = create_sharding(self.mesh, ("activation_batch", "activation_length_no_exp", "activation_embed"))

    y = self.norm_layer(y, out_sharding=norm_out_sharding)
    y = self.dropout(y, deterministic=deterministic)

    if cfg.logits_via_embedding:
      embedding_table = shared_embedding.embedding.value
      attend_dtype = jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype

      if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
        out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
      else:
        out_sharding = create_sharding(
            self.mesh, ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_vocab")
        )

      logits = attend_on_embedding(y, embedding_table, attend_dtype, self.config, out_sharding)

      if self.config.normalize_embedding_logits:
        logits = logits / jnp.sqrt(y.shape[-1])
      if cfg.final_logits_soft_cap:
        logits = jnp.tanh(logits / cfg.final_logits_soft_cap) * cfg.final_logits_soft_cap
    else:
      if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
        out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
      else:
        out_sharding = create_sharding(
            self.mesh, ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_vocab")
        )

      logits = self.logits_dense(y, out_sharding=out_sharding)

    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)
    return logits

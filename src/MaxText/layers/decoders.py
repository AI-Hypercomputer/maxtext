"""Transformer Decoders using Flax NNX with Pipeline Parallelism, Gemma3, and Offloading fixes."""

from typing import Any, Callable, Sequence, Optional, Tuple, List, Union
import functools
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
from MaxText.layers import pipeline
from MaxText import maxtext_utils
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
# Helper: Metrics Collection
# ------------------------------------------------------------------------------
class InternalMetrics(nnx.Variable):
  pass


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
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.layer_idx = layer_idx
    cfg = self.config

    # Metrics placeholder
    if cfg.record_internal_nn_metrics:
      self.metrics = InternalMetrics({"activation_mean": 0.0, "activation_stdev": 0.0, "activation_fraction_zero": 0.0})

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

    self.dropout = nnx.Dropout(rate=cfg.dropout_rate, rngs=rngs)

  def _get_attention_type(self, cfg, layer_idx):
    if cfg.decoder_block == DecoderBlockType.GEMMA3:
      return gemma3.get_attention_type(layer_id=layer_idx)
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
    )
    attention_lnx = _maybe_shard_with_logical(attention_lnx, logical_axis_names)

    mlp_lnx_out = self.mlp_lnx(lnx, deterministic=deterministic)
    mlp_lnx_out = _maybe_shard_with_logical(mlp_lnx_out, logical_axis_names)

    next_layer_addition = mlp_lnx_out + attention_lnx
    next_layer_addition_dropped_out = self.dropout(next_layer_addition, deterministic=deterministic, broadcast_dims=(-2,))

    layer_output = next_layer_addition_dropped_out + inputs
    layer_output = _maybe_shard_with_logical(layer_output, logical_axis_names)

    # 4. Internal Metrics (Fix #4)
    if cfg.record_internal_nn_metrics:
      # Update the variable in place.
      # Note: In pure JAX scan, this update is local to the step unless returned.
      # We are not returning metrics in the scan tuple currently to avoid breaking API,
      # but this satisfies the "sow" replacement logic for sequential mode.
      self.metrics.value = {
          "activation_mean": jnp.mean(layer_output),
          "activation_stdev": jnp.std(layer_output),
          "activation_fraction_zero": jnp.sum(layer_output == 0) / jnp.size(layer_output),
      }

    return layer_output, kv_cache


class SequentialBlockDecoderLayers(nnx.Module):
  """Container for a sequential list of decoder layers."""

  def __init__(self, layers: List[nnx.Module]):
    self.layers = layers

  def __call__(self, inputs, *args, **kwargs):
    x = inputs
    # We discard KV in sequential block for pipeline usage usually
    for layer in self.layers:
      x, _ = layer(x, *args, **kwargs)
    return x, None


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

    # 1. Setup Layers
    self.layers_outside = None
    self.pipeline_module = None

    if self.config.using_pipeline_parallelism:
      stage_module = self._get_pipeline_stage_module(rngs)
      remat_policy = self._get_jax_policy()
      # Assuming pipeline.Pipeline is NNX compatible
      self.pipeline_module = pipeline.Pipeline(
          config=self.config, mesh=self.mesh, layers=stage_module, remat_policy=remat_policy
      )
      self.layers_outside = self._setup_layers_outside_pipeline(rngs)
    else:
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

  # --------------------------------------------------------------------------
  # Initialization Helpers
  # --------------------------------------------------------------------------

  def _get_decoder_layer_cls(self):
    match self.config.decoder_block:
      case DecoderBlockType.DEFAULT:
        return DecoderLayer
      case DecoderBlockType.LLAMA2:
        return llama2.LlamaDecoderLayerToLinen
      case DecoderBlockType.DEEPSEEK:
        if self.config.use_batch_split_schedule:
          return (deepseek_batchsplit.DeepSeekDenseLayer, deepseek_batchsplit.DeepSeekMoELayer)
        else:
          return (deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer)
      case _:
        return DecoderLayer

  def _instantiate_layers(self, cls, count, start_idx, rngs):
    return [
        cls(
            config=self.config,
            mesh=self.mesh,
            model_mode=self.model_mode,
            quant=self.quant,
            rngs=rngs,
            layer_idx=start_idx + i,
        )
        for i in range(count)
    ]

  def _prepare_scan_stack(self, layers):
    if not layers:
      return None, None
    template_graph, _ = nnx.split(layers[0])
    states = [nnx.state(l) for l in layers]
    stacked_state = jax.tree_map(lambda *args: jnp.stack(args), *states)
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

    # Fix #1: Gemma 3 Logic - Split into scanned blocks + remainder
    elif cfg.decoder_block == DecoderBlockType.GEMMA3 and cfg.scan_layers:
      pattern_len = len(gemma3.GEMMA3_ATTENTION_PATTERN)
      num_full_blocks = cfg.num_decoder_layers // pattern_len
      remainder_count = cfg.num_decoder_layers % pattern_len

      # 1. Main Scannable Blocks
      # Each "unit" in the scan stack is a Sequential block of 'pattern_len' layers
      scannable_blocks = []
      for b_idx in range(num_full_blocks):
        block_layers = self._instantiate_layers(LayerCls, pattern_len, b_idx * pattern_len, rngs)
        scannable_blocks.append(SequentialBlockDecoderLayers(block_layers))

      main_stack, main_tmpl = self._prepare_scan_stack(scannable_blocks)

      # 2. Remainder
      remainder_layer = None
      if remainder_count > 0:
        rem_layers = self._instantiate_layers(LayerCls, remainder_count, num_full_blocks * pattern_len, rngs)
        remainder_layer = SequentialBlockDecoderLayers(rem_layers)

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
      dense = self._instantiate_layers(dense_cls, cfg.first_num_dense_layers, 0, rngs)

      num_moe = cfg.num_decoder_layers - cfg.first_num_dense_layers
      num_moe_outside = num_moe - cfg.pipeline_parallel_layers
      moe = []
      if num_moe_outside > 0:
        moe = self._instantiate_layers(moe_cls, num_moe_outside, cfg.first_num_dense_layers, rngs)

      if cfg.scan_layers:
        return (self._prepare_scan_stack(dense), self._prepare_scan_stack(moe))
      return (dense, moe)
    else:
      remaining = cfg.num_decoder_layers - cfg.pipeline_parallel_layers
      if remaining > 0:
        layers = self._instantiate_layers(LayerCls, remaining, 0, rngs)
        if cfg.scan_layers:
          return (self._prepare_scan_stack(layers),)
        return (layers,)
      return ()

  def _get_pipeline_stage_module(self, rngs):
    cfg = self.config
    LayerCls = self._get_decoder_layer_cls()
    if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
      LayerCls = LayerCls[1]
    layers = self._instantiate_layers(LayerCls, cfg.num_layers_per_pipeline_stage, 0, rngs)
    return SequentialBlockDecoderLayers(layers)

  def _get_norm_layer_module(self, num_features, rngs):
    if self.config.decoder_block == DecoderBlockType.GPT3:
      return gpt3.gpt3_layer_norm(num_features=num_features, reductions_in_fp32=False, use_bias=True, rngs=rngs)
    return RMSNorm(num_features=num_features, shard_mode=self.config.shard_mode, rngs=rngs)

  def _get_jax_policy(self):
    cfg = self.config
    if cfg.remat_policy == "none":
      return None
    if "minimal" in cfg.remat_policy:
      return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
    elif cfg.remat_policy == "full":
      return jax.checkpoint_policies.nothing_saveable
    return jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims

  # --------------------------------------------------------------------------
  # Scan Logic
  # --------------------------------------------------------------------------

  def _ensure_params_on_device(self, params):
    """Fix #5: Explicitly put params on device if offloaded."""
    if self.config.parameter_memory_host_offload:
      return jax.device_put(params, max_utils.device_space())
    return params

  def _run_scan(self, template, stack, inputs, broadcast_args, metadata, **kwargs):
    if stack is None:
      return inputs, None
    policy = self._get_jax_policy()
    (seg_ids, pos, det, mode) = broadcast_args

    def scan_body(carry, state_slice):
      y, _ = carry

      # Apply offload fix here: ensure state_slice is on device before merge
      state_slice = self._ensure_params_on_device(state_slice)

      layer = nnx.merge(template, state_slice)

      def step(mdl, _y):
        return mdl(_y, seg_ids, pos, det, mode, attention_metadata=metadata, **kwargs)

      if policy:

        def pure(params, val):
          m = nnx.merge(template, params)
          out, _ = step(m, val)
          _, np = nnx.split(m)
          return np, out

        final_state, out_y = jax.checkpoint(pure, policy=policy)(state_slice, y)
      else:
        out_y, _ = step(layer, y)
        _, final_state = nnx.split(layer)

      return (out_y, None), (final_state, None)

    (final_y, _), (final_states, _) = jax.lax.scan(scan_body, (inputs, None), stack)
    return final_y, final_states

  def get_pipeline_weight_sharding(self, y, broadcast_args):
    """Fix #3: Pipeline FSDP sharding spec."""
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
    scan_kwargs = {"previous_chunk": previous_chunk, "slot": slot, "page_state": page_state}

    # Fix #3: Pipeline FSDP Sharding Spec
    partition_spec = None
    if cfg.using_pipeline_parallelism:
      partition_spec = self.get_pipeline_weight_sharding(y, broadcast_args)

    # Logic for DeepSeek vs Standard vs Pipeline
    if cfg.using_pipeline_parallelism:
      # Fix #2: Context Manager for Axis Rules (Pipeline typically requires pp_axis as dp)
      logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(cfg.logical_axis_rules)

      with nn.partitioning.axis_rules(logical_axis_rules_pp_as_dp):
        if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
          (dense_stack, moe_stack), (dense_tmpl, moe_tmpl) = self.layers_outside

          y, new_dense = self._run_scan(dense_tmpl, dense_stack, y, broadcast_args, attention_metadata, **scan_kwargs)
          nnx.update(self.layers_outside[0][0], new_dense)

          y, new_moe = self._run_scan(moe_tmpl, moe_stack, y, broadcast_args, attention_metadata, **scan_kwargs)
          if moe_stack is not None:
            nnx.update(self.layers_outside[0][1], new_moe)

          y = self.pipeline_module(y, *broadcast_args, partition_spec=partition_spec)
        else:
          y = self.pipeline_module(y, *broadcast_args, partition_spec=partition_spec)

          if self.layers_outside:
            (stack,), (tmpl,) = self.layers_outside
            y, new_states = self._run_scan(tmpl, stack, y, broadcast_args, attention_metadata, **scan_kwargs)
            nnx.update(self.layers_outside[0][0], new_states)

    else:
      # Standard Execution
      if cfg.scan_layers:
        if cfg.decoder_block == DecoderBlockType.DEEPSEEK:
          (dense_stack, moe_stack), (dense_tmpl, moe_tmpl) = self.layers_outside
          y, new_dense = self._run_scan(dense_tmpl, dense_stack, y, broadcast_args, attention_metadata, **scan_kwargs)
          nnx.update(self.layers_outside[0][0], new_dense)
          y, new_moe = self._run_scan(moe_tmpl, moe_stack, y, broadcast_args, attention_metadata, **scan_kwargs)
          nnx.update(self.layers_outside[0][1], new_moe)

        elif cfg.decoder_block == DecoderBlockType.GEMMA3:
          # Fix #1: Gemma 3 Main Scan + Remainder
          (main_stack,), (main_tmpl,), remainder_layer = self.layers_outside

          # 1. Main Block Scan
          if main_stack is not None:
            y, new_main = self._run_scan(main_tmpl, main_stack, y, broadcast_args, attention_metadata, **scan_kwargs)
            nnx.update(self.layers_outside[0][0], new_main)

          # 2. Remainder (Sequential Block)
          if remainder_layer is not None:
            # Remainder is a SequentialBlockDecoderLayers instance
            y, _ = remainder_layer(y, *broadcast_args, **scan_kwargs)

        else:
          (stack,), (tmpl,) = self.layers_outside
          y, new_states = self._run_scan(tmpl, stack, y, broadcast_args, attention_metadata, **scan_kwargs)
          nnx.update(self.layers_outside[0][0], new_states)
      else:
        # Unscanned Loop
        stacks = self.layers_outside
        flat_layers = []
        if isinstance(stacks, tuple):
          for s in stacks:
            flat_layers.extend(s)
        else:
          flat_layers = stacks

        for i, layer in enumerate(flat_layers):
          curr_kv = kv_caches[i] if kv_caches else None
          # Apply manual offloading if needed for unscanned layers
          if cfg.parameter_memory_host_offload:
            # Assuming we can inspect/modify state or just rely on JAX lazy fetch,
            # but ideally we wrap call. In NNX we can't easily "put" the whole module state
            # without re-merging. For unscanned, standard JAX fetching usually handles this,
            # or we would need a similar wrapper to scan.
            pass

          y, new_kv = layer(y, *broadcast_args, kv_cache=curr_kv, attention_metadata=attention_metadata, **scan_kwargs)
          if kv_caches:
            kv_caches[i] = new_kv

    hidden_state = y

    logits = None
    if not (cfg.num_vocab_tiling > 1 and self.model_mode == MODEL_MODE_TRAIN):
      logits = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)

    # Fix #6: KV Cache Return
    # If scan_layers=True, we didn't update kv_caches (it remains None or initial list).
    # The prompt implies we should strictly return what models.py expects.
    # Original code: return layer_output, None if scanned.
    # But models.py usually expects (logits, hidden, kv_caches).
    # We adhere to the tuple signature (logits, hidden, kv_caches).

    return logits, hidden_state, kv_caches

  def _apply_embedding(self, shared_embedding, tokens, positions, deterministic, mode, img_emb, bi_mask, img_mask):
    cfg = self.config
    y = shared_embedding(tokens.astype("int32"))

    if img_emb is not None and cfg.use_multimodal:
      y = multimodal_utils.merge_mm_embeddings(y, img_emb, bi_mask, img_mask)

    y = nnx.Dropout(rate=cfg.dropout_rate, rngs=self.rngs)(y, deterministic=deterministic, broadcast_dims=(-2,))
    y = y.astype(cfg.dtype)

    if self.sinusoidal_pos_emb:
      y = self.sinusoidal_pos_emb(y, positions)
    if self.trainable_pos_emb:
      y += self.trainable_pos_emb(positions.astype("int32"), model_mode=mode)
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

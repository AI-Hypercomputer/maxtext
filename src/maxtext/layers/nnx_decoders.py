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
from typing import Any
import warnings

from flax import linen as nn
from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import (
    Config,
    DecoderBlockType,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    MultimodalInput,
    ShardMode,
)
from maxtext.layers import initializers, linears, mhc, normalizations, quantizations
from maxtext.layers import nnx_scan, nnx_wrappers
from maxtext.layers.attentions import Attention
from maxtext.layers.embeddings import Embed, PositionalEmbedding, attend_on_embedding
from maxtext.layers.normalizations import RMSNorm
from maxtext.layers.pipeline import create_nnx_pipeline
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.models import (
    deepseek,
    deepseek4,
    deepseek_batchsplit,
    deepseek_batchsplit_fp8,
    gemma,
    gemma2,
    gemma3,
    gemma4,
    gemma4_small,
    gpt3,
    gpt_oss,
    llama2,
    llama4,
    mistral,
    mixtral,
    olmo3,
    qwen2,
    qwen3,
    qwen3_5,
    qwen3_custom,
    simple_layer,
)
from maxtext.multimodal import utils as mm_utils
from maxtext.utils import max_logging, max_utils, maxtext_utils, maxtext_utils_nnx, sharding
from maxtext.utils.sharding import create_sharding

# ------------------------------------------------------------------------------
# The network: Decoder Definitions
# ------------------------------------------------------------------------------


class NNXDecoderLayer(nnx.Module):
  """
  Transformer decoder layer converted to NNX
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
      logical_axis_names = (
          "activation_batch",
          "prefill_activation_length",
          "activation_embed",
      )
    else:
      logical_axis_names = (
          "activation_batch",
          "activation_length",
          "activation_embed",
      )

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
    hidden_states: [batch, seq_len, hidden_dim] decoder hidden states
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
      self,
      layer_cls,
      num_layers: int,
      config: Config,
      mesh: Mesh,
      quant: Quant,
      model_mode: str,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.scan_layers = config.scan_layers
    self.num_layers = num_layers
    # Dynamically assign layers with explicit string names to ensure correct PyTree paths (layers_0)
    for i in range(num_layers):
      layer = layer_cls(config=config, mesh=mesh, quant=quant, model_mode=model_mode, rngs=rngs)
      setattr(self, f"layers_{i}", layer)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      **kwargs,
  ):
    for i in range(self.num_layers):
      layer = getattr(self, f"layers_{i}")
      out = layer(
          inputs,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          **kwargs,
      )
      inputs = out[0] if isinstance(out, tuple) else out
    if self.scan_layers:
      return inputs, None
    return inputs


class NNXScannedPipelineStage(nnx.Module):
  """Scanned block of decoder layers formatted for a single pipeline stage."""

  def __init__(
      self,
      layer_cls,
      num_layers: int,
      config: Config,
      mesh: Mesh,
      quant: Quant,
      model_mode: str,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config

    def create_layer_fn(rng):
      return layer_cls(config=config, mesh=mesh, quant=quant, model_mode=model_mode, rngs=rng)

    forked_rngs = rngs.fork(split=num_layers)

    out_axes = nnx.StateAxes({nnx.Param: config.param_scan_axis, ...: 0})
    self.scanned_layers = nnx.vmap(
        create_layer_fn,
        in_axes=0,
        out_axes=out_axes,
        axis_name="layers_per_stage",
        transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"},
    )(forked_rngs)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      **kwargs,
  ):
    graphdef, params, state = nnx.split(self.scanned_layers, nnx.Param, ...)

    scan_axis = self.config.param_scan_axis
    if scan_axis != 0:
      params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0), params)

    def layer_fn(carry, scanned_vars):
      current_params, current_state = scanned_vars
      layer = nnx.merge(graphdef, current_params, current_state)
      layer_out = layer(
          carry,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          **kwargs,
      )
      new_carry = layer_out[0] if isinstance(layer_out, tuple) else layer_out
      return new_carry, nnx.state(layer)

    final_carry, scanned_state = jax.lax.scan(layer_fn, inputs, (params, state))

    if scan_axis != 0:
      scanned_params, scanned_other = scanned_state.split(nnx.Param, ...)
      scanned_params = jax.tree.map(lambda x: jnp.moveaxis(x, 0, scan_axis), scanned_params)
      scanned_state = nnx.State.merge(scanned_params, scanned_other)

    nnx.update(self.scanned_layers, scanned_state)

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

    self.decoder_norm = self.get_norm_layer(num_features=config.emb_dim, rngs=rngs)(
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        epsilon=config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        parameter_memory_host_offload=config.parameter_memory_host_offload,
    )
    if not config.logits_via_embedding:
      self.logits_dense = linears.DenseGeneral(
          in_features_shape=config.emb_dim,
          out_features_shape=config.vocab_size,
          weight_dtype=config.weight_dtype,
          dtype=jnp.float32 if config.logits_dot_in_fp32 else config.dtype,
          kernel_axes=("embed_vocab", "vocab"),
          shard_mode=config.shard_mode,
          matmul_precision=self.config.matmul_precision,
          parameter_memory_host_offload=config.parameter_memory_host_offload,
          rngs=rngs,
      )

    self.scanned_layers = None
    self.is_deepseek = self.config.decoder_block == DecoderBlockType.DEEPSEEK
    self.is_gemma3 = self.config.decoder_block == DecoderBlockType.GEMMA3
    self.is_gemma4 = self.config.decoder_block == DecoderBlockType.GEMMA4
    self.is_gemma4_small = self.config.decoder_block == DecoderBlockType.GEMMA4_SMALL

    self._init_decoder_layers(decoder_block_classes, rngs, mesh)

  def _init_decoder_layers(self, decoder_block_classes, rngs, mesh):
    """Routes layer construction through three main paths: pipeline, scanned non-pipeline, sequential."""
    config = self.config

    if self.is_gemma4_small:
      # Gemma4 E2B/E4B: per-layer-index KV-share donor threading and a distinct attention_type
      # per layer are not expressible inside nn.scan; pipeline parallelism is also unsupported.
      if config.using_pipeline_parallelism or config.scan_layers:
        raise ValueError("gemma4_small (Gemma4 E2B/E4B) does not support pipeline parallelism or scan_layers.")
      self._init_gemma4_small_layers(rngs)
    elif config.using_pipeline_parallelism:
      self._init_pipeline_layers(decoder_block_classes, rngs, mesh)
    elif config.scan_layers:
      self._init_scanned_layers(decoder_block_classes, rngs, mesh)
    else:
      self._init_sequential_layers(decoder_block_classes, rngs)

  def _init_pipeline_layers(self, decoder_block_classes, rngs, mesh):
    """Initializes decoder layers with pipeline parallelism."""
    config = self.config
    assert not (config.engram_layers and self.is_deepseek), (
        "engram_layers + DeepSeek + pipeline_parallelism is not supported. "
        "engram interleaving is currently only implemented in the non-pipeline path."
    )

    def build_pipeline_stage_layers(rngs):
      return self._get_pipeline_stage_module(decoder_block_classes, rngs)

    self.pipeline_module = create_nnx_pipeline(
        config=config,
        stage_factory=build_pipeline_stage_layers,
        mesh=mesh,
        remat_policy=self.get_remat_policy(),
        rngs=rngs,
    )

    if self.is_deepseek:
      self._init_pipeline_deepseek(decoder_block_classes, rngs)
    else:
      self._init_pipeline_generic(decoder_block_classes, rngs)

  def _init_pipeline_deepseek(self, decoder_block_classes, rngs):
    """Initializes DeepSeek dense and MoE layers outside pipeline."""
    config = self.config
    assert len(decoder_block_classes) == 2
    dense_cls, moe_cls = decoder_block_classes
    if config.scan_layers:
      self.dense_layers = self._create_scanned_layers(
          dense_cls,
          length=config.first_num_dense_layers,
          metadata_axis_name="dense_layers",
          rngs=rngs,
      )
      num_moe_outside = (config.num_decoder_layers - config.first_num_dense_layers) - config.pipeline_parallel_layers
      if num_moe_outside > 0:
        self.moe_layers_outside_pipeline = self._create_scanned_layers(
            moe_cls,
            length=num_moe_outside,
            metadata_axis_name="moe_layers",
            rngs=rngs,
        )
    else:
      self.num_dense_layers = config.first_num_dense_layers
      for i in range(self.num_dense_layers):
        self._create_and_register_named_layer(dense_cls, rngs, "dense_layers", i)
      self.num_moe_outside_pipeline = (
          config.num_decoder_layers - config.first_num_dense_layers
      ) - config.pipeline_parallel_layers
      if self.num_moe_outside_pipeline > 0:
        for i in range(self.num_moe_outside_pipeline):
          self._create_and_register_named_layer(moe_cls, rngs, "moe_layers_outside_pipeline", i)

  def _init_pipeline_generic(self, decoder_block_classes, rngs):
    """Initializes generic decoder layers outside pipeline."""
    config = self.config
    remaining_layers = config.num_decoder_layers - config.pipeline_parallel_layers
    if remaining_layers > 0:
      base_cls = decoder_block_classes[0]
      if config.scan_layers:
        self.layers_outside_pipeline = self._create_scanned_layers(
            base_cls,
            length=remaining_layers,
            metadata_axis_name="layers",
            rngs=rngs,
        )
      else:
        self.num_layers_outside_pipeline = remaining_layers
        for i in range(self.num_layers_outside_pipeline):
          self._create_and_register_named_layer(base_cls, rngs, "layers_outside_pipeline", i)

  def _init_scanned_layers(self, decoder_block_classes, rngs, mesh):
    """Initializes decoder layers with scanning (non-pipeline)."""
    if self.is_deepseek:
      self._init_scanned_deepseek(decoder_block_classes, rngs)
    elif self.is_gemma3:
      self._init_scanned_gemma3(decoder_block_classes, rngs, mesh)
    elif self.is_gemma4:
      self._init_scanned_gemma4(decoder_block_classes, rngs, mesh)
    else:
      self._init_scanned_generic(decoder_block_classes, rngs)

  def _init_scanned_deepseek(self, decoder_block_classes, rngs):
    """Initializes scanned DeepSeek layers with optional Engram support."""
    config = self.config
    assert len(decoder_block_classes) == 2
    dense_cls, moe_cls = decoder_block_classes

    if config.engram_layers:
      self._init_scanned_deepseek_engram(dense_cls, moe_cls, rngs)
    else:
      self._init_scanned_deepseek_standard(dense_cls, moe_cls, rngs)

  def _init_scanned_deepseek_engram(self, dense_cls, moe_cls, rngs):
    """Initializes scanned DeepSeek layers with Engram interleaving."""
    config = self.config
    # 1. Create Dense Chunks (Direct setattr, NO nnx.Dict)
    current_idx = 0
    while current_idx < config.first_num_dense_layers:
      if current_idx in config.engram_layers:
        layer_name = f"dense_layers_engram_{current_idx}"
        setattr(
            self,
            layer_name,
            self._create_single_layer(dense_cls, rngs, layer_idx=current_idx),
        )
        current_idx += 1
      else:
        next_boundary = self._find_next_boundary(current_idx, config.first_num_dense_layers, config.engram_layers)
        chunk_name = f"dense_layers_{current_idx}_{next_boundary - 1}"
        setattr(
            self,
            chunk_name,
            self._create_scanned_layers(
                dense_cls,
                length=(next_boundary - current_idx),
                metadata_axis_name=chunk_name,
                rngs=rngs,
            ),
        )
        current_idx = next_boundary

    # 2. Create MoE Chunks (Direct setattr, NO nnx.Dict)
    current_idx = config.first_num_dense_layers
    while current_idx < config.num_decoder_layers:
      if current_idx in config.engram_layers:
        layer_name = f"moe_layers_engram_{current_idx}"
        setattr(
            self,
            layer_name,
            self._create_single_layer(moe_cls, rngs, layer_idx=current_idx),
        )
        current_idx += 1
      else:
        next_boundary = self._find_next_boundary(current_idx, config.num_decoder_layers, config.engram_layers)
        chunk_name = f"moe_layers_{current_idx}_{next_boundary - 1}"
        setattr(
            self,
            chunk_name,
            self._create_scanned_layers(
                moe_cls,
                length=(next_boundary - current_idx),
                metadata_axis_name=chunk_name,
                rngs=rngs,
            ),
        )
        current_idx = next_boundary

  def _init_scanned_deepseek_standard(self, dense_cls, moe_cls, rngs):
    """Initializes scanned DeepSeek layers without Engram interleaving."""
    config = self.config
    num_dense = config.first_num_dense_layers
    self.dense_layers = self._create_scanned_layers(
        dense_cls, length=num_dense, metadata_axis_name="dense_layers", rngs=rngs
    )
    num_moe = config.num_decoder_layers - config.first_num_dense_layers
    self.moe_layers = self._create_scanned_layers(moe_cls, length=num_moe, metadata_axis_name="moe_layers", rngs=rngs)

  def _init_scanned_gemma3(self, decoder_block_classes, rngs, mesh):
    """Initializes scanned Gemma3 layers."""
    config = self.config
    attention_pattern_length = len(gemma3.GEMMA3_ATTENTION_PATTERN)
    scan_length = config.num_decoder_layers // attention_pattern_length
    num_remaining_layers = config.num_decoder_layers % attention_pattern_length
    layer_kwargs = {"num_of_layers": attention_pattern_length}

    rem_layer_kwargs = {"num_of_layers": num_remaining_layers}

    RemattedGemma3Block = gemma3.Gemma3ScannableBlock

    if scan_length > 0:
      self.layers = self._create_scanned_layers(
          RemattedGemma3Block,
          length=scan_length,
          metadata_axis_name="layers",
          rngs=rngs,
          **layer_kwargs,
      )
    self.layers_remainder = RemattedGemma3Block(
        config=self.config,
        mesh=mesh,
        quant=self.quant,
        model_mode=self.model_mode,
        **rem_layer_kwargs,
        rngs=rngs,
    )  # pytype: disable=wrong-keyword-args

  def _init_scanned_gemma4(self, decoder_block_classes, rngs, mesh):
    """Initializes scanned Gemma4 layers."""
    config = self.config
    attention_pattern_length = len(gemma4.GEMMA4_ATTENTION_PATTERN)
    scan_length = config.num_decoder_layers // attention_pattern_length
    num_remaining_layers = config.num_decoder_layers % attention_pattern_length
    policy = self.get_remat_policy()
    # The pure-NNX decoder skips block-level remat (skip_block_remat=True below),
    # so the block rematerializes its own local/global layers instead.
    layer_kwargs = {
        "num_of_layers": attention_pattern_length,
        "remat_policy_fn": policy,
        "apply_internal_remat": True,
    }
    rem_layer_kwargs = {
        "num_of_layers": num_remaining_layers,
        "remat_policy_fn": policy,
        "apply_internal_remat": True,
    }

    RemattedGemma4Block = gemma4.Gemma4ScannableBlock

    if scan_length > 0:
      self.scanned_blocks = self._create_scanned_layers(
          RemattedGemma4Block,
          length=scan_length,
          metadata_axis_name="layers",
          rngs=rngs,
          **layer_kwargs,
      )
    self.layers_remainder = RemattedGemma4Block(
        config=self.config,
        mesh=mesh,
        quant=self.quant,
        model_mode=self.model_mode,
        **rem_layer_kwargs,
        rngs=rngs,
    )

  def _init_scanned_generic(self, decoder_block_classes, rngs):
    """Initializes scanned generic decoder layers."""
    config = self.config
    layer_cls = decoder_block_classes[0]
    num_layers = int(config.num_decoder_layers / config.inhomogeneous_layer_cycle_interval)
    layer_kwargs = {}
    if config.decoder_block == DecoderBlockType.LLAMA4:
      layer_kwargs = {
          "nope_layer_interval": self.config.nope_layer_interval,
          "interleave_moe_layer_step": self.config.interleave_moe_layer_step,
      }

    if num_layers > 0:
      self.layers = self._create_scanned_layers(
          layer_cls,
          length=num_layers,
          metadata_axis_name="layers",
          rngs=rngs,
          **layer_kwargs,
      )
    else:
      self.layers = nnx.List([])

  def _init_sequential_layers(self, decoder_block_classes, rngs):
    """Initializes decoder layers sequentially (no scanning)."""
    self.layers = nnx.List([])

    if self.is_deepseek:
      self._init_sequential_deepseek(decoder_block_classes, rngs)
    else:
      self._init_sequential_generic(decoder_block_classes, rngs)

  def _init_sequential_deepseek(self, decoder_block_classes, rngs):
    """Initializes sequential DeepSeek dense and MoE layers."""
    config = self.config
    dense_cls, moe_cls = decoder_block_classes
    for i in range(config.first_num_dense_layers):
      self._create_and_register_layer(dense_cls, rngs, "dense_layer", i)
    for i in range(config.num_decoder_layers - config.first_num_dense_layers):
      self._create_and_register_layer(moe_cls, rngs, "moe_layer", i)

  def _init_sequential_generic(self, decoder_block_classes, rngs):
    """Initializes sequential generic decoder layers with per-architecture layer_kwargs."""
    config = self.config
    layer_cls = decoder_block_classes[0]

    for lyr in range(config.num_decoder_layers):
      layer_kwargs = {}
      if config.decoder_block == DecoderBlockType.GEMMA3:
        layer_kwargs = {"attention_type": gemma3.get_attention_type(layer_id=lyr)}
      elif config.decoder_block == DecoderBlockType.GEMMA4:
        layer_kwargs = {"attention_type": gemma4.get_attention_type(layer_id=lyr)}
      elif config.decoder_block == DecoderBlockType.LLAMA4:
        layer_kwargs = {
            "is_nope_layer": llama4.determine_is_nope_layer(lyr, self.config.nope_layer_interval),
            "is_moe_layer": llama4.determine_is_moe_layer(lyr, self.config.interleave_moe_layer_step),
        }
      elif config.decoder_block in {
          DecoderBlockType.QWEN3_NEXT,
          DecoderBlockType.QWEN3_5,
      }:
        layer_kwargs = {"layer_idx": lyr}
      elif config.decoder_block == DecoderBlockType.GPT_OSS:
        layer_kwargs = {"attention_type": gpt_oss.get_attention_type(layer_id=lyr)}
      elif config.decoder_block == DecoderBlockType.OLMO3:
        layer_kwargs = {"attention_type": olmo3.get_attention_type(layer_id=lyr)}

      self._create_and_register_layer(layer_cls, rngs, "layers", lyr, **layer_kwargs)

  def _init_gemma4_small_layers(self, rngs):
    """Eagerly builds the Gemma4-small (E2B/E4B) per-layer-input embedder and one DISTINCT
    decoder layer per index.

    Each layer bakes its own attention_type + layer_idx at construction (these select head_dim,
    RoPE base, and KV-share role), so the layers are heterogeneous and cannot be folded into a
    scanned stack. Layers are registered both as named attrs (``layers_{i}``, matching the Linen
    checkpoint keys) and appended to ``self.layers`` for iteration, mirroring
    ``_create_and_register_layer``.
    """
    cfg = self.config
    self.layers = nnx.List([])
    # Only register the PLE submodule when it exists (mirrors the optional position_embedder
    # pattern); assigning None first would make nnx treat the attribute as static.
    if cfg.hidden_size_per_layer_input > 0 and cfg.vocab_size_per_layer_input > 0:
      self.per_layer_embedder = gemma4_small.Gemma4SmallPLE(config=cfg, mesh=self.mesh, rngs=rngs)

    layer_types = gemma4_small.build_layer_types(cfg.num_decoder_layers, cfg.model_name)
    for lyr in range(cfg.num_decoder_layers):
      layer = gemma4_small.Gemma4SmallDecoderLayer(
          config=cfg,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=self.model_mode,
          attention_type=layer_types[lyr],
          layer_idx=lyr,
          rngs=rngs,
      )
      setattr(self, f"layers_{lyr}", layer)
      self.layers.append(layer)

  def _get_pipeline_stage_module(self, decoder_blocks, rngs):
    """Retrieves the wrapper module formatted for single pipeline stage execution."""
    cfg = self.config
    base_stage_cls = decoder_blocks[1] if self.is_deepseek else decoder_blocks[0]

    if cfg.num_layers_per_pipeline_stage == 1:
      return self._create_single_layer(base_stage_cls, rngs)
    elif cfg.scan_layers_per_stage:
      return NNXScannedPipelineStage(
          base_stage_cls,
          cfg.num_layers_per_pipeline_stage,
          cfg,
          self.mesh,
          self.quant,
          self.model_mode,
          rngs=rngs,
      )
    return NNXSequentialPipelineStage(
        base_stage_cls,
        cfg.num_layers_per_pipeline_stage,
        cfg,
        self.mesh,
        self.quant,
        self.model_mode,
        rngs=rngs,
    )

  def _create_and_register_layer(self, layer_cls, rngs, base_name, i, **layer_kwargs):
    attr_name = f"{base_name}_{i}"
    layer = self._create_single_layer(layer_cls, rngs, **layer_kwargs)
    setattr(self, attr_name, layer)
    self.layers.append(layer)

  def _create_and_register_named_layer(self, layer_cls, rngs, base_name, i, **layer_kwargs):
    """Creates a layer registered ONLY via named attribute. Used by pipeline-outside paths
    to avoid double-registration when self.layers list is also tracked elsewhere."""
    attr_name = f"{base_name}_{i}"
    layer = self._create_single_layer(layer_cls, rngs, **layer_kwargs)
    setattr(self, attr_name, layer)

  def _create_single_layer(self, decoder_layer_class, rngs, **kwargs):
    """Helper to create a single layer (Linen or NNX)."""
    if issubclass(decoder_layer_class, nnx.Module):
      return decoder_layer_class(
          config=self.config,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=self.model_mode,
          rngs=rngs,
          **kwargs,
      )
    else:
      layer_linen = decoder_layer_class(
          config=self.config,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=self.model_mode,
          **kwargs,
      )
      return nnx_wrappers.ToNNX(layer_linen, rngs=rngs)

  def _create_scanned_layers(
      self,
      decoder_layer_class,
      length: int,
      metadata_axis_name: str,
      rngs: nnx.Rngs,
      **layer_kwargs,
  ):
    return nnx_scan.create_scanned_layers(
        lambda layer_rngs: decoder_layer_class(
            config=self.config,
            mesh=self.mesh,
            model_mode=self.model_mode,
            quant=self.quant,
            rngs=layer_rngs,
            **layer_kwargs,
        ),
        length=length,
        param_scan_axis=self.config.param_scan_axis,
        metadata_axis_name=metadata_axis_name,
        rngs=rngs,
    )

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

  def _apply_layers_sequentially(
      self,
      layers,
      x_in,
      *args,
      length: int,
      kv_caches_stacked=None,
      skip_block_remat: bool = False,
      unroll: int = 1,
      **kwargs,
  ):
    """Runs the layer stack using nnx.scan.

    Args:
      layers: The stacked NNX module whose params are scanned over.
      x_in: The carry (hidden state) fed into the first layer.
      *args: Positional args broadcast to every layer call.
      length: Number of scan iterations (= number of layers).
      kv_caches_stacked: Optional pytree whose leaves have shape [num_layers, ...].
        When provided, the i-th slice is passed as `kv_cache=` to layer i and the
        updated caches are returned as a third element of the tuple.
      skip_block_remat: When True, do not wrap the scanned body in jax.checkpoint.
        Used when the scanned module already applies its own (finer-grained,
        e.g. per-layer) remat internally, to avoid double rematerialization.
      unroll: Number of scan iterations to unroll into straight-line code
        (forwarded to jax.lax.scan). unroll >= length fully unrolls the loop.
      **kwargs: Keyword args forwarded to the layer (filtered by the layer signature).

    Returns:
      (final_carry, updated_layers) when kv_caches_stacked is None.
      (final_carry, updated_layers, returned_kv_stacked) otherwise.
    """
    if length == 0:
      return (
          x_in,
          layers,
          kv_caches_stacked if kv_caches_stacked is not None else None,
      )
    policy = self.get_remat_policy()
    prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
    graphdef, params, state = nnx.split(layers, nnx.Param, ...)

    scan_axis = self.config.param_scan_axis
    if scan_axis != 0:
      params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0), params)

    layer_cls = layers.__class__
    sig = inspect.signature(layer_cls.__call__)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or "kwargs" in sig.parameters}

    def _extract_matching_state(template, full):
      if isinstance(template, nnx.State):
        return nnx.State({k: _extract_matching_state(v, full[k]) for k, v in template.items()})
      elif isinstance(template, dict):
        return {k: _extract_matching_state(v, full[k]) for k, v in template.items()}
      return full

    dynamic_graph_init = bool(getattr(self, "disable_quant_stats_update", False))
    updated_graphdef = [graphdef]

    use_kv = kv_caches_stacked is not None

    def layer_fn(carry, scanned_vars):
      # Ensure metadata rank matches the sliced values
      scanned_vars = maxtext_utils_nnx.nnx_remove_scan_axis(scanned_vars, "layers")

      # Unpack the sliced variables for THIS layer
      if use_kv:
        current_params, current_state, kv_cache_layer = scanned_vars
      else:
        current_params, current_state = scanned_vars
        kv_cache_layer = None

      if self.config.parameter_memory_host_offload:
        current_params = jax.tree.map(
            lambda x: jax.device_put(x, max_utils.device_space()),
            current_params,
        )

      layer = nnx.merge(graphdef, current_params, current_state)

      # Build call kwargs, injecting per-layer kv_cache when available
      call_kwargs = dict(valid_kwargs)
      if kv_cache_layer is not None:
        call_kwargs["kv_cache"] = kv_cache_layer

      layer_out = layer(carry, *args, **call_kwargs)

      if isinstance(layer_out, tuple):
        new_carry = layer_out[0]
        updated_kv = layer_out[1] if len(layer_out) > 1 else None
      else:
        new_carry = layer_out
        updated_kv = None

      # Extract the updated state to return it
      if dynamic_graph_init:
        new_graphdef, updated_params, updated_state = nnx.split(layer, nnx.Param, ...)
        updated_graphdef[0] = new_graphdef
        returned_params = updated_params
        new_current_state = nnx.State.merge(returned_params, updated_state)
      else:
        new_current_state = nnx.state(layer)

      if use_kv:
        return new_carry, (new_current_state, updated_kv)
      return new_carry, new_current_state

    if skip_block_remat:
      # The scanned module applies its own remat internally; wrapping the whole
      # body again would double-remat and recompute the entire block.
      layer_fn_wrapped = layer_fn
    else:
      layer_fn_wrapped = jax.checkpoint(layer_fn, policy=policy, prevent_cse=prevent_cse)

    if use_kv:
      # If kv_caches is provided (e.g., from vLLM), we CANNOT use jax.lax.scan
      # because scanning requires stacking the kv_caches list, which creates a copy
      # and breaks the in-place memory updates required by vLLM's PagedAttention.
      # Therefore, we must unroll the loop statically when kv_caches is provided.

      # kv_caches_stacked is actually the original kv_caches list in this new flow
      kv_caches_list = kv_caches_stacked
      current_carry = x_in

      for i in range(length):
        # Statically slice the parameters and state for this layer
        current_params = jax.tree.map(lambda x, i=i: x[i], params)
        current_state = jax.tree.map(lambda x, i=i: x[i], state)

        # Call the layer
        current_carry, (_, updated_kv) = layer_fn_wrapped(
            current_carry, (current_params, current_state, kv_caches_list[i])
        )

        # Update the list in-place (mutates the list passed by reference)
        kv_caches_list[i] = updated_kv

      # We don't need to rebuild scanned_state or return it because during
      # inference with vLLM, parameters do not change and we don't need intermediates.
      return current_carry, layers, None
    else:
      params = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(params, length)
      state = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(state, length)

      final_carry, scanned_state = jax.lax.scan(layer_fn_wrapped, x_in, (params, state), unroll=unroll)
      returned_kv_stacked = None

      # Ensure metadata rank matches the stacked values
      scanned_state = maxtext_utils_nnx.nnx_add_scan_axis(scanned_state, "layers", 0)

      if scan_axis != 0:
        new_params, new_rest = scanned_state.split(nnx.Param, ...)
        new_params = maxtext_utils_nnx.nnx_sync_moveaxis(new_params, 0, scan_axis)
        scanned_state = nnx.merge_state(new_params, new_rest)

      returned_kv_stacked = None

    if dynamic_graph_init:
      # If graph changed, we need to merge with the new graphdef.
      # Note: scanned_state here is the full state (Params + rest).
      new_params, new_rest = scanned_state.split(nnx.Param, ...)
      out_layers = nnx.merge(updated_graphdef[0], new_params, new_rest)
    else:
      nnx.update(layers, scanned_state)
      out_layers = layers

    return final_carry, out_layers, returned_kv_stacked if use_kv else None

  def get_decoder_layers(self):
    """Retrieves decoder layer classes based on config using a dictionary lookup."""
    cfg = self.config

    def get_scannable(normal_cls, scannable_cls):
      return [scannable_cls] if cfg.scan_layers else [normal_cls]

    def get_deepseek():
      return [deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer]

    layer_map = {
        DecoderBlockType.DEFAULT: [NNXDecoderLayer],
        DecoderBlockType.LLAMA2: [llama2.LlamaDecoderLayer],
        DecoderBlockType.MISTRAL: [mistral.MistralDecoderLayer],
        DecoderBlockType.MIXTRAL: [mixtral.MixtralDecoderLayer],
        DecoderBlockType.GEMMA: [gemma.GemmaDecoderLayer],
        DecoderBlockType.GEMMA2: [gemma2.Gemma2DecoderLayer],
        DecoderBlockType.GEMMA3: [gemma3.Gemma3DecoderLayer],
        DecoderBlockType.GEMMA4: get_scannable(gemma4.Gemma4DecoderLayer, gemma4.Gemma4ScannableBlock),
        DecoderBlockType.GEMMA4_SMALL: [gemma4_small.Gemma4SmallDecoderLayer],
        DecoderBlockType.GPT3: [gpt3.Gpt3DecoderLayer],
        DecoderBlockType.QWEN2: [qwen2.Qwen2DecoderLayer],
        DecoderBlockType.QWEN3: [qwen3.Qwen3DecoderLayer],
        DecoderBlockType.QWEN3_MOE: [qwen3.Qwen3MoeDecoderLayer],
        DecoderBlockType.QWEN3_CUSTOM_MOE: [qwen3_custom.Qwen3CustomMoeDecoderLayer],
        DecoderBlockType.SIMPLE: [simple_layer.SimpleDecoderLayer],
        DecoderBlockType.SIMPLE_MLP: [simple_layer.SimpleMlpDecoderLayer],
        DecoderBlockType.DEEPSEEK: get_deepseek(),
        DecoderBlockType.DEEPSEEK4: get_scannable(deepseek4.DeepSeek4DecoderLayer, deepseek4.DeepSeek4ScannableBlock),
        DecoderBlockType.GPT_OSS: get_scannable(gpt_oss.GptOssDecoderLayer, gpt_oss.GptOssScannableBlock),
        DecoderBlockType.QWEN3_NEXT: get_scannable(qwen3.Qwen3NextDecoderLayer, qwen3.Qwen3NextScannableBlock),
        DecoderBlockType.QWEN3_5: get_scannable(qwen3_5.Qwen3_5DecoderLayer, qwen3_5.Qwen3_5ScannableBlock),
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
        "kv_proj",
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
      if cfg.remat_policy in {"minimal_with_context", "minimal_flash"}:
        if cfg.remat_policy == "minimal_flash":
          max_logging.log("WARNING: 'minimal_flash' will be deprecated soon, please use 'minimal_with_context' instead.")
        policy = self.minimal_policy(with_context=True)
      elif cfg.remat_policy == "minimal":
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
            "query_proj",
            "value_proj",
            "key_proj",
            "kv_proj",
            "qkv_proj",
            "context",
            "out_proj",
        )
      elif cfg.remat_policy == "save_dot_except_mlpwi":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "kv_proj",
            "qkv_proj",
            "out_proj",
            "mlpwo",
        )
      elif cfg.remat_policy == "save_dot_except_mlp":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "kv_proj",
            "qkv_proj",
            "out_proj",
        )
      elif cfg.remat_policy == "save_qkv_proj":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "kv_proj",
            "qkv_proj",
        )
      elif cfg.remat_policy == "qkv_proj_offloaded":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=[
                "query_proj",
                "value_proj",
                "key_proj",
                "kv_proj",
            ],
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
                "kv_proj",
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
    if self.config.decoder_block in {
        DecoderBlockType.DEFAULT,
        DecoderBlockType.LLAMA2,
        DecoderBlockType.MISTRAL,
        DecoderBlockType.MIXTRAL,
        DecoderBlockType.DEEPSEEK,
        DecoderBlockType.GEMMA,
        DecoderBlockType.GEMMA2,
        DecoderBlockType.GEMMA3,
        DecoderBlockType.GEMMA4,
        DecoderBlockType.GEMMA4_SMALL,
        DecoderBlockType.QWEN2,
        DecoderBlockType.QWEN3,
        DecoderBlockType.QWEN3_MOE,
        DecoderBlockType.QWEN3_CUSTOM_MOE,
        DecoderBlockType.GPT_OSS,
        DecoderBlockType.SIMPLE,
        DecoderBlockType.SIMPLE_MLP,
        DecoderBlockType.LLAMA4,
        DecoderBlockType.OLMO3,
    }:
      return functools.partial(
          RMSNorm,
          num_features=num_features,
          shard_mode=self.config.shard_mode,
          rngs=rngs,
      )
    elif self.config.decoder_block == DecoderBlockType.GPT3:
      return functools.partial(
          gpt3.Gpt3LayerNorm,
          num_features=num_features,
          reductions_in_fp32=False,
          use_bias=True,
          rngs=rngs,
      )
    elif self.config.decoder_block in {
        DecoderBlockType.QWEN3_NEXT,
        DecoderBlockType.QWEN3_5,
    }:
      return functools.partial(
          normalizations.Qwen3NextRMSNorm,
          num_features=num_features,
          shard_mode=self.config.shard_mode,
          rngs=rngs,
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
      multimodal_input=None,
  ):
    """Applies token and positional embeddings to the input tokens."""
    cfg = self.config

    y = shared_embedding(decoder_input_tokens.astype("int32"), model_mode=model_mode)

    # Merge the image embeddings with the text embeddings for multimodal models
    if multimodal_input is not None:
      image_embeddings = multimodal_input.image_embeddings
      bidirectional_mask = multimodal_input.bidirectional_mask
      image_masks = multimodal_input.image_masks
      video_embeddings = getattr(multimodal_input, "video_embeddings", None)
      video_masks = getattr(multimodal_input, "video_masks", None)
      bidirectional_mask_video = getattr(multimodal_input, "bidirectional_mask_video", None)
      audio_embeddings = multimodal_input.audio_embeddings
      audio_masks = multimodal_input.audio_masks

      if image_embeddings is not None and cfg.use_multimodal:
        if cfg.model_name in {
            "gemma3-4b",
            "gemma3-12b",
            "gemma3-27b",
            "gemma4-26b",
            "gemma4-31b",
            "gemma4-e2b",
            "gemma4-e4b",
            "llama4-17b-16e",
            "llama4-17b-128e",
            "qwen3-omni-30b-a3b",
            "qwen3-vl-2b",
            "qwen3-vl-4b",
            "qwen3.5-35b-a3b",
            "qwen3.5-397b-a17b",
        }:
          y = mm_utils.merge_mm_embeddings(
              text_embeddings=y,
              multimodal_embeddings=image_embeddings,
              mask=bidirectional_mask,
              token_masks=image_masks,
          )
        else:
          raise ValueError(f"Unsupported model_name for multimodal: {cfg.model_name}")

      if video_embeddings is not None and cfg.use_multimodal:
        if cfg.model_name in {"qwen3-omni-30b-a3b", "qwen3-vl-2b", "qwen3-vl-4b", "qwen3.5-35b-a3b", "qwen3.5-397b-a17b"}:
          y = mm_utils.merge_mm_embeddings(
              text_embeddings=y,
              multimodal_embeddings=video_embeddings,
              mask=bidirectional_mask_video,
              token_masks=video_masks,
          )
        else:
          raise ValueError(f"Unsupported model_name for video: {cfg.model_name}")

      if audio_embeddings is not None and cfg.use_audio:
        if cfg.model_name in {"qwen3-omni-30b-a3b"}:
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
      norm_out_sharding = create_sharding(
          self.mesh,
          ("activation_batch", "activation_length", "activation_embed"),
      )
    else:
      norm_out_sharding = None

    y = self.decoder_norm(y, out_sharding=norm_out_sharding)
    y = self.dropout(y, deterministic=deterministic)  # NNX call

    if model_mode in {MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE}:
      out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
    else:
      out_sharding = create_sharding(
          self.mesh,
          (
              "activation_embed_and_logits_batch",
              "activation_length",
              "activation_vocab",
          ),
      )

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      if isinstance(shared_embedding, nnx.Module):
        embedding_table = shared_embedding.embedding[...]
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

  def _build_linen_params(self, moe_stack: nnx.Module) -> dict:
    """
    Bridges NNX to Linen by creating a dictionary that mimics the exact variable
    structure expected by `deepseek_batchsplit.fetch_weights`.
    """
    state_dict = nnx.state(moe_stack, nnx.Param)

    return {
        "pre_self_attention_layer_norm": state_dict["pre_self_attention_layer_norm"],
        "post_self_attention_layer_norm": state_dict["post_self_attention_layer_norm"],
        "self_attention": state_dict["self_attention"],
        "DeepSeekMoeBlock_0": state_dict.get("moe_block", state_dict.get("DeepSeekMoeBlock_0")),
    }

  def _find_next_boundary(self, current_idx, end_idx, engram_indices):
    """Finds the next index boundary, either the next Engram layer index or the overall end index."""
    next_engrams = [l for l in engram_indices if l > current_idx]
    if next_engrams:
      return min(end_idx, *next_engrams)
    return end_idx

  def _apply_single_engram_layer(self, y, layer_name, *args, **kwargs):
    """Applies a single, unscanned Engram layer."""
    layer = getattr(self, layer_name)

    decoder_input_tokens = kwargs.get("decoder_input_tokens")
    layer_kwargs = kwargs.get("layer_kwargs", {})

    out = layer(y, *args, decoder_input_tokens=decoder_input_tokens, **layer_kwargs)
    if isinstance(out, tuple):
      y = out[0]
    else:
      y = out

    return y

  def _apply_scanned_chunk(self, y, current_idx, next_boundary, layer_stack, *args, **kwargs):
    """Applies a contiguous chunk of layers using scan over a state slice."""
    scan_length = next_boundary - current_idx
    if scan_length > 0:
      graphdef, state = nnx.split(layer_stack)
      params, rest = state.split(nnx.Param, ...)
      scan_axis = self.config.param_scan_axis

      # Slice the chunk state along the correct axes
      chunk_params = jax.tree.map(
          lambda x: jax.lax.dynamic_slice_in_dim(x, current_idx, scan_length, axis=scan_axis),
          params,
      )
      chunk_rest = jax.tree.map(
          lambda x: jax.lax.dynamic_slice_in_dim(x, current_idx, scan_length, axis=0),
          rest,
      )
      chunk_stack = nnx.merge(graphdef, chunk_params, chunk_rest)

      # Apply sequentially
      y, chunk_stack, _ = self._apply_layers_sequentially(
          chunk_stack,
          y,
          *args,
          length=scan_length,
          **kwargs.get("layer_kwargs", {}),
      )

      # Update the original stack state
      new_state = nnx.state(chunk_stack)
      new_params, new_rest = new_state.split(nnx.Param, ...)

      updated_params = jax.tree.map(
          lambda s, new_s: jax.lax.dynamic_update_slice_in_dim(s, new_s, current_idx, axis=scan_axis),
          params,
          new_params,
      )
      updated_rest = jax.tree.map(
          lambda s, new_s: jax.lax.dynamic_update_slice_in_dim(s, new_s, current_idx, axis=0),
          rest,
          new_rest,
      )

      nnx.update(layer_stack, updated_params, updated_rest)

    return y

  def _apply_interleaved_scanned_layers(self, y, layer_prefix, start_idx, end_idx, engram_indices, *args, **kwargs):
    """Applies a mix of scanned standard layers and unscanned Engram layers."""
    current_idx = start_idx
    while current_idx < end_idx:
      if current_idx in engram_indices:
        layer_name = f"{layer_prefix}_engram_{current_idx}"
        y = self._apply_single_engram_layer(y, layer_name, *args, **kwargs)
        current_idx += 1
      else:
        next_boundary = self._find_next_boundary(current_idx, end_idx, engram_indices)
        chunk_name = f"{layer_prefix}_{current_idx}_{next_boundary - 1}"
        chunk_stack = getattr(self, chunk_name)
        scan_length = next_boundary - current_idx

        y, chunk_stack, _ = self._apply_layers_sequentially(
            chunk_stack,
            y,
            *args,
            length=scan_length,
            **kwargs.get("layer_kwargs", {}),
        )
        current_idx = next_boundary
    return y

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
      kv_caches: list[jax.Array] | None = None,
      attention_metadata=None,
      deepstack_visual_embeds: None | list[jnp.ndarray] = None,
      multimodal_input: None | MultimodalInput = None,
  ):
    cfg = self.config
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    policy = self.get_remat_policy()

    # [batch, length] -> [batch, length, emb_dim]
    y = self._apply_embedding(
        shared_embedding,
        decoder_input_tokens,
        decoder_positions,
        deterministic,
        model_mode,
        multimodal_input=multimodal_input,
    )

    mhc_expand, mhc_reduce = mhc.get_functions(cfg.mhc_expansion_rate)
    if cfg.mhc_expansion_rate > 1:
      # (batch, length, emb_dim) --> (batch, length, mhc_expansion_rate, emb_dim)
      y = mhc_expand(y)

    layer_args = (decoder_segment_ids, decoder_positions, deterministic, model_mode)

    layer_kwargs = {}
    # Extract the bidirectional mask locally for layer configurations
    bidirectional_mask = multimodal_input.bidirectional_mask if multimodal_input is not None else None

    if cfg.decoder_block in {DecoderBlockType.GEMMA3, DecoderBlockType.GEMMA4}:
      layer_kwargs["bidirectional_mask"] = bidirectional_mask

    if attention_metadata is not None:
      layer_kwargs["attention_metadata"] = attention_metadata

    if cfg.using_pipeline_parallelism:
      logical_partition_spec = (
          self.pipeline_module.get_weight_sharding()
          if (cfg.pipeline_fsdp_ag_once or cfg.pipeline_fsdp_ag_per_repeat)
          else None
      )

      if self.is_deepseek:
        # Pre-pipeline: dense layers + outside-pipeline MoE layers under PP-as-DP axis rules.
        ds_layer_kwargs = {
            "previous_chunk": previous_chunk,
            "slot": slot,
        }
        logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(cfg.logical_axis_rules)
        with self.mesh, nn.partitioning.axis_rules(logical_axis_rules_pp_as_dp):
          if cfg.scan_layers:
            if getattr(self, "dense_layers", None) is not None and cfg.first_num_dense_layers > 0:
              y, self.dense_layers, _ = self._apply_layers_sequentially(
                  self.dense_layers,
                  y,
                  *layer_args,
                  length=cfg.first_num_dense_layers,
                  **ds_layer_kwargs,
              )
            if hasattr(self, "moe_layers_outside_pipeline") and self.moe_layers_outside_pipeline is not None:
              num_moe_outside = (cfg.num_decoder_layers - cfg.first_num_dense_layers) - cfg.pipeline_parallel_layers
              y, self.moe_layers_outside_pipeline, _ = self._apply_layers_sequentially(
                  self.moe_layers_outside_pipeline,
                  y,
                  *layer_args,
                  length=num_moe_outside,
                  **ds_layer_kwargs,
              )
          else:
            # Unscanned: iterate registered layers by name.
            for i in range(getattr(self, "num_dense_layers", 0)):
              layer = getattr(self, f"dense_layers_{i}")
              out = layer(y, *layer_args, **ds_layer_kwargs)
              y = out[0] if isinstance(out, tuple) else out
            for i in range(getattr(self, "num_moe_outside_pipeline", 0)):
              layer = getattr(self, f"moe_layers_outside_pipeline_{i}")
              out = layer(y, *layer_args, **ds_layer_kwargs)
              y = out[0] if isinstance(out, tuple) else out

        y = self.pipeline_module(
            y,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            logical_partition_spec=logical_partition_spec,
        )
      elif self.is_gemma4:
        y = self._apply_gemma4_scanned_blocks(
            y,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            bidirectional_mask,
            previous_chunk,
            slot,
            kv_caches=kv_caches,
            attention_metadata=attention_metadata,
        )
      else:
        # Standard pipeline run (non-DeepSeek, incl. Gemma4 — matches Linen decoders.py).
        # Gemma4 routes through the pipeline here; _apply_gemma4_scanned_blocks is
        # non-pipeline-only (its layers/layers_remainder are not built when
        # pipeline parallelism is enabled).
        y = self.pipeline_module(
            y,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            logical_partition_spec=logical_partition_spec,
        )

        # Remaining standard layers (outside the pipeline)
        if hasattr(self, "layers_outside_pipeline") or hasattr(self, "num_layers_outside_pipeline"):
          logical_axis_rules_pp_as_dp = sharding.logical_axis_rules_pp_act_as_dp(cfg.logical_axis_rules)
          with (
              self.mesh,
              nn.partitioning.axis_rules(logical_axis_rules_pp_as_dp),
          ):
            if cfg.scan_layers and hasattr(self, "layers_outside_pipeline"):
              remaining = cfg.num_decoder_layers - cfg.pipeline_parallel_layers
              y, self.layers_outside_pipeline, _ = self._apply_layers_sequentially(
                  self.layers_outside_pipeline,
                  y,
                  *layer_args,
                  length=remaining,
                  **layer_kwargs,
              )
            elif (not cfg.scan_layers) and hasattr(self, "num_layers_outside_pipeline"):
              for i in range(self.num_layers_outside_pipeline):
                layer = getattr(self, f"layers_outside_pipeline_{i}")
                out = layer(y, *layer_args, **layer_kwargs)
                y = out[0] if isinstance(out, tuple) else out

    else:
      if self.is_gemma4_small:
        y, kv_caches = self._apply_gemma4_small_layers(
            y,
            decoder_input_tokens,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            multimodal_input=multimodal_input,
            kv_caches=kv_caches,
            attention_metadata=attention_metadata,
            previous_chunk=previous_chunk,
            slot=slot,
        )
      elif cfg.scan_layers:
        if self.is_deepseek:
          layer_kwargs = {
              "previous_chunk": previous_chunk,
              "slot": slot,
          }

          if cfg.engram_layers:
            common_kwargs = {
                "layer_kwargs": layer_kwargs,
                "decoder_input_tokens": decoder_input_tokens,
            }

            y = self._apply_interleaved_scanned_layers(
                y,
                "dense_layers",
                0,
                cfg.first_num_dense_layers,
                cfg.engram_layers,
                *layer_args,
                **common_kwargs,
            )

            y = self._apply_interleaved_scanned_layers(
                y,
                "moe_layers",
                cfg.first_num_dense_layers,
                cfg.num_decoder_layers,
                cfg.engram_layers,
                *layer_args,
                **common_kwargs,
            )
          else:
            y, self.dense_layers, _ = self._apply_layers_sequentially(
                self.dense_layers,
                y,
                *layer_args,
                length=cfg.first_num_dense_layers,
                **layer_kwargs,
            )

            num_moe = cfg.num_decoder_layers - cfg.first_num_dense_layers

            if cfg.use_batch_split_schedule:
              policy = self.get_remat_policy()
              mock_params = self._build_linen_params(self.moe_layers)

              if cfg.use_qwix_quantization and not cfg.use_manual_quantization:
                y = deepseek_batchsplit_fp8.scan_batch_split_layers(
                    y,
                    mock_params,
                    decoder_positions,
                    decoder_segment_ids,
                    model_mode=model_mode,
                    mesh=self.mesh,
                    quant=self.quant,
                    cfg=cfg,
                    policy=policy,
                )
              else:
                # bf16 code path
                y = deepseek_batchsplit.scan_batch_split_layers(
                    y,
                    mock_params,
                    decoder_positions,
                    mesh=self.mesh,
                    cfg=cfg,
                    num_layers=num_moe,
                )
            else:
              y, self.moe_layers, _ = self._apply_layers_sequentially(
                  self.moe_layers,
                  y,
                  *layer_args,
                  length=num_moe,
                  **layer_kwargs,
              )
        elif self.is_gemma3:
          y = self._apply_gemma3_scanned_blocks(
              y,
              decoder_segment_ids,
              decoder_positions,
              deterministic,
              model_mode,
              bidirectional_mask,
              previous_chunk,
              slot,
          )
        elif self.is_gemma4:
          y = self._apply_gemma4_scanned_blocks(
              y,
              decoder_segment_ids,
              decoder_positions,
              deterministic,
              model_mode,
              bidirectional_mask,
              previous_chunk,
              slot,
          )
        else:
          scan_length = int(cfg.num_decoder_layers / cfg.inhomogeneous_layer_cycle_interval)
          if kv_caches is not None:
            # Pass the kv_caches list directly to avoid copying in jnp.stack,
            # which breaks vLLM PagedAttention in-place memory updates.
            # The _apply_layers_sequentially function will handle it by statically unrolling.
            y, self.layers, _ = self._apply_layers_sequentially(
                self.layers,
                y,
                *layer_args,
                length=scan_length,
                kv_caches_stacked=kv_caches,
                **layer_kwargs,
            )
            # kv_caches list is updated in-place inside _apply_layers_sequentially
          else:
            y, self.layers, _ = self._apply_layers_sequentially(
                self.layers,
                y,
                *layer_args,
                length=scan_length,
                **layer_kwargs,
            )
      else:
        prevent_cse = maxtext_utils.should_prevent_cse_in_remat(cfg)

        # Hoisted function to preserve XLA cache ID
        def pure_layer_fn(graphdef, state_in, y_in, kv_in):

          if cfg.parameter_memory_host_offload:
            state_in = jax.tree.map(
                lambda x: jax.device_put(x, max_utils.device_space()),
                state_in,
            )

          merged_layer = nnx.merge(graphdef, state_in)
          out_y, out_kv = merged_layer(y_in, *layer_args, kv_cache=kv_in, **layer_kwargs)
          return out_y, out_kv, nnx.state(merged_layer)

        checkpointed_fn = jax.checkpoint(pure_layer_fn, policy=policy, prevent_cse=prevent_cse)

        for lyr, layer in enumerate(self.layers):
          graphdef, state = nnx.split(layer)
          if kv_caches is not None:
            if cfg.decoder_block in (DecoderBlockType.QWEN3_NEXT, DecoderBlockType.QWEN3_5):
              if (lyr + 1) % cfg.inhomogeneous_layer_cycle_interval == 0:
                kv_cache = (
                    kv_caches["key_cache"][lyr],
                    kv_caches["value_cache"][lyr],
                )
              else:
                kv_cache = None
            else:
              kv_cache = kv_caches[lyr]
          else:
            kv_cache = None

          input_tokens = decoder_input_tokens if cfg.engram_layers else None
          if input_tokens is not None:
            layer_kwargs["decoder_input_tokens"] = input_tokens

          y, kv_cache, new_state = checkpointed_fn(graphdef, state, y, kv_cache)
          nnx.update(layer, new_state)

          if kv_caches is not None and kv_cache is not None:
            if cfg.decoder_block in (DecoderBlockType.QWEN3_NEXT, DecoderBlockType.QWEN3_5):
              if (lyr + 1) % cfg.inhomogeneous_layer_cycle_interval == 0:
                kv_caches["key_cache"][lyr] = kv_cache[0]
                kv_caches["value_cache"][lyr] = kv_cache[1]
            else:
              kv_caches[lyr] = kv_cache

          if deepstack_visual_embeds is not None and lyr < len(deepstack_visual_embeds):
            visual_embeds = deepstack_visual_embeds[lyr]
            if bidirectional_mask is not None and visual_embeds is not None:
              y = deepstack_process(y, bidirectional_mask, visual_embeds)

    assert isinstance(y, jax.Array)

    # After the final transformer layer, `y` holds the raw, un-normalized hidden state.
    if cfg.mhc_expansion_rate > 1:
      # (batch, length, mhc_expansion_rate, emb_dim) --> (batch, length, emb_dim)
      hidden_state = mhc_reduce(y)
    else:
      hidden_state = y

    # When invoking from vLLM with RPA attention, logit computation is deferred to a later stage.
    if cfg.attention == "vllm_rpa":
      logits = None

    # When in the Indexer Dense Warm-up stage, skip the expensive output head projection
    # for efficiency, as the main model is frozen and the LM loss is not needed.
    elif (
        cfg.use_indexer and cfg.indexer_loss_scaling_factor > 0.0 and not cfg.indexer_sparse_training
    ) and model_mode == MODEL_MODE_TRAIN:
      logits = None

    # When vocab tiling is enabled in training mode, full logits won't generate to reduce memory
    # Instead, we keep track on the hidden states, which has smaller size compared to full logits
    elif cfg.num_vocab_tiling > 1 and model_mode == MODEL_MODE_TRAIN:
      logits = None
      self.sow(nnx.Intermediate, "hidden_states", hidden_state)

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
      slot,
      kv_caches=None,
      attention_metadata=None,
  ):
    """Applies Gemma3 scanned decoder blocks, handling main scan and remainders."""

    cfg = self.config

    # Define the repeating pattern length and calculate how many full blocks to scan
    attention_pattern_length = len(gemma3.GEMMA3_ATTENTION_PATTERN)
    scan_length = cfg.num_decoder_layers // attention_pattern_length

    layer_args = (decoder_segment_ids, decoder_positions, deterministic, model_mode)
    layer_kwargs = {"bidirectional_mask": bidirectional_mask}
    if attention_metadata is not None:
      layer_kwargs["attention_metadata"] = attention_metadata

    # Apply the main scan over the full blocks
    if scan_length > 0:
      grouped_kv_caches = maxtext_utils.prepare_kv_caches_for_scan(
          kv_caches, scan_length, attention_pattern_length, stack=False
      )
      y, self.layers, _ = self._apply_layers_sequentially(
          self.layers, y, *layer_args, length=scan_length, kv_caches_stacked=grouped_kv_caches, **layer_kwargs
      )
      maxtext_utils.update_kv_caches_after_scan(
          kv_caches, grouped_kv_caches, scan_length, attention_pattern_length, stacked=False
      )

    # Apply any remaining layers that did not fit into a full scanned block
    num_remaining_layers = cfg.num_decoder_layers % attention_pattern_length
    if num_remaining_layers > 0:
      policy = self.get_remat_policy()
      prevent_cse = maxtext_utils.should_prevent_cse_in_remat(cfg)

      remainder_kv = None
      if kv_caches is not None:
        start_idx = scan_length * attention_pattern_length
        remainder_kv = tuple(kv_caches[start_idx : start_idx + num_remaining_layers])

      def pure_gemma_fn(graphdef, state_in, y_in, kv_in):
        merged_layer = nnx.merge(graphdef, state_in)
        call_kwargs = dict(layer_kwargs)
        if kv_in is not None:
          call_kwargs["kv_cache"] = kv_in
        call_kwargs["previous_chunk"] = previous_chunk
        call_kwargs["slot"] = slot
        out_res = merged_layer(y_in, *layer_args, **call_kwargs)
        if isinstance(out_res, tuple):
          out_y = out_res[0]
          out_kv = out_res[1] if len(out_res) > 1 else None
        else:
          out_y = out_res
          out_kv = None
        return out_y, out_kv, nnx.state(merged_layer)

      checkpointed_gemma_fn = jax.checkpoint(pure_gemma_fn, policy=policy, prevent_cse=prevent_cse)

      graphdef, state = nnx.split(self.layers_remainder)
      y, updated_remainder_kv, new_state = checkpointed_gemma_fn(graphdef, state, y, remainder_kv)
      nnx.update(self.layers_remainder, new_state)

      if kv_caches is not None and updated_remainder_kv is not None:
        start_idx = scan_length * attention_pattern_length
        for offset, updated_item in enumerate(updated_remainder_kv):
          kv_caches[start_idx + offset] = updated_item

    return y

  def _apply_gemma4_scanned_blocks(
      self,
      y,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      bidirectional_mask,
      previous_chunk,
      slot,
      kv_caches=None,
      attention_metadata=None,
  ):
    """Applies Gemma4 scanned decoder blocks, handling main scan and remainders."""

    cfg = self.config

    # Define the repeating pattern length and calculate how many full blocks to scan
    attention_pattern_length = len(gemma4.GEMMA4_ATTENTION_PATTERN)
    scan_length = cfg.num_decoder_layers // attention_pattern_length

    layer_args = (decoder_segment_ids, decoder_positions, deterministic, model_mode)
    layer_kwargs = {"bidirectional_mask": bidirectional_mask, "slot": slot, "previous_chunk": previous_chunk}
    if attention_metadata is not None:
      layer_kwargs["attention_metadata"] = attention_metadata

    # Apply the main scan over the full blocks. Gemma4ScannableBlock applies
    # per-layer remat internally (local scan + global layer), so skip the
    # block-level remat here to avoid double rematerialization. Unrolling the
    # block loop (one iteration per repeated block) lets XLA pipeline/free block
    # activations across iterations (memory + overlap knob).
    block_unroll = max(1, scan_length)
    if scan_length > 0:
      grouped_kv_caches = maxtext_utils.prepare_kv_caches_for_scan(
          kv_caches, scan_length, attention_pattern_length, stack=False
      )
      y, self.scanned_blocks, _ = self._apply_layers_sequentially(
          self.scanned_blocks,
          y,
          *layer_args,
          length=scan_length,
          kv_caches_stacked=grouped_kv_caches,
          skip_block_remat=True,
          unroll=block_unroll,
          **layer_kwargs,
      )
      maxtext_utils.update_kv_caches_after_scan(
          kv_caches, grouped_kv_caches, scan_length, attention_pattern_length, stacked=False
      )

    # Apply any remaining layers that did not fit into a full scanned block
    num_remaining_layers = cfg.num_decoder_layers % attention_pattern_length
    if num_remaining_layers > 0:
      policy = self.get_remat_policy()
      prevent_cse = maxtext_utils.should_prevent_cse_in_remat(cfg)

      remainder_kv = None
      if kv_caches is not None:
        start_idx = scan_length * attention_pattern_length
        remainder_kv = tuple(kv_caches[start_idx : start_idx + num_remaining_layers])

      def pure_gemma_fn(graphdef, state_in, y_in, kv_in):
        merged_layer = nnx.merge(graphdef, state_in)
        call_kwargs = dict(layer_kwargs)
        if kv_in is not None:
          call_kwargs["kv_cache"] = kv_in
        call_kwargs["previous_chunk"] = previous_chunk
        call_kwargs["slot"] = slot
        out_res = merged_layer(y_in, *layer_args, **call_kwargs)
        if isinstance(out_res, tuple):
          out_y = out_res[0]
          out_kv = out_res[1] if len(out_res) > 1 else None
        else:
          out_y = out_res
          out_kv = None
        return out_y, out_kv, nnx.state(merged_layer)

      checkpointed_gemma_fn = jax.checkpoint(pure_gemma_fn, policy=policy, prevent_cse=prevent_cse)

      graphdef, state = nnx.split(self.layers_remainder)
      y, updated_remainder_kv, new_state = checkpointed_gemma_fn(graphdef, state, y, remainder_kv)
      nnx.update(self.layers_remainder, new_state)

      if kv_caches is not None and updated_remainder_kv is not None:
        start_idx = scan_length * attention_pattern_length
        for offset, updated_item in enumerate(updated_remainder_kv):
          kv_caches[start_idx + offset] = updated_item

    return y

  def _apply_gemma4_small_layers(
      self,
      y,
      decoder_input_tokens,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      multimodal_input=None,
      kv_caches=None,
      attention_metadata=None,
      previous_chunk=None,
      slot=None,
  ):
    """Apply Gemma 4 small (E2B/E4B) decoder layers (pure-NNX)."""
    cfg = self.config
    bidirectional_mask_value = multimodal_input.bidirectional_mask if multimodal_input is not None else None

    per_layer_inputs = None
    if cfg.hidden_size_per_layer_input > 0 and cfg.vocab_size_per_layer_input > 0:
      per_layer_inputs = self.per_layer_embedder(decoder_input_tokens, y)

    layer_types = gemma4_small.build_layer_types(cfg.num_decoder_layers, cfg.model_name)
    num_kv_shared = cfg.num_kv_shared_layers
    shared_kv_states: dict[int, tuple[jax.Array, jax.Array]] = {}
    # tpu-inference allocates one kv_caches slot per non-shared layer; KV-shared layers reuse the donor's slot.
    cache_index_of = gemma4_small.kv_cache_slot_map(layer_types, num_kv_shared)

    for lyr in range(cfg.num_decoder_layers):
      layer = self.layers[lyr]
      donor_idx = gemma4_small.kv_donor_layer_idx(lyr, layer_types, num_kv_shared)
      is_donor = gemma4_small.is_kv_donor_layer(lyr, layer_types, num_kv_shared)

      shared_key = None
      shared_value = None
      if donor_idx is not None:
        if donor_idx not in shared_kv_states:
          raise RuntimeError(
              f"KV-shared layer {lyr} references donor {donor_idx} but no donor K/V "
              f"have been recorded yet. This indicates the layer iteration order is wrong."
          )
        shared_key, shared_value = shared_kv_states[donor_idx]

      # Donor layers expose their rotated, normed K/V to downstream shared layers, and reuse the
      # just-computed K/V in their own forward to avoid double-computing the K/V projection.
      if is_donor:
        donor_k, donor_v = layer.compute_shared_kv(y, decoder_positions)
        shared_kv_states[lyr] = (donor_k, donor_v)
        shared_key, shared_value = donor_k, donor_v

      ple_slice = per_layer_inputs[..., lyr, :] if per_layer_inputs is not None else None

      cache_idx = cache_index_of[lyr]
      kv_cache = kv_caches[cache_idx] if kv_caches is not None else None
      y, kv_cache = layer(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          bidirectional_mask=bidirectional_mask_value,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
          per_layer_input=ple_slice,
          shared_key=shared_key,
          shared_value=shared_value,
      )
      if kv_caches is not None and kv_cache is not None:
        kv_caches[cache_idx] = kv_cache

    return y, kv_caches


def decoder_as_linen(
    config: Config,
    mesh: Mesh,
    rngs: nnx.Rngs,
    model_mode: str,
    quant: None | Quant = None,
):
  """Creates a Decoder module"""
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

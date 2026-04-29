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
    deepseek_batchsplit_fp8,
    gemma,
    gemma2,
    gemma3,
    gemma4,
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

    if self.config.scan_layers:
      if self.is_deepseek:
        assert len(decoder_block_classes) == 2
        dense_cls, moe_cls = decoder_block_classes

        if config.engram_layers:
          # 1. Create Dense Chunks (Direct setattr, NO nnx.Dict)
          current_idx = 0
          while current_idx < config.first_num_dense_layers:
            if current_idx in config.engram_layers:
              layer_name = f"dense_layers_engram_{current_idx}"
              setattr(self, layer_name, self._create_single_layer(dense_cls, rngs, layer_idx=current_idx))
              current_idx += 1
            else:
              next_boundary = self._find_next_boundary(current_idx, config.first_num_dense_layers, config.engram_layers)
              chunk_name = f"dense_layers_{current_idx}_{next_boundary - 1}"
              setattr(
                  self,
                  chunk_name,
                  self._create_scanned_layers(
                      dense_cls, length=(next_boundary - current_idx), metadata_axis_name=chunk_name, rngs=rngs
                  ),
              )
              current_idx = next_boundary

          # 2. Create MoE Chunks (Direct setattr, NO nnx.Dict)
          current_idx = config.first_num_dense_layers
          while current_idx < config.num_decoder_layers:
            if current_idx in config.engram_layers:
              layer_name = f"moe_layers_engram_{current_idx}"
              setattr(self, layer_name, self._create_single_layer(moe_cls, rngs, layer_idx=current_idx))
              current_idx += 1
            else:
              next_boundary = self._find_next_boundary(current_idx, config.num_decoder_layers, config.engram_layers)
              chunk_name = f"moe_layers_{current_idx}_{next_boundary - 1}"
              setattr(
                  self,
                  chunk_name,
                  self._create_scanned_layers(
                      moe_cls, length=(next_boundary - current_idx), metadata_axis_name=chunk_name, rngs=rngs
                  ),
              )
              current_idx = next_boundary
        else:
          # Standard DeepSeek logic when Engrams are disabled
          num_dense = config.first_num_dense_layers
          self.dense_layers = self._create_scanned_layers(
              dense_cls, length=num_dense, metadata_axis_name="dense_layers", rngs=rngs
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
        )  # pytype: disable=wrong-keyword-args
      elif self.is_gemma4:
        attention_pattern_length = len(gemma4.GEMMA4_ATTENTION_PATTERN)
        scan_length = config.num_decoder_layers // attention_pattern_length
        num_remaining_layers = config.num_decoder_layers % attention_pattern_length
        layer_kwargs = {"num_of_layers": attention_pattern_length}

        rem_layer_kwargs = {"num_of_layers": num_remaining_layers}

        RemattedGemma4Block = gemma4.Gemma4ScannableBlock

        if scan_length > 0:
          self.layers = self._create_scanned_layers(
              RemattedGemma4Block, length=scan_length, metadata_axis_name="layers", rngs=rngs, **layer_kwargs
          )
        self.layers_remainder = RemattedGemma4Block(
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

        if num_layers > 0:
          self.layers = self._create_scanned_layers(
              layer_cls, length=num_layers, metadata_axis_name="layers", rngs=rngs, **layer_kwargs
          )
        else:
          self.layers = nnx.List([])

    else:
      self.layers = nnx.List([])

      if self.is_deepseek:
        dense_cls, moe_cls = decoder_block_classes
        for i in range(config.first_num_dense_layers):
          self._create_and_register_layer(dense_cls, rngs, "dense_layer", i)
        for i in range(config.num_decoder_layers - config.first_num_dense_layers):
          self._create_and_register_layer(moe_cls, rngs, "moe_layer", i)
      else:
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
          elif config.decoder_block == DecoderBlockType.QWEN3_NEXT:
            layer_kwargs = {"layer_idx": lyr}
          elif config.decoder_block == DecoderBlockType.GPT_OSS:
            layer_kwargs = {"attention_type": gpt_oss.get_attention_type(layer_id=lyr)}
          elif config.decoder_block == DecoderBlockType.OLMO3:
            layer_kwargs = {"attention_type": olmo3.get_attention_type(layer_id=lyr)}

          self._create_and_register_layer(layer_cls, rngs, "layers", lyr, **layer_kwargs)

  def _create_and_register_layer(self, layer_cls, rngs, base_name, i, **layer_kwargs):
    attr_name = f"{base_name}_{i}"
    layer = self._create_single_layer(layer_cls, rngs, **layer_kwargs)
    setattr(self, attr_name, layer)
    self.layers.append(layer)

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
    """Creates a scanned stack of layers using jax.lax.scan for memory-efficient initialization."""
    if length == 0:
      return None
    scan_axis = self.config.param_scan_axis

    # Fork rngs to get per-layer RNG states for scanning
    try:
      forked_rngs = rngs.fork(split=length)
    except:  # pylint: disable=bare-except
      pass

    rngs_graphdef, rngs_state = nnx.split(forked_rngs)

    first_rng_state = jax.tree.map(lambda x: x[0], rngs_state)
    ref_rngs = nnx.merge(rngs_graphdef, first_rng_state)
    ref_layer = decoder_layer_class(
        config=self.config, mesh=self.mesh, quant=self.quant, model_mode=self.model_mode, rngs=ref_rngs, **layer_kwargs
    )
    layer_graphdef, _, _ = nnx.split(ref_layer, nnx.Param, ...)
    del ref_layer

    def scan_body(carry, rng_state_slice):
      layer_rngs = nnx.merge(rngs_graphdef, rng_state_slice)
      layer = decoder_layer_class(
          config=self.config,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=self.model_mode,
          rngs=layer_rngs,
          **layer_kwargs,
      )
      _, params, rest = nnx.split(layer, nnx.Param, ...)
      return carry, (params, rest)

    _, (stacked_params, stacked_rest) = jax.lax.scan(scan_body, None, rngs_state)

    if scan_axis != 0:
      stacked_params = jax.tree.map(lambda x: jnp.moveaxis(x, 0, scan_axis), stacked_params)

    def _add_scan_metadata(state, axis):
      def _update_leaf(leaf):
        if hasattr(leaf, "replace") and hasattr(leaf, "value"):
          replace_kwargs = {}
          if hasattr(leaf, "get_metadata"):
            replace_kwargs.update(leaf.get_metadata())

          replace_kwargs[nnx.PARTITION_NAME] = metadata_axis_name
          replace_kwargs["param_scan_axis"] = axis

          for key in ["sharding", "out_sharding", "kernel_axes", "sharding_names"]:
            val = getattr(leaf, key, None)
            if val is None and key in replace_kwargs:
              val = replace_kwargs[key]

            if val is not None:
              if isinstance(val, str):
                val = (val,)
              if isinstance(val, tuple):
                l = list(val)
                # Safely insert the scan axis into the logical axes string
                if metadata_axis_name not in l:
                  insert_idx = min(axis, len(l))
                  l.insert(insert_idx, metadata_axis_name)
                  replace_kwargs[key] = tuple(l)

          return leaf.replace(**replace_kwargs)
        return leaf

      # We must use a custom is_leaf to catch the VariableState instances
      return jax.tree.map(_update_leaf, state, is_leaf=lambda x: hasattr(x, "replace") and hasattr(x, "value"))

    stacked_params = _add_scan_metadata(stacked_params, scan_axis)
    stacked_rest = _add_scan_metadata(stacked_rest, 0)

    return nnx.merge(layer_graphdef, stacked_params, stacked_rest)

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

  def _apply_layers_sequentially(self, layers, x_in, *args, length: int, kv_caches_stacked=None, **kwargs):
    """Runs the layer stack using nnx.scan.

    Args:
      layers: The stacked NNX module whose params are scanned over.
      x_in: The carry (hidden state) fed into the first layer.
      *args: Positional args broadcast to every layer call.
      length: Number of scan iterations (= number of layers).
      kv_caches_stacked: Optional pytree whose leaves have shape [num_layers, ...].
        When provided, the i-th slice is passed as `kv_cache=` to layer i and the
        updated caches are returned as a third element of the tuple.
      **kwargs: Keyword args forwarded to the layer (filtered by the layer signature).

    Returns:
      (final_carry, updated_layers) when kv_caches_stacked is None.
      (final_carry, updated_layers, returned_kv_stacked) otherwise.
    """
    if length == 0:
      return x_in, layers, kv_caches_stacked if kv_caches_stacked is not None else None
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

    use_kv = kv_caches_stacked is not None

    def layer_fn(carry, scanned_vars):
      # Unpack the sliced variables for THIS layer
      if use_kv:
        current_params, current_state, kv_cache_layer = scanned_vars
      else:
        current_params, current_state = scanned_vars
        kv_cache_layer = None

      if self.config.parameter_memory_host_offload:
        current_params = jax.tree.map(lambda x: jax.device_put(x, max_utils.device_space()), current_params)

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
      new_current_state = nnx.state(layer)

      if use_kv:
        return new_carry, (new_current_state, updated_kv)
      return new_carry, new_current_state

    layer_fn = jax.checkpoint(layer_fn, policy=policy, prevent_cse=prevent_cse)

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
        current_carry, (_, updated_kv) = layer_fn(current_carry, (current_params, current_state, kv_caches_list[i]))

        # Update the list in-place (mutates the list passed by reference)
        kv_caches_list[i] = updated_kv

      # We don't need to rebuild scanned_state or return it because during
      # inference with vLLM, parameters do not change and we don't need intermediates.
      return current_carry, layers, None
    else:
      final_carry, scanned_state = jax.lax.scan(layer_fn, x_in, (params, state))
      returned_kv_stacked = None

    if scan_axis != 0:
      new_params, new_rest = scanned_state.split(nnx.Param, ...)
      new_params = jax.tree.map(lambda x: jnp.moveaxis(x, 0, scan_axis), new_params)
      scanned_state = nnx.merge_state(new_params, new_rest)

    nnx.update(layers, scanned_state)
    return final_carry, layers, returned_kv_stacked if use_kv else None

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
        DecoderBlockType.GEMMA4,
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
          normalizations.RMSNorm, num_features=num_features, shard_mode=self.config.shard_mode, rngs=rngs
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
      audio_embeddings = multimodal_input.audio_embeddings
      audio_masks = multimodal_input.audio_masks

      if image_embeddings is not None and cfg.use_multimodal:
        if cfg.model_name in [
            "gemma3-4b",
            "gemma3-12b",
            "gemma3-27b",
            "gemma4-26b",
            "gemma4-31b",
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
          lambda x: jax.lax.dynamic_slice_in_dim(x, current_idx, scan_length, axis=scan_axis), params
      )
      chunk_rest = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, current_idx, scan_length, axis=0), rest)
      chunk_stack = nnx.merge(graphdef, chunk_params, chunk_rest)

      # Apply sequentially
      y, chunk_stack, _ = self._apply_layers_sequentially(
          chunk_stack, y, *args, length=scan_length, **kwargs.get("layer_kwargs", {})
      )

      # Update the original stack state
      new_state = nnx.state(chunk_stack)
      new_params, new_rest = new_state.split(nnx.Param, ...)

      updated_params = jax.tree.map(
          lambda s, new_s: jax.lax.dynamic_update_slice_in_dim(s, new_s, current_idx, axis=scan_axis), params, new_params
      )
      updated_rest = jax.tree.map(
          lambda s, new_s: jax.lax.dynamic_update_slice_in_dim(s, new_s, current_idx, axis=0), rest, new_rest
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
            chunk_stack, y, *args, length=scan_length, **kwargs.get("layer_kwargs", {})
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
      page_state: None | page_manager.PageState = None,
      multimodal_input: None | Any = None,
      kv_caches: list[jax.Array] | None = None,
      attention_metadata=None,
      deepstack_visual_embeds: None | list[jnp.ndarray] = None,
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

    if cfg.decoder_block in (DecoderBlockType.GEMMA3, DecoderBlockType.GEMMA4):
      layer_kwargs["bidirectional_mask"] = bidirectional_mask

    if attention_metadata is not None:
      layer_kwargs["attention_metadata"] = attention_metadata

    if cfg.scan_layers:
      if self.is_deepseek:
        layer_kwargs = {
            "previous_chunk": previous_chunk,
            "page_state": page_state,
            "slot": slot,
        }

        if cfg.engram_layers:
          common_kwargs = {
              "layer_kwargs": layer_kwargs,
              "decoder_input_tokens": decoder_input_tokens,
          }

          y = self._apply_interleaved_scanned_layers(
              y, "dense_layers", 0, cfg.first_num_dense_layers, cfg.engram_layers, *layer_args, **common_kwargs
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
              self.dense_layers, y, *layer_args, length=cfg.first_num_dense_layers, **layer_kwargs
          )

          num_moe = cfg.num_decoder_layers - cfg.first_num_dense_layers

          if cfg.use_batch_split_schedule:
            policy = self.get_remat_policy()
            mock_params = self._build_linen_params(self.moe_layers)

            if cfg.use_qwix_quantization:
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
                self.moe_layers, y, *layer_args, length=num_moe, **layer_kwargs
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
            page_state,
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
            page_state,
            slot,
        )
      else:
        scan_length = int(cfg.num_decoder_layers / cfg.inhomogeneous_layer_cycle_interval)
        if kv_caches is not None:
          # Pass the kv_caches list directly to avoid copying in jnp.stack,
          # which breaks vLLM PagedAttention in-place memory updates.
          # The _apply_layers_sequentially function will handle it by statically unrolling.
          y, self.layers, _ = self._apply_layers_sequentially(
              self.layers, y, *layer_args, length=scan_length, kv_caches_stacked=kv_caches, **layer_kwargs
          )
          # kv_caches list is updated in-place inside _apply_layers_sequentially
        else:
          y, self.layers, _ = self._apply_layers_sequentially(
              self.layers, y, *layer_args, length=scan_length, **layer_kwargs
          )
    else:
      prevent_cse = maxtext_utils.should_prevent_cse_in_remat(cfg)

      # Hoisted function to preserve XLA cache ID
      def pure_layer_fn(graphdef, state_in, y_in, kv_in):

        if cfg.parameter_memory_host_offload:
          state_in = jax.tree.map(lambda x: jax.device_put(x, max_utils.device_space()), state_in)

        merged_layer = nnx.merge(graphdef, state_in)
        out_y, out_kv = merged_layer(y_in, *layer_args, kv_cache=kv_in, **layer_kwargs)
        return out_y, out_kv, nnx.state(merged_layer)

      checkpointed_fn = jax.checkpoint(pure_layer_fn, policy=policy, prevent_cse=prevent_cse)

      for lyr, layer in enumerate(self.layers):
        graphdef, state = nnx.split(layer)
        if kv_caches is not None:
          if cfg.decoder_block == DecoderBlockType.QWEN3_NEXT:
            if (lyr + 1) % cfg.inhomogeneous_layer_cycle_interval == 0:
              kv_cache = (kv_caches["key_cache"][lyr], kv_caches["value_cache"][lyr])
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
          if cfg.decoder_block == DecoderBlockType.QWEN3_NEXT:
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
    elif (cfg.use_indexer and not cfg.indexer_sparse_training) and self.model_mode == MODEL_MODE_TRAIN:
      logits = None

    # When vocab tiling is enabled in training mode, full logits won't generate to reduce memory
    # Instead, we keep track on the hidden states, which has smaller size compared to full logits
    elif cfg.num_vocab_tiling > 1 and self.model_mode == MODEL_MODE_TRAIN:
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
      y, self.layers, _ = self._apply_layers_sequentially(self.layers, y, *layer_args, length=scan_length, **layer_kwargs)

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
      nnx.update(self.layers_remainder, new_state)

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
      page_state,
      slot,
  ):
    """Applies Gemma4 scanned decoder blocks, handling main scan and remainders."""

    cfg = self.config

    # Define the repeating pattern length and calculate how many full blocks to scan
    attention_pattern_length = len(gemma4.GEMMA4_ATTENTION_PATTERN)
    scan_length = cfg.num_decoder_layers // attention_pattern_length

    layer_args = (decoder_segment_ids, decoder_positions, deterministic, model_mode)
    layer_kwargs = {"bidirectional_mask": bidirectional_mask}

    # Apply the main scan over the full blocks
    if scan_length > 0:
      y, self.layers, _ = self._apply_layers_sequentially(self.layers, y, *layer_args, length=scan_length, **layer_kwargs)

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
      nnx.update(self.layers_remainder, new_state)

    return y


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

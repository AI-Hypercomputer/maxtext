#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Transformer models."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any, Callable, Optional


from flax import linen as nn
import functools
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
import common_types
from layers import attentions
from layers import embeddings
from layers import linears
from layers import normalizations, quantizations
from layers import pipeline
from page_manager import PageManager, PageState

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
Attention = attentions.Attention
RMSNorm = normalizations.RMSNorm
PositionalEmbedding = embeddings.PositionalEmbedding
Quant = quantizations.AqtQuantization

# ------------------------------------------------------------------------------
# The network: Decoder & Transformer Definitions
# ------------------------------------------------------------------------------


class DecoderLayer(nn.Module):
  """Transformer decoder layer."""

  config: Config
  mesh: Mesh
  quant: Optional[Quant] = None

  def setup(self):
    cfg = self.config
    self.is_paged_attention = cfg.attention == "paged"
    self.use_fused_qkv = cfg.fused_qkv

    print(f"\nDecoderLayer.setup() - Initializing layer with attention={cfg.attention}")
    
    # Configure attention parameters based on attention type
    attention_params = {
        "config": self.config,
        "num_query_heads": cfg.num_query_heads,
        "num_kv_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "max_target_length": cfg.max_target_length,
        "max_prefill_predict_length": cfg.max_prefill_predict_length,
        "attention_kernel": cfg.attention,
        "mesh": cfg.mesh,
        "dtype": cfg.dtype,
        "weight_dtype": cfg.weight_dtype,
        "dropout_rate": cfg.dropout_rate,
        "name": "self_attention",
        "float32_qk_product": cfg.float32_qk_product,
        "float32_logits": cfg.float32_logits,
        "quant": self.quant,
        "kv_quant": quantizations.configure_kv_quant(cfg),
    }
    
    # Add parameters specific to standard attention
    if not self.is_paged_attention:
        attention_params.update({
            "prefill_cache_axis_order": tuple([int(i) for i in cfg.prefill_cache_axis_order.split(",")]),
            "ar_cache_axis_order": tuple([int(i) for i in cfg.ar_cache_axis_order.split(",")]),
            "compute_axis_order": tuple([int(i) for i in cfg.compute_axis_order.split(",")]),
            "reshape_q": cfg.reshape_q,
            "use_ragged_attention": cfg.use_ragged_attention,
            "ragged_block_size": cfg.ragged_block_size,
        })
    
    # Create the Attention instance
    self.attention_layer = Attention(**attention_params)

  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      slot=None,
      true_length=None,
      page_state=None,
      key_pages=None,
      value_pages=None,
      layer_id: Optional[int] = None,
  ):
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    lnx = RMSNorm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_self_attention_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )(inputs)
    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_length", "activation_embed"))

    # Verify parameters for paged attention
    if self.is_paged_attention and layer_id is not None and model_mode != common_types.MODEL_MODE_TRAIN:
        print(f"DecoderLayer calling attention_layer - layer_id={layer_id}, mode={model_mode}")
        print(f"  page_state.sequence_lengths shape: {page_state.sequence_lengths.shape if page_state else 'None'}")
        
        if slot is not None:
            print(f"  slot: {slot}")
            if page_state is not None:
                seq_len = page_state.sequence_lengths[layer_id, slot]
                print(f"  Current sequence length: {seq_len}")
    
    attention_lnx = self.attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        model_mode=model_mode,
        page_state=page_state,
        key_pages=key_pages,  # PASS key_pages
        value_pages=value_pages, # PASS value_pages
        page_group_id=slot,
        true_length=true_length,
        layer_id=layer_id,
        use_fused_qkv=self.use_fused_qkv,
    )

    attention_lnx = nn.with_logical_constraint(attention_lnx, ("activation_batch", "activation_length", "activation_embed"))

    # MLP block.
    mlp_lnx = linears.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        config=cfg,
        quant=self.quant,
    )(lnx, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_length", "activation_embed"))

    next_layer_addition = mlp_lnx + attention_lnx

    next_layer_addition_dropped_out = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        next_layer_addition, deterministic=deterministic
    )

    layer_output = next_layer_addition_dropped_out + inputs
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )

    if cfg.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    return layer_output, attention_cache if self.is_paged_attention else None


class SequentialBlockDecoderLayers(nn.Module):
  """Sequential unscanned series of decoder layers."""

  decoder_layer: Any
  num_decoder_layers: int
  config: Config
  mesh: Mesh
  quant: Quant

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, decoder_segment_ids, decoder_positions, deterministic, model_mode) -> jnp.ndarray:
    for lyr in range(self.num_decoder_layers):
      inputs = self.decoder_layer(config=self.config, mesh=self.mesh, name=f"layers_{lyr}", quant=self.quant)(
          inputs,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
      )
    return inputs


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""

  config: Config
  shared_embedding: nn.Module
  mesh: Mesh
  quant: Optional[Quant] = None
  page_manager: Optional[PageManager] = None

  def setup(self):
    """Initialize decoder layer."""
    cfg = self.config
    self.decoder_layer = self.get_decoder_layers()
    self.norm_layer = self.get_norm_layer()

    if self.config.using_pipeline_parallelism:
      pipeline_stage_module = self.get_pipeline_stage_module(self.decoder_layer[0])
      remat_policy = self.get_remat_policy()
      self.pipeline_module = pipeline.Pipeline(
          config=self.config, mesh=self.mesh, layers=pipeline_stage_module, remat_policy=remat_policy
      )

  def get_remat_policy(self):
    cfg = self.config
    if cfg.remat_policy != "none":
      if cfg.remat_policy == "minimal":
        policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
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
        policy = jax.checkpoint_policies.offload_dot_with_no_batch_dims(offload_src="device", offload_dst="pinned_host")
      elif cfg.remat_policy == "custom":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=cfg.tensors_on_device,
            names_which_can_be_offloaded=cfg.tensors_to_offload,
            offload_src="device",
            offload_dst="pinned_host",
        )
      elif cfg.remat_policy == "minimal_flash":
        policy = jax.checkpoint_policies.save_from_both_policies(
            jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
            jax.checkpoint_policies.save_only_these_names(
                "context",
            ),
        )
      elif cfg.remat_policy == "save_out_proj":
        policy = jax.checkpoint_policies.save_only_these_names(
            "out_proj",
        )
      else:
        assert cfg.remat_policy == "full", "Remat policy needs to be on list of remat policies"
        policy = None
      return policy

  def set_remat_policy(self, block_layers, policy):
    RemattedBlockLayers = []
    for block_layer in block_layers:
      layer = nn.remat(
          block_layer,
          prevent_cse=not self.config.scan_layers,
          policy=policy,
          static_argnums=tuple(range(-len(block_layer.__call__.__code__.co_varnames) + 1, 0)),
      )
      RemattedBlockLayers.append(layer)
    return RemattedBlockLayers

  def get_decoder_layers(self):
    if self.config.decoder_block == "default":
      return [DecoderLayer]
    elif self.config.decoder_block == "llama2":
      from layers import llama2

      return [llama2.LlamaDecoderLayer]
    elif self.config.decoder_block == "mistral":
      # TODO(ranran): update to Mistral with sliding window attention
      from layers import mistral

      return [mistral.MistralDecoderLayer]
    elif self.config.decoder_block == "deepseek":
      from layers import deepseek

      return [deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer]
    elif self.config.decoder_block == "gemma":
      from layers import gemma

      return [gemma.GemmaDecoderLayer]
    elif self.config.decoder_block == "gemma2":
      from layers import gemma2

      return [gemma2.Gemma2DecoderLayer]
    elif self.config.decoder_block == "gpt3":
      from layers import gpt3

      return [gpt3.Gpt3DecoderLayer]
    elif self.config.decoder_block == "simple":
      from layers import simple_layer

      return [simple_layer.SimpleDecoderLayer]
    elif self.config.decoder_block == "simple_mlp":
      from layers import simple_layer

      return [simple_layer.SimpleMlpDecoderLayer]
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  def get_norm_layer(self):
    if self.config.decoder_block in ("default", "llama2", "mistral", "deepseek", "gemma", "gemma2", "simple", "simple_mlp"):
      return RMSNorm
    elif self.config.decoder_block == "gpt3":
      from layers import gpt3

      return functools.partial(gpt3.Gpt3LayerNorm, reductions_in_fp32=False, use_bias=True)
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  def scan_decoder_layers(self, cfg, decoder_layer, length, metdata_axis_name, mesh):
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
        in_axes=(
            nn.broadcast,
            nn.broadcast,
            nn.broadcast,
            nn.broadcast,
        ),
        length=length,
        metadata_params={nn.PARTITION_NAME: metdata_axis_name},
    )
    return scan_fn(config=cfg, mesh=mesh, name=metdata_axis_name, quant=self.quant)

  def get_pipeline_stage_module(self, base_stage):
    cfg = self.config
    if cfg.set_remat_policy_on_layers_per_stage:
      policy = self.get_remat_policy()
      base_stage = self.set_remat_policy([base_stage], policy)[0]
    if cfg.num_layers_per_pipeline_stage == 1:
      stage_module = base_stage(config=cfg, mesh=self.mesh, quant=self.quant)
    elif cfg.scan_layers:
      stage_module = self.scan_decoder_layers(
          cfg, base_stage, cfg.num_layers_per_pipeline_stage, "layers_per_stage", self.mesh
      )
    else:
      stage_module = SequentialBlockDecoderLayers(
          decoder_layer=base_stage,
          num_decoder_layers=cfg.num_layers_per_pipeline_stage,
          config=cfg,
          mesh=self.mesh,
          quant=self.quant,
      )
    return stage_module

  @nn.compact
  def __call__(
    self,
    decoder_input_tokens,
    decoder_positions,
    decoder_segment_ids=None,
    deterministic=False,
    model_mode=common_types.MODEL_MODE_TRAIN,
    slot=None,
    true_length=None,
    page_state=None,
    layer_id=None,
  ):
    cfg = self.config
    mesh = self.mesh
    assert decoder_input_tokens.ndim == 2

    y = self.shared_embedding(decoder_input_tokens.astype("int32"))
    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    if cfg.use_untrainable_positional_embedding:
        y = PositionalEmbedding(cfg.base_emb_dim)(y, decoder_positions)

    if cfg.trainable_position_size > 0:
        y += Embed(
            num_embeddings=cfg.trainable_position_size,
            features=cfg.emb_dim,
            dtype=cfg.dtype,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name="position_embedder",
            config=cfg,
        )(decoder_positions)

    # ONLY try to get variables from cache if they are available.
    key_pages = None
    value_pages = None
    if cfg.attention == 'paged' and model_mode != common_types.MODEL_MODE_TRAIN and 'cache' in self.variables:
      key_pages = self.variable("cache", "key_pages", lambda: None).value
      value_pages = self.variable("cache", "value_pages", lambda: None).value
      if page_state is None:  # Get from cache if function arg is None
          page_state = self.variable("cache", "page_state", lambda: None).value


    if cfg.using_pipeline_parallelism:
        if cfg.pipeline_fsdp_ag_once:
            partition_spec = self.pipeline_module.get_weight_sharding(
                y,
                decoder_segment_ids,
                decoder_positions,
                deterministic,
                model_mode,
            )
        else:
            partition_spec = None
        y = self.pipeline_module(
            y,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            partition_spec=partition_spec,
        )
    else:
        # Iterate through layers
        for lyr in range(cfg.num_decoder_layers):
            RemattedBlockLayers = self.set_remat_policy(self.decoder_layer, self.get_remat_policy())

            current_layer_class = RemattedBlockLayers[lyr % len(RemattedBlockLayers)]
            layer_instance = current_layer_class(config=cfg, mesh=mesh, name=f"layers_{lyr}", quant=self.quant)
            # Always pass lyr
            y, layer_cache = layer_instance(
                y,
                decoder_segment_ids,
                decoder_positions,
                deterministic,
                model_mode,
                slot=slot,
                true_length=true_length,
                page_state=page_state,
                key_pages=key_pages,
                value_pages=value_pages,
                layer_id=lyr,
            )
            if cfg.attention == "paged" and layer_cache is not None and model_mode != common_types.MODEL_MODE_TRAIN:
              key_pages = layer_cache['key_pages']
              value_pages = layer_cache['value_pages']
              page_state = layer_cache['page_state']


    y = self.get_norm_layer()(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="decoder_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )(y)
    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)

    if cfg.logits_via_embedding:
        logits = self.shared_embedding.attend(y)
        if self.config.normalize_embedding_logits:
            logits = logits / jnp.sqrt(y.shape[-1])
        if cfg.final_logits_soft_cap:
            logits = logits / cfg.final_logits_soft_cap
            logits = jnp.tanh(logits) * cfg.final_logits_soft_cap
    else:
        logits = linears.DenseGeneral(
            cfg.vocab_size,
            weight_dtype=cfg.weight_dtype,
            dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,
            kernel_axes=("embed", "vocab"),
            name="logits_dense",
            matmul_precision=self.config.matmul_precision,
        )(
            y
        )
    logits = nn.with_logical_constraint(
        logits,
        ("activation_embed_and_logits_batch", "activation_length", "activation_vocab"),
    )
    if self.config.cast_logits_to_fp32:
        logits = logits.astype(jnp.float32)

    # Sow ONLY if it is paged attention, NOT training, AND we are not in eval shape.
    if cfg.attention == 'paged' and model_mode != common_types.MODEL_MODE_TRAIN and 'cache' in self.variables:
      self.sow('cache', 'page_state', page_state)
      self.sow('cache', 'key_pages', key_pages)
      self.sow('cache', 'value_pages', value_pages)

    return logits

class Transformer(nn.Module):
  """An decoder-only Transformer model."""

  # Make new attributes required, so that all Transformer dependencies (train, decode, compile, etc) will error instead of silently use defaults.
  # pylint: disable=attribute-defined-outside-init
  config: Config
  mesh: Mesh
  quant: Quant

  def setup(self):
    """Initialize shared_embedding & decoder layers."""
    cfg = self.config
    mesh = self.mesh
    
    self.shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        name="token_embedder",
        config=cfg,
    )

    # Create fixed shape page state during initialization
    if cfg.attention == "paged":
        # Initialize page state once in setup, not dynamically in __call__
        self.page_state = {
            "decoder": {
                f"layers_{i}": {
                    "key_pages": jnp.zeros(
                        (cfg.num_pages, cfg.tokens_per_page, cfg.num_kv_heads, cfg.head_dim),
                        dtype=cfg.dtype
                    ),
                    "value_pages": jnp.zeros(
                        (cfg.num_pages, cfg.tokens_per_page, cfg.num_kv_heads, cfg.head_dim),
                        dtype=cfg.dtype
                    ),
                }
                for i in range(cfg.num_decoder_layers)
            },
            "page_manager": PageState(
                page_status=jnp.zeros((cfg.num_decoder_layers, cfg.num_pages), dtype=jnp.int32),
                page_map=jnp.full(
                    (cfg.num_decoder_layers, int(cfg.per_device_batch_size * jax.device_count()), 
                     cfg.max_pages_per_group), -1, dtype=jnp.int32
                ),
                sequence_lengths=jnp.zeros(
                    (cfg.num_decoder_layers, int(cfg.per_device_batch_size * jax.device_count())), 
                    dtype=jnp.int32
                ),
                num_pages_used=jnp.zeros(
                    (cfg.num_decoder_layers, int(cfg.per_device_batch_size * jax.device_count())), 
                    dtype=jnp.int32
                ),
                current_page=jnp.full(
                    (cfg.num_decoder_layers, int(cfg.per_device_batch_size * jax.device_count())), 
                    -1, dtype=jnp.int32
                ),
                current_page_position=jnp.zeros(
                    (cfg.num_decoder_layers, int(cfg.per_device_batch_size * jax.device_count())), 
                    dtype=jnp.int32
                ),
            )
        }
    else:
        self.page_state = None

    self.decoder = Decoder(
        config=cfg,
        shared_embedding=self.shared_embedding,
        mesh=mesh,
        quant=self.quant,
    )

  def __call__(
    self,
    decoder_input_tokens,
    decoder_positions,
    decoder_segment_ids=None,
    enable_dropout=True,
    model_mode=common_types.MODEL_MODE_TRAIN,
    slot=None,
    true_length=None,
    page_state=None,
    layer_id=None,
  ):
    """Applies Transformer decoder-branch on encoded-input and target."""
    if decoder_segment_ids is not None and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError(
            f"During autoregressive decoding we assume the tokens are in the active sequence"
            f" which is always {common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR}."
        )

    if self.config.attention == "paged" and page_state is None and model_mode != common_types.MODEL_MODE_TRAIN:
        # Create a dummy page state for debugging
        page_state = self.page_state["page_manager"]
        print(f"Using default page state with shape: {page_state.sequence_lengths.shape}")
        
    logits = self.decoder(
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=not enable_dropout,
        model_mode=model_mode,
        slot=slot,
        true_length=true_length,
        page_state=page_state,
        layer_id=layer_id,
    )
    return logits
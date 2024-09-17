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
import dataclasses
from inspect import signature
import time

from flax import linen as nn
from flax import nnx
from flax.nnx import bridge
import functools
import jax
import jax.numpy as jnp
import common_types
from nnx_layers import attentions
from nnx_layers import embeddings
#from layers import embeddings
#from layers import linears
from nnx_layers import linears
#from layers import normalizations, quantizations
from nnx_layers import normalizations
from layers import quantizations
from layers import pipeline


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

@jax.tree_util.register_static
class StaticStr(str): pass


@dataclasses.dataclass
class DecoderLayer(nnx.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: Config
  mesh: Mesh
  quant: Optional[Quant] = None
  name: str = "decoder_layer"
  rngs: nnx.Rngs | None = None
  
  def __post_init__(self) -> None:
    cfg = self.config
    mesh = self.mesh

    self.input_norm = RMSNorm(
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_self_attention_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.attention_layer = Attention(
      config=self.config,
      num_query_heads=cfg.num_query_heads,
      num_kv_heads=cfg.num_kv_heads,
      head_dim=cfg.head_dim,
      max_target_length=cfg.max_target_length,
      max_prefill_predict_length=cfg.max_prefill_predict_length,
      attention_kernel=cfg.attention,
      mesh=mesh,
      dtype=cfg.dtype,
      weight_dtype=cfg.weight_dtype,
      dropout_rate=cfg.dropout_rate,
      name="self_attention",
      quant=self.quant,
      kv_quant=quantizations.configure_kv_quant(cfg),
      prefill_cache_axis_order=tuple([int(i) for i in cfg.prefill_cache_axis_order.split(",")]),
      ar_cache_axis_order=tuple([int(i) for i in cfg.ar_cache_axis_order.split(",")]),
      compute_axis_order=tuple([int(i) for i in cfg.compute_axis_order.split(",")]),
      reshape_q=cfg.reshape_q,
      rngs=self.rngs
    )

    self.mlp_block = linears.MlpBlock(
        input_dim=cfg.emb_dim,
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        config=cfg,
        quant=self.quant,
        rngs=self.rngs
    )
    self.out_dropout = nnx.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,), 
                                   rngs=self.rngs)

  #@nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
  ):
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = self.input_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_length", "activation_embed"))

    attention_lnx = self.attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )
    attention_lnx = nn.with_logical_constraint(attention_lnx, ("activation_batch", "activation_length", "activation_embed"))

    # MLP block.
    mlp_lnx = self.mlp_block(lnx, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_length", "activation_embed"))

    next_layer_addition = mlp_lnx + attention_lnx
    
    next_layer_addition_dropped_out = self.out_dropout(
      next_layer_addition, deterministic=deterministic)

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

    return (layer_output, None) if cfg.scan_layers else layer_output

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
    
####################################################################################################

@dataclasses.dataclass
class Decoder(nnx.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""

  config: Config
  shared_embedding: nn.Module
  mesh: Mesh
  quant: Optional[Quant] = None
  rngs: nnx.Rngs | None = None

  def get_decoder_layer(self):
    if self.config.decoder_block == "default":
      return DecoderLayer
    elif self.config.decoder_block == "llama2":
      from layers import llama2

      return llama2.LlamaDecoderLayer
    elif self.config.decoder_block == "mistral":
      # TODO(ranran): update to Mistral with sliding window attention
      from nnx_layers import mistral

      return mistral.MistralDecoderLayer
    elif self.config.decoder_block == "gemma":
      from layers import gemma

      return gemma.GemmaDecoderLayer
    elif self.config.decoder_block == "gemma2":
      from layers import gemma2

      return gemma2.Gemma2DecoderLayer
    elif self.config.decoder_block == "gpt3":
      from layers import gpt3

      return gpt3.Gpt3DecoderLayer
    elif self.config.decoder_block == "simple":
      from layers import simple_layer

      return simple_layer.SimpleDecoderLayer
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  def get_norm_layer(self):
    if self.config.decoder_block in ("default", "llama2", "mistral", "gemma", "gemma2", "simple"):
      return RMSNorm
    elif self.config.decoder_block == "gpt3":
      from layers import gpt3

      return functools.partial(gpt3.Gpt3LayerNorm, reductions_in_fp32=False, use_bias=True)
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  def make_decoder_layers(self, cfg, decoder_layer, length, metdata_axis_name, mesh):
    #initializing = self.is_mutable_collection("params")
    initializing = True
    params_spec = cfg.param_scan_axis if initializing else ScanIn(cfg.param_scan_axis)
    cache_spec = 0
    
    # convert keyword args to positional arguments, nnx.remat layer only supports positional
    args = dict(config=cfg, mesh=mesh, quant=self.quant, rngs=self.rngs)
    arg_kws = list(signature(decoder_layer.__init__).parameters.keys())[1:] # skip "self"
    args = [args.get(kw, None) for kw in arg_kws]
    # call example:
    # decoder_layers(y, decoder_segment_ids, decoder_positions, deterministic, model_mode)
    return nnx.Scan.constructor(decoder_layer, length=cfg.base_num_decoder_layers, 
                                in_axes=(0, 0, None, None, None, None))(*args)
    
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
    #return bridge.ToNNX(scan_fn(config=cfg, mesh=mesh, name="layers", quant=self.quant), rngs=self.rngs)
    return scan_fn(config=cfg, mesh=mesh, name="layers", quant=self.quant)

  def scan_decoder_layers(self, cfg, decoder_layer, length, metdata_axis_name, mesh):
    #initializing = self.is_mutable_collection("params")
    initializing = True
    params_spec = cfg.param_scan_axis if initializing else ScanIn(cfg.param_scan_axis)
    cache_spec = 0
    #scan_fn = nn.scan(
    scan_fn = nnx.scan(
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
    #return bridge.ToNNX(scan_fn(config=cfg, mesh=mesh, name="layers", quant=self.quant), rngs=self.rngs)
    return scan_fn(config=cfg, mesh=mesh, name="layers", quant=self.quant)

  def __post_init__(self):
    cfg = self.config
    mesh = self.mesh

    # input dropout ################################################################################
    self.input_dropout = nnx.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,), 
                                     rngs=self.rngs)
    # input dropout ################################################################################

    # position embedding ###########################################################################
    if cfg.use_untrainable_positional_embedding:
      self.positional_embedding = PositionalEmbedding(cfg.emb_dim, rngs=self.rngs)
    if cfg.trainable_position_size > 0:
      self.trainable_position_embedder = Embed(
            num_embeddings=cfg.trainable_position_size,
            features=cfg.emb_dim,
            dtype=cfg.dtype,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name="position_embedder",
            config=cfg,
            rngs=self.rngs,
        )
    # position embedding ###########################################################################


    # block decoder layer ##########################################################################
    BlockLayer = self.get_decoder_layer()

    if cfg.remat_policy != "none":
      if cfg.remat_policy == "minimal":
        policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
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

    RemattedBlockLayer = nnx.remat(  # pylint: disable=invalid-name
        BlockLayer,
        prevent_cse=not cfg.scan_layers,
        policy=policy,
        static_argnums=(-1, -2, -3, -4, -5),
    )

    #if cfg.using_pipeline_parallelism:
    #    if cfg.num_layers_per_pipeline_stage == 1:
    #      stage_module = BlockLayer(config=cfg, mesh=mesh, quant=self.quant)
    #    elif cfg.scan_layers:
    #      stage_module = self.scan_decoder_layers(cfg, RemattedBlockLayer, cfg.num_layers_per_pipeline_stage, "layers_per_stage", mesh)
    #    elif not cfg.scan_layers:
    #      stage_module=SequentialBlockDecoderLayers(decoder_layer=RemattedBlockLayer, num_decoder_layers=cfg.num_layers_per_pipeline_stage, config=cfg, mesh=mesh,quant=self.quant)

    #    y = pipeline.Pipeline(config=cfg, mesh=mesh, layers=stage_module, remat_policy=policy)(
    #        y,
    #        decoder_segment_ids,
    #        decoder_positions,
    #        deterministic,
    #        model_mode,
    #    )
    #else:
    if True:
      if cfg.scan_layers:
        self.decoder_layers = self.make_decoder_layers(cfg, RemattedBlockLayer, cfg.num_decoder_layers, "layers", mesh)
      else:
        self.decoder_layers = []
        for lyr in range(cfg.num_decoder_layers):
          arg_keys = list(signature(BlockLayer.__init__).parameters.keys())[1:] # skip "self"
          args = dict(config=cfg, mesh=mesh, quant=self.quant, rngs=self.rngs, name=f"layers_{lyr}")
          args = [args.get(kw, None) for kw in arg_keys]
          self.decoder_layers.append(RemattedBlockLayer(*args))
    # block decoder layer ##########################################################################

    # norm layer ###################################################################################
    #self.norm_layer = bridge.ToNNX(self.get_norm_layer()(
    self.norm_layer = self.get_norm_layer()(
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="decoder_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    #), rngs=self.rngs)
    self.norm_layer_dropout = nnx.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,),
                                          rngs=self.rngs)
    # norm layer ###################################################################################

    # logits mapping ###############################################################################
    if not cfg.logits_via_embedding:
      self.logits_transpose = linears.DenseGeneral(
          cfg.emb_dim,
          cfg.vocab_size,
          weight_dtype=cfg.weight_dtype,
          dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
          kernel_axes=("embed", "vocab"),
          name="logits_dense",
          rngs=self.rngs)
    # logits mapping ###############################################################################

    # output #######################################################################################
    # output #######################################################################################

  #@nn.compact
  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      deterministic=False,
      model_mode=common_types.MODEL_MODE_TRAIN,
  ):
    cfg = self.config
    #mesh = self.mesh
    assert decoder_input_tokens.ndim == 2  # [batch, len]
    
    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype("int32"))
    self.sow(nnx.Intermediate, "rdyro_shared_embedding", y)

    # input dropout ################################################################################
    y = self.input_dropout(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)
    self.sow(nnx.Intermediate, "rdyro_input_droput", y)
    # input dropout ################################################################################

    # position embedding ###########################################################################
    if cfg.use_untrainable_positional_embedding:
      y = self.positional_embedding(y, decoder_positions)

    if cfg.trainable_position_size > 0:
      y += self.trainable_position_embedder(decoder_positions)
    # position embedding ###########################################################################


    # block decoder layer ##########################################################################
    self.sow(nnx.Intermediate, "rdyro_before_scan", y)
    if True:
      if cfg.scan_layers:
        #y, _ = nnx.scan(lambda y, layer: layer(y, decoder_segment_ids, decoder_positions, deterministic, model_mode))(y, self.decoder_layers)
        y, _ = self.decoder_layers(y, decoder_segment_ids, decoder_positions, deterministic, StaticStr(model_mode))
        #y, _ = self.decoder_layers(y, decoder_segment_ids, decoder_positions, deterministic, model_mode)
      else:
        layer: nnx.Module
        for lyr, layer in enumerate(self.decoder_layers):
          #layer.lazy_init(y, decoder_segment_ids, decoder_positions, deterministic, model_mode)
          y = layer(y, decoder_segment_ids, decoder_positions, deterministic, model_mode)
          self.sow(nnx.Intermediate, f"rdyro_after_layer_{lyr}", y)
    self.sow(nnx.Intermediate, "rdyro_after_scan", y)
    # block decoder layer ##########################################################################

    # norm layer ###################################################################################
    #self.norm_layer.lazy_init(y)
    y = self.norm_layer(y)
    y = self.norm_layer_dropout(y, deterministic=deterministic)
    self.sow(nnx.Intermediate, "rdyro_after_norm", y)
    # norm layer ###################################################################################

    # logits mapping ###############################################################################
    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      if self.config.normalize_embedding_logits:
        # Correctly normalize pre-softmax logits for this shared case.
        logits = logits / jnp.sqrt(y.shape[-1])
      if cfg.final_logits_soft_cap:
        logits = logits / cfg.final_logits_soft_cap
        logits = jnp.tanh(logits) * cfg.final_logits_soft_cap
    else:
      #self.logits_transpose.lazy_init(y)
      logits = self.logits_transpose(y)  # We do not quantize the logits matmul.
    self.sow(nnx.Intermediate, "rdyro_after_logits", logits)
    # logits mapping ###############################################################################

    # output #######################################################################################
    logits = nn.with_logical_constraint(logits, ("activation_embed_and_logits_batch", "activation_length", "activation_vocab"))
    logits = logits.astype(jnp.float32)
    return logits
    # output #######################################################################################


@dataclasses.dataclass
class Transformer(nnx.Module):
  """An decoder-only Transformer model."""

  # Make new attributes required, so that all Transformer dependencies (train, decode, compile, etc) will error instead of silently use defaults.
  # pylint: disable=attribute-defined-outside-init
  config: Config
  mesh: Mesh
  quant: Quant
  rngs: nnx.Rngs | None = None

  def __post_init__(self):
    """Initialize shared_embedding & decoder layers."""
    if self.rngs is None:
      self.rngs = nnx.Rngs(time.time_ns() % 2 ** 31)
    cfg = self.config
    mesh = self.mesh
    self.shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        #name="token_embedder",
        config=cfg,
        rngs=self.rngs,
    )

    self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding, mesh=mesh, 
                           quant=self.quant, rngs=self.rngs) 

  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      enable_dropout=True,
      model_mode=common_types.MODEL_MODE_TRAIN,
  ):
    """Applies Transformer decoder-branch on encoded-input and target."""

    if decoder_segment_ids is not None and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      raise ValueError(
          f"During autoregressive decoding we assume the tokens are in the active sequence"
          f" which is always {common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR}."
      )

    logits = self.decoder(
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=not enable_dropout,
        model_mode=model_mode,
    )
    return logits

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
import common_types
from layers import attentions
from layers import embeddings
from layers import linears
from layers import normalizations, quantizations
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


class DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: Config
  mesh: Mesh
  quant: Optional[Quant] = None

  @nn.compact
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
    lnx = RMSNorm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_self_attention_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )(inputs)
    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_length", "activation_embed"))

    attention_layer = Attention(
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
        quantize_kvcache=cfg.quantize_kvcache,
        prefill_cache_axis_order=tuple([int(i) for i in cfg.prefill_cache_axis_order.split(",")]),
        ar_cache_axis_order=tuple([int(i) for i in cfg.ar_cache_axis_order.split(",")]),
        compute_axis_order=tuple([int(i) for i in cfg.compute_axis_order.split(",")]),
        reshape_q=cfg.reshape_q,
        kv_quant_axis=cfg.kv_quant_axis,
    )

    attention_lnx = attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
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

    return layer_output, None if cfg.scan_layers else layer_output

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

  def get_decoder_layer(self):
    if self.config.decoder_block == "default":
      return DecoderLayer
    elif self.config.decoder_block == "llama2":
      from layers import llama2

      return llama2.LlamaDecoderLayer
    elif self.config.decoder_block == "mistral":
      # TODO(ranran): update to Mistral with sliding window attention
      from layers import mistral

      return mistral.MistralDecoderLayer
    elif self.config.decoder_block == "gemma":
      from layers import gemma

      return gemma.GemmaDecoderLayer
    elif self.config.decoder_block == "gpt3":
      from layers import gpt3

      return gpt3.Gpt3DecoderLayer
    elif self.config.decoder_block == "simple":
      from layers import simple_layer

      return simple_layer.SimpleDecoderLayer
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  def get_norm_layer(self):
    if self.config.decoder_block in ("default", "llama2", "mistral", "gemma"):
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
    return scan_fn(config=cfg, mesh=mesh, name="layers", quant=self.quant)

  @nn.compact
  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      deterministic=False,
      model_mode=common_types.MODEL_MODE_TRAIN,
  ):
    cfg = self.config
    mesh = self.mesh
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    # [batch, length] -> [batch, length, emb_dim]
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
      else:
        assert cfg.remat_policy == "full", "Remat policy needs to be on list of remat policies"
        policy = None

    RemattedBlockLayer = nn.remat(  # pylint: disable=invalid-name
        BlockLayer,
        prevent_cse=not cfg.scan_layers,
        policy=policy,
        static_argnums=(-1, -2, -3, -4, -5),
    )
    if cfg.using_pipeline_parallelism:
        if cfg.num_layers_per_pipeline_stage == 1:
          stage_module = BlockLayer(config=cfg, mesh=mesh, quant=self.quant)
        elif cfg.scan_layers:
          stage_module = self.scan_decoder_layers(cfg, RemattedBlockLayer, cfg.num_layers_per_pipeline_stage, "layers_per_stage", mesh)
        elif not cfg.scan_layers:
          stage_module=SequentialBlockDecoderLayers(decoder_layer=RemattedBlockLayer, num_decoder_layers=cfg.num_layers_per_pipeline_stage, config=cfg, mesh=mesh,quant=self.quant)

        y = pipeline.Pipeline(config=cfg, mesh=mesh, layers=stage_module, remat_policy=policy)(
            y,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
        )
    else:
      if cfg.scan_layers:
        y, _ = self.scan_decoder_layers(cfg, RemattedBlockLayer, cfg.num_decoder_layers, "layers", mesh)(
            y,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
        )
      else:
        for lyr in range(cfg.num_decoder_layers):
          y = RemattedBlockLayer(config=cfg, mesh=mesh, name=f"layers_{lyr}", quant=self.quant)(
              y,
              decoder_segment_ids,
              decoder_positions,
              deterministic,
              model_mode,
          )

    y = self.get_norm_layer()(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="decoder_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )(y)
    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      if self.config.normalize_embedding_logits:
        # Correctly normalize pre-softmax logits for this shared case.
        logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = linears.DenseGeneral(
          cfg.vocab_size,
          weight_dtype=cfg.weight_dtype,
          dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
          kernel_axes=("embed", "vocab"),
          name="logits_dense",
      )(
          y
      )  # We do not quantize the logits matmul.
    logits = nn.with_logical_constraint(logits, ("activation_embed_and_logits_batch", "activation_length", "activation_vocab"))
    logits = logits.astype(jnp.float32)
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
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        name="token_embedder",
        config=cfg,
    )

    self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding, mesh=mesh, quant=self.quant)

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

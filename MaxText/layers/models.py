# models.py (Corrected for TracerBoolConversionError)

import functools
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
import common_types
from layers import attentions
from layers import embeddings
from layers import linears
from layers import normalizations, quantizations
from layers import pipeline
from page_manager import PageManager, PageManagerLayer

from typing import Optional

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
      model_mode=common_types.MODEL_MODE_PREFILL,
      slot: Optional[int] = None,
      true_length: Optional[int] = None,
      page_manager: Optional[PageManager] = None,
  ):
    cfg = self.config
    mesh = self.mesh

    # Always do the initial constraint and layer norm
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
        float32_qk_product=cfg.float32_qk_product,
        float32_logits=cfg.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
        prefill_cache_axis_order=tuple([int(i) for i in cfg.prefill_cache_axis_order.split(",")]),
        ar_cache_axis_order=tuple([int(i) for i in cfg.ar_cache_axis_order.split(",")]),
        compute_axis_order=tuple([int(i) for i in cfg.compute_axis_order.split(",")]),
        reshape_q=cfg.reshape_q,
    )

    attention_lnx = attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        page_manager=page_manager,
        slot=slot,
        true_length=true_length
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

class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""

  config: Config
  shared_embedding: nn.Module
  mesh: Mesh
  quant: Optional[Quant] = None

  def setup(self):
        cfg = self.config

        # 1. Get base decoder layer classes
        decoder_layer_classes = self.get_decoder_layers()

        # 2. Apply remat transformation to the classes if needed
        if self.config.remat_policy != "none":
            transformed_classes = []
            for base_class in decoder_layer_classes:
                transformed_class = nn.remat(
                    base_class,
                    prevent_cse=not self.config.scan_layers,
                    policy=self.get_remat_policy(),
                    static_argnums=(3, 4),
                )
                transformed_classes.append(transformed_class)
            decoder_layer_classes = transformed_classes

        # 3. Create all layers at once instead of appending
        self.decoder_layers = [
            decoder_layer_classes[lyr % len(decoder_layer_classes)](
                config=cfg,
                mesh=self.mesh,
                name=f"layers_{lyr}",
                quant=self.quant
            )
            for lyr in range(cfg.num_decoder_layers)
        ]

        # 4. Initialize norm layer
        self.norm_layer = self.get_norm_layer()(
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            name="decoder_norm",
            epsilon=cfg.normalization_layer_epsilon,
            kernel_axes=("norm",),)

        # 5. Initialize page managers if needed
        if cfg.attention == "paged":
            self.page_managers = [
                PageManagerLayer(
                    config=cfg,
                    name=f"page_manager_{i}"
                )
                for i in range(cfg.num_decoder_layers)
            ]
        else:
            self.page_managers = None

        # 6. Optional: Initialize pipeline module if needed
        if self.config.using_pipeline_parallelism:
            pipeline_stage_module = self.get_pipeline_stage_module(self.decoder_layer[0])
            remat_policy = self.get_remat_policy()
            self.pipeline_module = pipeline.Pipeline(
                config=self.config,
                mesh=self.mesh,
                layers=pipeline_stage_module,
                remat_policy=remat_policy
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
      layer = nn.remat(  # pylint: disable=invalid-name
          block_layer,
          prevent_cse=not self.config.scan_layers,
          policy=policy,
          static_argnums=(3, 4),
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
            nn.broadcast,  # inputs
            nn.broadcast,  # decoder_segment_ids
            nn.broadcast,  # decoder_positions
            nn.broadcast,  # deterministic
            nn.broadcast,  # model_mode
            nn.broadcast,  # slot
            nn.broadcast,  # true_length
            nn.broadcast,  # page_manager
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
      model_mode=common_types.MODEL_MODE_PREFILL,
      slot=None,
      true_length=None,
  ):
      cfg = self.config
      mesh = self.mesh
      assert decoder_input_tokens.ndim == 2  # [batch, len]

      # [batch, length] -> [batch, length, emb_dim]
      y = self.shared_embedding(decoder_input_tokens.astype("int32"))
      y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
          y, deterministic=deterministic
      )
      y = y.astype(cfg.dtype)

      # Optional positional embedding configurations
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

      # Pipeline parallelism pathway
      if cfg.using_pipeline_parallelism:
          if cfg.pipeline_fsdp_ag_once:
              partition_spec = self.pipeline_module.get_weight_sharding(
                  y, decoder_segment_ids, decoder_positions, deterministic, model_mode
              )
          else:
              partition_spec = None
          y = self.pipeline_module(
              y,
              decoder_segment_ids,
              decoder_positions,
              deterministic,
              model_mode,
              partition_spec=partition_spec
          )
      else:
          # Process through decoder layers sequentially
          for lyr in range(cfg.num_decoder_layers):
              # Get the appropriate page manager if using paged attention
              page_manager = (
                  self.page_managers[lyr] if cfg.attention == "paged" else None
              )

              # Use pre-created layer instance from setup()
              y = self.decoder_layers[lyr](
                  y,
                  decoder_segment_ids,
                  decoder_positions,
                  deterministic,
                  model_mode=model_mode,
                  slot=slot,
                  true_length=true_length,
                  page_manager=page_manager
              )

      # Apply final normalization
      y = self.norm_layer(y)

      # Apply dropout after normalization
      y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
          y, deterministic=deterministic
      )

      # [batch, length, emb_dim] -> [batch, length, vocab_size]
      if cfg.logits_via_embedding:
          # Use the transpose of embedding matrix for logit transform
          logits = self.shared_embedding.attend(y)

          # Optional normalization of logits
          if self.config.normalize_embedding_logits:
              logits = logits / jnp.sqrt(y.shape[-1])

          # Optional soft cap on final logits
          if cfg.final_logits_soft_cap:
              logits = logits / cfg.final_logits_soft_cap
              logits = jnp.tanh(logits) * cfg.final_logits_soft_cap
      else:
          # Use separate dense layer for logits
          logits = linears.DenseGeneral(
              cfg.vocab_size,
              weight_dtype=cfg.weight_dtype,
              dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,
              kernel_axes=("embed", "vocab"),
              name="logits_dense",
              matmul_precision=self.config.matmul_precision,
          )(y)

      # Apply logical constraints to logits
      logits = nn.with_logical_constraint(
          logits,
          ("activation_embed_and_logits_batch", "activation_length", "activation_vocab")
      )

      # Final dtype casting if configured
      if self.config.cast_logits_to_fp32:
          logits = logits.astype(jnp.float32)

      return logits


class Transformer(nn.Module):
  """An decoder-only Transformer model."""

  config: Config
  mesh: Mesh
  quant: Quant

  def setup(self):
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

  # @functools.partial(
  #       jax.jit,
  #       static_argnames=('is_prefill', 'slot', 'true_length')
  # )
  @nn.compact
  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_segment_ids=None,
      enable_dropout=True,
      model_mode=common_types.MODEL_MODE_PREFILL,
      slot = None,
      true_length = None
  ):
    if decoder_segment_ids is not None and not model_mode:
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
      slot=slot,
      true_length=true_length,
    )
    return logits
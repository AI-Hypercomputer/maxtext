# Copyright 2023–2025 Google LLC
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

from typing import Any, Callable
import functools

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh

from flax import linen as nn
from flax import nnx
from flax.nnx import wrappers as nnx_wrappers

from MaxText.configs.types import PositionalEmbedding
from MaxText.common_types import DecoderBlockType, ShardMode, Config, EP_AS_CONTEXT
from MaxText.common_types import MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from MaxText import max_logging
from MaxText import max_utils
from MaxText.sharding import create_sharding
from MaxText.inference import page_manager
from MaxText.layers import linears
from MaxText.layers import normalizations
from MaxText.layers import quantizations
from MaxText.layers import pipeline
from MaxText import maxtext_utils
from MaxText import multimodal_utils
from MaxText import sharding
from MaxText.layers.attentions import attention_as_linen
from MaxText.layers.normalizations import rms_norm
from MaxText.layers.embeddings import Embed, attend_on_embedding, embed_as_linen, positional_embedding_as_linen
from MaxText.layers.quantizations import AqtQuantization as Quant

# Import specific layer definitions (assuming these files exist)
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
# The network: Decoder Definitions
# ------------------------------------------------------------------------------

class DecoderLayer(nnx.Module):
    """
    Transformer decoder layer converted to NNX.
    """
    def __init__(
        self,
        config: Config,
        mesh: Mesh,
        model_mode: str,
        rngs: nnx.Rngs,
        quant: None | Quant = None,
        name: str = "decoder_layer",
    ):
        self.config = config
        self.mesh = mesh
        self.model_mode = model_mode
        self.quant = quant
        
        cfg = self.config

        # 1. Pre-Self-Attention Norm
        self.pre_self_attention_norm = RMSNorm(
            num_features=cfg.emb_dim, 
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            epsilon=cfg.normalization_layer_epsilon,
            kernel_axes=("norm",),
        )

        # 2. Self Attention
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
            model_mode=model_mode,
        )

        # 3. MLP Block
        self.mlp = linears.MLPBlock(
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
            rngs=rngs
        )
        
        # 4. Dropout
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
        )

        if self.model_mode == MODEL_MODE_PREFILL:
            logical_axis_names = ("activation_batch", "prefill_activation_length", "activation_embed")
        elif self.config.expert_shard_attention_option == EP_AS_CONTEXT and self.model_mode == MODEL_MODE_TRAIN:
            logical_axis_names = ("activation_batch_no_exp", "activation_length", "activation_embed")
        else:
            logical_axis_names = ("activation_batch", "activation_length_no_exp", "activation_embed")

        inputs = _maybe_shard_with_logical(inputs, logical_axis_names)
        inputs = checkpoint_name(inputs, "decoder_layer_input")

        # --- Norm ---
        lnx = self.pre_self_attention_norm(inputs)
        lnx = _maybe_shard_with_logical(lnx, logical_axis_names)

        # --- Attention ---
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

        # --- MLP ---
        mlp_lnx = self.mlp(lnx, deterministic=deterministic)
        mlp_lnx = _maybe_shard_with_logical(mlp_lnx, logical_axis_names)

        # --- Residuals & Dropout ---
        next_layer_addition = mlp_lnx + attention_lnx
        next_layer_addition_dropped_out = self.dropout(
            next_layer_addition, deterministic=deterministic
        )

        layer_output = next_layer_addition_dropped_out + inputs
        layer_output = _maybe_shard_with_logical(layer_output, logical_axis_names)

        if cfg.record_internal_nn_metrics:
           self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
           self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
           self.sow(
               "intermediates",
               "activation_fraction_zero",
               jnp.sum(layer_output == 0) / jnp.size(layer_output),
           ) 

        if cfg.scan_layers:
            # For scanned layers, we typically return (carry, accumulator)
            # In MaxText scan convention: return output, None (as per instructions)
            return layer_output, None
        else:
            return layer_output, kv_cache


class Decoder(nnx.Module):
    """A stack of decoder layers as a part of an encoder-decoder architecture, using NNX."""

    def __init__(
        self,
        config: Config,
        mesh: Mesh,
        rngs: nnx.Rngs,
        quant: None | Quant = None,
        model_mode: str = MODEL_MODE_TRAIN,
    ):
        self.config = config
        self.mesh = mesh
        self.quant = quant
        self.model_mode = model_mode
        self.rngs = rngs

        # Initialize Layers
        decoder_block_classes = self.get_decoder_layers()
        
        # Norm Layer
        self.norm_layer = self.get_norm_layer(num_features=config.emb_dim, rngs=rngs)(
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
              embedding_init=nn.initializers.normal(stddev=1.0),
              config=config,
              mesh=self.mesh,
              rngs=rngs,
          )
           
        self.dropout = linears.Dropout(rate=config.dropout_rate, rngs=rngs, broadcast_dims=(-2,))
       
        self.positional_embedding = PositionalEmbedding(embedding_dims=config.base_emb_dim)

        # Output Logits Dense
        if not config.logits_via_embedding:
            self.logits_dense = linears.DenseGeneral(
              inputs_shape=(1, 1, config.emb_dim), # placeholder
              out_features_shape=config.vocab_size,
              weight_dtype=config.weight_dtype,
              dtype=jnp.float32 if config.logits_dot_in_fp32 else config.dtype,
              kernel_axes=("embed", "vocab"),
              shard_mode=config.shard_mode,
              matmul_precision=self.config.matmul_precision,
              parameter_memory_host_offload=config.parameter_memory_host_offload,
              rngs=rngs
            )

        # Logic for creating layer stack (Scanned vs Sequential)
        self.layers = []
        self.scanned_layers = None
        self.is_deepseek = (self.config.decoder_block == DecoderBlockType.DEEPSEEK)

        if self.config.scan_layers:
            if self.is_deepseek:
                assert len(decoder_block_classes) == 2
                dense_cls, moe_cls = decoder_block_classes
                
                # Create Dense Stack
                num_dense = config.first_num_dense_layers
                self.dense_stack = self._create_scanned_layers(
                    dense_cls, length=num_dense, rngs=rngs
                )
                
                # Create MoE Stack
                num_moe = config.num_decoder_layers - config.first_num_dense_layers
                self.moe_stack = self._create_scanned_layers(
                    moe_cls, length=num_moe, rngs=rngs
                )
            else:
                # Standard homogeneous stack
                layer_cls = decoder_block_classes[0]
                num_layers = config.num_decoder_layers
                self.layers_stack = self._create_scanned_layers(
                    layer_cls, length=num_layers, rngs=rngs
                )
        else:
            # Sequential (List of layers)
            # Using simple loop to create individual NNX layers
            if self.is_deepseek:
                dense_cls, moe_cls = decoder_block_classes
                for i in range(config.first_num_dense_layers):
                    self.layers.append(self._create_single_layer(dense_cls, rngs, name=f"dense_layer_{i}"))
                for i in range(config.num_decoder_layers - config.first_num_dense_layers):
                    self.layers.append(self._create_single_layer(moe_cls, rngs, name=f"moe_layer_{i}"))
            else:
                layer_cls = decoder_block_classes[0]
                for i in range(config.num_decoder_layers):
                    self.layers.append(self._create_single_layer(layer_cls, rngs, name=f"layers_{i}"))

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
    
    def _create_scanned_layers(self, decoder_layer_class, length: int, rngs: nnx.Rngs, **layer_kwargs):
        """Creates a VMapped stack of layers, forcing parameter init for Compact modules."""
        
        def create_layer_fn(rng):
            layer = decoder_layer_class(
                config=self.config,
                mesh=self.mesh,
                quant=self.quant,
                model_mode=self.model_mode,
                rngs=rng,
                **layer_kwargs
            )
            
            return layer
      
        nnx.split_rngs(rngs, splits=length)
        layers_vmapped = nnx.vmap(
          create_layer_fn,
          in_axes=0,
          out_axes=0,
          axis_name="layers",  # Named axis for sharding
          transform_metadata={nnx.PARTITION_NAME: "layers"},  # Sharding metadata
        )(rngs)

        return layers_vmapped
    
    def _apply_layers_sequentially(self, layers, x_in, *args, length: int, **kwargs):
      """Runs the layer stack using nnx.scan."""
      policy = self.get_remat_policy()
      prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
      graphdef, params, state = nnx.split(layers, nnx.Param, ...)  # state: the mutable state we carry (KV cache, RNGs, etc.)
      def layer_fn(carry, layer_params):
          layer = nnx.merge(graphdef, layer_params, other_state)
          # layer_i is a slice of the vmapped layers, representing one layer
          layer_out = layer(carry, *args, **kwargs)
          # handle tuple returns (output, aux) -> just take output for carry
          new_carry = layer_out[0] if isinstance(layer_out, tuple) else layer_out
          _, new_layer_state = nnx.split(layer)
          return new_carry, new_layer_state
      
      # Apply checkpointing to the step function
      if policy is not None:
          pure_layer_fn = jax.checkpoint(pure_layer_fn, policy=policy, prevent_cse=prevent_cse)
      layer_fn = jax.checkpoint(layer_fn, policy=policy, prevent_cse=prevent_cse)
      
      final_carry, scanned_state = jax.lax.scan(
          layer_fn,
          x_in,
          params
      )
      nnx.update(layers, scanned_state)

      return final_carry, None

    def get_decoder_layers(self):
      """Retrieves decoder layer classes based on config."""
      match self.config.decoder_block:
        case DecoderBlockType.DEFAULT:
          return [DecoderLayerNNX]
        case DecoderBlockType.LLAMA2:
          return [llama2.LlamaDecoderLayer]
        case DecoderBlockType.MISTRAL:
          return [mistral.MistralDecoderLayer]
        case DecoderBlockType.MIXTRAL:
          return [mixtral.MixtralDecoderLayer]
        case DecoderBlockType.DEEPSEEK:
          if self.config.use_batch_split_schedule:
            return [deepseek_batchsplit.DeepSeekDenseLayer, deepseek_batchsplit.DeepSeekMoELayer]
          else:
            return [deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer]
        case DecoderBlockType.GEMMA:
          return [gemma.GemmaDecoderLayer]
        case DecoderBlockType.GEMMA2:
          return [gemma2.Gemma2DecoderLayer]
        case DecoderBlockType.GEMMA3:
          return [gemma3.Gemma3DecoderLayer]
        case DecoderBlockType.GPT3:
          return [gpt3.Gpt3DecoderLayer]
        case DecoderBlockType.GPT_OSS:
          return [gpt_oss.GptOssScannableBlock] if self.config.scan_layers else [gpt_oss.GptOssDecoderLayer]
        case DecoderBlockType.QWEN3:
          return [qwen3.Qwen3DecoderLayer]
        case DecoderBlockType.QWEN3_MOE:
          return [qwen3.Qwen3MoeDecoderLayer]
        case DecoderBlockType.QWEN3_NEXT:
          return [qwen3.Qwen3NextScannableBlock] if self.config.scan_layers else [qwen3.Qwen3NextDecoderLayer]
        case DecoderBlockType.SIMPLE:
          return [simple_layer.SimpleDecoderLayer]
        case DecoderBlockType.SIMPLE_MLP:
          return [simple_layer.SimpleMlpDecoderLayer]
        case DecoderBlockType.LLAMA4:
          return [llama4.Llama4ScannableBlock] if self.config.scan_layers else [llama4.Llama4DecoderLayer]
        case _:
          raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}") 
  
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
      ):
        return functools.partial(rms_norm, num_features=num_features, shard_mode=self.config.shard_mode)
      elif self.config.decoder_block == DecoderBlockType.GPT3:
          return functools.partial(gpt3.Gpt3LayerNorm, num_features=num_features, reductions_in_fp32=False, use_bias=True, rngs=rngs)
      else:
        raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block.value=}")


    def _apply_embedding(
        self,
        shared_embedding,
        decoder_input_tokens,
        decoder_positions,
        deterministic,
        model_mode,
        image_embeddings=None,
        bidirectional_mask=None,
        image_masks=None,
    ):
        cfg = self.config
        
        # Handle shared embedding execution
        # If shared_embedding is NNX module, access params directly or call it
        if isinstance(shared_embedding, nnx.Module):
             y = shared_embedding(decoder_input_tokens.astype("int32"), model_mode=model_mode)
        else:
            # Fallback for Linen module passed in
            y = shared_embedding(decoder_input_tokens.astype("int32"), model_mode=model_mode)

        # Merge Multimodal
        if image_embeddings is not None and cfg.use_multimodal:
            y = multimodal_utils.merge_mm_embeddings(
                text_embeddings=y,
                vision_embeddings=image_embeddings,
                mask=bidirectional_mask,
                image_masks=image_masks,
            )

        y = self.dropout(y, deterministic=deterministic)
        y = y.astype(cfg.dtype)

        if cfg.use_untrainable_positional_embedding:
            y = self.positional_embedding(y, decoder_positions)

        if cfg.trainable_position_size > 0 and self.position_embedder:
            y += self.position_embedder(decoder_positions.astype("int32"), model_mode=model_mode)
        
        return y

    def apply_output_head(self, shared_embedding, y, deterministic, model_mode):
        cfg = self.config
        if cfg.shard_mode == ShardMode.EXPLICIT:
            norm_out_sharding = create_sharding(self.mesh, ("activation_batch", "activation_length_no_exp", "activation_embed"))
        else:
            norm_out_sharding = None

        y = self.norm_layer(y) # NNX call
        y = self.dropout(y, deterministic=deterministic) # NNX call

        if model_mode in (MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE):
            out_sharding = create_sharding(self.mesh, (None, None, "activation_vocab"))
        else:
            out_sharding = create_sharding(
                self.mesh, ("activation_embed_and_logits_batch", "activation_length_no_exp", "activation_vocab")
            )

        if cfg.logits_via_embedding:
            # NNX Access to embedding table
            if isinstance(shared_embedding, nnx.Module):
                # Assuming shared_embedding has an 'embedding' parameter which is a Variable/Param
                embedding_table = shared_embedding.embedding.value
            else:
                 # Legacy Linen
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
            # Use the NNX dense layer defined in __init__
            logits = self.logits_dense(y)
        
        if self.config.cast_logits_to_fp32:
            logits = logits.astype(jnp.float32)
        
        return logits

    def __call__(
        self,
        shared_embedding: Any, # Can be nnx.Module or Linen Module
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

        # 1. Apply Embeddings
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

        # 2. Prepare Arguments for Layers
        # Positional arguments that all layers accept
        layer_args = (
             decoder_segment_ids,
             decoder_positions,
             deterministic,
             model_mode
        )
        
        # Keyword arguments: Start with common ones
        layer_kwargs = {
            "previous_chunk": previous_chunk,
            "page_state": page_state,
            "slot": slot,
            "attention_metadata": attention_metadata,
        }

        # CONDITIONAL ARGUMENT LOGIC (Crucial Fix)
        # Only pass bidirectional_mask if the model is GEMMA3, matching original code behavior.
        if cfg.decoder_block == DecoderBlockType.GEMMA3:
            layer_kwargs["bidirectional_mask"] = bidirectional_mask

        # 3. Apply Layers (Scanned or Sequential)
        if cfg.scan_layers:
            if self.is_deepseek:
                # DeepSeek has two distinct stacks (Dense and MoE)
                # 1. Run Dense Stack
                y, _ = self._apply_layers_sequentially(
                    self.dense_stack, y, *layer_args, length=cfg.first_num_dense_layers, **layer_kwargs
                )
                
                # 2. Run MoE Stack
                num_moe = cfg.num_decoder_layers - cfg.first_num_dense_layers
                y, _ = self._apply_layers_sequentially(
                    self.moe_stack, y, *layer_args, length=num_moe, **layer_kwargs
                )
            else:
                # Standard Scanned Stack (Works for GPT3, Llama, Gemma, etc.)
                y, _ = self._apply_layers_sequentially(
                    self.layers_stack, y, *layer_args, length=cfg.num_decoder_layers, **layer_kwargs
                )
        else:
            # Unscanned Loop
            # self.layers is a flat list of initialized NNX modules created in __init__
            for i, layer in enumerate(self.layers):
                kv_cache = kv_caches[i] if kv_caches is not None else None
                
                # Call the layer
                out = layer(
                    y,
                    *layer_args,
                    kv_cache=kv_cache,
                    **layer_kwargs
                )
                
                # Handle Return formats: (output, kv_cache) or just output
                if isinstance(out, tuple):
                    y, kv_cache_out = out
                else:
                    y = out
                    kv_cache_out = None

                if kv_caches is not None:
                    kv_caches[i] = kv_cache_out

        assert isinstance(y, jax.Array)
        hidden_state = y

        # 4. Output Head / Logits
        if cfg.num_vocab_tiling > 1 and self.model_mode == MODEL_MODE_TRAIN:
            logits = None
        else:
            logits = self.apply_output_head(shared_embedding, hidden_state, deterministic, model_mode)

        return logits, hidden_state, kv_caches

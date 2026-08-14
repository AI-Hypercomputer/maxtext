# Copyright 2023–2026 Google LLC
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

"""Specialized layers for Gemma 4."""

"""
HIGH-LEVEL SUMMARY:
This module contains the specialized layer architectures for the Gemma 4 model,
written in JAX/Flax using maxtext.src layers. The primary features of Gemma 4
include a hybrid attention scheme (sliding window vs. global attention), a 
Routed and Shared Mixture of Experts (MoE) block, and sophisticated XLA-level scanning
methodology (jax.lax.scan with rematerialization hooks) to keep memory bounds 
low during training and inference.
"""

# Bring in jax arrays, transforms, and experimental XLA metadata controls 
# used for fine-tuning while-loop constraints (trip-count-one).
import jax
from jax.experimental import xla_metadata
# Used for manually tracking activation namespaces in auto-diff checkpoints
from jax.ad_checkpoint import checkpoint_name
# Used for logical mesh device layouts over distributed accelerators
from jax.sharding import Mesh
import jax.numpy as jnp

# Flax neural network apis (traditional state-based linen and newer nnx api)
from flax import linen as nn
from flax import nnx
from typing import Optional, Any

# MaxText internal configuration flags and constants
from maxtext.src.maxtext.common.common_types import Config, AttentionType, MODEL_MODE_PREFILL
# MaxText custom initialization bounds for linear projection layers
from maxtext.src.maxtext.layers import initializers
# MaxText Mixture of Experts layer primitive
from maxtext.src.maxtext.layers import moe
# Wrappers that allow JAX bounded loops (scan) natively alongside Flax modules
from maxtext.src.maxtext.layers import nnx_scan, nnx_wrappers
# Quantization definitions for low-precision inference and training (e.g. INT8, FP8)
from maxtext.src.maxtext.layers import quantizations
from maxtext.src.maxtext.layers.attentions import Attention
from maxtext.src.maxtext.layers.linears import MlpBlock

# Distributed tensor representation primitives
import jax.sharding
# Layernorm optimized for large language models, lacking bias offset
from maxtext.src.maxtext.layers.normalizations import RMSNorm
from maxtext.src.maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.src.maxtext.utils import max_utils
from maxtext.src.maxtext.utils import maxtext_utils


'''
HIGH-LEVEL SUMMARY (Attention Pattern):
Gemma 4 alternates between local (sliding window) and global attention.
This pattern determines how attention layers treat KV caching. Typically
models will repeat block sets of 5 sliding architectures and 1 global.
'''
# Defines the cyclical sequence of attention mechanics in Gemma 4
GEMMA4_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,  # Layer offset 0 uses sliding window
    AttentionType.LOCAL_SLIDING,  # Layer offset 1 uses sliding window
    AttentionType.LOCAL_SLIDING,  # Layer offset 2 uses sliding window
    AttentionType.LOCAL_SLIDING,  # Layer offset 3 uses sliding window
    AttentionType.LOCAL_SLIDING,  # Layer offset 4 uses sliding window
    AttentionType.GLOBAL,         # Layer offset 5 uses full global attention
)


# Helper function to easily resolve a layer index into its structural AttentionType
def get_attention_type(layer_id):
  # Perform modulo arithmetic to wrap the layer index relative to the pattern length (6)
  layer_id %= len(GEMMA4_ATTENTION_PATTERN)
  # Return either LOCAL_SLIDING or GLOBAL configured in the pattern mapping
  return GEMMA4_ATTENTION_PATTERN[layer_id]


'''
HIGH-LEVEL SUMMARY (Gemma4MoE):
This block combines standard Transformer Mixture of Experts (MoE) with a "Shared Expert".
Instead of routing tokens strictly to a sub-set of experts, a portion of the 
network operates as a "shared" FFN that works on all tokens synchronously.
The outputs of the routed experts and shared expert are then summed together.
'''
class Gemma4MoE(nnx.Module):
  """Gemma4 specific MoE block containing layer norms and a generic MoE block."""

  def __init__(
      self,
      config: Config, # Universal state object containing configurations (dims, epsilons)
      mesh: Mesh,     # Devices to shard execution over 
      rngs: nnx.Rngs, # Random key generators
      quant: None | Quant = None, # Precision parameters
  ):
    # Bind arguments onto instance namespace
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.quant = quant

    # Establish the underlying MoE block combining routed and shared experts.
    self.moe_block = moe.RoutedAndSharedMoE(
        config=config,
        mesh=mesh,
        # Default dense init mapping truncated_normal with fan-in scale derivation
        kernel_init=initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        weight_dtype=config.weight_dtype,
        dtype=config.dtype,
        quant=self.quant,
        rngs=self.rngs,
    )

    # Gemma4 MoE employs several custom pre/post norms uncharacteristic of standard transformers.
    # Instantiate parameter corresponding to scaling factor for router path computation
    self.pre_forward_scale_2 = nnx.Param(
        # Shape is embedding dimension, all initialized to contiguous 1.0 (float)
        jnp.ones((self.config.emb_dim,), dtype=self.config.weight_dtype),
        # Shard the parameter uniformly across the embedding dimension logical axis
        sharding=("embed",),
    )
    # RMSNorm initialized before feeding inputs into routed experts 
    self.pre_feedforward_layernorm_2 = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    # RMSNorm applied unconditionally to the shared expert output
    self.post_feedforward_layernorm_1 = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    # RMSNorm applied onto the dynamic routed experts output stream
    self.post_feedforward_layernorm_2 = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    # Gate calculation unscaled norm (no learned affine matrix 'scale', root-mean-square only)
    self.gate_norm = RMSNorm(
        num_features=self.config.emb_dim,
        epsilon=self.config.normalization_layer_epsilon,
        # MoE gates often need higher precision float32 representation for numerical stability
        dtype=jnp.float32 if self.config.float32_gate_logits else self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        kernel_axes=("norm",),
        with_scale=False, # Disable multiplicative weights since we supply custom router_scale
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs: jax.Array,
      original_inputs: jax.Array | None = None,
      intermediate_sharding: jax.sharding.NamedSharding | None = None,
      out_sharding: jax.sharding.NamedSharding | None = None,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    
    # 0. Execute the 'Shared Expert' (acts as an omnipresent MLP for all tokens)
    shared_experts = self.moe_block.shared_experts(
        inputs, intermediate_sharding=intermediate_sharding, out_sharding=out_sharding
    )
    # Apply post normalization on shared expert values
    shared_experts = self.post_feedforward_layernorm_1(shared_experts)

    # 1. Experts receive standard RMSNorm (with learned weight) before distribution
    routed_inputs = self.pre_feedforward_layernorm_2(original_inputs)

    # 2. Derive Gate features, utilizing a higher precision (float32 options) dtype
    gate_dtype = jnp.float32 if self.config.float32_gate_logits else self.config.dtype
    # Perform strict RMS calculation (no scales) over original token embeddings
    unscaled_norm = self.gate_norm(original_inputs)

    # Calculate constant root of embedding size as normalization factor bounds
    root_size = self.config.emb_dim**-0.5
    # Fetch parameters corresponding to pre-router gating scalar
    router_scale = jnp.asarray(self.pre_forward_scale_2.value, gate_dtype)
    # Merge unscaled normalization features, root factor, and router learned biases together
    gate_inputs = unscaled_norm * root_size * router_scale

    # 3. Pass evaluated gate inputs into the routed path with expected shardings
    routed_experts, load_balance_loss, moe_bias_updates = self.moe_block.routed_moe(
        routed_inputs, gate_inputs=gate_inputs, out_sharding=out_sharding
    )
    # Normalize routed elements sequentially post-MLP execution
    routed_experts = self.post_feedforward_layernorm_2(routed_experts)

    # Final outputs return the sum of dynamically routed experts and the ubiquitous shared expert.
    # We also return intermediate calculated load balance loss and updating biases for logging
    return routed_experts + shared_experts, load_balance_loss, moe_bias_updates


'''
HIGH-LEVEL SUMMARY (Gemma4DecoderLayer):
A single Decoder element in the architecture. This block constructs
attention configurations dependent on if it acts as a global or local
sliding window layer. It executes self-attention and MLP residual blocks.
'''
class Gemma4DecoderLayer(nnx.Module):
  """Transformer decoder layer for Gemma4."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      attention_type: AttentionType = AttentionType.LOCAL_SLIDING,
      layer_idx: int = 0,
  ):
    """Initializes the instance.

    Args:
      config: The Config object with model hyperparameters.
      mesh: The device mesh for distributed training.
      model_mode: One of MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, or MODEL_MODE_AUTOREGRESSIVE.
      rngs: The random number generators for initialization.
      quant: The quantization configuration.
      attention_type: The type of attention to use.
      layer_idx: The index of the layer in the block.
    """

    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.rngs = rngs
    # Capture requested Attention mechanics (LOCAL vs GLOBAL differences)
    self.attention_type = attention_type
    self.layer_idx = layer_idx

    # Dynamically extract sequence lengths depending on if we are training/inferencing
    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    # Generate an evaluation shape constraint useful for dummy initialization logic
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    # Establish RMSNorm prior to self attention q/k/v execution
    self.pre_self_attention_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    # Define standard configuration defaults for heads (MQA, GQA defaults)
    query_pre_attn_scalar = 1.0
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    share_kv_projections = False

    # Adjust KV mechanics exclusively when layer operates as a full GLOBAL window
    if attention_type == AttentionType.GLOBAL:
      # If global KV dims exist override local sequence configs
      if hasattr(config, "global_num_kv_heads") and config.global_num_kv_heads:
        num_kv_heads = config.global_num_kv_heads
      if hasattr(config, "global_head_dim") and config.global_head_dim:
        head_dim = config.global_head_dim
      if getattr(config, "share_kv_projections", False):
        share_kv_projections = True

    # Adjust RoPE rotary mechanisms depending on context (GLOBAL vs LOCAL view sizes differs RoPE constraints)
    if attention_type == AttentionType.GLOBAL:
      # Global contexts take proportional RoPE embeddings (e.g. 0.25 chunked representations)
      partial_rotary_factor = config.global_rope_proportion if hasattr(config, "global_rope_proportion") else 0.25
      max_timescale = (
          config.global_rope_max_timescale
          if hasattr(config, "global_rope_max_timescale") and config.global_rope_max_timescale > 0
          else config.rope_max_timescale
      )
    else:  # LOCAL_SLIDING path behavior
      # Sliding models take entirely scaled RoPE models (e.g. 1.0 multiplier)
      partial_rotary_factor = config.local_rope_proportion if hasattr(config, "local_rope_proportion") else 1.0
      max_timescale = (
          config.local_rope_max_timescale
          if hasattr(config, "local_rope_max_timescale") and config.local_rope_max_timescale > 0
          else config.rope_max_timescale
      )

    # Define main Attention module instance linking customized local/global values.
    self.self_attention = Attention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        mesh=mesh,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        dropout_rate=config.dropout_rate,
        float32_qk_product=config.float32_qk_product,
        float32_logits=config.float32_logits,
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(config),
        attention_type=self.attention_type, # Provide enum binding LOCAL/GLOBAL execution inside Attention
        sliding_window_size=config.sliding_window_size,
        attn_logits_soft_cap=config.attn_logits_soft_cap,
        use_qk_norm=True,  # Gemma 4 models natively embed QK normalizations 
        use_v_norm=True,
        query_pre_attn_scalar=query_pre_attn_scalar,
        share_kv_projections=share_kv_projections,
        rope_max_timescale=max_timescale,
        partial_rotary_factor=partial_rotary_factor,
        model_mode=model_mode,
        rngs=self.rngs,
    )

    # Standard configuration toggling for applying normalizations after self attention execution
    if self.config.use_post_attn_norm:
      self.post_self_attention_norm = RMSNorm(
          num_features=config.emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_self_attention_norm = None

    # Pre-MLP block normalization layer definition
    self.pre_ffw_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    # Assign MLP layer depending on architecture (Dense MlpBlock vs mixture of experts Gemma4MoE)
    if getattr(config, "num_experts", 1) > 1:
      self.mlp = Gemma4MoE(
          config=config,
          mesh=mesh,
          rngs=self.rngs,
          quant=self.quant,
      )
    else:
      self.mlp = MlpBlock(
          in_features=config.emb_dim,
          intermediate_dim=config.mlp_dim,
          activations=config.mlp_activations,
          intermediate_dropout_rate=config.dropout_rate,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          config=config,
          quant=self.quant,
          model_mode=model_mode,
          mesh=mesh,
          rngs=self.rngs,
      )

    # Normalization applied after finishing the feed forward (MLP) component
    if self.config.use_post_ffw_norm:
      self.post_ffw_norm = RMSNorm(
          num_features=config.emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_ffw_norm = None

    # Gemma4 unique affine transformation on individual entire decoder layer responses 
    self.layer_scalar = nnx.Param(jnp.ones((1,), dtype=config.weight_dtype), sharding=(None,))

    # Delineate specific tensor/sharding dimensions across differing run cycles (Prefill takes longer contextual lengths)
    if model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")


  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk=None,
      page_state=None,
      slot=None,
      bidirectional_mask=None,
      kv_cache=None,
      attention_metadata=None,
  ):
    cfg = self.config
    
    # Manage states passed down progressively through 'jax.lax.scan' executions where KV is carried
    is_scan_carry = False
    
    # Unpack inputs if it's a structural tuple returned from a previous layer `tuple[hidden, kvs, idx]`
    if isinstance(inputs, tuple) and len(inputs) == 3:
      hidden_states, stacked_kv_cache, layer_idx = inputs
      # Unroll corresponding element of cache cache according to current logical execution layer index
      kv_cache = stacked_kv_cache[layer_idx]
      # Remask hidden_states variable as input
      inputs = hidden_states
      is_scan_carry = True
    elif isinstance(inputs, tuple):
      inputs = inputs[0]
      
    # Apply safety constraints marking axis semantics for tensor dimensions explicitly
    inputs = nn.with_logical_constraint(inputs, self.activation_axis_names)
    # Checkpoint bounds help preserve memory tracing when executing gradients
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    # Pass layer inputs through the self attention root normalizing block 
    lnx = self.pre_self_attention_norm(inputs)
    lnx = nn.with_logical_constraint(lnx, self.activation_axis_names)

    # Gemma4 only applies bidirectional attention in sliding (local) layers,
    # not in full (global) attention layers where bidirectional limits leak.
    if self.attention_type != AttentionType.LOCAL_SLIDING:
      bidirectional_mask = None

    # Self-attention block execution with contextual matrices
    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic, # Flags randomized dropout behaviors vs constant execution
        model_mode=model_mode,
        bidirectional_mask=bidirectional_mask,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    
    # Optional post attention normalization logic based on Model Configs
    if cfg.use_post_attn_norm:
      attention_lnx = self.post_self_attention_norm(attention_lnx)
    attention_lnx = nn.with_logical_constraint(attention_lnx, self.activation_axis_names)

    # Add back the residual stream containing historical unaltered input vector
    attention_lnx += inputs
    residual = attention_lnx
    
    # Forward the newly computed additive vector into pre feed-forward standardization
    attn_output = self.pre_ffw_norm(attention_lnx)

    # MLP block execution route
    # Diverge processing depending on Dense Configuration vs Mixture of Experts definition
    if getattr(self.config, "num_experts", 1) > 1:
      mlp_lnx, load_balance_loss, _ = self.mlp(attn_output, original_inputs=attention_lnx)
      
      # Optional capture and internal logging/telemetry of Expert route balancing metrics
      if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
        self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)
    else:
      mlp_lnx = self.mlp(attn_output, deterministic=deterministic)

    # Optional post MLP normalization logic based on Model Configs
    if cfg.use_post_ffw_norm:
      mlp_lnx = self.post_ffw_norm(mlp_lnx)

    mlp_lnx = nn.with_logical_constraint(mlp_lnx, self.activation_axis_names)

    # Merge standard residual pipeline and MLP transformations 
    next_layer_addition = mlp_lnx + residual
    layer_output = next_layer_addition
    
    # Final layer transformation unique to Gemma 4 Architecture: Apply structural block level scalars multiplier
    layer_output = layer_output * jnp.asarray(self.layer_scalar.value, cfg.dtype)

    layer_output = nn.with_logical_constraint(layer_output, self.activation_axis_names)

    # Additional intermediate state sowing triggers to emit raw tensor statistics when flagged True
    if getattr(cfg, "record_internal_nn_metrics", False):
      # Writes calculated dimensions alongside active variables to intermediate stores 
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    # Wrap outputs inside historical scan stack arrays or emit plain elements directly
    if is_scan_carry:

      # Helper subfunction for inserting new kv sets securely into previous context stores 
      def update_cache(cache, val):
        if jnp.size(val) > 0:
          return cache.at[layer_idx].set(val)
        return cache

      # Replace historic instances natively with `jax.tree_util.tree_map` iterators over pytree boundaries
      stacked_kv_cache = jax.tree_util.tree_map(update_cache, stacked_kv_cache, kv_cache)
      
      # Returns (outputs, KV state, incremented index), empty internal state
      return (layer_output, stacked_kv_cache, layer_idx + 1), None
    else:
      # Return bare structures immediately representing finalized outputs
      return layer_output, kv_cache


# Wrapper assigning NNX definitions to traditional Linen architectures utilizing partitioning metrics
Gemma4DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    Gemma4DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


'''
HIGH-LEVEL SUMMARY (Gemma4ScannableBlock):
MaxText leverages standard XLA rematerialization mapping via `jax.lax.scan` for efficient throughput.
This ScannableBlock consolidates multiples Gemma4DecoderLayer instances mapping LOCAL architectures
and sequentially attaches a singular GLOBAL architecture, iterating recursively logic per block 
which prevents graph out of bounds errors.
'''
class Gemma4ScannableBlock(nnx.Module):
  """A repeatable block of Gemma4 decoder layers, scanning local layers."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      num_of_layers: int = 6, # Defaults to the explicit length of `GEMMA4_ATTENTION_PATTERN`
      remat_policy_fn: Any = None, # User-defined checkpoint logic overriding rematerialized loops
      apply_internal_remat: bool = False, # Flag defining if caller applies blocks (Linen) or internal remat isolates natively
  ):
    """Initializes the instance.

    Args:
      config: The Config object with model hyperparameters.
      mesh: The device mesh for distributed training.
      model_mode: One of MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, or MODEL_MODE_AUTOREGRESSIVE.
      rngs: The random number generators for initialization.
      quant: The quantization configuration.
      num_of_layers: The number of layers in the model.
      remat_policy_fn: The resolved rematerialization policy function.
      apply_internal_remat: When True, the block rematerializes its own local
        (scanned) and global layers, and the caller must NOT also apply
        block-level remat (that would double-rematerialize and make XLA treat the
        whole block as one unit). Both the pure-NNX and linen decoders set this
        and skip block-level remat, so remat happens per layer rather than over
        the whole block. When False, the block does not self-remat and relies on
        the caller's block-level remat instead.
    """
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    self.num_of_layers = num_of_layers
    self.remat_policy_fn = remat_policy_fn
    self.apply_internal_remat = apply_internal_remat

    # Prevent definitions violating maximum attention limits provided by static cycle mapping
    pattern_length = len(GEMMA4_ATTENTION_PATTERN)
    if not 0 <= num_of_layers <= pattern_length:
      raise ValueError(f"Gemma4ScannableBlock must contain between 0 and {pattern_length} layers; got {num_of_layers}.")

    # The block runs its local (sliding-window) layers first, then a single
    # global layer, matching GEMMA4_ATTENTION_PATTERN. Derive the per-type
    # counts from the pattern (well, its first num_of_layers entries) rather
    # than hardcoding the 5-local / 1-global split.
    active_pattern = GEMMA4_ATTENTION_PATTERN[:num_of_layers]
    # Dynamically accumulate the frequency of localized requests
    self.num_local = sum(1 for attn_type in active_pattern if attn_type == AttentionType.LOCAL_SLIDING)
    # Dynamically accumulate the frequency of global block requests
    self.num_global = sum(1 for attn_type in active_pattern if attn_type == AttentionType.GLOBAL)

    # num_of_layers can be 0: the decoders always construct a "remainder" block
    # for num_decoder_layers % pattern_length layers, which is 0 whenever the
    # layer count divides evenly (e.g. 31b = 60 layers). That block is built but
    # never applied, so num_local or num_global may legitimately be 0 here.
    if self.num_local > 0:
      # Creates an optimized Jax Scanned instantiation wrapper evaluating the defined model 
      self.local_layers = nnx_scan.create_scanned_layers(
          # Anonymous builder constructing distinct layers parameterized via passed instances of generic layer states
          lambda layer_rngs: Gemma4DecoderLayer(
              config=self.config,
              mesh=self.mesh,
              model_mode=self.model_mode,
              quant=self.quant,
              rngs=layer_rngs,
              attention_type=AttentionType.LOCAL_SLIDING,
              layer_idx=0,  # layer_idx is not used in the class execution directly
          ),
          length=self.num_local, # Scans execute precisely N-Times based on localized counts derived structurally
          param_scan_axis=self.config.param_scan_axis,
          metadata_axis_name="local_layers",
          rngs=self.rngs,
      )
    else:
      self.local_layers = None

    if self.num_global > 0:
      # Instantiate a single Decoder mapping the distinct AttentionType.GLOBAL requirements 
      self.global_layer = Gemma4DecoderLayer(
          config=self.config,
          mesh=self.mesh,
          model_mode=self.model_mode,
          rngs=self.rngs,
          quant=self.quant,
          attention_type=AttentionType.GLOBAL,
          layer_idx=5,  # layer_idx is not used in the class execution directly (visual marker)
      )
    else:
      self.global_layer = None

  def _run_layer(self, layer, y, layer_kwargs, kv_cache=None):
    """Invokes one ``Gemma4DecoderLayer``, returning ``(output, updated_kv_cache)``.

    This is the shared leaf used by the local scan, the global length-1 scan,
    and the external kv-cache unroll, so it runs in every mode (train / prefill
    / autoregressive). ``updated_kv_cache`` is ``None`` when the layer emits a bare
    output rather than an ``(output, kv_cache)`` tuple.
    """
    out = layer(y, **layer_kwargs, kv_cache=kv_cache)
    return out if isinstance(out, tuple) else (out, None)

  @property
  def _remat_enabled(self):
    """Whether the block rematerializes its own layers.

    False when the caller applies block-level remat instead
    (``apply_internal_remat=False``) or when remat is disabled
    (``remat_policy == "none"``). Note that ``remat_policy_fn``
    is ``None`` for both ``"none"`` and ``"full"``, so it
    cannot distinguish "no remat" from "full remat" on its own.
    """
    return self.apply_internal_remat and self.config.remat_policy != "none"

  def _scan_local_layers(self, y, layer_kwargs):
    """Runs the local (sliding-window) layers via a per-layer rematerialized ``jax.lax.scan``."""
    remat = self._remat_enabled
    # Invoke maxtext external scanner hooks to effectively distribute variables globally
    return nnx_scan.apply_scanned_layers(
        self.local_layers,
        y,
        length=self.num_local,
        param_scan_axis=self.config.param_scan_axis,
        # Replicates output mappings generated from lambda expressions directly via `_run_layer`
        apply_fn=lambda layer, carry: self._run_layer(layer, carry, layer_kwargs)[0],
        remat=remat,
        remat_policy=self.remat_policy_fn if remat else None,
        # prevent_cse is only consulted by jax.checkpoint, i.e. when remat=True;
        # its value is irrelevant otherwise. Common Subexpression Elimination guards redundant recomputations.
        prevent_cse=maxtext_utils.should_prevent_cse_in_remat(self.config) if remat else True,
    )

  def _scan_global_layer(self, y, layer_kwargs):
    """Runs the single global-attention layer inside a length-1 ``jax.lax.scan``.

    The length-1 scan is guarded by a trip-count-one while boundary and wraps
    the layer in its own ``jax.checkpoint``, which keeps only one layer's
    full-sequence-attention working set live at a time; without the boundary
    (blocks are unrolled) XLA co-schedules every block's backward working set
    and OOMs.
    """
    cfg = self.config
    # Split the state into Intermediates and everything else. Non-Intermediate
    # state (the large persistent weights/residuals) is carried through the scan
    # so it stays off the offload-bitcast-prone ys path. Intermediates instead go
    # in as scan xs and come out as ys: a sow can create or grow an Intermediate
    # during the call (e.g. MoE moe_lb_loss accumulates into a tuple), which would
    # break a carry's fixed pytree, and closing them over would mutate state from
    # the wrong trace level (nnx.merge aliases the variables). Routing through
    # xs/ys sidesteps both -- xs/ys have no matching-structure constraint and xs is
    # trace-local. For a dense layer that sows nothing (31b) the Intermediate
    # partition is empty and this is a no-op.
    graphdef_g, intermediate_g, other_g = nnx.split(self.global_layer, nnx.Intermediate, ...)
    intermediate_xs = jax.tree.map(lambda x: x[None], intermediate_g)

    # Nested local execution representing the unique sequence operation mapped within the trip-count unroll hook
    def run_global_layer(carry, intermediate_slice):
      hidden_states, other = carry
      # Combine components to re-acquire the object structure correctly
      layer = nnx.merge(graphdef_g, intermediate_slice, other)
      # Process outputs via isolated execution 
      new_hidden_states = self._run_layer(layer, hidden_states, layer_kwargs)[0]
      # Resplit dependencies post-execution tracking modifications 
      _, new_intermediate, new_other = nnx.split(layer, nnx.Intermediate, ...)
      return (new_hidden_states, new_other), new_intermediate

    # Offloaded (pinned-host) residuals can't cross the trip-count-one boundary,
    # so save would-be-offloaded tensors on device for the global layer instead;
    # the local-layer scan (a real multi-iteration scan) still offloads.
    global_remat_policy = self.remat_policy_fn
    offload_names = maxtext_utils.get_save_and_offload_names(cfg)
    if offload_names[0] or offload_names[1]:
      save_names, offload_to_device = offload_names
      # Modify XLA boundaries specifying tensor storage locations overriding pure native policies 
      global_remat_policy = jax.checkpoint_policies.save_only_these_names(*(save_names + offload_to_device))

    # Incorporate manual JAX Checkpointing ensuring computation paths execute logically tracking resources securely
    if self._remat_enabled:
      prevent_cse = maxtext_utils.should_prevent_cse_in_remat(self.config)
      run_global_layer = jax.checkpoint(
          run_global_layer,
          policy=global_remat_policy,
          prevent_cse=prevent_cse,
      )

    # Carry the non-Intermediate state through the loop instead of returning it as
    # a stacked [1, ...] result: slicing that result previously introduced a bitcast
    # between device and pinned-host memory under offload remat. Only the (tiny)
    # Intermediates ride the xs/ys path.
    # Specify manual directives guaranteeing single loops operate via While Loop primitives to prevent OOM
    with xla_metadata.set_xla_metadata(**{"skip-simplify-while-loops_trip-count-one": "true"}):
      # `jax.lax.scan` constructs the While Loop and prevents memory overlaps statically  
      (y, final_other), stacked_intermediate = jax.lax.scan(
          run_global_layer,
          (y, other_g),
          intermediate_xs,
          length=1,
      )

    # Squeeze the length-1 scan axis off the updated Intermediate state and write
    # it back to the module along with the carried non-Intermediate state.
    intermediate_state = jax.tree.map(lambda x: x[0], stacked_intermediate)
    nnx.update(self.global_layer, final_other, intermediate_state)
    return y

  def _forward_with_external_kv_cache(self, y, kv_cache, layer_kwargs):
    """Runs the block with externally-supplied per-layer kv caches (vLLM PagedAttention).

    Scanning would stack the kv-cache list, which copies it and breaks the
    in-place PagedAttention updates, so the layers are unrolled statically. The
    block's ``kv_cache`` is a per-layer list: the first ``num_local`` entries
    feed the local layers, followed by the single global layer. Returns
    ``(y, updated_kvs)`` with one updated cache per layer.
    """
    updated_kvs = []

    if self.local_layers is not None:
      # Slice the scanned local stack per layer, run it, collect the updated kv
      # caches, and re-stack the per-layer state. This circumvents `jax.lax.scan` entirely
      # accommodating inference engine specifics.
      graphdef, params, state = nnx.split(self.local_layers, nnx.Param, ...)
      scan_axis = self.config.param_scan_axis
      if scan_axis != 0:
        # Move structural dimensions temporarily to adapt isolated list-level executions
        params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0), params)
      per_layer_states = []
      
      # Step manually through individual layers computing updates synchronously 
      for i in range(self.num_local):
        current_params = jax.tree.map(lambda x, i=i: x[i], params)
        current_state = jax.tree.map(lambda x, i=i: x[i], state)
        layer = nnx.merge(graphdef, current_params, current_state)
        # Execute run operations tracking outputs against preconfigured unrolled dimensions
        y, new_kv = self._run_layer(layer, y, layer_kwargs, kv_cache[i])
        updated_kvs.append(new_kv)
        per_layer_states.append(nnx.state(layer))

      # Repack structures backwards into normalized constraints returning architectures identically
      stacked_state = jax.tree.map(lambda *xs: jnp.stack(xs), *per_layer_states)
      if scan_axis != 0:
        stacked_params, stacked_other = stacked_state.split(nnx.Param, ...)
        stacked_params = jax.tree.map(lambda x: jnp.moveaxis(x, 0, scan_axis), stacked_params)
        stacked_state = nnx.State.merge(stacked_params, stacked_other)
      nnx.update(self.local_layers, stacked_state)

    if self.global_layer is not None:
      # Pass through manually appended layer executing unrolled operations linearly as requested 
      y, new_kv = self._run_layer(self.global_layer, y, layer_kwargs, kv_cache[self.num_local])
      updated_kvs.append(new_kv)

    return y, tuple(updated_kvs)

  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      slot=None,
      page_state=None,
      previous_chunk=None,
      bidirectional_mask=None,
      kv_cache=None,
      attention_metadata=None,
  ):
    cfg = self.config
    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    # Arguments shared by every layer in the block. model_mode differentiates
    # train / prefill / autoregressive inside each layer; the block itself does
    # not branch on it.
    layer_kwargs = {
        "decoder_segment_ids": decoder_segment_ids,
        "decoder_positions": decoder_positions,
        "deterministic": deterministic,
        "model_mode": model_mode,
        "slot": slot,
        "previous_chunk": previous_chunk,
        "bidirectional_mask": bidirectional_mask,
        "attention_metadata": attention_metadata,
    }

    # Externally-supplied per-layer caches (vLLM PagedAttention) force a static
    # unroll; otherwise attention manages its own cache and we take the scanned
    # path (train and standard prefill/autoregressive alike).
    if kv_cache is not None:
      # Bypass scanning natively mapping logic sequentially internally tracking page boundaries
      return self._forward_with_external_kv_cache(inputs, kv_cache, layer_kwargs)

    y = inputs
    
    # Process dynamically mapped chunks of local executions (5 executions typically)
    if self.local_layers is not None:
      y = self._scan_local_layers(y, layer_kwargs)
      
    # Execute identical logic corresponding strictly towards unified global executions (1 execution typically)
    if self.global_layer is not None:
      y = self._scan_global_layer(y, layer_kwargs)

    # Strip auxiliary definitions relying exclusively structurally on return type conventions (boolean mapping)
    if cfg.scan_layers:
      return y, None
    return y


# Wrapper assigning NNX definitions to traditional Linen architectures utilizing partitioning metrics
Gemma4ScannableBlockToLinen = nnx_wrappers.to_linen_class(
    Gemma4ScannableBlock,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
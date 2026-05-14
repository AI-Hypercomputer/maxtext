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

"""Decoder Layer and Scannable Block definitions for DeepSeek-V4.

DeepSeek-V4 Decoder Layer Data Flow Guide:
`B` = batch_size, `S` = sequence_length, `k` = hc_mult (expansion rate), `D` = hidden_size

                     Parallel Streams Input [B, S, k, D]
                                    │
                                    ├───► [mHC Pre-norm & Mapping] ──► [B, S, k * D] ──► Flat-Norm
                                    │                                                         │
                                    │                                                [pre_alpha / pre_beta]
                                    │                                                         │
                                    │                                                         ▼
                                    │                                                  Sigmoid Logits
                                    │                                                         │
                                    │                                                         ▼
                                    │                                                 "pre" weights [B, S, k]
                                    │                                                         │
                                    ├─────────────────────────────────────────────────────────┼────────────┐
                                    ▼                                                         ▼            │
                         Parallel Streams [B, S, k, D]                                 Collapse Sum        │
                                    │                                                         │            │
                           [mHC Res-Mapping]                                                  ▼            │
                                    │                                                Collapsed [B, S, D]   │
                          [res_alpha / res_beta]                                              │            │
                                    │                                                  RMSNorm Pre-Attn    │
                                    ▼                                                         │            │
                          Sigmoid Logits [B, S, k, k]                                         ▼            │
                                    │                                                DeepSeekV4Attention   │
                             Sinkhorn-Knopp                                                   │            │
                                    │                                                         ▼            │
                         Doubly Stochastic "comb"                                    Attn Output [B, S, D] │
                                    │                                                         │            │
                                    ▼                                                [mHC Post-Mapping]    │
                                 Multiply                                            [post_alpha / beta]   │
                                    │                                                         │            │
                                    ▼                                                         ▼            │
                             Mixed Residual                                             Sigmoid Logits     │
                              [B, S, k, D]                                                    │            │
                                    │                                                         ▼            │
                                    │                                              "post" weights [B, S, k]│
                                    │                                                         │            │
                                    │                                                         ▼            │
                                    │                                                  Expanded Output     │
                                    │                                                   [B, S, k, D]       │
                                    │                                                         │            │
                                    └───────────────────────► ( + ) ◄─────────────────────────┘            │
                                                               │                                           │
                                                               ▼                                           │
                                                     Attention Site Output                                 │
                                                         [B, S, k, D]                                      │
                                                               │                                           │
                                                               ▼                                           │
                                                     Experts MoE FFN Site                                  │
                                              (Same flow: Collapse -> MoE -> Expand)                       │
                                                               │                                           │
                                                               ▼                                           │
                                                      Layer Output [B, S, k, D] ◄──────────────────────────┘
"""

from typing import Any, Optional
from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.common.common_types import Config, HyperConnectionType, MODEL_MODE_PREFILL
from maxtext.layers import initializers
from maxtext.layers import mhc
from maxtext.layers import moe
from maxtext.layers import nnx_wrappers
from maxtext.layers import quantizations
from maxtext.layers.attention_compressed import DeepSeekV4Attention
from maxtext.layers.normalizations import DeepSeekV4RMSNorm, DeepSeekV4UnweightedRMSNorm
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding
from maxtext.utils.sharding import maybe_shard_with_logical


def get_attention_type(compress_ratio: int) -> str:
  """Returns the attention type string corresponding to the given compression ratio."""
  if compress_ratio == 0:
    return "sliding_attention"
  elif compress_ratio == 4:
    return "compressed_sparse_attention"
  else:
    return "heavily_compressed_attention"


class DeepSeekV4DecoderLayer(nnx.Module):
  """Transformer decoder layer for DeepSeek-V4.

  This layer unconditionally implements routed and shared MoE and unconditionally
  applies Manifold-Constrained Hyper-Connections (mHC) to both attention and FFN block outputs.
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      compress_ratio: int = 4,
      layer_idx: int = 0,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    self.compress_ratio = compress_ratio
    self.layer_idx = layer_idx

    batch_size, sequence_length = max_utils.get_batch_seq_len_for_mode(self.config, self.model_mode)
    self.dummy_inputs_shape = (batch_size, sequence_length, self.config.emb_dim)

    # Pre-attention normalization layer
    self.pre_self_attention_layer_norm = DeepSeekV4RMSNorm(
        hidden_size=self.config.emb_dim,
        eps=self.config.normalization_layer_epsilon,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
    )

    # Compressed multi-head attention module.
    num_heads = (
        self.config.num_query_heads if self.config.num_query_heads is not None else self.config.num_attention_heads
    )
    attention_type = get_attention_type(self.compress_ratio)

    self.self_attention = DeepSeekV4Attention(
        hidden_size=self.config.emb_dim,
        q_lora_rank=self.config.q_lora_rank,
        head_dim=self.config.head_dim,
        num_heads=num_heads,
        config=config,
        layer_idx=layer_idx,
        eps=self.config.normalization_layer_epsilon,
        weight_dtype=self.config.weight_dtype,
        dtype=self.config.dtype,
        attention_type=attention_type,
        rngs=self.rngs,
    )

    # Manifold-constrained hyper-connection wrapper for attention block outputs.
    self.mhc_attention = mhc.ManifoldConstrainedHyperConnections(
        config=self.config,
        dim=self.config.emb_dim,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    # Pre-FFN normalization layer
    self.post_self_attention_layer_norm = DeepSeekV4RMSNorm(
        hidden_size=self.config.emb_dim,
        eps=self.config.normalization_layer_epsilon,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
    )

    # Routed sparse and shared experts mixture-of-experts FFN module.
    self.mlp = moe.RoutedAndSharedMoE(
        config=self.config,
        mesh=self.mesh,
        kernel_init=initializers.nd_dense_init(self.config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed_moe", None),
        weight_dtype=self.config.weight_dtype,
        dtype=self.config.dtype,
        quant=self.quant,
        rngs=self.rngs,
        layer_idx=self.layer_idx,
    )

    # Manifold-constrained hyper-connection wrapper for FFN block outputs.
    self.mhc_mlp = mhc.ManifoldConstrainedHyperConnections(
        config=self.config,
        dim=self.config.emb_dim,
        mesh=self.mesh,
        rngs=self.rngs,
    )

    self.out_sharding = create_sharding(self.mesh, self.logical_axis_names, rules=self.config.logical_axis_rules)

  @property
  def logical_axis_names(self):
    """Generate logical names for activations dynamically decoupling length dimensions."""
    length_name = "prefill_activation_norm_length" if self.model_mode == MODEL_MODE_PREFILL else "activation_norm_length"
    return ["activation_batch", length_name, "activation_embed"]

  def with_logical_constraint(self, x):
    """Applies sharding constraints over logical axes."""
    return maybe_shard_with_logical(
        x,
        logical_axes=self.logical_axis_names,
        mesh=self.mesh,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
        extra_stack_level=1,
        rules=self.config.logical_axis_rules,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: Optional[jnp.ndarray] = None,
      decoder_positions: Optional[jnp.ndarray] = None,
      deterministic: bool = True,
      model_mode: str = "train",
      previous_chunk: Optional[jnp.ndarray] = None,
      page_state: Any = None,
      slot: Any = None,
      bidirectional_mask: Optional[jnp.ndarray] = None,
      kv_cache: Any = None,
      attention_metadata: Any = None,
      cos: Optional[jnp.ndarray] = None,
      sin: Optional[jnp.ndarray] = None,
      position_ids: Optional[jnp.ndarray] = None,
      decoder_input_tokens: Optional[jnp.ndarray] = None,
  ):
    # inputs shape: [B, S, k, D] (where B = batch, S = sequence length, k = expansion rate, D = hidden dim)
    if isinstance(inputs, tuple):
      inputs = inputs[0]

    if decoder_positions is None and position_ids is not None:
      decoder_positions = position_ids
    if decoder_segment_ids is None:
      decoder_segment_ids = jnp.zeros(inputs.shape[:2], dtype=jnp.int32)

    # Apply constraint to inputs: [B, S, k, D] -> [B, S, k, D]
    x = self.with_logical_constraint(inputs)
    x = checkpoint_name(x, "decoder_layer_input")

    # 1. Attention hyper-connection block
    # intermediate_inputs: [B, S, k, D] -> [B, S, k, D]
    intermediate_inputs, _ = self.mhc_attention(
        norm_fn=self.pre_self_attention_layer_norm,
        branch_fn=self.self_attention,
        x=x,
        mhc_type=HyperConnectionType.ATTENTION,
        attention_mask=bidirectional_mask,
        cos=cos,
        sin=sin,
        position_ids=decoder_positions,
    )

    # 2. Experts MoE FFN hyper-connection block
    # Inputs: intermediate_inputs: [B, S, k, D], decoder_input_tokens (input_ids): [B, S]
    # Outputs output: [B, S, k, D]
    output, metadata = self.mhc_mlp(
        norm_fn=self.post_self_attention_layer_norm,
        branch_fn=self.mlp,
        x=intermediate_inputs,
        mhc_type=HyperConnectionType.MLP_MOE,
        input_ids=decoder_input_tokens,
    )

    load_balance_loss = metadata["load_balance_loss"]
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

    # Final output constraint application: [B, S, k, D] -> [B, S, k, D]
    output = self.with_logical_constraint(output)

    if self.config.scan_layers:
      return output, None
    else:
      return output, kv_cache


DeepSeekV4DecoderLayerToLinen = nnx_wrappers.to_linen_class(
    DeepSeekV4DecoderLayer,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class DeepSeekV4ScannableBlock(nnx.Module):
  """A repeating cyclical block of DeepSeek-V4 decoder layers for compiler scan loops."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: Optional[quantizations.AqtQuantization] = None,
      num_of_layers: int = 2,
      layer_offset: int = 0,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    self.num_of_layers = num_of_layers
    self.layer_offset = layer_offset

    for layer_id in range(self.num_of_layers):
      abs_layer_id = self.layer_offset + layer_id
      # Retrieve layer-specific compression ratio from configuration to support sliding window attention
      # at boundary layers and alternating compressed sparse/heavily compressed attention.
      compress_ratio = self.config.compress_ratios[abs_layer_id]
      layer_name = f"layers_{layer_id}"
      layer = DeepSeekV4DecoderLayer(
          config=self.config,
          mesh=self.mesh,
          model_mode=self.model_mode,
          rngs=self.rngs,
          quant=self.quant,
          compress_ratio=compress_ratio,
          layer_idx=abs_layer_id,
      )
      setattr(self, layer_name, layer)

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: jnp.ndarray,
      decoder_positions: jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      slot: Any = None,
      page_state: Any = None,
      previous_chunk: Optional[jnp.ndarray] = None,
      bidirectional_mask: Optional[jnp.ndarray] = None,
      decoder_input_tokens: Optional[jnp.ndarray] = None,
  ):
    y = inputs
    for layer_id in range(self.num_of_layers):
      y, _ = getattr(self, f"layers_{layer_id}")(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          page_state=page_state,
          slot=slot,
          bidirectional_mask=bidirectional_mask,
          decoder_input_tokens=decoder_input_tokens,
      )
    return y, None


DeepSeekV4ScannableBlockToLinen = nnx_wrappers.to_linen_class(
    DeepSeekV4ScannableBlock,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)


class DeepSeekV4HyperHead(nnx.Module):
  """Final learnable Manifold-Constrained Hyper-Connection (mHC) collapse head.

  This head collapses the parallel streams [B, S, k, D] down to a single
  sequence [B, S, D] before applying the final RMSNorm.
  """

  def __init__(self, config: Config, rngs: nnx.Rngs):
    self.config = config
    self.hc_mult = getattr(config, "mhc_expansion_rate", 4)
    self.eps = getattr(config, "hc_eps", 1e-6)
    self.dtype = config.dtype
    self.weight_dtype = config.weight_dtype
    self.matmul_precision = jax.lax.Precision(config.matmul_precision)

    # Scale-free unweighted RMSNorm
    self.input_norm = DeepSeekV4UnweightedRMSNorm(eps=config.normalization_layer_epsilon)

    # Parameter variables representing learnable linear projections
    scale_init = initializers.nd_dense_init(1.0, "fan_in", "normal")
    self.hc_fn = nnx.Param(
        scale_init(
            rngs.params(),
            (self.hc_mult * config.emb_dim, self.hc_mult),
            self.weight_dtype,
            in_axis=0,
            out_axis=1,
        ),
        out_sharding=("activation_embed", None),
    )
    self.hc_base = nnx.Param(
        initializers.default_bias_init(rngs.params(), (self.hc_mult,), self.weight_dtype),
        out_sharding=(None,),
    )
    self.hc_scale = nnx.Param(
        initializers.default_scalar_init(rngs.params(), (1,), self.weight_dtype),
        out_sharding=(None,),
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    # x shape: [B, S, k, D] where B = batch_size, S = sequence_length, k = hc_mult, D = emb_dim
    b, s, k, d = x.shape

    # 1. Flatten streams and apply scale-free normalization
    # [B, S, k, D] -> [B, S, k * D]
    flat = self.input_norm(jnp.reshape(x, (b, s, k * d)))

    # 2. Match precision and project flat features to mixing logits
    hc_fn = jnp.asarray(self.hc_fn[...], self.dtype)
    hc_base = jnp.asarray(self.hc_base[...], self.dtype)
    hc_scale = jnp.asarray(self.hc_scale[...], self.dtype)

    # mixes calculation: [B, S, k * D] @ [k * D, k] -> [B, S, k]
    mixes = jnp.einsum("bsm,mk -> bsk", flat, hc_fn, precision=self.matmul_precision)

    # mixes sigmoid weights calculation: [B, S, k]
    pre = jax.nn.sigmoid(mixes * hc_scale + hc_base[None, None, :]) + self.eps

    # 3. Collapse parallel streams: [B, S, k, D] * [B, S, k] -> [B, S, D]
    collapsed = jnp.einsum("bsed,bse -> bsd", x, pre, precision=self.matmul_precision)
    return collapsed

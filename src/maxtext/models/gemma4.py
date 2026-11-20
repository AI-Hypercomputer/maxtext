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

import functools

import jax
from jax.experimental import xla_metadata
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
import jax.numpy as jnp

from flax import nnx
from typing import Optional, Any

from maxtext.common.common_types import Config, AttentionType, MODEL_MODE_PREFILL, ShardMode
from maxtext.layers import initializers
from maxtext.layers import moe
from maxtext.layers import nnx_scan
from maxtext.layers import quantizations
from maxtext.layers.attentions import Attention
from maxtext.layers.linears import MlpBlock

import jax.sharding
from maxtext.layers.normalizations import RMSNorm, align_scale_with_normalized_axis
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils.sharding import create_sharding, get_logical_axis_rules, maybe_shard_with_logical, truncate_out_sharding, with_logical_constraint


GEMMA4_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def get_attention_type(layer_id):
  layer_id %= len(GEMMA4_ATTENTION_PATTERN)
  return GEMMA4_ATTENTION_PATTERN[layer_id]


def _mul_aligned_scale(
    x: jax.Array,
    scale: jax.Array,
    mesh: Mesh,
    shard_mode: ShardMode,
    check_vma: bool = True,
    out_sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Multiplies `x` `[..., K]` by aligned 1-D `scale` `[K]` on VPU while preserving `reduced` FSDP tags."""
  if shard_mode != ShardMode.EXPLICIT:
    return x * scale
  scale_spec = getattr(getattr(jax.typeof(scale), "sharding", None), "spec", None)
  if scale_spec is None or not scale_spec.reduced:
    return x * scale
  x_spec = getattr(getattr(jax.typeof(x), "sharding", None), "spec", None)
  if x_spec is None or mesh is None:
    eff_out = truncate_out_sharding(out_sharding, x.ndim) if out_sharding is not None else None
    return jnp.einsum("...k,k->...k", x, scale, out_sharding=eff_out)
  x_parts = tuple(x_spec.partitions) + (None,) * max(0, x.ndim - len(x_spec.partitions))
  contract_axes = {
      a
      for entry in x_parts[:-1]
      for a in ((entry,) if isinstance(entry, str) else (entry or ()))
      if mesh.shape.get(a, 1) > 1
  }
  reduced_set = frozenset(scale_spec.reduced)
  if not reduced_set.issubset(contract_axes):
    eff_out = truncate_out_sharding(out_sharding, x.ndim) if out_sharding is not None else None
    return jnp.einsum("...k,k->...k", x, scale, out_sharding=eff_out)
  reduced_axes = tuple(sorted(reduced_set))
  extra_axes = tuple(sorted(contract_axes - reduced_set))
  x_full_spec = jax.sharding.PartitionSpec(*x_parts)

  def _mul_local(x_l, s_l):
    s_v = jax.lax.pcast(s_l, reduced_axes, to="varying")
    if extra_axes:
      s_v = jax.lax.pcast(s_v, extra_axes, to="varying")
    return x_l * s_v

  return jax.shard_map(
      _mul_local,
      mesh=mesh,
      in_specs=(x_full_spec, scale_spec),
      out_specs=x_full_spec,
      check_vma=check_vma,
  )(x, scale)


class Gemma4MoE(nnx.Module):
  """Gemma4 specific MoE block containing layer norms and a generic MoE block."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
  ):
    self.config = config
    self.mesh = mesh
    self.rngs = rngs
    self.quant = quant

    self.moe_block = moe.RoutedAndSharedMoE(
        config=config,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal"),
        kernel_axes=("embed", None),
        weight_dtype=config.weight_dtype,
        dtype=config.dtype,
        quant=self.quant,
        rngs=self.rngs,
    )

    self.pre_forward_scale_2 = nnx.Param(
        jnp.ones((self.config.emb_dim,), dtype=self.config.weight_dtype),
        sharding=("embed",),
    )
    self.pre_feedforward_layernorm_2 = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        shard_mode=self.config.shard_mode,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.post_feedforward_layernorm_1 = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        shard_mode=self.config.shard_mode,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.post_feedforward_layernorm_2 = RMSNorm(
        num_features=self.config.emb_dim,
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        shard_mode=self.config.shard_mode,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )
    self.gate_norm = RMSNorm(
        num_features=self.config.emb_dim,
        epsilon=self.config.normalization_layer_epsilon,
        dtype=jnp.float32 if self.config.float32_gate_logits else self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        shard_mode=self.config.shard_mode,
        kernel_axes=("norm",),
        with_scale=False,
        rngs=self.rngs,
    )

  def __call__(
      self,
      inputs: jax.Array,
      original_inputs: jax.Array | None = None,
      intermediate_sharding: jax.sharding.NamedSharding | None = None,
      out_sharding: jax.sharding.NamedSharding | None = None,
      forced_routed_experts: jax.Array | None = None,
      token_mask: jax.Array | None = None,
  ) -> tuple[jax.Array, Optional[jax.Array], Optional[jax.Array]]:
    shared_experts = self.moe_block.shared_experts(
        inputs, intermediate_sharding=intermediate_sharding, out_sharding=out_sharding
    )
    shared_experts = self.post_feedforward_layernorm_1(shared_experts, out_sharding=out_sharding)

    # 1. Experts receive standard RMSNorm (with weight)
    routed_inputs = self.pre_feedforward_layernorm_2(original_inputs, out_sharding=out_sharding)

    # 2. Gate receives RMSNorm (without weight) * root_size * router_scale
    gate_dtype = jnp.float32 if self.config.float32_gate_logits else self.config.dtype
    unscaled_norm = self.gate_norm(original_inputs, out_sharding=out_sharding)

    root_size = self.config.emb_dim**-0.5
    router_scale = jnp.asarray(self.pre_forward_scale_2.value, gate_dtype)
    if self.config.shard_mode == ShardMode.EXPLICIT:
      # The router scale is declared "embed" but multiplies an activation whose last axis is
      # "activation_embed", two different mesh axes under the default rules.
      router_scale = align_scale_with_normalized_axis(router_scale, unscaled_norm)
    gate_inputs = _mul_aligned_scale(
        unscaled_norm * root_size,
        router_scale,
        self.mesh,
        self.config.shard_mode,
        self.config.check_vma,
        out_sharding,
    )

    # 3. Pass both to routed_moe
    routed_experts, load_balance_loss, moe_bias_updates = self.moe_block.routed_moe(
        routed_inputs,
        gate_inputs=gate_inputs,
        out_sharding=out_sharding,
        forced_routed_experts=forced_routed_experts,
        token_mask=token_mask,
    )
    routed_experts = self.post_feedforward_layernorm_2(routed_experts, out_sharding=out_sharding)

    return routed_experts + shared_experts, load_balance_loss, moe_bias_updates


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
    self.attention_type = attention_type
    self.layer_idx = layer_idx

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    if model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
    self.mlp_activation_axis_names = self.activation_axis_names[:-1] + ("activation_mlp",)

    # Physical shardings used to pin sublayer outputs under ShardMode.EXPLICIT. In
    # ShardMode.AUTO these are set to None so callees do not pin physical out_sharding
    # and GSPMD infers the layout via logical axis constraints.
    if config.shard_mode == ShardMode.EXPLICIT:
      self.out_sharding = create_sharding(mesh, self.activation_axis_names, rules=get_logical_axis_rules())
      self.mlp_intermediate_sharding = create_sharding(
          mesh, self.mlp_activation_axis_names, rules=get_logical_axis_rules()
      )
      self._maybe_shard_with_logical = functools.partial(
          maybe_shard_with_logical,
          mesh=mesh,
          shard_mode=config.shard_mode,
          debug_sharding=config.debug_sharding,
      )
    else:
      self.out_sharding = None
      self.mlp_intermediate_sharding = None
      self._maybe_shard_with_logical = with_logical_constraint

    self.pre_self_attention_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

    query_pre_attn_scalar = 1.0
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    share_kv_projections = False

    if attention_type == AttentionType.GLOBAL:
      if hasattr(config, "global_num_kv_heads") and config.global_num_kv_heads:
        num_kv_heads = config.global_num_kv_heads
      if hasattr(config, "global_head_dim") and config.global_head_dim:
        head_dim = config.global_head_dim
      if getattr(config, "share_kv_projections", False):
        share_kv_projections = True

    if attention_type == AttentionType.GLOBAL:
      partial_rotary_factor = config.global_rope_proportion if hasattr(config, "global_rope_proportion") else 0.25
      max_timescale = (
          config.global_rope_max_timescale
          if hasattr(config, "global_rope_max_timescale") and config.global_rope_max_timescale > 0
          else config.rope_max_timescale
      )
    else:  # LOCAL_SLIDING
      partial_rotary_factor = config.local_rope_proportion if hasattr(config, "local_rope_proportion") else 1.0
      max_timescale = (
          config.local_rope_max_timescale
          if hasattr(config, "local_rope_max_timescale") and config.local_rope_max_timescale > 0
          else config.rope_max_timescale
      )

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
        attention_type=self.attention_type,
        sliding_window_size=config.sliding_window_size,
        attn_logits_soft_cap=config.attn_logits_soft_cap,
        use_qk_norm=True,  # Gemma 4 models use query, key normalizations
        use_v_norm=True,
        query_pre_attn_scalar=query_pre_attn_scalar,
        share_kv_projections=share_kv_projections,
        rope_max_timescale=max_timescale,
        partial_rotary_factor=partial_rotary_factor,
        model_mode=model_mode,
        rngs=self.rngs,
    )

    if self.config.use_post_attn_norm:
      self.post_self_attention_norm = RMSNorm(
          num_features=config.emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          shard_mode=config.shard_mode,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_self_attention_norm = None

    self.pre_ffw_norm = RMSNorm(
        num_features=config.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=self.rngs,
    )

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

    if self.config.use_post_ffw_norm:
      self.post_ffw_norm = RMSNorm(
          num_features=config.emb_dim,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          shard_mode=config.shard_mode,
          kernel_axes=("norm",),
          rngs=self.rngs,
      )
    else:
      self.post_ffw_norm = None

    self.layer_scalar = nnx.Param(jnp.ones((1,), dtype=config.weight_dtype), sharding=(None,))

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
      forced_routed_experts: jnp.ndarray | None = None,
  ):
    cfg = self.config
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    is_scan_carry = False
    if isinstance(inputs, tuple) and len(inputs) == 3:
      hidden_states, stacked_kv_cache, layer_idx = inputs
      kv_cache = stacked_kv_cache[layer_idx]
      inputs = hidden_states
      is_scan_carry = True
    elif isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    lnx = self.pre_self_attention_norm(inputs, out_sharding=self.out_sharding)
    lnx = self._maybe_shard_with_logical(lnx, self.activation_axis_names)

    # Gemma4 only applies bidirectional attention in sliding (local) layers,
    # not in full (global) attention layers.
    if self.attention_type != AttentionType.LOCAL_SLIDING:
      bidirectional_mask = None

    # Self-attention block
    attention_lnx, kv_cache = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        bidirectional_mask=bidirectional_mask,
        out_sharding=self.out_sharding,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    if cfg.use_post_attn_norm:
      attention_lnx = self.post_self_attention_norm(attention_lnx, out_sharding=self.out_sharding)
    attention_lnx = self._maybe_shard_with_logical(attention_lnx, self.activation_axis_names)

    attention_lnx += inputs
    residual = attention_lnx
    attn_output = self.pre_ffw_norm(attention_lnx, out_sharding=self.out_sharding)

    # MLP block.
    if getattr(self.config, "num_experts", 1) > 1:
      mlp_lnx, load_balance_loss, _ = self.mlp(
          attn_output,
          original_inputs=attention_lnx,
          intermediate_sharding=self.mlp_intermediate_sharding,
          out_sharding=self.out_sharding,
          forced_routed_experts=forced_routed_experts,
          token_mask=moe.routing_token_mask(cfg, decoder_segment_ids, model_mode),
      )
      if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
        self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)
    else:
      mlp_lnx = self.mlp(
          attn_output,
          deterministic=deterministic,
          intermediate_sharding=self.mlp_intermediate_sharding,
          out_sharding=self.out_sharding,
      )

    if cfg.use_post_ffw_norm:
      mlp_lnx = self.post_ffw_norm(mlp_lnx, out_sharding=self.out_sharding)

    mlp_lnx = self._maybe_shard_with_logical(mlp_lnx, self.activation_axis_names)

    next_layer_addition = mlp_lnx + residual
    layer_output = next_layer_addition
    layer_output = layer_output * jnp.asarray(self.layer_scalar.value, cfg.dtype)

    layer_output = self._maybe_shard_with_logical(layer_output, self.activation_axis_names)

    if getattr(cfg, "record_internal_nn_metrics", False):
      self.sow(nnx.Intermediate, "activation_mean", jnp.mean(layer_output))
      self.sow(nnx.Intermediate, "activation_stdev", jnp.std(layer_output))
      self.sow(
          nnx.Intermediate,
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if is_scan_carry:

      def update_cache(cache, val):
        if jnp.size(val) > 0:
          return cache.at[layer_idx].set(val)
        return cache

      stacked_kv_cache = jax.tree_util.tree_map(update_cache, stacked_kv_cache, kv_cache)
      return (layer_output, stacked_kv_cache, layer_idx + 1), None
    else:
      return layer_output, kv_cache

  # ---------------------------------------------------------------------------
  # Generalized Batch-Split layer adapter (used only when config.enable_batch_split)
  # ---------------------------------------------------------------------------

  def batch_split_supported(self) -> bool:
    """Returns True if this Gemma4 MoE sub-layer supports `batch_split`."""
    cfg = self.config
    return (
        isinstance(self.mlp, Gemma4MoE)
        and getattr(cfg, "num_experts", 1) > 1
        and cfg.sparse_matmul
        and cfg.use_ring_of_experts
        and not cfg.te_moe_block
        and not getattr(cfg, "moe_use_fused_gmm3d", False)
        and cfg.num_moe_emb_chunks <= 0
        and not getattr(cfg, "record_internal_nn_metrics", False)
        and cfg.dropout_rate == 0.0
    )

  def batch_split_attn_and_route(
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
      forced_routed_experts: jnp.ndarray | None = None,
  ):
    """Phase 1/2 TC compute: Sliding/Global attention + norms + shared expert + MoE router gate."""
    del previous_chunk, page_state, slot, kv_cache, forced_routed_experts
    cfg = self.config
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)

    lnx = self.pre_self_attention_norm(inputs, out_sharding=self.out_sharding)
    lnx = self._maybe_shard_with_logical(lnx, self.activation_axis_names)

    if self.attention_type != AttentionType.LOCAL_SLIDING:
      bidirectional_mask = None

    attention_lnx, _ = self.self_attention(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        bidirectional_mask=bidirectional_mask,
        out_sharding=self.out_sharding,
        kv_cache=None,
        attention_metadata=attention_metadata,
    )
    if cfg.use_post_attn_norm and self.post_self_attention_norm is not None:
      attention_lnx = self.post_self_attention_norm(attention_lnx, out_sharding=self.out_sharding)
    attention_lnx = self._maybe_shard_with_logical(attention_lnx, self.activation_axis_names)

    residual = attention_lnx + inputs
    attn_output = self.pre_ffw_norm(residual, out_sharding=self.out_sharding)

    shared_out = self.mlp.moe_block.shared_experts(
        attn_output, intermediate_sharding=self.mlp_intermediate_sharding, out_sharding=self.out_sharding
    )
    shared_out = self.mlp.post_feedforward_layernorm_1(shared_out, out_sharding=self.out_sharding)

    routed_inputs = self.mlp.pre_feedforward_layernorm_2(residual, out_sharding=self.out_sharding)
    gate_dtype = jnp.float32 if cfg.float32_gate_logits else cfg.dtype
    unscaled_norm = self.mlp.gate_norm(residual, out_sharding=self.out_sharding)
    root_size = cfg.emb_dim**-0.5
    router_scale = jnp.asarray(self.mlp.pre_forward_scale_2.value, gate_dtype)
    if cfg.shard_mode == ShardMode.EXPLICIT:
      router_scale = align_scale_with_normalized_axis(router_scale, unscaled_norm)
    gate_inputs = _mul_aligned_scale(
        unscaled_norm * root_size,
        router_scale,
        self.mesh,
        cfg.shard_mode,
        cfg.check_vma,
        self.out_sharding,
    )

    moe_in, route_in, seq_lb_loss = self.mlp.moe_block.routed_moe.batch_split_route(
        routed_inputs,
        gate_inputs=gate_inputs,
    )

    post_ln2_scale = jnp.asarray(self.mlp.post_feedforward_layernorm_2.scale.get_value(), jnp.float32)
    if cfg.shard_mode == ShardMode.EXPLICIT:
      post_ln2_scale = align_scale_with_normalized_axis(post_ln2_scale, residual)

    post_ffw_scale = None
    if cfg.use_post_ffw_norm and self.post_ffw_norm is not None:
      post_ffw_scale = jnp.asarray(self.post_ffw_norm.scale.get_value(), jnp.float32)
      if cfg.shard_mode == ShardMode.EXPLICIT:
        post_ffw_scale = align_scale_with_normalized_axis(post_ffw_scale, residual)

    layer_scalar = jnp.asarray(self.layer_scalar.value, cfg.dtype)
    h_skip = (residual, shared_out, post_ln2_scale, post_ffw_scale, layer_scalar)
    return h_skip, moe_in, route_in, seq_lb_loss

  def batch_split_dispatch(self, moe_in, route_in):
    """Weight-free EP All-Gather of microbatch tokens and routing metadata on SC0."""
    return self.mlp.moe_block.routed_moe.batch_split_dispatch(moe_in, route_in)

  def batch_split_dispatch_bwd(self, moe_in, route_in, g_x_val_ag, g_route_ag_diff):
    """Weight-free backward of `batch_split_dispatch` (EP Reduce-Scatter on SC0)."""
    return self.mlp.moe_block.routed_moe.batch_split_dispatch_bwd(moe_in, route_in, g_x_val_ag, g_route_ag_diff)

  def batch_split_experts(
      self, moe_in_ref, x_val_ag, x_scale_ag, route_ag, decoder_input_tokens=None, return_wgrad_hooks: bool = False
  ):
    """Local expert permute + GMMs + unpermute without EP dispatch/combine."""
    del decoder_input_tokens
    return self.mlp.moe_block.routed_moe.batch_split_experts(
        moe_in_ref, x_val_ag, x_scale_ag, route_ag, return_wgrad_hooks=return_wgrad_hooks
    )

  def batch_split_shared_expert(self, moe_in):
    """Shared expert runs in `batch_split_attn_and_route` (`attn_output` pre-norm branch) and rides `h_skip`."""
    return jnp.zeros_like(moe_in)

  def batch_split_combine(self, y_partial, moe_in_ref):
    """Weight-free EP Reduce-Scatter of routed expert partial outputs on SC0."""
    routed_out = self.mlp.moe_block.routed_moe.batch_split_combine(y_partial, moe_in_ref)
    return self._maybe_shard_with_logical(routed_out, self.activation_axis_names)

  @staticmethod
  def _apply_rms_norm_pure(x: jax.Array, scale_fp32: jax.Array, norm_mod: RMSNorm, out_sharding) -> jax.Array:
    """Applies `norm_mod`'s exact RMSNorm formula using pre-fetched `scale_fp32` (no `nnx.Param` reads)."""
    x_f32 = jnp.asarray(x, jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True) + norm_mod.epsilon)
    eff_out_sharding = out_sharding if norm_mod.shard_mode == ShardMode.EXPLICIT else None
    if eff_out_sharding is not None:
      eff_out_sharding = truncate_out_sharding(eff_out_sharding, x.ndim)
    if norm_mod.scale_offset != 0.0:
      normed_f32 = x_f32 * inv_rms
      offset = jnp.asarray(norm_mod.scale_offset, jnp.float32)
      scale_reduced = jax.typeof(scale_fp32).sharding.spec.reduced
      if scale_reduced:
        offset = jax.sharding.reshard(offset, jax.sharding.PartitionSpec(reduced=scale_reduced))
      y = jnp.einsum("...k,k->...k", normed_f32, scale_fp32 + offset, out_sharding=eff_out_sharding)
      return jnp.asarray(y, norm_mod.dtype)
    y = jnp.asarray(x_f32 * inv_rms, norm_mod.dtype)
    return jnp.einsum("...k,k->...k", y, jnp.asarray(scale_fp32, norm_mod.dtype), out_sharding=eff_out_sharding)

  def batch_split_post_combine(self, mlp_sum, h_skip):
    """Weight-free Gemma4 post-combine norms, shared-expert addition, residual, and `layer_scalar`."""
    residual, shared_out, post_ln2_scale, post_ffw_scale, layer_scalar = h_skip
    routed_normed = self._apply_rms_norm_pure(
        mlp_sum, post_ln2_scale, self.mlp.post_feedforward_layernorm_2, self.out_sharding
    )
    mlp_lnx = routed_normed + shared_out
    if post_ffw_scale is not None and self.post_ffw_norm is not None:
      mlp_lnx = self._apply_rms_norm_pure(mlp_lnx, post_ffw_scale, self.post_ffw_norm, self.out_sharding)
    mlp_lnx = self._maybe_shard_with_logical(mlp_lnx, self.activation_axis_names)
    layer_output = (mlp_lnx + residual) * jnp.asarray(layer_scalar, self.config.dtype)
    return self._maybe_shard_with_logical(layer_output, self.activation_axis_names)

  def batch_split_combine_bwd(self, g_out, moe_in_ref, *, defer_dequant: bool = False):
    """Weight-free backward of `batch_split_combine` (EP All-Gather on SC0)."""
    return self.mlp.moe_block.routed_moe.batch_split_combine_bwd(g_out, moe_in_ref, defer_dequant=defer_dequant)

  def batch_split_finalize_bias_updates(self, summed_counts):
    """Finalizes microbatch-summed expert counts into routed bias updates."""
    return self.mlp.moe_block.routed_moe.batch_split_finalize_bias_updates(summed_counts)


class Gemma4ScannableBlock(nnx.Module):
  """A repeatable block of Gemma4 decoder layers, scanning local layers."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str,
      rngs: nnx.Rngs,
      quant: None | Quant = None,
      num_of_layers: int = 6,
      remat_policy_fn: Any = None,
      apply_internal_remat: bool = False,
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
    if config.shard_mode == ShardMode.EXPLICIT:
      self._maybe_shard_with_logical = functools.partial(
          maybe_shard_with_logical,
          mesh=mesh,
          shard_mode=config.shard_mode,
          debug_sharding=config.debug_sharding,
      )
    else:
      self._maybe_shard_with_logical = with_logical_constraint

    pattern_length = len(GEMMA4_ATTENTION_PATTERN)
    if not 0 <= num_of_layers <= pattern_length:
      raise ValueError(f"Gemma4ScannableBlock must contain between 0 and {pattern_length} layers; got {num_of_layers}.")

    # The block runs its local (sliding-window) layers first, then a single
    # global layer, matching GEMMA4_ATTENTION_PATTERN. Derive the per-type
    # counts from the pattern (well, its first num_of_layers entries) rather
    # than hardcoding the 5-local / 1-global split.
    active_pattern = GEMMA4_ATTENTION_PATTERN[:num_of_layers]
    self.num_local = sum(1 for attn_type in active_pattern if attn_type == AttentionType.LOCAL_SLIDING)
    self.num_global = sum(1 for attn_type in active_pattern if attn_type == AttentionType.GLOBAL)

    # num_of_layers can be 0: the decoders always construct a "remainder" block
    # for num_decoder_layers % pattern_length layers, which is 0 whenever the
    # layer count divides evenly (e.g. 31b = 60 layers). That block is built but
    # never applied, so num_local or num_global may legitimately be 0 here.
    if self.num_local > 0:
      self.local_layers = nnx_scan.create_scanned_layers(
          lambda layer_rngs: Gemma4DecoderLayer(
              config=self.config,
              mesh=self.mesh,
              model_mode=self.model_mode,
              quant=self.quant,
              rngs=layer_rngs,
              attention_type=AttentionType.LOCAL_SLIDING,
              layer_idx=0,  # layer_idx is not used in the class
          ),
          length=self.num_local,
          param_scan_axis=self.config.param_scan_axis,
          metadata_axis_name="local_layers",
          rngs=self.rngs,
      )
    else:
      self.local_layers = None

    if self.num_global > 0:
      self.global_layer = Gemma4DecoderLayer(
          config=self.config,
          mesh=self.mesh,
          model_mode=self.model_mode,
          rngs=self.rngs,
          quant=self.quant,
          attention_type=AttentionType.GLOBAL,
          layer_idx=5,  # layer_idx is not used in the class
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
    override = getattr(self, "remat_policy_override", None)
    remat_policy = self.config.remat_policy if override is None else override
    return self.apply_internal_remat and bool(remat_policy) and remat_policy != "none"

  def _scan_local_layers(self, y, layer_kwargs):
    """Runs the local (sliding-window) layers via a per-layer rematerialized ``jax.lax.scan``."""
    remat = self._remat_enabled
    override = getattr(self, "remat_policy_override", None)
    if override is None:
      local_remat_policy = self.remat_policy_fn
    else:
      local_remat_policy = getattr(self, "remat_policy_fn_override", None)
    return nnx_scan.apply_scanned_layers(
        self.local_layers,
        y,
        length=self.num_local,
        param_scan_axis=self.config.param_scan_axis,
        apply_fn=lambda layer, carry: self._run_layer(layer, carry, layer_kwargs)[0],
        remat=remat,
        remat_policy=local_remat_policy if remat else None,
        # prevent_cse is only consulted by jax.checkpoint, i.e. when remat=True;
        # its value is irrelevant otherwise.
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

    def run_global_layer(carry, intermediate_slice):
      hidden_states, other = carry
      layer = nnx.merge(graphdef_g, intermediate_slice, other)
      new_hidden_states = self._run_layer(layer, hidden_states, layer_kwargs)[0]
      _, new_intermediate, new_other = nnx.split(layer, nnx.Intermediate, ...)
      return (new_hidden_states, new_other), new_intermediate

    # Offloaded (pinned-host) residuals can't cross the trip-count-one boundary,
    # so save would-be-offloaded tensors on device for the global layer instead;
    # the local-layer scan (a real multi-iteration scan) still offloads.
    override = getattr(self, "remat_policy_override", None)
    if override is not None:
      global_remat_policy = getattr(self, "scan_safe_remat_policy_fn_override", None)
    else:
      global_remat_policy = self.remat_policy_fn
      offload_names = maxtext_utils.get_save_and_offload_names(cfg)
      if offload_names[0] or offload_names[1]:
        save_names, offload_to_device = offload_names
        global_remat_policy = jax.checkpoint_policies.save_only_these_names(*(save_names + offload_to_device))

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
    with xla_metadata.set_xla_metadata(**{"skip-simplify-while-loops_trip-count-one": "true"}):
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
      # caches, and re-stack the per-layer state.
      graphdef, params, state = nnx.split(self.local_layers, nnx.Param, ...)
      scan_axis = self.config.param_scan_axis
      if scan_axis != 0:
        params = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0), params)
      per_layer_states = []
      for i in range(self.num_local):
        current_params = jax.tree.map(lambda x, i=i: x[i], params)
        current_state = jax.tree.map(lambda x, i=i: x[i], state)
        layer = nnx.merge(graphdef, current_params, current_state)
        y, new_kv = self._run_layer(layer, y, layer_kwargs, kv_cache[i])
        updated_kvs.append(new_kv)
        per_layer_states.append(nnx.state(layer))

      stacked_state = jax.tree.map(lambda *xs: jnp.stack(xs), *per_layer_states)
      if scan_axis != 0:
        stacked_params, stacked_other = stacked_state.split(nnx.Param, ...)
        stacked_params = jax.tree.map(lambda x: jnp.moveaxis(x, 0, scan_axis), stacked_params)
        stacked_state = nnx.State.merge(stacked_params, stacked_other)
      nnx.update(self.local_layers, stacked_state)

    if self.global_layer is not None:
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
    # The per-layer constraints inside Gemma4DecoderLayer are model_mode aware; this
    # block-level one deliberately is not, matching the axis names this code has always
    # used so ShardMode.AUTO is unchanged.
    inputs = self._maybe_shard_with_logical(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
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
      return self._forward_with_external_kv_cache(inputs, kv_cache, layer_kwargs)

    y = inputs
    if self.local_layers is not None:
      y = self._scan_local_layers(y, layer_kwargs)
    if self.global_layer is not None:
      y = self._scan_global_layer(y, layer_kwargs)

    if cfg.scan_layers:
      return y, None
    return y

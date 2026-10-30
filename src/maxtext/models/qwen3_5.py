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

"""Qwen3.5 family of model decoder layers."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

import functools
import os
from typing import Any, cast

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.common.common_types import MODEL_MODE_TRAIN, Array, Config, ShardMode, get_weight_dtype
from maxtext.layers.linears import MlpBlock
from maxtext.layers.moe import backward_barrier_tie, barrier_tie, combine_ep_ubatch_bwd_fn
from maxtext.layers.normalizations import Qwen3NextRMSNorm
from maxtext.layers.quantizations import AqtQuantization as Quant
from maxtext.models.qwen3 import (
    Qwen3NextFullAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)
from maxtext.utils import max_utils
from maxtext.utils.sharding import create_sharding, get_logical_axis_rules, maybe_shard_with_logical

# -----------------------------------------
# Qwen3.5 Layer Implementations
# -----------------------------------------


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
  """Qwen3.5 GatedDeltaNet layer that is identical to Qwen3-Next GatedDeltaNet"""


class Qwen3_5FullAttention(Qwen3NextFullAttention):
  """Qwen3.5 Gated Attention layer that is identical to Qwen3-Next"""


class Qwen3_5SparseMoEBlock(Qwen3NextSparseMoeBlock):
  """Shares same MoE code as Qwen3-Next"""


def _noop_shard_with_logical(inputs, *args, **kwargs):
  return inputs


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _combine_bwd_bridge(combine_bwd_fn, unperm_prev, base_prev, x_mb1):
  """Zero-residual bridge routing `dx_mb1` back into `(d_unperm_prev, d_base_prev)` during `B_F1`."""
  del combine_bwd_fn, unperm_prev, base_prev
  return x_mb1


def _combine_bwd_bridge_fwd(combine_bwd_fn, unperm_prev, base_prev, x_mb1):
  del combine_bwd_fn, unperm_prev, base_prev
  return x_mb1, ()


def _combine_bwd_bridge_bwd(combine_bwd_fn, _, g_x_mb1):
  return combine_bwd_fn(g_x_mb1), g_x_mb1, jnp.zeros_like(g_x_mb1)


_combine_bwd_bridge.defvjp(_combine_bwd_bridge_fwd, _combine_bwd_bridge_bwd)


class Qwen3_5ScannableBlock(nnx.Module):
  """Scanned Structure for Text-only Architecture, explicitly invoking Qwen3_5 layers."""

  def __init__(self, config: Config, mesh: Mesh, model_mode: str, quant=None, *, rngs: nnx.Rngs):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.rngs = rngs
    cfg = self.config

    # Explicitly instantiate Qwen3_5DecoderLayer here
    for i in range(cfg.inhomogeneous_layer_cycle_interval):
      layer_rngs = self.rngs.fork()
      layer_name = f"layer_{i}"
      layer = Qwen3_5DecoderLayer(
          config=self.config,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=self.model_mode,
          layer_idx=i,
          rngs=layer_rngs,
      )
      setattr(self, layer_name, layer)

  def prefetch_sublayer_names(self) -> tuple[str, ...]:
    return tuple(f'layer_{i}' for i in range(self.config.inhomogeneous_layer_cycle_interval))

  def __call__(
      self,
      carry: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | tuple[Any, ...] = None,
      attention_metadata: None | dict[str, Any] = None,
      forced_routed_experts: jnp.ndarray | None = None,
  ) -> tuple[Array, None | tuple[Any, ...]]:
    cfg = self.config
    if getattr(cfg, "prefetch_fsdp_batch_split", False) and isinstance(carry, tuple) and len(carry) == 3:
      from maxtext.layers import fsdp_prefetch_inhomogeneous  # pylint: disable=import-outside-toplevel

      x_mb0, unperm_mb1, base_mb1 = carry
      act_sharding, _ = fsdp_prefetch_inhomogeneous._batchsplit_shardings_and_ep(self.mesh, cfg)
      for i in range(cfg.inhomogeneous_layer_cycle_interval):
        layer = getattr(self, f"layer_{i}")
        if forced_routed_experts is None:
          layer_forced = None
        elif isinstance(forced_routed_experts, tuple):
          layer_forced = (forced_routed_experts[0][i], forced_routed_experts[1][i])
        else:
          layer_forced = forced_routed_experts[i]
        if os.environ.get("MAXTEXT_TIE_F1_MB1_COMBINE", "0") == "1":
          x_mb0, unperm_mb1 = barrier_tie(x_mb0, unperm_mb1, backward=False)
        x_mb1 = fsdp_prefetch_inhomogeneous._combine_add_residual(
            base_mb1, unperm_mb1, self.mesh, cfg, act_sharding, cfg.shard_mode
        )
        x_mb0 = checkpoint_name(x_mb0, "decoder_layer_input")
        x_mb1 = checkpoint_name(x_mb1, "decoder_layer_input")
        moe_block = cast(Qwen3_5SparseMoEBlock, layer.mlp)
        w_tuple = moe_block.routed_experts.prepare_batchsplit_weights()
        bwd_tie_mode = os.environ.get("MAXTEXT_TIE_BWD_SHARDED_MOE", "all4")
        if (
            bwd_tie_mode == "all4"
            or (bwd_tie_mode == "top3" and i >= 1)
            or (bwd_tie_mode == "mid2" and i in (1, 2))
        ):
          w0_k, w1_k, wo_k, w0_b, w1_b, wo_b, fq_g = w_tuple
          if w1_k is not None:
            (x_mb0, x_mb1), (w0_k, w1_k, wo_k) = backward_barrier_tie((x_mb0, x_mb1), (w0_k, w1_k, wo_k))
          else:
            (x_mb0, x_mb1), (w0_k, wo_k) = backward_barrier_tie((x_mb0, x_mb1), (w0_k, wo_k))
          w_tuple = (w0_k, w1_k, wo_k, w0_b, w1_b, wo_b, fq_g)
        (x_mb0, unperm_mb1, base_mb1), _ = layer.call_batchsplit(
            (x_mb0, None, None, x_mb1),
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            previous_chunk,
            slot,
            kv_cache=None,
            attention_metadata=attention_metadata,
            forced_routed_experts=layer_forced,
            weights_tuple=w_tuple,
        )
      return (x_mb0, unperm_mb1, base_mb1), None
    x = carry
    # Inference with externally-managed (vLLM) caches hands this block one
    # entry per sub-layer of the cycle; the decoder groups the flat per-layer
    # list before the scan and ungroups the returned tuple afterwards.
    updated_kv_caches = []

    for i in range(cfg.inhomogeneous_layer_cycle_interval):
      layer = getattr(self, f"layer_{i}")
      # forced_routed_experts, when present, is shaped
      # [inhomogeneous_layer_cycle_interval, batch, seq, top_k]: one slice per
      # sub-layer in this cycle (see nnx_decoders.py's scan wiring).
      layer_forced_routed_experts = forced_routed_experts[i] if forced_routed_experts is not None else None
      layer_kv_cache = kv_cache[i] if (kv_cache is not None and i < len(kv_cache)) else None
      x, new_kv_cache = layer(
          x,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk,
          slot,
          kv_cache=layer_kv_cache,
          attention_metadata=attention_metadata,
          forced_routed_experts=layer_forced_routed_experts,
      )
      updated_kv_caches.append(new_kv_cache)

    if kv_cache is None:
      return x, None
    return x, tuple(updated_kv_caches)


class Qwen3_5DecoderLayer(nnx.Module):
  """
  This layer is a hybrid, capable of functioning as either:
  1. A standard attention + MoE layer.
  2. A linear attention + MoE layer.

  Attributes:
    config: The model configuration object.
    mesh: The device mesh for sharding.
    model_mode: The operational mode (e.g., 'train', 'prefill').
    layer_idx: The index of the current layer in the transformer stack.
    quant: Optional quantization configuration.
  """

  def __init__(
      self, config: Config, mesh: Mesh, model_mode: str, layer_idx: int, quant: None | Quant = None, *, rngs: nnx.Rngs
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    cfg = self.config
    is_full_attn = (layer_idx + 1) % cfg.inhomogeneous_layer_cycle_interval == 0
    self.layer_idx = layer_idx if (is_full_attn or model_mode != MODEL_MODE_TRAIN) else 0
    self.quant = quant
    self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")
    self.mlp_activation_axis_names = (
        "activation_batch",
        "activation_norm_length",
        "activation_mlp",
    )

    # Physical shardings used to pin sublayer outputs under ShardMode.EXPLICIT. In
    # ShardMode.AUTO the callees ignore these and let GSPMD infer the layout.
    if cfg.shard_mode == ShardMode.EXPLICIT:
      self.out_sharding = create_sharding(mesh, self.activation_axis_names, rules=get_logical_axis_rules())
      self.mlp_intermediate_sharding = create_sharding(
          mesh, self.mlp_activation_axis_names, rules=get_logical_axis_rules()
      )
      self._maybe_shard_with_logical = functools.partial(
          maybe_shard_with_logical,
          mesh=mesh,
          shard_mode=cfg.shard_mode,
          debug_sharding=cfg.debug_sharding,
      )
    else:
      self.out_sharding = None
      self.mlp_intermediate_sharding = None
      self._maybe_shard_with_logical = _noop_shard_with_logical

    # First LayerNorm, applied before the attention block.
    self.input_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=cfg.dtype,
        weight_dtype=get_weight_dtype(cfg, "norm"),
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )

    # Determine the type of attention mechanism for the current layer.
    is_full_attention_layer = is_full_attn

    # Conditionally instantiate either the Linear Attention or Full Attention block.
    if is_full_attention_layer:
      self.attention = Qwen3_5FullAttention(
          config=cfg,
          mesh=self.mesh,
          quant=self.quant,
          model_mode=model_mode,
          layer_idx=self.layer_idx,
          rngs=rngs,
      )
    else:
      batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
      dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)
      self.attention = Qwen3_5GatedDeltaNet(
          config=cfg,
          inputs_shape=dummy_inputs_shape,
          mesh=self.mesh,
          dtype=cfg.dtype,
          model_mode=model_mode,
          quant=self.quant,
          rngs=rngs,
      )

    # Second LayerNorm, applied before the MoE block (outputs fp32 into MoE router when float32_gate_logits=True).
    self.post_attention_layernorm = Qwen3NextRMSNorm(
        num_features=cfg.emb_dim,
        epsilon=cfg.normalization_layer_epsilon,
        dtype=jnp.float32 if cfg.float32_gate_logits else cfg.dtype,
        weight_dtype=jnp.float32 if cfg.float32_gate_logits else get_weight_dtype(cfg, "norm"),
        shard_mode=cfg.shard_mode,
        rngs=rngs,
    )

    if getattr(cfg, "num_experts", 1) > 1:
      self.mlp = Qwen3_5SparseMoEBlock(config=cfg, mesh=self.mesh, quant=self.quant, rngs=rngs)
    else:
      self.mlp = MlpBlock(
          config=cfg,
          mesh=self.mesh,
          in_features=cfg.emb_dim,
          intermediate_dim=cfg.mlp_dim,
          activations=tuple(cfg.mlp_activations),
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=get_weight_dtype(cfg, "mlp"),
          quant=self.quant,
          model_mode=model_mode,
          rngs=rngs,
      )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | dict[str, Array] = None,
      attention_metadata: None | dict[str, Any] = None,
      forced_routed_experts: jnp.ndarray | None = None,
  ):
    # Unpack inputs if it's a tuple (e.g. from a previous layer returning (hidden_states, kv_cache))
    if isinstance(inputs, tuple):
      inputs = inputs[0]
    if not getattr(self.config, "prefetch_fsdp_weights", False):
      inputs = checkpoint_name(inputs, "decoder_layer_input")
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)
    residual = inputs

    if isinstance(attention_metadata, dict):
      attention_metadata = attention_metadata.get(f"layer.{self.layer_idx}", attention_metadata.get(self.layer_idx))

    # First LayerNorm, applied before the attention block.
    hidden_states = self.input_layernorm(inputs, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    # Conditionally apply either the Linear Attention or Full Attention block.
    if isinstance(self.attention, Qwen3_5FullAttention):
      attention_output, new_kv_cache = cast(Qwen3_5FullAttention, self.attention)(
          hidden_states,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
          out_sharding=self.out_sharding,
      )
    else:
      attention_output, new_kv_cache = cast(Qwen3_5GatedDeltaNet, self.attention)(
          hidden_states,
          model_mode=model_mode,
          kv_cache=kv_cache,
          decoder_segment_ids=decoder_segment_ids,
          attention_metadata=attention_metadata,
          out_sharding=self.out_sharding,
      )

    # First residual connection after attention
    attention_output = self._maybe_shard_with_logical(attention_output, self.activation_axis_names)
    hidden_states = residual + attention_output
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    # Prepare for the MoE / MLP block by capturing the new residual
    residual = hidden_states

    # Second LayerNorm, applied before the MoE / MLP block.
    hidden_states = self.post_attention_layernorm(hidden_states, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)

    # Instantiate and call our `Qwen3_5SparseMoEBlock` or `MlpBlock`.
    if getattr(self.config, "num_experts", 1) > 1:
      # Padding tokens (segment id 0) are excluded from MoE routing when moe_exclude_padding_from_routing.
      token_mask = None
      if (
          self.config.moe_exclude_padding_from_routing
          and decoder_segment_ids is not None
          and model_mode == MODEL_MODE_TRAIN
      ):
        token_mask = decoder_segment_ids != 0
      mlp_output, load_balance_loss = self.mlp(
          hidden_states,
          deterministic=deterministic,
          forced_routed_experts=forced_routed_experts,
          intermediate_sharding=self.mlp_intermediate_sharding,
          out_sharding=self.out_sharding,
          token_mask=token_mask,
      )

      # We sow the load balancing loss so it can be collected and added to the total loss
      # during training.
      if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
        self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)
    else:
      mlp_output = self.mlp(
          hidden_states,
          deterministic=deterministic,
          intermediate_sharding=self.mlp_intermediate_sharding,
          out_sharding=self.out_sharding,
      )

    # Final residual connection (after the MoE block)
    mlp_output = self._maybe_shard_with_logical(mlp_output, self.activation_axis_names)
    layer_output = residual + mlp_output
    layer_output = self._maybe_shard_with_logical(
        layer_output,
        self.activation_axis_names,
    )
    return layer_output, new_kv_cache

  def _attn_and_shared(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: None | jnp.ndarray,
      decoder_positions: None | jnp.ndarray,
      deterministic: bool,
      model_mode: str,
      kv_cache: None | dict[str, Array] = None,
      attention_metadata: None | dict[str, Any] = None,
      forced_routed_experts: jnp.ndarray | None = None,
  ):
    """Runs `input_layernorm -> attention/GDN -> +residual -> post_attn_norm -> shared_expert + router_gate` on 1 ubatch."""
    inputs = self._maybe_shard_with_logical(inputs, self.activation_axis_names)
    residual = inputs
    if isinstance(attention_metadata, dict):
      attention_metadata = attention_metadata.get(f"layer.{self.layer_idx}", attention_metadata.get(self.layer_idx))
    hidden_states = self.input_layernorm(inputs, out_sharding=self.out_sharding)
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)
    if isinstance(self.attention, Qwen3_5FullAttention):
      attention_output, _ = cast(Qwen3_5FullAttention, self.attention)(
          hidden_states,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          kv_cache=kv_cache,
          attention_metadata=attention_metadata,
          out_sharding=self.out_sharding,
      )
    else:
      attention_output, _ = cast(Qwen3_5GatedDeltaNet, self.attention)(
          hidden_states,
          model_mode=model_mode,
          kv_cache=kv_cache,
          decoder_segment_ids=decoder_segment_ids,
          attention_metadata=attention_metadata,
          out_sharding=self.out_sharding,
      )
    attention_output = self._maybe_shard_with_logical(attention_output, self.activation_axis_names)
    hidden_states = residual + attention_output
    hidden_states = self._maybe_shard_with_logical(hidden_states, self.activation_axis_names)
    post_attn_residual = hidden_states

    normed = self.post_attention_layernorm(hidden_states, out_sharding=self.out_sharding)
    normed = self._maybe_shard_with_logical(normed, self.activation_axis_names)

    token_mask = None
    if (
        self.config.moe_exclude_padding_from_routing
        and decoder_segment_ids is not None
        and model_mode == MODEL_MODE_TRAIN
    ):
      token_mask = decoder_segment_ids != 0
    exclude_padding = token_mask is not None and self.config.moe_exclude_padding_from_routing

    moe_block = cast(Qwen3_5SparseMoEBlock, self.mlp)
    normed_bf16, gate_logits, pre_bias_logits, forced = moe_block.routed_experts.compute_gate(
        normed,
        forced_routed_experts=forced_routed_experts,
        token_mask=token_mask if exclude_padding else None,
    )
    if moe_block.use_shared_expert:
      shared_expert_output = moe_block.shared_expert(
          normed,
          deterministic=deterministic,
          intermediate_sharding=self.mlp_intermediate_sharding,
          out_sharding=self.out_sharding,
      )
      shared_gate_output = moe_block.shared_expert_gate(
          normed.astype(jnp.float32) if self.config.float32_gate_logits else normed
      )
      if self.config.float32_weight_sum:
        shared_gate = jax.nn.sigmoid(shared_gate_output.astype(jnp.float32))
        if exclude_padding:
          shared_gate = shared_gate * token_mask.astype(shared_gate.dtype)[..., None]
        shared_term = (shared_gate * shared_expert_output.astype(jnp.float32)).astype(self.config.dtype)
      else:
        shared_gate = jax.nn.sigmoid(shared_gate_output)
        if exclude_padding:
          shared_gate = shared_gate * token_mask.astype(shared_gate.dtype)[..., None]
        shared_term = shared_gate * shared_expert_output
      shared_term = self._maybe_shard_with_logical(shared_term, self.activation_axis_names)
      base_out = self._maybe_shard_with_logical(post_attn_residual + shared_term, self.activation_axis_names)
    else:
      base_out = post_attn_residual
    return base_out, normed_bf16, gate_logits, pre_bias_logits, forced

  def call_batchsplit(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic: bool,
      model_mode: str,
      previous_chunk=None,
      slot: None | int = None,
      kv_cache: None | dict[str, Array] = None,
      attention_metadata: None | dict[str, Any] = None,
      forced_routed_experts=None,
      weights_tuple=None,
  ):
    """4-stage batch-split schedule overlapping EP dispatch & combine with GDN/SplashAttn + SharedExpert + GMMs."""
    del previous_chunk, slot
    x_mb0, unperm_prev, base_prev, x_mb1 = inputs
    seg_mb0, seg_mb1 = (
        decoder_segment_ids if isinstance(decoder_segment_ids, tuple) else (decoder_segment_ids, decoder_segment_ids)
    )
    pos_mb0, pos_mb1 = (
        decoder_positions if isinstance(decoder_positions, tuple) else (decoder_positions, decoder_positions)
    )
    forced_mb0, forced_mb1 = (
        forced_routed_experts
        if isinstance(forced_routed_experts, tuple)
        else (forced_routed_experts, forced_routed_experts)
    )

    # F1: mb0 GDN/SplashAttn + SharedExpert + RouterGate (overlaps prev sub-layer's mb1 EP combine + next FSDP AG!)
    base_mb0, normed_mb0, logits_mb0, pre_bias_mb0, forced_mb0_out = self._attn_and_shared(
        x_mb0,
        seg_mb0,
        pos_mb0,
        deterministic,
        model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
        forced_routed_experts=forced_mb0,
    )

    moe_block = cast(Qwen3_5SparseMoEBlock, self.mlp)
    if weights_tuple is None:
      weights_tuple = moe_block.routed_experts.prepare_batchsplit_weights()
    if os.environ.get("MAXTEXT_BATCHSPLIT_EARLY_WEIGHT_AG", "0") == "1":
      weights_tuple, x_mb1 = moe_block.routed_experts.prefetch_batchsplit_weights(x_mb1, weights_tuple)

    # Bridge + Tie 1 (forward-only so mb0 backward EP dispatch RS overlaps B_F2 = mb1 attn+shared bwd!)
    if unperm_prev is not None:
      combine_bwd_fn = functools.partial(combine_ep_ubatch_bwd_fn, mesh=self.mesh, config=self.config)
      x_mb1_b = _combine_bwd_bridge(combine_bwd_fn, unperm_prev, base_prev, x_mb1)
    else:
      x_mb1_b = x_mb1
    (normed_mb0, logits_mb0), x_mb1_b = barrier_tie((normed_mb0, logits_mb0), x_mb1_b, backward=False)

    # F2: mb0 EP dispatch || mb1 GDN/SplashAttn + SharedExpert + RouterGate (TC)
    routed_mb0 = moe_block.routed_experts.batchsplit_dispatch(
        normed_mb0,
        logits_mb0,
        pre_bias_mb0,
        forced_routed_experts=forced_mb0_out,
    )
    base_mb1_out, normed_mb1, logits_mb1, pre_bias_mb1, forced_mb1_out = self._attn_and_shared(
        x_mb1_b,
        seg_mb1,
        pos_mb1,
        deterministic,
        model_mode,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
        forced_routed_experts=forced_mb1,
    )

    # Tie 2 (forward-only so mb1 backward EP dispatch RS overlaps B_F3 = mb0 wo+wi GMMs bwd!)
    r0_tok, r0_rest, r0_meta = routed_mb0
    (normed_mb1, logits_mb1), r0_tok = barrier_tie((normed_mb1, logits_mb1), r0_tok, backward=False)
    routed_mb0 = (r0_tok, r0_rest, r0_meta)

    # F3 + F4: mb0 GMMs || mb1 EP dispatch, then mb1 GMMs || mb0 EP combine + mb1 unpermute (deferred to next layer)
    routed_out_mb0, unperm_mb1_out, load_balance_loss, _ = moe_block.routed_experts.batchsplit_f3_f4(
        routed_mb0,
        normed_mb1,
        logits_mb1,
        pre_bias_mb1,
        weights_tuple,
        forced_mb1=forced_mb1_out,
    )
    if self.config.load_balance_loss_weight > 0.0 and load_balance_loss is not None:
      self.sow(nnx.Intermediate, "moe_lb_loss", load_balance_loss)

    if self.config.float32_weight_sum:
      x_mb0_out = (base_mb0.astype(jnp.float32) + routed_out_mb0.astype(jnp.float32)).astype(self.config.dtype)
    else:
      x_mb0_out = base_mb0 + routed_out_mb0
    x_mb0_out = self._maybe_shard_with_logical(x_mb0_out, self.activation_axis_names)
    return (x_mb0_out, unperm_mb1_out, base_mb1_out), None

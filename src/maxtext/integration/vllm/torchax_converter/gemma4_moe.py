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

"""Gemma4 MaxText to vLLM weight converter.

Supports gemma4-26b (MoE: 128 routed + 1 shared expert).

MaxText Gemma4 stores layers in a scanned-block structure:
  state['base']['decoder']['scanned_blocks']['layers_{slot}']
where slot ∈ [0..5].  Slots 0–4 are local-sliding-window attention layers
and slot 5 is a global attention layer.  The 'L' dimension (axis 1 of each
weight tensor) holds 'num_reps = num_layers // 6' repetitions of each slot.
Final vLLM layer index = rep * 6 + slot.

Global attention (slot 5) uses a shared KV projection — 'key' serves as
both K and V; there is no separate 'value' tensor.

Key names and tensor transformations are derived from the MaxText HF param mapping
at src/maxtext/checkpoint_conversion/utils/param_mapping.py.

Attention: Gemma4 uses SEPARATE q/k/v proj weights (not fused QKV).
MoE (26B): gate+up proj are fused into experts.gate_up_proj (E, 2*d_inner, d_model).
Embedding: MaxText stores embedding * sqrt(d_model); divide out before writing to vLLM.
"""

import gc
import logging

import jax
import jax.numpy as jnp
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import FusedMoEWeights
from tpu_inference.layers.common.process_weights.moe_weights import process_moe_weights

from maxtext.integration.vllm.torchax_converter.base import BaseMaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.base import timer
from maxtext.integration.vllm.torchax_converter.base import GREEN
from maxtext.integration.vllm.torchax_converter.base import RESET


class Gemma4MaxTextToVLLMConverter(BaseMaxTextToVLLMConverter):
  """Converts MaxText Gemma4 weights to the layout expected by a vLLM Gemma4 model."""

  NUM_SLOTS = 6  # 5 local + 1 global

  def __init__(self, config, mesh):
    super().__init__(config, mesh)
    assert self.num_layers % self.NUM_SLOTS == 0, f"num_layers {self.num_layers} must be divisible by {self.NUM_SLOTS}"
    self.num_reps = self.num_layers // self.NUM_SLOTS
    self.is_moe = config.model_name == "gemma4-26b"
    self.d_model = config.base_emb_dim

  # --- 1. Top-Level Entry Point ---

  def convert(self, model_state: dict):
    """Convert a MaxText Gemma4 model state into vLLM weight tensors."""
    logging.info(
        "\n%sStarting Gemma4 Conversion (is_moe=%s, num_layers=%d, num_reps=%d)...%s",
        GREEN,
        self.is_moe,
        self.num_layers,
        self.num_reps,
        RESET,
    )
    self.vllm_state = {}
    blocks = model_state["base"]["decoder"]["scanned_blocks"]
    prefix = "vllm_model.language_model.model.layers"

    with timer("Convert Global Weights"):
      self._convert_global(model_state)
    with timer("Convert Layer Norms"):
      self._convert_norms(blocks, prefix)
    with timer("Convert Attention Weights"):
      self._convert_attn_weights(blocks, prefix)
    if self.is_moe:
      with timer("Convert MoE Weights"):
        self._convert_moe_weights(blocks, prefix)
    else:
      with timer("Convert Dense MLP Weights"):
        self._convert_dense_mlp_weights(blocks, prefix)

    return self.vllm_state

  # --- Abstract method implementations (delegate to Gemma4-specific methods) ---

  def _convert_global(self, params):
    """Convert non-layered weights (embed_tokens, lm_head, final norm)."""
    # Gemma4 uses tied embeddings: no logits_dense; lm_head.weight = embed_tokens.weight.
    # MaxText stores embedding pre-multiplied by sqrt(hidden_size) (applied during HF->MaxText
    # conversion in param_mapping.py). vLLM/tpu-inference apply sqrt(hidden_size) at runtime,
    # so divide out the pre-multiplied factor to give vLLM the raw embedding.
    logging.info("_convert_global: embed_tokens (de-normalize) + lm_head (tied) + final_norm...")
    normalizer = self.d_model**0.5

    @jax.jit
    def _denorm_embed(x):
      return (x / normalizer).astype(x.dtype)

    raw_embedding = _denorm_embed(params["base"]["token_embedder"]["embedding"])
    self.vllm_state["vllm_model.language_model.model.embed_tokens.weight"] = raw_embedding
    self.vllm_state["vllm_model.language_model.lm_head.weight"] = raw_embedding  # tied
    self.vllm_state["vllm_model.language_model.model.norm.weight"] = params["base"]["decoder"]["decoder_norm"]["scale"]
    logging.info("_convert_global: done")

  def _convert_attn(self, params):
    """Satisfy abstract interface; Gemma4 uses _convert_attn_weights instead."""
    blocks = params["base"]["decoder"]["scanned_blocks"]
    prefix = "vllm_model.language_model.model.layers"
    self._convert_attn_weights(blocks, prefix)

  def _convert_moe(self, params):
    """Satisfy abstract interface; Gemma4 uses _convert_moe_weights/_convert_dense_mlp_weights."""
    blocks = params["base"]["decoder"]["scanned_blocks"]
    prefix = "vllm_model.language_model.model.layers"
    if self.is_moe:
      self._convert_moe_weights(blocks, prefix)
    else:
      self._convert_dense_mlp_weights(blocks, prefix)

  # --- 2. Static JIT helper ---

  @staticmethod
  @jax.jit
  def _pack_attn(q, k, v, o, qnorm, knorm):
    """Prepares separate q/k/v, o, and norms for all layers in a slot.

    Input shapes (MaxText scanned, scan axis at index 1):
      q/k/v: (d_model, L, nH, D)
      o:     (nH, L, D, d_model)   # scan axis is 1
      norms: (d_model, L)
    Returns: L × (nH*D, d_model) for q/k/v, L × (d_model, nH*D) for o.
    """
    # q/k/v: (d_model, L, nH, D) -> (L, nH, D, d_model) -> (L, nH*D, d_model)
    q = jnp.transpose(q, (1, 2, 3, 0)).reshape(q.shape[1], -1, q.shape[0])
    k = jnp.transpose(k, (1, 2, 3, 0)).reshape(k.shape[1], -1, k.shape[0])
    v = jnp.transpose(v, (1, 2, 3, 0)).reshape(v.shape[1], -1, v.shape[0])
    # o: (nH, L, D, d_model) -> (L, d_model, nH, D) -> (L, d_model, nH*D)
    o = jnp.transpose(o, (1, 3, 0, 2)).reshape(o.shape[1], o.shape[3], -1)
    # norms: (D, L) -> (L, D)
    qnorm = jnp.transpose(qnorm, (1, 0))
    knorm = jnp.transpose(knorm, (1, 0))
    return (
        jnp.unstack(q),
        jnp.unstack(k),
        jnp.unstack(v),
        jnp.unstack(o),
        jnp.unstack(qnorm),
        jnp.unstack(knorm),
    )

  # --- 3. Per-layer norms ---

  def _convert_norms(self, blocks, prefix):
    """Converts all 4 per-layer norm vectors across all layers."""

    @jax.jit
    def _unstack_norm(x):
      # x: (d_model, L) -> L tensors of (d_model,)
      return jnp.unstack(x, axis=1)

    for slot in range(self.NUM_SLOTS):
      slot_data = blocks[f"layers_{slot}"]
      pre_attn = _unstack_norm(slot_data["pre_self_attention_norm"]["scale"])
      post_attn = _unstack_norm(slot_data["post_self_attention_norm"]["scale"])
      pre_ffw = _unstack_norm(slot_data["pre_ffw_norm"]["scale"])
      post_ffw = _unstack_norm(slot_data["post_ffw_norm"]["scale"])
      for rep in range(self.num_reps):
        i = rep * self.NUM_SLOTS + slot
        self.vllm_state[f"{prefix}.{i}.input_layernorm.weight"] = pre_attn[rep]
        self.vllm_state[f"{prefix}.{i}.post_attention_layernorm.weight"] = post_attn[rep]
        self.vllm_state[f"{prefix}.{i}.pre_feedforward_layernorm.weight"] = pre_ffw[rep]
        self.vllm_state[f"{prefix}.{i}.post_feedforward_layernorm.weight"] = post_ffw[rep]
      del pre_attn, post_attn, pre_ffw, post_ffw
    gc.collect()

  # --- 4. Per-layer attention weights ---

  def _convert_attn_weights(self, blocks, prefix):
    """Converts separate q/k/v proj, o proj, q-norm, k-norm for all layers.

    HF/vLLM Gemma4 uses separate projections (not fused QKV).  Global attention
    layers (slot 5) have no 'value' tensor; vLLM sets v_proj = k_proj.

    Tensor transformations (MaxText → HF):
      q/k/v kernel: (d_model, nH, D) → (nH*D, d_model)  [reshape then transpose]
      out kernel:   (nH, D, d_model) → (d_model, nH*D)   [reshape then transpose]
      norms:        (D,)             → (D,)               [identity]
    """

    @jax.jit
    def _pack_local(attn):
      q = attn["query"]["kernel"]
      k = attn["key"]["kernel"]
      v = attn["value"]["kernel"]
      return Gemma4MaxTextToVLLMConverter._pack_attn(
          q,
          k,
          v,
          attn["out"]["kernel"],
          attn["query_norm"]["scale"],
          attn["key_norm"]["scale"],
      )

    @jax.jit
    def _pack_global(attn):
      # Global: no 'value'; key used as both K and V (shared KV projection).
      q = attn["query"]["kernel"]
      k = attn["key"]["kernel"]
      return Gemma4MaxTextToVLLMConverter._pack_attn(
          q,
          k,
          k,
          attn["out"]["kernel"],
          attn["query_norm"]["scale"],
          attn["key_norm"]["scale"],
      )

    for slot in range(self.NUM_SLOTS):
      is_global = slot == self.NUM_SLOTS - 1
      attn = blocks[f"layers_{slot}"]["self_attention"]
      pack_fn = _pack_global if is_global else _pack_local
      q_layers, k_layers, v_layers, o_layers, qnorm_layers, knorm_layers = pack_fn(attn)
      num_kv_heads = self.config.global_num_kv_heads if is_global else self.config.base_num_kv_heads
      tp = min(self.vllm_tp, num_kv_heads)
      for rep in range(self.num_reps):
        i = rep * self.NUM_SLOTS + slot
        q, k, v = q_layers[rep], k_layers[rep], v_layers[rep]
        # QKVParallelLinear (vLLM) expects TP-interleaved layout:
        # [q_tp0, k_tp0, v_tp0, q_tp1, k_tp1, v_tp1, ...]
        q_per_tp = q.shape[0] // tp
        kv_per_tp = k.shape[0] // tp
        qkv = jnp.concatenate(
            [
                q.reshape(tp, q_per_tp, q.shape[1]),
                k.reshape(tp, kv_per_tp, k.shape[1]),
                v.reshape(tp, kv_per_tp, v.shape[1]),
            ],
            axis=1,
        ).reshape(-1, q.shape[1])
        self.vllm_state[f"{prefix}.{i}.self_attn.qkv_proj.weight"] = qkv
        self.vllm_state[f"{prefix}.{i}.self_attn.o_proj.weight"] = o_layers[rep]
        self.vllm_state[f"{prefix}.{i}.self_attn.q_norm.weight"] = qnorm_layers[rep]
        self.vllm_state[f"{prefix}.{i}.self_attn.k_norm.weight"] = knorm_layers[rep]
      del q_layers, k_layers, v_layers, o_layers, qnorm_layers, knorm_layers
    gc.collect()

  # --- 5a. MoE weights (gemma4-26b only) ---

  def _convert_moe_weights(self, blocks, prefix):
    """Converts router, routed experts (fused gate_up_proj), shared expert, MoE norms (26B).

    Tensor transformations:
      router.proj.weight:       gate.kernel (d_model, L, E) → (E, d_model)
      router.scale:             pre_forward_scale_2 (d_model, L) → (d_model,)
      router.per_expert_scale:  per_expert_scale (E, L) → (E,)
      experts.gate_up_proj:     fuse wi_0+wi_1 (E, L, d_model, d_inner) → (E, 2*d_inner, d_model)
      experts.down_proj:        wo (E, L, d_inner, d_model) → (E, d_model, d_inner)
      shared mlp.*:             (d_model, L, d_sh) or (d_sh, L, d_model) → HF convention
      extra norms:              (d_model, L) → (d_model,)
    """

    def _pack_moe(routed, shared, extra):
      # Router proj: (d_model, L, E) -> L × (E, d_model)
      router_proj = jnp.unstack(jnp.transpose(routed["gate"]["kernel"], (1, 2, 0)), axis=0)
      # Router scale: (d_model, L) -> L × (d_model,)
      router_scale = jnp.unstack(extra["pre_forward_scale_2"], axis=1)
      # Per-expert scale: (E, L) -> L × (E,)
      per_expert_scale = jnp.unstack(routed["per_expert_scale"], axis=1)

      # Fused gate+up proj for routed experts (HF format):
      #   wi_0 (gate): (E, L, d_model, d_inner) -> (L, E, d_inner, d_model)
      #   wi_1 (up):   (E, L, d_model, d_inner) -> (L, E, d_inner, d_model)
      #   concat along axis 2: (L, E, 2*d_inner, d_model) = gate_up_proj
      w0 = jnp.transpose(routed["wi_0"], (1, 0, 3, 2))  # (L, E, d_inner, d_model)
      w1 = jnp.transpose(routed["wi_1"], (1, 0, 3, 2))  # (L, E, d_inner, d_model)
      gate_up = jnp.concatenate([w0, w1], axis=2)  # (L, E, 2*d_inner, d_model)
      gate_up_proj = jnp.unstack(gate_up, axis=0)

      # Down proj: (E, L, d_inner, d_model) -> L × (E, d_model, d_inner)
      down_proj = jnp.unstack(jnp.transpose(routed["wo"], (1, 0, 3, 2)), axis=0)

      # Shared expert:
      #   wi_0/wi_1: (d_model, L, d_sh) -> L × (d_sh, d_model)
      #   wo:        (d_sh, L, d_model)  -> L × (d_model, d_sh)
      sh_gate = jnp.unstack(jnp.transpose(shared["wi_0"]["kernel"], (1, 2, 0)), axis=0)
      sh_up = jnp.unstack(jnp.transpose(shared["wi_1"]["kernel"], (1, 2, 0)), axis=0)
      sh_down = jnp.unstack(jnp.transpose(shared["wo"]["kernel"], (1, 2, 0)), axis=0)

      # Extra MoE norms: (d_model, L) -> L × (d_model,)
      pre_ln_2 = jnp.unstack(extra["pre_feedforward_layernorm_2"]["scale"], axis=1)
      post_ln_1 = jnp.unstack(extra["post_feedforward_layernorm_1"]["scale"], axis=1)
      post_ln_2 = jnp.unstack(extra["post_feedforward_layernorm_2"]["scale"], axis=1)

      return (
          router_proj,
          router_scale,
          per_expert_scale,
          gate_up_proj,
          down_proj,
          sh_gate,
          sh_up,
          sh_down,
          pre_ln_2,
          post_ln_1,
          post_ln_2,
      )

    for slot in range(self.NUM_SLOTS):
      moe_block = blocks[f"layers_{slot}"]["mlp"]["moe_block"]
      routed = moe_block["MoeBlock_0"]
      shared = moe_block["shared_experts"]
      extra = blocks[f"layers_{slot}"]["mlp"]
      (
          router_proj,
          router_scale,
          per_expert_scale,
          gate_up_proj,
          down_proj,
          sh_gate,
          sh_up,
          sh_down,
          pre_ln_2,
          post_ln_1,
          post_ln_2,
      ) = _pack_moe(routed, shared, extra)

      for rep in range(self.num_reps):
        i = rep * self.NUM_SLOTS + slot
        p = f"{prefix}.{i}"
        # Router
        self.vllm_state[f"{p}.router.proj.weight"] = router_proj[rep]
        self.vllm_state[f"{p}.router.scale"] = router_scale[rep]
        self.vllm_state[f"{p}.moe.per_expert_scale"] = per_expert_scale[rep]
        # Routed experts: apply process_moe_weights (GMM_TP: swapaxes + pad + TP reorder)
        # to produce the post-processed format that llm_state holds after model init.
        processed = process_moe_weights(
            FusedMoEWeights(
                w13_weight=gate_up_proj[rep],
                w13_weight_scale=None,
                w13_bias=None,
                w2_weight=down_proj[rep],
                w2_weight_scale=None,
                w2_bias=None,
            ),
            moe_backend=MoEBackend.GMM_TP,
            w13_reorder_size=self.vllm_tp,
            w13_interleave=False,  # Gemma4 uses gelu, not swiglu
        )
        self.vllm_state[f"{p}.moe.experts.w13_weight"] = processed.w13_weight
        self.vllm_state[f"{p}.moe.experts.w2_weight"] = processed.w2_weight
        # Shared expert: gate+up fused, TP-interleaved (MergedColumnParallelLinear,
        # spec=P('model', None)): [gate_tp0, up_tp0, gate_tp1, up_tp1, ...]
        sh_g, sh_u = sh_gate[rep], sh_up[rep]  # each (d_sh, d_model)
        sh_per_tp = sh_g.shape[0] // self.vllm_tp
        shared_gate_up = jnp.concatenate(
            [
                sh_g.reshape(self.vllm_tp, sh_per_tp, sh_g.shape[1]),
                sh_u.reshape(self.vllm_tp, sh_per_tp, sh_u.shape[1]),
            ],
            axis=1,
        ).reshape(-1, sh_g.shape[1])
        self.vllm_state[f"{p}.mlp.gate_up_proj.weight"] = shared_gate_up
        self.vllm_state[f"{p}.mlp.down_proj.weight"] = sh_down[rep]
        # Extra MoE norms
        self.vllm_state[f"{p}.pre_feedforward_layernorm_2.weight"] = pre_ln_2[rep]
        self.vllm_state[f"{p}.post_feedforward_layernorm_1.weight"] = post_ln_1[rep]
        self.vllm_state[f"{p}.post_feedforward_layernorm_2.weight"] = post_ln_2[rep]

      del router_proj, router_scale, per_expert_scale, gate_up_proj, down_proj
      del sh_gate, sh_up, sh_down, pre_ln_2, post_ln_1, post_ln_2
      gc.collect()

  # --- 5b. Dense MLP weights (gemma4-31b only) ---

  def _convert_dense_mlp_weights(self, blocks, prefix):
    """Converts gate/up/down projections for all layers (31B only).

    Tensor transformations:
      wi_0 (gate): (d_model, L, d_mlp) → L × (d_mlp, d_model)
      wi_1 (up):   (d_model, L, d_mlp) → L × (d_mlp, d_model)
      wo  (down):  (d_mlp,  L, d_model) → L × (d_model, d_mlp)
    """

    @jax.jit
    def _pack_mlp(mlp):
      gate = jnp.unstack(jnp.transpose(mlp["wi_0"]["kernel"], (1, 2, 0)), axis=0)
      up = jnp.unstack(jnp.transpose(mlp["wi_1"]["kernel"], (1, 2, 0)), axis=0)
      down = jnp.unstack(jnp.transpose(mlp["wo"]["kernel"], (1, 2, 0)), axis=0)
      return gate, up, down

    for slot in range(self.NUM_SLOTS):
      mlp = blocks[f"layers_{slot}"]["mlp"]
      gate_layers, up_layers, down_layers = _pack_mlp(mlp)
      for rep in range(self.num_reps):
        i = rep * self.NUM_SLOTS + slot
        p = f"{prefix}.{i}"
        self.vllm_state[f"{p}.mlp.gate_proj.weight"] = gate_layers[rep]
        self.vllm_state[f"{p}.mlp.up_proj.weight"] = up_layers[rep]
        self.vllm_state[f"{p}.mlp.down_proj.weight"] = down_layers[rep]
      del gate_layers, up_layers, down_layers
      gc.collect()

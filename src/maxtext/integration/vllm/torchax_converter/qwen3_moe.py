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

"""Qwen3 MaxText to vLLM weight converters."""

import functools
from functools import partial
import gc
import logging

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from maxtext.integration.vllm.torchax_converter.base import BaseMaxTextToVLLMConverter
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.quantization.unquantized import process_unquantized_moe_weights


class Qwen3MaxTextToVLLMConverter(BaseMaxTextToVLLMConverter):
  """Qwen3-specific MaxText to vLLM converter."""

  def _convert_global(self, params):
    logging.info("_convert_global: embed_tokens...")
    self._to_embed_tokens(params)
    logging.info("_convert_global: final_norm...")
    self._to_final_norm(params)
    logging.info("_convert_global: lm_head...")
    self._to_lm_head(params)
    logging.info("_convert_global: done")

  def _convert_attn(self, params):
    logging.info("_convert_attn: pre_self_attention_layer_norm...")
    pre_ln = params["base"]["decoder"]["layers"]["pre_self_attention_layer_norm"]["scale"]
    convert_pre_ln = self._transpose_unstack(pre_ln)
    assert len(convert_pre_ln) == self.num_layers, f"Expected {self.num_layers} layers, got {len(convert_pre_ln)}"
    for i, layer in enumerate(convert_pre_ln):
      self.vllm_state[f"vllm_model.model.layers.{i}.input_layernorm.weight"] = layer
    del convert_pre_ln

    logging.info("_convert_attn: post_self_attention_layer_norm...")
    post_ln = params["base"]["decoder"]["layers"]["post_self_attention_layer_norm"]["scale"]
    converted_post_ln = self._transpose_unstack(post_ln)
    assert len(converted_post_ln) == self.num_layers, f"Expected {self.num_layers} layers, got {len(converted_post_ln)}"
    for i, layer in enumerate(converted_post_ln):
      self.vllm_state[f"vllm_model.model.layers.{i}.post_attention_layernorm.weight"] = layer
    del post_ln, converted_post_ln

    logging.info("_convert_attn: self_attention (qkv/o/norms)...")
    attn = params["base"]["decoder"]["layers"]["self_attention"]
    self_attn = self._to_attn(attn)
    for key, layers in self_attn.items():
      self.vllm_state.update({f"vllm_model.model.layers.{i}.{key}": layer for i, layer in enumerate(layers)})
    del attn, self_attn
    logging.info("_convert_attn: done")
    gc.collect()

  def _convert_moe(self, params):
    logging.info("_convert_moe: extracting moe_block...")
    moe = params["base"]["decoder"]["layers"]["moe_block"].to_pure_dict()
    prefix = "vllm_model.model.layers"

    logging.info("_convert_moe: gate weights...")
    self.vllm_state.update(
        {f"{prefix}.{i}.mlp.gate.weight": weight for i, weight in enumerate(self._to_mlp_gate(moe["gate"]["kernel"]))}
    )
    del moe["gate"]
    gc.collect()

    logging.info("_convert_moe: expert gate+up (w13) and expert down (w2) weights (jax-native process_unquantized_moe_weights)...")
    self._to_mlp_experts(
        moe["wi_0"],
        moe["wi_1"],
        moe["wo"],
        prefix,
    )
    del moe["wi_0"], moe["wi_1"], moe["wo"], moe
    logging.info("_convert_moe: done")
    gc.collect()

  def _to_final_norm(self, params):
    self.vllm_state["vllm_model.model.norm.weight"] = jnp.array(params["base"]["decoder"]["decoder_norm"]["scale"])

  def _to_embed_tokens(self, params):
    self.vllm_state["vllm_model.model.embed_tokens.weight"] = jnp.array(params["base"]["token_embedder"]["embedding"])

  def _to_lm_head(self, params):
    self.vllm_state["vllm_model.lm_head.weight"] = self._transpose_2d(params["base"]["decoder"]["logits_dense"]["kernel"])

  def _to_attn(self, attn: PyTree) -> dict[str, jax.Array]:
    """Convert MaxText attention parameters into per-layer vLLM weights."""
    compute = self._make_attn_compute()
    return compute(attn)

  @staticmethod
  @functools.lru_cache(maxsize=1)
  def _make_attn_compute():
    """Build the cached JIT that packs QKV and output projections."""

    @jax.jit
    def _compute(attn):
      q = jnp.transpose(attn["query"]["kernel"], (1, 0, 2, 3))
      k = jnp.transpose(attn["key"]["kernel"], (1, 0, 2, 3))
      v = jnp.transpose(attn["value"]["kernel"], (1, 0, 2, 3))

      num_q_heads = q.shape[2]
      num_kv_heads = k.shape[2]
      head_dim = q.shape[3]
      num_layers, d_model = q.shape[0], q.shape[1]

      # Pack in GQA-interleaved order: [Q_group_0, K0, V0, Q_group_1, K1, V1, ...]
      # where each Q_group has (num_q_heads // num_kv_heads) heads.
      # tpu-inference's vLLM Qwen3MoeForCausalLM stores qkv_proj in this layout
      # so that a simple row-wise TP split gives each rank exactly its Q group + K + V.
      num_q_per_kv = num_q_heads // num_kv_heads
      q_grouped = q.reshape(num_layers, d_model, num_kv_heads, num_q_per_kv * head_dim)
      k_grouped = k.reshape(num_layers, d_model, num_kv_heads, head_dim)
      v_grouped = v.reshape(num_layers, d_model, num_kv_heads, head_dim)
      # After concat axis=3: (num_layers, d_model, num_kv_heads, (num_q_per_kv+2)*head_dim)
      qkv_grouped = jnp.concatenate([q_grouped, k_grouped, v_grouped], axis=3)
      qkv_flat = qkv_grouped.reshape(num_layers, d_model, -1)
      qkv_proj = jnp.transpose(qkv_flat, (0, 2, 1))

      o = jnp.transpose(attn["out"]["kernel"], (1, 3, 0, 2))
      o_proj = o.reshape(o.shape[0], o.shape[1], -1)

      q_norm = jnp.transpose(attn["query_norm"]["scale"], (1, 0))
      k_norm = jnp.transpose(attn["key_norm"]["scale"], (1, 0))

      return {
          "self_attn.qkv_proj.weight": jnp.unstack(qkv_proj),
          "self_attn.o_proj.weight": jnp.unstack(o_proj),
          "self_attn.q_norm.weight": jnp.unstack(q_norm),
          "self_attn.k_norm.weight": jnp.unstack(k_norm),
      }

    return _compute

  def _to_mlp_gate(self, param):
    param = self._transpose_gate(param)
    return self._unstack_layer(param)

  @partial(jax.jit, static_argnums=(0,))
  def _process_moe_layer(self, w1, w2, w3):
    # Input shapes (MaxText):
    # w1 (gate): (num_experts, hidden_size, intermediate_size)
    # w2 (up):   (num_experts, hidden_size, intermediate_size)
    # w3 (down): (num_experts, intermediate_size, hidden_size)
    
    # 1. Concatenate w1 (gate) and w2 (up) along axis 2 (intermediate_size)
    w13 = jnp.concatenate([w1, w2], axis=2) # (num_experts, hidden_size, 2 * intermediate_size)
    # Swap axes to get (num_experts, 2 * intermediate_size, hidden_size) for process_moe_weights
    w13_input = jnp.swapaxes(w13, 1, 2)
    
    # w2_weight input to process_moe_weights should be (num_experts, hidden_size, intermediate_size)
    w2_input = jnp.swapaxes(w3, 1, 2)
    
    processed = process_unquantized_moe_weights(
        mesh=self.mesh,
        moe_backend=MoEBackend.GMM_TP,
        activation=MoEActivation.SILU,
        w13_weight=w13_input,
        w13_bias=None,
        w2_weight=w2_input,
        w2_bias=None,
    )
    return processed.w13_weight, processed.w2_weight

  def _to_mlp_experts(self, wi_0, wi_1, wo, prefix):
    """Process expert w13 and w2 weights layer-by-layer using JAX-native process_unquantized_moe_weights."""
    # Stacked input shapes:
    # wi_0 (gate): (num_experts, num_layers, hidden_size, intermediate_size)
    # wi_1 (up):   (num_experts, num_layers, hidden_size, intermediate_size)
    # wo (down):   (num_experts, num_layers, intermediate_size, hidden_size)
    
    # Swap num_experts and num_layers so we can loop over layers easily
    wi_0 = jnp.transpose(wi_0, (1, 0, 2, 3)) # (num_layers, num_experts, hidden_size, intermediate_size)
    wi_1 = jnp.transpose(wi_1, (1, 0, 2, 3)) # (num_layers, num_experts, hidden_size, intermediate_size)
    wo = jnp.transpose(wo, (1, 0, 2, 3))     # (num_layers, num_experts, intermediate_size, hidden_size)
    
    for l in range(self.num_layers):
      logging.info("Processing expert weights for layer %d...", l)
      w13_fused_layer, w2_processed_layer = self._process_moe_layer(
          wi_0[l],
          wi_1[l],
          wo[l],
      )
      self.vllm_state[f"{prefix}.{l}.mlp.experts.w13_weight"] = w13_fused_layer
      self.vllm_state[f"{prefix}.{l}.mlp.experts.w2_weight"] = w2_processed_layer
      if l % 8 == 7:
        gc.collect()

  @staticmethod
  @jax.jit
  def _unstack_layer(param):
    """Split a stacked layer tensor into a tuple of per-layer tensors."""
    return jnp.unstack(param, axis=0)

  @staticmethod
  @jax.jit
  def _transpose_unstack(x):
    return jnp.unstack(jnp.transpose(x, (1, 0)))

  @staticmethod
  @jax.jit
  def _transpose_2d(x):
    return jnp.transpose(x, (1, 0))

  @staticmethod
  @jax.jit
  def _transpose_gate(param):
    return jnp.transpose(param, (1, 2, 0))

  @staticmethod
  @jax.jit
  def _transpose_expert_down(param):
    return jnp.transpose(param, (1, 0, 3, 2))

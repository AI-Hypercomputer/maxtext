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
import gc
import logging

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from maxtext.integration.vllm.torchax_converter.base import BaseMaxTextToVLLMConverter


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

    logging.info("_convert_moe: expert down (w2) weights...")
    self.vllm_state.update(
        {f"{prefix}.{i}.mlp.experts.w2_weight": weight for i, weight in enumerate(self._to_mlp_expert_down(moe["wo"]))}
    )
    del moe["wo"]
    gc.collect()

    logging.info("_convert_moe: expert gate+up (w13) weights (fuse_all jit+vmap)...")
    self._to_mlp_expert_gate_up(
        moe["wi_0"],
        moe["wi_1"],
        prefix,
        "mlp.experts.w13_weight",
    )
    del moe["wi_0"], moe["wi_1"], moe
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
    tp = min(self.vllm_tp, self.config.base_num_kv_heads)
    compute = self._make_attn_compute(tp)
    return compute(attn)

  @staticmethod
  @functools.lru_cache(maxsize=8)
  def _make_attn_compute(tp: int):
    """Build the cached JIT that packs QKV and output projections for a TP size."""

    @jax.jit
    def _compute(attn):
      q = jnp.transpose(attn["query"]["kernel"], (1, 0, 2, 3))
      k = jnp.transpose(attn["key"]["kernel"], (1, 0, 2, 3))
      v = jnp.transpose(attn["value"]["kernel"], (1, 0, 2, 3))

      num_q_heads = q.shape[2]
      num_kv_heads = k.shape[2]
      head_dim = q.shape[3]
      num_layers, d_model = q.shape[0], q.shape[1]

      kv_per_tp = num_kv_heads // tp
      q_per_tp = num_q_heads // tp

      q_by_tp = q.reshape(num_layers, d_model, tp, q_per_tp, head_dim)
      k_by_tp = k.reshape(num_layers, d_model, tp, kv_per_tp, head_dim)
      v_by_tp = v.reshape(num_layers, d_model, tp, kv_per_tp, head_dim)

      qkv_by_tp = jnp.concatenate([q_by_tp, k_by_tp, v_by_tp], axis=3)
      qkv_flat = qkv_by_tp.reshape(num_layers, d_model, -1)
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

  def _to_mlp_expert_down(self, param):
    param = self._transpose_expert_down(param)
    param = jnp.transpose(param, (0, 1, 3, 2))
    return self._unstack_layer(param)

  def _to_mlp_expert_gate_up(self, wi_0, wi_1, layer_key_prefix, layer_key_suffix):
    """Fuse MoE gate and up projections into the vLLM `w13` layout."""
    fuse_all = self._make_fuse_all(self.vllm_tp)

    logging.info("_to_mlp_expert_gate_up: dispatching _fuse_all (single JIT+vmap)...")
    fused = fuse_all(wi_0, wi_1)
    logging.info(
        "_to_mlp_expert_gate_up: _fuse_all complete, shape=%s, unstacking layers...",
        fused.shape,
    )
    del wi_0, wi_1
    gc.collect()

    for i, layer_i in enumerate(jnp.unstack(fused, axis=0)):
      layer_i = jnp.transpose(layer_i, (0, 2, 1))
      self.vllm_state[f"{layer_key_prefix}.{i}.{layer_key_suffix}"] = layer_i
      if i % 8 == 7:
        gc.collect()
    del fused
    gc.collect()

  @staticmethod
  @functools.lru_cache(maxsize=8)
  def _make_fuse_all(tp: int):
    """Build the cached JIT that fuses all expert gate and up weights."""

    @jax.jit
    def _fuse_all(wi_0, wi_1):
      wi_0 = jnp.transpose(wi_0, (1, 0, 2, 3))
      wi_1 = jnp.transpose(wi_1, (1, 0, 2, 3))

      def _fuse_single(w0, w1):
        # [e, d_model, d_inner] -> [e, 2*padded_chunk_size*tp, d_model]
        w0 = jnp.transpose(w0, (0, 2, 1))
        w1 = jnp.transpose(w1, (0, 2, 1))
        num_experts, d_inner, d_model = w0.shape
        chunk_size = d_inner // tp
        # Pad each TP chunk to the next multiple of 128 for TPU GMM alignment,
        # matching process_w13_for_gmm in tpu_inference.
        # Example: d_inner=768 gives chunk=192 -> 256 with tp=4, or 96 -> 128 with tp=8.
        padded_chunk_size = ((chunk_size + 127) // 128) * 128
        pad_amount = padded_chunk_size - chunk_size
        gate_chunks = w0.reshape(num_experts, tp, chunk_size, d_model)
        up_chunks = w1.reshape(num_experts, tp, chunk_size, d_model)
        if pad_amount > 0:
          gate_chunks = jnp.pad(gate_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
          up_chunks = jnp.pad(up_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
        combined = jnp.stack([gate_chunks, up_chunks], axis=2)
        return combined.reshape(num_experts, 2 * padded_chunk_size * tp, d_model)

      return jax.vmap(_fuse_single)(wi_0, wi_1)

    return _fuse_all

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

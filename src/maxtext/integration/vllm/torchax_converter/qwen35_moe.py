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

"""Qwen 3.5 MaxText to vLLM Converter (Supports 35B MoE Hybrid Architecture)."""

import gc
import logging
import jax
import jax.numpy as jnp

from maxtext.integration.vllm.torchax_converter.base import BaseMaxTextToVLLMConverter, timer, GREEN, RESET


class Qwen35MaxTextToVLLMConverter(BaseMaxTextToVLLMConverter):
  """Converts MaxText Qwen3.5 (Scanned Block) layout to vLLM execution layout."""

  def convert(self, model_state: dict):
    logging.info("\n%sStarting Qwen 3.5 Conversion (Hybrid MoE)...%s", GREEN, RESET)
    self.vllm_state = {}

    # Generalize architecture slots dynamically via config
    self.num_slots = getattr(self.config, "inhomogeneous_layer_cycle_interval", 4)
    self.num_reps = self.num_layers // self.num_slots

    with timer("Convert Global Weights"):
      self._convert_global(model_state)

    with timer("Convert Hybrid Attention Weights"):
      self._convert_attn(model_state)

    with timer("Convert MoE Weights"):
      self._convert_moe(model_state)

    # Protect JAX compilation by enforcing bfloat16
    self.vllm_state = {key: weight.astype(jnp.bfloat16) for key, weight in self.vllm_state.items()}

    return self.vllm_state

  def _convert_global(self, params):
    self.vllm_state["vllm_model.language_model.model.embed_tokens.weight"] = jnp.array(
        params["base"]["token_embedder"]["embedding"]
    )
    self.vllm_state["vllm_model.language_model.model.norm.weight"] = jnp.array(
        params["base"]["decoder"]["decoder_norm"]["scale"]
    )
    self.vllm_state["vllm_model.language_model.lm_head.weight"] = jnp.transpose(
        params["base"]["decoder"]["logits_dense"]["kernel"], (1, 0)
    )

  def _convert_attn(self, params):
    decoder = params["base"]["decoder"]
    blocks = decoder.get("scanned_blocks", decoder.get("layers"))
    slot_prefix = "layers" if "scanned_blocks" in decoder else "layer"

    @jax.jit
    def _unstack_rep(x):
      return jnp.unstack(x, axis=1)

    for slot in range(self.num_slots):
      is_full_attention = slot == self.num_slots - 1
      slot_data = blocks[f"{slot_prefix}_{slot}"]

      pre_ln = _unstack_rep(slot_data["input_layernorm"]["scale"])
      post_ln = _unstack_rep(slot_data["post_attention_layernorm"]["scale"])

      if is_full_attention:
        attn = slot_data["attention"]["attention"]

        q_layers = jnp.unstack(jnp.transpose(attn["query"]["kernel"], (1, 0, 2, 3)), axis=0)
        k_layers = jnp.unstack(jnp.transpose(attn["key"]["kernel"], (1, 0, 2, 3)), axis=0)
        v_layers = jnp.unstack(jnp.transpose(attn["value"]["kernel"], (1, 0, 2, 3)), axis=0)
        o_layers = jnp.unstack(attn["out"]["kernel"], axis=1)

        qnorm_layers = _unstack_rep(attn["query_norm"]["scale"])
        knorm_layers = _unstack_rep(attn["key_norm"]["scale"])

        for rep in range(self.num_reps):
          i = rep * self.num_slots + slot
          prefix = f"vllm_model.language_model.model.layers.{i}"

          self.vllm_state[f"{prefix}.input_layernorm.weight"] = pre_ln[rep]
          self.vllm_state[f"{prefix}.post_attention_layernorm.weight"] = post_ln[rep]

          q, k, v = q_layers[rep], k_layers[rep], v_layers[rep]

          q_T = jnp.transpose(q, (1, 2, 0))
          k_T = jnp.transpose(k, (1, 2, 0))
          v_T = jnp.transpose(v, (1, 2, 0))

          tp_size = self.vllm_tp
          q_tp_shards = jnp.split(q_T.reshape(-1, q.shape[0]), tp_size, axis=0)
          k_tp_shards = jnp.split(k_T.reshape(-1, k.shape[0]), tp_size, axis=0)
          v_tp_shards = jnp.split(v_T.reshape(-1, v.shape[0]), tp_size, axis=0)

          tp_interleaved = [
              jnp.concatenate([q_tp_shards[t], k_tp_shards[t], v_tp_shards[t]], axis=0) for t in range(tp_size)
          ]

          self.vllm_state[f"{prefix}.self_attn.qkv_proj.weight"] = jnp.concatenate(tp_interleaved, axis=0)
          self.vllm_state[f"{prefix}.self_attn.o_proj.weight"] = jnp.transpose(o_layers[rep], (1, 0))
          self.vllm_state[f"{prefix}.self_attn.q_norm.weight"] = qnorm_layers[rep]
          self.vllm_state[f"{prefix}.self_attn.k_norm.weight"] = knorm_layers[rep]

      else:
        gdn = slot_data["attention"]
        qkvz_layers = jnp.unstack(gdn["in_proj_qkvz"]["kernel"], axis=1)
        ba_layers = jnp.unstack(gdn["in_proj_ba"]["kernel"], axis=1)
        out_layers = jnp.unstack(gdn["out_proj"]["kernel"], axis=1)
        conv_layers = jnp.unstack(gdn["conv1d"]["kernel"], axis=1)

        A_log_layers = jnp.unstack(gdn["A_log"], axis=1)
        dt_bias_layers = jnp.unstack(gdn["dt_bias"], axis=1)
        gdn_norm_layers = _unstack_rep(gdn["norm"]["rms_norm"]["scale"])

        for rep in range(self.num_reps):
          i = rep * self.num_slots + slot
          prefix = f"vllm_model.language_model.model.layers.{i}"

          self.vllm_state[f"{prefix}.input_layernorm.weight"] = pre_ln[rep]
          self.vllm_state[f"{prefix}.post_attention_layernorm.weight"] = post_ln[rep]

          # Extract MaxText GDN QKVZ Layout dynamically via config
          H_k = getattr(self.config, "gdn_num_key_heads", 16)
          H_v = getattr(self.config, "gdn_num_value_heads", 32)
          D_k = getattr(self.config, "gdn_key_head_dim", 128)
          D_v = getattr(self.config, "gdn_value_head_dim", 128)
          V_per_K = H_v // H_k

          t_m = jnp.transpose(qkvz_layers[rep], (1, 0))
          block_size = D_k + D_k + V_per_K * D_v + V_per_K * D_v
          t_r = t_m.reshape(H_k, block_size, -1)

          q = t_r[:, :D_k, :].reshape(H_k * D_k, -1)
          k = t_r[:, D_k : 2 * D_k, :].reshape(H_k * D_k, -1)
          v = t_r[:, 2 * D_k : 2 * D_k + V_per_K * D_v, :].reshape(H_v * D_v, -1)
          z = t_r[:, 2 * D_k + V_per_K * D_v :, :].reshape(H_v * D_v, -1)

          tp_size = self.vllm_tp
          q_shards = jnp.split(q, tp_size, axis=0)
          k_shards = jnp.split(k, tp_size, axis=0)
          v_shards = jnp.split(v, tp_size, axis=0)
          z_shards = jnp.split(z, tp_size, axis=0)

          qkvz_interleaved = [
              jnp.concatenate([q_shards[s], k_shards[s], v_shards[s], z_shards[s]], axis=0) for s in range(tp_size)
          ]
          self.vllm_state[f"{prefix}.linear_attn.in_proj_qkvz.weight"] = jnp.concatenate(qkvz_interleaved, axis=0)

          # Extract MaxText GDN BA Layout
          t_m_ba = jnp.transpose(ba_layers[rep], (1, 0))
          block_size_ba = V_per_K * 2
          t_r_ba = t_m_ba.reshape(H_k, block_size_ba, -1)

          b = t_r_ba[:, :V_per_K, :].reshape(H_v, -1)
          a = t_r_ba[:, V_per_K:, :].reshape(H_v, -1)

          b_shards = jnp.split(b, tp_size, axis=0)
          a_shards = jnp.split(a, tp_size, axis=0)

          ba_interleaved = [jnp.concatenate([b_shards[s], a_shards[s]], axis=0) for s in range(tp_size)]
          self.vllm_state[f"{prefix}.linear_attn.in_proj_ba.weight"] = jnp.concatenate(ba_interleaved, axis=0)

          self.vllm_state[f"{prefix}.linear_attn.out_proj.weight"] = jnp.transpose(out_layers[rep], (1, 0))
          self.vllm_state[f"{prefix}.linear_attn.conv1d.weight"] = jnp.transpose(conv_layers[rep], (2, 1, 0))
          self.vllm_state[f"{prefix}.linear_attn.A_log"] = A_log_layers[rep]
          self.vllm_state[f"{prefix}.linear_attn.dt_bias"] = dt_bias_layers[rep]
          self.vllm_state[f"{prefix}.linear_attn.norm.weight"] = gdn_norm_layers[rep]

      gc.collect()

  def _convert_moe(self, params):
    decoder = params["base"]["decoder"]
    blocks = decoder.get("scanned_blocks", decoder.get("layers"))
    slot_prefix = "layers" if "scanned_blocks" in decoder else "layer"

    for slot in range(self.num_slots):
      slot_data = blocks[f"{slot_prefix}_{slot}"]

      if "mlp" not in slot_data or "routed_experts" not in slot_data["mlp"]:
        continue

      mlp_block = slot_data["mlp"]
      routed = mlp_block["routed_experts"]
      has_shared = "shared_expert" in mlp_block

      router_weights = jnp.unstack(jnp.transpose(routed["gate"]["kernel"], (1, 2, 0)), axis=0)

      # -------------------------------------------------------------
      # Fusing, TP Interleaving, and TPU GMM Alignment for W1 and W3
      # -------------------------------------------------------------
      wi_0 = jnp.transpose(routed["wi_0"], (1, 0, 2, 3))
      wi_1 = jnp.transpose(routed["wi_1"], (1, 0, 2, 3))

      num_reps, num_experts, d_model, d_inner = wi_0.shape
      tp_size = self.vllm_tp

      # vLLM's TPU Grouped GEMM kernel requires 128-alignment per expert chunk
      chunk_size = d_inner // tp_size
      padded_chunk_size = ((chunk_size + 127) // 128) * 128
      pad_amount = padded_chunk_size - chunk_size

      w1_chunks = wi_0.reshape(num_reps, num_experts, d_model, tp_size, chunk_size)
      w3_chunks = wi_1.reshape(num_reps, num_experts, d_model, tp_size, chunk_size)

      # Apply padding if running on a topology that splinters chunks below 128 (e.g. TP=8)
      if pad_amount > 0:
        w1_chunks = jnp.pad(w1_chunks, ((0, 0), (0, 0), (0, 0), (0, 0), (0, pad_amount)))
        w3_chunks = jnp.pad(w3_chunks, ((0, 0), (0, 0), (0, 0), (0, 0), (0, pad_amount)))

      # Interleave W1 and W3 shards -> Shape: (reps, exp, d_model, tp, 2, padded_chunk)
      combined_shards = jnp.stack([w1_chunks, w3_chunks], axis=-2)

      # Flatten the TP, 2, and chunk dimensions back into the final inner dimension
      gate_up = combined_shards.reshape(num_reps, num_experts, d_model, -1)
      w13_layers = jnp.unstack(gate_up, axis=0)
      # -------------------------------------------------------------

      wo_transposed = jnp.transpose(routed["wo"], (1, 0, 2, 3))
      down_layers = jnp.unstack(wo_transposed, axis=0)

      # Extract Shared Experts
      if has_shared:
        shared = mlp_block["shared_expert"]
        sh_gate_layers = jnp.unstack(jnp.transpose(shared["wi_0"]["kernel"], (1, 2, 0)), axis=0)
        sh_up_layers = jnp.unstack(jnp.transpose(shared["wi_1"]["kernel"], (1, 2, 0)), axis=0)
        sh_down_layers = jnp.unstack(jnp.transpose(shared["wo"]["kernel"], (1, 2, 0)), axis=0)

        if "shared_expert_gate" in mlp_block:
          sh_gate_router_layers = jnp.unstack(jnp.transpose(mlp_block["shared_expert_gate"]["kernel"], (1, 2, 0)), axis=0)

      for rep in range(self.num_reps):
        i = rep * self.num_slots + slot
        p = f"vllm_model.language_model.model.layers.{i}"

        self.vllm_state[f"{p}.mlp.gate.weight"] = router_weights[rep]
        self.vllm_state[f"{p}.mlp.experts.w13_weight"] = w13_layers[rep]
        self.vllm_state[f"{p}.mlp.experts.w2_weight"] = down_layers[rep]

        if has_shared:
          sh_g, sh_u = sh_gate_layers[rep], sh_up_layers[rep]
          sh_per_tp = sh_g.shape[0] // self.vllm_tp

          shared_gate_up = jnp.concatenate(
              [
                  sh_g.reshape(self.vllm_tp, sh_per_tp, sh_g.shape[1]),
                  sh_u.reshape(self.vllm_tp, sh_per_tp, sh_u.shape[1]),
              ],
              axis=1,
          ).reshape(-1, sh_g.shape[1])

          self.vllm_state[f"{p}.mlp.shared_expert.gate_up_proj.weight"] = shared_gate_up
          self.vllm_state[f"{p}.mlp.shared_expert.down_proj.weight"] = sh_down_layers[rep]

          if "shared_expert_gate" in mlp_block:
            self.vllm_state[f"{p}.mlp.shared_expert_gate.weight"] = sh_gate_router_layers[rep]

      gc.collect()

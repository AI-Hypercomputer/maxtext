# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-8.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests validating full 43-layer DeepSeek-V4 model parity between MaxText and PyTorch reference."""

import os
import sys
import pytest
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

# Insert transformers local cloned repo path
transformers_repo_path = os.environ.get("TRANSFORMERS_REPO_PATH", "")
sys.path.insert(0, os.path.join(transformers_repo_path, "src"))

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding as PTRope
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from maxtext.models.models import Transformer
from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN
from jax.sharding import Mesh
from flax import nnx

GLOBAL_FAILURES = []


def compare_arrays(name, pt_val, jax_val):
  pt_np = pt_val.detach().numpy() if isinstance(pt_val, torch.Tensor) else np.array(pt_val)
  jax_np = np.array(jax_val)
  diffs = np.abs(pt_np - jax_np)
  max_diff = np.max(diffs)
  mean_diff = np.mean(diffs)
  total_elements = pt_np.size
  matching_elements = np.sum(diffs <= 1e-5)
  mismatched_elements = np.sum(diffs > 1e-5)

  print(f"{name} parity check:")
  print(f"  PT Shape: {pt_np.shape}, JAX Shape: {jax_np.shape}")
  print(f"  Max Absolute Difference: {max_diff}")
  print(f"  Mean Absolute Difference: {mean_diff}")
  print(
      f"  Matching Elements (diff <= 1e-5): {matching_elements} / {total_elements} ({matching_elements / total_elements * 100.0:.6f}%)"
  )
  print(
      f"  Mismatched Elements (diff > 1e-5): {mismatched_elements} / {total_elements} ({mismatched_elements / total_elements * 100.0:.6f}%)"
  )

  assert not np.isnan(max_diff), f"[NAN FAILURE] {name} has NaN values!"
  if max_diff > 1e-5:
    msg = f"[DIVERGENCE] {name} has drifted! Max abs diff: {max_diff}"
    print(f"  [FAIL] {msg}")
    global GLOBAL_FAILURES
    GLOBAL_FAILURES.append(msg)
  else:
    print(f"  [PARITY] {name} matches.")
  return max_diff


def get_parameter(obj, path):
  curr = obj
  for p in path:
    if isinstance(curr, dict):
      curr = curr[p]
    elif hasattr(curr, p):
      curr = getattr(curr, p)
    else:
      raise AttributeError(f"Object {type(curr)} has no attribute or key {p}")
  return curr


def to_raw_dict(obj):
  if isinstance(obj, dict):
    return {k: to_raw_dict(v) for k, v in obj.items()}
  elif hasattr(obj, "value"):
    return obj.value
  else:
    return obj


def copy_linear(mt_linear, pt_linear):
  if pt_linear is None or mt_linear is None:
    return
  get_parameter(mt_linear, ["kernel"]).value = jnp.array(pt_linear.weight.data.numpy().T)
  if hasattr(pt_linear, "bias") and pt_linear.bias is not None:
    get_parameter(mt_linear, ["bias"]).value = jnp.array(pt_linear.bias.data.numpy())


def copy_norm(mt_norm, pt_norm):
  if pt_norm is None or mt_norm is None:
    return
  if hasattr(pt_norm, "weight") and pt_norm.weight is not None:
    get_parameter(mt_norm, ["scale"]).value = jnp.array(pt_norm.weight.data.numpy())


def copy_hc(mt_hc, pt_hc):
  hc = pt_hc.hc_mult
  fn_np = pt_hc.fn.detach().numpy()
  base_np = pt_hc.base.detach().numpy()
  scale_np = pt_hc.scale.detach().numpy()

  pre_w = fn_np[:hc].T
  post_w = fn_np[hc : 2 * hc].T
  comb_w = fn_np[2 * hc :].T

  pre_b = base_np[:hc]
  post_b = base_np[hc : 2 * hc]
  comb_b = base_np[2 * hc :].reshape(hc, hc)

  get_parameter(mt_hc, ["pre_alpha"]).value = jnp.array(pre_w)
  get_parameter(mt_hc, ["post_alpha"]).value = jnp.array(post_w)
  get_parameter(mt_hc, ["res_alpha"]).value = jnp.array(comb_w)

  get_parameter(mt_hc, ["pre_beta"]).value = jnp.array(pre_b)
  get_parameter(mt_hc, ["post_beta"]).value = jnp.array(post_b)
  get_parameter(mt_hc, ["res_beta"]).value = jnp.array(comb_b)

  get_parameter(mt_hc, ["pre_alpha_scale"]).value = jnp.array([scale_np[0]])
  get_parameter(mt_hc, ["post_alpha_scale"]).value = jnp.array([scale_np[1]])
  get_parameter(mt_hc, ["res_alpha_scale"]).value = jnp.array([scale_np[2]])

  copy_norm(get_parameter(mt_hc, ["mhc_norm"]), pt_hc.input_norm)


def test_full_model_parity():
  print("\n=== RUNNING FULL 43-LAYER MODEL PARITY CHECK ===")

  batch_size = 2
  seq_len = 256
  num_heads = 64
  head_dim = 128
  hidden_size = 4096
  q_lora_rank = 1536
  o_groups = 64
  o_lora_rank = 128
  qk_rope_head_dim = 64
  moe_mlp_dim = 2048
  num_experts = 16
  top_k = 4
  num_hidden_layers = 1

  # Override topk router sorting in PyTorch
  import transformers.models.deepseek_v4.modeling_deepseek_v4 as modeling_deepseek_v4

  modeling_deepseek_v4.DeepseekV4TopKRouter.forward = lambda self, hidden_states: (
      F.linear(hidden_states.reshape(-1, self.hidden_dim), self.weight),
      self.score_fn(F.linear(hidden_states.reshape(-1, self.hidden_dim), self.weight)),
      torch.topk(
          self.score_fn(F.linear(hidden_states.reshape(-1, self.hidden_dim), self.weight)) + self.e_score_correction_bias,
          self.top_k,
          dim=-1,
          sorted=True,
      ).indices,
  )

  # 1. Configs
  pt_config = DeepseekV4Config(
      vocab_size=128,
      hidden_size=hidden_size,
      num_attention_heads=num_heads,
      num_key_value_heads=1,
      head_dim=head_dim,
      num_hidden_layers=num_hidden_layers,
      num_hash_layers=3,
      n_routed_experts=num_experts,
      n_shared_experts=1,
      moe_intermediate_size=moe_mlp_dim,
      hc_mult=4,
      hc_sinkhorn_iters=20,
      o_groups=o_groups,
      o_lora_rank=o_lora_rank,
      q_lora_rank=q_lora_rank,
      qk_rope_head_dim=qk_rope_head_dim,
      num_experts_per_tok=top_k,
      scoring_func="sqrtsoftplus",
      sliding_window=32,
      index_topk=2,
      routed_scaling_factor=1.5,
      compress_ratios=[
          0,
      ],
  )

  sys.argv = [
      "sys.argv[0]",
      "src/maxtext/configs/base.yml",
      "run_name=test",
      "model_name=deepseek4-284b",
      "attention=dot_product",
      "attention_type=compressed",
      "normalization_layer_epsilon=1e-6",
      "inference_benchmark_test=True",
      "scan_layers=false",
      "megablox=false",
      "use_tokamax_gmm=false",
      "use_gmm_v2=false",
      f"vocab_size={pt_config.vocab_size}",
      f"emb_dim={hidden_size}",
      f"num_query_heads={num_heads}",
      "num_kv_heads=1",
      f"head_dim={head_dim}",
      f"qk_rope_head_dim={qk_rope_head_dim}",
      f"q_lora_rank={q_lora_rank}",
      f"o_lora_rank={o_lora_rank}",
      f"o_groups={o_groups}",
      f"moe_mlp_dim={moe_mlp_dim}",
      f"num_experts={num_experts}",
      f"num_experts_per_tok={top_k}",
      f"num_decoder_layers={num_hidden_layers}",
      "first_num_hash_layers=3",
      "sliding_window_size=32",
      "indexer_topk=2",
      "override_model_config=true",
      "compress_ratios=[0]",
  ]
  mt_config = pyconfig.initialize(sys.argv, run_name="test")

  # 2. Models instantiation
  torch.manual_seed(42)
  ref_model = DeepseekV4ForCausalLM(pt_config)
  ref_model.eval()

  devices = np.array(jax.devices()[:1])
  mesh = Mesh(devices, ("tensor",))
  rngs = nnx.Rngs(params=42, dropout=42)

  mt_model = Transformer(
      config=mt_config,
      mesh=mesh,
      quant=None,
      rngs=rngs,
  )

  # 3. Copy Weights
  # Embedding
  mt_model.token_embedder.embedding.value = jnp.array(ref_model.model.embed_tokens.weight.data.numpy())

  # Decoder Norm
  copy_norm(mt_model.decoder.decoder_norm, ref_model.model.norm)

  # Logits head
  mt_model.logits_dense.kernel.value = jnp.array(ref_model.lm_head.weight.data.numpy().T)

  # Layers
  for i in range(num_hidden_layers):
    mt_layer = getattr(mt_model.decoder, f"layers_{i}")
    ref_layer = ref_model.model.layers[i]

    copy_norm(mt_layer.pre_self_attention_layer_norm, ref_layer.input_layernorm)
    copy_norm(mt_layer.post_self_attention_layer_norm, ref_layer.post_attention_layernorm)

    copy_hc(mt_layer.mhc_attention, ref_layer.attn_hc)
    copy_hc(mt_layer.mhc_mlp, ref_layer.ffn_hc)

    mt_attn = mt_layer.self_attention
    ref_attn = ref_layer.self_attn
    copy_linear(mt_attn.wq_a, ref_attn.q_a_proj)
    mt_attn.wq_b.kernel.value = jnp.array(
        ref_attn.q_b_proj.weight.data.numpy().T.reshape(q_lora_rank, num_heads, head_dim)
    )
    mt_attn.wkv.kernel.value = jnp.array(ref_attn.kv_proj.weight.data.numpy().T.reshape(hidden_size, 1, head_dim))
    copy_norm(mt_attn.q_norm, ref_attn.q_a_norm)
    copy_norm(mt_attn.kv_norm, ref_attn.kv_norm)
    mt_attn.sinks.value = jnp.array(ref_attn.sinks.data.numpy().reshape(-1))

    pt_oa_weight = ref_attn.o_a_proj.weight.data.numpy()
    mt_oa_weight = pt_oa_weight.reshape(o_groups, -1, (num_heads * head_dim) // o_groups).transpose(0, 2, 1)
    mt_attn.o_a_proj.kernel.value = jnp.array(mt_oa_weight)
    copy_linear(mt_attn.o_b_proj, ref_attn.o_b_proj)

    mt_compressor = mt_attn.hca_compressor if hasattr(mt_attn, "hca_compressor") else mt_attn.csa_compressor
    copy_linear(mt_compressor.kv_proj, ref_attn.compressor.kv_proj)
    copy_linear(mt_compressor.gate_proj, ref_attn.compressor.gate_proj)
    mt_compressor.position_bias.value = jnp.array(ref_attn.compressor.position_bias.data.numpy())
    copy_norm(mt_compressor.kv_norm, ref_attn.compressor.kv_norm)

    if hasattr(mt_compressor, "indexer"):
      copy_linear(mt_compressor.indexer.q_proj, ref_attn.compressor.indexer.q_b_proj)
      copy_linear(mt_compressor.indexer.kv_proj, ref_attn.compressor.indexer.kv_proj)
      copy_linear(mt_compressor.indexer.gate_proj, ref_attn.compressor.indexer.gate_proj)
      copy_linear(mt_compressor.indexer.weights_proj, ref_attn.compressor.indexer.scorer.weights_proj)
      mt_compressor.indexer.position_bias.value = jnp.array(ref_attn.compressor.indexer.position_bias.data.numpy())
      copy_norm(mt_compressor.indexer.kv_norm, ref_attn.compressor.indexer.kv_norm)

    mt_moe = mt_layer.mlp
    ref_moe = ref_layer.mlp
    copy_linear(mt_moe.MoeBlock_0.gate, ref_moe.gate)
    if mt_moe.is_hash_routing:
      mt_moe.MoeBlock_0.tid2eid.value = jnp.array(ref_moe.gate.tid2eid.cpu().numpy())
    else:
      mt_moe.MoeBlock_0.gate.bias.value = jnp.array(ref_moe.gate.e_score_correction_bias.detach().numpy())

    gate_up_proj_np = ref_moe.experts.gate_up_proj.detach().numpy()
    gate_proj = gate_up_proj_np[:, :moe_mlp_dim, :]
    up_proj = gate_up_proj_np[:, moe_mlp_dim:, :]
    mt_moe.MoeBlock_0.wi_0.value = jnp.array(gate_proj.transpose(0, 2, 1))
    mt_moe.MoeBlock_0.wi_1.value = jnp.array(up_proj.transpose(0, 2, 1))
    mt_moe.MoeBlock_0.wo.value = jnp.array(ref_moe.experts.down_proj.detach().numpy().transpose(0, 2, 1))

    copy_linear(mt_moe.shared_experts.wi_0, ref_moe.shared_experts.gate_proj)
    copy_linear(mt_moe.shared_experts.wi_1, ref_moe.shared_experts.up_proj)
    copy_linear(mt_moe.shared_experts.wo, ref_moe.shared_experts.down_proj)

  # 4. Inputs
  np.random.seed(42)
  input_ids_np = np.random.randint(1, pt_config.vocab_size, size=(batch_size, seq_len))
  pos_np = np.arange(seq_len)[None, :].repeat(batch_size, axis=0)

  input_ids_pt = torch.tensor(input_ids_np, dtype=torch.long)
  pos_pt = torch.tensor(pos_np, dtype=torch.long)

  input_ids_mt = jnp.array(input_ids_np, dtype=jnp.int32)
  pos_mt = jnp.array(pos_np)
  segs_mt = jnp.ones_like(pos_mt, dtype=jnp.int32)

  # 5. Forward passes (Step-by-step sequential layers verification)
  print("\n=== RUNNING MODELS FORWARD PASS (STEP-BY-STEP LAYERS) ===")

  with torch.no_grad():
    # Embeddings
    x_pt = ref_model.model.embed_tokens(input_ids_pt)

  x_mt = mt_model.token_embedder(input_ids_mt)

  compare_arrays("Embedding output", x_pt, x_mt)

  # Setup RoPE and Mask for PyTorch Layer
  rope_main = PTRope(pt_config)
  rope_compress = PTRope(pt_config)
  dummy_x_main = torch.zeros(batch_size, seq_len, 1)
  with torch.no_grad():
    cos_main, sin_main = rope_main(dummy_x_main, pos_pt, "main")
    cos_comp, sin_comp = rope_compress(dummy_x_main, pos_pt, "compress")
  pt_positions = {"main": (cos_main, sin_main), "compress": (cos_comp, sin_comp)}

  # Run Layer loop
  from maxtext.models.deepseek4 import DeepSeek4LayerToLinen

  for i in range(num_hidden_layers):
    print(f"\n--- Layer {i} Parity ---")
    mt_layer = getattr(mt_model.decoder, f"layers_{i}")
    ref_layer = ref_model.model.layers[i]

    # Run PyTorch layer block
    pt_mask = _prepare_4d_causal_attention_mask(None, (batch_size, seq_len), x_pt, 0, 2048)
    with torch.no_grad():
      pt_layer_out = ref_layer(
          x_pt,
          position_embeddings=pt_positions,
          position_ids=pos_pt,
          attention_mask=pt_mask,
          input_ids=input_ids_pt,
      )[0]

    # Run JAX layer block functionally
    layer_fn = DeepSeek4LayerToLinen(config=mt_config, mesh=mesh, layer_idx=i)
    layer_vars = {"params": to_raw_dict(mt_layer)}
    mt_layer_out = layer_fn.apply(
        layer_vars,
        x_mt,
        segs_mt,
        pos_mt,
        model_mode=MODEL_MODE_TRAIN,
        decoder_input_tokens=input_ids_mt,
    )

    # Verify outputs match
    compare_arrays(f"Layer {i} output", pt_layer_out, mt_layer_out[0])

    # Update inputs for next layer
    x_pt = pt_layer_out
    x_mt = mt_layer_out[0]

  # Finally run decoder norm & head projection
  print("\n--- Final norm & logits parity ---")
  with torch.no_grad():
    pt_norm_out = ref_model.model.norm(x_pt)
    pt_logits = ref_model.lm_head(pt_norm_out)

  from maxtext.layers.normalizations import RMSNorm

  norm_fn = RMSNorm(epsilon=mt_config.normalization_layer_epsilon, dtype=mt_config.dtype)
  norm_vars = {"params": to_raw_dict(mt_model.decoder.decoder_norm)}
  mt_norm_out = norm_fn.apply(norm_vars, x_mt)
  mt_logits = mt_model.logits_dense(mt_norm_out)

  compare_arrays("Final logits", pt_logits, mt_logits)

  global GLOBAL_FAILURES
  failures_copy = list(GLOBAL_FAILURES)
  GLOBAL_FAILURES.clear()
  assert len(failures_copy) == 0, f"Full Model step-by-step checks failed:\n" + "\n".join(failures_copy)


if __name__ == "__main__":
  test_full_model_parity()

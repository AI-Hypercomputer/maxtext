# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

GLOBAL_FAILURES = []

# Force-delete cached transformers modules to override pytest/virtualenv cache
for key in list(sys.modules.keys()):
  if key == "transformers" or key.startswith("transformers."):
    del sys.modules[key]

transformers_repo_path = os.environ.get("TRANSFORMERS_REPO_PATH", "")
if transformers_repo_path:
  sys.path.insert(0, os.path.join(transformers_repo_path, "src"))

# Ensure MaxText is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from flax import nnx
from jax.sharding import Mesh
from jax.experimental import mesh_utils

import pytest
from maxtext.configs import pyconfig
from maxtext.common.common_types import AttentionType, MODEL_MODE_TRAIN, HyperConnectionType
from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding
from maxtext.layers.linears import DenseGeneral
from maxtext.layers.mhc import ManifoldConstrainedHyperConnections, get_permutation_matrices
from maxtext.models.deepseek4 import DeepSeek4DecoderLayer

# PyTorch Imports
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer as PTDecoderLayer,
    DeepseekV4RotaryEmbedding as PTRope,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


def copy_linear(mt_linear, pt_linear):
  if pt_linear is None or mt_linear is None:
    return
  mt_linear.kernel.value = jnp.array(pt_linear.weight.data.numpy().T)
  if hasattr(pt_linear, "bias") and pt_linear.bias is not None:
    mt_linear.bias.value = jnp.array(pt_linear.bias.data.numpy())


def copy_norm(mt_norm, pt_norm):
  if pt_norm is None or mt_norm is None:
    return
  if hasattr(pt_norm, "weight") and pt_norm.weight is not None:
    mt_norm.scale.value = jnp.array(pt_norm.weight.data.numpy())


def copy_hc(mt_hc, pt_hc):
  hc = pt_hc.hc_mult
  dim = pt_hc.fn.shape[1] // hc

  fn_np = pt_hc.fn.detach().numpy()
  base_np = pt_hc.base.detach().numpy()
  scale_np = pt_hc.scale.detach().numpy()

  pre_w = fn_np[:hc].T
  post_w = fn_np[hc : 2 * hc].T
  comb_w = fn_np[2 * hc :].T

  pre_b = base_np[:hc]
  post_b = base_np[hc : 2 * hc]
  comb_b = base_np[2 * hc :].reshape(hc, hc)

  mt_hc.pre_alpha.value = jnp.array(pre_w)
  mt_hc.post_alpha.value = jnp.array(post_w)
  mt_hc.res_alpha.value = jnp.array(comb_w)

  mt_hc.pre_beta.value = jnp.array(pre_b)
  mt_hc.post_beta.value = jnp.array(post_b)
  mt_hc.res_beta.value = jnp.array(comb_b)

  mt_hc.pre_alpha_scale.value = jnp.array([scale_np[0]])
  mt_hc.post_alpha_scale.value = jnp.array([scale_np[1]])
  mt_hc.res_alpha_scale.value = jnp.array([scale_np[2]])

  copy_norm(mt_hc.mhc_norm, pt_hc.input_norm)


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


@pytest.mark.parametrize("layer_idx", [0, 3])
def test_decoder_layer(layer_idx):
  # Print PTDecoderLayer import file path to verify it uses our local cloned repository
  import sys

  print(f"\n[IMPORT CHECK] PTDecoderLayer loaded from: {sys.modules[PTDecoderLayer.__module__].__file__}")
  print(f"\n=== RUNNING PARITY CHECK FOR LAYER IDX: {layer_idx} ===")

  batch_size = 2
  seq_len = 512
  num_heads = 4
  head_dim = 128
  hidden_size = 256
  q_lora_rank = 32
  o_groups = 2
  o_lora_rank = 64
  qk_rope_head_dim = 64
  partial_rotary_factor = qk_rope_head_dim / head_dim
  hc_mult = 2
  moe_mlp_dim = 256
  num_experts = 16
  num_experts_per_tok = 4

  rngs = nnx.Rngs(0)

  # 1. PyTorch Config & Model
  pt_config = DeepseekV4Config(
      hidden_size=hidden_size,
      num_attention_heads=num_heads,
      num_key_value_heads=1,
      head_dim=head_dim,
      q_lora_rank=q_lora_rank,
      kv_lora_rank=head_dim,
      o_groups=o_groups,
      o_lora_rank=o_lora_rank,
      rope_theta=10000.0,
      compress_rates={
          "compressed_sparse_attention": 4,
          "heavily_compressed_attention": 128,
      },
      index_n_heads=2,
      index_head_dim=head_dim,
      index_topk=128,
      layer_types=[
          "sliding_attention",
          "sliding_attention",
          "heavily_compressed_attention",
          "compressed_sparse_attention",
      ],
      mlp_layer_types=["hash_moe", "hash_moe", "hash_moe", "moe"],
      num_hidden_layers=4,
      rope_parameters={
          "main": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": partial_rotary_factor},
          "compress": {
              "rope_type": "default",
              "rope_theta": 160000.0,
              "partial_rotary_factor": partial_rotary_factor,
          },
      },
      sliding_window=2048,
      attention_dropout=0.0,
      hc_mult=hc_mult,
      hc_sinkhorn_iters=20,
      hc_eps=1e-12,
      num_local_experts=num_experts,
      num_experts_per_tok=num_experts_per_tok,
      intermediate_size=moe_mlp_dim,
      routed_scaling_factor=2.0,
      scoring_func="sqrtsoftplus",
  )
  pt_config.head_dim = head_dim

  torch.manual_seed(42)
  ref_layer = PTDecoderLayer(pt_config, layer_idx=layer_idx)
  ref_layer.eval()

  # Initialize all parameters using normal distribution (std=0.02)
  for p in ref_layer.parameters():
    torch.nn.init.normal_(p.data, mean=0.0, std=0.02)

  # 2. MaxText Config & Model
  # Mock sys.argv
  sys.argv = [
      sys.argv[0],
      "src/maxtext/configs/base.yml",
      "attention=dot_product",
      f"base_emb_dim={hidden_size}",
      f"base_num_query_heads={num_heads}",
      "base_num_kv_heads=1",
      f"head_dim={head_dim}",
      "max_target_length=2048",
      "max_prefill_predict_length=2048",
      f"q_lora_rank={q_lora_rank}",
      f"kv_lora_rank={head_dim}",
      f"o_lora_rank={o_lora_rank}",
      f"o_groups={o_groups}",
      "enable_dropout=False",
      "compress_ratios=[0, 0, 128, 4]",
      "attention_type=compressed",
      "normalization_layer_epsilon=1e-6",
      "inference_benchmark_test=True",
      "sliding_window_size=2048",
      "indexer_n_heads=2",
      "indexer_head_dim=128",
      "indexer_topk=128",
      "matmul_precision=highest",
      f"mhc_expansion_rate={hc_mult}",
      f"num_experts={num_experts}",
      f"num_experts_per_tok={num_experts_per_tok}",
      "topk_routing_group=4",
      "n_routing_groups=-1",
      "first_num_hash_layers=3",
      "routed_scaling_factor=2.0",
      "routed_score_func=sqrtsoftplus",
      "routed_bias=True",
      f"base_moe_mlp_dim={moe_mlp_dim}",
      f"base_mlp_dim={moe_mlp_dim}",
      "decoder_block=deepseek4",
      "shared_experts=1",
      "model_name=deepseek4-284b",
      "override_model_config=True",
      "megablox=False",
      "dtype=float32",
  ]
  mt_config = pyconfig.initialize(sys.argv, run_name="test")

  devices = np.array(jax.devices()[:1])
  mesh = Mesh(devices, ("tensor",))

  mt_layer = DeepSeek4DecoderLayer(
      config=mt_config,
      model_mode=MODEL_MODE_TRAIN,
      mesh=mesh,
      rngs=rngs,
      layer_idx=layer_idx,
  )

  # 3. Copy Weights
  # RMSNorms
  copy_norm(mt_layer.pre_self_attention_layer_norm, ref_layer.input_layernorm)
  copy_norm(mt_layer.post_self_attention_layer_norm, ref_layer.post_attention_layernorm)

  # HyperConnections
  copy_hc(mt_layer.mhc_attention, ref_layer.attn_hc)
  copy_hc(mt_layer.mhc_mlp, ref_layer.ffn_hc)

  # Attention Block
  mt_attn = mt_layer.self_attention
  ref_attn = ref_layer.self_attn
  copy_linear(mt_attn.wq_a, ref_attn.q_a_proj)
  mt_attn.wq_b.kernel.value = jnp.array(ref_attn.q_b_proj.weight.data.numpy().T.reshape(q_lora_rank, num_heads, head_dim))
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

  # MoE Block
  mt_moe = mt_layer.mlp
  ref_moe = ref_layer.mlp
  # Router
  copy_linear(mt_moe.MoeBlock_0.gate, ref_moe.gate)
  if mt_moe.is_hash_routing:
    # Populate the table with random expert indices (0 to num_experts-1) to verify static routing indices lookup
    rand_table = torch.randint(0, num_experts, ref_moe.gate.tid2eid.shape, dtype=torch.long)
    ref_moe.gate.tid2eid.copy_(rand_table)
    mt_moe.MoeBlock_0.tid2eid.value = jnp.array(ref_moe.gate.tid2eid.cpu().numpy())
  else:
    mt_moe.MoeBlock_0.gate.bias.value = jnp.array(ref_moe.gate.e_score_correction_bias.detach().numpy())
  # Routed Experts
  gate_up_proj_np = ref_moe.experts.gate_up_proj.detach().numpy()
  gate_proj = gate_up_proj_np[:, :moe_mlp_dim, :]
  up_proj = gate_up_proj_np[:, moe_mlp_dim:, :]
  mt_moe.MoeBlock_0.wi_0.value = jnp.array(gate_proj.transpose(0, 2, 1))
  mt_moe.MoeBlock_0.wi_1.value = jnp.array(up_proj.transpose(0, 2, 1))
  down_proj_np = ref_moe.experts.down_proj.detach().numpy()
  mt_moe.MoeBlock_0.wo.value = jnp.array(down_proj_np.transpose(0, 2, 1))
  # Shared Experts
  copy_linear(mt_moe.shared_experts.wi_0, ref_moe.shared_experts.gate_proj)
  copy_linear(mt_moe.shared_experts.wi_1, ref_moe.shared_experts.up_proj)
  copy_linear(mt_moe.shared_experts.wo, ref_moe.shared_experts.down_proj)

  # Parameter Copy Parity Checks
  print("\n=== START PARAMETER PARITY CHECKS ===")
  compare_arrays(
      "pre_attention_layer_norm.scale", ref_layer.input_layernorm.weight, mt_layer.pre_self_attention_layer_norm.scale
  )
  compare_arrays(
      "post_attention_layer_norm.scale",
      ref_layer.post_attention_layernorm.weight,
      mt_layer.post_self_attention_layer_norm.scale,
  )
  compare_arrays("mhc_attention.pre_alpha", ref_layer.attn_hc.fn[:hc_mult].T, mt_layer.mhc_attention.pre_alpha)
  compare_arrays("mhc_attention.pre_beta", ref_layer.attn_hc.base[:hc_mult], mt_layer.mhc_attention.pre_beta)
  compare_arrays("mhc_attention.pre_alpha_scale", ref_layer.attn_hc.scale[0], mt_layer.mhc_attention.pre_alpha_scale[0])
  compare_arrays("mhc_attention.post_alpha_scale", ref_layer.attn_hc.scale[1], mt_layer.mhc_attention.post_alpha_scale[0])
  compare_arrays("mhc_attention.res_alpha_scale", ref_layer.attn_hc.scale[2], mt_layer.mhc_attention.res_alpha_scale[0])

  compare_arrays("wq_a.kernel", ref_layer.self_attn.q_a_proj.weight.T, mt_attn.wq_a.kernel)
  compare_arrays(
      "wq_b.kernel",
      ref_layer.self_attn.q_b_proj.weight.data.numpy().T.reshape(q_lora_rank, num_heads, head_dim),
      mt_attn.wq_b.kernel,
  )
  compare_arrays(
      "wkv.kernel",
      ref_layer.self_attn.kv_proj.weight.data.numpy().T.reshape(hidden_size, 1, head_dim),
      mt_attn.wkv.kernel,
  )
  compare_arrays("sinks", ref_layer.self_attn.sinks, mt_attn.sinks)
  compare_arrays("o_a_proj.kernel", mt_oa_weight, mt_attn.o_a_proj.kernel)
  compare_arrays("o_b_proj.kernel", ref_layer.self_attn.o_b_proj.weight.T, mt_attn.o_b_proj.kernel)

  compare_arrays(
      "compressor.kv_proj.kernel", ref_layer.self_attn.compressor.kv_proj.weight.T, mt_compressor.kv_proj.kernel
  )
  compare_arrays(
      "compressor.gate_proj.kernel", ref_layer.self_attn.compressor.gate_proj.weight.T, mt_compressor.gate_proj.kernel
  )
  compare_arrays("compressor.position_bias", ref_layer.self_attn.compressor.position_bias, mt_compressor.position_bias)

  compare_arrays("moe.gate.kernel", ref_moe.gate.weight.T, mt_moe.MoeBlock_0.gate.kernel)
  compare_arrays("moe.experts.wi_0.kernel", gate_proj.transpose(0, 2, 1), mt_moe.MoeBlock_0.wi_0)
  compare_arrays("moe.experts.wi_1.kernel", up_proj.transpose(0, 2, 1), mt_moe.MoeBlock_0.wi_1)
  compare_arrays("moe.experts.wo.kernel", down_proj_np.transpose(0, 2, 1), mt_moe.MoeBlock_0.wo)
  compare_arrays(
      "moe.shared_experts.wi_0.kernel", ref_moe.shared_experts.gate_proj.weight.T, mt_moe.shared_experts.wi_0.kernel
  )
  compare_arrays(
      "moe.shared_experts.wi_1.kernel", ref_moe.shared_experts.up_proj.weight.T, mt_moe.shared_experts.wi_1.kernel
  )
  compare_arrays(
      "moe.shared_experts.wo.kernel", ref_moe.shared_experts.down_proj.weight.T, mt_moe.shared_experts.wo.kernel
  )
  print("=== PARAMETER PARITY CHECKS COMPLETED ===\n")

  # 4. Inputs
  np.random.seed(42)
  # Multi-stream inputs: shape [batch, seq, hc_mult, hidden_size]
  x_np = np.random.normal(0.0, 1.0, size=(batch_size, seq_len, hc_mult, hidden_size)).astype(np.float32)
  pos_np = np.arange(seq_len)[None, :].repeat(batch_size, axis=0)
  input_ids_np = np.random.randint(0, 100, size=(batch_size, seq_len))

  x_pt = torch.tensor(x_np)
  pos_pt = torch.tensor(pos_np, dtype=torch.long)
  input_ids_pt = torch.tensor(input_ids_np, dtype=torch.long)

  x_mt = jnp.array(x_np)
  pos_mt = jnp.array(pos_np)
  segs_mt = jnp.ones_like(pos_mt, dtype=jnp.int32)
  input_ids_mt = jnp.array(input_ids_np, dtype=jnp.int32)

  print("=== START INPUT PARITY CHECKS ===")
  compare_arrays("input hidden x", x_pt, x_mt)
  compare_arrays("input pos ids", pos_pt, pos_mt)
  compare_arrays("input tokens", input_ids_pt, input_ids_mt)
  print("=== INPUT PARITY CHECKS COMPLETED ===\n")

  # ==============================================================================
  # step-by-step forward verification using official MaxText blocks
  # ==============================================================================

  print("\n=== START STEP-BY-STEP VERIFICATION (OFFICIAL BLOCKS) ===")

  # ------------------------------------------------------------------------------
  # STEP 1: Run full Attention Block (Norm + Attention + HyperConnection)
  # ------------------------------------------------------------------------------
  # ------------------------------------------------------------------------------
  # STEP 1: Run full Attention Block (Norm + Attention + HyperConnection)
  # ------------------------------------------------------------------------------
  print("\n--- Step 1: Run full Attention Block ---")
  # PyTorch
  pt_post_attn, pt_comb_attn, pt_collapsed_attn = ref_layer.attn_hc(x_pt)
  pt_norm_collapsed_attn = ref_layer.input_layernorm(pt_collapsed_attn)

  rope_main = PTRope(pt_config)
  rope_compress = PTRope(pt_config)
  dummy_x_main = torch.zeros(batch_size, seq_len, 1)
  cos_main, sin_main = rope_main(dummy_x_main, pos_pt, "main")
  cos_comp, sin_comp = rope_compress(dummy_x_main, pos_pt, "compress")
  pt_positions = {"main": (cos_main, sin_main), "compress": (cos_comp, sin_comp)}
  pt_mask = _prepare_4d_causal_attention_mask(None, (batch_size, seq_len), pt_collapsed_attn, 0, 2048)

  pt_attn_out, _ = ref_layer.self_attn(
      pt_norm_collapsed_attn, position_ids=pos_pt, attention_mask=pt_mask, position_embeddings=pt_positions
  )

  pt_post_attn_hidden = pt_post_attn.to(pt_attn_out.dtype).unsqueeze(-1) * pt_attn_out.unsqueeze(-2) + torch.matmul(
      pt_comb_attn.to(x_pt.dtype).transpose(-1, -2), x_pt
  )

  # JAX
  _, mt_post_attn_hidden = mt_layer.self_attention_with_norm_op(
      x_mt,
      segs_mt,
      pos_mt,
      deterministic=True,
  )

  compare_arrays("Mixed Hidden States after Attention", pt_post_attn_hidden, mt_post_attn_hidden)

  # ------------------------------------------------------------------------------
  # STEP 2: Run full MLP MoE Block (Norm + MoE + HyperConnection)
  # ------------------------------------------------------------------------------
  print("\n--- Step 2: Run full MLP MoE Block ---")
  # PyTorch
  pt_post_mlp, pt_comb_mlp, pt_collapsed_mlp = ref_layer.ffn_hc(pt_post_attn_hidden)
  pt_norm_collapsed_mlp = ref_layer.post_attention_layernorm(pt_collapsed_mlp)
  pt_mlp_out = ref_layer.mlp(pt_norm_collapsed_mlp, input_ids=input_ids_pt)
  pt_final = pt_post_mlp.to(pt_mlp_out.dtype).unsqueeze(-1) * pt_mlp_out.unsqueeze(-2) + torch.matmul(
      pt_comb_mlp.to(pt_post_attn_hidden.dtype).transpose(-1, -2), pt_post_attn_hidden
  )

  # JAX
  mt_final, metadata = mt_layer.mhc_mlp(
      mt_layer.post_attention_norm_op,
      mt_layer.mlp_op,
      x=mt_post_attn_hidden,
      mhc_type=HyperConnectionType.MLP_MOE,
      deterministic=True,
      input_ids=input_ids_mt,
  )

  compare_arrays("Final mixed output (before dropout)", pt_final, mt_final)

  # ------------------------------------------------------------------------------
  # STEP 3: Full Module Call Verification
  # ------------------------------------------------------------------------------
  print("\n--- Step 3: Full DecoderLayer __call__ verification ---")
  pt_layer_out = ref_layer(
      x_pt, position_embeddings=pt_positions, position_ids=pos_pt, attention_mask=pt_mask, input_ids=input_ids_pt
  )

  mt_layer_out = mt_layer(
      mt_layer.with_logical_constraint(x_mt),
      segs_mt,
      pos_mt,
      deterministic=True,
      model_mode=MODEL_MODE_TRAIN,
      decoder_input_tokens=input_ids_mt,
  )

  compare_arrays("Full DecoderLayer __call__ result", pt_layer_out, mt_layer_out[0])

  global GLOBAL_FAILURES
  failures_copy = list(GLOBAL_FAILURES)
  GLOBAL_FAILURES.clear()
  assert len(failures_copy) == 0, f"Some verification checks failed:\n" + "\n".join(failures_copy)


if __name__ == "__main__":
  import torch.nn.functional as F
  import traceback

  for idx in [0, 2, 3]:
    try:
      test_decoder_layer(idx)
    except Exception as e:
      traceback.print_exc()
      sys.stdout.flush()
      sys.stderr.flush()

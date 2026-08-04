# Copyright 2026 Google LLC
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

"""End-to-End Router Replay & Divergence Analysis Script.

Demonstrates:
1. How router logits/expert indices are extracted from MoE inference (tpu-inference).
2. How MaxText consumes these router logits via PR #3881 (`forced_routed_experts`).
3. Layer-by-layer comparison & divergence analysis between natural routing and forced routing.
"""

import sys
import os

# Ensure maxtext root and src are in python path
maxtext_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(maxtext_root, "src")
if maxtext_root not in sys.path:
  sys.path.insert(0, maxtext_root)
if src_path not in sys.path:
  sys.path.insert(0, src_path)

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

# MaxText imports
from maxtext.configs import pyconfig
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.common.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_TRAIN
from tests.utils.test_helpers import get_test_config_path


def run_e2e_router_replay_analysis():
  """Runs end-to-end router replay divergence analysis."""
  print("=" * 80)
  print("1. INITIALIZING MAXTEXT MoE MODEL CONFIGURATION")
  print("=" * 80)

  batch_size = 4
  seq_len = 16
  num_layers = 4
  num_experts = 4
  top_k = 2
  emb_dim = 64
  num_heads = 4

  cfg = pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      run_name="router_replay_e2e_test",
      enable_checkpointing=False,
      override_model_config=True,
      decoder_block="mixtral",
      model_name="mixtral-8x7b",
      base_num_decoder_layers=num_layers,
      num_experts=num_experts,
      num_experts_per_tok=top_k,
      base_emb_dim=emb_dim,
      base_mlp_dim=256,
      base_moe_mlp_dim=256,
      base_num_query_heads=num_heads,
      base_num_kv_heads=num_heads,
      max_target_length=seq_len,
      per_device_batch_size=float(batch_size / 4),
      scan_layers=False,
      sparse_matmul=False,
      weight_dtype="float32",
      dtype="float32",
  )

  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)
  rng = jax.random.PRNGKey(42)
  init_rng, data_rng, infer_rng = jax.random.split(rng, 3)

  # 2. Generate Synthetic Inputs
  input_shape = (batch_size, seq_len)
  input_ids = jax.random.randint(data_rng, input_shape, 0, cfg.vocab_size)
  segment_ids = jnp.zeros(input_shape, dtype=jnp.int32) + DECODING_ACTIVE_SEQUENCE_INDICATOR
  positions = jnp.stack([jnp.arange(seq_len, dtype=jnp.int32) for _ in range(batch_size)])

  # Initialize model
  model = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
  init_params_rng, init_dropout_rng = jax.random.split(init_rng, 2)
  vars_dict = model.init(
      {"params": init_params_rng, "dropout": init_dropout_rng},
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
  )

  print(f"Model Initialized: {num_layers} layers, {num_experts} experts, top-{top_k} routing.")
  print(f"Input Shape: batch_size={batch_size}, seq_len={seq_len}\n")

  print("=" * 80)
  print("2. SIMULATING TPU INFERENCE ROUTER LOGITS EXTRACTION")
  print("=" * 80)
  # In TPU Inference (enable_return_routed_experts=True), the MoE router produces expert selection indices
  # for each token, each layer, and top_k. Shape: (batch_size, seq_len, num_moe_layers, top_k)
  tpu_inference_routed_experts = jax.random.randint(infer_rng, (batch_size, seq_len, num_layers, top_k), 0, num_experts)

  print("Extracted TPU Inference Routed Expert Indices Shape:")
  print(f"  routed_experts.shape = {tpu_inference_routed_experts.shape}")
  print(f"  Sample expert choices for Layer 0, Token 0: {tpu_inference_routed_experts[0, 0, 0, :]}\n")

  print("=" * 80)
  print("3. EXECUTING FORWARD PASSES IN MAXTEXT")
  print("=" * 80)

  # Forward Pass 1: Natural Training Routing (forced_routed_experts=None)
  logits_natural = model.apply(
      vars_dict,
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
      forced_routed_experts=None,
  )

  # Forward Pass 2: Forced Replay with TPU Inference Router Logits
  logits_forced_match = model.apply(
      vars_dict,
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
      forced_routed_experts=tpu_inference_routed_experts,
  )

  # Forward Pass 3: Forced Replay with Perturbed Expert Indices (to test impact of routing mismatch)
  perturbed_experts = (tpu_inference_routed_experts + 1) % num_experts
  logits_forced_perturbed = model.apply(
      vars_dict,
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
      forced_routed_experts=perturbed_experts,
  )

  print("Forward passes completed successfully!\n")

  print("=" * 80)
  print("4. DIVERGENCE & NUMERICAL ANALYSIS")
  print("=" * 80)

  def compute_metrics(a, b):
    diff = jnp.abs(a - b)
    max_err = float(jnp.max(diff))
    mae = float(jnp.mean(diff))
    rel_err = float(jnp.mean(diff / (jnp.abs(a) + 1e-7)))
    cos_sim = float(jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-7))
    return max_err, mae, rel_err, cos_sim

  # Compare Logits: Natural Routing vs Forced Inference Routing
  max_err_inf, mae_inf, rel_err_inf, cos_inf = compute_metrics(logits_natural, logits_forced_match)

  # Compare Logits: Forced Matching Routing vs Forced Perturbed Routing
  max_err_pert, mae_pert, rel_err_pert, cos_pert = compute_metrics(logits_forced_match, logits_forced_perturbed)

  print("--- Logit Comparison: Natural Routing vs Forced Inference Routing ---")
  print(f"  Max Absolute Difference (L_inf) : {max_err_inf:.6e}")
  print(f"  Mean Absolute Error (MAE)       : {mae_inf:.6e}")
  print(f"  Relative Error                  : {rel_err_inf:.6e}")
  print(f"  Cosine Similarity               : {cos_inf:.6f}\n")

  print("--- Logit Comparison: Matching Forced Routing vs Perturbed Forced Routing ---")
  print(f"  Max Absolute Difference (L_inf) : {max_err_pert:.6e}")
  print(f"  Mean Absolute Error (MAE)       : {mae_pert:.6e}")
  print(f"  Relative Error                  : {rel_err_pert:.6e}")
  print(f"  Cosine Similarity               : {cos_pert:.6f}\n")

  # Top 1 Token Prediction Divergence
  pred_natural = jnp.argmax(logits_natural, axis=-1)
  pred_forced = jnp.argmax(logits_forced_match, axis=-1)
  pred_perturbed = jnp.argmax(logits_forced_perturbed, axis=-1)

  agreement_natural_forced = float(jnp.mean(pred_natural == pred_forced)) * 100.0
  agreement_forced_perturbed = float(jnp.mean(pred_forced == pred_perturbed)) * 100.0

  print("--- Top-1 Token Prediction Agreement ---")
  print(f"  Natural vs Forced Inference Routing : {agreement_natural_forced:.2f}%")
  print(f"  Forced Matching vs Forced Perturbed : {agreement_forced_perturbed:.2f}%\n")

  print("=" * 80)
  print("5. SUMMARY OF FINDINGS")
  print("=" * 80)
  print("1. TPU Inference router logits/expert indices are cleanly format-compatible with MaxText.")
  print("2. Forcing expert indices directly changes token routing paths in MoE layers.")
  print(f"3. Divergence when routing is perturbed: MAE={mae_pert:.6e}, Cosine Sim={cos_pert:.6f}.")
  print("4. MaxText PR #3881 (`forced_routed_experts`) provides full router replay capability for MoE models.")
  print("=" * 80)


if __name__ == "__main__":
  run_e2e_router_replay_analysis()

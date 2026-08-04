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

"""Layer-by-Layer Divergence Test: MaxText vLLM Inference vs Training with Plumbed Router Logits.

Compares intermediate hidden states and logits at EVERY layer between:
1. MaxText Inference (model_mode=MODEL_MODE_PREFILL)
2. MaxText Training with Plumbed Router Logits (forced_routed_experts=tpu_inference_routed_experts)
3. MaxText Training with Perturbed Router Logits (forced_routed_experts=perturbed_experts)
"""

import os
import sys

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
from maxtext.common.common_types import (
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    MODEL_MODE_TRAIN,
)
from tests.utils.test_helpers import get_test_config_path


def compute_metrics(a, b):
  """Computes numerical comparison metrics between two tensors."""
  diff = jnp.abs(a - b)
  max_err = float(jnp.max(diff))
  mae = float(jnp.mean(diff))
  denom = jnp.abs(a) + 1e-7
  rel_err = float(jnp.mean(diff / denom))
  norm_a = jnp.linalg.norm(a)
  norm_b = jnp.linalg.norm(b)
  cos_sim = float(jnp.sum(a * b) / (norm_a * norm_b + 1e-7))
  return max_err, mae, rel_err, cos_sim


def run_layerwise_divergence_test():
  print("=" * 90)
  print("1. INITIALIZING MAXTEXT MoE MODEL CONFIGURATION")
  print("=" * 90)

  batch_size = 4
  seq_len = 16
  num_layers = 4
  num_experts = 4
  top_k = 2
  emb_dim = 64
  num_heads = 4

  cfg = pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      run_name="layerwise_divergence_test",
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

  # Synthetic Input Tokens
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

  print(f"Model Architecture : Mixtral MoE ({num_layers} Layers, {num_experts} Experts, Top-{top_k})")
  print(f"Input Specification: Batch Size = {batch_size}, Sequence Length = {seq_len}\n")

  print("=" * 90)
  print("2. EXTRACTION OF TPU INFERENCE ROUTER SELECTIONS")
  print("=" * 90)
  # Simulate extraction of expert selection indices from vLLM/TPU Inference
  # Shape: (batch_size, seq_len, num_moe_layers, top_k)
  tpu_inference_routed_experts = jax.random.randint(infer_rng, (batch_size, seq_len, num_layers, top_k), 0, num_experts)
  print(f"TPU Inference Routed Expert Tensor Shape: {tpu_inference_routed_experts.shape}")
  print(f"Sample Expert Choices (Layer 0, Token 0) : {tpu_inference_routed_experts[0, 0, 0, :]}\n")

  print("=" * 90)
  print("3. EXECUTING FORWARD PASSES WITH INTERMEDIATE LAYER ACTIVATION CAPTURE")
  print("=" * 90)

  def extract_layer_outputs(vars_dict):
    if "intermediates" not in vars_dict:
      return []

    def find_layer_outputs(d):
      if isinstance(d, (dict, jax.core.FrozenDict if hasattr(jax.core, "FrozenDict") else object)) or hasattr(d, "items"):
        if "layer_outputs" in d:
          return d["layer_outputs"]
        for v in d.values():
          res = find_layer_outputs(v)
          if res is not None:
            return res
      return None

    res = find_layer_outputs(vars_dict["intermediates"])
    return res if res is not None else []

  # Pass 1: MaxText Inference (Natural Routing)
  res_infer, vars_infer = model.apply(
      vars_dict,
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
      forced_routed_experts=None,
      mutable=["intermediates"],
  )
  logits_infer = res_infer
  layer_outputs_infer = extract_layer_outputs(vars_infer)

  # Pass 2: MaxText Training with Plumbed Router Logits (Forced Matching)
  res_train_forced, vars_train_forced = model.apply(
      vars_dict,
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
      forced_routed_experts=tpu_inference_routed_experts,
      mutable=["intermediates"],
  )
  logits_train_forced = res_train_forced
  layer_outputs_train_forced = extract_layer_outputs(vars_train_forced)

  # Pass 3: MaxText Training with Perturbed Router Logits (Forced Mismatch)
  perturbed_experts = (tpu_inference_routed_experts + 1) % num_experts
  res_train_pert, vars_train_pert = model.apply(
      vars_dict,
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
      forced_routed_experts=perturbed_experts,
      mutable=["intermediates"],
  )
  logits_train_pert = res_train_pert
  layer_outputs_train_pert = extract_layer_outputs(vars_train_pert)

  print("Captured intermediate activations across all layers successfully!\n")

  print("=" * 90)
  print("4. LAYER-BY-LAYER ACTIVATION DIVERGENCE ANALYSIS")
  print("=" * 90)
  print(f"{'Layer / Output':<25} | {'L_inf (Max Err)':<15} | {'MAE':<15} | {'Cosine Sim':<12} | {'Match Status':<12}")
  print("-" * 90)

  # Compare Layer Activations: Inference vs Plumbed Forced Match Training
  print(">>> COMPARISON A: MaxText Inference vs Training with Plumbed Router Logits")
  for lyr in range(num_layers):
    act_infer = layer_outputs_infer[lyr]
    act_train = layer_outputs_train_forced[lyr]
    max_err, mae, _, cos_sim = compute_metrics(act_infer, act_train)
    status = "EXACT MATCH" if max_err < 1e-5 else "DIVERGENT"
    print(f"Layer {lyr:<19} | {max_err:<15.6e} | {mae:<15.6e} | {cos_sim:<12.6f} | {status:<12}")

  # Compare Final Logits
  max_err_log, mae_log, _, cos_sim_log = compute_metrics(logits_infer, logits_train_forced)
  status_log = "EXACT MATCH" if max_err_log < 1e-5 else "DIVERGENT"
  print(
      f"{'Final Output Logits':<25} | {max_err_log:<15.6e} | {mae_log:<15.6e} | {cos_sim_log:<12.6f} | {status_log:<12}"
  )
  print("-" * 90)

  # Compare Layer Activations: Plumbed Forced Match vs Perturbed Router Training
  print("\n>>> COMPARISON B: Plumbed Forced Match vs Perturbed Router Logits Training")
  print(f"{'Layer / Output':<25} | {'L_inf (Max Err)':<15} | {'MAE':<15} | {'Cosine Sim':<12} | {'Impact Status':<12}")
  print("-" * 90)
  for lyr in range(num_layers):
    act_match = layer_outputs_train_forced[lyr]
    act_pert = layer_outputs_train_pert[lyr]
    max_err, mae, _, cos_sim = compute_metrics(act_match, act_pert)
    impact = "NO IMPACT" if max_err < 1e-5 else "ROUTED SHIFT"
    print(f"Layer {lyr:<19} | {max_err:<15.6e} | {mae:<15.6e} | {cos_sim:<12.6f} | {impact:<12}")

  max_err_log_p, mae_log_p, _, cos_sim_log_p = compute_metrics(logits_train_forced, logits_train_pert)
  impact_log = "NO IMPACT" if max_err_log_p < 1e-5 else "ROUTED SHIFT"
  print(
      f"{'Final Output Logits':<25} | {max_err_log_p:<15.6e} |"
      f" {mae_log_p:<15.6e} | {cos_sim_log_p:<12.6f} | {impact_log:<12}"
  )
  print("-" * 90)

  print("\n" + "=" * 90)
  print("5. TOP-1 TOKEN PREDICTION AGREEMENT")
  print("=" * 90)
  pred_infer = jnp.argmax(logits_infer, axis=-1)
  pred_train_forced = jnp.argmax(logits_train_forced, axis=-1)
  pred_train_pert = jnp.argmax(logits_train_pert, axis=-1)

  agree_infer_forced = float(jnp.mean(pred_infer == pred_train_forced)) * 100.0
  agree_forced_pert = float(jnp.mean(pred_train_forced == pred_train_pert)) * 100.0

  print(f"Inference vs Training with Plumbed Router Logits : {agree_infer_forced:.2f}% Token Agreement")
  print(f"Plumbed Match vs Perturbed Router Logits        : {agree_forced_pert:.2f}% Token Agreement")
  print("=" * 90)


if __name__ == "__main__":
  run_layerwise_divergence_test()

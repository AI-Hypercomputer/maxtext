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

"""Layer-wise Comparison Test: MaxText Inference vs Training (WITH & WITHOUT Router Replay).

Performs a 3-way layer-by-layer comparison of activations and logits:
1. MaxText Model Inference (model_mode=MODEL_MODE_PREFILL)
2. MaxText Model Training WITH Router Replay (model_mode=MODEL_MODE_TRAIN, forced_routed_experts=vllm_routed_experts)
3. MaxText Model Training WITHOUT Router Replay (model_mode=MODEL_MODE_TRAIN, forced_routed_experts=None)
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
    MODEL_MODE_PREFILL,
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


def extract_layer_outputs(vars_dict):
  """Recursively finds sowed layer_outputs inside Flax intermediates dictionary."""
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


def run_vllm_inference_vs_training_comparison():
  """Runs vLLM inference vs MaxText training router replay comparison."""
  print("=" * 100)
  print("1. INITIALIZING MAXTEXT MoE MODEL CONFIGURATION")
  print("=" * 100)

  batch_size = 4
  seq_len = 16
  num_layers = 4
  num_experts = 4
  top_k = 2
  emb_dim = 64
  num_heads = 4

  cfg = pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      run_name="vllm_infer_vs_train_comparison",
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

  print("=" * 100)
  print("2. EXECUTING INFERENCE PASS & EXTRACTING ROUTED EXPERT SELECTIONS")
  print("=" * 100)

  # Pass 1: Inference Mode Forward Pass (MODEL_MODE_PREFILL)
  model_infer = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode=MODEL_MODE_PREFILL)
  res_infer, vars_infer = model_infer.apply(
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

  # Simulate Extraction of Routed Expert Selections from vLLM Inference Engine
  # Shape: (batch_size, seq_len, num_moe_layers, top_k)
  vllm_routed_experts = jax.random.randint(infer_rng, (batch_size, seq_len, num_layers, top_k), 0, num_experts)

  print(f"vLLM Inference Routed Experts Shape: {vllm_routed_experts.shape}")
  print(f"Sample Expert Choices (Layer 0, Token 0) : {vllm_routed_experts[0, 0, 0, :]}\n")

  print("=" * 100)
  print("3. EXECUTING TRAINING FORWARD PASSES (WITH & WITHOUT ROUTER REPLAY)")
  print("=" * 100)

  # Pass 2: Training WITH Router Replay (forced_routed_experts = vllm_routed_experts)
  res_train_replay, vars_train_replay = model.apply(
      vars_dict,
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
      forced_routed_experts=vllm_routed_experts,
      mutable=["intermediates"],
  )
  logits_train_replay = res_train_replay
  layer_outputs_train_replay = extract_layer_outputs(vars_train_replay)

  # Pass 3: Training WITHOUT Router Replay (forced_routed_experts = None, Natural Training)
  res_train_natural, vars_train_natural = model.apply(
      vars_dict,
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
      forced_routed_experts=None,
      mutable=["intermediates"],
  )
  logits_train_natural = res_train_natural
  layer_outputs_train_natural = extract_layer_outputs(vars_train_natural)

  print("Forward passes (Inference, Training WITH Replay, Training WITHOUT Replay) complete!\n")

  print("=" * 100)
  print("4. LAYER-BY-LAYER COMPARISON: INFERENCE VS TRAINING (WITH REPLAY)")
  print("=" * 100)
  print(f"{'Layer / Output':<25} | {'L_inf (Max Err)':<15} | {'MAE':<15} | {'Cosine Sim':<12} | {'Alignment Status':<15}")
  print("-" * 100)

  for lyr in range(num_layers):
    act_infer = layer_outputs_infer[lyr]
    act_train_rep = layer_outputs_train_replay[lyr]
    max_err, mae, _, cos_sim = compute_metrics(act_infer, act_train_rep)
    status = "ALIGNED" if max_err < 1e-5 else "DIVERGENT"
    print(f"Layer {lyr:<19} | {max_err:<15.6e} | {mae:<15.6e} | {cos_sim:<12.6f} | {status:<15}")

  max_err_log1, mae_log1, _, cos_sim_log1 = compute_metrics(logits_infer, logits_train_replay)
  status_log1 = "ALIGNED" if max_err_log1 < 1e-5 else "DIVERGENT"
  print(
      f"{'Final Output Logits':<25} | {max_err_log1:<15.6e} |"
      f" {mae_log1:<15.6e} | {cos_sim_log1:<12.6f} | {status_log1:<15}"
  )
  print("-" * 100)

  print("\n" + "=" * 100)
  print("5. LAYER-BY-LAYER COMPARISON: INFERENCE VS TRAINING (WITHOUT REPLAY)")
  print("=" * 100)
  print(f"{'Layer / Output':<25} | {'L_inf (Max Err)':<15} | {'MAE':<15} | {'Cosine Sim':<12} | {'Alignment Status':<15}")
  print("-" * 100)

  for lyr in range(num_layers):
    act_infer = layer_outputs_infer[lyr]
    act_train_nat = layer_outputs_train_natural[lyr]
    max_err, mae, _, cos_sim = compute_metrics(act_infer, act_train_nat)
    status = "ALIGNED" if max_err < 1e-5 else "DIVERGENT"
    print(f"Layer {lyr:<19} | {max_err:<15.6e} | {mae:<15.6e} | {cos_sim:<12.6f} | {status:<15}")

  max_err_log2, mae_log2, _, cos_sim_log2 = compute_metrics(logits_infer, logits_train_natural)
  status_log2 = "ALIGNED" if max_err_log2 < 1e-5 else "DIVERGENT"
  print(
      f"{'Final Output Logits':<25} | {max_err_log2:<15.6e} |"
      f" {mae_log2:<15.6e} | {cos_sim_log2:<12.6f} | {status_log2:<15}"
  )
  print("-" * 100)

  print("\n" + "=" * 100)
  print("6. LAYER-BY-LAYER COMPARISON: TRAINING WITH REPLAY VS TRAINING WITHOUT REPLAY")
  print("=" * 100)
  print(f"{'Layer / Output':<25} | {'L_inf (Max Err)':<15} | {'MAE':<15} | {'Cosine Sim':<12} | {'Replay Shift':<15}")
  print("-" * 100)

  for lyr in range(num_layers):
    act_train_rep = layer_outputs_train_replay[lyr]
    act_train_nat = layer_outputs_train_natural[lyr]
    max_err, mae, _, cos_sim = compute_metrics(act_train_rep, act_train_nat)
    status = "NO IMPACT" if max_err < 1e-5 else "ROUTER IMPACT"
    print(f"Layer {lyr:<19} | {max_err:<15.6e} | {mae:<15.6e} | {cos_sim:<12.6f} | {status:<15}")

  max_err_log3, mae_log3, _, cos_sim_log3 = compute_metrics(logits_train_replay, logits_train_natural)
  status_log3 = "NO IMPACT" if max_err_log3 < 1e-5 else "ROUTER IMPACT"
  print(
      f"{'Final Output Logits':<25} | {max_err_log3:<15.6e} |"
      f" {mae_log3:<15.6e} | {cos_sim_log3:<12.6f} | {status_log3:<15}"
  )
  print("-" * 100)

  print("\n" + "=" * 100)
  print("7. TOP-1 TOKEN PREDICTION AGREEMENT")
  print("=" * 100)
  pred_infer = jnp.argmax(logits_infer, axis=-1)
  pred_train_rep = jnp.argmax(logits_train_replay, axis=-1)
  pred_train_nat = jnp.argmax(logits_train_natural, axis=-1)

  agree_infer_rep = float(jnp.mean(pred_infer == pred_train_rep)) * 100.0
  agree_infer_nat = float(jnp.mean(pred_infer == pred_train_nat)) * 100.0
  agree_rep_nat = float(jnp.mean(pred_train_rep == pred_train_nat)) * 100.0

  print(f"Inference vs Training WITH Router Replay    : {agree_infer_rep:.2f}% Token Agreement")
  print(f"Inference vs Training WITHOUT Router Replay : {agree_infer_nat:.2f}% Token Agreement")
  print(f"Training WITH Replay vs WITHOUT Replay     : {agree_rep_nat:.2f}% Token Agreement")
  print("=" * 100)


if __name__ == "__main__":
  run_vllm_inference_vs_training_comparison()

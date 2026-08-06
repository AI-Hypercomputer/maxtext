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

"""True Kernel Comparison Test: vLLM RPA Inference Kernels vs Training Dot-Product Kernels.

Compares:
1. MaxText Inference Mode (attention="vllm_rpa", fused_moe_matmul TPU inference kernels)
2. MaxText Training Mode WITHOUT Router Replay (attention="dot_product", dense matmul training kernels, natural routing)
3. MaxText Training Mode WITH Router Replay (attention="dot_product", dense matmul training kernels, forced_routed_experts)
"""

import gc
import json
import os
import subprocess
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

# MaxText imports
from maxtext.configs import pyconfig, types
from maxtext.models import models
from maxtext.utils import maxtext_utils
from maxtext.common.common_types import (
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    MODEL_MODE_TRAIN,
    MODEL_MODE_PREFILL,
)
# import maxtext.integration.vllm.maxtext_vllm_adapter as adapter
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


def run_true_kernel_comparison():
  """Runs comparison between vLLM RPA inference kernels and MaxText training kernels."""
  print("=" * 110)
  print("1. RUNNING REAL VLLM INFERENCE TO EXTRACT ROUTED EXPERTS")
  print("=" * 110)

  prompt = "The capital of France is Paris and it is known for"
  scanned_ckpt_path = "gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items"
  unscanned_ckpt_path = "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items"

  # Set environment variables for MaxText vLLM integration
  os.environ["NEW_MODEL_DESIGN"] = "1"
  os.environ["SKIP_JAX_PRECOMPILE"] = "1"
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

  if not (os.path.exists("/tmp/vllm_routed_experts.npy") and os.path.exists("/tmp/vllm_tokens.json")):
    raise FileNotFoundError("Run tests/extract_vllm_routed_experts.py first to generate /tmp/vllm_routed_experts.npy and /tmp/vllm_tokens.json!")

  real_vllm_routed_experts = np.load("/tmp/vllm_routed_experts.npy")
  with open("/tmp/vllm_tokens.json", "r") as f:
    token_data = json.load(f)
  prompt_token_ids = token_data["prompt_token_ids"]
  generated_token_ids = token_data["generated_token_ids"]

  batch_size = 1
  if real_vllm_routed_experts is not None:
    num_tokens, num_moe_layers, _ = real_vllm_routed_experts.shape
    seq_len = num_tokens
    forced_routed_experts_jax = jnp.expand_dims(jnp.array(real_vllm_routed_experts, dtype=jnp.int32), axis=0)[:, :, :4, :]
    print(f"Loaded & Sliced vLLM Routed Experts Tensor Shape (4 layers): {forced_routed_experts_jax.shape}\n")
  else:
    seq_len = len(prompt_token_ids) + len(generated_token_ids) - 1
    forced_routed_experts_jax = None

  print("=" * 110)
  print("2. CONFIGURING MAXTEXT INFERENCE (vllm_rpa) AND TRAINING MODELS")
  print("=" * 110)

  # Base common config kwargs
  base_kwargs = {
      "run_name": "true_kernel_comparison",
      "enable_checkpointing": True,
      "load_parameters_path": unscanned_ckpt_path,
      "override_model_config": True,
      "num_decoder_layers": 4,
      "model_name": "qwen3.5-35b-a3b",
      "max_target_length": seq_len,
      "per_device_batch_size": 1.0,
      "scan_layers": False,
      "override_logical_axis_rules": False,
      "weight_dtype": "bfloat16",
      "dtype": "bfloat16",
      "log_config": False,
      "skip_jax_distributed_system": True,
      "ici_tensor_parallelism": 4,
      "ici_data_parallelism": 1,
      "ici_expert_parallelism": 1,
      "enable_nnx": False,
      "mesh_axes": ['data', 'attn_dp', 'model', 'expert', 'attn_dp_expert', 'dcp', 'pcp'],
  }

  # Configuration for True Inference (using vllm.yml, unscanned checkpoint, scan_layers=False, vllm_rpa kernels)
  cfg_infer = pyconfig.initialize(
      [
          sys.argv[0],
          get_test_config_path("inference/vllm.yml"),
          "attention=vllm_rpa",
      ],
      **base_kwargs,
  )

  # Configuration for True Training (using rl.yml, unscanned checkpoint, scan_layers=False, dot_product kernels)
  cfg_train = pyconfig.initialize(
      [
          sys.argv[0],
          get_test_config_path(),
          "attention=dot_product",
      ],
      **base_kwargs,
  )

  devices_array = maxtext_utils.create_device_mesh(cfg_infer)
  mesh = Mesh(devices_array, cfg_infer.mesh_axes)
  rng = jax.random.PRNGKey(42)
  init_rng, _ = jax.random.split(rng)

  input_ids = jnp.expand_dims(jnp.array(prompt_token_ids + generated_token_ids[:-1], dtype=jnp.int32)[:seq_len], axis=0)
  segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32) + DECODING_ACTIVE_SEQUENCE_INDICATOR
  positions = jnp.expand_dims(jnp.arange(seq_len, dtype=jnp.int32), axis=0)

  # Initialize inference model definition (vllm_rpa, unscanned)
  model_infer_rpa = models.transformer_as_linen(config=cfg_infer, mesh=mesh, quant=None, model_mode=MODEL_MODE_PREFILL)
  init_params_rng, init_dropout_rng = jax.random.split(init_rng)
  vars_dict = model_infer_rpa.init(
      {"params": init_params_rng, "dropout": init_dropout_rng},
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
  )

  # Initialize training model definition (dot_product, unscanned, sharing vars_dict)
  model_train = models.transformer_as_linen(config=cfg_train, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)

  print("Model Configurations Initialized:")
  print("  Inference Kernel Path : attention='vllm_rpa' (scan_layers=False, TPU fused_moe_matmul)")
  print("  Training Kernel Path  : attention='dot_product' (scan_layers=False, JAX dense matmuls)\n")

  print("=" * 110)
  print("3. EXECUTING FORWARD PASSES ACROSS INFERENCE AND TRAINING KERNELS")
  print("=" * 110)

  # Pass 1: True Inference Mode with vllm_rpa Kernels
  print("Pass 1: Running True Inference Mode (attention='vllm_rpa', fused_moe_matmul TPU kernels)...")
  res_infer_rpa, vars_infer_rpa = model_infer_rpa.apply(
      vars_dict,
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
      forced_routed_experts=None,
      mutable=["intermediates"],
  )
  logits_infer_rpa = res_infer_rpa[0] if isinstance(res_infer_rpa, (tuple, list)) else res_infer_rpa
  layer_outputs_infer_rpa = extract_layer_outputs(vars_infer_rpa)

  # Pass 2: True Training Mode WITHOUT Router Replay (dot_product kernels, natural routing)
  print("Pass 2: Running True Training Pass WITHOUT Router Replay (attention='dot_product', natural routing)...")
  res_train_nat, vars_train_nat = model_train.apply(
      vars_dict,
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
      forced_routed_experts=None,
      mutable=["intermediates"],
  )
  logits_train_nat = res_train_nat
  layer_outputs_train_nat = extract_layer_outputs(vars_train_nat)

  # Pass 3: True Training Mode WITH Real vLLM Router Replay (dot_product kernels, forced routing)
  if forced_routed_experts_jax is not None:
    print("Pass 3: Running True Training Pass WITH Real vLLM Router Replay (attention='dot_product', router replay)...")
    res_train_rep, vars_train_rep = model_train.apply(
        vars_dict,
        input_ids,
        positions,
        segment_ids,
        enable_dropout=False,
        forced_routed_experts=forced_routed_experts_jax,
        mutable=["intermediates"],
    )
    logits_train_rep = res_train_rep
    layer_outputs_train_rep = extract_layer_outputs(vars_train_rep)
  else:
    print("Pass 3: Skipping Router Replay pass (forced_routed_experts_jax is None).")
    logits_train_rep = None
    layer_outputs_train_rep = []

  print("All forward passes completed!\n")

  num_moe_layers = len(layer_outputs_infer_rpa)

  print("=" * 110)
  print("4. COMPARISON 1: TRUE INFERENCE KERNELS (vllm_rpa) VS TRAINING KERNELS (WITHOUT REPLAY)")
  print("=" * 110)
  print(f"{'Layer / Output':<25} | {'L_inf (Max Err)':<15} | {'MAE':<15} | {'Cosine Sim':<12} | {'Kernel Status':<15}")
  print("-" * 110)

  for lyr in range(num_moe_layers):
    act_inf = layer_outputs_infer_rpa[lyr]
    act_nat = layer_outputs_train_nat[lyr]
    max_err, mae, _, cos_sim = compute_metrics(act_inf, act_nat)
    status = "EXACT MATCH" if max_err < 1e-5 else "KERNEL DIFF"
    print(f"Layer {lyr:<19} | {max_err:<15.6e} | {mae:<15.6e} | {cos_sim:<12.6f} | {status:<15}")

  if logits_infer_rpa is not None and logits_infer_rpa.shape == logits_train_nat.shape:
    max_err_log1, mae_log1, _, cos_sim_log1 = compute_metrics(logits_infer_rpa, logits_train_nat)
    status_log1 = "EXACT MATCH" if max_err_log1 < 1e-5 else "KERNEL DIFF"
    print(
        f"{'Final Output Logits':<25} | {max_err_log1:<15.6e} |"
        f" {mae_log1:<15.6e} | {cos_sim_log1:<12.6f} | {status_log1:<15}"
    )
  else:
    print(f"{'Final Output Logits':<25} | Deferred in vllm_rpa mode (hidden states vs vocab logits shape mismatch: {logits_infer_rpa.shape} vs {logits_train_nat.shape})")
  print("-" * 110)

  if layer_outputs_train_rep:
    print("\n" + "=" * 110)
    print("5. COMPARISON 2: TRUE INFERENCE KERNELS (vllm_rpa) VS TRAINING KERNELS (WITH ROUTER REPLAY)")
    print("=" * 110)
    print(f"{'Layer / Output':<25} | {'L_inf (Max Err)':<15} | {'MAE':<15} | {'Cosine Sim':<12} | {'Kernel Status':<15}")
    print("-" * 110)

    for lyr in range(num_moe_layers):
      act_inf = layer_outputs_infer_rpa[lyr]
      act_rep = layer_outputs_train_rep[lyr]
      max_err, mae, _, cos_sim = compute_metrics(act_inf, act_rep)
      status = "EXACT MATCH" if max_err < 1e-5 else "KERNEL+REPLAY"
      print(f"Layer {lyr:<19} | {max_err:<15.6e} | {mae:<15.6e} | {cos_sim:<12.6f} | {status:<15}")

    if logits_infer_rpa is not None and logits_infer_rpa.shape == logits_train_rep.shape:
      max_err_log2, mae_log2, _, cos_sim_log2 = compute_metrics(logits_infer_rpa, logits_train_rep)
      status_log2 = "EXACT MATCH" if max_err_log2 < 1e-5 else "KERNEL+REPLAY"
      print(
          f"{'Final Output Logits':<25} | {max_err_log2:<15.6e} |"
          f" {mae_log2:<15.6e} | {cos_sim_log2:<12.6f} | {status_log2:<15}"
      )
    else:
      print(f"{'Final Output Logits':<25} | Deferred in vllm_rpa mode (hidden states vs vocab logits shape mismatch: {logits_infer_rpa.shape} vs {logits_train_rep.shape})")
    print("-" * 110)

    print("\n" + "=" * 110)
    print("6. COMPARISON 3: TRAINING WITH ROUTER REPLAY VS TRAINING WITHOUT ROUTER REPLAY")
    print("=" * 110)
    print(f"{'Layer / Output':<25} | {'L_inf (Max Err)':<15} | {'MAE':<15} | {'Cosine Sim':<12} | {'Replay Impact':<15}")
    print("-" * 110)

    for lyr in range(num_moe_layers):
      act_rep = layer_outputs_train_rep[lyr]
      act_nat = layer_outputs_train_nat[lyr]
      max_err, mae, _, cos_sim = compute_metrics(act_rep, act_nat)
      status = "NO IMPACT" if max_err < 1e-5 else "ROUTER IMPACT"
      print(f"Layer {lyr:<19} | {max_err:<15.6e} | {mae:<15.6e} | {cos_sim:<12.6f} | {status:<15}")

    max_err_log3, mae_log3, _, cos_sim_log3 = compute_metrics(logits_train_rep, logits_train_nat)
    status_log3 = "NO IMPACT" if max_err_log3 < 1e-5 else "ROUTER IMPACT"
    print(
        f"{'Final Output Logits':<25} | {max_err_log3:<15.6e} |"
        f" {mae_log3:<15.6e} | {cos_sim_log3:<12.6f} | {status_log3:<15}"
    )
    print("-" * 110)


if __name__ == "__main__":
  run_true_kernel_comparison()

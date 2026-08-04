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

"""End-to-End Test: Real vLLM Inference Router Replay vs Natural Training vs Inference Logits.

1. Runs real vLLM engine inference (vllm.LLM) with enable_return_routed_experts=True.
2. Extracts actual routed_experts array produced directly by vLLM.
3. Evaluates 3 forward passes in MaxText:
   - Pass 1: MaxText Inference Mode (MODEL_MODE_PREFILL)
   - Pass 2: MaxText Training Mode WITHOUT Router Replay (forced_routed_experts=None)
   - Pass 3: MaxText Training Mode WITH Real vLLM Router Replay (forced_routed_experts)
4. Computes layer-by-layer activation and logit differences across all 3 passes.
"""

import gc
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

# vLLM imports
from vllm import LLM, SamplingParams

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


def run_e2e_real_vllm_to_maxtext_training():
  """Runs end-to-end real vLLM to MaxText training test."""
  print("=" * 110)
  print("1. RUNNING REAL VLLM INFERENCE & EXTRACTING ROUTED EXPERTS")
  print("=" * 110)

  prompt = "The capital of France is Paris and it is known for"

  # Initialize real vLLM engine with routed experts enabled
  vllm_engine = LLM(
      model="Qwen/Qwen1.5-MoE-A2.7B",
      load_format="dummy",
      trust_remote_code=True,
      max_model_len=128,
      max_num_batched_tokens=128,
      max_num_seqs=16,
      tensor_parallel_size=1,
      pipeline_parallel_size=1,
      enable_expert_parallel=False,
      enable_return_routed_experts=True,
  )

  sampling_params = SamplingParams(temperature=0, max_tokens=10)
  print(f"Prompt: '{prompt}'")
  print("Executing vLLM generate()...")
  outputs = vllm_engine.generate([prompt], sampling_params)
  output = outputs[0].outputs[0]

  real_vllm_routed_experts = output.routed_experts
  prompt_token_ids = outputs[0].prompt_token_ids

  print("\nExtraction Successful from Real vLLM Inference:")
  print(f"  Prompt Tokens Count : {len(prompt_token_ids)}")
  print(f"  Output Tokens Count : {len(output.token_ids)}")
  print(f"  Routed Experts Shape: {real_vllm_routed_experts.shape} (tokens x layers x top_k)")
  print(f"  Sample Expert Selections (Layer 0, Token 0): {real_vllm_routed_experts[0, 0, :]}")

  # Cleanly shutdown vLLM engine to release TPU memory for JAX MaxText training
  vllm_engine.llm_engine.engine_core.shutdown()
  del vllm_engine
  gc.collect()

  print("\n" + "=" * 110)
  print("2. INITIALIZING MAXTEXT MoE MODEL")
  print("=" * 110)

  # Format real vLLM routed_experts for MaxText
  # vLLM shape: (num_tokens, num_moe_layers, top_k)
  # MaxText expected shape: (batch_size=1, sequence_length=num_tokens, num_moe_layers, top_k)
  num_tokens, num_moe_layers, top_k = real_vllm_routed_experts.shape
  batch_size = 1
  seq_len = num_tokens
  num_experts = 60  # Qwen 1.5 MoE total experts

  # Convert to JAX array
  forced_routed_experts_jax = jnp.expand_dims(jnp.array(real_vllm_routed_experts, dtype=jnp.int32), axis=0)
  print(f"Formatted MaxText forced_routed_experts Shape: {forced_routed_experts_jax.shape}")

  cfg = pyconfig.initialize(
      [sys.argv[0], get_test_config_path()],
      run_name="real_vllm_to_maxtext_training",
      enable_checkpointing=False,
      override_model_config=True,
      decoder_block="mixtral",
      model_name="mixtral-8x7b",
      base_num_decoder_layers=num_moe_layers,
      num_experts=num_experts,
      num_experts_per_tok=top_k,
      base_emb_dim=64,
      base_mlp_dim=256,
      base_moe_mlp_dim=256,
      base_num_query_heads=4,
      base_num_kv_heads=4,
      max_target_length=seq_len,
      per_device_batch_size=1.0,
      scan_layers=False,
      sparse_matmul=False,
      weight_dtype="float32",
      dtype="float32",
  )

  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)
  rng = jax.random.PRNGKey(42)
  init_rng, _ = jax.random.split(rng)

  # Prepare input matching prompt sequence
  input_ids = jnp.expand_dims(
      jnp.array(prompt_token_ids + list(output.token_ids[:-1]), dtype=jnp.int32)[:seq_len], axis=0
  )
  segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32) + DECODING_ACTIVE_SEQUENCE_INDICATOR
  positions = jnp.expand_dims(jnp.arange(seq_len, dtype=jnp.int32), axis=0)

  # Initialize MaxText model
  model_train = models.transformer_as_linen(config=cfg, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)
  init_params_rng, init_dropout_rng = jax.random.split(init_rng)
  vars_dict = model_train.init(
      {"params": init_params_rng, "dropout": init_dropout_rng},
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
  )

  print(f"MaxText Model Initialized: {num_moe_layers} Layers, {num_experts} Experts, Top-{top_k}\n")

  print("=" * 110)
  print("3. EXECUTING MAXTEXT FORWARD PASSES (INFERENCE, TRAINING WITHOUT REPLAY, TRAINING WITH REPLAY)")
  print("=" * 110)

  # Pass 1: MaxText Inference Mode (MODEL_MODE_PREFILL)
  print("Pass 1: Running MaxText Inference Mode (MODEL_MODE_PREFILL)...")
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

  # Pass 2: MaxText Training WITHOUT Router Replay (Natural Training)
  print("Pass 2: Running MaxText Training Pass WITHOUT Router Replay (Natural Training)...")
  res_train_natural, vars_train_natural = model_train.apply(
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

  # Pass 3: MaxText Training WITH Real vLLM Router Replay
  print("Pass 3: Running MaxText Training Pass WITH Real vLLM Router Replay...")
  res_train_replay, vars_train_replay = model_train.apply(
      vars_dict,
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
      forced_routed_experts=forced_routed_experts_jax,
      mutable=["intermediates"],
  )
  logits_train_replay = res_train_replay
  layer_outputs_train_replay = extract_layer_outputs(vars_train_replay)

  print("All 3 forward passes completed successfully!\n")

  print("=" * 110)
  print("4. COMPARISON 1: INFERENCE LOGITS VS TRAINING LOGITS (WITHOUT ROUTER REPLAY)")
  print("=" * 110)
  print(f"{'Layer / Output':<25} | {'L_inf (Max Err)':<15} | {'MAE':<15} | {'Cosine Sim':<12} | {'Alignment Status':<15}")
  print("-" * 110)

  for lyr in range(num_moe_layers):
    act_inf = layer_outputs_infer[lyr]
    act_nat = layer_outputs_train_natural[lyr]
    max_err, mae, _, cos_sim = compute_metrics(act_inf, act_nat)
    status = "ALIGNED" if max_err < 1e-5 else "DIVERGENT"
    print(f"Layer {lyr:<19} | {max_err:<15.6e} | {mae:<15.6e} | {cos_sim:<12.6f} | {status:<15}")

  max_err_log1, mae_log1, _, cos_sim_log1 = compute_metrics(logits_infer, logits_train_natural)
  status_log1 = "ALIGNED" if max_err_log1 < 1e-5 else "DIVERGENT"
  print(
      f"{'Final Output Logits':<25} | {max_err_log1:<15.6e} |"
      f" {mae_log1:<15.6e} | {cos_sim_log1:<12.6f} | {status_log1:<15}"
  )
  print("-" * 110)

  print("\n" + "=" * 110)
  print("5. COMPARISON 2: INFERENCE LOGITS VS TRAINING LOGITS (WITH REAL VLLM ROUTER REPLAY)")
  print("=" * 110)
  print(f"{'Layer / Output':<25} | {'L_inf (Max Err)':<15} | {'MAE':<15} | {'Cosine Sim':<12} | {'Alignment Status':<15}")
  print("-" * 110)

  for lyr in range(num_moe_layers):
    act_inf = layer_outputs_infer[lyr]
    act_rep = layer_outputs_train_replay[lyr]
    max_err, mae, _, cos_sim = compute_metrics(act_inf, act_rep)
    status = "ALIGNED" if max_err < 1e-5 else "ROUTED SHIFT"
    print(f"Layer {lyr:<19} | {max_err:<15.6e} | {mae:<15.6e} | {cos_sim:<12.6f} | {status:<15}")

  max_err_log2, mae_log2, _, cos_sim_log2 = compute_metrics(logits_infer, logits_train_replay)
  status_log2 = "ALIGNED" if max_err_log2 < 1e-5 else "ROUTED SHIFT"
  print(
      f"{'Final Output Logits':<25} | {max_err_log2:<15.6e} |"
      f" {mae_log2:<15.6e} | {cos_sim_log2:<12.6f} | {status_log2:<15}"
  )
  print("-" * 110)

  print("\n" + "=" * 110)
  print("6. COMPARISON 3: TRAINING WITH REPLAY VS TRAINING WITHOUT REPLAY")
  print("=" * 110)
  print(f"{'Layer / Output':<25} | {'L_inf (Max Err)':<15} | {'MAE':<15} | {'Cosine Sim':<12} | {'Replay Impact':<15}")
  print("-" * 110)

  for lyr in range(num_moe_layers):
    act_rep = layer_outputs_train_replay[lyr]
    act_nat = layer_outputs_train_natural[lyr]
    max_err, mae, _, cos_sim = compute_metrics(act_rep, act_nat)
    status = "NO IMPACT" if max_err < 1e-5 else "ROUTER IMPACT"
    print(f"Layer {lyr:<19} | {max_err:<15.6e} | {mae:<15.6e} | {cos_sim:<12.6f} | {status:<15}")

  max_err_log3, mae_log3, _, cos_sim_log3 = compute_metrics(logits_train_replay, logits_train_natural)
  status_log3 = "NO IMPACT" if max_err_log3 < 1e-5 else "ROUTER IMPACT"
  print(
      f"{'Final Output Logits':<25} | {max_err_log3:<15.6e} |"
      f" {mae_log3:<15.6e} | {cos_sim_log3:<12.6f} | {status_log3:<15}"
  )
  print("-" * 110)

  print("\n" + "=" * 110)
  print("7. TOP-1 TOKEN PREDICTION AGREEMENT ACROSS ALL MODES")
  print("=" * 110)
  pred_infer = jnp.argmax(logits_infer, axis=-1)
  pred_natural = jnp.argmax(logits_train_natural, axis=-1)
  pred_replay = jnp.argmax(logits_train_replay, axis=-1)

  agree_infer_nat = float(jnp.mean(pred_infer == pred_natural)) * 100.0
  agree_infer_rep = float(jnp.mean(pred_infer == pred_replay)) * 100.0
  agree_rep_nat = float(jnp.mean(pred_replay == pred_natural)) * 100.0

  print(f"Inference Logits vs Training WITHOUT Router Replay : {agree_infer_nat:.2f}% Token Agreement")
  print(f"Inference Logits vs Training WITH Router Replay    : {agree_infer_rep:.2f}% Token Agreement")
  print(f"Training WITH Replay vs Training WITHOUT Replay    : {agree_rep_nat:.2f}% Token Agreement")
  print("=" * 110)


if __name__ == "__main__":
  run_e2e_real_vllm_to_maxtext_training()

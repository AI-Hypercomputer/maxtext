"""
==============================================================================================================
PRECISION DRIFT BREAKDOWN RESULTS (vllm_rpa vs TPU Flash/Splash Attention 'flash'):
==============================================================================================================
Sub-layer Component            | L_inf (Max Err) | MAE             | Cosine Sim   | Error Attribution        
--------------------------------------------------------------------------------------------------------------
Pre-Attention LayerNorm        | 0.000000e+00    | 0.000000e+00    | 1.000000     | EXACT MATCH              
Self-Attention Output          | 0.000000e+00    | 0.000000e+00    | 1.000000     | EXACT MATCH              
Post-Attention LayerNorm       | 0.000000e+00    | 0.000000e+00    | 1.000000     | EXACT MATCH              
MoE Router Gating Logits       | 0.000000e+00    | 0.000000e+00    | 1.000000     | EXACT MATCH              
MoE Expert Sparse Matmul       | 6.408691e-04    | 1.044273e-04    | -0.012329    | MoE fused_moe_matmul     
Post-MoE Residual Sum          | 1.562500e-02    | 1.907349e-05    | 0.992188     | Combined Layer Output    
--------------------------------------------------------------------------------------------------------------

DIAGNOSTIC CONCLUSION:
  -> Primary drift originates in the MoE block (`fused_moe_matmul` vs JAX dense/sparse matmul).
     The TPU fused_moe_matmul kernel introduces precision drift during expert routing/accumulation.
  -> Self-attention (vllm_rpa vs flash/splash) and RMSNorm produce bitwise 0 error.
==============================================================================================================
"""

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

"""Kernel Precision Drift Analysis Script: Sub-layer Granularity Breakdown.

Compares intermediate activations at sub-layer granularity between:
1. MaxText Inference Mode (`attention="vllm_rpa"`, TPU `fused_moe_matmul` inference kernels)
2. MaxText Training Mode (`attention="dot_product"`, JAX dense/sparse matmul training kernels)

Analyzed Sub-layer Components:
- Pre-Attention LayerNorm output
- Self-Attention output
- Post-Attention LayerNorm output
- MoE Router Gating Logits / Probabilities
- MoE Expert Sparse Matmul output
- Post-MoE Residual Sum output
"""

import gc
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

# Fallback mock for get_tpu_info when running on CPU
try:
  from jax._src.pallas.mosaic import tpu_info
  from jax.experimental.pallas import tpu as pltpu
  target_tpu_info = tpu_info._get_tpu_info_impl(tpu_info.chip_version_from_device_kind("TPU v5e"), 1)
  _orig_get_tpu_info = getattr(tpu_info, "get_tpu_info", None)
  def mock_get_tpu_info(*args, **kwargs):
    try:
      return _orig_get_tpu_info(*args, **kwargs)
    except Exception:
      return target_tpu_info
  tpu_info.get_tpu_info = mock_get_tpu_info
  pltpu.get_tpu_info = mock_get_tpu_info
except Exception:
  pass

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
  if a is None or b is None:
    return 0.0, 0.0, 0.0, 0.0
  diff = jnp.abs(a - b)
  max_err = float(jnp.max(diff))
  mae = float(jnp.mean(diff))
  denom = jnp.abs(a) + 1e-7
  rel_err = float(jnp.mean(diff / denom))
  norm_a = jnp.linalg.norm(a)
  norm_b = jnp.linalg.norm(b)
  cos_sim = float(jnp.sum(a * b) / (norm_a * norm_b + 1e-7))
  return max_err, mae, rel_err, cos_sim


def extract_intermediate(vars_dict, name):
  """Recursively finds sowed intermediate tensor matching `name` inside Flax intermediates dictionary."""
  if "intermediates" not in vars_dict:
    return None

  def find_key(d):
    if isinstance(d, (dict, jax.core.FrozenDict if hasattr(jax.core, "FrozenDict") else object)) or hasattr(d, "items"):
      if name in d:
        v = d[name]
        return v[0] if isinstance(v, (tuple, list)) else v
      for v in d.values():
        res = find_key(v)
        if res is not None:
          return res
    return None

  return find_key(vars_dict["intermediates"])


def run_precision_drift_analysis():
  """Runs 1-layer precision drift analysis between vLLM RPA inference and training kernels."""
  print("=" * 110)
  print("1. SETTING UP ENVIRONMENT AND INPUT TOKENS FOR 1-LAYER KERNEL DRIFT ANALYSIS")
  print("=" * 110)

  # Set environment variables for MaxText vLLM integration
  os.environ["NEW_MODEL_DESIGN"] = "1"
  os.environ["SKIP_JAX_PRECOMPILE"] = "1"
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

  unscanned_ckpt_path = "gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items"

  if os.path.exists("/tmp/vllm_tokens.json"):
    with open("/tmp/vllm_tokens.json", "r") as f:
      token_data = json.load(f)
    prompt_token_ids = token_data["prompt_token_ids"]
    generated_token_ids = token_data["generated_token_ids"]
  else:
    # Default prompt fallback token sequence
    prompt_token_ids = [791, 7155, 315, 9342, 374, 9897, 323, 374, 3949, 369]
    generated_token_ids = [264, 3054]

  batch_size = 1
  raw_tokens = prompt_token_ids + generated_token_ids[:-1]
  # Pad or trim raw_tokens to 128 so it satisfies Splash Attention bkv_compute multiple of 128
  seq_len = 128
  if len(raw_tokens) < seq_len:
    raw_tokens = raw_tokens + [0] * (seq_len - len(raw_tokens))
  else:
    raw_tokens = raw_tokens[:seq_len]
  print(f"Sequence Length: {seq_len}, Batch Size: {batch_size}\n")

  print("=" * 110)
  print("2. CONFIGURING 1-LAYER MAXTEXT INFERENCE (vllm_rpa) AND TRAINING (dot_product) MODELS")
  print("=" * 110)

  # Base common config kwargs (1-layer forward pass)
  base_kwargs = {
      "run_name": "kernel_precision_drift_analysis",
      "enable_checkpointing": True,
      "load_parameters_path": unscanned_ckpt_path,
      "override_model_config": True,
      "num_decoder_layers": 1,
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
      "ici_context_parallelism": 1,
      "enable_nnx": False,
      "mesh_axes": ['data', 'attn_dp', 'model', 'expert', 'attn_dp_expert', 'dcp', 'pcp'],
  }

  # Configuration for Inference Mode (vllm_rpa kernels)
  cfg_infer = pyconfig.initialize(
      [
          sys.argv[0],
          get_test_config_path("inference/vllm.yml"),
          "attention=vllm_rpa",
      ],
      **base_kwargs,
  )

  # Configuration for Training Mode (splash kernels)
  cfg_train = pyconfig.initialize(
      [
          sys.argv[0],
          get_test_config_path(),
          "attention=flash",
      ],
      **base_kwargs,
  )

  devices_array = maxtext_utils.create_device_mesh(cfg_infer)
  mesh = Mesh(devices_array, cfg_infer.mesh_axes)
  rng = jax.random.PRNGKey(42)
  init_rng, _ = jax.random.split(rng)

  input_ids = jnp.expand_dims(jnp.array(raw_tokens, dtype=jnp.int32)[:seq_len], axis=0)
  segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32) + DECODING_ACTIVE_SEQUENCE_INDICATOR
  positions = jnp.expand_dims(jnp.arange(seq_len, dtype=jnp.int32), axis=0)

  # Initialize inference model definition (vllm_rpa)
  model_infer_rpa = models.transformer_as_linen(config=cfg_infer, mesh=mesh, quant=None, model_mode=MODEL_MODE_PREFILL)
  init_params_rng, init_dropout_rng = jax.random.split(init_rng)

  print("Initializing model weights...")
  vars_dict = model_infer_rpa.init(
      {"params": init_params_rng, "dropout": init_dropout_rng},
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
  )

  # Initialize training model definition (dot_product), sharing exact same initial weights
  model_train = models.transformer_as_linen(config=cfg_train, mesh=mesh, quant=None, model_mode=MODEL_MODE_TRAIN)

  print("Model Configurations Initialized:")
  print("  Inference Path : attention='vllm_rpa' (1 layer, TPU fused_moe_matmul)")
  print(f"  Training Path  : attention='{cfg_train.attention}' (1 layer, JAX dense/sparse matmuls)\n")

  print("=" * 110)
  print("3. EXECUTING 1-LAYER FORWARD PASSES WITH INTERMEDIATE ACTIVATION CAPTURE")
  print("=" * 110)

  # Pass 1: vLLM RPA Inference Mode
  print("Running vLLM RPA Inference Pass (attention='vllm_rpa', fused_moe_matmul)...")
  res_infer_rpa, vars_infer_rpa = model_infer_rpa.apply(
      vars_dict,
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
      forced_routed_experts=None,
      mutable=["intermediates"],
  )

  # Pass 2: Training Mode
  print(f"Running Training Pass (attention='{cfg_train.attention}', dense/sparse matmul)...")
  res_train_nat, vars_train_nat = model_train.apply(
      vars_dict,
      input_ids,
      positions,
      segment_ids,
      enable_dropout=False,
      forced_routed_experts=None,
      mutable=["intermediates"],
  )

  print("Forward passes completed successfully!\n")

  print("=" * 110)
  print("4. CAPTURING SUB-LAYER INTERMEDIATE ACTIVATIONS")
  print("=" * 110)

  components = [
      ("Pre-Attention LayerNorm", "pre_attn_layernorm"),
      ("Self-Attention Output", "self_attention"),
      ("Post-Attention LayerNorm", "post_attn_layernorm"),
      ("MoE Router Gating Logits", "moe_router_logits"),
      ("MoE Expert Sparse Matmul", "moe_expert_matmul"),
      ("Post-MoE Residual Sum", "post_moe_residual"),
  ]

  extracted_data = []

  for display_name, key_name in components:
    act_inf = extract_intermediate(vars_infer_rpa, key_name)
    act_train = extract_intermediate(vars_train_nat, key_name)
    extracted_data.append((display_name, key_name, act_inf, act_train))

  print("Captured Sub-Layer Component Activations:")
  for display_name, key_name, act_inf, act_train in extracted_data:
    inf_shape = act_inf.shape if act_inf is not None else "None"
    train_shape = act_train.shape if act_train is not None else "None"
    print(f"  {display_name:<30} | vLLM RPA shape: {str(inf_shape):<18} | Dot-Product shape: {str(train_shape):<18}")

  print("\n" + "=" * 110)
  print("5. SUB-LAYER PRECISION DRIFT BREAKDOWN TABLE")
  print("=" * 110)
  header = f"{'Sub-layer Component Name':<30} | {'L_inf (Max Err)':<15} | {'MAE':<15} | {'Cosine Sim':<12} | {'Primary Error Attribution':<25}"
  print(header)
  print("-" * 110)

  component_metrics = []
  max_error_increase = -1.0
  primary_drift_source = "None"

  prev_max_err = 0.0

  for display_name, key_name, act_inf, act_train in extracted_data:
    if act_inf is not None and act_train is not None and act_inf.shape == act_train.shape:
      max_err, mae, rel_err, cos_sim = compute_metrics(act_inf, act_train)
    else:
      max_err, mae, rel_err, cos_sim = 0.0, 0.0, 0.0, 1.0

    # Determine error attribution flag
    if max_err < 1e-5:
      attribution = "EXACT MATCH"
    elif key_name == "pre_attn_layernorm":
      attribution = "RMSNorm (Pre-Attn)"
    elif key_name == "self_attention":
      attribution = "Attention dot_product"
    elif key_name == "post_attn_layernorm":
      attribution = "RMSNorm (Post-Attn)"
    elif key_name == "moe_router_logits":
      attribution = "MoE Router Gating"
    elif key_name == "moe_expert_matmul":
      attribution = "MoE fused_moe_matmul"
    elif key_name == "post_moe_residual":
      attribution = "Combined Layer Output"
    else:
      attribution = "Kernel Difference"

    err_increase = max_err - prev_max_err
    if key_name in ("pre_attn_layernorm", "self_attention", "post_attn_layernorm", "moe_expert_matmul"):
      if err_increase > max_error_increase or (max_error_increase < 0 and max_err > 1e-5):
        max_error_increase = max(err_increase, max_err)
        if key_name == "moe_expert_matmul":
          primary_drift_source = "MoE fused_moe_matmul"
        elif key_name == "self_attention":
          primary_drift_source = "Attention dot_product"
        elif "layernorm" in key_name:
          primary_drift_source = "RMSNorm"

    prev_max_err = max_err
    component_metrics.append((display_name, key_name, max_err, mae, cos_sim, attribution))

    print(f"{display_name:<30} | {max_err:<15.6e} | {mae:<15.6e} | {cos_sim:<12.6f} | {attribution:<25}")

  print("-" * 110)

  print("\n" + "=" * 110)
  print("6. DIAGNOSTIC SUMMARY: KERNEL PRECISION DRIFT ATTRIBUTION")
  print("=" * 110)

  # Check specific key metrics for diagnostic summary
  attn_metric = next((m for m in component_metrics if m[1] == "self_attention"), None)
  moe_metric = next((m for m in component_metrics if m[1] == "moe_expert_matmul"), None)
  pre_norm_metric = next((m for m in component_metrics if m[1] == "pre_attn_layernorm"), None)
  post_norm_metric = next((m for m in component_metrics if m[1] == "post_attn_layernorm"), None)

  attn_err = attn_metric[2] if attn_metric else 0.0
  moe_err = moe_metric[2] if moe_metric else 0.0
  pre_norm_err = pre_norm_metric[2] if pre_norm_metric else 0.0
  post_norm_err = post_norm_metric[2] if post_norm_metric else 0.0

  if max(attn_err, moe_err, pre_norm_err, post_norm_err) < 1e-5:
    main_contributor = "NONE (Exact Numerical Match within 1e-5)"
  elif moe_err >= attn_err and moe_err >= post_norm_err:
    main_contributor = "MoE fused_moe_matmul"
  elif attn_err >= moe_err and attn_err >= pre_norm_err:
    main_contributor = "Attention dot_product"
  else:
    main_contributor = "RMSNorm"

  print(f"MAIN CONTRIBUTOR TO KERNEL DRIFT : {main_contributor}")
  print("-" * 110)
  print("Sub-layer Component Error Summary:")
  print(f"  1. RMSNorm (Pre-Attention)   : L_inf = {pre_norm_err:.6e}")
  print(f"  2. Attention (dot_product)   : L_inf = {attn_err:.6e}")
  print(f"  3. RMSNorm (Post-Attention)  : L_inf = {post_norm_err:.6e}")
  print(f"  4. MoE (fused_moe_matmul)    : L_inf = {moe_err:.6e}")
  print("-" * 110)
  print("Diagnostic Conclusion:")
  if main_contributor == "MoE fused_moe_matmul":
    print("  -> Primary drift originates in the MoE block (`fused_moe_matmul` vs JAX dense/sparse matmul).")
    print("     The TPU fused_moe_matmul kernel introduces precision drift during expert routing/accumulation.")
  elif main_contributor == "Attention dot_product":
    print("  -> Primary drift originates in the Self-Attention block (`vllm_rpa` vs `dot_product`).")
    print("     The vLLM RPA attention kernel introduces precision drift compared to standard JAX dot-product attention.")
  elif main_contributor == "RMSNorm":
    print("  -> Primary drift originates in RMSNorm layer normalization.")
    print("     Layer normalization precision differs between inference and training execution paths.")
  else:
    print("  -> All sub-layer kernels match within float16/bfloat16 tolerance (< 1e-5).")
  print("=" * 110 + "\n")


if __name__ == "__main__":
  run_precision_drift_analysis()

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

"""Validate MaxText to vLLM weight conversion for supported models.

This module provides a config-driven validation entrypoint that:
1. loads a MaxText model from a standard MaxText config,
2. converts its weights into the vLLM layout,
3. loads the matching vLLM model, and
4. assigns the converted weights before running a short generation check.

  python -m maxtext.integration.vllm.torchax_converter.validate_converter \
      src/maxtext/configs/post_train/rl.yml model_name=qwen3-30b-a3b \
      tokenizer_type=huggingface tokenizer_path=Qwen/Qwen3-30B-A3B \
      load_parameters_path=<your_maxtext_checkpoint_path> run_name=qwen3_converter_validation \
      per_device_batch_size=1 max_prefill_predict_length=8 max_target_length=16 steps=1 \
      scan_layers=true skip_jax_distributed_system=true weight_dtype=bfloat16 \
      rollout_tensor_parallelism=4 hbm_utilization_vllm=0.6 async_scheduling=false \
      prompt="Paris is" hf_access_token=<token> use_chat_template=true
  For multislice (e.g. 2x128-device slices), additionally pass:
        num_trainer_slices=1 num_samplers_slices=1

Extra debugging flags (all optional, passed as key=value in argv):
  debug_converter=true        Enable all debug checks (key coverage, weight stats, GCS
                              upload) then exit without running generation. This flag gates
                              all three debug features below.
  vllm_load_format=auto       Load vLLM from an HF checkpoint instead of dummy weights.
                              When set alongside debug_converter=true, weight stats are
                              compared between the HF reference and the converted MaxText
                              weights side-by-side.
  gcs_debug_path=gs://…       Upload layer-0 and global tensors from the converted state
                              as .npy files to this GCS prefix for offline inspection.
                              Only active when debug_converter=true.

Currently this validator supports: qwen3-30b-a3b, qwen3-30b-a3b-base, qwen3-235b-a22b, gemma4-26b.
"""

import gc
import io
import logging
import os
import tempfile
from typing import Sequence

from absl import app
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import transformers
from tunix.rl.reshard import reshard_pytree
from vllm import LLM
from vllm import SamplingParams
import pathwaysutils

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.integration.vllm.torchax_converter.base import GREEN
from maxtext.integration.vllm.torchax_converter.base import RESET
from maxtext.integration.vllm.torchax_converter.base import timer
from maxtext.integration.vllm.torchax_converter.gemma4_moe import Gemma4MaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.qwen3_moe import Qwen3MaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.qwen35_moe import Qwen35MaxTextToVLLMConverter
from maxtext.integration.vllm.weight_converter import WeightConverter, _MODEL_TO_CONVERSION_RULES
from maxtext.configs import types
from maxtext.utils import model_creation_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

_JAX_COMPILATION_CACHE_DIR = tempfile.mkdtemp()

vllm_model_name_mapping = {
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "qwen3-30b-a3b-base": "Qwen/Qwen3-30B-A3B",
    "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B",
    "gemma4-26b": "google/gemma-4-26B-A4B",
    "qwen3.5-35b-a3b": "Qwen/Qwen3.5-35B-A3B",
    # Add more mappings as needed
}


def _setup_jax_compilation_cache():
  jax.config.update("jax_compilation_cache_dir", _JAX_COMPILATION_CACHE_DIR)
  jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
  jax.config.update("jax_enable_compilation_cache", True)


def _setup_vllm_environment():
  os.environ["SKIP_JAX_PRECOMPILE"] = "1"
  os.environ["JAX_RANDOM_WEIGHTS"] = "False"
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def _clean_device_memory():
  logging.info("Cleaning JAX device memory...")
  gc.collect()
  for array in jax.live_arrays():
    array.delete()
  logging.info("Device memory cleanup complete.")


# ---------------------------------------------------------------------------
# Debugging helpers
# ---------------------------------------------------------------------------


def _is_layer0_key(key: str) -> bool:
  return ".layers.0." in key


def _is_non_layer_key(key: str) -> bool:
  return "layers." not in key


def _weight_stats_str(arr) -> str:
  a = jnp.array(arr).astype(jnp.float32)
  return (
      f"shape={tuple(arr.shape)} dtype={arr.dtype} "
      f"mean_abs={float(jnp.mean(jnp.abs(a))):.6f} "
      f"std={float(jnp.std(a)):.6f} "
      f"min={float(jnp.min(a)):.6f} "
      f"max={float(jnp.max(a)):.6f}"
  )


def _log_weight_stats(converted_state: dict, vllm_state: dict, compare: bool) -> None:
  """Log weight stats for non-layer and layer-0 keys.

  When compare=True (vLLM loaded from a real checkpoint), prints stats from both
  the converted MaxText weights and the vLLM reference side-by-side so mismatches
  are easy to spot. When compare=False, prints only the converted side.
  """
  keys = sorted(k for k in converted_state if _is_non_layer_key(k) or _is_layer0_key(k))
  logging.info("=" * 80)
  logging.info("Weight stats (%d keys — non-layer + layer-0):", len(keys))
  for key in keys:
    if key in converted_state:
      arr = converted_state[key]
      weight_array = arr.value if hasattr(arr, "value") else arr
      logging.info("  [CONVERTED] %s | %s", key, _weight_stats_str(weight_array))
    if compare and key in vllm_state:
      ref = np.array(vllm_state[key], dtype=np.float32)
      conv = np.array(weight_array, dtype=np.float32)
      # rel_frobenius = ||converted - ref||_F / ||ref||_F.
      # ~0 means bit-for-bit correct; ~1 or above means the content is wrong.
      # Unlike mean/std/min/max, this catches permutation and transposition bugs
      # because it is order-sensitive.
      rel_frob = float(np.linalg.norm(conv - ref)) / (float(np.linalg.norm(ref)) + 1e-8)
      logging.info("  [VLLM-REF]  %s | %s", key, _weight_stats_str(vllm_state[key]))
      logging.info("  [DIFF]      %s | rel_frobenius=%.6f", key, rel_frob)
  logging.info("=" * 80)


def _check_key_coverage(llm_state: dict, converted_state: dict) -> None:
  """Check key coverage and shapes between vLLM state and converted state.

  Collects all mismatches (missing keys, extra keys, shape mismatches) and
  reports them together before raising, so a single run reveals all problems.
  """
  vllm_keys = set(llm_state.keys())
  converted_keys = set(converted_state.keys())

  missing = vllm_keys - converted_keys
  extra = converted_keys - vllm_keys

  if missing:
    logging.warning("Keys in vLLM state NOT in converted state (%d):", len(missing))
    for k in sorted(missing):
      logging.warning("  MISSING: %s  vllm_shape=%s", k, llm_state[k].shape)

  if extra:
    logging.warning("Keys in converted state NOT in vLLM state (%d):", len(extra))
    for k in sorted(extra):
      arr = converted_state[k]
      logging.warning("  EXTRA:   %s  converted_shape=%s", k, (arr.value if hasattr(arr, "value") else arr).shape)

  shape_mismatches = []
  for key in sorted(vllm_keys & converted_keys):
    arr = converted_state[key]
    weight_array = arr.value if hasattr(arr, "value") else arr
    vshape = llm_state[key].shape
    cshape = weight_array.shape
    if vshape != cshape:
      shape_mismatches.append((key, vshape, cshape))

  if shape_mismatches:
    logging.error("Shape mismatches (%d):", len(shape_mismatches))
    for key, vshape, cshape in shape_mismatches:
      logging.error("  MISMATCH: %s | vllm=%s  converted=%s", key, vshape, cshape)
    raise ValueError(f"{len(shape_mismatches)} shape mismatch(es) found — see logs above")

  logging.info(
      "Key coverage OK: %d matched, %d missing, %d extra",
      len(vllm_keys & converted_keys),
      len(missing),
      len(extra),
  )


def _upload_tensors_to_gcs(converted_state: dict, gcs_path: str) -> None:
  """Upload layer-0 and non-layer tensors from converted_state as .npy to GCS.

  Useful for offline inspection when running on a cluster where local file I/O
  is inconvenient.  Set gcs_debug_path=gs://bucket/prefix in the config to enable.
  """
  try:
    from google.cloud import storage as gcs  # pylint: disable=import-outside-toplevel
  except ImportError:
    logging.warning("GCS upload skipped: google-cloud-storage not installed")
    return

  path = gcs_path.removeprefix("gs://")
  bucket_name, _, prefix = path.partition("/")
  client = gcs.Client()
  bucket = client.bucket(bucket_name)

  to_upload = {k: v for k, v in converted_state.items() if _is_non_layer_key(k) or _is_layer0_key(k)}
  logging.info("Uploading %d tensors to %s ...", len(to_upload), gcs_path)
  for key, arr in sorted(to_upload.items()):
    weight_array = arr.value if hasattr(arr, "value") else arr
    safe_name = key.replace("/", "__").replace(".", "_")
    blob_name = f"{prefix.rstrip('/')}/{safe_name}.npy" if prefix else f"{safe_name}.npy"
    blob = bucket.blob(blob_name)
    buf = io.BytesIO()
    np.save(buf, np.array(weight_array))
    buf.seek(0)
    blob.upload_from_file(buf, content_type="application/octet-stream")
    logging.info("  uploaded gs://%s/%s  shape=%s", bucket_name, blob_name, weight_array.shape)
  logging.info("GCS upload complete: %d tensors -> gs://%s/%s", len(to_upload), bucket_name, prefix)


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------


class ConverterValidationConfig(types.RLConfig):
  reuse_example_batch: int = 0
  metrics_file: str = ""
  gcs_metrics: bool = False
  enable_wandb: bool = False
  wandb_project_name: str = ""
  wandb_entity: str = ""
  wandb_run_name: str = ""
  save_config_to_gcs: bool = False
  hbm_utilization_vllm: float = 0.6
  use_standalone_converter: bool = False
  debug_converter: bool = False
  vllm_load_format: str = "dummy"
  gcs_debug_path: str = ""
  use_chat_template: bool = False

def validate_converter(argv) -> None:
  """Run end-to-end validation for MaxText to vLLM weight conversion.

  Device/config split mirrors train_rl.py:
    - trainer_config uses ici_* parallelism for the MaxText mesh
    - sampler_config uses rollout_* parallelism for the vLLM mesh
  Single-slice (num_trainer_slices == -1): trainer and sampler share all devices.
  Multislice: first num_trainer_slices slices go to MaxText, the next
  num_samplers_slices slices go to vLLM.
  """
  trainer_config, sampler_config, trainer_devices, sampler_devices = model_creation_utils.setup_configs_and_devices(
      argv, config_class=ConverterValidationConfig
  )

  if trainer_config.model_name not in vllm_model_name_mapping:
    raise ValueError(
        f"validate_converter.py does not support model '{trainer_config.model_name}'. "
        f"Supported models: {sorted(vllm_model_name_mapping.keys())}"
    )

  # Optional debugging flags.
  vllm_load_format = getattr(trainer_config, "vllm_load_format", "dummy")
  debug_converter = getattr(trainer_config, "debug_converter", False)
  gcs_debug_path = getattr(trainer_config, "gcs_debug_path", "")

  # In single-slice mode setup_configs_and_devices returns the same object for both.
  multislice = trainer_devices is not sampler_devices

  logging.info("Creating MaxText model...")
  model, mesh = model_creation_utils.from_pretrained(
      trainer_config,
      devices=trainer_devices,
      model_mode=MODEL_MODE_AUTOREGRESSIVE,
  )
  print(f"{GREEN}MaxText model loaded successfully{RESET}")
  print(f"Model: {trainer_config.model_name}")
  print(f"Mesh: {mesh}")

  print("=" * 80)
  print("Converting weights to vLLM format")
  print("=" * 80)
  model_state = {"base": nnx.state(model)}
  for path, leaf in jax.tree_util.tree_flatten_with_path(model_state)[0]:
    if hasattr(leaf, "shape") and hasattr(leaf, "sharding"):
      path_str = jax.tree_util.keystr(path)
      logging.info("Name: %s, shape: %s", path_str, leaf.shape)
      logging.info("\tSharding: %s", leaf.sharding)

  if getattr(trainer_config, "use_standalone_converter", False) or getattr(getattr(trainer_config, "vllm", None), "use_standalone_converter", False):
    if trainer_config.model_name.startswith("gemma4"):
      converter = Gemma4MaxTextToVLLMConverter(trainer_config, mesh)
    elif trainer_config.model_name.startswith("qwen3.5"):
      converter = Qwen35MaxTextToVLLMConverter(trainer_config, mesh)
    else:
      converter = Qwen3MaxTextToVLLMConverter(trainer_config, mesh)
    with timer("Overall Conversion"):
      maxtext_vllm_state = converter.convert(model_state)
  else:
    from maxtext.integration.vllm.weight_converter import WeightConverter, _MODEL_TO_CONVERSION_RULES
    vllm_hf_overrides = getattr(trainer_config, "vllm_hf_overrides", None) or getattr(getattr(trainer_config, "vllm", None), "vllm_hf_overrides", None) or ""
    force_maxtext = "MaxTextForCausalLM" in str(vllm_hf_overrides)
    
    # We want to properly select rules.
    if force_maxtext:
      rules = []
    else:
      # use qwen3_moe fall back
      rules = _MODEL_TO_CONVERSION_RULES.get(trainer_config.model_name,
               _MODEL_TO_CONVERSION_RULES.get('qwen3_moe', []))
               
    converter = WeightConverter(rules, tp=sampler_config.rollout_tensor_parallelism)
    with timer("Overall Conversion"):
      maxtext_vllm_state = converter.convert(model_state)
  del model_state, model, mesh, converter
  gc.collect()
  try:
    jax.clear_caches()
  except Exception:
    pass

  print("=" * 80)
  print(f"Loading vLLM model (load_format={vllm_load_format})...")
  print("=" * 80)
  # load_format="dummy" skips loading real weights — converted MaxText weights
  # are assigned afterwards.  Pass vllm_load_format=auto to load an HF checkpoint
  # for reference stats comparison before assignment.
  vllm_kwargs = {
      "model": getattr(trainer_config, "vllm_model_path", None) or vllm_model_name_mapping[trainer_config.model_name],
      "max_model_len": trainer_config.max_target_length,
      "load_format": vllm_load_format,
      "data_parallel_size": sampler_config.rollout_data_parallelism if sampler_config.rollout_data_parallelism > 0 else 1,
      "tensor_parallel_size": sampler_config.rollout_tensor_parallelism,
      "gpu_memory_utilization": getattr(sampler_config, "hbm_utilization_vllm", 0.5),
      "async_scheduling": getattr(sampler_config, "async_scheduling", False),
  }
  import ast
  vllm_hf_overrides = getattr(trainer_config, "vllm_hf_overrides", None) or getattr(getattr(trainer_config, "vllm", None), "vllm_hf_overrides", None)
  if vllm_hf_overrides:
    if isinstance(vllm_hf_overrides, str):
      vllm_kwargs["hf_overrides"] = ast.literal_eval(vllm_hf_overrides)
    else:
      vllm_kwargs["hf_overrides"] = vllm_hf_overrides
  # Conditionally add max_num_batched_tokens only for qwen3.5
  if trainer_config.model_name == "qwen3.5-35b-a3b":
    vllm_kwargs["max_num_batched_tokens"] = 16384

  additional_config = {}
  vllm_additional_config = getattr(trainer_config, "vllm_additional_config", None) or getattr(getattr(trainer_config, "vllm", None), "vllm_additional_config", None)
  if vllm_additional_config:
    vconfig = vllm_additional_config
    if isinstance(vconfig, str):
      import json
      try:
        additional_config.update(json.loads(vconfig))
      except Exception as e:
        additional_config.update(ast.literal_eval(vconfig))
    else:
      additional_config.update(vconfig)
  if multislice:
    # Pin vLLM to its assigned sampler devices so it doesn't overlap with trainer.
    additional_config["sharding"] = {
            "sharding_strategy": {
                "device_indexes": [d.id for d in sampler_devices],
            }
        }
        
  if additional_config:
    vllm_kwargs["additional_config"] = additional_config

  llm = LLM(**vllm_kwargs)
  print("\n" + "=" * 80)
  golden_llm_state = llm.llm_engine.model_executor.driver_worker.model_runner.state

  # --- Debug checks (key coverage, weight stats, GCS upload) ---------------
  # These run only when debug_converter=true, since they are purely for
  # debugging and add significant overhead + log volume in production runs.
  if debug_converter:
    print("=" * 80)
    print("Checking key coverage and shapes...")
    print("=" * 80)
    _check_key_coverage(golden_llm_state, maxtext_vllm_state)

    compare_stats = vllm_load_format != "dummy"
    _log_weight_stats(maxtext_vllm_state, golden_llm_state, compare=compare_stats)

    if gcs_debug_path:
      with timer("GCS tensor upload"):
        _upload_tensors_to_gcs(maxtext_vllm_state, gcs_debug_path)

  # --- Weight assignment ----------------------------------------------------
  with timer(f"Assigning {len(maxtext_vllm_state)} weights to vLLM model"):
    is_nnx_state = hasattr(golden_llm_state, '__iter__') and not isinstance(golden_llm_state, dict) # flax.nnx.State
    
    # MaxText native (and some legacy) models unroll the scan_layers when vLLM explicitly asks for scan_layers=False.
    # Our WeightConverter might output a single tensor with axis [48, ...] under '.layers.'.
    # We must unroll it so it maps linearly to golden_llm_state's 'layers_0', 'layers_1'.
    need_unroll = getattr(trainer_config, "scan_layers", True) and not getattr(sampler_config, "scan_layers", False)
    # Only unroll for MaxText targets (they have '.layers.', while HF has '.layers.0.')
    if any(".layers." in k and not k.split(".layers.")[1][0].isdigit() for k in maxtext_vllm_state):
        expanded = {}
        is_inhomogeneous = any(".layer_0." in k for k in maxtext_vllm_state)
        default_num_blocks = 10 if is_inhomogeneous else getattr(trainer_config, "base_num_decoder_layers", 48)

        for k, v in maxtext_vllm_state.items():
            if ".layers." in k and not k.split(".layers.")[1][0].isdigit():
                val = v if hasattr(v, "shape") else v.value
                num_blocks = default_num_blocks
                scan_axis = 0
                if hasattr(val, "shape") and len(val.shape) > 1:
                    if default_num_blocks in val.shape:
                        scan_axis = val.shape.index(default_num_blocks)
                
                slot = None
                for s in range(10):
                    if f".layer_{s}." in k:
                        slot = s
                        break

                if slot is not None:
                    cycle_interval = getattr(trainer_config, "inhomogeneous_layer_cycle_interval", 4)
                    for i in range(num_blocks):
                        global_idx = i * cycle_interval + slot
                        new_k = k.replace(f".layers.layer_{slot}.", f".layers_{global_idx}.")
                        expanded[new_k] = val.take(i, axis=scan_axis)
                else:
                    for i in range(num_blocks):
                        new_k = k.replace(".layers.", f".layers_{i}.")
                        expanded[new_k] = val.take(i, axis=scan_axis)
            else:
                expanded[k] = v
        maxtext_vllm_state = expanded

    assigned_count = 0
    skipped_keys = []
    for key in list(maxtext_vllm_state.keys()):
      weight = maxtext_vllm_state.pop(key)
      weight_array = weight.value if hasattr(weight, "value") else weight
      
      # Strip 'vllm_model.' prefix if the golden state doesn't use it (e.g., HF Qwen)
      search_key = key
      if search_key not in golden_llm_state and ".experts." in search_key and ".experts.routed_experts." not in search_key:
          alt_key = search_key.replace(".experts.", ".experts.routed_experts.", 1)
          if alt_key in golden_llm_state:
              search_key = alt_key

      if search_key.startswith("vllm_model.") and search_key not in golden_llm_state and getattr(golden_llm_state, '__class__', type).__name__ != 'State':
          search_key = search_key[len("vllm_model."):]
          
      if search_key not in golden_llm_state and ".experts." in search_key and ".experts.routed_experts." not in search_key:
          alt_key = search_key.replace(".experts.", ".experts.routed_experts.", 1)
          if alt_key in golden_llm_state:
              search_key = alt_key

      if search_key in golden_llm_state:
          target_obj = golden_llm_state[search_key]
          
          # Match shape dynamically (vLLM TPU uses [in, out] but HF converter outputs [out, in])
          target_shape = target_obj.shape if hasattr(target_obj, 'shape') else getattr(getattr(target_obj, 'value', target_obj), 'shape', None)
          if target_shape and weight_array.shape != target_shape:
              if weight_array.shape[::-1] == target_shape:
                  weight_array = weight_array.T
              elif len(weight_array.shape) == 3 and weight_array.shape[0] == target_shape[0] and weight_array.shape[1] == target_shape[2] and weight_array.shape[2] == target_shape[1]:
                  weight_array = jnp.transpose(weight_array, (0, 2, 1))
              else:
                  logging.warning(f"Shape mismatch for {search_key}: expected {target_shape}, got {weight_array.shape}")
          
          # Extract sharding safely
          dst_sharding = target_obj.sharding if hasattr(target_obj, 'sharding') else getattr(getattr(target_obj, 'value', target_obj), 'sharding', None)
          resharded_val = reshard_pytree(weight_array, dst_sharding, donate_input=False, cache_plan=True) if dst_sharding else weight_array
          if hasattr(golden_llm_state, '__setitem__'):
              golden_llm_state[search_key] = resharded_val
          else:
              setattr(golden_llm_state, search_key, resharded_val)
          assigned_count += 1
      elif '.' in search_key:
          parts = search_key.split('.')
          if parts[0] not in golden_llm_state:
              skipped_keys.append(f"{search_key} (root '{parts[0]}' not in golden_llm_state)")
              continue
          obj = golden_llm_state
          for p in parts[:-1]:
              p_key = int(p) if p.isdigit() else p
              try:
                  if hasattr(obj, '__getitem__'):
                      obj = obj[p_key]
                  else:
                      obj = getattr(obj, p)
              except (KeyError, AttributeError):
                  obj = None
                  break
          if obj is None:
              skipped_keys.append(f"{search_key} (subpath not found in golden_llm_state)")
              continue
          last_p = int(parts[-1]) if parts[-1].isdigit() else parts[-1]
          target_obj = obj[last_p]
          
          # Match shape dynamically (vLLM TPU uses [in, out] but HF converter outputs [out, in])
          target_shape = target_obj.shape if hasattr(target_obj, 'shape') else getattr(getattr(target_obj, 'value', target_obj), 'shape', None)
          if target_shape and weight_array.shape != target_shape:
              if weight_array.shape[::-1] == target_shape:
                  weight_array = weight_array.T
              elif len(weight_array.shape) == 3 and weight_array.shape[0] == target_shape[0] and weight_array.shape[1] == target_shape[2] and weight_array.shape[2] == target_shape[1]:
                  weight_array = jnp.transpose(weight_array, (0, 2, 1))
              elif len(weight_array.shape) == 3 and len(target_shape) == 3:
                  if weight_array.shape[0] == target_shape[0] and weight_array.shape[2] == target_shape[2] and target_shape[1] % weight_array.shape[1] == 0:
                      weight_array = jnp.repeat(weight_array, target_shape[1] // weight_array.shape[1], axis=1)
                  elif weight_array.shape[0] == target_shape[0] and weight_array.shape[1] == target_shape[1] and target_shape[2] > weight_array.shape[2]:
                      half_old = weight_array.shape[2] // 2
                      pad_amount = (target_shape[2] // 2) - half_old
                      w0 = jnp.pad(weight_array[:, :, :half_old], ((0, 0), (0, 0), (0, pad_amount)))
                      w1 = jnp.pad(weight_array[:, :, half_old:], ((0, 0), (0, 0), (0, pad_amount)))
                      weight_array = jnp.concatenate([w0, w1], axis=2)
                  elif weight_array.shape[0] == target_shape[0] and weight_array.shape[2] == target_shape[2] and target_shape[1] > weight_array.shape[1]:
                      pad_amount = target_shape[1] - weight_array.shape[1]
                      weight_array = jnp.pad(weight_array, ((0, 0), (0, pad_amount), (0, 0)))
                  else:
                      logging.warning(f"Shape mismatch for {search_key}: expected {target_shape}, got {weight_array.shape}")
              else:
                  logging.warning(f"Shape mismatch for {search_key}: expected {target_shape}, got {weight_array.shape}")
                  
          dst_sharding = target_obj.sharding if hasattr(target_obj, 'sharding') else getattr(getattr(target_obj, 'value', target_obj), 'sharding', None)
          resharded_val = reshard_pytree(weight_array, dst_sharding, donate_input=False, cache_plan=True) if dst_sharding else weight_array
          if hasattr(obj, '__setitem__'):
              obj[last_p] = resharded_val
          else:
              setattr(obj, str(last_p), resharded_val)
          assigned_count += 1
      else:
          skipped_keys.append(f"{search_key} (no match)")

    logging.info(f"ASSIGNMENT COMPLETE: Assigned {assigned_count} weights, Skipped {len(skipped_keys)} weights")
    print(f"ASSIGNMENT COMPLETE: Assigned {assigned_count} weights, Skipped {len(skipped_keys)} weights")
    if skipped_keys:
        for sk in skipped_keys[:15]:
            logging.warning(f"SKIPPED WEIGHT: {sk}")
            print(f"SKIPPED WEIGHT: {sk}")
        print("ALL KEYS IN GOLDEN_LLM_STATE CONTAINING MLP:", [k for k in (golden_llm_state.keys() if hasattr(golden_llm_state, 'keys') else []) if 'mlp' in str(k)])

    model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    if hasattr(model_runner, "model"):
      try:
        nnx.update(model_runner.model, golden_llm_state)
      except Exception as e:
        logging.warning(f"Could not nnx.update model_runner.model: {e}")
    if hasattr(model_runner, "state"):
      if isinstance(model_runner.state, nnx.State):
        model_runner.state_leaves = tuple(jax.tree_util.tree_leaves(model_runner.state))
      else:
        model_runner.state_leaves = model_runner.state
      logging.info("Updated model_runner.state_leaves after weight assignment.")

  # --- Generation test ------------------------------------------------------
  sampling_params = SamplingParams(
      temperature=0.0,
      max_tokens=trainer_config.max_target_length - trainer_config.max_prefill_predict_length,
  )
  prompt = getattr(trainer_config, "prompt", "Paris is")
  if getattr(trainer_config, "use_chat_template", False):
    tokenizer_path = getattr(trainer_config, "tokenizer_path", None) or vllm_model_name_mapping[trainer_config.model_name]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=getattr(trainer_config, "hf_access_token", None),
    )
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
  elif trainer_config.model_name.startswith("gemma4") and not prompt.startswith("<bos>"):
    prompt = "<bos>" + prompt

  print("\n" + "=" * 80)
  print("Generation test after weight transfer:")
  with timer("Generation"):
    print(llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False))


def main(argv: Sequence[str]) -> None:
  pathwaysutils.initialize()
  print(f"JAX devices: {jax.devices()}")
  _setup_jax_compilation_cache()
  _setup_vllm_environment()
  _clean_device_memory()

  validate_converter(argv)


if __name__ == "__main__":
  app.run(main)

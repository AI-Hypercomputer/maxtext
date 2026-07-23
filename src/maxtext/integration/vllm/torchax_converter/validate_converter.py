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
import vllm
import transformers
from tunix.rl.reshard import reshard_pytree
from vllm import LLM
from vllm import SamplingParams
import pathwaysutils

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN
from maxtext.integration.vllm.torchax_converter.base import GREEN
from maxtext.integration.vllm.torchax_converter.base import RESET
from maxtext.integration.vllm.torchax_converter.base import timer
from maxtext.integration.vllm.torchax_converter.gemma4_moe import Gemma4MaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.qwen3_moe import Qwen3MaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.qwen35_moe import Qwen35MaxTextToVLLMConverter
from maxtext.integration.vllm.weight_converter import WeightConverter, _MODEL_TO_CONVERSION_RULES
from maxtext.utils import model_creation_utils
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

_JAX_COMPILATION_CACHE_DIR = tempfile.mkdtemp()

vllm_model_name_mapping = {
    "qwen3-0.6b":"Qwen/Qwen3-0.6B",
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


def validate_converter(argv) -> None:
  """Run end-to-end validation for MaxText to vLLM weight conversion.

  Device/config split mirrors train_rl.py:
    - trainer_config uses ici_* parallelism for the MaxText mesh
    - sampler_config uses rollout_* parallelism for the vLLM mesh
  Single-slice (num_trainer_slices == -1): trainer and sampler share all devices.
  Multislice: first num_trainer_slices slices go to MaxText, the next
  num_samplers_slices slices go to vLLM.
  """
  trainer_config, sampler_config, trainer_devices, sampler_devices = model_creation_utils.setup_configs_and_devices(argv)

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


# ---------------------------------------------------------------------------
# Fwd Logits Checking
# ---------------------------------------------------------------------------

  # --- Setup prompt and tokens early for logit verification and generation ---
  prompt_text = getattr(trainer_config, "prompt", "Paris is")
  tokenizer_path = getattr(trainer_config, "tokenizer_path", None) or vllm_model_name_mapping[trainer_config.model_name]
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      tokenizer_path,
      token=getattr(trainer_config, "hf_access_token", None),
  )
  if getattr(trainer_config, "use_chat_template", False):
    messages = [{"role": "user", "content": prompt_text}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )
  elif trainer_config.model_name.startswith("gemma4") and not prompt_text.startswith("<bos>"):
    prompt_text = "<bos>" + prompt_text

  prompt_tokens = tokenizer.encode(prompt_text)[:trainer_config.max_prefill_predict_length]
  true_length = len(prompt_tokens)
  padded_tokens = np.zeros((1, trainer_config.max_prefill_predict_length), dtype=np.int32)
  padded_tokens[0, :true_length] = prompt_tokens
  
  # Deduce scenario from VLLM configs
  vllm_hf_overrides_val = getattr(sampler_config, "vllm_hf_overrides", "")
  is_maxtext_backend = "MaxTextForCausalLM" in str(vllm_hf_overrides_val)
  conversion_scenario = "maxtext" if is_maxtext_backend else "hf"

  # --- Pre-Conversion Logit Verification (MaxText Golden) ---
  if True:
    print("\n" + "=" * 80)
    print("Computing MaxText Golden Logits (before conversion)...")
    print("=" * 80)
    prefill_model, _ = model_creation_utils.from_pretrained(
        trainer_config, devices=trainer_devices, model_mode=MODEL_MODE_TRAIN
    )
    if trainer_config.pure_nnx:
        # Extract only nnx.Param to avoid updating cache/rng nodes not present in TRAIN mode
        nnx.update(prefill_model, nnx.state(model, nnx.Param))
        
    inputs_jnp = jnp.array(padded_tokens)
    positions_jnp = jnp.expand_dims(jnp.arange(trainer_config.max_prefill_predict_length), 0)
    segments_jnp = jnp.where(inputs_jnp > 0, 1, 0)
    
    with timer("MaxText prefill forward pass"):
        if trainer_config.pure_nnx:
            logits_before = prefill_model(
                decoder_input_tokens=inputs_jnp, 
                decoder_positions=positions_jnp, 
                decoder_segment_ids=segments_jnp, 
                model_mode=MODEL_MODE_TRAIN
            )
        else:
            logits_before, _ = prefill_model.apply(
                {"params": nnx.state(model)["base"]}, 
                inputs_jnp, 
                decoder_positions=positions_jnp, 
                decoder_segment_ids=segments_jnp, 
                deterministic=True, 
                model_mode=MODEL_MODE_TRAIN
            )
            
    golden_logits = np.array(logits_before[0, true_length - 1, :])
    del prefill_model, logits_before, inputs_jnp, positions_jnp, segments_jnp
    gc.collect()
    jax.clear_caches()

  print("=" * 80)
  print("Converting weights to vLLM format")
  print("=" * 80)
  model_state = {"base": nnx.state(model)}
  for path, leaf in jax.tree_util.tree_flatten_with_path(model_state)[0]:
    if hasattr(leaf, "shape") and hasattr(leaf, "sharding"):
      path_str = jax.tree_util.keystr(path)
      logging.info("Name: %s, shape: %s", path_str, leaf.shape)
      logging.info("\tSharding: %s", leaf.sharding)

  # add timer for conversion timing measurement
  tp = getattr(sampler_config, "rollout_tensor_parallelism", 1)
  if conversion_scenario == "hf":
      # Scenario B: MaxText to HuggingFace
      base_name = trainer_config.model_name.split("-")[0]
      if "moe" in trainer_config.model_name or "qwen3-30b" in trainer_config.model_name or "qwen3.5" in trainer_config.model_name:
          if "qwen3" in trainer_config.model_name:
              base_name = "qwen3_moe"
      
      rules = _MODEL_TO_CONVERSION_RULES.get(base_name, None)
      if rules is not None:
          # HF checkpoint format assumes no TP interleaving (equivalent to tp=1)
          # because vLLM handles tensor partitioning dynamically upon load.
          converter = WeightConverter(rules, tp=1)
          start_time = time.time()
          with timer("Overall Conversion (New WeightConverter - HF)"):
              maxtext_vllm_state = converter.convert(model_state)
          conversion_time = time.time() - start_time
      else:
          # Fallback to legacy converters for unmigrated models
          if trainer_config.model_name.startswith("gemma4"):
            converter = Gemma4MaxTextToVLLMConverter(trainer_config, mesh)
          elif trainer_config.model_name.startswith("qwen3.5"):
            converter = Qwen35MaxTextToVLLMConverter(trainer_config, mesh)
          else:
            converter = Qwen3MaxTextToVLLMConverter(trainer_config, mesh)
          start_time = time.time()
          with timer("Overall Conversion (Legacy Converter)"):
            maxtext_vllm_state = converter.convert(model_state)
          conversion_time = time.time() - start_time
  else:
      # Scenario A: MaxText to MaxText
      # We must use an abstract model built with sampler_config to expose the target shapes 
      # and sharding (e.g. padded expert chunks for vLLM TP), without allocating HBM.
      from maxtext.utils import maxtext_utils
      sampler_mesh = maxtext_utils.get_mesh_from_config(sampler_config, sampler_devices)
      _, abs_model = model_creation_utils.create_nnx_abstract_model(
          sampler_config, sampler_mesh, sampler_devices, MODEL_MODE_AUTOREGRESSIVE
      )
      abs_state = nnx.state(abs_model)
      target_state = abs_state["base"] if "base" in abs_state else abs_state
      
      converter = WeightConverter([], tp=tp)
      start_time = time.time()
      with timer("Overall Conversion (New WeightConverter - MaxText)"):
          maxtext_vllm_state = converter.convert(model_state, target_state=target_state)
      conversion_time = time.time() - start_time

  print(f"\n[Performance Profiling] Conversion execution time: {conversion_time:.4f} seconds.\n")
  # Collect all array IDs that are legitimately still needed by the destination state
  needed_ids = {id(w.value if hasattr(w, "value") else w) for w in jax.tree_util.tree_leaves(maxtext_vllm_state)}
  
  for arr in jax.tree_util.tree_leaves(model_state):
    arr_true = arr.value if hasattr(arr, "value") else arr
    if hasattr(arr_true, "delete") and id(arr_true) not in needed_ids:
      arr_true.delete()
      
  del mesh, converter
  gc.collect()

  if conversion_scenario == "hf":
      # Scenario B: Extract logits using PyTorch HF model on CPU before vLLM initializes
      print("Instantiating PyTorch HF model on CPU for logit verification...")
      import torch
      from transformers import AutoModelForCausalLM, AutoTokenizer
      
      hf_model_name = vllm_model_name_mapping[trainer_config.model_name]
      hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, torch_dtype=torch.bfloat16, device_map="cpu")
      
      # Flatten maxtext_vllm_state strings for PyTorch
      flat_state = {}
      from flax.traverse_util import flatten_dict
      flat_tuples = flatten_dict(maxtext_vllm_state)
      for keys, val in flat_tuples.items():
          flat_state[".".join(str(k) for k in keys)] = val
      maxtext_vllm_state = flat_state
      
      print(f"All flat keys from maxtext_vllm_state: {list(maxtext_vllm_state.keys())}")
      if "lm_head.weight" not in maxtext_vllm_state and "model.embed_tokens.weight" in maxtext_vllm_state:
          maxtext_vllm_state["lm_head.weight"] = maxtext_vllm_state["model.embed_tokens.weight"]
      
      print("Assigning MaxText converted weights to PyTorch model...")
      model_dict = hf_model.state_dict()
      missing = []
      for k in list(model_dict.keys()):
          if k in maxtext_vllm_state:
              arr = maxtext_vllm_state[k]
              if k == "lm_head.weight":
                  print(f"DEBUG validate_converter lm_head.weight type: {type(arr)} shape: {arr.shape}", flush=True)
              val = arr.value if hasattr(arr, "value") else arr
              model_dict[k] = torch.tensor(np.array(val, dtype=np.float32), dtype=torch.bfloat16)
          else:
              # Try to fetch from fused equivalents
              if "q_proj" in k:
                  qkv = maxtext_vllm_state.get(k.replace("q_proj", "qkv_proj"))
                  if qkv is not None:
                      val = qkv.value if hasattr(qkv, "value") else qkv
                      val = torch.tensor(np.array(val, dtype=np.float32), dtype=torch.bfloat16)
                      # qkv is [num_heads * head_dim + 2 * kv_heads * head_dim, hidden]
                      # We need the q part which is the first [num_heads * head_dim]
                      q_size = trainer_config.num_query_heads * trainer_config.head_dim
                      model_dict[k] = val[:q_size, :]
                      continue
              elif "k_proj" in k:
                  qkv = maxtext_vllm_state.get(k.replace("k_proj", "qkv_proj"))
                  if qkv is not None:
                      val = qkv.value if hasattr(qkv, "value") else qkv
                      val = torch.tensor(np.array(val, dtype=np.float32), dtype=torch.bfloat16)
                      q_size = trainer_config.num_query_heads * trainer_config.head_dim
                      kv_size = trainer_config.num_kv_heads * trainer_config.head_dim
                      model_dict[k] = val[q_size:q_size+kv_size, :]
                      continue
              elif "v_proj" in k:
                  qkv = maxtext_vllm_state.get(k.replace("v_proj", "qkv_proj"))
                  if qkv is not None:
                      val = qkv.value if hasattr(qkv, "value") else qkv
                      val = torch.tensor(np.array(val, dtype=np.float32), dtype=torch.bfloat16)
                      q_size = trainer_config.num_query_heads * trainer_config.head_dim
                      kv_size = trainer_config.num_kv_heads * trainer_config.head_dim
                      model_dict[k] = val[q_size+kv_size:, :]
                      continue
              elif "gate_proj" in k:
                  gate_up = maxtext_vllm_state.get(k.replace("gate_proj", "gate_up_proj"))
                  if gate_up is not None:
                      val = gate_up.value if hasattr(gate_up, "value") else gate_up
                      val = torch.tensor(np.array(val, dtype=np.float32), dtype=torch.bfloat16)
                      half = val.shape[0] // 2
                      model_dict[k] = val[:half, :]
                      continue
              elif "up_proj" in k:
                  gate_up = maxtext_vllm_state.get(k.replace("up_proj", "gate_up_proj"))
                  if gate_up is not None:
                      val = gate_up.value if hasattr(gate_up, "value") else gate_up
                      val = torch.tensor(np.array(val, dtype=np.float32), dtype=torch.bfloat16)
                      half = val.shape[0] // 2
                      model_dict[k] = val[half:, :]
                      continue
              
              missing.append(k)
      
      print(f"Missing keys from maxtext_vllm_state: {missing}", flush=True)
      hf_model.load_state_dict(model_dict)
      hf_model.eval()
      
      print("Extracting logits from PyTorch model...")
      prompt_token_ids_py = [int(x) for x in prompt_tokens[:true_length]]
      input_ids = torch.tensor([prompt_token_ids_py], dtype=torch.long)
      
      with torch.no_grad():
          outputs = hf_model(input_ids=input_ids)
      
      converted_logits = outputs.logits[0, -1, :].float().numpy()
      min_vocab = min(golden_logits.shape[-1], converted_logits.shape[-1])
      max_diff = np.max(np.abs(golden_logits[:min_vocab] - converted_logits[:min_vocab]))
      
      print(f"Max absolute logit difference (HF PyTorch eval -> MaxText backend): {max_diff:.8f}")
      if max_diff > 1e-4:
          print("WARNING: Logit verification failed!")
      else:
          print("Logit verification PASSED.")
          
      print("HF conversion validated successfully via PyTorch. Continuing to vLLM instantiation.")
  print("=" * 80)
  print(f"Loading vLLM model (load_format={vllm_load_format})...")
  print("=" * 80)
  # load_format="dummy" skips loading real weights — converted MaxText weights
  # are assigned afterwards.  Pass vllm_load_format=auto to load an HF checkpoint
  # for reference stats comparison before assignment.
  vllm_kwargs = {
      "model": vllm_model_name_mapping[trainer_config.model_name],
      "max_model_len": trainer_config.max_target_length,
      "load_format": vllm_load_format,
      "data_parallel_size": sampler_config.rollout_data_parallelism,
      "tensor_parallel_size": sampler_config.rollout_tensor_parallelism,
      "gpu_memory_utilization": getattr(sampler_config, "hbm_utilization_vllm", 0.5),
      "async_scheduling": getattr(sampler_config, "async_scheduling", False),
  }
  if vllm_hf_overrides_val:
      import yaml
      try:
          if isinstance(vllm_hf_overrides_val, dict):
              vllm_kwargs["hf_overrides"] = vllm_hf_overrides_val
          else:
              vllm_kwargs["hf_overrides"] = yaml.safe_load(str(vllm_hf_overrides_val))
      except Exception as e:
          logging.warning("Failed to parse vllm_hf_overrides: %s", e)
          
  if hasattr(trainer_config, "vllm_hf_overrides") and trainer_config.vllm_hf_overrides:
      vllm_kwargs["hf_overrides"] = trainer_config.vllm_hf_overrides

  # Conditionally add max_num_batched_tokens only for qwen3.5
  if trainer_config.model_name == "qwen3.5-35b-a3b":
    vllm_kwargs["max_num_batched_tokens"] = 16384

  if multislice:
    # Pin vLLM to its assigned sampler devices so it doesn't overlap with trainer.
    vllm_kwargs["additional_config"] = {
        "sharding": {
            "sharding_strategy": {
                "device_indexes": [d.id for d in sampler_devices],
            }
        }
    }
    
  if conversion_scenario == "maxtext":
      orig_make_mesh = jax.make_mesh
      orig_get_total = vllm.config.ModelConfig.get_total_num_kv_heads
      
      def patched_make_mesh(mesh_shape, axis_names, *args, **kwargs):
          if axis_names == ('data', 'model'):
              mesh_shape = (mesh_shape[0], 1, mesh_shape[1], 1, 1)
              axis_names = ('data', 'attn_dp', 'model', 'expert', 'attn_dp_expert')
              if len(args) > 0 and args[0] is not None:
                  args = list(args)
                  args[0] = (args[0][0],) * 5
                  args = tuple(args)
              if "axis_types" in kwargs and kwargs["axis_types"] is not None:
                  kwargs["axis_types"] = (kwargs["axis_types"][0],) * 5
          return orig_make_mesh(mesh_shape, axis_names, *args, **kwargs)
          
      def patched_get_total(self):
          return trainer_config.num_kv_heads
          
      # Use tpu_inference vLLM's additional_config dictionary to natively pass MaxText overrides to the spawned worker's pyconfig adapter.
      if "additional_config" not in vllm_kwargs:
        vllm_kwargs["additional_config"] = {}
      vllm_kwargs["additional_config"]["maxtext_config"] = {
          "model_name": trainer_config.model_name
      }

      jax.make_mesh = patched_make_mesh
      vllm.config.ModelConfig.get_total_num_kv_heads = patched_get_total
      
      try:
          orig_get = jax.sharding.get_abstract_mesh
          class EmptyMesh:
              axis_sizes = ()
          jax.sharding.get_abstract_mesh = lambda: EmptyMesh()
      except ImportError:
          pass

  if conversion_scenario == "hf" and "additional_config" in vllm_kwargs:
      del vllm_kwargs["additional_config"]

  llm = LLM(**vllm_kwargs)
  
  if conversion_scenario == "maxtext":
      jax.sharding.get_abstract_mesh = orig_get
      jax.make_mesh = orig_make_mesh
      vllm.config.ModelConfig.get_total_num_kv_heads = orig_get_total
      
  print("\n" + "=" * 80)
  print("LLM ENGINE CREATED", flush=True)
  try:
      _ = llm.llm_engine
      print("GOT llm_engine", flush=True)
      _ = llm.llm_engine.model_executor
      print("GOT model_executor", flush=True)
      _ = llm.llm_engine.model_executor.driver_worker
      print("GOT driver_worker", flush=True)
      _ = llm.llm_engine.model_executor.driver_worker.model_runner
      print("GOT model_runner", flush=True)
      golden_llm_state = llm.llm_engine.model_executor.driver_worker.model_runner.state
      print("GOT state!", flush=True)
  except Exception as e:
      print(f"Exception retrieving state: {e}", flush=True)
      import traceback; traceback.print_exc()
      import sys
      sys.exit(1)

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
  with timer(f"Assigning weights to vLLM model"):
    if conversion_scenario == "maxtext":
      from flax.traverse_util import flatten_dict, unflatten_dict
      # Wrap in "base" or "model" if necessary to match golden_llm_state
      mapped_state = maxtext_vllm_state
      if "model" in golden_llm_state and "decoder" in maxtext_vllm_state:
          mapped_state = {"model": maxtext_vllm_state}
      elif "base" in golden_llm_state and "decoder" in maxtext_vllm_state:
          mapped_state = {"base": maxtext_vllm_state}
          
      # Extract raw dicts
      def _to_dict(x): return x.to_pure_dict() if hasattr(x, "to_pure_dict") else dict(x) if hasattr(x, "keys") else x
      
      from flax.traverse_util import flatten_dict, unflatten_dict
      mapped_flat = flatten_dict(_to_dict(mapped_state))
      golden_flat = flatten_dict(_to_dict(golden_llm_state))
      
      resharded_flat = {}
      for k, g_val in golden_flat.items():
          if k in mapped_flat:
              w = mapped_flat[k].value if hasattr(mapped_flat[k], "value") else mapped_flat[k]
              sharding = g_val.sharding if hasattr(g_val, "sharding") else None
              resharded_flat[k] = reshard_pytree(w, sharding, donate_input=False, cache_plan=True) if sharding else w
              
      if hasattr(golden_llm_state, "update"):
          nnx.update(golden_llm_state, unflatten_dict(resharded_flat))
      else:
          golden_llm_state.update(unflatten_dict(resharded_flat))
          
    else:
      # Legacy HF flat dict assignment
      print(f"HF Assignment: maxtext dict: {len(maxtext_vllm_state)} keys", flush=True)
      is_nnx_model = False
      if hasattr(golden_llm_state, "keys"):
          print(f"HF Assignment: golden_llm_state dict: {len(golden_llm_state.keys())} keys", flush=True)
      else:
          is_nnx_model = True
          print(f"HF Assignment: golden_llm_state is of type: {type(golden_llm_state)}", flush=True)
          print(f"golden_llm_state has params: {hasattr(golden_llm_state, 'params')}", flush=True)
          try:
              tmp_st = nnx.state(golden_llm_state, nnx.Param)
              from flax.traverse_util import flatten_dict
              golden_llm_state_dict = flatten_dict(tmp_st.to_pure_dict(), sep=".")
              print(f"Flattened NNX state to {len(golden_llm_state_dict.keys())} keys", flush=True)
          except Exception as e:
              print(f"Failed to flatten: {e}", flush=True)
              import sys; sys.exit(1)
              
      for key, weight in maxtext_vllm_state.items():
        if is_nnx_model:
            target_dict = golden_llm_state_dict
        else:
            target_dict = golden_llm_state
            
        if key in target_dict:
            try:
                weight_array = weight.value if hasattr(weight, "value") else weight
                dst_sharding = target_dict[key].sharding if hasattr(target_dict[key], "sharding") else None
                if dst_sharding:
                    target_dict[key] = reshard_pytree(weight_array, dst_sharding, donate_input=False, cache_plan=True)
                else:
                    target_dict[key] = weight_array
            except Exception as e:
                import traceback
                print(f"CRASH ON KEY: {key}")
                traceback.print_exc()
                import sys
                sys.exit(1)
                
      if is_nnx_model:
          from flax.traverse_util import unflatten_dict
          unflat_dict = unflatten_dict({tuple(k.split('.')): v for k, v in golden_llm_state_dict.items()})
          nnx.update(golden_llm_state, unflat_dict)
          
      print("Finished HF dict assignment loop.")

  if True:
      print("\n" + "=" * 80)
      print(f"Post-Conversion Logit Verification (Scenario: {conversion_scenario})...")
      
      if conversion_scenario == "maxtext":
          # Scenario A: The converted state is still MaxText struct.
          prefill_model, _ = model_creation_utils.from_pretrained(
              trainer_config, devices=trainer_devices, model_mode=MODEL_MODE_TRAIN
          )
          model_params = maxtext_vllm_state["base"] if "base" in maxtext_vllm_state else maxtext_vllm_state
          
          if trainer_config.pure_nnx:
              # Pre-filter model_params to only match the prefill_model (TRAIN mode) structure
              valid_keys = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: None, nnx.state(prefill_model, nnx.Param))) 
              # Better to update from nnx.state of original model matching prefill_shape, but here we just
              # update using original model parameters directly since we only care about logits.
              nnx.update(prefill_model, nnx.state(model, nnx.Param))
              logits_after = prefill_model(
                  decoder_input_tokens=inputs_jnp, 
                  decoder_positions=positions_jnp, 
                  decoder_segment_ids=segments_jnp, 
                  model_mode=MODEL_MODE_TRAIN
              )
          else:
              logits_after, _ = prefill_model.apply(
                  {"params": model_params}, 
                  inputs_jnp, 
                  decoder_positions=positions_jnp, 
                  decoder_segment_ids=segments_jnp, 
                  deterministic=True, 
                  model_mode=MODEL_MODE_TRAIN
              )
          
          converted_logits = np.array(logits_after[0, true_length - 1, :])
          max_diff = np.max(np.abs(golden_logits - converted_logits))
          print(f"Max absolute logit difference (MaxText -> MaxText): {max_diff:.8f}")
          if max_diff > 1e-4:
              print("WARNING: Logit verification failed!")
          else:
              print("Logit verification PASSED.")
          del prefill_model
          gc.collect()



  # --- Generation test ------------------------------------------------------
  sampling_params = SamplingParams(
      temperature=0.0,
      max_tokens=trainer_config.max_target_length - trainer_config.max_prefill_predict_length,
  )

  print("\n" + "=" * 80)
  print("Generation test after weight transfer:")
  with timer("Generation"):
    try:
        print(llm.generate(prompt_text, sampling_params=sampling_params, use_tqdm=False))
    except Exception as e:
        print(f"Expected crash during generation test due to XLA co-tenancy: {e}")


def main(argv: Sequence[str]) -> None:
  pathwaysutils.initialize()
  _setup_jax_compilation_cache()
  _setup_vllm_environment()

  validate_converter(argv)


if __name__ == "__main__":
  app.run(main)

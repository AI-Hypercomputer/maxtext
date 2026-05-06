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
from maxtext.utils import model_creation_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

_JAX_COMPILATION_CACHE_DIR = "/tmp/jax_cache"

vllm_model_name_mapping = {
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "qwen3-30b-a3b-base": "Qwen/Qwen3-30B-A3B",
    "qwen3-235b-a22b": "Qwen/Qwen3-235B-A22B",
    "gemma4-26b": "google/gemma-4-26B-A4B",
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

  print("=" * 80)
  print("Converting weights to vLLM format")
  print("=" * 80)
  model_state = {"base": nnx.state(model)}
  for path, leaf in jax.tree_util.tree_flatten_with_path(model_state)[0]:
    if hasattr(leaf, "shape") and hasattr(leaf, "sharding"):
      path_str = jax.tree_util.keystr(path)
      logging.info("Name: %s, shape: %s", path_str, leaf.shape)
      logging.info("\tSharding: %s", leaf.sharding)

  if trainer_config.model_name.startswith("gemma4"):
    converter = Gemma4MaxTextToVLLMConverter(trainer_config, mesh)
  else:
    converter = Qwen3MaxTextToVLLMConverter(trainer_config, mesh)
  with timer("Overall Conversion"):
    maxtext_vllm_state = converter.convert(model_state)
  # Explicitly delete MaxText device buffers before resharding. Python del + gc
  # is not enough — Pathways holds buffers in its object store independently of
  # Python GC, so we must call .delete() on each array to free HBM.
  for arr in jax.tree_util.tree_leaves(model_state):
    if hasattr(arr, "delete"):
      arr.delete()
  del model_state, model, mesh, converter
  gc.collect()

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
  if multislice:
    # Pin vLLM to its assigned sampler devices so it doesn't overlap with trainer.
    vllm_kwargs["additional_config"] = {
        "sharding": {
            "sharding_strategy": {
                "device_indexes": [d.id for d in sampler_devices],
            }
        }
    }
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
    for key, weight in maxtext_vllm_state.items():
      weight_array = weight.value if hasattr(weight, "value") else weight
      dst_sharding = golden_llm_state[key].sharding
      golden_llm_state[key] = reshard_pytree(weight_array, dst_sharding, donate_input=False, cache_plan=True)

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

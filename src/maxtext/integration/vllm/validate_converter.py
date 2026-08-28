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

  python -m maxtext.integration.vllm.validate_converter \
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
  benchmark_weight_sync=true  Report wall time and HBM for each weight-sync phase,
                              blocking on device work so the numbers reflect
                              execution rather than dispatch. Also runs the reshard
                              before the debug_converter early return, so both arms
                              of the A/B cover convert + reshard + assign. Combine
                              with debug_converter=true to benchmark without paying
                              for generation:

                                # baseline
                                … use_weight_converter=false benchmark_weight_sync=true \
                                  debug_converter=true
                                # new converter, same model/checkpoint/command
                                … use_weight_converter=true  benchmark_weight_sync=true \
                                  debug_converter=true

                              Leave debug=false on the converter: its per-group
                              barrier serializes the sync and distorts timing.

Which conversion path runs is selected by config, and both WeightConverter modes
are covered:

  Mode 1 -- direct MaxText-to-MaxText (`WeightConverter(rules=None)`).
    Selected when `vllm_hf_overrides` names MaxTextForCausalLM (so vLLM runs the
    MaxText model) *and* `use_weight_converter=true`. Structural conversion only:
    scanned decoder layers are unrolled and MoE `wi_0`/`wi_1` are fused into the
    rollout's pre-fused `wi`. This is the path for qwen3.5-*.

      python -m maxtext.integration.vllm.validate_converter \
          src/maxtext/configs/post_train/rl.yml model_name=qwen3.5-35b-a3b \
          use_weight_converter=true debug_converter=true \
          vllm_hf_overrides='{"architectures":["MaxTextForCausalLM"]}' \
          vllm_additional_config='{"maxtext_config":{"model_name":"qwen3.5-35b-a3b",
              "model_call_mode":"inference","prefuse_moe_weights":true}}' \
          <plus the common flags above>

    Setting `use_weight_converter=false` with the same flags runs the legacy
    tunix `transfer_state_directly()` instead, for A/B comparison.

  Mode 2 -- MaxText-to-HuggingFace via torchax rules (`WeightConverter(rules=[...])`).
    Selected when vLLM runs its own HF-shaped model. Weights are renamed and
    restructured per `MODEL_TO_CONVERSION_RULES` (QKV fusion with GQA interleave,
    MoE gate+up fusion into `w13_weight`, norm/lm-head transposes).

      python -m maxtext.integration.vllm.validate_converter \
          src/maxtext/configs/post_train/rl.yml model_name=qwen3-30b-a3b \
          use_weight_converter=true debug_converter=true \
          <plus the common flags above>

Currently this validator supports: qwen3-30b-a3b, qwen3-30b-a3b-base, qwen3-235b-a22b, gemma4-26b.
"""

import ast
import collections
import contextlib
import gc
import io
import json
import logging
import os
import tempfile
import time
from typing import Sequence

from absl import app
import jax
import jax.numpy as jnp
from flax import nnx
from flax import traverse_util
import numpy as np
import transformers
import tunix.generate.utils as tunix_utils
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
from maxtext.integration.vllm.weight_converter import WeightConverter, MODEL_TO_CONVERSION_RULES
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
# tpu_inference / tunix compat shims
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _tpu_inference_compat_patches():
  """Local copy of the shims `train_rl.py` wraps its run in.

  Deliberately duplicated rather than imported: `train_rl.py` is the production
  entry point and pulls in the whole RL stack (datasets, grain, orbax, the GRPO
  learner), none of which this validator needs. **Keep in sync with
  `train_rl._tpu_inference_compat_patches`** -- if that one changes, this one
  must follow, or the benchmark stops reflecting production.

  Why the validator needs them at all: the legacy `transfer_state_directly`
  arm does not merely run *better* with these patches, it does not run without
  them. `transfer_state_directly` has no filter for non-Param nnx variables, so
  it feeds scanned scalar RNG counters (`...rngs.params.count`, source shape
  `(num_layers,)`, per-layer shape `()`) through the scanned-layer unroll,
  where `_bulk_align_and_unstack` indexes `arr.shape[scan_axis]` past the end
  of a rank-1 shape and raises IndexError. `_compat_bulk` clamps `scan_axis`.
  (`MaxTextToMaxTextConverter` drops those leaves up front, which is why it
  needs no equivalent -- so benchmarking the legacy arm unpatched would be
  benchmarking a configuration production never runs.)

  See `train_rl.py` for the rationale behind the other two shims
  (`with_sharding_constraint` fallback, and skipping the bf16->f32 upcast).
  """
  orig_wsc = jax.lax.with_sharding_constraint
  orig_apply_dtype_cast = tunix_utils._apply_dtype_cast  # pylint: disable=protected-access
  orig_bulk = tunix_utils._bulk_align_and_unstack  # pylint: disable=protected-access
  orig_unstack = tunix_utils._unstack_scanned_param  # pylint: disable=protected-access

  def _compat_wsc(x, shardings):
    try:
      return orig_wsc(x, shardings)
    except AssertionError:
      return jax.sharding.reshard(x, shardings)

  def _no_bf16_to_f32_cast(val, tgt_dtype, src_key):
    if hasattr(val, "dtype") and val.dtype == jnp.bfloat16 and tgt_dtype == jnp.float32:
      return val
    return orig_apply_dtype_cast(val, tgt_dtype, src_key)

  def _compat_bulk(arr, scan_axis, per_layer, key_path):
    if hasattr(arr, "shape") and len(arr.shape) <= scan_axis:
      scan_axis = len(arr.shape) - 1 if len(arr.shape) > 0 else 0
    return orig_bulk(arr, scan_axis, per_layer, key_path)

  def _compat_unstack(src_val, tgt_val, key_path, scan_axis=None):
    if scan_axis is not None and hasattr(src_val, "shape") and len(src_val.shape) <= scan_axis:
      scan_axis = len(src_val.shape) - 1 if len(src_val.shape) > 0 else 0
    res = orig_unstack(src_val, tgt_val, key_path, scan_axis=scan_axis)
    if isinstance(res, tuple) and len(res) == 1 and hasattr(src_val, "shape") and src_val.shape == tgt_val.shape:
      return res * 256
    return res

  jax.lax.with_sharding_constraint = _compat_wsc
  tunix_utils._apply_dtype_cast = _no_bf16_to_f32_cast  # pylint: disable=protected-access
  tunix_utils._bulk_align_and_unstack = _compat_bulk  # pylint: disable=protected-access
  tunix_utils._unstack_scanned_param = _compat_unstack  # pylint: disable=protected-access
  try:
    yield
  finally:
    jax.lax.with_sharding_constraint = orig_wsc
    tunix_utils._apply_dtype_cast = orig_apply_dtype_cast  # pylint: disable=protected-access
    tunix_utils._bulk_align_and_unstack = orig_bulk  # pylint: disable=protected-access
    tunix_utils._unstack_scanned_param = orig_unstack  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# Weight-sync benchmarking
# ---------------------------------------------------------------------------
#
# `timer` alone is not enough to compare the two sync paths. JAX dispatch is
# asynchronous, so a bare wall clock around `converter.convert()` measures how
# long it took to *enqueue* the conversion, not to run it -- and enqueue time is
# exactly what dispatch-count optimizations shrink. An un-blocked timer would
# therefore report a large speedup whether or not one actually occurred.
#
# It also matters that the two arms cover the same work. `transfer_state_directly`
# converts *and* reshards *and* assigns in a single call, whereas the converter
# path splits conversion from the reshard. Timing `convert()` against
# `transfer_state_directly()` compares two different amounts of work; the phases
# below are reported separately and summed so the totals are like-for-like.


def _hbm_snapshot():
  """Per-device (in_use, peak) bytes, or None if the backend won't report it."""
  snapshot = {}
  for device in jax.local_devices():
    try:
      stats = device.memory_stats()
    except Exception:  # pylint: disable=broad-except
      stats = None
    if not stats:
      return None
    snapshot[device.id] = (
        stats.get("bytes_in_use", 0),
        stats.get("peak_bytes_in_use", 0),
    )
  return snapshot


def _gib(n) -> float:
  return n / (1024**3)


class _SyncPhase:
  """Accumulates wall time and HBM for one phase of a weight sync."""

  totals = {}

  def __init__(self, label):
    self.label = label
    self._start = None
    self._before = None

  def block_on(self, tree):
    """Waits for `tree`'s device work, so the timer measures execution.

    Safe to call with a tree whose buffers were donated or deleted -- that is a
    reporting problem, not a reason to fail the run.
    """
    try:
      leaves = [leaf for leaf in jax.tree_util.tree_leaves(tree) if hasattr(leaf, "block_until_ready")]
      jax.block_until_ready(leaves)
    except Exception as exc:  # pylint: disable=broad-except
      logging.warning(
          "Could not block on '%s' outputs (%s); its time excludes any work " "still in flight.",
          self.label,
          exc,
      )

  def __enter__(self):
    self._before = _hbm_snapshot()
    self._start = time.perf_counter()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if exc_type is not None:
      # The phase did not complete. Reporting a duration here would print a
      # normal-looking timing line immediately above the traceback and invite
      # someone to record it as a result.
      print(
          f"[weight-sync] {self.label}: FAILED after " f"{time.perf_counter() - self._start:.4f} s (no measurement)",
          flush=True,
      )
      return False

    elapsed = time.perf_counter() - self._start
    after = _hbm_snapshot()
    _SyncPhase.totals[self.label] = elapsed

    line = f"[weight-sync] {self.label}: {elapsed:.4f} s"
    if self._before and after:
      # `peak_bytes_in_use` is a high-water mark that XLA never resets, so it is
      # meaningful only as an absolute figure at a fixed point in the run. That
      # still makes it a valid A/B statistic here: everything preceding the sync
      # (model load, vLLM boot) is identical across the two arms.
      in_use = max(v[0] for v in after.values())
      peak = max(v[1] for v in after.values())
      delta = max(after[d][0] - self._before.get(d, (0, 0))[0] for d in after)
      line += f" | HBM in_use {_gib(in_use):.2f} GiB " f"(delta {_gib(delta):+.2f}) | peak {_gib(peak):.2f} GiB"
    else:
      line += " | HBM stats unavailable on this backend"
    print(line, flush=True)
    logging.info(line)
    return False

  @classmethod
  def report(cls):
    if not cls.totals:
      return
    print("=" * 80, flush=True)
    total = sum(cls.totals.values())
    for label, seconds in cls.totals.items():
      print(f"[weight-sync] {label:<52} {seconds:8.4f} s", flush=True)
    print(f"[weight-sync] {'TOTAL':<52} {total:8.4f} s", flush=True)
    print("=" * 80, flush=True)


def _reshard_and_assign_converted(maxtext_vllm_state, golden_llm_state, llm):
  """Reshards a converted state onto the vLLM runner and assigns it.

  Extracted so the benchmark path can run (and time) the reshard without also
  running generation -- previously this lived inline *after* the
  `debug_converter` early return, so the phase it timed was unreachable in
  exactly the configuration used for A/B comparison.
  """
  if isinstance(golden_llm_state, nnx.State):
    state_dict = golden_llm_state.to_pure_dict() if hasattr(golden_llm_state, "to_pure_dict") else dict(golden_llm_state)
  else:
    state_dict = golden_llm_state

  src_flat = traverse_util.flatten_dict(maxtext_vllm_state)
  spec_flat = traverse_util.flatten_dict(state_dict)

  resharded_flat = tunix_utils._reshard_in_chunks(  # pylint: disable=protected-access
      src_flat,
      spec_flat,
      reshard_pytree,
      chunk_size=128,
      delete_spec_buffers=True,
  )
  resharded_weights = traverse_util.unflatten_dict(resharded_flat)

  if isinstance(golden_llm_state, nnx.State):
    nnx.update(golden_llm_state, resharded_weights)
  elif hasattr(golden_llm_state, "update"):
    golden_llm_state.update(resharded_weights)
  elif isinstance(golden_llm_state, dict):
    golden_llm_state.update(resharded_weights)
  else:
    llm.llm_engine.model_executor.driver_worker.model_runner.model.state = resharded_weights
  return golden_llm_state


def _sync_model_runner_state(llm, golden_llm_state) -> None:
  """Syncs the updated golden_llm_state into the model runner's model and leaves."""
  model_runner = getattr(
      getattr(
          getattr(getattr(llm, "llm_engine", None), "model_executor", None),
          "driver_worker",
          None,
      ),
      "model_runner",
      None,
  )
  if model_runner is None:
    return
  if hasattr(model_runner, "model") and isinstance(golden_llm_state, nnx.State):
    nnx.update(model_runner.model, golden_llm_state)
  if hasattr(model_runner, "state"):
    if isinstance(model_runner.state, nnx.State):
      model_runner.state_leaves = tuple(jax.tree_util.tree_leaves(model_runner.state))
    else:
      model_runner.state_leaves = model_runner.state
    logging.info("Updated model_runner.state_leaves after weight assignment.")


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


def _build_weight_converter(model_name: str, direct: bool, config, tp: int) -> WeightConverter:
  """Builds a WeightConverter in whichever of its two modes applies.

  Mode 1 (`direct=True`, `rules=None`): vLLM runs `MaxTextForCausalLM`, so the
  conversion is structural only -- unroll the scanned decoder layers and fuse
  MoE `wi_0`/`wi_1` into the rollout's pre-fused `wi`.

  Mode 2 (`direct=False`, rule table): vLLM runs its own HuggingFace-shaped
  model, so weights are renamed and restructured per `MODEL_TO_CONVERSION_RULES`.
  """
  if direct:
    logging.info("WeightConverter mode 1: direct MaxText-to-MaxText (rules=None).")
    print("WeightConverter mode 1: direct MaxText-to-MaxText (rules=None).", flush=True)
    return WeightConverter(rules=None, config=config, tp=tp)

  rules = MODEL_TO_CONVERSION_RULES.get(model_name, MODEL_TO_CONVERSION_RULES["qwen3_moe"])
  if rules is None:
    raise ValueError(
        f"{model_name} has no HuggingFace-target conversion rules, so mode 2 "
        "cannot be validated for it. Re-run with vllm_hf_overrides selecting "
        "MaxTextForCausalLM to validate the direct path (mode 1) instead."
    )
  logging.info("WeightConverter mode 2: torchax rules (%d rules, tp=%d).", len(rules), tp)
  print(f"WeightConverter mode 2: torchax rules ({len(rules)} rules, tp={tp}).", flush=True)
  return WeightConverter(rules=rules, tp=tp)


class ConverterValidationConfig(types.RLConfig):
  """Configuration dataclass for converter validation and benchmarking."""

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
  benchmark_weight_sync: bool = False
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
  benchmark_weight_sync = getattr(trainer_config, "benchmark_weight_sync", False)

  if len(trainer_devices) > sampler_config.rollout_tensor_parallelism:
    target_dev_count = sampler_config.rollout_tensor_parallelism
    # Group devices by host / task so subslice bounds align with host bounds (e.g. 2,2,1)
    by_host = collections.defaultdict(list)
    for d in trainer_devices:
      task = getattr(d, "logical_task", getattr(d, "task_id", getattr(d, "host_id", 0)))
      by_host[task].append(d)

    selected_devices = []
    for host_devs in by_host.values():
      selected_devices.extend(host_devs)
      if len(selected_devices) >= target_dev_count:
        break
    trainer_devices = selected_devices[:target_dev_count]
    sampler_devices = selected_devices[:target_dev_count]
    logging.info(
        "Clipping devices to rollout_tensor_parallelism=%d on host %s: %s",
        target_dev_count,
        getattr(trainer_devices[0], "logical_task", "unknown"),
        trainer_devices,
    )

  multislice = trainer_devices is not sampler_devices

  logging.info("Creating MaxText model with %d devices...", len(trainer_devices))
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

  print("=" * 80)
  print(f"Loading vLLM model (load_format={vllm_load_format})...")
  print("=" * 80)
  # load_format="dummy" skips loading real weights — converted MaxText weights
  # are assigned afterwards.  Pass vllm_load_format=auto to load an HF checkpoint
  # for reference stats comparison before assignment.
  dp_size = (
      sampler_config.rollout_data_parallelism
      if sampler_config.rollout_data_parallelism > 0
      else max(1, len(sampler_devices) // sampler_config.rollout_tensor_parallelism)
  )
  vllm_kwargs = {
      "model": getattr(trainer_config, "vllm_model_path", None) or vllm_model_name_mapping[trainer_config.model_name],
      "max_model_len": trainer_config.max_target_length,
      "load_format": vllm_load_format,
      "data_parallel_size": dp_size,
      "tensor_parallel_size": sampler_config.rollout_tensor_parallelism,
      "gpu_memory_utilization": 0.55,
      "num_gpu_blocks_override": 512,
      "async_scheduling": getattr(sampler_config, "async_scheduling", False),
  }
  vllm_hf_overrides = getattr(trainer_config, "vllm_hf_overrides", None) or getattr(
      getattr(trainer_config, "vllm", None), "vllm_hf_overrides", None
  )
  if vllm_hf_overrides:
    if isinstance(vllm_hf_overrides, str):
      vllm_kwargs["hf_overrides"] = ast.literal_eval(vllm_hf_overrides)
    else:
      vllm_kwargs["hf_overrides"] = vllm_hf_overrides
  # Conditionally add max_num_batched_tokens only for qwen3.5
  if trainer_config.model_name == "qwen3.5-35b-a3b":
    vllm_kwargs["max_num_batched_tokens"] = 16384

  additional_config = {}
  vllm_additional_config = getattr(trainer_config, "vllm_additional_config", None) or getattr(
      getattr(trainer_config, "vllm", None), "vllm_additional_config", None
  )
  if vllm_additional_config:
    vconfig = vllm_additional_config
    if isinstance(vconfig, str):
      try:
        additional_config.update(json.loads(vconfig))
      except ValueError:
        # Shell-quoted configs often arrive as a Python repr rather than JSON.
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

  vllm_hf_overrides = (
      getattr(trainer_config, "vllm_hf_overrides", None)
      or getattr(getattr(trainer_config, "vllm", None), "vllm_hf_overrides", None)
      or ""
  )
  force_maxtext = "MaxTextForCausalLM" in str(vllm_hf_overrides)
  use_weight_converter = additional_config.get("use_weight_converter", False) or getattr(
      trainer_config, "use_weight_converter", False
  )

  if getattr(trainer_config, "use_standalone_converter", False) or getattr(
      getattr(trainer_config, "vllm", None), "use_standalone_converter", False
  ):
    if trainer_config.model_name.startswith("gemma4"):
      converter = Gemma4MaxTextToVLLMConverter(trainer_config, mesh)
    elif trainer_config.model_name.startswith("qwen3.5"):
      converter = Qwen35MaxTextToVLLMConverter(trainer_config, mesh)
    else:
      converter = Qwen3MaxTextToVLLMConverter(trainer_config, mesh)
    with timer("Overall Conversion"):
      maxtext_vllm_state = converter.convert(model_state)
    del model_state, model, mesh, converter
  elif force_maxtext and not use_weight_converter:
    # Legacy Direct Sync path: transfer_state_directly from tunix
    logging.info(
        "Branch 2 (Direct Sync): Testing fallback tunix transfer_state_directly() (use_weight_converter is False)."
    )
    # transfer_state_directly converts, reshards and assigns in one call, so
    # this single phase is what the converter arm's convert + reshard phases
    # sum to. Reported under one label to make that correspondence explicit.
    with _SyncPhase("tunix transfer_state_directly (convert+reshard+assign)") as phase:
      tunix_utils.transfer_state_directly(
          src_state=model_state,
          dst_state=golden_llm_state,
          reshard_fn=reshard_pytree,
          delete_dst_buffers=True,
          reshard_chunk_size=128,
      )
      phase.block_on(golden_llm_state)
    del model_state, model, mesh
    maxtext_vllm_state = None
  else:
    # New WeightConverter pipeline. Both of its modes are exercised here:
    #   force_maxtext  -> mode 1, rules=None, direct MaxText-to-MaxText
    #   otherwise      -> mode 2, rule table, MaxText-to-HuggingFace (torchax)
    converter = _build_weight_converter(
        model_name=trainer_config.model_name,
        direct=force_maxtext,
        config=trainer_config,
        tp=sampler_config.rollout_tensor_parallelism,
    )
    with _SyncPhase("WeightConverter.convert (conversion only)") as phase:
      maxtext_vllm_state = converter.convert(model_state, target_state=golden_llm_state)
      phase.block_on(maxtext_vllm_state)
    del model_state, model, mesh, converter

  gc.collect()
  jax.clear_caches()

  # --- Debug checks (key coverage, weight stats, GCS upload) ---------------
  if debug_converter and maxtext_vllm_state is not None:
    print("=" * 80)
    print("Checking key coverage and shapes...")
    print("=" * 80)
    _check_key_coverage(golden_llm_state, maxtext_vllm_state)

    compare_stats = vllm_load_format != "dummy"
    _log_weight_stats(maxtext_vllm_state, golden_llm_state, compare=compare_stats)

    if gcs_debug_path:
      with timer("GCS tensor upload"):
        _upload_tensors_to_gcs(maxtext_vllm_state, gcs_debug_path)

  # --- Reshard (benchmark path) --------------------------------------------
  # Run and time the reshard *before* the debug_converter return, so the two
  # arms are compared over the same work. Without this, `debug_converter=true`
  # -- the natural A/B configuration -- times conversion alone in the converter
  # arm against convert+reshard+assign in the tunix arm.
  resharded_already = False
  if benchmark_weight_sync and use_weight_converter and maxtext_vllm_state is not None:
    with _SyncPhase("WeightConverter reshard+assign") as phase:
      _reshard_and_assign_converted(maxtext_vllm_state, golden_llm_state, llm)
      phase.block_on(golden_llm_state)
    resharded_already = True

  if benchmark_weight_sync:
    _SyncPhase.report()

  if debug_converter:
    # Documented behaviour: stop after the conversion checks. This makes the
    # converter testable on models whose *inference* path is still being fixed
    # elsewhere -- conversion correctness is pure weight math and does not
    # depend on decode working.
    print("debug_converter=true: conversion checks complete, skipping generation.", flush=True)
    logging.info("debug_converter=true: skipping weight assignment and generation.")
    return

  # --- Weight assignment ----------------------------------------------------
  if force_maxtext:
    if use_weight_converter and maxtext_vllm_state is not None and not resharded_already:
      with timer("Resharding and assigning converted weights to vLLM model"):
        _reshard_and_assign_converted(maxtext_vllm_state, golden_llm_state, llm)

    _sync_model_runner_state(llm, golden_llm_state)

    num_assigned = len(jax.tree_util.tree_leaves(golden_llm_state))
    logging.info("ASSIGNMENT COMPLETE: Assigned %d weights, Skipped 0 weights", num_assigned)
    print(f"ASSIGNMENT COMPLETE: Assigned {num_assigned} weights, Skipped 0 weights", flush=True)
    if hasattr(llm, "reset_prefix_cache") and callable(llm.reset_prefix_cache):
      llm.reset_prefix_cache()
  else:
    if isinstance(maxtext_vllm_state, dict) and any(isinstance(v, dict) for v in maxtext_vllm_state.values()):
      maxtext_vllm_state = {
          ".".join(str(k) for k in key): v for key, v in traverse_util.flatten_dict(maxtext_vllm_state).items()
      }

    with timer(f"Assigning {len(maxtext_vllm_state)} weights to vLLM model"):
      # MaxText native (and some legacy) models unroll the scan_layers when vLLM explicitly asks for scan_layers=False.
      # Our WeightConverter might output a single tensor with axis [48, ...] under '.layers.'.
      # We must unroll it so it maps linearly to golden_llm_state's 'layers_0', 'layers_1'.
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
        if (
            search_key.startswith("vllm_model.")
            and search_key not in golden_llm_state
            and getattr(golden_llm_state, "__class__", type).__name__ != "State"
        ):
          search_key = search_key[len("vllm_model.") :]
        if "model" in golden_llm_state and not search_key.startswith("model."):
          search_key = f"model.{search_key}"
        elif "model" not in golden_llm_state and search_key.startswith("model."):
          search_key = search_key[len("model.") :]

        if (
            search_key not in golden_llm_state
            and ".experts." in search_key
            and ".experts.routed_experts." not in search_key
        ):
          alt_key = search_key.replace(".experts.", ".experts.routed_experts.", 1)
          if alt_key in golden_llm_state:
            search_key = alt_key

        if search_key in golden_llm_state:
          target_obj = golden_llm_state[search_key]

          # Match shape dynamically (vLLM TPU uses [in, out] but HF converter outputs [out, in])
          target_shape = (
              target_obj.shape
              if hasattr(target_obj, "shape")
              else getattr(getattr(target_obj, "value", target_obj), "shape", None)
          )
          if target_shape and weight_array.shape != target_shape:
            if weight_array.shape[::-1] == target_shape:
              weight_array = weight_array.T
            elif (
                len(weight_array.shape) == 3
                and weight_array.shape[0] == target_shape[0]
                and weight_array.shape[1] == target_shape[2]
                and weight_array.shape[2] == target_shape[1]
            ):
              weight_array = jnp.transpose(weight_array, (0, 2, 1))
            else:
              logging.warning("Shape mismatch for %s: expected %s, got %s", search_key, target_shape, weight_array.shape)

          # Extract sharding safely
          dst_sharding = (
              target_obj.sharding
              if hasattr(target_obj, "sharding")
              else getattr(getattr(target_obj, "value", target_obj), "sharding", None)
          )
          if dst_sharding and getattr(weight_array, "sharding", None) != dst_sharding:
            resharded_val = reshard_pytree(weight_array, dst_sharding, donate_input=False, cache_plan=True)
          else:
            resharded_val = weight_array
          if hasattr(golden_llm_state, "__setitem__"):
            golden_llm_state[search_key] = resharded_val
          else:
            setattr(golden_llm_state, search_key, resharded_val)
          assigned_count += 1
        elif "." in search_key:
          parts = search_key.split(".")
          if parts[0] not in golden_llm_state:
            skipped_keys.append(f"{search_key} (root '{parts[0]}' not in golden_llm_state)")
            continue
          obj = golden_llm_state
          for p in parts[:-1]:
            p_key = int(p) if p.isdigit() else p
            try:
              if hasattr(obj, "__getitem__"):
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
          target_shape = (
              target_obj.shape
              if hasattr(target_obj, "shape")
              else getattr(getattr(target_obj, "value", target_obj), "shape", None)
          )
          if target_shape and weight_array.shape != target_shape:
            if weight_array.shape[::-1] == target_shape:
              weight_array = weight_array.T
            elif (
                len(weight_array.shape) == 3
                and weight_array.shape[0] == target_shape[0]
                and weight_array.shape[1] == target_shape[2]
                and weight_array.shape[2] == target_shape[1]
            ):
              weight_array = jnp.transpose(weight_array, (0, 2, 1))
            elif len(weight_array.shape) == 3 and len(target_shape) == 3:
              if (
                  weight_array.shape[0] == target_shape[0]
                  and weight_array.shape[2] == target_shape[2]
                  and target_shape[1] % weight_array.shape[1] == 0
              ):
                weight_array = jnp.repeat(weight_array, target_shape[1] // weight_array.shape[1], axis=1)
              elif (
                  weight_array.shape[0] == target_shape[0]
                  and weight_array.shape[1] == target_shape[1]
                  and target_shape[2] > weight_array.shape[2]
              ):
                tp = 4
                chunk_size = weight_array.shape[2] // (tp * 2)
                arr = weight_array.reshape(weight_array.shape[0], weight_array.shape[1], tp, 2, chunk_size)
                target_chunk_size = target_shape[2] // (tp * 2)
                pad_amount = target_chunk_size - chunk_size
                arr_pad = jnp.pad(arr, ((0, 0), (0, 0), (0, 0), (0, 0), (0, pad_amount)))
                weight_array = arr_pad.reshape(target_shape)
              elif (
                  weight_array.shape[0] == target_shape[0]
                  and weight_array.shape[2] == target_shape[2]
                  and target_shape[1] > weight_array.shape[1]
              ):
                pad_amount = target_shape[1] - weight_array.shape[1]
                weight_array = jnp.pad(weight_array, ((0, 0), (0, pad_amount), (0, 0)))
              else:
                logging.warning(
                    "Shape mismatch for %s: expected %s, got %s", search_key, target_shape, weight_array.shape
                )
            else:
              logging.warning("Shape mismatch for %s: expected %s, got %s", search_key, target_shape, weight_array.shape)

          dst_sharding = (
              target_obj.sharding
              if hasattr(target_obj, "sharding")
              else getattr(getattr(target_obj, "value", target_obj), "sharding", None)
          )
          if dst_sharding and getattr(weight_array, "sharding", None) != dst_sharding:
            resharded_val = reshard_pytree(weight_array, dst_sharding, donate_input=False, cache_plan=True)
          else:
            resharded_val = weight_array
          if hasattr(obj, "__setitem__"):
            obj[last_p] = resharded_val
          else:
            setattr(obj, str(last_p), resharded_val)
          assigned_count += 1
        else:
          skipped_keys.append(f"{search_key} (no match)")

      logging.info("ASSIGNMENT COMPLETE: Assigned %d weights, Skipped %d weights", assigned_count, len(skipped_keys))
      print(f"ASSIGNMENT COMPLETE: Assigned {assigned_count} weights, Skipped {len(skipped_keys)} weights")
      if skipped_keys:
        for sk in skipped_keys[:15]:
          logging.warning("SKIPPED WEIGHT: %s", sk)
          print(f"SKIPPED WEIGHT: {sk}")
        print(
            "ALL KEYS IN GOLDEN_LLM_STATE CONTAINING MLP:",
            [k for k in (golden_llm_state.keys() if hasattr(golden_llm_state, "keys") else []) if "mlp" in str(k)],
        )

      _sync_model_runner_state(llm, golden_llm_state)

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
  print("validate_converter completed successfully", flush=True)


def main(argv: Sequence[str]) -> None:
  pathwaysutils.initialize()
  print(f"JAX devices: {jax.devices()}")
  _setup_jax_compilation_cache()
  _setup_vllm_environment()
  _clean_device_memory()

  # Applied at the same scope train_rl.py applies them, so both arms of the
  # A/B run under the environment production actually uses.
  with _tpu_inference_compat_patches():
    validate_converter(argv)


if __name__ == "__main__":
  app.run(main)

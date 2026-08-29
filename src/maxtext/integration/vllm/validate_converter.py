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

    Setting `use_weight_converter=false use_standalone_converter=true` with the
    same flags instead runs the pre-WeightConverter per-model converter
    (`Qwen3MaxTextToVLLMConverter` / `Qwen35MaxTextToVLLMConverter` /
    `Gemma4MaxTextToVLLMConverter` in `torchax_converter/*_moe.py`), for A/B
    comparison. This is the real legacy baseline for mode 2: tunix's generic
    `transfer_state_with_mappings` has no MoE/expert coverage for these models,
    so `use_weight_converter=false` alone still runs the new rule-table
    converter for them.

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
import re
import tempfile
import time
from typing import Optional, Sequence

from absl import app
import jax
import jax.numpy as jnp
from flax import nnx
from flax import traverse_util
import numpy as np
import transformers
import tunix.generate.utils as tunix_utils
from tunix.generate import mappings
from tunix.generate.vllm_sampler import VllmConfig
import pathwaysutils
import maxtext.integration.vllm.maxtext_vllm_adapter as adapter

# Registers MaxTextForCausalLM with tpu_inference's model registry and applies
# patch_kv_cache_manager() (correct FP32 GDN recurrent-state cache dtype --
# see PR #4770, "Fix Qwen 3.5 35B RL gibberish output issue"). vLLM's
# vllm.general_plugins entry point is supposed to call this automatically, but
# both train_rl.py and vllm_decode.py call it explicitly rather than relying
# on that -- this validator must do the same, or vLLM boots without the patch
# and Qwen3.5's GDN recurrent state silently degrades to bf16 across decode
# steps, producing fluent-looking-but-wrong ("gibberish") generation despite
# perfectly correct weights.
adapter.register()

from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
from maxtext.integration.vllm.torchax_converter.base import GREEN
from maxtext.integration.vllm.torchax_converter.base import RESET
from maxtext.integration.vllm.torchax_converter.base import timer
from maxtext.integration.vllm.maxtext_vllm_rollout import (
    MaxTextVllmSampler,
    _create_model_converter,
    prepare_direct_sync_additional_config,
)
from maxtext.integration.vllm.torchax_converter.gemma4_moe import Gemma4MaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.qwen3_moe import Qwen3MaxTextToVLLMConverter
from maxtext.integration.vllm.torchax_converter.qwen35_moe import Qwen35MaxTextToVLLMConverter
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

  try:
    from tpu_inference.runner import kv_cache as tpu_kv_cache  # pylint: disable=import-outside-toplevel
    orig_get_kv_cache_shape_with_mesh = tpu_kv_cache.get_kv_cache_shape_with_mesh

    def _compat_kv_cache_shape(*args, **kwargs):
      from tpu_inference.layers.common.sharding import ShardingAxisName  # pylint: disable=import-outside-toplevel
      from tpu_inference import utils as common_utils  # pylint: disable=import-outside-toplevel

      mesh = kwargs.get("mesh", args[0] if len(args) > 0 else None)
      use_mla = kwargs.get("use_mla", args[6] if len(args) > 6 else False)
      kv_heads = kwargs.get("actual_num_kv_heads", args[3] if len(args) > 3 else None)

      if mesh is not None and kv_heads is not None and not use_mla:
        tp_axis_name = getattr(ShardingAxisName, "KV_HEAD", ShardingAxisName.ATTN_HEAD)
        model_cnt = common_utils.get_mesh_shape_product(mesh, tp_axis_name)
        if not model_cnt:
          tp_axis_name = ShardingAxisName.ATTN_HEAD
          model_cnt = common_utils.get_mesh_shape_product(mesh, tp_axis_name)
        if model_cnt and model_cnt > 0 and kv_heads % model_cnt != 0:
          padded_kv_heads = common_utils.get_padded_num_heads(kv_heads, model_cnt)
          if "actual_num_kv_heads" in kwargs:
            kwargs["actual_num_kv_heads"] = padded_kv_heads
          elif len(args) > 3:
            args = list(args)
            args[3] = padded_kv_heads

      return orig_get_kv_cache_shape_with_mesh(*args, **kwargs)

    tpu_kv_cache.get_kv_cache_shape_with_mesh = _compat_kv_cache_shape
  except (ImportError, AttributeError):
    tpu_kv_cache = None
    orig_get_kv_cache_shape_with_mesh = None

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
    if orig_get_kv_cache_shape_with_mesh is not None and tpu_kv_cache is not None:
      tpu_kv_cache.get_kv_cache_shape_with_mesh = orig_get_kv_cache_shape_with_mesh


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


# ---------------------------------------------------------------------------
# Debugging helpers
# ---------------------------------------------------------------------------


def _flatten_weight_dict(state) -> dict:
  """Flattens a weight tree into one flat, dotted-string-keyed dict of arrays.

  The debug helpers below (`_check_key_coverage`, `_log_weight_stats`,
  `_upload_tensors_to_gcs`) were written assuming a target already shaped
  like mode 2's flat HF dict (`.keys()` gives real parameter names directly).
  Mode 1's target is a nested `nnx.State` (`.keys()` gives one top-level
  attribute name, and indexing into it gives a sub-State, not a leaf array --
  the `AttributeError: No attribute 'shape' in State` these helpers hit
  otherwise). Routing both through this first normalizes them to the same
  flat shape regardless of which mode produced them.
  """
  if hasattr(state, "to_pure_dict"):
    state = state.to_pure_dict()
  flat = traverse_util.flatten_dict(state)
  return {".".join(str(k) for k in key): (v.value if hasattr(v, "value") else v) for key, v in flat.items()}


# Matches a layer index in either naming convention seen here: HF-style
# "...layers.3...." (mode 2) or MaxText's own "...layers_3...." (mode 1).
_LAYER_INDEX_RE = re.compile(r"layers[._](\d+)\b")


def _layer_index(key: str) -> Optional[int]:
  m = _LAYER_INDEX_RE.search(key)
  return int(m.group(1)) if m else None


def _is_layer0_key(key: str) -> bool:
  return _layer_index(key) == 0


def _is_non_layer_key(key: str) -> bool:
  return _layer_index(key) is None


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


def _leaf_path_signature(state) -> dict:
  """Maps every leaf's flattened path to (shape, dtype), for structural diffing.

  Paths are rendered manually rather than via `jax.tree_util.keystr`: some
  leaves (notably ones grafted in by a structurally-mismatched `nnx.update`)
  have a path whose single key is itself a raw Python string rather than a
  proper DictKey/GetAttrKey/SequenceKey entry -- `keystr` silently iterates a
  bare string character-by-character instead of treating it as one key,
  turning "model.decoder.decoder_norm.scale" into "['m']['o']['d']...". That
  silent mis-rendering would hide exactly the anomaly this function exists to
  surface, so such a key is rendered whole and tagged `#raw` instead.
  """
  flat, _ = jax.tree_util.tree_flatten_with_path(state)
  result = {}
  for path, leaf in flat:
    parts = []
    for p in path:
      if isinstance(p, str):
        parts.append(f"{p}#raw")
        continue
      key = getattr(p, "key", None)
      if key is None:
        key = getattr(p, "name", None)
      if key is None:
        key = getattr(p, "idx", None)
      parts.append(str(key) if key is not None else repr(p))
    result[".".join(parts)] = (getattr(leaf, "shape", None), getattr(leaf, "dtype", None))
  return result


def _check_leaf_structure_unchanged(before: dict, after: dict, label: str) -> None:
  """Warns if sampler.update_params() changed the *structure* of the rollout state.

  update_params() is documented to only overwrite leaf *values* in place --
  the rollout's own pytree shape is fixed the moment vLLM boots it. tpu_inference
  caches that initial pytree structure (`_state_treedef` in
  tpu_inference/models/common/model_loader.py) and reuses it on every forward
  pass; if the leaf count or path set differs after our sync, the model
  runner's cached compiled function fails with an opaque "Too many/few leaves
  for PyTreeDef" error at the *next* forward pass, with no indication of which
  key caused it. This check surfaces that mismatch immediately after sync,
  while the offending key(s) are still visible, instead of leaving it to
  surface (expensively, and without a key name) during generation.
  """
  added = sorted(set(after) - set(before))
  removed = sorted(set(before) - set(after))
  if len(after) == len(before) and not added and not removed:
    logging.info("%s: rollout state leaf structure unchanged (%d leaves).", label, len(after))
    return
  logging.error(
      "%s: rollout state leaf structure CHANGED by sampler.update_params() -- "
      "%d leaves before sync, %d after (%+d). The model runner's cached "
      "compiled function will fail with a PyTreeDef mismatch on the next "
      "forward pass unless this is resolved.",
      label,
      len(before),
      len(after),
      len(after) - len(before),
  )
  if added:
    logging.error("  %d new leaf path(s) after sync, e.g.: %s", len(added), added[:10])
  if removed:
    logging.error("  %d leaf path(s) that disappeared after sync, e.g.: %s", len(removed), removed[:10])
  changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
  if changed:
    logging.error("  %d leaf(ves) with changed (shape, dtype) at the same path, e.g.:", len(changed))
    for k in changed[:10]:
      logging.error("    %s: before=%s after=%s", k, before[k], after[k])


def _find_char_split_corruption(state) -> list:
  """Finds top-level `state` entries whose value is a dotted string iterated character-by-character.

  This is the corruption behind the mode-2 (native vLLM torch model) crash
  where `torch.func.functional_call`'s `swap_tensor` raises "{...} is not an
  instance of torch.Tensor" against a deeply nested single-character dict.

  Deliberately inspects `state` itself (one level, via `.items()`), not
  `jax.tree_util.tree_leaves(state)`: a fully-recursive flatten would walk
  *through* the corrupted dict down to whatever real value sits at its
  bottom, hiding the anomaly. A value is flagged only when it is a plain
  dict whose keys are *all* single characters -- e.g. `{'l': {'l': {'m':
  ...}}}`, which is exactly what `"vllm_model.language_model...weight"`
  looks like after being iterated one character at a time. Real leaves
  (arrays) and mode 1's nested `nnx.State`/`nnx.Variable` values never match
  this shape, so this has no false positives on the working path.
  """
  items = state.items() if hasattr(state, "items") else []
  return [(k, v) for k, v in items if isinstance(v, dict) and v and all(isinstance(c, str) and len(c) == 1 for c in v)]


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

  Weight sync itself is delegated to `MaxTextVllmSampler.update_params`, the
  same entry point train_rl.py's production rollout uses (via
  `MaxTextVllmRollout`). That keeps this validator honest about what actually
  ships: mode 1 (direct MaxText-to-MaxText) exercises tunix's legacy
  `transfer_state_directly` when `use_weight_converter=false` and the new
  `WeightConverter(rules=None)` when `true`; mode 2 (MaxText-to-HuggingFace)
  exercises tunix's legacy `transfer_state_with_mappings` (via
  `to_hf_key_mappings`) or the new `WeightConverter(rules=MODEL_TO_CONVERSION_RULES[...])`,
  selected the same way production selects them (`_create_model_converter`).
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

  # MaxTextVllmSampler needs a tokenizer up front (it wraps it immediately in
  # its constructor), not just for the chat-template branch of the old
  # llm.generate() call, so load it once and reuse it for both.
  tokenizer_path = getattr(trainer_config, "tokenizer_path", None) or vllm_model_name_mapping[trainer_config.model_name]
  try:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=getattr(trainer_config, "hf_access_token", None) or None,
    )
  except Exception as exc:  # pylint: disable=broad-except
    logging.warning("Failed to load tokenizer with token (%s), retrying with token=None...", exc)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, token=None)

  logging.info("Creating MaxText model with %d devices...", len(trainer_devices))
  # wrap_with_tunix_adapter=True gives `model` a "base" root (matching what
  # production's rollout_actor looks like) *and* a to_hf_mappings() method,
  # so the same model_state and the same mapping_config feed every sync path
  # below -- legacy direct, legacy HF-mapping, and both WeightConverter modes.
  model, mesh = model_creation_utils.from_pretrained(
      trainer_config,
      devices=trainer_devices,
      model_mode=MODEL_MODE_AUTOREGRESSIVE,
      wrap_with_tunix_adapter=True,
  )
  print(f"{GREEN}MaxText model loaded successfully{RESET}")
  print(f"Model: {trainer_config.model_name}")
  print(f"Mesh: {mesh}")

  print("=" * 80)
  print("Extracting MaxText weights")
  print("=" * 80)
  model_state = nnx.state(model, nnx.Param)

  for path, leaf in jax.tree_util.tree_flatten_with_path(model_state)[0]:
    if hasattr(leaf, "shape") and hasattr(leaf, "sharding"):
      path_str = jax.tree_util.keystr(path)
      logging.info("Name: %s, shape: %s", path_str, leaf.shape)
      logging.info("\tSharding: %s", leaf.sharding)

  try:
    mapping_config = mappings.MappingConfig.build(
        mapping_obj=getattr(trainer_config, "rollout_mapping_config", None),
        model=model,
        backend="vllm_jax",
    )
  except Exception as exc:  # pylint: disable=broad-except
    # Models without a registered legacy StandaloneVllmWeightMapping entry
    # (e.g. gemma4-26b, whose converter is always the standalone
    # Gemma4MaxTextToVLLMConverter -- never the legacy to_hf_key_mappings
    # path) raise here. That mapping is never consulted for such models, so
    # fall back to an empty one instead of failing the whole run.
    logging.warning(
        "Could not build legacy to_hf mapping_config for %s (%s); the sampler will rely "
        "entirely on its converter for weight sync.",
        trainer_config.model_name,
        exc,
    )
    mapping_config = mappings.MappingConfig()

  del model, mesh
  gc.collect()
  jax.clear_caches()

  # Resolve tensor/data/expert parallelism together (not just tensor/data) so that
  # whatever device budget is left over after tensor_parallel_size (e.g. because TP
  # is capped below the KV-head count) gets spent as expert_parallel_size -- sharding
  # MoE experts across those chips -- instead of silently falling back to plain
  # data_parallel_size, which replicates the entire model (all experts included) once
  # per replica. Reuses the same resolver train_rl.py uses for the production rollout
  # path, so validate_converter.py's parallelism math stays consistent with it.
  rollout_kwargs = model_creation_utils.get_rollout_kwargs_for_parallelism(sampler_config, len(sampler_devices))
  # The rollout mesh is deliberately separate from the trainer mesh above --
  # it spans only sampler_devices, exactly like the mesh MaxTextVllmRollout
  # receives in production. VllmSampler derives tensor/data parallel size,
  # expert_parallelism and the multislice device_indexes pin all from this
  # mesh (see VllmSampler._vllm_config), so no manual "sharding" bookkeeping
  # is needed here anymore.
  sampler_mesh = jax.sharding.Mesh(
      np.array(sampler_devices).reshape(-1, rollout_kwargs["tensor_parallel_size"]),
      ("data", "model"),
  )

  vllm_hf_overrides = getattr(trainer_config, "vllm_hf_overrides", None) or getattr(
      getattr(trainer_config, "vllm", None), "vllm_hf_overrides", None
  )
  # Mode 1 (direct MaxText-to-MaxText) is selected the same way production
  # selects it: vllm_hf_overrides names MaxTextForCausalLM, so vLLM
  # instantiates the MaxText adapter model instead of an HF-shaped one.
  direct_maxtext_sync = "MaxTextForCausalLM" in str(vllm_hf_overrides)

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

  use_weight_converter = additional_config.get("use_weight_converter", False) or getattr(
      trainer_config, "use_weight_converter", False
  )
  use_standalone_converter = getattr(trainer_config, "use_standalone_converter", False) or getattr(
      getattr(trainer_config, "vllm", None), "use_standalone_converter", False
  )

  # For direct MaxText sync with MoE + TP>1, forces the target's prefused
  # `wi` layout so each shard gets its local gate chunk followed by its local
  # up chunk -- required for correctness, and easy to omit by hand in
  # vllm_additional_config, so apply it the same way production does.
  additional_config = prepare_direct_sync_additional_config(
      additional_config,
      direct_maxtext_sync=direct_maxtext_sync,
      num_experts=getattr(trainer_config, "num_experts", 1),
      tensor_parallel_size=rollout_kwargs["tensor_parallel_size"],
  )

  init_with_random_weights = vllm_load_format == "dummy"
  # load_format="dummy" (the default) skips loading real weights -- converted
  # MaxText weights are assigned afterwards via sampler.update_params().  Pass
  # vllm_load_format=auto to load an HF checkpoint instead, for reference
  # stats comparison before assignment.
  vllm_engine_kwargs = {
      "model": getattr(trainer_config, "vllm_model_path", None) or vllm_model_name_mapping[trainer_config.model_name],
      "max_model_len": trainer_config.max_target_length,
      "num_gpu_blocks_override": 512,
      "async_scheduling": getattr(sampler_config, "async_scheduling", False),
  }
  if not init_with_random_weights:
    vllm_engine_kwargs["load_format"] = vllm_load_format
  if vllm_hf_overrides:
    vllm_engine_kwargs["hf_overrides"] = (
        ast.literal_eval(vllm_hf_overrides) if isinstance(vllm_hf_overrides, str) else vllm_hf_overrides
    )
  # Conditionally add max_num_batched_tokens only for qwen3.5
  if trainer_config.model_name == "qwen3.5-35b-a3b":
    vllm_engine_kwargs["max_num_batched_tokens"] = 16384

  vllm_config = VllmConfig(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
      mesh=sampler_mesh,
      hbm_utilization=getattr(trainer_config, "hbm_utilization_vllm", 0.6),
      init_with_random_weights=init_with_random_weights,
      tensor_parallel_size=rollout_kwargs["tensor_parallel_size"],
      data_parallel_size=rollout_kwargs["data_parallel_size"],
      expert_parallel_size=rollout_kwargs["expert_parallel_size"],
      reshard_chunk_size=16,
      mapping_config=mapping_config,
      additional_config=additional_config,
      engine_kwargs=vllm_engine_kwargs,
  )

  # Converter selection:
  #   use_standalone_converter=True  -> mode 2 legacy (pre-WeightConverter
  #     per-model converter: Qwen3MaxTextToVLLMConverter / Qwen35MaxTextToVLLMConverter /
  #     Gemma4MaxTextToVLLMConverter in torchax_converter/*_moe.py). This is the
  #     real "legacy" comparison point for mode 2 -- tunix's generic
  #     transfer_state_with_mappings (legacy to_hf_key_mappings) has no MoE/
  #     expert coverage for these models, so it's not a usable A/B baseline.
  #     These converters emit the same flat "vllm_model.*"-keyed dict either
  #     way, which is exactly what VllmSampler._assign_converted_state expects
  #     for an HF-shaped (flat) target, so they plug into `converter=` directly.
  #   otherwise -> the same helper train_rl.py uses in production, so
  #     mode 1/2 x legacy/new selection here can never drift from what ships:
  #       direct_maxtext_sync=True,  use_weight_converter=True  -> mode 1 new     (WeightConverter(rules=None))
  #       direct_maxtext_sync=True,  use_weight_converter=False -> mode 1 legacy  (tunix transfer_state_directly)
  #       direct_maxtext_sync=False, use_weight_converter=True  -> mode 2 new     (WeightConverter(rules=MODEL_TO_CONVERSION_RULES[...]))
  #       direct_maxtext_sync=False, use_weight_converter=False -> mode 2 "legacy" only in the sense of
  #         falling back toward tunix to_hf_key_mappings; for qwen3-30b-a3b/qwen3-235b-a22b (MoE,
  #         already have a MODEL_TO_CONVERSION_RULES entry) _create_model_converter keeps preferring
  #         the rule-table converter here regardless of this flag, for the reason above -- use
  #         use_standalone_converter=True instead to get an actual legacy mode 2 comparison.
  print("=" * 80)
  print(
      f"Building converter (direct_maxtext_sync={direct_maxtext_sync}, "
      f"use_weight_converter={use_weight_converter}, use_standalone_converter={use_standalone_converter})..."
  )
  print("=" * 80)
  if use_standalone_converter:
    if trainer_config.model_name.startswith("gemma4"):
      converter = Gemma4MaxTextToVLLMConverter(trainer_config, sampler_mesh)
    elif trainer_config.model_name.startswith("qwen3.5"):
      converter = Qwen35MaxTextToVLLMConverter(trainer_config, sampler_mesh)
    else:
      converter = Qwen3MaxTextToVLLMConverter(trainer_config, sampler_mesh)
  else:
    converter = _create_model_converter(
        model_name=trainer_config.model_name,
        config=trainer_config,
        mesh=sampler_mesh,
        use_hf_mapping=not direct_maxtext_sync,
        use_weight_converter=use_weight_converter,
        debug=debug_converter,
    )

  print("=" * 80)
  print(f"Booting vLLM via MaxTextVllmSampler (load_format={vllm_load_format})...")
  print("=" * 80)
  sampler = MaxTextVllmSampler(
      tokenizer=tokenizer,
      config=vllm_config,
      converter=converter,
      direct_maxtext_sync=direct_maxtext_sync,
      scan_axis=getattr(trainer_config, "param_scan_axis", 1),
      layer_pattern_length=getattr(trainer_config, "inhomogeneous_layer_cycle_interval", None),
  )
  golden_llm_state = sampler.transformer_state
  # Captured before any sync so a post-sync structural change (which will
  # otherwise surface only as an opaque PyTreeDef mismatch during generation)
  # can be caught and reported with the actual offending key(s) -- see
  # _check_leaf_structure_unchanged.
  pre_sync_leaf_signature = _leaf_path_signature(golden_llm_state)

  # --- Debug checks (key coverage, weight stats, GCS upload) ---------------
  # Calling converter.convert() directly here (rather than through
  # sampler.update_params()) keeps this a conversion-only cost when
  # benchmark_weight_sync is off -- WeightConverter.convert() is documented
  # pure, so this doesn't disturb the state sampler.update_params() converts
  # again below. Legacy paths (converter is None) have no intermediate dict
  # to inspect, so these checks only apply to the WeightConverter arms.
  maxtext_vllm_state = None
  if debug_converter and converter is not None:
    with _SyncPhase(f"{type(converter).__name__}.convert (conversion only)") as phase:
      maxtext_vllm_state = converter.convert(model_state, target_state=golden_llm_state)
      phase.block_on(maxtext_vllm_state)

    gc.collect()
    jax.clear_caches()

    # Normalize both to a flat dotted-key dict first: golden_llm_state is a
    # nested nnx.State for mode 1 (direct MaxText target) but already a flat
    # HF-shaped dict for mode 2, and maxtext_vllm_state mirrors whichever
    # shape the converter targeted -- these helpers only understand the flat
    # form.
    flat_golden_llm_state = _flatten_weight_dict(golden_llm_state)
    flat_maxtext_vllm_state = _flatten_weight_dict(maxtext_vllm_state)

    print("=" * 80)
    print("Checking key coverage and shapes...")
    print("=" * 80)
    _check_key_coverage(flat_golden_llm_state, flat_maxtext_vllm_state)

    compare_stats = vllm_load_format != "dummy"
    _log_weight_stats(flat_maxtext_vllm_state, flat_golden_llm_state, compare=compare_stats)

    if gcs_debug_path:
      with timer("GCS tensor upload"):
        _upload_tensors_to_gcs(flat_maxtext_vllm_state, gcs_debug_path)

  # --- Weight sync via sampler.update_params() ------------------------------
  # Legacy paths always pay for the full sync here (transfer_state_directly /
  # transfer_state_with_mappings can't be split into convert-only). The
  # WeightConverter arms only pay for it when generation will actually run,
  # or when benchmark_weight_sync explicitly asks to compare convert+reshard+assign
  # against the legacy arm's single all-in-one call.
  run_full_sync = (not debug_converter) or (converter is None) or benchmark_weight_sync
  sync_label = (
      f"sampler.update_params via {type(converter).__name__} (convert+reshard+assign)"
      if converter is not None
      else "sampler.update_params via legacy tunix sync (convert+reshard+assign)"
  )
  if run_full_sync:
    with _SyncPhase(sync_label) as phase:
      sampler.update_params(model_state)
      phase.block_on(sampler.transformer_state)
    _check_leaf_structure_unchanged(
        pre_sync_leaf_signature, _leaf_path_signature(sampler.transformer_state), sync_label
    )
    # Localizes the mode-2 "{...} is not an instance of torch.Tensor" crash:
    # does the corruption already exist in the state tunix just wrote, before
    # generation (and torchax/vLLM) ever touch it?
    corrupted = _find_char_split_corruption(sampler.transformer_state)
    if corrupted:
      logging.error(
          "CORRUPTION FOUND BEFORE GENERATION: %d entries in sampler.transformer_state "
          "already have the char-split pattern (dict keyed by single characters) right "
          "after sampler.update_params() -- the bug is in the weight-sync path (tunix "
          "and/or this converter), not downstream in torchax/vLLM. First key(s): %s",
          len(corrupted),
          [k for k, _ in corrupted][:5],
      )
    else:
      logging.info(
          "Post-sync state clean: no char-split corruption pattern found in %d "
          "top-level entries of sampler.transformer_state right after "
          "sampler.update_params(). If generation still crashes with the nested "
          "single-character dict error, the corruption is introduced downstream "
          "(torchax/vLLM), after tunix hands off a correct state.",
          len(list(sampler.transformer_state.items())) if hasattr(sampler.transformer_state, "items") else -1,
      )

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

  num_synced = len(jax.tree_util.tree_leaves(sampler.transformer_state))
  logging.info("ASSIGNMENT COMPLETE: synced %d weight leaves via sampler.update_params", num_synced)
  print(f"ASSIGNMENT COMPLETE: synced {num_synced} weight leaves via sampler.update_params", flush=True)

  # --- Generation test ------------------------------------------------------
  prompt = getattr(trainer_config, "prompt", "Paris is")
  if getattr(trainer_config, "use_chat_template", False):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
  elif trainer_config.model_name.startswith("gemma4") and not prompt.startswith("<bos>"):
    prompt = "<bos>" + prompt

  max_generation_steps = trainer_config.max_target_length - trainer_config.max_prefill_predict_length
  print("\n" + "=" * 80)
  print("Generation test after weight transfer:")
  with timer("Generation"):
    output = sampler([prompt], max_generation_steps=max_generation_steps, temperature=0.0)
  print(output.text)
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

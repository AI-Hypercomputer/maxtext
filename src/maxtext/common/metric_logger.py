# Copyright 2023–2025 Google LLC
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

# pylint: disable=bare-except, consider-using-generator
# pytype: disable=attribute-error
"""Logger that saves metrics to a local file, GCS and TensorBoard."""

import json
import os
import sys
import queue
import enum
import datetime

import numpy as np

import jax
import jax.numpy as jnp
from maxtext.utils.globals import EPS
from maxtext.common.gcloud_stub import mldiagnostics_modules
from maxtext.common.gcloud_stub import workload_monitor
from maxtext.common.managed_mldiagnostics import ManagedMLDiagnostics
from maxtext.utils import exceptions
from maxtext.utils import mllog_utils
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import elastic_utils
from collections import defaultdict

mldiag, _ = mldiagnostics_modules()
GCPWorkloadMonitor, _monitor_is_stub = workload_monitor()

# Mapping MaxText metrics to managed profiler metrics
_METRICS_TO_MANAGED = {
    "learning/current_learning_rate": "learning_rate",
    "learning/loss": "loss",
    "learning/grad_norm": "gradient_norm",
    "learning/total_weights": "total_weights",
    "perf/step_time_seconds": "step_time",
    "perf/per_device_tokens_per_sec": "throughput",
    "perf/per_device_tflops_per_sec": "tflops",
    # There are no mappings to the following metrics yet:
    # "latency", "mfu"
}


STEP_DIAGNOSTICS_KEYS = (
    ("raw_grad_norm", "learning/raw_grad_norm"),
    ("grad_norm", "learning/grad_norm"),
    ("param_norm", "learning/param_norm"),
    ("max_abs_grad", "diag/max_abs_grad"),
    ("moe_bias_checksum", "diag/moe_bias_checksum"),
    ("moe_bias_update_nonzero", "diag/moe_bias_update_nonzero"),
    ("moe_overflow", "diag/moe_overflow"),
)


def step_diagnostics_text(scalars):
  """The log_step_diagnostics part of the step line: 'diag: raw_grad_norm=... grad_norm=... ...' (%.9e; n/a if absent)."""
  parts = []
  for name, key in STEP_DIAGNOSTICS_KEYS:
    parts.append(f"{name}={float(scalars[key]):.9e}" if key in scalars else f"{name}=n/a")
  return "diag: " + " ".join(parts)


def _prepare_metrics_for_json(metrics, step, run_name):
  """Converts metric dictionary into json supported types (e.g. float)"""
  metrics_dict = {val: float(metrics["scalar"][val]) for val in metrics["scalar"]}
  metrics_dict["step"] = float(step)
  metrics_dict["run_name"] = run_name
  return metrics_dict


def _is_concrete_jax_array(val):
  return isinstance(val, jax.Array) and not isinstance(val, jax.core.Tracer)


def _materialize_metric_scalars(metrics):
  """Transfers device-resident JAX metric scalars to host in a single batch.

  On Pathways and multi-host setups, reading individual metric keys via float()
  or .item() triggers sequential device-to-host RPC round trips. Batching all
  concrete JAX array leaves into one jax.device_get() call issues the transfers
  concurrently so their latencies overlap while preserving dtypes and .item()
  compatibility.

  Args:
    metrics: Metric dictionary containing optional "scalar" and "scalars" dicts.
  """
  if not metrics:
    return
  scalar_targets = []
  jax_values = []

  scalar_dict = metrics.get("scalar")
  if isinstance(scalar_dict, dict):
    for key, val in scalar_dict.items():
      if _is_concrete_jax_array(val):
        scalar_targets.append((scalar_dict, key))
        jax_values.append(val)

  scalars_dict = metrics.get("scalars")
  if isinstance(scalars_dict, dict):
    for group_val in scalars_dict.values():
      if isinstance(group_val, dict):
        for subkey, subval in group_val.items():
          if _is_concrete_jax_array(subval):
            scalar_targets.append((group_val, subkey))
            jax_values.append(subval)

  if not jax_values:
    return

  host_values = jax.device_get(jax_values)
  for (target_dict, key), host_val in zip(scalar_targets, host_values):
    target_dict[key] = host_val


def _natural_path_sort_key(path_keys):
  """Sort key that orders numeric-suffixed path segments (e.g. layers_2 < layers_10) numerically."""
  result = []
  for k in path_keys:
    s = str(k)
    prefix, sep, suffix = s.rpartition("_")
    if sep and suffix.isdigit():
      result.append((prefix + sep, 0, int(suffix), ""))
    else:
      result.append((s, 1, 0, s))
  return tuple(result)


def _collect_per_layer_intermediates(intermediate_outputs, key):
  """Collects per-decoder-layer intermediate scalars for `key` in logical layer order.

  - Ignores leaves whose path contains `"mtp_block"` so decoder indices are not shifted.
  - Detects paired `("local_layers", "global_layer")` leaves from `Qwen3NextScannableBlock` /
    `Qwen3_5ScannableBlock` and interleaves `[local_0, ..., local_{k-1}, global]` per block.
  - Sorts unscanned `layers_<int>` keys numerically instead of lexicographically.
  """
  matched = []
  for path, val in jax.tree_util.tree_leaves_with_path(intermediate_outputs):
    path_keys = tuple(k.key for k in path if hasattr(k, "key"))
    if not path_keys or path_keys[-1] != key:
      continue
    if "mtp_block" in path_keys:
      continue
    matched.append((path_keys, jnp.asarray(val)))

  if not matched:
    return None

  global_entries = {}
  for path_keys, val in matched:
    if "global_layer" in path_keys and "local_layers" not in path_keys:
      idx = path_keys.index("global_layer")
      pair_key = (path_keys[:idx], path_keys[idx + 1 :])
      global_entries[pair_key] = val

  consumed_globals = set()
  entries = []
  for path_keys, val in matched:
    if "global_layer" in path_keys and "local_layers" not in path_keys:
      continue
    if "local_layers" in path_keys:
      idx = path_keys.index("local_layers")
      pair_key = (path_keys[:idx], path_keys[idx + 1 :])
      g_val = global_entries.get(pair_key)
      if g_val is not None and val.ndim >= 1 and g_val.shape == val.shape[:-1]:
        interleaved = jnp.concatenate([val, g_val[..., None]], axis=-1).reshape(-1)
        consumed_globals.add(pair_key)
        entries.append((path_keys[:idx] + path_keys[idx + 1 :], interleaved))
        continue
    entries.append((path_keys, jnp.ravel(val)))

  for path_keys, val in matched:
    if "global_layer" in path_keys and "local_layers" not in path_keys:
      idx = path_keys.index("global_layer")
      pair_key = (path_keys[:idx], path_keys[idx + 1 :])
      if pair_key not in consumed_globals:
        entries.append((path_keys, jnp.ravel(val)))

  entries.sort(key=lambda item: _natural_path_sort_key(item[0]))
  return jnp.concatenate([arr for _, arr in entries])


def record_activation_metrics(output_metrics, intermediate_outputs, config):
  """Adds the activation metrics to the metrics dict.

  Collects each metric by path suffix rather than a hardcoded path, so it works for
  both the Linen ("intermediates"-prefixed) and NNX (model-rooted) layouts and for
  scanned (including hybrid local/global scannable blocks) and unscanned decoders.
  """
  for label, key in (
      ("activ_fraction_zero", "activation_fraction_zero"),
      ("activ_mean", "activation_mean"),
      ("activ_stdev", "activation_stdev"),
      ("activ_amax", "activation_amax"),
      ("activ_mlp_wo_amax", "mlp_wo_input_amax"),
      ("activ_moe_wo_amax", "moe_wo_input_amax"),
  ):
    per_layer = _collect_per_layer_intermediates(intermediate_outputs, key)
    if per_layer is None:
      continue
    for layer_num in range(min(config.num_decoder_layers, per_layer.shape[0])):
      output_metrics["scalar"][f"{label}/layer_{layer_num:03d}"] = per_layer[layer_num]


def _is_nnx_var_leaf(x):
  """Duck-types NNX Variable / VariableState without importing flax."""
  return hasattr(x, "get_metadata") and (hasattr(x, "value") or hasattr(x, "get_value"))


def _path_to_keys(path):
  """Converts a JAX KeyPath into normalized string keys, stripping trailing value/raw_value."""
  keys = []
  for p in path:
    if hasattr(p, "key"):
      keys.append(str(p.key))
    elif hasattr(p, "name"):
      keys.append(str(p.name))
    elif hasattr(p, "idx"):
      keys.append(str(p.idx))
    else:
      keys.append(str(p))
  while keys and keys[-1] in ("value", "raw_value"):
    keys.pop()
  return tuple(keys)


def _extract_param_leaves(tree):
  """Extracts inexact array leaves and any NNX variable metadata keyed by normalized path."""
  if tree is None:
    return {}
  entries = {}
  leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=_is_nnx_var_leaf)
  for path, leaf in leaves_with_path:
    meta = {}
    if _is_nnx_var_leaf(leaf):
      raw_meta = leaf.get_metadata()
      if isinstance(raw_meta, dict):
        meta.update(raw_meta)
      for attr in ("sharding", "out_sharding", "kernel_axes", "sharding_names", "param_scan_axis"):
        if attr not in meta or meta[attr] is None:
          attr_val = getattr(leaf, attr, None)
          if attr_val is not None:
            meta[attr] = attr_val
      val = leaf.get_value() if hasattr(leaf, "get_value") else leaf.value
    else:
      val = leaf
    base_keys = _path_to_keys(path)
    if hasattr(val, "dtype") and hasattr(val, "ndim"):
      if jnp.issubdtype(val.dtype, jnp.inexact):
        entries[base_keys] = (val, meta)
    elif val is not None:
      sub_leaves, _ = jax.tree_util.tree_flatten_with_path(val)
      for sub_path, sub_val in sub_leaves:
        if hasattr(sub_val, "dtype") and hasattr(sub_val, "ndim") and jnp.issubdtype(sub_val.dtype, jnp.inexact):
          sub_keys = _path_to_keys(sub_path)
          entries[base_keys + sub_keys] = (sub_val, meta)
  return entries


def _find_axis_in_meta(meta, target_names, ndim):
  """Returns the array axis index matching any name in `target_names` from leaf sharding metadata."""
  if not meta or ndim <= 0:
    return None
  for meta_key in ("sharding", "kernel_axes", "out_sharding", "sharding_names"):
    axes = meta.get(meta_key)
    if axes is None:
      continue
    if isinstance(axes, str):
      axes = (axes,)
    if not isinstance(axes, (tuple, list)):
      continue
    for idx, ax in enumerate(axes):
      if idx >= ndim:
        break
      if ax in target_names:
        return idx
      if isinstance(ax, (tuple, list)) and any(sub in target_names for sub in ax):
        return idx
  return None


def _resolve_decoder_layer_mapping(path_keys, meta, ndim, config, prefix_local_k):
  """Resolves how a parameter/grad leaf maps onto logical decoder layer indices."""
  if "mtp_block" in path_keys:
    return None

  # Layout 1: unscanned `layers_<int>` in path keys.
  for k in path_keys:
    if k.startswith("layers_") and k[len("layers_") :].isdigit():
      return ("single", int(k[len("layers_") :]))

  scan_axis = meta.get("param_scan_axis", getattr(config, "param_scan_axis", 0))
  outer_names = ("layers", "decoder_layers", "scanned_blocks")

  # Layout 3: hybrid Qwen3NextScannableBlock / Qwen3_5ScannableBlock (`local_layers` + `global_layer`).
  if "local_layers" in path_keys:
    idx = path_keys.index("local_layers")
    prefix = path_keys[:idx]
    outer_axis = _find_axis_in_meta(meta, outer_names, ndim)
    inner_axis = _find_axis_in_meta(meta, ("local_layers",), ndim)
    is_outer_scanned = outer_axis is not None or any(k in ("layers", "scanned_blocks") for k in prefix)
    if is_outer_scanned:
      if outer_axis is None and inner_axis is None and ndim >= 2:
        outer_axis = min(scan_axis, ndim - 2)
        inner_axis = outer_axis + 1
      elif outer_axis is None and inner_axis is not None and ndim >= 2:
        cand = scan_axis if scan_axis < ndim else 0
        outer_axis = cand if cand != inner_axis else (0 if inner_axis != 0 else 1)
      elif inner_axis is None and outer_axis is not None and ndim >= 2:
        inner_axis = outer_axis + 1 if outer_axis + 1 < ndim else (0 if outer_axis != 0 else 1)
      if outer_axis is not None and inner_axis is not None and outer_axis != inner_axis:
        return ("hybrid_local", outer_axis, inner_axis)
    else:
      if inner_axis is None and ndim >= 1:
        inner_axis = scan_axis if scan_axis < ndim else 0
      if inner_axis is not None:
        offset = 0
        if "layers_remainder" in prefix:
          num_layers = getattr(config, "num_decoder_layers", 0)
          cfg_cycle = getattr(config, "inhomogeneous_layer_cycle_interval", 1)
          if cfg_cycle and cfg_cycle > 1 and num_layers > 0:
            offset = (num_layers // cfg_cycle) * cfg_cycle
        return ("scanned_1d", inner_axis, offset)
    return None

  if "global_layer" in path_keys:
    idx = path_keys.index("global_layer")
    prefix = path_keys[:idx]
    outer_axis = _find_axis_in_meta(meta, outer_names, ndim)
    is_outer_scanned = outer_axis is not None or any(k in ("layers", "scanned_blocks") for k in prefix)
    k_val = prefix_local_k.get(prefix)
    cfg_cycle = getattr(config, "inhomogeneous_layer_cycle_interval", None)
    if k_val is not None:
      cycle = int(cfg_cycle) if (cfg_cycle is not None and int(cfg_cycle) >= k_val + 1) else (k_val + 1)
      global_offset = k_val
    elif cfg_cycle is not None and int(cfg_cycle) > 1:
      cycle = int(cfg_cycle)
      global_offset = cycle - 1
    else:
      cycle = 1
      global_offset = 0
    if is_outer_scanned:
      if outer_axis is None and ndim >= 1:
        outer_axis = scan_axis if scan_axis < ndim else 0
      if outer_axis is not None:
        return ("hybrid_global", outer_axis, cycle, global_offset)
    else:
      return ("single", global_offset)
    return None

  # Layout 2: standard scanned decoder.
  layer_axis = _find_axis_in_meta(meta, outer_names, ndim)
  if layer_axis is None and any(k in ("layers", "scanned_blocks") for k in path_keys) and ndim >= 1:
    layer_axis = scan_axis if scan_axis < ndim else 0
  if layer_axis is not None:
    return ("scanned_1d", layer_axis, 0)

  return None


def _reduce_leaf_for_axes(arr, keep_axes, compute_max):
  """Reduces sum-of-squares (and optionally absmax) over all axes except `keep_axes`."""
  if arr.size == 0:
    zero = jnp.float32(0.0)
    if not keep_axes:
      return zero, (zero if compute_max else None)
    shape = tuple(arr.shape[a] for a in keep_axes)
    zeros = jnp.zeros(shape, dtype=jnp.float32)
    return zeros, (zeros if compute_max else None)

  arr_f32 = arr.astype(jnp.float32)
  reduce_axes = tuple(i for i in range(arr_f32.ndim) if i not in keep_axes)
  if reduce_axes:
    sq = jnp.sum(jnp.square(arr_f32), axis=reduce_axes)
    amax = jnp.max(jnp.abs(arr_f32), axis=reduce_axes) if compute_max else None
  else:
    sq = jnp.square(arr_f32)
    amax = jnp.abs(arr_f32) if compute_max else None

  # When keeping 2 axes (outer, inner), transpose if outer > inner so dim 0 is always outer.
  if len(keep_axes) == 2 and keep_axes[0] > keep_axes[1]:
    sq = jnp.swapaxes(sq, 0, 1)
    if amax is not None:
      amax = jnp.swapaxes(amax, 0, 1)
  return sq, amax


def compute_per_module_norms(raw_grads, model_params, config):
  """Computes per-module and per-decoder-layer grad_norm, grad_max, and param_norm float32 scalars."""
  grad_entries = _extract_param_leaves(raw_grads)
  param_entries = _extract_param_leaves(model_params)

  all_paths = list(grad_entries.keys())
  seen = set(all_paths)
  for path_keys in param_entries:
    if path_keys not in seen:
      all_paths.append(path_keys)
      seen.add(path_keys)

  # Pre-pass: discover K (local layers per block) for each hybrid scannable-block prefix.
  prefix_local_k = {}
  for path_keys in all_paths:
    if "local_layers" not in path_keys or "mtp_block" in path_keys:
      continue
    g, g_meta = grad_entries.get(path_keys, (None, {}))
    p, p_meta = param_entries.get(path_keys, (None, {}))
    ref = p if p is not None else g
    meta = {k: v for k, v in g_meta.items() if v is not None}
    meta.update({k: v for k, v in p_meta.items() if v is not None})
    idx = path_keys.index("local_layers")
    prefix = path_keys[:idx]
    inner_axis = _find_axis_in_meta(meta, ("local_layers",), ref.ndim)
    if inner_axis is None:
      outer_axis = _find_axis_in_meta(meta, ("layers", "decoder_layers", "scanned_blocks"), ref.ndim)
      is_outer_scanned = outer_axis is not None or any(k in ("layers", "scanned_blocks") for k in prefix)
      scan_axis = meta.get("param_scan_axis", getattr(config, "param_scan_axis", 0))
      if is_outer_scanned and ref.ndim >= 2:
        outer_ax = outer_axis if outer_axis is not None else min(scan_axis, ref.ndim - 2)
        inner_axis = outer_ax + 1 if outer_ax + 1 < ref.ndim else (0 if outer_ax != 0 else 1)
      elif not is_outer_scanned and ref.ndim >= 1:
        inner_axis = scan_axis if scan_axis < ref.ndim else 0
    if inner_axis is not None and inner_axis < ref.ndim:
      prefix_local_k[prefix] = ref.shape[inner_axis]

  metrics = {}
  layer_grad_sq = defaultdict(list)
  layer_grad_max = defaultdict(list)
  layer_param_sq = defaultdict(list)

  for path_keys in all_paths:
    name = "/".join(path_keys)
    g, g_meta = grad_entries.get(path_keys, (None, {}))
    p, p_meta = param_entries.get(path_keys, (None, {}))
    meta = {k: v for k, v in g_meta.items() if v is not None}
    meta.update({k: v for k, v in p_meta.items() if v is not None})
    ref = p if p is not None else g

    mapping = _resolve_decoder_layer_mapping(path_keys, meta, ref.ndim, config, prefix_local_k)
    if mapping is None or mapping[0] == "single":
      keep_axes = ()
    elif mapping[0] in ("scanned_1d", "hybrid_global"):
      keep_axes = (mapping[1],)
    elif mapping[0] == "hybrid_local":
      keep_axes = (mapping[1], mapping[2])
    else:
      keep_axes = ()

    def _record_layer_slices(sq_arr, amax_arr, target_sq, target_max, leaf_mapping, leaf_shape):
      if leaf_mapping is None:
        return
      kind = leaf_mapping[0]
      if kind == "single":
        layer_idx = leaf_mapping[1]
        target_sq[layer_idx].append(sq_arr)
        if target_max is not None and amax_arr is not None:
          target_max[layer_idx].append(amax_arr)
      elif kind == "scanned_1d":
        axis, offset = leaf_mapping[1], leaf_mapping[2]
        for i in range(leaf_shape[axis]):
          layer_idx = offset + i
          target_sq[layer_idx].append(sq_arr[i])
          if target_max is not None and amax_arr is not None:
            target_max[layer_idx].append(amax_arr[i])
      elif kind == "hybrid_local":
        outer_axis, inner_axis = leaf_mapping[1], leaf_mapping[2]
        num_blocks = leaf_shape[outer_axis]
        num_local = leaf_shape[inner_axis]
        cfg_cycle = getattr(config, "inhomogeneous_layer_cycle_interval", None)
        cycle = int(cfg_cycle) if (cfg_cycle is not None and int(cfg_cycle) >= num_local + 1) else (num_local + 1)
        for b in range(num_blocks):
          for k in range(num_local):
            layer_idx = b * cycle + k
            target_sq[layer_idx].append(sq_arr[b, k])
            if target_max is not None and amax_arr is not None:
              target_max[layer_idx].append(amax_arr[b, k])
      elif kind == "hybrid_global":
        outer_axis, cycle, global_offset = leaf_mapping[1], leaf_mapping[2], leaf_mapping[3]
        num_blocks = leaf_shape[outer_axis]
        for b in range(num_blocks):
          layer_idx = b * cycle + global_offset
          target_sq[layer_idx].append(sq_arr[b])
          if target_max is not None and amax_arr is not None:
            target_max[layer_idx].append(amax_arr[b])

    if g is not None:
      g_keep = keep_axes if all(a < g.ndim for a in keep_axes) else ()
      g_sq, g_max = _reduce_leaf_for_axes(g, g_keep, compute_max=True)
      metrics[f"grad_norm/{name}"] = jnp.sqrt(jnp.sum(g_sq))
      metrics[f"grad_max/{name}"] = jnp.max(g_max)
      if g_keep == keep_axes:
        _record_layer_slices(g_sq, g_max, layer_grad_sq, layer_grad_max, mapping, g.shape)

    if p is not None:
      p_keep = keep_axes if all(a < p.ndim for a in keep_axes) else ()
      p_sq, _ = _reduce_leaf_for_axes(p, p_keep, compute_max=False)
      metrics[f"param_norm/{name}"] = jnp.sqrt(jnp.sum(p_sq))
      if p_keep == keep_axes:
        _record_layer_slices(p_sq, None, layer_param_sq, None, mapping, p.shape)

  for layer_idx in sorted(layer_grad_sq):
    metrics[f"layer_grad_norm/layer_{layer_idx:03d}"] = jnp.sqrt(jnp.sum(jnp.stack(layer_grad_sq[layer_idx])))
  for layer_idx in sorted(layer_grad_max):
    metrics[f"layer_grad_max/layer_{layer_idx:03d}"] = jnp.max(jnp.stack(layer_grad_max[layer_idx]))
  for layer_idx in sorted(layer_param_sq):
    metrics[f"layer_param_norm/layer_{layer_idx:03d}"] = jnp.sqrt(jnp.sum(jnp.stack(layer_param_sq[layer_idx])))

  return metrics


class MetadataKey(enum.Enum):
  PER_DEVICE_TFLOPS = "per_device_tflops"
  PER_DEVICE_TOKENS = "per_device_tokens"


_UNSET = object()


class MetricLogger:
  """
  Logger for saving metrics to a local file, GCS and TensorBoard.
  """

  # Lazily resolved local CPU device for host-side LR schedule evaluation
  # (_UNSET: not yet queried; None: unavailable or disabled after fallback).
  _host_cpu_device = _UNSET

  def __init__(self, config, learning_rate_schedule, start_step=0):
    self.writer = max_utils.initialize_summary_writer(config.tensorboard_dir, config.run_name, config.enable_tensorboard)
    self.config = config
    self.metadata = {}
    self.running_gcs_metrics = [] if config.gcs_metrics else None
    self.performance_metric_queue = self.get_performance_metric_queue(config)
    self.learning_rate_schedule = learning_rate_schedule
    self.start_step = start_step
    self.cumulative_eval_metrics = {"scalar": defaultdict(float)}
    # self.buffered_metrics is a polymorphic deferred-write queue. Entries are one of:
    #   ("train", train_step, metrics, step_time_delta)
    #   ("eval", eval_step, metrics, step_time_delta)
    self.buffered_metrics = []
    # Number of eval steps accumulated since the last reset_eval_metrics(). Used by
    # buffer_and_write_metrics to detect the eval→train transition and trigger finalization.
    self._pending_eval_step_count = 0
    # Wall-clock start of the in-flight eval loop, set by mark_eval_loop_start(). Used to emit
    # the mllog validation_time from _finalize_eval_metrics, where the eval metrics are already
    # being materialized, so measuring it costs no extra device synchronization.
    self._eval_loop_start = None
    if self.config.managed_mldiagnostics:
      ManagedMLDiagnostics(config)  # Initialize the MLRun instance.

    if self.config.enable_wandb and jax.process_index() == 0:
      import wandb  # pylint: disable=import-outside-toplevel # pytype: disable=import-error # lazy import: wandb is an optional dependency

      wandb.init(
          project=config.wandb_project_name,
          name=config.wandb_run_name,
          resume="allow",
      )  # Initialize wandb logger.

  def mark_eval_loop_start(self):
    """Records the wall-clock start of an eval loop for the mllog validation_time event."""
    self._eval_loop_start = datetime.datetime.now()

  def reset_eval_metrics(self):
    """Resets the cumulative metrics dictionary for a new evaluation run."""
    self.cumulative_eval_metrics = {"scalar": defaultdict(float)}
    self._pending_eval_step_count = 0

  def write_metrics(self, metrics, step, metric_type="train"):
    """Entry point for all metrics writing. metric_type is one of 'train', 'eval', 'running_eval'."""
    if metrics:
      _materialize_metric_scalars(metrics)
      self.log_metrics(metrics, step, metric_type)

      if self.config.enable_tensorboard and metric_type != "running_eval":
        self.write_metrics_to_tensorboard(metrics, step, metric_type)

      if self.config.metrics_file:
        self.write_metrics_locally(metrics, step)

      if self.config.gcs_metrics and jax.process_index() == 0:
        self.write_metrics_for_gcs(metrics, step, metric_type)

      if self.config.managed_mldiagnostics:
        self.write_metrics_to_managed_mldiagnostics(metrics, step)

      if self.config.enable_wandb and jax.process_index() == 0:
        self.write_metrics_to_wandb(metrics, step)

      if metric_type == "train":
        self._maybe_abort_after_write_metrics(metrics)

  def log_metrics(self, metrics, step, metric_type):
    """Logs metrics via max_logging."""
    if metric_type == "train":
      self._log_training_metrics(metrics, step)
    elif metric_type == "running_eval":
      self._log_running_eval_metrics(metrics, step)
    else:
      self._log_eval_metrics(metrics, step)

  def _log_training_metrics(self, metrics, step):
    """Handles training-specific metric logging."""
    # Skip logging if in profiler activation/deactivation steps
    # TODO(b/456828037): Switch to subprocess profiling to avoid timing artifacts at boundary steps.
    scalars = metrics["scalar"]
    loss = scalars["learning/loss"]
    is_rampup = step < self.config.rampup_end_step
    is_metric_hidden_step = self.config.hide_profiler_step_metric and self._is_profiler_boundary_step(step)

    # Start building the log parts
    log_parts = []
    if is_rampup:
      log_parts.append("[Rampup Batch Size Phase]")

    if is_metric_hidden_step:
      log_parts.append(
          f"completed profiler activation/deactivation step: {step}",
      )
    else:
      log_parts.extend(
          [
              f"completed step: {step}",
              f"seconds: {scalars['perf/step_time_seconds']:.3f}",
          ]
      )
      if elastic_utils.elastic_enabled(self.config):
        log_parts.extend(
            [
                f"live slice count: {len(elastic_utils.live_slice_indices(self.config))}",
            ]
        )

    # Add performance metrics only if strictly NOT in rampup phase
    # TODO(b/452468482): Enable performance metric (TFLOPs, Tokens/s) tracking during batch size rampup.
    if not is_rampup and not is_metric_hidden_step:
      log_parts.extend(
          [
              f"TFLOP/s/device: {scalars['perf/per_device_tflops_per_sec']:.3f}",
              f"Tokens/s/device: {scalars['perf/per_device_tokens_per_sec']:.3f}",
          ]
      )

    lm_loss = scalars.get("learning/lm_loss", 0.0)
    perplexity = scalars.get("learning/perplexity", 0.0)
    log_parts.extend(
        [
            f"total_weights: {scalars['learning/total_weights']}",
            f"loss: {loss:.3f}",
            f"lm_loss: {lm_loss:.3f}",
            f"perplexity: {perplexity:.3f}",
        ]
    )
    if "learning/dpo_loss" in scalars:
      log_parts.append(f"dpo_loss: {scalars['learning/dpo_loss']:.3f}")
    if "learning/reward_accuracy" in scalars:
      log_parts.append(f"reward_accuracy: {scalars['learning/reward_accuracy']:.3f}")

    if self.config.num_experts > 1:
      moe_lb_loss = scalars.get("learning/moe_lb_loss", 0.0)
      log_parts.append(f"moe_lb_loss: {moe_lb_loss:.6f}")
      if "learning/moe_lb_loss_unweighted" in scalars:
        log_parts.append(f"moe_lb_loss_unweighted: {scalars['learning/moe_lb_loss_unweighted']:.6f}")
      if "learning/moe_max_load_ratio" in scalars:
        log_parts.append(f"moe_max_load_ratio: {scalars['learning/moe_max_load_ratio']:.4f}")
        log_parts.append(f"moe_max_load_ratio_mean: {scalars['learning/moe_max_load_ratio_mean']:.4f}")

    if getattr(self.config, "mtp_num_layers", 0) > 0:
      mtp_loss = scalars.get("learning/mtp_loss", 0.0)
      log_parts.append(f"main_model_loss: {loss - mtp_loss:.3f}")
      log_parts.append(f"mtp_loss: {mtp_loss:.3f}")

    if getattr(self.config, "use_indexer", False):
      indexer_l = scalars.get("learning/indexer_loss", 0.0)
      log_parts.append(f"indexer_loss: {float(indexer_l)}")

    if getattr(self.config, "log_step_diagnostics", False):
      log_parts.append(step_diagnostics_text(scalars))

    max_logging.log(", ".join(log_parts))

    # Logged here rather than in the train loop: metrics are flushed one step late, so reading
    # the flag does not add a per-step host sync (which the in-layer fallback exists to avoid).
    fallback_layers = scalars.get("learning/moe_dropless_fallback_layers")
    if fallback_layers is not None and int(fallback_layers) > 0:
      max_logging.log(
          f"Step {step}: MoE ragged buffer overflow detected in {int(fallback_layers)} layer(s); "
          f"in-layer dropless fallback was used (no tokens dropped)."
      )

  def _log_eval_metrics(self, metrics, step):
    """Logs the final accumulated eval summary at the end of an eval run."""
    scalars = metrics["scalar"]
    log_parts = [
        f"Completed eval after train step {step}",
        f"loss={scalars['eval/avg_loss']:.3f}",
        f"perplexity={scalars['eval/avg_perplexity']:.3f}",
        f"total_weights={scalars['eval/total_weights']}",
        f"avg_z_loss={scalars.get('eval/avg_z_loss', 0.0):.3f}",
    ]
    if self.config.num_experts > 1:
      log_parts.append(f"avg_moe_lb_loss={scalars['eval/avg_moe_lb_loss']:.3f}")
      if "eval/avg_moe_lb_loss_unweighted" in scalars:
        log_parts.append(f"avg_moe_lb_loss_unweighted={scalars['eval/avg_moe_lb_loss_unweighted']:.6f}")
    if getattr(self.config, "mtp_num_layers", 0) > 0:
      log_parts.extend(
          [
              f"avg_mtp_loss={scalars['eval/avg_mtp_loss']:.3f}",
              f"avg_mtp_acceptance_rate={scalars['eval/avg_mtp_acceptance_rate_percent']:.2f}%",
          ]
      )
    if "eval/avg_dpo_reward_accuracy" in scalars:
      log_parts.append(f"dpo_reward_accuracy={scalars['eval/avg_dpo_reward_accuracy']:.3f}")

    if getattr(self.config, "use_indexer", False):
      indexer_l = scalars.get("eval/avg_indexer_loss", 0.0)
      log_parts.append(f"avg_indexer_loss={indexer_l:.3f}")

    max_logging.log(", ".join(log_parts))

    fallback_layers = int(scalars.get("eval/moe_dropless_fallback_layers", 0))
    if fallback_layers > 0:
      max_logging.log(
          f"Eval after train step {step}: MoE ragged buffer overflow detected in {fallback_layers} layer(s) "
          f"summed over eval steps; in-layer dropless fallback was used (no tokens dropped)."
      )

  def _log_running_eval_metrics(self, metrics, step):
    """Logs a per-eval-step running average (deferred by one eval step)."""
    scalars = metrics["scalar"]
    log_parts = [
        f"Completed eval step: {step}",
        f"seconds: {scalars['eval/step_time_seconds']:.3f}",
        f"running loss={scalars['eval/avg_loss']:.3f}",
        f"running perplexity={scalars['eval/avg_perplexity']:.3f}",
        f"running total_weights={scalars['eval/total_weights']}",
    ]
    if getattr(self.config, "mtp_num_layers", 0) > 0:
      log_parts.extend(
          [
              f"running mtp_loss={scalars['eval/avg_mtp_loss']:.3f}",
              f"running mtp_acceptance_rate={scalars['eval/avg_mtp_acceptance_rate_percent']:.2f}%",
          ]
      )
    max_logging.log(", ".join(log_parts))

  def _is_profiler_boundary_step(self, step):
    """Determines if the current step is a profiler start/stop boundary that should be hidden."""
    if len(self.config.profiler) == 0:
      return False
    skip_steps = self.config.skip_first_n_steps_for_profiler
    profiler_steps = self.config.profiler_steps
    # Steps immediately before/at start, and at/immediately after end of profiling
    boundary_steps = {
        skip_steps,
        skip_steps + 1,
        skip_steps + profiler_steps,
        skip_steps + profiler_steps + 1,
    }
    return step in boundary_steps

  def _maybe_abort_after_write_metrics(self, metrics):
    """This function checks whether we have nan or inf values in training"""
    loss = metrics["scalar"].get("learning/loss")
    if self.config.abort_on_nan_loss and np.isnan(loss):
      max_logging.log("Aborting training due to NaN loss.")
      sys.exit(1)
    if self.config.abort_on_inf_loss and np.isinf(loss):
      max_logging.log("Aborting training due to Inf loss.")
      sys.exit(1)

  def write_metrics_locally(self, metrics, step):
    """Writes metrics locally for testing."""
    with open(self.config.metrics_file, "a", encoding="utf8") as local_metrics_file:
      if step == 0:
        local_metrics_file.truncate(0)

      metrics_dict = _prepare_metrics_for_json(metrics, step, self.config.run_name)
      local_metrics_file.write(str(json.dumps(metrics_dict)) + "\n")

  def write_metrics_for_gcs(self, metrics, step, metric_type):
    """Writes metrics to GCS."""
    metrics_dict_step = _prepare_metrics_for_json(metrics, step, self.config.run_name)
    self.running_gcs_metrics.append(metrics_dict_step)
    if metric_type == "train" and (step + 1) % self.config.log_period == 0 or step == self.config.steps - 1:
      start_step = (step // self.config.log_period) * self.config.log_period
      metrics_filename = f"metrics_step_{start_step:06}_to_step_{step:06}.txt"
      with open(metrics_filename, "wt", encoding="utf8") as metrics_for_gcs:
        for metrics_step in self.running_gcs_metrics:
          metrics_for_gcs.write(str(json.dumps(metrics_step)) + "\n")

      gcs_filename = os.path.join(self.config.metrics_dir, metrics_filename)
      max_logging.log(f"Moving file {metrics_filename} to GCS...")
      gcs_utils.upload_blob(gcs_filename, metrics_filename)
      max_logging.log(f"File {metrics_filename} moved successfully!")
      self.running_gcs_metrics = []  # reset running_metrics to empty list

  def write_metrics_to_tensorboard(self, metrics, step, metric_type):
    """Writes metrics to TensorBoard."""
    if jax.process_index() == 0:
      for metric_name in metrics.get("scalar", []):
        self.writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name]), step)
      for metric_name in metrics.get("scalars", []):
        self.writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

    if metric_type == "train":
      full_log = step % self.config.log_period == 0

      if full_log and jax.process_index() == 0:
        max_logging.log(f"To see full metrics 'tensorboard --logdir={self.config.tensorboard_dir}'")
        self.writer.flush()

  def write_metrics_to_managed_mldiagnostics(self, metrics, step):
    """Write metrics to managed profiler."""
    if (step + 1) % self.config.log_period == 0 or step == self.config.steps - 1:
      for metric_name in metrics.get("scalar", []):
        value = metrics["scalar"][metric_name]
        # For NumPy/JAX array objects (including single-element arrays), use .item()
        # to extract the native Python scalar.
        if hasattr(value, "item"):
          value = value.item()
        mapped_metric_name = _METRICS_TO_MANAGED.get(metric_name, metric_name)
        mldiag.metrics.record(mapped_metric_name, value, step=int(step))

  def write_metrics_to_wandb(self, metrics, step):
    """Write metrics to weights and biases (wandb)."""
    import wandb  # pylint: disable=import-outside-toplevel # pytype: disable=import-error # lazy import: wandb is an optional dependency

    flat_metrics = {}
    for key, val in metrics.get("scalar", {}).items():
      flat_metrics[key] = float(val)
    for key, val in metrics.get("scalars", {}).items():
      for subkey, subval in val.items():
        flat_metrics[f"{key}/{subkey}"] = float(subval)
    wandb.log(flat_metrics, step=step)

  def write_setup_info_to_tensorboard(self, params):
    """Writes setup information like train config params, num model params, and XLA flags to TensorBoard."""
    num_model_parameters = max_utils.calculate_num_params_from_pytree(params)
    self.metadata[MetadataKey.PER_DEVICE_TFLOPS], _, _ = maxtext_utils.calculate_tflops_training_per_device(self.config)
    self.metadata[MetadataKey.PER_DEVICE_TOKENS] = maxtext_utils.calculate_tokens_training_per_device(self.config)
    max_logging.log(f"number parameters: {num_model_parameters/1e9:.3f} billion")
    if not self.config.enable_tensorboard:
      return
    max_utils.add_text_to_summary_writer("num_model_parameters", str(num_model_parameters), self.writer)
    max_utils.add_text_to_summary_writer("libtpu_init_args", os.getenv("LIBTPU_INIT_ARGS", ""), self.writer)
    maxtext_utils.add_config_to_summary_writer(self.config, self.writer)

  def get_performance_metric_queue(self, config):
    """Records heartbeat metrics and performance metrics to GCP."""
    performance_metric_queue = None

    # Early return if monitoring is stubbed
    if _monitor_is_stub:
      if config.report_heartbeat_metric_for_gcp_monitoring or config.report_performance_metric_for_gcp_monitoring:
        max_logging.log("[DECOUPLED NO-OP] skipping GCP workload monitoring threads.")
      return performance_metric_queue

    if config.report_heartbeat_metric_for_gcp_monitoring or config.report_performance_metric_for_gcp_monitoring:
      gcp_workload_monitor = GCPWorkloadMonitor(config.run_name)
      if config.report_heartbeat_metric_for_gcp_monitoring:
        gcp_workload_monitor.start_heartbeat_reporting_thread(config.heartbeat_reporting_interval_in_seconds)
      if config.report_performance_metric_for_gcp_monitoring:
        performance_metric_queue = queue.Queue()
        gcp_workload_monitor.start_performance_reporting_thread(performance_metric_queue)
    return performance_metric_queue

  def buffer_and_write_metrics(self, metrics, step, step_time_delta=None, is_training=True):
    """Per-step entry point for both train and eval metrics.

    Flushes the single deferred entry from the previous call, then queues this
    step's metrics. The buffer is a queue of length 1: a new call means the
    previous dispatch is already submitted, so its result is safe to
    materialize in a single batched jax.device_get() transfer.

    For train, record_train_metrics runs eagerly on the current step and
    evaluates self.learning_rate_schedule(step) on a local host CPU device when
    available so no accelerator op is enqueued behind p_train_step (avoiding
    device-queue stalls and extra Pathways proxy round trips).

    For eval, _accumulate_eval_metrics stays deferred to
    _flush_one_buffered_entry and batches device-to-host scalar transfers so
    the eval dispatch loop is not serialized.
    """
    if self.buffered_metrics:
      self._flush_one_buffered_entry(self.buffered_metrics.pop(0))
    if is_training:
      self.record_train_metrics(metrics, step, step_time_delta.total_seconds())
      self.buffered_metrics.append(("train", step, metrics, step_time_delta))
      if self._pending_eval_step_count > 0 or getattr(self.config, "enable_mllog", False):
        # Under enable_mllog, flush this step's entry now so tracked_stats stays ahead of
        # eval_stop/run_stop, and on steps without an eval also lands inside the step's own
        # block_start/block_stop pair. The .item() in the flush is the only synchronization
        # this adds, and the device is already the bottleneck, so it costs no step time.
        # On eval steps train.py has already emitted block_stop/eval_start, so tracked_stats
        # falls inside the eval interval instead; the 6.1 checker does not constrain it there.
        self._flush_one_buffered_entry(self.buffered_metrics.pop(0))
      if self._pending_eval_step_count > 0:
        self._finalize_eval_metrics(step)
    else:
      self._pending_eval_step_count += 1
      self.buffered_metrics.append(("eval", step, metrics, step_time_delta))

  def _flush_one_buffered_entry(self, entry):
    """Dispatches a single buffered entry to the writer."""
    kind = entry[0]
    if kind == "train":
      _, step, metrics, step_time_delta = entry
      self.write_metrics(metrics, step)
      # tracked_stats calls .item() on learning/loss. Deferring it to here means the step it
      # belongs to has already completed, so the materialization does not block the train loop.
      mllog_utils.tracked_stats(
          self.config,
          step + 1,
          step_time_delta.total_seconds(),
          metrics["scalar"]["learning/loss"],
          start_step=self.start_step,
      )
    elif kind == "eval":
      _, eval_step, raw_metrics, step_time_delta = entry
      # _accumulate_eval_metrics calls float() that materialize the metrics, deferred to here
      self._accumulate_eval_metrics(raw_metrics)
      running_count = eval_step + 1  # eval_step is 0-indexed
      cumulative = self.cumulative_eval_metrics["scalar"]
      running_avg_loss = cumulative["eval/total_loss"] / (cumulative["eval/total_weights"] + EPS)
      snapshot = {
          "scalar": {
              "eval/avg_loss": running_avg_loss,
              "eval/avg_perplexity": np.exp(running_avg_loss),
              "eval/total_weights": cumulative["eval/total_weights"],
              "eval/avg_mtp_loss": cumulative["eval/mtp_loss"] / running_count,
              "eval/avg_mtp_acceptance_rate_percent": (cumulative["eval/mtp_acceptance_rate_percent"] / running_count),
              "eval/step_time_seconds": step_time_delta.total_seconds(),
          }
      }
      self.write_metrics(snapshot, eval_step, metric_type="running_eval")

  def _accumulate_eval_metrics(self, metrics):
    """Accumulates one eval step's raw metrics into cumulative_eval_metrics."""
    _materialize_metric_scalars(metrics)
    scalar = metrics.get("scalar", {})
    self.cumulative_eval_metrics["scalar"]["eval/total_loss"] += float(scalar.get("evaluation/total_loss", 0.0))
    self.cumulative_eval_metrics["scalar"]["eval/total_weights"] += float(scalar.get("evaluation/total_weights", 0.0))
    self.cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] += float(scalar.get("evaluation/moe_lb_loss", 0.0))
    if "evaluation/moe_lb_loss_unweighted" in scalar:
      self.cumulative_eval_metrics["scalar"]["eval/moe_lb_loss_unweighted"] += float(
          scalar["evaluation/moe_lb_loss_unweighted"]
      )
    self.cumulative_eval_metrics["scalar"]["eval/indexer_loss"] += float(scalar.get("evaluation/indexer_loss", 0.0))
    self.cumulative_eval_metrics["scalar"]["eval/mtp_loss"] += float(scalar.get("evaluation/mtp_loss", 0.0))
    self.cumulative_eval_metrics["scalar"]["eval/mtp_acceptance_rate_percent"] += float(
        scalar.get("evaluation/mtp_acceptance_rate_percent", 0.0)
    )
    self.cumulative_eval_metrics["scalar"]["eval/z_loss"] += float(scalar.get("evaluation/z_loss", 0.0))
    if "evaluation/moe_dropless_fallback_layers" in scalar:
      self.cumulative_eval_metrics["scalar"]["eval/moe_dropless_fallback_layers"] += float(
          scalar["evaluation/moe_dropless_fallback_layers"]
      )

  def _evaluate_learning_rate_on_host(self, step):
    """Evaluates learning_rate_schedule on local host CPU when available.

    Requires a registered local CPU JAX backend (for example,
    `JAX_PLATFORMS=proxy,cpu` or `pathways,cpu` under Pathways). The CPU
    device lookup is cached per logger instance; if no CPU backend is
    available or host-CPU evaluation raises a device/runtime error, this logs
    once, falls back to direct evaluation, and disables further host-CPU
    attempts for this logger.

    Args:
      step: Current integer training step.

    Returns:
      The learning rate value as a host NumPy scalar/array when produced on
      CPU, or the schedule's native return value otherwise.
    """
    if self._host_cpu_device is _UNSET:
      try:
        cpu_devices = jax.local_devices(backend="cpu")
        self._host_cpu_device = cpu_devices[0] if cpu_devices else None
      except RuntimeError:
        self._host_cpu_device = None
      if self._host_cpu_device is None:
        max_logging.log(
            "No local CPU JAX backend available; learning_rate_schedule will"
            " run on the default device. Add `cpu` to JAX_PLATFORMS (e.g."
            " `proxy,cpu`) to evaluate it on the host."
        )

    if self._host_cpu_device is not None:
      try:
        with jax.default_device(self._host_cpu_device):
          lr = self.learning_rate_schedule(step)
      except (RuntimeError, ValueError) as err:
        prefix = "Host-CPU learning_rate_schedule evaluation failed"
        max_logging.log(f"{prefix} ({err}); falling back to default device.")
        self._host_cpu_device = None
      else:
        if _is_concrete_jax_array(lr):
          sharding = getattr(lr, "sharding", None)
          devices = getattr(sharding, "device_set", None)
          if devices and all(d.platform == "cpu" for d in devices):
            return np.asarray(lr)
        return lr

    return self.learning_rate_schedule(step)

  def record_train_metrics(self, metrics, step, step_time):
    """Records training metrics for the current step."""
    metrics["scalar"].update({"perf/step_time_seconds": step_time})
    lr = self._evaluate_learning_rate_on_host(step)
    metrics["scalar"].update({"learning/current_learning_rate": lr})
    if step >= self.config.rampup_end_step:
      metrics["scalar"].update({"perf/per_device_tflops": self.metadata[MetadataKey.PER_DEVICE_TFLOPS]})
      metrics["scalar"].update(
          {"perf/per_device_tflops_per_sec": (self.metadata[MetadataKey.PER_DEVICE_TFLOPS] / step_time)}
      )
      metrics["scalar"].update({"perf/per_device_tokens": self.metadata[MetadataKey.PER_DEVICE_TOKENS]})
      metrics["scalar"].update(
          {"perf/per_device_tokens_per_sec": (self.metadata[MetadataKey.PER_DEVICE_TOKENS] / step_time)}
      )
    if self.performance_metric_queue:
      self.performance_metric_queue.put(step_time)

  def _finalize_eval_metrics(self, train_step):
    """Computes final averaged eval metrics and writes them at train_step."""
    eval_step_count = self._pending_eval_step_count
    cumulative = self.cumulative_eval_metrics["scalar"]
    eval_loss = cumulative["eval/total_loss"] / (cumulative["eval/total_weights"] + EPS)
    cumulative["eval/avg_loss"] = eval_loss
    cumulative["eval/avg_perplexity"] = np.exp(eval_loss)
    cumulative["eval/avg_moe_lb_loss"] = cumulative["eval/moe_lb_loss"] / eval_step_count
    if "eval/moe_lb_loss_unweighted" in cumulative:
      cumulative["eval/avg_moe_lb_loss_unweighted"] = cumulative["eval/moe_lb_loss_unweighted"] / eval_step_count
    cumulative["eval/avg_indexer_loss"] = cumulative["eval/indexer_loss"] / eval_step_count
    cumulative["eval/avg_mtp_loss"] = cumulative["eval/mtp_loss"] / eval_step_count
    cumulative["eval/avg_mtp_acceptance_rate_percent"] = cumulative["eval/mtp_acceptance_rate_percent"] / eval_step_count
    cumulative["eval/avg_z_loss"] = cumulative["eval/z_loss"] / eval_step_count

    self.write_metrics(self.cumulative_eval_metrics, train_step, metric_type="eval")
    self._pending_eval_step_count = 0
    # Emitted here rather than at the end of the eval loop in train.py: reaching this point
    # required materializing every eval metric, so the elapsed time already covers the eval
    # device work without a dedicated jax.block_until_ready() draining the pipeline. Must stay
    # ahead of eval_stop, which may emit the terminal run_stop.
    if self._eval_loop_start is not None:
      mllog_utils.validation_time(train_step + 1, (datetime.datetime.now() - self._eval_loop_start).total_seconds())
      self._eval_loop_start = None
    mllog_utils.eval_stop(self.config, train_step + 1, eval_loss, self.start_step)
    if self.config.target_eval_loss and eval_loss <= self.config.target_eval_loss:
      raise exceptions.StopTraining(f"Target loss {self.config.target_eval_loss=} is achieved.")

  def flush_metrics_and_cleanup(self):
    """
    This is a terminal operation that uploads any buffered metrics to GCS
    and/or TensorBoard before closing the writer objects. Once called, the
    logger instance should not be used to add or write more metrics as the
    underlying writer objects (e.g., TensorBoard SummaryWriter) will be closed.
    """
    for entry in self.buffered_metrics:
      self._flush_one_buffered_entry(entry)
    self.buffered_metrics = []

    mllog_utils.sync_log(force=True)
    max_utils.close_summary_writer(self.writer)

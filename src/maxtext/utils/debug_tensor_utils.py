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

"""Tensor distribution debugging utilities for MaxText."""

import functools
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np


def _compute_stats(x: Any) -> dict[str, Any]:
  """Computes mean, std, min, max, l2_norm, percentiles, NaNs, and Infs for a tensor."""
  x_f32 = jnp.asarray(x, dtype=jnp.float32)
  x_flat = jnp.ravel(x_f32)

  mean = jnp.mean(x_flat)
  std = jnp.std(x_flat)
  min_val = jnp.min(x_flat)
  max_val = jnp.max(x_flat)
  l2_norm = jnp.linalg.norm(x_flat)

  # Non-optional percentiles: 1%, 5%, 25%, 50%, 75%, 95%, 99%
  pct_qs = jnp.array([1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0])
  pcts = jnp.percentile(x_flat, pct_qs)

  nan_count = jnp.sum(jnp.isnan(x_flat))
  inf_count = jnp.sum(jnp.isinf(x_flat))

  return {
      "mean": mean,
      "std": std,
      "min": min_val,
      "max": max_val,
      "l2_norm": l2_norm,
      "p01": pcts[0],
      "p05": pcts[1],
      "p25": pcts[2],
      "p50": pcts[3],
      "p75": pcts[4],
      "p95": pcts[5],
      "p99": pcts[6],
      "nans": nan_count,
      "infs": inf_count,
      "nan_count": nan_count,
      "inf_count": inf_count,
  }


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _debug_tensor_vjp(x: Any, name: str, step: int | jax.Array, enabled: bool) -> Any:
  """Custom VJP wrapper to tap tensor distributions in forward and backward passes."""
  return x


def _debug_tensor_fwd(x: Any, name: str, step: int | jax.Array, enabled: bool):
  """Forward rule for _debug_tensor_vjp computing and printing tensor statistics."""
  if enabled:
    stats = _compute_stats(x)
    shape_str = getattr(x, "shape", ())
    dtype_str = getattr(x, "dtype", type(x).__name__)
    is_moe_routing = name.endswith(("router_weights", "combine_weights", "gate_logits", "expert_weights")) or (
        "moe" in name and name.endswith(("weights", "logits"))
    )
    if is_moe_routing and hasattr(x, "ndim") and x.ndim >= 2:
      x_f32 = jnp.asarray(x, dtype=jnp.float32)
      mean_expert_weights = jnp.mean(x_f32, axis=tuple(range(x.ndim - 1)))
      jax.debug.print(
          "[DEBUG_TENSOR FWD] step={step} name={name} shape={shape}"
          " dtype={dtype} mean={mean:.6e} std={std:.6e} min={min:.6e}"
          " max={max:.6e} l2_norm={l2_norm:.6e} p01={p01:.6e} p05={p05:.6e}"
          " p25={p25:.6e} p50={p50:.6e} p75={p75:.6e} p95={p95:.6e}"
          " p99={p99:.6e} nan_count={nan_count} inf_count={inf_count}"
          " expert_weights={expert_weights}",
          step=step,
          name=name,
          shape=shape_str,
          dtype=str(dtype_str),
          mean=stats["mean"],
          std=stats["std"],
          min=stats["min"],
          max=stats["max"],
          l2_norm=stats["l2_norm"],
          p01=stats["p01"],
          p05=stats["p05"],
          p25=stats["p25"],
          p50=stats["p50"],
          p75=stats["p75"],
          p95=stats["p95"],
          p99=stats["p99"],
          nan_count=stats["nan_count"],
          inf_count=stats["inf_count"],
          expert_weights=mean_expert_weights,
      )
    else:
      jax.debug.print(
          "[DEBUG_TENSOR FWD] step={step} name={name} shape={shape}"
          " dtype={dtype} mean={mean:.6e} std={std:.6e} min={min:.6e}"
          " max={max:.6e} l2_norm={l2_norm:.6e} p01={p01:.6e} p05={p05:.6e}"
          " p25={p25:.6e} p50={p50:.6e} p75={p75:.6e} p95={p95:.6e}"
          " p99={p99:.6e} nan_count={nan_count} inf_count={inf_count}",
          step=step,
          name=name,
          shape=shape_str,
          dtype=str(dtype_str),
          mean=stats["mean"],
          std=stats["std"],
          min=stats["min"],
          max=stats["max"],
          l2_norm=stats["l2_norm"],
          p01=stats["p01"],
          p05=stats["p05"],
          p25=stats["p25"],
          p50=stats["p50"],
          p75=stats["p75"],
          p95=stats["p95"],
          p99=stats["p99"],
          nan_count=stats["nan_count"],
          inf_count=stats["inf_count"],
      )
  return x, None


def _debug_tensor_bwd(name: str, step: int | jax.Array, enabled: bool, res: Any, g: Any):
  """Backward rule for _debug_tensor_vjp computing and printing gradient statistics."""
  if enabled and g is not None:
    stats = _compute_stats(g)
    shape_str = getattr(g, "shape", ())
    dtype_str = getattr(g, "dtype", type(g).__name__)
    jax.debug.print(
        "[DEBUG_TENSOR BWD] step={step} name={name}/grad shape={shape}"
        " dtype={dtype} mean={mean:.6e} std={std:.6e} min={min:.6e}"
        " max={max:.6e} l2_norm={l2_norm:.6e} p01={p01:.6e} p05={p05:.6e}"
        " p25={p25:.6e} p50={p50:.6e} p75={p75:.6e} p95={p95:.6e} p99={p99:.6e}"
        " nan_count={nan_count} inf_count={inf_count}",
        step=step,
        name=name,
        shape=shape_str,
        dtype=str(dtype_str),
        mean=stats["mean"],
        std=stats["std"],
        min=stats["min"],
        max=stats["max"],
        l2_norm=stats["l2_norm"],
        p01=stats["p01"],
        p05=stats["p05"],
        p25=stats["p25"],
        p50=stats["p50"],
        p75=stats["p75"],
        p95=stats["p95"],
        p99=stats["p99"],
        nan_count=stats["nan_count"],
        inf_count=stats["inf_count"],
    )
  return (g,)


_debug_tensor_vjp.defvjp(_debug_tensor_fwd, _debug_tensor_bwd)


def _get_active_step() -> int | jax.Array:
  """Returns current active step from telemetry scope if active, else 0."""
  try:
    # pylint: disable=import-outside-toplevel
    from maxtext.utils import debug_tensor_interceptors

    _, step = debug_tensor_interceptors.get_active_telemetry_context()
    return step
  except (ImportError, AttributeError):
    return 0


def should_debug_tensor(
    config: Any,
    name: str,
    step: int | jax.Array | None = None,
) -> bool:
  """Determines if debug logging should be enabled for a given tensor name and step."""
  if step is None:
    step = _get_active_step()
  if config is None:
    return False
  if not getattr(config, "debug_tensor_distribution", False):
    return False
  layers_filter = getattr(config, "debug_tensor_distribution_layers", "all")
  if layers_filter and layers_filter != "all":
    allowed_layers = [l.strip() for l in layers_filter.split(",") if l.strip()]

    def _matches(layer_pat: str, tensor_name: str) -> bool:
      if layer_pat in tensor_name:
        if layer_pat.isdigit():
          parts = tensor_name.replace("/", "_").split("_")
          return layer_pat in parts
        return True
      return False

    if not any(_matches(layer, name) for layer in allowed_layers):
      return False
  interval = getattr(config, "debug_tensor_distribution_step_interval", 1)
  if isinstance(step, (int, np.integer)) and interval > 1:
    if step % interval != 0:
      return False
  return True


def debug_tensor(
    x: Any,
    name: str,
    step: int | jax.Array | None = None,
    enabled: bool | Any = True,
) -> Any:
  """Identity function instrumented with FWD and BWD distribution logging.

  When enabled=False, statically returns x directly at trace time with zero
  Jaxpr overhead.
  When enabled is a Config object, automatically evaluates should_debug_tensor.
  """
  if step is None:
    step = _get_active_step()
  if not isinstance(enabled, (bool, np.bool_)):
    enabled = should_debug_tensor(enabled, name, step)
  if not enabled:
    return x
  return _debug_tensor_vjp(x, name, step, True)


def debug_tensor_from_config(
    x: Any,
    name: str,
    config: Any,
    step: int | jax.Array | None = None,
) -> Any:
  """Convenience wrapper around debug_tensor using MaxText Config."""
  return debug_tensor(x, name, step=step, enabled=config)

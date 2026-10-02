# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ML Diagnostics Scalar Backend for Tunix RL.

Implements the metrax / Tunix LoggingBackend protocol (log_scalar, close)
to stream training and evaluation scalar metrics directly into Google Cloud ML
Diagnostics.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import jax
import numpy as np

try:
  import google_cloud_mldiagnostics as mldiag

  mldiag_metrics = getattr(mldiag, "metrics", None)
  metric_types = getattr(mldiag, "metric_types", None)
  _HAS_MLDIAG = mldiag is not None and mldiag_metrics is not None
except ImportError:
  try:
    # pylint: disable=g-import-not-at-top
    from maxtext.common.gcloud_stub import mldiagnostics_modules

    mldiag, _ = mldiagnostics_modules()
    mldiag_metrics = getattr(mldiag, "metrics", None) if mldiag else None
    metric_types = getattr(mldiag, "metric_types", None) if mldiag else None
    _HAS_MLDIAG = mldiag is not None and mldiag_metrics is not None
  except Exception:  # pylint: disable=broad-exception-caught
    mldiag = None
    mldiag_metrics = None
    metric_types = None
    _HAS_MLDIAG = False

from maxtext.common import managed_mldiagnostics

_EXACT_METRIC_MAP = {
    "loss": "LOSS",
    "learning_rate": "LEARNING_RATE",
    "grad_norm": "GRADIENT_NORM",
    "total_weights": "TOTAL_WEIGHTS",
    "step_time": "STEP_TIME",
    "global_step_time": "STEP_TIME",
    "throughput": "THROUGHPUT",
    "latency": "LATENCY",
}


def _normalize_metric_event(event: str) -> str:
  """Normalizes enum string representations in metric event names.

  In Python 3.11+, enum interpolation in f-strings can format enums as
  'Mode.TRAIN' or 'Mode.EVAL' instead of their string values ('train',
  'eval'). This normalizes them back to lowercase standard modes.

  Args:
    event: The raw metric event name.

  Returns:
    The normalized metric event name.
  """
  return (
      event.replace("Mode.TRAIN", "train")
      .replace("Mode.Train", "train")
      .replace("Mode.EVAL", "eval")
      .replace("Mode.Eval", "eval")
  )


def _extract_scalar(value: Any) -> int | float | None:
  """Extracts a primitive Python float or int from an array, tensor, or scalar.

  Args:
    value: A scalar, array, or tensor value.

  Returns:
    An int or float scalar value, or None if extraction fails or value is bool.
  """
  if isinstance(value, (str, bytes)):
    return None
  try:
    if hasattr(value, "item"):
      val = value.item()
    elif isinstance(value, (int, float, np.number)):
      val = value
    else:
      val = float(value)

    if isinstance(val, (str, bytes, bool, np.bool_)):
      return None
    if isinstance(val, (int, np.integer)):
      return int(val)
    if isinstance(val, (float, np.floating)):
      return float(val)
    return float(val)
  except (TypeError, ValueError, AttributeError, RuntimeError):
    return None


class MLDiagScalarBackend:
  """Routes all scalar metrics to Google Cloud ML Diagnostics."""

  def __init__(self, config: Any | None = None) -> None:
    """Initializes the ML Diagnostics scalar backend."""
    if not _HAS_MLDIAG:
      logging.warning(
          "google_cloud_mldiagnostics is not installed; MLDiagScalarBackend is"
          " disabled."
      )
    if config is not None:
      managed_mldiagnostics.ManagedMLDiagnostics(config)

  def log_scalar(
      self,
      event: str,
      value: Any,
      step: int | None = None,
      **kwargs: Any,
  ) -> None:
    """Logs a single scalar metric to Google Cloud ML Diagnostics.

    Args:
      event: Hierarchical metric name (e.g. 'actor/train/loss',
        'rewards/train/score/mean').
      value: Metric scalar value (can be float, int, or jnp/np array).
      step: Training or evaluation step index.
      **kwargs: Additional metadata keywords.
    """
    if not _HAS_MLDIAG or mldiag_metrics is None or jax.process_index() != 0:
      return

    val = _extract_scalar(value)
    if val is None or math.isnan(val) or math.isinf(val):
      return

    try:
      event = _normalize_metric_event(event)
      metric_key = event
      if event in _EXACT_METRIC_MAP:
        enum_attr = _EXACT_METRIC_MAP[event]
        if metric_types is not None and hasattr(
            metric_types.MetricType, enum_attr
        ):
          metric_key = getattr(metric_types.MetricType, enum_attr)

      int_step = int(step) if step is not None else None
      timestamp = kwargs.get("timestamp")
      if timestamp is not None:
        mldiag_metrics.record(
            metric_key, val, step=int_step, timestamp=timestamp
        )
      else:
        mldiag_metrics.record(metric_key, val, step=int_step)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Failed to record metric '%s' to ML Diagnostics: %s", event, e
      )

  def close(self) -> None:
    """Closes the ML Diagnostics backend."""
    pass

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layer-by-layer activation similarity and diagnostic utilities for validation agent."""

from typing import Any, Dict, List, Optional
import numpy as np


def compute_layer_statistics(tensor_or_array: Any) -> Dict[str, float]:
  """Computes summary statistics (mean, std, min, max) for a layer activation array/tensor."""
  try:
    arr = np.asarray(tensor_or_array, dtype=np.float64)
    if arr.size == 0:
      return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "has_nan_inf": False}
    has_nan_inf = bool(np.isnan(arr).any() or np.isinf(arr).any())
    return {
        "mean": float(np.mean(arr)) if not has_nan_inf else float("nan"),
        "std": float(np.std(arr)) if not has_nan_inf else float("nan"),
        "min": float(np.min(arr)) if not has_nan_inf else float("nan"),
        "max": float(np.max(arr)) if not has_nan_inf else float("nan"),
        "has_nan_inf": has_nan_inf,
    }
  except Exception:  # pylint: disable=broad-exception-caught
    return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "has_nan_inf": True}


def compute_cosine_similarity(arr1: Any, arr2: Any) -> float:
  """Computes cosine similarity between two layer activation arrays."""
  try:
    a = np.asarray(arr1, dtype=np.float64).flatten()
    b = np.asarray(arr2, dtype=np.float64).flatten()
    min_len = min(a.size, b.size)
    if min_len == 0:
      return 0.0
    a = a[:min_len]
    b = b[:min_len]
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
      return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
  except Exception:  # pylint: disable=broad-exception-caught
    return 0.0


def analyze_layer_divergence(
    hf_hidden_states: Optional[List[Any]] = None,
    mt_intermediates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  """Analyzes layer-by-layer activation divergence between HuggingFace and MaxText models.

  Returns a dictionary containing:
    - 'layers': List of per-layer statistics and cosine similarity.
    - 'first_divergence_layer': Index of the first layer where cosine similarity drops below 0.98.
    - 'summary_table': Formatted ASCII table suitable for stdout and JSON report inclusion.
  """
  layers_data = []
  first_divergence_layer = None

  num_layers = 0
  if hf_hidden_states:
    num_layers = len(hf_hidden_states)
  elif mt_intermediates and "hidden_states" in mt_intermediates:
    num_layers = len(mt_intermediates["hidden_states"])

  for idx in range(num_layers):
    layer_label = "Embedding" if idx == 0 else f"Layer_{idx - 1:03d}"
    row = {"layer_index": idx, "label": layer_label}

    hf_arr = hf_hidden_states[idx] if (hf_hidden_states and idx < len(hf_hidden_states)) else None
    if hf_arr is not None:
      row["hf_stats"] = compute_layer_statistics(hf_arr)

    mt_arr = None
    if mt_intermediates and "hidden_states" in mt_intermediates:
      hs_list = mt_intermediates["hidden_states"]
      if idx < len(hs_list):
        mt_arr = hs_list[idx]

    if mt_arr is not None:
      row["mt_stats"] = compute_layer_statistics(mt_arr)

    if hf_arr is not None and mt_arr is not None:
      cos_sim = compute_cosine_similarity(hf_arr, mt_arr)
      row["cosine_similarity"] = cos_sim
      if cos_sim < 0.98 and first_divergence_layer is None:
        first_divergence_layer = idx
    else:
      row["cosine_similarity"] = None

    layers_data.append(row)

  table_lines = [
      "--- Layer-by-Layer Activation Summary ---",
      "| Layer       | HF Mean    | HF StdDev  | MT Mean    | MT StdDev  | CosSim   | Status       |",
      "|-------------|------------|------------|------------|------------|----------|--------------|",
  ]

  for row in layers_data:
    label = row["label"]
    hf_mean = row.get("hf_stats", {}).get("mean", 0.0)
    hf_std = row.get("hf_stats", {}).get("std", 0.0)
    mt_mean = row.get("mt_stats", {}).get("mean", 0.0)
    mt_std = row.get("mt_stats", {}).get("std", 0.0)
    cossim_val = row.get("cosine_similarity")
    cossim_str = f"{cossim_val:8.4f}" if cossim_val is not None else "     N/A"

    status = "OK"
    if row.get("hf_stats", {}).get("has_nan_inf") or row.get("mt_stats", {}).get("has_nan_inf"):
      status = "INVALID (NaN)"
    elif cossim_val is not None and cossim_val < 0.98:
      status = "DIVERGED"

    table_lines.append(
        f"| {label:<11} | {hf_mean:10.4f} | {hf_std:10.4f} | {mt_mean:10.4f} | {mt_std:10.4f} | {cossim_str} |"
        f" {status:<12} |"
    )

  summary_table = "\n".join(table_lines)

  return {
      "layers": layers_data,
      "first_divergence_layer": first_divergence_layer,
      "summary_table": summary_table,
  }

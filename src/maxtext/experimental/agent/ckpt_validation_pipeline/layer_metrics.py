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
    arr = np.asarray(tensor_or_array, dtype=np.float32)
    if arr.size == 0:
      return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "has_nan_inf": False}
    has_nan_inf = bool(not np.isfinite(arr).all())
    return {
        "mean": float(np.mean(arr)) if not has_nan_inf else None,
        "std": float(np.std(arr)) if not has_nan_inf else None,
        "min": float(np.min(arr)) if not has_nan_inf else None,
        "max": float(np.max(arr)) if not has_nan_inf else None,
        "has_nan_inf": has_nan_inf,
    }
  except Exception:  # pylint: disable=broad-exception-caught
    return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "has_nan_inf": True}


def compute_cosine_similarity(arr1: Any, arr2: Any) -> float:
  """Computes cosine similarity between two layer activation arrays."""
  try:
    a = np.ravel(np.asarray(arr1, dtype=np.float32))
    b = np.ravel(np.asarray(arr2, dtype=np.float32))
    if a.size != b.size:
      return None
    if a.size == 0:
      return None
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
      return None
    return float(np.dot(a, b) / (norm_a * norm_b))
  except Exception:  # pylint: disable=broad-exception-caught
    return None


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

      has_nan_inf = row.get("hf_stats", {}).get("has_nan_inf", False) or row.get("mt_stats", {}).get("has_nan_inf", False)
      is_diverged = (cos_sim < 0.98) if not np.isnan(cos_sim) else True
      if (has_nan_inf or is_diverged) and first_divergence_layer is None:
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
    hf_mean = row.get("hf_stats", {}).get("mean")
    hf_std = row.get("hf_stats", {}).get("std")
    mt_mean = row.get("mt_stats", {}).get("mean")
    mt_std = row.get("mt_stats", {}).get("std")
    cossim_val = row.get("cosine_similarity")
    cossim_str = f"{cossim_val:8.4f}" if cossim_val is not None else "     N/A"

    hf_mean_str = f"{hf_mean:10.4f}" if hf_mean is not None else "       NaN"
    hf_std_str = f"{hf_std:10.4f}" if hf_std is not None else "       NaN"
    mt_mean_str = f"{mt_mean:10.4f}" if mt_mean is not None else "       NaN"
    mt_std_str = f"{mt_std:10.4f}" if mt_std is not None else "       NaN"

    status = "OK"
    if row.get("hf_stats", {}).get("has_nan_inf") or row.get("mt_stats", {}).get("has_nan_inf"):
      status = "INVALID (NaN)"
    elif cossim_val is not None and cossim_val < 0.98:
      status = "DIVERGED"

    table_lines.append(
        f"| {label:<11} | {hf_mean_str} | {hf_std_str} | {mt_mean_str} | {mt_std_str} | {cossim_str} |"
        f" {status:<12} |"
    )

  summary_table = "\n".join(table_lines)

  return {
      "layers": layers_data,
      "first_divergence_layer": first_divergence_layer,
      "summary_table": summary_table,
  }

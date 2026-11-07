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

"""Model-agnostic FP64 math, tensor comparison, and Orbax loader utilities for layer bisection."""

import abc
import importlib
from typing import Any
import numpy as np


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6, zero_centered: bool = False) -> np.ndarray:
  """FP64 Root Mean Square Layer Normalization (supports standard and zero-centered weights)."""
  x64 = x.astype(np.float64)
  w64 = w.astype(np.float64)
  scale = (1.0 + w64) if zero_centered else w64
  variance = np.mean(x64 * x64, axis=-1, keepdims=True)
  return x64 * (1.0 / np.sqrt(variance + eps)) * scale


def silu(x: np.ndarray) -> np.ndarray:
  """FP64 SiLU (Swish) activation."""
  x64 = x.astype(np.float64)
  return x64 / (1.0 + np.exp(-x64))


def swiglu(gate: np.ndarray, up: np.ndarray, limit: float | None = None) -> np.ndarray:
  """FP64 SwiGLU activation with optional symmetric clamping."""
  g = gate.astype(np.float64)
  u = up.astype(np.float64)
  if limit is not None and limit > 0:
    g = np.clip(g, -limit, limit)
    u = np.clip(u, -limit, limit)
  return silu(g) * u


def tensor_diff_metrics(ref: np.ndarray, cand: np.ndarray) -> dict[str, float | int]:
  """Computes max absolute diff, relative L2 norm, cosine similarity, and worst token position."""
  if ref.shape != cand.shape:
    raise ValueError(f"Shape mismatch in tensor_diff_metrics: ref {ref.shape} vs cand {cand.shape}")
  r = ref.astype(np.float64)
  c = cand.astype(np.float64)
  diff = r - c
  max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
  ref_norm = float(np.linalg.norm(r))
  diff_norm = float(np.linalg.norm(diff))
  rel_l2 = diff_norm / (ref_norm + 1e-12)
  cand_norm = float(np.linalg.norm(c))
  cos_sim = float(np.sum(r * c) / (ref_norm * cand_norm + 1e-12)) if ref_norm > 0 and cand_norm > 0 else 1.0

  # Identify the token position with the largest L2 error when rank >= 2 (e.g. [B, S, D] or [S, D]).
  worst_pos = 0
  if diff.ndim >= 2:
    if diff.ndim >= 3:
      per_pos = np.linalg.norm(diff.reshape(-1, diff.shape[-2], diff.shape[-1]), axis=(0, 2))
    else:
      per_pos = np.linalg.norm(diff, axis=-1)
    worst_pos = int(np.argmax(per_pos))

  return {
      "max_abs": max_abs,
      "rel_l2": rel_l2,
      "cos_sim": cos_sim,
      "worst_pos": worst_pos,
  }


def load_zarr_tensor(
    ckpt_items: str,
    key: str,
    layer: int | None = None,
    scanned: bool = False,
    scan_axis: int = 1,
) -> np.ndarray:
  """Loads a tensor from an Orbax Zarr3 + OCDbt checkpoint in FP64 (scanned or unscanned)."""
  import tensorstore as ts  # pylint: disable=import-outside-toplevel

  spec = {
      "driver": "zarr3",
      "kvstore": {"driver": "ocdbt", "base": ckpt_items, "path": key},
  }
  arr = np.asarray(ts.open(spec, read=True).result().read().result(), dtype=np.float64)
  if scanned and layer is not None:
    effective_axis = scan_axis if arr.ndim > scan_axis else 0
    arr = np.take(arr, indices=layer, axis=effective_axis)
  return arr


class LayerAdapter(abc.ABC):
  """Abstract interface for optional single-layer FP64 weight recomputation."""

  @abc.abstractmethod
  def load_layer_weights(
      self, ckpt_items: str, layer: int, scanned: bool = False, scan_axis: int = 1
  ) -> dict[str, np.ndarray]:
    """Loads converted checkpoint weights for the given layer index."""

  @abc.abstractmethod
  def forward_layer_fp64(self, x: np.ndarray, weights: dict[str, np.ndarray], eps: float = 1e-6) -> np.ndarray:
    """Runs the layer's FP64 reference math on input x."""


class StandardSwiGLUMLPAdapter(LayerAdapter):
  """Default adapter for a standard pre-norm SwiGLU MLP sub-block (scanned or unscanned)."""

  def __init__(self, prefix_template: str | None = None):
    self.prefix_template = prefix_template

  def load_layer_weights(
      self, ckpt_items: str, layer: int, scanned: bool = False, scan_axis: int = 1
  ) -> dict[str, np.ndarray]:
    if self.prefix_template:
      prefix = self.prefix_template.format(layer=layer)
    elif scanned:
      prefix = "params.params.decoder.layers"
    else:
      prefix = f"params.params.decoder.layers_{layer}"
    return {
        "norm": load_zarr_tensor(
            ckpt_items,
            f"{prefix}.post_self_attention_layer_norm.scale",
            layer=layer,
            scanned=scanned,
            scan_axis=scan_axis,
        ),
        "wi_0": load_zarr_tensor(
            ckpt_items,
            f"{prefix}.mlp.wi_0.kernel",
            layer=layer,
            scanned=scanned,
            scan_axis=scan_axis,
        ),
        "wi_1": load_zarr_tensor(
            ckpt_items,
            f"{prefix}.mlp.wi_1.kernel",
            layer=layer,
            scanned=scanned,
            scan_axis=scan_axis,
        ),
        "wo": load_zarr_tensor(
            ckpt_items,
            f"{prefix}.mlp.wo.kernel",
            layer=layer,
            scanned=scanned,
            scan_axis=scan_axis,
        ),
    }

  def forward_layer_fp64(self, x: np.ndarray, weights: dict[str, np.ndarray], eps: float = 1e-6) -> np.ndarray:
    x64 = x.astype(np.float64)
    normed = rmsnorm(x64, weights["norm"], eps=eps)
    gate = normed @ weights["wi_0"]
    up = normed @ weights["wi_1"]
    return x64 + (swiglu(gate, up) @ weights["wo"])


def resolve_adapter(adapter_spec: str | None) -> LayerAdapter:
  """Resolves 'module.path:ClassName' (or returns StandardSwiGLUMLPAdapter if None)."""
  if not adapter_spec or adapter_spec == "default":
    return StandardSwiGLUMLPAdapter()
  if ":" not in adapter_spec:
    raise ValueError(f"Invalid --adapter '{adapter_spec}'. Expected 'module_path:ClassName'.")
  mod_name, cls_name = adapter_spec.split(":", 1)
  mod = importlib.import_module(mod_name)
  cls: Any = getattr(mod, cls_name)
  return cls()

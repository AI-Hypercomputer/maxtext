# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical checkpoint conversion test helpers for Kimi-K3."""

from typing import Any, Mapping, Optional

from flax import nnx
import jax.numpy as jnp
import numpy as np

from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING

_HF_PREFIX = "language_model."


def _find_state_dict_key(hf_key: str, pt_state_dict: Mapping[str, Any]) -> Optional[str]:
  """Returns the `pt_state_dict` key that `hf_key` refers to, or None.

  `pt_state_dict` is usually the state dict of a single reference submodule (e.g. a KDA
  attention block with keys like `b_proj.weight`), so its keys are suffixes of the full HF
  checkpoint names. Matching is exact, then with the `language_model.` prefix stripped,
  then on a `.`-boundary suffix (so `f_b_proj.weight` never matches `b_proj.weight`). If
  several suffixes match, the longest (most specific) one wins.
  """
  if hf_key in pt_state_dict:
    return hf_key
  if hf_key.startswith(_HF_PREFIX) and hf_key[len(_HF_PREFIX) :] in pt_state_dict:
    return hf_key[len(_HF_PREFIX) :]
  candidates = [k for k in pt_state_dict if hf_key.endswith("." + k)]
  return max(candidates, key=len) if candidates else None


def _to_numpy(tensor: Any) -> np.ndarray:
  """torch.Tensor (or array-like) -> float32 NumPy array, without importing torch here."""
  if hasattr(tensor, "detach"):
    return tensor.detach().cpu().float().numpy()
  return np.asarray(tensor, dtype=np.float32)


def convert_pytorch_module_to_maxtext_params(
    pt_state_dict: Mapping[str, Any],
    hf_config: dict[str, Any],
    maxtext_config: Any = None,
    prefix_filter: Optional[str] = None,
    target_shapes: Optional[Mapping[str, tuple[int, ...]]] = None,
) -> dict[str, np.ndarray]:
  """Converts a PyTorch module state_dict to a MaxText parameter dictionary.

  Runs the production `PARAM_MAPPING["kimi-k3"]` / `HOOK_FNS["kimi-k3"]` (HF -> MaxText,
  unscanned). Key lookup differs from `to_maxtext` (see `_find_state_dict_key`), and mapping
  entries with no matching tensor are skipped, because callers pass a single submodule's
  state dict. `to_maxtext` calls hooks with the MaxText target shape; pass `target_shapes`
  (MaxText key -> shape) to do the same here, otherwise hooks use their built-in shapes.

  Args:
    pt_state_dict: The state dictionary of the PyTorch reference module.
    hf_config: The Hugging Face configuration dictionary.
    maxtext_config: The MaxText configuration object / mock.
    prefix_filter: Optional prefix string to filter MaxText parameter keys.
    target_shapes: Optional MaxText key -> target shape, forwarded to the hooks.

  Returns:
    A dictionary mapping MaxText parameter keys to transformed NumPy arrays.
  """
  param_map = PARAM_MAPPING["kimi-k3"](hf_config, maxtext_config, scan_layers=False)
  hook_map = HOOK_FNS["kimi-k3"](hf_config, maxtext_config, scan_layers=False, saving_to_hf=False)
  target_shapes = target_shapes or {}

  def _convert(mt_key, hf_key):
    matching_key = _find_state_dict_key(hf_key, pt_state_dict)
    if matching_key is None:
      return None
    tensor_np = _to_numpy(pt_state_dict[matching_key])
    if mt_key in hook_map:
      tensor_np = hook_map[mt_key](tensor_np, target_shapes.get(mt_key))
    return tensor_np

  converted_params = {}
  for mt_key, hf_target in param_map.items():
    if prefix_filter and not mt_key.startswith(prefix_filter):
      continue
    if isinstance(hf_target, str):
      tensor_np = _convert(mt_key, hf_target)
      if tensor_np is not None:
        converted_params[mt_key] = tensor_np
    else:  # Routed experts: one HF tensor per expert, stacked on a leading expert axis.
      stacked = [t for t in (_convert(mt_key, k) for k in hf_target) if t is not None]
      if stacked:
        converted_params[mt_key] = np.stack(stacked, axis=0)

  return converted_params


def _path_to_key(path) -> str:
  """Flattens a PyTree path tuple into a hyphen-delimited parameter key."""
  parts = []
  for p in path:
    if hasattr(p, "key"):
      parts.append(str(p.key))
    elif hasattr(p, "name"):
      parts.append(str(p.name))
    elif hasattr(p, "idx"):
      parts.append(str(p.idx))
    else:
      parts.append(str(p))
  return "-".join(parts)


def load_params_into_nnx(
    module: nnx.Module,
    converted: dict[str, np.ndarray],
    prefix: str,
    *,
    strict: bool = True,
) -> list[str]:
  """Copies converted weights into `module` by walking its `nnx.Param` tree.

  Each `nnx.Param` reachable from `module` at path `a.b.c` is matched against
  `f"{prefix}-a-b-c"` in `converted`. Returns the list of keys loaded. With
  `strict=True`, raises if any param under `prefix` has no converted weight or if any
  converted key under `prefix` was never consumed.
  """
  loaded: list[str] = []
  missing: list[str] = []
  for path, node in nnx.iter_graph(module):
    if not isinstance(node, nnx.Param):
      continue
    key = f"{prefix}-{_path_to_key(path)}"
    if key not in converted:
      missing.append(key)
      continue
    src = np.asarray(converted[key])
    dst_shape = tuple(node.shape)
    if tuple(src.shape) != dst_shape:
      raise ValueError(f"shape mismatch for {key}: converted {src.shape} vs module {dst_shape}")
    node[...] = jnp.asarray(src, dtype=node.dtype)
    loaded.append(key)

  if strict:
    if missing:
      raise KeyError(f"no converted weight for module params: {missing}")
    unconsumed = sorted(k for k in converted if k.startswith(prefix + "-") and k not in loaded)
    if unconsumed:
      raise KeyError(f"converted keys never loaded into module: {unconsumed}")
  return loaded

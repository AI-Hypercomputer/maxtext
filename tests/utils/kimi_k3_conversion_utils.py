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
import ml_dtypes
import numpy as np

from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING
from maxtext.layers import moe

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


_NON_PARAM_COLLECTION_PREFIXES = ("MoEBiasVar-",)


def _to_params_key(mt_key: str) -> str:
  """`MoEBiasVar-decoder-...` -> `params-decoder-...`; other keys unchanged."""
  for collection_prefix in _NON_PARAM_COLLECTION_PREFIXES:
    if mt_key.startswith(collection_prefix):
      return "params-" + mt_key[len(collection_prefix) :]
  return mt_key


def _to_numpy(tensor: Any) -> np.ndarray:
  """torch.Tensor (or array-like) -> NumPy array of the same dtype, without a top-level torch import.

  torch bfloat16 has no NumPy dtype, so it is reinterpreted bit-for-bit as `ml_dtypes.bfloat16`
  rather than upcast, keeping bf16 weight rounding visible to the parity tests.
  """
  if hasattr(tensor, "detach"):
    tensor = tensor.detach().cpu()
    if str(tensor.dtype) == "torch.bfloat16":
      import torch  # pylint: disable=import-outside-toplevel  # already loaded: `tensor` is a torch.Tensor

      return tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return tensor.numpy()
  return np.asarray(tensor)


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
    A dictionary mapping MaxText parameter keys to transformed NumPy arrays. Keys of non-`nnx.Param`
    collections (`MoEBiasVar-...`) are returned under `params-...`, i.e. by module path.
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
  for map_key, hf_target in param_map.items():
    # Non-`nnx.Param` collections (Kimi-K3's `MoEBiasVar` router bias) are keyed under `params-` here,
    # so callers can address every weight by its module path (see `load_params_into_nnx`).
    mt_key = _to_params_key(map_key)
    if prefix_filter and not mt_key.startswith(prefix_filter):
      continue
    if isinstance(hf_target, str):
      tensor_np = _convert(map_key, hf_target)
      if tensor_np is not None:
        converted_params[mt_key] = tensor_np
    else:  # Routed experts: one HF tensor per expert, stacked on a leading expert axis.
      stacked = [t for t in (_convert(map_key, k) for k in hf_target) if t is not None]
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
    # Kimi-K3's router bias is a `moe.MoEBiasVar` (outside `nnx.Param`, like DeepSeek's).
    if not isinstance(node, (nnx.Param, moe.MoEBiasVar)):
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


def convert_hf_state_dict_via_to_maxtext(
    pt_state_dict: Mapping[str, Any],
    hf_config: dict[str, Any],
    maxtext_config: Any,
    target_shapes: Mapping[str, tuple[int, ...]],
    *,
    lazy: bool = False,
) -> dict[str, np.ndarray]:
  """Converts a full HF state dict through the production `to_maxtext` per-parameter loop.

  Unlike `convert_pytorch_module_to_maxtext_params` (unscanned only, test-side stacking), this
  runs `to_maxtext`'s own loaders (`_get_hf_loading_function` / `_get_maxtext_weight`), so it
  covers the scanned layout (`maxtext_config.scan_layers=True`): layer stacking on
  `param_scan_axis`, nested (expert, block) stacking and the `MoEBiasVar` collection.

  Args:
    pt_state_dict: Full HF state dict (keys with or without the `language_model.` prefix).
    hf_config: The Hugging Face (text) config dict.
    maxtext_config: MaxText config; `scan_layers` selects the layout.
    target_shapes: MaxText checkpoint key (with collection prefix, e.g. `MoEBiasVar-...`) -> shape,
      as returned by `to_maxtext.get_maxtext_model_info`.
    lazy: Build `LazyTensor`s (the `--lazy_load_tensors` path) and materialize them afterwards.

  Returns:
    MaxText checkpoint key (with collection prefix) -> converted NumPy array.
  """
  # pylint: disable=import-outside-toplevel,protected-access
  from maxtext.checkpoint_conversion import to_maxtext
  from maxtext.checkpoint_conversion.utils.utils import validate_and_filter_param_map_keys

  scan_layers = bool(maxtext_config.scan_layers)
  param_map = PARAM_MAPPING["kimi-k3"](hf_config, maxtext_config, scan_layers=scan_layers)
  hook_map = HOOK_FNS["kimi-k3"](hf_config, maxtext_config, scan_layers=scan_layers, saving_to_hf=False)
  keys = list(target_shapes)
  index = {k: (i, tuple(target_shapes[k])) for i, k in enumerate(keys)}

  def getter(hf_key, dtype=None):
    del dtype
    matching_key = _find_state_dict_key(hf_key, pt_state_dict)
    if matching_key is None:
      raise KeyError(f"HF key {hf_key} not in the state dict")
    return _to_numpy(pt_state_dict[matching_key]).astype(np.float32)

  weights = [None] * len(keys)
  for mt_key in validate_and_filter_param_map_keys(param_map.keys(), set(keys)):
    idx, shape = index[mt_key]
    load_fn = to_maxtext._get_hf_loading_function(
        param_map[mt_key], getter, hook_map.get(mt_key), shape, maxtext_config, mt_key
    )
    to_maxtext._get_maxtext_weight(load_fn, idx, shape, mt_key, weights, "float32", lazy)
  return {k: np.asarray(w) for k, w in zip(keys, weights)}


def load_checkpoint_params_into_nnx(module: nnx.Module, converted: Mapping[str, np.ndarray]) -> list[str]:
  """`load_params_into_nnx` for checkpoint-keyed weights (`params-...` / `MoEBiasVar-...` keys)."""
  return load_params_into_nnx(module, {_to_params_key(k): v for k, v in converted.items()}, prefix="params")

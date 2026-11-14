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

"""Test helpers: import the Ming-Image-Ant PyTorch reference (written for transformers 4.57) under newer transformers.

The reference repo is `MING_IMAGE_REPO` (default `~/workspace/diffusion/ming/Ming-Image-Ant`). `import_reference()`
installs small compatibility shims (symbols removed in transformers 5) and returns the reference modules, or None when
the repo / torch / transformers are unavailable. Also loads NNX models from HF-keyed state dicts through the
`*_param_mapping` functions exactly like `to_maxtext_diffusers.convert_diffusers_component`.
"""

import importlib
import os
import sys

from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict
import jax.numpy as jnp
import numpy as np

REPO = os.environ.get("MING_IMAGE_REPO", os.path.expanduser("~/workspace/diffusion/ming/Ming-Image-Ant"))


def _default_rope_init(config, device=None, seq_len=None):  # pylint: disable=unused-argument
  """transformers 4.57 `_compute_default_rope_parameters` (removed from ROPE_INIT_FUNCTIONS in transformers 5)."""
  import torch  # pylint: disable=import-outside-toplevel

  base = getattr(config, "rope_theta", None)
  if base is None:  # transformers 5 configs keep it in `rope_parameters`.
    base = config.rope_parameters["rope_theta"]
  partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
  head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
  dim = int(head_dim * partial_rotary_factor)
  inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
  return inv_freq, 1.0


def _install_shims():
  """Restores transformers-4.x symbols the reference imports."""
  import transformers  # pylint: disable=import-outside-toplevel
  from transformers import modeling_rope_utils, pytorch_utils  # pylint: disable=import-outside-toplevel
  from transformers.utils import import_utils  # pylint: disable=import-outside-toplevel

  if not hasattr(import_utils, "is_torch_fx_available"):
    import_utils.is_torch_fx_available = lambda: False
  for name, value in (("ALL_LAYERNORM_LAYERS", []), ("is_torch_greater_or_equal_than_1_13", True)):
    if not hasattr(pytorch_utils, name):
      setattr(pytorch_utils, name, value)
  if not hasattr(transformers.utils, "is_flash_attn_greater_or_equal_2_10"):
    transformers.utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
  if "default" not in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
  if REPO not in sys.path:
    sys.path.insert(0, REPO)


def import_reference(*module_names):
  """The reference modules (e.g. "modeling_bailing_moe_v2"), or None if unavailable."""
  if not os.path.isdir(REPO):
    return None
  try:
    import torch  # pylint: disable=import-outside-toplevel,unused-import
    import transformers  # pylint: disable=import-outside-toplevel,unused-import
  except ImportError:
    return None
  _install_shims()
  return tuple(importlib.import_module(name) for name in module_names)


def load_with_mapping(model, mapping_and_hooks, hf_state: dict) -> set:
  """Fills `model`'s params from `hf_state` (`{hf_key: np.ndarray}`) via a `(mapping, hooks)` pair.

  Mirrors `convert_diffusers_component`: list values are loaded one key at a time (hook with `target_shape[1:]`) and
  stacked along axis 0. Asserts that every model param is covered and shapes match. Returns the HF keys used.
  """
  mapping, hooks = mapping_and_hooks
  params = nnx.state(model, nnx.Param).to_pure_dict()
  flat = flatten_dict(params)
  used, new_flat = set(), {}
  for path, leaf in flat.items():
    key = "-".join(map(str, path))
    assert key in mapping, f"model param {key} not covered by the mapping"
    hook = hooks.get(key)
    hf_keys = mapping[key]

    def load(hf_key, target_shape, hook=hook):
      used.add(hf_key)
      weight = np.asarray(hf_state[hf_key])
      return hook(weight, target_shape) if hook is not None else weight

    if isinstance(hf_keys, list):
      value = np.stack([load(k, leaf.shape[1:]) for k in hf_keys], axis=0)
    else:
      value = load(hf_keys, leaf.shape)
    assert value.shape == tuple(leaf.shape), f"{key}: {value.shape} vs {leaf.shape}"
    new_flat[path] = jnp.asarray(value, dtype=leaf.dtype)
  nnx.update(model, unflatten_dict(new_flat))
  return used


def torch_state(module, prefix: str = "") -> dict:
  """`{prefix + name: float32 numpy}` of a torch module's state dict."""
  return {prefix + k: v.detach().float().cpu().numpy() for k, v in module.state_dict().items()}

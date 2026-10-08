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

from typing import Any, Dict, Optional
import numpy as np
import torch

from maxtext.checkpoint_conversion.utils.param_mapping import HOOK_FNS, PARAM_MAPPING


def convert_pytorch_module_to_maxtext_params(  # pylint: disable=too-many-nested-blocks
    pt_state_dict: Dict[str, torch.Tensor],
    hf_config: Dict[str, Any],
    maxtext_config: Any = None,
    prefix_filter: Optional[str] = None,
) -> Dict[str, np.ndarray]:
  """Converts a PyTorch module state_dict to a MaxText parameter dictionary.

  Uses the exact KIMI_K3_MAXTEXT_TO_HF_PARAM_MAPPING and
  KIMI_K3_MAXTEXT_TO_HF_PARAM_HOOK_FN from `param_mapping.py` to ensure that
  both model forward passes and parameter conversions are tested simultaneously.

  Args:
    pt_state_dict: The state dictionary of the PyTorch reference module.
    hf_config: The Hugging Face configuration dictionary.
    maxtext_config: The MaxText configuration object / mock.
    prefix_filter: Optional prefix string to filter MaxText parameter keys.

  Returns:
    A dictionary mapping MaxText parameter keys to transformed NumPy arrays.
  """
  mapping_fn = PARAM_MAPPING["kimi-k3"]
  hook_fn_factory = HOOK_FNS["kimi-k3"]

  param_map = mapping_fn(hf_config, maxtext_config, scan_layers=False)
  hook_map = hook_fn_factory(hf_config, maxtext_config, scan_layers=False, saving_to_hf=False)

  converted_params = {}

  for mt_key, hf_target in param_map.items():
    if prefix_filter and not mt_key.startswith(prefix_filter):
      continue

    # Handle atomic string mapping
    if isinstance(hf_target, str):
      matching_key = None
      if hf_target in pt_state_dict:
        matching_key = hf_target
      elif hf_target.startswith("language_model.") and hf_target[len("language_model.") :] in pt_state_dict:
        matching_key = hf_target[len("language_model.") :]
      else:
        for k in pt_state_dict.keys():
          if (
              hf_target == k
              or hf_target.endswith("." + k)
              or hf_target.endswith(k)
              or ("." in hf_target and ".".join(hf_target.split(".")[-2:]) == k)
          ):
            matching_key = k
            break

      if matching_key is not None and matching_key in pt_state_dict:
        tensor_np = pt_state_dict[matching_key].detach().cpu().float().numpy()
        if mt_key in hook_map:
          tensor_np = hook_map[mt_key](tensor_np)
        converted_params[mt_key] = tensor_np

    # Handle expert stacking (list of strings)
    elif isinstance(hf_target, list):
      stacked_tensors = []
      for expert_hf_key in hf_target:
        matching_key = None
        if expert_hf_key in pt_state_dict:
          matching_key = expert_hf_key
        elif expert_hf_key.startswith("language_model.") and expert_hf_key[len("language_model.") :] in pt_state_dict:
          matching_key = expert_hf_key[len("language_model.") :]
        else:
          for k in pt_state_dict.keys():
            if expert_hf_key.endswith(k):
              matching_key = k
              break
        if matching_key is not None and matching_key in pt_state_dict:
          tensor_np = pt_state_dict[matching_key].detach().cpu().float().numpy()
          if mt_key in hook_map:
            tensor_np = hook_map[mt_key](tensor_np)
          stacked_tensors.append(tensor_np)
      if stacked_tensors:
        converted_params[mt_key] = np.stack(stacked_tensors, axis=0)

  return converted_params

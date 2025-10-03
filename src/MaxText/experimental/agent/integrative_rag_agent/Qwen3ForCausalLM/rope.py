
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file contains the ROPE_INIT_FUNCTIONS mapping, which is a dictionary that maps the "rope_type" string field
in the rope config to the corresponding function to compute the RoPE parameters from the model config.
"""

# RoPE parameter computation functions reused from Qwen3ForCausalLM project.
# See:
# - Qwen3ForCausalLM.modeling_utils._compute_default_rope_parameters
# - Qwen3ForCausalLM.rope_utils._compute_linear_scaling_rope_parameters
# - Qwen3ForCausalLM.modeling_utils._compute_dynamic_ntk_parameters
# - Qwen3ForCausalLM.modeling_utils._compute_yarn_parameters
# - Qwen3ForCausalLM.positional_embeddings._compute_longrope_parameters
# - Qwen3ForCausalLM.modeling_utils._compute_llama3_parameters
from .modeling_utils import (
    _compute_default_rope_parameters,
    _compute_dynamic_ntk_parameters,
    _compute_llama3_parameters,
    _compute_yarn_parameters,
)
from .positional_embeddings import _compute_longrope_parameters
from .rope_utils import _compute_linear_scaling_rope_parameters


# This maps the "rope_type" string field in rope config to the corresponding function to compute the RoPE parameters
# from the model config. You can append new {'rope_type': callable} pairs to this dictionary to enable custom RoPE
# parameterizations, as long as the callable has the same signature.
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
}

from typing import Optional

from .. import max_logging
from ..common_types import Config as PretrainedConfig
from .rope import _check_received_keys


def _validate_linear_scaling_rope_parameters(config: PretrainedConfig, ignore_keys: Optional[set] = None):
  """Validates the linear scaling RoPE parameters in the config."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    max_logging.log(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RoPE utils."""

from typing import Optional

from MaxText.common_types import Config
from MaxText import max_logging as logging
from .rope_utils import ROPE_VALIDATION_FUNCTIONS


def rope_config_validation(config: Config, ignore_keys: Optional[set] = None):
  """
  Validate the RoPE config arguments, given a `Config` object
  """
  rope_scaling = getattr(config, "rope_scaling", None)  # not a default parameter in `Config`
  if rope_scaling is None:
    return

  # BC: "rope_type" was originally "type"
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
  validation_fn = ROPE_VALIDATION_FUNCTIONS.get(rope_type)
  if validation_fn is not None:
    validation_fn(config, ignore_keys=ignore_keys)
  else:
    logging.warning(
        f"Missing validation function mapping in `ROPE_VALIDATION_FUNCTIONS` for 'rope_type'='{rope_type}'"
    )


# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""RoPE configuration validation"""

import logging
from typing import Optional, Set

from MaxText.common_types import Config
from MaxText import max_logging


def _check_received_keys(
    rope_type: str,
    received_keys: Set[str],
    required_keys: Set[str],
    optional_keys: Optional[Set[str]] = None,
    ignore_keys: Optional[Set[str]] = None,
):
    """Compare the received keys in `config.rope_scaling` against the expected and optional keys"""
    # BC: "rope_type" was originally "type" -- let's check for "rope_type" when "type" is present
    if "type" in received_keys:
        received_keys -= {"type"}
        required_keys.add("rope_type")

    # Some models need to store model-specific keys, and we don't want to throw warning at them
    if ignore_keys is not None:
        received_keys -= ignore_keys

    missing_keys = required_keys - received_keys
    if missing_keys:
        raise KeyError(f"Missing required keys in `rope_scaling` for 'rope_type'='{rope_type}': {missing_keys}")

    if optional_keys is not None:
        unused_keys = received_keys - required_keys - optional_keys
    else:
        unused_keys = received_keys - required_keys
    if unused_keys:
        max_logging.log(
            level=logging.WARNING, f"Unrecognized keys in `rope_scaling` for 'rope_type'='{rope_type}': {unused_keys}"
        )


def _validate_default_rope_parameters(config: Config, ignore_keys: Optional[Set[str]] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)


def _validate_linear_scaling_rope_parameters(config: Config, ignore_keys: Optional[Set[str]] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        max_logging.log(level=logging.WARNING, f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")


def _validate_dynamic_scaling_rope_parameters(config: Config, ignore_keys: Optional[Set[str]] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {"original_max_position_embeddings"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        max_logging.log(level=logging.WARNING, f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")


def _validate_yarn_parameters(config: Config, ignore_keys: Optional[Set[str]] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    optional_keys = {
        "attention_factor",
        "beta_fast",
        "beta_slow",
        "original_max_position_embeddings",
        "mscale",
        "mscale_all_dim",
        "truncate",
    }
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        max_logging.log(level=logging.WARNING, f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
        max_logging.log(
            level=logging.WARNING,
            f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}",
        )
    beta_fast = rope_scaling.get("beta_fast")
    if beta_fast is not None and not isinstance(beta_fast, float):
        max_logging.log(level=logging.WARNING, f"`rope_scaling`'s beta_fast field must be a float, got {beta_fast}")
    beta_slow = rope_scaling.get("beta_slow")
    if beta_slow is not None and not isinstance(beta_slow, float):
        max_logging.log(level=logging.WARNING, f"`rope_scaling`'s beta_slow field must be a float, got {beta_slow}")

    if (beta_fast or 32) < (beta_slow or 1):
        max_logging.log(
            level=logging.WARNING,
            f"`rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
            f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)",
        )


def _validate_longrope_parameters(config: Config, ignore_keys: Optional[Set[str]] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "short_factor", "long_factor"}
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {"attention_factor", "factor", "original_max_position_embeddings"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.emb_dim // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    short_factor = rope_scaling.get("short_factor")
    if not isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor):
        max_logging.log(
            level=logging.WARNING, f"`rope_scaling`'s short_factor field must be a list of numbers, got {short_factor}"
        )
    if len(short_factor) != dim // 2:
        max_logging.log(
            level=logging.WARNING,
            f"`rope_scaling`'s short_factor field must have length {dim // 2}, got {len(short_factor)}",
        )

    long_factor = rope_scaling.get("long_factor")
    if not isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor):
        max_logging.log(
            level=logging.WARNING, f"`rope_scaling`'s long_factor field must be a list of numbers, got {long_factor}"
        )
    if len(long_factor) != dim // 2:
        max_logging.log(
            level=logging.WARNING,
            f"`rope_scaling`'s long_factor field must have length {dim // 2}, got {len(long_factor)}",
        )

    # Handle Phi3 divergence: prefer the use of `attention_factor` and/or `factor` over
    # `original_max_position_embeddings` to compute internal variables. The latter lives outside `rope_scaling` and is
    # unique to longrope (= undesirable)
    if hasattr(config, "original_max_position_embeddings"):
        max_logging.log(
            level=logging.WARNING,
            "This model has set a `original_max_position_embeddings` field, to be used together with "
            "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_scaling`"
            "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
            "as it is compatible with most model architectures.",
        )
    else:
        factor = rope_scaling.get("factor")
        if factor is None:
            max_logging.log(level=logging.WARNING, "Missing required keys in `rope_scaling`: 'factor'")
        elif not isinstance(factor, float) or factor < 1.0:
            max_logging.log(level=logging.WARNING, f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

        attention_factor = rope_scaling.get("attention_factor")
        if attention_factor is not None:
            if not isinstance(attention_factor, float) or attention_factor < 0.0:
                max_logging.log(
                    level=logging.WARNING,
                    f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}",
                )


def _validate_llama3_parameters(config: Config, ignore_keys: Optional[Set[str]] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        max_logging.log(level=logging.WARNING, f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    if low_freq_factor is None or not isinstance(low_freq_factor, float):
        max_logging.log(
            level=logging.WARNING, f"`rope_scaling`'s low_freq_factor field must be a float, got {low_freq_factor}"
        )
    if high_freq_factor is None or not isinstance(high_freq_factor, float):
        max_logging.log(
            level=logging.WARNING, f"`rope_scaling`'s high_freq_factor field must be a float, got {high_freq_factor}"
        )
    if high_freq_factor <= low_freq_factor:
        max_logging.log(
            level=logging.WARNING,
            "`rope_scaling`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
            f"{high_freq_factor} and low_freq_factor={low_freq_factor}",
        )

    original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
    if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
        max_logging.log(
            level=logging.WARNING,
            "`rope_scaling`'s original_max_position_embeddings field must be an integer, got "
            f"{original_max_position_embeddings}",
        )
    if original_max_position_embeddings >= config.max_target_length:
        max_logging.log(
            level=logging.WARNING,
            "`rope_scaling`'s original_max_position_embeddings field must be less than max_target_length, got "
            f"{original_max_position_embeddings} and max_target_length={config.max_target_length}",
        )


# Like `ROPE_INIT_FUNCTIONS`, this validation function mapping can be dynamically updated for custom RoPE types.
ROPE_VALIDATION_FUNCTIONS = {
    "default": _validate_default_rope_parameters,
    "linear": _validate_linear_scaling_rope_parameters,
    "dynamic": _validate_dynamic_scaling_rope_parameters,
    "yarn": _validate_yarn_parameters,
    "longrope": _validate_longrope_parameters,
    "llama3": _validate_llama3_parameters,
}


def rope_config_validation(config: Config, ignore_keys: Optional[Set[str]] = None):
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
        max_logging.log(
            level=logging.WARNING,
            f"Missing validation function mapping in `ROPE_VALIDATION_FUNCTIONS` for 'rope_type'='{rope_type}'",
        )

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
""" RoPE configuration validation functions """

from typing import Optional, Set

from MaxText import max_logging
from MaxText import max_utils


def _check_received_keys(
    rope_type: str,
    received_keys: Set,
    required_keys: Set,
    optional_keys: Optional[Set] = None,
    ignore_keys: Optional[Set] = None,
):
  """Compare the received keys in `config.rope_scaling` against the expected and optional keys."""
  # BC: "rope_type" was originally "type" -- let's check for "rope_type" when "type" is present
  if "type" in received_keys:
    received_keys -= {"type"}
    required_keys.add("rope_type")

  # Some models need to store model-specific keys, and we don't want to throw warning at them
  if ignore_keys is not None:
    received_keys -= ignore_keys

  missing_keys = required_keys - received_keys
  if missing_keys:
    raise KeyError(f"Missing required keys in `rope_scaling` for 'rope_type'='{rope_type}': {missing_keys}")

  if optional_keys is not None:
    unused_keys = received_keys - required_keys - optional_keys
  else:
    unused_keys = received_keys - required_keys
  if unused_keys:
    max_logging.log(f"Unrecognized keys in `rope_scaling` for 'rope_type'='{rope_type}': {unused_keys}")


def _validate_default_rope_parameters(config: max_utils.Config, ignore_keys: Optional[Set] = None):
  """Validates default RoPE parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)


def _validate_linear_scaling_rope_parameters(config: max_utils.Config, ignore_keys: Optional[Set] = None):
  """Validates linear scaling RoPE parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    max_logging.log(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")


def _validate_dynamic_scaling_rope_parameters(config: max_utils.Config, ignore_keys: Optional[Set] = None):
  """Validates dynamic scaling RoPE parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor"}
  # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
  optional_keys = {"original_max_position_embeddings"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    max_logging.log(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")


def _validate_yarn_parameters(config: max_utils.Config, ignore_keys: Optional[Set] = None):
  """Validates YaRN RoPE parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor"}
  optional_keys = {
      "attention_factor",
      "beta_fast",
      "beta_slow",
      "original_max_position_embeddings",
      "mscale",
      "mscale_all_dim",
      "truncate",
  }
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    max_logging.log(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

  attention_factor = rope_scaling.get("attention_factor")
  if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
    max_logging.log(f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}")
  beta_fast = rope_scaling.get("beta_fast")
  if beta_fast is not None and not isinstance(beta_fast, float):
    max_logging.log(f"`rope_scaling`'s beta_fast field must be a float, got {beta_fast}")
  beta_slow = rope_scaling.get("beta_slow")
  if beta_slow is not None and not isinstance(beta_slow, float):
    max_logging.log(f"`rope_scaling`'s beta_slow field must be a float, got {beta_slow}")

  if (beta_fast or 32) < (beta_slow or 1):
    max_logging.log(
        f"`rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
        f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
    )


def _validate_longrope_parameters(config: max_utils.Config, ignore_keys: Optional[Set] = None):
  """Validates LongRoPE parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "short_factor", "long_factor"}
  # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
  optional_keys = {"attention_factor", "factor", "original_max_position_embeddings"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

  partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
  head_dim = getattr(config, "head_dim", config.emb_dim // config.num_query_heads)
  dim = int(head_dim * partial_rotary_factor)

  short_factor = rope_scaling.get("short_factor")
  if not isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor):
    max_logging.log(f"`rope_scaling`'s short_factor field must be a list of numbers, got {short_factor}")
  if len(short_factor) != dim // 2:
    max_logging.log(f"`rope_scaling`'s short_factor field must have length {dim // 2}, got {len(short_factor)}")

  long_factor = rope_scaling.get("long_factor")
  if not isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor):
    max_logging.log(f"`rope_scaling`'s long_factor field must be a list of numbers, got {long_factor}")
  if len(long_factor) != dim // 2:
    max_logging.log(f"`rope_scaling`'s long_factor field must have length {dim // 2}, got {len(long_factor)}")

  # Handle Phi3 divergence: prefer the use of `attention_factor` and/or `factor` over
  # `original_max_position_embeddings` to compute internal variables. The latter lives outside `rope_scaling` and is
  # unique to longrope (= undesirable)
  if hasattr(config, "original_max_position_embeddings"):
    # MaxText does not have a log_once equivalent.
    max_logging.log(
        "This model has set a `original_max_position_embeddings` field, to be used together with "
        "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_scaling`"
        "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
        "as it is compatible with most model architectures."
    )
  else:
    factor = rope_scaling.get("factor")
    if factor is None:
      max_logging.log("Missing required keys in `rope_scaling`: 'factor'")
    elif not isinstance(factor, float) or factor < 1.0:
      max_logging.log(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None:
      if not isinstance(attention_factor, float) or attention_factor < 0.0:
        max_logging.log(
            f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )


def _validate_llama3_parameters(config: max_utils.Config, ignore_keys: Optional[Set] = None):
  """Validates Llama3 RoPE parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    max_logging.log(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

  low_freq_factor = rope_scaling["low_freq_factor"]
  high_freq_factor = rope_scaling["high_freq_factor"]
  if low_freq_factor is None or not isinstance(low_freq_factor, float):
    max_logging.log(f"`rope_scaling`'s low_freq_factor field must be a float, got {low_freq_factor}")
  if high_freq_factor is None or not isinstance(high_freq_factor, float):
    max_logging.log(f"`rope_scaling`'s high_freq_factor field must be a float, got {high_freq_factor}")
  if high_freq_factor <= low_freq_factor:
    max_logging.log(
        "`rope_scaling`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
        f"{high_freq_factor} and low_freq_factor={low_freq_factor}"
    )

  original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
  if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
    max_logging.log(
        "`rope_scaling`'s original_max_position_embeddings field must be an integer, got "
        f"{original_max_position_embeddings}"
    )
  if original_max_position_embeddings >= config.max_position_embeddings:
    max_logging.log(
        "`rope_scaling`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
        f"{original_max_position_embeddings} and max_position_embeddings={config.max_position_embeddings}"
    )


# Like `ROPE_INIT_FUNCTIONS`, this validation function mapping can be dynamically updated for custom RoPE types.
ROPE_VALIDATION_FUNCTIONS = {
    "default": _validate_default_rope_parameters,
    "linear": _validate_linear_scaling_rope_parameters,
    "dynamic": _validate_dynamic_scaling_rope_parameters,
    "yarn": _validate_yarn_parameters,
    "longrope": _validate_longrope_parameters,
    "llama3": _validate_llama3_parameters,
}


def rope_config_validation(config: max_utils.Config, ignore_keys: Optional[Set] = None):
  """Validate the RoPE config arguments, given a `Config` object."""
  rope_scaling = getattr(config, "rope_scaling", None)  # not a default parameter in `Config`
  if rope_scaling is None:
    return

  # BC: "rope_type" was originally "type"
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
  validation_fn = ROPE_VALIDATION_FUNCTIONS.get(rope_type)
  if validation_fn is not None:
    validation_fn(config, ignore_keys=ignore_keys)
  else:
    max_logging.log(
        f"Missing validation function mapping in `ROPE_VALIDATION_FUNCTIONS` for 'rope_type'='{rope_type}'"
    )

import jax.numpy as jnp
from jax import Array


def rotate_half(x: Array) -> Array:
  """Rotates half the hidden dims of the input."""
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return jnp.concatenate((-x2, x1), axis=-1)

# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
""" Flax Qwen3Moe model."""

from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array

# from ..rope_utils import rotate_half - Reused from generated_code.Qwen3MoeForCausalLM.rope_utils.rotate_half
from ..rope_utils import rotate_half


def apply_rotary_pos_emb(
    q: Array,
    k: Array,
    cos: Array,
    sin: Array,
    position_ids: Optional[Array] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[Array, Array]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`jax.Array`): The query tensor.
        k (`jax.Array`): The key tensor.
        cos (`jax.Array`): The cosine part of the rotary embedding.
        sin (`jax.Array`): The sine part of the rotary embedding.
        position_ids (`jax.Array`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos and
            sin so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos and sin have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos and sin broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(jax.Array)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # The position_ids argument is deprecated and unused.
    # This is to align with the original implementation.
    del position_ids

    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
""" RoPE scaling utilities"""
from typing import Optional, Set

from MaxText import max_logging as logging


logger = logging.get_logger(__name__)


def _check_received_keys(
    rope_type: str,
    received_keys: Set[str],
    required_keys: Set[str],
    optional_keys: Optional[Set[str]] = None,
    ignore_keys: Optional[Set[str]] = None,
):
  """Compare the received keys in `config.rope_scaling` against the expected and optional keys"""
  # BC: "rope_type" was originally "type" -- let's check for "rope_type" when "type" is present
  if "type" in received_keys:
    received_keys -= {"type"}
    required_keys.add("rope_type")

  # Some models need to store model-specific keys, and we don't want to throw warning at them
  if ignore_keys is not None:
    received_keys -= ignore_keys

  missing_keys = required_keys - received_keys
  if missing_keys:
    raise KeyError(
        f"Missing required keys in `rope_scaling` for 'rope_type'='{rope_type}': {missing_keys}"
    )

  if optional_keys is not None:
    unused_keys = received_keys - required_keys - optional_keys
  else:
    unused_keys = received_keys - required_keys
  if unused_keys:
    logger.warning(
        f"Unrecognized keys in `rope_scaling` for 'rope_type'='{rope_type}': {unused_keys}"
    )

from typing import Optional

from MaxText.common_types import Config


def _validate_default_rope_parameters(config: Config, ignore_keys: Optional[set] = None):
  """Validate default RoPE parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

from typing import Optional, Set
from absl import logging

from maxtext.common_types import Config
# from maxtext.layers.rope_utils import _check_received_keys is an implicit dependency
# and is assumed to be in the same file.


def _validate_dynamic_scaling_rope_parameters(config: Config, ignore_keys: Optional[Set[str]] = None):
  """Validate the dynamic scaling RoPE parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor"}
  # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
  optional_keys = {"original_max_position_embeddings"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    logging.warning("`rope_scaling`'s factor field must be a float >= 1, got %s", factor)

from typing import Optional, Set

from absl import logging
from maxtext.common_types import PretrainedConfig
# from maxtext.layers.rope import _check_received_keys, assumed to be in the same file or a utility file.


def _validate_linear_scaling_rope_parameters(config: PretrainedConfig, ignore_keys: Optional[Set[str]] = None):
  """Validates parameters for linear scaling RoPE."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    logging.warning("`rope_scaling`'s factor field must be a float >= 1, got %s", factor)

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""
This file is a JAX version of the huggingface transformers file at
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
"""
from typing import Optional, Set

from absl import logging

from maxtext.common_types import Config
# from maxtext.layers.rope_utils import _check_received_keys is used
# but not explicitly imported as per MaxText style.
# Assuming it is available in the same module or a common utility file.
from .rope_utils import _check_received_keys


def _validate_llama3_parameters(config: Config, ignore_keys: Optional[Set[str]] = None):
  """Validates Llama3 parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    logging.warning("`rope_scaling`'s factor field must be a float >= 1, got %s", factor)

  low_freq_factor = rope_scaling["low_freq_factor"]
  high_freq_factor = rope_scaling["high_freq_factor"]
  if low_freq_factor is None or not isinstance(low_freq_factor, float):
    logging.warning("`rope_scaling`'s low_freq_factor field must be a float, got %s", low_freq_factor)
  if high_freq_factor is None or not isinstance(high_freq_factor, float):
    logging.warning("`rope_scaling`'s high_freq_factor field must be a float, got %s", high_freq_factor)
  if high_freq_factor <= low_freq_factor:
    logging.warning(
        "`rope_scaling`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor=%s and"
        " low_freq_factor=%s",
        high_freq_factor,
        low_freq_factor,
    )

  original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
  if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
    logging.warning(
        "`rope_scaling`'s original_max_position_embeddings field must be an integer, got %s",
        original_max_position_embeddings,
    )
  if original_max_position_embeddings >= config.max_position_embeddings:
    logging.warning(
        "`rope_scaling`'s original_max_position_embeddings field must be less than max_position_embeddings, got %s and"
        " max_position_embeddings=%s",
        original_max_position_embeddings,
        config.max_position_embeddings,
    )

from typing import Optional, Set

from absl import logging

from MaxText.common_types import Config
# Assuming _check_received_keys is available in the same module.
# This function is defined in the same source file and is expected to be converted as well.
from . import rope_utils


def _validate_longrope_parameters(config: Config, ignore_keys: Optional[Set[str]] = None):
  """Validates the configuration parameters for longrope RoPE scaling."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "short_factor", "long_factor"}
  # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
  optional_keys = {"attention_factor", "factor", "original_max_position_embeddings"}
  received_keys = set(rope_scaling.keys())
  rope_utils._check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

  partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
  head_dim = getattr(config, "head_dim", config.emb_dim // config.num_query_heads)
  dim = int(head_dim * partial_rotary_factor)

  short_factor = rope_scaling.get("short_factor")
  if not isinstance(short_factor, list) or not all(isinstance(x, (int, float)) for x in short_factor):
    logging.warning("`rope_scaling`'s short_factor field must be a list of numbers, got %s", short_factor)
  if len(short_factor) != dim // 2:
    logging.warning(
        "`rope_scaling`'s short_factor field must have length %d, got %d", dim // 2, len(short_factor)
    )

  long_factor = rope_scaling.get("long_factor")
  if not isinstance(long_factor, list) or not all(isinstance(x, (int, float)) for x in long_factor):
    logging.warning("`rope_scaling`'s long_factor field must be a list of numbers, got %s", long_factor)
  if len(long_factor) != dim // 2:
    logging.warning("`rope_scaling`'s long_factor field must have length %d, got %d", dim // 2, len(long_factor))

  # Handle Phi3 divergence: prefer the use of `attention_factor` and/or `factor` over
  # `original_max_position_embeddings` to compute internal variables. The latter lives outside `rope_scaling` and is
  # unique to longrope (= undesirable)
  if hasattr(config, "original_max_position_embeddings"):
    logging.warning(
        "This model has set a `original_max_position_embeddings` field, to be used together with "
        "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_scaling`"
        "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
        "as it is compatible with most model architectures."
    )
  else:
    factor = rope_scaling.get("factor")
    if factor is None:
      logging.warning("Missing required keys in `rope_scaling`: 'factor'")
    elif not isinstance(factor, float) or factor < 1.0:
      logging.warning("`rope_scaling`'s factor field must be a float >= 1, got %f", factor)

    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None:
      if not isinstance(attention_factor, float) or attention_factor < 0.0:
        logging.warning(
            "`rope_scaling`'s attention_factor field must be a float greater than 0, got %f", attention_factor
        )

from typing import Optional, Set
from absl import logging

from MaxText.common_types import Config
# The function `_check_received_keys` is assumed to be available in the same module.

def _validate_yarn_parameters(config: Config, ignore_keys: Optional[Set[str]] = None):
  """Validate YaRN parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor"}
  optional_keys = {
      "attention_factor",
      "beta_fast",
      "beta_slow",
      "original_max_position_embeddings",
      "mscale",
      "mscale_all_dim",
      "truncate",
  }
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    logging.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

  attention_factor = rope_scaling.get("attention_factor")
  if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
    logging.warning(
        f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
    )
  beta_fast = rope_scaling.get("beta_fast")
  if beta_fast is not None and not isinstance(beta_fast, float):
    logging.warning(f"`rope_scaling`'s beta_fast field must be a float, got {beta_fast}")
  beta_slow = rope_scaling.get("beta_slow")
  if beta_slow is not None and not isinstance(beta_slow, float):
    logging.warning(f"`rope_scaling`'s beta_slow field must be a float, got {beta_slow}")

  if (beta_fast or 32) < (beta_slow or 1):
    logging.warning(
        f"`rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
        f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
    )

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
""" RoPE validation functions """

from __future__ import annotations
from typing import Optional, Set

from absl import logging
from MaxText import max_utils


def _check_received_keys(
    rope_type: str,
    received_keys: Set[str],
    required_keys: Set[str],
    optional_keys: Optional[Set[str]] = None,
    ignore_keys: Optional[Set[str]] = None,
):
  """Compare the received keys in `config.rope_scaling` against the expected and optional keys"""
  # BC: "rope_type" was originally "type" -- let's check for "rope_type" when "type" is present
  if "type" in received_keys:
    received_keys -= {"type"}
    required_keys.add("rope_type")

  # Some models need to store model-specific keys, and we don't want to throw warning at them
  if ignore_keys is not None:
    received_keys -= ignore_keys

  missing_keys = required_keys - received_keys
  if missing_keys:
    raise KeyError(f"Missing required keys in `rope_scaling` for 'rope_type'='{rope_type}': {missing_keys}")

  if optional_keys is not None:
    unused_keys = received_keys - required_keys - optional_keys
  else:
    unused_keys = received_keys - required_keys
  if unused_keys:
    logging.warning("Unrecognized keys in `rope_scaling` for 'rope_type'='%s': %s", rope_type, unused_keys)


def _validate_default_rope_parameters(config: max_utils.Config, ignore_keys: Optional[Set[str]] = None):
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)


def _validate_linear_scaling_rope_parameters(config: max_utils.Config, ignore_keys: Optional[Set[str]] = None):
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    logging.warning("`rope_scaling`'s factor field must be a float >= 1, got %s", factor)


def _validate_dynamic_scaling_rope_parameters(config: max_utils.Config, ignore_keys: Optional[Set[str]] = None):
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor"}
  # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
  optional_keys = {"original_max_position_embeddings"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    logging.warning("`rope_scaling`'s factor field must be a float >= 1, got %s", factor)


def _validate_yarn_parameters(config: max_utils.Config, ignore_keys: Optional[Set[str]] = None):
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor"}
  optional_keys = {
      "attention_factor",
      "beta_fast",
      "beta_slow",
      "original_max_position_embeddings",
      "mscale",
      "mscale_all_dim",
      "truncate",
  }
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    logging.warning("`rope_scaling`'s factor field must be a float >= 1, got %s", factor)

  attention_factor = rope_scaling.get("attention_factor")
  if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
    logging.warning("`rope_scaling`'s attention_factor field must be a float greater than 0, got %s", attention_factor)
  beta_fast = rope_scaling.get("beta_fast")
  if beta_fast is not None and not isinstance(beta_fast, float):
    logging.warning("`rope_scaling`'s beta_fast field must be a float, got %s", beta_fast)
  beta_slow = rope_scaling.get("beta_slow")
  if beta_slow is not None and not isinstance(beta_slow, float):
    logging.warning("`rope_scaling`'s beta_slow field must be a float, got %s", beta_slow)

  if (beta_fast or 32) < (beta_slow or 1):
    logging.warning(
        "`rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast=%s "
        "(defaults to 32 if None) and beta_slow=%s (defaults to 1 if None)",
        beta_fast,
        beta_slow,
    )


def _validate_longrope_parameters(config: max_utils.Config, ignore_keys: Optional[Set[str]] = None):
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "short_factor", "long_factor"}
  # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
  optional_keys = {"attention_factor", "factor", "original_max_position_embeddings"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

  partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
  head_dim = getattr(config, "head_dim", config.emb_dim // config.num_attention_heads)
  dim = int(head_dim * partial_rotary_factor)

  short_factor = rope_scaling.get("short_factor")
  if not isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor):
    logging.warning("`rope_scaling`'s short_factor field must be a list of numbers, got %s", short_factor)
  if len(short_factor) != dim // 2:
    logging.warning("`rope_scaling`'s short_factor field must have length %s, got %s", dim // 2, len(short_factor))

  long_factor = rope_scaling.get("long_factor")
  if not isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor):
    logging.warning("`rope_scaling`'s long_factor field must be a list of numbers, got %s", long_factor)
  if len(long_factor) != dim // 2:
    logging.warning("`rope_scaling`'s long_factor field must have length %s, got %s", dim // 2, len(long_factor))

  # Handle Phi3 divergence: prefer the use of `attention_factor` and/or `factor` over
  # `original_max_position_embeddings` to compute internal variables. The latter lives outside `rope_scaling` and is
  # unique to longrope (= undesirable)
  if hasattr(config, "original_max_position_embeddings"):
    logging.warning(
        "This model has set a `original_max_position_embeddings` field, to be used together with "
        "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_scaling`"
        "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
        "as it is compatible with most model architectures."
    )
  else:
    factor = rope_scaling.get("factor")
    if factor is None:
      logging.warning("Missing required keys in `rope_scaling`: 'factor'")
    elif not isinstance(factor, float) or factor < 1.0:
      logging.warning("`rope_scaling`'s factor field must be a float >= 1, got %s", factor)

    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None:
      if not isinstance(attention_factor, float) or attention_factor < 0.0:
        logging.warning(
            "`rope_scaling`'s attention_factor field must be a float greater than 0, got %s", attention_factor
        )


def _validate_llama3_parameters(config: max_utils.Config, ignore_keys: Optional[Set[str]] = None):
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
  required_keys = {"rope_type", "factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor"}
  received_keys = set(rope_scaling.keys())
  _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    logging.warning("`rope_scaling`'s factor field must be a float >= 1, got %s", factor)

  low_freq_factor = rope_scaling["low_freq_factor"]
  high_freq_factor = rope_scaling["high_freq_factor"]
  if low_freq_factor is None or not isinstance(low_freq_factor, float):
    logging.warning("`rope_scaling`'s low_freq_factor field must be a float, got %s", low_freq_factor)
  if high_freq_factor is None or not isinstance(high_freq_factor, float):
    logging.warning("`rope_scaling`'s high_freq_factor field must be a float, got %s", high_freq_factor)
  if high_freq_factor <= low_freq_factor:
    logging.warning(
        "`rope_scaling`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
        "%s and low_freq_factor=%s",
        high_freq_factor,
        low_freq_factor,
    )

  original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
  if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
    logging.warning(
        "`rope_scaling`'s original_max_position_embeddings field must be an integer, got "
        "%s",
        original_max_position_embeddings,
    )
  if original_max_position_embeddings >= config.max_position_embeddings:
    logging.warning(
        "`rope_scaling`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
        "%s and max_position_embeddings=%s",
        original_max_position_embeddings,
        config.max_position_embeddings,
    )


# Like `ROPE_INIT_FUNCTIONS`, this validation function mapping can be dynamically updated for custom RoPE types.
ROPE_VALIDATION_FUNCTIONS = {
    "default": _validate_default_rope_parameters,
    "linear": _validate_linear_scaling_rope_parameters,
    "dynamic": _validate_dynamic_scaling_rope_parameters,
    "yarn": _validate_yarn_parameters,
    "longrope": _validate_longrope_parameters,
    "llama3": _validate_llama3_parameters,
}

from typing import Optional, Set

from absl import logging

from maxtext.common_types import Config
# The ROPE_VALIDATION_FUNCTIONS dictionary is defined in the same file.
# from . import ROPE_VALIDATION_FUNCTIONS


def rope_config_validation(config: Config, ignore_keys: Optional[Set] = None):
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
        "Missing validation function mapping in `ROPE_VALIDATION_FUNCTIONS` for 'rope_type'='%s'", rope_type
    )

from typing import Optional, Tuple

import jax.numpy as jnp

from maxtext.common_types import Array, Config


def _compute_dynamic_ntk_parameters(
    config: Config,
    seq_len: Optional[int | Array] = None,
) -> Tuple[Array, float]:
  """Computes the inverse frequencies with NTK scaling.

  Credits to the Reddit users /u/bloc97 and /u/emozilla.

  Args:
    config: The model configuration.
    seq_len: The current sequence length, used to update the dynamic RoPE at
      inference time.

  Returns:
    A tuple of (inv_freq, attention_factor), containing the inverse
    frequencies for the RoPE embeddings and the post-processing scaling factor
    applied to the computed cos/sin (unused in this type of RoPE).
  """
  # TODO (joao): use the new `original_max_position_embeddings` from rope_scaling
  base = config.rope_theta
  partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
  head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
  dim = int(head_dim * partial_rotary_factor)
  max_position_embeddings = config.max_position_embeddings
  factor = config.rope_scaling["factor"]

  attention_factor = 1.0  # Unused in this type of RoPE

  # seq_len: default to max_position_embeddings, e.g. at init time
  if seq_len is None:
    seq_len = max_position_embeddings
  elif isinstance(seq_len, (jnp.ndarray, Array)):
    seq_len = jnp.maximum(
        seq_len,
        jnp.array(max_position_embeddings, dtype=seq_len.dtype),
    )
  else:
    seq_len = max(seq_len, max_position_embeddings)

  # Compute the inverse frequencies
  base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
  inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
  return inv_freq, attention_factor

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""
 Jax implementation of RoPE scaling utils.
"""
from typing import Optional

from maxtext.common_types import Array, Config
from .rope_scaling_utils import _compute_default_rope_parameters


def _compute_linear_scaling_rope_parameters(
    config: Config,
    seq_len: Optional[int] = None,
) -> tuple[Array, float]:
  """Computes the inverse frequencies with linear scaling.

  Credits to the Reddit user /u/kaiokendev

  Args:
    config: The model configuration.
    seq_len: The current sequence length. Unused for this type of RoPE.

  Returns:
    A tuple of (Array, float), containing the inverse frequencies for the RoPE
    embeddings and the post-processing scaling factor applied to the computed
    cos/sin (unused in this type of RoPE).
  """
  factor = config.rope_scaling["factor"]

  # Gets the default RoPE parameters
  inv_freq, attention_factor = _compute_default_rope_parameters(config, seq_len=seq_len)

  # Then applies linear scaling to the frequencies.
  # NOTE: originally, scaling was applied to the position_ids. However, we get
  # `embs = inv_freq @ position_ids`, so applying scaling to the inverse
  # frequencies is equivalent.
  inv_freq = inv_freq / factor
  return inv_freq, attention_factor

import math
from typing import Optional

import jax.numpy as jnp
from jax import Array

# The following import is for type annotation only and does not introduce a runtime dependency.
# In a real application, this would likely be replaced by a MaxText-native config object.
from transformers import PretrainedConfig

# Reused from MaxText.layers.rope_scaling_utils._compute_default_rope_parameters
from MaxText.layers.rope_scaling_utils import _compute_default_rope_parameters


def _compute_llama3_parameters(
    config: PretrainedConfig, seq_len: Optional[int] = None
) -> tuple[Array, float]:
  """Computes the inverse frequencies for llama 3.1.

  Args:
    config: The model configuration.
    seq_len: The current sequence length. Unused for this type of RoPE.

  Returns:
    Tuple of (Array, float), containing the inverse frequencies for the RoPE
    embeddings and the post-processing scaling factor applied to the computed
    cos/sin.
  """
  # Gets the default RoPE parameters
  inv_freq, attention_factor = _compute_default_rope_parameters(config, seq_len)

  factor = config.rope_scaling["factor"]  # `8` in the original implementation
  low_freq_factor = config.rope_scaling[
      "low_freq_factor"
  ]  # `1` in the original implementation
  high_freq_factor = config.rope_scaling[
      "high_freq_factor"
  ]  # `4` in the original implementation
  old_context_len = config.rope_scaling[
      "original_max_position_embeddings"
  ]  # `8192` in the original implementation

  low_freq_wavelen = old_context_len / low_freq_factor
  high_freq_wavelen = old_context_len / high_freq_factor

  wavelen = 2 * math.pi / inv_freq
  # wavelen < high_freq_wavelen: do nothing
  # wavelen > low_freq_wavelen: divide by factor
  inv_freq_llama = jnp.where(
      wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
  )
  # otherwise: interpolate between the two, using a smooth factor
  smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
      high_freq_factor - low_freq_factor
  )
  smoothed_inv_freq = (
      1 - smooth_factor
  ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
  is_medium_freq = jnp.logical_and(
      wavelen >= high_freq_wavelen, wavelen <= low_freq_wavelen
  )
  inv_freq_llama = jnp.where(
      is_medium_freq, smoothed_inv_freq, inv_freq_llama
  )

  return inv_freq_llama, attention_factor

import math
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array

# Assuming config is a dataclass or similar object with attributes used below.
# In MaxText, this would likely be `from maxtext.common_types import Config`.
from maxtext.common_types import Config


def _compute_longrope_parameters(
    config: Config, seq_len: Optional[int] = None
) -> Tuple[Array, float]:
  """
    Computes the inverse frequencies with LongRoPE scaling. Please refer to the
    [original implementation](https://github.com/microsoft/LongRoPE)
    Args:
        config:
            The model configuration.
        seq_len (`int`, *optional*):
            The current sequence length.
    Returns:
        Tuple of (`jax.Array`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
  # TODO (joao): use the new `original_max_position_embeddings` from rope_scaling
  base = config.rope_theta
  partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
  head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
  dim = int(head_dim * partial_rotary_factor)
  long_factor = config.rope_scaling["long_factor"]
  short_factor = config.rope_scaling["short_factor"]
  factor = config.rope_scaling.get("factor")
  attention_factor = config.rope_scaling.get("attention_factor")

  # NOTE: Phi3 (and potentially other models) modify `max_position_embeddings` and have a
  # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
  # values to compute the default attention scaling factor, instead of using `factor`.
  if hasattr(config, "original_max_position_embeddings"):
    original_max_position_embeddings = config.original_max_position_embeddings
    factor = config.max_position_embeddings / config.original_max_position_embeddings
  else:
    original_max_position_embeddings = config.max_position_embeddings

  # Sets the attention factor as suggested in the paper
  if attention_factor is None:
    if factor <= 1.0:
      attention_factor = 1.0
    else:
      attention_factor = math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))

  # Compute the inverse frequencies -- scaled based on the target sequence length
  if seq_len and seq_len > original_max_position_embeddings:
    ext_factors = jnp.array(long_factor, dtype=jnp.float32)
  else:
    ext_factors = jnp.array(short_factor, dtype=jnp.float32)
  inv_freq_shape = jnp.arange(0, dim, 2, dtype=jnp.int64).astype(jnp.float32) / dim
  inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

  return inv_freq, attention_factor

import math
from typing import Optional, Tuple

from jax import Array
import jax.numpy as jnp

from ..configs.base_config import Config


def _compute_yarn_parameters(
    config: Config, seq_len: Optional[int] = None
) -> Tuple[Array, float]:
    """
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://huggingface.co/papers/2309.00071)
    Args:
        config (Config):
            The model configuration.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (Array, float), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """

    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    factor = config.rope_scaling["factor"]
    attention_factor = config.rope_scaling.get("attention_factor")
    mscale = config.rope_scaling.get("mscale")
    mscale_all_dim = config.rope_scaling.get("mscale_all_dim")

    # NOTE: DeekSeek-V3 (and potentially other models) modify `max_position_embeddings` and have a
    # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
    # values to compute the default attention scaling factor, instead of using `factor`.
    if "original_max_position_embeddings" in config.rope_scaling:
        original_max_position_embeddings = config.rope_scaling["original_max_position_embeddings"]
        factor = config.max_position_embeddings / original_max_position_embeddings
    else:
        original_max_position_embeddings = config.max_position_embeddings

    def get_mscale(scale, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    # Sets the attention factor as suggested in the paper
    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)

    # Optional config options
    # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
    beta_fast = config.rope_scaling.get("beta_fast", 32)
    beta_slow = config.rope_scaling.get("beta_slow", 1)

    # Compute the inverse frequencies
    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
        """Find dimension range bounds based on rotations"""
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001  # Prevent singularity

        linear_func = (jnp.arange(dim, dtype=jnp.float32) - min_val) / (max_val - min_val)
        ramp_func = jnp.clip(linear_func, 0, 1)
        return ramp_func

    # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
    # to expand the possible context length. In other words, interpolation = apply scaling factor.
    pos_freqs = base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    truncate = config.rope_scaling.get("truncate", True)
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2)
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""
This file contains the RoPE initialization functions and their mappings.
Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/common/rope_scaling_utils.py
"""

import math
from typing import Optional

import jax.numpy as jnp
from maxtext.common_types import Array, Config


def _compute_default_rope_parameters(
    config: Config, seq_len: Optional[int] = None
) -> tuple[Array, float]:
  """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config (Config):
            The model configuration.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`jnp.ndarray`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
  base = config.rope_theta
  partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
  head_dim = (
      getattr(config, "head_dim", None)
      or config.hidden_size // config.num_attention_heads
  )
  dim = int(head_dim * partial_rotary_factor)

  attention_factor = 1.0  # Unused in this type of RoPE

  # Compute the inverse frequencies
  inv_freq = 1.0 / (
      base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
  )
  return inv_freq, attention_factor


def _compute_linear_scaling_rope_parameters(
    config: Config, seq_len: Optional[int] = None
) -> tuple[Array, float]:
  """
    Computes the inverse frequencies with linear scaling. Credits to the Reddit user /u/kaiokendev
    Args:
        config (Config):
            The model configuration.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`jnp.ndarray`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
  factor = config.rope_scaling["factor"]

  # Gets the default RoPE parameters
  inv_freq, attention_factor = _compute_default_rope_parameters(config, seq_len)

  # Then applies linear scaling to the frequencies.
  # NOTE: originally, scaling was applied to the position_ids. However, we get `embs = inv_freq @ position_ids`, so
  # applying scaling to the inverse frequencies is equivalent.
  inv_freq /= factor
  return inv_freq, attention_factor


def _compute_dynamic_ntk_parameters(
    config: Config, seq_len: Optional[int] = None
) -> tuple[Array, float]:
  """
    Computes the inverse frequencies with NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
    Args:
        config (Config):
            The model configuration.
        seq_len (`int`, *optional*):
            The current sequence length, used to update the dynamic RoPE at inference time.
    Returns:
        Tuple of (`jnp.ndarray`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
  # TODO (joao): use the new `original_max_position_embeddings` from rope_scaling
  base = config.rope_theta
  partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
  head_dim = getattr(
      config, "head_dim", config.hidden_size // config.num_attention_heads
  )
  dim = int(head_dim * partial_rotary_factor)
  max_position_embeddings = config.max_position_embeddings
  factor = config.rope_scaling["factor"]

  attention_factor = 1.0  # Unused in this type of RoPE

  # seq_len: default to max_position_embeddings, e.g. at init time
  if seq_len is None:
    seq_len = max_position_embeddings
  elif isinstance(seq_len, jnp.ndarray):
    seq_len = jnp.maximum(
        seq_len,
        jnp.array(max_position_embeddings, dtype=seq_len.dtype),
    )
  else:
    seq_len = max(seq_len, max_position_embeddings)

  # Compute the inverse frequencies
  base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (
      dim / (dim - 2)
  )
  inv_freq = 1.0 / (
      base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
  )
  return inv_freq, attention_factor


def _compute_yarn_parameters(
    config: Config, seq_len: Optional[int] = None
) -> tuple[Array, float]:
  """
    Computes the inverse frequencies with NTK scaling. Please refer to the
    [original paper](https://huggingface.co/papers/2309.00071)
    Args:
        config (Config):
            The model configuration.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`jnp.ndarray`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """

  base = config.rope_theta
  partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
  head_dim = getattr(
      config, "head_dim", config.hidden_size // config.num_attention_heads
  )
  dim = int(head_dim * partial_rotary_factor)
  factor = config.rope_scaling["factor"]
  attention_factor = config.rope_scaling.get("attention_factor")
  mscale = config.rope_scaling.get("mscale")
  mscale_all_dim = config.rope_scaling.get("mscale_all_dim")

  # NOTE: DeekSeek-V3 (and potentially other models) modify `max_position_embeddings` and have a
  # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
  # values to compute the default attention scaling factor, instead of using `factor`.
  if "original_max_position_embeddings" in config.rope_scaling:
    original_max_position_embeddings = config.rope_scaling[
        "original_max_position_embeddings"
    ]
    factor = config.max_position_embeddings / original_max_position_embeddings
  else:
    original_max_position_embeddings = config.max_position_embeddings

  def get_mscale(scale, mscale=1):
    if scale <= 1:
      return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

  # Sets the attention factor as suggested in the paper
  if attention_factor is None:
    if mscale and mscale_all_dim:
      attention_factor = float(
          get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
      )
    else:
      attention_factor = get_mscale(factor)

  # Optional config options
  # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
  beta_fast = config.rope_scaling.get("beta_fast") or 32
  beta_slow = config.rope_scaling.get("beta_slow") or 1

  # Compute the inverse frequencies
  def find_correction_dim(
      num_rotations, dim, base, max_position_embeddings
  ):
    """Inverse dimension formula to find the dimension based on the number of rotations"""
    return (
        dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
    ) / (2 * math.log(base))

  def find_correction_range(
      low_rot, high_rot, dim, base, max_position_embeddings, truncate
  ):
    """Find dimension range bounds based on rotations"""
    low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
      low = math.floor(low)
      high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)

  def linear_ramp_factor(min_val, max_val, dim):
    if min_val == max_val:
      max_val += 0.001  # Prevent singularity

    linear_func = (jnp.arange(dim, dtype=jnp.float32) - min_val) / (
        max_val - min_val
    )
    ramp_func = jnp.clip(linear_func, 0, 1)
    return ramp_func

  # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
  # to expand the possible context length. In other words, interpolation = apply scaling factor.
  pos_freqs = base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
  inv_freq_extrapolation = 1.0 / pos_freqs
  inv_freq_interpolation = 1.0 / (factor * pos_freqs)

  truncate = config.rope_scaling.get("truncate", True)
  low, high = find_correction_range(
      beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate
  )

  # Get n-dimensional rotational scaling corrected for extrapolation
  inv_freq_extrapolation_factor = 1 - linear_ramp_factor(
      low, high, dim // 2
  )
  inv_freq = (
      inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
      + inv_freq_extrapolation * inv_freq_extrapolation_factor
  )
  return inv_freq, attention_factor


def _compute_longrope_parameters(
    config: Config, seq_len: Optional[int] = None
) -> tuple[Array, float]:
  """
    Computes the inverse frequencies with LongRoPE scaling. Please refer to the
    [original implementation](https://github.com/microsoft/LongRoPE)
    Args:
        config (Config):
            The model configuration.
        seq_len (`int`, *optional*):
            The current sequence length.
    Returns:
        Tuple of (`jnp.ndarray`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
  # TODO (joao): use the new `original_max_position_embeddings` from rope_scaling
  base = config.rope_theta
  partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
  head_dim = getattr(
      config, "head_dim", config.hidden_size // config.num_attention_heads
  )
  dim = int(head_dim * partial_rotary_factor)
  long_factor = config.rope_scaling["long_factor"]
  short_factor = config.rope_scaling["short_factor"]
  factor = config.rope_scaling.get("factor")
  attention_factor = config.rope_scaling.get("attention_factor")

  # NOTE: Phi3 (and potentially other models) modify `max_position_embeddings` and have a
  # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
  # values to compute the default attention scaling factor, instead of using `factor`.
  if hasattr(config, "original_max_position_embeddings"):
    original_max_position_embeddings = config.original_max_position_embeddings
    factor = config.max_position_embeddings / config.original_max_position_embeddings
  else:
    original_max_position_embeddings = config.max_position_embeddings

  # Sets the attention factor as suggested in the paper
  if attention_factor is None:
    if factor <= 1.0:
      attention_factor = 1.0
    else:
      attention_factor = math.sqrt(
          1 + math.log(factor) / math.log(original_max_position_embeddings)
      )

  # Compute the inverse frequencies -- scaled based on the target sequence length
  if seq_len and seq_len > original_max_position_embeddings:
    ext_factors = jnp.array(long_factor, dtype=jnp.float32)
  else:
    ext_factors = jnp.array(short_factor, dtype=jnp.float32)
  inv_freq_shape = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
  inv_freq = 1.0 / (ext_factors * base**inv_freq_shape)

  return inv_freq, attention_factor


def _compute_llama3_parameters(
    config: Config, seq_len: Optional[int] = None
) -> tuple[Array, float]:
  """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config (Config):
            The model configuration.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`jnp.ndarray`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
  # Gets the default RoPE parameters
  inv_freq, attention_factor = _compute_default_rope_parameters(config, seq_len)

  factor = config.rope_scaling["factor"]  # `8` in the original implementation
  low_freq_factor = config.rope_scaling[
      "low_freq_factor"
  ]  # `1` in the original implementation
  high_freq_factor = config.rope_scaling[
      "high_freq_factor"
  ]  # `4` in the original implementation
  old_context_len = config.rope_scaling[
      "original_max_position_embeddings"
  ]  # `8192` in the original implementation

  low_freq_wavelen = old_context_len / low_freq_factor
  high_freq_wavelen = old_context_len / high_freq_factor

  wavelen = 2 * math.pi / inv_freq
  # wavelen < high_freq_wavelen: do nothing
  # wavelen > low_freq_wavelen: divide by factor
  inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
  # otherwise: interpolate between the two, using a smooth factor
  smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
      high_freq_factor - low_freq_factor
  )
  smoothed_inv_freq = (
      (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
  )
  is_medium_freq = jnp.logical_and(
      jnp.logical_not(wavelen < high_freq_wavelen),
      jnp.logical_not(wavelen > low_freq_wavelen),
  )
  inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

  return inv_freq_llama, attention_factor


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

# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""
JAX port of HuggingFace's rope_utils.py
"""
from functools import wraps
from typing import Callable

import jax.numpy as jnp

from maxtext.common_types import Array


def dynamic_rope_update(rope_forward: Callable) -> Callable:
  """Decorator function to update the RoPE parameters in the forward pass, if the model is using a dynamic RoPE
    (i.e. a RoPE implementation that may recompute its frequencies in the forward pass).

    Args:
        rope_forward (Callable):
            The forward pass of the RoPE implementation.

    Returns:
        The decorated forward pass.
  """

  def longrope_frequency_update(self, position_ids: Array):
    """Longrope uses long factor if sequence is larger than original pretraining length, short otherwise."""
    seq_len = jnp.max(position_ids) + 1
    if hasattr(self.config, "original_max_position_embeddings"):
      original_max_position_embeddings = self.config.original_max_position_embeddings
    else:
      original_max_position_embeddings = self.config.max_position_embeddings

    # Note: This conditional logic is stateful and not JAX JIT-compatible.
    # It is a direct translation of the original PyTorch logic.
    if seq_len > original_max_position_embeddings:
      if not hasattr(self, "long_inv_freq"):
        # The `device` argument is omitted in the JAX version.
        self.long_inv_freq, _ = self.rope_init_fn(self.config, seq_len=original_max_position_embeddings + 1)
      # Direct attribute assignment replaces register_buffer
      self.inv_freq = self.long_inv_freq
    else:
      # .to(device) is not needed in JAX.
      # Direct attribute assignment replaces register_buffer
      self.inv_freq = self.original_inv_freq

  def dynamic_frequency_update(self, position_ids: Array):
    """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
    """
    seq_len = jnp.max(position_ids) + 1

    # Note: This conditional logic is stateful and not JAX JIT-compatible.
    # It is a direct translation of the original PyTorch logic.
    if seq_len > self.max_seq_len_cached:  # growth
      # The `device` argument is omitted in the JAX version.
      inv_freq, self.attention_scaling = self.rope_init_fn(self.config, seq_len=seq_len)
      # Direct attribute assignment replaces register_buffer
      self.inv_freq = inv_freq
      self.max_seq_len_cached = seq_len

    if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
      # .to(device) is not needed in JAX.
      # Direct attribute assignment replaces register_buffer
      self.inv_freq = self.original_inv_freq
      self.max_seq_len_cached = self.original_max_seq_len

  @wraps(rope_forward)
  def wrapper(self, x: Array, position_ids: Array):
    # The `device=x.device` argument is removed as it's not needed in JAX.
    if "dynamic" in self.rope_type:
      dynamic_frequency_update(self, position_ids)
    elif self.rope_type == "longrope":
      longrope_frequency_update(self, position_ids)
    return rope_forward(self, x, position_ids)

  return wrapper

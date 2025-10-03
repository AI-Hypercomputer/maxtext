
from typing import Optional, Tuple

from maxtext.common_types import Array
from maxtext.max_utils import Config

# The following import is for a function assumed to be in the same file/module.
# It is not a pre-existing MaxText module.
from .rope_utils import _compute_default_rope_parameters


def _compute_linear_scaling_rope_parameters(
    config: Config,
    seq_len: Optional[int] = None,
) -> Tuple[Array, float]:
  """Computes the inverse frequencies with linear scaling.

  Credits to the Reddit user /u/kaiokendev

  Args:
    config: The model configuration.
    seq_len: The current sequence length. Unused for this type of RoPE.

  Returns:
    Tuple of (Array, float), containing the inverse frequencies for the RoPE
    embeddings and the post-processing scaling factor applied to the computed
    cos/sin (unused in this type of RoPE).
  """
  factor = config.rope_scaling["factor"]

  # Gets the default RoPE parameters
  inv_freq, attention_factor = _compute_default_rope_parameters(config, seq_len)

  # Then applies linear scaling to the frequencies.
  # NOTE: originally, scaling was applied to the position_ids. However, we get
  # `embs = inv_freq @ position_ids`, so applying scaling to the inverse
  # frequencies is equivalent.
  inv_freq /= factor
  return inv_freq, attention_factor

from typing import Optional, Set

from maxtext.common_types import Config
from maxtext import max_logging
from .rope_utils import _check_received_keys


def _validate_yarn_parameters(config: Config, ignore_keys: Optional[Set] = None):
  """Validate the YaRN RoPE scaling parameters."""
  rope_scaling = config.rope_scaling
  rope_type = rope_scaling.get(
      "rope_type", rope_scaling.get("type", None)
  )  # BC: "rope_type" was originally "type"
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
  _check_received_keys(
      rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys
  )

  factor = rope_scaling["factor"]
  if factor is None or not isinstance(factor, float) or factor < 1.0:
    max_logging.warning(
        f"`rope_scaling`'s factor field must be a float >= 1, got {factor}"
    )

  attention_factor = rope_scaling.get("attention_factor")
  if attention_factor is not None and (
      not isinstance(attention_factor, float) or attention_factor < 0
  ):
    max_logging.warning(
        "`rope_scaling`'s attention_factor field must be a float greater than"
        f" 0, got {attention_factor}"
    )
  beta_fast = rope_scaling.get("beta_fast")
  if beta_fast is not None and not isinstance(beta_fast, float):
    max_logging.warning(
        f"`rope_scaling`'s beta_fast field must be a float, got {beta_fast}"
    )
  beta_slow = rope_scaling.get("beta_slow")
  if beta_slow is not None and not isinstance(beta_slow, float):
    max_logging.warning(
        f"`rope_scaling`'s beta_slow field must be a float, got {beta_slow}"
    )

  if (beta_fast or 32) < (beta_slow or 1):
    max_logging.warning(
        "`rope_scaling`'s beta_fast field must be greater than beta_slow, got"
        f" beta_fast={beta_fast} (defaults to 32 if None) and"
        f" beta_slow={beta_slow} (defaults to 1 if None)"
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
""" RoPE validation functions"""

from typing import Optional, Set

from MaxText.common_types import Config
from MaxText.maxtext_utils import max_logging


def _check_received_keys(
    rope_type: str,
    received_keys: Set,
    required_keys: Set,
    optional_keys: Optional[Set] = None,
    ignore_keys: Optional[Set] = None,
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
        max_logging.warning(f"Unrecognized keys in `rope_scaling` for 'rope_type'='{rope_type}': {unused_keys}")


def _validate_default_rope_parameters(config: Config, ignore_keys: Optional[Set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)


def _validate_linear_scaling_rope_parameters(config: Config, ignore_keys: Optional[Set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        max_logging.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")


def _validate_dynamic_scaling_rope_parameters(config: Config, ignore_keys: Optional[Set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor"}
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {"original_max_position_embeddings"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        max_logging.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")


def _validate_yarn_parameters(config: Config, ignore_keys: Optional[Set] = None):
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
        max_logging.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    attention_factor = rope_scaling.get("attention_factor")
    if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
        max_logging.warning(
            f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
        )
    beta_fast = rope_scaling.get("beta_fast")
    if beta_fast is not None and not isinstance(beta_fast, float):
        max_logging.warning(f"`rope_scaling`'s beta_fast field must be a float, got {beta_fast}")
    beta_slow = rope_scaling.get("beta_slow")
    if beta_slow is not None and not isinstance(beta_slow, float):
        max_logging.warning(f"`rope_scaling`'s beta_slow field must be a float, got {beta_slow}")

    if (beta_fast or 32) < (beta_slow or 1):
        max_logging.warning(
            f"`rope_scaling`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
            f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
        )


def _validate_longrope_parameters(config: Config, ignore_keys: Optional[Set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "short_factor", "long_factor"}
    # TODO (joao): update logic for the inclusion of `original_max_position_embeddings`
    optional_keys = {"attention_factor", "factor", "original_max_position_embeddings"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)

    short_factor = rope_scaling.get("short_factor")
    if not isinstance(short_factor, list) or not all(isinstance(x, (int, float)) for x in short_factor):
        max_logging.warning(f"`rope_scaling`'s short_factor field must be a list of numbers, got {short_factor}")
    if len(short_factor) != dim // 2:
        max_logging.warning(f"`rope_scaling`'s short_factor field must have length {dim // 2}, got {len(short_factor)}")

    long_factor = rope_scaling.get("long_factor")
    if not isinstance(long_factor, list) or not all(isinstance(x, (int, float)) for x in long_factor):
        max_logging.warning(f"`rope_scaling`'s long_factor field must be a list of numbers, got {long_factor}")
    if len(long_factor) != dim // 2:
        max_logging.warning(f"`rope_scaling`'s long_factor field must have length {dim // 2}, got {len(long_factor)}")

    # Handle Phi3 divergence: prefer the use of `attention_factor` and/or `factor` over
    # `original_max_position_embeddings` to compute internal variables. The latter lives outside `rope_scaling` and is
    # unique to longrope (= undesirable)
    if hasattr(config, "original_max_position_embeddings"):
        # MaxText does not have a `warning_once` equivalent, using standard warning.
        max_logging.warning(
            "This model has set a `original_max_position_embeddings` field, to be used together with "
            "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_scaling`"
            "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
            "as it is compatible with most model architectures."
        )
    else:
        factor = rope_scaling.get("factor")
        if factor is None:
            max_logging.warning("Missing required keys in `rope_scaling`: 'factor'")
        elif not isinstance(factor, float) or factor < 1.0:
            max_logging.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

        attention_factor = rope_scaling.get("attention_factor")
        if attention_factor is not None:
            if not isinstance(attention_factor, float) or attention_factor < 0.0:
                max_logging.warning(
                    f"`rope_scaling`'s attention_factor field must be a float greater than 0, got {attention_factor}"
                )


def _validate_llama3_parameters(config: Config, ignore_keys: Optional[Set] = None):
    rope_scaling = config.rope_scaling
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", None))  # BC: "rope_type" was originally "type"
    required_keys = {"rope_type", "factor", "original_max_position_embeddings", "low_freq_factor", "high_freq_factor"}
    received_keys = set(rope_scaling.keys())
    _check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    factor = rope_scaling["factor"]
    if factor is None or not isinstance(factor, float) or factor < 1.0:
        max_logging.warning(f"`rope_scaling`'s factor field must be a float >= 1, got {factor}")

    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    if low_freq_factor is None or not isinstance(low_freq_factor, float):
        max_logging.warning(f"`rope_scaling`'s low_freq_factor field must be a float, got {low_freq_factor}")
    if high_freq_factor is None or not isinstance(high_freq_factor, float):
        max_logging.warning(f"`rope_scaling`'s high_freq_factor field must be a float, got {high_freq_factor}")
    if high_freq_factor <= low_freq_factor:
        max_logging.warning(
            "`rope_scaling`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
            f"{high_freq_factor} and low_freq_factor={low_freq_factor}"
        )

    original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
    if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
        max_logging.warning(
            "`rope_scaling`'s original_max_position_embeddings field must be an integer, got "
            f"{original_max_position_embeddings}"
        )
    if original_max_position_embeddings >= config.max_position_embeddings:
        max_logging.warning(
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

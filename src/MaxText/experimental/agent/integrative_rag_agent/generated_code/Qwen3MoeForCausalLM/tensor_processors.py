
# Copyright 2024 The ggml.ai team and The HuggingFace Inc. team. and pygguf author (github.com/99991)
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
This file contains the MambaTensorProcessor, a JAX-based tensor processor for Mamba models.
"""

from typing import Any, Dict

import jax.numpy as jnp

from .gguf_utils import GGUFTensor, TensorProcessor
from maxtext.common_types import Array


class MambaTensorProcessor(TensorProcessor):
  """A tensor processor for Mamba models."""

  def __init__(self, config: Dict[str, Any] | None = None):
    super().__init__(config=config)

  def process(self, weights: Array, name: str, **kwargs: Any) -> GGUFTensor:
    """Processes a Mamba tensor."""
    if "ssm_conv1d.weight" in name:
      # for compatibility tensor ssm_conv1d must be (5120, 1, 4]) dim,
      # quantized one is (5120, 4)
      weights = jnp.expand_dims(weights, axis=1)
    if "ssm_a" in name:
      # Original exponential implementation
      # https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L2975-L2977
      weights = jnp.log(-weights)
    return GGUFTensor(weights, name, {})

# Copyright 2024 The ggml.ai team and The HuggingFace Inc. team. and pygguf author (github.com/99991)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tensor Processors for GGUF."""

from typing import Any

from . import GGUFTensor, TensorProcessor
from maxtext.common_types import Array


class NemotronTensorProcessor(TensorProcessor):
  """A tensor processor for Nemotron models."""

  def __init__(self, config: Any = None):
    super().__init__(config=config)

  # ref : https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L4666
  def process(self, weights: Array, name: str, **kwargs: Any) -> GGUFTensor:
    """Processes a tensor, applying Nemotron-specific transformations."""
    if "norm.weight" in name:
      weights = weights - 1
    return GGUFTensor(weights, name, {})

from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

from maxtext.common_types import Array

# Assuming GGUFTensor and TensorProcessor are defined in the same file/module
# and have been converted to JAX-compatible versions.
from .gguf_utils import GGUFTensor, TensorProcessor


class Qwen2MoeTensorProcessor(TensorProcessor):
  """Tensor processor for Qwen2Moe models."""

  def __init__(self, config: Dict[str, Any] | None = None):
    """Initializes the Qwen2MoeTensorProcessor.

    Args:
      config: A dictionary containing model configuration.
    """
    super().__init__(config=config)

  def process(self, weights: Array, name: str, **kwargs: Any) -> GGUFTensor:
    """Processes a tensor, handling special cases for MoE experts and shared expert gates.

    Args:
      weights: The tensor weights as a JAX array.
      name: The name of the tensor.
      **kwargs: Additional arguments, expected to contain 'tensor_key_mapping'
        and 'parsed_parameters' for MoE tensors.

    Returns:
      A GGUFTensor object containing the processed weights and name.
    """
    if "_exp" in name:
      tensor_key_mapping = kwargs.get("tensor_key_mapping")
      parsed_parameters = kwargs.get("parsed_parameters")
      if tensor_key_mapping and parsed_parameters is not None:
        self._split_moe_expert_tensor(weights, parsed_parameters, name, tensor_key_mapping)
        # Return a GGUFTensor with a None name to indicate it's been handled
        return GGUFTensor(weights, None, {})
    if "ffn_gate_inp_shexp" in name:
      # for compatibility tensor shared_expert_gate must be (1, 2048) dim,
      # quantized one is (2048)
      weights = jnp.expand_dims(weights, axis=0)
    return GGUFTensor(weights, name, {})

  def _split_moe_expert_tensor(
      self,
      weights: np.ndarray,
      parsed_parameters: Dict[str, Any],
      name: str,
      tensor_key_mapping: Dict[str, str],
  ) -> None:
    """Splits a merged MoE expert tensor back into individual expert tensors.

    This method modifies the `parsed_parameters` dictionary in-place.

    Args:
      weights: The merged tensor containing weights for all experts.
      parsed_parameters: A dictionary to be updated with the split tensors.
      name: The original name of the merged tensor.
      tensor_key_mapping: A mapping from the GGUF name to the target model's
        name format.
    """
    # Original merge implementation
    # https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L1994-L2022
    name = tensor_key_mapping[name]
    w_counter = self.config.get("num_experts", 60)
    for i in range(w_counter):
      temp_name = name.replace("mlp.experts.", f"mlp.experts.{i}.")
      exp_weight = weights[i]
      # jnp.array creates a copy, which is the desired behavior.
      parsed_parameters["tensors"][temp_name] = jnp.array(exp_weight)

from typing import Any, Dict, Optional

from maxtext.common_types import Array
from maxtext.layers.gguf import GGUFTensor, TensorProcessor


class T5TensorProcessor(TensorProcessor):
  """Tensor processor for T5 model."""

  def __init__(self, config: Optional[Dict[str, Any]] = None):
    """Initializes the T5TensorProcessor.

    Args:
      config: Configuration dictionary.
    """
    super().__init__(config=config)

  def process(self, weights: Array, name: str, **kwargs) -> GGUFTensor:
    """Processes a T5 tensor to extract the block ID.

    Args:
      weights: The tensor weights.
      name: The name of the tensor.
      **kwargs: Additional keyword arguments.

    Returns:
      A GGUFTensor containing the weights, name, and extracted metadata.
    """
    bid = None
    for chunk in name.split("."):
      if chunk.isdigit():
        bid = int(chunk)
        break
    return GGUFTensor(weights, name, {"bid": bid})

# Copyright 2024 The ggml.ai team and The HuggingFace Inc. team. and pygguf author (github.com/99991)
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
"""Tensor processors for GGUF conversion."""

from typing import Dict, Type
# Assuming these processors are defined in a new module: MaxText.gguf.tensor_processors
from .tensor_processors import (
    BloomTensorProcessor,
    Gemma2TensorProcessor,
    GPT2TensorProcessor,
    LlamaTensorProcessor,
    MambaTensorProcessor,
    NemotronTensorProcessor,
    Qwen2MoeTensorProcessor,
    T5TensorProcessor,
    TensorProcessor,
)


TENSOR_PROCESSORS: Dict[str, Type[TensorProcessor]] = {
    "llama": LlamaTensorProcessor,
    "qwen2moe": Qwen2MoeTensorProcessor,
    "qwen3moe": Qwen2MoeTensorProcessor,
    "bloom": BloomTensorProcessor,
    "t5": T5TensorProcessor,
    "t5encoder": T5TensorProcessor,
    "gpt2": GPT2TensorProcessor,
    "mamba": MambaTensorProcessor,
    "nemotron": NemotronTensorProcessor,
    "gemma2": Gemma2TensorProcessor,
    "gemma3": Gemma2TensorProcessor,
}

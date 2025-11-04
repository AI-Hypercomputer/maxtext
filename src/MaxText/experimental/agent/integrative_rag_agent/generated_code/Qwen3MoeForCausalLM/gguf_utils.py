
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
""" GGUF conversion utilities."""

from typing import Any, List

# Assuming `integrations._gguf_parse_value` is available in the JAX environment.
from .integrations import _gguf_parse_value


def read_field(reader: Any, field: str) -> List[Any]:
  """Reads a specific field from a GGUF reader object.

  Args:
    reader: The GGUFReader object.
    field: The name of the field to read.

  Returns:
    A list of parsed values for the given field, or an empty list if the
    field does not exist.
  """
  if field not in reader.fields:
    return []
  value = reader.fields[field]
  return [
      _gguf_parse_value(value.parts[_data_index], value.types)
      for _data_index in value.data
  ]

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
""" GGUF constants """

from .integrations import GGUF_CONFIG_MAPPING, GGUF_TOKENIZER_MAPPING


GGUF_TO_TRANSFORMERS_MAPPING = {
    "ignore": {
        "GGUF": {
            "version": "version",
            "tensor_count": "tensor_count",
            "kv_count": "kv_count",
        },
        "general": {"file_type": "file_type", "quantization_version": "quantization_version"},
    },
    "config": GGUF_CONFIG_MAPPING,
    "tokenizer": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer"]},
    "tokenizer_config": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer_config"]},
}

from typing import NamedTuple
from jax import Array


class GGUFTensor(NamedTuple):
  """Represents a tensor read from a GGUF file.

  Attributes:
    weights: The tensor data as a JAX array.
    name: The name of the tensor.
    metadata: A dictionary of metadata associated with the tensor.
  """

  weights: Array
  name: str
  metadata: dict

from typing import Any, Dict, NamedTuple, Optional

import jax


# GGUFTensor is a dependency from the source file, defined here for clarity
# as it is not an existing JAX module.
class GGUFTensor(NamedTuple):
  """A named tuple to hold tensor data and metadata."""

  weights: jax.Array
  name: str
  metadata: Dict[str, Any]


class TensorProcessor:
  """Base class for processing tensors from a GGUF file."""

  config: Dict[str, Any]

  def __init__(self, config: Optional[Dict[str, Any]] = None):
    """Initializes the TensorProcessor.

    Args:
      config: A dictionary containing model configuration.
    """
    self.config = config or {}

  def process(
      self, weights: jax.Array, name: str, **kwargs: Any
  ) -> GGUFTensor:
    """Processes a tensor, wrapping it in a GGUFTensor.

    This base implementation is a pass-through.

    Args:
      weights: The tensor weights as a JAX array.
      name: The name of the tensor.
      **kwargs: Additional keyword arguments (unused in the base class).

    Returns:
      A GGUFTensor object containing the weights, name, and empty metadata.
    """
    return GGUFTensor(weights, name, {})
GGUF_SUPPORTED_ARCHITECTURES = list(GGUF_TO_TRANSFORMERS_MAPPING["config"].keys())
from typing import Any, Dict, NamedTuple, Optional

import jax.numpy as jnp


# The GGUFTensor and TensorProcessor classes are dependencies of the code block
# and are converted from the full source file.
class GGUFTensor(NamedTuple):
  """A named tuple to hold tensor data and metadata from a GGUF file."""

  weights: jnp.ndarray
  name: str
  metadata: Dict[str, Any]


class TensorProcessor:
  """Base class for processing tensors from a GGUF file."""

  def __init__(self, config: Optional[Dict[str, Any]] = None):
    self.config = config or {}

  def process(
      self, weights: jnp.ndarray, name: str, **kwargs: Any
  ) -> GGUFTensor:
    """Processes a tensor, returning a GGUFTensor object."""
    return GGUFTensor(weights, name, {})


class BloomTensorProcessor(TensorProcessor):
  """Tensor processor for Bloom model weights."""

  def __init__(self, config: Optional[Dict[str, Any]] = None):
    super().__init__(config=config)

  def process(
      self, weights: jnp.ndarray, name: str, **kwargs: Any
  ) -> GGUFTensor:
    """
    Processes Bloom model tensors, specifically handling the reshaping of
    the fused QKV attention weights and biases.
    """
    if "attn_qkv" in name:
      num_heads = self.config["n_head"]
      n_embed = self.config["hidden_size"]
      if "weight" in name:
        weights = self._reverse_reshape_weights(weights, num_heads, n_embed)
      else:
        weights = self._reverse_reshape_bias(weights, num_heads, n_embed)
    return GGUFTensor(weights, name, {})

  def _reverse_reshape_weights(
      self, weights: jnp.ndarray, n_head: int, n_embed: int
  ) -> jnp.ndarray:
    """
    Reverses the reshaping of the fused QKV weights to match the GGUF format.

    Original reshape implementation:
    https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L972-L985
    """
    q, k, v = jnp.array_split(weights, 3, axis=0)

    q = q.reshape(n_head, n_embed // n_head, n_embed)
    k = k.reshape(n_head, n_embed // n_head, n_embed)
    v = v.reshape(n_head, n_embed // n_head, n_embed)
    qkv_weights = jnp.stack([q, k, v], axis=1)

    return qkv_weights.reshape(n_head * 3 * (n_embed // n_head), n_embed)

  def _reverse_reshape_bias(
      self, weights: jnp.ndarray, n_head: int, n_embed: int
  ) -> jnp.ndarray:
    """
    Reverses the reshaping of the fused QKV bias to match the GGUF format.

    Original reshape implementation:
    https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L986-L998
    """
    q_bias, k_bias, v_bias = jnp.array_split(weights, 3)

    q_bias = q_bias.reshape(n_head, n_embed // n_head)
    k_bias = k_bias.reshape(n_head, n_embed // n_head)
    v_bias = v_bias.reshape(n_head, n_embed // n_head)

    qkv_bias = jnp.stack([q_bias, k_bias, v_bias], axis=1).flatten()
    return qkv_bias

from typing import Any, Optional

import jax.numpy as jnp

# The GGUFTensor and TensorProcessor classes are assumed to be defined in the same file,
# similar to the original PyTorch source.
from .gguf_utils import GGUFTensor, TensorProcessor


class GPT2TensorProcessor(TensorProcessor):
  """Tensor processor for GPT2."""

  def __init__(self, config: Optional[dict[str, Any]] = None):
    super().__init__(config=config)

  def process(
      self, weights: jnp.ndarray, name: Optional[str], **kwargs: Any
  ) -> GGUFTensor:
    """Processes a GPT2 tensor, applying necessary transformations."""
    # Original transpose implementation
    # https://github.com/ggerganov/llama.cpp/blob/a38b884c6c4b0c256583acfaaabdf556c62fabea/convert_hf_to_gguf.py#L2060-L2061
    if name is not None and (
        "attn_qkv.weight" in name
        or "ffn_down.weight" in name
        or "ffn_up.weight" in name
        or "attn_output.weight" in name
    ):
      weights = weights.T

    # Handle special case for output.weight
    if name == "output.weight":
      # output.weight has conflicts with attn_output.weight in name checking
      # Store the tensor directly and signal to skip further processing
      name = "lm_head.weight"
      parsed_parameters = kwargs.get("parsed_parameters", {})
      parsed_parameters["tensors"][name] = jnp.copy(weights)
      name = None  # Signal to skip further processing
    return GGUFTensor(weights, name, {})

from typing import Any, Dict

from maxtext.common_types import Array
# The GGUFTensor and TensorProcessor classes are assumed to be in the same module
# as they are in the source file.
from . import GGUFTensor, TensorProcessor


class Gemma2TensorProcessor(TensorProcessor):
  """Tensor processor for Gemma2."""

  def __init__(self, config: Dict[str, Any] | None = None):
    """Initializes the Gemma2TensorProcessor.

    Args:
      config: An optional dictionary containing model configuration.
    """
    super().__init__(config=config)

  # ref: https://github.com/ggerganov/llama.cpp/blob/d79d8f39b4da6deca4aea8bf130c6034c482b320/convert_hf_to_gguf.py#L3191
  # ref: https://github.com/huggingface/transformers/blob/fc37f38915372c15992b540dfcbbe00a916d4fc6/src/transformers/models/gemma/modeling_gemma.py#L89
  def process(self, weights: Array, name: str, **kwargs: Any) -> GGUFTensor:
    """Processes a tensor, applying Gemma2-specific transformations.

    Args:
      weights: The tensor weights to process.
      name: The name of the tensor.
      **kwargs: Additional keyword arguments.

    Returns:
      A GGUFTensor object with the processed weights.
    """
    if "norm.weight" in name:
      weights = weights - 1
    return GGUFTensor(weights, name, {})

# Copyright 2024 The ggml.ai team and The HuggingFace Inc. team. and pygguf author (github.com/99991)
# https://github.com/99991/pygguf
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

import re
from typing import Any, List, NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from .integrations import (
    GGUF_CONFIG_MAPPING,
    GGUF_TOKENIZER_MAPPING,
    _gguf_parse_value,
)
from .utils.import_utils import is_gguf_available
from .utils.logging import get_logger


logger = get_logger(__name__)


GGUF_TO_TRANSFORMERS_MAPPING = {
    "ignore": {
        "GGUF": {
            "version": "version",
            "tensor_count": "tensor_count",
            "kv_count": "kv_count",
        },
        "general": {"file_type": "file_type", "quantization_version": "quantization_version"},
    },
    "config": GGUF_CONFIG_MAPPING,
    "tokenizer": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer"]},
    "tokenizer_config": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer_config"]},
}

GGUF_SUPPORTED_ARCHITECTURES = list(GGUF_TO_TRANSFORMERS_MAPPING["config"].keys())


class GGUFTensor(NamedTuple):
    weights: jnp.ndarray
    name: str
    metadata: dict


class TensorProcessor:
    def __init__(self, config=None):
        self.config = config or {}

    def process(self, weights, name, **kwargs):
        return GGUFTensor(weights, name, {})


class LlamaTensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        if ".attn_k." in name or ".attn_q." in name:
            num_heads = self.config.get("num_attention_heads")
            num_kv_heads = self.config.get("num_key_value_heads")

            if None in (num_heads, num_kv_heads):
                return GGUFTensor(weights, name, {})
            if ".attn_q." in name:
                weights = self._reverse_permute_weights(weights, num_heads, num_heads)
            elif ".attn_k." in name:
                weights = self._reverse_permute_weights(weights, num_heads, num_kv_heads)
        return GGUFTensor(weights, name, {})

    def _reverse_permute_weights(
        self, weights: jnp.ndarray, n_head: int, num_kv_heads: Optional[int] = None
    ) -> jnp.ndarray:
        # Original permutation implementation
        # https://github.com/ggerganov/llama.cpp/blob/a38b884c6c4b0c256583acfaaabdf556c62fabea/convert_hf_to_gguf.py#L1402-L1408
        if num_kv_heads is not None and n_head != num_kv_heads:
            n_head = num_kv_heads

        dim = weights.shape[0] // n_head // 2
        w = weights.reshape(n_head, dim, 2, *weights.shape[1:])
        return jnp.swapaxes(w, 1, 2).reshape(weights.shape)


class Qwen2MoeTensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        if "_exp" in name:
            tensor_key_mapping = kwargs.get("tensor_key_mapping")
            parsed_parameters = kwargs.get("parsed_parameters")
            if tensor_key_mapping:
                self._split_moe_expert_tensor(weights, parsed_parameters, name, tensor_key_mapping)
                return GGUFTensor(weights, None, {})
        if "ffn_gate_inp_shexp" in name:
            # for compatibility tensor shared_expert_gate must be (1, 2048) dim,
            # quantized one is (2048)
            weights = jnp.expand_dims(weights, axis=0)
        return GGUFTensor(weights, name, {})

    def _split_moe_expert_tensor(
        self, weights: jnp.ndarray, parsed_parameters: dict[str, dict], name: str, tensor_key_mapping: dict
    ):
        # Original merge implementation
        # https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L1994-L2022
        name = tensor_key_mapping[name]
        w_counter = self.config.get("num_experts", 60)
        for i in range(0, w_counter):
            temp_name = name.replace("mlp.experts.", f"mlp.experts.{i}.")
            exp_weight = weights[i]
            parsed_parameters["tensors"][temp_name] = jnp.asarray(exp_weight)


class BloomTensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        if "attn_qkv" in name:
            num_heads = self.config["n_head"]
            n_embed = self.config["hidden_size"]
            if "weight" in name:
                weights = self._reverse_reshape_weights(weights, num_heads, n_embed)
            else:
                weights = self._reverse_reshape_bias(weights, num_heads, n_embed)
        return GGUFTensor(weights, name, {})

    def _reverse_reshape_weights(self, weights: jnp.ndarray, n_head: int, n_embed: int):
        # Original reshape implementation
        # https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L972-L985
        q, k, v = jnp.array_split(weights, 3, axis=0)

        q = q.reshape(n_head, n_embed // n_head, n_embed)
        k = k.reshape(n_head, n_embed // n_head, n_embed)
        v = v.reshape(n_head, n_embed // n_head, n_embed)
        qkv_weights = jnp.stack([q, k, v], axis=1)

        return qkv_weights.reshape(n_head * 3 * (n_embed // n_head), n_embed)

    def _reverse_reshape_bias(self, weights: jnp.ndarray, n_head: int, n_embed: int):
        # Original reshape implementation
        # https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L986-L998
        q_bias, k_bias, v_bias = jnp.array_split(weights, 3)

        q_bias = q_bias.reshape(n_head, n_embed // n_head)
        k_bias = k_bias.reshape(n_head, n_embed // n_head)
        v_bias = v_bias.reshape(n_head, n_embed // n_head)

        qkv_bias = jnp.stack([q_bias, k_bias, v_bias], axis=1).flatten()
        return qkv_bias


class T5TensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        bid = None
        for chunk in name.split("."):
            if chunk.isdigit():
                bid = int(chunk)
                break
        return GGUFTensor(weights, name, {"bid": bid})


class GPT2TensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        # Original transpose implementation
        # https://github.com/ggerganov/llama.cpp/blob/a38b884c6c4b0c256583acfaaabdf556c62fabea/convert_hf_to_gguf.py#L2060-L2061
        if (
            "attn_qkv.weight" in name
            or "ffn_down.weight" in name
            or "ffn_up.weight" in name
            or "attn_output.weight" in name
        ):
            weights = weights.T

        # Handle special case for output.weight
        if name == "output.weight":
            # output.weight has conflicts with attn_output.weight in name checking
            # Store the tensor directly and signal to skip further processing
            name = "lm_head.weight"
            parsed_parameters = kwargs.get("parsed_parameters", {})
            parsed_parameters["tensors"][name] = jnp.asarray(weights)
            name = None  # Signal to skip further processing
        return GGUFTensor(weights, name, {})


class MambaTensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        if "ssm_conv1d.weight" in name:
            # for compatibility tensor ssm_conv1d must be (5120, 1, 4]) dim,
            # quantized one is (5120, 4)
            weights = jnp.expand_dims(weights, axis=1)
        if "ssm_a" in name:
            # Original exponential implementation
            # https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L2975-L2977
            weights = jnp.log(-weights)
        return GGUFTensor(weights, name, {})


class NemotronTensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    # ref : https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L4666
    def process(self, weights, name, **kwargs):
        if "norm.weight" in name:
            weights = weights - 1
        return GGUFTensor(weights, name, {})


class Gemma2TensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    # ref: https://github.com/ggerganov/llama.cpp/blob/d79d8f39b4da6deca4aea8bf130c6034c482b320/convert_hf_to_gguf.py#L3191
    # ref: https://github.com/huggingface/transformers/blob/fc37f38915372c15992b540dfcbbe00a916d4fc6/src/transformers/models/gemma/modeling_gemma.py#L89
    def process(self, weights, name, **kwargs):
        if "norm.weight" in name:
            weights = weights - 1
        return GGUFTensor(weights, name, {})


TENSOR_PROCESSORS = {
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


def read_field(reader, field):
    if field not in reader.fields:
        return []
    value = reader.fields[field]
    return [_gguf_parse_value(value.parts[_data_index], value.types) for _data_index in value.data]


# modified from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/loader.py#L1115-L1147
def get_gguf_hf_weights_map(
    param_names: List[str],
    model_type: str,
    num_layers: int,
    qual_name: str = "",
):
    """
    GGUF uses this naming convention for their tensors from HF checkpoint:
    `blk.N.BB.weight` and `blk.N.BB.bias`
    where N signifies the block number of a layer, and BB signifies the
    attention/mlp layer components.
    See "Standardized tensor names" in
    https://github.com/ggerganov/ggml/blob/master/docs/gguf.md for details.
    """
    if is_gguf_available():
        from gguf import MODEL_ARCH_NAMES, get_tensor_name_map
    else:
        logger.error(
            "Loading a GGUF checkpoint in JAX, requires GGUF>=0.10.0 to be installed. Please see "
            "https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install gguf>=0.10.0 to load a GGUF checkpoint in JAX.")

    # hack: ggufs have a different name for cohere
    if model_type == "cohere":
        model_type = "command-r"
    elif model_type == "qwen2_moe":
        model_type = "qwen2moe"
    elif model_type == "qwen3_moe":
        model_type = "qwen3moe"
    elif model_type == "gemma3_text":
        model_type = "gemma3"
    arch = None
    for key, value in MODEL_ARCH_NAMES.items():
        if value == model_type:
            arch = key
            break
    if arch is None:
        raise NotImplementedError(
            f"Unknown gguf model_type: {model_type} in gguf-py. "
            "This might because you're using an outdated version of gguf-py package, "
            "you can install `gguf` package from source refer to "
            "https://github.com/ggerganov/llama.cpp/tree/master/gguf-py#development"
        )
    name_map = get_tensor_name_map(arch, num_layers)

    # Use a dummy conversion to get the mapping, because
    # hf => gguf and gguf => hf mappings are reversed
    gguf_to_hf_name_map = {}
    for hf_name in param_names:
        # An exception for qwen2moe/qwen3moe model, where the expert layers are packed
        if model_type in ("qwen2moe", "qwen3moe") and "mlp.experts." in hf_name:
            hf_name = re.sub(r"mlp.experts.\d+.", "mlp.experts.", hf_name)

        name, suffix = hf_name, ""
        if hf_name.endswith(".weight") or hf_name.endswith(".bias"):
            name, suffix = hf_name.rsplit(".", 1)
            suffix = "." + suffix

        gguf_name = name_map.get_name(name)
        if gguf_name is None:
            continue

        gguf_to_hf_name_map[gguf_name + suffix] = qual_name + hf_name

    return gguf_to_hf_name_map


def load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False, target_param_names: Optional[List[str]] = None):
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed
    tokenizer and config attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        return_tensors (`bool`, defaults to `False`):
            Whether to read the tensors from the file and return them. Not doing so is faster
            and only loads the metadata in memory.
        target_param_names (`Optional[List[str]]`, defaults to `None`):
            A list of parameter names in the target JAX model. This is required if `return_tensors` is True
            to correctly map GGUF tensor names to JAX parameter names.
    """
    if is_gguf_available():
        from gguf import GGUFReader, dequantize
    else:
        logger.error(
            "Loading a GGUF checkpoint in JAX, requires GGUF>=0.10.0 to be installed. Please see "
            "https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install gguf>=0.10.0 to load a GGUF checkpoint in JAX.")

    reader = GGUFReader(gguf_checkpoint_path)
    fields = reader.fields
    reader_keys = list(fields.keys())

    parsed_parameters = {k: {} for k in GGUF_TO_TRANSFORMERS_MAPPING}

    architecture = read_field(reader, "general.architecture")[0]
    # NOTE: Some GGUF checkpoints may miss `general.name` field in metadata
    model_name = read_field(reader, "general.name")

    updated_architecture = None
    # in llama.cpp mistral models use the same architecture as llama. We need
    # to add this patch to ensure things work correctly on our side.
    if "llama" in architecture and "mistral" in model_name:
        updated_architecture = "mistral"
    # FIXME: Currently this implementation is only for flan-t5 architecture.
    # It needs to be developed for supporting legacy t5.
    elif "t5" in architecture or "t5encoder" in architecture:
        parsed_parameters["config"]["is_gated_act"] = True
        if "t5encoder" in architecture:
            parsed_parameters["config"]["architectures"] = ["T5EncoderModel"]
        updated_architecture = "t5"
    else:
        updated_architecture = architecture

    if "qwen2moe" in architecture:
        updated_architecture = "qwen2_moe"
    elif "qwen3moe" in architecture:
        updated_architecture = "qwen3_moe"

    # For stablelm architecture, we need to set qkv_bias and use_parallel_residual from tensors
    # If `qkv_bias=True`, qkv_proj with bias will be present in the tensors
    # If `use_parallel_residual=False`, ffn_norm will be present in the tensors
    if "stablelm" in architecture:
        attn_bias_name = {"attn_q.bias", "attn_k.bias", "attn_v.bias"}
        ffn_norm_name = "ffn_norm"
        qkv_bias = any(bias_name in tensor.name for tensor in reader.tensors for bias_name in attn_bias_name)
        use_parallel_residual = any(ffn_norm_name in tensor.name for tensor in reader.tensors)
        parsed_parameters["config"]["use_qkv_bias"] = qkv_bias
        parsed_parameters["config"]["use_parallel_residual"] = not use_parallel_residual

    if architecture not in GGUF_SUPPORTED_ARCHITECTURES and updated_architecture not in GGUF_SUPPORTED_ARCHITECTURES:
        raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")

    # Handle tie_word_embeddings, if lm_head.weight is not present in tensors,
    # tie_word_embeddings is true otherwise false
    exceptions = ["falcon", "bloom"]
    parsed_parameters["config"]["tie_word_embeddings"] = (
        all("output.weight" != tensor.name for tensor in reader.tensors) or architecture in exceptions
    )

    # List all key-value pairs in a columnized format
    for gguf_key, field in reader.fields.items():
        gguf_key = gguf_key.replace(architecture, updated_architecture)
        split = gguf_key.split(".")
        prefix = split[0]
        config_key = ".".join(split[1:])

        value = [_gguf_parse_value(field.parts[_data_index], field.types) for _data_index in field.data]

        if len(value) == 1:
            value = value[0]

        if isinstance(value, str) and architecture in value:
            value = value.replace(architecture, updated_architecture)

        for parameter, parameter_renames in GGUF_TO_TRANSFORMERS_MAPPING.items():
            if prefix in parameter_renames and config_key in parameter_renames[prefix]:
                renamed_config_key = parameter_renames[prefix][config_key]
                if renamed_config_key == -1:
                    continue

                if renamed_config_key is not None:
                    parsed_parameters[parameter][renamed_config_key] = value

                if gguf_key in reader_keys:
                    reader_keys.remove(gguf_key)

        if gguf_key in reader_keys:
            logger.info(f"Some keys were not parsed and added into account {gguf_key} | {value}")

    # Gemma3 GGUF checkpoint only contains weights of text backbone
    if parsed_parameters["config"].get("model_type") == "gemma3":
        parsed_parameters["config"]["model_type"] = "gemma3_text"

    # retrieve config vocab_size from tokenizer
    # Please refer to https://github.com/huggingface/transformers/issues/32526 for more details
    if "vocab_size" not in parsed_parameters["config"]:
        tokenizer_parameters = parsed_parameters["tokenizer"]
        if "tokens" in tokenizer_parameters:
            parsed_parameters["config"]["vocab_size"] = len(tokenizer_parameters["tokens"])
        else:
            logger.warning(
                "Can't find a way to retrieve missing config vocab_size from tokenizer parameters. "
                "This will use default value from model config class and cause unexpected behavior."
            )

    if return_tensors:
        if target_param_names is None:
            raise ValueError("`target_param_names` must be provided when `return_tensors` is True.")

        parsed_parameters["tensors"] = {}
        config = parsed_parameters.get("config", {})
        num_hidden_layers = config.get("num_hidden_layers")
        if num_hidden_layers is None:
            # Fallback for architectures that use a different key
            num_hidden_layers = config.get("n_layer") or config.get("num_layers")

        if num_hidden_layers is None:
            raise ValueError("Could not determine `num_hidden_layers` from GGUF metadata.")

        tensor_key_mapping = get_gguf_hf_weights_map(
            target_param_names, model_type=updated_architecture, num_layers=num_hidden_layers
        )

        ProcessorClass = TENSOR_PROCESSORS.get(architecture, TensorProcessor)
        processor = ProcessorClass(config=config)

        for tensor in tqdm(reader.tensors, desc="Converting and de-quantizing GGUF tensors..."):
            name = tensor.name
            weights = dequantize(tensor.data, tensor.tensor_type)

            result = processor.process(
                weights=weights,
                name=name,
                tensor_key_mapping=tensor_key_mapping,
                parsed_parameters=parsed_parameters,
            )

            weights = result.weights
            name = result.name

            if name not in tensor_key_mapping:
                continue

            name = tensor_key_mapping[name]

            parsed_parameters["tensors"][name] = jnp.asarray(weights)

    if len(reader_keys) > 0:
        logger.info(f"Some keys of the GGUF file were not considered: {reader_keys}")

    return parsed_parameters

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
This file contains the LlamaTensorProcessor class, which is responsible for processing Llama model tensors.
"""
from typing import Optional

import jax.numpy as jnp

# pylint: disable=g-importing-member
# Re-used from generated_code.Qwen3MoeForCausalLM.gguf_utils
from .gguf_utils import GGUFTensor, TensorProcessor


class LlamaTensorProcessor(TensorProcessor):
    """
    A tensor processor for Llama models.
    """

    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        if ".attn_k." in name or ".attn_q." in name:
            num_heads = self.config.get("num_attention_heads")
            num_kv_heads = self.config.get("num_key_value_heads")

            if None in (num_heads, num_kv_heads):
                return GGUFTensor(weights, name, {})
            if ".attn_q." in name:
                weights = self._reverse_permute_weights(weights, num_heads, num_heads)
            elif ".attn_k." in name:
                weights = self._reverse_permute_weights(weights, num_heads, num_kv_heads)
        return GGUFTensor(weights, name, {})

    def _reverse_permute_weights(
        self, weights: jnp.ndarray, n_head: int, num_kv_heads: Optional[int] = None
    ) -> jnp.ndarray:
        # Original permutation implementation
        # https://github.com/ggerganov/llama.cpp/blob/a38b884c6c4b0c256583acfaaabdf556c62fabea/convert_hf_to_gguf.py#L1402-L1408
        if num_kv_heads is not None and n_head != num_kv_heads:
            n_head = num_kv_heads

        dim = weights.shape[0] // n_head // 2
        w = weights.reshape(n_head, dim, 2, *weights.shape[1:])
        return w.swapaxes(2, 1).reshape(weights.shape)

from packaging import version

# The following globals are assumed to be defined in the same file,
# similar to the source PyTorch file.
# from . import _is_gguf_available, _gguf_version, GGUF_MIN_VERSION

def is_gguf_available(min_version: str = GGUF_MIN_VERSION) -> bool:
  """Checks if gguf is available and its version is at least `min_version`."""
  return _is_gguf_available and version.parse(_gguf_version) >= version.parse(min_version)

import re
from typing import Any, Dict, Optional

from flax.core.frozen_dict import FrozenDict
from flax.traverse_util import flatten_dict

from .utils import is_gguf_available
from .utils.logging import get_logger


logger = get_logger(__name__)


def get_gguf_hf_weights_map(
    params: Dict[str, Any] | FrozenDict,
    config: Any,
    model_type: Optional[str] = None,
    num_layers: Optional[int] = None,
):
  """Creates a mapping from GGUF tensor names to JAX parameter names.

  GGUF uses this naming convention for their tensors from HF checkpoint:
  `blk.N.BB.weight` and `blk.N.BB.bias`
  where N signifies the block number of a layer, and BB signifies the
  attention/mlp layer components.
  See "Standardized tensor names" in
  https://github.com/ggerganov/ggml/blob/master/docs/gguf.md for details.

  Args:
    params: The parameter PyTree of the JAX model.
    config: The model configuration object.
    model_type: Optional string specifying the model type.
    num_layers: Optional integer specifying the number of hidden layers.

  Returns:
    A dictionary mapping GGUF tensor names to their corresponding JAX parameter
    names.
  """
  if is_gguf_available():
    from gguf import MODEL_ARCH_NAMES, get_tensor_name_map
  else:
    logger.error(
        "Loading a GGUF checkpoint in JAX, requires GGUF>=0.10.0 to be"
        " installed. Please see "
        "https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for"
        " installation instructions."
    )
    raise ImportError(
        "Please install gguf>=0.10.0 to load a GGUF checkpoint in JAX."
    )

  model_type = config.model_type if model_type is None else model_type
  num_layers = config.num_hidden_layers if num_layers is None else num_layers
  # hack: ggufs have a different name for cohere
  if model_type == "cohere":
    model_type = "command-r"
  elif model_type == "qwen2_moe":
    model_type = "qwen2moe"
  elif model_type == "qwen3_moe":
    model_type = "qwen3moe"
  elif model_type == "gemma3_text":
    model_type = "gemma3"
  arch = None
  for key, value in MODEL_ARCH_NAMES.items():
    if value == model_type:
      arch = key
      break
  if arch is None:
    raise NotImplementedError(
        f"Unknown gguf model_type: {model_type} in gguf-py. "
        "This might because you're using an outdated version of gguf-py"
        " package, you can install `gguf` package from source refer to "
        "https://github.com/ggerganov/llama.cpp/tree/master/gguf-py#development"
    )
  name_map = get_tensor_name_map(arch, num_layers)

  # Use a dummy conversion to get the mapping, because
  # hf => gguf and gguf => hf mappings are reversed
  gguf_to_hf_name_map = {}

  # In JAX/Flax, parameters are a nested dictionary (PyTree).
  # We flatten it to get a dict of {('path', 'to', 'param'): tensor}.
  flat_params = flatten_dict(params)

  for path_tuple in flat_params:
    hf_name = ".".join(map(str, path_tuple))

    hf_name_to_map = hf_name
    # An exception for qwen2moe/qwen3moe model, where the expert layers are
    # packed
    if (
        model_type in ("qwen2moe", "qwen3moe")
        and "mlp.experts." in hf_name_to_map
    ):
      hf_name_to_map = re.sub(
          r"mlp.experts.\d+.", "mlp.experts.", hf_name_to_map
      )

    name, suffix = hf_name_to_map, ""
    if hf_name_to_map.endswith(".weight") or hf_name_to_map.endswith(".bias"):
      name, suffix = hf_name_to_map.rsplit(".", 1)
      suffix = "." + suffix

    gguf_name = name_map.get_name(name)
    if gguf_name is None:
      continue

    gguf_to_hf_name_map[gguf_name + suffix] = hf_name

  # The recursive part of the original PyTorch function is not needed in
  # JAX/Flax. In PyTorch, model.state_dict() might miss some parameters from
  # submodules in complex model compositions, requiring recursion as a fallback.
  # In Flax, the parameter PyTree contains all parameters from all submodules,
  # and flatten_dict() gives us a complete, flat list of all parameter names.
  # Therefore, the recursion would be redundant.

  return gguf_to_hf_name_map

from typing import Any, Dict, Optional

import numpy as np
import jax.numpy as jnp
from tqdm.auto import tqdm

from .integrations import (
    GGUF_CONFIG_MAPPING,
)
from .utils import is_jax_available
from .utils.import_utils import is_gguf_available
from .utils.logging import get_logger
from .gguf_utils_common import (
    GGUF_SUPPORTED_ARCHITECTURES,
    GGUF_TO_TRANSFORMERS_MAPPING,
    TENSOR_PROCESSORS,
    TensorProcessor,
    _gguf_parse_value,
    get_gguf_hf_weights_map,
    read_field,
)


logger = get_logger(__name__)


def load_gguf_checkpoint(
    gguf_checkpoint_path: str, return_tensors: bool = False, model_to_load: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed
    tokenizer and config attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        return_tensors (`bool`, defaults to `False`):
            Whether to read the tensors from the file and return them. Not doing so is faster
            and only loads the metadata in memory.
    """
    if is_gguf_available() and is_jax_available():
        from gguf import GGUFReader, dequantize
    else:
        logger.error(
            "Loading a GGUF checkpoint in JAX, requires both JAX and GGUF>=0.10.0 to be installed. Please see "
            "https://github.com/google/jax and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise ImportError("Please install jax and gguf>=0.10.0 to load a GGUF checkpoint in JAX.")

    reader = GGUFReader(gguf_checkpoint_path)
    fields = reader.fields
    reader_keys = list(fields.keys())

    parsed_parameters = {k: {} for k in GGUF_TO_TRANSFORMERS_MAPPING}

    architecture = read_field(reader, "general.architecture")[0]
    # NOTE: Some GGUF checkpoints may miss `general.name` field in metadata
    model_name = read_field(reader, "general.name")

    updated_architecture = None
    # in llama.cpp mistral models use the same architecture as llama. We need
    # to add this patch to ensure things work correctly on our side.
    if "llama" in architecture and "mistral" in model_name:
        updated_architecture = "mistral"
    # FIXME: Currently this implementation is only for flan-t5 architecture.
    # It needs to be developed for supporting legacy t5.
    elif "t5" in architecture or "t5encoder" in architecture:
        parsed_parameters["config"]["is_gated_act"] = True
        if "t5encoder" in architecture:
            parsed_parameters["config"]["architectures"] = ["T5EncoderModel"]
        updated_architecture = "t5"
    else:
        updated_architecture = architecture

    if "qwen2moe" in architecture:
        updated_architecture = "qwen2_moe"
    elif "qwen3moe" in architecture:
        updated_architecture = "qwen3_moe"

    # For stablelm architecture, we need to set qkv_bias and use_parallel_residual from tensors
    # If `qkv_bias=True`, qkv_proj with bias will be present in the tensors
    # If `use_parallel_residual=False`, ffn_norm will be present in the tensors
    if "stablelm" in architecture:
        attn_bias_name = {"attn_q.bias", "attn_k.bias", "attn_v.bias"}
        ffn_norm_name = "ffn_norm"
        qkv_bias = any(bias_name in tensor.name for tensor in reader.tensors for bias_name in attn_bias_name)
        use_parallel_residual = any(ffn_norm_name in tensor.name for tensor in reader.tensors)
        parsed_parameters["config"]["use_qkv_bias"] = qkv_bias
        parsed_parameters["config"]["use_parallel_residual"] = not use_parallel_residual

    if architecture not in GGUF_SUPPORTED_ARCHITECTURES and updated_architecture not in GGUF_SUPPORTED_ARCHITECTURES:
        raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")

    # Handle tie_word_embeddings, if lm_head.weight is not present in tensors,
    # tie_word_embeddings is true otherwise false
    exceptions = ["falcon", "bloom"]
    parsed_parameters["config"]["tie_word_embeddings"] = (
        all("output.weight" != tensor.name for tensor in reader.tensors) or architecture in exceptions
    )

    # List all key-value pairs in a columnized format
    for gguf_key, field in reader.fields.items():
        gguf_key = gguf_key.replace(architecture, updated_architecture)
        split = gguf_key.split(".")
        prefix = split[0]
        config_key = ".".join(split[1:])

        value = [_gguf_parse_value(field.parts[_data_index], field.types) for _data_index in field.data]

        if len(value) == 1:
            value = value[0]

        if isinstance(value, str) and architecture in value:
            value = value.replace(architecture, updated_architecture)

        for parameter, parameter_renames in GGUF_TO_TRANSFORMERS_MAPPING.items():
            if prefix in parameter_renames and config_key in parameter_renames[prefix]:
                renamed_config_key = parameter_renames[prefix][config_key]
                if renamed_config_key == -1:
                    continue

                if renamed_config_key is not None:
                    parsed_parameters[parameter][renamed_config_key] = value

                if gguf_key in reader_keys:
                    reader_keys.remove(gguf_key)

        if gguf_key in reader_keys:
            logger.info(f"Some keys were not parsed and added into account {gguf_key} | {value}")

    # Gemma3 GGUF checkpoint only contains weights of text backbone
    if parsed_parameters["config"]["model_type"] == "gemma3":
        parsed_parameters["config"]["model_type"] = "gemma3_text"

    # retrieve config vocab_size from tokenizer
    # Please refer to https://github.com/huggingface/transformers/issues/32526 for more details
    if "vocab_size" not in parsed_parameters["config"]:
        tokenizer_parameters = parsed_parameters["tokenizer"]
        if "tokens" in tokenizer_parameters:
            parsed_parameters["config"]["vocab_size"] = len(tokenizer_parameters["tokens"])
        else:
            logger.warning(
                "Can't find a way to retrieve missing config vocab_size from tokenizer parameters. "
                "This will use default value from model config class and cause unexpected behavior."
            )

    if return_tensors:
        parsed_parameters["tensors"] = {}

        tensor_key_mapping = get_gguf_hf_weights_map(model_to_load)
        config = parsed_parameters.get("config", {})

        ProcessorClass = TENSOR_PROCESSORS.get(architecture, TensorProcessor)
        processor = ProcessorClass(config=config)

        for tensor in tqdm(reader.tensors, desc="Converting and de-quantizing GGUF tensors..."):
            name = tensor.name
            weights = dequantize(tensor.data, tensor.tensor_type)

            result = processor.process(
                weights=weights,
                name=name,
                tensor_key_mapping=tensor_key_mapping,
                parsed_parameters=parsed_parameters,
            )

            weights = result.weights
            name = result.name

            if name not in tensor_key_mapping:
                continue

            name = tensor_key_mapping[name]

            parsed_parameters["tensors"][name] = jnp.array(np.copy(weights))

    if len(reader_keys) > 0:
        logger.info(f"Some keys of the GGUF file were not considered: {reader_keys}")

    return parsed_parameters

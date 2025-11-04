
from enum import Enum


class QuantizationMethod(str, Enum):
  BITS_AND_BYTES = "bitsandbytes"
  GPTQ = "gptq"
  AWQ = "awq"
  AQLM = "aqlm"
  VPTQ = "vptq"
  QUANTO = "quanto"
  EETQ = "eetq"
  HIGGS = "higgs"
  HQQ = "hqq"
  COMPRESSED_TENSORS = "compressed-tensors"
  FBGEMM_FP8 = "fbgemm_fp8"
  TORCHAO = "torchao"
  BITNET = "bitnet"
  SPQR = "spqr"
  FP8 = "fp8"
  QUARK = "quark"
  FPQUANT = "fp_quant"
  AUTOROUND = "auto-round"
  MXFP4 = "mxfp4"

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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
"""Quantization configurations."""

from enum import Enum


class AWQLinearVersion(str, Enum):
  """
  The version of the AWQ quantization algorithm to use.
  """

  GEMM = "gemm"
  GEMV = "gemv"
  EXLLAMA = "exllama"
  IPEX = "ipex"

  @staticmethod
  def from_str(version: str):
    """
    Gets the AWQLinearVersion from a string.

    Args:
      version: The string representation of the version.

    Returns:
      The AWQLinearVersion.
    """
    version = version.lower()
    if version == "gemm":
      return AWQLinearVersion.GEMM
    elif version == "gemv":
      return AWQLinearVersion.GEMV
    elif version == "exllama":
      return AWQLinearVersion.EXLLAMA
    elif version == "ipex":
      return AWQLinearVersion.IPEX
    else:
      raise ValueError(f"Unknown AWQLinearVersion {version}")

from enum import Enum


class AwqBackendPackingMethod(str, Enum):
  """Backend packing method for AWQ quantization."""

  AUTOAWQ = "autoawq"
  LLMAWQ = "llm-awq"

import copy
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Union


class QuantizationMethod(str, Enum):
  """Enum for different quantization methods."""

  BITS_AND_BYTES = "bitsandbytes"
  GPTQ = "gptq"
  AWQ = "awq"
  AQLM = "aqlm"
  VPTQ = "vptq"
  QUANTO = "quanto"
  EETQ = "eetq"
  HIGGS = "higgs"
  HQQ = "hqq"
  COMPRESSED_TENSORS = "compressed-tensors"
  FBGEMM_FP8 = "fbgemm_fp8"
  TORCHAO = "torchao"
  BITNET = "bitnet"
  SPQR = "spqr"
  FP8 = "fp8"
  QUARK = "quark"
  FPQUANT = "fp_quant"
  AUTOROUND = "auto-round"
  MXFP4 = "mxfp4"


@dataclass
class QuantizationConfigMixin:
  """Mixin class for quantization config."""

  quant_method: QuantizationMethod

  @classmethod
  def from_dict(
      cls, config_dict: Dict[str, Any], return_unused_kwargs: bool = False, **kwargs
  ) -> Any:
    """Instantiates a [`QuantizationConfigMixin`] from a Python dictionary of parameters.

    Args:
        config_dict: Dictionary that will be used to instantiate the
          configuration object.
        return_unused_kwargs: Whether or not to return a list of unused keyword
          arguments. Used for `from_pretrained` method in `PreTrainedModel`.
        **kwargs: Additional parameters from which to initialize the
          configuration object.

    Returns:
        The configuration object instantiated from those parameters.
    """
    config = cls(**config_dict)

    to_remove = []
    for key, value in kwargs.items():
      if hasattr(config, key):
        setattr(config, key, value)
        to_remove.append(key)
    for key in to_remove:
      kwargs.pop(key, None)

    if return_unused_kwargs:
      return config, kwargs
    else:
      return config

  def to_json_file(self, json_file_path: Union[str, os.PathLike]):
    """Saves this instance to a JSON file.

    Args:
        json_file_path: Path to the JSON file in which this configuration
          instance's parameters will be saved.
    """
    with open(json_file_path, "w", encoding="utf-8") as writer:
      config_dict = self.to_dict()
      json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

      writer.write(json_string)

  def to_dict(self) -> Dict[str, Any]:
    """Serializes this instance to a Python dictionary.

    Returns:
        Dictionary of all the attributes that make up this configuration
        instance.
    """
    return copy.deepcopy(self.__dict__)

  def __iter__(self):
    """Allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin."""
    for attr, value in copy.deepcopy(self.__dict__).items():
      yield attr, value

  def __repr__(self):
    return f"{self.__class__.__name__} {self.to_json_string()}"

  def to_json_string(self, use_diff: bool = True) -> str:
    """Serializes this instance to a JSON string.

    Args:
        use_diff: If set to `True`, only the difference between the config
          instance and the default `PretrainedConfig()` is serialized to JSON
          string.

    Returns:
        String containing all the attributes that make up this configuration
        instance in JSON format.
    """
    if use_diff is True:
      config_dict = self.to_diff_dict()
    else:
      config_dict = self.to_dict()
    return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

  def to_diff_dict(self) -> Dict[str, Any]:
    """Returns a dictionary containing the difference between the current configuration and the default.

    This method is intended to be overridden by subclasses. The base
    implementation returns the full configuration dictionary.

    Returns:
        A dictionary representing the difference from the default configuration.
    """
    return self.to_dict()

  def update(self, **kwargs) -> Dict[str, Any]:
    """Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes.

    Args:
        **kwargs: Dictionary of attributes to tentatively update this class.

    Returns:
        Dictionary containing all the key-value pairs that were not used to
        update the instance.
    """
    to_remove = []
    for key, value in kwargs.items():
      if hasattr(self, key):
        setattr(self, key, value)
        to_remove.append(key)

    # Remove all the attributes that were updated, without modifying the input dict
    unused_kwargs = {
        key: value for key, value in kwargs.items() if key not in to_remove
    }
    return unused_kwargs

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

# Note: The following two classes are dependencies from the original file.
# As no direct MaxText matches were found, they are included here for completeness.
class QuantizationMethod(str, Enum):
  """
  The quantization methods supported by the transformers library.
  """

  BITS_AND_BYTES = "bitsandbytes"
  GPTQ = "gptq"
  AWQ = "awq"
  AQLM = "aqlm"
  VPTQ = "vptq"
  QUANTO = "quanto"
  EETQ = "eetq"
  HIGGS = "higgs"
  HQQ = "hqq"
  COMPRESSED_TENSORS = "compressed-tensors"
  FBGEMM_FP8 = "fbgemm_fp8"
  TORCHAO = "torchao"
  BITNET = "bitnet"
  SPQR = "spqr"
  FP8 = "fp8"
  QUARK = "quark"
  FPQUANT = "fp_quant"
  AUTOROUND = "auto-round"
  MXFP4 = "mxfp4"


@dataclass
class QuantizationConfigMixin:
  """
  Mixin class for quantization config
  """

  quant_method: QuantizationMethod


class AqlmConfig(QuantizationConfigMixin):
  """
  This is a wrapper class about `aqlm` parameters.

  Attributes:
      in_group_size (`int`, *optional*, defaults to 8):
          The group size along the input dimension.
      out_group_size (`int`, *optional*, defaults to 1):
          The group size along the output dimension. It's recommended to always use 1.
      num_codebooks (`int`, *optional*, defaults to 1):
          Number of codebooks for the Additive Quantization procedure.
      nbits_per_codebook (`int`, *optional*, defaults to 16):
          Number of bits encoding a single codebook vector. Codebooks size is 2**nbits_per_codebook.
      linear_weights_not_to_quantize (`Optional[list[str]]`, *optional*):
          List of full paths of `nn.Linear` weight parameters that shall not be quantized.
      kwargs (`dict[str, Any]`, *optional*):
          Additional parameters from which to initialize the configuration object.
  """

  def __init__(
      self,
      in_group_size: int = 8,
      out_group_size: int = 1,
      num_codebooks: int = 1,
      nbits_per_codebook: int = 16,
      linear_weights_not_to_quantize: Optional[List[str]] = None,
      **kwargs,
  ):
    self.quant_method = QuantizationMethod.AQLM
    self.in_group_size = in_group_size
    self.out_group_size = out_group_size
    self.num_codebooks = num_codebooks
    self.nbits_per_codebook = nbits_per_codebook
    self.linear_weights_not_to_quantize = linear_weights_not_to_quantize

    self.post_init()

  def post_init(self):
    """
    Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
    """
    if not isinstance(self.in_group_size, int):
      raise TypeError("in_group_size must be an int")
    if not isinstance(self.out_group_size, int):
      raise TypeError("out_group_size must be an int")
    if not isinstance(self.num_codebooks, int):
      raise TypeError("num_codebooks must be an int")
    if not isinstance(self.nbits_per_codebook, int):
      raise TypeError("nbits_per_codebook must be an int")

    if self.linear_weights_not_to_quantize is not None and not isinstance(self.linear_weights_not_to_quantize, list):
      raise ValueError("linear_weights_not_to_quantize must be a list of strings")

    if self.linear_weights_not_to_quantize is None:
      self.linear_weights_not_to_quantize = []

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Type

import jax
import jax.numpy as jnp
from flax import linen as nn

# Assuming the following imports and placeholders for a MaxText environment
# In a real scenario, these would be imported from shared modules.
# Re-used from `src.MaxText.layers.quantizations.Quantization`
from dataclasses import dataclass
from enum import Enum

# A placeholder for the PreTrainedModel type hint
PreTrainedModel = nn.Module


class QuantizationMethod(str, Enum):
  """
    The quantization methods supported by the transformers library.
    """

  BITS_AND_BYTES = "bitsandbytes"
  GPTQ = "gptq"
  AWQ = "awq"
  AQLM = "aqlm"
  VPTQ = "vptq"
  QUANTO = "quanto"
  EETQ = "eetq"
  HIGGS = "higgs"
  HQQ = "hqq"
  COMPRESSED_TENSORS = "compressed_tensors"
  FBGEMM_FP8 = "fbgemm_fp8"
  TORCHAO = "torchao"
  BITNET = "bitnet"
  SPQR = "spqr"
  FP8 = "fp8"
  QUARK = "quark"
  FPQUANT = "fpquant"
  AUTOROUND = "autoround"
  MXFP4 = "mxfp4"


@dataclass
class QuantizationConfigMixin:
  """
    Mixin class for quantization config
    """

  quant_method: QuantizationMethod


# Placeholder for a utility that doesn't have a direct equivalent in the provided dict.
def get_module_from_name(model: nn.Module, name: str) -> tuple[nn.Module, str]:
  """
    Gets a submodule from a Flax model given its dot-separated name.
    This is a simplified placeholder. A real implementation would need to
    handle nested modules within lists, dicts, etc.
    """
  parts = name.split(".")
  module = model
  for part in parts[:-1]:
    module = getattr(module, part)
  return module, parts[-1]


class HfQuantizer(ABC):
  """
    Abstract class of the HuggingFace quantizer. Supports for now quantizing HF transformers models for inference and/or quantization.
    This class is used only for transformers.PreTrainedModel.from_pretrained and cannot be easily used outside the scope of that method
    yet.

    Attributes
        quantization_config (`...quantization_config.QuantizationConfigMixin`):
            The quantization config that defines the quantization parameters of your model that you want to quantize.
        modules_to_not_convert (`list[str]`, *optional*):
            The list of module names to not convert when quantizing the model.
        required_packages (`list[str]`, *optional*):
            The list of required pip packages to install prior to using the quantizer
        requires_calibration (`bool`):
            Whether the quantization method requires to calibrate the model before using it.
        requires_parameters_quantization (`bool`):
            Whether the quantization method requires to create a new Parameter. For example, for bitsandbytes, it is
            required to create a new xxxParameter in order to properly quantize the model.
    """

  requires_calibration = False
  required_packages = None
  requires_parameters_quantization = False

  def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
    self.quantization_config = quantization_config

    # -- Handle extra kwargs below --
    self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])
    self.pre_quantized = kwargs.pop("pre_quantized", True)

    if not self.pre_quantized and self.requires_calibration:
      raise ValueError(
          f"The quantization method {quantization_config.quant_method} does require the model to be pre-quantized."
          f" You explicitly passed `pre_quantized=False` meaning your model weights are not quantized. Make sure to "
          f"pass `pre_quantized=True` while knowing what you are doing."
      )

  def update_jax_dtype(self, jax_dtype: jnp.dtype) -> jnp.dtype:
    """
        Some quantization methods require to explicitly set the dtype of the model to a
        target dtype. You need to override this method in case you want to make sure that behavior is
        preserved

        Args:
            jax_dtype (`jnp.dtype`):
                The input dtype that is passed in `from_pretrained`
        """
    return jax_dtype

  def update_device_map(self, device_map: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
        Override this method if you want to pass a override the existing device map with a new
        one. E.g. for bitsandbytes, since `accelerate` is a hard requirement, if no device_map is
        passed, the device_map is set to `"auto"``

        Args:
            device_map (`Union[dict, str]`, *optional*):
                The device_map that is passed through the `from_pretrained` method.
        """
    # JAX/MaxText handles device placement via Mesh, so this is a no-op for compatibility.
    return device_map

  def adjust_target_dtype(self, jax_dtype: jnp.dtype) -> jnp.dtype:
    """
        Override this method if you want to adjust the `target_dtype` variable used in `from_pretrained`
        to compute the device_map in case the device_map is a `str`. E.g. for bitsandbytes we force-set `target_dtype`
        to `torch.int8` and for 4-bit we pass a custom enum `accelerate.CustomDtype.int4`.

        Args:
            jax_dtype (`jnp.dtype`, *optional*):
                The jax_dtype that is used to compute the device_map.
        """
    # JAX/MaxText handles device placement via Mesh, so this is a no-op for compatibility.
    return jax_dtype

  def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
    """
        Override this method if you want to adjust the `missing_keys`.

        Args:
            missing_keys (`list[str]`, *optional*):
                The list of missing keys in the checkpoint compared to the state dict of the model
        """
    return missing_keys

  def update_unexpected_keys(self, model, unexpected_keys: List[str], prefix: str) -> List[str]:
    """
        Override this method if you want to adjust the `unexpected_keys`.

        Args:
            unexpected_keys (`list[str]`, *optional*):
                The list of unexpected keys in the checkpoint compared to the state dict of the model
        """
    return unexpected_keys

  def update_missing_keys_after_loading(self, model, missing_keys: List[str], prefix: str) -> List[str]:
    """
        Override this method if you want to adjust the `missing_keys` after loading the model params,
        but before the model is post-processed.

        Args:
            missing_keys (`list[str]`, *optional*):
                The list of missing keys in the checkpoint compared to the state dict of the model
        """
    return missing_keys

  def update_expected_keys(self, model, expected_keys: List[str], loaded_keys: List[str]) -> List[str]:
    """
        Override this method if you want to adjust the `update_expected_keys`.

        Args:
            expected_keys (`list[str]`, *optional*):
                The list of the expected keys in the initialized model.
            loaded_keys (`list[str]`, *optional*):
                The list of the loaded keys in the checkpoint.
        """
    return expected_keys

  def get_special_dtypes_update(self, model: "PreTrainedModel", params: Dict, jax_dtype: jnp.dtype) -> Dict[str, jnp.dtype]:
    """
        returns dtypes for modules that are not quantized - used for the computation of the device_map in case
        one passes a str as a device_map. The method will use the `modules_to_not_convert` that is modified
        in `_process_model_before_weight_loading`.

        Args:
            model (`PreTrainedModel`):
                The model to quantize
            params (`Dict`):
                The model's parameters as a PyTree.
            jax_dtype (`jnp.dtype`):
                The dtype passed in `from_pretrained` method.
        """
    from flax.traverse_util import flatten_dict

    flat_params = flatten_dict(params)
    return {
        ".".join(k): jax_dtype
        for k, _ in flat_params.items()
        if any(m in ".".join(k) for m in self.modules_to_not_convert)
    }

  def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
    """adjust max_memory argument for infer_auto_device_map() if extra memory is needed for quantization"""
    # JAX/MaxText handles memory management differently, this is a no-op for compatibility.
    return max_memory

  def check_quantized_param(
      self,
      model: "PreTrainedModel",
      param_value: jnp.ndarray,
      param_name: str,
      state_dict: Dict[str, Any],
      **kwargs,
  ) -> bool:
    """
        checks if a loaded state_dict component is part of quantized param + some validation; only defined if
        requires_parameters_quantization == True for quantization methods that require to create a new parameters
        for quantization.
        """
    return False

  def create_quantized_param(self, *args, **kwargs) -> jnp.ndarray:
    """
        takes needed components from state_dict and creates quantized param; only applicable if
        requires_parameters_quantization == True
        """
    if not self.requires_parameters_quantization:
      raise AttributeError(f"`.create_quantized_param()` method is not supported by quantizer class {self.__class__.__name__}.")

  def validate_environment(self, *args, **kwargs):
    """
        This method is used to potentially check for potential conflicts with arguments that are
        passed in `from_pretrained`. You need to define it for all future quantizers that are integrated with transformers.
        If no explicit check are needed, simply return nothing.
        """
    return

  def update_tp_plan(self, config):
    "updates the tp plan for the scales"
    return config

  def preprocess_model(self, model: "PreTrainedModel", **kwargs):
    """
        Setting model attributes and/or converting model before weights loading. At this point
        the model should be initialized on the meta device so you can freely manipulate the skeleton
        of the model in order to replace modules in-place. Make sure to override the abstract method `_process_model_before_weight_loading`.

        Args:
            model (`PreTrainedModel`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_before_weight_loading`.
        """
    model.is_quantized = True
    model.quantization_method = self.quantization_config.quant_method
    if self.pre_quantized:
      self._convert_model_for_quantization(model)
    return self._process_model_before_weight_loading(model, **kwargs)

  def postprocess_model(self, model: "PreTrainedModel", **kwargs):
    """
        Post-process the model post weights loading.
        Make sure to override the abstract method `_process_model_after_weight_loading`.

        Args:
            model (`PreTrainedModel`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_after_weight_loading`.
        """
    return self._process_model_after_weight_loading(model, **kwargs)

  def remove_quantization_config(self, model):
    """
        Remove the quantization config from the model.
        """
    if hasattr(model, "hf_quantizer"):
      del model.hf_quantizer
    if hasattr(model.config, "quantization_config"):
      del model.config.quantization_config
    if hasattr(model.config, "_pre_quantization_dtype"):
      del model.config._pre_quantization_dtype
    if hasattr(model, "quantization_method"):
      del model.quantization_method
    model.is_quantized = False

  def dequantize(self, model):
    """
        Potentially dequantize the model to retrieve the original model, with some loss in accuracy / performance.
        Note not all quantization schemes support this.
        """
    model = self._dequantize(model)

    # Delete quantizer and quantization config
    del model.hf_quantizer
    del model.config.quantization_config
    del model.config._pre_quantization_dtype
    del model.quantization_method
    model.is_quantized = False

    return model

  def get_cuda_warm_up_factor(self):
    """
        The factor to be used in `caching_allocator_warmup` to get the number of bytes to pre-allocate to warm up cuda.
        A factor of 2 means we allocate all bytes in the empty model (since we allocate in fp16), a factor of 4 means
        we allocate half the memory of the weights residing in the empty model, etc...
        """
    # This is a PyTorch/CUDA-specific concept and does not directly apply to JAX's memory management.
    # Returning a default value for compatibility.
    return 4

  def _dequantize(self, model):
    raise NotImplementedError(
        f"{self.quantization_config.quant_method} has no implementation of `dequantize`, please raise an issue on GitHub."
    )

  def update_param_name(self, param_name: str) -> str:
    """
        Override this method if you want to adjust the `param_name`.
        """
    return param_name

  @staticmethod
  def get_modules_to_not_convert(
      model: "PreTrainedModel",
      skip_modules: Optional[List[str]] = None,
      keep_in_fp32_modules: Optional[List[str]] = None,
      add_default_skips: bool = False,
  ):
    # Assuming a JAX equivalent of `get_keys_to_not_convert` exists.
    from ..integrations import get_keys_to_not_convert

    if skip_modules is None or add_default_skips:
      modules_to_not_convert = get_keys_to_not_convert(model)
    else:
      modules_to_not_convert = []

    if skip_modules is not None:
      modules_to_not_convert.extend(skip_modules)

    if keep_in_fp32_modules is not None:
      modules_to_not_convert.extend(keep_in_fp32_modules)

    return modules_to_not_convert

  @property
  def is_qat_trainable(self) -> bool:
    """Flag indicating whether the quantized model can carry out quantization aware training"""
    return False

  @property
  def is_compileable(self) -> bool:
    """Flag indicating whether the quantized model can be compiled"""
    return False

  @abstractmethod
  def _process_model_before_weight_loading(self, model, **kwargs): ...

  @abstractmethod
  def _process_model_after_weight_loading(self, model, **kwargs): ...

  @abstractmethod
  def is_serializable(self, safe_serialization=None): ...

  @property
  @abstractmethod
  def is_trainable(self): ...

  def _convert_model_for_quantization(self, model):
    # JAX/Flax models have a static structure defined at initialization.
    # Unlike PyTorch, you cannot dynamically replace submodules in an already
    # instantiated model instance. This method, which relies on PyTorch's
    # dynamic module patching, does not have a direct equivalent in JAX.
    # The idiomatic JAX approach is to define the model with quantized layers
    # from the start, usually controlled by a configuration object.
    # Raising NotImplementedError to indicate that a JAX-specific implementation
    # is required, which would likely involve returning a new model definition
    # rather than modifying one in-place.
    raise NotImplementedError(
        "_convert_model_for_quantization is a PyTorch-specific concept involving "
        "in-place model surgery, which is not supported in JAX/Flax. "
        "Quantized models in JAX are typically defined with quantized layers from the start."
    )


# Placeholder for Llama4TextMLP
class Llama4TextMLP(nn.Module):
  config: Any

  @nn.compact
  def __call__(self, x):
    # Dummy implementation
    return x


class SequentialLlama4TextExperts(nn.Module):
  """
    A module that implements a compressed version of a list of expert modules.
    This is specifically designed to work with Llama4TextExperts in MoE layers.
    """

  config: Any

  def setup(self):
    self.experts = [Llama4TextMLP(self.config) for _ in range(self.config.num_local_experts)]
    self.num_experts = self.config.num_local_experts

  def __call__(
      self,
      hidden_states: jnp.ndarray,
  ) -> jnp.ndarray:
    hidden_states = hidden_states.reshape(self.num_experts, -1, hidden_states.shape[-1])
    # JAX arrays are immutable. We build a list of outputs and then stack them.
    outputs = [self.experts[i](hidden_states[i]) for i in range(self.num_experts)]
    routed_out = jnp.stack(outputs, axis=0)
    return routed_out


MODULES_TO_PATCH_FOR_QUANTIZATION: Dict[str, Dict[str, Union[Type[nn.Module], List[QuantizationMethod]]]] = {
    "Llama4TextExperts": {
        "module_name": SequentialLlama4TextExperts,
        "quantization_methods": [
            QuantizationMethod.COMPRESSED_TENSORS,
            QuantizationMethod.BITS_AND_BYTES,
        ],
    }
}

from typing import Any

def _quantization_type(weight: Any) -> str | None:
  """Generates a string representation of a weight's quantization type."""
  try:
    # Analogous to torchao.dtypes.AffineQuantizedTensor
    from aqt.jax.v2.aqt_tensor import QTensor
  except ImportError:
    return None

  if isinstance(weight, QTensor):
    # AQT's QTensor does not have a `_quantization_type()` method.
    # We create a representative string from its attributes.
    return f"{type(weight).__name__}(qvalue_dtype={weight.qvalue.dtype})"

  # The torchao.quantization.linear_activation_quantized_tensor.LinearActivationQuantizedTensor
  # has no direct JAX/AQT equivalent as a tensor wrapper type.

  return None

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
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
A new class created for quantization.
"""

import flax.linen as nn
import jax.numpy as jnp

from MaxText.common_types import Array, Config
# Re-used module: src.MaxText.layers.linears.mlp_block
from MaxText.layers import linears
# Re-used module: src.MaxText.layers.quantizations.QuantizationMethod
from MaxText.layers.quantizations import QuantizationMethod


class SequentialLlama4TextExperts(nn.Module):
  """
  A module that implements a compressed version of a list of expert modules.
  This is specifically designed to work with Llama4TextExperts in MoE layers.
  This is the JAX/Flax equivalent of the PyTorch SequentialLlama4TextExperts class.
  """

  config: Config

  def setup(self):
    """Initializes a list of expert MLP modules."""
    # The original PyTorch implementation uses Llama4TextMLP, which is a SwiGLU MLP.
    # The MaxText `mlp_block` with appropriate activations serves as the equivalent.
    self.experts = [
        linears.mlp_block(
            config=self.config,
            in_features=self.config.hidden_size,
            intermediate_dim=self.config.intermediate_size,
            activations=self.config.mlp_activations,
            use_pre_norm=False,
            name=f"expert_{i}",
        )
        for i in range(self.config.num_local_experts)
    ]
    self.num_experts = self.config.num_local_experts

  def __call__(self, hidden_states: Array, deterministic: bool = True) -> Array:
    """
    Applies each expert to a slice of the input hidden_states.

    Args:
      hidden_states: The input tensor, expected to be of shape [num_tokens, hidden_dim],
        where num_tokens is a multiple of num_experts.
      deterministic: A flag to control dropout behavior in submodules.

    Returns:
      The output tensor of shape [num_experts, tokens_per_expert, hidden_dim].
    """
    original_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(self.num_experts, -1, original_shape[-1])

    expert_outputs = [
        self.experts[i](hidden_states[i], deterministic=deterministic) for i in range(self.num_experts)
    ]
    routed_out = jnp.stack(expert_outputs, axis=0)

    return routed_out


MODULES_TO_PATCH_FOR_QUANTIZATION = {
    "Llama4TextExperts": {
        "module_name": SequentialLlama4TextExperts,
        "quantization_methods": [
            QuantizationMethod.COMPRESSED_TENSORS,
            QuantizationMethod.BITS_AND_BYTES,
        ],
    }
}

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
A JAX equivalent of the HuggingFace HfQuantizer abstract base class.
"""

import abc
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

import jax.numpy as jnp
from flax.traverse_util import flatten_dict

if TYPE_CHECKING:
    from flax.linen import Module as PreTrainedModel
    from ..utils.quantization_config import QuantizationConfigMixin
else:
    PreTrainedModel = Any
    QuantizationConfigMixin = Any


class HfQuantizer(abc.ABC):
    """
    Abstract class of the HuggingFace quantizer. Supports for now quantizing HF transformers models for inference and/or quantization.
    This class is used only for transformers.PreTrainedModel.from_pretrained and cannot be easily used outside the scope of that method
    yet.

    Attributes
        quantization_config (`transformers.utils.quantization_config.QuantizationConfigMixin`):
            The quantization config that defines the quantization parameters of your model that you want to quantize.
        modules_to_not_convert (`list[str]`, *optional*):
            The list of module names to not convert when quantizing the model.
        required_packages (`list[str]`, *optional*):
            The list of required pip packages to install prior to using the quantizer
        requires_calibration (`bool`):
            Whether the quantization method requires to calibrate the model before using it.
        requires_parameters_quantization (`bool`):
            Whether the quantization method requires to create a new Parameter. For example, for bitsandbytes, it is
            required to create a new xxxParameter in order to properly quantize the model.
    """

    requires_calibration = False
    required_packages = None
    requires_parameters_quantization = False

    def __init__(self, quantization_config: "QuantizationConfigMixin", **kwargs):
        self.quantization_config = quantization_config

        # -- Handle extra kwargs below --
        self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])
        self.pre_quantized = kwargs.pop("pre_quantized", True)

        if not self.pre_quantized and self.requires_calibration:
            raise ValueError(
                f"The quantization method {quantization_config.quant_method} does require the model to be pre-quantized."
                f" You explicitly passed `pre_quantized=False` meaning your model weights are not quantized. Make sure to "
                f"pass `pre_quantized=True` while knowing what you are doing."
            )

    def update_jax_dtype(self, jax_dtype: jnp.dtype) -> jnp.dtype:
        """
        Some quantization methods require to explicitly set the dtype of the model to a
        target dtype. You need to override this method in case you want to make sure that behavior is
        preserved

        Args:
            jax_dtype (`jnp.dtype`):
                The input dtype that is passed in `from_pretrained`
        """
        return jax_dtype

    def update_device_map(self, device_map: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Override this method if you want to pass a override the existing device map with a new
        one. E.g. for bitsandbytes, since `accelerate` is a hard requirement, if no device_map is
        passed, the device_map is set to `"auto"``

        Args:
            device_map (`Union[dict, str]`, *optional*):
                The device_map that is passed through the `from_pretrained` method.
        """
        return device_map

    def adjust_target_dtype(self, jax_dtype: jnp.dtype) -> jnp.dtype:
        """
        Override this method if you want to adjust the `target_dtype` variable used in `from_pretrained`
        to compute the device_map in case the device_map is a `str`. E.g. for bitsandbytes we force-set `target_dtype`
        to `jnp.int8`.

        Args:
            jax_dtype (`jnp.dtype`, *optional*):
                The jax_dtype that is used to compute the device_map.
        """
        return jax_dtype

    def update_missing_keys(self, model: PreTrainedModel, missing_keys: List[str], prefix: str) -> List[str]:
        """
        Override this method if you want to adjust the `missing_keys`.

        Args:
            missing_keys (`list[str]`, *optional*):
                The list of missing keys in the checkpoint compared to the state dict of the model
        """
        return missing_keys

    def update_unexpected_keys(self, model: PreTrainedModel, unexpected_keys: List[str], prefix: str) -> List[str]:
        """
        Override this method if you want to adjust the `unexpected_keys`.

        Args:
            unexpected_keys (`list[str]`, *optional*):
                The list of unexpected keys in the checkpoint compared to the state dict of the model
        """
        return unexpected_keys

    def update_missing_keys_after_loading(
        self, model: PreTrainedModel, missing_keys: List[str], prefix: str
    ) -> List[str]:
        """
        Override this method if you want to adjust the `missing_keys` after loading the model params,
        but before the model is post-processed.

        Args:
            missing_keys (`list[str]`, *optional*):
                The list of missing keys in the checkpoint compared to the state dict of the model
        """
        return missing_keys

    def update_expected_keys(self, model: PreTrainedModel, expected_keys: List[str], loaded_keys: List[str]) -> List[str]:
        """
        Override this method if you want to adjust the `update_expected_keys`.

        Args:
            expected_keys (`list[str]`, *optional*):
                The list of the expected keys in the initialized model.
            loaded_keys (`list[str]`, *optional*):
                The list of the loaded keys in the checkpoint.
        """
        return expected_keys

    def get_special_dtypes_update(self, model: PreTrainedModel, jax_dtype: jnp.dtype) -> Dict[str, jnp.dtype]:
        """
        returns dtypes for modules that are not quantized - used for the computation of the device_map in case
        one passes a str as a device_map. The method will use the `modules_to_not_convert` that is modified
        in `_process_model_before_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            jax_dtype (`jnp.dtype`):
                The dtype passed in `from_pretrained` method.
        """
        if not hasattr(model, "params"):
            return {}

        flattened_params = flatten_dict(model.params, sep=".")
        return {
            name: jax_dtype
            for name in flattened_params.keys()
            if any(m in name for m in self.modules_to_not_convert)
        }

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        """adjust max_memory argument for infer_auto_device_map() if extra memory is needed for quantization"""
        return max_memory

    def check_quantized_param(
        self,
        model: PreTrainedModel,
        param_value: jnp.ndarray,
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        """
        checks if a loaded state_dict component is part of quantized param + some validation; only defined if
        requires_parameters_quantization == True for quantization methods that require to create a new parameters
        for quantization.
        """
        return False

    def create_quantized_param(self, *args, **kwargs) -> jnp.ndarray:
        """
        takes needed components from state_dict and creates quantized param; only applicable if
        requires_parameters_quantization == True
        """
        if not self.requires_parameters_quantization:
            raise AttributeError(
                f"`.create_quantized_param()` method is not supported by quantizer class {self.__class__.__name__}."
            )

    def validate_environment(self, *args, **kwargs):
        """
        This method is used to potentially check for potential conflicts with arguments that are
        passed in `from_pretrained`. You need to define it for all future quantizers that are integrated with transformers.
        If no explicit check are needed, simply return nothing.
        """
        return

    def update_tp_plan(self, config):
        "updates the tp plan for the scales"
        return config

    def preprocess_model(self, model: PreTrainedModel, **kwargs):
        """
        Setting model attributes and/or converting model before weights loading. At this point
        the model should be initialized on the meta device so you can freely manipulate the skeleton
        of the model in order to replace modules in-place. Make sure to override the abstract method `_process_model_before_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_before_weight_loading`.
        """
        model.is_quantized = True
        model.quantization_method = self.quantization_config.quant_method
        if self.pre_quantized:
            self._convert_model_for_quantization(model)
        return self._process_model_before_weight_loading(model, **kwargs)

    def postprocess_model(self, model: PreTrainedModel, **kwargs):
        """
        Post-process the model post weights loading.
        Make sure to override the abstract method `_process_model_after_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_after_weight_loading`.
        """
        return self._process_model_after_weight_loading(model, **kwargs)

    def remove_quantization_config(self, model: PreTrainedModel):
        """
        Remove the quantization config from the model.
        """
        if hasattr(model, "hf_quantizer"):
            del model.hf_quantizer
        if hasattr(model.config, "quantization_config"):
            del model.config.quantization_config
        if hasattr(model.config, "_pre_quantization_dtype"):
            del model.config._pre_quantization_dtype
        if hasattr(model, "quantization_method"):
            del model.quantization_method
        model.is_quantized = False

    def dequantize(self, model: PreTrainedModel):
        """
        Potentially dequantize the model to retrieve the original model, with some loss in accuracy / performance.
        Note not all quantization schemes support this.
        """
        model = self._dequantize(model)

        # Delete quantizer and quantization config
        del model.hf_quantizer
        del model.config.quantization_config
        del model.config._pre_quantization_dtype
        del model.quantization_method
        model.is_quantized = False

        return model

    def get_cuda_warm_up_factor(self):
        """
        The factor to be used in `caching_allocator_warmup` to get the number of bytes to pre-allocate to warm up cuda.
        A factor of 2 means we allocate all bytes in the empty model (since we allocate in fp16), a factor of 4 means
        we allocate half the memory of the weights residing in the empty model, etc...
        """
        # By default we return 4, i.e. half the model size (this corresponds to the case where the model is not
        # really pre-processed, i.e. we do not have the info that weights are going to be 8 bits before actual
        # weight loading)
        return 4

    def _dequantize(self, model: PreTrainedModel):
        raise NotImplementedError(
            f"{self.quantization_config.quant_method} has no implementation of `dequantize`, please raise an issue on GitHub."
        )

    def update_param_name(self, param_name: str) -> str:
        """
        Override this method if you want to adjust the `param_name`.
        """
        return param_name

    @staticmethod
    def get_modules_to_not_convert(
        model: PreTrainedModel,
        skip_modules: Optional[List[str]] = None,
        keep_in_fp32_modules: Optional[List[str]] = None,
        add_default_skips: bool = False,
    ):
        # This import will need to be adapted to the JAX/MaxText ecosystem.
        from ..integrations import get_keys_to_not_convert

        if skip_modules is None or add_default_skips:
            modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            modules_to_not_convert = []

        if skip_modules is not None:
            modules_to_not_convert.extend(skip_modules)

        if keep_in_fp32_modules is not None:
            modules_to_not_convert.extend(keep_in_fp32_modules)

        return modules_to_not_convert

    @property
    def is_qat_trainable(self) -> bool:
        """Flag indicating whether the quantized model can carry out quantization aware training"""
        return False

    @property
    def is_compileable(self) -> bool:
        """Flag indicating whether the quantized model can be compiled"""
        return False

    @abc.abstractmethod
    def _process_model_before_weight_loading(self, model: PreTrainedModel, **kwargs): ...

    @abc.abstractmethod
    def _process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs): ...

    @abc.abstractmethod
    def is_serializable(self, safe_serialization=None): ...

    @property
    @abc.abstractmethod
    def is_trainable(self): ...

    def _convert_model_for_quantization(self, model: PreTrainedModel):
        """
        In-place model modification is not a JAX-native pattern. Model architecture is static after initialization.
        Subclasses should handle model conversion logic within `_process_model_before_weight_loading` by defining
        the correct quantized layers from the start, potentially using configuration flags.
        """
        raise NotImplementedError(
            "In-place model conversion is not supported in JAX. Quantization logic should be handled during model definition."
        )

import importlib
from typing import Any

from absl import logging
from packaging import version
import jax
import jax.numpy as jnp
from flax.linen import Module as nn_Module

from maxtext.layers.quantizations import QuantizationConfigMixin
# The following imports are placeholders for JAX-compatible equivalents
# of the original PyTorch utilities. Their implementation is assumed to exist.
# from maxtext.utils.integrations import replace_with_aqlm_linear
# from maxtext.utils import is_aqlm_available

# Placeholder for a JAX-compatible `replace_with_aqlm_linear` function.
# In a real scenario, this would likely involve transforming the model's
# parameter structure or using a different model definition.
def replace_with_aqlm_linear(model: nn_Module, **kwargs) -> nn_Module:
  """Placeholder for a function that replaces linear layers with AQLM layers."""
  logging.warning(
      "replace_with_aqlm_linear is a placeholder and does not perform any model modification."
  )
  return model

# Placeholder for a JAX-compatible `is_aqlm_available` function.
def is_aqlm_available() -> bool:
  """Checks if the 'aqlm' package is available."""
  return importlib.util.find_spec("aqlm") is not None


class AqlmHfQuantizer:
  """
  Quantizer of the AQLM method. Enables the loading of prequantized models.
  """

  requires_calibration = True
  required_packages = ["aqlm"]
  optimum_quantizer = None

  def __init__(
      self, quantization_config: QuantizationConfigMixin, **kwargs: Any
  ):
    self.quantization_config = quantization_config

  def validate_environment(self, *args: Any, **kwargs: Any):
    # The 'accelerate' check is PyTorch-specific and not relevant for JAX.
    if not is_aqlm_available():
      raise ImportError(
          "Using `aqlm` quantization requires AQLM: `pip install aqlm[gpu,cpu]`"
      )

  def update_jax_dtype(self, jax_dtype: jnp.dtype | None) -> jnp.dtype:
    """Updates the jax_dtype if it is not specified."""
    if jax_dtype is None:
      if len(jax.devices("gpu")) > 0:
        jax_dtype = jnp.float16
        logging.info(
            "CUDA available. Assuming AQLM inference on GPU and loading the"
            " model in `jnp.float16`. To overwrite it, set `jax_dtype`"
            " manually."
        )
      else:
        jax_dtype = jnp.float32
        logging.info(
            "CUDA is unavailable. Assuming AQLM inference on CPU and loading"
            " the model in `jnp.float32`. To overwrite it, set `jax_dtype`"
            " manually."
        )
    return jax_dtype

  def _process_model_before_weight_loading(
      self,
      model: nn_Module,
      **kwargs: Any,
  ) -> nn_Module:
    # In JAX, model modification is not in-place. We assume a function that
    # transforms the model definition or state.
    model = replace_with_aqlm_linear(
        model,
        quantization_config=self.quantization_config,
        linear_weights_not_to_quantize=self.quantization_config.linear_weights_not_to_quantize,
    )
    # Assuming model has a config attribute, similar to Hugging Face models.
    model.config.quantization_config = self.quantization_config
    return model

  def _process_model_after_weight_loading(
      self, model: nn_Module, **kwargs: Any
  ) -> nn_Module:
    return model

  @property
  def is_trainable(self) -> bool:
    """Returns whether the quantized model is trainable."""
    aqlm_supports_training = version.parse(
        importlib.metadata.version("aqlm")
    ) >= version.parse("1.0.2")
    if aqlm_supports_training:
      return True
    else:
      logging.warning(
          "Currently installed `aqlm` version"
          f" ({importlib.metadata.version('aqlm')}) doesn't support training."
          " If you wish to train a quantized model, please update `aqlm` with"
          " `pip install aqlm>=1.0.2`"
      )
      return False

  def is_serializable(self, safe_serialization: Any = None) -> bool:
    """Returns whether the quantized model is serializable."""
    return True

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING, Any, Dict

import jax.numpy as jnp

from ..utils import is_auto_round_available, logging
from ..utils.quantization_config import QuantizationConfigMixin
from .base import HfQuantizer


if TYPE_CHECKING:
    from flax.linen import Module as PreTrainedModel


logger = logging.get_logger(__name__)


class AutoRoundQuantizer(HfQuantizer):
    """
    Quantizer of the AutoRound method. (https://huggingface.co/papers/2309.05516)
    """

    # AutoRound requires data calibration - we support only inference
    requires_calibration = True
    required_packages = ["auto_round"]

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.device_map: Dict[str, Any] | None = None
        self.used_backends: Any = None

    def validate_environment(self, *args, **kwargs):
        self.device_map = kwargs.get("device_map")
        if not is_auto_round_available():
            raise ImportError(
                "Loading an AutoRound quantized model requires auto-round library (`pip install 'auto-round>=0.5'`)"
            )

    def update_jax_dtype(self, jax_dtype: jnp.dtype) -> jnp.dtype:
        if jax_dtype is None:
            jax_dtype = jnp.bfloat16
            logger.info("Loading the model in `jnp.bfloat16`. To overwrite it, set `jax_dtype` manually.")
        return jax_dtype

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs) -> "PreTrainedModel":
        if hasattr(model, "main_input_name") and model.main_input_name != "input_ids":
            logger.warning("AutoRound offers only limited support for models that are not strictly text-based.")
        from auto_round.inference.convert_model import convert_hf_model, infer_target_device

        if self.pre_quantized:
            target_device = infer_target_device(self.device_map)
            model, used_backends = convert_hf_model(model, target_device)
            self.used_backends = used_backends
        return model

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs) -> "PreTrainedModel":
        if self.pre_quantized:
            from auto_round.inference.convert_model import post_init

            post_init(model, self.used_backends)
        else:
            raise ValueError("AutoRound only sports pre-quantized models.")
        return model

    @property
    def is_trainable(self) -> bool:
        return False

    def is_serializable(self, safe_serialization=None):
        ## for gptq/awq models, the quantization config will be changed
        return True

import importlib.metadata
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from absl import logging
from packaging import version
import jax
import jax.numpy as jnp

# Reused modules from MaxText
# generated_code.Qwen3MoeForCausalLM.quantization.HfQuantizer
from transformers.utils.quantization_config import HfQuantizer
# generated_code.Qwen3MoeForCausalLM.quantization.AWQLinearVersion
from transformers.utils.quantization_config import AWQLinearVersion


if TYPE_CHECKING:
  from flax.linen import Module as PreTrainedModel


# Placeholder for utility functions that would need a JAX implementation
# These are not provided in the JAX_MODULES_DICT
def is_auto_awq_available():
  """Checks if the autoawq package is available."""
  try:
    import autoawq  # noqa: F401

    return True
  except ImportError:
    return False


def is_accelerate_available():
  """Checks if the accelerate package is available."""
  try:
    import accelerate  # noqa: F401

    return True
  except ImportError:
    return False


def _is_gpu_available():
  """Checks if a GPU device is available in JAX."""
  return any(d.platform == "gpu" for d in jax.devices())


# Placeholders for model transformation functions. In a real JAX implementation,
# these would likely be replaced by defining the model with quantized layers from the start.
def replace_with_awq_linear(model, quantization_config, modules_to_not_convert):
  logging.warning("replace_with_awq_linear is a placeholder and does not modify the model.")
  return model, True  # Assume replacement happened for logic flow


def replace_quantization_scales(model, model_type):
  logging.warning("replace_quantization_scales is a placeholder and does not modify the model.")
  return model


def fuse_awq_modules(model, quantization_config):
  logging.warning("fuse_awq_modules is a placeholder and does not modify the model.")
  return model


def post_init_awq_exllama_modules(model, exllama_config):
  logging.warning("post_init_awq_exllama_modules is a placeholder and does not modify the model.")
  return model


def post_init_awq_ipex_modules(model):
  logging.warning("post_init_awq_ipex_modules is a placeholder and does not modify the model.")
  return model


logger = logging


class AwqQuantizer(HfQuantizer):
  """
  4-bit quantization for Activation-aware Weight Quantization(AWQ) (https://huggingface.co/papers/2306.00978)
  """

  # AWQ requires data calibration - we support only inference
  requires_calibration = True

  required_packages = ["awq", "accelerate"]

  def __init__(self, quantization_config, **kwargs):
    super().__init__(quantization_config, **kwargs)

  def validate_environment(self, device_map: Optional[Dict[str, Any]] = None, **kwargs):
    if not is_auto_awq_available():
      raise ImportError("Loading an AWQ quantized model requires auto-awq library (`pip install autoawq`)")

    if not is_accelerate_available():
      raise ImportError("Loading an AWQ quantized model requires accelerate (`pip install accelerate`)")

    if self.quantization_config.version == AWQLinearVersion.GEMM and not _is_gpu_available():
      logger.warning("No GPU found, consider switching to the IPEX version for CPU-only execution.")
      self.quantization_config.version = AWQLinearVersion.IPEX

    if self.quantization_config.version == AWQLinearVersion.IPEX:
      if version.parse(importlib.metadata.version("autoawq")) < version.parse("0.2.6"):
        raise RuntimeError(
            "To use IPEX backend, you need autoawq>0.2.6. Please install the latest version or from source."
        )
      # The concept of device_map is PyTorch-specific. In JAX, device placement is
      # handled by the mesh. We adapt the checks to be about the general environment.
      if not any(d.platform == "cpu" for d in jax.devices()):
        logger.warning("You have loaded an IPEX AWQ model but have no CPU device available.")
    else:  # GEMM or other GPU-based versions
      if not _is_gpu_available():
        raise RuntimeError(
            "GPU is required to run AWQ quantized model. You can use IPEX version AWQ if you have an Intel CPU"
        )

      # This check warns if the user might be accidentally running on CPU.
      # In JAX, this is less likely if a mesh is configured, but the warning is still useful.
      if device_map is None and not _is_gpu_available():
        logger.warning(
            "You have loaded an AWQ model on CPU and have a GPU device available, make sure to set "
            "your model on a GPU device in order to run your model."
        )
      elif device_map is not None:
        # The original check is for "cpu" or "disk" in device_map values.
        # This is a PyTorch-specific concept. The JAX equivalent is to check if the user
        # is trying to use a non-GPU device with a GPU-only kernel.
        # This is hard to map directly, so we'll omit the detailed device_map check,
        # as the primary check for GPU availability already covers the main failure case.
        pass

  def update_jax_dtype(self, jax_dtype: Optional[jnp.dtype]) -> jnp.dtype:
    if jax_dtype is None:
      jax_dtype = jnp.float16
      logger.info("Loading the model in `jnp.float16`. To overwrite it, set `jax_dtype` manually.")
    elif jax_dtype == jnp.bfloat16 and _is_gpu_available():
      logger.warning("`jnp.bfloat16` is not supported for AWQ CUDA/XPU kernels yet. Casting to `jnp.float16`.")
      jax_dtype = jnp.float16
    elif jax_dtype != jnp.float16 and _is_gpu_available():
      logger.warning("We suggest you to set `jax_dtype=jnp.float16` for better efficiency on CUDA/XPU with AWQ.")
    return jax_dtype

  def _process_model_before_weight_loading(
      self, model: "PreTrainedModel", keep_in_fp32_modules: Optional[List[str]] = None, **kwargs
  ):
    # In JAX, model modification is not typically done in-place. These functions
    # would return a new model definition or modified parameters.
    # For this translation, we call placeholder functions to maintain the logic flow.
    self.modules_to_not_convert = self.get_modules_to_not_convert(
        model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules, add_default_skips=True
    )

    model, has_been_replaced = replace_with_awq_linear(
        model, quantization_config=self.quantization_config, modules_to_not_convert=self.modules_to_not_convert
    )

    model = replace_quantization_scales(model, model.config.model_type)

    if not has_been_replaced:
      logger.warning(
          "You are loading an AWQ model but no linear modules were found in your model."
          " Please double check your model architecture, or submit an issue on github if you think this is a bug."
      )

  def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
    if self.quantization_config.do_fuse:
      model = fuse_awq_modules(model, self.quantization_config)
      model._awq_is_fused = True  # TODO: consider storing this flag in model.config instead

    if self.quantization_config.version == AWQLinearVersion.EXLLAMA:
      model = post_init_awq_exllama_modules(model, self.quantization_config.exllama_config)

    if self.quantization_config.version == AWQLinearVersion.IPEX:
      model = post_init_awq_ipex_modules(model)

  def is_serializable(self, safe_serialization: Optional[bool] = None) -> bool:
    # AWQ through auto-awq has been always serializable, except if the model is fused.
    if self.quantization_config.do_fuse:
      logger.warning("You cannot save an AWQ model that uses fused modules!")
      return False

    if self.quantization_config.version == AWQLinearVersion.EXLLAMA:
      logger.warning("You cannot save an AWQ model that uses Exllama backend!")
      return False

    return True

  @property
  def is_trainable(self) -> bool:
    # AWQ supports PEFT fine-tuning from version 0.2.0
    MIN_AWQ_VERSION_FOR_PEFT = "0.2.0"
    return version.parse(importlib.metadata.version("autoawq")) >= version.parse(MIN_AWQ_VERSION_FOR_PEFT)

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" BitNetHfQuantizer class for MaxText. """

from typing import Any, Dict, List, Optional, Union

from absl import logging
import jax
import jax.numpy as jnp

# from ..integrations import replace_with_bitnet_linear # Assumed to be in maxtext.layers.quantizations.bitnet
# from ..utils import is_accelerate_available # Assumed to be in maxtext.utils.import_utils
# from .base import HfQuantizer # Assumed to be in maxtext.layers.quantizations.base


class BitNetHfQuantizer(HfQuantizer):
  """
    1.58-bit quantization from BitNet quantization method:
    Before loading: it converts the linear layers into BitLinear layers during loading.

    Check out the paper introducing this method: https://huggingface.co/papers/2402.17764
    """

  requires_parameters_quantization = False
  requires_calibration = True

  required_packages = ["accelerate"]

  def __init__(self, quantization_config: Any, **kwargs: Any):
    super().__init__(quantization_config, **kwargs)
    self.quantization_config = quantization_config

  def validate_environment(self, *args: Any, **kwargs: Any) -> None:
    if not is_accelerate_available():
      raise ImportError("Loading a BitNet quantized model requires accelerate (`pip install accelerate`)")

    if kwargs.get("from_tf", False) or kwargs.get("from_pt", False):
      raise ValueError(
          "Loading ternary weights from tf/pt is currently not supported, please make"
          " sure the weights are in Flax/JAX format."
      )

    if not jax.devices("gpu"):
      logging.warning(
          "You don't have a GPU available to load the model, the inference will be slow because of weight unpacking"
      )
      return

    device_map = kwargs.get("device_map")
    if device_map is None:
      logging.warning(
          "You have loaded a BitNet model on CPU and have a CUDA device available, make sure to set "
          "your model on a GPU device in order to run your model."
      )
    elif device_map is not None:
      if isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
        raise ValueError(
            "You are attempting to load a BitNet model with a device_map that contains a CPU or disk device."
            "This is not supported. Please remove the CPU or disk device from the device_map."
        )

  def _process_model_after_weight_loading(self, model: Any, **kwargs: Any) -> Any:
    return model

  def _process_model_before_weight_loading(
      self,
      model: Any,
      keep_in_fp32_modules: Optional[List[str]] = None,
      **kwargs: Any,
  ) -> Any:
    from maxtext.layers.quantizations.bitnet import replace_with_bitnet_linear

    self.modules_to_not_convert = self.get_modules_to_not_convert(
        model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
    )

    model = replace_with_bitnet_linear(
        model,
        modules_to_not_convert=self.modules_to_not_convert,
        quantization_config=self.quantization_config,
        pre_quantized=self.pre_quantized,
    )
    return model

  def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
    max_memory = {key: val * 0.90 for key, val in max_memory.items()}
    return max_memory

  def adjust_target_dtype(self, target_dtype: jnp.dtype) -> jnp.dtype:
    target_dtype = jnp.int8
    return target_dtype

  def is_serializable(self, safe_serialization: Any = None) -> bool:
    return True

  @property
  def is_trainable(self) -> bool:
    return (
        self.quantization_config.linear_class == "autobitlinear"
        and self.quantization_config.quantization_mode == "online"
    )

  @property
  def is_qat_trainable(self) -> bool:
    """Flag indicating whether the quantized model can carry out quantization aware training"""
    return (
        self.quantization_config.linear_class == "autobitlinear"
        and self.quantization_config.quantization_mode == "online"
    )

import functools
import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from absl import logging
from packaging import version

# from generated_code.Qwen3MoeForCausalLM.quantization import HfQuantizer
from transformers.utils.quantization_config import HfQuantizer

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    from flax.linen import Module as FlaxModule


class Bnb4BitHfQuantizer(HfQuantizer):
    """
    4-bit quantization from bitsandbytes.py quantization method.

    This class is a JAX-based adaptation of the Hugging Face Bnb4BitHfQuantizer.
    However, the core `bitsandbytes` library is built on PyTorch and CUDA, and
    does not have a JAX backend. Therefore, most methods in this class are

    placeholders that raise `NotImplementedError` or return default values. It
    preserves the API structure for compatibility but is not functional for
    actual 4-bit quantization in a pure JAX environment. For quantization in
    JAX, consider using libraries like AQT.
    """

    use_keep_in_fp32_modules = True
    requires_parameters_quantization = True
    requires_calibration = False

    required_packages = ["bitsandbytes", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if self.quantization_config.llm_int8_skip_modules is not None:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

    def validate_environment(self, *args, **kwargs):
        """
        Validates the environment for bitsandbytes quantization. This method is
        deeply tied to the PyTorch, bitsandbytes, and accelerate ecosystems and
        is not applicable in a pure JAX environment.
        """
        raise NotImplementedError(
            "Bnb4BitHfQuantizer and the bitsandbytes library are not supported in JAX. "
            "Please use a JAX-native quantization method like AQT."
        )

    def adjust_target_dtype(self, target_dtype: "jnp.dtype") -> "jnp.dtype":
        """
        This method relies on `accelerate.utils.CustomDtype`, which is part of
        the PyTorch ecosystem and not available in JAX.
        """
        raise NotImplementedError("adjust_target_dtype is not applicable in JAX.")

    def check_quantized_param(
        self,
        model: "FlaxModule",
        param_value: "jax.Array",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        """
        This method checks for bitsandbytes-specific parameter types
        (`bnb.nn.Params4bit`), which do not exist in JAX.
        """
        return False

    def create_quantized_param(
        self,
        model: "FlaxModule",
        param_value: "jax.Array",
        param_name: str,
        target_device: "jax.Device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        """
        This method's logic is to create and load `bitsandbytes.nn.Params4bit`
        objects, which are PyTorch-specific. This functionality is not
        available in JAX.
        """
        raise NotImplementedError("create_quantized_param for bitsandbytes is not supported in JAX.")

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer.adjust_max_memory
    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        # need more space for buffers that are created during quantization
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer.update_torch_dtype
    def update_jax_dtype(self, jax_dtype: "jnp.dtype") -> "jnp.dtype":
        import jax.numpy as jnp

        if jax_dtype is None:
            # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
            logging.info(
                "Overriding jax_dtype=%s with `jax_dtype=jnp.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 4-bit. "
                "Pass your own jax_dtype to specify the dtype of the remaining non-linear layers or pass"
                " jax_dtype=jnp.float16 to remove this warning.",
                jax_dtype,
            )
            jax_dtype = jnp.float16
        return jax_dtype

    def update_device_map(self, device_map):
        # Device mapping is handled differently in JAX (via Mesh) and this
        # accelerate-specific logic does not apply.
        if device_map is None:
            logging.info(
                "The device_map was not initialized. JAX uses a Mesh for device placement. "
                "This method is a placeholder for API compatibility."
            )
        return device_map

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer._process_model_before_weight_loading
    def _process_model_before_weight_loading(
        self,
        model: "FlaxModule",
        device_map,
        keep_in_fp32_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        # This method relies on in-place model modification (`replace_with_bnb_linear`)
        # which is not a JAX/Flax pattern. JAX models are defined with their
        # quantized layers from the start.
        raise NotImplementedError("_process_model_before_weight_loading for bitsandbytes is not supported in JAX.")

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer._process_model_after_weight_loading with 8bit->4bit
    def _process_model_after_weight_loading(self, model: "FlaxModule", **kwargs):
        model.is_loaded_in_4bit = True
        model.is_4bit_serializable = self.is_serializable()
        return model

    def is_serializable(self, safe_serialization=None):
        # This check is specific to the bitsandbytes library version.
        # Since bitsandbytes is not used in JAX, we return False.
        logging.warning(
            "bitsandbytes serialization is not available in JAX. "
            "Saving 4-bit models is not supported through this quantizer."
        )
        return False

    @functools.cached_property
    def is_bnb_supports_quant_storage_module(self) -> bool:
        """
        Determines if a JAX equivalent of bitsandbytes would support
        the `module` parameter in `Params4bit.from_prequantized`.
        This is a placeholder as bitsandbytes is not JAX-compatible.
        """
        return False

    @property
    def is_trainable(self) -> bool:
        return True

    def _dequantize(self, model):
        # This method relies on in-place model modification (`dequantize_and_replace`)
        # which is not a JAX/Flax pattern.
        raise NotImplementedError("_dequantize for bitsandbytes is not supported in JAX.")

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
A JAX-based Bnb8BitHfQuantizer.
This is a non-functional placeholder for API compatibility, as the underlying
'bitsandbytes' library is not JAX-compatible.
"""

import importlib
from typing import TYPE_CHECKING, Any, Optional, Union

from absl import logging
from packaging import version

import jax
import jax.numpy as jnp

from .base import HfQuantizer


if TYPE_CHECKING:
    from flax.linen import Module as FlaxModule

# MaxText matched dependencies:
# No matching modules found for the core functionality of this file.
# The original PyTorch code relies heavily on the `bitsandbytes` library,
# which is CUDA-specific and not compatible with JAX.
# Therefore, most methods are replaced with `NotImplementedError`.


class Bnb8BitHfQuantizer(HfQuantizer):
    """
    8-bit quantization from bitsandbytes quantization method:
        before loading: converts transformer layers into Linear8bitLt during loading: load 16bit weight and pass to the
        layer object after: quantizes individual weights in Linear8bitLt into 8bit at fitst .cuda() call
    saving:
        from state dict, as usual; saves weights and 'SCB' component
    loading:
        need to locate SCB component and pass to the Linear8bitLt object

    This is a non-functional placeholder for API compatibility, as the underlying
    'bitsandbytes' library is not JAX-compatible.
    """

    use_keep_in_fp32_modules = True
    requires_parameters_quantization = True
    requires_calibration = False

    required_packages = ["bitsandbytes", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if self.quantization_config.llm_int8_skip_modules is not None:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

    def validate_environment(self, *args, **kwargs):
        raise NotImplementedError(
            "8-bit quantization with bitsandbytes is not supported in JAX. "
            "The `bitsandbytes` library is a PyTorch/CUDA-specific library. "
            "Please consider using a JAX-native quantization library like AQT."
        )

    def adjust_max_memory(self, max_memory: dict[str, Union[int, str]]) -> dict[str, Union[int, str]]:
        # need more space for buffers that are created during quantization
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def update_jax_dtype(self, jax_dtype: jnp.dtype) -> jnp.dtype:
        if jax_dtype is None:
            # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
            logging.info(
                "Overriding jax_dtype=%s with `jax_dtype=jnp.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
                "Pass your own jax_dtype to specify the dtype of the remaining non-linear layers or pass"
                " jax_dtype=jnp.float16 to remove this warning.",
                jax_dtype,
            )
            jax_dtype = jnp.float16
        return jax_dtype

    def update_device_map(self, device_map):
        if device_map is None:
            if len(jax.devices("gpu")) > 0:
                device_map = {"": jax.devices("gpu")[0]}
            elif len(jax.devices("tpu")) > 0:
                device_map = {"": jax.devices("tpu")[0]}
            else:
                device_map = {"": "cpu"}
            logging.info(
                "The device_map was not initialized. "
                f"Setting device_map to {device_map}. "
                "If you want to use the model for inference, please set device_map ='auto' "
            )
        return device_map

    def adjust_target_dtype(self, target_dtype: jnp.dtype) -> jnp.dtype:
        if target_dtype != jnp.int8:
            logging.info("target_dtype {target_dtype} is replaced by `jnp.int8` for 8-bit BnB quantization")
        return jnp.int8

    def check_quantized_param(
        self,
        model: "FlaxModule",
        param_value: jax.Array,
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ):
        raise NotImplementedError("Bitsandbytes is not supported in JAX.")

    def create_quantized_param(
        self,
        model: "FlaxModule",
        param_value: jax.Array,
        param_name: str,
        target_device: Any,
        state_dict: dict[str, Any],
        unexpected_keys: Optional[list[str]] = None,
    ):
        """
        combines logic from _load_state_dict_into_meta_model and .integrations.bitsandbytes.py::set_module_quantized_tensor_to_device()
        needs aux items from state dicts, if found - removes them from unexpected_keys
        """
        raise NotImplementedError("Bitsandbytes is not supported in JAX.")

    def _process_model_after_weight_loading(self, model: "FlaxModule", **kwargs):
        model.is_loaded_in_8bit = True
        model.is_8bit_serializable = self.is_serializable()
        return model

    def _process_model_before_weight_loading(
        self,
        model: "FlaxModule",
        device_map,
        keep_in_fp32_modules: Optional[list[str]] = None,
        **kwargs,
    ):
        raise NotImplementedError(
            "In-place model modification for quantization (`replace_with_bnb_linear`) is not a JAX-native pattern. "
            "In JAX/Flax, quantized layers should be defined at model initialization."
        )

    def is_serializable(self, safe_serialization=None):
        logging.warning(
            "You are calling `save_pretrained` to a 8-bit converted model, but your `bitsandbytes` version doesn't support it. "
            "If you want to save 8-bit models, make sure to have `bitsandbytes>0.37.2` installed. You will most likely face errors or"
            " unexpected behaviours."
        )
        return False

    @property
    def is_trainable(self) -> bool:
        try:
            is_trainable = version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.37.0")
        except importlib.metadata.PackageNotFoundError:
            is_trainable = False
        return is_trainable

    def _dequantize(self, model):
        raise NotImplementedError(
            "Dequantization of bitsandbytes models is not supported in JAX as it relies on in-place model modification."
        )

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
JAX implementation of the `CompressedTensorsHfQuantizer`.
Functionally equivalent to the PyTorch version, but adapted for JAX/Flax.
"""

import os
import re
from typing import Any, List

from absl import logging
import jax.numpy as jnp

# Assuming these utils are available in the JAX environment
from ..utils import is_compressed_tensors_available, is_torch_available
from ..utils.quantization_config import CompressedTensorsConfig
from .base import HfQuantizer


class CompressedTensorsHfQuantizer(HfQuantizer):
    """
    Quantizer for the compressed_tensors package.  Loads and restores models to
    quantized state with compressed_tensors
    """

    requires_calibration = True
    required_packages = ["compressed_tensors"]

    def __init__(self, quantization_config: CompressedTensorsConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires the compressed-tensors library: "
                "`pip install compressed-tensors`"
            )

        # Call post_init here to ensure proper config setup when `run_compressed`
        # is provided directly via CompressedTensorsConfig, and to avoid duplicate logging.

        quantization_config.post_init()
        from compressed_tensors.compressors import ModelCompressor

        self.compressor = ModelCompressor.from_compression_config(quantization_config)
        self.run_compressed = quantization_config.run_compressed
        self.quantization_config = quantization_config

    def update_missing_keys_after_loading(self, model: Any, missing_keys: List[str], prefix: str) -> List[str]:
        """
        Update missing keys after loading the model. This is necessary for compressed tensors
        to load the model correctly. We expect weights to be present in missing keys.
        The weight's are re-constructed by ModelCompressor in _process_model_after_weight_loading

        This function cleans up expected missing keys and returns the remaining missing keys
        """

        if self.run_compressed:
            return missing_keys

        # We expect some keys to be missing for
        # compressed models
        # This is fine as the weights are reconstructed by ModelCompressor
        # in _process_model_after_weight_loading

        expected_missing_keys = self.compressor.get_missing_module_keys(model)
        return [
            key for key in missing_keys if not any(re.match(f".*{pattern}", key) for pattern in expected_missing_keys)
        ]

    def update_unexpected_keys(self, model: Any, unexpected_keys: List[str], prefix: str) -> List[str]:
        """
        Override this method if you want to adjust the `unexpected_keys`.

        Args:
            unexpected_keys (`list[str]`, *optional*):
                The list of unexpected keys in the checkpoint compared to the state dict of the model
        """

        if self.run_compressed:
            return unexpected_keys

        # We expect some unexpected keys in model
        # safetensors file for compressed models
        keys_to_ignore = self.compressor.get_unexpected_file_keys(model)
        return [key for key in unexpected_keys if not any(re.match(f".*{pattern}", key) for pattern in keys_to_ignore)]

    def validate_environment(self, *args, **kwargs):
        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires the compressed-tensors library: "
                "`pip install compressed-tensors`"
            )
        if not is_torch_available():
            # torch already should be installed as part of compressed tensors
            raise ImportError("torch is required for using compressed-tensors quantization")

    def update_jax_dtype(self, jax_dtype: jnp.dtype) -> jnp.dtype:
        if jax_dtype is None:
            logging.info("Loading model using jnp.float16 for compressed-tensors quantization")
            jax_dtype = jnp.float16
        elif jax_dtype != jnp.float16:
            logging.info(
                "We suggest you to set `jax_dtype=jnp.float16` for better efficiency with compressed_tensors."
            )
        return jax_dtype

    def _process_model_before_weight_loading(self, model: Any, **kwargs):
        # NOTE: This function originally modified the model in-place, which is not
        # a JAX-native pattern. The `compressed_tensors` library would need a
        # JAX-compatible functional API to work correctly.
        from compressed_tensors.quantization import apply_quantization_config

        ct_quantization_config = self.compressor.quantization_config

        if self.run_compressed:
            apply_quantization_config(model, ct_quantization_config, run_compressed=True)
        elif not self.quantization_config.is_quantization_compressed:
            apply_quantization_config(model, ct_quantization_config)

    def _process_model_after_weight_loading(self, model: Any, **kwargs):
        """Decompress loaded model if necessary - need for qat"""
        # NOTE: This function originally modified the model in-place, which is not
        # a JAX-native pattern. The `compressed_tensors` library would need a
        # JAX-compatible functional API to work correctly.

        if (
            self.quantization_config.is_quantization_compressed and not self.run_compressed
        ) or self.quantization_config.is_sparsification_compressed:
            config = kwargs.get("config")
            cache_path = config._name_or_path

            if not os.path.exists(cache_path):
                from transformers.utils import cached_file

                config_file_path = cached_file(cache_path, "config.json")
                cache_path = os.path.sep.join(config_file_path.split(os.path.sep)[:-1])

            if self.quantization_config.is_quantization_compressed and not self.run_compressed:
                from compressed_tensors.quantization import QuantizationStatus

                self.compressor.quantization_config.quantization_status = QuantizationStatus.FROZEN
            self.compressor.decompress(model_path=cache_path, model=model)

    def update_tp_plan(self, config: Any) -> Any:
        additional_plan = {
            "layers.*.feed_forward.experts.*.gate_proj.weight": "local_colwise",
            "layers.*.feed_forward.experts.*.gate_proj.weight_scale": "local_colwise",
            "layers.*.feed_forward.experts.*.up_proj.weight": "local_colwise",
            "layers.*.feed_forward.experts.*.up_proj.weight_scale": "local_colwise",
            "layers.*.feed_forward.experts.*.down_proj.weight": "local_rowwise",
        }
        if config.get_text_config() is not None and config.get_text_config().base_model_tp_plan is not None:
            config.get_text_config().base_model_tp_plan.update(additional_plan)

        return config

    @property
    def is_trainable(self) -> bool:
        return True

    def is_qat_trainable(self) -> bool:
        """Loaded Models can carry out quantization aware training"""
        # models need to be decompressed carry out qat
        return not self.run_compressed or not self.quantization_config.is_quantization_compressed

    def is_serializable(self, safe_serialization: Any = None) -> bool:
        """Models quantized using compressed tensors can be saved to disk"""
        return True

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING, Any, Optional, Dict

import jax
import jax.numpy as jnp
import numpy as np

from .base import HfQuantizer
from ..utils import is_accelerate_available, is_eetq_available, logging
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

# Import torch for eetq interop
try:
    import torch
except ImportError:
    pass


logger = logging.get_logger(__name__)


class EetqHfQuantizer(HfQuantizer):
    """
    8-bit quantization from EETQ quantization method:
        before loading: converts transformer layers into W8A16Linear during loading: load 16bit weight and pass to the
        layer object after: quantizes individual weights in Linear8bitLt into 8bit at first .cuda() call
    """

    requires_parameters_quantization = True
    requires_calibration = False

    required_packages = ["eetq", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_eetq_available():
            raise ImportError(
                "Using `eetq` 8-bit quantization requires eetq."
                "Please install the latest version of eetq from : https://github.com/NetEase-FuXi/EETQ"
            )

        try:
            import eetq  # noqa: F401
        except ImportError as exc:
            if "shard_checkpoint" in str(exc):
                # EETQ 1.0.0 is currently broken with the latest transformers because it tries to import the removed
                # shard_checkpoint function, see https://github.com/NetEase-FuXi/EETQ/issues/34.
                # TODO: Update message once eetq releases a fix
                raise ImportError(
                    "You are using a version of EETQ that is incompatible with the current transformers version. "
                    "Either downgrade transformers to <= v4.46.3 or, if available, upgrade EETQ to > v1.0.0."
                ) from exc
            else:
                raise

        if not is_accelerate_available():
            raise ImportError("Loading an EETQ quantized model requires accelerate (`pip install accelerate`)")

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting into 8-bit weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        if not jax.device_count("gpu") > 0:
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded an EETQ model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model."
            )
        elif device_map is not None:
            if isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
                raise ValueError(
                    "You are attempting to load an EETQ model with a device_map that contains a CPU or disk device."
                    " This is not supported. Please remove the CPU or disk device from the device_map."
                )

    def update_jax_dtype(self, jax_dtype: "jnp.dtype") -> "jnp.dtype":
        if jax_dtype is None:
            jax_dtype = jnp.float16
            logger.info(
                "Overriding jax_dtype=%s with `jax_dtype=jnp.float16` due to "
                "requirements of `eetq` to enable model loading in 8-bit. "
                "Pass your own jax_dtype to specify the dtype of the remaining non-linear layers or pass"
                " jax_dtype=jnp.float16 to remove this warning.",
                jax_dtype,
            )
        elif jax_dtype != jnp.float16:
            logger.info("We suggest you to set `jax_dtype=jnp.float16` for better efficiency with EETQ.")
        return jax_dtype

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "jax.Array",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        from eetq import EetqLinear

        module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, EetqLinear):
            if self.pre_quantized or tensor_name == "bias":
                if tensor_name == "weight" and param_value.dtype != jnp.int8:
                    raise ValueError("Expect quantized weights but got an unquantized weight")
                return False
            else:
                if tensor_name == "weight_scale":
                    raise ValueError("Expect unquantized weights but got a quantized weight_scale")
                return True
        return False

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "jax.Array",
        param_name: str,
        target_device: "jax.Device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[list[str]] = None,
    ):
        """
        quantizes weights into qweight and weight_scales
        """
        from eetq import quantize_and_preprocess_weights

        module, tensor_name = get_module_from_name(model, param_name)

        # Eetq operates on torch tensors, so we need to convert back and forth.
        param_value_pt = torch.from_numpy(np.array(param_value))
        new_value_pt, weight_scale_pt = quantize_and_preprocess_weights(param_value_pt)

        new_value = jnp.array(new_value_pt.numpy())
        weight_scale = jnp.array(weight_scale_pt.numpy())

        # In JAX/Flax/NNX, we cannot modify modules in-place like in PyTorch.
        # This translation assumes the `module` object is a mutable container (like an NNX module)
        # that allows attribute assignment to update its state during model loading.
        # The distinction between buffers and parameters is handled by making `weight_scales`
        # a parameter and the quantized `weight` a simple attribute (state).
        # Device placement is handled by the JAX runtime, so `.to(target_device)` is omitted.
        setattr(module, tensor_name, new_value)
        setattr(module, "weight_scales", weight_scale)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        keep_in_fp32_modules: Optional[list[str]] = None,
        **kwargs,
    ):
        from ..integrations import replace_with_eetq_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        model = replace_with_eetq_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )

        model.config.quantization_config = self.quantization_config

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return True

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.linen import Module

# MaxText imports
from maxtext.common_types import Array, Config, DType

# The following imports are placeholders for HuggingFace-specific utilities
# and modules that would need a JAX equivalent in a real application.
# In a real MaxText integration, these would be replaced by MaxText's own
# model definitions and utility functions for model traversal and modification.


class HfQuantizer:
  """Placeholder for the HuggingFace HfQuantizer base class."""

  def __init__(self, quantization_config, **kwargs):
    self.pre_quantized = kwargs.get("pre_quantized", True)
    self.quantization_config = quantization_config

  def get_modules_to_not_convert(self, *args, **kwargs) -> List[str]:
    return []


class PreTrainedModel(Module):
  """Placeholder for the HuggingFace PreTrainedModel base class."""

  config: Config
  _tp_plan: Optional[Dict] = None


class FbgemmFp8Linear(Module):
  """Placeholder for a JAX equivalent of FbgemmFp8Linear."""

  pass


class FbgemmFp8Llama4TextExperts(Module):
  """Placeholder for a JAX equivalent of FbgemmFp8Llama4TextExperts."""

  pass


def get_module_from_name(model: Module, name: str) -> Tuple[Module, str]:
  """
  Placeholder for a utility to retrieve a submodule from a model by its name.
  """
  parts = name.split(".")
  module = model
  for part in parts[:-1]:
    if hasattr(module, part):
      module = getattr(module, part)
    else:
      # This is a simplified traversal. A real implementation would need to handle
      # lists of layers, etc., properly.
      return model, name  # Fallback
  return module, parts[-1]


def replace_with_fbgemm_fp8_linear(model: PreTrainedModel, **kwargs) -> PreTrainedModel:
  """
  Placeholder for a function that replaces standard linear layers with their
  FP8 equivalents. In JAX, this is typically handled by conditional logic
  within the model definition itself, rather than by in-place replacement.
  """
  return model


class _Logger:
  """Placeholder for a logging utility."""

  def warning_once(self, *args, **kwargs):
    print("WARNING:", *args)

  def info(self, *args, **kwargs):
    print("INFO:", *args)


logger = _Logger()


def _quantize_fp8_per_row(tensor: Array) -> Tuple[Array, Array]:
  """
  Quantizes a tensor to FP8 per row.
  This is a JAX implementation of the logic similar to fbgemm.quantize_fp8_per_row.
  """
  if tensor.ndim < 2:
    raise ValueError("Input tensor must have at least 2 dimensions for per-row quantization.")

  # The max value for e4m3fn is 448.0
  max_fp8_val = jnp.finfo(jnp.float8_e4m3fn).max

  # Calculate scales per row
  max_abs = jnp.max(jnp.abs(tensor), axis=-1, keepdims=True)
  scales = max_abs / max_fp8_val

  # To avoid division by zero for rows of zeros
  scales = jnp.where(scales == 0, 1.0, scales)

  # Quantize by dividing by the scale
  quantized_tensor = (tensor / scales).astype(jnp.float8_e4m3fn)

  return quantized_tensor, scales


class FbgemmFp8HfQuantizer(HfQuantizer):
  """
  FP8 quantization using fbgemm kernels.
  """

  requires_parameters_quantization = True
  requires_calibration = False

  required_packages = ["fbgemm-gpu", "accelerate"]

  def __init__(self, quantization_config, **kwargs):
    super().__init__(quantization_config, **kwargs)
    self.quantization_config = quantization_config

  def validate_environment(self, *args, **kwargs):
    # This method is highly specific to the PyTorch/CUDA environment.
    # In JAX, device availability and capabilities are handled differently.
    # A direct translation is not meaningful. For example, a JAX check might be:
    # if jax.default_backend() not in ['gpu', 'tpu']:
    #     raise RuntimeError("FP8 quantization requires a GPU or TPU backend.")
    pass

  def update_jax_dtype(self, jax_dtype: Optional[DType]) -> DType:
    if jax_dtype is None:
      jax_dtype = jnp.bfloat16
      logger.info(
          "Overriding jax_dtype=%s with `jnp.bfloat16` due to "
          "requirements to enable model loading in fp8. "
          "Pass your own jax_dtype to specify the dtype of the remaining non-linear layers or pass"
          " jax_dtype=jnp.bfloat16 to remove this warning.",
          jax_dtype,
      )
    elif jax_dtype == jnp.float16:
      raise ValueError("You cannot use FP8 with jax_dtype=jnp.float16. We recommend you passing jax_dtype=jnp.bfloat16")
    return jax_dtype

  def check_quantized_param(
      self,
      model: PreTrainedModel,
      param_value: Array,
      param_name: str,
      state_dict: Dict[str, Any],
      **kwargs,
  ) -> bool:
    module, tensor_name = get_module_from_name(model, param_name)

    if isinstance(module, FbgemmFp8Linear):
      if self.pre_quantized or tensor_name == "bias":
        if tensor_name == "weight" and param_value.dtype != jnp.float8_e4m3fn:
          raise ValueError("Expect quantized weights but got an unquantized weight")
        return False
      else:
        if tensor_name == "weight_scale":
          raise ValueError("Expect unquantized weights but got a quantized weight_scale")
        return True
    if isinstance(module, FbgemmFp8Llama4TextExperts):
      if self.pre_quantized or tensor_name == "bias":
        return False
      else:
        if tensor_name == "gate_up_proj_scale" or tensor_name == "down_proj_scale":
          raise ValueError("Expect unquantized weights but got a quantized weight_scale")
        return True
    return False

  def create_quantized_param(
      self,
      model: PreTrainedModel,
      param_value: Array,
      param_name: str,
      target_device: Optional[jax.Device] = None,
      state_dict: Optional[Dict[str, Any]] = None,
      unexpected_keys: Optional[List[str]] = None,
  ) -> Dict[str, Array]:
    """
    Quantizes weights into weight and weight_scale.
    In JAX, this method cannot modify the model in-place. Instead, it returns a
    dictionary of the new quantized parameters that should be updated in the model's state.
    """
    module, tensor_name = get_module_from_name(model, param_name)
    new_params = {}

    if isinstance(module, FbgemmFp8Llama4TextExperts):
      if tensor_name == "gate_up_proj":
        # Process each expert separately
        # Transpose the second and third dimension
        transposed_param = jnp.transpose(param_value, (0, 2, 1))

        # Reshape to 2D for quantization
        original_shape = transposed_param.shape
        flattened_param = transposed_param.reshape(-1, original_shape[-1])

        # Quantize using per row
        new_value_flat, weight_scale_flat = _quantize_fp8_per_row(flattened_param)

        # Reshape back to original dimensions
        new_value = new_value_flat.reshape(original_shape)
        new_value = jnp.transpose(new_value, (0, 2, 1))
        weight_scale = weight_scale_flat.reshape(original_shape[0], 1, original_shape[1])
      elif tensor_name == "down_proj":
        # Process each expert separately
        # Transpose the weights for proper quantization
        transposed_param = jnp.transpose(param_value, (0, 2, 1))

        # Reshape to 2D for quantization
        original_shape = transposed_param.shape
        flattened_param = transposed_param.reshape(-1, original_shape[-1])

        # Quantize using per row
        new_value_flat, weight_scale_flat = _quantize_fp8_per_row(flattened_param)

        # Reshape back to original dimensions
        new_value = new_value_flat.reshape(original_shape)
        new_value = jnp.transpose(new_value, (0, 2, 1))
        weight_scale = weight_scale_flat.reshape(original_shape[0], original_shape[1], 1)

      new_params[f"{tensor_name}_scale"] = weight_scale
      new_params[tensor_name] = new_value
    else:
      new_value, weight_scale = _quantize_fp8_per_row(param_value)
      new_params[f"{tensor_name}_scale"] = weight_scale.reshape(weight_scale.shape[0], 1)
      new_params[tensor_name] = new_value

    # The caller is responsible for updating the model's parameters with the returned dictionary.
    # The target_device argument is not used as JAX handles device placement via sharding/jitting.
    # The unexpected_keys logic is part of the HF loading loop and is handled by the caller.
    return new_params

  def _process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs) -> PreTrainedModel:
    return model

  def _process_model_before_weight_loading(
      self,
      model: PreTrainedModel,
      keep_in_fp32_modules: Optional[List[str]] = None,
      **kwargs,
  ) -> PreTrainedModel:
    # In JAX/Flax, model architecture is static. Layers are not replaced at runtime.
    # This functionality is achieved by defining the model to conditionally use
    # quantized layers based on the config. This method is a hook for that process.
    tp_plan = model._tp_plan
    self.modules_to_not_convert = self.get_modules_to_not_convert(
        model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
    )

    config = model.config
    model = replace_with_fbgemm_fp8_linear(
        model,
        modules_to_not_convert=self.modules_to_not_convert,
        quantization_config=self.quantization_config,
        pre_quantized=self.pre_quantized,
        config=config,
        tp_plan=tp_plan,
    )

    model.config.quantization_config = self.quantization_config
    return model

  def update_missing_keys(self, model: PreTrainedModel, missing_keys: List[str], prefix: str) -> List[str]:
    # This logic is highly dependent on the model's structure and how it's traversed.
    # A JAX equivalent would require a way to iterate through named submodules,
    # which is not a standard Flax pattern. This is a conceptual translation.
    not_missing_keys = []
    # for name, module in model.named_modules():  # Placeholder for JAX model traversal
    #     if isinstance(module, (FbgemmFp8Linear, FbgemmFp8Llama4TextExperts)):
    #         for missing in missing_keys:
    #             if (
    #                 (name in missing or name in f"{prefix}.{missing}")
    #                 and not missing.endswith(".weight")
    #                 and not missing.endswith(".bias")
    #             ):
    #                 not_missing_keys.append(missing)
    return [k for k in missing_keys if k not in not_missing_keys]

  def update_tp_plan(self, config: Config) -> Config:
    # This logic is about updating sharding configurations, which is relevant in JAX.
    # The specific plan is for a "Llama4" model.
    if "Llama4" in config.model_name:
      text_plan = {
          # We are using a different tp plan with local_colwise and local_rowwise for the attention because fbgemm operations cannot be parallelized
          # With local_colwise and local_rowwise, all the operations are done locally, and we add a gather operation to gather the results instead of
          # using dtensors
          "layers.*.self_attn.q_proj.weight": "local_colwise",
          "layers.*.self_attn.q_proj.weight_scale": "local_colwise",
          "layers.*.self_attn.k_proj.weight": "local_colwise",
          "layers.*.self_attn.k_proj.weight_scale": "local_colwise",
          "layers.*.self_attn.v_proj.weight": "local_colwise",
          "layers.*.self_attn.v_proj.weight_scale": "local_colwise",
          "layers.*.self_attn.o_proj.weight": "local_rowwise",
          "layers.*.self_attn": "gather",
          # We keep the same sequence_parallel plan for layernorms
          "layers.*.input_layernorm.weight": "sequence_parallel",
          "layers.*.post_attention_layernorm.weight": "sequence_parallel",
          "norm.weight": "sequence_parallel",
          # We keep the same local_colwise and local_rowwise plan for the feed forward shared expert
          # We also add scales for the shared expert, for local_colwise the scale is also local_colwise
          # For local_rowwise the scale is replicated, so we don't need to add it
          "layers.*.feed_forward.shared_expert.gate_proj.weight": "local_colwise",
          "layers.*.feed_forward.shared_expert.gate_proj.weight_scale": "local_colwise",
          "layers.*.feed_forward.shared_expert.up_proj.weight": "local_colwise",
          "layers.*.feed_forward.shared_expert.up_proj.weight_scale": "local_colwise",
          "layers.*.feed_forward.shared_expert.down_proj.weight": "local_rowwise",
          "layers.*.feed_forward.experts": "local",
          "layers.*.feed_forward": "gather",
          "layers.*.feed_forward.experts.*.gate_proj.weight": "local_colwise",
          "layers.*.feed_forward.experts.*.gate_proj.weight_scale": "local_colwise",
          "layers.*.feed_forward.experts.*.up_proj.weight": "local_colwise",
          "layers.*.feed_forward.experts.*.up_proj.weight_scale": "local_colwise",
          "layers.*.feed_forward.experts.*.down_proj.weight": "local_rowwise",
          # For Fused implementation we use local_packed_rowwise for the gate_up_proj, and the same for the packed scales
          # We use local_colwise for the down_proj, and the scales are replicated so we don't add them
          "layers.*.feed_forward.experts.gate_up_proj": "local_packed_rowwise",
          "layers.*.feed_forward.experts.gate_up_proj_scale": "local_packed_rowwise",
          "layers.*.feed_forward.experts.down_proj": "local_colwise",
      }
      # In MaxText, this would modify the logical_axis_rules or a similar config attribute.
      # Assuming a similar structure for demonstration.
      if hasattr(config, "get_text_config") and config.get_text_config() is not None:
        config.get_text_config().base_model_tp_plan = text_plan
      else:
        config.base_model_tp_plan = text_plan
      return config

    return config

  def is_serializable(self, safe_serialization: Any = None) -> bool:
    return True

  @property
  def is_trainable(self) -> bool:
    return False

# Copyright 2024 Google LLC
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

"""A JAX/Flax implementation of the Hugging Face FineGrainedFP8HfQuantizer."""

from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp

# MaxText modules used.
# from .. import max_logging (Path: src/MaxText/max_logging.py)
# from .base import HfQuantizer (Path: generated_code/maxtext/quantization/base.py)
# from .quantizers_utils import get_module_from_name (Path: generated_code/maxtext/quantization/quantizers_utils.py)
# from ..layers.models import PreTrainedModel (Path: generated_code/maxtext/layers/models.py)
from maxtext import max_logging
from maxtext.layers.models import PreTrainedModel
from maxtext.layers.quantizations.base import HfQuantizer
from maxtext.layers.quantizations.quantizers_utils import get_module_from_name

logger = max_logging.get_logger(__name__)


class FineGrainedFP8HfQuantizer(HfQuantizer):
  """FP8 quantization implementation supporting both standard and MoE models.

  Supports e4m3fn format.
  This is a JAX/Flax adaptation of the Hugging Face FineGrainedFP8HfQuantizer.
  The core quantization logic is in `create_quantized_param`. Other methods
  are provided for API compatibility with the HF quantization workflow but may
  not be idiomatic in a pure JAX/Flax environment.
  """

  requires_parameters_quantization = True
  requires_calibration = False

  def __init__(self, quantization_config: Any, **kwargs):
    super().__init__(quantization_config, **kwargs)
    self.quantization_config = quantization_config

  def validate_environment(self, *args, **kwargs):
    """Validates the environment for FP8 quantization.

    Removes PyTorch/Accelerate specific checks.
    """
    try:
      _ = jnp.float8_e4m3fn
    except AttributeError:
      raise ImportError("Using fp8 quantization requires a JAX version that supports jnp.float8_e4m3fn.")

    devices = jax.devices()
    if not any(d.platform in ["gpu", "tpu"] for d in devices):
      raise RuntimeError("No GPU or TPU found. A GPU or TPU is needed for FP8 quantization.")

    if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
      raise ValueError(
          "Converting into FP8 weights from tf/flax weights is currently not supported, "
          "please make sure the weights are in PyTorch format."
      )

    # The device_map logic from the original PyTorch implementation is specific to the `accelerate`
    # library and does not have a direct equivalent in JAX's sharding model.
    # Environment validation related to device_map is omitted.

  def update_jax_dtype(self, jax_dtype: Optional[jnp.dtype]) -> jnp.dtype:
    """Updates the jax_dtype of the model if it is not set."""
    if jax_dtype is None:
      logger.info("Setting jax_dtype to jnp.float32 as no jax_dtype was specified in from_pretrained")
      jax_dtype = jnp.float32
    return jax_dtype

  def create_quantized_param(
      self,
      model: PreTrainedModel,
      param_value: jnp.ndarray,
      param_name: str,
      target_device: jax.Device,
      state_dict: Dict[str, Any],
      unexpected_keys: Optional[List[str]] = None,
  ):
    """Quantizes weights to FP8 format using Block-wise quantization.

    This method is adapted for a JAX workflow. Instead of modifying the model
    in-place, it returns a dictionary of new parameters. The calling function
    is responsible for updating the model's parameter tree.
    """
    # In JAX, device placement is handled by the caller (e.g., via pjit).
    # We assume param_value is already on the correct device.

    # Get FP8 min/max values
    fp8_min = jnp.finfo(jnp.float8_e4m3fn).min
    fp8_max = jnp.finfo(jnp.float8_e4m3fn).max

    block_size_m, block_size_n = self.quantization_config.weight_block_size

    rows, cols = param_value.shape[-2:]

    if rows % block_size_m != 0 or cols % block_size_n != 0:
      raise ValueError(
          f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_size_m}, {block_size_n})"
      )
    param_value_orig_shape = param_value.shape

    # Reshape and transpose for block-wise processing
    reshaped_param = param_value.reshape(
        -1, rows // block_size_m, block_size_m, cols // block_size_n, block_size_n
    )
    permuted_param = jnp.transpose(reshaped_param, (0, 1, 3, 2, 4))

    # Calculate scaling factor for each block
    max_abs = jnp.max(jnp.abs(permuted_param), axis=(-1, -2))
    scale = fp8_max / max_abs
    scale_orig_shape = scale.shape
    scale_expanded = scale[..., None, None]

    # Quantize the weights
    quantized_param_permuted = jnp.clip(
        permuted_param * scale_expanded, a_min=fp8_min, a_max=fp8_max
    ).astype(jnp.float8_e4m3fn)

    # Transpose back and reshape to original matrix shape
    quantized_param_transposed = jnp.transpose(quantized_param_permuted, (0, 1, 3, 2, 4))
    quantized_param = quantized_param_transposed.reshape(param_value_orig_shape)

    # Reshape scale to match the number of blocks and compute reciprocal
    scale_inv = 1.0 / scale.reshape(scale_orig_shape).squeeze()

    # In JAX, we don't load parameters into the model in-place.
    # Instead, we would update the state_dict that will be used to initialize the model's state.
    state_dict[param_name] = quantized_param
    state_dict[param_name.rsplit(".", 1)[0] + ".weight_scale_inv"] = scale_inv

  def check_quantized_param(
      self,
      model: PreTrainedModel,
      param_value: jnp.ndarray,
      param_name: str,
      state_dict: Dict[str, Any],
      **kwargs,
  ) -> bool:
    """Checks if a parameter from a state dictionary needs to be quantized.

    This method's utility depends on a JAX equivalent of FP8Linear and get_module_from_name.
    """
    # This import is a placeholder for the JAX equivalent of FP8Linear
    from ..integrations.finegrained_fp8 import FP8Linear

    module, tensor_name = get_module_from_name(model, param_name)

    if isinstance(module, FP8Linear):
      if self.pre_quantized or tensor_name == "bias":
        if tensor_name == "weight" and param_value.dtype != jnp.float8_e4m3fn:
          raise ValueError("Expect quantized weights but got an unquantized weight")
        return False
      else:
        if tensor_name == "weight_scale_inv":
          raise ValueError("Expect unquantized weights but got a quantized weight_scale")
        return True
    return False

  def _process_model_before_weight_loading(
      self,
      model: PreTrainedModel,
      keep_in_fp32_modules: Optional[List[str]] = None,
      **kwargs,
  ):
    """Processes the model before loading weights.

    In JAX/Flax, model architecture is typically defined declaratively.
    This kind of in-place replacement is not idiomatic.
    This logic would live inside the model's `setup()` method, where it would
    conditionally instantiate an FP8Linear layer based on the config.
    """
    # This import is a placeholder for the JAX equivalent of this utility
    from ..integrations.finegrained_fp8 import replace_with_fp8_linear

    self.modules_to_not_convert = self.get_modules_to_not_convert(
        model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
    )

    model = replace_with_fp8_linear(
        model,
        modules_to_not_convert=self.modules_to_not_convert,
        quantization_config=self.quantization_config,
    )

    model.config.quantization_config = self.quantization_config

  def _process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs) -> PreTrainedModel:
    return model

  def update_missing_keys(self, model: PreTrainedModel, missing_keys: List[str], prefix: str) -> List[str]:
    """Updates the list of missing keys.

    This is highly specific to the HF loading mechanism.
    A JAX/Flax loading pipeline might have a different way of handling this.
    """
    # This import is a placeholder for the JAX equivalent of FP8Linear
    from ..integrations import FP8Linear

    not_missing_keys = []
    # `model.named_modules()` is a PyTorch concept. A JAX/Flax equivalent would be
    # traversing the parameter PyTree or using `module.named_children()`.
    # This translation is conceptual.
    for name, module in model.named_modules():
      if isinstance(module, FP8Linear):
        for missing in missing_keys:
          if (
              (name in missing or name in f"{prefix}.{missing}")
              and not missing.endswith(".weight")
              and not missing.endswith(".bias")
          ):
            not_missing_keys.append(missing)
    return [k for k in missing_keys if k not in not_missing_keys]

  def update_tp_plan(self, config: Any) -> Any:
    """Updates the tensor parallelism plan for a given model configuration."""
    if "Qwen3" in config.__class__.__name__:
      text_plan = {
          "layers.*.self_attn.q_proj.weight": "local_colwise",
          "layers.*.self_attn.q_proj.weight_scale_inv": "local_colwise",
          "layers.*.self_attn.k_proj.weight": "local_colwise",
          "layers.*.self_attn.k_proj.weight_scale_inv": "local_colwise",
          "layers.*.self_attn.v_proj.weight": "local_colwise",
          "layers.*.self_attn.v_proj.weight_scale_inv": "local_colwise",
          "layers.*.self_attn.o_proj.weight": "local_rowwise",
          "layers.*.self_attn.o_proj.weight_scale_inv": "local_rowwise",
          "layers.*.self_attn": "gather",
          "layers.*.mlp.gate_proj.weight": "local_colwise",
          "layers.*.mlp.gate_proj.weight_scale_inv": "local_colwise",
          "layers.*.mlp.up_proj.weight": "local_colwise",
          "layers.*.mlp.up_proj.weight_scale_inv": "local_colwise",
          "layers.*.mlp.down_proj.weight": "local_rowwise",
          "layers.*.mlp.down_proj.weight_scale_inv": "local_rowwise",
          "layers.*.mlp": "gather",
      }

      config.base_model_tp_plan = text_plan

    return config

  def is_serializable(self, safe_serialization: Optional[bool] = None) -> bool:
    return True

  @property
  def is_trainable(self) -> bool:
    return False

  def get_cuda_warm_up_factor(self) -> int:
    # Pre-processing is done cleanly, so we can allocate everything here
    return 2

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import jax
import jax.numpy as jnp

from ..utils import is_fp_quant_available, is_qutlass_available, logging
from ..utils.quantization_config import QuantizationConfigMixin
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)


class FPQuantHfQuantizer(HfQuantizer):
    """
    Quantizer for the FP-Quant method. Enables the loading of prequantized models and in-flight quantization of full-precision models.
    """

    requires_calibration = False
    requires_parameters_quantization = True
    is_qat_trainable = False
    required_packages = ["fp_quant"]

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, device_map: Optional[dict[str, Any]], **kwargs):
        if not jax.devices("gpu"):
            raise NotImplementedError("FPQuant quantization is only supported on GPU. Please use a different quantizer.")

        if not is_qutlass_available() and not self.quantization_config.pseudoquantization:
            raise ImportError(
                "Using `fp_quant` with real quantization requires a **Blackwell GPU** and qutlass: `git clone https://github.com/IST-DASLab/qutlass.git && cd qutlass && pip install --no-build-isolation .`. You can use `FPQuantConfig(pseudoquantization=True, ...)` to use Triton-based pseudo-quantization. It doesn't provide any speedups but emulates the quantization behavior of the real quantization."
            )

        if self.quantization_config.pseudoquantization:
            logger.warning(
                "Using pseudo-quantization for FP-Quant. This doesn't provide any speedups but emulates the quantization behavior of the real quantization."
            )

        if not is_fp_quant_available():
            raise ImportError("Using `fp_quant` quantization requires fp_quant: `pip install fp_quant`")

        if device_map is None:
            raise ValueError(
                "You are attempting to load a FPQuant model without setting device_map."
                " Please set device_map comprised of 'cuda' devices."
            )
        elif isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
            raise ValueError(
                "You are attempting to load a FPQuant model with a device_map that contains a CPU or disk device."
                " This is not supported. Please remove the CPU or disk device from the device_map."
            )

    def update_jax_dtype(self, jax_dtype: jnp.dtype) -> jnp.dtype:
        if jax_dtype is None:
            logger.info("`jax_dtype` is None. Setting `jax_dtype=jnp.bfloat16` for qutlass compatibility.")
            jax_dtype = jnp.bfloat16
        elif jax_dtype != jnp.bfloat16:
            raise ValueError(
                f"Invalid `jax_dtype` {jax_dtype}. fp_quant quantization only supports `jax_dtype=jnp.bfloat16`."
            )

        return jax_dtype

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: jnp.ndarray,
        param_name: str,
        target_device: jax.Device,
        state_dict: dict[str, Any],
        unexpected_keys: Optional[list[str]] = None,
    ):
        module, _ = get_module_from_name(model, param_name)

        # The module holds either:
        #  * `weight` when `store_master_weights=True`
        #  * `qweight` and `scales` when `store_master_weights=False` and `pseudoquantization=False`
        #  * `dqweight` when `store_master_weights=False` and `pseudoquantization=True`

        if param_name.endswith(".qweight"):
            # Loading a real quantized checkpoint without master weights
            module.qweight = param_value
            module.weight = None
            module.dqweight = None
            return

        if param_name.endswith(".dqweight"):
            # Loading a pseudo-quantized checkpoint without master weights
            module.dqweight = param_value
            module.weight = None
            module.qweight = None
            module.scales = None
            return

        # Loading master weights or an unquantized checkpoint
        module.weight = param_value
        # Let pre-forward handle the quantization and set None where necessary
        module.pre_forward()

        if unexpected_keys is not None and param_name in unexpected_keys:
            unexpected_keys.remove(param_name)

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        from fp_quant import replace_with_fp_quant_linear

        from ..integrations.fp_quant import adapt_fp_quant_config

        replace_with_fp_quant_linear(
            model,
            fp_quant_linear_config=adapt_fp_quant_config(self.quantization_config),
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def update_missing_keys(self, model: "PreTrainedModel", missing_keys: list[str], prefix: str) -> list[str]:
        from fp_quant import FPQuantLinear

        fp_quant_names = {name for name, module in model.named_modules() if isinstance(module, FPQuantLinear)}

        def should_exclude(key: str) -> bool:
            if key.endswith(".weight") or key.endswith(".bias"):
                return False
            full_key = f"{prefix}.{key}"
            return any(name in key or name in full_key for name in fp_quant_names)

        return [key for key in missing_keys if not should_exclude(key)]

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None) -> bool:
        return False

    def is_serializable(self, safe_serialization: Optional[bool] = None) -> bool:
        return True

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: jnp.ndarray,
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        from fp_quant import FPQuantLinear

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, FPQuantLinear) and tensor_name in ["weight", "qweight", "dqweight"]:
            # Only quantize weights of FPQuantLinear modules that are not already quantized
            return True
        else:
            return False

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" GptqHfQuantizer """
import importlib
from typing import TYPE_CHECKING, Any, Dict

from absl import logging
from packaging import version

import jax
import jax.numpy as jnp

from .base import HfQuantizer
from ..utils import is_auto_gptq_available, is_gptqmodel_available, is_optimum_available
from ..utils.quantization_config import GPTQConfig, QuantizationConfigMixin


if TYPE_CHECKING:
    from flax.linen import Module as FlaxModule


class GptqHfQuantizer(HfQuantizer):
    """
    Quantizer of the GPTQ method - for GPTQ the quantizer support calibration of the model through
    `auto_gptq` or `gptqmodel` package. Quantization is done under the hood for users if they load a non-prequantized model.
    """

    requires_calibration = False
    required_packages = ["optimum", "auto_gptq", "gptqmodel"]
    optimum_quantizer = None

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if not is_optimum_available():
            raise ImportError("Loading a GPTQ quantized model requires optimum (`pip install optimum`)")
        from optimum.gptq import GPTQQuantizer

        self.optimum_quantizer = GPTQQuantizer.from_dict(self.quantization_config.to_dict_optimum())

    def validate_environment(self, *args, **kwargs):
        if not is_optimum_available():
            raise ImportError("Loading a GPTQ quantized model requires optimum (`pip install optimum`)")
        if is_auto_gptq_available() and is_gptqmodel_available():
            logging.warning("Detected gptqmodel and auto-gptq, will use gptqmodel")

        gptq_supports_cpu = (
            is_auto_gptq_available()
            and version.parse(importlib.metadata.version("auto-gptq")) > version.parse("0.4.2")
        ) or is_gptqmodel_available()
        if not gptq_supports_cpu and not jax.devices("gpu"):
            raise RuntimeError("GPU is required to quantize or run quantize model.")
        elif not (is_auto_gptq_available() or is_gptqmodel_available()):
            raise ImportError(
                "Loading a GPTQ quantized model requires gptqmodel (`pip install gptqmodel`) or auto-gptq (`pip install auto-gptq`) library. "
            )
        elif is_auto_gptq_available() and version.parse(importlib.metadata.version("auto_gptq")) < version.parse(
            "0.4.2"
        ):
            raise ImportError(
                "You need a version of auto_gptq >= 0.4.2 to use GPTQ: `pip install --upgrade auto-gptq` or use gptqmodel by `pip install gptqmodel>=1.4.3`."
            )
        elif is_gptqmodel_available() and (
            version.parse(importlib.metadata.version("gptqmodel")) < version.parse("1.4.3")
            or version.parse(importlib.metadata.version("optimum")) < version.parse("1.23.99")
        ):
            raise ImportError("The gptqmodel version should be >= 1.4.3, optimum version should >= 1.24.0")

    def update_jax_dtype(self, jax_dtype: jnp.dtype) -> jnp.dtype:
        if jax_dtype is None:
            jax_dtype = jnp.float16
            logging.info("Loading the model in `jnp.float16`. To overwrite it, set `jax_dtype` manually.")
        elif jax_dtype != jnp.float16:
            logging.info("We suggest you to set `jax_dtype=jnp.float16` for better efficiency with GPTQ.")
        return jax_dtype

    def update_device_map(self, device_map: Dict[str, Any]) -> Dict[str, Any]:
        if device_map is None:
            device_map = {"": "cpu"}
        # Only with auto-gptq do not support CPU, we should move the model to cuda if available.
        if not is_gptqmodel_available() and device_map in ("cpu", {"": "cpu"}):
            device_map = {"": 0}
        return device_map

    def _process_model_before_weight_loading(self, model: "FlaxModule", **kwargs):
        if model.__class__.main_input_name != "input_ids":
            raise RuntimeError("We can only quantize pure text model.")

        if self.pre_quantized:
            # compat: latest optimum has gptqmodel refactor
            if version.parse(importlib.metadata.version("optimum")) <= version.parse("1.23.99"):
                model = self.optimum_quantizer.convert_model(model)
            else:
                model = self.optimum_quantizer.convert_model(model, **kwargs)

    def _process_model_after_weight_loading(self, model: "FlaxModule", **kwargs):
        if self.pre_quantized:
            model = self.optimum_quantizer.post_init_model(model)
        else:
            if self.quantization_config.tokenizer is None:
                self.quantization_config.tokenizer = model.name_or_path

            self.optimum_quantizer.quantize_model(model, self.quantization_config.tokenizer)
            model.config.quantization_config = GPTQConfig.from_dict(self.optimum_quantizer.to_dict())

    @property
    def is_trainable(self) -> bool:
        return True

    def is_serializable(self, safe_serialization=None):
        return True

from typing import Any, Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp
from absl import logging
from tqdm import tqdm

from ..utils.quantization_config import QuantizationConfigMixin
# HfQuantizer is a JAX-compatible abstract base class.
# src.MaxText.layers.quantizations.HfQuantizer was used as a reference.
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_accelerate_available, is_flute_available, is_hadamard_available


if TYPE_CHECKING:
  from ..modeling_flax_utils import FlaxPreTrainedModel


class HiggsHfQuantizer(HfQuantizer):
  """
    Quantizer of the HIGGS method. Enables the loading of prequantized models and in-flight quantization of full-precision models.
    """

  requires_calibration = False
  requires_parameters_quantization = True
  required_packages = ["flute-kernel", "fast_hadamard_transform"]

  def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
    super().__init__(quantization_config, **kwargs)
    self.quantization_config = quantization_config

  def validate_environment(self, device_map, **kwargs):
    if not jax.devices("gpu"):
      raise NotImplementedError("HIGGS quantization is only supported on GPU. Please use a different quantizer.")

    if not is_accelerate_available():
      raise ImportError("Using `higgs` quantization requires Accelerate: `pip install accelerate`")

    if not is_flute_available():
      raise ImportError("Using `higgs` quantization requires FLUTE: `pip install flute-kernel>=0.3.0`")

    if not is_hadamard_available():
      raise ImportError(
          "Using `higgs` quantization requires fast_hadamard_transform: `pip install fast_hadamard_transform`"
      )

    if device_map is None:
      raise ValueError(
          "You are attempting to load a HIGGS model without setting device_map."
          " Please set device_map comprised of 'cuda' devices."
      )
    elif isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
      raise ValueError(
          "You are attempting to load a HIGGS model with a device_map that contains a CPU or disk device."
          " This is not supported. Please remove the CPU or disk device from the device_map."
      )

  def update_jax_dtype(self, jax_dtype: jnp.dtype) -> jnp.dtype:
    if jax_dtype is None:
      logging.info("`jax_dtype` is None. Setting `jax_dtype=jnp.float16` for FLUTE compatibility.")
      jax_dtype = jnp.float16
    elif jax_dtype != jnp.float16 and jax_dtype != jnp.bfloat16:
      raise ValueError(
          f"Invalid `jax_dtype` {jax_dtype}. HIGGS quantization only supports `jax_dtype=jnp.float16` or `jax_dtype=jnp.bfloat16`."
      )

    return jax_dtype

  def create_quantized_param(
      self,
      model: "FlaxPreTrainedModel",
      param_value: jax.Array,
      param_name: str,
      target_device: jax.Device,
      state_dict: dict[str, Any],
      unexpected_keys: Optional[list[str]] = None,
  ):
    from ..integrations import quantize_with_higgs

    """
        Quantizes weights into weight and weight_scale
        """
    # In JAX, device placement is handled differently. We assume the param_value is already on the correct device.
    flute_dict = quantize_with_higgs(
        param_value,
        self.quantization_config.bits,
        self.quantization_config.p,
        self.quantization_config.group_size,
        self.quantization_config.hadamard_size,
    )
    del param_value

    module, _ = get_module_from_name(model, param_name)
    module_name = ".".join(param_name.split(".")[:-1])

    # In JAX/Flax, we don't modify module._parameters or _buffers directly.
    # Instead, we modify the state_dict which will be used to initialize the model.
    # This assumes the calling function (e.g., from_pretrained) will handle the modified state_dict.
    for key, value in flute_dict.items():
      new_param_name = f"{module_name}.{key}"
      if key == "tune_metadata":
        # The PyTorch code sets an attribute on the module. In JAX, this kind of state
        # should be handled explicitly. We store it in the config.
        # We also assume a JAX-compatible `to_dict` method exists.
        self.quantization_config.tune_metadata[module_name] = value.to_dict()
        # We might also need to store it in a 'buffers' collection if the model needs it at runtime.
        state_dict[new_param_name] = value
      else:
        state_dict[new_param_name] = value

    # The original parameter is now replaced by the quantized components.
    if param_name in state_dict:
      del state_dict[param_name]

    if unexpected_keys is not None and param_name in unexpected_keys:
      unexpected_keys.remove(param_name)

  def _process_model_before_weight_loading(
      self,
      model: "FlaxPreTrainedModel",
      keep_in_fp32_modules: Optional[list[str]] = None,
      **kwargs,
  ):
    from ..integrations import replace_with_higgs_linear

    self.modules_to_not_convert = self.get_modules_to_not_convert(
        model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
    )

    # In JAX, model modification is not in-place. This function should return a new model definition.
    # The calling function is responsible for using the new model.
    model = replace_with_higgs_linear(
        model,
        quantization_config=self.quantization_config,
        modules_to_not_convert=self.modules_to_not_convert,
    )
    model.config.quantization_config = self.quantization_config
    return model

  def _process_model_after_weight_loading(self, model: "FlaxPreTrainedModel", **kwargs):
    from flute.tune import TuneMetaData, maybe_tune_and_repack
    from flute.utils import make_workspace_streamk

    from ..integrations import HiggsLinear

    # This method is highly imperative and modifies the model in-place in PyTorch.
    # A JAX-idiomatic approach would involve updating the parameter and state PyTrees.
    # We simulate the imperative API, assuming the `model` object is a mutable
    # container for its parameters and state, which is not standard in Flax/JAX.

    flute_workspaces = {}
    # In JAX, we traverse the model's structure to find modules.
    # This assumes a utility similar to `model.named_modules()` exists for Flax models.
    flute_modules = {name: module for name, module in model.named_modules() if isinstance(module, HiggsLinear)}
    for name, module in tqdm(flute_modules.items(), desc="Repacking HIGGS modules", leave=False):
      # Every HiggsLinear needs a "workspace": a buffer for the unpacking operation.
      # This buffer needs to be on the same device as the weights, but can be reused across modules otherwise.
      device = module.weight.device()
      if device not in flute_workspaces:
        flute_workspaces[device] = make_workspace_streamk(device=device)
      module.workspace = flute_workspaces[device]

      # FLUTE weights are packed in a way that is optimized for a specific number of SMs (GPU streaming multiprocessors).
      # If the model is loaded on a different device than the one it was saved on, we need to repack the weights.
      module.tune_metadata = TuneMetaData.from_dict(self.quantization_config.tune_metadata[name])

      # In JAX, arrays are immutable. `maybe_tune_and_repack` would need to return new arrays.
      # The assignment `module.weight.data = ...` is not possible. We re-assign the whole attribute.
      # This assumes the module's parameter attributes can be reassigned after initialization.
      new_weight, new_tune_metadata = maybe_tune_and_repack(
          weight=module.weight,
          scales=module.scales,
          metadata=module.tune_metadata,
      )
      module.weight = new_weight
      module.tune_metadata = new_tune_metadata

      self.quantization_config.tune_metadata[name] = module.tune_metadata.to_dict()

  def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
    from ..integrations import HiggsLinear

    higgs_names = {name for name, module in model.named_modules() if isinstance(module, HiggsLinear)}

    def should_update(key: str) -> bool:
      if key.endswith(".weight") or key.endswith(".bias"):
        return False
      full_key = f"{prefix}.{key}"
      return any(name in key or name in full_key for name in higgs_names)

    return [key for key in missing_keys if not should_update(key)]

  @property
  def is_trainable(self) -> bool:
    return False

  def is_serializable(self, safe_serialization=None):
    return True

  def check_quantized_param(
      self,
      model: "FlaxPreTrainedModel",
      param_value: jax.Array,
      param_name: str,
      state_dict: dict[str, Any],
      **kwargs,
  ) -> bool:
    from ..integrations import HiggsLinear

    module, tensor_name = get_module_from_name(model, param_name)
    if isinstance(module, HiggsLinear) and tensor_name == "weight" and param_value.dtype != jnp.int16:
      # Only quantize weights of HiggsLinear modules that are not already quantized
      return True
    else:
      return False

  def _dequantize(self, model):
    from ..integrations import dequantize_higgs

    model = dequantize_higgs(model)
    return model

from typing import Any, Dict, List, Optional, Type

import jax
import jax.numpy as jnp
from absl import logging
from flax.linen import Dense, Module

# Assuming JAX equivalent of HfQuantizer exists.
# from .base import HfQuantizer
# Assuming JAX equivalent of PreTrainedModel exists.
# from ..modeling_utils import PreTrainedModel
# Assuming JAX equivalent of quantizers_utils exists.
# from .quantizers_utils import get_module_from_name
# Assuming JAX equivalent of integrations exists.
# from ..integrations import prepare_for_hqq_linear
# Assuming JAX equivalent of is_hqq_available exists.
# from ..utils import is_hqq_available

# The following are placeholders for functionality that is not standard in JAX/Flax
# and would need a custom implementation based on the specific model structure.


class HfQuantizer:
  """Placeholder for the base HfQuantizer class."""

  def __init__(self, quantization_config, **kwargs):
    self.quantization_config = quantization_config
    self.pre_quantized = kwargs.get("pre_quantized", True)


class PreTrainedModel(Module):
  """Placeholder for the base PreTrainedModel class."""

  pass


def get_module_from_name(model: Module, name: str) -> tuple[Module, str]:
  """Placeholder for get_module_from_name."""
  parts = name.split(".")
  module = model
  for part in parts[:-1]:
    module = getattr(module, part)
  tensor_name = parts[-1]
  return module, tensor_name


def is_hqq_available():
  """Placeholder for is_hqq_available."""
  try:
    import hqq  # pylint: disable=g-import-not-at-top

    return True
  except ImportError:
    return False


def prepare_for_hqq_linear(model, quantization_config):
  """Placeholder for prepare_for_hqq_linear."""
  return model


def find_parent(model: Module, name: str) -> Module:
  """Finds the parent of a module in a Flax/NNX model."""
  module_tree = name.split(".")[:-1]
  parent = model
  for m in module_tree:
    parent = getattr(parent, m)
  return parent


def remove_hook_from_module(module, *args, **kwargs):
  """Placeholder for accelerate.hooks.remove_hook_from_module."""
  # In JAX, hooks are not a standard concept. This might be a no-op
  # or require a different approach depending on the JAX framework used.
  logging.warning("remove_hook_from_module is not a standard JAX operation and is treated as a no-op.")
  return module


logger = logging


class HqqHfQuantizer(HfQuantizer):
  """
  HQQ quantizer base HF class.
  flax.linen.Dense modules are first tagged with quant_config in _process_model_before_weight_loading().
  The actual quantization and offloading to the GPU is done in check_quantized_param().
  """

  use_keep_in_fp32_modules = False
  requires_parameters_quantization = True
  requires_calibration = False
  required_packages = ["hqq"]

  def __init__(self, quantization_config, **kwargs):
    super().__init__(quantization_config, **kwargs)
    self.jax_dtype = None
    self.using_multi_gpu = False

  def validate_environment(self, *args, **kwargs):
    if not (is_hqq_available()):
      raise ImportError(
          "A valid HQQ version (>=0.2.1) is not available. Please follow the instructions to install it:"
          " `https://github.com/mobiusml/hqq/`."
      )

    if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
      raise ValueError(
          "Converting weights from tf/flax weights is currently not supported, please make"
          " sure the weights are in PyTorch format."
      )

    if self.jax_dtype is None:
      if "torch_dtype" in kwargs:  # Keep torch_dtype for API compatibility with HF
        self.jax_dtype = kwargs["torch_dtype"]
      elif "jax_dtype" in kwargs:
        self.jax_dtype = kwargs["jax_dtype"]
      else:
        self.jax_dtype = jnp.float32
        logger.info("Setting jax_dtype to jnp.float32 as the default value since it was not specified.")

    device_map = kwargs.get("device_map")
    if isinstance(device_map, dict):
      if "cpu" in device_map.values() or "disk" in device_map.values():
        raise ValueError(
            "You are attempting to use an HQQ model with a device_map that contains a CPU or disk device."
            " This is not supported. Please remove the CPU or disk device from the device_map."
        )
      else:
        # In JAX, multi-device is handled by the mesh. This is an approximation.
        self.using_multi_gpu = len(set(device_map.values())) > 1

  def update_missing_keys(
      self, model: "PreTrainedModel", missing_keys: list[str], prefix: str, **kwargs
  ) -> list[str]:
    if self.pre_quantized:
      return [key for key in missing_keys if ("kernel" not in key)]
    else:
      return missing_keys

  # Adds missing keys for HQQLinear modules that are loaded but the model with initialized with nn.Dense
  def update_expected_keys(
      self, model: "PreTrainedModel", expected_keys: list[str], loaded_keys: list[str]
  ) -> list[str]:
    if not self.pre_quantized:
      return expected_keys

    # This traversal is tricky in Flax. Assuming an nnx-like or custom traversal.
    # This placeholder will need a real implementation based on the model structure.
    def _find_hqq_quantizable_layers(module: Module, layers: set, prefix: str = ""):
      """Recursively finds Dense layers and adds their path to the layers set."""
      # This is a simplified traversal for nnx-like models where submodules are attributes.
      # It might not work for complex structures like ModuleList.
      for name, child in vars(module).items():
        if isinstance(child, Module):
          full_name = f"{prefix}.{name}" if prefix else name
          if isinstance(child, Dense):
            layers.add(full_name)
          _find_hqq_quantizable_layers(child, layers, prefix=full_name)

    new_keys = set(expected_keys)
    if is_hqq_available():
      from hqq.core.quantize import HQQLinear  # Assuming JAX version

      # valid modules are Dense layers that have HQQLinear state_dict. We ignore skip_modules and any layers with Dense state_dict() params
      _valid_modules = set()
      _find_hqq_quantizable_layers(model, _valid_modules)

      # Remove skipped modules
      _skipped_modules = set()
      for _module in _valid_modules:
        for _skip_module in model.config.quantization_config["skip_modules"]:
          if _skip_module in _module:
            _skipped_modules.add(_module)
      _valid_modules -= _skipped_modules

      # Append new expected layers based on _ref_keys
      # Assuming the JAX HQQLinear has a similar method to get param names.
      _ref_keys = HQQLinear(
          linear_layer=None,
          quant_config=None,
          compute_dtype=jnp.float16,
          del_orig=False,
      ).state_dict_keys() - {"bias"}

      # Clean-up
      _rm_keys = set()
      for key in new_keys:
        if any(_module in key for _module in _valid_modules):
          _rm_keys.add(key)
      new_keys -= _rm_keys
      # At this point, new_keys contains all the keys of the layers that are NOT HQQLinear or nn.Dense

      # Re-populate Dense/HQQLinear
      for _module in _valid_modules:
        if _module + ".kernel" in loaded_keys:
          new_keys.add(_module + ".kernel")
        else:
          new_keys.update({_module + "." + _ref_key for _ref_key in _ref_keys})
        if _module + ".bias" in loaded_keys:
          new_keys.add(_module + ".bias")

    return list(new_keys)

  def check_quantized_param(
      self,
      model: "PreTrainedModel",
      param_value: jax.Array,
      param_name: str,
      state_dict: Dict[str, Any],
      **kwargs,
  ) -> bool:
    if is_hqq_available():
      from hqq.core.quantize import HQQLinear  # Assuming JAX version
    module, tensor_name = get_module_from_name(model, param_name)

    if self.pre_quantized:
      return (isinstance(module, (Dense, HQQLinear))) and tensor_name != "kernel"
    else:
      return (
          isinstance(module, Dense)
          and tensor_name == "kernel"
          # bias doesn't need to be quantized, we use this as a workaround to avoid loading bias into HQQLinear assuming it was loaded
          # in the state_dict directly with the weight because hqq overwrote load_state_dict for this layer
          or (isinstance(module, HQQLinear) and tensor_name == "bias")
      )

  def create_quantized_param(
      self,
      model: "PreTrainedModel",
      param_value: jax.Array,
      param_name: str,
      target_device: Any,
      state_dict: Dict[str, Any],
      unexpected_keys: List[str],
  ):
    """
    Each nn.Dense layer is processed here.
    We first check if the corresponding module state_dict contains already HQQ quantized parameters.
    If not, we create a temp linear layer with the module state_dict params and use it for quantization
    """

    if is_hqq_available():
      from hqq.core.quantize import HQQLinear  # Assuming JAX version

      # TODO: This is a compatibility hack. HQQ-quantized linear layers do not have a `kernel` attribute,
      # but some models attempt to access `kernel.dtype` during the forward pass. To prevent runtime errors,
      # we patch HQQLinear with a dummy `kernel` property that returns an empty tensor with the correct dtype.
      @property
      def kernel(_self: HQQLinear):
        return jnp.empty(0, dtype=_self.compute_dtype)

      HQQLinear.kernel = kernel

    module, tensor_name = get_module_from_name(model, param_name)
    layer_name = ".".join(param_name.split(".")[:-1])
    parent_module = find_parent(model, layer_name)
    node = layer_name.split(".")[-1]

    if tensor_name == "bias":
      # this should already be set
      return

    # set module state_dict
    module_state_dict = {}
    for k, v in state_dict.items():
      if layer_name + "." in k:
        module_state_dict[k.split(".")[-1]] = v
        if unexpected_keys is not None and k in unexpected_keys:
          unexpected_keys.remove(k)

    if self.pre_quantized:
      if isinstance(module, HQQLinear):
        return
      else:
        hqq_layer = HQQLinear(
            linear_layer=None,
            quant_config=None,
            compute_dtype=self.jax_dtype,
            del_orig=False,
        )

      # In JAX, we would update the parameters of the module.
      # Assuming an nnx-like model where we can set attributes.
      for key, value in module_state_dict.items():
        setattr(hqq_layer, key, value)

      if self.using_multi_gpu:
        hqq_layer = self._patch_layer_for_multigpu(hqq_layer)

      setattr(parent_module, node, hqq_layer)
      return

    # Step 1: populate module with weight/bias from module state dict
    for key, tensor in module_state_dict.items():
      setattr(module, key, tensor)

    # Step 2: Replace module with either HQQLinear or move it to device. We do this via setattr on the parent as doing on it on the module
    # directly doesn't work.
    quant_config = model.config.quantization_config["quant_config"]
    skip_modules = model.config.quantization_config["skip_modules"]
    module_name = layer_name
    module_tag = ".".join(module_name.split(".")[-2:])
    module_quant_config = None
    if "weight_quant_params" in quant_config:
      module_quant_config = quant_config
    elif module_tag in quant_config:
      module_quant_config = quant_config[module_tag]

    for skip_module in skip_modules:
      if skip_module in module_name:
        module_quant_config = None
        break

    if module_quant_config is not None:
      hqq_layer = HQQLinear(
          module,
          quant_config=module_quant_config,
          compute_dtype=self.jax_dtype,
          del_orig=True,
      )

      if self.using_multi_gpu:
        hqq_layer = self._patch_layer_for_multigpu(hqq_layer)

      setattr(parent_module, node, hqq_layer)

    else:
      # In JAX, this would involve casting the parameters, not the module.
      if hasattr(module, "kernel") and module.kernel is not None:
        module.kernel = module.kernel.astype(self.jax_dtype)
      if hasattr(module, "bias") and module.bias is not None:
        module.bias = module.bias.astype(self.jax_dtype)
      setattr(parent_module, node, module)

  # Remove accelerate hook and uses a simpler forward pass. Otherwise, this breaks with multi-gpu
  def _patch_layer_for_multigpu(self, hqq_layer):
    hqq_layer = remove_hook_from_module(hqq_layer)

    def forward_with_device(self, x):
      out = jnp.matmul(x, self.dequantize().T)
      if self.bias is not None:
        out += self.bias
      return out

    hqq_layer.__call__ = lambda x: forward_with_device(hqq_layer, x)
    return hqq_layer

  def _process_model_before_weight_loading(
      self,
      model: "PreTrainedModel",
      **kwargs,
  ):
    # Add the corresponding quant_config to each valid module. This allows us to do the actual nn.Dense -> HQQLinear conversion in create_quantized_param().
    # prepare_for_hqq_linear() also sets the right quantization config inside the model (model.config.quantization_config) and the layers (hqq_layer.quant_config)
    model = prepare_for_hqq_linear(model, quantization_config=self.quantization_config)

  def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
    model.is_hqq_quantized = True
    model.is_hqq_serializable = self.is_serializable()
    return model

  def is_serializable(self, safe_serialization=None):
    return True

  @property
  def is_trainable(self) -> bool:
    return True

from typing import Any, Optional

import jax
import jax.numpy as jnp

from ..integrations import (  # MaxText-matched dependency: src.MaxText.layers.models.gemma3
    Mxfp4GptOssExperts,
    dequantize,
    load_and_swizzle_mxfp4,
    quantize_to_mxfp4,
    replace_with_mxfp4_linear,
)
from ..models.gpt_oss.modeling_gpt_oss import (  # MaxText-matched dependency: src.MaxText.layers.gpt_oss
    GptOssExperts,
)
from ..modeling_utils import PreTrainedModel
from ..utils import (
    is_accelerate_available,
    is_kernels_available,
    is_torch_available,
    is_triton_available,
    logging,
)
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


logger = logging.get_logger(__name__)


class Mxfp4HfQuantizer(HfQuantizer):
    """
    FP4 quantization using fbgemm kernels
    """

    requires_parameters_quantization = True
    # to remove if we decide to allow quantizing weights with this method
    requires_calibration = False

    required_packages = ["accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_torch_available():
            raise ImportError(
                "Using mxfp4 quantization requires torch"
                "Please install the latest version of torch ( pip install --upgrade torch )"
            )

        if self.quantization_config.dequantize:
            return

        if not len(jax.devices("gpu")) > 0:
            if self.pre_quantized:
                logger.warning_once(
                    "Using MXFP4 quantized models requires a GPU, we will default to dequantizing the model to bf16"
                )
                self.quantization_config.dequantize = True
                return
            else:
                raise RuntimeError("Quantizing a model using MXFP4 requires a GPU")

        if not is_accelerate_available():
            raise ImportError("Using mxfp4 requires Accelerate: `pip install accelerate`")

        # This is a placeholder for a JAX-compatible way to get compute capability.
        # A library like `pynvml` could be used here if needed.
        # For now, we assume a compatible GPU is present if jax.devices("gpu") finds one.
        # compute_capability = torch.cuda.get_device_capability()
        # gpu_is_supported = compute_capability >= (7, 5)
        gpu_is_supported = True
        kernels_available = is_triton_available("3.4.0") and is_kernels_available()

        if self.pre_quantized:
            # On unsupported GPUs or without kernels, we will dequantize the model to bf16
            if not gpu_is_supported:
                logger.warning_once(
                    "MXFP4 quantization is only supported on GPUs with compute capability >= 7.5 (e.g T4, A100, L4, H100, or B200). "
                    "We will default to dequantizing the model to bf16."
                )
                self.quantization_config.dequantize = True
                return

            if not kernels_available:
                logger.warning_once(
                    "MXFP4 quantization requires triton >= 3.4.0 and kernels installed, we will default to dequantizing the model to bf16"
                )
                self.quantization_config.dequantize = True
                return
        elif not gpu_is_supported:
            # we can't quantize the model in this case so we raise an error
            raise ValueError(
                "MXFP4 quantization is only supported on GPUs with compute capability >= 7.5 (e.g T4, A100, L4, H100, or B200)"
            )
        elif not kernels_available:
            # we can't quantize the model in this case so we raise an error
            raise ValueError("MXFP4 quantization requires triton >= 3.4.0 and triton_kernels installed")

        if not self.pre_quantized:
            from kernels import get_kernel

            global triton_kernels_hub
            triton_kernels_hub = get_kernel("kernels-community/triton_kernels")

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded an FP4 model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model. To remove this warning, pass device_map = 'cuda'. "
            )
        elif device_map is not None:
            if (
                not self.pre_quantized
                and isinstance(device_map, dict)
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                raise ValueError(
                    "You are attempting to load an FP4 model with a device_map that contains a CPU or disk device."
                    "This is not supported when the model is quantized on the fly. "
                    "Please use a quantized checkpoint or remove the CPU or disk device from the device_map."
                )

    def update_jax_dtype(self, jax_dtype: jnp.dtype) -> jnp.dtype:
        if jax_dtype is None:
            jax_dtype = jnp.bfloat16
            logger.info(
                "Overriding jax_dtype=%s with `jax_dtype=jnp.bfloat16` due to "
                "requirements of `fbgemm-gpu` to enable model loading in fp4. "
                "Pass your own jax_dtype to specify the dtype of the remaining non-linear layers or pass"
                " jax_dtype=jnp.bfloat16 to remove this warning.",
                jax_dtype,
            )
        return jax_dtype

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: jax.Array,
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ):
        # if we are dequantizing, the model doesn't have scales, and blocks only params like gate_up_proj and down_proj so we need to handle this case differently
        if self.quantization_config.dequantize and ("blocks" in param_name or "scales" in param_name):
            module, tensor_name = get_module_from_name(model, param_name[: -len("_blocks")])
        else:
            module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, Mxfp4GptOssExperts) or (
            isinstance(module, GptOssExperts) and self.quantization_config.dequantize
        ):
            if tensor_name in ["down_proj_bias", "gate_up_proj_bias"]:
                return False
            return True
        return False

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: jax.Array,
        param_name: str,
        target_device: jax.Device,
        state_dict: dict[str, Any],
        unexpected_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        if not self.pre_quantized:
            PrecisionConfig, FlexCtx, InFlexData = (
                triton_kernels_hub.matmul_ogs.PrecisionConfig,
                triton_kernels_hub.matmul_ogs.FlexCtx,
                triton_kernels_hub.matmul_ogs.InFlexData,
            )
            module, _ = get_module_from_name(model, param_name)
            # In JAX, device placement is handled differently. We assume the context is correct.
            if isinstance(module, Mxfp4GptOssExperts):
                if "gate_up_proj" in param_name:
                    right_pad = module.gate_up_proj_right_pad
                    bottom_pad = module.gate_up_proj_bottom_pad
                    # PyTorch pad: (pad_left, pad_right, pad_top, pad_bottom, ...) for last dims
                    # JAX pad: ((before_0, after_0), (before_1, after_1), ...)
                    pad_width = ((0, 0), (0, bottom_pad), (0, right_pad))
                    loaded_weight = jnp.pad(param_value, pad_width, mode="constant", constant_values=0)
                    triton_weight_tensor, weight_scale, blocks_data = quantize_to_mxfp4(loaded_weight)
                    module.gate_up_proj_precision_config = PrecisionConfig(
                        weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())
                    )
                    module.gate_up_proj = triton_weight_tensor
                    module.gate_up_proj_blocks = blocks_data
                elif "down_proj" in param_name:
                    right_pad = module.down_proj_right_pad
                    bottom_pad = module.down_proj_bottom_pad
                    pad_width = ((0, 0), (0, bottom_pad), (0, right_pad))
                    loaded_weight = jnp.pad(param_value, pad_width, mode="constant", constant_values=0)
                    loaded_weight = jax.device_put(loaded_weight, target_device)
                    triton_weight_tensor, weight_scale, blocks_data = quantize_to_mxfp4(loaded_weight)
                    module.down_proj_precision_config = PrecisionConfig(
                        weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())
                    )
                    module.down_proj = triton_weight_tensor
                    module.down_proj_blocks = blocks_data

        # we take this path if already quantized but not in a compatible way
        # The params going here are either gate_up_proj_blocks, or down_proj_blocks, or gate_up_proj_scales, or down_proj_scales
        else:
            empty_param = kwargs.get("empty_param")
            casting_dtype = kwargs.get("casting_dtype")
            to_contiguous = kwargs.get("to_contiguous")
            rank = kwargs.get("rank")
            device_mesh = kwargs.get("device_mesh")
            if ("blocks" in param_name or "scales" in param_name) and self.quantization_config.dequantize:
                # blocks and scales have the same length that's this works for both
                module, _ = get_module_from_name(model, param_name[: -len("_blocks")])
            else:
                module, _ = get_module_from_name(model, param_name)

            shard_kwargs = {
                "empty_param": empty_param,
                "casting_dtype": casting_dtype,
                "to_contiguous": to_contiguous,
                "rank": rank,
                "device_mesh": device_mesh,
                "model": model,
            }

            if isinstance(module, Mxfp4GptOssExperts) or (
                isinstance(module, GptOssExperts) and self.quantization_config.dequantize
            ):
                if self.quantization_config.dequantize:
                    # dq_param_name is the name of the parameter without the blocks or scales suffix, it's used in this case since we don't switch linears
                    # so we only have the original param name
                    dq_param_name = param_name[: -len("_blocks")]
                    dequantize(module, param_name, param_value, target_device, dq_param_name, **shard_kwargs)
                else:
                    load_and_swizzle_mxfp4(
                        module,
                        param_name,
                        param_value,
                        target_device,
                        **shard_kwargs,
                    )

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        # we are not really dequantizing, we are just removing everthing related to quantization here
        if self.quantization_config.dequantize:
            self.remove_quantization_config(model)
        # clean cache due to triton ops
        # JAX manages memory automatically, so empty_cache is not needed.
        pass

    def update_expected_keys(self, model: "PreTrainedModel", expected_keys: list[str], checkpoint_keys: list[str]):
        # Replace expected_keys for experts' gate_up_proj and down_proj with their _blocks and _scales variants
        new_expected_keys = []
        for key in expected_keys:
            if key.endswith(".mlp.experts.gate_up_proj"):
                base = key[: -len("gate_up_proj")]
                new_expected_keys.append(base + "gate_up_proj_blocks")
                new_expected_keys.append(base + "gate_up_proj_scales")
            elif key.endswith(".mlp.experts.down_proj"):
                base = key[: -len("down_proj")]
                new_expected_keys.append(base + "down_proj_blocks")
                new_expected_keys.append(base + "down_proj_scales")
            else:
                new_expected_keys.append(key)
        return new_expected_keys

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        keep_in_fp32_modules: Optional[list[str]] = None,
        **kwargs,
    ):
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        use_kernels = kwargs.get("use_kernels", False)
        # if we are using kernels, we can't use the quantized model, since the forward pass is different and needs special handling
        if use_kernels:
            logger.warning_once(
                "You are using full precision kernels, we will dequantize the model to bf16. "
                "To use the quantized model with quantization kernels, please set use_kernels=False"
            )
            self.quantization_config.dequantize = True

        config = model.config
        model = replace_with_mxfp4_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            config=config,
        )

        model.config.quantization_config = self.quantization_config

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
        not_missing_keys = []
        # In Flax, we iterate through submodules differently
        for name, module in model.iter_submodules():
            if isinstance(module, Mxfp4GptOssExperts):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")
                        and not missing.endswith(".bias")
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]

    def update_tp_plan(self, config):
        if "GptOssConfig" in config.__class__.__name__:
            if getattr(config, "base_model_tp_plan", None) is not None:
                config.base_model_tp_plan.update(
                    {
                        "layers.*.mlp.experts.gate_up_proj_blocks": "grouped_gemm",
                        "layers.*.mlp.experts.gate_up_proj_scales": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj_blocks": "grouped_gemm",
                        "layers.*.mlp.experts.down_proj_scales": "grouped_gemm",
                    }
                )
        return config

    def update_param_name(self, param_name: str) -> str:
        if self.quantization_config.dequantize:
            if "_blocks" in param_name:
                return param_name.replace("_blocks", "")
            elif "_scales" in param_name:
                return param_name.replace("_scales", "")
        return param_name

    def is_serializable(self, safe_serialization=None):
        logger.warning_once("MXFP4 quantization is not serializable using safetensors for now")
        return False

    @property
    def is_trainable(self) -> bool:
        logger.warning_once(
            "MXFP4 quantization don't support training, please consider dequantizing the model first by passing quantization_config=Mxfp4Config(dequantize=True) to .from_pretrained()"
        )
        return False

import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from absl import logging
from packaging import version

# Assuming HfQuantizer is the JAX equivalent from the base file
# from .base import HfQuantizer
# Re-used from generated_code.Qwen3MoeForCausalLM.quantization.HfQuantizer
from jetstream.engine.quantization import HfQuantizer
# Assuming get_module_from_name is the JAX equivalent
# from .quantizers_utils import get_module_from_name
# Re-used from generated_code.Qwen3MoeForCausalLM.dynamic_module_utils.get_module_from_name
from jetstream.engine.model_utils import get_module_from_name
import jax.numpy as jnp

if TYPE_CHECKING:
    # Assuming PreTrainedModel is the JAX equivalent base model class
    from ..modeling_flax_utils import FlaxPreTrainedModel

# Assuming these utility functions are available in the JAX environment
from ..utils import (
    is_accelerate_available,
    is_optimum_quanto_available,
)
from ..utils.quantization_config import QuantoConfig


logger = logging

# This is a placeholder for accelerate's CustomDtype to make the logic translatable.
# In a real JAX environment, the loader would need to interpret these.
class CustomDtype:
    """Placeholder for accelerate.utils.CustomDtype."""

    FP8 = "float8"
    INT4 = "int4"
    INT2 = "int2"


class QuantoHfQuantizer(HfQuantizer):
    """
    Quantizer for the quanto library
    """

    required_packages = ["quanto", "accelerate"]
    requires_parameters_quantization = True
    requires_calibration = False

    def __init__(self, quantization_config: QuantoConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.post_init()

    def post_init(self):
        r"""
        Safety checker
        """
        if self.quantization_config.activations is not None and not self.pre_quantized:
            raise ValueError(
                "We don't support quantizing the activations with transformers library."
                "Use quanto library for more complex use cases such as activations quantization, calibration and quantization aware training."
            )

    def validate_environment(self, *args, **kwargs):
        if not is_optimum_quanto_available():
            raise ImportError(
                "Loading an optimum-quanto quantized model requires optimum-quanto library (`pip install optimum-quanto`)"
            )
        if not is_accelerate_available():
            raise ImportError(
                "Loading an optimum-quanto quantized model requires accelerate library (`pip install accelerate`)"
            )

    def update_device_map(self, device_map: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if device_map is None:
            device_map = {"": "cpu"}
            logger.info(
                "The device_map was not initialized. "
                "Setting device_map to {'':'cpu'}. "
                "If you want to use the model for inference, please set device_map ='auto'"
            )
        return device_map

    def update_jax_dtype(self, jax_dtype: Optional[jnp.dtype]) -> jnp.dtype:
        if jax_dtype is None:
            logger.info("You did not specify `jax_dtype` in `from_pretrained`. Setting it to `jnp.float32`.")
            jax_dtype = jnp.float32
        return jax_dtype

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        # This method relies on iterating through modules, which is not a standard Flax pattern.
        # A utility function `named_modules` would be needed to traverse the Flax model structure.
        # Assuming such a utility exists for this loading pipeline.
        if is_optimum_quanto_available():
            from optimum.quanto import QModuleMixin

        # Placeholder for a function that can iterate through Flax modules similar to PyTorch's named_modules
        def named_modules(module, name=""):
            # This is a simplified implementation and might need to be more robust
            # to handle various Flax module structures (e.g., ModuleList).
            yield name, module
            for sub_name, sub_module in module.__dict__.items():
                if hasattr(sub_module, "children"):  # A way to identify submodules
                    yield from named_modules(sub_module, name=f"{name}.{sub_name}" if name else sub_name)

        not_missing_keys = []
        for name, module in named_modules(model):
            if isinstance(module, QModuleMixin):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")
                        and not missing.endswith(".bias")
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]

    def check_quantized_param(
        self,
        model: "FlaxPreTrainedModel",
        param_value: jnp.ndarray,
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        """
        Check if a parameter needs to be quantized.
        """
        if is_optimum_quanto_available():
            from optimum.quanto import QModuleMixin

        device_map = kwargs.get("device_map")
        param_device = kwargs.get("param_device")
        # we don't quantize the model if the module is going to be offloaded to the cpu
        if device_map is not None and param_device is not None:
            device_map_values = set(device_map.values())
            if param_device == "cpu" and len(device_map_values) > 1:
                if not (device_map_values == {"cpu"} or device_map_values == {"cpu", "disk"}):
                    return False

        module, tensor_name = get_module_from_name(model, param_name)
        # We only quantize the weights and the bias is not quantized.
        if isinstance(module, QModuleMixin) and "weight" in tensor_name:
            # if the weights are quantized, don't need to recreate it again with `create_quantized_param`
            return not module.frozen
        else:
            return False

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def create_quantized_param(
        self,
        model: "FlaxPreTrainedModel",
        param_value: jnp.ndarray,
        param_name: str,
        target_device: Any,  # JAX device object
        *args,
        **kwargs,
    ):
        """
        Create the quantized parameter by calling .freeze() after setting it to the module.
        This is not an idiomatic JAX pattern as it involves in-place model modification.
        """
        # JAX equivalent of accelerate.utils.set_module_tensor_to_device
        # This is a stateful operation, not idiomatic JAX, but required for this API.
        def set_module_tensor_to_device(model, tensor_name, _, value):
            module, name = get_module_from_name(model, tensor_name)
            setattr(module, name, value)

        set_module_tensor_to_device(model, param_name, target_device, param_value)
        module, _ = get_module_from_name(model, param_name)
        module.freeze()
        # In JAX, requires_grad is not an attribute of the tensor.
        # This line is omitted as JAX handles gradients via transformations (e.g., jax.grad).
        # module.weight.requires_grad = False

    def adjust_target_dtype(self, target_dtype: jnp.dtype) -> Union[jnp.dtype, str]:
        if version.parse(importlib.metadata.version("accelerate")) > version.parse("0.27.0"):
            mapping = {
                "int8": jnp.int8,
                "float8": CustomDtype.FP8,
                "int4": CustomDtype.INT4,
                "int2": CustomDtype.INT2,
            }
            target_dtype = mapping[self.quantization_config.weights]
            return target_dtype
        else:
            raise ValueError(
                "You are using `device_map='auto'` on an optimum-quanto quantized model. To automatically compute"
                " the appropriate device map, you should upgrade your `accelerate` library,"
                "`pip install --upgrade accelerate` or install it from source."
            )

    def _process_model_before_weight_loading(
        self, model: "FlaxPreTrainedModel", keep_in_fp32_modules: Optional[List[str]] = None, **kwargs
    ):
        # In JAX, model modification is not in-place. This function would return a new model.
        from ..integrations import replace_with_quanto_layers

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        model, _ = replace_with_quanto_layers(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        model.config.quantization_config = self.quantization_config
        return model

    def _process_model_after_weight_loading(self, model: "FlaxPreTrainedModel", **kwargs) -> "FlaxPreTrainedModel":
        return model

    @property
    def is_trainable(self) -> bool:
        return True

    def is_serializable(self, safe_serialization=None) -> bool:
        return False

# Copyright 2025 Advanced Micro Devices, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Quark HfQuantizer."""

from typing import Any, Dict

import jax
from absl import logging
from flax.core import freeze
from flax.traverse_util import flatten_dict, unflatten_dict

# Placeholder for HfQuantizer base class, assuming it exists in the target project
from .base import HfQuantizer
from ..utils import is_quark_available


# Helper function to mimic the imperative style of accelerate.utils.set_module_tensor_to_device
# This is not purely functional but directly translates the original logic for a weight loading script.
def _set_module_tensor_to_device(
    model_params: Dict[str, Any], tensor_name: str, device: Any, value: jax.Array
) -> Dict[str, Any]:
  """
  Sets a tensor value in a nested dictionary of parameters.
  'device' is ignored as placement is handled by sharding annotations in JAX.
  This function returns a new dictionary with the updated value.
  """
  _ = device  # Unused in JAX implementation
  flat_params = flatten_dict(model_params, sep=".")
  flat_params[tensor_name] = value
  return unflatten_dict(flat_params, sep=".")


# Mapped from transformers.models.auto.quantization.CHECKPOINT_KEYS
CHECKPOINT_KEYS = {
    "weight_scale": "weight_quantizer.scale",
    "bias_scale": "bias_quantizer.scale",
    "input_scale": "input_quantizer.scale",
    "output_scale": "output_quantizer.scale",
    "weight_zero_point": "weight_quantizer.zero_point",
    "bias_zero_point": "bias_quantizer.zero_point",
    "input_zero_point": "input_quantizer.zero_point",
    "output_zero_point": "output_quantizer.zero_point",
}


class QuarkHfQuantizer(HfQuantizer):
  """
  Quark quantizer (https://quark.docs.amd.com/latest/).
  """

  requires_calibration = True  # On-the-fly quantization with quark is not supported for now.
  required_packages = ["quark"]

  # Checkpoints are expected to be already quantized when loading a quark model. However, as some keys from
  # the checkpoint might mismatch the model parameters keys, we use the `create_quantized_param` method
  # to load the checkpoints, remapping the keys.
  requires_parameters_quantization = True

  def __init__(self, quantization_config, **kwargs):
    super().__init__(quantization_config, **kwargs)

    self.json_export_config = quantization_config.json_export_config

  def validate_environment(self, *args, **kwargs):
    if not is_quark_available():
      raise ImportError(
          "Loading a Quark quantized model requires the `quark` library but it was not found in the"
          " environment. Please refer to https://quark.docs.amd.com/latest/install.html."
      )

  def _process_model_before_weight_loading(self, model, **kwargs):
    # Assuming a JAX-compatible API exists in the quark library
    from quark.jax.export.api import _map_to_quark

    _map_to_quark(
        model,
        self.quantization_config.quant_config,
        pack_method=self.json_export_config.pack_method,
        custom_mode=self.quantization_config.custom_mode,
    )

    return model

  def check_quantized_param(
      self,
      model,
      param_value: jax.Array,
      param_name: str,
      state_dict: Dict[str, Any],
      **kwargs,
  ) -> bool:
    return True

  def create_quantized_param(
      self, model, param, param_name, param_device, state_dict, unexpected_keys
  ):
    postfix = param_name.split(".")[-1]

    if postfix in CHECKPOINT_KEYS:
      param_name = param_name.replace(postfix, CHECKPOINT_KEYS[postfix])

    # In JAX, we work with parameter dictionaries (PyTrees) which are immutable.
    # The equivalent of `set_module_tensor_to_device` is to return an updated
    # parameter dictionary. However, to maintain the original API structure
    # which implies side-effects, we assume the calling context will handle
    # the returned updated model parameters.
    # The original PyTorch function has no return, but its type hint suggests it should.
    # We will return the updated model parameters.
    return _set_module_tensor_to_device(model, param_name, param_device, value=param)

  def _process_model_after_weight_loading(self, model, **kwargs):
    return model

  def is_serializable(self, safe_serialization=None):
    return False

  @property
  def is_trainable(self):
    return False

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""SpQRHfQuantizer Module"""
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
from absl import logging

from transformers.utils.quantization_config import QuantizationConfigMixin

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_flax_utils import FlaxPreTrainedModel

from ..integrations import replace_with_spqr_linear
from ..utils import is_accelerate_available, is_spqr_available


logger = logging


class SpQRHfQuantizer(HfQuantizer):
    """
    Quantizer of the SpQR method. Enables the loading of prequantized models.
    """

    requires_calibration = True

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not jax.devices("gpu"):
            raise RuntimeError("GPU is required to run SpQR quantized model.")

        if not is_accelerate_available():
            raise ImportError("Using `spqr` quantization requires Accelerate: `pip install accelerate`")

        if not is_spqr_available():
            raise ImportError("Using `spqr` quantization requires SpQR: `pip install spqr_quant[gpu]`")

    def update_jax_dtype(self, jax_dtype: Optional[jnp.dtype]) -> jnp.dtype:
        if jax_dtype is None:
            jax_dtype = jnp.float16
            logger.info("Assuming SpQR inference on GPU and loading the model in `jnp.float16`.")
        elif jax_dtype != jnp.float16:
            raise ValueError(
                "You cannot use any type other than jnp.float16 for SpQR. Please either leave it None or set it to"
                " jnp.float16 explicitly."
            )
        return jax_dtype

    def _process_model_before_weight_loading(
        self,
        model: "FlaxPreTrainedModel",
        keep_in_fp32_modules: Optional[list[str]] = None,
        **kwargs,
    ):
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        model = replace_with_spqr_linear(
            model,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.modules_to_not_convert,
        )
        model.config.quantization_config = self.quantization_config
        return model

    def _process_model_after_weight_loading(self, model: "FlaxPreTrainedModel", **kwargs):
        return model

    @property
    def is_trainable(self):
        return False

    def is_serializable(self, safe_serialization=None):
        return True

import importlib
import re
import types
from typing import Any, Dict, List, Optional, Union

from absl import logging
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
from packaging import version

# MaxText dependencies
from MaxText.layers import linears
from MaxText.layers.quantizations import QuantizationConfigMixin

# Reused from generated_code.Qwen3MoeForCausalLM.dynamic_module_utils.get_module_from_name
from transformers.utils.quantization_base import HfQuantizer, get_module_from_name


# Placeholder for a JAX equivalent of torchao. This would need to be replaced
# with a real JAX quantization library like AQT.
def is_torchao_available():
  """Checks if a JAX equivalent for torchao is available."""
  # This is a placeholder. In a real scenario, this would check for a specific
  # JAX quantization library.
  logging.warning("Using a placeholder for `is_torchao_available`. A real JAX quantization library should be used.")
  return True


def _linear_extra_repr(self):
  # This is a placeholder for pretty-printing and is not functionally critical.
  return f"in_features={self.in_features}, out_features={self.out_features}, weight=Quantized"


def quantize_(module, config):
  # This function in PyTorch modifies the module in-place.
  # A JAX equivalent would need to return a new module or new parameters.
  # This is a conceptual placeholder.
  logging.warning("torchao.quantization.quantize_ is not implemented in JAX. This is a placeholder.")
  if isinstance(module, linears.DenseGeneral):
    # Simulate the side-effect for repr
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
  return module


def autoquant(
    model,
    qtensor_class_list=None,
    set_inductor_config=False,
    **kwargs,
):
  logging.warning("torchao.autoquant is not implemented in JAX. This is a placeholder.")
  return model


class AOBaseConfig:
  """Placeholder for torchao.core.config.AOBaseConfig."""

  pass


class ModuleFqnToConfig:
  """Placeholder for torchao.quantization.ModuleFqnToConfig."""

  def __init__(self, module_fqn_to_config):
    self.module_fqn_to_config = module_fqn_to_config


class CustomDtype:
  """Placeholder for accelerate.utils.CustomDtype."""

  INT4 = "int4"


# End of placeholders


def fuzzy_match_size(config_name: str) -> Optional[str]:
  """
    Extract the size digit from strings like "4weight", "8weight".
    Returns the digit as an integer if found, otherwise None.
    """
  config_name = config_name.lower()

  str_match = re.search(r"(\d)weight", config_name)

  if str_match:
    return str_match.group(1)

  return None


class TorchAoHfQuantizer(HfQuantizer):
  """
    Quantizer for a JAX equivalent of torchao.
    """

  requires_parameters_quantization = True
  requires_calibration = False
  required_packages = ["aqt-jax"]  # Assuming AQT or a similar library is the JAX target

  def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
    super().__init__(quantization_config, **kwargs)

  def validate_environment(self, *args, **kwargs):
    if not is_torchao_available():
      raise ImportError("Loading a torchao-style quantized model requires a JAX quantization library.")

    self.offload = False
    device_map = kwargs.get("device_map")
    if isinstance(device_map, dict):
      if ("disk" in device_map.values() or "cpu" in device_map.values()) and len(device_map) > 1:
        self.offload = True
        if self.pre_quantized and "disk" in device_map.values():
          raise ValueError(
              "You are attempting to perform disk offload with a pre-quantized model. "
              "This is not supported yet. Please remove the disk device from the device_map."
          )

  def update_jax_dtype(self, jax_dtype: jnp.dtype) -> jnp.dtype:
    if self.quantization_config.quant_type == "int4_weight_only":
      if jax_dtype is not None and jax_dtype != jnp.bfloat16:
        logging.warning(
            "Setting jax_dtype to %s for int4_weight_only quantization, but only bfloat16 is supported right now. "
            "Please set the jax_dtype to bfloat16.",
            jax_dtype,
        )
      if jax_dtype is None:
        logging.warning(
            "Setting jax_dtype to jnp.bfloat16 for int4_weight_only quantization since only bfloat16 is supported "
            "right now. Please set jax_dtype=jnp.bfloat16 to remove this warning."
        )
        jax_dtype = jnp.bfloat16
    if self.quantization_config.quant_type == "int8_dynamic_activation_int8_weight":
      if jax_dtype is None:
        logging.info(
            "Setting jax_dtype to jnp.float32 for int8_dynamic_activation_int8_weight quantization as no jax_dtype "
            "was specified in from_pretrained"
        )
        # we need to set the jax_dtype, otherwise we have dtype mismatch when performing the quantized linear op
        jax_dtype = jnp.float32
    return jax_dtype

  def adjust_target_dtype(self, jax_dtype: jnp.dtype) -> Union[jnp.dtype, str]:
    # This method is heavily tied to accelerate, we provide a conceptual JAX equivalent.
    # The version check is removed as it's not relevant for JAX.
    # Import AOBaseConfig directly since we know we have the right version
    if self.quantization_config._get_ao_version() > version.Version("0.9.0"):
      quant_type = self.quantization_config.quant_type
      if isinstance(quant_type, AOBaseConfig):
        # Extract size digit using fuzzy match on the class name
        config_name = quant_type.__class__.__name__
        size_digit = fuzzy_match_size(config_name)

        # Map the extracted digit to appropriate dtype
        if size_digit == "4":
          return CustomDtype.INT4  # Representing int4 as a string
        else:
          # Default to int8
          return jnp.int8

    # Original mapping for non-AOBaseConfig types
    map_to_target_dtype = {
        "int4_weight_only": CustomDtype.INT4,
        "int8_weight_only": jnp.int8,
        "int8_dynamic_activation_int8_weight": jnp.int8,
        "autoquant": None,
    }
    return map_to_target_dtype[self.quantization_config.quant_type]

  def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
    # need more space for the quantization parameters (e.g. scale). Tested with int4 wo and group size = 128
    max_memory = {key: val * 0.9 for key, val in max_memory.items()}
    return max_memory

  def _process_model_before_weight_loading(
      self, model, keep_in_fp32_modules: Optional[List[str]] = None, **kwargs
  ):
    self.modules_to_not_convert = self.get_modules_to_not_convert(
        model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
    )
    if self.quantization_config.include_input_output_embeddings:
      # Assuming the JAX model has a similar API to Hugging Face PyTorch models
      if hasattr(model, "get_input_embeddings") and hasattr(model, "get_output_embeddings"):
        input_emb = model.get_input_embeddings()
        input_emb_names = [name for name, module in model.named_modules() if id(module) == id(input_emb)]
        output_emb = model.get_output_embeddings()
        output_emb_names = [name for name, module in model.named_modules() if id(module) == id(output_emb)]
        self.modules_to_not_convert = [
            x for x in self.modules_to_not_convert if x not in input_emb_names + output_emb_names
        ]
    return

  def check_quantized_param(
      self,
      model,
      param_value: jax.Array,
      param_name: str,
      state_dict: Dict[str, Any],
      **kwargs,
  ) -> bool:
    if self.quantization_config.quant_type == "autoquant":
      return False

    param_device = kwargs.pop("param_device", None)
    # check if the param_name is not in self.modules_to_not_convert
    if any((key + "." in param_name) or (key == param_name) for key in self.modules_to_not_convert):
      return False
    elif param_device == "cpu" and self.offload:
      # We don't quantize weights that we offload
      return False
    else:
      # we only quantize the weight of nn.Dense and nn.Embed
      module, tensor_name = get_module_from_name(model, param_name)
      _QUANTIZABLE = [linears.DenseGeneral]
      # Assuming nn.Embed is the JAX equivalent for nn.Embedding
      # if self.quantization_config.include_input_output_embeddings:
      #     _QUANTIZABLE.append(nn.Embed)
      return isinstance(module, tuple(_QUANTIZABLE)) and (tensor_name == "kernel")

  def create_quantized_param(
      self,
      model,
      param_value: jax.Array,
      param_name: str,
      target_device: jax.Device,
      state_dict: Dict[str, Any],
      unexpected_keys: List[str],
  ):
    """
        This method is part of a PyTorch-native loading pipeline that relies on in-place model modification,
        which is not idiomatic in JAX. A JAX-native approach would define the model with quantized layers
        from the start and load quantized weights directly into a parameter PyTree. This implementation
        conceptually follows the original logic but relies on non-functional placeholders and assumptions
        about a mutable model object for API compatibility.
        """
    if self.quantization_config.quant_type == "autoquant":
      return

    module, tensor_name = get_module_from_name(model, param_name)
    if self.pre_quantized:
      # In JAX, parameters are not part of the module object itself.
      # This simulates setting a parameter in a stateful wrapper.
      # The parameter would be moved to the target device by JAX's sharding mechanism.
      setattr(module, tensor_name, param_value)
      if isinstance(module, linears.DenseGeneral):
        module.extra_repr = types.MethodType(_linear_extra_repr, module)
    else:
      assert isinstance(self.quantization_config, TorchAoConfig)
      # This simulates setting a parameter and then quantizing the module in-place.
      setattr(module, tensor_name, param_value)

      # The concept of tying weights and then quantizing requires careful handling in JAX's
      # functional paradigm, likely at the parameter initialization stage.
      if hasattr(model, "get_input_embeddings") and hasattr(model, "tie_weights"):
        input_embed = model.get_input_embeddings()
        if self.quantization_config.untie_embedding_weights and id(module) == id(input_embed):
          model.tie_weights()
          # This direct config modification is an anti-pattern in JAX.
          model.config.tie_word_embeddings = False

      # handle ModuleFqnToConfig, introduced in torchao 0.12.0+
      if self.quantization_config._get_ao_version() >= version.Version("0.12.0"):
        config = self.quantization_config.get_apply_tensor_subclass()
        if isinstance(config, ModuleFqnToConfig):
          module_fqn, _ = param_name.rsplit(".", 1)
          c = None
          if module_fqn in config.module_fqn_to_config:
            c = config.module_fqn_to_config[module_fqn]
          else:
            c = config.module_fqn_to_config.get("_default", None)
          if c is not None:
            # filter_fn: not filtering out any modules
            quantize_(module, c)  # Placeholder call
          return

      quantize_(module, self.quantization_config.get_apply_tensor_subclass())  # Placeholder call

  def _process_model_after_weight_loading(self, model, **kwargs):
    """No process required for torchao quantized model"""
    if self.quantization_config.quant_type == "autoquant":
      # `torch.compile` is analogous to `jax.jit`. The `autoquant` part is specific
      # to the torchao library and has no direct JAX equivalent.
      logging.warning(
          "torchao.autoquant is a PyTorch-specific feature and is not available in JAX. "
          "This is a conceptual placeholder."
      )
      model = jax.jit(model)
      model = autoquant(
          model,
          **self.quantization_config.quant_type_kwargs,
      )
      return model
    return

  def is_serializable(self, safe_serialization=None) -> bool:
    if safe_serialization:
      logging.warning(
          "torchao-style quantized models do not support safe serialization, please set `safe_serialization` to False"
      )
      return False
    _is_torchao_serializable = version.parse(importlib.metadata.version("huggingface_hub")) >= version.parse("0.25.0")
    if not _is_torchao_serializable:
      logging.warning("torchao-style quantized models are only serializable after huggingface_hub >= 0.25.0 ")
    if self.offload and self.quantization_config.modules_to_not_convert is None:
      logging.warning(
          "The model contains offloaded modules and these modules are not quantized. We don't recommend saving the "
          "model as we won't be able to reload them. If you want to specify modules to not quantize, please specify "
          "modules_to_not_convert in the quantization_config."
      )
      return False
    return _is_torchao_serializable

  def get_cuda_warm_up_factor(self):
    """
        This factor is used in caching_allocator_warmup to determine how many bytes to pre-allocate for CUDA warmup.
        - A factor of 2 means we pre-allocate the full memory footprint of the model.
        - A factor of 4 means we pre-allocate half of that, and so on

        However, when using a JAX equivalent of TorchAO, calculating memory usage with param.size * param.itemsize
        doesn't give the correct size for quantized weights (like int4 or int8).

        To correct for this:
        - Use a division factor of 8 for int4 weights
        - Use a division factor of 4 for int8 weights
        """
    if self.quantization_config._get_ao_version() > version.Version("0.9.0"):
      quant_type = self.quantization_config.quant_type
      # For autoquant case, it will be treated in the string implementation below in map_to_target_dtype
      if isinstance(quant_type, AOBaseConfig):
        # Extract size digit using fuzzy match on the class name
        config_name = quant_type.__class__.__name__
        size_digit = fuzzy_match_size(config_name)

        if size_digit == "4":
          return 8
        else:
          return 4

    # Original mapping for non-AOBaseConfig types
    map_to_target_dtype = {
        "int4_weight_only": 8,
        "int8_weight_only": 4,
        "int8_dynamic_activation_int8_weight": 4,
        "autoquant": 4,
    }

    return map_to_target_dtype[self.quantization_config.quant_type]

  @property
  def is_trainable(self) -> bool:
    supported_quant_types_for_training = [
        "int8_weight_only",
        "int8_dynamic_activation_int8_weight",
    ]
    return self.quantization_config.quant_type in supported_quant_types_for_training

  @property
  def is_compileable(self) -> bool:
    return True

from typing import TYPE_CHECKING, Optional, List

from absl import logging
import jax
import jax.numpy as jnp

from transformers.utils.quantization_config import QuantizationConfigMixin
from .base import HfQuantizer
from ..integrations.vptq import is_accelerate_available, is_vptq_available, replace_with_vptq_linear

if TYPE_CHECKING:
    from flax.linen import Module as FlaxModule


class VptqHfQuantizer(HfQuantizer):
    """
    Quantizer of the VPTQ method. Enables the loading of prequantized models.
    """

    requires_calibration = True
    required_packages = ["vptq"]

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Using `vptq` quantization requires Accelerate: `pip install accelerate`")

        if not is_vptq_available():
            raise ImportError("Using `vptq` quantization requires VPTQ>=0.0.4: `pip install -U vptq`")

    def update_jax_dtype(self, jax_dtype: jnp.dtype) -> jnp.dtype:
        if jax_dtype is None:
            if len(jax.devices("gpu")) > 0:
                jax_dtype = jnp.float16
                logging.info(
                    "CUDA available. Assuming VPTQ inference on GPU and loading the model in `jnp.float16`. To overwrite it, set `jax_dtype` manually."
                )
            else:
                import vptq

                device_availability = getattr(vptq, "device_availability", lambda device: False)
                if device_availability("cpu") is True:
                    raise RuntimeError("No GPU found. Please wait for the next release of VPTQ to use CPU inference")
                jax_dtype = jnp.float32
                logging.info("No GPU found. Assuming VPTQ inference on CPU and loading the model in `jnp.float32`.")
        return jax_dtype

    def _process_model_before_weight_loading(
        self,
        model: "FlaxModule",
        keep_in_fp32_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        we don't have param like modules_to_not_convert to indicate which layers should not be quantized
        because `quantization_config` include the layers that should be quantized
        """
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        replace_with_vptq_linear(
            model,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.modules_to_not_convert,
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "FlaxModule", **kwargs):
        return model

    @property
    def is_trainable(self) -> bool:
        return False

    def is_serializable(self, safe_serialization=None):
        return True
from .quantizer_aqlm import AqlmHfQuantizer
from .quantizer_auto_round import AutoRoundQuantizer
from .quantizer_awq import AwqQuantizer
from .quantizer_bitnet import BitNetHfQuantizer
from .quantizer_bnb_4bit import Bnb4BitHfQuantizer
from .quantizer_bnb_8bit import Bnb8BitHfQuantizer
from .quantizer_compressed_tensors import CompressedTensorsHfQuantizer
from .quantizer_eetq import EetqHfQuantizer
from .quantizer_fbgemm_fp8 import FbgemmFp8HfQuantizer
from .quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
from .quantizer_fp_quant import FPQuantHfQuantizer
from .quantizer_gptq import GptqHfQuantizer
from .quantizer_higgs import HiggsHfQuantizer
from .quantizer_hqq import HqqHfQuantizer
from .quantizer_mxfp4 import Mxfp4HfQuantizer
from .quantizer_quanto import QuantoHfQuantizer
from .quantizer_quark import QuarkHfQuantizer
from .quantizer_spqr import SpQRHfQuantizer
from .quantizer_torchao import TorchAoHfQuantizer
from .quantizer_vptq import VptqHfQuantizer


AUTO_QUANTIZER_MAPPING = {
    "awq": AwqQuantizer,
    "bitsandbytes_4bit": Bnb4BitHfQuantizer,
    "bitsandbytes_8bit": Bnb8BitHfQuantizer,
    "gptq": GptqHfQuantizer,
    "aqlm": AqlmHfQuantizer,
    "quanto": QuantoHfQuantizer,
    "quark": QuarkHfQuantizer,
    "fp_quant": FPQuantHfQuantizer,
    "eetq": EetqHfQuantizer,
    "higgs": HiggsHfQuantizer,
    "hqq": HqqHfQuantizer,
    "compressed-tensors": CompressedTensorsHfQuantizer,
    "fbgemm_fp8": FbgemmFp8HfQuantizer,
    "torchao": TorchAoHfQuantizer,
    "bitnet": BitNetHfQuantizer,
    "vptq": VptqHfQuantizer,
    "spqr": SpQRHfQuantizer,
    "fp8": FineGrainedFP8HfQuantizer,
    "auto-round": AutoRoundQuantizer,
    "mxfp4": Mxfp4HfQuantizer,
}
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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
""" Auto HfQuantizer class."""

import warnings
from typing import Optional, Union

from MaxText import max_logging as logging
from MaxText.layers.quantizations.configs import (
    AutoRoundConfig,
    AwqConfig,
    CompressedTensorsConfig,
    FbgemmFp8Config,
    GPTQConfig,
    Mxfp4Config,
    QuantizationConfigMixin,
)
from MaxText.layers.quantizations.quantization_utils import QuantizationMethod

from .auto_factory import (
    AUTO_QUANTIZATION_CONFIG_MAPPING,
    AUTO_QUANTIZER_MAPPING,
    AutoQuantizationConfig,
)

logger = logging.get_logger(__name__)


class AutoHfQuantizer:
    """
    The Auto-HF quantizer class that takes care of automatically instantiating to the correct
    `HfQuantizer` given the `QuantizationConfig`.
    """

    @classmethod
    def from_config(cls, quantization_config: Union[QuantizationConfigMixin, dict], **kwargs):
        # Convert it to a QuantizationConfig if the q_config is a dict
        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        quant_method = quantization_config.quant_method

        # Again, we need a special care for bnb as we have a single quantization config
        # class for both 4-bit and 8-bit quantization
        if quant_method == QuantizationMethod.BITS_AND_BYTES:
            if quantization_config.load_in_8bit:
                quant_method += "_8bit"
            else:
                quant_method += "_4bit"

        if quant_method not in AUTO_QUANTIZER_MAPPING:
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )

        target_cls = AUTO_QUANTIZER_MAPPING[quant_method]
        return target_cls(quantization_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        quantization_config = AutoQuantizationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_config(quantization_config)

    @classmethod
    def merge_quantization_configs(
        cls,
        quantization_config: Union[dict, QuantizationConfigMixin],
        quantization_config_from_args: Optional[QuantizationConfigMixin],
    ):
        """
        handles situations where both quantization_config from args and quantization_config from model config are present.
        """
        if quantization_config_from_args is not None:
            warning_msg = (
                "You passed `quantization_config` or equivalent parameters to `from_pretrained` but the model you're"
                " loading"
                " already has a `quantization_config` attribute. The `quantization_config` from the model will be used."
            )
        else:
            warning_msg = ""

        if isinstance(quantization_config, dict):
            # Convert the config based on the type of quantization_config_from_args (e.g., AutoRoundConfig), which takes priority before automatic configuration dispatch.
            if isinstance(quantization_config_from_args, AutoRoundConfig):
                quantization_config = AutoRoundConfig.from_dict(quantization_config)
            else:
                quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        if (
            quantization_config_from_args is not None
            and quantization_config.__class__.__name__ != quantization_config_from_args.__class__.__name__
        ):
            raise ValueError(
                f"The model is quantized with {quantization_config.__class__.__name__} but you are passing a"
                f" {quantization_config_from_args.__class__.__name__} config. "
                "Please make sure to pass the same quantization config class to `from_pretrained` with different"
                " loading attributes."
            )

        if (
            isinstance(
                quantization_config,
                (GPTQConfig, AwqConfig, AutoRoundConfig, FbgemmFp8Config, CompressedTensorsConfig, Mxfp4Config),
            )
            and quantization_config_from_args is not None
        ):
            # special case for GPTQ / AWQ / FbgemmFp8 config collision
            loading_attr_dict = quantization_config_from_args.get_loading_attributes()
            for attr, val in loading_attr_dict.items():
                setattr(quantization_config, attr, val)

            warning_msg += (
                f"However, loading attributes (e.g. {list(loading_attr_dict.keys())}) will be overwritten with the one"
                " you passed to `from_pretrained`. The rest will be ignored."
            )

        if warning_msg != "" and not isinstance(quantization_config, Mxfp4Config):
            warnings.warn(warning_msg)
        else:
            # in the case of mxfp4, we don't want to print the warning message, bit confusing for users
            logger.info(warning_msg)
        return quantization_config

    @staticmethod
    def supports_quant_method(quantization_config_dict):
        quant_method = quantization_config_dict.get("quant_method", None)
        if quantization_config_dict.get("load_in_8bit", False) or quantization_config_dict.get("load_in_4bit", False):
            suffix = "_4bit" if quantization_config_dict.get("load_in_4bit", False) else "_8bit"
            quant_method = QuantizationMethod.BITS_AND_BYTES + suffix
        elif quant_method is None:
            raise ValueError(
                "The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the"
                " model has been correctly quantized"
            )

        if quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING:
            logger.warning(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}. Hence, we will skip the quantization. "
                "To remove the warning, you can delete the quantization_config attribute in config.json"
            )
            return False
        return True

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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
import warnings
from typing import Optional, Union

from ..models.auto.configuration_auto import AutoConfig
from ..utils import logging
from ..utils.quantization_config import (
    AqlmConfig,
    AutoRoundConfig,
    AwqConfig,
    BitNetQuantConfig,
    BitsAndBytesConfig,
    CompressedTensorsConfig,
    EetqConfig,
    FbgemmFp8Config,
    FineGrainedFP8Config,
    FPQuantConfig,
    GPTQConfig,
    HiggsConfig,
    HqqConfig,
    Mxfp4Config,
    QuantizationConfigMixin,
    QuantizationMethod,
    QuantoConfig,
    QuarkConfig,
    SpQRConfig,
    TorchAoConfig,
    VptqConfig,
)
from .base import HfQuantizer
from .quantizer_aqlm import AqlmHfQuantizer
from .quantizer_auto_round import AutoRoundQuantizer
from .quantizer_awq import AwqQuantizer
from .quantizer_bitnet import BitNetHfQuantizer
from .quantizer_bnb_4bit import Bnb4BitHfQuantizer
from .quantizer_bnb_8bit import Bnb8BitHfQuantizer
from .quantizer_compressed_tensors import CompressedTensorsHfQuantizer
from .quantizer_eetq import EetqHfQuantizer
from .quantizer_fbgemm_fp8 import FbgemmFp8HfQuantizer
from .quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
from .quantizer_fp_quant import FPQuantHfQuantizer
from .quantizer_gptq import GptqHfQuantizer
from .quantizer_higgs import HiggsHfQuantizer
from .quantizer_hqq import HqqHfQuantizer
from .quantizer_mxfp4 import Mxfp4HfQuantizer
from .quantizer_quanto import QuantoHfQuantizer
from .quantizer_quark import QuarkHfQuantizer
from .quantizer_spqr import SpQRHfQuantizer
from .quantizer_torchao import TorchAoHfQuantizer
from .quantizer_vptq import VptqHfQuantizer


AUTO_QUANTIZER_MAPPING = {
    "awq": AwqQuantizer,
    "bitsandbytes_4bit": Bnb4BitHfQuantizer,
    "bitsandbytes_8bit": Bnb8BitHfQuantizer,
    "gptq": GptqHfQuantizer,
    "aqlm": AqlmHfQuantizer,
    "quanto": QuantoHfQuantizer,
    "quark": QuarkHfQuantizer,
    "fp_quant": FPQuantHfQuantizer,
    "eetq": EetqHfQuantizer,
    "higgs": HiggsHfQuantizer,
    "hqq": HqqHfQuantizer,
    "compressed-tensors": CompressedTensorsHfQuantizer,
    "fbgemm_fp8": FbgemmFp8HfQuantizer,
    "torchao": TorchAoHfQuantizer,
    "bitnet": BitNetHfQuantizer,
    "vptq": VptqHfQuantizer,
    "spqr": SpQRHfQuantizer,
    "fp8": FineGrainedFP8HfQuantizer,
    "auto-round": AutoRoundQuantizer,
    "mxfp4": Mxfp4HfQuantizer,
}

AUTO_QUANTIZATION_CONFIG_MAPPING = {
    "awq": AwqConfig,
    "bitsandbytes_4bit": BitsAndBytesConfig,
    "bitsandbytes_8bit": BitsAndBytesConfig,
    "eetq": EetqConfig,
    "gptq": GPTQConfig,
    "aqlm": AqlmConfig,
    "quanto": QuantoConfig,
    "quark": QuarkConfig,
    "fp_quant": FPQuantConfig,
    "hqq": HqqConfig,
    "compressed-tensors": CompressedTensorsConfig,
    "fbgemm_fp8": FbgemmFp8Config,
    "higgs": HiggsConfig,
    "torchao": TorchAoConfig,
    "bitnet": BitNetQuantConfig,
    "vptq": VptqConfig,
    "spqr": SpQRConfig,
    "fp8": FineGrainedFP8Config,
    "auto-round": AutoRoundConfig,
    "mxfp4": Mxfp4Config,
}

logger = logging.get_logger(__name__)


class AutoQuantizationConfig:
    """
    The Auto-HF quantization config class that takes care of automatically dispatching to the correct
    quantization config given a quantization config stored in a dictionary.
    """

    @classmethod
    def from_dict(cls, quantization_config_dict: dict):
        quant_method = quantization_config_dict.get("quant_method")
        # We need a special care for bnb models to make sure everything is BC ..
        if quantization_config_dict.get("load_in_8bit", False) or quantization_config_dict.get("load_in_4bit", False):
            suffix = "_4bit" if quantization_config_dict.get("load_in_4bit", False) else "_8bit"
            quant_method = QuantizationMethod.BITS_AND_BYTES + suffix
        elif quant_method is None:
            raise ValueError(
                "The model's quantization config from the arguments has no `quant_method` attribute. Make sure that the model has been correctly quantized"
            )

        if quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING:
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )

        target_cls = AUTO_QUANTIZATION_CONFIG_MAPPING[quant_method]
        return target_cls.from_dict(quantization_config_dict)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if getattr(model_config, "quantization_config", None) is None:
            raise ValueError(
                f"Did not found a `quantization_config` in {pretrained_model_name_or_path}. Make sure that the model is correctly quantized."
            )
        quantization_config_dict = model_config.quantization_config
        quantization_config = cls.from_dict(quantization_config_dict)
        # Update with potential kwargs that are passed through from_pretrained.
        quantization_config.update(**kwargs)
        return quantization_config

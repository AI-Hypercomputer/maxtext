
from dataclasses import dataclass, field
from typing import List, Optional

# Assuming QuantizationConfigMixin and QuantizationMethod are available from a JAX-equivalent module.
# Based on the FULL_FILE_CODE, these would be located in a file like `.../quantization_configs.py`
from .quantization_configs import QuantizationConfigMixin, QuantizationMethod


@dataclass
class BitNetQuantConfig(QuantizationConfigMixin):
  """
  Configuration class for applying BitNet quantization.

  Args:
      modules_to_not_convert (`Optional[List]`, *optional*):
          Optionally, provides a list of full paths of `nn.Linear` weight parameters
          that shall not be quantized. Defaults to None.
      linear_class (`str`, *optional*, defaults to `"bitlinear"`):
          The type of linear class to use. Can be either `bitlinear` or `autobitlinear`.
      quantization_mode (`str`, *optional*, defaults to `"offline"`):
          The quantization mode to use. Can be either `online` or `offline`.
          In `online` mode, the weight quantization parameters are calculated dynamically
          during each forward pass (e.g., based on the current weight values). This can
          adapt to weight changes during training (Quantization-Aware Training - QAT).
          In `offline` mode, quantization parameters are pre-calculated *before* inference.
          These parameters are then fixed and loaded into the quantized model. This
          generally results in lower runtime overhead compared to online quantization.
      use_rms_norm (`bool`, *optional*, defaults to `False`):
          Whether to apply RMSNorm on the activations before quantization. This matches the original BitNet paper's approach
          of normalizing activations before quantization/packing.
      rms_norm_eps (`float`, *optional*, defaults to 1e-06):
          The epsilon value used in the RMSNorm layer for numerical stability.
  """

  modules_to_not_convert: Optional[List[str]] = None
  linear_class: str = "bitlinear"
  quantization_mode: str = "offline"
  use_rms_norm: bool = False
  rms_norm_eps: float = 1e-6
  # The parent dataclass `QuantizationConfigMixin` has `quant_method` as a required field.
  # We use `field(init=False)` to exclude it from the constructor and set it in `__post_init__`.
  quant_method: QuantizationMethod = field(init=False)

  def __post_init__(self):
    """
    Safety checker that arguments are correct and sets the quant_method.
    """
    if self.linear_class not in ["bitlinear", "autobitlinear"]:
      raise ValueError(
          "linear_class must be either 'bitlinear' or 'autobitlinear', but"
          f" got {self.linear_class}"
      )
    if self.quantization_mode not in ["online", "offline"]:
      raise ValueError(
          "quantization_mode must be either 'online' or 'offline', but got"
          f" {self.quantization_mode}"
      )
    self.quant_method = QuantizationMethod.BITNET

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" A JAX equivalent of the HuggingFace BitsAndBytesConfig class."""
import copy
import json
import importlib.metadata
from packaging import version
from typing import Any, List, Optional, Union

import jax.numpy as jnp

from . import QuantizationConfigMixin, QuantizationMethod
from ..utils import logging


logger = logging.get_logger(__name__)


class BitsAndBytesConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `bitsandbytes`.

    This replaces `load_in_8bit` or `load_in_4bit`therefore both options are mutually exclusive.

    Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
    then more arguments will be added to this class.

    Args:
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 8-bit quantization with LLM.int8().
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from
            `bitsandbytes`.
        llm_int8_threshold (`float`, *optional*, defaults to 6.0):
            This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix
            Multiplication for Transformers at Scale` paper: https://huggingface.co/papers/2208.07339 Any hidden states value
            that is above this threshold will be considered an outlier and the operation on those values will be done
            in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but
            there are some exceptional systematic outliers that are very differently distributed for large models.
            These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of
            magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6,
            but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        llm_int8_skip_modules (`list[str]`, *optional*):
            An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as
            Jukebox that has several heads in different places and not necessarily at the last position. For example
            for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.
        llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`):
            This flag is used for advanced use cases and users that are aware of this feature. If you want to split
            your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
            this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
            operations will not be run on CPU.
        llm_int8_has_fp16_weight (`bool`, *optional*, defaults to `False`):
            This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not
            have to be converted back and forth for the backward pass.
        bnb_4bit_compute_dtype (`jnp.dtype` or str, *optional*, defaults to `jnp.float32`):
            This sets the computational type which might be different than the input type. For example, inputs might be
            fp32, but computation can be set to bf16 for speedups.
        bnb_4bit_quant_type (`str`,  *optional*, defaults to `"fp4"`):
            This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types
            which are specified by `fp4` or `nf4`.
        bnb_4bit_use_double_quant (`bool`, *optional*, defaults to `False`):
            This flag is used for nested quantization where the quantization constants from the first quantization are
            quantized again.
        bnb_4bit_quant_storage (`jnp.dtype` or str, *optional*, defaults to `jnp.uint8`):
            This sets the storage type to pack the quanitzed 4-bit prarams.
        kwargs (`dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(
        self,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        llm_int8_threshold: float = 6.0,
        llm_int8_skip_modules: Optional[List[str]] = None,
        llm_int8_enable_fp32_cpu_offload: bool = False,
        llm_int8_has_fp16_weight: bool = False,
        bnb_4bit_compute_dtype: Optional[Union[jnp.dtype, str]] = None,
        bnb_4bit_quant_type: str = "fp4",
        bnb_4bit_use_double_quant: bool = False,
        bnb_4bit_quant_storage: Optional[Union[jnp.dtype, str]] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.BITS_AND_BYTES

        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")

        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self.llm_int8_threshold = llm_int8_threshold
        self.llm_int8_skip_modules = llm_int8_skip_modules
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

        if bnb_4bit_compute_dtype is None:
            self.bnb_4bit_compute_dtype = jnp.float32
        elif isinstance(bnb_4bit_compute_dtype, str):
            self.bnb_4bit_compute_dtype = getattr(jnp, bnb_4bit_compute_dtype)
        elif isinstance(bnb_4bit_compute_dtype, jnp.dtype):
            self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        else:
            raise ValueError("bnb_4bit_compute_dtype must be a string or a jnp.dtype")

        if bnb_4bit_quant_storage is None:
            self.bnb_4bit_quant_storage = jnp.uint8
        elif isinstance(bnb_4bit_quant_storage, str):
            if bnb_4bit_quant_storage not in ["float16", "float32", "int8", "uint8", "float64", "bfloat16"]:
                raise ValueError(
                    "`bnb_4bit_quant_storage` must be a valid string (one of 'float16', 'float32', 'int8', 'uint8', 'float64', 'bfloat16') "
                )
            self.bnb_4bit_quant_storage = getattr(jnp, bnb_4bit_quant_storage)
        elif isinstance(bnb_4bit_quant_storage, jnp.dtype):
            self.bnb_4bit_quant_storage = bnb_4bit_quant_storage
        else:
            raise ValueError("bnb_4bit_quant_storage must be a string or a jnp.dtype")

        if kwargs:
            logger.info(f"Unused kwargs: {list(kwargs.keys())}. These kwargs are not used in {self.__class__}.")

        self.post_init()

    @property
    def load_in_4bit(self):
        return self._load_in_4bit

    @load_in_4bit.setter
    def load_in_4bit(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("load_in_4bit must be a boolean")

        if self.load_in_8bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        self._load_in_4bit = value

    @property
    def load_in_8bit(self):
        return self._load_in_8bit

    @load_in_8bit.setter
    def load_in_8bit(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("load_in_8bit must be a boolean")

        if self.load_in_4bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        self._load_in_8bit = value

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.load_in_4bit, bool):
            raise TypeError("load_in_4bit must be a boolean")

        if not isinstance(self.load_in_8bit, bool):
            raise TypeError("load_in_8bit must be a boolean")

        if not isinstance(self.llm_int8_threshold, float):
            raise TypeError("llm_int8_threshold must be a float")

        if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
            raise TypeError("llm_int8_skip_modules must be a list of strings")
        if not isinstance(self.llm_int8_enable_fp32_cpu_offload, bool):
            raise TypeError("llm_int8_enable_fp32_cpu_offload must be a boolean")

        if not isinstance(self.llm_int8_has_fp16_weight, bool):
            raise TypeError("llm_int8_has_fp16_weight must be a boolean")

        if self.bnb_4bit_compute_dtype is not None and not isinstance(self.bnb_4bit_compute_dtype, jnp.dtype):
            raise TypeError("bnb_4bit_compute_dtype must be jnp.dtype")

        if not isinstance(self.bnb_4bit_quant_type, str):
            raise TypeError("bnb_4bit_quant_type must be a string")

        if not isinstance(self.bnb_4bit_use_double_quant, bool):
            raise TypeError("bnb_4bit_use_double_quant must be a boolean")

        # TODO(b/346363083): bitsandbytes is a PyTorch-specific library. This check is not applicable in a JAX environment.
        # It is commented out to maintain structural similarity while acknowledging the difference in ecosystems.
        # if self.load_in_4bit and not version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse(
        #     "0.39.0"
        # ):
        #     raise ValueError(
        #         "4 bit quantization requires bitsandbytes>=0.39.0 - please upgrade your bitsandbytes version"
        #     )

    def is_quantizable(self):
        r"""
        Returns `True` if the model is quantizable, `False` otherwise.
        """
        return self.load_in_8bit or self.load_in_4bit

    def quantization_method(self):
        r"""
        This method returns the quantization method used for the model. If the model is not quantizable, it returns
        `None`.
        """
        if self.load_in_8bit:
            return "llm_int8"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "fp4":
            return "fp4"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "nf4":
            return "nf4"
        else:
            return None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["bnb_4bit_compute_dtype"] = output["bnb_4bit_compute_dtype"].name
        output["bnb_4bit_quant_storage"] = output["bnb_4bit_quant_storage"].name
        output["load_in_4bit"] = self.load_in_4bit
        output["load_in_8bit"] = self.load_in_8bit

        return output

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = BitsAndBytesConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict

from dataclasses import dataclass
from typing import List, Optional

# Reused modules:
# generated_code.Qwen3MoeForCausalLM.quantization.QuantizationConfigMixin
# generated_code.Qwen3MoeForCausalLM.quantization.QuantizationMethod
from generated_code.Qwen3MoeForCausalLM.quantization import (
    QuantizationConfigMixin,
    QuantizationMethod,
)


@dataclass
class EetqConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `eetq`.

    Args:
        weights (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights. Supported value is only "int8"
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision.
    """

    def __init__(
        self,
        weights: str = "int8",
        modules_to_not_convert: Optional[List[str]] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.EETQ
        self.weights = weights
        self.modules_to_not_convert = modules_to_not_convert
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_weights = ["int8"]
        if self.weights not in accepted_weights:
            raise ValueError(f"Only support weights in {accepted_weights} but found {self.weights}")

from dataclasses import dataclass, field
from typing import List, Optional

# Re-used modules:
# src.MaxText.layers.quantizations.QuantizationConfigMixin
# src.MaxText.layers.quantizations.QuantizationMethod
from ..quantizations import QuantizationConfigMixin, QuantizationMethod


@dataclass
class FPQuantConfig(QuantizationConfigMixin):
  """
    FPQuantConfig is a configuration class for quantization using the FPQuant method.

    Attributes:
        forward_dtype (`str`, *optional*, defaults to `"mxfp4"`):
            The dtype to use for the forward pass.
        forward_method (`str`, *optional*, defaults to `"abs_max"`):
            The scaling to use for the forward pass. Can be `"abs_max"` or `"quest"`. `"abs_max"` is better for PTQ, `"quest"` is better for QAT.
        backward_dtype (`str`, *optional*, defaults to `"bf16"`):
            The dtype to use for the backward pass.
        store_master_weights (`bool`, *optional*, defaults to `False`):
            Whether to store the master weights. Needed for QAT over layer weights.
        hadamard_group_size (`int`, *optional*, defaults to 32):
            The group size for the hadamard transform before quantization for `"quest"` it matches the MXFP4 group size (32).
        pseudoquantization (`bool`, *optional*, defaults to `False`):
            Whether to use Triton-based pseudo-quantization. Is mandatory for non-Blackwell GPUs. Doesn't provide any speedup. For debugging purposes.
        modules_to_not_convert (`list`, *optional*):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision.
    """

  forward_dtype: str = "mxfp4"
  forward_method: str = "abs_max"
  backward_dtype: str = "bf16"
  store_master_weights: bool = False
  hadamard_group_size: int = 32
  pseudoquantization: bool = False
  modules_to_not_convert: Optional[List[str]] = None
  quant_method: QuantizationMethod = field(init=False)

  def __post_init__(self):
    """
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
    self.quant_method = QuantizationMethod.FPQUANT
    if self.forward_dtype not in ["mxfp4"]:
      raise ValueError("Only 'mxfp4' is supported for forward_dtype for now.")
    if self.forward_method not in ["abs_max", "quest"]:
      raise ValueError(
          "Only 'abs_max' and 'quest' are supported for forward_method for now."
      )
    if self.backward_dtype not in ["bf16"]:
      raise ValueError("Only 'bf16' is supported for backward_dtype for now.")
    if self.hadamard_group_size not in [32]:
      raise ValueError(
          "Only a hadamard_group_size of 32 is supported for now."
      )
    if self.modules_to_not_convert is None:
      self.modules_to_not_convert = ["lm_head"]

from typing import Optional, Tuple

# Reused from generated_code.Qwen3MoeForCausalLM.quantization.QuantizationConfigMixin
from .quantization import QuantizationConfigMixin
# Reused from generated_code.Qwen3MoeForCausalLM.quantization.QuantizationMethod
from .quantization import QuantizationMethod


class FineGrainedFP8Config(QuantizationConfigMixin):
    """
    FineGrainedFP8Config is a configuration class for fine-grained FP8 quantization used mainly for deepseek models.

    Args:
        activation_scheme (`str`, *optional*, defaults to `"dynamic"`):
            The scheme used for activation, the defaults and only support scheme for now is "dynamic".
        weight_block_size (`typing.Tuple[int, int]`, *optional*, defaults to `(128, 128)`):
            The size of the weight blocks for quantization, default is (128, 128).
        modules_to_not_convert (`list`, *optional*):
            A list of module names that should not be converted during quantization.
    """

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        weight_block_size: Tuple[int, int] = (128, 128),
        modules_to_not_convert: Optional[list] = None,
        **kwargs,
    ):
        self.modules_to_not_convert = modules_to_not_convert
        self.activation_scheme = activation_scheme
        self.weight_block_size = weight_block_size
        super().__init__(quant_method=QuantizationMethod.FP8)
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        self.activation_scheme = self.activation_scheme.lower()
        if self.activation_scheme not in ["dynamic"]:
            raise ValueError(f"Activation scheme {self.activation_scheme} not supported")
        if len(self.weight_block_size) != 2:
            raise ValueError("weight_block_size must be a tuple of two integers")
        if self.weight_block_size[0] <= 0 or self.weight_block_size[1] <= 0:
            raise ValueError("weight_block_size must be a tuple of two positive integers")

from typing import Any, Optional

# Reused from generated_code.Qwen3MoeForCausalLM.quantization.QuantizationConfigMixin
from .quantization import QuantizationConfigMixin
# Reused from generated_code.Qwen3MoeForCausalLM.quantization.QuantizationMethod
from .quantization import QuantizationMethod


class HiggsConfig(QuantizationConfigMixin):
  """
    HiggsConfig is a configuration class for quantization using the HIGGS method.

    Args:
        bits (int, *optional*, defaults to 4):
            Number of bits to use for quantization. Can be 2, 3 or 4. Default is 4.
        p (int, *optional*, defaults to 2):
            Quantization grid dimension. 1 and 2 are supported. 2 is always better in practice. Default is 2.
        modules_to_not_convert (`list`, *optional*, default to ["lm_head"]):
            List of linear layers that should not be quantized.
        hadamard_size (int, *optional*, defaults to 512):
            Hadamard size for the HIGGS method. Default is 512. Input dimension of matrices is padded to this value. Decreasing this below 512 will reduce the quality of the quantization.
        group_size (int, *optional*, defaults to 256):
            Group size for the HIGGS method. Can be 64, 128 or 256. Decreasing it barely affects the performance. Default is 256. Must be a divisor of hadamard_size.
        tune_metadata ('dict', *optional*, defaults to {}):
            Module-wise metadata (gemm block shapes, GPU metadata, etc.) for saving the kernel tuning results. Default is an empty dictionary. Is set automatically during tuning.
    """

  def __init__(
      self,
      bits: int = 4,
      p: int = 2,
      modules_to_not_convert: Optional[list[str]] = None,
      hadamard_size: int = 512,
      group_size: int = 256,
      tune_metadata: Optional[dict[str, Any]] = None,
      **kwargs,
  ):
    if tune_metadata is None:
      tune_metadata = {}
    self.quant_method = QuantizationMethod.HIGGS
    self.bits = bits
    self.p = p
    self.modules_to_not_convert = modules_to_not_convert
    self.hadamard_size = hadamard_size
    self.group_size = group_size
    self.tune_metadata = tune_metadata

    self.post_init()

  def post_init(self):
    r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
    if self.bits not in [2, 3, 4]:
      raise ValueError("bits must be 2, 3, or 4")
    if self.p not in [1, 2]:
      raise ValueError("p must be 1 or 2. 2 is always better in practice")
    if self.group_size not in [64, 128, 256]:
      raise ValueError("group_size must be 64, 128, or 256")
    if self.hadamard_size % self.group_size != 0:
      raise ValueError("hadamard_size must be divisible by group_size")

import json
from typing import Any, Optional

from absl import logging

# Assuming these are defined in the MaxText quantization library
from MaxText.layers.quantizations import QuantizationConfigMixin, QuantizationMethod
# Assuming a utility function to check for HQQ availability exists
from MaxText.utils import is_hqq_available


class HqqConfig(QuantizationConfigMixin):
    """
    This is wrapper around hqq's BaseQuantizeConfig.

    Args:
        nbits (`int`, *optional*, defaults to 4):
            Number of bits. Supported values are (8, 4, 3, 2, 1).
        group_size (`int`, *optional*, defaults to 64):
            Group-size value. Supported values are any value that is divisible by weight.shape[axis]).
        view_as_float (`bool`, *optional*, defaults to `False`):
            View the quantized weight as float (used in distributed training) if set to `True`.
        axis (`Optional[int]`, *optional*):
            Axis along which grouping is performed. Supported values are 0 or 1.
        dynamic_config (dict, *optional*):
            Parameters for dynamic configuration. The key is the name tag of the layer and the value is a quantization config.
            If set, each layer specified by its id will use its dedicated quantization configuration.
        skip_modules (`list[str]`, *optional*, defaults to `['lm_head']`):
            List of `nn.Linear` layers to skip.
        kwargs (`dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(
        self,
        nbits: int = 4,
        group_size: int = 64,
        view_as_float: bool = False,
        axis: Optional[int] = None,
        dynamic_config: Optional[dict] = None,
        skip_modules: list[str] = ["lm_head"],
        **kwargs,
    ):
        if is_hqq_available():
            from hqq.core.quantize import BaseQuantizeConfig as HQQBaseQuantizeConfig
        else:
            raise ImportError(
                "A valid HQQ version (>=0.2.1) is not available. Please follow the instructions to install it: `https://github.com/mobiusml/hqq/`."
            )

        for deprecated_key in ["quant_zero", "quant_scale", "offload_meta"]:
            if deprecated_key in kwargs:
                logging.info(
                    "%s is deprecated. This parameter will be ignored in quantization settings.", deprecated_key
                )

        if axis is None:
            axis = 1
            logging.info("Setting axis=1 as faster backends such as TorchAO or BitBlas are only compatible with it.")

        if axis not in [0, 1]:
            raise ValueError("Invalid axis value. Only 0 and 1 are allowed.")

        if dynamic_config is not None:
            self.quant_config = {}
            for key in dynamic_config:
                self.quant_config[key] = HQQBaseQuantizeConfig(**dynamic_config[key])
        else:
            self.quant_config = HQQBaseQuantizeConfig(
                **{
                    "nbits": nbits,
                    "group_size": group_size,
                    "view_as_float": view_as_float,
                    "axis": axis,
                }
            )

        self.quant_method = QuantizationMethod.HQQ
        self.skip_modules = skip_modules

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        pass

    @classmethod
    def from_dict(cls, config: dict[str, Any]):
        """
        Override from_dict, used in AutoQuantizationConfig.from_dict in quantizers/auto.py
        """
        # Create an instance without calling __init__ to avoid re-running logic
        instance = cls.__new__(cls)
        instance.quant_config = config["quant_config"]
        instance.skip_modules = config["skip_modules"]
        instance.quant_method = QuantizationMethod.HQQ
        return instance

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return {
            "quant_config": self.quant_config,
            "quant_method": self.quant_method,
            "skip_modules": self.skip_modules,
        }

    def __repr__(self):
        config_dict = self.to_dict()

        def default_serializer(o):
            if hasattr(o, "to_dict"):
                return o.to_dict()
            if hasattr(o, "__dict__"):
                return o.__dict__
            return str(o)

        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True, default=default_serializer)}\n"

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.
        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = HqqConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            # The HQQBaseQuantizeConfig object might not have a proper __eq__ method.
            # Comparing string representations is a robust fallback.
            if str(value) != str(default_config_dict[key]):
                serializable_config_dict[key] = value

        return serializable_config_dict

import dataclasses
from typing import List, Optional

# Used from: generated_code.Qwen3MoeForCausalLM.quantization.QuantizationConfigMixin
from . import QuantizationConfigMixin
# Used from: generated_code.Qwen3MoeForCausalLM.quantization.QuantizationMethod
from . import QuantizationMethod


@dataclasses.dataclass
class QuantoConfig(QuantizationConfigMixin):
  """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `quanto`.

    Args:
        weights (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights after quantization. Supported values are ("float8","int8","int4","int2")
        activations (`str`, *optional*):
            The target dtype for the activations after quantization. Supported values are (None,"int8","float8")
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
    """

  def __init__(
      self,
      weights: str = "int8",
      activations: Optional[str] = None,
      modules_to_not_convert: Optional[List[str]] = None,
      **kwargs,
  ):
    self.quant_method = QuantizationMethod.QUANTO
    self.weights = weights
    self.activations = activations
    self.modules_to_not_convert = modules_to_not_convert
    self.post_init()

  def post_init(self):
    r"""
        Safety checker that arguments are correct
        """
    accepted_weights = ["float8", "int8", "int4", "int2"]
    accepted_activations = [None, "int8", "float8"]
    if self.weights not in accepted_weights:
      raise ValueError(f"Only support weights in {accepted_weights} but found {self.weights}")
    if self.activations not in accepted_activations:
      raise ValueError(f"Only support weights in {accepted_activations} but found {self.activations}")

from typing import Any
from absl import logging
from packaging import version

# Assuming these are available from a utils file and the current file's JAX equivalent.
from ..utils import is_quark_available
from .quantization_config import QuantizationConfigMixin, QuantizationMethod


class QuarkConfig(QuantizationConfigMixin):
  """Configuration class for Quark quantization."""

  def __init__(self, **kwargs: Any):
    if is_quark_available():
      from quark import __version__ as quark_version
      # Assuming a JAX backend for quark exists that mirrors the torch API
      from quark.jax.export.config.config import JsonExporterConfig
      from quark.jax.export.main_export.quant_config_parser import (
          QuantConfigParser,
      )
      from quark.jax.quantization.config.config import Config
    else:
      raise ImportError(
          "Quark is not installed. Please refer to"
          " https://quark.docs.amd.com/latest/install.html."
      )
    # This might be e.g. `"fp8"` or `"awq"`.
    self.custom_mode = kwargs["quant_method"]
    self.legacy = "export" not in kwargs

    if self.custom_mode in ["awq", "fp8"]:
      # Legacy (quark<1.0) or custom export.
      self.quant_config = QuantConfigParser.from_custom_config(
          kwargs, is_bias_quantized=False
      )
      self.json_export_config = JsonExporterConfig()
    else:
      self.quant_config = Config.from_dict(kwargs)

      if "export" in kwargs:
        # TODO: Remove this check once configuration version is handled
        # natively by Quark.
        if "min_kv_scale" in kwargs["export"] and version.parse(
            quark_version
        ) < version.parse("0.8"):
          min_kv_scale = kwargs["export"].pop("min_kv_scale")
          logging.warning(
              "The parameter `min_kv_scale=%s` was found in the model"
              " config.json's `quantization_config.export` configuration, but"
              " this parameter is supported only for quark>=0.8. Ignoring this"
              " configuration parameter. Please update the `amd-quark`"
              " package.",
              min_kv_scale,
          )

        self.json_export_config = JsonExporterConfig(**kwargs["export"])
      else:
        # Legacy (quark<1.0) or custom export.
        self.json_export_config = JsonExporterConfig()

    self.quant_method = QuantizationMethod.QUARK

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class QuantizationMethod(str, Enum):
    """
    The quantization methods supported.
    """
    SPQR = "spqr"


@dataclass
class QuantizationConfigMixin:
    """
    Mixin class for quantization config.
    """
    quant_method: QuantizationMethod


class SpQRConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about `spqr` parameters. Refer to the original publication for more details.

    Args:
        bits (`int`, *optional*, defaults to 3):
            Specifies the bit count for the weights and first order zero-points and scales.
            Currently only bits = 3 is supported.
        beta1 (`int`, *optional*, defaults to 16):
            SpQR tile width. Currently only beta1 = 16 is supported.
        beta2 (`int`, *optional*, defaults to 16):
            SpQR tile height. Currently only beta2 = 16 is supported.
        shapes (`Optional`, *optional*):
            A dictionary holding the shape of each object. We need this because it's impossible
            to deduce the exact size of the parameters just from bits, beta1, beta2.
        modules_to_not_convert (`Optional[list[str]]`, *optional*):
            Optionally, provides a list of full paths of `nn.Linear` weight parameters that shall not be quantized.
            Defaults to None.
        kwargs (`dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(
        self,
        bits: int = 3,
        beta1: int = 16,
        beta2: int = 16,
        shapes: Optional[Dict[str, int]] = None,
        modules_to_not_convert: Optional[List[str]] = None,
        **kwargs,
    ):
        if shapes is None:
            shapes = {}
        self.shapes = shapes
        self.quant_method = QuantizationMethod.SPQR
        self.bits = bits
        self.beta1 = beta1
        self.beta2 = beta2
        self.modules_to_not_convert = modules_to_not_convert
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.bits, int):
            raise TypeError("bits must be an int")
        if not isinstance(self.beta1, int):
            raise TypeError("beta1 must be an int")
        if not isinstance(self.beta2, int):
            raise TypeError("beta2 must be an int")

        if self.bits != 3:
            raise ValueError("SpQR currently only supports bits = 3")
        if self.beta1 != 16:
            raise ValueError("SpQR currently only supports beta1 = 16")
        if self.beta2 != 16:
            raise ValueError("SpQR currently only supports beta2 = 16")
        if not isinstance(self.shapes, dict):
            raise TypeError("shapes must be a dict")

from __future__ import annotations

import dataclasses
import importlib.metadata
from dataclasses import dataclass, is_dataclass
from inspect import Parameter, signature
from typing import Any, Optional, Union

from packaging import version

# MaxText-matched dependencies:
# From generated_code.Qwen3MoeForCausalLM.quantization.QuantizationConfigMixin
from .quantization_config import QuantizationConfigMixin
# From generated_code.Qwen3MoeForCausalLM.quantization.QuantizationMethod
from .quantization_config import QuantizationMethod


# A placeholder for a JAX quantization library availability check, similar to is_torchao_available
def is_aot_available():
  """Checks if a JAX quantization library (e.g., aqt) is available."""
  try:
    importlib.metadata.version("aqt-jax")
    return True
  except importlib.metadata.PackageNotFoundError:
    return False


# A placeholder for a base JAX quantization config class, similar to AOBaseConfig
class AOTBaseConfig:
  pass


# A placeholder for a JAX quantization library's config-to-dict utility
def config_to_dict(config: Any) -> dict[str, Any]:
  """Serializes a JAX quantization config object to a dictionary."""
  if is_dataclass(config):
    return {"class_name": config.__class__.__name__, "params": dataclasses.asdict(config)}
  raise TypeError(f"Object of type {type(config)} is not JSON serializable")


# A placeholder for a JAX quantization library's dict-to-config utility
def config_from_dict(config_dict: dict[str, Any]) -> AOTBaseConfig:
  """Deserializes a dictionary into a JAX quantization config object."""
  # In a real implementation, this would use a registry to map class_name to a class
  # and instantiate it with params. This is a simplified placeholder.
  print(f"Warning: config_from_dict is a placeholder and cannot reconstruct complex objects from {config_dict}")
  return AOTBaseConfig()


@dataclass
class TorchAoConfig(QuantizationConfigMixin):
  """This is a config class for torchao-like quantization/sparsity techniques in JAX.

  Args:
    quant_type (`Union[str, AOTBaseConfig]`):
        The type of quantization we want to use. Can be either:
        - A string: currently supporting: `int4_weight_only`, `int8_weight_only`
          and `int8_dynamic_activation_int8_weight`.
        - An AOTBaseConfig instance: for more advanced configuration options.
    modules_to_not_convert (`list`, *optional*, default to `None`):
        The list of modules to not quantize, useful for quantizing models that
        explicitly require to have some modules left in their original precision.
    include_input_output_embeddings (`bool`, default to `False`):
        Whether to include embedding in quantization or not, input embedding
        will be removed from the module_not_to_convert list as well if this
        flag is set.
    untie_embedding_weights (`bool`, default to `False`):
        Whether to untie the weights when we are quantizing input embedding
        weights that is tied to other weights.
    quant_type_kwargs (`dict[str, Any]`, *optional*):
        The keyword arguments for the chosen type of quantization.
  """

  quant_type: Union[str, "AOTBaseConfig"]
  modules_to_not_convert: Optional[list] = None
  include_input_output_embeddings: bool = False
  untie_embedding_weights: bool = False
  quant_type_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    """Validate configuration and set defaults."""
    self.quant_method = QuantizationMethod.TORCHAO
    self._get_aot_version()  # Check for JAX AOT library availability

    if isinstance(self.quant_type, str):
      self._validate_string_quant_type()
    elif not isinstance(self.quant_type, AOTBaseConfig):
      # This check is simplified as AOTBaseConfig is a placeholder.
      # In a real scenario, it would check against a known base class for JAX quant configs.
      print(f"Warning: quant_type is an object of type {type(self.quant_type)}, "
            "but not a recognized AOTBaseConfig. Proceeding with caution.")

  @staticmethod
  def _get_aot_version() -> version.Version:
    """Centralized check for JAX AOT availability and version requirements."""
    if not is_aot_available():
      raise ValueError("TorchAoConfig requires a JAX AOT library to be installed, e.g., `pip install aqt-jax`")
    # Using 'aqt-jax' as a stand-in for a 'torchao' equivalent in JAX.
    return version.parse(importlib.metadata.version("aqt-jax"))

  def _validate_string_quant_type(self):
    """Validate string quant_type."""
    # In a real JAX implementation, this would map to JAX functions/configs.
    # For this translation, we just check against a list of supported names.
    # The kwarg validation against a function signature is removed as it's too specific.
    supported_methods = [
        "int4_weight_only",
        "int8_weight_only",
        "int8_dynamic_activation_int8_weight",
        "autoquant",
    ]

    if self.quant_type not in supported_methods:
      raise ValueError(
          f"Unsupported string quantization type: {self.quant_type}. "
          f"Supported types: {', '.join(supported_methods)}"
      )

  def to_dict(self) -> dict[str, Any]:
    """Convert configuration to a dictionary."""
    d = super().to_dict()

    if not isinstance(self.quant_type, str):
      # Handle AOTBaseConfig serialization
      # For now we assume there is 1 config per Transformer.
      d["quant_type"] = {"default": config_to_dict(self.quant_type)}

    return d

  @classmethod
  def from_dict(cls, config_dict: dict[str, Any], return_unused_kwargs: bool = False, **kwargs) -> "TorchAoConfig":
    """Create configuration from a dictionary."""
    cls._get_aot_version()  # Check for availability
    config_dict = config_dict.copy()
    quant_type = config_dict.pop("quant_type")

    if isinstance(quant_type, str):
      instance = cls(quant_type=quant_type, **config_dict)
    else:
      # Handle object deserialization
      if not (isinstance(quant_type, dict) and len(quant_type) == 1 and "default" in quant_type):
        raise ValueError("Expected only one key 'default' in quant_type dictionary")
      quant_type_dict = quant_type["default"]

      # Deserialize quant_type if needed
      quant_type_obj = config_from_dict(quant_type_dict)
      instance = cls(quant_type=quant_type_obj, **config_dict)

    # This part is to maintain compatibility with the mixin's from_dict signature
    to_remove = []
    for key, value in kwargs.items():
      if hasattr(instance, key):
        setattr(instance, key, value)
        to_remove.append(key)
    for key in to_remove:
      kwargs.pop(key, None)

    if return_unused_kwargs:
      return instance, kwargs
    else:
      return instance

from typing import Tuple

# The QuantizationConfigMixin is defined in the same file in the provided FULL_FILE_CODE.
# No import is required.
from .quantization_configs import QuantizationConfigMixin


class VptqLayerConfig(QuantizationConfigMixin):
    """
    This is used to explain vptq config params for each layer
    Args:
        enable_norm (`bool`, *optional*, defaults to `True`): to control if we have scale/bias for fp-weight
        enable_perm (`bool`, *optional*, defaults to `True`): to perm input_channel or not
        group_num (`int`, *optional*, defaults to `1`): how many single groups for vector-quantization
        group_size (`int`, *optional*, defaults to `-1`): depends on out-features
        indices_as_float (`bool`, *optional*, defaults to `False`): for Finetuning
        is_indice_packed (`bool`, *optional*, defaults to `True`): should always be True
        num_centroids (`list`, *optional*, defaults to `[-1, -1]`): centroid numbers of clusters
        num_res_centroids (`list`, *optional*, defaults to `[-1, -1]`): ditto for residual
        outlier_size (`int`, *optional*, defaults to `1`): outliers
        vector_lens (`list`, *optional*, defaults to `[-1, -1]`): centroid vector length in quantization
    """

    def __init__(
        self,
        enable_norm: bool = True,
        enable_perm: bool = True,
        group_num: int = 1,
        group_size: int = -1,
        in_features: int = -1,
        indices_as_float: bool = False,
        is_indice_packed: bool = True,
        num_centroids: Tuple = (-1, -1),
        num_res_centroids: Tuple = (-1, -1),
        out_features: int = -1,
        outlier_size: int = 0,
        vector_lens: Tuple = (-1, -1),
        **kwargs,
    ):
        self.enable_norm = enable_norm
        self.enable_perm = enable_perm
        self.group_num = group_num
        self.group_size = group_size
        self.in_features = in_features
        self.indices_as_float = indices_as_float
        self.is_indice_packed = is_indice_packed
        self.num_centroids = num_centroids
        self.num_res_centroids = num_res_centroids
        self.out_features = out_features
        self.outlier_size = outlier_size
        self.vector_lens = vector_lens
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.is_indice_packed is False:
            raise ValueError("is_indice_packed should always be True")

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .quantization_config import QuantizationConfigMixin, VptqLayerConfig
from .quantization_config import QuantizationMethod


@dataclass
class VptqConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about `vptq` parameters.

    Args:
        enable_proxy_error (`bool`, *optional*, defaults to `False`): calculate proxy error for each layer
        config_for_layers (`Dict`, *optional*, defaults to `{}`): quantization params for each layer
        shared_layer_config (`Dict`, *optional*, defaults to `{}`): shared quantization params among layers
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
    """
    enable_proxy_error: bool = False
    config_for_layers: Dict[str, Any] = field(default_factory=dict)
    shared_layer_config: Dict[str, Any] = field(default_factory=dict)
    modules_to_not_convert: Optional[List[str]] = None
    quant_method: QuantizationMethod = field(init=False)

    def __post_init__(self):
        r"""
        Safety checker that arguments are correct
        """
        self.quant_method = QuantizationMethod.VPTQ
        for layer_param in self.config_for_layers.values():
            VptqLayerConfig(**layer_param)
        if self.enable_proxy_error is True:
            raise ValueError("enable_proxy_error should always be False until we support training")
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
from __future__ import annotations

import dataclasses
from typing import Any

from .quantization_utils import QuantizationConfigMixin, QuantizationMethod


@dataclasses.dataclass
class AutoRoundConfig(QuantizationConfigMixin):
    """This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded AutoRound quantization.

    Args:
        bits (`int`, *optional*, defaults to 4):
            The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
        group_size (`int`, *optional*, defaults to 128): Group-size value
        sym (`bool`, *optional*, defaults to `True`): Symmetric quantization or not
        backend (`str`, *optional*, defaults to `"auto"`): The kernel to use, e.g., ipex,marlin, exllamav2, triton, etc. Ref. https://github.com/intel/auto-round?tab=readme-ov-file#specify-backend
    """

    bits: int = 4
    group_size: int = 128
    sym: bool = True
    backend: str = "auto"
    packing_format: str = "auto_round:gptq"

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
        backend: str = "auto",
        **kwargs,
    ):
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.backend = backend
        self.packing_format = "auto_round:gptq"
        super().__init__(quant_method=QuantizationMethod.AUTOROUND)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.post_init()

    def post_init(self):
        r"""Safety checker that arguments are correct."""
        if self.bits not in [2, 3, 4, 8]:
            raise ValueError(f"Only support quantization to [2,3,4,8] bits but found {self.bits}")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")

    def get_loading_attributes(self) -> dict[str, Any]:
        loading_attibutes_dict = {"backend": self.backend}
        return loading_attibutes_dict

    def to_dict(self) -> dict[str, Any]:
        config_dict = super().to_dict()
        return config_dict

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        quant_method = config_dict["quant_method"]
        if "auto-round" not in quant_method and "gptq" not in quant_method and "awq" not in quant_method:
            raise NotImplementedError(
                "Failed to convert to auto_round format. Only `gptqv1`, `awq`, and `auto-round` formats are supported."
            )

        if "gptq" in quant_method and "meta" in config_dict:
            raise NotImplementedError("Failed to convert gptq format to auto_round format. Only supports `gptqv1`")

        if "awq" in quant_method and config_dict.get("version", "gemm") != "gemm":
            raise NotImplementedError(
                "Failed to convert awq format to auto_round format. Only supports awq format with gemm version"
            )

        if "auto-round" not in quant_method:
            config_dict["packing_format"] = f"auto_round:{quant_method}"

        return super().from_dict(config_dict, return_unused_kwargs=return_unused_kwargs, **kwargs)

import copy
import importlib.metadata
from typing import Any, Optional

from packaging import version

# The following imports are assumed to be available in the JAX environment.
# They are based on the structure of the original PyTorch file.
# from ..utils import is_auto_awq_available
# from .base import (
#     QuantizationConfigMixin,
#     QuantizationMethod,
#     AWQLinearVersion,
#     AwqBackendPackingMethod,
#     ExllamaVersion,
# )
# Since no JAX modules were provided, these dependencies are assumed to be defined elsewhere.
# For this code block to be self-contained for review, placeholder definitions would be needed.
# For example:
# from enum import Enum
# class QuantizationConfigMixin: pass
# class QuantizationMethod(str, Enum): AWQ = "awq"
# class AWQLinearVersion(str, Enum): GEMM = "gemm"; GEMV = "gemv"; EXLLAMA = "exllama"; IPEX = "ipex"; @staticmethod def from_str(v): return AWQLinearVersion(v.lower())
# class AwqBackendPackingMethod(str, Enum): AUTOAWQ = "autoawq"; LLMAWQ = "llm-awq"
# class ExllamaVersion(int, Enum): ONE = 1; TWO = 2
# def is_auto_awq_available(): return False


class AwqConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `auto-awq` library awq quantization relying on auto_awq backend.

    Args:
        bits (`int`, *optional*, defaults to 4):
            The number of bits to quantize to.
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        zero_point (`bool`, *optional*, defaults to `True`):
            Whether to use zero point quantization.
        version (`AWQLinearVersion`, *optional*, defaults to `AWQLinearVersion.GEMM`):
            The version of the quantization algorithm to use. GEMM is better for big batch_size (e.g. >= 8) otherwise,
            GEMV is better (e.g. < 8 ). GEMM models are compatible with Exllama kernels.
        backend (`AwqBackendPackingMethod`, *optional*, defaults to `AwqBackendPackingMethod.AUTOAWQ`):
            The quantization backend. Some models might be quantized using `llm-awq` backend. This is useful for users
            that quantize their own models using `llm-awq` library.
        do_fuse (`bool`, *optional*, defaults to `False`):
            Whether to fuse attention and mlp layers together for faster inference
        fuse_max_seq_len (`int`, *optional*):
            The Maximum sequence length to generate when using fusing.
        modules_to_fuse (`dict`, *optional*, default to `None`):
            Overwrite the natively supported fusing scheme with the one specified by the users.
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
            Note you cannot quantize directly with transformers, please refer to `AutoAWQ` documentation for quantizing HF models.
        exllama_config (`dict[str, Any]`, *optional*):
            You can specify the version of the exllama kernel through the `version` key, the maximum sequence
            length through the `max_input_len` key, and the maximum batch size through the `max_batch_size` key.
            Defaults to `{"version": 2, "max_input_len": 2048, "max_batch_size": 8}` if unset.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        version: AWQLinearVersion = AWQLinearVersion.GEMM,
        backend: AwqBackendPackingMethod = AwqBackendPackingMethod.AUTOAWQ,
        do_fuse: Optional[bool] = None,
        fuse_max_seq_len: Optional[int] = None,
        modules_to_fuse: Optional[dict] = None,
        modules_to_not_convert: Optional[list] = None,
        exllama_config: Optional[dict[str, int]] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.AWQ

        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.backend = backend
        self.fuse_max_seq_len = fuse_max_seq_len
        self.modules_to_not_convert = modules_to_not_convert
        self.exllama_config = exllama_config

        self.modules_to_fuse = modules_to_fuse
        if do_fuse is None:
            self.do_fuse = modules_to_fuse is not None and len(modules_to_fuse) > 0
        else:
            self.do_fuse = do_fuse
        self.fuse_max_seq_len = fuse_max_seq_len

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.backend not in [AwqBackendPackingMethod.AUTOAWQ, AwqBackendPackingMethod.LLMAWQ]:
            raise ValueError(
                f"Only supported quantization backends in {AwqBackendPackingMethod.AUTOAWQ} and {AwqBackendPackingMethod.LLMAWQ} - not recognized backend {self.backend}"
            )

        self.version = AWQLinearVersion.from_str(self.version)
        if self.version not in [
            AWQLinearVersion.GEMM,
            AWQLinearVersion.GEMV,
            AWQLinearVersion.EXLLAMA,
            AWQLinearVersion.IPEX,
        ]:
            raise ValueError(
                f"Only supported versions are in [AWQLinearVersion.GEMM, AWQLinearVersion.GEMV, AWQLinearVersion.EXLLAMA, AWQLinearVersion.IPEX] - not recognized version {self.version}"
            )

        if self.backend == AwqBackendPackingMethod.LLMAWQ:
            # Hardware-specific checks for CUDA/XPU and compute capability are removed
            # as they are PyTorch-specific and not idiomatic for a JAX configuration class.
            # Such checks should be handled by the specific JAX kernel implementation.
            pass

        if self.do_fuse and self.fuse_max_seq_len is None:
            raise ValueError(
                "You cannot enable fused modules without specifying a `fuse_max_seq_len`, make sure to pass a valid `fuse_max_seq_len` for your usecase"
            )

        if self.do_fuse:
            awq_version_supports_fusing = False
            MIN_AWQ_VERSION = "0.1.7"
            if is_auto_awq_available():
                awq_version_supports_fusing = version.parse(importlib.metadata.version("autoawq")) >= version.parse(
                    MIN_AWQ_VERSION
                )

            if not awq_version_supports_fusing:
                raise ValueError(
                    f"You current version of `autoawq` does not support module fusing, please upgrade `autoawq` package to at least {MIN_AWQ_VERSION}."
                )

        if self.modules_to_not_convert is not None:
            awq_version_supports_non_conversion = False
            MIN_AWQ_VERSION = "0.1.8"
            if is_auto_awq_available():
                awq_version_supports_non_conversion = version.parse(
                    importlib.metadata.version("autoawq")
                ) >= version.parse(MIN_AWQ_VERSION)

            if not awq_version_supports_non_conversion:
                raise ValueError(
                    f"You current version of `autoawq` does not support module quantization skipping, please upgrade `autoawq` package to at least {MIN_AWQ_VERSION}."
                )

        if self.do_fuse and self.modules_to_fuse is not None:
            required_keys = [
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "mlp",
                "attention",
                "layernorm",
                "use_alibi",
            ]
            if not all(key in self.modules_to_fuse for key in required_keys):
                raise ValueError(
                    f"Required fields are missing in the fusing mapping, required fields are {required_keys}"
                )

        if self.version == AWQLinearVersion.EXLLAMA:
            awq_version_supports_exllama = False
            MIN_AWQ_VERSION = "0.2.0"
            if is_auto_awq_available():
                awq_version_supports_exllama = version.parse(importlib.metadata.version("autoawq")) >= version.parse(
                    MIN_AWQ_VERSION
                )

            if not awq_version_supports_exllama:
                raise ValueError(
                    f"You current version of `autoawq` does not support exllama backend, "
                    f"please upgrade `autoawq` package to at least {MIN_AWQ_VERSION}."
                )

            if self.exllama_config is None:
                self.exllama_config = {"version": ExllamaVersion.TWO, "max_input_len": 2048, "max_batch_size": 8}
            else:
                if "version" not in self.exllama_config:
                    raise ValueError("`exllama_config` needs to have a `version` key.")
                elif self.exllama_config["version"] not in [ExllamaVersion.ONE, ExllamaVersion.TWO]:
                    exllama_version = self.exllama_config["version"]
                    raise ValueError(
                        f"Only supported versions are in [ExllamaVersion.ONE, ExllamaVersion.TWO] - not recognized version {exllama_version}"
                    )

    def get_loading_attributes(self):
        attibutes_dict = copy.deepcopy(self.__dict__)
        loading_attibutes = ["version", "do_fuse", "modules_to_fuse", "fuse_max_seq_len", "exllama_config"]
        loading_attibutes_dict = {i: j for i, j in attibutes_dict.items() if i in loading_attibutes}
        return loading_attibutes_dict

from __future__ import annotations

import copy
import json
from enum import Enum
from typing import Any, Optional, Union

from absl import logging

# Re-implementation of huggingface/transformers/src/transformers/utils/import_utils.py
# This is a placeholder for a utility function that checks for library availability.
def is_compressed_tensors_available():
  try:
    import compressed_tensors

    return True
  except ImportError:
    return False


# Re-implementation of huggingface/transformers/src/transformers/quantizers/base.py
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


# Re-implementation of huggingface/transformers/src/transformers/quantizers/base.py
class QuantizationConfigMixin:
  """
  Mixin class for quantization config
  """

  quant_method: QuantizationMethod

  @classmethod
  def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
    """
    Instantiates a [`QuantizationConfigMixin`] from a Python dictionary of parameters.

    Args:
        config_dict (`dict[str, Any]`):
            Dictionary that will be used to instantiate the configuration object.
        return_unused_kwargs (`bool`,*optional*, defaults to `False`):
            Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
            `PreTrainedModel`.
        kwargs (`dict[str, Any]`):
            Additional parameters from which to initialize the configuration object.

    Returns:
        [`QuantizationConfigMixin`]: The configuration object instantiated from those parameters.
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

  def to_dict(self) -> dict[str, Any]:
    """
    Serializes this instance to a Python dictionary. Returns:
        `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
    """
    return copy.deepcopy(self.__dict__)


class CompressedTensorsConfig(QuantizationConfigMixin):
  """
  This is a wrapper class that handles compressed-tensors quantization config options.
  It is a wrapper around `compressed_tensors.QuantizationConfig`
  Args:
      config_groups (`typing.dict[str, typing.Union[ForwardRef('QuantizationScheme'), typing.list[str]]]`, *optional*):
          dictionary mapping group name to a quantization scheme definition
      format (`str`, *optional*, defaults to `"dense"`):
          format the model is represented as. Set `run_compressed` True to execute model as the
          compressed format if not `dense`
      quantization_status (`QuantizationStatus`, *optional*, defaults to `"initialized"`):
          status of model in the quantization lifecycle, ie 'initialized', 'calibration', 'frozen'
      kv_cache_scheme (`typing.Union[QuantizationArgs, NoneType]`, *optional*):
          specifies quantization of the kv cache. If None, kv cache is not quantized.
      global_compression_ratio (`typing.Union[float, NoneType]`, *optional*):
          0-1 float percentage of model compression
      ignore (`typing.Union[typing.list[str], NoneType]`, *optional*):
          layer names or types to not quantize, supports regex prefixed by 're:'
      sparsity_config (`typing.dict[str, typing.Any]`, *optional*):
          configuration for sparsity compression
      quant_method (`str`, *optional*, defaults to `"compressed-tensors"`):
          do not override, should be compressed-tensors
      run_compressed (`bool`, *optional*, defaults to `True`): alter submodules (usually linear) in order to
          emulate compressed model execution if True, otherwise use default submodule
  """

  def __init__(
      self,
      config_groups: Optional[dict[str, Union["QuantizationScheme", list[str]]]] = None,  # noqa: F821
      format: str = "dense",
      quantization_status: "QuantizationStatus" = "initialized",  # noqa: F821
      kv_cache_scheme: Optional["QuantizationArgs"] = None,  # noqa: F821
      global_compression_ratio: Optional[float] = None,
      ignore: Optional[list[str]] = None,
      sparsity_config: Optional[dict[str, Any]] = None,
      quant_method: str = "compressed-tensors",
      run_compressed: bool = True,
      **kwargs,
  ):
    if is_compressed_tensors_available():
      from compressed_tensors.config import SparsityCompressionConfig
      from compressed_tensors.quantization import QuantizationConfig
    else:
      raise ImportError(
          "compressed_tensors is not installed and is required for compressed-tensors quantization. Please install it"
          " with `pip install compressed-tensors`."
      )
    self.quantization_config = None
    self.sparsity_config = None

    self.run_compressed = run_compressed

    # parse from dict to load nested QuantizationScheme objects
    if config_groups or kv_cache_scheme:
      self.quantization_config = QuantizationConfig.model_validate(
          {
              "config_groups": config_groups,
              "quant_method": quant_method,
              "format": format,
              "quantization_status": quantization_status,
              "kv_cache_scheme": kv_cache_scheme,
              "global_compression_ratio": global_compression_ratio,
              "ignore": ignore,
              **kwargs,
          }
      )

    if sparsity_config:
      self.sparsity_config = SparsityCompressionConfig.load_from_registry(
          sparsity_config.get("format"), **sparsity_config
      )

    self.quant_method = QuantizationMethod.COMPRESSED_TENSORS

  def post_init(self):
    if self.run_compressed:
      if self.is_sparsification_compressed:
        logging.warning(
            "`run_compressed` is only supported for quantized_compressed models"
            " and not for sparsified models. Setting `run_compressed=False`"
        )
        self.run_compressed = False
      elif not self.is_quantization_compressed:
        logging.warning("`run_compressed` is only supported for compressed models. Setting `run_compressed=False`")
        self.run_compressed = False

  @classmethod
  def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
    """
    Instantiates a [`CompressedTensorsConfig`] from a Python dictionary of parameters.
    Optionally unwraps any args from the nested quantization_config

    Args:
        config_dict (`dict[str, Any]`):
            Dictionary that will be used to instantiate the configuration object.
        return_unused_kwargs (`bool`,*optional*, defaults to `False`):
            Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
            `PreTrainedModel`.
        kwargs (`dict[str, Any]`):
            Additional parameters from which to initialize the configuration object.

    Returns:
        [`QuantizationConfigMixin`]: The configuration object instantiated from those parameters.

    """

    if "quantization_config" in config_dict:
      config_dict = dict(
          sparsity_config=config_dict.get("sparsity_config"),
          **config_dict["quantization_config"],
      )

    return super().from_dict(config_dict, return_unused_kwargs=return_unused_kwargs, **kwargs)

  def to_dict(self) -> dict[str, Any]:
    """
    Quantization config to be added to config.json

    Serializes this instance to a Python dictionary. Returns:
        `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
    """
    quantization_config = {}
    if self.quantization_config is not None:
      quantization_config = self.quantization_config.model_dump()
    else:
      quantization_config["quant_method"] = QuantizationMethod.COMPRESSED_TENSORS

    if self.sparsity_config is not None:
      quantization_config["sparsity_config"] = self.sparsity_config.model_dump()
    else:
      quantization_config["sparsity_config"] = {}

    return quantization_config

  def to_diff_dict(self) -> dict[str, Any]:
    """
    Removes all attributes from config which correspond to the default config attributes for better readability and
    serializes to a Python dictionary.
    Returns:
        `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
    """
    config_dict = self.to_dict()

    # get the default config dict
    default_config_dict = CompressedTensorsConfig().to_dict()

    serializable_config_dict = {}

    # only serialize values that differ from the default config
    for key, value in config_dict.items():
      if key not in default_config_dict or value != default_config_dict[key]:
        serializable_config_dict[key] = value

    return serializable_config_dict

  def get_loading_attributes(self):
    return {"run_compressed": self.run_compressed}

  @property
  def is_quantized(self):
    return bool(self.quantization_config) and bool(self.quantization_config.config_groups)

  @property
  def is_quantization_compressed(self):
    from compressed_tensors.quantization import QuantizationStatus

    return self.is_quantized and self.quantization_config.quantization_status == QuantizationStatus.COMPRESSED

  @property
  def is_sparsification_compressed(self):
    from compressed_tensors.config import (
        CompressionFormat,
        SparsityCompressionConfig,
    )

    return (
        isinstance(self.sparsity_config, SparsityCompressionConfig)
        and self.sparsity_config.format != CompressionFormat.dense.value
    )

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


# Replicated from full file for dependency
class QuantizationMethod(str, Enum):
  """
  The quantization methods supported.
  """

  FBGEMM_FP8 = "fbgemm_fp8"


# Replicated from full file for dependency
class QuantizationConfigMixin:
  """Mixin class for quantization config."""

  pass


@dataclass
class FbgemmFp8Config(QuantizationConfigMixin):
  """
  This is a wrapper class about all possible attributes and features that you can play with a model that has been
  loaded using fbgemm fp8 quantization.

  Attributes:
      activation_scale_ub: The activation scale upper bound. This is used when
        quantizing the input activation.
      modules_to_not_convert: The list of modules to not quantize, useful for
        quantizing models that explicitly require to have some modules left in
        their original precision.
      quant_method: The quantization method, fixed to FBGEMM_FP8 for this
        config.
  """

  activation_scale_ub: float = 1200.0
  modules_to_not_convert: Optional[List[str]] = None
  quant_method: QuantizationMethod = field(init=False)

  def __post_init__(self):
    self.quant_method = QuantizationMethod.FBGEMM_FP8

  def get_loading_attributes(self) -> dict[str, Any]:
    """Returns a dictionary of attributes required for loading a model."""
    return {"activation_scale_ub": self.activation_scale_ub}

import copy
import dataclasses
import importlib.metadata
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from absl import logging
from packaging import version

# MaxText-matched dependencies:
# from ..utils import is_auto_gptq_available, is_gptqmodel_available
# from .quantization_config import QuantizationConfigMixin, QuantizationMethod
# The following are placeholder imports assuming the dependencies are available in the target MaxText environment.
from .quantization_config import QuantizationConfigMixin, QuantizationMethod
from .utils import is_auto_gptq_available, is_gptqmodel_available


class ExllamaVersion(int, Enum):
  ONE = 1
  TWO = 2


@dataclasses.dataclass
class GPTQConfig(QuantizationConfigMixin):
  """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum` api for gptq quantization relying on auto_gptq backend.

    Args:
        bits (`int`):
            The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
        tokenizer (`str` or `PreTrainedTokenizerBase`, *optional*):
            The tokenizer used to process the dataset. You can pass either:
                - A custom tokenizer object.
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        dataset (`Union[list[str]]`, *optional*):
            The dataset used for quantization. You can provide your own dataset in a list of string or just use the
            original datasets used in GPTQ paper ['wikitext2','c4','c4-new']
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        damp_percent (`float`, *optional*, defaults to 0.1):
            The percent of the average Hessian diagonal to use for dampening. Recommended value is 0.1.
        desc_act (`bool`, *optional*, defaults to `False`):
            Whether to quantize columns in order of decreasing activation size. Setting it to False can significantly
            speed up inference but the perplexity may become slightly worse. Also known as act-order.
        sym (`bool`, *optional*, defaults to `True`):
            Whether to use symmetric quantization.
        true_sequential (`bool`, *optional*, defaults to `True`):
            Whether to perform sequential quantization even within a single Transformer block. Instead of quantizing
            the entire block at once, we perform layer-wise quantization. As a result, each layer undergoes
            quantization using inputs that have passed through the previously quantized layers.
        checkpoint_format (`str`, *optional*, defaults to `"gptq"`):
            GPTQ weight format. `gptq`(v1) is supported by both gptqmodel and auto-gptq. `gptq_v2` is gptqmodel only.
        meta (`dict[str, any]`, *optional*):
            Properties, such as tooling:version, that do not directly contributes to quantization or quant inference are stored in meta.
            i.e. `meta.quantizer`: ["optimum:_version_", "gptqmodel:_version_"]
        backend (`str`, *optional*):
            Controls which gptq kernel to be used. Valid values for gptqmodel are `auto`, `auto_trainable` and more. For auto-gptq, only
            valid value is None and `auto_trainable`. Ref gptqmodel backends: https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/utils/backend.py
        use_cuda_fp16 (`bool`, *optional*, defaults to `False`):
            Whether or not to use optimized cuda kernel for fp16 model. Need to have model in fp16. Auto-gptq only.
        model_seqlen (`int`, *optional*):
            The maximum sequence length that the model can take.
        block_name_to_quantize (`str`, *optional*):
            The transformers block name to quantize. If None, we will infer the block name using common patterns (e.g. model.layers)
        module_name_preceding_first_block (`list[str]`, *optional*):
            The layers that are preceding the first Transformer block.
        batch_size (`int`, *optional*, defaults to 1):
            The batch size used when processing the dataset
        pad_token_id (`int`, *optional*):
            The pad token id. Needed to prepare the dataset when `batch_size` > 1.
        use_exllama (`bool`, *optional*):
            Whether to use exllama backend. Defaults to `True` if unset. Only works with `bits` = 4.
        max_input_length (`int`, *optional*):
            The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input
            length. It is specific to the exllama backend with act-order.
        exllama_config (`dict[str, Any]`, *optional*):
            The exllama config. You can specify the version of the exllama kernel through the `version` key. Defaults
            to `{"version": 1}` if unset.
        cache_block_outputs (`bool`, *optional*, defaults to `True`):
            Whether to cache block outputs to reuse as inputs for the succeeding block.
        modules_in_block_to_quantize (`list[list[str]]`, *optional*):
            List of list of module names to quantize in the specified block. This argument is useful to exclude certain linear modules from being quantized.
            The block to quantize can be specified by setting `block_name_to_quantize`. We will quantize each list sequentially. If not set, we will quantize all linear layers.
            Example: `modules_in_block_to_quantize =[["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"]]`.
            In this example, we will first quantize the q,k,v layers simultaneously since they are independent.
            Then, we will quantize `self_attn.o_proj` layer with the q,k,v layers quantized. This way, we will get
            better results since it reflects the real input `self_attn.o_proj` will get when the model is quantized.
    """

  bits: int
  tokenizer: Any = None
  dataset: Optional[Union[List[str], str]] = None
  group_size: int = 128
  damp_percent: float = 0.1
  desc_act: bool = False
  sym: bool = True
  true_sequential: bool = True
  checkpoint_format: str = "gptq"
  meta: Optional[Dict[str, Any]] = None
  backend: Optional[str] = None
  use_cuda_fp16: bool = False
  model_seqlen: Optional[int] = None
  block_name_to_quantize: Optional[str] = None
  module_name_preceding_first_block: Optional[List[str]] = None
  batch_size: int = 1
  pad_token_id: Optional[int] = None
  use_exllama: Optional[bool] = None
  max_input_length: Optional[int] = None
  exllama_config: Optional[Dict[str, Any]] = None
  cache_block_outputs: bool = True
  modules_in_block_to_quantize: Optional[List[List[str]]] = None
  quant_method: QuantizationMethod = dataclasses.field(default=QuantizationMethod.GPTQ, init=False)

  def __post_init__(self):
    self.checkpoint_format = self.checkpoint_format.lower()
    if isinstance(self.backend, str):
      self.backend = self.backend.lower()

    r"""
        Safety checker that arguments are correct
        """
    if self.bits not in [2, 3, 4, 8]:
      raise ValueError(f"Only support quantization to [2,3,4,8] bits but found {self.bits}")
    if self.group_size != -1 and self.group_size <= 0:
      raise ValueError("group_size must be greater than 0 or equal to -1")
    if not (0 < self.damp_percent < 1):
      raise ValueError("damp_percent must between 0 and 1.")
    if self.dataset is not None:
      if isinstance(self.dataset, str):
        if self.dataset in ["ptb", "ptb-new"]:
          raise ValueError(
              f"""{self.dataset} dataset was deprecated. You can only choose between
                        ['wikitext2','c4','c4-new']"""
          )
        if self.dataset not in ["wikitext2", "c4", "c4-new"]:
          raise ValueError(
              f"""You have entered a string value for dataset. You can only choose between
                        ['wikitext2','c4','c4-new'], but we found {self.dataset}"""
          )
      elif not isinstance(self.dataset, list):
        raise ValueError(
            f"""dataset needs to be either a list of string or a value in
                    ['wikitext2','c4','c4-new'], but we found {self.dataset}"""
        )

    # make sure backend is back/forward compatible with both gptqmodel (full) and auto-gptq (partial)
    if is_gptqmodel_available():
      # convert auto-gptq control into gptqmodel backend
      if self.backend is None:
        self.backend = "auto_trainable" if self.use_exllama is not None and not self.use_exllama else "auto"
    else:
      # convert gptqmodel backend `auto_trainable` into auto-gptq control
      if self.backend == "auto_trainable":
        self.use_exllama = False

    # auto-gptq specific kernel control logic
    if self.use_exllama is None:
      # New default behaviour
      self.use_exllama = True

    if self.exllama_config is None:
      self.exllama_config = {"version": ExllamaVersion.ONE}
    else:
      if "version" not in self.exllama_config:
        raise ValueError("`exllama_config` needs to have a `version` key.")
      elif self.exllama_config["version"] not in [ExllamaVersion.ONE, ExllamaVersion.TWO]:
        exllama_version = self.exllama_config["version"]
        raise ValueError(
            f"Only supported versions are in [ExllamaVersion.ONE, ExllamaVersion.TWO] - not recognized version {exllama_version}"
        )

    if self.bits == 4 and self.use_exllama:
      if self.exllama_config["version"] == ExllamaVersion.ONE:
        logging.info(
            "You have activated exllama backend. Note that you can get better inference "
            "speed using exllamav2 kernel by setting `exllama_config`."
        )
      elif self.exllama_config["version"] == ExllamaVersion.TWO:
        if is_auto_gptq_available():
          optimum_version = version.parse(importlib.metadata.version("optimum"))
          autogptq_version = version.parse(importlib.metadata.version("auto_gptq"))
          if optimum_version <= version.parse("1.13.2") or autogptq_version <= version.parse("0.4.2"):
            raise ValueError(
                "You need optimum > 1.13.2 and auto-gptq > 0.4.2 . Make sure to have that version installed - detected"
                f" version : optimum {optimum_version} and autogptq {autogptq_version}"
            )
    if self.modules_in_block_to_quantize is not None:
      optimum_version = version.parse(importlib.metadata.version("optimum"))
      if optimum_version < version.parse("1.15.0"):
        raise ValueError(
            "You current version of `optimum` does not support `modules_in_block_to_quantize` quantization argument,"
            " please upgrade `optimum` package to a version superior than 1.15.0 ."
        )

  def get_loading_attributes(self):
    attibutes_dict = dataclasses.asdict(self)
    loading_attibutes = [
        "use_exllama",
        "exllama_config",
        "use_cuda_fp16",
        "max_input_length",
        "backend",
    ]
    loading_attibutes_dict = {i: j for i, j in attibutes_dict.items() if i in loading_attibutes}
    return loading_attibutes_dict

  def to_dict(self):
    config_dict = super().to_dict()
    config_dict.pop("disable_exllama", None)
    return config_dict

  def to_dict_optimum(self):
    """
    Get compatible dict for optimum gptq config
    """
    quant_dict = self.to_dict()
    # make it compatible with optimum config
    quant_dict["disable_exllama"] = not self.use_exllama
    return quant_dict

  @classmethod
  def from_dict_optimum(cls, config_dict):
    """
    Get compatible class with optimum gptq config dict
    """
    config_dict = config_dict.copy()
    if "disable_exllama" in config_dict:
      config_dict["use_exllama"] = not config_dict["disable_exllama"]
      # switch to None to not trigger the warning
      config_dict.pop("disable_exllama")

    known_keys = {f.name for f in dataclasses.fields(cls)}
    filtered_config_dict = {k: v for k, v in config_dict.items() if k in known_keys}

    config = cls(**filtered_config_dict)
    return config

from typing import Optional

# Re-used from generated_code.Qwen3MoeForCausalLM.quantization.QuantizationConfigMixin
from .quantization import QuantizationConfigMixin
# Re-used from generated_code.Qwen3MoeForCausalLM.quantization.QuantizationMethod
from .quantization import QuantizationMethod


class Mxfp4Config(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using mxfp4 quantization.

    Args:
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision.
    """

    def __init__(
        self,
        modules_to_not_convert: Optional[list] = None,
        dequantize: bool = False,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.MXFP4
        self.modules_to_not_convert = modules_to_not_convert
        self.dequantize = dequantize

    def get_loading_attributes(self):
        return {
            "dequantize": self.dequantize,
        }
from MaxText.layers.quantization_configs import (
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
    QuantoConfig,
    QuarkConfig,
    SpQRConfig,
    TorchAoConfig,
    VptqConfig,
)


AUTO_QUANTIZATION_CONFIG_MAPPING = {
    "awq": AwqConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.AwqConfig
    "bitsandbytes_4bit": BitsAndBytesConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.BitsAndBytesConfig
    "bitsandbytes_8bit": BitsAndBytesConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.BitsAndBytesConfig
    "eetq": EetqConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.EetqConfig
    "gptq": GPTQConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.GPTQConfig
    "aqlm": AqlmConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization.AqlmConfig
    "quanto": QuantoConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.QuantoConfig
    "quark": QuarkConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.QuarkConfig
    "fp_quant": FPQuantConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.FPQuantConfig
    "hqq": HqqConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.HqqConfig
    "compressed-tensors": CompressedTensorsConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.CompressedTensorsConfig
    "fbgemm_fp8": FbgemmFp8Config,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.FbgemmFp8Config
    "higgs": HiggsConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.HiggsConfig
    "torchao": TorchAoConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.TorchAoConfig
    "bitnet": BitNetQuantConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.BitNetQuantConfig
    "vptq": VptqConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.VptqConfig
    "spqr": SpQRConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.SpQRConfig
    "fp8": FineGrainedFP8Config,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.FineGrainedFP8Config
    "auto-round": AutoRoundConfig,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.AutoRoundConfig
    "mxfp4": Mxfp4Config,  # Reused from generated_code.Qwen3MoeForCausalLM.quantization_configs.Mxfp4Config
}
from ..models.auto.configuration_auto import AutoConfig
from ..utils.quantization_config import QuantizationMethod

# The following mappings are assumed to be defined in the same file in the JAX/MaxText version,
# similar to the PyTorch source file.
# from . import AUTO_QUANTIZATION_CONFIG_MAPPING, AUTO_QUANTIZER_MAPPING


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

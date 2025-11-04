
# coding=utf-8
# Copyright 2024 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Flax modeling file."""

import collections
import copy
import functools
import gc
import importlib.metadata
import inspect
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
from abc import abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from enum import Enum
from functools import partial, wraps
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax.linen import partitioning as nn_partitioning
from flax.traverse_util import flatten_dict, unflatten_dict
from huggingface_hub import split_torch_state_dict_into_shards
from packaging import version

# Re-used from src.MaxText.layers.linears.MlpBlock
from src.MaxText.layers.linears import MlpBlock
# Re-used from src.MaxText.layers.quantizations
from src.MaxText.layers import quantizations
# Re-used from src.MaxText.layers.quantizations.AqtQuantization
from src.MaxText.layers.quantizations import AqtQuantization
# Re-used from src.MaxText.layers.quantizations.Fp8Quantization
from src.MaxText.layers.quantizations import Fp8Quantization
# Re-used from src.MaxText.layers.quantizations.Quantization
from src.MaxText.layers.quantizations import Quantization
# Re-used from generated_code.Qwen3MoeForCausalLM.quantization.QuantizationMethod
from generated_code.Qwen3MoeForCausalLM.quantization import QuantizationMethod
# Re-used from generated_code.Qwen3MoeForCausalLM.quantization_configs.BitsAndBytesConfig
from generated_code.Qwen3MoeForCausalLM.quantization_configs import (
    BitsAndBytesConfig,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.quantization_configs.QuantizationConfigMixin
from generated_code.Qwen3MoeForCausalLM.quantization_configs import (
    QuantizationConfigMixin,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.configuration
from generated_code.Qwen3MoeForCausalLM.configuration import PretrainedConfig
# Re-used from generated_code.Qwen3MoeForCausalLM.distributed_utils
from generated_code.Qwen3MoeForCausalLM.distributed_utils import (
    DistributedConfig,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.dynamic_module_utils
from generated_code.Qwen3MoeForCausalLM.dynamic_module_utils import (
    custom_object_save,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.generation
from generated_code.Qwen3MoeForCausalLM.generation import (
    CompileConfig,
    GenerationConfig,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.distributed_utils
from generated_code.Qwen3MoeForCausalLM.distributed_utils import (
    is_deepspeed_zero3_enabled,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.distributed_utils
from generated_code.Qwen3MoeForCausalLM.distributed_utils import (
    _get_parameter_tp_plan,
    distribute_model,
    initialize_tensor_parallelism,
    repack_weights,
    replace_state_dict_local_with_dtensor,
    shard_and_distribute_module,
    verify_tp_plan,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.losses
from generated_code.Qwen3MoeForCausalLM.losses import LOSS_MAPPING
# Re-used from generated_code.Qwen3MoeForCausalLM.attention_utils
from generated_code.Qwen3MoeForCausalLM.attention_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.attention_utils
from generated_code.Qwen3MoeForCausalLM.attention_utils import (
    lazy_import_flash_attention,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.quantization
from generated_code.Qwen3MoeForCausalLM.quantization import AutoHfQuantizer, HfQuantizer
# Re-used from generated_code.Qwen3MoeForCausalLM.dynamic_module_utils
from generated_code.Qwen3MoeForCausalLM.dynamic_module_utils import (
    get_module_from_name,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.hub_utils
from generated_code.Qwen3MoeForCausalLM.hub_utils import auto_conversion
# Re-used from generated_code.Qwen3MoeForCausalLM.constants
from generated_code.Qwen3MoeForCausalLM.constants import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    DUMMY_INPUTS,
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.hub_utils
from generated_code.Qwen3MoeForCausalLM.hub_utils import PushToHubMixin
# Re-used from generated_code.Qwen3MoeForCausalLM.hub_utils
from generated_code.Qwen3MoeForCausalLM.hub_utils import (
    cached_file,
    copy_func,
    download_url,
    extract_commit_hash,
    has_file,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.package_utils
from generated_code.Qwen3MoeForCausalLM.package_utils import (
    is_accelerate_available,
    is_bitsandbytes_available,
    is_flash_attn_2_available,
    is_flash_attn_3_available,
    is_kernels_available,
    is_offline_mode,
    is_optimum_available,
    is_peft_available,
    is_remote_url,
    is_safetensors_available,
    is_torch_flex_attn_available,
    is_torch_greater_or_equal,
    is_torch_mlu_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import logging
# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import strtobool
# Re-used from generated_code.Qwen3MoeForCausalLM.hub_utils
from generated_code.Qwen3MoeForCausalLM.hub_utils import (
    create_and_tag_model_card,
    get_checkpoint_shard_files,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.import_utils
from generated_code.Qwen3MoeForCausalLM.import_utils import (
    ENV_VARS_TRUE_VALUES,
    is_huggingface_hub_greater_or_equal,
    is_sagemaker_mp_enabled,
    is_torch_fx_proxy,
    is_torchdynamo_compiling,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.constants
from generated_code.Qwen3MoeForCausalLM.constants import (
    XLA_DOWNCAST_BF16,
    XLA_USE_BF16,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.distributed_utils
from generated_code.Qwen3MoeForCausalLM.distributed_utils import (
    is_fsdp_enabled,
    is_local_dist_rank_0,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.peft
from generated_code.Qwen3MoeForCausalLM.peft import find_adapter_config_file
# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import (
    SpecificPreTrainedModelType,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.constants
from generated_code.Qwen3MoeForCausalLM.constants import VLMS
# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import (
    no_init_weights,
    set_quantized_state,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.distributed_utils
from generated_code.Qwen3MoeForCausalLM.distributed_utils import set_zero3_state
# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import (
    get_parameter_device,
    get_parameter_dtype,
    get_state_dict_dtype,
    load_sharded_checkpoint,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.tensor_utils
from generated_code.Qwen3MoeForCausalLM.tensor_utils import str_to_jnp_dtype
# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import load_state_dict
# Re-used from generated_code.Qwen3MoeForCausalLM.tensor_utils
from generated_code.Qwen3MoeForCausalLM.tensor_utils import (
    _end_ptr,
    _find_disjoint,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import (
    _get_tied_weight_keys,
    _find_identical,
    _infer_parameter_dtype,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.dynamic_module_utils
from generated_code.Qwen3MoeForCausalLM.dynamic_module_utils import (
    _load_parameter_into_model,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import (
    _load_state_dict_into_meta_model,
    load_shard_file,
    load_shard_files_with_threadpool,
    _add_variant,
    _get_resolved_checkpoint_files,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import (
    _find_missing_and_unexpected_keys,
    _find_mismatched_keys,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import ModuleUtilsMixin
# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import EmbeddingAccessMixin
# Re-used from generated_code.Qwen3MoeForCausalLM.peft
from generated_code.Qwen3MoeForCausalLM.peft import PeftAdapterMixin
# Re-used from generated_code.Qwen3MoeForCausalLM.attention_utils
from generated_code.Qwen3MoeForCausalLM.attention_utils import (
    flash_attention_forward,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import unwrap_model
# Re-used from generated_code.Qwen3MoeForCausalLM.distributed_utils
from generated_code.Qwen3MoeForCausalLM.distributed_utils import (
    is_accelerator_device,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.model_utils
from generated_code.Qwen3MoeForCausalLM.model_utils import (
    get_disk_only_shard_files,
)
# Re-used from generated_code.Qwen3MoeForCausalLM.attention_utils
from generated_code.Qwen3MoeForCausalLM.attention_utils import (
    AttentionInterface,
    ALL_ATTENTION_FUNCTIONS,
)

if is_accelerate_available():
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.hooks import add_hook_to_module
    from accelerate.utils import (
        check_tied_parameters_on_same_device,
        extract_model_from_parallel,
        get_balanced_memory,
        get_max_memory,
        load_offloaded_weights,
        offload_weight,
        save_offload_index,
    )

    accelerate_version = version.parse(importlib.metadata.version("accelerate"))
    if accelerate_version >= version.parse("0.31"):
        from accelerate.utils.modeling import get_state_dict_from_offload

if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.flax import load_file as safe_load_file
    from safetensors.flax import save_file as safe_save_file


if is_kernels_available():
    from kernels import get_kernel


logger = logging.get_logger(__name__)


_init_weights = True
_is_quantized = False
_is_ds_init_called = False
# Re-used from generated_code.Qwen3MoeForCausalLM.distributed_utils._jax_distributed_available
_torch_distributed_available = _jax_distributed_available

# Re-used from generated_code.Qwen3MoeForCausalLM.distributed_utils._is_dtensor_available
_is_dtensor_available = _is_dtensor_available
if _is_dtensor_available:
    from torch.distributed.tensor import DTensor


def restore_default_torch_dtype(func):
    """
    Decorator to restore the default jax dtype
    at the end of the function. Serves
    as a backup in case calling the function raises
    an error after the function has changed the default dtype but before it could restore it.
    """

    @wraps(func)
    def _wrapper(*args, **kwargs):
        old_dtype = jax.config.jax_default_dtype
        try:
            return func(*args, **kwargs)
        finally:
            jax.config.update("jax_default_dtype", old_dtype)

    return _wrapper


def get_torch_context_manager_or_global_device():
    """
    In JAX, device management is typically handled by the JAX runtime itself
    or explicitly through `jax.device_put`. This function is kept for API
    compatibility but is a no-op in the JAX context.
    """
    return None


def set_initialized_submodules(model, state_dict_keys):
    """
    Sets the `_is_hf_initialized` flag in all submodules of a given model when all its weights are in the loaded state
    dict.
    """
    state_dict_keys = set(state_dict_keys)
    not_initialized_submodules = {}
    # In Flax, we don't have a direct equivalent of named_modules that includes the root.
    # We can iterate through attributes. This is a simplified version.
    for module_name, module in model.named_children():
        if not isinstance(module, nn.Module):
            continue

        try:
            # This is an approximation. Getting a module's state_dict in Flax
            # requires the params tree, which we don't have here.
            # We assume if any key starts with the module name, it's relevant.
            module_keys = {
                k for k in state_dict_keys if k.startswith(f"{module_name}.")
            }
            # This check is difficult and not very reliable in Flax without params.
            # We'll assume for now that if we find keys, it's initialized.
            if len(module_keys) > 0:
                module._is_hf_initialized = True
            else:
                not_initialized_submodules[module_name] = module
        except Exception:
            not_initialized_submodules[module_name] = module

    # Check the root model itself
    root_keys = {k for k in state_dict_keys if "." not in k}
    if len(root_keys) > 0:
        model._is_hf_initialized = True
    else:
        not_initialized_submodules[""] = model

    return not_initialized_submodules


class PipelineParallel(Enum):
    inputs: 0
    outputs: 1


class PreTrainedAudioTokenizerBase(PreTrainedModel):
    """
    Class that additionally defines the behavior of any `audio_tokenizer` to be added.
    Characteristic for any of them:
        1. Encode raw audio into discrete audio codebooks (with x channels)
        2. Decode from discrete audio codebooks back to raw audio
    It is possible that they can decode in different ways given a different representation
    but they are forced to support 2. nonetheless, e.g. see `DAC`.
    """

    @abstractmethod
    def encode(self, input_values: jnp.ndarray, *args, **kwargs):
        """
        Encode raw audio retrieved from a respective `FeatureExtractor` into discrete audio codebooks (with x channels)
        """
        pass

    @abstractmethod
    def decode(self, audio_codes: jnp.ndarray, *args, **kwargs):
        """Decode from discrete audio codebooks back to raw audio"""
        pass


class FlaxPreTrainedModel(
    nn.Module,
    EmbeddingAccessMixin,
    ModuleUtilsMixin,
    PushToHubMixin,
    PeftAdapterMixin,
):
    r"""
    Base class for all models.

    [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings,
        - prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **load_tf_weights** (`Callable`) -- A python *method* for loading a TensorFlow checkpoint in a PyTorch model,
          taking as arguments:

            - **model** ([`PreTrainedModel`]) -- An instance of the model on which to load the TensorFlow checkpoint.
            - **config** ([`PretrainedConfig`]) -- An instance of the configuration associated to the model.
            - **path** (`str`) -- A path to the TensorFlow checkpoint.

        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **is_parallelizable** (`bool`) -- A flag indicating whether this model supports model parallelization.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
        - **can_record_outputs** (dict):"""

    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    model_tags = None

    _checkpoint_conversion_mapping = {}  # used for BC support in VLMs, not meant to be used by new models

    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None

    _keep_in_fp32_modules = None
    # the _keep_in_fp32_modules will avoid casting to anything other than float32, except bfloat16
    # to also prevent bfloat16 casting, use the _keep_in_fp32_modules_strict flag
    _keep_in_fp32_modules_strict = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None

    is_parallelizable = False
    supports_gradient_checkpointing = False
    _is_stateful = False

    # Flash Attention support
    _supports_flash_attn = False

    # SDPA support
    _supports_sdpa = False

    # Flex Attention support
    _supports_flex_attn = False

    _can_compile_fullgraph = False

    # A tensor parallel plan to be applied to the model when TP is enabled. For
    # top-level models, this attribute is currently defined in respective model
    # code. For base models, this attribute comes from
    # `config.base_model_tp_plan` during `__init__`.
    # It should identify the layers exactly: if you want to TP model.language_model.layers.fc1
    # by passing `tp_plan` to the init, it should be {"model.language_model.layers.fc1":"colwise"}
    # for example.
    _tp_plan = None

    # tensor parallel degree to which model is sharded to.
    _tp_size = None

    # A pipeline parallel plan specifying the layers which may not be present
    # on all ranks when PP is enabled. For top-level models, this attribute is
    # currently defined in respective model code. For base models, this
    # attribute comes from `config.base_model_pp_plan` during `post_init`.
    #
    # The variable names for the inputs and outputs of the specified layers can
    # be indexed using the `PipelineParallel` enum as follows:
    # - `_pp_plan["layers"][PipelineParallel.inputs]`
    # - `_pp_plan["layers"][PipelineParallel.outputs]`
    _pp_plan = None

    # This flag signal that the model can be used as an efficient backend in TGI and vLLM
    # In practice, it means that they support attention (mask) interface functions, fully pass the kwargs
    # through all modules up to the Attention layer, can slice logits with Tensor, and have a default TP plan
    _supports_attention_backend = False
    _can_record_outputs = None

    @property
    def dummy_inputs(self) -> Dict[str, jnp.ndarray]:
        """
        `dict[str, jnp.ndarray]`: Dummy inputs to do a forward pass in the network.
        """
        return {"input_ids": jnp.array(DUMMY_INPUTS, dtype=jnp.int32)}

    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a Flax model.
        """
        return "flax"

    def __init__(
        self,
        config: PretrainedConfig,
        *inputs,
        dtype: jnp.dtype = jnp.float32,
        **kwargs,
    ):
        super().__init__(dtype=dtype)
        if not isinstance(config, PretrainedConfig):
            raise TypeError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config
        self.dtype = dtype

        # Check the attention implementation is supported, or set it if not yet set (on the internal attr, to avoid
        # setting it recursively)
        self.config._attn_implementation_internal = self._check_and_adjust_attn_implementation(
            self.config._attn_implementation, is_init_check=True
        )

        # for initialization of the loss
        loss_type = self.__class__.__name__
        if loss_type not in LOSS_MAPPING:
            loss_groups = f"({'|'.join(LOSS_MAPPING)})"
            loss_type = re.findall(loss_groups, self.__class__.__name__)
            if len(loss_type) > 0:
                loss_type = loss_type[0]
            else:
                loss_type = None
        self.loss_type = loss_type

        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        self.generation_config = (
            GenerationConfig.from_model_config(config)
            if self.can_generate()
            else None
        )
        # Overwrite the class attribute to make it an instance attribute, so models like
        # `InstructBlipForConditionalGeneration` can dynamically update it without modifying the class attribute
        # when a different component (e.g. language_model) is used.
        self._keep_in_fp32_modules = copy.copy(self.__class__._keep_in_fp32_modules)
        self._keep_in_fp32_modules_strict = copy.copy(
            self.__class__._keep_in_fp32_modules_strict
        )

        self._no_split_modules = self._no_split_modules or []
        _CAN_RECORD_REGISTRY[str(self.__class__)] = self._can_record_outputs

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).

        This is also used when the user is running distributed code. We add hooks to the modules here, according to
        the model's tp_plan!
        """
        # In Flax, initialization happens during model.init(), so this is a placeholder.
        # The logic from PyTorch's init_weights and _backward_compatibility_gradient_checkpointing
        # will be handled differently in the JAX ecosystem.
        pass

    def dequantize(self):
        """
        Potentially dequantize the model in case it has been quantized by a quantization method that support
        dequantization.
        """
        hf_quantizer = getattr(self, "hf_quantizer", None)

        if hf_quantizer is None:
            raise ValueError(
                "You need to first quantize your model in order to dequantize it"
            )

        return hf_quantizer.dequantize(self)

    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(
            self.config, "gradient_checkpointing", False
        ):
            self.gradient_checkpointing_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "gradient_checkpointing")

    def add_model_tags(self, tags: Union[List[str], str]) -> None:
        r"""
        Add custom tags into the model that gets pushed to the Hugging Face Hub. Will
        not overwrite existing tags in the model.

        Args:
            tags (`Union[list[str], str]`):
                The desired tags to inject in the model

        Examples:

        
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence

from MaxText.common_types import Array, Config
from MaxText.layers import linears


class SequentialLlama4TextExperts(nn.Module):
  """A module that implements a compressed version of a list of expert modules.

  This is specifically designed to work with Llama4TextExperts in MoE layers.
  """

  config: Config

  def setup(self):
    """Initializes the expert modules."""
    self.num_experts = self.config.num_local_experts

    # The original PyTorch code uses Llama4TextMLP, which is a gated MLP (SwiGLU)
    # with no dropout. The MaxText mlp_block can replicate this.
    # We assume config.mlp_activations is set appropriately (e.g., ('silu', 'linear')).
    mlp_activations = self.config.mlp_activations
    if not isinstance(mlp_activations, Sequence):
      mlp_activations = (mlp_activations,)

    self.experts = [
        # Reused from src.MaxText.layers.linears.mlp_block
        linears.mlp_block(
            config=self.config,
            in_features=self.config.hidden_size,
            intermediate_dim=self.config.intermediate_size,
            activations=mlp_activations,
            name=f"expert_{i}",
        )
        for i in range(self.num_experts)
    ]

  def __call__(self, hidden_states: Array) -> Array:
    """Forward pass for the sequential experts.

    Args:
      hidden_states: Input tensor.

    Returns:
      The output of the experts.
    """
    hidden_states = hidden_states.reshape(self.num_experts, -1, hidden_states.shape[-1])

    # Apply each expert to its corresponding slice of the input.
    # The original Llama4TextMLP has no dropout, so we pass deterministic=True.
    expert_outputs = [self.experts[i](hidden_states[i], deterministic=True) for i in range(self.num_experts)]

    # Stack the results back into a single tensor.
    routed_out = jnp.stack(expert_outputs, axis=0)
    return routed_out

# Copyright 2024 The MaxText Authors.
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
This file is a JAX/Flax adaptation of HuggingFace's `transformers.modeling_utils.PreTrainedModel`.
The original source code can be found at:
https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py
"""

from typing import Callable, Dict

import flax.linen as nn

# from .. import PretrainedConfig # Path: src/MaxText/pyconfig.py
# from .hub_utils import PushToHubMixin # Path: src/MaxText/hub_utils.py
# from .model_utils import EmbeddingAccessMixin, ModuleUtilsMixin, OutputRecorder # Path: src/MaxText/layers/models.py
# from .peft import PeftAdapterMixin # Path: src/MaxText/peft.py


class FlaxPreTrainedModel(nn.Module, EmbeddingAccessMixin, ModuleUtilsMixin, PushToHubMixin, PeftAdapterMixin):
    r"""
    Base class for all models.

    [`FlaxPreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models as well as a few methods common to all models to:

        - resize the input embeddings,
        - prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **load_tf_weights** (`Callable`) -- A python *method* for loading a TensorFlow checkpoint in a JAX model,
          taking as arguments:

            - **model** ([`FlaxPreTrainedModel`]) -- An instance of the model on which to load the TensorFlow checkpoint.
            - **config** ([`PretrainedConfig`]) -- An instance of the configuration associated to the model.
            - **path** (`str`) -- A path to the TensorFlow checkpoint.

        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **is_parallelizable** (`bool`) -- A flag indicating whether this model supports model parallelization.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
        - **can_record_outputs** (dict):"""

    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    model_tags = None

    _checkpoint_conversion_mapping = {}  # used for BC support in VLMs, not meant to be used by new models

    _auto_class = None
    _no_split_modules = None
    _skip_keys_device_placement = None

    _keep_in_fp32_modules = None
    # the _keep_in_fp32_modules will avoid casting to anything other than float32, except bfloat16
    # to also prevent bfloat16 casting, use the _keep_in_fp32_modules_strict flag
    _keep_in_fp32_modules_strict = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None
    # a list of `state_dict` keys that are potentially tied to another key in the state_dict.
    _tied_weights_keys = None

    is_parallelizable = False
    supports_gradient_checkpointing = False
    _is_stateful = False

    # Flash Attention support
    _supports_flash_attn = False

    # SDPA support
    _supports_sdpa = False

    # Flex Attention support
    _supports_flex_attn = False

    _can_compile_fullgraph = False

    # A tensor parallel plan to be applied to the model when TP is enabled. For
    # top-level models, this attribute is currently defined in respective model
    # code. For base models, this attribute comes from
    # `config.base_model_tp_plan` during `__init__`.
    # It should identify the layers exactly: if you want to TP model.language_model.layers.fc1
    # by passing `tp_plan` to the init, it should be {"model.language_model.layers.fc1":"colwise"}
    # for example.
    _tp_plan = None

    # tensor parallel degree to which model is sharded to.
    _tp_size = None

    # A pipeline parallel plan specifying the layers which may not be present
    # on all ranks when PP is enabled. For top-level models, this attribute is
    # currently defined in respective model code. For base models, this
    # attribute comes from `config.base_model_pp_plan` during `post_init`.
    #
    # The variable names for the inputs and outputs of the specified layers can
    # be indexed using the `PipelineParallel` enum as follows:
    # - `_pp_plan["layers"][PipelineParallel.inputs]`
    # - `_pp_plan["layers"][PipelineParallel.outputs]`
    _pp_plan = None

    # This flag signal that the model can be used as an efficient backend in TGI and vLLM
    # In practice, it means that they support attention (mask) interface functions, fully pass the kwargs
    # through all modules up to the Attention layer, can slice logits with Tensor, and have a default TP plan
    _supports_attention_backend = False
    _can_record_outputs = None

    @property
    def can_record_outputs(self) -> Dict[str, "OutputRecorder"]:
        """
        Maps output names (e.g., "attentions", "hidden_states")
        to either:
            - A module class (e.g., `LlamaDecoderLayer`), using default index conventions:
                * index=0 for "hidden_states"
                * index=1 for "attentions"
            - Or an `OutputRecorder(...)` with `target_class`, optional `index`, and `layer_name`.

        Examples:
            These two are equivalent:

        
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
from typing import Type

import flax.linen as nn

from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import auto_docstring
from .configuration_qwen3_moe import Qwen3MoeConfig


@auto_docstring
class FlaxQwen3MoePreTrainedModel(FlaxPreTrainedModel):
    config_class = Qwen3MoeConfig
    base_model_prefix: str = "model"
    module_class: Type[nn.Module] = None


from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from ..layers import embeddings
# MaxText modules used:
# From src.MaxText.layers.decoders: DecoderLayer (via MAXTEXT_MATCHED_DEPENDENCIES)
# From src.MaxText.layers.normalizations: rms_norm
from ..layers import normalizations
from .configuration_qwen3_moe import Qwen3MoeConfig
from .modeling_flax_qwen3_moe_decoder_layer import Qwen3MoeDecoderLayer
from .modeling_flax_qwen3_moe_rotary_embedding import Qwen3MoeRotaryEmbedding


@dataclass
class MoeModelOutputWithPast:
  """
    Base class for model's outputs that may also contain a past key/value states.

    Args:
        last_hidden_state (`jax.Array` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(jax.Array))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(jax.Array)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
  """

  last_hidden_state: jax.Array
  past_key_values: Optional[Tuple[Tuple[jax.Array, jax.Array]]] = None


class Qwen3MoeModel(nn.Module):
  config: Qwen3MoeConfig
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.vocab_size = self.config.vocab_size
    self.embed_tokens = nn.Embed(
        num_embeddings=self.config.vocab_size,
        features=self.config.hidden_size,
        embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        dtype=self.dtype,
    )
    self.layers = [
        Qwen3MoeDecoderLayer(config=self.config, name=str(i), dtype=self.dtype)
        for i in range(self.config.num_hidden_layers)
    ]
    self.norm = normalizations.RMSNorm(
        self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, name="norm"
    )
    self.rotary_emb = Qwen3MoeRotaryEmbedding(config=self.config, dtype=self.dtype)

  def __call__(
      self,
      input_ids: Optional[jnp.ndarray] = None,
      attention_mask: Optional[jnp.ndarray] = None,
      position_ids: Optional[jnp.ndarray] = None,
      past_key_values: Optional[Tuple[Tuple[jax.Array, jax.Array]]] = None,
      inputs_embeds: Optional[jnp.ndarray] = None,
      use_cache: Optional[bool] = None,
      deterministic: bool = True,
  ) -> MoeModelOutputWithPast:
    if (input_ids is None) and (inputs_embeds is None):
      raise ValueError("You must specify either input_ids or inputs_embeds")

    if inputs_embeds is None:
      inputs_embeds = self.embed_tokens(input_ids.astype("i4"))

    past_key_values_length = 0
    if past_key_values is not None:
      past_key_values_length = past_key_values[0][0].shape[2]

    if position_ids is None:
      cache_position = jnp.arange(
          past_key_values_length,
          past_key_values_length + inputs_embeds.shape[1],
          dtype=jnp.int32,
      )
      position_ids = jnp.expand_dims(cache_position, axis=0)

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    next_past_key_values = [] if use_cache else None

    for i, decoder_layer in enumerate(self.layers):
      layer_past_key_values = past_key_values[i] if past_key_values is not None else None

      layer_outputs = decoder_layer(
          hidden_states,
          position_embeddings=position_embeddings,
          attention_mask=attention_mask,
          position_ids=position_ids,
          past_key_values=layer_past_key_values,
          use_cache=use_cache,
          deterministic=deterministic,
      )
      hidden_states = layer_outputs[0]

      if use_cache:
        next_past_key_values.append(layer_outputs[1])

    hidden_states = self.norm(hidden_states)

    if use_cache:
      past_key_values = tuple(next_past_key_values)

    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )

from typing import Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

# From generated_code.Qwen3MoeForCausalLM.configuration
from .configuration_qwen3_moe import Qwen3MoeConfig
# From generated_code.Qwen3MoeForCausalLM.modeling
from .modeling import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
# From generated_code.Qwen3MoeForCausalLM.modeling_flax
from .modeling_flax import FlaxQwen3MoeModel, FlaxQwen3MoePreTrainedModel
# From generated_code.Qwen3MoeForCausalLM.cache_utils
from .cache_utils import Cache
# From generated_code.Qwen3MoeForCausalLM.losses
from .losses import ForCausalLMLoss, load_balancing_loss_func


class Qwen3MoeForCausalLM(FlaxQwen3MoePreTrainedModel):
  """
  Flax Qwen3MoeForCausalLM class.
  """

  config: Qwen3MoeConfig
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    # Reused from generated_code.Qwen3MoeForCausalLM.modeling_flax.Qwen3MoeModel
    self.model = FlaxQwen3MoeModel(self.config, dtype=self.dtype)
    self.vocab_size = self.config.vocab_size
    self.lm_head = nn.Dense(
        features=self.config.vocab_size,
        use_bias=False,
        dtype=self.dtype,
        kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        name="lm_head",
    )
    self.router_aux_loss_coef = self.config.router_aux_loss_coef
    self.num_experts = self.config.num_experts
    self.num_experts_per_tok = self.config.num_experts_per_tok
    # Reused from generated_code.Qwen3MoeForCausalLM.losses.ForCausalLMLoss
    self.loss_fct = ForCausalLMLoss()

  def __call__(
      self,
      input_ids: Optional[jnp.ndarray] = None,
      attention_mask: Optional[jnp.ndarray] = None,
      position_ids: Optional[jnp.ndarray] = None,
      past_key_values: Optional[Cache] = None,
      inputs_embeds: Optional[jnp.ndarray] = None,
      labels: Optional[jnp.ndarray] = None,
      use_cache: Optional[bool] = None,
      output_router_logits: Optional[bool] = None,
      cache_position: Optional[jnp.ndarray] = None,
      logits_to_keep: Union[int, jnp.ndarray] = 0,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
      **kwargs,
  ) -> MoeCausalLMOutputWithPast:
    r"""
    labels (`jnp.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    """
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits
    if isinstance(logits_to_keep, int):
      if logits_to_keep > 0:
        hidden_states_for_logits = hidden_states[:, -logits_to_keep:, :]
      else:
        hidden_states_for_logits = hidden_states
    else:
      hidden_states_for_logits = hidden_states[:, logits_to_keep, :]

    logits = self.lm_head(hidden_states_for_logits)

    loss = None
    if labels is not None:
      loss = self.loss_fct(logits, labels)

    aux_loss = None
    if output_router_logits:
      # Reused from generated_code.Qwen3MoeForCausalLM.losses.load_balancing_loss_func
      aux_loss = load_balancing_loss_func(
          outputs.router_logits,
          self.num_experts,
          self.num_experts_per_tok,
          attention_mask,
      )
      if labels is not None and loss is not None:
        loss += self.router_aux_loss_coef * aux_loss

    # Reused from generated_code.Qwen3MoeForCausalLM.modeling.MoeCausalLMOutputWithPast
    return MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )

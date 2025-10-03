
# In JAX, weight initialization is handled explicitly during `model.init()`.
# This global flag is not used in the JAX implementation but is kept for
# structural consistency during translation.
_init_weights: bool = True

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

"""
Configuration base class and utilities.
This file is converted from HuggingFace Transformers.
"""

import copy
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import jax.numpy as jnp


# type hinting: specifying the type of config class that inherits from PretrainedConfig
SpecificPretrainedConfigType = TypeVar("SpecificPretrainedConfigType", bound="PretrainedConfig")


class PretrainedConfig:
  # no-format
  r"""
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.

    <Tip>

    A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
    initialize a model does **not** load the model weights. It only affects the model's configuration.

    </Tip>

    Class attributes (overridden by derived classes):

    - **model_type** (`str`) -- An identifier for the model type, serialized into the JSON file.
    - **has_no_defaults_at_init** (`bool`) -- Whether the config class can be initialized without providing input arguments.
    - **attribute_map** (`Dict[str, str]`) -- A dict that maps model specific attribute names to the standardized
      naming of attributes.

    Common attributes (present in all subclasses):

    - **vocab_size** (`int`) -- The number of tokens in the vocabulary.
    - **hidden_size** (`int`) -- The hidden size of the model.
    - **num_attention_heads** (`int`) -- The number of attention heads.
    - **num_hidden_layers** (`int`) -- The number of blocks in the model.
    """

  model_type: str = ""
  base_config_key: str = ""
  sub_configs: Dict[str, Type["PretrainedConfig"]] = {}
  has_no_defaults_at_init: bool = False
  attribute_map: Dict[str, str] = {}
  base_model_tp_plan: Optional[Dict[str, Any]] = None
  base_model_pp_plan: Optional[Dict[str, tuple[List[str]]]] = None
  _auto_class: Optional[str] = None

  def __setattr__(self, key, value):
    if key in super().__getattribute__("attribute_map"):
      key = super().__getattribute__("attribute_map")[key]
    super().__setattr__(key, value)

  def __getattribute__(self, key):
    if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
      key = super().__getattribute__("attribute_map")[key]
    return super().__getattribute__(key)

  def __init__(
      self,
      *,
      # All models common arguments
      output_hidden_states: bool = False,
      output_attentions: bool = False,
      return_dict: bool = True,
      # JAX specific parameters
      dtype: Optional[Union[str, jnp.dtype]] = None,
      # Common arguments
      pruned_heads: Optional[Dict[int, List[int]]] = None,
      tie_word_embeddings: bool = True,
      chunk_size_feed_forward: int = 0,
      is_encoder_decoder: bool = False,
      is_decoder: bool = False,
      cross_attention_hidden_size: Optional[int] = None,
      add_cross_attention: bool = False,
      tie_encoder_decoder: bool = False,
      # Fine-tuning task arguments
      architectures: Optional[List[str]] = None,
      finetuning_task: Optional[str] = None,
      id2label: Optional[Dict[int, str]] = None,
      label2id: Optional[Dict[str, int]] = None,
      num_labels: Optional[int] = None,
      task_specific_params: Optional[Dict[str, Any]] = None,
      problem_type: Optional[str] = None,
      # Tokenizer kwargs
      tokenizer_class: Optional[str] = None,
      prefix: Optional[str] = None,
      bos_token_id: Optional[int] = None,
      pad_token_id: Optional[int] = None,
      eos_token_id: Optional[int] = None,
      sep_token_id: Optional[int] = None,
      decoder_start_token_id: Optional[int] = None,
      **kwargs,
  ):
    # Validation for some arguments
    if label2id is not None and not isinstance(label2id, dict):
      raise ValueError("Argument label2id should be a dictionary.")
    if id2label is not None and not isinstance(id2label, dict):
      raise ValueError("Argument id2label should be a dictionary.")
    if num_labels is not None and id2label is not None and len(id2label) != num_labels:
      warnings.warn(
          f"You passed `num_labels={num_labels}` which is incompatible to "
          f"the `id2label` map of length `{len(id2label)}`."
      )
    if problem_type is not None and problem_type not in (
        "regression",
        "single_label_classification",
        "multi_label_classification",
    ):
      raise ValueError(
          f"The config parameter `problem_type` was not understood: received {problem_type} "
          "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
      )
    if dtype is not None and isinstance(dtype, str):
      dtype = getattr(jnp, dtype)

    # Attributes common for all models
    self.return_dict = return_dict
    self.output_hidden_states = output_hidden_states
    self.dtype = dtype
    self._output_attentions = output_attentions  # has public property

    # Less common kwargs, only used by some models
    self.pruned_heads = pruned_heads if pruned_heads is not None else {}
    self.tie_word_embeddings = tie_word_embeddings
    self.chunk_size_feed_forward = chunk_size_feed_forward

    # Encoder-decoder models attributes
    self.is_encoder_decoder = is_encoder_decoder
    self.is_decoder = is_decoder  # used in encoder-decoder models to differentiate encoder from decoder
    self.cross_attention_hidden_size = cross_attention_hidden_size
    self.add_cross_attention = add_cross_attention
    self.tie_encoder_decoder = tie_encoder_decoder

    # Fine-tuning task attributes
    self.architectures = architectures
    self.finetuning_task = finetuning_task
    self.id2label = id2label
    self.label2id = label2id
    self.task_specific_params = task_specific_params
    self.problem_type = problem_type

    if self.id2label is None:
      self._create_id_label_maps(num_labels if num_labels is not None else 2)
    else:
      # Keys are always strings in JSON so convert ids to int here.
      self.id2label = {int(key): value for key, value in self.id2label.items()}

    # Tokenizer attributes
    self.tokenizer_class = tokenizer_class
    self.prefix = prefix
    self.bos_token_id = bos_token_id
    self.pad_token_id = pad_token_id
    self.eos_token_id = eos_token_id
    self.sep_token_id = sep_token_id
    self.decoder_start_token_id = decoder_start_token_id

    # Retrocompatibility: Parameters for sequence generation.
    for parameter_name, default_value in self._get_global_generation_defaults().items():
      setattr(self, parameter_name, kwargs.pop(parameter_name, default_value))

    # Name or path to the pretrained checkpoint
    self._name_or_path = str(kwargs.pop("name_or_path", ""))
    self._commit_hash = kwargs.pop("_commit_hash", None)

    # Attention implementation to use, if relevant (it sets it recursively on sub-configs)
    self._attn_implementation = kwargs.pop("attn_implementation", None)

    # Additional attributes without default values
    for key, value in kwargs.items():
      try:
        setattr(self, key, value)
      except AttributeError as err:
        # Using warnings instead of logger to avoid dependency
        warnings.warn(f"Can't set {key} with value {value} for {self}\n{err}")

  def _create_id_label_maps(self, num_labels: int):
    self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
    self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

  @property
  def name_or_path(self) -> Optional[str]:
    return getattr(self, "_name_or_path", None)

  @name_or_path.setter
  def name_or_path(self, value):
    self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

  @property
  def output_attentions(self):
    """
    `bool`: Whether or not the model should returns all attentions.
    """
    return self._output_attentions

  @output_attentions.setter
  def output_attentions(self, value: bool):
    # If we set `output_attentions` explicitly before the attn implementation, dispatch eager
    if value and self._attn_implementation is None:
      self._attn_implementation = "eager"
    if value and self._attn_implementation != "eager":
      raise ValueError(
          "The `output_attentions` attribute is not supported when using the `attn_implementation` set to "
          f"{self._attn_implementation}. Please set it to 'eager' instead."
      )
    self._output_attentions = value

  @property
  def use_return_dict(self) -> bool:
    """
    `bool`: Whether or not return a dataclass instead of tuples.
    """
    return self.return_dict

  @property
  def num_labels(self) -> int:
    """
    `int`: The number of labels for classification models.
    """
    return len(self.id2label)

  @num_labels.setter
  def num_labels(self, num_labels: int):
    # we do not store `num_labels` attribute in config, but instead
    # compute it based on the length of the `id2label` map
    if self.id2label is None or self.num_labels != num_labels:
      self._create_id_label_maps(num_labels)

  @property
  def _attn_implementation(self):
    return self._attn_implementation_internal

  @_attn_implementation.setter
  def _attn_implementation(self, value: Optional[Union[str, dict]]):
    """We set it recursively on the sub-configs as well"""
    # Set if for current config
    attn_implementation = value if not isinstance(value, dict) else value.get("", self._attn_implementation)
    self._attn_implementation_internal = attn_implementation

    # Set it recursively on the subconfigs
    for subconfig_key in self.sub_configs:
      subconfig = getattr(self, subconfig_key, None)
      if subconfig is not None:
        sub_implementation = (
            value if not isinstance(value, dict) else value.get(subconfig_key, subconfig._attn_implementation)
        )
        subconfig._attn_implementation = sub_implementation

  # The from_pretrained and save_pretrained methods are heavily tied to the Hugging Face Hub
  # and file downloading/caching logic, which is out of scope for a core JAX model.
  # Users in a JAX ecosystem typically handle model loading and saving via explicit
  # checkpointing utilities. We will implement from_dict and from_json_file for basic serialization.

  @classmethod
  def from_dict(
      cls: Type[SpecificPretrainedConfigType], config_dict: Dict[str, Any], **kwargs
  ) -> SpecificPretrainedConfigType:
    """
    Instantiates a [`PretrainedConfig`] from a Python dictionary of parameters.

    Args:
        config_dict (`Dict[str, Any]`):
            Dictionary that will be used to instantiate the configuration object.
        kwargs (`Dict[str, Any]`):
            Additional parameters from which to initialize the configuration object.

    Returns:
        [`PretrainedConfig`]: The configuration object instantiated from those parameters.
    """
    return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

    config_dict["attn_implementation"] = kwargs.pop("attn_implementation", None)

    config = cls(**config_dict)

    if hasattr(config, "pruned_heads"):
      config.pruned_heads = {int(key): value for key, value in config.pruned_heads.items()}

    # Update config with kwargs if needed
    to_remove = []
    for key, value in kwargs.items():
      if hasattr(config, key):
        current_attr = getattr(config, key)
        # To authorize passing a custom subconfig as kwarg in models that have nested configs.
        if isinstance(current_attr, PretrainedConfig) and isinstance(value, dict):
          value = current_attr.__class__(**value)
        setattr(config, key, value)
        if key != "dtype":
          to_remove.append(key)
    for key in to_remove:
      kwargs.pop(key, None)

    if return_unused_kwargs:
      return config, kwargs
    else:
      return config

  @classmethod
  def from_json_file(
      cls: Type[SpecificPretrainedConfigType], json_file: Union[str, os.PathLike]
  ) -> SpecificPretrainedConfigType:
    """
    Instantiates a [`PretrainedConfig`] from the path to a JSON file of parameters.

    Args:
        json_file (`str` or `os.PathLike`):
            Path to the JSON file containing the parameters.

    Returns:
        [`PretrainedConfig`]: The configuration object instantiated from that JSON file.

    """
    config_dict = cls._dict_from_json_file(json_file)
    return cls(**config_dict)

  @classmethod
  def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
    with open(json_file, "r", encoding="utf-8") as reader:
      text = reader.read()
    return json.loads(text)

  def __eq__(self, other):
    return isinstance(other, PretrainedConfig) and (self.__dict__ == other.__dict__)

  def __repr__(self):
    return f"{self.__class__.__name__} {self.to_json_string()}"

  def __iter__(self):
    yield from self.__dict__

  def to_diff_dict(self) -> Dict[str, Any]:
    """
    Removes all attributes from the configuration that correspond to the default
    config attributes for better readability.
    """
    config_dict = self.to_dict()

    # Get the default config dict
    default_config_dict = PretrainedConfig().to_dict()

    # get class specific config dict
    class_config_dict = self.__class__().to_dict() if not self.has_no_defaults_at_init else {}

    serializable_config_dict = {}

    for key, value in config_dict.items():
      if (
          key not in default_config_dict
          or value != default_config_dict[key]
          or (key in default_config_dict and value != class_config_dict.get(key, value))
      ):
        serializable_config_dict[key] = value

    self._remove_keys_not_serialized(serializable_config_dict)

    if "_name_or_path" in serializable_config_dict:
      del serializable_config_dict["_name_or_path"]

    self.dict_jax_dtype_to_str(serializable_config_dict)

    return serializable_config_dict

  def to_dict(self) -> Dict[str, Any]:
    """
    Serializes this instance to a Python dictionary.
    """
    output = copy.deepcopy(self.__dict__)
    if hasattr(self.__class__, "model_type"):
      output["model_type"] = self.__class__.model_type

    for key, value in output.items():
      if isinstance(value, PretrainedConfig):
        value = value.to_dict()
      output[key] = value

    self._remove_keys_not_serialized(output)
    self.dict_jax_dtype_to_str(output)

    return output

  def to_json_string(self, use_diff: bool = True) -> str:
    """
    Serializes this instance to a JSON string.
    """
    if use_diff is True:
      config_dict = self.to_diff_dict()
    else:
      config_dict = self.to_dict()
    return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

  def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
    """
    Save this instance to a JSON file.
    """
    with open(json_file_path, "w", encoding="utf-8") as writer:
      writer.write(self.to_json_string(use_diff=use_diff))

  def update(self, config_dict: Dict[str, Any]):
    """
    Updates attributes of this class with attributes from `config_dict`.
    """
    for key, value in config_dict.items():
      setattr(self, key, value)

  def dict_jax_dtype_to_str(self, d: Dict[str, Any]) -> None:
    """
    Converts jnp.dtype to a string.
    """
    if d.get("dtype") is not None and not isinstance(d["dtype"], str):
      d["dtype"] = str(d["dtype"])
    for value in d.values():
      if isinstance(value, dict):
        self.dict_jax_dtype_to_str(value)

  def _remove_keys_not_serialized(self, d: Dict[str, Any]) -> None:
    """
    Removes keys that should not be serialized.
    """
    if "_auto_class" in d:
      del d["_auto_class"]
    if "_output_attentions" in d:
      d["output_attentions"] = d.pop("_output_attentions")
    if "_commit_hash" in d:
      del d["_commit_hash"]
    if "_attn_implementation_internal" in d:
      del d["_attn_implementation_internal"]
    if "base_model_tp_plan" in d:
      del d["base_model_tp_plan"]
    if "base_model_pp_plan" in d:
      del d["base_model_pp_plan"]
    for value in d.values():
      if isinstance(value, dict):
        self._remove_keys_not_serialized(value)

  @staticmethod
  def _get_global_generation_defaults() -> Dict[str, Any]:
    return {
        "max_length": 20,
        "min_length": 0,
        "do_sample": False,
        "early_stopping": False,
        "num_beams": 1,
        "num_beam_groups": 1,
        "diversity_penalty": 0.0,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "typical_p": 1.0,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "num_return_sequences": 1,
        "output_scores": False,
        "return_dict_in_generate": False,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "remove_invalid_values": False,
        "exponential_decay_length_penalty": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
    }
_hf_api_to_flash_mapping = {
    "dropout": "dropout_p",
    "sliding_window": "window_size",
}
from typing import Any, Dict, List, Optional

# Reused from Qwen3ForCausalLM.rope.rope_config_validation
from Qwen3ForCausalLM.rope import rope_config_validation


class Qwen3MoeConfig:
  r"""
    This is the configuration class to store the configuration of a [`Qwen3MoeModel`]. It is used to instantiate a
    Qwen3MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of [Qwen/Qwen3-15B-A2B](https://huggingface.co/Qwen/Qwen3-15B-A2B).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3MoE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen3MoeModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.

        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer.
        moe_intermediate_size (`int`, *optional*, defaults to 768):
            Intermediate size of the routed expert.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 128):
            Number of routed experts.
        norm_topk_prob (`bool`, *optional*, defaults to `False`):
            Whether to normalize the topk probabilities.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        mlp_only_layers (`list[int]`, *optional*, defaults to `[]`):
            Indicate which layers use Qwen3MoeMLP rather than Qwen3MoeSparseMoeBlock
            The list contains layer index, from 0 to num_layers-1 if we have num_layers layers
            If `mlp_only_layers` is empty, `decoder_sparse_step` is used to determine the sparsity.

    
_tf_version = "N/A"
_tf_available = False
import copy
import re
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints

import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ..configuration_utils import PretrainedConfig
from ..generation import GenerationConfig
from ..utils import DUMMY_INPUTS, logging
from ..utils.generic import OutputRecorder

# In a JAX/Flax environment, mixins like PushToHubMixin and PeftAdapterMixin would
# require specific JAX implementations. For this translation, we focus on the core
# model structure and utilities from ModuleUtilsMixin and EmbeddingAccessMixin,
# integrating their relevant functionalities directly into the base class.


logger = logging.get_logger(__name__)


class PreTrainedModel(nn.Module):
  r"""
  Base class for all models.

  [`PreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
  downloading and saving models as well as a few methods common to all models to:

      - resize the input embeddings,
      - prune heads in the self-attention heads.

  Class attributes (overridden by derived classes):

      - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
        for this model architecture.
      - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
        classes of the same architecture adding modules on top of the base model.
      - **is_parallelizable** (`bool`) -- A flag indicating whether this model supports model parallelization.
      - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
        models, `pixel_values` for vision models and `input_values` for speech models).
  """

  config: PretrainedConfig
  dtype: jnp.dtype = jnp.float32

  config_class: Type[PretrainedConfig] = None
  base_model_prefix: str = ""
  main_input_name: str = "input_ids"
  model_tags: Optional[List[str]] = None

  _checkpoint_conversion_mapping: Dict[str, str] = {}

  _auto_class: Optional[str] = None
  # The following attributes are PyTorch-specific and related to device placement or saving.
  # They don't have a direct equivalent in JAX/Flax's sharding and checkpointing system.
  # _no_split_modules = None
  # _skip_keys_device_placement = None
  # _keep_in_fp32_modules = None
  # _keep_in_fp32_modules_strict = None

  # These attributes are related to loading/saving checkpoints. In a JAX framework,
  # this logic is typically handled by external checkpointing utilities, not the model class itself.
  _keys_to_ignore_on_load_missing: Optional[List[str]] = None
  _keys_to_ignore_on_load_unexpected: Optional[List[str]] = None
  _keys_to_ignore_on_save: Optional[List[str]] = None
  _tied_weights_keys: Optional[List[str]] = None

  is_parallelizable: bool = False
  supports_gradient_checkpointing: bool = False
  _is_stateful: bool = False

  # Attention implementation support flags
  _supports_flash_attn: bool = False
  _supports_sdpa: bool = False
  _supports_flex_attn: bool = False

  # JAX/Flax uses logical axis rules for sharding, not explicit TP/PP plans on the model.
  # _tp_plan = None
  # _tp_size = None
  # _pp_plan = None

  _supports_attention_backend: bool = False
  _can_record_outputs: Optional[Dict[str, Any]] = None

  @property
  def can_record_outputs(self) -> Dict[str, OutputRecorder]:
    """
    Maps output names (e.g., "attentions", "hidden_states")
    to an `OutputRecorder`.
    """
    return self._can_record_outputs or {}

  @property
  def dummy_inputs(self) -> Dict[str, jnp.ndarray]:
    """
    `Dict[str, jnp.ndarray]`: Dummy inputs to do a forward pass in the network.
    """
    return {"input_ids": jnp.array(DUMMY_INPUTS)}

  @property
  def framework(self) -> str:
    """
    :str: Identifies that this is a Flax model.
    """
    return "flax"

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    # For BC we keep the original `config_class` definition in case
    # there is a `config_class` attribute (e.g. remote code models),
    # otherwise we derive it from the annotated `config` attribute.

    # defined in this particular subclass
    child_annotation = cls.__dict__.get("__annotations__", {}).get("config", None)
    child_attribute = cls.__dict__.get("config_class", None)

    # defined in the class (this subclass or any parent class)
    full_annotation = get_type_hints(cls).get("config", None)
    full_attribute = cls.config_class

    # priority (child class_config -> child annotation -> global class_config -> global annotation)
    if child_attribute is not None:
      cls.config_class = child_attribute
    elif child_annotation is not None:
      cls.config_class = child_annotation
    elif full_attribute is not None:
      cls.config_class = full_attribute
    elif full_annotation is not None:
      cls.config_class = full_annotation

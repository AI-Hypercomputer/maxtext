
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
Flax PEFT model.
"""
from typing import Any, Dict, List, Optional, Union

import jax.numpy as jnp


# Note on JAX/Flax Conversion:
# The original PyTorch `PeftAdapterMixin` relies heavily on the `peft` library, which
# performs dynamic, in-place modifications of a model's architecture (a concept often
# called "model surgery"). It finds specific layers (e.g., `torch.nn.Linear`) and
# replaces them with PEFT-compatible layers (e.g., `peft.tuners.lora.Linear`) that
# contain the adapter weights.
#
# This dynamic, object-oriented approach is fundamentally incompatible with JAX's
# functional and static computation graph paradigm. In JAX and Flax, a model's
# architecture is defined statically before compilation. Modifying the model's
# structure at runtime is not a supported or idiomatic practice.
#
# A JAX-native approach to implementing parameter-efficient fine-tuning methods
# like LoRA involves the following principles:
# 1.  **Static Architecture**: The model is defined from the outset to include the
#     potential for adapter layers. For example, a LoRA-enabled linear layer would
#     have the base weights and the LoRA A/B matrices as part of its static definition.
# 2.  **Parameter Management**: Adapter weights are managed as part of the model's
#     parameter PyTree. Loading an adapter involves loading its weights from a file
#     and merging them into the main parameter PyTree. Deleting an adapter means
#     removing its corresponding parameters from the PyTree.
# 3.  **Conditional Forward Pass**: The active adapter is selected by passing an
#     argument (e.g., `active_adapter_name`) through the model's `apply` method.
#     The adapter layers within the model then conditionally execute their logic
#     based on this argument.
#
# Given these fundamental differences, a direct line-by-line translation of
# `PeftAdapterMixin` is not feasible. The following code provides a class with the
# same API structure, but the methods raise `NotImplementedError` and include
# comments explaining how the equivalent functionality would be achieved in JAX.
# This serves as a conceptual guide for re-implementing adapter support in a
# JAX-idiomatic way.


class PeftAdapterMixin:
  """
    A conceptual JAX-based mixin for handling PEFT adapters.

    This class preserves the API of the original PyTorch `PeftAdapterMixin` but does
    not provide a functional implementation due to paradigm differences between
    PyTorch and JAX. See the note at the top of the file for a detailed explanation.
    """

  _hf_peft_config_loaded: bool = False

  def load_adapter(
      self,
      peft_model_id: Optional[str] = None,
      adapter_name: Optional[str] = None,
      # JAX/Flax does not have a direct equivalent of torch.device.
      # Sharding and device placement are handled by the JAX mesh.
      device_map: Optional[str] = "auto",
      # Other arguments are kept for API consistency.
      **kwargs,
  ) -> None:
    """
        Conceptual method for loading adapter weights.

        In JAX, this would involve:
        1. Downloading the adapter configuration and weights (e.g., from Hugging Face Hub).
        2. Parsing the weights into a JAX PyTree.
        3. Merging this PyTree of adapter weights into the model's main parameter PyTree.
           The model must already have been defined with the structure to hold these weights.
        """
    raise NotImplementedError(
        "Dynamic adapter loading via model surgery is not supported in JAX. "
        "A JAX-native implementation would load weights into a pre-defined model structure."
    )

  def add_adapter(self, adapter_config, adapter_name: Optional[str] = None) -> None:
    """
        Conceptual method for adding a new, trainable adapter.

        In JAX, this would involve:
        1. Defining a new set of adapter parameters (e.g., LoRA A/B matrices) based on the `adapter_config`.
        2. Initializing these parameters (e.g., with zeros or random values).
        3. Merging the new parameter PyTree into the model's main parameter PyTree under the specified `adapter_name`.
        """
    raise NotImplementedError(
        "Dynamic adapter addition via model surgery is not supported in JAX. "
        "A JAX-native implementation would initialize and merge new adapter parameters into the state."
    )

  def set_adapter(self, adapter_name: Union[List[str], str]) -> None:
    """
        Conceptual method for setting the active adapter.

        In JAX, this functionality is typically not achieved by modifying the model's state.
        Instead, the name of the active adapter(s) should be passed as an argument to the
        model's `apply` or `__call__` method during the forward pass. This method could
        be adapted to set a default `active_adapter` attribute on the model instance.
        """
    raise NotImplementedError(
        "Setting an active adapter in JAX is typically handled by passing an argument to the forward pass, "
        "not by modifying the model's internal state."
    )

  def disable_adapters(self) -> None:
    """
        Conceptual method for disabling all adapters.

        In JAX, this would be equivalent to calling the model's forward pass with the
        `active_adapter_name` argument set to `None`.
        """
    raise NotImplementedError(
        "Disabling adapters in JAX is achieved by passing `active_adapter_name=None` to the forward pass."
    )

  def enable_adapters(self) -> None:
    """
        Conceptual method for enabling adapters.

        In JAX, this would be equivalent to calling the model's forward pass with a
        specific `active_adapter_name`. This method is less meaningful in a functional paradigm.
        """
    raise NotImplementedError(
        "Enabling adapters in JAX is the default state; control is achieved by specifying an `active_adapter_name`."
    )

  def active_adapters(self) -> List[str]:
    """
        Conceptual method for getting the currently active adapters.

        In a JAX model, the active adapter is not part of the model's persistent state but is
        an argument to the forward pass. This method's utility is limited in a JAX context.
        """
    raise NotImplementedError("The active adapter is a runtime argument in JAX, not a persistent model state.")

  def get_adapter_state_dict(self, adapter_name: Optional[str] = None) -> Dict[str, jnp.ndarray]:
    """
        Conceptual method for extracting an adapter's parameters.

        This is feasible in JAX. It would involve:
        1. Traversing the model's full parameter PyTree.
        2. Filtering the PyTree to include only the parameters associated with the specified `adapter_name`.
        3. Returning the filtered PyTree of adapter weights.
        """
    raise NotImplementedError(
        "This can be implemented by filtering the model's parameter PyTree for adapter-specific keys."
    )

  def delete_adapter(self, adapter_names: Union[List[str], str]) -> None:
    """
        Conceptual method for deleting an adapter.

        In JAX, this would not change the model's architecture. It would involve:
        1. Traversing the model's parameter PyTree.
        2. Removing the parameters associated with the specified `adapter_names`.
        3. The model structure, which contains the slots for these parameters, would remain unchanged.
        """
    raise NotImplementedError(
        "Deleting an adapter in JAX means removing its parameters from the parameter PyTree, not altering the model structure."
    )

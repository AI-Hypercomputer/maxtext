"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = "Google LLC"
__version__ = "2025.04.25"
__description__ = (
    "MaxText is a high performance, highly scalable, open-source LLM written in pure Python/Jax and "
    "targeting Google Cloud TPUs and GPUs for training and **inference."
)


# maxtext/__init__.py

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.layers import models

# Import the global store functionality
from MaxText import global_store

Transformer = models.Transformer

def from_pretrained(
    config: pyconfig.HyperParameters,
    store_global_state: bool = True,
    **kwargs
) -> Transformer:
    """Load a pretrained MaxText model from checkpoint.
    
    This function loads a model from a checkpoint and optionally stores
    the mesh and init_rng in a global store for later retrieval in train.py.
    
    Args:
        config: Optional config object. If None, will load from checkpoint
        store_global_state: Whether to store mesh/init_rng in global store
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Transformer: The loaded model instance (only the model)
    
    Example:
        # Basic usage - mesh and init_rng are stored globally
        model = from_pretrained(config)

        # Disable global storage for testing
        model = from_pretrained(config, store_global_state=False)
    """
    mesh, init_rng = maxtext_utils.initialize_platform(config)
    model = maxtext_utils.create_model(config, mesh)
    checkpoint_manager, learning_rate_schedule, tx = maxtext_utils.create_training_tools(config, model, mesh, init_rng)
    
    # Store in global store if requested
    if store_global_state:
        global_store.store_global_state(mesh, init_rng, checkpoint_manager, learning_rate_schedule, tx)    

    # Return only the model
    return model

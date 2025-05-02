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

import datetime
import os
import sys
import functools
import time
import queue

from typing import Sequence
from absl import app
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import grain.python as grain
import jax
import numpy as np
import orbax.checkpoint
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager

from MaxText import checkpointing
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import max_logging
from MaxText import optimizers
from MaxText import profiler
from MaxText import pyconfig
import pathwaysutils  # pylint: disable=unused-import
import tensorflow as tf

from MaxText.metric_logger import MetricLogger
from MaxText.utils import gcs_utils

from MaxText.vertex_tensorboard import VertexTensorboardManager
# Placeholder: internal

from MaxText.input_pipeline.input_pipeline_interface import create_data_iterator
from MaxText.layers import models

from MaxText.gcp_workload_monitor import GCPWorkloadMonitor

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jax.experimental import checkify

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from MaxText.layers import quantizations

from ml_goodput_measurement import goodput
from ml_goodput_measurement import monitoring
from MaxText.train import setup_mesh_and_model

# from MaxText.layers.models import Transformer
# Transformer = models.Transformer

def from_pretrained(config):
    """
    Create the model and mesh and other artifacts
     needed for running training with a config that specifies a model_name, checkpoint
    """
    model, mesh, init_rng, writer, checkpoint_manager, learning_rate_schedule, tx = setup_mesh_and_model(config)
    return model, mesh, init_rng, writer, checkpoint_manager, learning_rate_schedule, tx


# def setup_mesh_and_model(config, devices=None):
#   """Set up the mesh and the model for training

#   Args:
#     config
#     devices

#   Returns:
#     init_rng: RNG key
#     writer: Summary writer for tensorboard
#     checkpoint_manager: Orbax checkpointer
#     state_mesh_annotations: the mesh annotations for the train state
#     model:
#     mesh:
#     learning_rate_schedule:
#     tx:
#   """

#   init_rng = random.PRNGKey(config.init_weights_seed)
#   writer = max_utils.initialize_summary_writer(config.tensorboard_dir, config.run_name)

#   # Mesh definition
#   devices_array = maxtext_utils.create_device_mesh(config, devices)
#   mesh = Mesh(devices_array, config.mesh_axes)

#   # Model and Optimizer definition
#   quant = quantizations.configure_quantization(config)
#   model = Transformer(config, mesh, quant=quant)
#   learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
#   tx = optimizers.get_optimizer(config, learning_rate_schedule)
#   logger = checkpointing.setup_checkpoint_logger(config)
#   if config.enable_emergency_checkpoint:
#     if config.use_replicator_service:
#       checkpoint_manager = checkpointing.create_orbax_emergency_replicator_checkpoint_manager(
#           config.local_checkpoint_directory,
#           config.local_checkpoint_period,
#           mesh,
#       )
#     else:
#       abstract_state, _, _ = maxtext_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
#       checkpoint_manager = checkpointing.create_orbax_emergency_checkpoint_manager(
#           config.local_checkpoint_directory,
#           config.checkpoint_dir,
#           mesh,
#           abstract_state,
#           config.local_checkpoint_period,
#           config.checkpoint_period,
#           logger,
#       )
#   else:
#     # TODO(b/368121306): Remove this once zarr3 support is plumbed on the backend
#     use_ocdbt = config.checkpoint_storage_use_ocdbt
#     use_zarr3 = config.checkpoint_storage_use_zarr3
#     if config.enable_single_controller:
#       use_ocdbt, use_zarr3 = False, False
#     checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
#         config.checkpoint_dir,
#         config.enable_checkpointing,
#         config.async_checkpointing,
#         config.checkpoint_period,
#         config.dataset_type,
#         logger,
#         use_ocdbt,
#         use_zarr3,
#     )

#   return model, mesh, init_rng, writer, checkpoint_manager, learning_rate_schedule, tx
    
    
    







#     # # 1. Resolve configuration (mesh axes, vocab size, num layers, etc.)
#     # config = get_config(model_name)

#     # # 2. Instantiate an *empty* parameter pytree matching the model architecture
#     # init_rng = jax.random.PRNGKey(config.init_weights_seed)
#     # # We rely on your existing abstract_state extractor:
#     # from MaxText.max_utils import get_abstract_state
#     # abstract_state, _, _ = get_abstract_state(
#     #     Transformer(config, mesh=None),  # no mesh needed for shape inference
#     #     None,  # optimizer
#     #     config,
#     #     init_rng,
#     #     mesh=None,
#     #     is_training=False
#     # )
#     # params = abstract_state.params

#     # # 3. Create an Orbax checkpoint manager pointed at checkpoint_path
#     # ckpt_manager = get_checkpoint_manager(checkpoint_path)

#     # # 4. Load the weights into `params`
#     # params = load_checkpoint(ckpt_manager, checkpoint_path, params)

#     # # 5. Wrap in our high-level model API
#     # return MaxTextModel(config, params)


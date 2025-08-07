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

# pylint: disable=bare-except, consider-using-generator
""" Utils that are only interesting for training in MaxText. """

import os
import jax
import tensorflow as tf
import pathwaysutils  # pylint: disable=unused-import
from typing import Any, Sequence, Tuple
from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration
from MaxText.layers import quantizations
from MaxText.layers import models
from MaxText import optimizers
from MaxText import checkpointing
from MaxText import maxtext_utils
from MaxText import max_logging
from MaxText import max_utils
from MaxText import pyconfig
from MaxText.input_pipeline.input_pipeline_interface import create_data_iterator
from MaxText.utils.goodput_utils import (
    GoodputEvent,
    maybe_record_goodput,
    maybe_monitor_goodput,
    create_goodput_recorder,
)
from MaxText.vertex_tensorboard import VertexTensorboardManager
import MaxText as mt


def get_transformer_model(config, mesh, quant):
  if config.model_fsdp_ag_once:
    return models.ZeroOneTransformer(config, mesh, quant=quant)
  else:
    return models.Transformer(config, mesh, quant=quant)


def create_model(config, mesh):
  """Instantiates and returns the model object, sharded across the mesh."""
  # Model definition
  quant = quantizations.configure_quantization(config)
  model = get_transformer_model(config, mesh, quant)
  model = quantizations.maybe_quantize_model(model, config)
  return model


def create_training_tools(config, model, mesh):
  """Creates the init_rng, optimizer, learning rate schedule, and checkpoint manager."""
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)
  logger = checkpointing.setup_checkpoint_logger(config)
  if config.enable_emergency_checkpoint:
    if config.use_replicator_service:
      checkpoint_manager = checkpointing.create_orbax_emergency_replicator_checkpoint_manager(
          config.local_checkpoint_directory,
          config.local_checkpoint_period,
          mesh,
      )
    else:
      abstract_state, _, _ = maxtext_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
      checkpoint_manager = checkpointing.create_orbax_emergency_checkpoint_manager(
          config.local_checkpoint_directory,
          config.checkpoint_dir,
          mesh,
          abstract_state,
          config.local_checkpoint_period,
          config.checkpoint_period,
          logger,
      )
  else:
    # TODO(b/368121306): Remove this once zarr3 support is plumbed on the backend
    use_ocdbt = config.checkpoint_storage_use_ocdbt
    use_zarr3 = config.checkpoint_storage_use_zarr3
    if config.enable_single_controller:
      use_ocdbt, use_zarr3 = False, False

    checkpoint_dir = ""
    if config.enable_checkpointing:
      checkpoint_dir = config.checkpoint_dir
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
        config.dataset_type,
        logger,
        use_ocdbt,
        use_zarr3,
    )

  return init_rng, checkpoint_manager, learning_rate_schedule, tx


def jit_train_step(config, model, state, state_mesh_shardings, data_sharding, train_step):
  """Returns a JIT-compiled train step function, which is loaded from a file if specified in the config."""
  functional_train, in_shardings, out_shardings, static_argnums, donate_argnums = (
      maxtext_utils.get_functional_train_with_signature(train_step, data_sharding, state_mesh_shardings, model, config)
  )

  # Define the compilation of functional_train, either by loading the compiled version or wrapping a new one in a jit
  if config.compiled_trainstep_file != "":
    print("Loading the compiled function...", flush=True)
    # Need to pass train signature and state to determine i/o shapes of train_state for now.
    p_train_step = maxtext_utils.load_compiled(config, functional_train, state)
    print("Loaded compiled function!", flush=True)
  else:
    p_train_step = jax.jit(
        functional_train,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )

  return p_train_step


def jit_eval_step(config, model, state_mesh_shardings, data_sharding, eval_step):
  """Returns a JIT-compiled eval step function."""
  functional_eval, in_shardings, out_shardings, static_argnums, donate_argnums = (
      maxtext_utils.get_functional_eval_with_signature(eval_step, data_sharding, state_mesh_shardings, model, config)
  )

  p_eval_step = None
  if config.compiled_trainstep_file == "":
    p_eval_step = jax.jit(
        functional_eval,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )

  return p_eval_step


def jit_train_and_eval_step(
    config, model, mesh, state, state_mesh_shardings, train_step, eval_step=None, eval_data_iterator=None
):
  """Returns a JIT-compiled train and eval step function."""
  data_sharding = maxtext_utils.get_input_data_sharding(config, mesh)
  p_train_step = jit_train_step(config, model, state, state_mesh_shardings, data_sharding, train_step)
  p_eval_step = None
  if eval_data_iterator:
    p_eval_step = jit_eval_step(config, model, state_mesh_shardings, data_sharding, eval_step)

  return p_train_step, p_eval_step


def validate_train_config(config):
  """Validates the configuration is set correctly for training."""

  assert config.run_name, "Erroring out, need a real run_name"
  if config.dataset_path and not config.dataset_path.startswith("gs://"):
    max_logging.log("WARNING: 'dataset_path' might be pointing your local file system")
  if not config.base_output_directory.startswith("gs://"):
    max_logging.log("WARNING: 'base_output_directory' might be pointing your local file system")
  assert config.steps > 0, "You must set steps or learning_rate_schedule_steps to a positive integer."

  if config.quantization in ("fp8", "nanoo_fp8"):
    # pylint: disable=line-too-long
    assert config.gradient_accumulation_steps == 1, (
        "fp8 can't be used with gradient_accumulation_steps right now. Please use other quantization or set "
        "gradient_accumulation_steps to 1"
    )

  # Check if GPU Flash Attention is being used with sequence packing
  if config.attention == "cudnn_flash_te" and config.packing and config.dataset_type != "synthetic":
    raise ValueError(
        "cudnn_flash_te only supports BSHD format. The THD (seq packing) support is going to be available in "
        "Transformer Engine 2.0 release. "
        "Please disable sequence packing (set packing=False) or use a different attention mechanism. "
        "With synthetic data, the format is not important as packing is not applied."
    )


def initialize(argv: Sequence[str]) -> Tuple[pyconfig.HyperParameters, Any, Any]:
  """Initialization of hyperparameters and utilities"""
  pathwaysutils.initialize()
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  # TODO: mazumdera@ : ensure missing mandatory fields in base.yml are filled in in argv,
  # or fill in here
  config = pyconfig.initialize(argv)
  jax.config.update("jax_use_shardy_partitioner", config.shardy)
  max_utils.print_system_information()
  validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path or ""
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  # Goodput configurations
  maybe_monitor_goodput(config)
  recorder = create_goodput_recorder(config)

  # Stack traces configurations
  debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=config.collect_stack_trace,
          stack_trace_to_cloud=config.stack_trace_to_cloud,
          stack_trace_interval_seconds=config.stack_trace_interval_seconds,
      )
  )
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  return config, recorder, diagnostic_config

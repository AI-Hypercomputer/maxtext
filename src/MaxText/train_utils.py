# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=bare-except, consider-using-generator
""" Utils that are only interesting for training in MaxText. """

import os
from typing import Any, Iterator

import jax

import optax

from pydantic import BaseModel, Field, ConfigDict

from MaxText import checkpointing
from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import model_creation_utils
from MaxText import optimizers
from MaxText import sharding
from MaxText.data_loader import create_dataloader, DataLoader
from MaxText.dpo_utils import _merge_dpo_state
from MaxText.input_pipeline.input_pipeline_interface import create_data_iterator
from MaxText.rampup_batch import create_rampup_manager, RampupBatchManager
from MaxText.utils.goodput_utils import GoodputEvent
from MaxText.utils.goodput_utils import maybe_record_goodput

# Fix for Pydantic resolving TrainState annotations
ArrayTree = Any


class TrainContext(BaseModel):
  """Training context."""

  model_config = ConfigDict(arbitrary_types_allowed=True)

  init_rng: jax.Array = Field(description="PRNG key initialized for the training loop.")
  checkpoint_manager: checkpointing.CheckpointManager | None = Field(
      description="Orbax CheckpointManager for saving/restoring checkpoints."
  )
  state_mesh_shardings: Any = Field(description="TrainState containing sharding specifications for the model state.")
  model: Any = Field(description="The initialized Flax (Linen or NNX) model instance.")
  mesh: jax.sharding.Mesh = Field(description="JAX Mesh object defining the device topology.")
  learning_rate_schedule: optax.Schedule | None = Field(description="Optax schedule function for learning rate.")
  data_iterator: Iterator[Any] = Field(description="Iterator for training data.")
  data_loader: DataLoader = Field(description="DataLoader instance handling sharding and batching.")
  rampup_manager: RampupBatchManager | None = Field(description="Manager class for handling batch size rampup.")
  eval_data_iterator: Iterator[Any] | None = Field(description="Iterator for evaluation data.")
  state: Any = Field(description="Current TrainState containing parameters and optimizer state.")


# Explicitly rebuild the model to resolve possible ForwardRefs
TrainContext.model_rebuild()


def create_training_tools(config, model, mesh):
  """Creates the init_rng, optimizer, learning rate schedule, and checkpoint manager."""
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  # pass in model for muon
  tx = optimizers.get_optimizer(config, learning_rate_schedule, model)
  logger = checkpointing.setup_checkpoint_logger(config)
  if config.enable_multi_tier_checkpointing:
    checkpoint_manager = checkpointing.create_orbax_emergency_replicator_checkpoint_manager(
        config.local_checkpoint_directory,
        config.local_checkpoint_period,
        mesh,
    )
  elif config.enable_emergency_checkpoint:
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
        config.enable_continuous_checkpointing,
        config.max_num_checkpoints_to_keep,
    )

  return init_rng, checkpoint_manager, learning_rate_schedule, tx


def jit_train_step(config, model, state, state_mesh_shardings, data_sharding, train_step, params_shardings):
  """Returns a JIT-compiled train step function, which is loaded from a file if specified in the config."""
  (
      functional_train,
      in_shardings,
      out_shardings,
      static_argnums,
      donate_argnums,
  ) = maxtext_utils.get_functional_train_with_signature(
      train_step, data_sharding, state_mesh_shardings, model, config, params_shardings
  )

  # Define the compilation of functional_train, either by loading the compiled version or wrapping a new one in a jit
  if config.compiled_trainstep_file != "":
    max_logging.log("Loading the compiled function...")
    execution_devices = model.mesh.devices.flatten().tolist()
    # Need to pass train signature and state to determine i/o shapes of train_state for now.
    p_train_step = maxtext_utils.load_compiled(config, functional_train, state, execution_devices)
    max_logging.log("Loaded compiled function!")
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
  (
      functional_eval,
      in_shardings,
      out_shardings,
      static_argnums,
      donate_argnums,
  ) = maxtext_utils.get_functional_eval_with_signature(eval_step, data_sharding, state_mesh_shardings, model, config)

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
    config,
    model,
    mesh,
    state,
    state_mesh_shardings,
    train_step,
    eval_step=None,
    eval_data_iterator=None,
    params_shardings=None,
):
  """Returns a JIT-compiled train and eval step function."""
  data_sharding = sharding.get_input_data_sharding(config, mesh)
  p_train_step = jit_train_step(config, model, state, state_mesh_shardings, data_sharding, train_step, params_shardings)
  p_eval_step = None
  if eval_data_iterator:
    p_eval_step = jit_eval_step(config, model, state_mesh_shardings, data_sharding, eval_step)

  return p_train_step, p_eval_step


def setup_train_loop(config, recorder, devices=None) -> TrainContext:
  """Sets up prerequisites for the training loop.

  Sets up checkpoint_manager, PRNG keys, Mesh, Model and optimizer.
  Sets up data iterator and tokenizer, initializes the model.

  Args:
    config: pyconfig.HyperParameters
    recorder: GoodputRecorder
    devices: List of devices to use.

  Returns:
      TrainContext: A dataclass containing the training context.
  """

  with maybe_record_goodput(recorder, GoodputEvent.TPU_INIT):
    model = model_creation_utils.from_config(config, devices)
    mesh = model.mesh
    init_rng, checkpoint_manager, learning_rate_schedule, tx = create_training_tools(config, model, mesh)

  with maybe_record_goodput(recorder, GoodputEvent.TRAINING_PREPARATION):
    data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
    rampup_manager = create_rampup_manager(config, checkpoint_manager)
    data_loader = create_dataloader(config, mesh, data_iterator, recorder, rampup_manager)
    context_parallel_size = mesh.shape["context"]
    # Check if context parallelism is being used with sequence packing
    if context_parallel_size > 1 and config.packing and config.dataset_type != "synthetic":
      raise ValueError(
          "Context parallelism cannot be used with sequence packing. "
          "Disable sequence packing (set packing=False). "
          "Context parallelism with packing support will be added soon."
      )

    # Apply reordering wrapper to data iterators if context parallelism is enabled
    with jax.set_mesh(mesh):
      if context_parallel_size > 1 and config.context_parallel_load_balance:
        data_iterator = map(maxtext_utils.get_reorder_callable(context_parallel_size, config.shard_mode), data_iterator)
        if eval_data_iterator:
          eval_data_iterator = map(
              maxtext_utils.get_reorder_callable(context_parallel_size, config.shard_mode),
              eval_data_iterator,
          )

    state, _, state_mesh_shardings, data_iterator = maxtext_utils.setup_training_state(
        model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
    )

    # TODO(aireenmei, hengtaoguo): support sharding in vit for multimodal
    if not config.using_pipeline_parallelism and not config.use_multimodal:
      # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage
      sharding.assert_params_sufficiently_sharded(state.params, mesh, config.sharding_tolerance)

    # print weights sharding info under debug sharding mode
    if config.debug_sharding:
      max_utils.print_non_trivial_mesh_axis(model.mesh)
      maxtext_utils.print_state_mesh_shardings_params(state, state_mesh_shardings, model.mesh)

    if config.use_dpo:
      abstract_state, _, _ = maxtext_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
      max_logging.log(
          "Restoring reference parameters for DPO from" f" '{os.path.join(str(config.checkpoint_dir), str(0))}'"
      )
      try:
        step0_restored, _ = checkpointing.load_state_if_possible(
            checkpoint_manager,
            data_iterator,
            load_parameters_from_path="",
            load_full_state_from_path="",
            checkpoint_storage_concurrent_gb=config.checkpoint_storage_concurrent_gb,
            abstract_unboxed_pre_state=abstract_state,
            enable_single_replica_ckpt_restoring=False,
            dataset_type=config.dataset_type,
            step=0,
            use_ocdbt=config.checkpoint_storage_use_ocdbt,
            use_zarr3=config.checkpoint_storage_use_zarr3,
            enable_orbax_v1=config.enable_orbax_v1,
            checkpoint_conversion_fn=config.checkpoint_conversion_fn,
            source_checkpoint_layout=config.source_checkpoint_layout,
        )
      except FileNotFoundError:
        step0_restored = None
      if step0_restored is not None:
        reference_params = step0_restored["items"].params["params"]
        state = _merge_dpo_state(state, reference_params)
      else:
        max_logging.log(
            "Could not restore reference parameters for DPO from" f" '{os.path.join(str(config.checkpoint_dir), str(0))}'"
        )

  return TrainContext(
      init_rng=init_rng,
      checkpoint_manager=checkpoint_manager,
      state_mesh_shardings=state_mesh_shardings,
      model=model,
      mesh=mesh,
      learning_rate_schedule=learning_rate_schedule,
      data_iterator=data_iterator,
      data_loader=data_loader,
      rampup_manager=rampup_manager,
      eval_data_iterator=eval_data_iterator,
      state=state,
  )


def validate_train_config(config):
  """Validates the configuration is set correctly for 'train.py'."""

  assert config.run_name, "Erroring out, need a real run_name"
  if config.dataset_path and not config.dataset_path.startswith("gs://"):
    max_logging.log("WARNING: 'dataset_path' might be pointing your local file system")
  if not config.base_output_directory.startswith("gs://"):
    max_logging.log("WARNING: 'base_output_directory' might be pointing your local file" " system")
  assert config.steps > 0, "You must set steps or learning_rate_schedule_steps to a positive integer."

  if config.quantization in ("fp8", "nanoo_fp8"):
    # pylint: disable=line-too-long
    assert config.gradient_accumulation_steps == 1, (
        "fp8 can't be used with gradient_accumulation_steps right now. Please"
        " use other quantization or set gradient_accumulation_steps to 1"
    )

  if config.packing and config.dataset_type == "synthetic":
    max_logging.log(
        "WARNING: Sequence packing is essentially ignored for synthetic data. "
        "Please use a real dataset to use sequence packing."
    )

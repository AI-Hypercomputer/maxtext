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
"""Utils that are only interesting for training in MaxText."""

import subprocess
import jax
import functools
from functools import partial

from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxtext.common import checkpointing
from maxtext.common import train_state_nnx
from maxtext.common.common_types import ReorderStrategy
from maxtext.common.data_loader import create_dataloader
from maxtext.common.goodput import GoodputEvent, maybe_record_goodput
from maxtext.optimizers import optimizers
from maxtext.trainers.diloco import diloco
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import model_creation_utils
from maxtext.utils import sharding
from maxtext.utils.rampup_batch import create_rampup_manager


def create_training_optimizer(config, model):
  """Creates the optimizer and learning rate schedule."""
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  # pass in model for muon
  tx = optimizers.get_optimizer(config, learning_rate_schedule, model)
  return learning_rate_schedule, tx


def create_checkpoint_manager(config, mesh, init_state_fn):
  """Creates the init_rng, optimizer, learning rate schedule, and checkpoint manager."""
  # pass in model for muon
  logger = checkpointing.setup_checkpoint_logger(config)
  if config.enable_multi_tier_checkpointing:
    checkpoint_manager = checkpointing.create_orbax_emergency_replicator_checkpoint_manager(
        config.local_checkpoint_directory,
        config.local_checkpoint_period,
        mesh,
    )
  elif config.enable_emergency_checkpoint:
    abstract_state, _, _ = maxtext_utils.get_abstract_state(config, mesh, init_state_fn, is_training=True)
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
    if config.enable_single_controller and not config.colocated_python_checkpointing:
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
        config.checkpoint_storage_concurrent_gb,
        config.enable_single_controller,
        config.colocated_python_checkpointing,
        config.enable_single_replica_ckpt_restoring,
        config.enable_autocheckpoint,
        config.checkpoint_todelete_subdir,
        config.checkpoint_todelete_full_path,
    )

  return checkpoint_manager


def jit_train_step(config, model, state, state_mesh_shardings, data_sharding, train_step, params_shardings, mesh=None):
  """Returns a JIT-compiled train step function, which is loaded from a file if specified in the config."""
  if config.enable_diloco:
    functional_train = train_step
    in_shardings = (state_mesh_shardings, data_sharding, None)  # State, batch, rng
    out_shardings = (state_mesh_shardings, None)  # State, metrics
    static_argnums = ()  # We partial out the static argnums of model and config
    donate_argnums = 0  # This is the index of the state - we allow the compiler to make use of this memory.
  else:
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
    # For NNX, model is the GraphDef (no .mesh); use the mesh passed explicitly instead.
    execution_mesh = mesh if mesh is not None else model.mesh
    execution_devices = execution_mesh.devices.flatten().tolist()
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
  if config.enable_diloco:
    train_step_partial = functools.partial(train_step, model, config, state_mesh_shardings, params_shardings)
    train_step = diloco.build_diloco_train_step(config, train_step_partial, mesh=mesh)
  data_sharding = sharding.get_input_data_sharding(config, mesh)
  p_train_step = jit_train_step(
      config, model, state, state_mesh_shardings, data_sharding, train_step, params_shardings, mesh=mesh
  )
  p_eval_step = None
  if eval_data_iterator:
    p_eval_step = jit_eval_step(config, model, state_mesh_shardings, data_sharding, eval_step)

  return p_train_step, p_eval_step


def setup_train_loop(config, recorder, devices=None):
  """Set up prerequisites for the training loop -

      checkpoint_manager, PRNG keys, Mesh, Model and optimizer.
      Set up data iterator and tokenizer, initialize the model.

  Args: config recorder

  Returns:
    init_rng:
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    data_iterator:
    data_loader:
    rampup_manager: the class managing rampup batch sizes
    train_state: the initialized train state. For NNX, this is a TrainStateNNX instance
  """
  # pylint: disable=import-outside-toplevel
  from maxtext.input_pipeline.input_pipeline_interface import create_data_iterator

  with maybe_record_goodput(recorder, GoodputEvent.TPU_INIT):
    is_training = True
    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    mesh = maxtext_utils.get_mesh_from_config(config, devices)
    context_parallel_size = mesh.shape.get(config.context_sharding, 1)
    if config.pure_nnx:
      # Create abstract NNX model.
      _create_model_partial, model = model_creation_utils.create_nnx_abstract_model(config, mesh, devices)
    else:
      model = model_creation_utils.from_config(config, devices)
    learning_rate_schedule, tx = create_training_optimizer(config, model)

    if config.pure_nnx:
      # For NNX, the train state is wrapped in the TrainStateNNX module.
      def create_train_state_fn():
        model = _create_model_partial()
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        return train_state_nnx.TrainStateNNX(model, optimizer)

      init_state_fn = create_train_state_fn
    else:
      init_state_fn = partial(maxtext_utils.init_initial_state, model, tx, config, is_training, init_rng)
    checkpoint_manager = create_checkpoint_manager(config, mesh, init_state_fn)
    if checkpoint_manager is not None:
      checkpoint_step = checkpoint_manager.latest_step()
      if checkpoint_step is not None:
        validate_completed_steps(checkpoint_step + 1, config.steps)

  with maybe_record_goodput(recorder, GoodputEvent.TRAINING_PREPARATION):
    data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
    rampup_manager = create_rampup_manager(config, checkpoint_manager)
    # Validate context parallelism with packing configuration
    context_parallel_strategy = config.context_parallel_strategy.lower()
    if context_parallel_size > 1 and config.packing:
      if config.dataset_type == "synthetic":
        raise ValueError(
            "Context parallelism with sequence packing is not supported with synthetic data. "
            "Please disable sequence packing (set packing=False)."
        )
      if context_parallel_strategy not in ("all_gather", "ring"):
        raise ValueError(
            "Context parallelism with sequence packing supports context_parallel_strategy='all_gather' or 'ring'."
        )
      if (
          config.hardware in ("gpu", "gpu_multiprocess")
          and config.attention == "cudnn_flash_te"
          and not (context_parallel_strategy == "ring" and config.context_parallel_load_balance)
      ):
        raise ValueError("Packing is only supported for load balanced ring attention with context parallelism for GPU.")

    # Apply reordering wrapper to data iterators if context parallelism is enabled
    with jax.set_mesh(mesh):
      if context_parallel_size > 1 and config.context_parallel_load_balance:

        # Determine load balancing reorder strategy based on whether packing is enabled
        if config.context_parallel_reorder_strategy == ReorderStrategy.AUTO:
          reorder_strategy = (
              ReorderStrategy.STRIPED
              if config.packing and context_parallel_strategy == "ring"
              else ReorderStrategy.DUAL_CHUNK_SWAP
          )
        else:
          reorder_strategy = config.context_parallel_reorder_strategy

        reorder_fn = maxtext_utils.get_reorder_callable(
            context_parallel_size, config.shard_mode, reorder_strategy, config.hardware
        )
        data_iterator = map(reorder_fn, data_iterator)
        if eval_data_iterator:
          eval_data_iterator = map(reorder_fn, eval_data_iterator)

    # Create data_loader AFTER reordering wrapper is applied
    data_loader = create_dataloader(config, mesh, data_iterator, recorder, rampup_manager)

    state, _, state_mesh_shardings, data_iterator = maxtext_utils.setup_training_state(
        data_iterator, config, mesh, checkpoint_manager, init_state_fn
    )
    if config.pure_nnx:
      with nn_partitioning.axis_rules(config.logical_axis_rules):
        # We only need the graphdef here; it's merged with state below. Avoid
        # nnx.get_abstract_model: it eagerly builds a NamedSharding for every variable
        # under jax.set_mesh(mesh) and rejects any logical name missing from
        # logical_axis_rules (e.g. concat_embed on the MTP kernel). Tracing shapes
        # without a mesh skips sharding resolution, so it avoids the crash.
        state_graphdef = nnx.graphdef(nnx.eval_shape(init_state_fn))
        _, state_params, _ = nnx.split(state.model, nnx.Param, ...)
        _, state_mesh_shardings_params, _ = nnx.split(state_mesh_shardings.model, nnx.Param, ...)
    else:
      state_params = state.params
      state_mesh_shardings_params = state_mesh_shardings.params

    if config.enable_diloco:
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
        state, outer_opt_state_sharding = diloco.build_diloco_state(config, lambda: state, mesh=mesh)

        # create state_mesh_shardings for the DilocoState
        inner_state_shardings = diloco.add_diloco_to_sharding(state_mesh_shardings)
        state_mesh_shardings = diloco.DiLoCoTrainState(
            inner_state_shardings,
            state_mesh_shardings.params,
            outer_opt_state_sharding,
            jax.sharding.NamedSharding(mesh=state_mesh_shardings.step.mesh, spec=jax.sharding.PartitionSpec()),
        )

    # TODO(aireenmei, hengtaoguo): support sharding in vit for multimodal
    if not config.using_pipeline_parallelism and not config.use_multimodal:
      # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage
      sharding.assert_params_sufficiently_sharded(state_params, mesh, config.sharding_tolerance)

    # print weights sharding info under debug sharding mode
    if config.debug_sharding:
      if config.pure_nnx:
        # TODO: Study how to get logical annotations of NNX module. Because of eager sharding, we
        # probably already lost the logical partition info at this moment.
        logical_annotations_params = None
      else:
        logical_annotations = maxtext_utils.get_logical_annotations(config, mesh, init_state_fn)
        logical_annotations_params = logical_annotations.params

      max_utils.print_non_trivial_mesh_axis(model.mesh)
      maxtext_utils.print_shardings_params(state_params, state_mesh_shardings_params, mesh, logical_annotations_params)

  if config.pure_nnx:
    train_state = nnx.merge(state_graphdef, state)
    model = train_state.model
  else:
    train_state = state

  return (
      init_rng,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      data_loader,
      rampup_manager,
      eval_data_iterator,
      train_state,
  )


def validate_train_config(config):
  """Validates the configuration is set correctly for 'train.py'."""

  if getattr(config, "use_dpo", False):
    raise ValueError("Legacy DPO implementation in train.py is removed. Please use post-training train_dpo.py instead.")

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


def validate_completed_steps(completed_steps: int, config_steps: int):
  """Raises RuntimeError if training has already completed up to config_steps."""
  if completed_steps >= config_steps:
    raise RuntimeError(
        f"Requested training up to step {config_steps}, but a checkpoint already exists at step {completed_steps - 1} "
        f"(which means {completed_steps} steps have been completed). "
        f"Did you mean to continue training past step {completed_steps} (you should set steps > {completed_steps}) "
        f"or to not load the checkpoint (use enable_checkpointing=False?)"
    )


def maybe_apply_dcn_throttling(config):
  """Applies programmatic traffic control (tc) bandwidth limit if configured."""
  if not config.dcn_bandwidth_limit:
    return

  interface = config.dcn_bandwidth_interface

  # Always clean up any existing traffic control rule on the interface first.
  try:
    subprocess.run(
        ["tc", "qdisc", "del", "dev", interface, "root"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    max_logging.error(f"Failed to clean up existing traffic control on {interface}: {e}")

  rate = config.dcn_bandwidth_limit
  burst = config.dcn_bandwidth_burst
  latency = config.dcn_bandwidth_latency

  max_logging.log(f"Applying tc egress limit of {rate} (burst: {burst}, latency: {latency}) on {interface}...")
  try:
    cmd = ["tc", "qdisc", "add", "dev", interface, "root", "tbf", "rate", rate, "burst", burst, "latency", latency]
    subprocess.run(cmd, check=True)
    max_logging.log("DCN Bandwidth throttling applied successfully.")
  except Exception as e:  # pylint: disable=broad-exception-caught
    max_logging.error(f"Failed to apply DCN bandwidth throttling: {e}")


def maybe_cleanup_dcn_throttling(config):
  """Cleans up traffic control (tc) rules."""
  if not config.dcn_bandwidth_limit:
    return

  interface = config.dcn_bandwidth_interface
  max_logging.log(f"Cleaning up tc egress limit on {interface}...")
  try:
    subprocess.run(
        ["tc", "qdisc", "del", "dev", interface, "root"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    max_logging.log("DCN Bandwidth throttling cleaned up successfully.")
  except Exception as e:  # pylint: disable=broad-exception-caught
    max_logging.error(f"Failed to clean up DCN bandwidth throttling: {e}")

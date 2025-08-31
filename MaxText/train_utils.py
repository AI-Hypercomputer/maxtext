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

from collections.abc import Sequence
import os
from typing import overload

from flax import nnx
import flax.linen as nn
from functools import partial
from etils import epath
import jax
from orbax import checkpoint as ocp
from jax.sharding import Mesh
from MaxText import checkpointing
from MaxText import max_logging
from MaxText import max_utils
from MaxText import maxtext_utils
from MaxText import optimizers
from MaxText import pyconfig
from MaxText.layers import quantizations
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.dpo_utils import _merge_dpo_state
from MaxText.input_pipeline.input_pipeline_interface import create_data_iterator
from MaxText.layers import models
from MaxText.utils.goodput_utils import GoodputEvent
from MaxText.utils.goodput_utils import maybe_record_goodput


@overload
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
) -> nn.Module: ...
@overload
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: nnx.Rngs,
) -> models.Transformer: ...
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: nnx.Rngs | None = None,
) -> nn.Module | models.Transformer:
  """Load a pretrained MaxText model from checkpoint.

  This function loads a model from a checkpoint.

  Args:
      config: Config object.
      devices: Sequence of devices to use for the model. If None, use all
        available devices.

  Returns:
      Transformer: The loaded model instance (only the model)

  Example:
      model = from_config(config)
  """
  devices_array = maxtext_utils.create_device_mesh(config, devices)
  mesh = Mesh(devices_array, config.mesh_axes)
  model = create_model(config, mesh, model_mode=model_mode, rngs=rngs)

  # Return only the model
  return model


def get_transformer_model(config, mesh, quant, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None):
  """Returns the transformer model based on the configuration."""
  if config.model_fsdp_ag_once:
    if rngs is not None:
      raise NotImplementedError
    else:
      return models.ZeroOneTransformer(config, mesh, quant=quant, model_mode=model_mode)
  else:
    if rngs is not None:
      return models.Transformer(config, mesh, quant=quant, rngs=rngs, model_mode=model_mode)
    else:
      return models.transformer_as_linen(config, mesh, quant=quant, model_mode=model_mode)


def create_model(config, mesh, model_mode: str = MODEL_MODE_TRAIN, rngs: nnx.Rngs | None = None):
  """Instantiates and returns the model object, sharded across the mesh."""
  # Model definition
  quant = quantizations.configure_quantization(config)
  model = get_transformer_model(config, mesh, quant, model_mode=model_mode, rngs=rngs)
  model = quantizations.maybe_quantize_model(model, config)
  return model


def create_nnx_model(config):
  """Creates a NNX model with sharded parameters, possibly loading from a checkpoint."""

  def create_model():
    init_rng = jax.random.PRNGKey(config.init_weights_seed)
    return from_config(config, rngs=nnx.Rngs(params=init_rng, dropout=1))

  abstract_model = nnx.eval_shape(create_model)
  graphdef, abstract_state = nnx.split(abstract_model)
  specs = nnx.get_partition_spec(abstract_state)
  mesh = abstract_model.mesh

  # JIT a function that creates the model state with proper sharding from the start.
  # By providing out_shardings, we instruct JAX to produce sharded output directly,
  # avoiding a large intermediate allocation on a single device.
  with nn.logical_axis_rules(config.logical_axis_rules):
    out_shardings = nn.logical_to_mesh_sharding(specs, mesh)

  @partial(jax.jit, out_shardings=out_shardings)
  def create_sharded_state():
    # This will be JIT-compiled. JAX knows the output sharding and can
    # initialize the parameters directly on the target devices in a sharded way.
    model = create_model()
    return nnx.state(model)

  with mesh:
    # Create the model with sharded parameters.
    sharded_state = create_sharded_state()
    model = nnx.merge(graphdef, sharded_state)

    if config.load_parameters_path:
      target_for_restore = jax.tree.map(
          lambda v: v.value,
          sharded_state,
          is_leaf=lambda n: isinstance(n, nnx.Variable),
      )

      try:
        ckptr = ocp.Checkpointer(
            ocp.PyTreeCheckpointHandler(
                restore_concurrent_gb=None,
                save_concurrent_gb=None,
                use_ocdbt=True,
                use_zarr3=True,
            )
        )
        # This is a memory optimization. We don't want to restore the entire checkpoint - only the params.
        # Rather than passing the entire abstract state, which could unnecessarily restore opt_state and
        # waste memory, we instead restore the params field of the checkpoint (which itself may be a dictionary
        #  containing a key named 'params').
        restore_args = ocp.checkpoint_utils.construct_restore_args(target_for_restore)
        restored = ckptr.restore(
            epath.Path(config.load_parameters_path),
            item={"params": {"params": target_for_restore}},
            transforms={},
            restore_args={"params": {"params": restore_args}},
        )
        checkpoint = restored["params"]["params"]

        if checkpoint:
          nnx.update(model, checkpoint)

      except Exception as e:
        raise ValueError(f"Checkpoint loading failed: {e}")

    return model, mesh


def create_training_tools(config, model, mesh):
  """Creates the init_rng, optimizer, learning rate schedule, and checkpoint manager."""
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)
  logger = checkpointing.setup_checkpoint_logger(config)
  if config.enable_multi_tier_checkpointing:
    checkpoint_manager = (
        checkpointing.create_orbax_emergency_replicator_checkpoint_manager(
            config.local_checkpoint_directory,
            config.local_checkpoint_period,
            mesh,
        )
    )
  elif config.enable_emergency_checkpoint:
    abstract_state, _, _ = maxtext_utils.get_abstract_state(
        model, tx, config, init_rng, mesh, is_training=True
    )
    checkpoint_manager = (
        checkpointing.create_orbax_emergency_checkpoint_manager(
            config.local_checkpoint_directory,
            config.checkpoint_dir,
            mesh,
            abstract_state,
            config.local_checkpoint_period,
            config.checkpoint_period,
            logger,
        )
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


def jit_train_step(
    config, model, state, state_mesh_shardings, data_sharding, train_step
):
  """Returns a JIT-compiled train step function, which is loaded from a file if specified in the config."""
  (
      functional_train,
      in_shardings,
      out_shardings,
      static_argnums,
      donate_argnums,
  ) = maxtext_utils.get_functional_train_with_signature(
      train_step, data_sharding, state_mesh_shardings, model, config
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


def jit_eval_step(
    config, model, state_mesh_shardings, data_sharding, eval_step
):
  """Returns a JIT-compiled eval step function."""
  (
      functional_eval,
      in_shardings,
      out_shardings,
      static_argnums,
      donate_argnums,
  ) = maxtext_utils.get_functional_eval_with_signature(
      eval_step, data_sharding, state_mesh_shardings, model, config
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
    config,
    model,
    mesh,
    state,
    state_mesh_shardings,
    train_step,
    eval_step=None,
    eval_data_iterator=None,
):
  """Returns a JIT-compiled train and eval step function."""
  data_sharding = maxtext_utils.get_input_data_sharding(config, mesh)
  p_train_step = jit_train_step(
      config, model, state, state_mesh_shardings, data_sharding, train_step
  )
  p_eval_step = None
  if eval_data_iterator:
    p_eval_step = jit_eval_step(
        config, model, state_mesh_shardings, data_sharding, eval_step
    )

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
    state: the initialized train state
  """

  with maybe_record_goodput(recorder, GoodputEvent.TPU_INIT):
    model = from_config(config, devices)
    mesh = model.mesh
    init_rng, checkpoint_manager, learning_rate_schedule, tx = (
        create_training_tools(config, model, mesh)
    )

  with maybe_record_goodput(recorder, GoodputEvent.TRAINING_PREPARATION):
    data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
    context_parallel_size = mesh.shape["context"]
    # Check if context parallelism is being used with sequence packing
    if (
        context_parallel_size > 1
        and config.packing
        and config.dataset_type != "synthetic"
    ):
      raise ValueError(
          "Context parallelism cannot be used with sequence packing except for"
          " synthetic data where packing is not applied. Either disable"
          " sequence packing (set packing=False) or disable context"
          " parallelism. Context parallelism with packing support will be added"
          " soon."
      )

    # Apply reordering wrapper to data iterators if context parallelism is enabled
    with mesh:
      if context_parallel_size > 1 and config.context_parallel_load_balance:
        data_iterator = map(
            max_utils.get_reorder_callable(context_parallel_size), data_iterator
        )
        if eval_data_iterator:
          eval_data_iterator = map(
              max_utils.get_reorder_callable(context_parallel_size),
              eval_data_iterator,
          )

    state, _, state_mesh_shardings, data_iterator = (
        maxtext_utils.setup_training_state(
            model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
        )
    )

    # TODO(aireenmei, hengtaoguo): support sharding in vit for multimodal
    if not config.using_pipeline_parallelism and not config.use_multimodal:
      # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage
      maxtext_utils.assert_params_sufficiently_sharded(
          state.params, mesh, config.sharding_tolerance
      )

    if config.use_dpo:
      abstract_state, _, _ = maxtext_utils.get_abstract_state(
          model, tx, config, init_rng, mesh, is_training=True
      )
      max_logging.log(
          "Restoring reference parameters for DPO from"
          f" '{os.path.join(str(config.checkpoint_dir), str(0))}'"
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
            "Could not restore reference parameters for DPO from"
            f" '{os.path.join(str(config.checkpoint_dir), str(0))}'"
        )

  return (
      init_rng,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  )


def validate_train_config(config):
  """Validates the configuration is set correctly for 'train.py'."""

  assert config.run_name, "Erroring out, need a real run_name"
  if config.dataset_path and not config.dataset_path.startswith("gs://"):
    max_logging.log(
        "WARNING: 'dataset_path' might be pointing your local file system"
    )
  if not config.base_output_directory.startswith("gs://"):
    max_logging.log(
        "WARNING: 'base_output_directory' might be pointing your local file"
        " system"
    )
  assert (
      config.steps > 0
  ), "You must set steps or learning_rate_schedule_steps to a positive integer."

  if config.quantization in ("fp8", "nanoo_fp8"):
    # pylint: disable=line-too-long
    assert config.gradient_accumulation_steps == 1, (
        "fp8 can't be used with gradient_accumulation_steps right now. Please"
        " use other quantization or set gradient_accumulation_steps to 1"
    )

  # Check if GPU Flash Attention is being used with sequence packing
  if (
      config.attention == "cudnn_flash_te"
      and config.packing
      and config.dataset_type != "synthetic"
  ):
    raise ValueError(
        "cudnn_flash_te only supports BSHD format. The THD (seq packing)"
        " support is going to be available in Transformer Engine 2.0 release."
        " Please disable sequence packing (set packing=False) or use a"
        " different attention mechanism. With synthetic data, the format is not"
        " important as packing is not applied."
    )

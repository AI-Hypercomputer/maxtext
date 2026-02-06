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

"""
Save a Cross Ahead of Time Compiled (XAOT) version of train.py's train step
Generates shaped versions of state and data without ever constructing them, so its possible
to compile with target hardware (e.g. hundreds/thousands of chips), without using the hardware.
This helpfully detects if your configuration would run into memory problems (OOM) on the target hardware,
before having to use the target hardware - you will see the same OOM error message during this compilation
as you would on the target hardware.
"""

from typing import Sequence
import os
import pickle
import functools

from absl import app

import jax
from jax.experimental.topologies import get_topology_desc
from jax.sharding import Mesh, AxisType
from jax.experimental.serialize_executable import serialize

from flax.linen import partitioning as nn_partitioning

from MaxText import accelerator_to_spec_map
from MaxText import train
from MaxText import optimizers
from MaxText import pyconfig
from MaxText import sharding
from MaxText.common_types import MODEL_MODE_TRAIN, ShardMode
from maxtext.models import models
from maxtext.layers import quantizations
from maxtext.utils import gcs_utils
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.trainers.diloco import diloco

# pylint: disable=too-many-positional-arguments

Transformer = models.transformer_as_linen


def validate_config(config):
  """Validates the config is is setup correctly to compile, returning a useful error message if not."""
  assert (
      config.compile_topology != ""
  ), "You must pass your desired target hardware in compile_topology, e.g. compile_topology=v5e-256"
  assert config.compile_topology_num_slices > 0, "You must set compile_topology_num_slices to a positive integer"


def get_topology_mesh(config):
  """Get the target hardware devices, and create configured mesh with them"""
  target_hardware = accelerator_to_spec_map.get_system_characteristics(config.compile_topology)
  if target_hardware.platform == "gpu":
    # Disable sharded autotuning. This is an optimization to distribute
    # autotuning across the fleet, but can cause hangs with AoT compilation.
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_shard_autotuning=false"
    jax.config.update("mock_num_gpu_processes", config.compile_topology_num_slices)
    topology_devices = jax.devices()
  else:
    topology_devices = get_topology_desc(
        platform=target_hardware.platform,
        topology_name=target_hardware.topology_name,
        chip_config_name=target_hardware.chip_config_name,
        chips_per_host_bounds=target_hardware.chips_per_host_bounds,
        num_slices=config.compile_topology_num_slices,
        wrap=target_hardware.wrap,
    ).devices
  if config.shard_mode == ShardMode.EXPLICIT:
    jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)
  topology_device_mesh = maxtext_utils.create_device_mesh(config, topology_devices)
  mesh_axis_type = AxisType.Explicit if config.shard_mode == ShardMode.EXPLICIT else AxisType.Auto
  topology_mesh = Mesh(topology_device_mesh, config.mesh_axes, axis_types=(mesh_axis_type,) * len(config.mesh_axes))
  return topology_mesh


def get_shaped_inputs(topology_mesh, config):
  """Get shaped abstractions of inputs to train_step: state, batch and rng"""
  # Construct the model and optimizer to get shaped versions of the state
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, topology_mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
  # The learning_rate_schedule is baked into the compiled object.
  learning_rate_schedule = maxtext_utils.create_learning_rate_schedule(config)
  # pass in model for muon
  tx = optimizers.get_optimizer(config, learning_rate_schedule, model)

  # Shaped RNG keys
  _, example_rng = jax.random.split(jax.random.PRNGKey(0), 2)
  shaped_rng = jax.ShapeDtypeStruct(example_rng.shape, example_rng.dtype)

  # Shaped state
  abstract_state, _, state_mesh_shardings = maxtext_utils.get_abstract_state(
      model, tx, config, example_rng, topology_mesh
  )

  # unsharded logical annotations
  logical_annotations = maxtext_utils.get_logical_annotations(model, tx, config, example_rng, topology_mesh)

  # Shaped batch
  shaped_batch = maxtext_utils.get_shaped_batch(config)

  shaped_train_args = (abstract_state, shaped_batch, shaped_rng)
  shaped_train_kwargs = {}
  return shaped_train_args, shaped_train_kwargs, state_mesh_shardings, logical_annotations, model


def jit_and_compile(
    func,
    func_input_args,
    func_input_kwargs,
    mesh,
    in_shardings,
    out_shardings,
    static_argnums,
    donate_argnums,
    config,
    logical_axis_rules,
):
  """Jit, lower, and compile func."""
  with jax.set_mesh(mesh), logical_axis_rules:
    jitted = jax.jit(
        func,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )
    maxtext_utils.maybe_dump_jaxpr(config, jitted, func_input_args)
    lowered = jitted.lower(*func_input_args, **func_input_kwargs)
  compiled = lowered.compile()
  return compiled


def save_compiled(compiled, save_name):
  """Serialize and save the compiled function."""
  serialized, _, _ = serialize(compiled)
  with open(save_name, "wb") as f:
    pickle.dump(serialized, f)


def is_oom(argv: Sequence[str]) -> bool:
  """Function returns a boolean indicating whether OOM happens"""
  # Parse and validate configuration
  config = pyconfig.initialize(argv)
  validate_config(config)

  # Create target mesh
  topology_mesh = get_topology_mesh(config)

  # Print system information after building the compile topology to avoid
  # prematurely initializing the backend.
  max_utils.print_system_information()

  # Get shaped inputs
  (
      shaped_train_args,
      shaped_train_kwargs,
      state_mesh_shardings,
      _,
      model,
  ) = get_shaped_inputs(topology_mesh, config)

  # Get data sharding
  data_sharding = sharding.get_input_data_sharding(config, topology_mesh)

  # Get function to compile and shardings
  func_to_compile, in_shard, out_shard, static_argnums, donate_argnums = (
      maxtext_utils.get_functional_train_with_signature(
          train.train_step, data_sharding, state_mesh_shardings, model, config
      )
  )

  try:
    _ = jit_and_compile(
        func_to_compile,
        shaped_train_args,
        shaped_train_kwargs,
        topology_mesh,
        in_shard,
        out_shard,
        static_argnums,
        donate_argnums,
        config,
        nn_partitioning.axis_rules(config.logical_axis_rules),
    )
    return False
  except Exception as e:
    # return true if OOM error happens
    # OOM error looks like
    # jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Allocation ...
    # jax.errors.JaxRuntimeError: INTERNAL: RET_CHECK failure ...
    message = str(e).lower()
    if "resource_exhausted" in message or "hbm" in message:
      return True
    raise e


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  )
  print("Starting train_compile.py...", flush=True)

  # Parse and validate configuration
  config = pyconfig.initialize(argv)
  validate_config(config)

  # Create target mesh
  topology_mesh = get_topology_mesh(config)

  # Print system information after building the compile topology to avoid
  # prematurely initializing the backend.
  max_utils.print_system_information()

  # Get shaped inputs
  (
      shaped_train_args,
      shaped_train_kwargs,
      state_mesh_shardings,
      logical_annotations,
      model,
  ) = get_shaped_inputs(topology_mesh, config)

  # Get data sharding
  data_sharding = sharding.get_input_data_sharding(config, topology_mesh)
  if config.enable_diloco:
    # Build abstract DiLoCo state and shardings for AOT compilation
    abstract_state = shaped_train_args[0]
    diloco_state, state_mesh_shardings, inner_state_shardings = diloco.build_abstract_diloco_state(
        config, abstract_state, state_mesh_shardings, topology_mesh
    )
    shaped_train_args = (diloco_state, shaped_train_args[1], shaped_train_args[2])

    # Wrap train_step with diloco
    train_step_partial = functools.partial(train.train_step, model, config, inner_state_shardings, None)
    train_step_fn = diloco.build_diloco_train_step(config, train_step_partial)

    # For DiLoCo, the train_step_fn is already fully wrapped and takes (state, batch, prng)
    func_to_compile = train_step_fn
    func_to_compile.__name__ = "train_step"
    in_shard = (state_mesh_shardings, data_sharding, None)  # State, batch, rng
    out_shard = (state_mesh_shardings, None)  # State, metrics
    static_argnums = ()
    donate_argnums = 0
  else:
    # Get function to compile and shardings
    func_to_compile, in_shard, out_shard, static_argnums, donate_argnums = (
        maxtext_utils.get_functional_train_with_signature(
            train.train_step, data_sharding, state_mesh_shardings, model, config
        )
    )

  # print weights sharding info under debug sharding mode
  if config.debug_sharding:
    max_utils.print_non_trivial_mesh_axis(topology_mesh)
    maxtext_utils.print_shardings_params(
        shaped_train_args[0].params,
        state_mesh_shardings.params,
        topology_mesh,
        logical_annotations.params,
    )

  # Compile
  print("Jitting and compiling train step...", flush=True)
  compiled = jit_and_compile(
      func_to_compile,
      shaped_train_args,
      shaped_train_kwargs,
      topology_mesh,
      in_shard,
      out_shard,
      static_argnums,
      donate_argnums,
      config,
      nn_partitioning.axis_rules(config.logical_axis_rules),
  )
  print("Jitting and compilation complete!", flush=True)

  # Serialize and save the compiled object
  if config.compiled_trainstep_file != "":
    print("Saving compiled object...")
    save_compiled(compiled, config.compiled_trainstep_file)
    print(f"Successfully saved compiled object as {config.compiled_trainstep_file}")
  print("Finished train_compile.py successfully!", flush=True)
  print(f"Cost analysis: {compiled.cost_analysis()}")
  print(f"Memory analysis: {compiled.memory_analysis()}")

  # Dump HLO if requested
  if config.dump_hlo:
    gcs_utils.upload_dump(
        config.dump_hlo_local_dir,
        config.dump_hlo_gcs_dir,
        module_name=config.dump_hlo_module_name,
        delete_local_after=config.dump_hlo_delete_local_after,
        all_host_upload=config.dump_hlo_upload_all,
    )


if __name__ == "__main__":
  app.run(main)

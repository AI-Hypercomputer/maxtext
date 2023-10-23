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

""" 
Save a Cross Ahead of Time Compiled (XAOT) version of train.py's train step
Generates shaped versions of state and data without ever constructing them, so its possible
to compile with target hardware (e.g. hundreds/thousands of chips), without using the hardware.
This helpfully detects if your configuration would run into memory problems (OOM) on the target hardware,
before having to use the target hardware - you will see the same OOM error message during this compilation
as you would on the target hardware.
"""

import max_utils
from layers import Transformer
import pyconfig
import jax
import numpy as np
from typing import Sequence
from jax.experimental.topologies import get_topology_desc
from absl import app
from jax.sharding import Mesh
from jax.experimental.serialize_executable import serialize, deserialize_and_load
import pickle
from hardware_map import UserFacingNameToSystemCharacteristics, get_system_characteristics

import train

def validate_config(config):
  """ Validates the config is is setup correctly to compile, returning a useful error message if not. """
  assert config.compile_topology != '', "You must pass your desired target hardware in compile_topology, e.g. compile_topology=v5e-256"

def get_topology_mesh(config):
  #target_hardware = UserFacingNameToSystemCharacteristics[config.compile_topology]
  target_hardware = get_system_characteristics(config.compile_topology)
  topology_devices = get_topology_desc(
      platform=target_hardware.platform,
      topology_name=target_hardware.topology_name,
      chip_config_name=target_hardware.chip_config_name,
      chips_per_host_bounds=target_hardware.chips_per_host_bounds,
      num_slices=config.compile_topology_num_slices,
  ).devices
  topology_device_mesh = max_utils.create_device_mesh(config, topology_devices)
  topology_mesh = Mesh(topology_device_mesh, config.mesh_axes)
  return topology_mesh

def get_shaped_inputs(topology_mesh, config):
  model = Transformer(config, topology_mesh)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(config) # WARNING!!! This learning_rate_schedule is what is really used at runtime
  tx = max_utils.get_optimizer(config, learning_rate_schedule)
  shaped_train_args, shaped_train_kwargs, state_mesh_annotations = max_utils.gen_shaped_input_data(model, tx, config, topology_mesh)
  return shaped_train_args, shaped_train_kwargs, state_mesh_annotations, model

def get_train_step_and_shardings(model, config, topology_mesh, state_mesh_annotations):
    func_to_compile = train.get_functional_train_step(train.train_step, model, config)
    in_shardings, out_shardings, static_argnums, donate_argnums = max_utils.get_train_shardings(topology_mesh, state_mesh_annotations, config)
    return func_to_compile, in_shardings, out_shardings, static_argnums, donate_argnums

def jit_and_compile(func, func_input_args, func_input_kwargs, mesh, in_shardings, out_shardings, static_argnums, donate_argnums):
    # Jit, lower, and compile func, using topology devices
    with mesh:
        jitted = jax.jit(
          func,
          in_shardings=in_shardings,
          out_shardings=out_shardings,
          static_argnums=static_argnums,
          donate_argnums=donate_argnums
        )
        lowered = jitted.lower(*func_input_args, **func_input_kwargs)
    compiled = lowered.compile()
    return compiled

def save_compiled(compiled, save_name):
    # Serialize and save the compiled object
    serialized, in_tree, out_tree = serialize(compiled)
    with open(save_name, "wb") as f:
        pickle.dump(serialized, f)

def main(argv: Sequence[str]) -> None:
  print("Starting train_compile.py...", flush=True)

  # Parse and validate configuration
  pyconfig.initialize(argv)
  config = pyconfig.config
  validate_config(config)

  # Create target mesh
  topology_mesh = get_topology_mesh(config)

  # Get shaped inputs
  shaped_train_args, shaped_train_kwargs, state_mesh_annotations, model = get_shaped_inputs(topology_mesh, config)

  # Get function to compile and shardings
  func_to_compile, in_shardings, out_shardings, static_argnums, donate_argnums = get_train_step_and_shardings(model, config, topology_mesh, state_mesh_annotations)

  # Compile
  print("Jitting and compiling train step...", flush=True)
  compiled = jit_and_compile(func_to_compile, shaped_train_args, shaped_train_kwargs, topology_mesh, in_shardings, out_shardings, static_argnums, donate_argnums)
  print("Jitting and compilation complete!", flush=True)

  # Serialize and save the compiled object
  if config.compiled_save_file != '':
    print(f"Saving compiled object...")
    save_compiled(compiled, config.compiled_save_file)
    print(f"Successfully saved compiled object as {config.compiled_save_file}")
  print("Finished train_compile.py successfully!", flush=True)

if __name__ == "__main__":
  app.run(main)
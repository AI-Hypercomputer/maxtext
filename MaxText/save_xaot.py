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
to compile with target hardware (e.g. a large cluster), without using the hardware.
"""

import max_utils
import pyconfig
import jax
import numpy as np
from typing import Sequence
from jax.experimental.topologies import get_topology_desc
from absl import app
from jax.sharding import Mesh
from jax.experimental.serialize_executable import serialize, deserialize_and_load
import pickle

import train

# Ideally this map exists either in XAOT or we can programmatically call it from elsewhere
# so the chip layout is consistent across cloud technologies
def get_topology_mesh(config):
  if config.topology=='v4-8':
    topology_devices = get_topology_desc(
        platform='tpu',
        topology_name=f'v4:2x2x1',
        chip_config_name='megacore',
        chips_per_host_bounds=(2, 2, 1),
        num_slices=config.topology_num_slices,
    ).devices
  elif config.topology=='v4-16':
    print("excitement v4-16", flush=True)
    topology_devices = get_topology_desc(
    platform='tpu',
    topology_name=f'v4:2x2x2',
    chip_config_name='megacore',
    chips_per_host_bounds=(2, 2, 2),
    num_slices=config.topology_num_slices,
).devices
  elif config.topology == 'v5e-16':
    print("excitement v5e-16")
    topology_devices = get_topology_desc(
        platform='tpu',
        topology_name=f'v5e:2x2',
        chips_per_host_bounds=(2, 2, 1),
        num_slices=config.topology_num_slices,
    ).devices
  elif config.topology == 'v5e-256':
    print("excitement v5e-256")
    topology_devices = get_topology_desc(
        platform='tpu',
        topology_name=f'v5e:8x8',
        chips_per_host_bounds=(8, 8, 1),
        num_slices=config.topology_num_slices,
    ).devices
  topology_device_mesh = max_utils.create_device_mesh(config, topology_devices)
  topology_mesh = Mesh(topology_device_mesh, config.mesh_axes)
  return topology_mesh

def xaot_compile_and_save(func, compiled_name, func_input_args, func_input_kwargs, in_shardings, out_shardings, static_argnums, donate_argnums, mesh):
    def jit_and_compile(func, func_input_args, func_input_kwargs, mesh, in_shardings, out_shardings):
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
        return jitted, lowered, compiled

    def save_compiled(compiled, save_name):
        # Serialize and save the compiled object
        serialized, in_tree, out_tree = serialize(compiled)
        with open(save_name, "wb") as f:
            pickle.dump(serialized, f)
    print("Jitting train step so it can be saved...", flush=True)
    jitted, lowered, compiled = jit_and_compile(func, func_input_args, func_input_kwargs, mesh, in_shardings, out_shardings)
    print("Jitting train step!!!", flush=True)
    save_compiled(compiled, compiled_name) # Serialize and save the compiled object

def save_train_xaot(config):
    print("Saving compiled xaot...", flush=True)
    topology_mesh = get_topology_mesh(config)
    model = max_utils.get_model(config, topology_mesh)
    learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
    tx = max_utils.get_optimizer(config, learning_rate_schedule)

    func_to_xaot = train.get_partial_train_step_func(train.train_step, model, config)
    shaped_train_args, shaped_train_kwargs, state_mesh_annotations = max_utils.gen_shaped_input_data(model, tx, config, topology_mesh)
    in_shardings, out_shardings = max_utils.get_shardings(topology_mesh, state_mesh_annotations, config)
    static_argnums=()
    donate_argnums=0
    xaot_compile_and_save(func_to_xaot, config.xaot_save_name, shaped_train_args, shaped_train_kwargs, in_shardings, out_shardings, static_argnums, donate_argnums, topology_mesh)
    print("Saved compiled xaot!!!", flush=True)



def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  save_train_xaot(pyconfig.config)

if __name__ == "__main__":
  app.run(main)
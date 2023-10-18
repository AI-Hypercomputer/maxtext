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
This also saves the config.
Generates shaped versions of state and data without ever constructing them, so its possible
to compile with target hardware (e.g. a large cluster), without using the hardware.
"""

import max_utils
import pyconfig
import jax
import numpy as np
from typing import Sequence
from absl import app

import train

def save_compiled_full(func, compiled_name, func_input_args, func_input_kwargs, in_shardings, out_shardings, static_argnums, donate_argnums, mesh):
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
    topology_mesh = max_utils.get_topology_mesh(config)
    model = max_utils.get_model(config, topology_mesh)
    learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
    tx = max_utils.get_optimizer(config, learning_rate_schedule)

    func_to_xaot = train.get_partial_train_step_func(train.train_step, model, config)
    shaped_train_args, shaped_train_kwargs, state_mesh_annotations = max_utils.gen_shaped_input_data(model, tx, config, topology_mesh)
    in_shardings, out_shardings = max_utils.get_shardings(topology_mesh, state_mesh_annotations, config)
    static_argnums=()
    donate_argnums=0
    compiled_name='go me'
    save_compiled_full(func_to_xaot, compiled_name, shaped_train_args, shaped_train_kwargs, in_shardings, out_shardings, static_argnums, donate_argnums, topology_mesh)
    print("Saved compiled xaot!!!", flush=True)



def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  save_train_xaot(pyconfig.config)

if __name__ == "__main__":
  app.run(main)
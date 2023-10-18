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
from typing import Sequence
from absl import app

def save_xaot(config):
    print("Saving compiled xaot...", flush=True)
    topology_mesh = max_utils.get_topology_mesh(config)
    # topology_data_iterator, _ = create_data_iterator_with_tokenizer(config, topology_mesh)
    # func_input_args, func_input_kwargs, state_mesh_annotations = max_utils.gen_input_data(model, tx, config, init_rng, topology_mesh, data_iterator)
    # in_shardings, out_shardings = max_utils.get_shardings(topology_mesh, state_mesh_annotations, config)
    # static_argnums=(0,1)
    # donate_argnums=2
    # max_utils.save_compiled_full(train_step, compiled_name, func_input_args, func_input_kwargs, in_shardings, out_shardings, static_argnums, donate_argnums, topology_mesh)
    # print("Saved compiled xaot!!!", flush=True)



def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  save_xaot(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
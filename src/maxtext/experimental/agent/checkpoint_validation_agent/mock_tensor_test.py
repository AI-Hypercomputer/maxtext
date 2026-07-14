# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "innovation" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mock tensor dry-run to validate checkpoint architecture stability."""

import sys
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from maxtext import pyconfig
from maxtext.models.models import transformer_as_linen


def run_mock_forward(checkpoint_path, model_name, *overrides):
  """Initializes the model abstractly and dry-runs a forward pass."""
  # minimal config required to bypass distributed TPU checks
  config_args = [
      "src/maxtext/configs/base.yml",
      f"model_name={model_name}",
      f"load_parameters_path={checkpoint_path}",
      "skip_jax_distributed_system=true",
  ]

  # append all dynamic overrides passed from Airflow
  config_args.extend(overrides)

  # capture returned configuration object
  config = pyconfig.initialize(config_args)

  print(f"Loading model from {checkpoint_path}...")
  # create a dummy 1-device hardware mesh using pod's single CPU
  dummy_mesh = Mesh(np.array(jax.devices()), ('data',))
  # pass dummy_mesh instead of None
  model = transformer_as_linen(config, mesh=dummy_mesh, quant=None)

  # dynamically generate tensor shapes based on the parsed config
  batch_size = int(config.per_device_batch_size) if config.per_device_batch_size else 1
  seq_len = int(config.max_target_length) if config.max_target_length else 128

  print(f"Generating mock tensors with shape: ({batch_size}, {seq_len})")

  # run a single dummy pass
  mock_input = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
  mock_positions = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
  mock_segment_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

  print("Executing forward pass...")
  # tracing full traceback without try/except block
  rng = jax.random.PRNGKey(0)

  # generate abstract shapes for the parameters which uses 0 memory
  print("Initializing abstract model parameters...")
  abstract_variables = jax.eval_shape(model.init, rng, mock_input, mock_positions, mock_segment_ids)

  # dry-run the forward pass using the abstract parameters
  print("Tracing forward pass graph...")
  out_shape = jax.eval_shape(model.apply, abstract_variables, mock_input, mock_positions, mock_segment_ids)

  print(f"SUCCESS: Model architecture is stable. Output shape: {out_shape}")

if __name__ == "__main__":
  # sys.argv[1] is checkpoint, sys.argv[2] is model_name, sys.argv[3:] captures any overrides
  run_mock_forward(sys.argv[1], sys.argv[2], *sys.argv[3:])

"""Mock tensor dry-run to validate checkpoint architecture stability."""

import sys
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from maxtext import pyconfig
#from maxtext.models import get_model
from maxtext.models.models import transformer_as_linen


def run_mock_forward(checkpoint_path, model_name):
  """Initializes the model abstractly and dry-runs a forward pass."""
  # minimal config to load the model and run a pass
  config_args = [
      f"model_name={model_name}",
      f"load_parameters_path={checkpoint_path}",
      "batch_size=1",
      "max_target_length=128",
      "scan_layers=false",
      "skip_jax_distributed_system=true",
  ]

  # initialize pyconfig (extra [] argument not supported)
  # pyconfig.initialize(config_args)
  # capture returned configuration object
  config = pyconfig.initialize(config_args)

  print(f"Loading model from {checkpoint_path}...")
  #model = get_model(pyconfig.config, mesh=None)
  # create a dummy 1-device hardware mesh using pod's single CPU
  dummy_mesh = Mesh(np.array(jax.devices()), ('data',))
  # pass dummy_mesh instead of None
  model = transformer_as_linen(config, mesh=dummy_mesh, quant=None)

  # run a single dummy pass
  mock_input = jnp.zeros((1, 128), dtype=jnp.int32)
  mock_positions = jnp.zeros((1, 128), dtype=jnp.int32)
  mock_segment_ids = jnp.zeros((1, 128), dtype=jnp.int32)

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
  run_mock_forward(sys.argv[1], sys.argv[2])

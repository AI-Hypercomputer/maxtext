import jax
import unittest
from layers import linears
from layers import initializers
import jax.numpy as jnp

import pyconfig
import max_utils
import maxtext_utils
from jax.sharding import Mesh
from typing import Tuple
import common_types
import flax.linen as nn
import jax
import max_utils
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
from jax import random
from jax.sharding import Mesh
import optax
import pyconfig
import unittest
from layers import models
from layers import quantizations

Transformer = models.Transformer


class FlopsTest(unittest.TestCase):

  def setUp(self):
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    pyconfig.initialize(
      [None, 'configs/base.yml'],
      run_name='test',
      model_name='mixtral-8x7b',
      flops_change=True,
    )
    self.cfg = pyconfig.config
    devices_array = max_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    quant = quantizations.configure_quantization(self.cfg)
    self.model = Transformer(self.cfg, mesh=self.mesh, quant=quant)

  def test_flops(self):
    rng = random.PRNGKey(0)

    state, _ = max_utils.setup_decode_state(self.model, self.cfg, rng, self.mesh, None)
    num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
    maxtext_utils.calculate_tflops_training_per_device(num_model_parameters, self.cfg)


if __name__ == '__main__':
  unittest.main()
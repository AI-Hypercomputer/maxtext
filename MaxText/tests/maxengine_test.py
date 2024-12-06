"""
Copyright 2024 Google LLC

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

""" Tests for the maxengine """

import jax
from jax import numpy as jnp
import numpy as np
import unittest
import pyconfig
from maxengine import MaxEngine


class MaxEngineTest(unittest.TestCase):
  """Tests for MaxEngine."""

  # TODO: add unit test for the MaxEngine.

  def test_stack_and_unstack_prefill_cache(self):
    pyconfig.initialize(
        [None, "configs/base.yml"],
        enable_checkpointing=False,
        stack_prefill_result_cache=True,
    )
    config = pyconfig.config
    engine = MaxEngine(config, jax.devices())
    num_layers = engine.config.num_decoder_layers
    input = {
        "decoder": {},
    }
    for i in range(num_layers):
      input["decoder"][f"layers_{i}"] = {
          "a": jnp.ones((1, 10)),
          "b": jnp.ones((1, 9)),
      }

    expected_stacked = {
        "a": jnp.ones((num_layers, 1, 10)),
        "b": jnp.ones((num_layers, 1, 9)),
    }
    got_stacked = engine._maybe_stack_prefill_result_cache(input)
    jax.tree.map(np.testing.assert_array_equal, got_stacked, expected_stacked)

    got_unstacked = engine._maybe_unstack_prefill_result_cache(got_stacked)
    jax.tree.map(np.testing.assert_array_equal, got_unstacked, input)


if __name__ == "__main__":
  unittest.main()

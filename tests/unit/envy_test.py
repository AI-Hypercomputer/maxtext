# Copyright 2026 Google LLC
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

"""Tests for Envy MoE."""

import sys
import unittest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN, MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE
from maxtext.layers import quantizations
from maxtext.models import models
from maxtext.utils import maxtext_utils
from tests.utils.test_helpers import get_test_config_path
import numpy as np


class EnvyTest(unittest.TestCase):
  """Compile and verification tests for Envy MoE."""

  def setUp(self):
    super().setUp()
    self.cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        run_name="test",
        enable_checkpointing=False,
        model_name="envy-test",
        dtype="float32",
        per_device_batch_size=1.0 / jax.device_count(),
    )
    self.rng = jax.random.PRNGKey(123)

    devices_array = maxtext_utils.create_device_mesh(self.cfg, devices=[jax.devices()[0]])
    mesh = Mesh(devices_array, self.cfg.mesh_axes)
    quant = quantizations.configure_quantization(self.cfg)
    self.model = models.transformer_as_linen(config=self.cfg, mesh=mesh, quant=quant, model_mode=MODEL_MODE_TRAIN)
    self.example_batch = {
        "inputs": jnp.array([[5, 12, 18, 3, 22]], dtype=jnp.int32),
        "inputs_position": jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32),
        "inputs_segmentation": jnp.array([[1, 1, 1, 1, 1]], dtype=jnp.int32),
        "targets": jnp.array([[12, 18, 3, 22, 1]], dtype=jnp.int32),
        "targets_position": jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32),
        "targets_segmentation": jnp.array([[1, 1, 1, 1, 0]], dtype=jnp.int32),
    }

  def test_model_initialization_and_forward(self):
    """Verifies that the Envy model initializes and computes a training forward pass without error."""
    model_vars = self.model.init(
        {"params": self.rng, "aqt": self.rng},
        self.example_batch["inputs"],
        self.example_batch["inputs_position"],
        enable_dropout=False,
    )

    logits, _ = self.model.apply(
        model_vars,
        self.example_batch["inputs"],
        self.example_batch["inputs_position"],
        decoder_segment_ids=self.example_batch["inputs_segmentation"],
        enable_dropout=False,
        rngs={"dropout": self.rng, "aqt": self.rng},
        mutable="intermediates",
    )

    self.assertEqual(logits.shape, (1, 5, self.cfg.vocab_size))
    # Make sure logits are not all zeros or NaNs
    self.assertFalse(np.isnan(logits).any())
    self.assertFalse((logits == 0).all())


if __name__ == "__main__":
  unittest.main()

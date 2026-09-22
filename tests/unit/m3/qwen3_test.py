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

"""Unit tests for Qwen3 dense model in m3 format."""

import unittest
from absl.testing import absltest
import jax
import jax.numpy as jnp

from maxtext.common.common_types import (
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
)
from maxtext.configs import pyconfig
from maxtext.m3.models.qwen3 import Qwen3Model, create_qwen3_model
from maxtext.utils import model_creation_utils


class Qwen3M3Test(unittest.TestCase):
  """Unit tests for Qwen3 M3 model construction and execution."""

  def setUp(self):
    """Sets up test configuration and device mesh."""
    super().setUp()
    self.config = pyconfig.initialize(
        [
            None,
            "src/maxtext/configs/base.yml",
            "model_name=qwen3-0.6b",
            "use_m3_model=true",
            "override_model_config=true",
            "num_decoder_layers=2",
            "emb_dim=64",
            "mlp_dim=128",
            "num_query_heads=4",
            "num_kv_heads=2",
            "head_dim=16",
            "vocab_size=256",
            "max_prefill_predict_length=16",
            "max_target_length=32",
            "scan_layers=false",
            "weight_dtype=bfloat16",
        ]
    )
    devices = jax.devices()[:1]
    self.mesh = jax.sharding.Mesh(devices, ("data",))

  def test_model_creation_utils_routing(self):
    """Verifies that model_creation_utils.create_model routes to Qwen3Model when use_m3_model=true."""
    with jax.set_mesh(self.mesh):
      model = model_creation_utils.create_model(self.config, self.mesh)
      self.assertIsInstance(model, Qwen3Model)

  def test_qwen3_forward_pass_train(self):
    """Verifies train forward pass executes and produces correct logits shape."""
    with jax.set_mesh(self.mesh):
      model = create_qwen3_model(self.config, self.mesh, model_mode=MODEL_MODE_TRAIN)
      batch_size, seq_len = 2, 8
      tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))
      logits = model(tokens, positions, model_mode=MODEL_MODE_TRAIN)
      self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))

  def test_qwen3_forward_pass_prefill(self):
    """Verifies prefill forward pass executes and returns prompt logits."""
    with jax.set_mesh(self.mesh):
      model = create_qwen3_model(self.config, self.mesh, model_mode=MODEL_MODE_PREFILL)
      batch_size, seq_len = 1, 4
      tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))
      logits = model(tokens, positions, model_mode=MODEL_MODE_PREFILL, slot=0)
      self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))


if __name__ == "__main__":
  absltest.main()

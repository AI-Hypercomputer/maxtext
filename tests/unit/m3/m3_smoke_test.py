# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke test for m3 model registry routing and forward pass under Blaze."""

from types import SimpleNamespace

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.m3 import models as m3_models
from maxtext.m3.core.interfaces import CausalLMOutput
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.models.qwen3.modeling_qwen3 import Qwen3Model
from maxtext.utils import model_creation_utils
from tests.utils import m3_test_utils


class M3SmokeTest(absltest.TestCase):
  """Smoke test verifying m3 registry routing and forward execution under Blaze."""

  def setUp(self):
    super().setUp()
    self.mesh = create_mesh(devices=jax.devices()[:1])

  def test_qwen3_routing_and_forward_pass(self):
    """Verifies use_m3_model=True constructs Qwen3Model and executes forward pass."""
    cfg = m3_test_utils.tiny_config("qwen3-0.6b")
    with jax.set_mesh(self.mesh):
      model = model_creation_utils.create_model(cfg, self.mesh, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
      self.assertIsInstance(model, Qwen3Model)

      batch_size, seq_len = 2, 8
      tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
      positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (batch_size, seq_len))
      output = model(tokens, positions, model_mode=MODEL_MODE_TRAIN)
      self.assertIsInstance(output, CausalLMOutput)
      self.assertEqual(output.logits.shape, (batch_size, seq_len, cfg.vocab_size))

  def test_unregistered_model_raises(self):
    with self.assertRaisesRegex(ValueError, "does not support the m3 backend"):
      m3_models.create_model(SimpleNamespace(model_name="not-a-model"), self.mesh, rngs=nnx.Rngs(0))


if __name__ == "__main__":
  absltest.main()

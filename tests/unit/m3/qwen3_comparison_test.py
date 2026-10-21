# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parity of m3 Qwen3 against legacy MaxText Qwen3.

Both models are built from one seed on one mesh and legacy's weights are
copied into m3. Checked: parameter paths, shapes and dtypes; forward logits;
training loss and per-parameter gradients. The path check also runs at
qwen3-0.6b's real dimensions, which `override_model_config` would otherwise
mask.
"""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from maxtext.m3.core.interfaces import CausalLMOutput
from maxtext.m3.models.qwen3.modeling_qwen3 import Qwen3Model
from maxtext.utils import model_creation_utils
from tests.utils import m3_test_utils as u


@pytest.mark.tpu_backend
@pytest.mark.tpu_only
class Qwen3ComparisonTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = u.legacy_mesh(u.tiny_config("qwen3-0.6b"))
    self.enter_context(jax.set_mesh(self.mesh))

  def test_create_model_routes_to_m3(self):
    cfg = u.tiny_config("qwen3-0.6b")
    model = model_creation_utils.create_model(cfg, self.mesh, rngs=nnx.Rngs(0))
    self.assertIsInstance(model, Qwen3Model)
    batch = u.random_batch(cfg)
    output = model(batch["inputs"], batch["positions"], batch["segment_ids"])
    self.assertIsInstance(output, CausalLMOutput)
    self.assertEqual(output.logits.shape, (2, 8, cfg.vocab_size))
    self.assertIsNone(output.aux_loss)

  def test_parameters_match_at_full_size(self):
    legacy, m3, _ = u.build_pair("qwen3-0.6b", self.mesh, full_size=True)
    u.assert_params_match(self, legacy, m3)

  def test_parameters_match_with_untied_head(self):
    legacy, m3, _ = u.build_pair("qwen3-0.6b", self.mesh, logits_via_embedding=False)
    u.assert_params_match(self, legacy, m3)
    self.assertIn(("decoder", "logits_dense", "kernel"), u.flat_params(m3))

  def test_forward_logits_match(self):
    legacy, m3, cfg = u.build_pair("qwen3-0.6b", self.mesh)
    batch = u.random_batch(cfg)
    logits_legacy = u.legacy_logits(legacy, batch)
    logits_m3 = m3(batch["inputs"], batch["positions"], batch["segment_ids"]).logits
    self.assertLess(float(jnp.max(jnp.abs(logits_legacy - logits_m3))), 1e-4)

  def test_loss_and_gradients_match(self):
    legacy, m3, cfg = u.build_pair("qwen3-0.6b", self.mesh)
    batch = u.random_batch(cfg, seed=1)
    loss_legacy, grads_legacy = nnx.value_and_grad(u.legacy_loss)(legacy, batch, cfg)
    loss_m3, grads_m3 = nnx.value_and_grad(u.m3_loss)(m3, batch, cfg)
    u.assert_losses_close(self, loss_legacy, loss_m3)
    u.assert_grads_close(self, grads_legacy, grads_m3)


if __name__ == "__main__":
  absltest.main()

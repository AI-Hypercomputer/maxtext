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

"""Parity of m3 Qwen3-MoE against legacy MaxText Qwen3-MoE.

Same protocol as `qwen3_comparison_test.py`, with `sparse_matmul=false` on
both sides so legacy's dense expert path is compared against m3's `dense`
executor. The loss comparison includes the load-balance loss: legacy's comes
out as sown `moe_lb_loss` intermediates, m3's as `CausalLMOutput.aux_loss`.
"""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from maxtext.m3.core import moe
from maxtext.m3.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel
from maxtext.utils import model_creation_utils
from tests.utils import m3_test_utils as u

_MODEL = "qwen3-30b-a3b"


@pytest.mark.tpu_backend
@pytest.mark.tpu_only
class Qwen3MoeComparisonTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = u.legacy_mesh(u.tiny_config(_MODEL))
    self.enter_context(jax.set_mesh(self.mesh))

  def test_create_model_routes_to_m3(self):
    cfg = u.tiny_config(_MODEL)
    model = model_creation_utils.create_model(cfg, self.mesh, rngs=nnx.Rngs(0))
    self.assertIsInstance(model, Qwen3MoeModel)
    batch = u.random_batch(cfg)
    output = model(batch["inputs"], batch["positions"], batch["segment_ids"])
    self.assertEqual(output.logits.shape, (2, 8, cfg.vocab_size))
    self.assertGreater(float(output.aux_loss), 0.0)

  def test_parameters_match(self):
    legacy, m3, _ = u.build_pair(_MODEL, self.mesh)
    u.assert_params_match(self, legacy, m3)
    paths = u.flat_params(m3)
    self.assertIn(("decoder", "layers", 0, "moe_block", "gate", "kernel"), paths)
    self.assertIn(("decoder", "layers", 0, "moe_block", "wi_0"), paths)

  def test_forward_logits_match(self):
    legacy, m3, cfg = u.build_pair(_MODEL, self.mesh)
    batch = u.random_batch(cfg)
    logits_legacy = u.legacy_logits(legacy, batch)
    logits_m3 = m3(batch["inputs"], batch["positions"], batch["segment_ids"]).logits
    self.assertLess(float(jnp.max(jnp.abs(logits_legacy - logits_m3))), 1e-4)

  def test_loss_with_aux_and_gradients_match(self):
    legacy, m3, cfg = u.build_pair(_MODEL, self.mesh)
    batch = u.random_batch(cfg, seed=1)
    loss_legacy, grads_legacy = nnx.value_and_grad(u.legacy_loss)(legacy, batch, cfg)
    loss_m3, grads_m3 = nnx.value_and_grad(u.m3_loss)(m3, batch, cfg)
    u.assert_losses_close(self, loss_legacy, loss_m3)
    u.assert_grads_close(self, grads_legacy, grads_m3)
    # The router only gets gradient through the aux loss and combine weights; make sure it is not silently zero.
    gate = u.flat_params(grads_m3)[("decoder", "layers", 0, "moe_block", "gate", "kernel")]
    self.assertGreater(float(jnp.max(jnp.abs(gate))), 0.0)

  def test_ragged_dot_executor_matches_dense(self):
    legacy, m3, cfg = u.build_pair(_MODEL, self.mesh)
    del legacy
    batch = u.random_batch(cfg)
    dense = m3(batch["inputs"], batch["positions"], batch["segment_ids"]).logits
    for _, node in nnx.iter_graph(m3):
      if hasattr(node, "executor"):
        node.executor = moe.RAGGED_DOT
    ragged = m3(batch["inputs"], batch["positions"], batch["segment_ids"]).logits
    self.assertLess(float(jnp.max(jnp.abs(dense - ragged))), 5e-2)


if __name__ == "__main__":
  absltest.main()

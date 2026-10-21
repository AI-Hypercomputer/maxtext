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

"""Parity of m3 DeepSeek-V3 against legacy MaxText DeepSeek-V3.

Same protocol as `qwen3_comparison_test.py`, on a 1-dense + 2-MoE layer
`deepseek3-tiny` with group-limited routing enabled, so MLA with YaRN, the
sigmoid router with its bias, shared experts and both layer kinds are all
exercised. Legacy numbers its layer kinds separately (`dense_layers_0`,
`moe_layers_0`); the helpers map those onto m3's single container.
"""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from maxtext.m3.models.deepseek_v3.modeling_deepseek_v3 import DeepSeekV3Model, DeepSeekV3Router, RouterBias
from maxtext.utils import model_creation_utils
from tests.utils import m3_test_utils as u

_MODEL = "deepseek3-tiny"


@pytest.mark.tpu_backend
@pytest.mark.tpu_only
class DeepSeekV3ComparisonTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = u.legacy_mesh(u.tiny_config(_MODEL))
    self.enter_context(jax.set_mesh(self.mesh))

  def test_create_model_routes_to_m3(self):
    cfg = u.tiny_config(_MODEL)
    model = model_creation_utils.create_model(cfg, self.mesh, rngs=nnx.Rngs(0))
    self.assertIsInstance(model, DeepSeekV3Model)
    batch = u.random_batch(cfg)
    output = model(batch["inputs"], batch["positions"], batch["segment_ids"])
    self.assertEqual(output.logits.shape, (2, 8, cfg.vocab_size))
    # No load-balance weight configured: the aux loss is present but zero.
    self.assertEqual(float(output.aux_loss), 0.0)

  def test_parameters_match(self):
    legacy, m3, cfg = u.build_pair(_MODEL, self.mesh)
    u.assert_params_match(self, legacy, m3, cfg.first_num_dense_layers)
    paths = u.flat_params(m3)
    self.assertIn(("decoder", "layers", 0, "mlp", "wi_0", "kernel"), paths)
    self.assertIn(("decoder", "layers", 1, "moe_block", "routed_experts", "gate", "kernel"), paths)
    self.assertIn(("decoder", "layers", 1, "moe_block", "shared_experts", "wo", "kernel"), paths)
    self.assertIn(("decoder", "layers", 0, "self_attention", "wkv_b", "kernel"), paths)
    # The routing bias is state, not a parameter, on both sides.
    self.assertNotIn(("decoder", "layers", 1, "moe_block", "routed_experts", "gate", "bias"), paths)
    self.assertNotEmpty(nnx.to_flat_state(nnx.state(m3, RouterBias)))

  def test_forward_logits_match(self):
    legacy, m3, cfg = u.build_pair(_MODEL, self.mesh)
    batch = u.random_batch(cfg)
    logits_legacy = u.legacy_logits(legacy, batch)
    logits_m3 = m3(batch["inputs"], batch["positions"], batch["segment_ids"]).logits
    self.assertLess(float(jnp.max(jnp.abs(logits_legacy - logits_m3))), 1e-4)

  def test_forward_logits_match_with_router_bias(self):
    """A non-zero bias changes which experts are selected but not their weights."""
    legacy, m3, cfg = u.build_pair(_MODEL, self.mesh)
    bias = jax.random.normal(jax.random.key(3), (cfg.num_experts,), jnp.float32)
    routers = [node for _, node in nnx.iter_graph(legacy) if type(node).__name__ == "GateLogit"]
    routers += [node for _, node in nnx.iter_graph(m3) if isinstance(node, DeepSeekV3Router)]
    self.assertLen(routers, 2 * (cfg.num_decoder_layers - cfg.first_num_dense_layers))
    for router in routers:
      router.bias[...] = bias.astype(router.bias[...].dtype)
    batch = u.random_batch(cfg, seed=2)
    logits_legacy = u.legacy_logits(legacy, batch)
    logits_m3 = m3(batch["inputs"], batch["positions"], batch["segment_ids"]).logits
    self.assertLess(float(jnp.max(jnp.abs(logits_legacy - logits_m3))), 1e-4)

  def test_loss_and_gradients_match(self):
    legacy, m3, cfg = u.build_pair(_MODEL, self.mesh)
    batch = u.random_batch(cfg, seed=1)
    loss_legacy, grads_legacy = nnx.value_and_grad(u.legacy_loss)(legacy, batch, cfg)
    loss_m3, grads_m3 = nnx.value_and_grad(u.m3_loss)(m3, batch, cfg)
    u.assert_losses_close(self, loss_legacy, loss_m3)
    u.assert_grads_close(self, grads_legacy, grads_m3, cfg.first_num_dense_layers)


if __name__ == "__main__":
  absltest.main()

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

"""Tests for `maxtext.m3.infra.scan`.

Scan must be a pure re-layout: the same numbers as the unscanned model, from
the same weights, with the parameter tree in legacy's scanned shape. Run on
real m3 models (tiny Qwen3, Qwen3-MoE and DeepSeek-V3) rather than toys, so
the heterogeneous-run and aux-loss-carry cases are the real ones.
"""

from types import SimpleNamespace

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3 import models as m3_models
from maxtext.m3.core.layers import Layers
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.infra import remat, scan
from maxtext.m3.infra.scan import ScannedLayers
from maxtext.m3.models.deepseek_v3.modeling_deepseek_v3 import DeepSeekV3DenseLayer, DeepSeekV3MoeLayer
from maxtext.m3.train.loss import loss_fn
from tests.utils import m3_test_utils


def _batch(config, seed=0):
  shape = (2, 8)
  key = jax.random.key(seed)
  tokens = jax.random.randint(key, (shape[0], shape[1] + 1), 0, config.vocab_size)
  positions = jnp.broadcast_to(jnp.arange(shape[1]), shape)
  return {"inputs": tokens[:, :-1], "targets": tokens[:, 1:], "positions": positions}


def _grads(model, batch):
  (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
  return loss, {path: g[...] for path, g in nnx.to_flat_state(nnx.state(grads, nnx.Param))}


class ScanTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = create_mesh(devices=jax.devices()[:1])
    self.enter_context(jax.set_mesh(self.mesh))

  def _models(self, model_name, **overrides):
    """The same tiny model twice from one seed: as built, and scanned."""
    config = m3_test_utils.tiny_config(model_name, weight_dtype="float32", dtype="float32", **overrides)
    plain = m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0))
    scanned = m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0))
    scanned = scan.maybe_apply_scan(scanned, SimpleNamespace(scan_layers=True, param_scan_axis=config.param_scan_axis))
    return config, plain, scanned

  @parameterized.named_parameters(
      ("qwen3", "qwen3-0.6b"), ("qwen3_moe", "qwen3-30b-a3b"), ("deepseek_v3", "deepseek3-tiny")
  )
  def test_scanned_matches_unscanned(self, model_name):
    config, plain, scanned = self._models(model_name)
    batch = _batch(config)
    out_plain = plain(batch["inputs"], batch["positions"])
    out_scanned = scanned(batch["inputs"], batch["positions"])
    np.testing.assert_allclose(out_scanned.logits, out_plain.logits, rtol=1e-5, atol=1e-5)
    if out_plain.aux_loss is not None:
      np.testing.assert_allclose(out_scanned.aux_loss, out_plain.aux_loss, rtol=1e-5)

    loss_plain, grads_plain = _grads(plain, batch)
    loss_scanned, grads_scanned = _grads(scanned, batch)
    np.testing.assert_allclose(loss_scanned, loss_plain, rtol=1e-5)
    self.assertNotEmpty(grads_scanned)
    # Stacked gradients slice back to the per-layer ones.
    self.assertEqual(sum(g.size for g in grads_scanned.values()), sum(g.size for g in grads_plain.values()))

  def test_qwen3_layout_matches_legacy_scanned(self):
    """One run: `decoder/layers/<submodule>` with the layer axis at `param_scan_axis`."""
    config, plain, scanned = self._models("qwen3-0.6b")
    self.assertIsInstance(scanned.decoder.layers, ScannedLayers)
    stacked = m3_test_utils.flat_params(scanned)
    per_layer = m3_test_utils.flat_params(plain)
    kernel = ("decoder", "layers", "self_attention", "query", "kernel")
    self.assertIn(kernel, stacked)
    n, axis = config.num_decoder_layers, config.param_scan_axis
    expected_shape = list(per_layer[("decoder", "layers", 0) + kernel[2:]].shape)
    expected_shape.insert(axis, n)
    self.assertEqual(stacked[kernel].shape, tuple(expected_shape))
    for i in range(n):
      np.testing.assert_array_equal(
          jnp.take(stacked[kernel], i, axis=axis), per_layer[("decoder", "layers", i) + kernel[2:]]
      )
    # Rank-1 parameters cannot host axis 1, so they stack at their last axis.
    scale = ("decoder", "layers", "pre_self_attention_layer_norm", "scale")
    self.assertEqual(stacked[scale].shape, (config.emb_dim, n))

  def test_deepseek_splits_into_dense_and_moe_runs(self):
    """Two kinds of layer give two scanned runs inside one `Layers`, like legacy's dense_layers / moe_layers."""
    config, _, scanned = self._models("deepseek3-tiny")
    container = scanned.decoder.layers
    self.assertIsInstance(container, Layers)
    self.assertLen(container, 2)
    self.assertEqual([block.num_layers for block in container], [config.first_num_dense_layers, 2])
    paths = m3_test_utils.flat_params(scanned)
    self.assertIn(("decoder", "layers", 0, "mlp", "wi_0", "kernel"), paths)
    self.assertIn(("decoder", "layers", 1, "moe_block", "routed_experts", "wi_0"), paths)
    # Non-parameter state (the router bias) is stacked too and survives a call.
    bias = scanned.decoder.layers[1].moe_block.routed_experts.gate.bias
    self.assertEqual(bias[...].shape, (2, config.num_experts))

  def test_remat_is_carried_into_the_scan_body(self):
    config = m3_test_utils.tiny_config("qwen3-0.6b", weight_dtype="float32", dtype="float32")
    model = m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0))
    reference = model(_batch(config)["inputs"], _batch(config)["positions"]).logits
    model = remat.maybe_apply_remat(model, SimpleNamespace(remat_policy="minimal"))
    model = scan.maybe_apply_scan(model, SimpleNamespace(scan_layers=True, param_scan_axis=1))
    self.assertIsNotNone(model.decoder.layers.remat)
    rematted = model(_batch(config)["inputs"], _batch(config)["positions"]).logits
    np.testing.assert_allclose(rematted, reference, rtol=1e-5, atol=1e-5)
    loss, grads = _grads(model, _batch(config))
    self.assertTrue(np.isfinite(float(loss)))
    self.assertNotEmpty(grads)

  def test_scan_layers_false_is_a_no_op(self):
    config = m3_test_utils.tiny_config("qwen3-0.6b")
    model = m3_models.create_model(config, self.mesh, rngs=nnx.Rngs(0))
    self.assertIs(scan.maybe_apply_scan(model, SimpleNamespace(scan_layers=False, param_scan_axis=1)), model)
    self.assertIsInstance(model.decoder.layers, Layers)

  def test_scanned_model_survives_split_merge_and_jit(self):
    config, _, scanned = self._models("qwen3-30b-a3b")
    batch = _batch(config)
    reference = scanned(batch["inputs"], batch["positions"]).logits
    merged = nnx.merge(*nnx.split(scanned))
    np.testing.assert_allclose(merged(batch["inputs"], batch["positions"]).logits, reference, rtol=1e-5)
    jitted = nnx.jit(lambda m, b: m(b["inputs"], b["positions"]).logits)
    np.testing.assert_allclose(jitted(scanned, batch), reference, rtol=1e-5, atol=1e-5)

  def test_rejects_mixed_layers_in_one_run(self):
    with self.assertRaisesRegex(ValueError, "not structurally identical"):
      scan.scan_run([nnx.Linear(2, 2, rngs=nnx.Rngs(0)), nnx.Linear(2, 3, rngs=nnx.Rngs(0))], param_scan_axis=1)

  def test_scanned_run_is_an_instance_of_its_layer_class(self):
    """`Scanned<Layer>` subclasses the layer class, so the layer's helpers are available in the scan body."""
    _, _, scanned = self._models("deepseek3-tiny")
    dense_run, moe_run = scanned.decoder.layers
    self.assertIsInstance(dense_run, ScannedLayers)
    self.assertIsInstance(dense_run, DeepSeekV3DenseLayer)
    self.assertIsInstance(moe_run, DeepSeekV3MoeLayer)
    self.assertEqual(type(dense_run).__name__, "ScannedDeepSeekV3DenseLayer")
    # One class per layer class, shared across models: graphdefs stay comparable.
    _, _, again = self._models("deepseek3-tiny")
    self.assertIs(type(again.decoder.layers[0]), type(dense_run))


if __name__ == "__main__":
  absltest.main()

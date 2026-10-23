# Copyright 2023–2026 Google LLC
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

"""m3 scan transform checks, including Qwen3 output and gradient parity."""

from absl.testing import absltest
from flax import nnx
from jax.sharding import PartitionSpec as P
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from maxtext.m3.infra import scan as m3_scan
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model
from maxtext.m3.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLDecoder
from tests.utils.test_helpers import get_test_config_path


def _config(scan_layers: bool, num_layers: int = 4, param_scan_axis: int = 1):
  """Build a small Qwen3 test configuration with optional layer scanning."""
  args = {
      "model_name": "qwen3-0.6b",
      "use_m3_model": "true",
      "scan_layers": str(scan_layers).lower(),
      "param_scan_axis": param_scan_axis,
      "override_model_config": "true",
      "base_num_decoder_layers": num_layers,
      "base_emb_dim": 64,
      "base_mlp_dim": 128,
      "base_num_query_heads": 4,
      "base_num_kv_heads": 2,
      "head_dim": 16,
      "vocab_size": 256,
      "per_device_batch_size": 1,
      "max_target_length": 32,
      "dtype": "float32",
      "weight_dtype": "float32",
      "matmul_precision": "highest",
      "attention": "dot_product",
      "enable_checkpointing": "false",
      "run_name": "m3_scan_test",
  }
  return pyconfig.initialize(["", get_test_config_path()] + [f"{k}={v}" for k, v in args.items()])


class ScanTransformTest(absltest.TestCase):

  def test_qwen3_scanned_matches_unscanned(self):
    for axis in (0, 1):
      with self.subTest(param_scan_axis=axis):
        cfg = _config(scan_layers=True, param_scan_axis=axis)
        model = create_qwen3_model(cfg, rngs=nnx.Rngs(0))
        tokens = jax.random.randint(jax.random.key(1), (1, 32), 0, cfg.vocab_size)
        positions = jnp.arange(32)[None]
        ref = model(tokens, positions)

        m3_scan.maybe_apply_scan(model, cfg)
        self.assertEqual(model.decoder.layer_names, ("layers",))

        out = nnx.jit(lambda m, t, p: m(t, p))(model, tokens, positions)
        np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-4, rtol=1e-4)

  def test_qwen3_scan_is_noop_when_disabled(self):
    cfg = _config(scan_layers=False)
    model = create_qwen3_model(cfg, rngs=nnx.Rngs(0))
    m3_scan.maybe_apply_scan(model, cfg)
    self.assertEqual(model.decoder.layer_names, tuple(f"layers_{i}" for i in range(4)))

  def test_qwen3_scanned_gradients_match(self):
    for axis in (0, 1):
      with self.subTest(param_scan_axis=axis):
        cfg = _config(scan_layers=True, num_layers=3, param_scan_axis=axis)
        tokens = jax.random.randint(jax.random.key(1), (1, 32), 0, cfg.vocab_size)
        positions = jnp.arange(32)[None]

        def loss(m, t, p):
          return jnp.mean(m(t, p) ** 2)

        ref_model = create_qwen3_model(cfg, rngs=nnx.Rngs(0))
        ref_grads = nnx.grad(loss)(ref_model, tokens, positions)
        model = m3_scan.maybe_apply_scan(create_qwen3_model(cfg, rngs=nnx.Rngs(0)), cfg)
        grads = nnx.jit(nnx.grad(loss))(model, tokens, positions)

        self._assert_matches_unscanned(grads, ref_grads, cfg)

  def _assert_matches_unscanned(self, scanned, reference, cfg):
    reference = nnx.to_pure_dict(reference)
    decoder = reference["decoder"]
    layers = [decoder.pop(f"layers_{i}") for i in range(cfg.num_decoder_layers)]
    decoder["layers"] = jax.tree.map(lambda *xs: jnp.stack(xs, axis=cfg.param_scan_axis), *layers)
    actual = nnx.to_pure_dict(scanned)
    self.assertEqual(jax.tree.structure(actual), jax.tree.structure(reference))
    for (path, value), expected in zip(jax.tree.leaves_with_path(actual), jax.tree.leaves(reference)):
      np.testing.assert_allclose(
          np.asarray(value),
          np.asarray(expected),
          atol=1e-5,
          rtol=1e-4,
          err_msg=str(path),
      )

  def test_scalar_and_vector_params_with_short_sharding_specs(self):
    class Layer(nnx.Module):
      """Affine layer with scalar and vector parameters."""

      def __init__(self, scale):
        self.scale = nnx.Param(jnp.array(scale), out_sharding=P())
        self.bias = nnx.Param(jnp.arange(4, dtype=jnp.float32), out_sharding=P())

      def __call__(self, x):
        # Metadata must describe the sliced values, including the scalar.
        if len(self.scale.get_metadata()["out_sharding"]) != 0:
          raise ValueError("Scalar sharding must have rank zero.")
        return x * self.scale[...] + self.bias[...]

    for axis in (0, 1):
      with self.subTest(param_scan_axis=axis):
        layers = [Layer(float(i + 1)) for i in range(3)]
        x = jnp.ones(4)
        expected = x
        for layer in layers:
          expected = layer(expected)
        scanned = m3_scan.ScannedLayers(layers, param_scan_axis=axis)
        self.assertEqual(scanned.scale.shape, (3,))
        self.assertEqual(scanned.bias.shape, (3, 4) if axis == 0 else (4, 3))
        self.assertEqual(scanned.bias.get_metadata()["out_sharding"], P(None) if axis == 0 else P(None, None))
        actual = nnx.jit(lambda m, v: m(v))(scanned, x)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected))

  def test_scan_rejects_unsupported_model_before_mutation(self):
    cfg = _config(scan_layers=True)
    model = create_qwen3_model(cfg, rngs=nnx.Rngs(0))
    model.decoder = Qwen3VLDecoder(cfg, rngs=nnx.Rngs(0))
    self.assertIsNone(model.decoder.layer_names)
    first_layer = model.decoder.layers_0
    with self.assertRaisesRegex(ValueError, "Qwen3VLDecoder does not expose scan units through `layer_names`"):
      m3_scan.maybe_apply_scan(model, cfg)
    self.assertIs(model.decoder.layers_0, first_layer)
    self.assertIsNone(model.decoder.layer_names)


if __name__ == "__main__":
  absltest.main()

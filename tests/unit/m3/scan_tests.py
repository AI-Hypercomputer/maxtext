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

"""Tests for m3 scan and Qwen3's legacy-compatible scanned checkpoint layout."""

from types import SimpleNamespace

from absl.testing import absltest, parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.infra import scan
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model
from maxtext.m3.models.qwen3_vl.modeling_qwen3_vl import create_qwen3_vl_model


def tiny_config(scan_layers=False, param_scan_axis=1):
  """Only the configuration consumed by the dense Qwen3 model."""
  return SimpleNamespace(
      scan_layers=scan_layers,
      param_scan_axis=param_scan_axis,
      num_decoder_layers=3,
      emb_dim=16,
      mlp_dim=32,
      num_query_heads=4,
      num_kv_heads=2,
      head_dim=4,
      vocab_size=32,
      dtype=jnp.float32,
      weight_dtype=jnp.float32,
      normalization_layer_epsilon=1e-6,
      use_qk_norm=True,
      rope_max_timescale=1000000,
      float32_logits=True,
      logits_dot_in_fp32=True,
      normalize_embedding_logits=False,
      final_logits_soft_cap=0,
      cast_logits_to_fp32=True,
  )


def flat_params(model):
  return {p: v[...] for p, v in nnx.to_flat_state(nnx.state(model, nnx.Param))}


class ScanTest(parameterized.TestCase):

  def test_qwen3_vl_rejects_scan(self):
    with self.assertRaisesRegex(ValueError, "Qwen3-VL.*scan_layers=false"):
      create_qwen3_vl_model(tiny_config(scan_layers=True), rngs=nnx.Rngs(0))

  @parameterized.parameters(0, 1, 2)
  def test_qwen3_scan_forward_gradients_and_checkpoint_layout(self, axis):
    # Parameters are placed from their sharding metadata at construction, which
    # needs a mesh in scope; one device is enough (README rule 3).
    with jax.set_mesh(create_mesh(devices=jax.devices()[:1])):
      plain = create_qwen3_model(tiny_config(), rngs=nnx.Rngs(0))
      config = tiny_config(True, axis)
      scanned = scan.maybe_apply_scan(create_qwen3_model(config, rngs=nnx.Rngs(0)), config)
    p_plain, p_scan = flat_params(plain), flat_params(scanned)
    for path, value in p_plain.items():
      if path[:3] == ("decoder", "layers", "layers"):
        i = path[3]
        target = path[:2] + path[4:]
        np.testing.assert_array_equal(jnp.take(p_scan[target], i, axis=min(axis, value.ndim)), value)
      else:
        np.testing.assert_array_equal(p_scan[path], value)

    tokens = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    positions = jnp.array([[0, 1, 0, 1], [0, 1, 2, 3]])
    segments = jnp.array([[1, 1, 2, 2], [1, 1, 1, 1]])

    def logits(m):
      return m(tokens, positions, decoder_segment_ids=segments)

    np.testing.assert_allclose(logits(scanned), logits(plain), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(nnx.jit(logits)(scanned), logits(plain), rtol=1e-5, atol=1e-5)
    self.assertIn("scan[", str(jax.make_jaxpr(lambda: logits(scanned))()))

    def loss(m):
      return -jnp.mean(jax.nn.log_softmax(logits(m))[..., 0])

    loss_plain, grad_plain = nnx.value_and_grad(loss)(plain)
    loss_scan, grad_scan = nnx.value_and_grad(loss)(scanned)
    np.testing.assert_allclose(loss_scan, loss_plain, rtol=1e-5)
    g_scan = flat_params(grad_scan)
    for path, value in flat_params(grad_plain).items():
      if path[:3] == ("decoder", "layers", "layers"):
        target = path[:2] + path[4:]
        actual = jnp.take(g_scan[target], path[3], axis=min(axis, value.ndim))
      else:
        actual = g_scan[path]
      np.testing.assert_allclose(actual, value, rtol=2e-4, atol=2e-6, err_msg=str(path))


class CounterLayer(nnx.Module):
  """A stateful layer with scalar and rank-one parameters."""

  def __init__(self, offset):
    self.scale = nnx.Param(jnp.array(float(offset)))
    self.bias = nnx.Param(jnp.ones(2), out_sharding=jax.sharding.PartitionSpec())
    self.calls = nnx.Variable(jnp.array(0), out_sharding=jax.sharding.PartitionSpec())

  def __call__(self, carry, *, increment=1):
    self.calls[...] += increment
    return carry * self.scale[...] + self.bias[...]


class ScanInfrastructureTest(absltest.TestCase):

  def test_state_and_metadata_survive_repeated_calls(self):
    layers = nnx.Sequential(CounterLayer(1), CounterLayer(2))
    scanned = scan.scan_layers(layers, param_scan_axis=1)
    self.assertEqual(scanned.bias.get_metadata()["out_sharding"], jax.sharding.PartitionSpec(None, None))
    for _ in range(2):
      np.testing.assert_allclose(scanned(jnp.zeros(2), increment=2), layers(jnp.zeros(2), increment=2))
    np.testing.assert_array_equal(scanned.calls[...], [4, 4])
    self.assertEqual(scanned.calls.get_metadata()["out_sharding"], jax.sharding.PartitionSpec(None))

  def test_heterogeneous_runs_and_disabled_transform(self):
    model = nnx.Dict(
        blocks=nnx.Sequential(nnx.Linear(2, 2, rngs=nnx.Rngs(0)), nnx.Linear(2, 2, use_bias=False, rngs=nnx.Rngs(1)))
    )
    x = jnp.ones((1, 2))
    reference = model.blocks(x)
    self.assertIs(scan.maybe_apply_scan(model, SimpleNamespace(scan_layers=False)), model)
    scan.apply_scan(model, param_scan_axis=1)
    self.assertIsInstance(model.blocks, nnx.Sequential)
    np.testing.assert_allclose(model.blocks(x), reference, rtol=1e-5)
    scan.apply_scan(model, param_scan_axis=1)
    np.testing.assert_allclose(model.blocks(x), reference, rtol=1e-5)

  def test_named_sharding_is_preserved(self):
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ("data",))
    self.enter_context(jax.set_mesh(mesh))
    spec = jax.sharding.PartitionSpec("data")

    def make_layer():
      layer = CounterLayer(1)
      layer.bias = nnx.Param(jax.device_put(jnp.ones(2), jax.sharding.NamedSharding(mesh, spec)), out_sharding=spec)
      return layer

    scanned = scan.scan_run([make_layer() for _ in range(2)], param_scan_axis=1)
    expected = jax.sharding.PartitionSpec("data", None)
    self.assertEqual(scanned.bias[...].sharding.spec, expected)
    np.testing.assert_allclose(nnx.jit(lambda m: m(jnp.zeros(2)))(scanned), [2, 2])
    self.assertEqual(scanned.bias.get_metadata()["out_sharding"], expected)


if __name__ == "__main__":
  absltest.main()

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

"""Tests for the FSDP shard_map DenseGeneral dot (dense_fsdp_shard_map_dot): forward and gradients must match GSPMD."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import dataclasses  # pylint: disable=wrong-import-position
import unittest  # pylint: disable=wrong-import-position
from unittest import mock  # pylint: disable=wrong-import-position

from flax import nnx  # pylint: disable=wrong-import-position
from flax.linen import partitioning as nn_partitioning  # pylint: disable=wrong-import-position
import jax  # pylint: disable=wrong-import-position
import jax.numpy as jnp  # pylint: disable=wrong-import-position
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
import pytest  # pylint: disable=wrong-import-position

from maxtext.layers import linears  # pylint: disable=wrong-import-position

_RULES = (
    ("activation_batch", ("fsdp", "expert")),
    ("activation_norm_length", None),
    ("embed", "fsdp"),
    ("mlp", None),
    ("heads", None),
    ("kv", None),
)


def _mesh():
  devices = np.array(jax.devices()[:8]).reshape(4, 2)
  return Mesh(devices, ("fsdp", "expert"))


def _settings(mesh, **kw):
  return linears.DenseWgradReduceScatterConfig(enabled=True, mesh=mesh, **kw)


@pytest.mark.cpu_only
class FsdpShardMapDotTest(unittest.TestCase):
  """Compares DenseGeneral with and without the shard_map FSDP dot on an 8-device CPU mesh."""

  def setUp(self):
    super().setUp()
    if len(jax.devices()) < 8:
      self.skipTest("needs 8 (CPU) devices")
    self.mesh = _mesh()
    self.addCleanup(lambda: linears.configure_dense_wgrad_reduce_scatter(object(), None))

  # pylint: disable=protected-access
  def _run(self, kernel_axes, in_features, out_features, axis, settings, x_shape):
    """Loss + grads of a DenseGeneral under `settings` (None: GSPMD path)."""
    with self.mesh, nn_partitioning.axis_rules(_RULES):
      module = linears.DenseGeneral(
          in_features_shape=in_features,
          out_features_shape=out_features,
          axis=axis,
          kernel_axes=kernel_axes,
          dtype=jnp.float32,
          weight_dtype=jnp.float32,
          mesh=self.mesh,
          rngs=nnx.Rngs(0),
      )
      graphdef, state = nnx.split(module)
      x = jax.random.normal(jax.random.PRNGKey(1), x_shape, jnp.float32)
      x = jax.device_put(x, NamedSharding(self.mesh, P(("fsdp", "expert"))))
      linears._DENSE_WGRAD_RS = settings if settings is not None else linears.DenseWgradReduceScatterConfig()
      calls = []
      shard_map_dot = linears._fsdp_shard_map_dot

      def _spy(*args, **kwargs):
        out = shard_map_dot(*args, **kwargs)
        calls.append(out is not None)
        return out

      def loss_fn(state, x):
        out = nnx.merge(graphdef, state)(x)
        return jnp.sum(out * out), out

      with mock.patch.object(linears, "_fsdp_shard_map_dot", _spy):
        (loss, out), grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(state, x)
      # The shard_map path must actually be taken when enabled (and never when disabled).
      self.assertEqual(calls, [True] if settings is not None else [])
      grads = jax.tree.map(lambda g: np.asarray(g.value if hasattr(g, "value") else g), grads)
      return np.asarray(loss), np.asarray(out), grads

  def _check(self, kernel_axes, in_features, out_features, axis, x_shape, **kw):
    ref = self._run(kernel_axes, in_features, out_features, axis, None, x_shape)
    for flatten in (True, False):
      got = self._run(
          kernel_axes, in_features, out_features, axis, _settings(self.mesh, flatten_scatter_dim=flatten, **kw), x_shape
      )
      np.testing.assert_allclose(got[0], ref[0], rtol=1e-5)
      np.testing.assert_allclose(got[1], ref[1], rtol=1e-5, atol=1e-5)
      jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5), got[2], ref[2])

  def test_kernel_sharded_on_dim0(self):
    # e.g. the MLP wi kernel [embed, mlp] sharded on embed (fsdp).
    self._check(("embed", "mlp"), 32, 24, -1, (8, 4, 32))

  def test_kernel_sharded_on_dim1(self):
    # e.g. the MLP wo kernel [mlp, embed] sharded on embed: the scatter dim is not the leading dim.
    self._check(("mlp", "embed"), 24, 32, -1, (8, 4, 24))

  def test_three_dim_kernel(self):
    # e.g. attention out-projection [heads, kv, embed] contracted over (heads, kv).
    self._check(("heads", "kv", "embed"), (4, 8), 32, (-2, -1), (8, 4, 4, 8))

  def test_pinned_reduce_scatter_traces(self):
    # The SparseCore pin wraps the psum_scatter in a compute_on region (not executable on CPU; check the trace only).
    with self.mesh, nn_partitioning.axis_rules(_RULES):
      module = linears.DenseGeneral(32, 24, kernel_axes=("embed", "mlp"), mesh=self.mesh, rngs=nnx.Rngs(0))
      graphdef, state = nnx.split(module)
      x = jnp.ones((8, 4, 32), jnp.float32)
      linears._DENSE_WGRAD_RS = _settings(self.mesh, sparse_core_id=1)
      jaxpr = str(jax.make_jaxpr(jax.grad(lambda s, x: jnp.sum(nnx.merge(graphdef, s)(x))))(state, x))
    self.assertIn("all_gather", jaxpr)
    # Backward: reduce-scatter over fsdp (inside the pinned compute_on region) and then psum over expert.
    self.assertIn("compute_type=tpu_sparsecore", jaxpr)
    self.assertIn("reduce_scatter", jaxpr)
    self.assertIn("psum_invariant[axes=('expert',)]", jaxpr)

  def test_falls_back_when_kernel_not_sharded(self):
    with self.mesh, nn_partitioning.axis_rules(_RULES):
      kernel = jnp.zeros((24, 16), jnp.float32)
      self.assertIsNone(
          linears._fsdp_shard_map_dot(jnp.zeros((8, 24)), kernel, ("mlp", "heads"), (1,), "default", _settings(self.mesh))
      )

  def test_configure_from_config(self):
    cfg = dataclasses.make_dataclass(
        "Cfg",
        [
            ("dense_fsdp_shard_map_dot", bool),
            ("dense_fsdp_shard_map_max_kernel_elems", int),
            ("dense_wgrad_rs_sparse_core_id", int),
            ("dense_wgrad_rs_flatten_scatter_dim", bool),
            ("use_qwix_quantization", bool),
            ("quantization", str),
            ("weight_quantization_calibration_method", str),
        ],
    )(True, 128, 1, False, True, "fp8_full", "fixed,-224,224")
    linears.configure_dense_wgrad_reduce_scatter(cfg, self.mesh)
    s = linears._DENSE_WGRAD_RS
    self.assertTrue(s.enabled)
    self.assertEqual(
        (s.max_kernel_elems, s.sparse_core_id, s.flatten_scatter_dim, s.fp8_fixed_absmax), (128, 1, False, 224.0)
    )


if __name__ == "__main__":
  unittest.main()

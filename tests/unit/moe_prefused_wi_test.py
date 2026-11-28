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

"""RoutedMoE with `prefuse_moe_weights=True` and `moe_unsliced_prefused_wi=True`
matches the separate wi_0 / wi_1 weights in forward and backward."""

import unittest

from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
import pytest

from maxtext.configs import pyconfig
from maxtext.layers import moe
from maxtext.layers.initializers import nd_dense_init
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from tests.utils.test_helpers import get_test_config_path


class PrefusedWiSparseMatmulTest(unittest.TestCase):
  """Fused wi (prefuse_moe_weights + moe_unsliced_prefused_wi) vs separate wi_0 / wi_1 under sparse_matmul + gmm_v2."""

  def _make_config(self, prefuse, tensor_parallel=False, sparse_matmul=True, unsliced=None):
    """Small qwen3-moe-style RoutedMoE (gmm_v2) with fused or separate wi, on an all-fsdp or all-tensor mesh."""
    num_devices = jax.device_count()
    cfg = pyconfig.initialize(
        [None, get_test_config_path()],
        run_name=f"moe_prefused_wi_{prefuse}_{tensor_parallel}",
        enable_checkpointing=False,
        model_name="qwen3-30b-a3b",
        num_experts=((8 + num_devices - 1) // num_devices) * num_devices,
        num_experts_per_tok=2,
        override_model_config=True,
        base_emb_dim=512,
        base_mlp_dim=1024,
        base_moe_mlp_dim=1024,
        dtype="bfloat16",
        weight_dtype="float32",
        sparse_matmul=sparse_matmul,
        megablox=False,
        use_tokamax_gmm=True,
        use_gmm_v2=True,
        prefuse_moe_weights=prefuse,
        moe_unsliced_prefused_wi=prefuse if unsliced is None else unsliced,
        shard_exp_on_fsdp=not tensor_parallel,
        ici_fsdp_parallelism=1 if tensor_parallel else num_devices,
        ici_tensor_parallelism=num_devices if tensor_parallel else 1,
        per_device_batch_size=1,
        max_target_length=64,
        log_config=False,
    )
    return cfg

  def _make_model(self, prefuse, tensor_parallel=False):
    """RoutedMoE (plus its config and mesh) built from `_make_config`, with its parameters sharded on the mesh."""
    cfg = self._make_config(prefuse, tensor_parallel)
    mesh = Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
    # The logical -> mesh axis rules are read from the flax context, as in training; without it the
    # parameters (and the whole layer) would be replicated.
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      model = moe.RoutedMoE(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          intermediate_dim=cfg.mlp_dim,
          weight_dtype=cfg.weight_dtype,
          dtype=cfg.dtype,
          rngs=nnx.Rngs(params=0),
      )
      self._shard_params(model, mesh)
    return cfg, mesh, model

  def _shard_params(self, model, mesh):
    """Places the parameters on the mesh with their declared sharding (the Param metadata alone leaves them replicated)."""
    params = nnx.state(model, nnx.Param)
    shardings = sharding.nnx_construct_named_sharding(params, mesh)
    nnx.update(model, jax.tree.map(jax.device_put, params, shardings))

  def _assert_prefused_matches_separate_weights(self):
    """Fused wi (moe_unsliced_prefused_wi) and separate wi_0 / wi_1 give the same outputs and gradients."""
    cfg, mesh, split_model = self._make_model(prefuse=False)
    _, _, fused_model = self._make_model(prefuse=True)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      fused_model.wi = nnx.Param(
          jnp.concatenate([split_model.wi_0[...], split_model.wi_1[...]], axis=-1),
          out_sharding=fused_model.wi_kernel_axes,
      )
      fused_model.wo = split_model.wo
      fused_model.gate = split_model.gate
      self._shard_params(fused_model, mesh)
      batch_sharding = sharding.create_sharding(mesh, ("activation_batch", None, None))
    # Sharded weights, so the FSDP all-gathers are really in the compiled graphs being compared.
    self.assertFalse(split_model.wi_0[...].sharding.is_fully_replicated)
    self.assertFalse(fused_model.wi[...].sharding.is_fully_replicated)

    inputs = jax.random.normal(jax.random.PRNGKey(1), (jax.device_count(), 64, cfg.emb_dim), jnp.bfloat16)
    cotangent = jax.random.normal(jax.random.PRNGKey(2), inputs.shape, jnp.float32)
    inputs, cotangent = jax.device_put(inputs, batch_sharding), jax.device_put(cotangent, batch_sharding)

    def loss(model, x):
      out, _, _ = model(x)
      return jnp.sum(out.astype(jnp.float32) * cotangent), out

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      run = nnx.jit(nnx.value_and_grad(loss, argnums=(0, 1), has_aux=True))
      (_, out_split), (grads_split, dx_split) = run(split_model, inputs)
      (_, out_fused), (grads_fused, dx_fused) = run(fused_model, inputs)

    # Compared elementwise (the scalar loss is a near-zero sum of signed terms), against each array's own scale:
    # the values here are small (outputs ~1e-4), so a fixed absolute tolerance would accept anything.
    def assert_close(actual, expected):
      actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
      scale = float(np.abs(expected).max())
      self.assertGreater(scale, 0.0)
      np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-2 * scale)

    assert_close(out_fused, out_split)
    assert_close(dx_fused, dx_split)
    assert_close(grads_fused["wi"][...], jnp.concatenate([grads_split["wi_0"][...], grads_split["wi_1"][...]], axis=-1))
    assert_close(grads_fused["wo"][...], grads_split["wo"][...])

  @pytest.mark.tpu_only
  def test_prefused_matches_separate_weights(self):
    self._assert_prefused_matches_separate_weights()

  def test_tensor_parallel_raises(self):
    """With the mlp dim sharded over the tensor axis a shard would get a piece of [w0 | w1], so the layer raises."""
    if jax.device_count() < 2:
      self.skipTest("needs at least 2 devices for tensor parallelism")
    cfg, mesh, fused_model = self._make_model(prefuse=True, tensor_parallel=True)
    inputs = jnp.zeros((jax.device_count(), 64, cfg.emb_dim), jnp.bfloat16)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(cfg.logical_axis_rules):
      with self.assertRaisesRegex(
          ValueError, "moe_unsliced_prefused_wi=True is not supported with the expert MLP dim sharded"
      ):
        fused_model(inputs)

  def test_config_requirements(self):
    """The flag is rejected at config time, instead of silently slicing, when its static requirements are unmet."""
    with self.assertRaisesRegex(ValueError, "moe_unsliced_prefused_wi=True requires .*sparse_matmul=True"):
      self._make_config(prefuse=True, sparse_matmul=False)
    with self.assertRaisesRegex(ValueError, "moe_unsliced_prefused_wi=True requires .*prefuse_moe_weights=True"):
      self._make_config(prefuse=False, unsliced=True)

  def test_split_gate_up(self):
    """_split_gate_up halves the last dim, and its custom VJP is the concatenation of the two half-gradients."""
    out = jax.random.normal(jax.random.PRNGKey(0), (6, 8), jnp.float32)
    c_gate = jax.random.normal(jax.random.PRNGKey(1), (6, 4), jnp.float32)
    c_up = jax.random.normal(jax.random.PRNGKey(2), (6, 4), jnp.float32)
    gate, up = moe._split_gate_up(out)  # pylint: disable=protected-access
    np.testing.assert_array_equal(gate, out[:, :4])
    np.testing.assert_array_equal(up, out[:, 4:])

    def loss(x, split):
      g, u = split(x)
      return jnp.sum(g * c_gate) + jnp.sum(u * c_up)

    grad = jax.grad(loss)(out, moe._split_gate_up)  # pylint: disable=protected-access
    grad_ref = jax.grad(loss)(out, lambda x: (x[:, :4], x[:, 4:]))
    np.testing.assert_array_equal(grad, grad_ref)
    np.testing.assert_array_equal(grad, jnp.concatenate([c_gate, c_up], axis=-1))


if __name__ == "__main__":
  unittest.main()

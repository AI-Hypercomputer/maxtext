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

"""Unit tests for Qwen3.5 sub-layer rematerialization."""

import os
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig
from maxtext.models import qwen3_5
from maxtext.utils.globals import MAXTEXT_PKG_DIR
import numpy as np


class Qwen3_5SublayerRematTest(unittest.TestCase):

  def test_sublayer_remat(self):
    base_config_path = os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")
    cfg_off = pyconfig.initialize(
        ["", base_config_path],
        run_name="foo",
        skip_jax_distributed_system=True,
        override_model_config=True,
        scan_layers=False,
        qwen3_5_sublayer_remat=False,
        base_num_decoder_layers=4,
        inhomogeneous_layer_cycle_interval=4,
        base_emb_dim=16,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        head_dim=8,
        base_mlp_dim=32,
        base_moe_mlp_dim=32,
        gdn_key_head_dim=8,
        gdn_value_head_dim=8,
        gdn_num_key_heads=2,
        gdn_num_value_heads=2,
        gdn_chunk_size=16,
        num_experts=2,
        dtype="float32",
        weight_dtype="float32",
        remat_policy="full",
    )
    cfg_on = pyconfig.initialize(
        ["", base_config_path],
        run_name="foo",
        skip_jax_distributed_system=True,
        override_model_config=True,
        scan_layers=False,
        qwen3_5_sublayer_remat=True,
        base_num_decoder_layers=4,
        inhomogeneous_layer_cycle_interval=4,
        base_emb_dim=16,
        base_num_query_heads=2,
        base_num_kv_heads=2,
        head_dim=8,
        base_mlp_dim=32,
        base_moe_mlp_dim=32,
        gdn_key_head_dim=8,
        gdn_value_head_dim=8,
        gdn_num_key_heads=2,
        gdn_num_value_heads=2,
        gdn_chunk_size=16,
        num_experts=2,
        dtype="float32",
        weight_dtype="float32",
        remat_policy="full",
    )
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))

    with jax.default_prng_impl("threefry2x32"):
      block_off = qwen3_5.Qwen3_5ScannableBlock(config=cfg_off, mesh=mesh, model_mode="train", rngs=nnx.Rngs(0))
      block_on = qwen3_5.Qwen3_5ScannableBlock(config=cfg_on, mesh=mesh, model_mode="train", rngs=nnx.Rngs(0))

    g_off, params_off, rest_off = nnx.split(block_off, nnx.Param, ...)
    g_on, params_on, rest_on = nnx.split(block_on, nnx.Param, ...)

    self.assertEqual(jax.tree_util.tree_structure(params_off), jax.tree_util.tree_structure(params_on))

    x = jnp.ones((2, 16, 16))
    decoder_positions = jnp.broadcast_to(jnp.arange(16), (2, 16))

    def forward_and_loss(xi, params, graphdef, rest):
      block_merged = nnx.merge(graphdef, params, rest)
      out, _ = block_merged(xi, None, decoder_positions, deterministic=True, model_mode="train")
      return jnp.sum(out)

    grad_fn = jax.value_and_grad(forward_and_loss, argnums=(0, 1))

    with jax.default_matmul_precision("highest"):
      loss_off, (dx_off, grads_off) = grad_fn(x, params_off, g_off, rest_off)
      loss_on, (dx_on, grads_on) = grad_fn(x, params_on, g_on, rest_on)

    np.testing.assert_allclose(loss_off, loss_on, rtol=1e-5, atol=1e-5)

    # Scale-aware tolerance to account for fusion/reassociation differences
    # across XLA compilation strategies and JAX PRNG implementations.
    # The max observed ratio to 1e-4 tol is ~0.24-0.3, so we use 5e-4 (a 5x margin).
    dx_max = np.maximum(np.max(np.abs(dx_off)), 1e-6)
    np.testing.assert_allclose(dx_off, dx_on, rtol=5e-4, atol=dx_max * 5e-4 + 5e-4)

    grads_off_flat, _ = jax.tree_util.tree_flatten(grads_off)
    grads_on_flat, _ = jax.tree_util.tree_flatten(grads_on)
    for go, gn in zip(grads_off_flat, grads_on_flat):
      g_max = np.maximum(np.max(np.abs(go)), 1e-6)
      np.testing.assert_allclose(go, gn, rtol=5e-4, atol=g_max * 5e-4 + 5e-4)


if __name__ == "__main__":
  unittest.main()

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

  def _make_cfg(self, *, scan_layers: bool, sublayer_remat: bool, num_layers: int = 4, cycle: int = 4):
    base_config_path = os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")
    return pyconfig.initialize(
        ["", base_config_path],
        run_name="foo",
        skip_jax_distributed_system=True,
        override_model_config=True,
        decoder_block="qwen3_5",
        scan_layers=scan_layers,
        qwen3_5_sublayer_remat=sublayer_remat,
        base_num_decoder_layers=num_layers,
        inhomogeneous_layer_cycle_interval=cycle,
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

  def test_sublayer_remat_produces_checkpoint(self):
    cfg_on = self._make_cfg(scan_layers=False, sublayer_remat=True, num_layers=1, cycle=1)
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))

    with jax.default_prng_impl("threefry2x32"):
      layer = qwen3_5.Qwen3_5DecoderLayer(
          config=cfg_on, mesh=mesh, quant=None, model_mode="train", layer_idx=0, rngs=nnx.Rngs(0)
      )

    x = jnp.ones((4, 16, 16))
    decoder_positions = jnp.broadcast_to(jnp.arange(16), (4, 16))
    graphdef, state = nnx.split(layer)

    def forward(xi, layer_state):
      layer_merged = nnx.merge(graphdef, layer_state)
      out, kv = layer_merged(xi, decoder_positions=decoder_positions, deterministic=True, model_mode="train")
      self.assertEqual(out.shape, xi.shape)
      self.assertIsNone(kv)
      return jnp.sum(out)

    jaxpr = jax.make_jaxpr(forward)(x, state)
    self.assertIn("attn_subblock_out", str(jaxpr))

  def _run_scannable_block_remat_test(self, scan_layers: bool):
    cfg_off = self._make_cfg(scan_layers=scan_layers, sublayer_remat=False)
    cfg_on = self._make_cfg(scan_layers=scan_layers, sublayer_remat=True)
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))

    with jax.default_prng_impl("threefry2x32"):
      block_off = qwen3_5.Qwen3_5ScannableBlock(
          config=cfg_off, mesh=mesh, model_mode="train", apply_internal_remat=True, rngs=nnx.Rngs(0)
      )
      block_on = qwen3_5.Qwen3_5ScannableBlock(
          config=cfg_on, mesh=mesh, model_mode="train", apply_internal_remat=True, rngs=nnx.Rngs(0)
      )

    _, params_off, _ = nnx.split(block_off, nnx.Param, ...)
    _, params_on, _ = nnx.split(block_on, nnx.Param, ...)
    self.assertEqual(jax.tree_util.tree_structure(params_off), jax.tree_util.tree_structure(params_on))

    # Batch size 4 (divisible by 4 virtual CPU devices, != 2 so accidental tuple unpacking fails).
    x = jnp.ones((4, 16, 16))
    decoder_positions = jnp.broadcast_to(jnp.arange(16), (4, 16))

    def forward_and_loss(xi, block_mod, is_scanned=scan_layers):
      res = block_mod(xi, None, decoder_positions, deterministic=True, model_mode="train")
      out = res[0] if is_scanned else res
      self.assertEqual(out.shape, xi.shape)
      return jnp.sum(out)

    grad_fn = nnx.value_and_grad(forward_and_loss, argnums=(0, nnx.DiffState(1, nnx.Param)))

    with jax.default_matmul_precision("highest"):
      loss_off, (dx_off, grads_off) = grad_fn(x, block_off)
      loss_on, (dx_on, grads_on) = grad_fn(x, block_on)

    np.testing.assert_allclose(loss_off, loss_on, rtol=1e-5, atol=1e-5)

    # Scale-aware tolerance to account for fusion/reassociation differences
    # across XLA compilation strategies and JAX PRNG implementations.
    dx_max = np.maximum(np.max(np.abs(dx_off)), 1e-6)
    np.testing.assert_allclose(dx_off, dx_on, rtol=5e-4, atol=dx_max * 5e-4 + 5e-4)

    grads_off_flat, _ = jax.tree_util.tree_flatten(grads_off)
    grads_on_flat, _ = jax.tree_util.tree_flatten(grads_on)
    self.assertGreater(len(grads_off_flat), 0)
    self.assertEqual(len(grads_off_flat), len(grads_on_flat))
    for go, gn in zip(grads_off_flat, grads_on_flat):
      g_max = np.maximum(np.max(np.abs(go)), 1e-6)
      np.testing.assert_allclose(go, gn, rtol=5e-4, atol=g_max * 5e-4 + 5e-4)

  def test_sublayer_remat_scan_layers_true(self):
    self._run_scannable_block_remat_test(scan_layers=True)

  def test_sublayer_remat_scan_layers_false(self):
    self._run_scannable_block_remat_test(scan_layers=False)


if __name__ == "__main__":
  unittest.main()

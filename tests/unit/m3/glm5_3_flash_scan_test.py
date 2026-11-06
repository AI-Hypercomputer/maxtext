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

"""Tier 1 parity, path-transparency, and invariant tests for GLM-5.3-Flash super-block scan."""

# pylint: disable=protected-access

import dataclasses
import importlib.util
import os

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.configs import glm5_3_flash_config as glm5_config
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
import numpy as np
import pytest

try:
  HAS_TOKAMAX_KDA = "TEST_SRCDIR" in os.environ or importlib.util.find_spec("tokamax._src.ops.kda") is not None
except (ImportError, ValueError):
  HAS_TOKAMAX_KDA = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TOKAMAX_KDA, reason="tokamax KDA not available"),
]


def _make_cfg() -> glm5_config.Glm5Config:
  """Constructs a mini 11-layer Glm5Config for super-block scan tests."""
  f = (
      "hidden_size:64 intermediate_size:128 moe_intermediate_size:32"
      " n_routed_experts:8 num_experts_per_tok:2 q_lora_rank:32 kv_lora_rank:16"
      " num_attention_heads:4 num_key_value_heads:4 qk_nope_head_dim:16"
      " v_head_dim:16 index_n_heads:2 index_head_dim:16 index_topk:8"
      " linear_num_key_heads:4 linear_num_value_heads:4"
      " linear_key_head_dim:16 linear_value_head_dim:16"
  )
  k = {x.split(":")[0]: int(x.split(":")[1]) for x in f.split()}
  return glm5_config.Glm5Config.mini_config(
      num_hidden_layers=11,
      kda_implementation="xla",
      dtype="float32",
      weight_dtype="float32",
      vocab_size=256,
      **k,
  )


class Glm5ScanTest(parameterized.TestCase):
  """Unit tests for Glm5SuperBlock, scan_super_blocks, and apply_glm5_remat."""

  def setUp(self):
    super().setUp()
    self.enter_context(nnx.use_eager_sharding(True))
    np.random.seed(42)
    self.cfg = _make_cfg()

  def _assert_allclose(self, a, b, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol, rtol=rtol)

  def test_scanned_unscanned_and_remat_parity(self):
    """Verifies forward/backward parity with distinct per-block router bias across scan/remat modes."""
    cfg = self.cfg
    num_sb = 2
    sb = modeling_glm5_3_flash.create_stacked_super_block(cfg, num_sb, rngs=nnx.Rngs(42), place_on_devices=False)

    # Initialize all nnx.Param leaves with deterministic non-zero values
    param_vars = jax.tree.leaves(
        nnx.state(sb, nnx.Param),
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    for i, var in enumerate(param_vars):
      var.value = (
          jax.random.normal(
              jax.random.PRNGKey(i + 10),
              var.value.shape,
              dtype=var.value.dtype,
          )
          * 0.1
      )

    # Set distinct non-zero e_score_correction_bias for block 0 vs block 1
    bias_0 = jnp.array([10.0, -10.0, 5.0, -5.0, 0.0, 0.0, 0.0, 0.0])
    bias_1 = jnp.array([-10.0, 10.0, -5.0, 5.0, 5.0, -5.0, 0.0, 0.0])
    for s in range(4):
      slot = getattr(sb, f"slot_{s}")
      b = slot.mlp.gate.e_score_correction_bias
      b.value = b.value.at[0].set(bias_0)
      b.value = b.value.at[1].set(bias_1)

    streams = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 4, cfg.hidden_size))

    # Single super-block __call__ 5-tuple unpacking check
    single_sb = modeling_glm5_3_flash.Glm5SuperBlock(cfg, rngs=nnx.Rngs(0))
    s_out, s_r, s_rt, s_idx, _, _, _ = single_sb(streams, return_aux=False)
    self.assertEqual(s_out.shape, streams.shape)
    self.assertEqual(
        s_r.shape,
        (4, streams.shape[0] * streams.shape[1], cfg.n_routed_experts),
    )
    self.assertEqual(
        s_rt.shape,
        (4, streams.shape[0], streams.shape[1], cfg.num_experts_per_tok),
    )
    self.assertIsInstance(s_idx, tuple)
    self.assertTrue(all(x is None for x in s_idx))

    single_sb_dbg = modeling_glm5_3_flash.Glm5SuperBlock(
        dataclasses.replace(cfg, enable_debug_stats=True), rngs=nnx.Rngs(0)
    )
    _, _, _, s_idx_dbg, _, _, _ = single_sb_dbg(streams, return_aux=False)
    self.assertIsInstance(s_idx_dbg, tuple)
    self.assertTrue(any(x is not None for x in s_idx_dbg))
    for x in s_idx_dbg:
      if x is not None:
        self.assertEqual(x.shape[:2], streams.shape[:2])

    # Run in 3 modes: scanned, unscanned, remat
    pol = jax.checkpoint_policies.nothing_saveable
    out_s, r_s, lb_s, rt_s, idx_s, _, _ = modeling_glm5_3_flash.scan_super_blocks(
        sb,
        streams,
        scan_layers=True,
        remat_policy=None,
        return_router_logits=True,
        return_aux=False,
    )
    out_u, r_u, lb_u, rt_u, idx_u, _, _ = modeling_glm5_3_flash.scan_super_blocks(
        sb,
        streams,
        scan_layers=False,
        remat_policy=None,
        return_router_logits=True,
        return_aux=False,
    )
    out_r, r_r, _, rt_r, _, _, _ = modeling_glm5_3_flash.scan_super_blocks(
        sb,
        streams,
        scan_layers=True,
        remat_policy=pol,
        return_router_logits=True,
        return_aux=False,
    )

    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.scan_super_blocks(sb, streams, scan_layers=True)  # pylint: disable=missing-kwoa

    self.assertIsNone(idx_s)
    self.assertIsNone(idx_u)

    # Flat sequence equivalence test (P2-37)
    fk = cfg.first_k_dense_replace
    flat_layers = [modeling_glm5_3_flash.Glm5DecoderLayer(cfg, fk + i, rngs=nnx.Rngs(0)) for i in range(8)]
    for b in range(num_sb):
      for s in range(4):
        layer_idx = 4 * b + s
        slot = getattr(sb, f"slot_{s}")
        sliced = modeling_glm5_3_flash._slice_state(nnx.state(slot), b)
        nnx.update(flat_layers[layer_idx], sliced)

    flat_streams = streams
    lb_flat = modeling_glm5_3_flash.init_moe_lb_stats(cfg.n_routed_experts)
    for layer in flat_layers:
      flat_streams, r_l, _, _, _, _ = layer(flat_streams, return_aux=False)
      lb_flat = modeling_glm5_3_flash.accumulate_moe_lb_stats(
          lb_flat,
          r_l,
          num_experts=cfg.n_routed_experts,
          top_k=cfg.num_experts_per_tok,
          attention_mask=None,
      )

    self._assert_allclose(out_s, flat_streams)
    self._assert_allclose(lb_s[0], lb_flat[0])
    self._assert_allclose(lb_s[1], lb_flat[1])
    self._assert_allclose(lb_s[2], lb_flat[2])

    # Test return_router_logits=False (P2-34)
    out_no_r, r_no_r, lb_no_r, rt_no_r, _, _, _ = modeling_glm5_3_flash.scan_super_blocks(
        sb,
        streams,
        scan_layers=True,
        return_router_logits=False,
        return_aux=False,
    )
    self.assertEqual(r_no_r, [])
    self.assertEqual(rt_no_r, [])
    self._assert_allclose(out_s, out_no_r)
    self._assert_allclose(lb_s[0], lb_no_r[0])
    self._assert_allclose(lb_s[1], lb_no_r[1])
    self._assert_allclose(lb_s[2], lb_no_r[2])

    self._assert_allclose(out_s, out_u)
    self._assert_allclose(out_s, out_r)
    self.assertEqual(len(r_s), 8)
    self.assertEqual(len(r_u), 8)
    self.assertEqual(len(r_r), 8)
    self.assertEqual(len(rt_s), 8)
    self.assertEqual(len(rt_u), 8)
    self.assertEqual(len(rt_r), 8)
    for logit_s, logit_u, logit_r in zip(r_s, r_u, r_r):
      self._assert_allclose(logit_s, logit_u)
      self._assert_allclose(logit_s, logit_r)
    for topk_s, topk_u, topk_r in zip(rt_s, rt_u, rt_r):
      np.testing.assert_array_equal(np.array(topk_s), np.array(topk_u))
      np.testing.assert_array_equal(np.array(topk_s), np.array(topk_r))
    self._assert_allclose(lb_s[0], lb_u[0])
    self._assert_allclose(lb_s[1], lb_u[1])
    self._assert_allclose(lb_s[2], lb_u[2])

    # Invariant: Distinct router bias changes expert routing compared to zero bias
    sb_zero = modeling_glm5_3_flash.create_stacked_super_block(cfg, num_sb, rngs=nnx.Rngs(42), place_on_devices=False)
    nnx.update(sb_zero, nnx.state(sb, nnx.Param))
    out_zero, r_zero, _, _, _, _, _ = modeling_glm5_3_flash.scan_super_blocks(
        sb_zero,
        streams,
        scan_layers=True,
        remat_policy=None,
        return_router_logits=True,
        return_aux=False,
    )
    self.assertFalse(np.allclose(np.array(out_s), np.array(out_zero)))
    self.assertFalse(np.allclose(np.array(r_s[1]), np.array(r_zero[1])))

    # Backward parity: input gradients and all param gradients match
    def loss_scan(block, x):
      out, r_logits, _, _, _, _, _ = modeling_glm5_3_flash.scan_super_blocks(
          block,
          x,
          scan_layers=True,
          remat_policy=None,
          return_router_logits=True,
          return_aux=False,
      )
      return jnp.mean(out) + sum(jnp.mean(rl) for rl in r_logits)

    def loss_unscan(block, x):
      out, r_logits, _, _, _, _, _ = modeling_glm5_3_flash.scan_super_blocks(
          block,
          x,
          scan_layers=False,
          remat_policy=None,
          return_router_logits=True,
          return_aux=False,
      )
      return jnp.mean(out) + sum(jnp.mean(rl) for rl in r_logits)

    def loss_remat(block, x):
      out, r_logits, _, _, _, _, _ = modeling_glm5_3_flash.scan_super_blocks(
          block,
          x,
          scan_layers=True,
          remat_policy=pol,
          return_router_logits=True,
          return_aux=False,
      )
      return jnp.mean(out) + sum(jnp.mean(rl) for rl in r_logits)

    wrt = nnx.DiffState(0, (nnx.Param, modeling_glm5_3_flash.Glm5RouterBias))
    l_s, (gp_s, gx_s) = nnx.value_and_grad(loss_scan, argnums=(wrt, 1))(sb, streams)
    l_u, (gp_u, gx_u) = nnx.value_and_grad(loss_unscan, argnums=(wrt, 1))(sb, streams)
    l_r, (gp_r, gx_r) = nnx.value_and_grad(loss_remat, argnums=(wrt, 1))(sb, streams)

    self._assert_allclose(l_s, l_u)
    self._assert_allclose(l_s, l_r)
    self._assert_allclose(gx_s, gx_u)
    self._assert_allclose(gx_s, gx_r)

    leaves_s = jax.tree.leaves(nnx.state(gp_s, nnx.Param))
    leaves_u = jax.tree.leaves(nnx.state(gp_u, nnx.Param))
    leaves_r = jax.tree.leaves(nnx.state(gp_r, nnx.Param))
    self.assertEqual(len(leaves_s), len(leaves_u))
    self.assertEqual(len(leaves_s), len(leaves_r))
    for ps, pu, pr in zip(leaves_s, leaves_u, leaves_r):
      self._assert_allclose(ps, pu)
      self._assert_allclose(ps, pr)

    # Invariant: Glm5RouterBias receives zero gradient
    bias_leaves = jax.tree.leaves(nnx.state(gp_s, modeling_glm5_3_flash.Glm5RouterBias))
    self.assertNotEmpty(bias_leaves)
    for b_grad in bias_leaves:
      np.testing.assert_allclose(np.array(b_grad), 0.0, atol=0.0, rtol=0.0)

    # Verify remat is active in the traced computation
    jaxpr = jax.make_jaxpr(
        lambda x: modeling_glm5_3_flash.scan_super_blocks(
            sb,
            x,
            scan_layers=True,
            remat_policy=pol,
            return_router_logits=True,
            return_aux=False,
        )[0]
    )(streams)
    jaxpr_str = str(jaxpr)
    self.assertTrue(
        "remat2" in jaxpr_str or "checkpoint" in jaxpr_str,
        f"Expected remat2 or checkpoint in jaxpr, got:\n{jaxpr_str[:500]}...",
    )

  def _assert_stacked_super_block_sharding(
      self,
      sb: modeling_glm5_3_flash.Glm5SuperBlock,
      mesh: jax.sharding.Mesh,
  ) -> None:
    """Asserts expected sharding specs on stacked super-block parameters."""
    gate = sb.slot_0.mlp.shared_experts.gate_proj.kernel
    expected_spec = jax.sharding.PartitionSpec(None, "fsdp", "tensor")
    expected_sh = jax.sharding.NamedSharding(mesh, expected_spec)
    self.assertEqual(gate.out_sharding, expected_spec)
    self.assertEqual(gate.sharding, expected_sh)
    self.assertEqual(gate.value.sharding, expected_sh)
    bias = sb.slot_0.mlp.gate.e_score_correction_bias
    self.assertEqual(bias.out_sharding, jax.sharding.PartitionSpec(None, None))

  def test_return_aux_scanned_unscanned_parity(self):
    """F49 Invariant: Parity for scanned vs unscanned return_aux execution."""
    cfg = self.cfg
    num_sb = 2
    sb = modeling_glm5_3_flash.create_stacked_super_block(cfg, num_sb, rngs=nnx.Rngs(0), place_on_devices=False)

    # Initialize all nnx.Param leaves with deterministic model-fan-in (1/sqrt(hidden_size)) scale
    fan_in_scale = jax.lax.rsqrt(jnp.float32(cfg.hidden_size))
    param_vars = jax.tree.leaves(
        nnx.state(sb, nnx.Param),
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )
    for i, var in enumerate(param_vars):
      var.value = (
          jax.random.normal(
              jax.random.PRNGKey(i + 10),
              var.value.shape,
              dtype=var.value.dtype,
          )
          * fan_in_scale
      )

    # Set distinct non-zero e_score_correction_bias for block 0 vs block 1
    bias_0 = jnp.array([1e-3, -1e-3, 5e-4, -5e-4, 0.0, 0.0, 0.0, 0.0])
    bias_1 = jnp.array([-2e-2, 2e-2, -1e-2, 1e-2, 2e-2, -2e-2, 0.0, 0.0])
    for s in range(4):
      slot = getattr(sb, f"slot_{s}")
      b = slot.mlp.gate.e_score_correction_bias
      b.value = b.value.at[0].set(bias_0)
      b.value = b.value.at[1].set(bias_1)

    streams = jax.random.uniform(
        jax.random.key(1),
        (1, 16, cfg.mhc_expansion_rate, cfg.hidden_size),
        dtype=jnp.dtype(self.cfg.dtype),
    )

    # Under eager execution (jax.disable_jit()), scan_layers=True and scan_layers=False
    # execute identical operations and match bit-for-bit (assert_array_equal).
    # Under JIT compilation, XLA compiles scan_layers=True as a while-loop body and
    # unrolls scan_layers=False, introducing a ~1-ULP float32 FMA/reassociation difference
    # in attn_hc. Uniform model fan-in scaling (1/sqrt(hidden_size)) is used rather than
    # per-leaf lecun_normal(in_axis=-2) because stacked ShortConv1d (in_axis=1), 3D expert
    # gate_up_proj (in_axis=2), and mHC scale leaves do not have hidden_size on axis -2.

    with jax.disable_jit():
      (
          out_s_eager,
          r_s_eager,
          _,
          rt_s_eager,
          idx_s_eager,
          rw_s_eager,
          rp_s_eager,
      ) = modeling_glm5_3_flash.scan_super_blocks(
          sb,
          streams,
          scan_layers=True,
          remat_policy=None,
          return_router_logits=True,
          return_aux=True,
      )
      (
          out_u_eager,
          r_u_eager,
          _,
          rt_u_eager,
          idx_u_eager,
          rw_u_eager,
          rp_u_eager,
      ) = modeling_glm5_3_flash.scan_super_blocks(
          sb,
          streams,
          scan_layers=False,
          remat_policy=None,
          return_router_logits=True,
          return_aux=True,
      )

      np.testing.assert_array_equal(np.array(out_s_eager), np.array(out_u_eager))
      for i in range(8):
        np.testing.assert_array_equal(np.array(r_s_eager[i]), np.array(r_u_eager[i]))
        np.testing.assert_array_equal(np.array(rp_s_eager[i]), np.array(rp_u_eager[i]))
        np.testing.assert_array_equal(np.array(rt_s_eager[i]), np.array(rt_u_eager[i]))
        np.testing.assert_array_equal(np.array(rw_s_eager[i]), np.array(rw_u_eager[i]))
      for b in range(2):
        self.assertEqual(len(idx_s_eager[b]), 4)
        self.assertEqual(len(idx_u_eager[b]), 4)
        for s in range(4):
          if idx_s_eager[b][s] is not None:
            self.assertIsNotNone(idx_u_eager[b][s])
            np.testing.assert_array_equal(np.array(idx_s_eager[b][s]), np.array(idx_u_eager[b][s]))
          else:
            self.assertIsNone(idx_u_eager[b][s])

    # JIT / Compiled execution (default)
    out_s, r_s, _, rt_s, idx_s, rw_s, rp_s = modeling_glm5_3_flash.scan_super_blocks(
        sb,
        streams,
        scan_layers=True,
        remat_policy=None,
        return_router_logits=True,
        return_aux=True,
    )
    # Unscanned
    out_u, r_u, _, rt_u, idx_u, rw_u, rp_u = modeling_glm5_3_flash.scan_super_blocks(
        sb,
        streams,
        scan_layers=False,
        remat_policy=None,
        return_router_logits=True,
        return_aux=True,
    )

    np.testing.assert_allclose(np.array(out_s), np.array(out_u), atol=1e-5)
    self.assertEqual(len(r_s), 8)
    self.assertEqual(len(rt_s), 8)
    self.assertEqual(len(rw_s), 8)
    self.assertEqual(len(rp_s), 8)
    self.assertEqual(len(idx_s), 2)
    self.assertEqual(len(idx_u), 2)

    for i in range(8):
      np.testing.assert_array_equal(np.array(rt_s[i]), np.array(rt_u[i]))
      np.testing.assert_allclose(np.array(r_s[i]), np.array(r_u[i]), atol=1e-5)
      np.testing.assert_allclose(np.array(rw_s[i]), np.array(rw_u[i]), atol=1e-5)
      np.testing.assert_allclose(np.array(rp_s[i]), np.array(rp_u[i]), atol=1e-5)

    for i in range(2):
      self.assertEqual(len(idx_s[i]), 4)
      self.assertEqual(len(idx_u[i]), 4)
      for s in range(4):
        if idx_s[i][s] is not None:
          self.assertIsNotNone(idx_u[i][s])
          np.testing.assert_array_equal(np.array(idx_s[i][s]), np.array(idx_u[i][s]))
        else:
          self.assertIsNone(idx_u[i][s])

    self.assertFalse(np.array_equal(np.array(idx_s[0][0]), np.array(idx_s[1][0])))
    self.assertFalse(np.array_equal(np.array(idx_s[0][0])[0, 4, :], np.array(idx_s[0][0])[0, 15, :]))
    self.assertFalse(np.array_equal(np.array(rt_s[0]), np.array(rt_s[4])))
    self.assertGreater(len(np.unique(np.array(rt_s[0]))), 2)
    self.assertFalse(np.allclose(np.array(rw_s[0]), 1.0 / cfg.n_routed_experts))

  def test_path_transparency_and_sharding(self):
    """Verifies parameter path transparency and outer-axis replication in sharding."""
    cfg = self.cfg
    single = modeling_glm5_3_flash.Glm5SuperBlock(cfg, rngs=nnx.Rngs(0))
    single_params = nnx.state(single, nnx.Param)

    stacked = modeling_glm5_3_flash.create_stacked_super_block(cfg, 2, rngs=nnx.Rngs(0), place_on_devices=False)
    stacked_params = nnx.state(stacked, nnx.Param)

    # Parameter treedef must match exactly (100% path transparent)
    self.assertEqual(jax.tree.structure(single_params), jax.tree.structure(stacked_params))

    for p_single, p_stacked in zip(jax.tree.leaves(single_params), jax.tree.leaves(stacked_params)):
      self.assertEqual(p_stacked.shape, (2, *p_single.shape))

    # Sharding test: stacked parameters must have PartitionSpec(None, *spec)
    mesh = m3_sharding.create_mesh(devices=jax.devices()[:1])
    sh = glm5_config.Glm5Sharding(use_tp=True)
    with jax.sharding.set_mesh(mesh):
      sharded = modeling_glm5_3_flash.create_stacked_super_block(
          cfg, 2, rngs=nnx.Rngs(0), sharding=sh, place_on_devices=True
      )
    self._assert_stacked_super_block_sharding(sharded, mesh)

  def test_out_sharding_prepend_and_device_placement(self):
    """Verifies prepend axis sharding placement and negative error on missing mesh."""
    cfg = self.cfg
    mesh = m3_sharding.create_mesh(devices=jax.devices()[:1])
    sh = glm5_config.Glm5Sharding(use_tp=True)
    with jax.sharding.set_mesh(mesh):
      sharded = modeling_glm5_3_flash.create_stacked_super_block(
          cfg, 2, rngs=nnx.Rngs(0), sharding=sh, place_on_devices=True
      )
    self._assert_stacked_super_block_sharding(sharded, mesh)

    # Negative test: place_on_devices=True without NamedSharding mesh raises ValueError
    with self.assertRaisesRegex(
        ValueError,
        r"place_on_devices=True requires a non-empty NamedSharding mesh on" r" stacked parameters",
    ):
      modeling_glm5_3_flash.create_stacked_super_block(cfg, 2, rngs=nnx.Rngs(0), sharding=None, place_on_devices=True)

  def test_slice_state_strips_leading_out_sharding(self):
    """Verifies _slice_state strips the leading stacked axis from out_sharding."""
    cfg = self.cfg
    mesh = m3_sharding.create_mesh(devices=jax.devices()[:1])
    sh = glm5_config.Glm5Sharding(use_tp=True)
    with jax.sharding.set_mesh(mesh):
      sb = modeling_glm5_3_flash.create_stacked_super_block(cfg, 2, rngs=nnx.Rngs(0), sharding=sh, place_on_devices=False)
    _, params, non_params = nnx.split(sb, nnx.Param, ...)
    p0 = modeling_glm5_3_flash._slice_state(params, 0)
    gate = p0["slot_0"]["mlp"]["shared_experts"]["gate_proj"]["kernel"]
    self.assertEqual(gate.out_sharding, jax.sharding.PartitionSpec("fsdp", "tensor"))
    np0 = modeling_glm5_3_flash._slice_state(non_params, 0)
    bias = np0["slot_0"]["mlp"]["gate"]["e_score_correction_bias"]
    self.assertEqual(bias.out_sharding, jax.sharding.PartitionSpec(None))

  def test_scan_super_blocks_unstacked_raises(self):
    """Verifies scan_super_blocks raises ValueError when given an unstacked super-block."""
    cfg = self.cfg
    single_sb = modeling_glm5_3_flash.Glm5SuperBlock(cfg, rngs=nnx.Rngs(0))
    streams = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 4, cfg.hidden_size))
    with self.assertRaisesRegex(ValueError, "not stacked"):
      modeling_glm5_3_flash.scan_super_blocks(
          single_sb,
          streams,
          scan_layers=True,
          return_router_logits=True,
          return_aux=False,
      )

  def test_scan_super_blocks_enable_debug_stats(self):
    """Verifies scan_super_blocks with enable_debug_stats=True returns non-None all_idx."""
    cfg = dataclasses.replace(self.cfg, enable_debug_stats=True)
    num_sb = 2
    sb = modeling_glm5_3_flash.create_stacked_super_block(cfg, num_sb, rngs=nnx.Rngs(0), place_on_devices=False)
    streams = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 4, cfg.hidden_size))
    _, _, _, _, all_idx, _, _ = modeling_glm5_3_flash.scan_super_blocks(
        sb,
        streams,
        scan_layers=True,
        return_router_logits=True,
        return_aux=False,
    )
    _, _, _, _, all_idx_unscan, _, _ = modeling_glm5_3_flash.scan_super_blocks(
        sb,
        streams,
        scan_layers=False,
        return_router_logits=True,
        return_aux=False,
    )
    self.assertIsNotNone(all_idx)
    self.assertLen(all_idx, num_sb)
    self.assertLen(all_idx_unscan, num_sb)
    for b in range(num_sb):
      self.assertLen(all_idx[b], 4)
      self.assertLen(all_idx_unscan[b], 4)
      for s in range(4):
        if all_idx[b][s] is not None:
          self.assertEqual(all_idx[b][s].shape[:2], streams.shape[:2])
          np.testing.assert_array_equal(np.asarray(all_idx[b][s]), np.asarray(all_idx_unscan[b][s]))
        else:
          self.assertIsNone(all_idx_unscan[b][s])

  def test_create_stacked_super_block_abstract_eval_shape(self):
    """Verifies create_stacked_super_block works under nnx.eval_shape without device_put."""
    cfg = self.cfg
    mesh = m3_sharding.create_mesh(devices=jax.devices()[:1])
    sh = glm5_config.Glm5Sharding(use_tp=True)
    with jax.sharding.set_mesh(mesh):
      abstract_sb = nnx.eval_shape(
          lambda: modeling_glm5_3_flash.create_stacked_super_block(
              cfg, 2, rngs=nnx.Rngs(0), sharding=sh, place_on_devices=False
          )
      )

    gate = abstract_sb.slot_0.mlp.shared_experts.gate_proj.kernel
    self.assertEqual(
        gate.shape,
        (2, cfg.hidden_size, cfg.moe_intermediate_size * cfg.n_shared_experts),
    )
    self.assertEqual(gate.dtype, jnp.float32)
    self.assertEqual(gate.out_sharding, jax.sharding.PartitionSpec(None, "fsdp", "tensor"))
    self.assertEqual(gate.sharding.spec, jax.sharding.PartitionSpec(None, "fsdp", "tensor"))

    # Negative test: missing sharding metadata when sharding is passed raises ValueError
    class UnshardedSharding:

      def __call__(self, name: str, logical_axes: tuple[object, ...]) -> None:
        return None

    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.create_stacked_super_block(
          cfg,
          2,
          rngs=nnx.Rngs(0),
          sharding=UnshardedSharding(),  # type: ignore[arg-type]
          place_on_devices=False,
      )

  def test_dynamic_first_k_dense_replace(self):
    """Verifies dynamic slot indices fk + s when first_k_dense_replace != 3."""
    k_dims = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "qk_nope_head_dim": 16,
        "v_head_dim": 16,
        "index_n_heads": 2,
        "index_head_dim": 16,
        "index_topk": 8,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
    }
    cfg_fk1 = glm5_config.Glm5Config.mini_config(
        num_hidden_layers=11,
        first_k_dense_replace=1,
        kda_implementation="xla",
        dtype="float32",
        weight_dtype="float32",
        vocab_size=256,
        **k_dims,
    )
    sb_fk1 = modeling_glm5_3_flash.Glm5SuperBlock(cfg_fk1, rngs=nnx.Rngs(0))
    self.assertEqual(sb_fk1.slot_0.layer_idx, 1)
    self.assertEqual(sb_fk1.slot_1.layer_idx, 2)
    self.assertEqual(sb_fk1.slot_2.layer_idx, 3)
    self.assertEqual(sb_fk1.slot_3.layer_idx, 4)

    sec, b_idx, s_idx = modeling_glm5_3_flash.layer_to_block_slot(1, first_k_dense_replace=1, num_hidden_layers=11)
    self.assertEqual(sec, "super_block")
    self.assertEqual(b_idx, 0)
    self.assertEqual(s_idx, 0)

    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.layer_to_block_slot(11, first_k_dense_replace=1, num_hidden_layers=11)

  def test_layer_to_block_slot_and_guards(self):
    """Verifies layer index mapping across all 45 layers and negative validation guards."""
    first_k_dense = 3
    total_layers = 45

    for l in range(total_layers):
      sec, b_idx, s_idx = modeling_glm5_3_flash.layer_to_block_slot(l, first_k_dense, num_hidden_layers=total_layers)
      if l < 3:
        self.assertEqual(sec, "prefix")
        self.assertEqual(b_idx, l)
        self.assertIsNone(s_idx)
      elif l < 43:
        self.assertEqual(sec, "super_block")
        self.assertEqual(b_idx, (l - 3) // 4)
        self.assertEqual(s_idx, (l - 3) % 4)
      else:
        self.assertEqual(sec, "suffix")
        self.assertEqual(b_idx, l - 43)
        self.assertIsNone(s_idx)

    # Negative guard: layer index >= num_hidden_layers
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.layer_to_block_slot(total_layers, first_k_dense, num_hidden_layers=total_layers)

    # Negative guard: negative layer index
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.layer_to_block_slot(-1, first_k_dense, num_hidden_layers=total_layers)

    # Negative guard: non-positive num_hidden_layers
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.layer_to_block_slot(0, first_k_dense, num_hidden_layers=0)
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.layer_to_block_slot(0, first_k_dense, num_hidden_layers=-1)

    # Negative guard: missing num_hidden_layers raises TypeError
    with self.assertRaises(TypeError):
      modeling_glm5_3_flash.layer_to_block_slot(0, 3)  # pylint: disable=no-value-for-parameter

    # Negative guard: non-positive num_super_blocks in factory
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.create_stacked_super_block(self.cfg, 0, rngs=nnx.Rngs(0), place_on_devices=False)
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.create_stacked_super_block(self.cfg, -2, rngs=nnx.Rngs(0), place_on_devices=False)

    # Negative guard: non-periodic schedule in factory (F19)
    bad_lt = list(self.cfg.layer_types)
    bad_lt[7] = "full_attention" if bad_lt[7] == "linear_attention" else "linear_attention"
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.create_stacked_super_block(
          dataclasses.replace(self.cfg, layer_types=tuple(bad_lt)),
          2,
          rngs=nnx.Rngs(0),
          place_on_devices=False,
      )

    # Negative guard: fk + 4 * num_super_blocks > num_hidden_layers (P2-36)
    with self.assertRaisesRegex(ValueError, "exceeds num_hidden_layers"):
      modeling_glm5_3_flash.create_stacked_super_block(self.cfg, 20, rngs=nnx.Rngs(0), place_on_devices=False)

    # apply_glm5_remat requires callable policy (F20)
    class DummyModel:
      pass

    dummy = DummyModel()
    modeling_glm5_3_flash.apply_glm5_remat(dummy, policy=jax.checkpoint_policies.nothing_saveable)
    self.assertEqual(dummy.remat_policy, jax.checkpoint_policies.nothing_saveable)
    with self.assertRaises(ValueError):
      modeling_glm5_3_flash.apply_glm5_remat(dummy, policy=True)


if __name__ == "__main__":
  absltest.main()

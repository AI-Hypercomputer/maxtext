# Copyright 2023-2026 Google LLC
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

"""Unit and numerical parity tests for GLM-5.3-Flash MoE."""

from __future__ import annotations

import dataclasses
import itertools
import math
import os
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.configs import glm5_3_flash_config as glm5_config
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash
import numpy as np
import pytest

try:
  import torch
  from transformers.models.glm5_next import configuration_glm5_next as hf_cfg
  from transformers.models.glm5_next import modeling_glm5_next as hf_model

  HAS_TORCH = True
except ImportError:
  if "TEST_SRCDIR" in os.environ:
    raise
  HAS_TORCH = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="Torch or transformers not available"),
]


def _to_np(x):
  return x.detach().cpu().numpy()


@unittest.skipUnless(HAS_TORCH, "Torch or transformers not available")
class Glm5MoeTest(absltest.TestCase):
  """Numerical parity and invariant tests for Clamped-SwiGLU MLP and MoE."""

  def setUp(self):
    super().setUp()
    self.enter_context(nnx.use_eager_sharding(True))
    self.enter_context(torch.enable_grad())
    torch.manual_seed(42)
    np.random.seed(42)

  def _assert_allclose(self, a, b, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol, rtol=rtol)

  def _mini_cfg(self) -> glm5_config.Glm5Config:
    return glm5_config.Glm5Config.mini_config(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        dtype="float32",
    )

  def _hf_cfg(self, c: glm5_config.Glm5Config) -> hf_cfg.Glm5NextTextConfig:
    return hf_cfg.Glm5NextTextConfig(
        hidden_size=c.hidden_size,
        intermediate_size=c.intermediate_size,
        moe_intermediate_size=c.moe_intermediate_size,
        n_routed_experts=c.n_routed_experts,
        num_experts_per_tok=c.num_experts_per_tok,
        n_group=c.n_group,
        topk_group=c.topk_group,
        norm_topk_prob=c.norm_topk_prob,
        routed_scaling_factor=c.routed_scaling_factor,
        swiglu_limit=c.mlp_clamp_limit,
    )

  def _copy_mlp(self, hf_m: hf_model.Glm5NextTextMLP, jax_m: modeling_glm5_3_flash.Glm5MLP):
    for p in ("gate", "up", "down"):
      w = _to_np(getattr(hf_m, f"{p}_proj").weight).T
      getattr(jax_m, f"{p}_proj").kernel.value = jnp.array(w)

  def _copy_moe(
      self,
      hf_m: hf_model.Glm5NextTextMoE,
      jax_m: modeling_glm5_3_flash.Glm5MoE,
      cfg: glm5_config.Glm5Config,
  ):
    """Copies HuggingFace MoE weights into JAX Glm5MoE module."""
    self._copy_mlp(hf_m.shared_experts, jax_m.shared_experts)
    jax_m.gate.weight.value = jnp.array(_to_np(hf_m.gate.weight).T)
    b_w = _to_np(hf_m.gate.e_score_correction_bias)
    jax_m.gate.e_score_correction_bias.value = jnp.array(b_w)
    gu_w = (
        _to_np(hf_m.experts.gate_up_proj)
        .transpose(0, 2, 1)
        .reshape(cfg.n_routed_experts, cfg.hidden_size, 2, cfg.moe_intermediate_size)
    )
    jax_m.experts.gate_up_proj.value = jnp.array(gu_w)
    d_w = _to_np(hf_m.experts.down_proj).transpose(0, 2, 1)
    jax_m.experts.down_proj.value = jnp.array(d_w)

  def test_mlp_clamped_swiglu(self):
    """Verifies forward/backward parity and clamp saturation for Glm5MLP."""
    cfg = self._mini_cfg()
    hf_mlp = hf_model.Glm5NextTextMLP(self._hf_cfg(cfg)).to(torch.float32)
    jax_mlp = modeling_glm5_3_flash.Glm5MLP(cfg, cfg.intermediate_size, rngs=nnx.Rngs(0))
    self._copy_mlp(hf_mlp, jax_mlp)

    # Standard range inputs
    x_np = np.random.randn(2, 4, cfg.hidden_size).astype(np.float32)
    x_jax, x_pt = jnp.array(x_np), torch.from_numpy(x_np)
    self._assert_allclose(jax_mlp(x_jax), _to_np(hf_mlp(x_pt)))

    cot_np = np.random.randn(*x_np.shape).astype(np.float32)
    cot_pt, cot_j = torch.from_numpy(cot_np), jnp.array(cot_np)
    x_pt_g = torch.from_numpy(x_np).requires_grad_(True)
    (hf_mlp(x_pt_g) * cot_pt).sum().backward()

    vg = nnx.value_and_grad(
        lambda m, x: jnp.sum(m(x) * cot_j),
        argnums=(0, 1),
    )
    _, (grads, grad_x) = vg(jax_mlp, x_jax)
    self._assert_allclose(grad_x, _to_np(x_pt_g.grad))
    for p in ("gate", "up", "down"):
      gw = getattr(hf_mlp, f"{p}_proj").weight.grad.numpy().T
      self._assert_allclose(getattr(grads, f"{p}_proj").kernel.value, gw)

    # Clamp saturation: large inputs forcing gate > 10 and |up| > 10
    x_lg = (torch.randn(2, 4, cfg.hidden_size) * 50.0).requires_grad_(True)
    hf_mlp.zero_grad()
    (hf_mlp(x_lg) * cot_pt).sum().backward()
    _, (_, grad_x_lg) = vg(jax_mlp, jnp.array(_to_np(x_lg)))
    self._assert_allclose(grad_x_lg, _to_np(x_lg.grad))

  def test_moe_forward_backward_and_router_bias(self):
    """Verifies forward/backward parity and non-Param router bias."""
    cfg = self._mini_cfg()
    hf_moe = hf_model.Glm5NextTextMoE(self._hf_cfg(cfg)).to(torch.float32)
    with torch.no_grad():
      for p in hf_moe.parameters():
        p.normal_(0, 0.02)
    jax_moe = modeling_glm5_3_flash.Glm5MoE(cfg, rngs=nnx.Rngs(0))
    self._copy_moe(hf_moe, jax_moe, cfg)

    # 1. Non-Param invariant check
    self.assertIsInstance(
        jax_moe.gate.e_score_correction_bias,
        modeling_glm5_3_flash.Glm5RouterBias,
    )
    self.assertNotIsInstance(jax_moe.gate.e_score_correction_bias, nnx.Param)

    x_np = np.random.randn(2, 4, cfg.hidden_size).astype(np.float32)
    x_jax, x_pt = jnp.array(x_np), torch.from_numpy(x_np)
    self._assert_allclose(jax_moe(x_jax)[0], _to_np(hf_moe(x_pt)))

    cot_np = np.random.randn(*x_np.shape).astype(np.float32)
    cot_pt, cot_j = torch.from_numpy(cot_np), jnp.array(cot_np)
    x_pt_g = torch.from_numpy(x_np).requires_grad_(True)
    (hf_moe(x_pt_g) * cot_pt).sum().backward()

    vg = nnx.value_and_grad(
        lambda m, x: jnp.sum(m(x)[0] * cot_j),
        argnums=(0, 1),
    )
    _, (grads, grad_x) = vg(jax_moe, x_jax)
    self._assert_allclose(grad_x, _to_np(x_pt_g.grad))
    gw_g = _to_np(hf_moe.gate.weight.grad).T
    self._assert_allclose(grads.gate.weight.value, gw_g)
    for p in ("gate_up_proj", "down_proj"):
      if p == "gate_up_proj":
        exp_g = (
            _to_np(hf_moe.experts.gate_up_proj.grad)
            .transpose(0, 2, 1)
            .reshape(
                cfg.n_routed_experts,
                cfg.hidden_size,
                2,
                cfg.moe_intermediate_size,
            )
        )
      else:
        exp_g = _to_np(hf_moe.experts.down_proj.grad).transpose(0, 2, 1)
      self._assert_allclose(getattr(grads.experts, p).value, exp_g)
    self.assertFalse(hasattr(grads.gate, "e_score_correction_bias"))

    # 2. Router bias shifting expert selection
    bias_np = np.array([10.0, -10.0, 5.0, 0.0, -5.0, 2.0, 1.0, -1.0])
    jax_moe.gate.e_score_correction_bias.value = jnp.array(bias_np)
    hf_moe.gate.e_score_correction_bias.copy_(torch.from_numpy(bias_np))
    self._assert_allclose(jax_moe(x_jax)[0], _to_np(hf_moe(x_pt)))

  def test_moe_bfloat16_fp32_combine(self):
    """Verifies that bfloat16 Glm5MoE accumulates weighted combine in float32."""
    cfg = glm5_config.Glm5Config.mini_config(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=5,
        dtype="bfloat16",
        weight_dtype="float32",
    )
    jax_moe = modeling_glm5_3_flash.Glm5MoE(cfg, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(42), (2, 4, cfg.hidden_size), dtype=jnp.bfloat16)
    out, (r_logits, topk_indices, _, _) = jax_moe(x)
    self.assertEqual(out.dtype, jnp.bfloat16)
    self.assertEqual(r_logits.dtype, jnp.float32)
    self.assertEqual(topk_indices.shape, (2, 4, cfg.num_experts_per_tok))
    self.assertTrue(jnp.all(jnp.isfinite(out)))
    self.assertGreater(float(jnp.max(jnp.abs(out))), 1e-4)

    c = cfg
    h_1 = jnp.ones((1, c.hidden_size), dtype=jnp.bfloat16)
    topk_idx_1 = jnp.arange(c.num_experts_per_tok, dtype=jnp.int32)[None, :]
    topk_weights_1 = jnp.array([[1.0] + [0.002] * (c.num_experts_per_tok - 1)], dtype=jnp.float32)

    gu_w = np.zeros(jax_moe.experts.gate_up_proj[...].shape, dtype=np.float32)
    gu_w[:, 0, 0, 0] = 1.0
    gu_w[:, 0, 1, 0] = float(1.0 + math.exp(-1.0))
    d_w = np.zeros(jax_moe.experts.down_proj[...].shape, dtype=np.float32)
    d_w[:, 0, :] = 1.0
    jax_moe.experts.gate_up_proj[...] = jnp.array(gu_w)
    jax_moe.experts.down_proj[...] = jnp.array(d_w)

    expert_outs_bf16 = jnp.ones((1, c.num_experts_per_tok, c.hidden_size), dtype=jnp.bfloat16)
    ref_fp32_combine = jnp.sum(
        expert_outs_bf16.astype(jnp.float32) * topk_weights_1[:, :, None],
        axis=1,
    ).astype(jnp.bfloat16)
    naive_bf16_combine = expert_outs_bf16[:, 0, :] * topk_weights_1[:, 0, None].astype(jnp.bfloat16)
    for k_i in range(1, c.num_experts_per_tok):
      naive_bf16_combine = (
          naive_bf16_combine + (expert_outs_bf16[:, k_i, :] * topk_weights_1[:, k_i, None].astype(jnp.bfloat16))
      ).astype(jnp.bfloat16)

    out_1 = jax_moe.experts(h_1, topk_idx_1, topk_weights_1)
    self.assertEqual(out_1.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(out_1), np.asarray(ref_fp32_combine))
    self.assertFalse(np.array_equal(np.asarray(out_1), np.asarray(naive_bf16_combine)))

  def test_moe_grouped_routing_parity(self):
    """Verifies forward and backward parity for n_group=4, topk_group=2 with non-zero router bias."""
    cfg = glm5_config.Glm5Config.mini_config(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        n_routed_experts=16,
        num_experts_per_tok=4,
        n_group=4,
        topk_group=2,
        dtype="float32",
    )
    hf_moe = hf_model.Glm5NextTextMoE(self._hf_cfg(cfg)).to(torch.float32)
    with torch.no_grad():
      for p in hf_moe.parameters():
        p.normal_(0, 0.02)
      bias_np = np.random.randn(16).astype(np.float32)
      hf_moe.gate.e_score_correction_bias.copy_(torch.from_numpy(bias_np))

    jax_moe = modeling_glm5_3_flash.Glm5MoE(cfg, rngs=nnx.Rngs(0))
    self._copy_moe(hf_moe, jax_moe, cfg)

    x_np = np.random.randn(2, 4, cfg.hidden_size).astype(np.float32)
    x_jax, x_pt = jnp.array(x_np), torch.from_numpy(x_np)
    self._assert_allclose(jax_moe(x_jax)[0], _to_np(hf_moe(x_pt)), atol=1e-5)

    cot_np = np.random.randn(*x_np.shape).astype(np.float32)
    cot_pt, cot_j = torch.from_numpy(cot_np), jnp.array(cot_np)
    x_pt_g = torch.from_numpy(x_np).requires_grad_(True)
    (hf_moe(x_pt_g) * cot_pt).sum().backward()

    vg = nnx.value_and_grad(
        lambda m, x: jnp.sum(m(x)[0] * cot_j),
        argnums=(0, 1),
    )
    _, (grads, grad_x) = vg(jax_moe, x_jax)
    self._assert_allclose(grad_x, _to_np(x_pt_g.grad), atol=1e-5)
    self._assert_allclose(grads.gate.weight.value, _to_np(hf_moe.gate.weight.grad).T, atol=1e-5)

  def test_load_balancing_loss(self):
    """Verifies load balancing loss matching HF load_balancing_loss_func and differentiability."""
    b, s, e, k = 2, 4, 8, 2
    logits_np = [np.random.randn(b * s, e).astype(np.float32) for _ in range(2)]
    l_jax = tuple(jnp.array(x) for x in logits_np)
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=np.int32)

    for m in (None, mask):
      jax_m = None if m is None else jnp.array(m)
      loss_val = modeling_glm5_3_flash.compute_moe_load_balancing_loss(l_jax, e, k, attention_mask=jax_m)
      self.assertTrue(jnp.isfinite(loss_val))
      self.assertGreater(float(loss_val), 0.0)

      hf_loss = hf_model.load_balancing_loss_func(
          tuple(torch.from_numpy(x) for x in logits_np),
          num_experts=e,
          top_k=k,
          attention_mask=None if m is None else torch.from_numpy(m),
      )
      self._assert_allclose(loss_val, hf_loss.numpy(), atol=1e-5, rtol=1e-5)

    zero_mask = jnp.zeros((b, s), dtype=jnp.int32)
    zero_mask_loss = modeling_glm5_3_flash.compute_moe_load_balancing_loss(l_jax, e, k, attention_mask=zero_mask)
    self.assertTrue(np.isfinite(float(zero_mask_loss)))
    self.assertEqual(float(zero_mask_loss), 0.0)

    # Uniform routing test
    uniform_logits = jnp.zeros((e, e), dtype=jnp.float32)
    for r in range(e):
      for c_i in range(k):
        uniform_logits = uniform_logits.at[r, (r + c_i) % e].set(1e-3 * (k - c_i))
    uniform_loss = modeling_glm5_3_flash.compute_moe_load_balancing_loss((uniform_logits,), e, k)
    self.assertAlmostEqual(float(uniform_loss), float(k), places=3)

    # Differentiability check: verify loss produces non-zero gradients wrt logits
    g_lg = jax.grad(lambda lg: modeling_glm5_3_flash.compute_moe_load_balancing_loss((lg,), e, k))(l_jax[0])
    self.assertTrue(jnp.all(jnp.isfinite(g_lg)))
    self.assertGreater(float(jnp.max(jnp.abs(g_lg))), 1e-4)

    with self.assertRaisesRegex(ValueError, "non-empty"):
      modeling_glm5_3_flash.compute_moe_load_balancing_loss((), e, k)

  def test_moe_sharding_and_guards(self):
    """Verifies sharding metadata and dense einsum reference parity."""
    cfg = self._mini_cfg()
    mesh = m3_sharding.create_mesh(devices=jax.devices()[:1])
    shd = glm5_config.Glm5Sharding(use_tp=True)
    with jax.sharding.set_mesh(mesh):
      moe = modeling_glm5_3_flash.Glm5MoE(cfg, rngs=nnx.Rngs(0), sharding=shd)
    P, NS = jax.sharding.PartitionSpec, jax.sharding.NamedSharding
    exp = moe.experts
    self.assertEqual(exp.gate_up_proj.sharding, NS(mesh, P("expert", "fsdp", None, "tensor")))
    self.assertEqual(exp.down_proj.sharding, NS(mesh, P("expert", "tensor", "fsdp")))
    self.assertIsNotNone(moe.gate.weight.sharding)
    self.assertIsNotNone(moe.shared_experts.gate_proj.kernel.sharding)

    # Constructor guard: scoring_func
    with self.assertRaisesRegex(ValueError, "must be 'sigmoid'"):
      modeling_glm5_3_flash.Glm5TopkRouter(dataclasses.replace(cfg, scoring_func="softmax"), rngs=nnx.Rngs(0))

    # Dense einsum reference parity for Glm5Experts
    exp_test = modeling_glm5_3_flash.Glm5Experts(cfg, rngs=nnx.Rngs(1))
    x_test = jax.random.normal(jax.random.PRNGKey(0), (4, cfg.hidden_size), dtype=jnp.float32)
    topk_idx = jnp.array([[0, 2], [1, 3], [2, 0], [3, 1]], dtype=jnp.int32)
    topk_w = jax.nn.softmax(
        jax.random.normal(jax.random.PRNGKey(1), (4, 2), dtype=jnp.float32),
        axis=-1,
    )

    out_exp = exp_test(x_test, topk_idx, topk_w)

    # Compute dense reference
    gu_arr = exp_test.gate_up_proj.value
    d_arr = exp_test.down_proj.value
    ref_out = np.zeros((4, cfg.hidden_size), dtype=np.float32)
    for n in range(4):
      for k_i in range(2):
        e_idx = int(topk_idx[n, k_i])
        w_gu = gu_arr[e_idx]  # [H, 2, I]
        gu_val = np.einsum("h,hti->ti", np.array(x_test[n]), np.array(w_gu))
        gate = np.minimum(gu_val[0], cfg.mlp_clamp_limit)
        up = np.clip(gu_val[1], -cfg.mlp_clamp_limit, cfg.mlp_clamp_limit)
        act = (gate / (1.0 + np.exp(-gate))) * up  # silu(gate) * up
        w_d = d_arr[e_idx]  # [I, H]
        out_k = np.einsum("i,ih->h", act, np.array(w_d))
        ref_out[n] += float(topk_w[n, k_i]) * out_k
    self._assert_allclose(out_exp, ref_out, atol=1e-5)

  def test_ep_sharding_guard(self):
    """Verifies ep>1 construction-time and runtime guards on a real 4-device mesh."""
    if len(jax.devices()) < 4:
      if "TEST_SRCDIR" in os.environ:
        self.fail("Expected >= 4 JAX devices under Blaze XLA_FLAGS, got" f" {len(jax.devices())}")
      self.skipTest(f"Requires >= 4 JAX devices, got {len(jax.devices())}")
    cfg = self._mini_cfg()
    mesh_ep4 = m3_sharding.create_mesh(ici={"expert": 4}, devices=jax.devices()[:4])
    shd_ep4 = glm5_config.Glm5Sharding.from_mesh(mesh_ep4)

    # 1. Verify Glm5MoE constructor raises NotImplementedError
    with self.assertRaisesRegex(NotImplementedError, "expert parallelism"):
      modeling_glm5_3_flash.Glm5MoE(cfg, sharding=shd_ep4, rngs=nnx.Rngs(0))

    # 2. Verify Glm5Experts constructor raises NotImplementedError
    with self.assertRaisesRegex(NotImplementedError, "expert parallelism"):
      modeling_glm5_3_flash.Glm5Experts(cfg, sharding=shd_ep4, rngs=nnx.Rngs(0))

    # 3. Construct Glm5Experts without sharding and place x on mesh_ep4
    exp_guard = modeling_glm5_3_flash.Glm5Experts(cfg, rngs=nnx.Rngs(0))
    x_test = jax.random.normal(jax.random.PRNGKey(0), (4, cfg.hidden_size), dtype=jnp.float32)
    topk_idx = jnp.array([[0, 2], [1, 3], [2, 0], [3, 1]], dtype=jnp.int32)
    topk_w = jax.nn.softmax(
        jax.random.normal(jax.random.PRNGKey(1), (4, 2), dtype=jnp.float32),
        axis=-1,
    )
    x_ep4 = jax.device_put(
        x_test,
        jax.sharding.NamedSharding(mesh_ep4, jax.sharding.PartitionSpec("expert", None)),
    )
    with self.assertRaisesRegex(NotImplementedError, "requires shard_map EP dispatch"):
      jax.jit(lambda m, inp, idx, w: m(inp, idx, w))(exp_guard, x_ep4, topk_idx, topk_w)

  def test_experts_tp_sharding_no_weight_all_gather(self):
    """Verifies that TP sharding on Glm5Experts does not all-gather full expert weights."""
    # pylint: disable=too-many-function-args
    if len(jax.devices()) < 4:
      if "TEST_SRCDIR" in os.environ:
        self.fail("Expected >= 4 JAX devices under Blaze XLA_FLAGS, got" f" {len(jax.devices())}")
      self.skipTest(f"Requires >= 4 JAX devices, got {len(jax.devices())}")
    cfg = self._mini_cfg()

    devices = jax.devices()[:4]
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(1, 2, 2, 1),
        ("data", "fsdp", "tensor", "expert"),
    )

    def _require_mesh_devices(m: jax.sharding.Mesh) -> None:
      if len(jax.devices()) < m.size:
        raise RuntimeError(f"Require at least {m.size} devices")

    _require_mesh_devices(mesh)

    # Positive control (F61 / E83): prove the mesh.size RuntimeError guard is reachable when jax.devices() < mesh.size.
    # Mock jax.devices around only this single call per crew_coder_standards.md.
    with mock.patch.object(jax, "devices", return_value=devices[:2]):
      with self.assertRaisesRegex(RuntimeError, "Require at least 4 devices"):
        _require_mesh_devices(mesh)

    def _check_no_expert_weight_all_gather(
        hlo_text: str,
        forbidden_shapes: tuple[str, ...],
    ) -> None:
      violations = []
      for line in hlo_text.splitlines():
        if "all-gather" in line:
          compact = line.replace(" ", "")
          for shape in forbidden_shapes:
            if shape in compact:
              violations.append((shape, line.strip()))
      if violations:
        raise AssertionError(f"Detected prohibited expert weight all-gather in HLO: {violations}")

    # Positive Controls: Synthetic Negative and Positive
    synthetic_hlo_fail = "%all-gather.1 = f32[8,64,32] all-gather(%x), dimensions={2}"
    with self.assertRaisesRegex(AssertionError, "8,64,32"):
      _check_no_expert_weight_all_gather(synthetic_hlo_fail, ("8,64,32",))

    synthetic_hlo_fail_slice = "%all-gather.2 = f32[8,32,32] all-gather(%x), dimensions={2}"
    with self.assertRaisesRegex(AssertionError, "8,32,32"):
      _check_no_expert_weight_all_gather(synthetic_hlo_fail_slice, ("8,32,32",))

    synthetic_hlo_pass = "%reshape.1 = f32[8,64,32] reshape(%x)"
    _check_no_expert_weight_all_gather(synthetic_hlo_pass, ("8,64,32",))

    shd = glm5_config.Glm5Sharding.from_mesh(mesh)

    rngs_ref = nnx.Rngs(42)
    moe_ref = modeling_glm5_3_flash.Glm5MoE(cfg, rngs=rngs_ref)

    with jax.sharding.set_mesh(mesh):
      rngs_shd = nnx.Rngs(42)
      moe_shd = modeling_glm5_3_flash.Glm5MoE(cfg, rngs=rngs_shd, sharding=shd)

    # Sync weights to exact match
    state_ref = nnx.state(moe_ref)
    state_shd = nnx.state(moe_shd)
    for r_var, s_var in zip(
        jax.tree.leaves(state_ref, is_leaf=lambda x: isinstance(x, nnx.Variable)),
        jax.tree.leaves(state_shd, is_leaf=lambda x: isinstance(x, nnx.Variable)),
        strict=True,
    ):
      s_var.value = jax.device_put(r_var.value, s_var.value.sharding)

    experts = moe_shd.experts
    # Check that gate_up_proj shards axis -1 across "tensor"
    self.assertEqual(experts.gate_up_proj.value.sharding.spec[-1], "tensor")
    local_shape = experts.gate_up_proj.value.addressable_shards[0].data.shape
    self.assertEqual(local_shape[-1], 16)
    self.assertEqual(local_shape[0], 8)
    self.assertEqual(local_shape[2], 2)

    # Verify numerical parity of Glm5MoE output against unsharded reference
    b, s, h = 2, 4, cfg.hidden_size
    x = jax.random.normal(jax.random.PRNGKey(0), (b, s, h), dtype=jnp.float32)
    out_ref, _ = moe_ref(x)

    x_sharded = jax.device_put(
        x,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", "fsdp", None)),
    )
    out_shd, _ = jax.jit(moe_shd)(x_sharded)
    np.testing.assert_allclose(np.array(out_shd), np.array(out_ref), rtol=1e-5, atol=1e-5)

    # Derive forbidden shapes dynamically from each expert weight
    def _derive_forbidden_shapes(weight_var: nnx.Variable) -> set[str]:
      g_shape = tuple(weight_var.value.shape)
      s_shape = tuple(weight_var.value.addressable_shards[0].data.shape)
      axes_choices = []
      for g, s in zip(g_shape, s_shape):
        if g == s:
          axes_choices.append([g])
        else:
          axes_choices.append([s, g])
      shapes = set()
      for comb in itertools.product(*axes_choices):
        if comb != s_shape:
          shapes.add(",".join(str(d) for d in comb))
      return shapes

    forbidden_shapes = tuple(
        sorted(_derive_forbidden_shapes(experts.gate_up_proj) | _derive_forbidden_shapes(experts.down_proj))
    )

    # Lower and compile experts with module passed as explicit jit argument
    h_flat = x_sharded.reshape(-1, h)
    _, topk_weights, topk_indices, _ = moe_shd.gate(h_flat)

    def _experts_hlo(experts_mod: Any) -> str:
      graphdef, state = nnx.split(experts_mod)

      @jax.jit
      def _call(st, inp, idx, w):
        mod = nnx.merge(graphdef, st)
        return mod(inp, idx, w)

      lowered = _call.lower(state, h_flat, topk_indices, topk_weights)
      return lowered.compile().as_text()

    # Real compilation check: no forbidden shapes present
    real_hlo = _experts_hlo(experts)
    _check_no_expert_weight_all_gather(real_hlo, forbidden_shapes)

    # Regressed positive control using the exact same _experts_hlo helper
    class _RegressedGlm5Experts(modeling_glm5_3_flash.Glm5Experts):

      def __call__(self, hidden_states, topk_indices, topk_weights):
        gathered_gate_up = jax.lax.with_sharding_constraint(
            self.gate_up_proj.value,
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        )
        self.gate_up_proj.value = gathered_gate_up
        return super().__call__(hidden_states, topk_indices, topk_weights)

    with jax.sharding.set_mesh(mesh):
      regressed_experts = _RegressedGlm5Experts(cfg, sharding=shd, rngs=nnx.Rngs(42))
    reg_state = nnx.state(regressed_experts)
    exp_state = nnx.state(experts)
    for r_var, e_var in zip(
        jax.tree.leaves(reg_state, is_leaf=lambda x: isinstance(x, nnx.Variable)),
        jax.tree.leaves(exp_state, is_leaf=lambda x: isinstance(x, nnx.Variable)),
        strict=True,
    ):
      r_var.value = jax.device_put(e_var.value, r_var.value.sharding)

    ctrl_hlo = _experts_hlo(regressed_experts)
    self.assertIn("all-gather", ctrl_hlo)
    with self.assertRaises(AssertionError):
      _check_no_expert_weight_all_gather(ctrl_hlo, forbidden_shapes)

  def test_router_param_dtype_independent_of_weight_dtype(self):
    """Verifies router parameters use router_param_dtype (float32) when weight_dtype is bfloat16."""
    cfg = glm5_config.Glm5Config.mini_config(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        dtype="bfloat16",
        weight_dtype="bfloat16",
        router_param_dtype="float32",
    )
    moe = modeling_glm5_3_flash.Glm5MoE(cfg, rngs=nnx.Rngs(0))
    self.assertEqual(moe.gate.weight[...].dtype, jnp.float32)
    self.assertEqual(moe.gate.e_score_correction_bias[...].dtype, jnp.float32)
    self.assertEqual(moe.experts.gate_up_proj[...].dtype, jnp.bfloat16)
    self.assertEqual(moe.experts.down_proj[...].dtype, jnp.bfloat16)
    self.assertEqual(moe.shared_experts.gate_proj.kernel[...].dtype, jnp.bfloat16)
    self.assertEqual(moe.shared_experts.up_proj.kernel[...].dtype, jnp.bfloat16)
    self.assertEqual(moe.shared_experts.down_proj.kernel[...].dtype, jnp.bfloat16)


if __name__ == "__main__":
  absltest.main()

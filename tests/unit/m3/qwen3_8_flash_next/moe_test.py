# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Qwen3.8 Sparse MoE block."""

from typing import Any
import unittest

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.qwen3_8_flash_next import moe  # pylint: disable=no-name-in-module
import numpy as np
import pytest

torch: Any
configuration_qwen4_exp: Any
modeling_qwen4_exp: Any

try:
  import torch  # pylint: disable=g-import-not-at-top
  from transformers.models.qwen4_exp import configuration_qwen4_exp  # pylint: disable=g-import-not-at-top
  from transformers.models.qwen4_exp import modeling_qwen4_exp  # pylint: disable=g-import-not-at-top

  _HAS_QWEN4_EXP = True
except ImportError:
  torch = None
  configuration_qwen4_exp = None
  modeling_qwen4_exp = None
  _HAS_QWEN4_EXP = False


@pytest.mark.cpu_only
class Qwen3_8MoETest(absltest.TestCase):

  def _make_moe(self, **kwargs) -> moe.Qwen3_8SparseMoeBlock:
    """Helper to instantiate Qwen3_8SparseMoeBlock with test defaults."""
    cfg = {
        "emb_dim": 32,
        "moe_mlp_dim": 16,
        "shared_expert_intermediate_size": 16,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.0,
        "dtype": jnp.float32,
        "weight_dtype": jnp.float32,
        "rngs": nnx.Rngs(42),
    }
    cfg.update(kwargs)
    return moe.Qwen3_8SparseMoeBlock(**cfg)

  def test_moe_parameter_tree_contract(self):
    mod = self._make_moe(emb_dim=64, moe_mlp_dim=32, shared_expert_intermediate_size=48, num_experts=16)
    for p, shape in [
        (mod.gate.linear.kernel.value, (64, 16)),
        (mod.experts.gate_up_proj.value, (16, 64, 2 * 32)),
        (mod.experts.down_proj.value, (16, 32, 64)),
        (mod.shared_expert.gate_proj.kernel.value, (64, 48)),
        (mod.shared_expert.up_proj.kernel.value, (64, 48)),
        (mod.shared_expert.down_proj.kernel.value, (48, 64)),
        (mod.shared_expert_gate.kernel.value, (64, 1)),
    ]:
      self.assertEqual(p.shape, shape)
      self.assertEqual(p.dtype, jnp.float32)

    with self.assertRaises(ValueError):
      self._make_moe(shared_expert_intermediate_size=0)

  def test_router_topk_normalization_and_shapes(self):
    b, s, d, k = 2, 4, 32, 2
    mod = self._make_moe(emb_dim=d, num_experts_per_tok=k, norm_topk_prob=True)
    x = jax.random.normal(jax.random.key(0), (b, s, d))

    out = mod(x)
    self.assertIsInstance(out, jax.Array)
    self.assertEqual(out.shape, (b, s, d))
    self.assertTrue(np.all(np.isfinite(out)))

    res = mod(x, return_router_topk=True)
    self.assertIsInstance(res, tuple)
    comb, weights, indices, probs = res
    self.assertEqual(comb.shape, (b, s, d))
    self.assertEqual(weights.shape, (b, s, k))
    self.assertEqual(indices.shape, (b, s, k))
    self.assertEqual(probs.shape, (b, s, mod.num_experts))
    np.testing.assert_allclose(comb, out, atol=1e-5)

    # Probabilities normalized to 1.0 along top_k
    sum_weights = jnp.sum(weights, axis=-1)
    np.testing.assert_allclose(sum_weights, np.ones((b, s)), atol=1e-5)
    # Full router probs normalized to 1.0 along num_experts and non-negative
    sum_probs = jnp.sum(probs, axis=-1)
    np.testing.assert_allclose(sum_probs, np.ones((b, s)), atol=1e-5)
    self.assertTrue(np.all(probs >= 0.0))
    # Expert indices are within valid range [0, num_experts - 1]
    self.assertTrue(np.all((indices >= 0) & (indices < mod.num_experts)))

  def test_experts_ragged_dot_matches_reference_forward(self):
    n, d, m, e, k = 8, 32, 16, 12, 2
    experts = moe.Qwen3_8Experts(
        num_experts=e, emb_dim=d, moe_mlp_dim=m, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=nnx.Rngs(0)
    )
    x = jax.random.normal(jax.random.key(1), (n, d))

    # Random top-k routing
    top_k_idx = jax.random.randint(jax.random.key(2), (n, k), 0, e)
    top_k_w = jax.random.uniform(jax.random.key(3), (n, k))
    top_k_w = top_k_w / jnp.sum(top_k_w, axis=-1, keepdims=True)

    actual = experts(x, top_k_idx, top_k_w)
    expected = experts.reference_forward(x, top_k_idx, top_k_w)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    # Skewed load with group_size == 0 for several experts
    skewed_idx = jnp.zeros((n, k), dtype=jnp.int32).at[:, 1].set(1)
    actual_skewed = experts(x, skewed_idx, top_k_w)
    expected_skewed = experts.reference_forward(x, skewed_idx, top_k_w)
    np.testing.assert_allclose(actual_skewed, expected_skewed, rtol=1e-5, atol=1e-5)

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_pytorch_moe_weight_copy_parity(self):
    if torch is None or modeling_qwen4_exp is None or configuration_qwen4_exp is None:
      self.fail("transformers qwen4_exp reference modules are missing")
    torch.manual_seed(0)
    b, s, h, i, e, k, shared = 2, 3, 32, 16, 8, 2, 16
    hf_cfg = configuration_qwen4_exp.Qwen4ExpTextConfig(
        hidden_size=h,
        moe_intermediate_size=i,
        shared_expert_intermediate_size=shared,
        num_experts=e,
        num_experts_per_tok=k,
        norm_topk_prob=True,
        hidden_act="silu",
    )
    pt_moe = modeling_qwen4_exp.Qwen4ExpTextSparseMoeBlock(hf_cfg)
    torch.manual_seed(0)
    with torch.no_grad():
      for p in pt_moe.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02)
    jax_moe = moe.Qwen3_8SparseMoeBlock(
        emb_dim=h,
        moe_mlp_dim=i,
        shared_expert_intermediate_size=shared,
        num_experts=e,
        num_experts_per_tok=k,
        norm_topk_prob=True,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )

    # Weight copy with HF -> MaxText layout transpositions:
    # gate_up_proj: HF (E, 2I, H) -> MaxText (E, H, 2I)
    # down_proj: HF (E, H, I) -> MaxText (E, I, H)
    jax_moe.gate.linear.kernel.value = jnp.array(pt_moe.gate.weight.detach().cpu().numpy().T)
    jax_moe.experts.gate_up_proj.value = jnp.swapaxes(
        jnp.array(pt_moe.experts.gate_up_proj.detach().cpu().numpy()), -1, -2
    )
    jax_moe.experts.down_proj.value = jnp.swapaxes(jnp.array(pt_moe.experts.down_proj.detach().cpu().numpy()), -1, -2)
    jax_moe.shared_expert.gate_proj.kernel.value = jnp.array(
        pt_moe.shared_expert.gate_proj.weight.detach().cpu().numpy().T
    )
    jax_moe.shared_expert.up_proj.kernel.value = jnp.array(pt_moe.shared_expert.up_proj.weight.detach().cpu().numpy().T)
    jax_moe.shared_expert.down_proj.kernel.value = jnp.array(
        pt_moe.shared_expert.down_proj.weight.detach().cpu().numpy().T
    )
    jax_moe.shared_expert_gate.kernel.value = jnp.array(pt_moe.shared_expert_gate.weight.detach().cpu().numpy().T)

    rng = np.random.default_rng(42)
    x = rng.standard_normal((b, s, h)).astype(np.float32)

    pt_out = pt_moe(torch.from_numpy(x)).detach().cpu().numpy()
    pt_logits, pt_w, pt_idx = pt_moe.gate(torch.from_numpy(x))
    pt_probs = torch.nn.functional.softmax(pt_logits, dim=-1).detach().cpu().numpy()

    jax_out, jax_w, jax_idx, jax_probs = jax_moe(jnp.array(x), return_router_topk=True)

    np.testing.assert_allclose(np.array(jax_probs), pt_probs.reshape(b, s, e), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(jax_w), pt_w.detach().cpu().numpy().reshape(b, s, k), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.array(jax_idx), pt_idx.detach().cpu().numpy().reshape(b, s, k))
    np.testing.assert_allclose(np.array(jax_out), pt_out, rtol=1e-6, atol=1e-6)

  def test_shared_expert_gating_behavior(self):
    b, s, d = 2, 4, 32
    mod = self._make_moe(emb_dim=d)
    # Use positive inputs so uniform gate weights reliably saturate sigmoid to 0 or 1.
    x = jnp.abs(jax.random.normal(jax.random.key(2), (b, s, d))) + 0.1

    flat_x = x.reshape(-1, d)
    _, weights, indices, _ = mod(x, return_router_topk=True)
    k = mod.num_experts_per_tok
    routed_only = mod.experts(flat_x, indices.reshape(-1, k), weights.reshape(-1, k)).reshape(b, s, d)
    shared_only = mod.shared_expert(flat_x).reshape(b, s, d)

    # Large negative gate bias zeroes out shared expert
    mod.shared_expert_gate.kernel.value = jnp.full_like(mod.shared_expert_gate.kernel.value, -100.0)
    out_zero = mod(x)
    self.assertIsInstance(out_zero, jax.Array)
    np.testing.assert_allclose(out_zero, routed_only, atol=1e-4)

    # Large positive gate bias passes 100% of shared expert
    mod.shared_expert_gate.kernel.value = jnp.full_like(mod.shared_expert_gate.kernel.value, 100.0)
    out_one = mod(x)
    self.assertIsInstance(out_one, jax.Array)
    np.testing.assert_allclose(out_one, routed_only + shared_only, atol=1e-4)

  def test_moe_backward_gradients_finite_and_nonzero(self):
    b, s, d = 2, 4, 32
    mod = self._make_moe(emb_dim=d, moe_mlp_dim=16, shared_expert_intermediate_size=16)
    x = jax.random.normal(jax.random.key(0), (b, s, d))

    def loss_fn(m, inp):
      return jnp.sum(m(inp))

    loss_val, (grads, grad_x) = nnx.value_and_grad(loss_fn, argnums=(0, 1))(mod, x)
    self.assertTrue(jnp.isfinite(loss_val))
    self.assertTrue(jnp.all(jnp.isfinite(grad_x)))
    self.assertTrue(jnp.any(grad_x != 0.0))

    for p in jax.tree_util.tree_leaves(grads):
      self.assertTrue(jnp.all(jnp.isfinite(jnp.asarray(p))))

    for g in (
        grads.gate.linear.kernel,
        grads.experts.gate_up_proj,
        grads.experts.down_proj,
        grads.shared_expert.gate_proj.kernel,
        grads.shared_expert.up_proj.kernel,
        grads.shared_expert.down_proj.kernel,
        grads.shared_expert_gate.kernel,
    ):
      self.assertTrue(jnp.any(jnp.asarray(getattr(g, "value", g)) != 0.0))


if __name__ == "__main__":
  absltest.main()

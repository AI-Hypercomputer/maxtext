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

"""Unit and parity tests for GLM-5.3-Flash KDA linear attention and primitives."""

from __future__ import annotations

import math
import os
from typing import Any
import unittest

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from maxtext.m3.configs.glm5_3_flash_config import Glm5Config
from maxtext.m3.configs.glm5_3_flash_config import Glm5Sharding
from maxtext.m3.configs.glm5_3_flash_config import validate_no_packing
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.glm5_3_flash.modeling_glm5_3_flash import Glm5KdaAttention
from maxtext.m3.models.glm5_3_flash.modeling_glm5_3_flash import Glm5RMSNormGated
from maxtext.m3.models.glm5_3_flash.modeling_glm5_3_flash import Glm5ShortConv1d
from maxtext.m3.models.glm5_3_flash.modeling_glm5_3_flash import l2norm
import numpy as np
import pytest

try:
  import torch
  from transformers.models.glm5_next import modeling_glm5_next
  from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
  from transformers.models.glm5_next.modeling_glm5_next import causal_conv1d_fn
  from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextLinearAttention
  from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextRMSNormGated
  from transformers.models.glm5_next.modeling_glm5_next import l2norm as hf_l2norm

  HAS_TORCH = True
except ImportError:
  if "TEST_SRCDIR" in os.environ:
    raise
  HAS_TORCH = False

pytestmark = [
    pytest.mark.scheduled_only,
    pytest.mark.skipif(not HAS_TORCH, reason="Torch or transformers not available"),
]

FLAGS = flags.FLAGS
flags.DEFINE_bool("run_tpu_mosaic_test", False, "Run TPU Mosaic parity tests.")


@unittest.skipUnless(HAS_TORCH, "Torch or transformers not available")
class Glm5KdaPrimitivesTest(parameterized.TestCase):
  """Tests forward/backward numerical parity and invariants of KDA primitives."""

  def setUp(self):
    super().setUp()
    self.enter_context(nnx.use_eager_sharding(True))
    self.enter_context(torch.enable_grad())
    torch.manual_seed(42)
    np.random.seed(42)

  def test_l2norm_parity_and_eps(self):
    """Verifies forward/backward parity of l2norm vs HF and zero-norm behavior."""
    shape = (2, 4, 8, 16)
    x_np = np.random.randn(*shape).astype(np.float32)
    cot_np = np.random.randn(*shape).astype(np.float32)
    eps = 1e-6

    # HF PyTorch reference
    x_pt = torch.tensor(x_np, requires_grad=True)
    out_pt = hf_l2norm(x_pt, dim=-1, eps=eps)
    cot_pt = torch.tensor(cot_np)
    (out_pt * cot_pt).sum().backward()

    # JAX implementation
    x_jax = jnp.array(x_np)
    cot_jax = jnp.array(cot_np)
    out_jax = l2norm(x_jax, axis=-1, eps=eps)

    def l2norm_loss(x):
      return jnp.sum(l2norm(x, axis=-1, eps=eps) * cot_jax)

    grad_x = jax.grad(l2norm_loss)(x_jax)

    # 1. Forward parity
    np.testing.assert_allclose(
        np.array(out_jax),
        out_pt.detach().numpy(),
        atol=1e-6,
        rtol=1e-6,
    )

    # 2. Backward parity
    np.testing.assert_allclose(
        np.array(grad_x),
        x_pt.grad.numpy(),
        atol=1e-6,
        rtol=1e-6,
    )

    # 3. FLA near-zero eps behavior: zero vector produces zero norm (not NaN or Inf)
    zeros_jax = jnp.zeros((2, 8))
    normed_zeros = l2norm(zeros_jax, axis=-1, eps=eps)
    self.assertFalse(np.isnan(np.array(normed_zeros)).any())
    self.assertFalse(np.isinf(np.array(normed_zeros)).any())
    np.testing.assert_allclose(np.array(normed_zeros), 0.0, atol=1e-7)
    grad_zeros = jax.grad(lambda z: jnp.sum(l2norm(z, axis=-1, eps=eps)))(zeros_jax)
    self.assertTrue(np.isfinite(np.array(grad_zeros)).all())
    # For f(z) = sum_i z_i / sqrt(sum_k z_k^2 + eps), df/dz_j at z=0 is eps**(-0.5).
    np.testing.assert_allclose(np.array(grad_zeros), 1.0 / np.sqrt(eps), rtol=1e-5)
    orig_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
      x64 = jnp.ones((2, 4), dtype=jnp.float64)
      self.assertEqual(l2norm(x64, axis=-1, eps=eps).dtype, jnp.float64)
    finally:
      jax.config.update("jax_enable_x64", orig_x64)

  def test_rmsnorm_gated_parity(self):
    """Verifies forward and backward parity of Glm5RMSNormGated vs HF."""
    hidden_size = 128
    eps = 1e-5

    pt_norm = Glm5NextTextRMSNormGated(hidden_size=hidden_size, eps=eps)
    jax_norm = Glm5RMSNormGated(hidden_size=hidden_size, eps=eps)

    # Initialize with non-trivial weights
    weight_np = np.random.randn(hidden_size).astype(np.float32)
    with torch.no_grad():
      pt_norm.weight.copy_(torch.tensor(weight_np))
    jax_norm.weight[...] = jnp.array(weight_np)

    b, s = 2, 8
    x_np = np.random.randn(b, s, hidden_size).astype(np.float32)
    g_np = np.random.randn(b, s, hidden_size).astype(np.float32)
    cot_np = np.random.randn(b, s, hidden_size).astype(np.float32)

    x_pt = torch.tensor(x_np, requires_grad=True)
    g_pt = torch.tensor(g_np, requires_grad=True)
    out_pt = pt_norm(x_pt, g_pt)
    cot_pt = torch.tensor(cot_np)
    (out_pt * cot_pt).sum().backward()

    x_jax = jnp.array(x_np)
    g_jax = jnp.array(g_np)
    cot_jax = jnp.array(cot_np)
    out_jax = jax_norm(x_jax, g_jax)

    # 1. Forward parity
    np.testing.assert_allclose(
        np.array(out_jax),
        out_pt.detach().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )

    # 2. Backward parity
    def norm_loss(model, x, g):
      return jnp.sum(model(x, g) * cot_jax)

    grad_fn = nnx.value_and_grad(norm_loss, argnums=(0, 1, 2))
    _, (grads, grad_x, grad_g) = grad_fn(jax_norm, x_jax, g_jax)

    np.testing.assert_allclose(
        np.array(grad_x),
        x_pt.grad.numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(grad_g),
        g_pt.grad.numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(grads.weight[...]),
        pt_norm.weight.grad.numpy(),
        atol=1e-5,
        rtol=1e-5,
    )

  @parameterized.named_parameters(
      ("seq1", 1),
      ("seq4", 4),
      ("seq16", 16),
  )
  def test_short_conv1d_parity_causality_and_sharding(self, seq_len: int):
    """Verifies forward/backward parity, strict causality, and sharding for Glm5ShortConv1d."""
    b = 2
    dim = 32
    kernel_size = 4

    jax_conv = Glm5ShortConv1d(dim=dim, kernel_size=kernel_size, rngs=nnx.Rngs(0))
    weight_np = np.random.randn(dim, 1, kernel_size).astype(np.float32)
    jax_conv.weight[...] = jnp.array(weight_np)

    w_pt = torch.tensor(weight_np.squeeze(1), requires_grad=True)

    x_np = np.random.randn(b, seq_len, dim).astype(np.float32)
    cot_np = np.random.randn(b, seq_len, dim).astype(np.float32)

    # HF causal_conv1d expects channel-first [b, dim, seq_len]
    x_pt = torch.tensor(x_np.transpose(0, 2, 1), requires_grad=True)
    out_pt = causal_conv1d_fn(
        x_pt,
        weight=w_pt,
        bias=None,
        activation="silu",
    ).transpose(1, 2)
    cot_pt = torch.tensor(cot_np)
    (out_pt * cot_pt).sum().backward()

    x_jax = jnp.array(x_np)
    cot_jax = jnp.array(cot_np)
    out_jax = jax_conv(x_jax)

    # 1. Forward parity
    np.testing.assert_allclose(
        np.array(out_jax),
        out_pt.detach().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )

    # 2. Backward parity
    def conv_loss(model, x):
      return jnp.sum(model(x) * cot_jax)

    grad_fn = nnx.value_and_grad(conv_loss, argnums=(0, 1))
    _, (grads, grad_x) = grad_fn(jax_conv, x_jax)

    np.testing.assert_allclose(
        np.array(grad_x),
        x_pt.grad.numpy().transpose(0, 2, 1),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.array(grads.weight[...]),
        w_pt.grad.numpy()[:, None, :],
        atol=1e-5,
        rtol=1e-5,
    )

    # 3. Strict causality: perturbing input at timestep >= t0 does not affect outputs at t < t0
    if seq_len >= 4:
      t0 = seq_len // 2
      x_perturbed = np.array(x_np, copy=True)
      noise = np.random.randn(b, seq_len - t0, dim).astype(np.float32)
      x_perturbed[:, t0:, :] += noise
      out_perturbed = jax_conv(jnp.array(x_perturbed))

      # Negative assertion: pre-t0 outputs are completely invariant to post-t0 perturbations
      np.testing.assert_allclose(
          np.array(out_perturbed)[:, :t0, :],
          np.array(out_jax)[:, :t0, :],
          atol=1e-6,
          rtol=1e-6,
      )
      # Positive liveness check: post-t0 outputs are perturbed
      diff = np.array(out_perturbed)[:, t0:, :] - np.array(out_jax)[:, t0:, :]
      max_diff = np.max(np.abs(diff))
      self.assertGreater(max_diff, 1e-4)

    # 4. Logical sharding check
    self.assertIsInstance(
        jax_conv.weight.sharding,
        jax.sharding.SingleDeviceSharding,
    )
    sharding = Glm5Sharding(use_tp=True)
    mesh = m3_sharding.create_mesh(devices=jax.devices()[:1])
    with jax.sharding.set_mesh(mesh):
      sharded_conv = Glm5ShortConv1d(
          dim=dim,
          kernel_size=kernel_size,
          sharding=sharding,
          rngs=nnx.Rngs(0),
      )
    expected_conv_spec = P("tensor", None, None)
    self.assertEqual(
        sharded_conv.weight.sharding,
        jax.sharding.NamedSharding(mesh, expected_conv_spec),
    )

  def test_short_conv1d_random_init_and_guards(self):
    """Verifies Glm5ShortConv1d rejects missing rngs and non-trivial init produces live outputs & grads."""
    dim = 32
    kernel_size = 4
    with self.assertRaisesRegex(ValueError, "Glm5ShortConv1d requires rngs"):
      Glm5ShortConv1d(dim=dim, kernel_size=kernel_size, rngs=None)

    conv = Glm5ShortConv1d(dim=dim, kernel_size=kernel_size, rngs=nnx.Rngs(0))
    w = np.array(conv.weight[...])
    bound = 1.0 / np.sqrt(kernel_size)
    self.assertTrue((w >= -bound).all() and (w <= bound).all())
    self.assertFalse(np.allclose(w, 0.0))

    b, s = 2, 8
    x_jax = jax.random.normal(jax.random.PRNGKey(42), (b, s, dim))

    def conv_loss(model, x):
      return jnp.sum(model(x))

    grad_fn = nnx.value_and_grad(conv_loss, argnums=(0, 1))
    out, (grads, grad_x) = grad_fn(conv, x_jax)

    self.assertFalse(np.allclose(np.array(out), 0.0))
    self.assertFalse(np.allclose(np.array(grad_x), 0.0))
    self.assertFalse(np.allclose(np.array(grads.weight[...]), 0.0))

  def test_short_conv1d_bfloat16_parity(self):
    """Verifies Glm5ShortConv1d on bfloat16 input matches PyTorch causal_conv1d_fn in fp32."""
    b, seq_len, dim, kernel_size = 2, 16, 64, 4
    weight_np = np.random.randn(dim, 1, kernel_size).astype(np.float32)
    x_np = np.random.randn(b, seq_len, dim).astype(np.float32)

    # HF PyTorch reference
    w_pt = torch.tensor(weight_np.squeeze(1), dtype=torch.float32)
    x_pt = torch.tensor(x_np.transpose(0, 2, 1), dtype=torch.bfloat16)
    out_pt = causal_conv1d_fn(
        x_pt,
        weight=w_pt,
        bias=None,
        activation="silu",
    ).transpose(1, 2)
    out_pt_np = out_pt.to(torch.float32).detach().numpy()

    # JAX implementation with bfloat16 input
    jax_conv = Glm5ShortConv1d(dim=dim, kernel_size=kernel_size, rngs=nnx.Rngs(0))
    jax_conv.weight[...] = jnp.array(weight_np)
    x_jax = jnp.array(x_np, dtype=jnp.bfloat16)
    out_jax = jax_conv(x_jax)
    self.assertEqual(out_jax.dtype, jnp.bfloat16)

    out_jax_np = np.array(out_jax, dtype=np.float32)
    rel_fro = np.linalg.norm(out_jax_np - out_pt_np) / np.linalg.norm(out_pt_np)
    self.assertLess(rel_fro, 2.5e-3)


@unittest.skipUnless(HAS_TORCH, "Torch or transformers not available")
class Glm5KdaAttentionTest(parameterized.TestCase):
  """Tests forward/backward numerical parity and invariants of Glm5KdaAttention."""

  def setUp(self):
    super().setUp()
    self.enter_context(nnx.use_eager_sharding(True))
    self.enter_context(torch.enable_grad())
    torch.manual_seed(42)
    np.random.seed(42)

  def _assert_allclose(self, a, b, atol=1e-5, rtol=1e-5):
    a_np, b_np = np.array(a), np.array(b)
    self.assertTrue(np.all(np.isfinite(a_np)), "Non-finite values in actual")
    self.assertTrue(np.all(np.isfinite(b_np)), "Non-finite values in reference")
    np.testing.assert_allclose(a_np, b_np, atol=atol, rtol=rtol)

  def _assert_grad_close(self, name: str, a, b):
    """Asserts parameter gradient parity against PyTorch reference."""
    # HF Glm5NextTextForgetGate and Tokamax _activate_gate compute internal gate activations in float32.
    a_np, b_np = np.array(a), np.array(b)
    self.assertTrue(np.all(np.isfinite(a_np)), f"Non-finite JAX grad: {name}")
    self.assertTrue(np.all(np.isfinite(b_np)), f"Non-finite PT grad: {name}")
    atol = 1e-5 * max(1.0, float(np.max(np.abs(b_np))))
    np.testing.assert_allclose(a_np, b_np, atol=atol, rtol=1e-5, err_msg=name)
    denom = max(float(np.linalg.norm(b_np)), 1e-12)
    rel_frob = float(np.linalg.norm(a_np - b_np)) / denom
    self.assertLessEqual(rel_frob, 1e-6, f"{name} rel_frob={rel_frob}")

  def _mini_cfg(
      self,
      kda_implementation="xla",
      dtype="float32",
      weight_dtype=None,
      **kwargs,
  ) -> Glm5Config:
    w_dtype = weight_dtype if weight_dtype is not None else dtype
    return Glm5Config.mini_config(
        hidden_size=64,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        kda_implementation=kda_implementation,
        dtype=dtype,
        weight_dtype=w_dtype,
        **kwargs,
    )

  def _hf_layer(self) -> Glm5NextTextLinearAttention:
    """Constructs a mini HF Glm5NextTextLinearAttention reference layer."""
    cfg = Glm5NextTextConfig(
        hidden_size=64,
        linear_num_heads=4,
        linear_head_dim=16,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        rms_norm_eps=1e-5,
        layer_types=["linear_attention"],
        num_hidden_layers=1,
    )
    layer = Glm5NextTextLinearAttention(cfg, layer_idx=0).to(torch.float32)
    # Glm5NextTextForgetGate allocates A_log and dt_bias with torch.empty()
    # and relies on PreTrainedModel._init_weights; initialize them explicitly.
    with torch.no_grad():
      torch.nn.init.normal_(layer.forget_gate.A_log, mean=0.0, std=0.1)
      torch.nn.init.uniform_(layer.forget_gate.dt_bias, a=-2.0, b=-0.5)
    return layer

  @parameterized.named_parameters(
      ("seq8_xla", 8),
      ("seq64_xla", 64),
  )
  def test_kda_attention_parity(self, seq_len: int):
    """Verifies forward/backward gradient parity against HF LinearAttention."""
    hf_layer = self._hf_layer()
    cfg = self._mini_cfg(kda_implementation="xla")
    jax_layer = Glm5KdaAttention(cfg, rngs=nnx.Rngs(0))

    fg = hf_layer.forget_gate
    projs = ("q", "k", "v", "b", "o", "f_a", "f_b", "g_a", "g_b")
    hf_p = {
        "q": hf_layer.q_proj,
        "k": hf_layer.k_proj,
        "v": hf_layer.v_proj,
        "b": hf_layer.b_proj,
        "o": hf_layer.o_proj,
        "f_a": fg.f_a_proj,
        "f_b": fg.f_b_proj,
        "g_a": hf_layer.g_a_proj,
        "g_b": hf_layer.g_b_proj,
    }

    sls = (("q", slice(64)), ("k", slice(64, 128)), ("v", slice(128, None)))
    with torch.no_grad():
      for p in projs:
        w_pt = hf_p[p].weight.detach().numpy().T
        getattr(jax_layer, f"{p}_proj").kernel[...] = jnp.array(w_pt)

      cw = hf_layer.conv1d.weight.detach().numpy()
      for c, s in sls:
        getattr(jax_layer, f"{c}_conv1d").weight[...] = jnp.array(cw[s])
      jax_layer.A_log[...] = jnp.array(fg.A_log.detach().numpy())
      jax_layer.dt_bias[...] = jnp.array(fg.dt_bias.detach().numpy())
      o_w = hf_layer.o_norm.weight.detach().numpy()
      jax_layer.o_norm.weight[...] = jnp.array(o_w)

    x_np = np.random.randn(2, seq_len, 64).astype(np.float32)
    with torch.no_grad():
      out_pt = hf_layer(torch.from_numpy(x_np))
    out_jax = jax_layer(jnp.array(x_np))
    self._assert_allclose(out_jax, out_pt.numpy())

    cot_np = np.random.randn(2, seq_len, 64).astype(np.float32)
    # Use HF recurrent_kimi_delta_attention for backward reference
    # (chunk_kimi_delta_attention overflows exp() before masking at full 64-token chunks).
    m = modeling_glm5_next
    orig_fn = m.chunk_kimi_delta_attention
    m.chunk_kimi_delta_attention = m.recurrent_kimi_delta_attention
    try:
      x_pt = torch.from_numpy(x_np).requires_grad_(True)
      (hf_layer(x_pt) * torch.from_numpy(cot_np)).sum().backward()
    finally:
      m.chunk_kimi_delta_attention = orig_fn
    cot_jax = jnp.array(cot_np)

    def loss(m_l, x):
      return jnp.sum(m_l(x) * cot_jax)

    vg = nnx.value_and_grad(loss, argnums=(0, 1))
    _, (grads, grad_x) = vg(jax_layer, jnp.array(x_np))
    self._assert_allclose(grad_x, x_pt.grad.numpy())

    cg = hf_layer.conv1d.weight.grad.numpy()
    for p in projs:
      w_grad = hf_p[p].weight.grad.numpy().T
      ja = getattr(grads, f"{p}_proj").kernel[...]
      self._assert_grad_close(f"{p}_proj", ja, w_grad)
    for c, s in sls:
      ca = getattr(grads, f"{c}_conv1d").weight[...]
      self._assert_grad_close(f"{c}_conv1d", ca, cg[s])
    self._assert_grad_close("A_log", grads.A_log[...], fg.A_log.grad.numpy())
    dtb_g = fg.dt_bias.grad.numpy()
    self._assert_grad_close("dt_bias", grads.dt_bias[...], dtb_g)
    o_g = hf_layer.o_norm.weight.grad.numpy()
    self._assert_grad_close("o_norm", grads.o_norm.weight[...], o_g)

  def test_kda_attention_initialization(self):
    """Verifies dt_bias initialization range and live non-zero outputs and gradients."""
    cfg = self._mini_cfg(kda_implementation="xla")
    layer = Glm5KdaAttention(cfg, rngs=nnx.Rngs(0))
    dt_b = np.array(layer.dt_bias[...])
    self.assertTrue((dt_b >= -7.0).all() and (dt_b <= -2.0).all())

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64))

    def loss(m, inp):
      return jnp.sum(m(inp))

    vg = nnx.value_and_grad(loss, argnums=(0, 1))
    out, (grads, grad_x) = vg(layer, x)
    self.assertGreater(float(jnp.max(jnp.abs(out))), 1e-4)
    self.assertGreater(float(jnp.max(jnp.abs(grad_x))), 1e-4)
    for p in ("q", "k", "v", "b", "o", "f_a", "f_b", "g_a", "g_b"):
      self.assertGreater(
          float(jnp.max(jnp.abs(getattr(grads, f"{p}_proj").kernel[...]))),
          1e-4,
      )
    for c in ("q", "k", "v"):
      self.assertGreater(
          float(jnp.max(jnp.abs(getattr(grads, f"{c}_conv1d").weight[...]))),
          1e-4,
      )
    self.assertGreater(float(jnp.max(jnp.abs(grads.dt_bias[...]))), 1e-4)
    self.assertGreater(float(jnp.max(jnp.abs(grads.o_norm.weight[...]))), 1e-4)

  def test_kda_attention_xla_chunked_vs_xla(self):
    """Verifies parity between xla_chunked and xla backends at seq_len=192 (multi-chunk)."""
    # Parity contract tolerances per tokamax/_src/ops/kda/xla_chunked_test.py:102,222
    cfg_xla = self._mini_cfg(kda_implementation="xla")
    cfg_chunked = self._mini_cfg(kda_implementation="xla_chunked")
    layer_xla = Glm5KdaAttention(cfg_xla, rngs=nnx.Rngs(0))
    layer_chunked = Glm5KdaAttention(cfg_chunked, rngs=nnx.Rngs(0))

    rng = np.random.RandomState(42)
    for cv in (layer_xla.q_conv1d, layer_xla.k_conv1d, layer_xla.v_conv1d):
      cv.weight[...] = jnp.array(rng.randn(64, 1, 4).astype(np.float32))
    layer_xla.A_log[...] = jnp.array(rng.randn(4).astype(np.float32))
    layer_xla.dt_bias[...] = jnp.array(rng.randn(64).astype(np.float32))
    nnx.update(layer_chunked, nnx.state(layer_xla))

    x = jnp.array(rng.randn(2, 192, 64).astype(np.float32))
    cot = jnp.array(rng.randn(2, 192, 64).astype(np.float32))
    self._assert_allclose(layer_chunked(x), layer_xla(x), atol=2e-4, rtol=2e-4)

    def loss(m_l, inp):
      return jnp.sum(m_l(inp) * cot)

    vg = nnx.value_and_grad(loss, argnums=(0, 1))
    _, (g_xla, gx_xla) = vg(layer_xla, x)
    _, (g_chk, gx_chk) = vg(layer_chunked, x)

    def _tol(a, b):
      self._assert_allclose(a, b, atol=1e-3, rtol=1e-3)

    _tol(gx_chk, gx_xla)
    for p in ("q", "k", "v", "b", "o", "f_a", "f_b", "g_a", "g_b"):
      _tol(
          getattr(g_chk, f"{p}_proj").kernel[...],
          getattr(g_xla, f"{p}_proj").kernel[...],
      )
    for c in ("q", "k", "v"):
      _tol(
          getattr(g_chk, f"{c}_conv1d").weight[...],
          getattr(g_xla, f"{c}_conv1d").weight[...],
      )
    _tol(g_chk.A_log[...], g_xla.A_log[...])
    _tol(g_chk.dt_bias[...], g_xla.dt_bias[...])
    _tol(g_chk.o_norm.weight[...], g_xla.o_norm.weight[...])

    cfg_unbounded = self._mini_cfg(kda_implementation="xla", linear_lower_bound=None)
    layer_unbounded = Glm5KdaAttention(cfg_unbounded, rngs=nnx.Rngs(0))
    a_log_np = np.asarray(layer_unbounded.A_log[...])
    self.assertTrue(np.all((a_log_np >= 0.0) & (a_log_np <= math.log(16.0))))
    self.assertTrue(np.any(a_log_np > 0.0))

  def test_right_padding_mask_invariance(self):
    """Verifies trailing right-padded tokens do not affect valid tokens."""
    jax_layer = Glm5KdaAttention(self._mini_cfg(), rngs=nnx.Rngs(0))
    b, s_val, s_pad = 2, 6, 4
    s_tot = s_val + s_pad

    x_val = np.random.randn(b, s_val, 64).astype(np.float32)
    x_pad = np.zeros((b, s_tot, 64), dtype=np.float32)
    x_pad[:, :s_val, :] = x_val
    x_pad[:, s_val:, :] = np.random.randn(b, s_pad, 64).astype(np.float32)

    seg_ids = jnp.array(
        [[1] * s_val + [0] * s_pad for _ in range(b)],
        dtype=jnp.int32,
    )
    out_val = jax_layer(jnp.array(x_val))
    out_pad = jax_layer(jnp.array(x_pad), decoder_segment_ids=seg_ids)

    self._assert_allclose(out_pad[:, :s_val, :], out_val)
    self._assert_allclose(out_pad[:, s_val:, :], 0.0, atol=1e-7)

    cot_val = jnp.array(np.random.randn(b, s_val, 64).astype(np.float32))
    cot_pad_tail = jnp.array(np.random.randn(b, s_tot - s_val, 64).astype(np.float32))
    cot_pad = jnp.concatenate([cot_val, cot_pad_tail], axis=1)

    def v_loss(m, x):
      return jnp.sum(m(x) * cot_val)

    def p_loss(m, x, s):
      return jnp.sum(m(x, decoder_segment_ids=s) * cot_pad)

    v_vg = nnx.value_and_grad(v_loss, argnums=(0, 1))
    p_vg = nnx.value_and_grad(p_loss, argnums=(0, 1))
    _, (g_val, gx_val) = v_vg(jax_layer, jnp.array(x_val))
    _, (g_pad, gx_pad) = p_vg(jax_layer, jnp.array(x_pad), seg_ids)

    self._assert_allclose(gx_pad[:, :s_val, :], gx_val)
    self._assert_allclose(gx_pad[:, s_val:, :], 0.0, atol=1e-7)
    g_val_flat = dict(nnx.state(g_val, nnx.Param).flat_state())
    g_pad_flat = dict(nnx.state(g_pad, nnx.Param).flat_state())
    self.assertLen(g_val_flat, 15)
    self.assertEqual(set(g_val_flat.keys()), set(g_pad_flat.keys()))
    for path, val_var in g_val_flat.items():
      self._assert_allclose(g_pad_flat[path][...], val_var[...])

    # Positive assertions: test is provably non-vacuous
    self.assertGreater(float(jnp.max(jnp.abs(out_val))), 1e-4)
    self.assertGreater(float(jnp.max(jnp.abs(gx_val))), 1e-4)
    self.assertGreater(float(jnp.max(jnp.abs(g_val.q_conv1d.weight[...]))), 1e-4)

  def test_kda_sharding(self):
    """Verifies logical-to-mesh sharding placement on KDA parameters and shard_map execution."""
    sharding = Glm5Sharding(use_tp=True)
    mesh = m3_sharding.create_mesh(devices=jax.devices()[:1])
    with jax.sharding.set_mesh(mesh):
      layer = Glm5KdaAttention(
          self._mini_cfg(weight_dtype="bfloat16"),
          sharding=sharding,
          rngs=nnx.Rngs(0),
      )

      # Verify that jax.jit with shard_map executes cleanly inside active mesh
      x_test = jnp.zeros((2, 8, 64), dtype=jnp.float32)
      sharded_fn = jax.jit(lambda m, inp: m(inp))
      out_sharded = sharded_fn(layer, x_test)
      self.assertEqual(out_sharded.shape, (2, 8, 64))

    self.assertEqual(layer.A_log[...].dtype, jnp.float32)
    self.assertEqual(layer.dt_bias[...].dtype, jnp.float32)
    self.assertEqual(layer.q_conv1d.weight[...].dtype, jnp.float32)

    def _sharded(t, s):
      self.assertEqual(t.sharding, jax.sharding.NamedSharding(mesh, s))

    for p in (layer.q_proj, layer.k_proj, layer.v_proj, layer.b_proj):
      _sharded(p.kernel, P("fsdp", "tensor"))
    for cv in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d):
      _sharded(cv.weight, P("tensor", None, None))
    for p in (layer.f_a_proj, layer.g_a_proj):
      _sharded(p.kernel, P("fsdp", None))
    for p in (layer.f_b_proj, layer.g_b_proj):
      _sharded(p.kernel, P(None, "tensor"))
    _sharded(layer.A_log, P("tensor"))
    _sharded(layer.dt_bias, P("tensor"))
    _sharded(layer.o_norm.weight, P(None))
    _sharded(layer.o_proj.kernel, P("tensor", "fsdp"))

  def test_kda_multi_device_sharding_parity(self):
    """Verifies numerical parity between sharded and unsharded Glm5KdaAttention forward and backward."""
    if len(jax.devices()) < 4:
      if "TEST_SRCDIR" in os.environ:
        self.fail("Expected >= 4 JAX devices under Blaze XLA_FLAGS, got" f" {len(jax.devices())}")
      self.skipTest(f"Requires >= 4 JAX devices, got {len(jax.devices())}")
    mesh = m3_sharding.create_mesh(
        ici={"data": 2, "fsdp": 1, "tensor": 2, "expert": 1},
        devices=jax.devices()[:4],
    )
    sharding = Glm5Sharding.from_mesh(mesh)
    cfg = self._mini_cfg()

    layer_ref = Glm5KdaAttention(cfg, rngs=nnx.Rngs(0))
    rng = np.random.RandomState(42)
    for cv in (layer_ref.q_conv1d, layer_ref.k_conv1d, layer_ref.v_conv1d):
      cv.weight[...] = jnp.array(rng.randn(*cv.weight[...].shape).astype(np.float32))
    layer_ref.A_log[...] = jnp.array(rng.randn(*layer_ref.A_log[...].shape).astype(np.float32))
    layer_ref.dt_bias[...] = jnp.array(rng.randn(*layer_ref.dt_bias[...].shape).astype(np.float32))

    with jax.sharding.set_mesh(mesh):
      layer_sharded = Glm5KdaAttention(cfg, sharding=sharding, rngs=nnx.Rngs(0))
      ref_dict = dict(nnx.state(layer_ref, nnx.Param).flat_state())
      sharded_state = nnx.state(layer_sharded, nnx.Param)
      sharded_dict = dict(sharded_state.flat_state())
      for path, ref_var in ref_dict.items():
        target_var = sharded_dict[path]
        target_shd = target_var.sharding if target_var.sharding is not None else jax.sharding.NamedSharding(mesh, P())
        target_var[...] = jax.device_put(ref_var[...], target_shd)
      nnx.update(layer_sharded, sharded_state)

    x_np = rng.randn(4, 8, 64).astype(np.float32)
    cot_np = rng.randn(4, 8, 64).astype(np.float32)
    x = jnp.array(x_np)
    cot = jnp.array(cot_np)

    def loss(m, inp, c):
      return jnp.sum(m(inp) * c)

    vg = nnx.value_and_grad(loss, argnums=(0, 1))
    _, (g_ref, gx_ref) = vg(layer_ref, x, cot)
    out_ref = layer_ref(x)

    x_sharded = jax.device_put(x, jax.sharding.NamedSharding(mesh, P(("data", "fsdp"), None, None)))
    cot_sharded = jax.device_put(cot, jax.sharding.NamedSharding(mesh, P(("data", "fsdp"), None, None)))

    with jax.sharding.set_mesh(mesh):
      vg_sharded = jax.jit(vg)
      _, (g_sharded, gx_sharded) = vg_sharded(layer_sharded, x_sharded, cot_sharded)
      out_sharded = jax.jit(lambda m, inp: m(inp))(layer_sharded, x_sharded)

    def _assert_frob_rel(actual, ref, name, tol=1e-5):
      a_np = np.array(actual, dtype=np.float64)
      r_np = np.array(ref, dtype=np.float64)
      self.assertTrue(np.all(np.isfinite(a_np)), f"Non-finite actual: {name}")
      self.assertTrue(np.all(np.isfinite(r_np)), f"Non-finite ref: {name}")
      denom = max(float(np.linalg.norm(r_np)), 1e-12)
      frob_rel = float(np.linalg.norm(a_np - r_np)) / denom
      self.assertLessEqual(frob_rel, tol, f"{name} frob_rel={frob_rel} > {tol}")

    _assert_frob_rel(out_sharded, out_ref, "output")
    _assert_frob_rel(gx_sharded, gx_ref, "gx")
    g_ref_flat = dict(nnx.state(g_ref, nnx.Param).flat_state())
    g_sharded_flat = dict(nnx.state(g_sharded, nnx.Param).flat_state())
    self.assertLen(g_ref_flat, 15)
    self.assertEqual(set(g_ref_flat.keys()), set(g_sharded_flat.keys()))
    for path, ref_var in g_ref_flat.items():
      _assert_frob_rel(g_sharded_flat[path][...], ref_var[...], ".".join(map(str, path)))

    # Positive check: verify gradients are non-trivial
    self.assertGreater(float(np.linalg.norm(np.array(g_ref.A_log[...]))), 1e-4)
    self.assertGreater(float(np.linalg.norm(np.array(g_ref.dt_bias[...]))), 1e-4)
    self.assertGreater(float(np.linalg.norm(np.array(gx_ref))), 1e-4)

  def test_loud_guards(self):
    """Verifies loud ValueError checks for packing and backend selection."""
    with self.assertRaisesRegex(ValueError, "FU-D5"):
      validate_no_packing(np.array([[1, 2]]))

    err = "kda_implementation must be one of"
    for invalid_val in (["xla"], 123, "unsupported_backend"):
      bad_cfg = self._mini_cfg()
      object.__setattr__(bad_cfg, "kda_implementation", invalid_val)
      with self.assertRaisesRegex(ValueError, err):
        Glm5KdaAttention(bad_cfg, rngs=nnx.Rngs(0))

    with self.assertRaisesRegex(ValueError, "linear_lower_bound cannot be None"):
      bad_cfg = self._mini_cfg(kda_implementation="mosaic")
      object.__setattr__(bad_cfg, "linear_lower_bound", None)
      Glm5KdaAttention(bad_cfg, rngs=nnx.Rngs(0))

  def test_mosaic_backend(self):
    """Tests TPU Mosaic kernel execution under Tokamax fp32 contract or asserts loud failure on CPU."""
    cfg_mosaic = self._mini_cfg(kda_implementation="mosaic", dtype="float32", weight_dtype="float32")
    cfg_xla = self._mini_cfg(kda_implementation="xla", dtype="float32", weight_dtype="float32")
    test_rngs = nnx.Rngs(0)

    if FLAGS.run_tpu_mosaic_test and jax.devices()[0].platform == "tpu":

      def _assert_mosaic_fp32_parity(
          name: str,
          actual: Any,
          expected: Any,
          *,
          atol: float,
          rtol: float,
          max_rms_ratio: float,
      ) -> None:
        self.assertTrue(bool(jnp.all(jnp.isfinite(actual))), msg=f"{name} non-finite")
        actual_f64 = np.asarray(actual, dtype=np.float64)
        expected_f64 = np.asarray(expected, dtype=np.float64)
        diff = np.abs(expected_f64 - actual_f64)
        max_abs = float(np.max(diff))
        max_rel = float(np.max(diff / (np.abs(expected_f64) + 1e-12)))
        rms_error = float(np.sqrt(np.mean(np.square(diff))))
        ref_rms = float(np.sqrt(np.mean(np.square(expected_f64))))
        rms_ratio = rms_error / (ref_rms + 1e-8)
        print(
            f"[KDA_MOSAIC_PARITY] {name}: max_abs={max_abs:.6e}" f" max_rel={max_rel:.6e} rms_ratio={rms_ratio:.6e}",
            flush=True,
        )
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            atol=atol,
            rtol=rtol,
            err_msg=f"{name} allclose mismatch",
        )
        self.assertLessEqual(
            rms_ratio,
            max_rms_ratio,
            msg=f"{name} rms_ratio={rms_ratio:.6e} > {max_rms_ratio:.6e}",
        )

      # Tokamax pallas_mosaic_tpu fp32 contract
      # (tokamax/_src/ops/kda/pallas_mosaic_tpu_kernel_test.py:264-270,300-304):
      # forward atol=rtol=5e-3, rms_error_ratio <= 5e-3;
      # backward atol=rtol=1e-2, rms_error_ratio <= 8e-3 (gate grads <= 2e-2).
      with jax.default_matmul_precision("highest"):
        layer_mosaic = Glm5KdaAttention(cfg_mosaic, rngs=nnx.Rngs(0))
        layer_xla = Glm5KdaAttention(cfg_xla, rngs=nnx.Rngs(0))
        rng = np.random.RandomState(0)
        for cv in (layer_xla.q_conv1d, layer_xla.k_conv1d, layer_xla.v_conv1d):
          cv.weight[...] = jnp.array(rng.randn(64, 1, 4))
        layer_xla.A_log[...] = jnp.array(rng.randn(4).astype(np.float32))
        layer_xla.dt_bias[...] = jnp.array(rng.randn(64).astype(np.float32))
        nnx.update(layer_mosaic, nnx.state(layer_xla))
        x = jax.random.normal(jax.random.key(0), (2, 192, 64), dtype=jnp.float32)
        cot = jax.random.normal(jax.random.key(1), (2, 192, 64), dtype=jnp.float32)
        y_m = layer_mosaic(x)
        y_x = layer_xla(x)
        _assert_mosaic_fp32_parity(
            "forward_y",
            y_m,
            y_x,
            atol=5e-3,
            rtol=5e-3,
            max_rms_ratio=5e-3,
        )

        def loss(m_l, inp):
          return jnp.sum(m_l(inp) * cot)

        vg = nnx.value_and_grad(loss, argnums=(0, 1))
        _, (g_m, gx_m) = vg(layer_mosaic, x)
        _, (g_x, gx_x) = vg(layer_xla, x)
        _assert_mosaic_fp32_parity(
            "grad_x",
            gx_m,
            gx_x,
            atol=1e-2,
            rtol=1e-2,
            max_rms_ratio=8e-3,
        )

        flat_g_m = dict(nnx.state(g_m, nnx.Param).flat_state())
        flat_g_x = dict(nnx.state(g_x, nnx.Param).flat_state())
        self.assertNotEmpty(flat_g_m)
        self.assertEqual(set(flat_g_m.keys()), set(flat_g_x.keys()))
        gate_roots = ("f_a_proj", "f_b_proj", "A_log", "dt_bias", "b_proj")
        for key in sorted(flat_g_m.keys()):
          leaf_name = ".".join(str(k) for k in key)
          is_gate = str(key[0]) in gate_roots
          leaf_tol = 2e-2 if is_gate else 1e-2
          leaf_rms = 2e-2 if is_gate else 8e-3
          _assert_mosaic_fp32_parity(
              f"grad_{leaf_name}",
              flat_g_m[key][...],
              flat_g_x[key][...],
              atol=leaf_tol,
              rtol=leaf_tol,
              max_rms_ratio=leaf_rms,
          )
    else:
      layer_m = Glm5KdaAttention(cfg_mosaic, rngs=test_rngs)
      with self.assertRaises((NotImplementedError, ValueError)):
        layer_m(jnp.zeros((2, 192, 64), dtype=jnp.float32))


if __name__ == "__main__":
  absltest.main()

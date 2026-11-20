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

"""Unit tests for the Qwen3.8-Flash-Next m3 model against the PyTorch reference.

Sections:
  * Hyper-connection and normalization layers
  * Gated DeltaNet layer
  * Gated DeltaNet recurrence numerics
  * Qwen Sparse Attention and indexer
  * Sparse MoE block
  * SplitMix64 n-gram hashing and PLE
  * Decoder backbone, scan parity, and m3 registration
  * MTP draft block and integrated loss
"""

import functools
import importlib
import importlib.util
import os
import re
import types
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.configs import pyconfig  # pylint: disable=no-name-in-module
from maxtext.m3 import models as m3_models  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next  # pylint: disable=no-name-in-module
from maxtext.utils import globals as maxtext_globals  # pylint: disable=no-name-in-module
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

# Tolerances for fp32 JAX-vs-PyTorch parity of the chunked gated delta rule. The two
# implementations accumulate the recurrence in different orders, so isolated elements
# can drift slightly past 1e-5 depending on the CPU backend (observed: 1 / 32768
# elements off by 1.17e-5). Real kernel bugs are orders of magnitude larger.
_GDN_PT_PARITY_RTOL = 1e-4
_GDN_PT_PARITY_ATOL = 5e-5


def _assert_allclose_to_scale(actual: Any, desired: Any, rtol: float = 1e-4, atol_scale: float = 1e-5) -> None:
  """Like assert_allclose, with atol relative to max|desired| (gradients here reach O(100s))."""
  desired = np.asarray(desired)
  np.testing.assert_allclose(np.asarray(actual), desired, rtol=rtol, atol=atol_scale * float(np.abs(desired).max()))


# =================================================================================================
# Hyper-connection and normalization layers
# =================================================================================================


@pytest.mark.cpu_only
class Qwen3_8HyperConnectionTest(absltest.TestCase):
  """Validates Qwen3.8 hyper-connection primitives against PyTorch reference."""

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_pytorch_rms_norm_parity(self):
    """Verifies grouped and ungrouped zero-centered RMSNorm match PyTorch."""
    assert torch is not None and modeling_qwen4_exp is not None
    rng = np.random.default_rng(42)
    dim, group_size = 64, 4
    x = rng.standard_normal((2, 4, dim)).astype(np.float32)
    weight = rng.standard_normal((dim,)).astype(np.float32) * 0.5

    pt_norm = modeling_qwen4_exp.Qwen4ExpTextRMSNorm(dim, group_size=group_size, eps=1e-6)
    with torch.no_grad():
      pt_norm.weight.copy_(torch.from_numpy(weight))
    pt_out = pt_norm(torch.from_numpy(x)).detach().cpu().numpy()

    jax_norm = modeling_qwen3_8_flash_next.Qwen3_8RMSNorm(dim, group_size=group_size, eps=1e-6, rngs=nnx.Rngs(0))
    jax_norm.scale[...] = jnp.array(weight)
    jax_out = np.array(jax_norm(jnp.array(x)))
    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-6, atol=1e-6)

    dim_ungrouped = 16
    x_ungrouped = rng.standard_normal((2, 3, dim_ungrouped)).astype(np.float32)
    w_ungrouped = rng.standard_normal((dim_ungrouped,)).astype(np.float32) * 0.5
    pt_zero = modeling_qwen4_exp.Qwen4ExpTextRMSNorm(dim_ungrouped, group_size=None, eps=1e-6)
    with torch.no_grad():
      pt_zero.weight.copy_(torch.from_numpy(w_ungrouped))
    pt_zero_out = pt_zero(torch.from_numpy(x_ungrouped)).detach().cpu().numpy()

    jax_zero = modeling_qwen3_8_flash_next.Qwen3_8RMSNorm(dim_ungrouped, eps=1e-6, rngs=nnx.Rngs(0))
    jax_zero.scale[...] = jnp.array(w_ungrouped)
    jax_zero_out = np.array(jax_zero(jnp.array(x_ungrouped)))
    np.testing.assert_allclose(jax_zero_out, pt_zero_out, rtol=1e-6, atol=1e-6)

    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8RMSNorm(dim=10, group_size=3)

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_pytorch_rms_norm_gated_parity(self):
    """Verifies gated RMSNorm with sigmoid and silu activations match PyTorch."""
    assert torch is not None and modeling_qwen4_exp is not None
    rng = np.random.default_rng(42)
    dim = 16
    x = rng.standard_normal((2, 3, dim)).astype(np.float32)
    gate = rng.standard_normal((2, 3, dim)).astype(np.float32)
    weight = rng.standard_normal((dim,)).astype(np.float32)

    for act in ("sigmoid", "silu"):
      pt_gated = modeling_qwen4_exp.Qwen4ExpTextRMSNormGated(dim, eps=1e-6, activation=act)
      with torch.no_grad():
        pt_gated.weight.copy_(torch.from_numpy(weight))
      pt_out = pt_gated(torch.from_numpy(x), gate=torch.from_numpy(gate)).detach().cpu().numpy()

      jax_gated = modeling_qwen3_8_flash_next.Qwen3_8RMSNormGated(dim, eps=1e-6, activation=act, rngs=nnx.Rngs(0))
      jax_gated.scale[...] = jnp.array(weight)
      jax_out = np.array(jax_gated(jnp.array(x), gate=jnp.array(gate)))
      np.testing.assert_allclose(jax_out, pt_out, rtol=1e-6, atol=1e-6)

    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8RMSNormGated(dim=16, activation="unsupported_gate")

    with self.assertRaises(TypeError):
      getattr(jax_gated, "__call__")(jnp.array(x))  # pylint: disable=no-value-for-parameter

  def test_rms_norm_gated_requires_gate(self):
    """Verifies gate argument is required for Qwen3_8RMSNormGated."""
    dim = 16
    x = jnp.ones((2, 3, dim), dtype=jnp.float32)
    norm = modeling_qwen3_8_flash_next.Qwen3_8RMSNormGated(dim, rngs=nnx.Rngs(0))
    with self.assertRaises(TypeError):
      getattr(norm, "__call__")(x)  # pylint: disable=no-value-for-parameter

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_pytorch_qwen4_exp_weight_copy_parity(self):
    """Verifies GatedResidual matches PyTorch reference with weight copy."""
    assert torch is not None and modeling_qwen4_exp is not None and configuration_qwen4_exp is not None
    hf_cfg = configuration_qwen4_exp.Qwen4ExpTextConfig(
        hidden_size=16,
        hc_count=4,
        hc_lowrank=8,
        rms_norm_eps=1e-6,
        output_gate_type="sigmoid",
    )
    rng = np.random.default_rng(42)
    emb_dim, hc_count, hc_lowrank = 16, 4, 8
    hc_dim = hc_count * emb_dim
    x = rng.standard_normal((2, 3, hc_dim)).astype(np.float32)

    # 1. use_combine=False (final mixer / mix_input)
    pt_mixer = modeling_qwen4_exp.Qwen4ExpTextGatedResidual(hf_cfg, use_combine=False)
    jax_mixer = modeling_qwen3_8_flash_next.Qwen3_8GatedResidual(
        emb_dim=emb_dim,
        hc_count=hc_count,
        hc_lowrank=hc_lowrank,
        use_combine=False,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    jax_mixer.hc_norm.scale[...] = jnp.array(pt_mixer.hc_norm.weight.detach().cpu().numpy())
    jax_mixer.input_mix_weight_down.kernel[...] = jnp.array(
        pt_mixer.input_mix_weight_down.weight.detach().cpu().numpy().T
    )
    jax_mixer.input_mix_weight_up.kernel[...] = jnp.array(pt_mixer.input_mix_weight_up.weight.detach().cpu().numpy().T)
    self.assertIsNone(jax_mixer.block_inject_weight)

    pt_mixer_out = pt_mixer(torch.from_numpy(x)).detach().cpu().numpy()
    jax_mixer_out = np.array(jax_mixer.mix_input(jnp.array(x)))
    np.testing.assert_allclose(jax_mixer_out, pt_mixer_out, rtol=1e-6, atol=1e-6)

    # 2. use_combine=True (before_block, combine, apply_block)
    pt_gr = modeling_qwen4_exp.Qwen4ExpTextGatedResidual(hf_cfg, use_combine=True)
    jax_gr = modeling_qwen3_8_flash_next.Qwen3_8GatedResidual(
        emb_dim=emb_dim,
        hc_count=hc_count,
        hc_lowrank=hc_lowrank,
        use_combine=True,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    jax_gr.hc_norm.scale[...] = jnp.array(pt_gr.hc_norm.weight.detach().cpu().numpy())
    jax_gr.input_mix_weight_down.kernel[...] = jnp.array(pt_gr.input_mix_weight_down.weight.detach().cpu().numpy().T)
    jax_gr.input_mix_weight_up.kernel[...] = jnp.array(pt_gr.input_mix_weight_up.weight.detach().cpu().numpy().T)
    assert jax_gr.block_inject_weight is not None
    assert pt_gr.block_inject_weight is not None
    jax_gr.block_inject_weight.kernel[...] = jnp.array(pt_gr.block_inject_weight.weight.detach().cpu().numpy().T)

    pt_mixed, pt_hyper, pt_inject = pt_gr(torch.from_numpy(x))
    jax_mixed, jax_inject = jax_gr.before_block(jnp.array(x))

    np.testing.assert_allclose(np.array(jax_mixed), pt_mixed.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.array(jax_inject), pt_inject.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)

    # Combine parity with HF decoder-layer algebra
    block_out = rng.standard_normal((2, 3, emb_dim)).astype(np.float32)
    pt_inj = torch.from_numpy(block_out).unsqueeze(-2) * pt_inject.unsqueeze(-1)
    pt_combined = (pt_hyper + pt_inj.flatten(-2)).detach().cpu().numpy()

    jax_combined = np.array(jax_gr.combine(jnp.array(x), jnp.array(block_out), jax_inject))
    np.testing.assert_allclose(jax_combined, pt_combined, rtol=1e-6, atol=1e-6)

    jax_applied = np.array(jax_gr.apply_block(jnp.array(x), lambda m: jnp.array(block_out)))
    np.testing.assert_allclose(jax_applied, pt_combined, rtol=1e-6, atol=1e-6)

    with self.assertRaises(ValueError):
      jax_gr.before_block(jnp.array(x[..., :-1]))

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_l2norm_parity(self):
    """Verifies l2norm matches PyTorch modeling_qwen4_exp.l2norm."""
    assert torch is not None and modeling_qwen4_exp is not None
    rng = np.random.default_rng(42)
    x = rng.standard_normal((2, 8, 16)).astype(np.float32)
    pt_out = modeling_qwen4_exp.l2norm(torch.from_numpy(x)).detach().cpu().numpy()
    jax_out = np.array(modeling_qwen3_8_flash_next.l2norm(jnp.array(x)))
    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-6, atol=1e-6)

  def test_init_hyper_connection_streams(self):
    """Verifies stream replication across hc_count streams."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((2, 5, 16)).astype(np.float32)
    streams = modeling_qwen3_8_flash_next.init_hyper_connection_streams(jnp.array(emb), hc_count=4)
    self.assertEqual(streams.shape, (2, 5, 64))
    for c in range(4):
      np.testing.assert_allclose(np.array(streams[..., c * 16 : (c + 1) * 16]), emb, rtol=1e-6)


# =================================================================================================
# Gated DeltaNet layer
# =================================================================================================


@pytest.mark.cpu_only
class Qwen3_8GDNTest(absltest.TestCase):
  """Tests Qwen3.8 GDN layer and recurrence kernels against PyTorch reference."""

  def test_qwen3_8_gdn_parameter_tree_contract(self):
    """Verifies shapes and dtypes of all GDN parameters."""
    emb_dim, num_k_heads, num_v_heads = 256, 2, 4
    head_k_dim, head_v_dim = 64, 64
    val_dim = num_v_heads * head_v_dim
    conv_dim = (num_k_heads * head_k_dim) * 2 + val_dim
    mod = modeling_qwen3_8_flash_next.Qwen3_8GatedDeltaNet(
        emb_dim=emb_dim,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        chunk_size=16,
        output_gate_activation="silu",
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(42),
    )
    for p, shape in (
        (mod.in_proj_qkv.kernel[...], (emb_dim, conv_dim)),
        (mod.in_proj_z.kernel[...], (emb_dim, val_dim)),
        (mod.in_proj_b.kernel[...], (emb_dim, num_v_heads)),
        (mod.in_proj_a.kernel[...], (emb_dim, num_v_heads)),
        (mod.conv1d.kernel[...], (4, 1, conv_dim)),
        (mod.A_log[...], (num_v_heads,)),
        (mod.dt_bias[...], (num_v_heads,)),
        (mod.norm.scale[...], (head_v_dim,)),
        (mod.out_proj.kernel[...], (val_dim, emb_dim)),
    ):
      self.assertEqual(p.shape, shape)
    for p in (mod.A_log[...], mod.dt_bias[...], mod.norm.scale[...]):
      self.assertEqual(p.dtype, jnp.float32)

  def test_invalid_head_ratio_raises(self):
    """Verifies that num_v_heads not divisible by num_k_heads raises ValueError."""
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8GatedDeltaNet(
          emb_dim=64,
          num_k_heads=3,
          num_v_heads=4,
          output_gate_activation="sigmoid",
          rngs=nnx.Rngs(0),
      )

  def test_chunk_vs_recurrent_and_padding(self):
    """Verifies chunked scan matches recurrent forward across seq lens, including padding."""
    rng = np.random.default_rng(42)
    b, h, d_k, d_v = 2, 4, 32, 32
    for s in (64, 70):
      q, k, v = (jnp.array(rng.standard_normal((b, s, h, d), dtype=np.float32)) for d in (d_k, d_k, d_v))
      beta = jnp.array(rng.uniform(0.1, 0.9, size=(b, s, h)).astype(np.float32))
      g = jnp.array(-rng.uniform(0.01, 0.5, size=(b, s, h)).astype(np.float32))
      chunk_out, chunk_state = modeling_qwen3_8_flash_next.jax_chunk_gated_delta_rule(
          q, k, v, g, beta, chunk_size=64, output_final_state=True
      )
      rec_out, rec_state = modeling_qwen3_8_flash_next.jax_recurrent_gated_delta_rule(
          q, k, v, g, beta, output_final_state=True
      )
      self.assertIsNotNone(chunk_state)
      self.assertIsNotNone(rec_state)
      np.testing.assert_allclose(np.array(chunk_out), np.array(rec_out), rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(np.array(chunk_state), np.array(rec_state), rtol=1e-5, atol=1e-5)

  def _make_gdn_layer(self, conv_kernel_size: int) -> modeling_qwen3_8_flash_next.Qwen3_8GatedDeltaNet:
    return modeling_qwen3_8_flash_next.Qwen3_8GatedDeltaNet(
        emb_dim=64,
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=conv_kernel_size,
        chunk_size=64,
        output_gate_activation="sigmoid",
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )

  def test_layer_padding_mask_state(self):
    """Verifies right-padding leaks into neither the outputs nor the state, including through conv1d."""
    b, s_unpad, s_pad, emb_dim = 1, 16, 20, 64
    mod = self._make_gdn_layer(conv_kernel_size=4)
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((b, s_pad, emb_dim), dtype=np.float32))
    mask = jnp.concatenate([jnp.ones((b, s_unpad)), jnp.zeros((b, s_pad - s_unpad))], axis=-1)

    out_pad, state_pad = mod(x, attention_mask=mask, output_final_state=True)
    out_unpad, state_unpad = mod(x[:, :s_unpad, :], output_final_state=True)
    self.assertIsNotNone(state_pad)
    self.assertIsNotNone(state_unpad)
    np.testing.assert_allclose(np.array(out_pad[:, :s_unpad, :]), np.array(out_unpad), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(state_pad), np.array(state_unpad), rtol=1e-5, atol=1e-5)

  def test_layer_single_step_decode(self):
    """Verifies a single recurrent decode step matches the full-sequence forward pass."""
    b, s, emb_dim = 1, 16, 64
    # Kernel size 1 keeps this check valid without a conv cache, which pre-training doesn't need.
    mod = self._make_gdn_layer(conv_kernel_size=1)
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((b, s + 1, emb_dim), dtype=np.float32))

    _, prefix_state = mod(x[:, :s, :], output_final_state=True)
    decode_out, decode_state = mod(x[:, s : s + 1, :], initial_state=prefix_state, output_final_state=True)
    full_out, full_state = mod(x, output_final_state=True)
    self.assertIsNotNone(decode_state)
    self.assertIsNotNone(full_state)
    np.testing.assert_allclose(np.array(decode_out), np.array(full_out[:, s : s + 1, :]), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(decode_state), np.array(full_state), rtol=1e-5, atol=1e-5)

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_chunk_gated_delta_rule_pytorch_parity(self):
    """Verifies JAX chunked gated delta rule matches PyTorch reference kernel."""
    if torch is None or modeling_qwen4_exp is None:
      self.fail("torch and modeling_qwen4_exp must not be None")
    rng = np.random.default_rng(42)
    b, s, h, d_k, d_v = 2, 128, 4, 32, 32
    q, k, v = (rng.standard_normal((b, s, h, d), dtype=np.float32) for d in (d_k, d_k, d_v))
    beta = rng.uniform(0.1, 0.9, size=(b, s, h)).astype(np.float32)
    g = -rng.uniform(0.01, 0.5, size=(b, s, h)).astype(np.float32)

    t_inputs = tuple(torch.from_numpy(x) for x in (q, k, v, g, beta))
    pt_out, pt_state = modeling_qwen4_exp.torch_chunk_gated_delta_rule(
        *t_inputs, chunk_size=64, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    j_inputs = tuple(jnp.array(x) for x in (q, k, v, g, beta))
    jax_out, jax_state = modeling_qwen3_8_flash_next.jax_chunk_gated_delta_rule(
        *j_inputs, chunk_size=64, output_final_state=True
    )
    self.assertIsNotNone(jax_state)
    self.assertIsNotNone(pt_state)
    if jax_state is None or pt_state is None:
      self.fail("jax_state and pt_state must not be None")
    self.assertEqual(jax_state.dtype, jnp.float32)
    np.testing.assert_allclose(
        np.array(jax_out), pt_out.detach().cpu().numpy(), rtol=_GDN_PT_PARITY_RTOL, atol=_GDN_PT_PARITY_ATOL
    )
    np.testing.assert_allclose(
        np.array(jax_state), pt_state.detach().cpu().numpy(), rtol=_GDN_PT_PARITY_RTOL, atol=_GDN_PT_PARITY_ATOL
    )

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_qwen3_8_gdn_layer_pytorch_weight_copy_parity(self):
    """Verifies full Qwen3_8GatedDeltaNet matches PyTorch module under weight copy."""
    if torch is None or modeling_qwen4_exp is None or configuration_qwen4_exp is None:
      self.fail("torch, modeling_qwen4_exp, and configuration_qwen4_exp must not be None")
    emb_dim, num_k_heads, num_v_heads, head_k_dim, head_v_dim, seq_len = 64, 2, 4, 32, 32, 512
    hf_cfg = configuration_qwen4_exp.Qwen4ExpTextConfig(
        hidden_size=emb_dim,
        linear_num_key_heads=num_k_heads,
        linear_num_value_heads=num_v_heads,
        linear_key_head_dim=head_k_dim,
        linear_value_head_dim=head_v_dim,
        linear_conv_kernel_dim=4,
        output_gate_type="sigmoid",
    )
    pt_gdn = modeling_qwen4_exp.Qwen4ExpTextGatedDeltaNet(hf_cfg, layer_idx=0)
    jax_gdn = modeling_qwen3_8_flash_next.Qwen3_8GatedDeltaNet(
        emb_dim=emb_dim,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=4,
        chunk_size=64,
        output_gate_activation="sigmoid",
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    for j_p, p_p in (
        (jax_gdn.in_proj_qkv.kernel, pt_gdn.in_proj_qkv.weight.T),
        (jax_gdn.in_proj_z.kernel, pt_gdn.in_proj_z.weight.T),
        (jax_gdn.in_proj_b.kernel, pt_gdn.in_proj_b.weight.T),
        (jax_gdn.in_proj_a.kernel, pt_gdn.in_proj_a.weight.T),
        (jax_gdn.out_proj.kernel, pt_gdn.out_proj.weight.T),
        (jax_gdn.A_log, pt_gdn.A_log),
        (jax_gdn.dt_bias, pt_gdn.dt_bias),
        (jax_gdn.norm.scale, pt_gdn.norm.weight),
    ):
      j_p[...] = jnp.array(p_p.detach().cpu().numpy())
    jax_gdn.conv1d.kernel[...] = jnp.array(pt_gdn.conv1d.weight.detach().cpu().permute(2, 1, 0).numpy())

    rng = np.random.default_rng(123)
    x = rng.standard_normal((2, seq_len, emb_dim)).astype(np.float32)
    pt_out = pt_gdn(torch.from_numpy(x)).detach().cpu().numpy()
    jax_out = np.array(jax_gdn(jnp.array(x)))
    np.testing.assert_allclose(jax_out, pt_out, rtol=_GDN_PT_PARITY_RTOL, atol=_GDN_PT_PARITY_ATOL)

  def test_block_doubling_unit_lower_inverse_parity_and_validation(self):
    """Verifies the block-doubling unit-lower inverse matches solve_triangular / fp64 in kernel and layer."""
    rng = np.random.default_rng(7)
    for c_size in (16, 64):
      raw = rng.standard_normal((2, 3, c_size, c_size), dtype=np.float32) * 0.1
      s_mat = jnp.tril(jnp.array(raw), k=-1)
      rhs = jnp.array(rng.standard_normal((2, 3, c_size, 24), dtype=np.float32))
      expected = jax.scipy.linalg.solve_triangular(
          jnp.eye(c_size, dtype=jnp.float32) + s_mat, rhs, lower=True, unit_diagonal=True
      )
      actual = jnp.matmul(
          modeling_qwen3_8_flash_next._block_doubling_unit_lower_inverse(s_mat, c_size),  # pylint: disable=protected-access
          rhs,
          precision=jax.lax.Precision.HIGHEST,
      )
      np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5)

    # Correlated keys + weak decay: entries of (I+S) stay <= 1 but the Neumann powers N^(2^j) blow up to 1e7+, so a
    # product-form inverse cancels catastrophically in float32. Block doubling must stay accurate vs an fp64 solve.
    d_k = 32
    for c_size in (16, 48, 64):
      rhs64 = rng.standard_normal((c_size, 24))
      for case_name, key_cos, beta_val, g_val in (
          ("identical_keys_weak_decay", 1.0, 0.9, -0.05),
          ("correlated_keys_weak_decay", 0.5, 0.9, -0.05),
          ("identical_keys_no_decay", 1.0, 1.0, 0.0),
      ):
        with self.subTest(case_name=case_name, chunk_size=c_size):
          base = rng.standard_normal((1, d_k))
          keys = np.sqrt(key_cos) * base + np.sqrt(1.0 - key_cos) * rng.standard_normal((c_size, d_k))
          keys = keys / np.linalg.norm(keys, axis=-1, keepdims=True)
          g_cumsum = np.cumsum(np.full((c_size,), g_val))
          decay = np.tril(np.exp(np.tril(g_cumsum[:, None] - g_cumsum[None, :])), k=-1)
          s64 = np.tril((beta_val * keys) @ keys.T * decay, k=-1)
          s32 = jnp.array(s64, dtype=jnp.float32)
          expected = np.linalg.solve(np.eye(c_size) + s64, rhs64)
          actual = jnp.matmul(
              modeling_qwen3_8_flash_next._block_doubling_unit_lower_inverse(s32, c_size),  # pylint: disable=protected-access
              jnp.array(rhs64, dtype=jnp.float32),
              precision=jax.lax.Precision.HIGHEST,
          )
          np.testing.assert_allclose(np.array(actual), expected, rtol=1e-5, atol=1e-5)

    with self.assertRaises(ValueError):
      q0 = jnp.zeros((1, 16, 2, 16), dtype=jnp.float32)
      b0 = jnp.zeros((1, 16, 2), dtype=jnp.float32)
      modeling_qwen3_8_flash_next.jax_chunk_gated_delta_rule(q0, q0, q0, b0, b0, chunk_inverse="invalid")
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8GatedDeltaNet(
          output_gate_activation="silu", chunk_inverse="invalid", rngs=nnx.Rngs(0)
      )

    emb_dim = 64
    kw = {
        "emb_dim": emb_dim,
        "num_k_heads": 2,
        "num_v_heads": 4,
        "head_k_dim": 32,
        "head_v_dim": 32,
        "chunk_size": 16,
        "output_gate_activation": "silu",
        "dtype": jnp.float32,
        "weight_dtype": jnp.float32,
    }
    mod_solve = modeling_qwen3_8_flash_next.Qwen3_8GatedDeltaNet(**kw, chunk_inverse="solve", rngs=nnx.Rngs(5))
    mod_block_doubling = modeling_qwen3_8_flash_next.Qwen3_8GatedDeltaNet(
        **kw, chunk_inverse="block_doubling", rngs=nnx.Rngs(5)
    )
    nnx.update(mod_block_doubling, nnx.state(mod_solve))

    x = jnp.array(rng.standard_normal((2, 70, emb_dim), dtype=np.float32))
    out_s, st_s = mod_solve(x, output_final_state=True)
    out_n, st_n = mod_block_doubling(x, output_final_state=True)
    np.testing.assert_allclose(np.array(out_n), np.array(out_s), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(st_n), np.array(st_s), rtol=1e-5, atol=1e-5)

    def loss_fn(mod: modeling_qwen3_8_flash_next.Qwen3_8GatedDeltaNet, inp: jax.Array) -> jax.Array:
      out = mod(inp)
      assert isinstance(out, jax.Array)
      return jnp.sum(out * jnp.arange(1, emb_dim + 1, dtype=jnp.float32))

    _, (grads_s, dx_s) = nnx.value_and_grad(loss_fn, argnums=(0, 1))(mod_solve, x)
    _, (grads_n, dx_n) = nnx.value_and_grad(loss_fn, argnums=(0, 1))(mod_block_doubling, x)
    _assert_allclose_to_scale(dx_n, dx_s)
    _assert_allclose_to_scale(grads_n.in_proj_qkv.kernel[...], grads_s.in_proj_qkv.kernel[...])


# =================================================================================================
# Gated DeltaNet recurrence numerics
# =================================================================================================


@pytest.mark.cpu_only
class Qwen3_8GDNNumericsTest(absltest.TestCase):
  """Validates numerical stability and PyTorch parity at long sequences."""

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_chunked_scan_numerics_long_seq(self):
    """Verifies JAX vs PyTorch chunked scan parity at seq_len=8192."""
    if torch is None or modeling_qwen4_exp is None:
      self.fail("torch and modeling_qwen4_exp must not be None")
    rng = np.random.default_rng(2026)
    b, seq_len, h, d_k, d_v = 1, 8192, 2, 32, 32
    q, k, v = (rng.standard_normal((b, seq_len, h, d), dtype=np.float32) * 0.1 for d in (d_k, d_k, d_v))
    beta = rng.uniform(0.1, 0.9, size=(b, seq_len, h)).astype(np.float32)
    g = -rng.uniform(0.01, 0.2, size=(b, seq_len, h)).astype(np.float32)

    t_inputs = tuple(torch.from_numpy(x) for x in (q, k, v, g, beta))
    pt_out, pt_state = modeling_qwen4_exp.torch_chunk_gated_delta_rule(
        *t_inputs, chunk_size=64, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    j_inputs = tuple(jnp.array(x) for x in (q, k, v, g, beta))
    jax_out, jax_state = modeling_qwen3_8_flash_next.jax_chunk_gated_delta_rule(
        *j_inputs, chunk_size=64, output_final_state=True
    )
    self.assertIsNotNone(jax_state)
    self.assertIsNotNone(pt_state)
    if jax_state is None or pt_state is None:
      self.fail("jax_state and pt_state must not be None")
    np.testing.assert_allclose(
        np.array(jax_out), pt_out.detach().cpu().numpy(), rtol=_GDN_PT_PARITY_RTOL, atol=_GDN_PT_PARITY_ATOL
    )
    np.testing.assert_allclose(
        np.array(jax_state), pt_state.detach().cpu().numpy(), rtol=_GDN_PT_PARITY_RTOL, atol=_GDN_PT_PARITY_ATOL
    )

  def test_chunk_gated_delta_rule_backward_grad_finite_large_decay(self):
    """Checks backward grad through chunked scan is finite under large decay."""
    rng = np.random.default_rng(2026)
    b, s, h, d_k, d_v = 1, 128, 2, 32, 32
    q, k, v = (jnp.array(rng.standard_normal((b, s, h, d), dtype=np.float32) * 0.1) for d in (d_k, d_k, d_v))
    beta = jnp.array(rng.uniform(0.1, 0.9, size=(b, s, h)).astype(np.float32))
    g = jnp.full((b, s, h), -5.0, dtype=jnp.float32)

    def loss_fn(q_in: jax.Array, k_in: jax.Array, v_in: jax.Array, g_in: jax.Array, beta_in: jax.Array) -> jax.Array:
      out, _ = modeling_qwen3_8_flash_next.jax_chunk_gated_delta_rule(q_in, k_in, v_in, g_in, beta_in, chunk_size=64)
      return jnp.sum(out)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
    for grad in grads:
      self.assertTrue(np.all(np.isfinite(np.array(grad))))

  def test_block_doubling_chunked_scan_numerics_long_seq_and_grad(self):
    """Checks the block-doubling chunk inverse matches solve_triangular at seq_len=8192 and under large decay."""
    rng = np.random.default_rng(2027)
    b, seq_len, h, d_k, d_v = 1, 8192, 2, 32, 32
    q, k, v = (jnp.array(rng.standard_normal((b, seq_len, h, d), dtype=np.float32) * 0.1) for d in (d_k, d_k, d_v))
    beta = jnp.array(rng.uniform(0.1, 0.9, size=(b, seq_len, h)).astype(np.float32))
    g = jnp.array(-rng.uniform(0.01, 0.2, size=(b, seq_len, h)).astype(np.float32))

    solve_out, solve_state = modeling_qwen3_8_flash_next.jax_chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=64, output_final_state=True, chunk_inverse="solve"
    )
    block_doubling_out, block_doubling_state = modeling_qwen3_8_flash_next.jax_chunk_gated_delta_rule(
        q, k, v, g, beta, chunk_size=64, output_final_state=True, chunk_inverse="block_doubling"
    )
    np.testing.assert_allclose(np.array(block_doubling_out), np.array(solve_out), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(block_doubling_state), np.array(solve_state), rtol=1e-5, atol=1e-5)

    s_short = 128
    q_s, k_s, v_s = (x[:, :s_short] for x in (q, k, v))
    beta_s = beta[:, :s_short]
    g_s = jnp.full((b, s_short, h), -5.0, dtype=jnp.float32)

    def make_loss(inv_mode: str):
      def _loss(q_in: jax.Array, k_in: jax.Array, v_in: jax.Array, g_in: jax.Array, b_in: jax.Array) -> jax.Array:
        out, _ = modeling_qwen3_8_flash_next.jax_chunk_gated_delta_rule(
            q_in, k_in, v_in, g_in, b_in, chunk_size=64, chunk_inverse=inv_mode
        )
        return jnp.sum(out)

      return _loss

    grads_solve = jax.grad(make_loss("solve"), argnums=(0, 1, 2, 3, 4))(q_s, k_s, v_s, g_s, beta_s)
    grads_block_doubling = jax.grad(make_loss("block_doubling"), argnums=(0, 1, 2, 3, 4))(q_s, k_s, v_s, g_s, beta_s)
    for g_solve, g_block_doubling in zip(grads_solve, grads_block_doubling):
      self.assertTrue(np.all(np.isfinite(np.array(g_block_doubling))))
      np.testing.assert_allclose(np.array(g_block_doubling), np.array(g_solve), rtol=1e-5, atol=1e-5)

  def test_block_doubling_chunk_inverse_correlated_keys_weak_decay(self):
    """Checks the MXU chunk inverse matches solve_triangular for correlated keys with weak decay.

    Identical or strongly correlated l2-normed keys with beta ~0.9 and g ~-0.05 keep (I+S)^-1 at O(1) while the
    Neumann powers N^(2^j) grow past 1e7, which breaks a product-form inverse in float32; the block-doubling inverse
    must stay at solve parity in outputs, final state and gradients.
    """
    rng = np.random.default_rng(2028)
    b, s, h, d_k, d_v = 1, 256, 2, 32, 32
    base_key = rng.standard_normal((1, 1, h, d_k)).astype(np.float32)
    q = jnp.array(rng.standard_normal((b, s, h, d_k), dtype=np.float32))
    v = jnp.array(rng.standard_normal((b, s, h, d_v), dtype=np.float32) * 0.1)
    beta = jnp.full((b, s, h), 0.9, dtype=jnp.float32)
    g = jnp.full((b, s, h), -0.05, dtype=jnp.float32)

    def make_loss(inv_mode: str):
      def _loss(q_in: jax.Array, k_in: jax.Array, v_in: jax.Array, g_in: jax.Array, b_in: jax.Array) -> jax.Array:
        out, _ = modeling_qwen3_8_flash_next.jax_chunk_gated_delta_rule(
            q_in, k_in, v_in, g_in, b_in, chunk_size=64, chunk_inverse=inv_mode
        )
        return jnp.sum(out)

      return _loss

    for case_name, key_noise in (("identical_keys", 0.0), ("correlated_keys", 0.3)):
      with self.subTest(case_name=case_name):
        k = jnp.array(base_key + key_noise * rng.standard_normal((b, s, h, d_k)).astype(np.float32))
        results = {
            inv_mode: modeling_qwen3_8_flash_next.jax_chunk_gated_delta_rule(
                q, k, v, g, beta, chunk_size=64, output_final_state=True, chunk_inverse=inv_mode
            )
            for inv_mode in ("solve", "block_doubling")
        }
        for got, want in zip(results["block_doubling"], results["solve"]):
          self.assertTrue(np.all(np.isfinite(np.array(got))))
          np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-4)
        grads_solve = jax.grad(make_loss("solve"), argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
        grads_block_doubling = jax.grad(make_loss("block_doubling"), argnums=(0, 1, 2, 3, 4))(q, k, v, g, beta)
        for g_solve, g_block_doubling in zip(grads_solve, grads_block_doubling):
          self.assertTrue(np.all(np.isfinite(np.array(g_block_doubling))))
          # Two different fp32 inverse algorithms through a 256-step recurrence differ by accumulation-order noise
          # (~1e-7 absolute); the product-form inverse this guards against is off by O(1) here.
          _assert_allclose_to_scale(g_block_doubling, g_solve, atol_scale=1e-4)


# =================================================================================================
# Qwen Sparse Attention and indexer
# =================================================================================================


def _make_hf_config(**kwargs) -> Any:
  """Constructs a test Qwen4ExpTextConfig."""
  if configuration_qwen4_exp is None:
    raise RuntimeError("transformers.models.qwen4_exp is not available")
  rope = {
      "rope_type": "default",
      "rope_theta": 10000.0,
      "partial_rotary_factor": 0.25,
      "mrope_interleaved": True,
      "mrope_section": [2, 1, 1],
  }
  cfg = {
      "hidden_size": 64,
      "num_attention_heads": 4,
      "num_key_value_heads": 2,
      "head_dim": 32,
      "indexer_n_heads": 2,
      "indexer_kv_heads": 1,
      "indexer_head_dim": 32,
      "indexer_budget": 16,
      "indexer_compress_ratio": 4,
      "rms_norm_eps": 1e-6,
      "attention_bias": False,
      "attention_dropout": 0.0,
      "rope_parameters": rope,
  }
  cfg.update(kwargs)
  return configuration_qwen4_exp.Qwen4ExpTextConfig(**cfg)


def _copy_weights(pt: Any, jx: modeling_qwen3_8_flash_next.Qwen3_8SparseAttention):
  """Copies PyTorch attention weights to JAX QSA module."""

  def cv(t: Any) -> jax.Array:
    return jnp.array(t.detach().cpu().numpy())

  jx.q_proj.kernel[...] = cv(pt.q_proj.weight.T)
  jx.k_proj.kernel[...] = cv(pt.k_proj.weight.T)
  jx.v_proj.kernel[...] = cv(pt.v_proj.weight.T)
  jx.o_proj.kernel[...] = cv(pt.o_proj.weight.T)
  jx.q_norm.scale[...] = cv(pt.q_norm.weight)
  jx.k_norm.scale[...] = cv(pt.k_norm.weight)
  jx.indexer.index_qk_proj.kernel[...] = cv(pt.indexer.index_qk_proj.weight.T)
  jx.indexer.q_layernorm.scale[...] = cv(pt.indexer.q_layernorm.weight)
  jx.indexer.k_layernorm.scale[...] = cv(pt.indexer.k_layernorm.weight)


@pytest.mark.cpu_only
class Qwen3_8QSAAttentionTest(absltest.TestCase):
  """Tests QSA attention, indexer, and mRoPE against PyTorch reference."""

  def _make_jax_qsa(
      self, emb_dim: int = 64, rng_seed: int = 0, token_budget: int = 16
  ) -> modeling_qwen3_8_flash_next.Qwen3_8SparseAttention:
    """Constructs a test JAX QSA module."""
    args = (emb_dim, 4, 2, 32, 0.25, 10000.0, (2, 1, 1), 2, 1, 32, token_budget, 4, 1e-6, jnp.float32, jnp.float32)
    return modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(*args, rngs=nnx.Rngs(rng_seed))

  def test_qsa_parameter_tree_contract(self):
    """Verifies shapes and dtypes of all QSA parameters."""
    m = self._make_jax_qsa()
    self.assertEqual((m.q_proj.kernel[...].shape, m.k_proj.kernel[...].shape), ((64, 256), (64, 64)))
    self.assertEqual((m.v_proj.kernel[...].shape, m.o_proj.kernel[...].shape), ((64, 64), (128, 64)))
    self.assertEqual((m.q_norm.scale[...].shape, m.k_norm.scale[...].shape), ((32,), (32,)))
    self.assertEqual(
        (m.indexer.index_qk_proj.kernel[...].shape, m.indexer.q_layernorm.scale[...].shape), ((64, 96), (32,))
    )
    self.assertEqual((m.indexer.k_layernorm.scale[...].shape, m.q_proj.kernel[...].dtype), ((32,), jnp.float32))

  def _make_pt_and_jax(
      self, seed: int = 0, token_budget: int = 16
  ) -> tuple[Any, Any, modeling_qwen3_8_flash_next.Qwen3_8SparseAttention]:
    """Constructs parity-initialized PyTorch and JAX attention modules."""
    if torch is None or modeling_qwen4_exp is None or configuration_qwen4_exp is None:
      raise RuntimeError("PyTorch and transformers.models.qwen4_exp must be available")
    torch.manual_seed(seed)
    cfg = _make_hf_config(indexer_budget=token_budget)
    pt_r = modeling_qwen4_exp.Qwen4ExpTextRotaryEmbedding(cfg)
    pt_a = modeling_qwen4_exp.Qwen4ExpTextAttention(cfg, layer_idx=3)
    jax_a = self._make_jax_qsa(rng_seed=seed, token_budget=token_budget)
    _copy_weights(pt_a, jax_a)
    return pt_r, pt_a, jax_a

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_rotary_embedding_and_partial_mrope_pytorch_parity(self):
    """Verifies partial interleaved mRoPE matches PyTorch reference."""
    if torch is None or modeling_qwen4_exp is None or configuration_qwen4_exp is None:
      self.fail("transformers qwen4_exp reference modules are missing")
    hf_cfg = _make_hf_config()
    pt_rope = modeling_qwen4_exp.Qwen4ExpTextRotaryEmbedding(hf_cfg)
    jax_rope = modeling_qwen3_8_flash_next.Qwen3_8RotaryEmbedding(32, 0.25, 10000.0, (2, 1, 1))
    b, s = 2, 16
    pos_2d = np.tile(np.arange(s), (b, 1))
    pos_3d = np.stack([pos_2d, pos_2d + 7, pos_2d * 2], axis=0)
    x_np = np.ones((b, s, 32), dtype=np.float32)

    for case_name, pos in (("2d", pos_2d), ("3d", pos_3d)):
      with self.subTest(case_name=case_name):
        pt_cos, pt_sin = pt_rope(torch.from_numpy(x_np), torch.from_numpy(pos))
        jax_cos, jax_sin = jax_rope(jnp.array(x_np), jnp.array(pos))
        np.testing.assert_allclose(np.array(jax_cos), pt_cos.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.array(jax_sin), pt_sin.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_qsa_attention_pytorch_weight_copy_parity_at_32(self):
    """Verifies output and selected-block parity vs HF attention at seq_len=32."""
    if torch is None or modeling_qwen4_exp is None or configuration_qwen4_exp is None:
      self.fail("transformers qwen4_exp reference modules are missing")
    b, s, emb_dim = 1, 32, 64
    pt_rope, pt_attn, jax_attn = self._make_pt_and_jax(seed=0)

    x = np.random.default_rng(42).standard_normal((b, s, emb_dim)).astype(np.float32)
    pos = np.tile(np.arange(s), (b, 1))

    pt_cos, pt_sin = pt_rope(torch.from_numpy(x), torch.from_numpy(pos))
    mask = torch.tril(torch.ones((b, 1, s, s), dtype=torch.bool))
    pt_float_mask = torch.where(mask, torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min)

    pt_out, _ = pt_attn(torch.from_numpy(x), (pt_cos, pt_sin), attention_mask=pt_float_mask)
    jax_out, jax_blocks = jax_attn(jnp.array(x), position_ids=jnp.array(pos), return_selected_blocks=True)

    pt_tok_mask = pt_attn.indexer(torch.from_numpy(x), (pt_cos, pt_sin), pt_float_mask, None)
    pt_tok_mask_arr = pt_tok_mask.detach().cpu().numpy()
    pt_tok_mask_np = pt_tok_mask_arr if pt_tok_mask_arr.dtype == np.bool_ else (pt_tok_mask_arr == 0.0)
    for q_idx in range(s):
      n_complete = (q_idx + 1) // 4
      if n_complete > 0:
        hf_sel = {m for m in range(n_complete) if np.all(pt_tok_mask_np[0, 0, q_idx, m * 4 : (m + 1) * 4])}
        jax_sel = {int(idx) for idx in np.array(jax_blocks[0, q_idx]) if idx >= 0}
        self.assertEqual(hf_sel, jax_sel)
    np.testing.assert_allclose(np.array(jax_out), pt_out.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

  def test_indexer_packed_sequence_and_partial_block_handling(self):
    """Verifies packed-sequence isolation and partial block selection (block_topk < n_complete)."""
    b, s, emb_dim = 1, 28, 64
    # Document A = 1 on [0..7] (blocks 0, 1), Document B = 2 on [8..25], Padding = 0 on [26..27]
    seg_ids = np.zeros((b, s), dtype=np.int32)
    seg_ids[0, :8], seg_ids[0, 8:25] = 1, 2

    x = np.random.default_rng(123).standard_normal((b, s, emb_dim)).astype(np.float32)
    pos = np.tile(np.arange(s), (b, 1))

    # indexer_budget=8 gives block_topk = 8 // 4 = 2.
    # At query t=20, Document B has 3 candidate blocks: [8..11] (2), [12..15] (3), [16..19] (4).
    # Since block_topk = 2 < 3, top-k selection picks exactly 2 blocks from {2, 3, 4}.
    jax_attn = self._make_jax_qsa(emb_dim=emb_dim, rng_seed=1, token_budget=8)

    pt_out: Any = None
    pt_tok_mask_np: Any = None
    if _HAS_QWEN4_EXP:
      if torch is None or modeling_qwen4_exp is None or configuration_qwen4_exp is None:
        self.fail("transformers qwen4_exp reference modules are missing")
      pt_rope, pt_attn, _ = self._make_pt_and_jax(seed=1, token_budget=8)
      _copy_weights(pt_attn, jax_attn)

      pt_cos, pt_sin = pt_rope(torch.from_numpy(x), torch.from_numpy(pos))
      same = (seg_ids[:, :, None] == seg_ids[:, None, :]) & (seg_ids[:, :, None] != 0)
      pt_valid = same & np.tril(np.ones((s, s), dtype=bool))[None, :, :]
      pt_float_mask = torch.where(
          torch.from_numpy(pt_valid[:, None, :, :]), torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min
      )

      pt_out, _ = pt_attn(torch.from_numpy(x), (pt_cos, pt_sin), attention_mask=pt_float_mask)
      pt_tok_mask = pt_attn.indexer(torch.from_numpy(x), (pt_cos, pt_sin), pt_float_mask, None)
      pt_tok_mask_arr = pt_tok_mask.detach().cpu().numpy()
      pt_tok_mask_np = pt_tok_mask_arr if pt_tok_mask_arr.dtype == np.bool_ else (pt_tok_mask_arr == 0.0)

    jax_out, jax_blocks = jax_attn(
        jnp.array(x), position_ids=jnp.array(pos), segment_ids=jnp.array(seg_ids), return_selected_blocks=True
    )
    cos, sin = jax_attn.rotary_emb(jnp.array(x), jnp.array(pos))
    tok_mask, _ = jax_attn.indexer(jnp.array(x), (cos, sin), segment_ids=jnp.array(seg_ids))
    tok_mask_np = np.array(tok_mask)[0, 0, 20]

    num_blocks = s // 4
    jax_active_blocks = {m for m in range(num_blocks) if np.all(tok_mask_np[m * 4 : (m + 1) * 4])}

    # Pure-JAX invariants:
    self.assertLen(jax_active_blocks, 2)
    self.assertTrue(jax_active_blocks.issubset({2, 3, 4}))
    self.assertNotIn(0, jax_active_blocks)
    self.assertNotIn(1, jax_active_blocks)
    self.assertFalse(np.any(tok_mask_np[:8]))  # Document A tokens NEVER visible
    self.assertTrue(tok_mask_np[20])  # Tail token visible
    self.assertFalse(np.any(tok_mask_np[21:]))  # Future tokens not visible
    selected_jax_blocks = {int(idx) + 2 for idx in np.array(jax_blocks[0, 20]) if idx >= 0}
    self.assertEqual(selected_jax_blocks, jax_active_blocks)

    if _HAS_QWEN4_EXP:
      hf_active_blocks = {m for m in range(num_blocks) if np.all(pt_tok_mask_np[0, 0, 20, m * 4 : (m + 1) * 4])}
      self.assertEqual(jax_active_blocks, hf_active_blocks)
      np.testing.assert_allclose(np.array(jax_out)[:, :25], pt_out[:, :25].detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

  def test_attention_mask_folding_and_init_validations(self):
    """Verifies 2D mask folding equivalence, 4D mask rejection, and init validation raises."""
    b, s, emb_dim = 2, 16, 64
    x = np.random.default_rng(0).standard_normal((b, s, emb_dim)).astype(np.float32)
    pos = np.tile(np.arange(s), (b, 1))

    # 1. 2D mask folding equivalence
    mask_2d = np.ones((b, s), dtype=np.int32)
    mask_2d[0, 12:] = 0
    mask_2d[1, 14:] = 0

    jax_attn = self._make_jax_qsa(emb_dim=emb_dim, rng_seed=42)
    out_mask, blocks_mask = jax_attn(
        jnp.array(x), position_ids=jnp.array(pos), attention_mask=jnp.array(mask_2d), return_selected_blocks=True
    )
    folded_seg = (mask_2d != 0).astype(np.int32)
    out_seg, blocks_seg = jax_attn(
        jnp.array(x), position_ids=jnp.array(pos), segment_ids=jnp.array(folded_seg), return_selected_blocks=True
    )
    np.testing.assert_allclose(np.array(out_mask), np.array(out_seg), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.array(blocks_mask), np.array(blocks_seg))

    # 2. 4D mask rejection
    mask_4d = jnp.zeros((b, 1, s, s), dtype=jnp.float32)
    with self.assertRaises(ValueError):
      jax_attn(jnp.array(x), attention_mask=mask_4d)

    cos, sin = jax_attn.rotary_emb(jnp.array(x), jnp.array(pos))
    with self.assertRaises(ValueError):
      jax_attn.indexer(jnp.array(x), (cos, sin), attention_mask=mask_4d)

    # 3. Init validation raises
    # Qwen3_8RotaryEmbedding: length != 3 or sum != dim // 2
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8RotaryEmbedding(head_dim=32, mrope_section=(2, 2))
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8RotaryEmbedding(head_dim=32, mrope_section=(2, 2, 1))
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8RotaryEmbedding(head_dim=64, mrope_section=(11, 11, 10))

    # Qwen3_8QSAIndexer: index_kv_heads != 1, compress_ratio <= 0, token_budget < compress_ratio
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(index_kv_heads=2, rngs=nnx.Rngs(0))
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(compress_ratio=0, rngs=nnx.Rngs(0))
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(token_budget=2, compress_ratio=4, rngs=nnx.Rngs(0))

    # Qwen3_8SparseAttention: num_query_heads % num_kv_heads != 0
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(num_query_heads=5, num_kv_heads=2, rngs=nnx.Rngs(0))

  def test_indexer_reshape_mask_mode_bit_exact_parity(self):
    """Verifies mask_mode='reshape' is bit-identical to 'gather' across budget and packing modes."""
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(mask_mode="invalid", rngs=nnx.Rngs(0))

    b, emb_dim = 2, 64
    rng = np.random.default_rng(7)
    rope = modeling_qwen3_8_flash_next.Qwen3_8RotaryEmbedding(32, 0.25, 10000.0, (2, 1, 1))

    # Case 1: s <= token_budget (unpacked and packed)
    s_short = 32
    x_short = jnp.array(rng.standard_normal((b, s_short, emb_dim), dtype=np.float32))
    pos_short = jnp.tile(jnp.arange(s_short), (b, 1))
    cos_s, sin_s = rope(x_short, pos_short)

    idx_gather_le = modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(
        emb_dim, 2, 1, 32, 64, 4, 1e-6, jnp.float32, jnp.float32, mask_mode="gather", rngs=nnx.Rngs(3)
    )
    idx_reshape_le = modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(
        emb_dim, 2, 1, 32, 64, 4, 1e-6, jnp.float32, jnp.float32, mask_mode="reshape", rngs=nnx.Rngs(3)
    )
    nnx.update(idx_reshape_le, nnx.state(idx_gather_le))

    seg_short = jnp.ones((b, s_short), dtype=jnp.int32).at[0, 16:28].set(2).at[0, 28:].set(0)
    mask_short = jnp.ones((b, s_short), dtype=jnp.int32).at[1, 24:].set(0)
    for seg_in, mask_in in ((None, None), (seg_short, None), (seg_short, mask_short)):
      m_g, b_g = idx_gather_le(x_short, (cos_s, sin_s), attention_mask=mask_in, segment_ids=seg_in)
      m_r, b_r = idx_reshape_le(x_short, (cos_s, sin_s), attention_mask=mask_in, segment_ids=seg_in)
      np.testing.assert_array_equal(np.array(m_r), np.array(m_g))
      np.testing.assert_array_equal(np.array(b_r), np.array(b_g))

    m_none, b_none = idx_reshape_le(x_short, (cos_s, sin_s), return_token_mask=False)
    self.assertIsNone(m_none)
    np.testing.assert_array_equal(np.array(b_none), np.array(idx_gather_le(x_short, (cos_s, sin_s))[1]))

    # Case 2: s > token_budget (unpacked non-multiple of compress_ratio, and packed fallback)
    s_long = 70
    x_long = jnp.array(rng.standard_normal((b, s_long, emb_dim), dtype=np.float32))
    pos_long = jnp.tile(jnp.arange(s_long), (b, 1))
    cos_l, sin_l = rope(x_long, pos_long)

    idx_gather_gt = modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(
        emb_dim, 2, 1, 32, 16, 4, 1e-6, jnp.float32, jnp.float32, mask_mode="gather", rngs=nnx.Rngs(5)
    )
    idx_reshape_gt = modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(
        emb_dim, 2, 1, 32, 16, 4, 1e-6, jnp.float32, jnp.float32, mask_mode="reshape", rngs=nnx.Rngs(5)
    )
    nnx.update(idx_reshape_gt, nnx.state(idx_gather_gt))

    m_g_unpacked, b_g_unpacked = idx_gather_gt(x_long, (cos_l, sin_l))
    m_r_unpacked, b_r_unpacked = idx_reshape_gt(x_long, (cos_l, sin_l))
    np.testing.assert_array_equal(np.array(m_r_unpacked), np.array(m_g_unpacked))
    np.testing.assert_array_equal(np.array(b_r_unpacked), np.array(b_g_unpacked))

    seg_long = jnp.ones((b, s_long), dtype=jnp.int32).at[0, 30:60].set(2).at[0, 60:].set(0)
    m_g_packed, b_g_packed = idx_gather_gt(x_long, (cos_l, sin_l), segment_ids=seg_long)
    m_r_packed, b_r_packed = idx_reshape_gt(x_long, (cos_l, sin_l), segment_ids=seg_long)
    np.testing.assert_array_equal(np.array(m_r_packed), np.array(m_g_packed))
    np.testing.assert_array_equal(np.array(b_r_packed), np.array(b_g_packed))

  def test_indexer_reshape_packed_multi_doc_parity(self):
    """Verifies gather-free packed reshape matches gather bit-identically at S=64 and S=512 with >=3 unequal docs."""
    b, emb_dim = 2, 64
    rng = np.random.default_rng(29)
    rope = modeling_qwen3_8_flash_next.Qwen3_8RotaryEmbedding(32, 0.25, 10000.0, (2, 1, 1))

    for s, budget, doc_splits_b0, doc_splits_b1 in (
        (64, 16, (0, 17, 39, 58), (0, 11, 29, 53, 64)),
        (512, 64, (0, 93, 247, 419, 490), (0, 130, 311, 477, 512)),
    ):
      with self.subTest(seq_len=s, token_budget=budget):
        x = jnp.array(rng.standard_normal((b, s, emb_dim), dtype=np.float32))
        pos = jnp.tile(jnp.arange(s), (b, 1))
        cos, sin = rope(x, pos)

        seg_np = np.zeros((b, s), dtype=np.int32)
        for doc_id, (start, end) in enumerate(zip(doc_splits_b0[:-1], doc_splits_b0[1:]), start=1):
          seg_np[0, start:end] = doc_id
        for doc_id, (start, end) in enumerate(zip(doc_splits_b1[:-1], doc_splits_b1[1:]), start=1):
          seg_np[1, start:end] = doc_id
        seg = jnp.array(seg_np)
        attn_mask = (seg != 0).astype(jnp.int32)

        idx_gather = modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(
            emb_dim, 2, 1, 32, budget, 4, 1e-6, jnp.float32, jnp.float32, mask_mode="gather", rngs=nnx.Rngs(31)
        )
        idx_reshape = modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(
            emb_dim, 2, 1, 32, budget, 4, 1e-6, jnp.float32, jnp.float32, mask_mode="reshape", rngs=nnx.Rngs(31)
        )
        nnx.update(idx_reshape, nnx.state(idx_gather))

        for mask_in in (None, attn_mask):
          m_g, b_g = idx_gather(x, (cos, sin), attention_mask=mask_in, segment_ids=seg)
          m_r, b_r = idx_reshape(x, (cos, sin), attention_mask=mask_in, segment_ids=seg)
          np.testing.assert_array_equal(np.array(m_r), np.array(m_g))
          np.testing.assert_array_equal(np.array(b_r), np.array(b_g))

  def test_flash_attention_kernel_parity_and_validation(self):
    """Verifies attention_kernel='flash' matches 'dense' for both s <= token_budget and s > token_budget."""
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(attention_kernel="invalid", rngs=nnx.Rngs(0))

    b, s, emb_dim = 2, 32, 64
    args_64 = (emb_dim, 4, 2, 32, 0.25, 10000.0, (2, 1, 1), 2, 1, 32, 64, 4, 1e-6, jnp.float32, jnp.float32)
    dense_attn = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(*args_64, attention_kernel="dense", rngs=nnx.Rngs(11))
    flash_attn = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
        *args_64, indexer_mask_mode="reshape", attention_kernel="flash", rngs=nnx.Rngs(11)
    )
    nnx.update(flash_attn, nnx.state(dense_attn))

    rng = np.random.default_rng(19)
    x = jnp.array(rng.standard_normal((b, s, emb_dim), dtype=np.float32))
    pos = jnp.tile(jnp.arange(s), (b, 1))
    mask_2d = jnp.ones((b, s), dtype=jnp.int32).at[0, 26:].set(0).at[1, 29:].set(0)
    seg_2d = jnp.ones((b, s), dtype=jnp.int32).at[0, 12:24].set(2).at[0, 24:].set(0).at[1, 20:].set(2)

    for case_name, mask_in, seg_in in (
        ("unpacked", None, None),
        ("mask_2d", mask_2d, None),
        ("packed_seg", None, seg_2d),
    ):
      with self.subTest(case_name=case_name):
        # Padding positions attend to nothing under flash; only non-padding outputs must match dense.
        valid = jnp.ones((b, s), dtype=bool)
        if mask_in is not None:
          valid &= mask_in != 0
        if seg_in is not None:
          valid &= seg_in != 0
        out_d, blk_d = dense_attn(
            x, position_ids=pos, attention_mask=mask_in, segment_ids=seg_in, return_selected_blocks=True
        )
        out_f, blk_f = flash_attn(
            x, position_ids=pos, attention_mask=mask_in, segment_ids=seg_in, return_selected_blocks=True
        )
        np.testing.assert_array_equal(np.array(blk_f), np.array(blk_d))
        self.assertTrue(np.all(np.isfinite(np.array(out_f))))
        np.testing.assert_allclose(
            np.array(out_f)[np.array(valid)], np.array(out_d)[np.array(valid)], rtol=1e-4, atol=1e-4
        )

        def loss_fn(
            mod: modeling_qwen3_8_flash_next.Qwen3_8SparseAttention,
            inp: jax.Array,
            mask_in: jax.Array | None = mask_in,
            seg_in: jax.Array | None = seg_in,
            valid: jax.Array = valid,
        ) -> jax.Array:
          out = mod(inp, position_ids=pos, attention_mask=mask_in, segment_ids=seg_in)
          assert isinstance(out, jax.Array)
          return jnp.sum(jnp.where(valid[:, :, None], out, 0.0) * jnp.arange(1, emb_dim + 1, dtype=jnp.float32))

        _, (grads_d, dx_d) = nnx.value_and_grad(loss_fn, argnums=(0, 1))(dense_attn, x)
        _, (grads_f, dx_f) = nnx.value_and_grad(loss_fn, argnums=(0, 1))(flash_attn, x)
        self.assertTrue(np.all(np.isfinite(np.array(dx_f))))
        _assert_allclose_to_scale(dx_f, dx_d)
        _assert_allclose_to_scale(grads_f.q_proj.kernel[...], grads_d.q_proj.kernel[...])

  def _check_flash_block_sparse_dispatch_parity(self, dt: Any, rng_seed: int, packed_only: bool | None = None) -> None:
    """Verifies `s > token_budget` block-sparse flash attention forward/backward parity against dense."""
    b, s_long, emb_dim, budget_short = 2, 48, 64, 16
    rng = np.random.default_rng(rng_seed)
    pos_long = jnp.tile(jnp.arange(s_long, dtype=jnp.int32), (b, 1))
    args_long = (emb_dim, 4, 2, 32, 0.25, 10000.0, (2, 1, 1), 2, 1, 32, budget_short, 4, 1e-6, dt, jnp.float32)
    dense_long = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
        *args_long, attention_kernel="dense", rngs=nnx.Rngs(23)
    )
    flash_long = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
        *args_long, indexer_mask_mode="reshape", attention_kernel="flash", rngs=nnx.Rngs(23)
    )
    nnx.update(flash_long, nnx.state(dense_long))

    x_long = jnp.array(rng.standard_normal((b, s_long, emb_dim), dtype=np.float32), dtype=dt)
    mask_long = jnp.ones((b, s_long), dtype=jnp.int32).at[0, 40:].set(0).at[1, 44:].set(0)
    seg_long = jnp.ones((b, s_long), dtype=jnp.int32).at[0, 18:36].set(2).at[0, 36:].set(0).at[1, 22:].set(2)
    all_cases = (
        ("unpacked", None, None),
        ("mask_2d", mask_long, None),
        ("packed_seg", None, seg_long),
        ("both_mask_and_seg", mask_long, seg_long),
    )
    cases = all_cases if packed_only is None else (all_cases[2:] if packed_only else all_cases[:2])

    for case_name, mask_in, seg_in in cases:
      with self.subTest(mode="block_sparse", dtype=str(dt), case_name=case_name):
        valid_l = jnp.ones((b, s_long), dtype=bool)
        if mask_in is not None:
          valid_l &= mask_in != 0
        if seg_in is not None:
          valid_l &= seg_in != 0
        out_d, blk_d = dense_long(
            x_long, position_ids=pos_long, attention_mask=mask_in, segment_ids=seg_in, return_selected_blocks=True
        )
        out_f, blk_f = flash_long(
            x_long, position_ids=pos_long, attention_mask=mask_in, segment_ids=seg_in, return_selected_blocks=True
        )
        np.testing.assert_array_equal(np.array(blk_f), np.array(blk_d))
        self.assertTrue(np.all(np.isfinite(np.array(out_f, dtype=np.float32))))
        if dt == jnp.float32:
          np.testing.assert_allclose(
              np.array(out_f)[np.array(valid_l)], np.array(out_d)[np.array(valid_l)], rtol=1e-4, atol=1e-4
          )
        else:
          _assert_allclose_to_scale(
              jnp.where(valid_l[:, :, None], out_f, 0.0),
              jnp.where(valid_l[:, :, None], out_d, 0.0),
              rtol=2e-2,
              atol_scale=2e-2,
          )

        def loss_long(
            mod: modeling_qwen3_8_flash_next.Qwen3_8SparseAttention,
            inp: jax.Array,
            mask_in: jax.Array | None = mask_in,
            seg_in: jax.Array | None = seg_in,
            valid_l: jax.Array = valid_l,
        ) -> jax.Array:
          out = mod(inp, position_ids=pos_long, attention_mask=mask_in, segment_ids=seg_in)
          assert isinstance(out, jax.Array)
          w_out = jnp.arange(1, emb_dim + 1, dtype=jnp.float32) / float(emb_dim)
          return jnp.sum(jnp.where(valid_l[:, :, None], out.astype(jnp.float32), 0.0) * w_out)

        _, (grads_d, dx_d) = nnx.value_and_grad(loss_long, argnums=(0, 1))(dense_long, x_long)
        _, (grads_f, dx_f) = nnx.value_and_grad(loss_long, argnums=(0, 1))(flash_long, x_long)
        self.assertTrue(np.all(np.isfinite(np.array(dx_f, dtype=np.float32))))
        tol = 1e-4 if dt == jnp.float32 else 2.5e-2
        _assert_allclose_to_scale(dx_f, dx_d, rtol=tol, atol_scale=tol)
        _assert_allclose_to_scale(grads_f.q_proj.kernel[...], grads_d.q_proj.kernel[...], rtol=tol, atol_scale=tol)
        _assert_allclose_to_scale(grads_f.k_proj.kernel[...], grads_d.k_proj.kernel[...], rtol=tol, atol_scale=tol)
        _assert_allclose_to_scale(grads_f.v_proj.kernel[...], grads_d.v_proj.kernel[...], rtol=tol, atol_scale=tol)

  def test_flash_block_sparse_attention_dispatch_parity_fp32(self):
    """Verifies `s > token_budget` block-sparse flash attention fp32 parity against dense."""
    self._check_flash_block_sparse_dispatch_parity(jnp.float32, rng_seed=29)

  def test_flash_block_sparse_attention_dispatch_parity_bf16_unpacked(self):
    """Verifies `s > token_budget` block-sparse flash attention bf16 unpacked and 2D-masked parity."""
    self._check_flash_block_sparse_dispatch_parity(jnp.bfloat16, rng_seed=31, packed_only=False)

  def test_flash_block_sparse_attention_dispatch_parity_bf16_packed(self):
    """Verifies `s > token_budget` block-sparse flash attention bf16 packed-segment parity."""
    self._check_flash_block_sparse_dispatch_parity(jnp.bfloat16, rng_seed=31, packed_only=True)

  def _check_qsa_nondivisible_or_multiword_case(
      self, *, s_case: int, budget_case: int, bq_case: int, bkv_case: int, rng_seed: int
  ) -> None:
    """Verifies block-sparse flash parity against dense and corrupt-index negative control."""
    rng = np.random.default_rng(rng_seed)
    b, emb_dim = 2, 64
    args_c = (emb_dim, 4, 2, 32, 0.25, 10000.0, (2, 1, 1), 2, 1, 32, budget_case, 4, 1e-6, jnp.float32, jnp.float32)
    dense_c = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(*args_c, attention_kernel="dense", rngs=nnx.Rngs(31))
    flash_c = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(*args_c, attention_kernel="flash", rngs=nnx.Rngs(31))
    nnx.update(flash_c, nnx.state(dense_c))
    x_c = jnp.array(rng.standard_normal((b, s_case, emb_dim), dtype=np.float32))
    pos_c = jnp.tile(jnp.arange(s_case, dtype=jnp.int32), (b, 1))
    out_d, blk_d = dense_c(x_c, position_ids=pos_c, return_selected_blocks=True)
    out_f, blk_f = flash_c(x_c, position_ids=pos_c, return_selected_blocks=True)
    np.testing.assert_array_equal(np.array(blk_f), np.array(blk_d))
    np.testing.assert_allclose(np.array(out_f), np.array(out_d), rtol=1e-4, atol=1e-4)

    cos_c, sin_c = flash_c.rotary_emb(x_c, pos_c)
    q_gate = flash_c.q_proj(x_c).reshape(b, s_case, 4, 64)
    q_st = flash_c.q_norm(q_gate[..., :32])
    k_st = flash_c.k_norm(flash_c.k_proj(x_c).reshape(b, s_case, 2, 32))
    v_st = flash_c.v_proj(x_c).reshape(b, s_case, 2, 32)
    q_st, k_st = modeling_qwen3_8_flash_next.apply_rotary_pos_emb(q_st, k_st, cos=cos_c, sin=sin_c, unsqueeze_dim=2)
    ctx_good = flash_c.flash_block_sparse_attention(q_st, k_st, v_st, blk_f, block_q=bq_case, block_kv=bkv_case)
    blk_corrupt = jnp.full_like(blk_f, -1)
    ctx_corrupt = flash_c.flash_block_sparse_attention(q_st, k_st, v_st, blk_corrupt, block_q=bq_case, block_kv=bkv_case)
    self.assertGreater(float(jnp.max(jnp.abs(ctx_good - ctx_corrupt))), 0.05)

  def test_qsa_flash_block_sparse_tail_packing_sharding_and_negative_controls(self):
    """Verifies 32-bit bitmap packing with duplicate/-1 slots and non-multiple-of-c tail block (`s=70`)."""
    rng = np.random.default_rng(41)
    b, s, topk, num_blocks, num_words = 2, 16, 12, 80, 3
    raw_idx = rng.integers(-2, num_blocks, size=(b, s, topk), dtype=np.int32)
    raw_idx[:, :, -2:] = raw_idx[:, :, :2]
    sel_idx = jnp.array(raw_idx, dtype=jnp.int32)
    packed = modeling_qwen3_8_flash_next.pack_qsa_block_mask_32bit(sel_idx, num_words)
    self.assertEqual(packed.shape, (b, s, num_words))
    self.assertEqual(packed.dtype, jnp.uint32)
    bits = ((packed[:, :, :, None] >> jnp.arange(32, dtype=jnp.uint32)) & jnp.uint32(1)) != 0
    unpacked_bool = bits.reshape(b, s, num_words * 32)[:, :, :num_blocks]
    m_local = jnp.arange(num_blocks, dtype=jnp.int32)[None, None, :]
    ref_bool = jnp.any((sel_idx[:, :, :, None] == m_local) & (sel_idx[:, :, :, None] >= 0), axis=2)
    np.testing.assert_array_equal(np.array(unpacked_bool), np.array(ref_bool))
    self._check_qsa_nondivisible_or_multiword_case(s_case=70, budget_case=20, bq_case=20, bkv_case=20, rng_seed=41)

  def test_qsa_flash_block_sparse_multiword_s256_and_corrupt_control(self):
    """Verifies multi-word `fast_unpacked` slice (`s=256, b_kv=128`) and corrupt-index negative control."""
    self._check_qsa_nondivisible_or_multiword_case(s_case=256, budget_case=64, bq_case=64, bkv_case=128, rng_seed=42)

  def test_qsa_flash_block_sparse_packed_2doc_isolation(self):
    """Verifies packed 2-doc output and gradients match concatenated per-doc runs and leak without `segment_ids`."""
    rng = np.random.default_rng(43)
    emb_dim, len1, len2 = 64, 28, 24
    args_p = (emb_dim, 4, 2, 32, 0.25, 10000.0, (2, 1, 1), 2, 1, 32, 16, 4, 1e-6, jnp.float32, jnp.float32)
    flash_p = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(*args_p, attention_kernel="flash", rngs=nnx.Rngs(53))
    x1 = jnp.array(rng.standard_normal((1, len1, emb_dim), dtype=np.float32))
    x2 = jnp.array(rng.standard_normal((1, len2, emb_dim), dtype=np.float32))
    x_pack = jnp.concatenate([x1, x2], axis=1)
    pos1 = jnp.arange(len1, dtype=jnp.int32)[None, :]
    pos2 = jnp.arange(len2, dtype=jnp.int32)[None, :]
    pos_pack = jnp.concatenate([pos1, pos2], axis=1)
    seg_pack = jnp.concatenate([jnp.ones((1, len1), dtype=jnp.int32), jnp.full((1, len2), 2, dtype=jnp.int32)], axis=1)
    w_probe = jnp.arange(1, emb_dim + 1, dtype=jnp.float32) / float(emb_dim)

    def packed_loss(
        mod: modeling_qwen3_8_flash_next.Qwen3_8SparseAttention, inp: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
      out = mod(inp, position_ids=pos_pack, segment_ids=seg_pack)
      assert isinstance(out, jax.Array)
      return jnp.sum(out * w_probe), out

    def unpacked_loss(
        mod: modeling_qwen3_8_flash_next.Qwen3_8SparseAttention, inp1: jax.Array, inp2: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
      out1 = mod(inp1, position_ids=pos1)
      out2 = mod(inp2, position_ids=pos2)
      assert isinstance(out1, jax.Array) and isinstance(out2, jax.Array)
      out_cat = jnp.concatenate([out1, out2], axis=1)
      return jnp.sum(out_cat * w_probe), out_cat

    (_, out_pack), (g_pack, dx_pack) = nnx.value_and_grad(packed_loss, argnums=(0, 1), has_aux=True)(flash_p, x_pack)
    (_, out_cat), (g_cat, dx1, dx2) = nnx.value_and_grad(unpacked_loss, argnums=(0, 1, 2), has_aux=True)(flash_p, x1, x2)
    dx_cat = jnp.concatenate([dx1, dx2], axis=1)
    np.testing.assert_allclose(np.array(out_pack), np.array(out_cat), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.array(dx_pack), np.array(dx_cat), rtol=1e-4, atol=1e-4)
    _assert_allclose_to_scale(g_pack.q_proj.kernel[...], g_cat.q_proj.kernel[...])

    out_leaked = flash_p(x_pack, position_ids=pos_pack, segment_ids=None)
    assert isinstance(out_leaked, jax.Array)
    self.assertGreater(float(jnp.max(jnp.abs(out_leaked[:, len1:, :] - out_cat[:, len1:, :]))), 0.05)

  def test_qsa_indexer_token_mask_free_parity_and_jaxpr(self):
    """Verifies `Qwen3_8QSAIndexer(return_token_mask=False)` bit-exact parity and zero `bool[B, S, S]` allocation."""
    rng = np.random.default_rng(45)
    emb_dim, b_ix, s_ix = 64, 2, 48
    args_p = (emb_dim, 4, 2, 32, 0.25, 10000.0, (2, 1, 1), 2, 1, 32, 16, 4, 1e-6, jnp.float32, jnp.float32)
    flash_p = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(*args_p, attention_kernel="flash", rngs=nnx.Rngs(53))
    x_ix = jnp.array(rng.standard_normal((b_ix, s_ix, emb_dim), dtype=np.float32))
    pos_ix = jnp.tile(jnp.arange(s_ix, dtype=jnp.int32), (b_ix, 1))
    cos_ix, sin_ix = flash_p.rotary_emb(x_ix, pos_ix)
    mask_ix = jnp.ones((b_ix, s_ix), dtype=jnp.int32).at[0, 38:].set(0).at[1, :6].set(0)
    seg_ix = jnp.ones((b_ix, s_ix), dtype=jnp.int32).at[:, 20:36].set(2).at[:, 36:].set(3)
    for m_in, s_in in ((None, None), (mask_ix, None), (None, seg_ix), (mask_ix, seg_ix)):
      tmask_t, blk_t = flash_p.indexer(
          x_ix, (cos_ix, sin_ix), attention_mask=m_in, segment_ids=s_in, return_token_mask=True
      )
      tmask_f, blk_f = flash_p.indexer(
          x_ix, (cos_ix, sin_ix), attention_mask=m_in, segment_ids=s_in, return_token_mask=False
      )
      self.assertIsNotNone(tmask_t)
      self.assertIsNone(tmask_f)
      np.testing.assert_array_equal(np.array(blk_f), np.array(blk_t))
    ix_jaxpr = str(
        jax.make_jaxpr(
            lambda x_i, m_i, s_i: flash_p.indexer(
                x_i, (cos_ix, sin_ix), attention_mask=m_i, segment_ids=s_i, return_token_mask=False
            )[1]
        )(x_ix, mask_ix, seg_ix)
    )
    self.assertNotIn("bool[2,48,48]", ix_jaxpr)

  def test_qsa_flash_block_sparse_outer_scan_remat_no_branch_residuals(self):
    """Verifies outer `lax.scan` + custom remat policies never stack 6D/7D branch residuals or dense masks."""
    rng = np.random.default_rng(47)
    emb_dim, b_rm, s_rm = 64, 2, 48
    pos_rm = jnp.tile(jnp.arange(s_rm, dtype=jnp.int32), (b_rm, 1))
    seg_rm = jnp.ones((b_rm, s_rm), dtype=jnp.int32).at[:, 24:].set(2)
    args_rm = (emb_dim, 4, 2, 32, 0.25, 10000.0, (2, 1, 1), 2, 1, 32, 16, 4, 1e-6, jnp.float32, jnp.float32)
    flash_rm = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(*args_rm, attention_kernel="flash", rngs=nnx.Rngs(73))
    gd_rm, params_rm, rest_rm = nnx.split(flash_rm, nnx.Param, ...)
    stacked_params = jax.tree.map(lambda p: jnp.stack([p, p], axis=0), params_rm)
    x_rm = jnp.array(rng.standard_normal((b_rm, s_rm, emb_dim), dtype=np.float32))

    for pol in (
        jax.checkpoint_policies.save_only_these_names("query_proj"),
        jax.checkpoint_policies.nothing_saveable,
        jax.checkpoint_policies.everything_saveable,
    ):

      def scanned_loss(
          h_in: jax.Array,
          sp_in: Any,
          pol_fn: Any = pol,
      ) -> jax.Array:
        def body_fn(carry: jax.Array, p_slice: Any) -> tuple[jax.Array, None]:
          @functools.partial(jax.checkpoint, policy=pol_fn, prevent_cse=True)
          def _step(c_in: jax.Array, ps: Any) -> jax.Array:
            mod = nnx.merge(gd_rm, ps, rest_rm)
            out = mod(c_in, position_ids=pos_rm, segment_ids=seg_rm)
            assert isinstance(out, jax.Array)
            return c_in + out

          return _step(carry, p_slice), None

        h_out, _ = jax.lax.scan(body_fn, h_in, sp_in)
        return jnp.sum(h_out)

      jaxpr_str = str(jax.make_jaxpr(jax.grad(scanned_loss, argnums=(0, 1)))(x_rm, stacked_params))
      for forbidden in (
          "f32[3,3,",
          "f32[4,4,",
          "f32[3,2,12,2,2,12]",
          "f32[4,2,16,2,2,16]",
          "f32[2,4,48,48]",
          "bool[2,48,48]",
      ):
        self.assertNotIn(forbidden, jaxpr_str)

  def test_qsa_flash_block_sparse_4device_sharded_parity(self):
    """Verifies 4-device CPU sharded parity (data=2, tensor=2) and TP head sharding for s > token_budget."""
    if len(jax.devices()) < 4:
      self.skipTest("Requires >= 4 JAX devices")
    rng = np.random.default_rng(61)
    emb_dim = 64
    w_probe = jnp.arange(1, emb_dim + 1, dtype=jnp.float32) / float(emb_dim)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("data", "tensor"))
    P = jax.sharding.PartitionSpec

    def ns(spec: jax.sharding.PartitionSpec) -> jax.sharding.NamedSharding:
      return jax.sharding.NamedSharding(mesh, spec)

    b_sh, s_sh = 2, 48
    args_sh = (emb_dim, 4, 2, 32, 0.25, 10000.0, (2, 1, 1), 2, 1, 32, 16, 4, 1e-6, jnp.float32, jnp.float32)
    dense_sh = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(*args_sh, attention_kernel="dense", rngs=nnx.Rngs(61))
    flash_sh = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(*args_sh, attention_kernel="flash", rngs=nnx.Rngs(61))
    nnx.update(flash_sh, nnx.state(dense_sh))
    x_sh = jax.device_put(
        jnp.array(rng.standard_normal((b_sh, s_sh, emb_dim), dtype=np.float32)), ns(P("data", None, None))
    )
    pos_sh = jax.device_put(jnp.tile(jnp.arange(s_sh, dtype=jnp.int32), (b_sh, 1)), ns(P("data", None)))
    seg_sh = jax.device_put(jnp.ones((b_sh, s_sh), dtype=jnp.int32).at[:, 24:].set(2), ns(P("data", None)))
    with jax.set_mesh(mesh):
      out_sh_d = dense_sh(x_sh, position_ids=pos_sh, segment_ids=seg_sh)
      out_sh_f = flash_sh(x_sh, position_ids=pos_sh, segment_ids=seg_sh)
      np.testing.assert_allclose(np.array(out_sh_f), np.array(out_sh_d), rtol=1e-4, atol=1e-4)

      def sh_loss(
          mod: modeling_qwen3_8_flash_next.Qwen3_8SparseAttention,
          inp: jax.Array,
      ) -> jax.Array:
        out = mod(inp, position_ids=pos_sh, segment_ids=seg_sh)
        assert isinstance(out, jax.Array)
        return jnp.sum(out * w_probe)

      sh_grad_fn = nnx.jit(nnx.value_and_grad(sh_loss, argnums=(0, 1)))
      _, (g_sh_d, dx_sh_d) = sh_grad_fn(dense_sh, x_sh)
      _, (g_sh_f, dx_sh_f) = sh_grad_fn(flash_sh, x_sh)
      np.testing.assert_allclose(np.array(dx_sh_f), np.array(dx_sh_d), rtol=1e-4, atol=1e-4)
      _assert_allclose_to_scale(g_sh_f.q_proj.kernel[...], g_sh_d.q_proj.kernel[...])

      # Verify TP head sharding (h_kv=2 -> 1 per TP rank on tensor=2) inside flash_block_sparse_attention HLO.
      q_sh = jax.device_put(jnp.ones((b_sh, s_sh, 4, 32), dtype=jnp.float32), ns(P("data", None, "tensor", None)))
      k_sh = jax.device_put(jnp.ones((b_sh, s_sh, 2, 32), dtype=jnp.float32), ns(P("data", None, "tensor", None)))
      v_sh = jax.device_put(jnp.ones((b_sh, s_sh, 2, 32), dtype=jnp.float32), ns(P("data", None, "tensor", None)))
      blk_sh = jax.device_put(jnp.zeros((b_sh, s_sh, 4), dtype=jnp.int32), ns(P("data", None, None)))
      hlo_tp = (
          jax.jit(
              lambda q_i, k_i, v_i, b_i: jnp.sum(
                  jax.grad(
                      lambda qq, kk, vv: jnp.sum(
                          flash_sh.flash_block_sparse_attention(qq, kk, vv, b_i, block_q=16, block_kv=16)
                      ),
                      argnums=(0, 1, 2),
                  )(q_i, k_i, v_i)[0]
              )
          )
          .lower(q_sh, k_sh, v_sh, blk_sh)
          .compile()
          .as_text()
          or ""
      )
      self.assertIn("f32[1,16,1,2,16]", hlo_tp)
      self.assertNotIn("f32[1,16,2,2,16]", hlo_tp)

  def test_qsa_flash_block_sparse_sequence_sharding_and_custom_strategy(self):
    """Verifies flash_block_sparse_attention preserves length_axes sharding and respects custom Sharding."""
    if len(jax.devices()) < 4:
      self.skipTest("Requires 4 devices.")
    rng = np.random.default_rng(1309)
    b, s, h_q, d = 2, 64, 4, 32
    q_np = rng.standard_normal((b, s, h_q, d), dtype=np.float32)
    k_np = rng.standard_normal((b, s, 2, d), dtype=np.float32)
    v_np = rng.standard_normal((b, s, 2, d), dtype=np.float32)
    blk_np = rng.integers(0, s // 4, size=(b, s, 4), dtype=np.int32)
    seg_np = np.ones((b, s), dtype=np.int32)
    seg_np[:, 40:] = 2
    w_out = jnp.array(rng.standard_normal((b, s, h_q * d), dtype=np.float32))

    mod_default = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
        emb_dim=64,
        num_query_heads=h_q,
        num_kv_heads=2,
        head_dim=d,
        mrope_section=(2, 1, 1),
        index_n_heads=2,
        index_kv_heads=1,
        index_head_dim=16,
        token_budget=16,
        compress_ratio=4,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        attention_kernel="flash",
        rngs=nnx.Rngs(params=0),
    )

    def unsharded_loss(q_i, k_i, v_i, seg_i):
      out = mod_default.flash_block_sparse_attention(
          q_i, k_i, v_i, jnp.asarray(blk_np), segment_ids=seg_i, block_q=16, block_kv=16
      )
      return jnp.sum(out * w_out)

    q_ref, k_ref, v_ref = jnp.asarray(q_np), jnp.asarray(k_np), jnp.asarray(v_np)
    seg_ref = jnp.asarray(seg_np)
    loss_ref, (dq_ref, dk_ref, dv_ref) = jax.jit(jax.value_and_grad(unsharded_loss, argnums=(0, 1, 2)))(
        q_ref, k_ref, v_ref, seg_ref
    )

    # 1. Context + Tensor (2, 2) mesh: query & packed bitmap stay sharded on `context` (0 query all-gathers)
    #    and also handles GQA head repetition when head_shards (2) > num_kv_heads (1).
    P = jax.sharding.PartitionSpec
    mesh_ct = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("context", "tensor"))

    def ns_ct(spec):
      return jax.sharding.NamedSharding(mesh_ct, spec)

    with jax.set_mesh(mesh_ct):
      q_ct = jax.device_put(q_ref, ns_ct(P(None, "context", "tensor", None)))
      k_ct = jax.device_put(k_ref, ns_ct(P(None, None, "tensor", None)))
      v_ct = jax.device_put(v_ref, ns_ct(P(None, None, "tensor", None)))
      blk_ct = jax.device_put(jnp.asarray(blk_np), ns_ct(P(None, "context", None)))
      seg_ct = jax.device_put(seg_ref, ns_ct(P(None, "context")))

      w_ct = jax.device_put(w_out, ns_ct(P(None, "context", None)))
      loss_ct, (dq_ct, dk_ct, dv_ct) = jax.jit(
          jax.value_and_grad(
              lambda qq, kk, vv: jnp.sum(
                  mod_default.flash_block_sparse_attention(
                      qq, kk, vv, blk_ct, segment_ids=seg_ct, block_q=16, block_kv=16
                  )
                  * w_ct
              ),
              argnums=(0, 1, 2),
          )
      )(q_ct, k_ct, v_ct)
      np.testing.assert_allclose(float(loss_ct), float(loss_ref), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(dq_ct), np.asarray(dq_ref), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(dk_ct), np.asarray(dk_ref), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(dv_ct), np.asarray(dv_ref), rtol=1e-4, atol=1e-4)

      # Unpacked forward HLO with pre-replicated KV has zero all-gathers (positive control: forcing None on Q emits one).
      fwd_hlo = (
          jax.jit(
              lambda qq, kk, vv, bb: mod_default.flash_block_sparse_attention(qq, kk, vv, bb, block_q=16, block_kv=16)
          )
          .lower(q_ct, k_ct, v_ct, blk_ct)
          .compile()
          .as_text()
          or ""
      )
      pos_hlo = (
          jax.jit(lambda qq: jnp.sum(jax.lax.with_sharding_constraint(qq, P(None, None, "tensor", None))))
          .lower(q_ct)
          .compile()
          .as_text()
          or ""
      )
      self.assertIn("all-gather(", pos_hlo)
      self.assertNotIn("all-gather(", fwd_hlo)
      # Each context shard only holds 2 local Q tiles (32 tokens = 2 * 16), never all 4 Q tiles.
      self.assertIn("f32[2,2,16,1,2,32]", fwd_hlo)
      self.assertNotIn("f32[4,2,16,1,2,32]", fwd_hlo)

      # Eager call on SingleDeviceSharding inputs inside `with jax.set_mesh` also succeeds cleanly.
      out_eager = mod_default.flash_block_sparse_attention(
          q_ref, k_ref, v_ref, jnp.asarray(blk_np), block_q=16, block_kv=16
      )
      self.assertEqual(out_eager.shape, (b, s, h_q * d))

    # GQA head repetition when head_shards (2) > num_kv_heads (1), compared against unsharded reference.
    k1_ref, v1_ref = k_ref[:, :, :1, :], v_ref[:, :, :1, :]
    l1_ref, (dq1_ref, dk1_ref, dv1_ref) = jax.jit(
        jax.value_and_grad(
            lambda qq, kk, vv: jnp.sum(
                mod_default.flash_block_sparse_attention(qq, kk, vv, jnp.asarray(blk_np), block_q=16, block_kv=16) * w_out
            ),
            argnums=(0, 1, 2),
        )
    )(q_ref, k1_ref, v1_ref)
    with jax.set_mesh(mesh_ct):
      k1_ct = jax.device_put(k1_ref, ns_ct(P(None, None, None, None)))
      v1_ct = jax.device_put(v1_ref, ns_ct(P(None, None, None, None)))
      l1_ct, (dq1_ct, dk1_ct, dv1_ct) = jax.jit(
          jax.value_and_grad(
              lambda qq, kk, vv: jnp.sum(
                  mod_default.flash_block_sparse_attention(qq, kk, vv, blk_ct, block_q=16, block_kv=16) * w_ct
              ),
              argnums=(0, 1, 2),
          )
      )(q_ct, k1_ct, v1_ct)
      np.testing.assert_allclose(float(l1_ct), float(l1_ref), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(dq1_ct), np.asarray(dq1_ref), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(dk1_ct), np.asarray(dk1_ref), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(dv1_ct), np.asarray(dv1_ref), rtol=1e-4, atol=1e-4)

    # 2. Custom EP=CP Sharding (`batch -> fsdp`, `length -> expert`) + `tensor_sequence` head sharding.
    class _EpCpSharding(modeling_qwen3_8_flash_next.DefaultSharding):
      """Custom sharding mapping activation batch off `expert` and length onto `expert`."""

      def map_axis(self, axis, tensor_name, tensor_type):
        if tensor_type == modeling_qwen3_8_flash_next.TensorType.ACTIVATION:
          if axis == "batch":
            return ("data", "fsdp", "fsdp_transpose")
          if axis == "length":
            return ("context", "expert")
        return super().map_axis(axis, tensor_name, tensor_type)

    ep_cp = _EpCpSharding()
    mod_epcp = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
        emb_dim=64,
        num_query_heads=h_q,
        num_kv_heads=2,
        head_dim=d,
        mrope_section=(2, 1, 1),
        index_n_heads=2,
        index_kv_heads=1,
        index_head_dim=16,
        token_budget=16,
        compress_ratio=4,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        sharding=ep_cp,
        attention_kernel="flash",
        rngs=nnx.Rngs(params=0),
    )
    mesh_ep = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))

    def ns_ep(spec):
      return jax.sharding.NamedSharding(mesh_ep, spec)

    with jax.set_mesh(mesh_ep):
      q_ep = jax.device_put(q_ref, ns_ep(P("fsdp", "expert", None, None)))
      k_ep = jax.device_put(k_ref, ns_ep(P("fsdp", None, None, None)))
      v_ep = jax.device_put(v_ref, ns_ep(P("fsdp", None, None, None)))
      blk_ep = jax.device_put(jnp.asarray(blk_np), ns_ep(P("fsdp", "expert", None)))
      w_ep = jax.device_put(w_out, ns_ep(P("fsdp", "expert", None)))
      l_ep, (dq_ep, dk_ep, dv_ep) = jax.jit(
          jax.value_and_grad(
              lambda qq, kk, vv: jnp.sum(
                  mod_epcp.flash_block_sparse_attention(qq, kk, vv, blk_ep, block_q=16, block_kv=16) * w_ep
              ),
              argnums=(0, 1, 2),
          )
      )(q_ep, k_ep, v_ep)
    loss_unpacked_ref, (dq_u_ref, dk_u_ref, dv_u_ref) = jax.jit(
        jax.value_and_grad(
            lambda qq, kk, vv: jnp.sum(
                mod_epcp.flash_block_sparse_attention(qq, kk, vv, jnp.asarray(blk_np), block_q=16, block_kv=16) * w_out
            ),
            argnums=(0, 1, 2),
        )
    )(q_ref, k_ref, v_ref)
    with jax.set_mesh(mesh_ep):
      np.testing.assert_allclose(float(l_ep), float(loss_unpacked_ref), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(dq_ep), np.asarray(dq_u_ref), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(dk_ep), np.asarray(dk_u_ref), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(dv_ep), np.asarray(dv_u_ref), rtol=1e-4, atol=1e-4)
      ep_hlo = (
          jax.jit(lambda qq, kk, vv, bb: mod_epcp.flash_block_sparse_attention(qq, kk, vv, bb, block_q=16, block_kv=16))
          .lower(q_ep, k_ep, v_ep, blk_ep)
          .compile()
          .as_text()
          or ""
      )
      self.assertNotIn("all-gather(", ep_hlo)
      self.assertIn("f32[2,1,16,2,2,32]", ep_hlo)

    # 3. `tensor_sequence` head sharding via `get_heads_mesh_axes`.
    mesh_tsp = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "tensor_sequence"))

    def ns_tsp(spec):
      return jax.sharding.NamedSharding(mesh_tsp, spec)

    with jax.set_mesh(mesh_tsp):
      q_tsp = jax.device_put(q_ref, ns_tsp(P("fsdp", None, "tensor_sequence", None)))
      k_tsp = jax.device_put(k_ref, ns_tsp(P("fsdp", None, "tensor_sequence", None)))
      v_tsp = jax.device_put(v_ref, ns_tsp(P("fsdp", None, "tensor_sequence", None)))
      blk_tsp = jax.device_put(jnp.asarray(blk_np), ns_tsp(P("fsdp", None, None)))
      tsp_hlo = (
          jax.jit(
              lambda qq, kk, vv, bb: mod_default.flash_block_sparse_attention(qq, kk, vv, bb, block_q=16, block_kv=16)
          )
          .lower(q_tsp, k_tsp, v_tsp, blk_tsp)
          .compile()
          .as_text()
          or ""
      )
      self.assertIn("f32[4,1,16,1,2,32]", tsp_hlo)
      self.assertNotIn("f32[4,1,16,2,2,32]", tsp_hlo)

  @unittest.skipIf(
      importlib.util.find_spec("libtpu") is None or os.getenv("JAX_PLATFORMS") == "proxy",
      "requires libtpu virtual TPU topology support",
  )
  def test_qsa_flash_attention_virtual_tpu7x_tpu_custom_call(self):
    """Verifies virtual TPU7x _flash_attention lowers to tpu_custom_call when batch_axes == ()."""
    topo_mod = importlib.import_module("jax.experimental.topologies")
    tpu_topo = topo_mod.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1")
    emb_dim = 64
    attn_short = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
        emb_dim=emb_dim,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=128,
        mrope_section=(8, 4, 4),
        index_n_heads=2,
        index_kv_heads=1,
        index_head_dim=32,
        token_budget=256,
        compress_ratio=4,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float32,
        attention_kernel="flash",
        rngs=nnx.Rngs(params=13),
    )
    mesh_tpu_ct = jax.sharding.Mesh(
        np.array(tpu_topo.devices).reshape(4, 2),
        ("context", "tensor"),
    )
    P = jax.sharding.PartitionSpec
    gd_s, p_s, r_s = nnx.split(attn_short, nnx.Param, ...)
    p_abs = jax.tree.map(
        lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=jax.sharding.NamedSharding(mesh_tpu_ct, P())),
        p_s,
    )
    x_abs = jax.ShapeDtypeStruct(
        (2, 256, emb_dim),
        jnp.bfloat16,
        sharding=jax.sharding.NamedSharding(mesh_tpu_ct, P(None, "context", None)),
    )
    with jax.set_mesh(mesh_tpu_ct):
      splash_tpu_hlo = jax.jit(lambda pp, xi: nnx.merge(gd_s, pp, r_s)(xi)).lower(p_abs, x_abs).compile().as_text() or ""
    self.assertIn("tpu_custom_call", splash_tpu_hlo)

  def test_qsa_pack_32bit_scatter_free(self):
    """Verifies scatter-free and sort-free 32-bit QSA block-mask packing across <=16 and >16 words."""
    rng = np.random.default_rng(1709)
    for num_words in (3, 20):
      max_blk = num_words * 32
      raw = rng.integers(-4, max_blk + 8, size=(2, 12, 64), dtype=np.int32)
      raw[0, 0, :32] = 31
      raw[0, 0, 32:] = max_blk - 1
      raw[1, 0, :] = -1
      sel = jnp.asarray(raw, dtype=jnp.int32)
      pack_jit = jax.jit(lambda x, nw=num_words: modeling_qwen3_8_flash_next.pack_qsa_block_mask_32bit(x, nw))
      packed = pack_jit(sel)
      self.assertEqual(packed.shape, (2, 12, num_words))
      self.assertEqual(packed.dtype, jnp.uint32)
      bits = ((packed[:, :, :, None] >> jnp.arange(32, dtype=jnp.uint32)) & jnp.uint32(1)) != 0
      unpacked_bits = bits.reshape(2, 12, max_blk)
      m_grid = jnp.arange(max_blk, dtype=jnp.int32)[None, None, :]
      ref_bits = jnp.any((sel[:, :, :, None] == m_grid) & (sel[:, :, :, None] >= 0), axis=2)
      np.testing.assert_array_equal(np.asarray(unpacked_bits), np.asarray(ref_bits))

      pack_hlo = pack_jit.lower(sel).compile().as_text() or ""
      self.assertNotIn("scatter(", pack_hlo)
      self.assertNotIn("sort(", pack_hlo)

  _qsa_remat_ref_cache: dict[tuple[float, bool], tuple[Any, float, Any]] = {}

  def _check_qsa_saved_bitmap_remat(self, pol_name: str, aux_coeff: float, packed_segments: bool) -> None:
    """Verifies scanned Qwen3.8 remat loss/grad parity, saved bitmap tags, and zero backward indexer recompute."""
    rng = np.random.default_rng(1713)
    toks_r = jnp.asarray(rng.integers(1, 64, size=(2, 48), dtype=np.int32))
    tgts_r = jnp.asarray(rng.integers(1, 64, size=(2, 48), dtype=np.int32))
    seg_arg = jnp.ones((2, 48), dtype=jnp.int32).at[:, 24:].set(2) if packed_segments else None

    cache_key = (aux_coeff, packed_segments)
    cached = type(self)._qsa_remat_ref_cache.get(cache_key)
    if cached is None:
      cfg_none = _get_mini_config(
          scan_layers=True,
          base_num_decoder_layers=4,
          use_ple_layer=False,
          qsa_attention_kernel="flash",
          index_select_budget=16,
          remat_policy="none",
      )
      m_none = m3_models.create_model(cfg_none, mesh=None)
      ref_state = nnx.state(m_none)
      gd_n, p_n, rest_n = nnx.split(m_none, nnx.Param, ...)

      def _ref_loss(params: Any) -> jax.Array:
        loss_val, _ = modeling_qwen3_8_flash_next.compute_total_loss(
            nnx.merge(gd_n, params, rest_n),
            toks_r,
            tgts_r,
            segment_ids=seg_arg,
            router_aux_coeff=aux_coeff,
            use_mtp=False,
        )
        return loss_val

      l_ref_arr, g_ref = jax.jit(jax.value_and_grad(_ref_loss))(p_n)
      cached = (ref_state, float(l_ref_arr), g_ref)
      type(self)._qsa_remat_ref_cache[cache_key] = cached

    ref_state, l_ref, g_ref = cached
    cfg_pol = _get_mini_config(
        scan_layers=True,
        base_num_decoder_layers=4,
        use_ple_layer=False,
        qsa_attention_kernel="flash",
        index_select_budget=16,
        remat_policy=pol_name,
    )
    m_pol = m3_models.create_model(cfg_pol, mesh=None)
    nnx.update(m_pol, ref_state)
    gd_p, p_p, rest_p = nnx.split(m_pol, nnx.Param, ...)

    def _pol_loss(params: Any) -> jax.Array:
      loss_val, _ = modeling_qwen3_8_flash_next.compute_total_loss(
          nnx.merge(gd_p, params, rest_p),
          toks_r,
          tgts_r,
          segment_ids=seg_arg,
          router_aux_coeff=aux_coeff,
          use_mtp=False,
      )
      return loss_val

    pol_vag = jax.value_and_grad(_pol_loss)
    l_pol, g_pol = jax.jit(pol_vag)(p_p)
    np.testing.assert_allclose(float(l_pol), float(l_ref), rtol=1e-5, atol=1e-5)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4), g_pol, g_ref)

    def _all_jaxpr_text(jpr: Any) -> str:
      parts = [str(jpr)]
      for eq in jpr.eqns:
        for val in eq.params.values():
          if hasattr(val, "jaxpr"):
            parts.append(_all_jaxpr_text(val.jaxpr))
          elif isinstance(val, (list, tuple)):
            for item in val:
              if hasattr(item, "jaxpr"):
                parts.append(_all_jaxpr_text(item.jaxpr))
      return "\n".join(parts)

    fwd_txt = _all_jaxpr_text(jax.make_jaxpr(_pol_loss)(p_p).jaxpr)
    bwd_txt = _all_jaxpr_text(jax.make_jaxpr(pol_vag)(p_p).jaxpr)
    self.assertIn("qsa_packed_mask", fwd_txt)
    if packed_segments:
      self.assertIn("qsa_local_pos", fwd_txt)
    # Positive control: rematerialized sublayers + VJP run > 2x forward dot_general ops,
    # whereas detached QSA index_qk_proj ([2, 48, 96]) is NEVER recomputed or differentiated in backward.
    self.assertGreater(bwd_txt.count("dot_general["), 2 * fwd_txt.count("dot_general["))
    self.assertLessEqual(bwd_txt.count("[2,48,96]"), fwd_txt.count("[2,48,96]"))

  # Explicit per-case methods distribute across shards without adding new parameterized dep under Bazel strict_deps.
  def test_qsa_saved_bitmap_remat_custom_aux_unpacked(self):
    """Verifies `remat_policy='custom'` with router aux loss on unpacked sequences."""
    self._check_qsa_saved_bitmap_remat("custom", 0.01, False)

  def test_qsa_saved_bitmap_remat_custom_aux_packed(self):
    """Verifies `remat_policy='custom'` with router aux loss on multi-segment packed sequences."""
    self._check_qsa_saved_bitmap_remat("custom", 0.01, True)

  def test_qsa_saved_bitmap_remat_custom_noaux_unpacked(self):
    """Verifies `remat_policy='custom'` without router aux loss on unpacked sequences."""
    self._check_qsa_saved_bitmap_remat("custom", 0.0, False)

  def test_qsa_saved_bitmap_remat_custom_noaux_packed(self):
    """Verifies `remat_policy='custom'` without router aux loss on multi-segment packed sequences."""
    self._check_qsa_saved_bitmap_remat("custom", 0.0, True)

  def test_qsa_saved_bitmap_remat_full_aux_unpacked(self):
    """Verifies `remat_policy='full'` with router aux loss on unpacked sequences."""
    self._check_qsa_saved_bitmap_remat("full", 0.01, False)

  def test_qsa_saved_bitmap_remat_full_aux_packed(self):
    """Verifies `remat_policy='full'` with router aux loss on multi-segment packed sequences."""
    self._check_qsa_saved_bitmap_remat("full", 0.01, True)

  def test_qsa_saved_bitmap_remat_full_noaux_unpacked(self):
    """Verifies `remat_policy='full'` without router aux loss on unpacked sequences."""
    self._check_qsa_saved_bitmap_remat("full", 0.0, False)

  def test_qsa_saved_bitmap_remat_full_noaux_packed(self):
    """Verifies `remat_policy='full'` without router aux loss on multi-segment packed sequences."""
    self._check_qsa_saved_bitmap_remat("full", 0.0, True)

  def test_qsa_epcp_sharded_kv_and_context_tensor_splash_4dev(self):
    """Verifies 4-device ('context', 'tensor') Splash batch_axes==() and ('fsdp', 'expert') EP=CP KV parity."""
    if len(jax.devices()) < 4:
      self.skipTest("Requires >= 4 devices (set XLA_FLAGS=--xla_force_host_platform_device_count=4).")
    rng = np.random.default_rng(1719)
    emb_dim = 64

    # 1. _flash_attention (s <= token_budget) on 4-device ('context', 'tensor') where batch_axes == ().
    attn_short = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
        emb_dim=emb_dim,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=128,
        mrope_section=(8, 4, 4),
        index_n_heads=2,
        index_kv_heads=1,
        index_head_dim=32,
        token_budget=256,
        compress_ratio=4,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.float32,
        attention_kernel="flash",
        rngs=nnx.Rngs(params=13),
    )
    mesh_ct = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("context", "tensor"))
    x_ct = jnp.ones((1, 128, emb_dim), dtype=jnp.bfloat16)
    with jax.set_mesh(mesh_ct):
      out_ct = nnx.jit(lambda m, xi: m(xi))(attn_short, x_ct)
      self.assertEqual(out_ct.shape, (1, 128, emb_dim))

    # 2. Full Qwen3_8SparseAttention.__call__ on 4-device ('fsdp', 'expert') EP=CP mesh (kv_sharded_on_length=True).
    class _EpCpFullSharding(modeling_qwen3_8_flash_next.DefaultSharding):
      """EP=CP sharding mapping activation batch to fsdp and length to expert."""

      def map_axis(self, axis, tensor_name, tensor_type):
        if tensor_type == modeling_qwen3_8_flash_next.TensorType.ACTIVATION:
          if axis == "batch":
            return ("data", "fsdp", "fsdp_transpose")
          if axis == "length":
            return ("context", "expert")
        return super().map_axis(axis, tensor_name, tensor_type)

    ep_shd = _EpCpFullSharding()
    b_ep, s_ep = 2, 64
    args_ep = (emb_dim, 4, 2, 32, 0.25, 10000.0, (2, 1, 1), 2, 1, 32, 16, 4, 1e-6, jnp.float32, jnp.float32)
    dense_ep = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
        *args_ep, sharding=ep_shd, attention_kernel="dense", rngs=nnx.Rngs(91)
    )
    flash_ep = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
        *args_ep, sharding=ep_shd, attention_kernel="flash", rngs=nnx.Rngs(91)
    )
    nnx.update(flash_ep, nnx.state(dense_ep))
    x_ep_np = rng.standard_normal((b_ep, s_ep, emb_dim), dtype=np.float32)
    w_ep_probe = jnp.arange(1, emb_dim + 1, dtype=jnp.float32) / float(emb_dim)
    pos_ep0 = jnp.tile(jnp.arange(s_ep, dtype=jnp.int32), (b_ep, 1))
    seg_ep0 = jnp.ones((b_ep, s_ep), dtype=jnp.int32).at[:, 36:].set(2)

    l_d_u, (g_d_u, dx_d_u) = nnx.jit(
        nnx.value_and_grad(lambda m, xi: jnp.sum(m(xi, position_ids=pos_ep0) * w_ep_probe), argnums=(0, 1))
    )(dense_ep, jnp.asarray(x_ep_np))
    l_d_p, (g_d_p, dx_d_p) = nnx.jit(
        nnx.value_and_grad(
            lambda m, xi: jnp.sum(m(xi, position_ids=pos_ep0, segment_ids=seg_ep0) * w_ep_probe), argnums=(0, 1)
        )
    )(dense_ep, jnp.asarray(x_ep_np))

    mesh_ep4 = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))
    P = jax.sharding.PartitionSpec

    def _ns_ep4(sp):
      return jax.sharding.NamedSharding(mesh_ep4, sp)

    with jax.set_mesh(mesh_ep4):
      x_ep_sh = jax.device_put(jnp.asarray(x_ep_np), _ns_ep4(P("fsdp", "expert", None)))
      pos_ep_sh = jax.device_put(pos_ep0, _ns_ep4(P("fsdp", "expert")))
      seg_ep_sh = jax.device_put(seg_ep0, _ns_ep4(P("fsdp", "expert")))
      l_f_u, (g_f_u, dx_f_u) = nnx.jit(
          nnx.value_and_grad(lambda m, xi: jnp.sum(m(xi, position_ids=pos_ep_sh) * w_ep_probe), argnums=(0, 1))
      )(flash_ep, x_ep_sh)
      l_f_p, (g_f_p, dx_f_p) = nnx.jit(
          nnx.value_and_grad(
              lambda m, xi: jnp.sum(m(xi, position_ids=pos_ep_sh, segment_ids=seg_ep_sh) * w_ep_probe),
              argnums=(0, 1),
          )
      )(flash_ep, x_ep_sh)
      np.testing.assert_allclose(float(l_f_u), float(l_d_u), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(float(l_f_p), float(l_d_p), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(dx_f_u), np.asarray(dx_d_u), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(dx_f_p), np.asarray(dx_d_p), rtol=1e-4, atol=1e-4)
      _assert_allclose_to_scale(g_f_u.k_proj.kernel[...], g_d_u.k_proj.kernel[...])
      _assert_allclose_to_scale(g_f_u.v_proj.kernel[...], g_d_u.v_proj.kernel[...])
      _assert_allclose_to_scale(g_f_p.k_proj.kernel[...], g_d_p.k_proj.kernel[...])
      _assert_allclose_to_scale(g_f_p.v_proj.kernel[...], g_d_p.v_proj.kernel[...])

      # Unpacked indexer under EP=CP emits zero cross-expert all-reduce ops.
      cos_ep, sin_ep = flash_ep.rotary_emb(x_ep_sh, pos_ep_sh)
      gd_ix, p_ix, r_ix = nnx.split(flash_ep.indexer, nnx.Param, ...)
      p_ix_sh = jax.tree.map(lambda a: jax.device_put(a, _ns_ep4(P())), p_ix)
      ix_ep_hlo = (
          jax.jit(lambda pp, xi, ci, si: nnx.merge(gd_ix, pp, r_ix)(xi, (ci, si), return_token_mask=False)[1])
          .lower(p_ix_sh, x_ep_sh, cos_ep, sin_ep)
          .compile()
          .as_text()
          or ""
      )
      self.assertEqual(len(re.findall(r"=\s*[^=]*\ball-reduce(?:-start)?\(", ix_ep_hlo)), 0)

  def test_qsa_tiled_indexer_bounds_and_unpacked_chunk_parity(self):
    """Verifies `tiled_qsa_top_k` argument validation and bit-exact unpacked chunk parity across fp32/bf16."""
    rng = np.random.default_rng(1409)
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(block_chunk_size=0, rngs=nnx.Rngs(0))
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(block_chunk_size=-4, rngs=nnx.Rngs(0))

    emb_dim, h_idx, d_idx, c = 64, 4, 32, 4
    rope = modeling_qwen3_8_flash_next.Qwen3_8RotaryEmbedding(
        head_dim=64, partial_rotary_factor=0.5, rope_theta=10000.0, mrope_section=(6, 5, 5)
    )

    q_v = jnp.ones((1, 16, h_idx, d_idx), dtype=jnp.float32)
    bk_v = jnp.ones((1, 4, d_idx), dtype=jnp.float32)
    bv_v = jnp.ones((1, 4), dtype=bool)
    bseg_v = jnp.ones((1, 4), dtype=jnp.int32)
    gs_v = jnp.arange(4, dtype=jnp.int32)[None, :] * c
    seg_v = jnp.ones((1, 16), dtype=jnp.int32)
    for bad_kw, err_pat in (
        ({"k_top": 2, "block_chunk_size": 0, "index_head_dim": d_idx}, "block_chunk_size > 0"),
        ({"k_top": -1, "block_chunk_size": 2, "index_head_dim": d_idx}, "1 <= k_top <= num_blocks"),
        ({"k_top": 0, "block_chunk_size": 2, "index_head_dim": d_idx}, "1 <= k_top <= num_blocks"),
        ({"k_top": 5, "block_chunk_size": 2, "index_head_dim": d_idx}, "1 <= k_top <= num_blocks"),
        ({"k_top": 2, "block_chunk_size": 2, "index_head_dim": d_idx + 4}, "index_head_dim"),
    ):
      with self.assertRaisesRegex(ValueError, err_pat):
        modeling_qwen3_8_flash_next.tiled_qsa_top_k(q_v, bk_v, bv_v, bseg_v, gs_v, seg_v, compress_ratio=c, **bad_kw)

    for dt in (jnp.float32, jnp.bfloat16):
      idx = modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(
          emb_dim=emb_dim,
          index_n_heads=h_idx,
          index_kv_heads=1,
          index_head_dim=d_idx,
          token_budget=16,
          compress_ratio=c,
          dtype=dt,
          weight_dtype=dt,
          rngs=nnx.Rngs(11),
      )
      with self.assertRaises(ValueError):
        x0 = jnp.zeros((1, 8, emb_dim), dtype=dt)
        cos0, sin0 = rope(x0, jnp.arange(8, dtype=jnp.int32)[None, :])
        idx(x0, (cos0, sin0), return_token_mask=False, block_chunk_size=0)

      b, s = 2, 80  # M = 20 != d_idx // 2 (16), breaking RoPE half-dim symmetry.
      x = jnp.array(rng.standard_normal((b, s, emb_dim)), dtype=dt)
      pos = jnp.tile(jnp.arange(s, dtype=jnp.int32), (b, 1))
      cos, sin = rope(x, pos)
      _, blk_dense = idx(x, (cos, sin), return_token_mask=True)
      for mc in (None, 1, 3, 4, 7):
        tmask_f, blk_tiled = jax.jit(
            lambda xi, ci, si, mod_i=idx, mc_i=mc: mod_i(xi, (ci, si), return_token_mask=False, block_chunk_size=mc_i)
        )(x, cos, sin)
        self.assertIsNone(tmask_f)
        np.testing.assert_array_equal(np.asarray(blk_tiled), np.asarray(blk_dense))

  def test_qsa_tiled_indexer_packed_padding_all_masked_and_ties(self):
    """Verifies packed multi-doc padding, all-masked finite gradients, and exact tie-breaking."""
    rng = np.random.default_rng(1411)
    emb_dim, h_idx, d_idx, c = 64, 4, 32, 4
    rope = modeling_qwen3_8_flash_next.Qwen3_8RotaryEmbedding(
        head_dim=64, partial_rotary_factor=0.5, rope_theta=10000.0, mrope_section=(6, 5, 5)
    )
    idx_f32 = modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(
        emb_dim=emb_dim,
        index_n_heads=h_idx,
        index_kv_heads=1,
        index_head_dim=d_idx,
        token_budget=16,
        compress_ratio=c,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(17),
    )
    b, s = 2, 80
    x = jnp.array(rng.standard_normal((b, s, emb_dim)), dtype=jnp.float32)
    pos = jnp.tile(jnp.arange(s, dtype=jnp.int32), (b, 1))
    cos, sin = rope(x, pos)
    seg = jnp.ones((b, s), dtype=jnp.int32)
    seg = seg.at[0, 18:38].set(2).at[0, 38:44].set(0).at[0, 44:].set(3).at[1, :5].set(0).at[1, 31:].set(2)
    amask = jnp.ones((b, s), dtype=jnp.int32).at[0, 72:].set(0).at[1, 27:31].set(0)
    for m_in, s_in in ((amask, None), (None, seg), (amask, seg)):
      _, blk_d = idx_f32(x, (cos, sin), attention_mask=m_in, segment_ids=s_in, return_token_mask=True)
      _, blk_t = idx_f32(
          x, (cos, sin), attention_mask=m_in, segment_ids=s_in, return_token_mask=False, block_chunk_size=3
      )
      np.testing.assert_array_equal(np.asarray(blk_t), np.asarray(blk_d))

    attn_mask_check = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
        emb_dim,
        4,
        2,
        32,
        0.25,
        10000.0,
        (2, 1, 1),
        2,
        1,
        32,
        16,
        4,
        1e-6,
        jnp.float32,
        jnp.float32,
        attention_kernel="flash",
        rngs=nnx.Rngs(83),
    )
    attn_mask_check.indexer.block_chunk_size = 2
    x_m = jnp.array(rng.standard_normal((2, 48, emb_dim)), dtype=jnp.float32)
    pos_m = jnp.tile(jnp.arange(48, dtype=jnp.int32), (2, 1))
    for am_case in (
        jnp.ones((2, 48), dtype=jnp.int32).at[0, :].set(0),
        jnp.zeros((2, 48), dtype=jnp.int32),
    ):
      (_, sel_m), (g_m, dx_m) = nnx.jit(
          nnx.value_and_grad(
              lambda mod, xi, am=am_case: (
                  jnp.sum(mod(xi, position_ids=pos_m, attention_mask=am, return_selected_blocks=True)[0]),
                  mod(xi, position_ids=pos_m, attention_mask=am, return_selected_blocks=True)[1],
              ),
              argnums=(0, 1),
              has_aux=True,
          )
      )(attn_mask_check, x_m)
      np.testing.assert_array_equal(np.asarray(sel_m[0]), -1)
      self.assertTrue(np.all(np.isfinite(np.asarray(dx_m))))
      for leaf in jax.tree_util.tree_leaves(nnx.state(g_m)):
        self.assertTrue(np.all(np.isfinite(np.asarray(leaf))))

    x_zero = jnp.zeros((b, s, emb_dim), dtype=jnp.float32)
    _, blk_zero_d = idx_f32(x_zero, (cos, sin), segment_ids=seg, return_token_mask=True)
    _, blk_zero_t = idx_f32(x_zero, (cos, sin), segment_ids=seg, return_token_mask=False, block_chunk_size=3)
    np.testing.assert_array_equal(np.asarray(blk_zero_t), np.asarray(blk_zero_d))

    m_tie = s // c
    q_tie = jnp.ones((b, s, h_idx, d_idx), dtype=jnp.float32)
    bk_neg = -jnp.ones((b, m_tie, d_idx), dtype=jnp.float32)
    bv_all = jnp.ones((b, m_tie), dtype=bool)
    gs_all = jnp.broadcast_to(jnp.arange(m_tie, dtype=jnp.int32)[None, :] * c, (b, m_tie))
    seg_all = jnp.ones((b, s), dtype=jnp.int32)
    bseg_all = jnp.ones((b, m_tie), dtype=jnp.int32)
    bk_dyadic = jnp.broadcast_to(
        jnp.array([0.5, 1.25], dtype=jnp.float32)[None, None, :, None], (b, m_tie // 2, 2, d_idx)
    ).reshape(b, m_tie, d_idx)
    bk_dup = jnp.broadcast_to(
        jnp.array(rng.standard_normal((b, 1, 2, d_idx)), dtype=jnp.float32), (b, m_tie // 2, 2, d_idx)
    ).reshape(b, m_tie, d_idx)
    for exact_vals, bk_case in ((True, bk_neg), (True, bk_dyadic), (False, bk_dup)):
      raw_ref = jnp.einsum("bthd,bmd->bthm", q_tie, bk_case)
      sc_ref = jnp.sum(jax.nn.relu(raw_ref), axis=2) / jnp.sqrt(float(d_idx))
      valid_ref = (gs_all + c - 1)[:, None, :] <= jnp.arange(s)[None, :, None]
      dv, di = jax.lax.top_k(jnp.where(valid_ref, sc_ref, -1e9), 5)
      tv, ti = modeling_qwen3_8_flash_next.tiled_qsa_top_k(
          q_tie,
          bk_case,
          bv_all,
          bseg_all,
          gs_all,
          seg_all,
          compress_ratio=c,
          index_head_dim=d_idx,
          k_top=5,
          block_chunk_size=3,
      )
      if exact_vals:
        np.testing.assert_array_equal(np.asarray(tv), np.asarray(dv))
      else:
        np.testing.assert_allclose(np.asarray(tv), np.asarray(dv), rtol=1e-6, atol=1e-6)
      np.testing.assert_array_equal(np.asarray(ti), np.asarray(di))

  def test_qsa_tiled_indexer_short_seqs_hlo_and_tail_boundary(self):
    """Verifies short/non-divisible sequences, dense-score-free HLO, and tail-chunk boundary negative control."""
    rng = np.random.default_rng(1413)
    b, s, emb_dim, h_idx, d_idx, c = 2, 80, 64, 4, 32, 4
    rope = modeling_qwen3_8_flash_next.Qwen3_8RotaryEmbedding(
        head_dim=64, partial_rotary_factor=0.5, rope_theta=10000.0, mrope_section=(6, 5, 5)
    )
    idx_f32 = modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(
        emb_dim=emb_dim,
        index_n_heads=h_idx,
        index_kv_heads=1,
        index_head_dim=d_idx,
        token_budget=16,
        compress_ratio=c,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(17),
    )
    for s_case, mc_case in ((12, None), (12, 2), (3, None), (3, 1), (70, None), (70, 4), (70, 5)):
      x_c = jnp.array(rng.standard_normal((b, s_case, emb_dim)), dtype=jnp.float32)
      pos_c = jnp.tile(jnp.arange(s_case, dtype=jnp.int32), (b, 1))
      cos_c, sin_c = rope(x_c, pos_c)
      _, blk_cd = idx_f32(x_c, (cos_c, sin_c), return_token_mask=True)
      _, blk_ct = idx_f32(x_c, (cos_c, sin_c), return_token_mask=False, block_chunk_size=mc_case)
      self.assertEqual(blk_ct.shape, (b, s_case, idx_f32.block_topk))
      np.testing.assert_array_equal(np.asarray(blk_ct), np.asarray(blk_cd))

    x = jnp.array(rng.standard_normal((b, s, emb_dim)), dtype=jnp.float32)
    pos = jnp.tile(jnp.arange(s, dtype=jnp.int32), (b, 1))
    cos, sin = rope(x, pos)
    gd_i, p_i, rest_i = nnx.split(idx_f32, nnx.Param, ...)
    hlo_tiled = (
        jax.jit(lambda pp, xi, ci, si: nnx.merge(gd_i, pp, rest_i)(xi, (ci, si), return_token_mask=False)[1])
        .lower(p_i, x, cos, sin)
        .compile()
        .as_text()
        or ""
    )
    hlo_dense = (
        jax.jit(lambda pp, xi, ci, si: nnx.merge(gd_i, pp, rest_i)(xi, (ci, si), return_token_mask=True)[1])
        .lower(p_i, x, cos, sin)
        .compile()
        .as_text()
        or ""
    )
    self.assertIn("f32[2,80,4,20]", hlo_dense)
    self.assertNotIn("f32[2,80,4,20]", hlo_tiled)
    self.assertNotIn("f32[2,80,20]", hlo_tiled)
    self.assertIn("f32[2,80,4,4]", hlo_tiled)

    # Negative control: boundary-crossing winner in partial tail chunk (M=17, mc=4 -> block 16 in chunk 4).
    s_nc, m_nc, mc_nc = 70, 17, 4
    q_nc = jnp.zeros((b, s_nc, h_idx, d_idx), dtype=jnp.float32).at[:, :, :, 0].set(1.0)
    bk_nc = jnp.full((b, m_nc, d_idx), 0.1, dtype=jnp.float32)
    bk_nc = bk_nc.at[:, 4, 0].set(5.0).at[:, 16, 0].set(9.0)
    bv_nc = jnp.ones((b, m_nc), dtype=bool)
    gs_nc = jnp.broadcast_to(jnp.arange(m_nc, dtype=jnp.int32)[None, :] * c, (b, m_nc))
    seg_nc = jnp.ones((b, s_nc), dtype=jnp.int32)
    bseg_nc = jnp.ones((b, m_nc), dtype=jnp.int32)
    _, ti_ok = modeling_qwen3_8_flash_next.tiled_qsa_top_k(
        q_nc,
        bk_nc,
        bv_nc,
        bseg_nc,
        gs_nc,
        seg_nc,
        compress_ratio=c,
        index_head_dim=d_idx,
        k_top=4,
        block_chunk_size=mc_nc,
    )
    np.testing.assert_array_equal(np.asarray(ti_ok[:, -1, :2]), np.array([[16, 4], [16, 4]], dtype=np.int32))
    bv_drop_tail = bv_nc.at[:, 16:].set(False)
    _, ti_dropped = modeling_qwen3_8_flash_next.tiled_qsa_top_k(
        q_nc,
        bk_nc,
        bv_drop_tail,
        bseg_nc,
        gs_nc,
        seg_nc,
        compress_ratio=c,
        index_head_dim=d_idx,
        k_top=4,
        block_chunk_size=mc_nc,
    )
    self.assertFalse(np.array_equal(np.asarray(ti_dropped[:, -1, :]), np.asarray(ti_ok[:, -1, :])))
    self.assertNotIn(16, np.asarray(ti_dropped[:, -1, :]).tolist()[0])

  def _check_qsa_tiled_indexer_outer_scan_remat_policies(self, policies: tuple[Any, ...]) -> None:
    """Verifies outer `lax.scan` + remat policies emit zero stacked `[num_chunks, B, S, mc]` residuals."""
    rng = np.random.default_rng(1417)
    b, s_rm, emb_dim = 2, 48, 64
    pos_rm = jnp.tile(jnp.arange(s_rm, dtype=jnp.int32), (b, 1))
    seg_rm = jnp.ones((b, s_rm), dtype=jnp.int32).at[:, 24:].set(2)
    attn_rm = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
        emb_dim,
        4,
        2,
        32,
        0.25,
        10000.0,
        (2, 1, 1),
        2,
        1,
        32,
        16,
        4,
        1e-6,
        jnp.float32,
        jnp.float32,
        attention_kernel="flash",
        rngs=nnx.Rngs(79),
    )
    attn_rm.indexer.block_chunk_size = 2  # M=12, mc=2 -> num_chunks=6
    gd_rm, params_rm, rest_rm = nnx.split(attn_rm, nnx.Param, ...)
    stacked_p = jax.tree.map(lambda a: jnp.stack([a, a], axis=0), params_rm)
    x_rm = jnp.array(rng.standard_normal((b, s_rm, emb_dim)), dtype=jnp.float32)

    bad_remat_toks = (
        "f32[6,2,48",
        "i32[6,2,48",
        "s32[6,2,48",
        "bool[6,2,48",
        "pred[6,2,48",
        "f32[2,48,2,12]",
        "f32[2,48,12]",
    )
    for pol in policies:

      def scanned_loss(h_in: jax.Array, sp_in: Any, pol_fn: Any = pol) -> jax.Array:
        def body_fn(carry: jax.Array, p_slice: Any) -> tuple[jax.Array, None]:
          @functools.partial(jax.checkpoint, policy=pol_fn, prevent_cse=True)
          def _step(c_in: jax.Array, ps: Any) -> jax.Array:
            mod = nnx.merge(gd_rm, ps, rest_rm)
            out = mod(c_in, position_ids=pos_rm, segment_ids=seg_rm)
            assert isinstance(out, jax.Array)
            return c_in + out

          return _step(carry, p_slice), None

        h_out, _ = jax.lax.scan(body_fn, h_in, sp_in)
        return jnp.sum(h_out)

      jitted_grad = jax.jit(jax.grad(scanned_loss, argnums=(0, 1)))
      lowered = jitted_grad.lower(x_rm, stacked_p)
      lowered_str = lowered.as_text() or ""
      hlo_str = lowered.compile().as_text() or ""
      for bad_tok in bad_remat_toks:
        self.assertNotIn(bad_tok, hlo_str)
      for mlir_bad in ("tensor<6x2x48x", "tensor<2x48x2x12xf32>", "tensor<2x48x12xf32>"):
        self.assertNotIn(mlir_bad, lowered_str)
      _, g_params = jitted_grad(x_rm, stacked_p)
      for leaf in jax.tree_util.tree_leaves(g_params["indexer"]):
        self.assertEqual(float(jnp.max(jnp.abs(leaf))), 0.0)

  def test_qsa_tiled_indexer_outer_scan_remat_save_and_offload(self):
    """Verifies tiled QSA indexer outer scan + named save/offload remat policies."""
    self._check_qsa_tiled_indexer_outer_scan_remat_policies(
        (
            jax.checkpoint_policies.save_only_these_names("query_proj"),
            jax.checkpoint_policies.save_and_offload_only_these_names(
                names_which_can_be_saved=[],
                names_which_can_be_offloaded=["query_proj"],
                offload_src="device",
                offload_dst="pinned_host",
            ),
        )
    )

  def test_qsa_tiled_indexer_outer_scan_remat_nothing_and_everything(self):
    """Verifies tiled QSA indexer outer scan + `nothing_saveable` and `everything_saveable` remat policies."""
    self._check_qsa_tiled_indexer_outer_scan_remat_policies(
        (
            jax.checkpoint_policies.nothing_saveable,
            jax.checkpoint_policies.everything_saveable,
        )
    )

  def test_qsa_tiled_indexer_4device_sequence_sharding_hlo(self):
    """Verifies tiled QSA indexer preserves length sharding with zero new collectives on 4-device CPU meshes."""
    if len(jax.devices()) < 4:
      self.skipTest("Requires 4 devices.")
    rng = np.random.default_rng(1419)
    b, s, emb_dim, h_idx, d_idx, c = 2, 80, 64, 4, 32, 4  # M = 20 != d_idx // 2 (16)
    idx = modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(
        emb_dim=emb_dim,
        index_n_heads=h_idx,
        index_kv_heads=1,
        index_head_dim=d_idx,
        token_budget=16,
        compress_ratio=c,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(29),
    )
    rope = modeling_qwen3_8_flash_next.Qwen3_8RotaryEmbedding(
        head_dim=64, partial_rotary_factor=0.5, rope_theta=10000.0, mrope_section=(6, 5, 5)
    )
    x_ref = jnp.array(rng.standard_normal((b, s, emb_dim)), dtype=jnp.float32)
    pos_ref = jnp.tile(jnp.arange(s, dtype=jnp.int32), (b, 1))
    cos_ref, sin_ref = rope(x_ref, pos_ref)
    seg_ref = jnp.ones((b, s), dtype=jnp.int32).at[:, 44:].set(2)
    _, blk_ref = idx(x_ref, (cos_ref, sin_ref), segment_ids=seg_ref, return_token_mask=True)
    gd_i, p_i, rest_i = nnx.split(idx, nnx.Param, ...)
    P = jax.sharding.PartitionSpec

    def _count_ag(hlo: str) -> int:
      return len(re.findall(r"=\s*[^=]*\ball-gather(?:-start)?\(", hlo))

    def _count_ar(hlo: str) -> int:
      return len(re.findall(r"=\s*[^=]*\ball-reduce(?:-start)?\(", hlo))

    for axes, x_spec, expected_local_tile, forbidden_full in (
        (("context", "tensor"), P(None, "context", None), "f32[2,40,4,4]", "f32[2,40,4,20]"),
        (("fsdp", "expert"), P("fsdp", "expert", None), "f32[1,40,4,4]", "f32[1,40,4,20]"),
    ):
      mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), axes)

      def _ns(sp, m=mesh):
        return jax.sharding.NamedSharding(m, sp)

      seg_spec = P(x_spec[0], x_spec[1])
      repl_x_spec = P(x_spec[0], None, None)
      with jax.set_mesh(mesh):
        x_s = jax.device_put(x_ref, _ns(x_spec))
        cos_s = jax.device_put(cos_ref, _ns(x_spec))
        sin_s = jax.device_put(sin_ref, _ns(x_spec))
        seg_s = jax.device_put(seg_ref, _ns(seg_spec))
        p_s = jax.tree.map(lambda a, ns_fn=_ns: jax.device_put(a, ns_fn(P())), p_i)

        fn_tiled = jax.jit(
            lambda pp, xi, ci, si, sgi: nnx.merge(gd_i, pp, rest_i)(
                xi, (ci, si), segment_ids=sgi, return_token_mask=False, block_chunk_size=4
            )[1]
        )
        fn_dense = jax.jit(
            lambda pp, xi, ci, si, sgi: nnx.merge(gd_i, pp, rest_i)(
                xi, (ci, si), segment_ids=sgi, return_token_mask=False, block_chunk_size=20
            )[1]
        )
        fn_pos_ctrl = jax.jit(
            lambda pp, xi, ci, si, sgi, rspec=repl_x_spec: nnx.merge(gd_i, pp, rest_i)(
                jax.lax.with_sharding_constraint(xi, rspec),
                (ci, si),
                segment_ids=sgi,
                return_token_mask=False,
                block_chunk_size=4,
            )[1]
        )
        blk_s = fn_tiled(p_s, x_s, cos_s, sin_s, seg_s)
        np.testing.assert_array_equal(np.asarray(blk_s), np.asarray(blk_ref))

        hlo_t = fn_tiled.lower(p_s, x_s, cos_s, sin_s, seg_s).compile().as_text() or ""
        hlo_d = fn_dense.lower(p_s, x_s, cos_s, sin_s, seg_s).compile().as_text() or ""
        hlo_pos = fn_pos_ctrl.lower(p_s, x_s, cos_s, sin_s, seg_s).compile().as_text() or ""

        self.assertLessEqual(_count_ag(hlo_t), _count_ag(hlo_d))
        self.assertLessEqual(_count_ar(hlo_t), _count_ar(hlo_d))
        self.assertGreater(_count_ag(hlo_pos), _count_ag(hlo_t))
        self.assertIn(expected_local_tile, hlo_t)
        self.assertNotIn(forbidden_full, hlo_t)
        self.assertIn(forbidden_full, hlo_d)

  def test_qsa_tiled_indexer_4device_explicit_sharding_modes(self):
    """Verifies `tiled_qsa_top_k` under `ShardMode.EXPLICIT` across 4-device axis combinations."""
    if len(jax.devices()) < 4:
      self.skipTest("Requires 4 devices.")
    rng = np.random.default_rng(1421)
    b, s, h_idx, d_idx, c = 2, 80, 4, 32, 4
    P = jax.sharding.PartitionSpec
    m_blk = s // c
    q_e0 = jnp.array(rng.standard_normal((b, s, h_idx, d_idx)), dtype=jnp.bfloat16)
    bk_e0 = jnp.array(rng.standard_normal((b, m_blk, d_idx)), dtype=jnp.bfloat16)
    bv_e0 = jnp.ones((b, m_blk), dtype=bool)
    gs_e0 = jnp.broadcast_to(jnp.arange(m_blk, dtype=jnp.int32)[None, :] * c, (b, m_blk))
    bs_e0 = jnp.ones((b, m_blk), dtype=jnp.int32)
    seg_e0 = jnp.ones((b, s), dtype=jnp.int32)
    _, ti_ref = modeling_qwen3_8_flash_next.tiled_qsa_top_k(
        q_e0, bk_e0, bv_e0, bs_e0, gs_e0, seg_e0, compress_ratio=c, index_head_dim=d_idx, k_top=4, block_chunk_size=4
    )

    for expl_axes, q_sp, bk_sp, bm_sp, seg_sp in (
        (
            ("fsdp", "context"),
            P("fsdp", "context", None, None),
            P("fsdp", None, None),
            P("fsdp", None),
            P("fsdp", "context"),
        ),
        (("data", "tensor"), P("data", None, "tensor", None), P("data", None, None), P("data", None), P("data", None)),
        (
            ("data", "tensor"),
            P("data", None, None, "tensor"),
            P("data", None, "tensor"),
            P("data", None),
            P("data", None),
        ),
        (
            ("context", "tensor"),
            P(None, "context", "tensor", None),
            P(None, None, None),
            P(None, None),
            P(None, "context"),
        ),
        (
            ("context", "expert"),
            P("expert", "context", None, None),
            P("expert", "context", None),
            P("expert", None),
            P("expert", "context"),
        ),
    ):
      mesh_expl = jax.sharding.Mesh(
          np.array(jax.devices()[:4]).reshape(2, 2),
          expl_axes,
          axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
      )

      def _ns_e(sp, m=mesh_expl):
        return jax.sharding.NamedSharding(m, sp)

      with jax.set_mesh(mesh_expl):
        q_e = jax.device_put(q_e0, _ns_e(q_sp))
        bk_e = jax.device_put(bk_e0, _ns_e(bk_sp))
        bv_e = jax.device_put(bv_e0, _ns_e(bm_sp))
        bs_e = jax.device_put(bs_e0, _ns_e(bm_sp))
        gs_e = jax.device_put(gs_e0, _ns_e(bm_sp))
        seg_e = jax.device_put(seg_e0, _ns_e(seg_sp))
        _, ti_e = jax.jit(
            lambda qq, kk, bv_i=bv_e, bs_i=bs_e, gs_i=gs_e, sg_i=seg_e: modeling_qwen3_8_flash_next.tiled_qsa_top_k(
                qq, kk, bv_i, bs_i, gs_i, sg_i, compress_ratio=c, index_head_dim=d_idx, k_top=4, block_chunk_size=4
            )
        )(q_e, bk_e)
        np.testing.assert_array_equal(np.asarray(ti_e), np.asarray(ti_ref))
        gq, gbk = jax.jit(
            jax.grad(
                lambda qq, kk, bv_i=bv_e, bs_i=bs_e, gs_i=gs_e, sg_i=seg_e: jnp.sum(
                    modeling_qwen3_8_flash_next.tiled_qsa_top_k(
                        qq,
                        kk,
                        bv_i,
                        bs_i,
                        gs_i,
                        sg_i,
                        compress_ratio=c,
                        index_head_dim=d_idx,
                        k_top=4,
                        block_chunk_size=4,
                    )[0]
                ),
                argnums=(0, 1),
            )
        )(q_e, bk_e)
        self.assertEqual(gq.sharding.spec, q_sp)
        self.assertEqual(gbk.sharding.spec, bk_sp)
        self.assertEqual(float(jnp.max(jnp.abs(gq))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(gbk))), 0.0)

  def test_qsa_tiled_indexer_default_off_lowered_hlo_identity(self):
    """Verifies return_token_mask=True and single-chunk flash emit identical lowered HLO to untiled parent."""

    def _norm_stablehlo(txt: str) -> str:
      txt = re.sub(r"module @\S+", "module @jit_fn", txt)
      return re.sub(r" loc\([^)]*\)", "", txt)

    def _ref_parent_indexer(mod, hidden_states, position_embeddings, segment_ids=None, return_token_mask=True):
      bsz, seq, _ = hidden_states.shape
      cr = mod.compress_ratio
      full_cos, full_sin = position_embeddings
      eff_mod = mod
      if not return_token_mask:
        hidden_states = jax.lax.stop_gradient(hidden_states)
        full_cos = jax.lax.stop_gradient(full_cos)
        full_sin = jax.lax.stop_gradient(full_sin)
        gd_m, p_m, r_m = nnx.split(mod, nnx.Param, ...)
        eff_mod = nnx.merge(gd_m, jax.tree.map(jax.lax.stop_gradient, p_m), r_m)
      curr_cos, curr_sin = full_cos[:, -seq:, :], full_sin[:, -seq:, :]
      qk = eff_mod.index_qk_proj(hidden_states)
      q_dim = mod.index_n_heads * mod.index_head_dim
      q = eff_mod.q_layernorm(qk[..., :q_dim].reshape(bsz, seq, mod.index_n_heads, mod.index_head_dim))
      q_rot = modeling_qwen3_8_flash_next.apply_rotary_pos_emb(q, cos=curr_cos, sin=curr_sin, unsqueeze_dim=2)
      assert isinstance(q_rot, jax.Array)
      raw_keys = qk[..., q_dim:].reshape(bsz, seq, mod.index_head_dim)
      unpacked = segment_ids is None
      num_blocks = max(1, seq // cr)
      if return_token_mask:
        if segment_ids is None:
          segment_ids = jnp.ones((bsz, seq), dtype=jnp.int32)
        same_doc = (segment_ids[:, :, None] == segment_ids[:, None, :]) & (segment_ids[:, :, None] != 0)
        doc_start = jnp.argmax(same_doc, axis=-1)
        local_pos, end_t = jnp.arange(seq)[None, :] - doc_start, jnp.clip(jnp.arange(seq) + cr - 1, 0, seq - 1)
        is_block_start = (
            (segment_ids != 0)
            & ((local_pos % cr) == 0)
            & ((jnp.arange(seq) + cr - 1 < seq)[None, :])
            & (segment_ids[:, end_t] == segment_ids)
        )
        group_starts = jax.vmap(lambda mk: jnp.nonzero(mk, size=num_blocks, fill_value=0)[0])(is_block_start)
        block_valid = jnp.arange(num_blocks)[None, :] < jnp.sum(is_block_start.astype(jnp.int32), axis=-1, keepdims=True)
        block_seg_ids = jnp.take_along_axis(segment_ids, group_starts, axis=1)
        block_token_idx = jnp.clip(group_starts[:, :, None] + jnp.arange(cr)[None, None, :], 0, seq - 1)
        block_keys_raw = jax.vmap(lambda k, ix: k[ix])(raw_keys, block_token_idx)
        pooled_keys = eff_mod.k_layernorm(jnp.mean(block_keys_raw.astype(jnp.float32), axis=2).astype(raw_keys.dtype))
        cos_blk = jax.vmap(lambda cb, ix: cb[ix])(full_cos, group_starts)
        sin_blk = jax.vmap(lambda sb, ix: sb[ix])(full_sin, group_starts)
        res = modeling_qwen3_8_flash_next.apply_rotary_pos_emb(
            pooled_keys[:, :, None, :], cos=cos_blk, sin=sin_blk, unsqueeze_dim=2
        )
        assert isinstance(res, jax.Array)
        block_keys = res.squeeze(2)
      else:
        same_doc = None
        if segment_ids is None:
          segment_ids = jnp.ones((bsz, seq), dtype=jnp.int32)
        # pylint: disable=protected-access
        doc_start = (
            jnp.zeros((bsz, seq), dtype=jnp.int32)
            if unpacked
            else modeling_qwen3_8_flash_next._compute_qsa_doc_start(segment_ids)
        )
        # pylint: enable=protected-access
        local_pos, end_t = jnp.arange(seq)[None, :] - doc_start, jnp.clip(jnp.arange(seq) + cr - 1, 0, seq - 1)
        is_block_start = (
            (segment_ids != 0)
            & ((local_pos % cr) == 0)
            & ((jnp.arange(seq) + cr - 1 < seq)[None, :])
            & (segment_ids[:, end_t] == segment_ids)
        )
        group_starts = jax.vmap(lambda mk: jnp.nonzero(mk, size=num_blocks, fill_value=0)[0])(is_block_start)
        block_valid = jnp.arange(num_blocks)[None, :] < jnp.sum(is_block_start.astype(jnp.int32), axis=-1, keepdims=True)
        block_seg_ids = jnp.take_along_axis(segment_ids, group_starts, axis=1)
        block_token_idx = jnp.clip(group_starts[:, :, None] + jnp.arange(cr)[None, None, :], 0, seq - 1)
        block_keys_raw = jnp.take_along_axis(raw_keys[:, :, None, :], block_token_idx[:, :, :, None], axis=1)
        pooled_keys = eff_mod.k_layernorm(jnp.mean(block_keys_raw.astype(jnp.float32), axis=2).astype(raw_keys.dtype))
        cos_blk = jnp.take_along_axis(full_cos, group_starts[:, :, None], axis=1)
        sin_blk = jnp.take_along_axis(full_sin, group_starts[:, :, None], axis=1)
        res = modeling_qwen3_8_flash_next.apply_rotary_pos_emb(
            pooled_keys[:, :, None, :], cos=cos_blk, sin=sin_blk, unsqueeze_dim=2
        )
        assert isinstance(res, jax.Array)
        block_keys = res.squeeze(2)

      k_top = min(mod.block_topk, num_blocks)
      raw_scores = jnp.einsum("bthd,bmd->bthm", q_rot.astype(jnp.float32), block_keys.astype(jnp.float32))
      scores = jnp.sum(jax.nn.relu(raw_scores), axis=2) / jnp.sqrt(float(mod.index_head_dim))
      valid_q_block = (
          block_valid[:, None, :]
          & (block_seg_ids[:, None, :] == segment_ids[:, :, None])
          & ((group_starts + cr - 1)[:, None, :] <= jnp.arange(seq)[None, :, None])
          & (segment_ids[:, :, None] != 0)
      )
      top_vals, top_indices = jax.lax.top_k(jnp.where(valid_q_block, scores, -1e9), k_top)
      doc_local_idx = (jnp.take_along_axis(group_starts[:, None, :], top_indices, axis=2) - doc_start[:, :, None]) // cr
      selected_block_indices = jnp.where(top_vals > -1e8, doc_local_idx, -1).astype(jnp.int32)
      if mod.block_topk > k_top:
        selected_block_indices = jnp.pad(
            selected_block_indices, ((0, 0), (0, 0), (0, mod.block_topk - k_top)), constant_values=-1
        )
      if not return_token_mask:
        return None, selected_block_indices
      assert same_doc is not None
      m_local = jnp.arange(num_blocks)[None, None, :]
      in_topk = jnp.any(
          (selected_block_indices[:, :, :, None] == m_local) & (selected_block_indices[:, :, :, None] >= 0), axis=2
      )
      ar_k, ar_q = jnp.arange(seq)[None, None, :], jnp.arange(seq)[None, :, None]
      is_selected = jnp.take_along_axis(
          in_topk, jnp.clip((ar_k - doc_start[:, :, None]) // cr, 0, num_blocks - 1), axis=-1
      )
      in_tail = ar_k >= (doc_start + ((jnp.arange(seq)[None, :] - doc_start + 1) // cr) * cr)[:, :, None]
      token_mask = (same_doc & (ar_k <= ar_q) & (is_selected | in_tail))[:, None, :, :]
      return token_mask, selected_block_indices

    b, s, emb_dim, h_idx, d_idx, c = 2, 48, 64, 4, 32, 4  # num_blocks = 12 > block_topk = 4 > chunk = 2
    idx = modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(
        emb_dim=emb_dim,
        index_n_heads=h_idx,
        index_kv_heads=1,
        index_head_dim=d_idx,
        token_budget=16,
        compress_ratio=c,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(7),
    )
    idx_init_chunk = modeling_qwen3_8_flash_next.Qwen3_8QSAIndexer(
        emb_dim=emb_dim,
        index_n_heads=h_idx,
        index_kv_heads=1,
        index_head_dim=d_idx,
        token_budget=16,
        compress_ratio=c,
        block_chunk_size=2,
        dtype=jnp.bfloat16,
        weight_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(7),
    )
    rope = modeling_qwen3_8_flash_next.Qwen3_8RotaryEmbedding(
        head_dim=64, partial_rotary_factor=0.5, rope_theta=10000.0, mrope_section=(6, 5, 5)
    )
    x = jnp.ones((b, s, emb_dim), dtype=jnp.bfloat16)
    pos = jnp.tile(jnp.arange(s, dtype=jnp.int32), (b, 1))
    cos, sin = rope(x, pos)
    seg = jnp.ones((b, s), dtype=jnp.int32).at[:, 28:].set(2)
    gd, params, rest = nnx.split(idx, nnx.Param, ...)
    gd_ic, params_ic, rest_ic = nnx.split(idx_init_chunk, nnx.Param, ...)

    hlo_default = _norm_stablehlo(
        jax.jit(lambda p, xi, ci, si, sgi: nnx.merge(gd, p, rest)(xi, (ci, si), segment_ids=sgi, return_token_mask=True))
        .lower(params, x, cos, sin, seg)
        .as_text()
        or ""
    )
    hlo_call_chunk2 = _norm_stablehlo(
        jax.jit(
            lambda p, xi, ci, si, sgi: nnx.merge(gd, p, rest)(
                xi, (ci, si), segment_ids=sgi, return_token_mask=True, block_chunk_size=2
            )
        )
        .lower(params, x, cos, sin, seg)
        .as_text()
        or ""
    )
    hlo_init_chunk2 = _norm_stablehlo(
        jax.jit(
            lambda p, xi, ci, si, sgi: nnx.merge(gd_ic, p, rest_ic)(xi, (ci, si), segment_ids=sgi, return_token_mask=True)
        )
        .lower(params_ic, x, cos, sin, seg)
        .as_text()
        or ""
    )
    hlo_ref_dense = _norm_stablehlo(
        jax.jit(
            lambda p, xi, ci, si, sgi: _ref_parent_indexer(
                nnx.merge(gd, p, rest), xi, (ci, si), segment_ids=sgi, return_token_mask=True
            )
        )
        .lower(params, x, cos, sin, seg)
        .as_text()
        or ""
    )
    hlo_single_chunk = _norm_stablehlo(
        jax.jit(
            lambda p, xi, ci, si, sgi: nnx.merge(gd, p, rest)(
                xi, (ci, si), segment_ids=sgi, return_token_mask=False, block_chunk_size=12
            )[1]
        )
        .lower(params, x, cos, sin, seg)
        .as_text()
        or ""
    )
    hlo_ref_nomask = _norm_stablehlo(
        jax.jit(
            lambda p, xi, ci, si, sgi: _ref_parent_indexer(
                nnx.merge(gd, p, rest), xi, (ci, si), segment_ids=sgi, return_token_mask=False
            )[1]
        )
        .lower(params, x, cos, sin, seg)
        .as_text()
        or ""
    )
    hlo_tiled_active = _norm_stablehlo(
        jax.jit(
            lambda p, xi, ci, si, sgi: nnx.merge(gd, p, rest)(xi, (ci, si), segment_ids=sgi, return_token_mask=False)[1]
        )
        .lower(params, x, cos, sin, seg)
        .as_text()
        or ""
    )

    self.assertEqual(hlo_default, hlo_ref_dense)
    self.assertEqual(hlo_default, hlo_call_chunk2)
    self.assertEqual(hlo_default, hlo_init_chunk2)
    self.assertEqual(hlo_single_chunk, hlo_ref_nomask)
    self.assertNotIn("stablehlo.while", hlo_default)
    # Positive control: active tiled path (return_token_mask=False) emits stablehlo.while and differs from dense.
    self.assertIn("stablehlo.while", hlo_tiled_active)
    self.assertNotEqual(hlo_tiled_active, hlo_ref_nomask)

    # Default Qwen3_8SparseAttention (attention_kernel='dense') lowered HLO must also be unaffected by block_chunk_size.
    attn_def = self._make_jax_qsa(emb_dim=emb_dim, rng_seed=11, token_budget=16)
    attn_c2 = self._make_jax_qsa(emb_dim=emb_dim, rng_seed=11, token_budget=16)
    attn_c2.indexer.block_chunk_size = 2
    x32 = jnp.ones((b, s, emb_dim), dtype=jnp.float32)
    gd_a1, p_a1, r_a1 = nnx.split(attn_def, nnx.Param, ...)
    gd_a2, p_a2, r_a2 = nnx.split(attn_c2, nnx.Param, ...)
    hlo_attn_def = _norm_stablehlo(
        jax.jit(lambda p, xi, pi, sgi: nnx.merge(gd_a1, p, r_a1)(xi, pi, segment_ids=sgi))
        .lower(p_a1, x32, pos, seg)
        .as_text()
        or ""
    )
    hlo_attn_c2 = _norm_stablehlo(
        jax.jit(lambda p, xi, pi, sgi: nnx.merge(gd_a2, p, r_a2)(xi, pi, segment_ids=sgi))
        .lower(p_a2, x32, pos, seg)
        .as_text()
        or ""
    )
    self.assertEqual(hlo_attn_def, hlo_attn_c2)
    self.assertNotIn("stablehlo.while", hlo_attn_def)


# =================================================================================================
# Sparse MoE block
# =================================================================================================


@pytest.mark.cpu_only
class Qwen3_8MoETest(absltest.TestCase):

  def _make_moe(self, **kwargs) -> modeling_qwen3_8_flash_next.Qwen3_8SparseMoeBlock:
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
    return modeling_qwen3_8_flash_next.Qwen3_8SparseMoeBlock(**cfg)

  def test_moe_parameter_tree_contract(self):
    mod = self._make_moe(emb_dim=64, moe_mlp_dim=32, shared_expert_intermediate_size=48, num_experts=16)
    for p, shape in [
        (mod.gate.linear.kernel.value, (64, 16)),
        (mod.experts.gate_up_proj.value, (16, 64, 2, 32)),
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
    experts = modeling_qwen3_8_flash_next.Qwen3_8Experts(
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

  def test_direct_in_mesh_init_uses_sharded_gmm_executor(self):
    if len(jax.devices()) < 4:
      self.skipTest("Needs 4 devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count=4).")
    n, d, m, e, k = 16, 32, 16, 8, 2
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("expert", "tensor"))
    with jax.set_mesh(mesh):
      moe_block = self._make_moe(
          emb_dim=d, moe_mlp_dim=m, shared_expert_intermediate_size=m, num_experts=e, num_experts_per_tok=k
      )
      self.assertEqual(moe_block.experts.executor.kind, "gmm_v2")
      self.assertEqual(moe_block.experts.executor.experts, "expert")
      self.assertEqual(moe_block.experts.executor.mlp, "tensor")
      x = jax.random.normal(jax.random.key(21), (n, d), dtype=jnp.float32)
      top_k_idx = jax.random.randint(jax.random.key(22), (n, k), 0, e)
      top_k_w = jax.random.uniform(jax.random.key(23), (n, k), dtype=jnp.float32)
      top_k_w = top_k_w / jnp.sum(top_k_w, axis=-1, keepdims=True)
      actual = jax.jit(lambda mod, inp, idx, w: mod(inp, idx, w))(moe_block.experts, x, top_k_idx, top_k_w)
      expected = moe_block.experts.reference_forward(x, top_k_idx, top_k_w)
      np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-4)

    # Outside a mesh, direct init without sharding/executor keeps the unsharded ragged_dot default.
    unsharded = modeling_qwen3_8_flash_next.Qwen3_8Experts(
        num_experts=e, emb_dim=d, moe_mlp_dim=m, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=nnx.Rngs(0)
    )
    self.assertEqual(unsharded.executor.kind, "ragged_dot")

  def test_experts_shard_map_matches_reference_forward_on_cpu_mesh(self):
    if len(jax.devices()) < 4:
      self.skipTest("Needs 4 devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count=4).")
    n, d, m, e, k = 16, 32, 16, 12, 2
    experts = modeling_qwen3_8_flash_next.Qwen3_8Experts(
        num_experts=e, emb_dim=d, moe_mlp_dim=m, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=nnx.Rngs(7)
    )
    x = jax.random.normal(jax.random.key(11), (n, d))
    top_k_idx = jax.random.randint(jax.random.key(12), (n, k), 0, e)
    top_k_w = jax.random.uniform(jax.random.key(13), (n, k))
    top_k_w = top_k_w / jnp.sum(top_k_w, axis=-1, keepdims=True)

    def ref_loss_fn(mod, inp):
      return jnp.sum(mod.reference_forward(inp, top_k_idx, top_k_w) * jnp.arange(1, d + 1, dtype=jnp.float32))

    ref_out = experts.reference_forward(x, top_k_idx, top_k_w)
    ref_loss, (ref_grads, ref_dx) = nnx.value_and_grad(ref_loss_fn, argnums=(0, 1))(experts, x)

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]), ("fsdp",))
    token_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp", None))

    x_sharded = jax.device_put(x, token_sharding)
    idx_sharded = jax.device_put(top_k_idx, token_sharding)
    w_sharded = jax.device_put(top_k_w, token_sharding)
    gu_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "fsdp", None, None))
    dn_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, None, "fsdp"))
    experts.gate_up_proj.value = jax.device_put(experts.gate_up_proj.value, gu_sharding)
    experts.down_proj.value = jax.device_put(experts.down_proj.value, dn_sharding)

    with jax.set_mesh(mesh):
      actual_out = jax.jit(lambda mod, inp, idx, w: mod(inp, idx, w))(experts, x_sharded, idx_sharded, w_sharded)

      def sharded_loss_fn(mod, inp):
        return jnp.sum(mod(inp, idx_sharded, w_sharded) * jnp.arange(1, d + 1, dtype=jnp.float32))

      actual_loss, (actual_grads, actual_dx) = jax.jit(nnx.value_and_grad(sharded_loss_fn, argnums=(0, 1)))(
          experts, x_sharded
      )

    np.testing.assert_allclose(np.asarray(actual_out), np.asarray(ref_out), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(actual_loss), np.asarray(ref_loss), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(actual_dx), np.asarray(ref_dx), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(actual_grads.gate_up_proj.value), np.asarray(ref_grads.gate_up_proj.value), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(actual_grads.down_proj.value), np.asarray(ref_grads.down_proj.value), rtol=1e-5, atol=1e-5
    )

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
    jax_moe = modeling_qwen3_8_flash_next.Qwen3_8SparseMoeBlock(
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
    # gate_up_proj: HF (E, 2I, H) -> MaxText (E, H, 2, I)
    # down_proj: HF (E, H, I) -> MaxText (E, I, H)
    jax_moe.gate.linear.kernel.value = jnp.array(pt_moe.gate.weight.detach().cpu().numpy().T)
    jax_moe.experts.gate_up_proj.value = jnp.swapaxes(
        jnp.array(pt_moe.experts.gate_up_proj.detach().cpu().numpy()), -1, -2
    ).reshape(e, h, 2, i)
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


# =================================================================================================
# SplitMix64 n-gram hashing and PLE
# =================================================================================================


@pytest.mark.cpu_only
class Qwen3_8PLETest(absltest.TestCase):

  def test_splitmix64_known_vectors(self):
    multipliers, moduli, table_offsets, padded_vocab_size = modeling_qwen3_8_flash_next.compute_ple_buffers(
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=20_000_000,
        make_ngram_vocab_size_divisible_by=128,
        seed=1234,
    )
    expected_multipliers = [23703573157769, 20109073645365, 8052911324071]
    expected_moduli = [
        int(x)
        for x in (
            "20000003,20000023,20000033,20000047,20000059,20000063,20000069,20000077,"
            "20000081,20000093,20000107,20000147,20000153,20000159,20000161,20000171"
        ).split(",")
    ]
    expected_offsets = [
        int(x)
        for x in (
            "0,20000003,40000026,60000059,80000106,100000165,120000228,140000297,"
            "160000374,180000455,200000548,220000655,240000802,260000955,280001114,"
            "300001275"
        ).split(",")
    ]
    self.assertEqual(multipliers, expected_multipliers)
    self.assertEqual(moduli, expected_moduli)
    self.assertEqual(table_offsets, expected_offsets)
    self.assertEqual(padded_vocab_size, 320_001_536)
    self.assertEqual(2560 // 16, 160)
    self.assertTrue(modeling_qwen3_8_flash_next._is_prime(20000003))  # pylint: disable=protected-access
    self.assertFalse(modeling_qwen3_8_flash_next._is_prime(20000001))  # pylint: disable=protected-access

  def test_ple_parameter_tree_contract(self):
    mod = modeling_qwen3_8_flash_next.Qwen3_8PLELayer(
        eos_token_id=3,
        emb_dim=32,
        vocab_size=1000,
        ple_embed_dim=64,
        heads_per_ngram=2,
        ngram_vocab_size_base=100,
        make_ngram_vocab_size_divisible_by=16,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(42),
    )
    self.assertIsNotNone(mod.ple_embedding.ngram_embedding)
    self.assertEqual(mod.ple_embedding.ngram_embedding.embedding.value.shape, (432, 16))
    self.assertEqual(mod.key_proj.kernel.value.shape, (64, 128))
    self.assertEqual(mod.value_proj.kernel.value.shape, (64, 32))
    self.assertEqual(mod.norm_key.scale.value.shape, (128,))
    self.assertEqual(mod.conv1d.kernel.value.shape, (4, 1, 128))
    np.testing.assert_array_equal(mod.conv1d.kernel.value, 0.0)

    # Ensure static integer buffers are NOT in nnx.Param state
    params = nnx.state(mod, nnx.Param)
    self.assertIn("embedding", params["ple_embedding"]["ngram_embedding"])
    self.assertNotIn("layer_multipliers", params["ple_embedding"])
    self.assertNotIn("head_vocab_sizes", params["ple_embedding"])

  def test_r2_uint32_invariants_raise_on_overflow(self):
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8NGramEmbedding(
          eos_token_id=3, ngram_vocab_size_base=(1 << 31) + 1, rngs=nnx.Rngs(0)
      )
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8NGramEmbedding(
          eos_token_id=3, ngram_vocab_size_base=(1 << 28) + 1, rngs=nnx.Rngs(0)
      )
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8NGramEmbedding(eos_token_id=1 << 24, rngs=nnx.Rngs(0))
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8NGramEmbedding(eos_token_id=3, vocab_size=(1 << 24) + 1, rngs=nnx.Rngs(0))

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_pytorch_ple_weight_copy_parity(self):
    self.assertIsNotNone(torch)
    self.assertIsNotNone(modeling_qwen4_exp)
    self.assertIsNotNone(configuration_qwen4_exp)
    cfg_args = {
        "vocab_size": 1024,
        "ple_embed_dim": 32,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 1000,
        "make_ngram_vocab_size_divisible_by": 4,
        "eos_token_id": 3,
    }
    hf_cfg = configuration_qwen4_exp.Qwen4ExpTextConfig(hidden_size=16, hc_count=4, **cfg_args)
    hf_ple = modeling_qwen4_exp.Qwen4ExpTextPLELayer(hf_cfg, layer_idx=1, ple_layer_index=0)
    jax_ple = modeling_qwen3_8_flash_next.Qwen3_8PLELayer(
        emb_dim=16, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=nnx.Rngs(0), **cfg_args
    )

    # 1. Assert buffer arrays match 100% identically
    emb_j, emb_h = jax_ple.ple_embedding, hf_ple.ple_embedding
    self.assertEqual(list(emb_j.layer_multipliers), emb_h.layer_multipliers.tolist())
    self.assertEqual(list(emb_j.head_vocab_sizes), emb_h.ngram_heads_vocab_sizes.tolist())
    self.assertEqual(list(emb_j.head_offsets), emb_h.ngram_heads_offsets.tolist())
    self.assertEqual(emb_j.padded_vocab_size, emb_h.ngram_embedding.num_embeddings)

    # 2. Copy weights from PyTorch to JAX
    for j_w, p_w, tr in [
        (emb_j.ngram_embedding.embedding, emb_h.ngram_embedding.weight, None),
        (jax_ple.conv1d.kernel, hf_ple.conv1d.weight, (2, 1, 0)),
        (jax_ple.norm_key.scale, hf_ple.norm_key.weight, None),
        (jax_ple.norm_query.scale, hf_ple.norm_query.weight, None),
        (jax_ple.norm_conv.scale, hf_ple.norm_conv.weight, None),
        (jax_ple.key_proj.kernel, hf_ple.key_proj.weight, (1, 0)),
        (jax_ple.value_proj.kernel, hf_ple.value_proj.weight, (1, 0)),
    ]:
      arr = p_w.detach().cpu().numpy()
      j_w.value = jnp.array(arr.transpose(tr) if tr else arr)

    # 3. Test on batch with eos_token_id=3 in the middle of sequences
    input_ids = np.array([[10, 20, 3, 40, 50, 60, 3, 70], [5, 3, 15, 25, 35, 45, 55, 65]], dtype=np.int64)
    rng = np.random.default_rng(42)
    hidden_states = rng.standard_normal((2, 8, 4 * 16)).astype(np.float32)

    pt_emb = hf_ple.ple_embedding(torch.from_numpy(input_ids), past_key_values=None).detach().cpu().numpy()
    jax_emb = np.array(jax_ple.ple_embedding(jnp.array(input_ids)))
    np.testing.assert_allclose(jax_emb, pt_emb, atol=1e-5, rtol=1e-5)

    pt_out = hf_ple(torch.from_numpy(hidden_states), torch.from_numpy(input_ids), None).detach().cpu().numpy()
    jax_out = np.array(jax_ple(jnp.array(hidden_states), jnp.array(input_ids)))
    np.testing.assert_allclose(jax_out, pt_out, atol=1e-5, rtol=1e-5)

  def test_ngram_ids_empty_sequence(self):
    """Verifies an empty sequence yields no n-gram positions rather than the prepended context."""
    mod = modeling_qwen3_8_flash_next.Qwen3_8PLELayer(
        eos_token_id=3,
        emb_dim=16,
        vocab_size=1000,
        ple_embed_dim=32,
        heads_per_ngram=2,
        ngram_vocab_size_base=100,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    num_heads = 2 * 2  # (ngram_size - 1) n-gram orders x heads_per_ngram.
    self.assertEqual(mod.ple_embedding.compute_ngram_ids(jnp.zeros((2, 0), dtype=jnp.int32)).shape, (2, 0, num_heads))
    self.assertEqual(mod.ple_embedding.compute_ngram_ids(jnp.zeros((2, 5), dtype=jnp.int32)).shape, (2, 5, num_heads))

  def test_conv_mask_right_padding_does_not_leak(self):
    """Verifies the dilated short conv leaks nothing into right-padded positions."""
    b, s, s_valid, d, hc = 1, 12, 3, 16, 4
    mod = modeling_qwen3_8_flash_next.Qwen3_8PLELayer(
        eos_token_id=3,
        emb_dim=d,
        vocab_size=1000,
        ple_embed_dim=32,
        heads_per_ngram=2,
        ngram_vocab_size_base=100,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    mod.conv1d.kernel.value = jnp.ones_like(mod.conv1d.kernel.value)
    rng = np.random.default_rng(0)
    input_ids = jnp.array(rng.integers(4, 1000, size=(b, s)), dtype=jnp.int32)
    hidden = jnp.array(rng.standard_normal((b, s, hc * d)), dtype=jnp.float32)
    mask = jnp.concatenate([jnp.ones((b, s_valid)), jnp.zeros((b, s - s_valid))], axis=-1)

    out_masked = mod(hidden, input_ids, conv_mask=mask)
    out_unpadded = mod(hidden[:, :s_valid], input_ids[:, :s_valid])
    np.testing.assert_allclose(np.array(out_masked[:, s_valid:]), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.array(out_masked[:, :s_valid]), np.array(out_unpadded), rtol=1e-5, atol=1e-5)

  def test_rank_handling_and_conv_mask(self):
    b, s, d, hc = 2, 4, 16, 4
    mod = modeling_qwen3_8_flash_next.Qwen3_8PLELayer(
        eos_token_id=3,
        emb_dim=d,
        vocab_size=1000,
        ple_embed_dim=32,
        heads_per_ngram=2,
        ngram_vocab_size_base=100,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    mod.conv1d.kernel.value = jnp.ones_like(mod.conv1d.kernel.value)
    input_ids = jnp.zeros((b, s), dtype=jnp.int32)
    h_4d = jnp.ones((b, s, hc, d))
    h_3d = jnp.ones((b, s, hc * d))

    self.assertEqual(mod(h_4d, input_ids).shape, (b, s, hc, d))
    self.assertEqual(mod(h_3d, input_ids).shape, (b, s, hc * d))

    mask = jnp.array([[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    out_masked = mod(h_3d, input_ids, conv_mask=mask)
    np.testing.assert_allclose(out_masked[0, 0], 0.0, atol=1e-6)
    self.assertFalse(np.allclose(out_masked[1, 0], 0.0, atol=1e-3))


# =================================================================================================
# Decoder backbone, scan parity, and m3 registration
# =================================================================================================


def _get_mini_config(
    scan_layers: bool = False,
    dtype: str = "float32",
    weight_dtype: str = "float32",
    **kwargs,
):
  """Builds a test config initialized from qwen3.8-flash-next-mini.yml."""
  return pyconfig.initialize(
      [None, f"{maxtext_globals.MAXTEXT_CONFIGS_DIR}/base.yml"],
      model_name="qwen3.8-flash-next-mini",
      override_model_config=True,
      scan_layers=scan_layers,
      packing=False,
      dtype=dtype,
      weight_dtype=weight_dtype,
      run_name="backbone_test",
      **kwargs,
  )


def _copy_hf_weights_to_jax(hf_model, jax_model):
  """Copies weights from PyTorch Qwen4ExpForCausalLM to Qwen3_8FlashNextModel."""
  d, pm = jax_model.decoder, hf_model.model

  def to_j(t, tr=False):
    return jnp.array(t.detach().cpu().numpy().T if tr else t.detach().cpu().numpy())

  def cp(dst, src, *keys, tr=True, attr="kernel"):
    for k in keys:
      getattr(getattr(dst, k), attr)[...] = to_j(getattr(src, k).weight, tr)

  d.token_embedder.embedding[...] = to_j(pm.embed_tokens.weight)
  jax_model.lm_head.kernel[...] = to_j(hf_model.lm_head.weight, True)
  cp(d.hyper_connection_mixer, pm.hyper_connection_mixer, "input_mix_weight_down", "input_mix_weight_up")
  cp(d.hyper_connection_mixer, pm.hyper_connection_mixer, "hc_norm", tr=False, attr="scale")
  for i in range(d.num_layers):
    jl, pl = getattr(d, f"layers_{i}"), pm.layers[i]
    for jh, ph in (
        (jl.attn_hyper_connection, pl.attn_hyper_connection),
        (jl.mlp_hyper_connection, pl.mlp_hyper_connection),
    ):
      cp(jh, ph, "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")
      cp(jh, ph, "hc_norm", tr=False, attr="scale")
    if jl.ple is not None:
      jl.ple.ple_embedding.ngram_embedding.embedding[...] = to_j(pl.ple.ple_embedding.ngram_embedding.weight)
      jl.ple.conv1d.kernel[...] = jnp.array(pl.ple.conv1d.weight.detach().cpu().permute(2, 1, 0).numpy())
      cp(jl.ple, pl.ple, "norm_key", "norm_query", "norm_conv", tr=False, attr="scale")
      cp(jl.ple, pl.ple, "key_proj", "value_proj")
    if jl.layer_type == "linear_attention":
      cp(jl.linear_attn, pl.linear_attn, "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
      jl.linear_attn.A_log[...], jl.linear_attn.dt_bias[...] = to_j(pl.linear_attn.A_log), to_j(pl.linear_attn.dt_bias)
      cp(jl.linear_attn, pl.linear_attn, "norm", tr=False, attr="scale")
      jl.linear_attn.conv1d.kernel[...] = jnp.array(pl.linear_attn.conv1d.weight.detach().cpu().permute(2, 1, 0).numpy())
    else:
      cp(jl.self_attn, pl.self_attn, "q_proj", "k_proj", "v_proj", "o_proj")
      cp(jl.self_attn, pl.self_attn, "q_norm", "k_norm", tr=False, attr="scale")
      cp(jl.self_attn.indexer, pl.self_attn.indexer, "q_layernorm", "k_layernorm", tr=False, attr="scale")
      cp(jl.self_attn.indexer, pl.self_attn.indexer, "index_qk_proj")
    for d_l, w in (
        (jl.mlp.gate.linear, pl.mlp.gate.weight),
        (jl.mlp.shared_expert_gate, pl.mlp.shared_expert_gate.weight),
    ):
      d_l.kernel[...] = to_j(w, True)
    jl.mlp.experts.down_proj[...] = jnp.swapaxes(to_j(pl.mlp.experts.down_proj), -1, -2)
    jl.mlp.experts.gate_up_proj[...] = jnp.swapaxes(to_j(pl.mlp.experts.gate_up_proj), -1, -2).reshape(
        jl.mlp.experts.gate_up_proj.shape
    )
    cp(jl.mlp.shared_expert, pl.mlp.shared_expert, "gate_proj", "up_proj", "down_proj")


def _make_matching_hf_config(c) -> Any:
  """Constructs matching HuggingFace configuration from MaxText mini config."""
  if configuration_qwen4_exp is None:
    raise RuntimeError("transformers qwen4_exp not available")
  l = int(c.num_decoder_layers)
  rp = {
      "rope_theta": float(c.rope_max_timescale),
      "partial_rotary_factor": float(c.partial_rotary_factor),
      "mrope_interleaved": True,
      "mrope_section": list(c.mrope_section),
  }
  return configuration_qwen4_exp.Qwen4ExpTextConfig(
      vocab_size=int(c.vocab_size),
      hidden_size=int(c.emb_dim),
      num_hidden_layers=l,
      head_dim=int(c.head_dim),
      num_attention_heads=int(c.num_query_heads),
      num_key_value_heads=int(c.num_kv_heads),
      hc_count=int(c.hc_count),
      shared_expert_intermediate_size=int(c.mlp_dim),
      moe_intermediate_size=int(c.moe_mlp_dim),
      attention_bias=False,
      num_experts=int(c.num_experts),
      num_experts_per_tok=int(c.num_experts_per_tok),
      hc_lowrank=int(c.hc_lowrank),
      norm_topk_prob=bool(c.norm_topk_prob),
      output_gate_type=str(c.gdn_output_gate_activation),
      attention_dropout=0.0,
      layer_types=(["linear_attention"] * 3 + ["full_attention"]) * (l // 4),
      rope_parameters=rp,
      ple_layer_ids=[2],
      linear_num_key_heads=int(c.gdn_num_key_heads),
      linear_num_value_heads=int(c.gdn_num_value_heads),
      linear_key_head_dim=int(c.gdn_key_head_dim),
      linear_value_head_dim=int(c.gdn_value_head_dim),
      linear_conv_kernel_dim=int(c.gdn_conv_kernel_dim),
      indexer_compress_ratio=int(c.index_select_block_size),
      indexer_n_heads=int(c.index_n_heads),
      indexer_kv_heads=int(c.index_kv_heads),
      indexer_head_dim=int(c.index_head_dim),
      indexer_budget=int(c.index_select_budget),
      rms_norm_eps=float(c.normalization_layer_epsilon),
      seed=int(c.ple_seed),
      ple_embed_dim=int(c.ple_embed_dim),
      ple_conv_kernel_size=int(c.ple_conv_kernel_size),
      ngram_size=int(c.ngram_size),
      heads_per_ngram=int(c.heads_per_ngram),
      ngram_vocab_size_base=int(c.ngram_vocab_size_base),
      eos_token_id=int(c.ple_eos_token_id),
      make_ngram_vocab_size_divisible_by=int(c.make_ngram_vocab_size_divisible_by),
  )


@pytest.mark.cpu_only
class Qwen3_8BackboneTest(absltest.TestCase):
  """Backbone tests for scanned/unscanned parity and PyTorch reference match."""

  def test_scanned_vs_unscanned_parity_2_cycles(self):
    """Verifies scanned vs unscanned parity, 2-segment inputs, and registry dispatch."""
    cu, cs = (_get_mini_config(scan_layers=s, base_num_decoder_layers=8) for s in (False, True))
    mu, ms = (m3_models.create_model(c, mesh=None) for c in (cu, cs))
    self.assertIsInstance(mu, modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel)
    self.assertTrue({"qwen3.8-flash-next", "qwen3.8-flash-next-mini"}.issubset(m3_models.MODEL_REGISTRY))
    modeling_qwen3_8_flash_next.copy_unscanned_to_scanned(mu, ms)
    toks = jnp.array([[3, 17, 42, 88, 3, 50, 61, 99]], dtype=jnp.int32)
    pos, cmask = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (1, 8)), jnp.ones((1, 8), dtype=jnp.int32)
    (lu, au), (ls, as_) = (m(toks, decoder_positions=pos, decoder_segment_ids=cmask, return_aux=True) for m in (mu, ms))
    # 8 fp32 layers of GDN + MoE + QSA accumulate rounding near 1e-5, so cross-path checks use atol=5e-5.
    np.testing.assert_allclose(lu, ls, atol=5e-5, rtol=1e-5)
    jax.tree.map(lambda u, s: np.testing.assert_allclose(u, s, atol=5e-5, rtol=1e-5), au, as_)
    s2 = cmask.at[:, 4:].set(2)
    np.testing.assert_allclose(mu(toks, decoder_segment_ids=s2), ms(toks, decoder_segment_ids=s2), atol=5e-5, rtol=1e-5)

    logits_hs, pre_mixer = mu(toks, decoder_positions=pos, decoder_segment_ids=cmask, return_hidden_states=True)
    self.assertEqual(pre_mixer.shape, (1, 8, int(cu.hc_count) * int(cu.emb_dim)))
    np.testing.assert_allclose(
        mu.lm_head(mu.decoder.hyper_connection_mixer.mix_input(pre_mixer)), logits_hs, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(logits_hs, lu, atol=1e-5, rtol=1e-5)

  def test_scanned_vs_unscanned_bfloat16_parity(self):
    """Verifies scanned vs unscanned parity under bfloat16."""
    cu = _get_mini_config(
        scan_layers=False,
        base_num_decoder_layers=4,
        dtype="bfloat16",
        weight_dtype="bfloat16",
        num_experts=2,
        num_experts_per_tok=2,
        index_select_budget=64,
    )
    cs = _get_mini_config(
        scan_layers=True,
        base_num_decoder_layers=4,
        dtype="bfloat16",
        weight_dtype="bfloat16",
        num_experts=2,
        num_experts_per_tok=2,
        index_select_budget=64,
    )
    mu, ms = (m3_models.create_model(c, mesh=None) for c in (cu, cs))
    modeling_qwen3_8_flash_next.copy_unscanned_to_scanned(mu, ms)
    toks = jnp.array([[3, 17, 42, 88]], dtype=jnp.int32)
    pos, cmask = jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (1, 4)), jnp.ones((1, 4), dtype=jnp.int32)
    out_u = mu(toks, decoder_positions=pos, decoder_segment_ids=cmask)
    out_s = ms(toks, decoder_positions=pos, decoder_segment_ids=cmask)
    self.assertEqual(out_u.dtype, jnp.bfloat16)
    self.assertEqual(out_s.dtype, jnp.bfloat16)
    self.assertTrue(jnp.all(jnp.isfinite(out_u)))
    self.assertTrue(jnp.all(jnp.isfinite(out_s)))
    # Scanned and unscanned paths compile to different XLA programs, so bf16
    # rounding diverges across 4 layers and a handful of elements drift by up to
    # ~0.2. Exact cross-path logic is covered by the fp32 parity test above; here
    # bound the element-wise and mean error so gross bf16 regressions still fail
    # without depending on exact fusion decisions.
    # TODO(chengnuojin): Follow up with the XLA team on the bf16 drift introduced by the
    # ReshapeMover change that keeps data-reordering rearranges before widening converts,
    # and tighten these bounds again if it is resolved.
    out_u32, out_s32 = np.asarray(out_u, np.float32), np.asarray(out_s, np.float32)
    np.testing.assert_allclose(out_u32, out_s32, atol=2.5e-1, rtol=1e-1)
    self.assertLess(float(np.mean(np.abs(out_u32 - out_s32))), 5e-2)

  def test_scanned_cycles_fail_fast_validation(self):
    """Verifies fail-fast validation in scanned cycle construction."""
    with self.assertRaises(ValueError):
      c = _get_mini_config(scan_layers=True, ple_layer_id=2)
      m3_models.create_model(c, mesh=None)
    with self.assertRaises(ValueError):
      c = _get_mini_config(scan_layers=True, inhomogeneous_layer_cycle_interval=2)
      m3_models.create_model(c, mesh=None)
    with self.assertRaises(ValueError):
      c = _get_mini_config(scan_layers=True, base_num_decoder_layers=6)
      m3_models.create_model(c, mesh=None)

  def test_copy_unscanned_to_scanned_validation(self):
    """Verifies copy_unscanned_to_scanned rejects mismatched layer counts and PLE placement."""
    ms = m3_models.create_model(_get_mini_config(scan_layers=True, base_num_decoder_layers=4), mesh=None)
    for cu in (
        _get_mini_config(scan_layers=False, base_num_decoder_layers=8),
        _get_mini_config(scan_layers=False, base_num_decoder_layers=4, ple_layer_id=2),
    ):
      with self.assertRaises(ValueError):
        modeling_qwen3_8_flash_next.copy_unscanned_to_scanned(m3_models.create_model(cu, mesh=None), ms)
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.copy_unscanned_to_scanned(ms, ms)

  def test_scan_remat_matches_no_remat(self):
    """Verifies the checkpointed scanned cycle (remat) matches the uncheckpointed one (loss and gradients)."""
    cfgs = [_get_mini_config(scan_layers=True, base_num_decoder_layers=8, remat_policy=p) for p in ("full", "none")]
    m_remat, m_plain = (m3_models.create_model(c, mesh=None) for c in cfgs)
    nnx.update(m_plain, nnx.state(m_remat))
    self.assertTrue(m_remat.decoder.remat)
    self.assertFalse(m_plain.decoder.remat)
    toks = jnp.array([[3, 17, 42, 88, 3, 50, 61, 99]], dtype=jnp.int32)
    targets = jnp.array([[17, 42, 88, 3, 50, 61, 99, 5]], dtype=jnp.int32)
    seg = jnp.ones((1, 8), dtype=jnp.int32).at[:, 4:].set(2)

    def loss_fn(model):
      loss, _ = modeling_qwen3_8_flash_next.compute_total_loss(
          model, toks, targets, segment_ids=seg, router_aux_coeff=0.01, use_mtp=False
      )
      return loss

    def loss_and_grad(model, expect_remat):
      with mock.patch.object(jax, "checkpoint", wraps=jax.checkpoint) as checkpoint:
        out = nnx.value_and_grad(loss_fn)(model)
      # The scan body is the only jax.checkpoint in the model, so only the remat path may wrap it.
      self.assertEqual(checkpoint.called, expect_remat)
      return out

    (l_remat, g_remat), (l_plain, g_plain) = loss_and_grad(m_remat, True), loss_and_grad(m_plain, False)
    np.testing.assert_allclose(l_remat, l_plain, atol=1e-6, rtol=1e-6)
    # Recomputing the cycle in the backward pass changes XLA's fusion/accumulation order, so allow f32 noise.
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-4), g_remat, g_plain)

  def test_shard_batch_skips_indivisible_batches(self):
    """Verifies shard_batch constrains evenly divisible batches and leaves indivisible ones unconstrained."""
    if len(jax.devices()) < 4:
      self.skipTest("Needs 4 devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count=4).")
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]), ("fsdp",))
    with jax.set_mesh(mesh):
      for batch, constrained in ((8, True), (3, False), (1, False)):
        jaxpr = str(jax.make_jaxpr(modeling_qwen3_8_flash_next.shard_batch)(jnp.zeros((batch, 16))))
        self.assertEqual("sharding_constraint" in jaxpr, constrained, msg=f"batch={batch}")

    class _EpCpBatchLenSharding(modeling_qwen3_8_flash_next.DefaultSharding):
      """Maps activation batch onto fsdp and length onto expert."""

      def map_axis(self, axis, tensor_name, tensor_type):
        if tensor_type == modeling_qwen3_8_flash_next.TensorType.ACTIVATION:
          if axis == "batch":
            return ("fsdp",)
          if axis == "length":
            return ("expert",)
        return super().map_axis(axis, tensor_name, tensor_type)

    mesh_ep = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))
    ep_sharding = _EpCpBatchLenSharding()
    with jax.set_mesh(mesh_ep):
      jaxpr_ep = str(
          jax.make_jaxpr(lambda a: modeling_qwen3_8_flash_next.shard_batch(a, ep_sharding))(jnp.zeros((4, 16, 8)))
      )
      self.assertIn("spec=P('fsdp', 'expert', None)", jaxpr_ep)

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_pytorch_full_model_weight_copy_parity(self):
    """Verifies fp32 forward parity against HuggingFace Qwen4ExpForCausalLM."""
    self.assertIsNotNone(torch)
    self.assertIsNotNone(modeling_qwen4_exp)
    self.assertIsNotNone(configuration_qwen4_exp)
    cu, cs = (_get_mini_config(scan_layers=s, base_num_decoder_layers=8) for s in (False, True))
    mu, ms = (m3_models.create_model(c, mesh=None) for c in (cu, cs))
    torch.manual_seed(0)
    hf_model = modeling_qwen4_exp.Qwen4ExpForCausalLM(_make_matching_hf_config(cu)).eval()
    with torch.no_grad():
      for layer in hf_model.model.layers:
        if hasattr(layer, "ple") and layer.ple is not None:
          layer.ple.conv1d.weight.normal_(std=0.02)
      for mod in hf_model.modules():
        if type(mod).__name__.endswith("RMSNorm") and hasattr(mod, "weight") and mod.weight is not None:
          mod.weight.normal_(std=0.02)
    _copy_hf_weights_to_jax(hf_model, mu)
    modeling_qwen3_8_flash_next.copy_unscanned_to_scanned(mu, ms)
    input_ids = np.array([[10, 20, 3, 40, 50, 60, 3, 70], [5, 3, 15, 25, 35, 45, 55, 65]], dtype=np.int64)
    hf_masks = {}
    for idx in (3, 7):
      hf_model.model.layers[idx].self_attn.indexer.register_forward_hook(
          lambda _, __, o, i=idx: hf_masks.__setitem__(i, o)
      )
    pt_out = hf_model(torch.from_numpy(input_ids), output_router_logits=True)
    (lu, au), (ls, as_) = (m(jnp.array(input_ids, dtype=jnp.int32), return_aux=True) for m in (mu, ms))
    for out in (lu, ls):
      np.testing.assert_allclose(np.array(out), pt_out.logits.detach().cpu().numpy(), atol=5e-5, rtol=1e-5)
    for i, topk in enumerate(
        torch.topk(l.view(2, 8, -1), k=2, dim=-1).indices.cpu().numpy() for l in pt_out.router_logits
    ):
      for aux in (au, as_):
        np.testing.assert_array_equal(np.array(aux[i]["router_indices"]), topk)
    for l_idx in (3, 7):
      mask, jax_b = hf_masks[l_idx].detach().cpu().numpy(), np.array(au[l_idx]["selected_block_indices"])
      mask = mask if mask.dtype == bool else (mask == 0.0)
      for b, q in ((b, q) for b in range(2) for q in range(3, 8)):
        hf_sel = {m for m in range((q + 1) // 4) if np.all(mask[b, 0, q, m * 4 : (m + 1) * 4])}
        self.assertEqual(hf_sel, {int(idx) for idx in jax_b[b, q] if idx >= 0})

  def test_kernel_path_flags_match_defaults_end_to_end(self):
    """Verifies the TPU kernel-path config flags match the reference paths end-to-end on the mini backbone."""
    common_kw = {
        "scan_layers": True,
        "base_num_decoder_layers": 4,
        "num_experts": 2,
        "num_experts_per_tok": 2,
        "use_ple_layer": False,
        "index_select_budget": 64,
        "sparse_matmul": True,
    }
    cfg_default = _get_mini_config(**common_kw)
    self.assertEqual(
        (
            cfg_default.gdn_chunk_inverse,
            cfg_default.qsa_indexer_mask,
            cfg_default.qsa_attention_kernel,
        ),
        ("block_doubling", "gather", "dense"),
    )
    # Reference paths (triangular solve, gather mask, dense attention, dense MoE) vs. the TPU kernel paths.
    cfg_base = _get_mini_config(**common_kw, gdn_chunk_inverse="solve")
    cfg_opt = _get_mini_config(
        **common_kw,
        gdn_chunk_inverse="block_doubling",
        qsa_indexer_mask="reshape",
        qsa_attention_kernel="flash",
    )
    mod_base = m3_models.create_model(cfg_base, mesh=None)
    mod_opt = m3_models.create_model(cfg_opt, mesh=None)
    nnx.update(mod_opt, nnx.state(mod_base))

    toks = jnp.array([[3, 17, 42, 88, 3, 50, 61, 99]], dtype=jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(8, dtype=jnp.int32), (1, 8))
    seg = jnp.array([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=jnp.int32)

    logits_base, aux_base = mod_base(toks, decoder_positions=pos, decoder_segment_ids=seg, return_aux=True)
    logits_opt, aux_opt = mod_opt(toks, decoder_positions=pos, decoder_segment_ids=seg, return_aux=True)
    np.testing.assert_allclose(np.array(logits_opt), np.array(logits_base), rtol=1e-4, atol=1e-4)
    jax.tree.map(
        lambda b, o: np.testing.assert_allclose(np.array(o), np.array(b), rtol=1e-4, atol=1e-4), aux_base, aux_opt
    )


# =================================================================================================
# MTP draft block and integrated loss
# =================================================================================================


# pylint: disable=no-name-in-module


def _create_mini_config(
    num_layers: int = 4, mtp_num_layers: int = 1, scan_layers: bool = False, dtype=jnp.float32, **kwargs: Any
) -> types.SimpleNamespace:
  """Creates a mini-twin config SimpleNamespace for fast MTP and loss unit testing."""
  base = {
      "emb_dim": 64,
      "num_decoder_layers": num_layers,
      "vocab_size": 128,
      "eos_token_id": 127,
      "ple_eos_token_id": 127,
      "hc_count": 4,
      "hc_lowrank": 16,
      "scan_layers": scan_layers,
      "dtype": dtype,
      "weight_dtype": dtype,
      "normalization_layer_epsilon": 1e-6,
      "inhomogeneous_layer_cycle_interval": 4,
      "gdn_num_key_heads": 2,
      "gdn_num_value_heads": 4,
      "gdn_key_head_dim": 16,
      "gdn_value_head_dim": 16,
      "gdn_conv_kernel_dim": 4,
      "gdn_chunk_size": 16,
      "num_query_heads": 4,
      "num_kv_heads": 1,
      "head_dim": 16,
      "partial_rotary_factor": 0.25,
      "rope_max_timescale": 1e4,
      "mrope_section": (1, 1, 0),
      "index_n_heads": 2,
      "index_kv_heads": 1,
      "index_head_dim": 16,
      "index_select_budget": 32,
      "index_select_block_size": 4,
      "moe_mlp_dim": 32,
      "mlp_dim": 32,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "norm_topk_prob": True,
      "routed_scaling_factor": 1.0,
      "use_ple_layer": False,
      "heads_per_ngram": 2,
      "ple_seed": 0,
      "ple_kernel_size": 3,
      "split_ngram_parts": 2,
      "ple_hash_seed": 0,
      "ngram_vocab_size_base": 500,
      "mtp_num_layers": mtp_num_layers,
      "mtp_num_streams": 4,
      "mtp_loss_scaling_factor": 0.1,
      "load_balance_loss_weight": 0.001,
      "gdn_output_gate_activation": "silu",
      "gdn_chunk_inverse": "block_doubling",
      "qsa_indexer_mask": "gather",
      "qsa_attention_kernel": "dense",
      "ple_layer_ids": [],
  }
  base.update(kwargs)
  return types.SimpleNamespace(**base)


def _copy_layer_and_mixer_weights(pt_layer, pt_mixer, jx_layer, jx_mixer):
  """Copies decoder layer and mixer parameters from PyTorch to JAX modules."""

  def to_j(t, tr=False):
    return jnp.array(t.detach().cpu().numpy().T if tr else t.detach().cpu().numpy())

  def cp(dst, src, *keys, tr=True, attr="kernel"):
    for k in keys:
      getattr(getattr(dst, k), attr).value = to_j(getattr(src, k).weight, tr)

  for jh, ph in (
      (jx_layer.attn_hyper_connection, pt_layer.attn_hyper_connection),
      (jx_layer.mlp_hyper_connection, pt_layer.mlp_hyper_connection),
  ):
    cp(jh, ph, "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")
    cp(jh, ph, "hc_norm", tr=False, attr="scale")
  cp(jx_layer.self_attn, pt_layer.self_attn, "q_proj", "k_proj", "v_proj", "o_proj")
  cp(jx_layer.self_attn, pt_layer.self_attn, "q_norm", "k_norm", tr=False, attr="scale")
  cp(jx_layer.self_attn.indexer, pt_layer.self_attn.indexer, "q_layernorm", "k_layernorm", tr=False, attr="scale")
  cp(jx_layer.self_attn.indexer, pt_layer.self_attn.indexer, "index_qk_proj")
  for d_l, w in (
      (jx_layer.mlp.gate.linear, pt_layer.mlp.gate.weight),
      (jx_layer.mlp.shared_expert_gate, pt_layer.mlp.shared_expert_gate.weight),
  ):
    d_l.kernel.value = to_j(w, True)
  jx_layer.mlp.experts.down_proj.value = jnp.swapaxes(to_j(pt_layer.mlp.experts.down_proj), -1, -2)
  jx_layer.mlp.experts.gate_up_proj.value = jnp.swapaxes(to_j(pt_layer.mlp.experts.gate_up_proj), -1, -2).reshape(
      jx_layer.mlp.experts.gate_up_proj.shape
  )
  cp(jx_layer.mlp.shared_expert, pt_layer.mlp.shared_expert, "gate_proj", "up_proj", "down_proj")
  cp(jx_mixer, pt_mixer, "input_mix_weight_down", "input_mix_weight_up")
  cp(jx_mixer, pt_mixer, "hc_norm", tr=False, attr="scale")


@pytest.mark.cpu_only
class Qwen3_8MTPLossTest(absltest.TestCase):

  def test_mtp_block_shapes_and_parity(self):
    cfg = _create_mini_config()
    block = modeling_qwen3_8_flash_next.Qwen3_8MTPBlock(cfg, rngs=nnx.Rngs(params=0))
    self.assertEqual((block.pre_fc_norm_embedding.scale.shape, block.pre_fc_norm_hidden.scale.shape), ((64,), (256,)))
    self.assertEqual((block.fc_embedding.kernel.shape, block.fc_hidden.kernel.shape), ((64, 64), (64, 64)))
    b, s, c, d = 2, 8, 4, 64
    hc_hidden = jax.random.normal(jax.random.PRNGKey(1), (b, s, c * d))
    next_embeds = jax.random.normal(jax.random.PRNGKey(2), (b, s, d))
    out = block(hc_hidden, next_embeds)
    assert isinstance(out, jax.Array)
    self.assertEqual(out.shape, (b, s, d))
    e_proj = block.fc_embedding(block.pre_fc_norm_embedding(next_embeds))
    h_proj = block.fc_hidden(block.pre_fc_norm_hidden(hc_hidden).reshape(b, s, c, d))
    h_mtp_in = (h_proj + jnp.expand_dims(e_proj, axis=-2)).reshape(b, s, c * d)
    ref_out = block.hyper_connection_mixer.mix_input(block.layers_0(h_mtp_in, return_aux=False))
    np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-5)

  @unittest.skipUnless(_HAS_QWEN4_EXP, "requires internal transformers qwen4_exp")
  def test_pytorch_weight_copy_parity(self):
    assert torch is not None and modeling_qwen4_exp is not None and configuration_qwen4_exp is not None
    cfg, (b, s, c, dim) = _create_mini_config(), (2, 4, 4, 64)
    rp = {
        "rope_theta": float(cfg.rope_max_timescale),
        "partial_rotary_factor": float(cfg.partial_rotary_factor),
        "mrope_interleaved": True,
        "mrope_section": list(cfg.mrope_section),
    }
    hf_cfg = configuration_qwen4_exp.Qwen4ExpTextConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=dim,
        num_hidden_layers=4,
        head_dim=cfg.head_dim,
        num_attention_heads=cfg.num_query_heads,
        num_key_value_heads=cfg.num_kv_heads,
        hc_count=c,
        shared_expert_intermediate_size=cfg.mlp_dim,
        moe_intermediate_size=cfg.moe_mlp_dim,
        attention_bias=False,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        hc_lowrank=cfg.hc_lowrank,
        norm_topk_prob=cfg.norm_topk_prob,
        output_gate_type=cfg.gdn_output_gate_activation,
        attention_dropout=0.0,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        rope_parameters=rp,
        ple_layer_ids=[],
        linear_num_key_heads=cfg.gdn_num_key_heads,
        linear_num_value_heads=cfg.gdn_num_value_heads,
        linear_key_head_dim=cfg.gdn_key_head_dim,
        linear_value_head_dim=cfg.gdn_value_head_dim,
        linear_conv_kernel_dim=cfg.gdn_conv_kernel_dim,
        indexer_compress_ratio=cfg.index_select_block_size,
        indexer_n_heads=cfg.index_n_heads,
        indexer_kv_heads=cfg.index_kv_heads,
        indexer_head_dim=cfg.index_head_dim,
        indexer_budget=cfg.index_select_budget,
        rms_norm_eps=cfg.normalization_layer_epsilon,
        seed=cfg.ple_seed,
    )
    torch.manual_seed(42)
    pt_norm_emb, pt_norm_hid = (
        modeling_qwen4_exp.Qwen4ExpTextRMSNorm(d, group_size=None, eps=1e-6) for d in (dim, c * dim)
    )
    pt_fc_emb, pt_fc_hid = (torch.nn.Linear(dim, dim, bias=False) for _ in range(2))
    pt_layer = modeling_qwen4_exp.Qwen4ExpTextDecoderLayer(hf_cfg, layer_idx=3)
    pt_mixer = modeling_qwen4_exp.Qwen4ExpTextGatedResidual(hf_cfg, use_combine=False)

    with torch.no_grad():
      for p in (*pt_layer.parameters(), *pt_mixer.parameters()):
        p.normal_(std=0.02)

    jx_block = modeling_qwen3_8_flash_next.Qwen3_8MTPBlock(cfg, rngs=nnx.Rngs(params=0))
    jx_block.pre_fc_norm_embedding.scale.value = jnp.array(pt_norm_emb.weight.detach().cpu().numpy())
    jx_block.pre_fc_norm_hidden.scale.value = jnp.array(pt_norm_hid.weight.detach().cpu().numpy())
    jx_block.fc_embedding.kernel.value = jnp.array(pt_fc_emb.weight.detach().cpu().numpy().T)
    jx_block.fc_hidden.kernel.value = jnp.array(pt_fc_hid.weight.detach().cpu().numpy().T)
    _copy_layer_and_mixer_weights(pt_layer, pt_mixer, jx_block.layers_0, jx_block.hyper_connection_mixer)

    pt_hc, pt_nxt = torch.randn(b, s, c * dim), torch.randn(b, s, dim)
    pt_fused = pt_fc_hid(pt_norm_hid(pt_hc).view(b, s, c, dim)) + pt_fc_emb(pt_norm_emb(pt_nxt)).unsqueeze(-2)
    pt_mtp_in = pt_fused.view(b, s, c * dim)
    cos, sin = modeling_qwen4_exp.Qwen4ExpTextRotaryEmbedding(hf_cfg)(
        pt_mtp_in, torch.arange(s).unsqueeze(0).expand(b, -1)
    )
    mask = torch.tril(torch.ones((b, 1, s, s), dtype=torch.bool))
    pt_mask = torch.where(mask, torch.zeros((), dtype=torch.float32), torch.finfo(torch.float32).min)
    pt_out = pt_mixer(pt_layer(pt_mtp_in, position_embeddings=(cos, sin), attention_mask=pt_mask)).detach().cpu().numpy()

    jx_out = np.array(
        jx_block(
            jnp.array(pt_hc.numpy()),
            jnp.array(pt_nxt.numpy()),
            positions=jnp.broadcast_to(jnp.arange(s, dtype=jnp.int32), (b, s)),
        )
    )
    self.assertGreater(np.max(np.abs(pt_out)), 0.0)
    np.testing.assert_allclose(jx_out, pt_out, rtol=1e-5, atol=1e-5)

  def test_checkpoint_pinned_shape_contract(self):
    cfg = _create_mini_config(
        emb_dim=2560,
        hc_count=4,
        hc_lowrank=320,
        head_dim=256,
        mrope_section=(11, 11, 10),
        index_head_dim=128,
        num_query_heads=24,
        num_kv_heads=2,
        index_n_heads=4,
        index_kv_heads=1,
        num_experts=512,
        num_experts_per_tok=10,
        moe_mlp_dim=640,
        mlp_dim=640,
        use_ple_layer=True,
    )
    shapes = nnx.eval_shape(lambda: modeling_qwen3_8_flash_next.Qwen3_8MTPBlock(cfg, rngs=nnx.Rngs(params=0)))
    self.assertIsNone(shapes.layers_0.ple)
    self.assertEqual((shapes.fc_embedding.kernel.shape, shapes.fc_hidden.kernel.shape), ((2560, 2560), (2560, 2560)))
    self.assertEqual(
        (shapes.pre_fc_norm_embedding.scale.shape, shapes.pre_fc_norm_hidden.scale.shape), ((2560,), (10240,))
    )
    self.assertEqual(
        (
            shapes.hyper_connection_mixer.hc_norm.scale.shape,
            shapes.hyper_connection_mixer.input_mix_weight_down.kernel.shape,
            shapes.hyper_connection_mixer.input_mix_weight_up.kernel.shape,
        ),
        ((10240,), (10240, 320), (320, 10240)),
    )
    self.assertEqual(
        (shapes.layers_0.self_attn.q_norm.scale.shape, shapes.layers_0.self_attn.k_norm.scale.shape), ((256,), (256,))
    )
    self.assertEqual(
        (
            shapes.layers_0.self_attn.indexer.q_layernorm.scale.shape,
            shapes.layers_0.self_attn.indexer.k_layernorm.scale.shape,
        ),
        ((128,), (128,)),
    )
    self.assertEqual(
        (
            shapes.layers_0.mlp.experts.gate_up_proj.shape,
            shapes.layers_0.mlp.experts.down_proj.shape,
            shapes.layers_0.mlp.gate.linear.kernel.shape,
            shapes.layers_0.mlp.shared_expert_gate.kernel.shape,
            shapes.layers_0.mlp.shared_expert.gate_proj.kernel.shape,
            shapes.layers_0.mlp.shared_expert.up_proj.kernel.shape,
            shapes.layers_0.mlp.shared_expert.down_proj.kernel.shape,
        ),
        (
            (512, 2560, 2, 640),
            (512, 640, 2560),
            (2560, 512),
            (2560, 1),
            (2560, 640),
            (2560, 640),
            (640, 2560),
        ),
    )

  def test_phase_gating(self):
    cfg = _create_mini_config()
    model = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    self.assertIsNotNone(model.mtp)
    tokens = jax.random.randint(jax.random.PRNGKey(3), (2, 8), 0, 128)
    self.assertEqual(model(tokens, return_mtp=False).shape, (2, 8, 128))
    logits, mtp_logits = model(tokens, return_mtp=True)
    self.assertEqual((logits.shape, mtp_logits.shape), ((2, 8, 128), (2, 8, 128)))
    logits_aux, aux_dict = model(tokens, return_mtp=True, return_aux=True)
    self.assertEqual(logits_aux.shape, (2, 8, 128))
    self.assertTrue({"layer_aux", "mtp_logits", "mtp_aux"}.issubset(aux_dict))
    self.assertIn("router_probs", aux_dict["mtp_aux"])

  def test_cross_entropy_loss(self):
    logits, targets = jnp.zeros((2, 4, 8), dtype=jnp.float32), jnp.zeros((2, 4), dtype=jnp.int32)
    np.testing.assert_allclose(modeling_qwen3_8_flash_next.cross_entropy_loss(logits, targets), np.log(8.0), rtol=1e-5)
    mask = jnp.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=jnp.float32)
    np.testing.assert_allclose(
        modeling_qwen3_8_flash_next.cross_entropy_loss(logits, targets, mask=mask), np.log(8.0), rtol=1e-5
    )
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.cross_entropy_loss(jnp.array([]), jnp.array([]))
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.cross_entropy_loss(jnp.zeros((2, 4, 8)), jnp.zeros((2, 5), dtype=jnp.int32))
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.cross_entropy_loss(
          jnp.zeros((2, 4, 8)), jnp.zeros((2, 4), dtype=jnp.int32), mask=jnp.zeros((2, 5))
      )

  def test_router_load_balancing_loss(self):
    b, s, e = 2, 4, 4
    r_probs, r_idx = jnp.full((b, s, e), 1.0 / e, dtype=jnp.float32), jnp.tile(
        jnp.array([0, 1], dtype=jnp.int32), (b, s, 1)
    )
    aux = [{"router_probs": r_probs, "router_indices": r_idx}]
    lb_loss = modeling_qwen3_8_flash_next.router_load_balancing_loss(aux, num_experts=e)
    self.assertTrue(jnp.isfinite(lb_loss) and lb_loss > 0.0)
    for bad_aux in (None, [], [{"bad": 1}]):
      with self.assertRaises(ValueError):
        modeling_qwen3_8_flash_next.router_load_balancing_loss(bad_aux, 4)

  def test_router_load_balancing_loss_hf_equivalence(self):
    b, s, e, k = 2, 8, 4, 2
    rng = np.random.default_rng(42)
    layer_aux = []
    for _ in range(3):
      logits = rng.standard_normal((b, s, e)).astype(np.float32)
      probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
      indices = np.argsort(-probs, axis=-1)[:, :, :k].astype(np.int32)
      layer_aux.append({"router_probs": jnp.array(probs), "router_indices": jnp.array(indices)})
    mtp_logits = rng.standard_normal((b, s, e)).astype(np.float32)
    mtp_probs = np.exp(mtp_logits) / np.sum(np.exp(mtp_logits), axis=-1, keepdims=True)
    mtp_indices = np.argsort(-mtp_probs, axis=-1)[:, :, :k].astype(np.int32)
    mtp_aux = {"router_probs": jnp.array(mtp_probs), "router_indices": jnp.array(mtp_indices)}

    mask = jnp.array([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]], dtype=jnp.float32)
    mtp_mask = jnp.pad(mask[:, 1:], ((0, 0), (0, 1)))

    aux_dict = {"layer_aux": layer_aux, "mtp_aux": mtp_aux}
    jax_loss = modeling_qwen3_8_flash_next.router_load_balancing_loss(
        aux_dict, num_experts=e, mask=mask, mtp_mask=mtp_mask
    )

    items = [(lay, mask) for lay in layer_aux] + [(mtp_aux, mtp_mask)]
    tokens_sum = np.zeros(e, dtype=np.float64)
    probs_sum = np.zeros(e, dtype=np.float64)
    total_tokens = 0.0
    for l_aux, l_mask in items:
      rp = np.array(l_aux["router_probs"], dtype=np.float64)
      ri = np.array(l_aux["router_indices"])
      lm = np.array(l_mask, dtype=np.float64)
      one_hot = np.zeros((*ri.shape, e), dtype=np.float64)
      np.put_along_axis(one_hot, np.expand_dims(ri, -1), 1.0, axis=-1)
      tokens_sum += np.sum(one_hot * lm[:, :, None, None], axis=(0, 1, 2))
      probs_sum += np.sum(rp * lm[:, :, None], axis=(0, 1))
      total_tokens += np.sum(lm)

    ref_loss = e * np.sum((tokens_sum / total_tokens) * (probs_sum / total_tokens))
    np.testing.assert_allclose(float(jax_loss), ref_loss, rtol=1e-5, atol=1e-5)

  def test_compute_total_loss_and_gradients(self):
    cfg = _create_mini_config()
    model = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    inp, tgt = (jax.random.randint(jax.random.PRNGKey(k), (2, 16), 0, 128) for k in (4, 5))
    (total_l, metrics), grads = nnx.value_and_grad(
        lambda m: modeling_qwen3_8_flash_next.compute_total_loss(
            m, inp, tgt, router_aux_coeff=0.001, mtp_loss_coeff=0.1, use_mtp=True
        ),
        has_aux=True,
    )(model)
    self.assertTrue(jnp.isfinite(total_l))
    self.assertTrue({"total_loss", "primary_loss", "router_loss", "mtp_loss"}.issubset(metrics))
    self.assertGreater(jnp.sum(jnp.abs(grads.lm_head.kernel[...])), 0.0)
    self.assertGreater(jnp.sum(jnp.abs(grads.mtp.fc_embedding.kernel[...])), 0.0)
    self.assertGreater(jnp.sum(jnp.abs(grads.decoder.layers_0.mlp.experts.gate_up_proj[...])), 0.0)
    self.assertGreater(jnp.sum(jnp.abs(grads.decoder.layers_0.mlp.gate.linear.kernel[...])), 0.0)
    self.assertGreater(jnp.sum(jnp.abs(grads.mtp.layers_0.mlp.gate.linear.kernel[...])), 0.0)

  def test_compute_total_loss_single_token(self):
    cfg = _create_mini_config(num_layers=4)
    model = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    toks = jnp.array([[5]], dtype=jnp.int32)
    total_l, mets = modeling_qwen3_8_flash_next.compute_total_loss(model, toks, toks, use_mtp=True)
    self.assertTrue(jnp.isfinite(total_l))
    self.assertEqual(float(mets["mtp_loss"]), 0.0)

  def test_compute_total_loss_scanned(self):
    cfg_unscanned = _create_mini_config(num_layers=4, scan_layers=False)
    cfg_scanned = _create_mini_config(num_layers=4, scan_layers=True)
    unscanned_model = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(cfg_unscanned, rngs=nnx.Rngs(params=0))
    scanned_model = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(cfg_scanned, rngs=nnx.Rngs(params=42))
    modeling_qwen3_8_flash_next.copy_unscanned_to_scanned(unscanned_model, scanned_model)
    toks = jax.random.randint(jax.random.PRNGKey(6), (2, 8), 0, 128)
    loss_u, mets_u = modeling_qwen3_8_flash_next.compute_total_loss(
        unscanned_model, toks, toks, router_aux_coeff=0.001, use_mtp=True
    )
    loss_s, mets_s = modeling_qwen3_8_flash_next.compute_total_loss(
        scanned_model, toks, toks, router_aux_coeff=0.001, use_mtp=True
    )
    self.assertTrue(jnp.isfinite(loss_s))
    np.testing.assert_allclose(loss_u, loss_s, rtol=1e-5, atol=1e-5)
    for k, val_u in mets_u.items():
      np.testing.assert_allclose(val_u, mets_s[k], rtol=1e-5, atol=1e-5)

  def test_packed_document_boundary_mtp_loss(self):
    cfg = _create_mini_config()
    model = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    toks, seg_ids = jnp.array([[10, 20, 30, 40]], dtype=jnp.int32), jnp.array([[1, 1, 2, 0]], dtype=jnp.int32)
    toks_modified = jnp.array([[10, 20, 99, 40]], dtype=jnp.int32)
    _, m1 = modeling_qwen3_8_flash_next.compute_total_loss(model, toks, toks, segment_ids=seg_ids, use_mtp=True)
    _, m2 = modeling_qwen3_8_flash_next.compute_total_loss(
        model, toks_modified, toks_modified, segment_ids=seg_ids, use_mtp=True
    )
    np.testing.assert_allclose(float(m1["mtp_loss"]), float(m2["mtp_loss"]), rtol=1e-5)
    _, u1 = modeling_qwen3_8_flash_next.compute_total_loss(model, toks, toks, use_mtp=True)
    _, u2 = modeling_qwen3_8_flash_next.compute_total_loss(model, toks_modified, toks_modified, use_mtp=True)
    self.assertNotEqual(float(u1["mtp_loss"]), float(u2["mtp_loss"]))

  def test_document_boundary_next_tokens_masking(self):
    cfg = _create_mini_config()
    model = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(cfg, rngs=nnx.Rngs(params=0))
    toks = jnp.array([[10, 20, 30, 40]], dtype=jnp.int32)
    seg_ids = jnp.array([[1, 1, 2, 0]], dtype=jnp.int32)
    logits, mtp_logits = model(toks, decoder_segment_ids=seg_ids, return_mtp=True)
    self.assertEqual(logits.shape, (1, 4, 128))
    self.assertEqual(mtp_logits.shape, (1, 4, 128))
    same_seg = (seg_ids != 0) & (seg_ids == jnp.pad(seg_ids[:, 1:], ((0, 0), (0, 1))))
    self.assertTrue(bool(same_seg[0, 0]))
    self.assertFalse(bool(same_seg[0, 1]))
    self.assertFalse(bool(same_seg[0, 2]))
    self.assertFalse(bool(same_seg[0, 3]))

  def test_validations_and_error_handling(self):
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(_create_mini_config(mtp_num_layers=2), rngs=nnx.Rngs(params=0))
    model = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(_create_mini_config(), rngs=nnx.Rngs(params=0))
    toks = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    with self.assertRaises(ValueError):
      model(toks, return_mtp=True, return_hidden_states=True)

    block = modeling_qwen3_8_flash_next.Qwen3_8MTPBlock(_create_mini_config(), rngs=nnx.Rngs(params=0))
    with self.assertRaises(ValueError):
      block(jnp.zeros((2, 4, 100)), jnp.zeros((2, 4, 64)))
    with self.assertRaises(ValueError):
      block(jnp.zeros((2, 4, 256)), jnp.zeros((2, 4, 100)))

    cfg_no_aux = _create_mini_config()
    delattr(cfg_no_aux, "load_balance_loss_weight")
    model_no_aux = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(cfg_no_aux, rngs=nnx.Rngs(params=0))
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.compute_total_loss(model_no_aux, toks, toks)

    cfg_no_mtp = _create_mini_config()
    delattr(cfg_no_mtp, "mtp_loss_scaling_factor")
    model_no_mtp = modeling_qwen3_8_flash_next.Qwen3_8FlashNextModel(cfg_no_mtp, rngs=nnx.Rngs(params=0))
    with self.assertRaises(ValueError):
      modeling_qwen3_8_flash_next.compute_total_loss(model_no_mtp, toks, toks, router_aux_coeff=0.001)

  def test_qwen3_8_experts_lecun_normal_std_scales_by_fan_in_not_num_experts(self):
    """Verifies 4D gate_up_proj and 3D down_proj initializers treat expert/split dims as batch axes."""
    num_experts, emb_dim, moe_mlp_dim = 32, 256, 128
    experts = modeling_qwen3_8_flash_next.Qwen3_8Experts(
        num_experts=num_experts,
        emb_dim=emb_dim,
        moe_mlp_dim=moe_mlp_dim,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(params=123),
    )
    gu_std = float(jnp.std(experts.gate_up_proj[...]))
    dn_std = float(jnp.std(experts.down_proj[...]))
    np.testing.assert_allclose(gu_std, 1.0 / np.sqrt(emb_dim), rtol=0.1)
    np.testing.assert_allclose(dn_std, 1.0 / np.sqrt(moe_mlp_dim), rtol=0.1)


if __name__ == "__main__":
  absltest.main()

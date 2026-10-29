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
    np.testing.assert_allclose(np.array(jax_out), pt_out.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(jax_state), pt_state.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

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
    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)


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
    np.testing.assert_allclose(np.array(jax_out), pt_out.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(jax_state), pt_state.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

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
    pt_tok_mask_np = pt_tok_mask.detach().cpu().numpy() == 0.0
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
      pt_tok_mask_np = pt_tok_mask.detach().cpu().numpy() == 0.0

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

  def test_experts_shard_map_matches_reference_forward_on_cpu_mesh(self):
    if len(jax.devices()) < 4:
      self.skipTest("Needs 4 devices (e.g. XLA_FLAGS=--xla_force_host_platform_device_count=4).")
    n, d, m, e, k = 16, 32, 16, 12, 2
    experts = modeling_qwen3_8_flash_next.Qwen3_8Experts(
        num_experts=e,
        emb_dim=d,
        moe_mlp_dim=m,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(7),
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
    weight_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "fsdp", None))

    x_sharded = jax.device_put(x, token_sharding)
    idx_sharded = jax.device_put(top_k_idx, token_sharding)
    w_sharded = jax.device_put(top_k_w, token_sharding)
    experts.gate_up_proj.value = jax.device_put(experts.gate_up_proj.value, weight_sharding)
    experts.down_proj.value = jax.device_put(experts.down_proj.value, weight_sharding)

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
        np.asarray(actual_grads.gate_up_proj.value),
        np.asarray(ref_grads.gate_up_proj.value),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_grads.down_proj.value),
        np.asarray(ref_grads.down_proj.value),
        rtol=1e-5,
        atol=1e-5,
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
    for k in ("gate_up_proj", "down_proj"):
      getattr(jl.mlp.experts, k)[...] = jnp.swapaxes(to_j(getattr(pl.mlp.experts, k)), -1, -2)
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
    np.testing.assert_allclose(out_u, out_s, atol=1e-1, rtol=1e-1)

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
  for k in ("gate_up_proj", "down_proj"):
    getattr(jx_layer.mlp.experts, k).value = jnp.swapaxes(to_j(getattr(pt_layer.mlp.experts, k)), -1, -2)
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
            (512, 2560, 1280),
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


if __name__ == "__main__":
  absltest.main()

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
"""

from typing import Any
import unittest

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next  # pylint: disable=no-name-in-module
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


if __name__ == "__main__":
  absltest.main()

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


if __name__ == "__main__":
  absltest.main()

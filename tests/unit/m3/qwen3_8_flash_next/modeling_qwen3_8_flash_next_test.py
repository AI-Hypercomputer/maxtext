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
"""

from typing import Any
import unittest

from absl.testing import absltest
from flax import nnx
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


if __name__ == "__main__":
  absltest.main()

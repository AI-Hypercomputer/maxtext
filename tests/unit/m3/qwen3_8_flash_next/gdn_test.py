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

"""Unit tests for Qwen3.8-Flash-Next Gated DeltaNet (GDN) linear attention."""

from typing import Any
import unittest

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
from maxtext.m3.models.qwen3_8_flash_next import gdn  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3_8_flash_next import gdn_recurrence  # pylint: disable=no-name-in-module
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
class Qwen3_8GDNTest(absltest.TestCase):
  """Tests Qwen3.8 GDN layer and recurrence kernels against PyTorch reference."""

  def test_qwen3_8_gdn_parameter_tree_contract(self):
    """Verifies shapes and dtypes of all GDN parameters."""
    emb_dim, num_k_heads, num_v_heads = 256, 2, 4
    head_k_dim, head_v_dim = 64, 64
    val_dim = num_v_heads * head_v_dim
    conv_dim = (num_k_heads * head_k_dim) * 2 + val_dim
    mod = gdn.Qwen3_8GatedDeltaNet(
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
      gdn.Qwen3_8GatedDeltaNet(
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
      chunk_out, chunk_state = gdn_recurrence.jax_chunk_gated_delta_rule(
          q, k, v, g, beta, chunk_size=64, output_final_state=True
      )
      rec_out, rec_state = gdn_recurrence.jax_recurrent_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
      self.assertIsNotNone(chunk_state)
      self.assertIsNotNone(rec_state)
      np.testing.assert_allclose(np.array(chunk_out), np.array(rec_out), rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(np.array(chunk_state), np.array(rec_state), rtol=1e-5, atol=1e-5)

  def test_layer_mask_state_and_decode(self):
    """Verifies right-padding does not leak into state, and single decode step matches."""
    b, s_unpad, s_pad, emb_dim = 1, 16, 20, 64
    mod = gdn.Qwen3_8GatedDeltaNet(
        emb_dim=emb_dim,
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=1,
        chunk_size=64,
        output_gate_activation="sigmoid",
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    rng = np.random.default_rng(42)
    x = jnp.array(rng.standard_normal((b, s_pad, emb_dim), dtype=np.float32))
    mask = jnp.concatenate([jnp.ones((b, s_unpad)), jnp.zeros((b, s_pad - s_unpad))], axis=-1)

    _, state_pad = mod(x, attention_mask=mask, output_final_state=True)
    _, state_unpad = mod(x[:, :s_unpad, :], output_final_state=True)
    self.assertIsNotNone(state_pad)
    self.assertIsNotNone(state_unpad)
    np.testing.assert_allclose(np.array(state_pad), np.array(state_unpad), rtol=1e-5, atol=1e-5)

    x_next = x[:, s_unpad : s_unpad + 1, :]
    decode_out, decode_state = mod(x_next, initial_state=state_unpad, output_final_state=True)
    full_out, full_state = mod(x[:, : s_unpad + 1, :], output_final_state=True)
    self.assertIsNotNone(decode_state)
    self.assertIsNotNone(full_state)
    np.testing.assert_allclose(
        np.array(decode_out), np.array(full_out[:, s_unpad : s_unpad + 1, :]), rtol=1e-5, atol=1e-5
    )
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
    jax_out, jax_state = gdn_recurrence.jax_chunk_gated_delta_rule(*j_inputs, chunk_size=64, output_final_state=True)
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
    jax_gdn = gdn.Qwen3_8GatedDeltaNet(
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


if __name__ == "__main__":
  absltest.main()

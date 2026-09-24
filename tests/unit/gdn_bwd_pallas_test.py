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

"""Unit tests for gdn_bwd_pallas with canonical GDN backward pass."""

# pylint: disable=protected-access

import functools
import sys
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.configs import types as config_types
from maxtext.kernels.gdn import gdn_bwd_pallas
from maxtext.models import qwen3
from maxtext.utils import maxtext_utils


def _init_fwd_inputs(
    key: jax.Array,
    batch_size: int,
    seq_len: int,
    dim_size: int,
    num_v_heads: int,
    conv_kernel_size: int,
    with_bias: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array, jax.Array]:
  """Initializes test inputs for forward GDN pass."""
  k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
  qkv = jax.random.normal(k1, (batch_size, seq_len, dim_size), dtype=jnp.float32)
  b = jax.random.normal(k2, (batch_size, seq_len, num_v_heads), dtype=jnp.float32)
  a = jax.random.normal(k3, (batch_size, seq_len, num_v_heads), dtype=jnp.float32)
  conv_weight = jax.random.normal(k4, (conv_kernel_size, 1, dim_size), dtype=jnp.float32)
  conv_bias = jax.random.normal(k5, (dim_size,), dtype=jnp.float32) if with_bias else None
  a_log = jax.random.normal(k6, (num_v_heads,), dtype=jnp.float32)
  dt_bias = jax.random.normal(k7, (num_v_heads,), dtype=jnp.float32)
  return qkv, b, a, conv_weight, conv_bias, a_log, dt_bias


def _init_bwd_inputs(
    key: jax.Array,
    batch_size: int,
    seq_len: int,
    dim_size: int,
    num_v_heads: int,
    head_v_dim: int,
    conv_kernel_size: int,
    with_bias: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array, jax.Array, jax.Array]:
  """Initializes test inputs for backward GDN pass including output cotangent do."""
  k_fwd, k_do = jax.random.split(key, 2)
  qkv, b, a, conv_weight, conv_bias, a_log, dt_bias = _init_fwd_inputs(
      k_fwd, batch_size, seq_len, dim_size, num_v_heads, conv_kernel_size, with_bias=with_bias
  )
  do = jax.random.normal(k_do, (batch_size, seq_len, num_v_heads, head_v_dim), dtype=jnp.float32)
  return qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do


class GdnBwdPallasTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    gdn_bwd_pallas.ensure_cpu_interpret_registered()

  def test_chunk_forward_matches_jax_reference(self):
    key = jax.random.PRNGKey(42)
    chunk_size = 64
    num_kq_heads = 2
    num_v_heads = 4
    kq_head_dim = 128
    v_head_dim = 128
    repeats = num_v_heads // num_kq_heads

    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
    q = jax.random.normal(k1, (chunk_size, num_kq_heads, kq_head_dim), dtype=jnp.float32)
    k = jax.random.normal(k2, (chunk_size, num_kq_heads, kq_head_dim), dtype=jnp.float32)
    v = jax.random.normal(k3, (chunk_size, num_v_heads, v_head_dim), dtype=jnp.float32)
    b_val = jax.random.normal(k4, (chunk_size, num_v_heads), dtype=jnp.float32)
    a_val = jax.random.normal(k5, (chunk_size, num_v_heads), dtype=jnp.float32)
    a_log_val = jax.random.normal(k6, (num_v_heads,), dtype=jnp.float32)
    dt_bias_val = jax.random.normal(k7, (num_v_heads,), dtype=jnp.float32)
    state_prev = jax.random.normal(k8, (num_v_heads, kq_head_dim, v_head_dim), dtype=jnp.float32)

    out_emit, state_emit, t_inv = gdn_bwd_pallas.chunk_forward_with_tinv(
        q,
        k,
        v,
        b_val,
        a_val,
        a_log_val,
        dt_bias_val,
        state_prev,
        kq_head_dim=kq_head_dim,
        repeats=repeats,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
    )

    self.assertEqual(t_inv.shape, (num_v_heads, chunk_size, chunk_size))

    # Reference computation using qwen3.jax_chunk_gated_delta_rule
    q_4d = q[None, :, :, :]
    k_4d = k[None, :, :, :]
    v_4d = v[None, :, :, :]
    q_rep_4d = jnp.repeat(q_4d, repeats, axis=2)
    k_rep_4d = jnp.repeat(k_4d, repeats, axis=2)
    beta_3d = jax.nn.sigmoid(b_val)[None, :, :]
    log_g_3d = (-jnp.exp(a_log_val) * jax.nn.softplus(a_val + dt_bias_val))[None, :, :]
    state_4d = state_prev[None, :, :, :]

    expected_out, expected_state = qwen3.jax_chunk_gated_delta_rule(
        query=q_rep_4d,
        key=k_rep_4d,
        value=v_4d,
        g=log_g_3d,
        beta=beta_3d,
        chunk_size=chunk_size,
        initial_state=state_4d,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32,
    )

    np.testing.assert_allclose(out_emit, expected_out[0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(state_emit, expected_state[0], rtol=5e-3, atol=5e-3)

  def test_compute_forward_conv_and_states(self):
    batch_size = 1
    chunk_size = 16
    num_chunks = 2
    seq_len = num_chunks * chunk_size
    num_k_heads = 1
    num_v_heads = 2
    head_k_dim = 128
    head_v_dim = 128
    conv_kernel_size = 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias = _init_fwd_inputs(
        jax.random.PRNGKey(456), batch_size, seq_len, dim_size, num_v_heads, conv_kernel_size
    )

    qkv_conv, chunk_states, t_inv = gdn_bwd_pallas._compute_forward_conv_and_states(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=None,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
    )

    self.assertEqual(
        chunk_states.shape,
        (batch_size, num_chunks, num_v_heads, head_k_dim, head_v_dim),
    )
    self.assertEqual(
        t_inv.shape,
        (batch_size, num_chunks, num_v_heads, chunk_size, chunk_size),
    )

    conv_input = jnp.pad(qkv.astype(jnp.float32), ((0, 0), (conv_kernel_size - 1, 0), (0, 0)))
    expected_conv_out = jax.lax.conv_general_dilated(
        lhs=conv_input,
        rhs=conv_weight.astype(jnp.float32),
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=dim_size,
        precision=jax.lax.Precision.HIGHEST,
    )
    expected_conv_out = expected_conv_out + conv_bias
    expected_qkv_conv = jax.nn.silu(expected_conv_out)
    np.testing.assert_allclose(qkv_conv, expected_qkv_conv, rtol=1e-5, atol=1e-5)

  def test_decoupled_conv1d_gdn_kernel_gradient_against_autodiff(self):
    """Compares gdn_decoupled_conv1d custom VJP against JAX autodiff on pure JAX."""
    batch_size = 1
    chunk_size = 64
    num_chunks = 2
    seq_len = num_chunks * chunk_size
    num_k_heads = 2
    num_v_heads = 4
    head_k_dim = 128
    head_v_dim = 128
    conv_kernel_size = 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do = _init_bwd_inputs(
        jax.random.PRNGKey(789),
        batch_size,
        seq_len,
        dim_size,
        num_v_heads,
        head_v_dim,
        conv_kernel_size,
        with_bias=True,
    )

    # 1. Golden Reference Gradient via Autodiff on pure JAX implementation
    def loss_pure(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out, _ = gdn_bwd_pallas.pure_jax_decoupled_conv1d_gdn(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=cw_in,
          conv_bias=cb_in,
          a_log=al_in,
          dt_bias=dt_in,
          conv_state=None,
          recurrent_state=None,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          conv_kernel_size=conv_kernel_size,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=True,
      )
      return jnp.sum(out * do)

    exp_dqkv, exp_db, exp_da, exp_dcw, exp_dcb, exp_dal, exp_ddt = jax.grad(loss_pure, argnums=(0, 1, 2, 3, 4, 5, 6))(
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias
    )

    # 2. Kernel Gradients via GDN Kernel custom VJP
    def loss_gdn_kernel(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out, _ = gdn_bwd_pallas.gdn_decoupled_conv1d(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=cw_in,
          conv_bias=cb_in,
          a_log=al_in,
          dt_bias=dt_in,
          conv_state=None,
          recurrent_state=None,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          conv_kernel_size=conv_kernel_size,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=True,
          compute_dtype=jnp.float32,
      )
      return jnp.sum(out * do)

    act_dqkv, act_db, act_da, act_dcw, act_dcb, act_dal, act_ddt = jax.grad(
        loss_gdn_kernel, argnums=(0, 1, 2, 3, 4, 5, 6)
    )(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias)

    print("\n--- GDN Kernel Custom VJP vs Pure JAX Autodiff Breakdown ---")
    comparisons = [
        ("beta (d_b)", exp_db, act_db),
        ("alpha (d_a)", exp_da, act_da),
        ("a_log (d_a_log)", exp_dal, act_dal),
        ("dt_bias (d_dt_bias)", exp_ddt, act_ddt),
        ("qkv (d_qkv)", exp_dqkv, act_dqkv),
        ("conv_weight (d_conv_weight)", exp_dcw, act_dcw),
        ("conv_bias (d_conv_bias)", exp_dcb, act_dcb),
    ]

    for name, exp_g, act_g in comparisons:
      self.assertIsNotNone(act_g, f"{name} actual gradient is None")
      abs_diff = float(jnp.max(jnp.abs(exp_g - act_g)))
      rel_diff = abs_diff / (float(jnp.max(jnp.abs(exp_g))) + 1e-7)
      status = "✅ MATCH" if rel_diff < 1e-3 else "❌ DIVERGED"
      print(f"  {name:<28}: MaxAbsDiff = {abs_diff:.2e} | RelDiff =" f" {rel_diff:.2e} | {status}")
      self.assertLess(
          rel_diff,
          1e-3,
          f"{name} relative difference {rel_diff:.2e} exceeds tolerance 1e-3",
      )

    print("✅ All 7 parameter gradients match Pure JAX autodiff within 0.1% on" " CPU!")

  def test_decoupled_conv1d_gdn_kernel_conv_bias_none(self):
    """Verifies GDN kernel backward executes correctly when conv_bias is None."""
    batch_size = 1
    chunk_size = 32
    num_chunks = 2
    seq_len = num_chunks * chunk_size
    num_k_heads = 1
    num_v_heads = 2
    head_k_dim = 64
    head_v_dim = 64
    conv_kernel_size = 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    qkv, b, a, conv_weight, _, a_log, dt_bias, do = _init_bwd_inputs(
        jax.random.PRNGKey(101),
        batch_size,
        seq_len,
        dim_size,
        num_v_heads,
        head_v_dim,
        conv_kernel_size,
        with_bias=False,
    )

    def loss_fn(qkv_in, b_in, a_in, cw_in, al_in, dt_in):
      out, _ = gdn_bwd_pallas.gdn_decoupled_conv1d(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=cw_in,
          conv_bias=None,
          a_log=al_in,
          dt_bias=dt_in,
          conv_state=None,
          recurrent_state=None,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          conv_kernel_size=conv_kernel_size,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=True,
          compute_dtype=jnp.float32,
      )
      return jnp.sum(out * do)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(qkv, b, a, conv_weight, a_log, dt_bias)
    for g in grads:
      self.assertIsNotNone(g)
      self.assertFalse(np.any(np.isnan(np.array(g))))

  def test_decoupled_conv1d_gdn_kernel_multi_batch(self):
    """Verifies GDN kernel backward handles batch_size > 1."""
    batch_size = 2
    chunk_size = 32
    num_chunks = 2
    seq_len = num_chunks * chunk_size
    num_k_heads = 1
    num_v_heads = 2
    head_k_dim = 64
    head_v_dim = 64
    conv_kernel_size = 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do = _init_bwd_inputs(
        jax.random.PRNGKey(202),
        batch_size,
        seq_len,
        dim_size,
        num_v_heads,
        head_v_dim,
        conv_kernel_size,
        with_bias=True,
    )

    def loss_fn(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out, _ = gdn_bwd_pallas.gdn_decoupled_conv1d(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=cw_in,
          conv_bias=cb_in,
          a_log=al_in,
          dt_bias=dt_in,
          conv_state=None,
          recurrent_state=None,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          conv_kernel_size=conv_kernel_size,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=True,
          compute_dtype=jnp.float32,
      )
      return jnp.sum(out * do)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias)
    for g in grads:
      self.assertIsNotNone(g)
      self.assertFalse(np.any(np.isnan(np.array(g))))

  def test_chunk_state_forward_matches_chunk_forward(self):
    """Verifies chunk_state_forward matches chunk_forward_with_tinv state."""
    key = jax.random.PRNGKey(999)
    chunk_size = 64
    num_kq_heads = 2
    num_v_heads = 4
    kq_head_dim = 128
    v_head_dim = 128
    repeats = num_v_heads // num_kq_heads

    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
    q = jax.random.normal(k1, (chunk_size, num_kq_heads, kq_head_dim), dtype=jnp.float32)
    k = jax.random.normal(k2, (chunk_size, num_kq_heads, kq_head_dim), dtype=jnp.float32)
    v = jax.random.normal(k3, (chunk_size, num_v_heads, v_head_dim), dtype=jnp.float32)
    b_val = jax.random.normal(k4, (chunk_size, num_v_heads), dtype=jnp.float32)
    a_val = jax.random.normal(k5, (chunk_size, num_v_heads), dtype=jnp.float32)
    a_log_val = jax.random.normal(k6, (num_v_heads,), dtype=jnp.float32)
    dt_bias_val = jax.random.normal(k7, (num_v_heads,), dtype=jnp.float32)
    state_prev = jax.random.normal(k8, (num_v_heads, kq_head_dim, v_head_dim), dtype=jnp.float32)

    _, expected_state, t_inv = gdn_bwd_pallas.chunk_forward_with_tinv(
        q,
        k,
        v,
        b_val,
        a_val,
        a_log_val,
        dt_bias_val,
        state_prev,
        kq_head_dim=kq_head_dim,
        repeats=repeats,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
    )

    # 1. Test chunk_state_forward with t_inv=None (computes t_inv internally)
    state_uncached, t_inv_computed = gdn_bwd_pallas.chunk_state_forward(
        k=k,
        v=v,
        b_val=b_val,
        a_val=a_val,
        a_log_val=a_log_val,
        dt_bias_val=dt_bias_val,
        state_prev=state_prev,
        t_inv=None,
        repeats=repeats,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
    )
    np.testing.assert_allclose(state_uncached, expected_state, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(t_inv_computed, t_inv, rtol=1e-6, atol=1e-6)

    # 2. Test chunk_state_forward with t_inv provided
    state_cached, t_inv_returned = gdn_bwd_pallas.chunk_state_forward(
        k=k,
        v=v,
        b_val=b_val,
        a_val=a_val,
        a_log_val=a_log_val,
        dt_bias_val=dt_bias_val,
        state_prev=state_prev,
        t_inv=t_inv,
        repeats=repeats,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
    )
    np.testing.assert_allclose(state_cached, expected_state, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(t_inv_returned, t_inv, rtol=1e-6, atol=1e-6)

  def test_compute_forward_conv_and_states_with_cached_tinv(self):
    """Verifies _compute_forward_conv_and_states with cached_t_inv matches uncached version."""
    batch_size = 1
    chunk_size = 16
    num_chunks = 2
    seq_len = num_chunks * chunk_size
    num_k_heads = 1
    num_v_heads = 2
    head_k_dim = 64
    head_v_dim = 64
    conv_kernel_size = 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias = _init_fwd_inputs(
        jax.random.PRNGKey(888), batch_size, seq_len, dim_size, num_v_heads, conv_kernel_size
    )

    qkv_conv_ref, chunk_states_ref, t_inv_ref = gdn_bwd_pallas._compute_forward_conv_and_states(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=None,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
    )

    qkv_conv_cached, chunk_states_cached, t_inv_cached = gdn_bwd_pallas._compute_forward_conv_and_states(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=None,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        cached_t_inv=t_inv_ref,
    )

    np.testing.assert_allclose(qkv_conv_cached, qkv_conv_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(chunk_states_cached, chunk_states_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(t_inv_cached, t_inv_ref, rtol=1e-6, atol=1e-6)

  def test_decoupled_conv1d_gdn_kernel_bwd_with_cached_tinv_in_residuals(self):
    """Verifies _gdn_decoupled_conv1d_bwd gives identical grads with cached t_inv."""
    batch_size = 1
    chunk_size = 32
    num_chunks = 2
    seq_len = num_chunks * chunk_size
    num_k_heads = 1
    num_v_heads = 2
    head_k_dim = 64
    head_v_dim = 64
    conv_kernel_size = 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim
    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do = _init_bwd_inputs(
        jax.random.PRNGKey(777),
        batch_size,
        seq_len,
        dim_size,
        num_v_heads,
        head_v_dim,
        conv_kernel_size,
        with_bias=True,
    )

    _, chunk_states, t_inv = gdn_bwd_pallas._compute_forward_conv_and_states(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=None,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
    )

    res_none = (
        qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        None,
        None,
        None,
    )
    res_cached = (
        qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        None,
        None,
        t_inv,
    )
    res_cached_all = (
        qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        None,
        None,
        t_inv,
        chunk_states,
    )
    cotangents = (do, (None, None))

    grads_none = gdn_bwd_pallas._gdn_decoupled_conv1d_bwd(
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32,
        residuals=res_none,
        cotangents=cotangents,
    )

    grads_cached = gdn_bwd_pallas._gdn_decoupled_conv1d_bwd(
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32,
        residuals=res_cached,
        cotangents=cotangents,
    )

    grads_cached_all = gdn_bwd_pallas._gdn_decoupled_conv1d_bwd(
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32,
        residuals=res_cached_all,
        cotangents=cotangents,
    )

    for g_none, g_cached, g_cached_all in zip(grads_none, grads_cached, grads_cached_all):
      if g_none is not None and g_cached is not None:
        np.testing.assert_allclose(g_none, g_cached, rtol=1e-5, atol=1e-5)
      if g_none is not None and g_cached_all is not None:
        np.testing.assert_allclose(g_none, g_cached_all, rtol=1e-5, atol=1e-5)

  def test_run_local_gdn_decoupled_fwd_returns_cached_chunk_states(self):
    """Verifies _run_local_gdn_decoupled_fwd returns properly shaped chunk_states."""
    batch_size = 1
    chunk_size = 32
    num_chunks = 2
    seq_len = num_chunks * chunk_size
    num_k_heads = 1
    num_v_heads = 2
    head_k_dim = 64
    head_v_dim = 64
    conv_kernel_size = 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias = _init_fwd_inputs(
        jax.random.PRNGKey(101), batch_size, seq_len, dim_size, num_v_heads, conv_kernel_size
    )

    (_, _), t_inv, chunk_states = gdn_bwd_pallas._run_local_gdn_decoupled_fwd(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        conv_state=None,
        recurrent_state=None,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32,
    )

    self.assertIsNotNone(chunk_states)
    self.assertIsNotNone(t_inv)
    self.assertEqual(
        chunk_states.shape,
        (batch_size, num_chunks, num_v_heads, head_k_dim, head_v_dim),
    )
    self.assertEqual(
        t_inv.shape,
        (batch_size, num_chunks, num_v_heads, chunk_size, chunk_size),
    )

    # Verify against golden pure JAX _compute_forward_conv_and_states
    _, exp_chunk_states, exp_t_inv = gdn_bwd_pallas._compute_forward_conv_and_states(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=None,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        compute_dtype=jnp.float32,
    )
    np.testing.assert_allclose(chunk_states, exp_chunk_states, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(t_inv, exp_t_inv, rtol=1e-5, atol=1e-5)

  def test_decoupled_conv1d_gdn_kernel_gradient_with_initial_states(self):
    """Verifies custom VJP gradients when initial conv_state and recurrent_state are provided."""
    batch_size = 1
    chunk_size = 32
    num_chunks = 2
    seq_len = num_chunks * chunk_size
    num_k_heads = 1
    num_v_heads = 2
    head_k_dim = 64
    head_v_dim = 64
    conv_kernel_size = 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    key = jax.random.PRNGKey(999)
    k_in, k_cs, k_rs = jax.random.split(key, 3)
    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do = _init_bwd_inputs(
        k_in,
        batch_size,
        seq_len,
        dim_size,
        num_v_heads,
        head_v_dim,
        conv_kernel_size,
        with_bias=True,
    )
    conv_state = jax.random.normal(k_cs, (batch_size, conv_kernel_size - 1, dim_size), dtype=jnp.float32)
    recurrent_state = jax.random.normal(
        k_rs,
        (batch_size, num_v_heads, head_k_dim, head_v_dim),
        dtype=jnp.float32,
    )

    def loss_pure(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out, _ = gdn_bwd_pallas.pure_jax_decoupled_conv1d_gdn(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=cw_in,
          conv_bias=cb_in,
          a_log=al_in,
          dt_bias=dt_in,
          conv_state=conv_state,
          recurrent_state=recurrent_state,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          conv_kernel_size=conv_kernel_size,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=True,
          compute_dtype=jnp.float32,
      )
      return jnp.sum(out * do)

    def loss_gdn_kernel(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out, _ = gdn_bwd_pallas.gdn_decoupled_conv1d(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=cw_in,
          conv_bias=cb_in,
          a_log=al_in,
          dt_bias=dt_in,
          conv_state=conv_state,
          recurrent_state=recurrent_state,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          conv_kernel_size=conv_kernel_size,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=True,
          compute_dtype=jnp.float32,
      )
      return jnp.sum(out * do)

    exp_grads = jax.grad(loss_pure, argnums=(0, 1, 2, 3, 4, 5, 6))(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias)
    act_grads = jax.grad(loss_gdn_kernel, argnums=(0, 1, 2, 3, 4, 5, 6))(
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias
    )

    for exp_g, act_g in zip(exp_grads, act_grads):
      self.assertIsNotNone(act_g)
      np.testing.assert_allclose(exp_g, act_g, rtol=1e-3, atol=1e-3)

  def test_gdn_kernel_bwd_multi_group_head_parallel(self):
    """Verifies multi-group head-parallel grid dispatch matches reference."""
    batch_size = 1
    chunk_size = 32
    num_chunks = 2
    seq_len = num_chunks * chunk_size
    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 64
    head_v_dim = 64
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    key = jax.random.PRNGKey(1234)
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)

    qkv = jax.random.normal(k1, (batch_size, seq_len, dim_size), dtype=jnp.float32)
    b = jax.random.normal(k2, (batch_size, seq_len, num_v_heads), dtype=jnp.float32)
    a = jax.random.normal(k3, (batch_size, seq_len, num_v_heads), dtype=jnp.float32)
    a_log = jax.random.normal(k4, (num_v_heads,), dtype=jnp.float32)
    dt_bias = jax.random.normal(k5, (num_v_heads,), dtype=jnp.float32)
    do = jax.random.normal(k6, (batch_size, seq_len, num_v_heads, head_v_dim), dtype=jnp.float32)
    chunk_states = jax.random.normal(k7, (batch_size, num_chunks, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32)
    t_inv = jax.random.normal(k8, (batch_size, num_chunks, num_v_heads, chunk_size, chunk_size), dtype=jnp.float32)

    # 1. Dispatch with head_tile = 4 -> 2 head groups
    # pylint: disable=unbalanced-tuple-unpacking
    dy1, db1, da1, dal1, ddt1 = gdn_bwd_pallas.pallas_gdn_bwd_kernel(
        qkv_conv=qkv,
        b=b,
        a=a,
        a_log=a_log,
        dt_bias=dt_bias,
        do=do,
        chunk_states=chunk_states,
        t_inv=t_inv,
        num_v_heads=num_v_heads,
        kq_head_dim=head_k_dim,
        v_head_dim=head_v_dim,
        chunk_size=chunk_size,
        head_tile=4,
    )

    # 2. Dispatch with head_tile = 8 -> 1 head group
    # pylint: disable=unbalanced-tuple-unpacking
    dy2, db2, da2, dal2, ddt2 = gdn_bwd_pallas.pallas_gdn_bwd_kernel(
        qkv_conv=qkv,
        b=b,
        a=a,
        a_log=a_log,
        dt_bias=dt_bias,
        do=do,
        chunk_states=chunk_states,
        t_inv=t_inv,
        num_v_heads=num_v_heads,
        kq_head_dim=head_k_dim,
        v_head_dim=head_v_dim,
        chunk_size=chunk_size,
        head_tile=8,
    )

    np.testing.assert_allclose(dy1, dy2, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(db1, db2, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(da1, da2, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(dal1, dal2, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(ddt1, ddt2, rtol=1e-3, atol=1e-3)

  def test_gdn_kernel_bwd_variable_length_segment_ids_reset(self):
    """Verifies segment_ids document boundaries reset carried state gradient to prevent leakage."""
    batch_size = 1
    chunk_size = 32
    num_chunks = 2
    seq_len = num_chunks * chunk_size
    num_k_heads = 1
    num_v_heads = 2
    head_k_dim = 64
    head_v_dim = 64
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    key = jax.random.PRNGKey(5678)
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)

    qkv = jax.random.normal(k1, (batch_size, seq_len, dim_size), dtype=jnp.float32)
    b = jax.random.normal(k2, (batch_size, seq_len, num_v_heads), dtype=jnp.float32)
    a = jax.random.normal(k3, (batch_size, seq_len, num_v_heads), dtype=jnp.float32)
    a_log = jax.random.normal(k4, (num_v_heads,), dtype=jnp.float32)
    dt_bias = jax.random.normal(k5, (num_v_heads,), dtype=jnp.float32)
    chunk_states = jax.random.normal(k6, (batch_size, num_chunks, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32)
    t_inv = jax.random.normal(k7, (batch_size, num_chunks, num_v_heads, chunk_size, chunk_size), dtype=jnp.float32)

    # Only chunk 1 has non-zero incoming gradients; chunk 0 do is all zeros
    do_chunk1 = jax.random.normal(k8, (batch_size, chunk_size, num_v_heads, head_v_dim), dtype=jnp.float32)
    do_chunk0 = jnp.zeros((batch_size, chunk_size, num_v_heads, head_v_dim), dtype=jnp.float32)
    do = jnp.concatenate([do_chunk0, do_chunk1], axis=1)

    # 1. No document reset: gradient flows backwards from chunk 1 into chunk 0
    # pylint: disable=unbalanced-tuple-unpacking
    dy_no_reset, _, _, _, _ = gdn_bwd_pallas.pallas_gdn_bwd_kernel(
        qkv_conv=qkv,
        b=b,
        a=a,
        a_log=a_log,
        dt_bias=dt_bias,
        do=do,
        chunk_states=chunk_states,
        t_inv=t_inv,
        num_v_heads=num_v_heads,
        kq_head_dim=head_k_dim,
        v_head_dim=head_v_dim,
        chunk_size=chunk_size,
        segment_ids=None,
    )
    # Chunk 0 gradient is non-zero due to recurrent state carrying gradients from chunk 1
    self.assertGreater(float(jnp.max(jnp.abs(dy_no_reset[:, :chunk_size, :]))), 1e-4)

    # 2. With segment_ids boundary between chunk 0 (doc 0) and chunk 1 (doc 1)
    seg_doc0 = jnp.zeros((batch_size, chunk_size), dtype=jnp.int32)
    seg_doc1 = jnp.ones((batch_size, chunk_size), dtype=jnp.int32)
    segment_ids = jnp.concatenate([seg_doc0, seg_doc1], axis=1)

    # pylint: disable=unbalanced-tuple-unpacking
    dy_reset, _, _, _, _ = gdn_bwd_pallas.pallas_gdn_bwd_kernel(
        qkv_conv=qkv,
        b=b,
        a=a,
        a_log=a_log,
        dt_bias=dt_bias,
        do=do,
        chunk_states=chunk_states,
        t_inv=t_inv,
        num_v_heads=num_v_heads,
        kq_head_dim=head_k_dim,
        v_head_dim=head_v_dim,
        chunk_size=chunk_size,
        segment_ids=segment_ids,
    )
    # Chunk 0 gradient is strictly zero because boundary reset eliminated cross-document leakage!
    np.testing.assert_allclose(
        dy_reset[:, :chunk_size, :],
        jnp.zeros_like(dy_reset[:, :chunk_size, :]),
        atol=1e-6,
    )

  def test_decoupled_conv1d_gdn_kernel_bwd_with_head_tile(self):
    """Verifies decoupled_conv1d_gdn_bwd_kernel forwards head_tile correctly."""
    batch_size = 1
    chunk_size = 32
    num_chunks = 2
    seq_len = num_chunks * chunk_size
    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 64
    head_v_dim = 64
    conv_kernel_size = 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    key = jax.random.PRNGKey(999)
    k_in, k_cs, k_ti = jax.random.split(key, 3)
    pre_conv_qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do = _init_bwd_inputs(
        k_in,
        batch_size,
        seq_len,
        dim_size,
        num_v_heads,
        head_v_dim,
        conv_kernel_size,
        with_bias=True,
    )
    chunk_states = jax.random.normal(
        k_cs, (batch_size, num_chunks, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32
    )
    t_inv = jax.random.normal(k_ti, (batch_size, num_chunks, num_v_heads, chunk_size, chunk_size), dtype=jnp.float32)

    res1 = gdn_bwd_pallas.decoupled_conv1d_gdn_bwd_kernel(
        pre_conv_qkv=pre_conv_qkv,
        b=b,
        a=a,
        a_log=a_log,
        dt_bias=dt_bias,
        do=do,
        chunk_states=chunk_states,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        t_inv=t_inv,
        num_v_heads=num_v_heads,
        kq_head_dim=head_k_dim,
        v_head_dim=head_v_dim,
        chunk_size=chunk_size,
        head_tile=4,
    )

    res2 = gdn_bwd_pallas.decoupled_conv1d_gdn_bwd_kernel(
        pre_conv_qkv=pre_conv_qkv,
        b=b,
        a=a,
        a_log=a_log,
        dt_bias=dt_bias,
        do=do,
        chunk_states=chunk_states,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        t_inv=t_inv,
        num_v_heads=num_v_heads,
        kq_head_dim=head_k_dim,
        v_head_dim=head_v_dim,
        chunk_size=chunk_size,
        head_tile=8,
    )

    for g1, g2 in zip(res1, res2):
      if g1 is not None and g2 is not None:
        np.testing.assert_allclose(g1, g2, rtol=2e-2, atol=2.5e-1)

  def test_decoupled_conv1d_gdn_equal_heads_mha(self):
    """Verifies GDN backward pass when num_v_heads == num_k_heads (1:1 head ratio)."""
    batch_size = 1
    seq_len = 32
    num_k_heads = 2
    num_v_heads = 2
    head_k_dim = 64
    head_v_dim = 64
    conv_kernel_size = 4
    chunk_size = 16
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    key = jax.random.PRNGKey(777)
    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do = _init_bwd_inputs(
        key, batch_size, seq_len, dim_size, num_v_heads, head_v_dim, conv_kernel_size
    )

    def layer_fn(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out, _ = gdn_bwd_pallas.gdn_decoupled_conv1d(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=cw_in,
          conv_bias=cb_in,
          a_log=al_in,
          dt_bias=dt_in,
          conv_state=None,
          recurrent_state=None,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          conv_kernel_size=conv_kernel_size,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=True,
          compute_dtype=jnp.float32,
      )
      return jnp.sum(out * do)

    grads = jax.grad(layer_fn, argnums=(0, 1, 2, 3, 4, 5, 6))(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias)
    for g in grads:
      self.assertIsNotNone(g)
      self.assertFalse(np.any(np.isnan(np.array(g))))

  def test_gdn_custom_remat_policy_preserves_residuals(self):
    """Verifies that GDN forward residuals are preserved under full remat policy.

    When remat_policy='full', an outer jax.checkpoint normally recomputes all
    intermediate operations. With _get_gdn_aware_remat_policy, the forward
    Pallas kernel (_run_local_gdn_decoupled_fwd) runs exactly once and its
    residuals (chunk_states, t_inv) are saved in memory, preventing duplicate
    forward execution during backward autodiff.
    """
    batch_size = 1
    chunk_size = 32
    num_chunks = 2
    seq_len = num_chunks * chunk_size
    num_k_heads = 2
    num_v_heads = 4
    head_k_dim = 64
    head_v_dim = 64
    conv_kernel_size = 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    key = jax.random.PRNGKey(42)
    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do = _init_bwd_inputs(
        key, batch_size, seq_len, dim_size, num_v_heads, head_v_dim, conv_kernel_size
    )

    def layer_fn(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out, _ = gdn_bwd_pallas.gdn_decoupled_conv1d(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=cw_in,
          conv_bias=cb_in,
          a_log=al_in,
          dt_bias=dt_in,
          conv_state=None,
          recurrent_state=None,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          conv_kernel_size=conv_kernel_size,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=True,
          compute_dtype=jnp.float32,
      )
      return jnp.sum(out * do)

    # 1. Baseline uncheckpointed gradients
    baseline_grads = jax.grad(layer_fn, argnums=(0, 1, 2, 3, 4, 5, 6))(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias)

    # 2. Test with default full remat (policy=None)
    ckpt_full_fn = jax.checkpoint(layer_fn, policy=None)
    full_grads = jax.grad(ckpt_full_fn, argnums=(0, 1, 2, 3, 4, 5, 6))(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias)
    for g_base, g_full in zip(baseline_grads, full_grads):
      np.testing.assert_allclose(g_full, g_base, rtol=1e-4, atol=1e-4)

    # 3. Test with GDN custom remat policy (saving gdn residuals and output)
    class DummyConfig:
      use_gdn_kernel = True
      remat_policy = "custom"
      tensors_on_device = ["decoder_layer_input", "gdn"]
      tensors_to_offload = []

    save_names, offload_names = maxtext_utils.get_save_and_offload_names(DummyConfig())
    self.assertIn("gdn_core_attn_out", save_names)
    self.assertIn("gdn_chunk_states", save_names)
    self.assertIn("gdn_t_inv", save_names)

    policy = jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=save_names,
        names_which_can_be_offloaded=offload_names,
        offload_src="device",
        offload_dst="pinned_host",
    )
    ckpt_custom_fn = jax.checkpoint(layer_fn, policy=policy)
    custom_grads = jax.grad(ckpt_custom_fn, argnums=(0, 1, 2, 3, 4, 5, 6))(
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias
    )
    for g_base, g_custom in zip(baseline_grads, custom_grads):
      np.testing.assert_allclose(g_custom, g_base, rtol=1e-4, atol=1e-4)

    # 4. Verify that the JAXPR under custom remat policy retains named residuals
    jaxpr = jax.make_jaxpr(jax.grad(ckpt_custom_fn))(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias)
    jaxpr_str = str(jaxpr)
    self.assertIn("name=gdn_core_attn_out", jaxpr_str)

  def test_gdn_granular_remat_requires_gdn_kernel(self):
    """Verifies that setting gdn, gdn_conv or gdn_states to device/offload raises ValueError when use_gdn_kernel=False."""
    cfg = object.__new__(config_types.MaxTextConfig)
    object.__setattr__(cfg, "gdn", "device")
    object.__setattr__(cfg, "gdn_conv", "remat")
    object.__setattr__(cfg, "gdn_states", "remat")
    object.__setattr__(cfg, "use_gdn_kernel", False)

    with self.assertRaisesRegex(ValueError, "requires `use_gdn_kernel=True`"):
      config_types.MaxTextConfig.validate_gdn_remat_requires_kernel(cfg)

    # When use_gdn_kernel is True, the check passes
    object.__setattr__(cfg, "use_gdn_kernel", True)
    res = config_types.MaxTextConfig.validate_gdn_remat_requires_kernel(cfg)
    self.assertIs(res, cfg)

    # gdn_states alone (gdn=remat gdn_conv=remat) is also granular GDN remat and needs the kernel.
    object.__setattr__(cfg, "gdn", "remat")
    object.__setattr__(cfg, "gdn_states", "device")
    object.__setattr__(cfg, "use_gdn_kernel", False)
    with self.assertRaisesRegex(ValueError, "requires `use_gdn_kernel=True`"):
      config_types.MaxTextConfig.validate_gdn_remat_requires_kernel(cfg)
    object.__setattr__(cfg, "use_gdn_kernel", True)
    self.assertIs(config_types.MaxTextConfig.validate_gdn_remat_requires_kernel(cfg), cfg)

  def test_gdn_states_remat_names(self):
    """`gdn_states` expands to exactly the fwd-kernel outputs the Pallas bwd kernel consumes (+ CP-only m_local)."""

    class DummyConfig:
      use_gdn_kernel = True
      remat_policy = "custom"
      tensors_on_device = ["decoder_layer_input", "gdn_states"]
      tensors_to_offload = []

    save_names, offload_names = maxtext_utils.get_save_and_offload_names(DummyConfig())
    self.assertEqual(offload_names, [])
    self.assertCountEqual(
        save_names,
        ["decoder_layer_input", "gdn_states", "gdn_core_attn_out", "gdn_t_inv", "gdn_chunk_states", "gdn_m_local"],
    )
    # The other 5 `gdn` residuals must NOT be saved: that is what distinguishes gdn_states from gdn=device.
    for name in ("gdn_qkv", "gdn_b", "gdn_a", "gdn_conv_state", "gdn_recurrent_state", "gdn_conv_out", "gdn_fwd_conv"):
      self.assertNotIn(name, save_names)

    class OffloadConfig:
      use_gdn_kernel = True
      remat_policy = "custom"
      tensors_on_device = ["decoder_layer_input"]
      tensors_to_offload = ["gdn_states"]

    save_names, offload_names = maxtext_utils.get_save_and_offload_names(OffloadConfig())
    self.assertEqual(save_names, ["decoder_layer_input"])
    self.assertCountEqual(
        offload_names, ["gdn_states", "gdn_core_attn_out", "gdn_t_inv", "gdn_chunk_states", "gdn_m_local"]
    )

  def test_pallas_gdn_bwd_kernel_dht_and_dh0_against_autodiff(self):
    """Verifies pallas_gdn_bwd_kernel with non-zero d_recurrent_state (dht), return_dh0=True, and conv_state."""
    batch_size = 1
    chunk_size = 64
    num_chunks = 2
    seq_len = num_chunks * chunk_size
    num_k_heads = 2
    num_v_heads = 4
    head_k_dim = 128
    head_v_dim = 128
    conv_kernel_size = 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do = _init_bwd_inputs(
        jax.random.PRNGKey(2026),
        batch_size,
        seq_len,
        dim_size,
        num_v_heads,
        head_v_dim,
        conv_kernel_size,
        with_bias=True,
    )
    k_cs, k_h0, k_dht = jax.random.split(jax.random.PRNGKey(2027), 3)
    conv_state = jax.random.normal(k_cs, (batch_size, conv_kernel_size - 1, dim_size), dtype=jnp.float32) * 0.2
    h0 = jax.random.normal(k_h0, (batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32) * 0.2
    dht = jax.random.normal(k_dht, (batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32) * 0.2

    def loss_pure(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in, cs_in, h0_in):
      out, (_, ht) = gdn_bwd_pallas.pure_jax_decoupled_conv1d_gdn(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=cw_in,
          conv_bias=cb_in,
          a_log=al_in,
          dt_bias=dt_in,
          conv_state=cs_in,
          recurrent_state=h0_in,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          conv_kernel_size=conv_kernel_size,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=True,
      )
      return jnp.sum(out * do) + jnp.sum(ht * dht)

    exp_grads = jax.grad(loss_pure, argnums=tuple(range(9)))(
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, h0
    )
    exp_dqkv, exp_db, exp_da, exp_dcw, exp_dcb, exp_dal, exp_ddt, exp_dcs, exp_dh0 = exp_grads

    qkv_conv, chunk_states, t_inv = gdn_bwd_pallas._compute_forward_conv_and_states(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        recurrent_state=h0,
        conv_state=conv_state,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
    )
    dy_conv, d_b, d_a, d_a_log, d_dt_bias, dh0 = gdn_bwd_pallas.pallas_gdn_bwd_kernel(
        qkv_conv=qkv_conv,
        b=b,
        a=a,
        a_log=a_log,
        dt_bias=dt_bias,
        do=do,
        chunk_states=chunk_states,
        t_inv=t_inv,
        num_v_heads=num_v_heads,
        kq_head_dim=head_k_dim,
        v_head_dim=head_v_dim,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
        d_recurrent_state=dht,
        return_dh0=True,
    )
    conv_out, _ = gdn_bwd_pallas.conv1d_silu_fwd(
        qkv=qkv, conv_weight=conv_weight, conv_bias=conv_bias, kernel_size=conv_kernel_size, conv_state=conv_state
    )
    d_qkv, d_cw, d_cb, d_cs = gdn_bwd_pallas.conv1d_silu_bwd(
        qkv=qkv,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        dy=dy_conv,
        kernel_size=conv_kernel_size,
        conv_out=conv_out,
        conv_state=conv_state,
        return_d_conv_state=True,
    )

    tol = 5e-4 if jax.default_backend() == "tpu" else 1e-4
    np.testing.assert_allclose(dh0, exp_dh0, rtol=tol, atol=tol)
    np.testing.assert_allclose(d_cs, exp_dcs, rtol=tol, atol=tol)
    np.testing.assert_allclose(d_qkv, exp_dqkv, rtol=tol, atol=tol)
    np.testing.assert_allclose(d_b, exp_db, rtol=tol, atol=tol)
    np.testing.assert_allclose(d_a, exp_da, rtol=tol, atol=tol)
    np.testing.assert_allclose(d_cw, exp_dcw, rtol=tol, atol=tol)
    np.testing.assert_allclose(d_cb, exp_dcb, rtol=tol, atol=tol)
    np.testing.assert_allclose(d_a_log, exp_dal, rtol=tol, atol=tol)
    np.testing.assert_allclose(d_dt_bias, exp_ddt, rtol=tol, atol=tol)

  def test_gdn_decoupled_conv1d_cp_axis_against_single_device(self):
    """Verifies gdn_decoupled_conv1d with cp_axis_name='context' across 4 shards vs CP=1."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 devices, got {len(devices)}")

    batch_size = 1
    chunk_size = 16
    cp_size = 4
    num_chunks = cp_size * 2  # 2 chunks per rank
    seq_len = num_chunks * chunk_size
    num_k_heads = 2
    num_v_heads = 4
    head_k_dim = 64
    head_v_dim = 64
    conv_kernel_size = 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim

    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do = _init_bwd_inputs(
        jax.random.PRNGKey(2030),
        batch_size,
        seq_len,
        dim_size,
        num_v_heads,
        head_v_dim,
        conv_kernel_size,
        with_bias=True,
    )
    k_cs, k_h0, k_dcs, k_dht = jax.random.split(jax.random.PRNGKey(2031), 4)
    conv_state = jax.random.normal(k_cs, (batch_size, conv_kernel_size - 1, dim_size), dtype=jnp.float32) * 0.2
    h0 = jax.random.normal(k_h0, (batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32) * 0.2
    dcs = jax.random.normal(k_dcs, (batch_size, conv_kernel_size - 1, dim_size), dtype=jnp.float32) * 0.2
    dht = jax.random.normal(k_dht, (batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32) * 0.2

    def loss_single(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in, cs_in, h0_in):
      out, (next_cs, next_rs) = gdn_bwd_pallas.gdn_decoupled_conv1d(
          qkv_in,
          b_in,
          a_in,
          cw_in,
          cb_in,
          al_in,
          dt_in,
          cs_in,
          h0_in,
          num_k_heads,
          num_v_heads,
          head_k_dim,
          head_v_dim,
          conv_kernel_size,
          chunk_size,
          True,
          jnp.float32,
          None,
      )
      loss = jnp.sum(out * do) + jnp.sum(next_cs * dcs) + jnp.sum(next_rs * dht)
      return loss, (out, next_cs, next_rs)

    (_, (ref_out, ref_cs, ref_rs)), ref_grads = jax.value_and_grad(loss_single, argnums=tuple(range(9)), has_aux=True)(
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, h0
    )

    mesh = jax.sharding.Mesh(np.array(devices[:cp_size]), ("context",))
    P = jax.sharding.PartitionSpec

    @jax.jit
    def run_cp(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in, cs_in, h0_in):
      def loss_cp(qkv_, b_, a_, cw_, cb_, al_, dt_, cs_, h0_):
        @functools.partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(
                P(None, "context", None),
                P(None, "context", None),
                P(None, "context", None),
                P(),
                P(),
                P(),
                P(),
                P(None, None, None),
                P(None, None, None, None),
            ),
            out_specs=(
                P(None, "context", None, None),
                (P(None, None, None), P(None, None, None, None)),
            ),
            check_vma=False,
        )
        def _mapped(qkv_loc, b_loc, a_loc, cw_loc, cb_loc, al_loc, dt_loc, cs_loc, h0_loc):
          return gdn_bwd_pallas.gdn_decoupled_conv1d(
              qkv_loc,
              b_loc,
              a_loc,
              cw_loc,
              cb_loc,
              al_loc,
              dt_loc,
              cs_loc,
              h0_loc,
              num_k_heads,
              num_v_heads,
              head_k_dim,
              head_v_dim,
              conv_kernel_size,
              chunk_size,
              True,
              jnp.float32,
              "context",
          )

        out, (next_cs, next_rs) = _mapped(qkv_, b_, a_, cw_, cb_, al_, dt_, cs_, h0_)
        loss = jnp.sum(out * do) + jnp.sum(next_cs * dcs) + jnp.sum(next_rs * dht)
        return loss, (out, next_cs, next_rs)

      return jax.value_and_grad(loss_cp, argnums=tuple(range(9)), has_aux=True)(
          qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in, cs_in, h0_in
      )

    (_, (cp_out, cp_cs, cp_rs)), cp_grads = run_cp(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, conv_state, h0)

    np.testing.assert_allclose(cp_out, ref_out, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cp_cs, ref_cs, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cp_rs, ref_rs, rtol=1e-4, atol=1e-4)
    for idx_g, (g_cp, g_ref) in enumerate(zip(cp_grads, ref_grads)):
      np.testing.assert_allclose(
          g_cp,
          g_ref,
          rtol=1e-4,
          atol=1e-4,
          err_msg=f"CP=4 gradient arg {idx_g} diverged from CP=1",
      )

  def test_bwd_block_specs_and_symbolic_zero_dht_dh0_omission(self):
    """Verifies make_bwd_block_specs conditional dht/dh0 specs and SymbolicZero omission in gdn_decoupled_conv1d."""
    bwd_api = sys.modules[gdn_bwd_pallas.gdn_decoupled_conv1d.__module__]

    in_00, out_00, n_in_00, n_out_00 = gdn_bwd_pallas.make_bwd_block_specs(
        num_chunks=2,
        chunk_size=16,
        dim_size=128,
        num_v_heads=4,
        kq_head_dim=16,
        v_head_dim=16,
        has_dht=False,
        has_dh0=False,
    )
    self.assertEqual(n_in_00, 9)
    self.assertEqual(n_out_00, 5)
    self.assertLen(in_00, 9)
    self.assertLen(out_00, 5)

    _, _, n_in_10, n_out_10 = gdn_bwd_pallas.make_bwd_block_specs(
        num_chunks=2,
        chunk_size=16,
        dim_size=128,
        num_v_heads=4,
        kq_head_dim=16,
        v_head_dim=16,
        has_dht=True,
        has_dh0=False,
    )
    self.assertEqual((n_in_10, n_out_10), (10, 5))

    _, _, n_in_01, n_out_01 = gdn_bwd_pallas.make_bwd_block_specs(
        num_chunks=2,
        chunk_size=16,
        dim_size=128,
        num_v_heads=4,
        kq_head_dim=16,
        v_head_dim=16,
        has_dht=False,
        has_dh0=True,
    )
    self.assertEqual((n_in_01, n_out_01), (9, 6))

    _, _, n_in_11, n_out_11 = gdn_bwd_pallas.make_bwd_block_specs(
        num_chunks=2,
        chunk_size=16,
        dim_size=128,
        num_v_heads=4,
        kq_head_dim=16,
        v_head_dim=16,
        has_dht=True,
        has_dh0=True,
    )
    self.assertEqual((n_in_11, n_out_11), (10, 6))

    batch_size, seq_len, chunk_size = 1, 32, 16
    num_k_heads, num_v_heads = 2, 4
    head_k_dim, head_v_dim = 16, 16
    conv_kernel_size = 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim
    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do = _init_bwd_inputs(
        jax.random.PRNGKey(3030),
        batch_size,
        seq_len,
        dim_size,
        num_v_heads,
        head_v_dim,
        conv_kernel_size,
        with_bias=True,
    )
    recorded_calls = []
    orig_kernel = bwd_api.pallas_gdn_bwd_kernel

    def wrapped_kernel(*args, **kwargs):
      recorded_calls.append(
          (
              kwargs.get("d_recurrent_state") is not None,
              bool(kwargs.get("return_dh0", False)),
          )
      )
      return orig_kernel(*args, **kwargs)

    with mock.patch.object(bwd_api, "pallas_gdn_bwd_kernel", side_effect=wrapped_kernel):

      def loss_no_states(qkv_in):
        out, _ = gdn_bwd_pallas.gdn_decoupled_conv1d(
            qkv_in,
            b,
            a,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            None,
            None,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            chunk_size,
            True,
            jnp.float32,
            None,
        )
        return jnp.sum(out * do)

      _ = jax.grad(loss_no_states)(qkv)

    self.assertEqual(recorded_calls, [(False, False)])

  def _run_unpacked_solo_baseline(
      self,
      qkv: jax.Array,
      b: jax.Array,
      a: jax.Array,
      conv_weight: jax.Array,
      conv_bias: jax.Array | None,
      a_log: jax.Array,
      dt_bias: jax.Array,
      do: jax.Array,
      segment_ids: jax.Array,
      *,
      num_k_heads: int,
      num_v_heads: int,
      head_k_dim: int,
      head_v_dim: int,
      conv_kernel_size: int,
      chunk_size: int,
      use_qk_norm_in_gdn: bool = True,
  ):
    """Oracle #1: Loops over each packed document in isolation using segment_ids=None."""
    batch_size, seq_len, _ = qkv.shape
    seg_np = np.asarray(segment_ids)

    out_ref = np.zeros((batch_size, seq_len, num_v_heads, head_v_dim), dtype=np.float32)
    dqkv_ref = np.zeros_like(np.asarray(qkv), dtype=np.float32)
    db_ref = np.zeros_like(np.asarray(b), dtype=np.float32)
    da_ref = np.zeros_like(np.asarray(a), dtype=np.float32)
    dcw_ref = np.zeros_like(np.asarray(conv_weight), dtype=np.float32)
    dcb_ref = np.zeros_like(np.asarray(conv_bias), dtype=np.float32) if conv_bias is not None else None
    dal_ref = np.zeros_like(np.asarray(a_log), dtype=np.float32)
    ddt_ref = np.zeros_like(np.asarray(dt_bias), dtype=np.float32)

    @functools.lru_cache(maxsize=8)
    def get_compiled_solo_step(l_pad: int):
      @jax.jit
      def solo_step(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in, do_target):
        def solo_loss(q_a, b_a, a_a, cw_a, cb_a, al_a, dt_a):
          out_solo, _ = gdn_bwd_pallas.gdn_decoupled_conv1d(
              qkv=q_a,
              b=b_a,
              a=a_a,
              conv_weight=cw_a,
              conv_bias=cb_a,
              a_log=al_a,
              dt_bias=dt_a,
              conv_state=None,
              recurrent_state=None,
              num_k_heads=num_k_heads,
              num_v_heads=num_v_heads,
              head_k_dim=head_k_dim,
              head_v_dim=head_v_dim,
              conv_kernel_size=conv_kernel_size,
              chunk_size=chunk_size,
              use_qk_norm_in_gdn=use_qk_norm_in_gdn,
              compute_dtype=jnp.float32,
              segment_ids=None,
          )
          return (
              jnp.sum(out_solo.astype(jnp.float32) * do_target.astype(jnp.float32)),
              out_solo,
          )

        (_, out_solo), grads_solo = jax.value_and_grad(solo_loss, argnums=(0, 1, 2, 3, 4, 5, 6), has_aux=True)(
            qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in
        )
        return out_solo, grads_solo

      return solo_step

    for b_idx in range(batch_size):
      row = seg_np[b_idx]
      idx = 0
      while idx < seq_len:
        seg_val = int(row[idx])
        if seg_val <= 0:
          idx += 1
          continue
        end_idx = idx + 1
        while end_idx < seq_len and int(row[end_idx]) == seg_val:
          end_idx += 1
        l_d = end_idx - idx
        l_pad = max(((l_d + chunk_size - 1) // chunk_size) * chunk_size, chunk_size)
        pad_amt = l_pad - l_d

        qkv_d = jnp.pad(
            qkv[b_idx : b_idx + 1, idx:end_idx],
            ((0, 0), (0, pad_amt), (0, 0)),
        )
        b_d = jnp.pad(
            b[b_idx : b_idx + 1, idx:end_idx],
            ((0, 0), (0, pad_amt), (0, 0)),
        )
        a_d = jnp.pad(
            a[b_idx : b_idx + 1, idx:end_idx],
            ((0, 0), (0, pad_amt), (0, 0)),
        )
        do_d = jnp.pad(
            do[b_idx : b_idx + 1, idx:end_idx],
            ((0, 0), (0, pad_amt), (0, 0), (0, 0)),
        )

        out_solo, grads_solo = get_compiled_solo_step(l_pad)(
            qkv_d, b_d, a_d, conv_weight, conv_bias, a_log, dt_bias, do_d
        )

        out_ref[b_idx, idx:end_idx] = np.asarray(out_solo[0, :l_d], dtype=np.float32)
        dqkv_ref[b_idx, idx:end_idx] = np.asarray(grads_solo[0][0, :l_d], dtype=np.float32)
        db_ref[b_idx, idx:end_idx] = np.asarray(grads_solo[1][0, :l_d], dtype=np.float32)
        da_ref[b_idx, idx:end_idx] = np.asarray(grads_solo[2][0, :l_d], dtype=np.float32)
        dcw_ref += np.asarray(grads_solo[3], dtype=np.float32)
        if dcb_ref is not None and grads_solo[4] is not None:
          dcb_ref += np.asarray(grads_solo[4], dtype=np.float32)
        dal_ref += np.asarray(grads_solo[5], dtype=np.float32)
        ddt_ref += np.asarray(grads_solo[6], dtype=np.float32)

        idx = end_idx

    return out_ref, (dqkv_ref, db_ref, da_ref, dcw_ref, dcb_ref, dal_ref, ddt_ref)

  def _step_by_step_recurrent_packed_gdn(
      self,
      qkv: jax.Array,
      b: jax.Array,
      a: jax.Array,
      conv_weight: jax.Array,
      conv_bias: jax.Array | None,
      a_log: jax.Array,
      dt_bias: jax.Array,
      segment_ids: jax.Array,
      *,
      num_k_heads: int,
      num_v_heads: int,
      head_k_dim: int,
      head_v_dim: int,
      conv_kernel_size: int,
      use_qk_norm_in_gdn: bool = True,
  ) -> jax.Array:
    """Oracle #2: Token-by-token (C=1) sequential scan with explicit state reset at segment boundaries."""
    batch_size, _, dim_size = qkv.shape
    halo_len = conv_kernel_size - 1
    key_dim = num_k_heads * head_k_dim
    repeats = num_v_heads // num_k_heads
    scale = 1.0 / jnp.sqrt(float(head_k_dim))

    w_3d = conv_weight.astype(jnp.float32) if conv_weight.ndim == 3 else conv_weight[:, None, :].astype(jnp.float32)
    cb_f32 = conv_bias.astype(jnp.float32) if conv_bias is not None else jnp.zeros((dim_size,), dtype=jnp.float32)
    a_log_f32 = a_log.astype(jnp.float32)
    dt_bias_f32 = dt_bias.astype(jnp.float32)

    init_conv_hist = jnp.zeros((batch_size, halo_len, dim_size), dtype=jnp.float32)
    init_conv_seg = jnp.zeros((batch_size, halo_len), dtype=jnp.int32)
    init_state = jnp.zeros((batch_size, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32)
    init_active_seg = jnp.zeros((batch_size,), dtype=jnp.int32)

    def step_fn(carry, token_inputs):
      conv_hist, conv_seg, state, active_seg = carry
      qkv_t, b_t, a_t, seg_t = token_inputs
      seg_i32 = jnp.maximum(seg_t.astype(jnp.int32), 0)
      is_valid = seg_i32 > 0

      # Reset recurrent state if a new positive segment starts
      same_rec_doc = is_valid & (seg_i32 == active_seg)
      state_curr = jnp.where(same_rec_doc[:, None, None, None], state, 0.0)

      # 4-tap Conv1D with per-tap same-document check
      window_x = jnp.concatenate([conv_hist, qkv_t[:, None, :].astype(jnp.float32)], axis=1)
      window_s = jnp.concatenate([conv_seg, seg_i32[:, None]], axis=1)
      z_t = cb_f32[None, :]
      for k in range(conv_kernel_size):
        tap_same = is_valid & (window_s[:, k] == seg_i32)
        z_t = z_t + jnp.where(tap_same[:, None], window_x[:, k, :], 0.0) * w_3d[k, 0, :]
      z_t = jnp.where(is_valid[:, None], z_t, 0.0)
      qkv_conv_t = jax.nn.silu(z_t)

      q_c, k_c, v_c = jnp.split(qkv_conv_t, [key_dim, 2 * key_dim], axis=-1)
      q_t = q_c.reshape(batch_size, num_k_heads, head_k_dim)
      k_t = k_c.reshape(batch_size, num_k_heads, head_k_dim)
      v_t = v_c.reshape(batch_size, num_v_heads, head_v_dim)

      if use_qk_norm_in_gdn:
        q_t = q_t * jax.lax.rsqrt(jnp.sum(q_t * q_t, axis=-1, keepdims=True) + 1e-6)
        k_t = k_t * jax.lax.rsqrt(jnp.sum(k_t * k_t, axis=-1, keepdims=True) + 1e-6)
      q_t = jnp.repeat(q_t * scale, repeats, axis=1)
      k_t = jnp.repeat(k_t, repeats, axis=1)

      beta_t = jax.nn.sigmoid(b_t.astype(jnp.float32))
      g_t = -jnp.exp(a_log_f32)[None, :] * jax.nn.softplus(a_t.astype(jnp.float32) + dt_bias_f32[None, :])

      decay_t = jnp.exp(g_t)[:, :, None, None]
      s_decayed = state_curr * decay_t
      v_pred = jnp.einsum("bhkd,bhk->bhd", s_decayed, k_t)
      delta = (v_t - v_pred) * beta_t[:, :, None]
      s_updated = s_decayed + jnp.einsum("bhk,bhd->bhkd", k_t, delta)
      out_t = jnp.einsum("bhkd,bhk->bhd", s_updated, q_t)

      out_t = jnp.where(is_valid[:, None, None], out_t, 0.0)
      next_state = jnp.where(is_valid[:, None, None, None], s_updated, state)
      next_active_seg = jnp.where(is_valid, seg_i32, active_seg)
      next_conv_hist = window_x[:, 1:, :]
      next_conv_seg = window_s[:, 1:]
      return (next_conv_hist, next_conv_seg, next_state, next_active_seg), out_t

    xs = (
        jnp.swapaxes(qkv, 0, 1),
        jnp.swapaxes(b, 0, 1),
        jnp.swapaxes(a, 0, 1),
        jnp.swapaxes(segment_ids, 0, 1),
    )
    _, out_seq = jax.lax.scan(
        step_fn,
        (init_conv_hist, init_conv_seg, init_state, init_active_seg),
        xs,
    )
    return jnp.swapaxes(out_seq, 0, 1)

  def _assert_packed_matches_oracles(
      self,
      segment_ids: jax.Array,
      *,
      chunk_size: int = 64,
      num_k_heads: int = 2,
      num_v_heads: int = 4,
      head_k_dim: int | None = None,
      head_v_dim: int | None = None,
      conv_kernel_size: int = 4,
      dtype: jnp.dtype = jnp.float32,
      rtol: float = 2e-3,
      atol: float = 2e-3,
      check_oracle2: bool = True,
      seed: int = 2026,
  ):
    """Verifies packed forward + 7 backward gradients against Oracle #1, Oracle #2, and Pure JAX."""
    default_hd = 128 if jax.default_backend() == "tpu" else 32
    if head_k_dim is None:
      head_k_dim = default_hd
    if head_v_dim is None:
      head_v_dim = default_hd
    batch_size, seq_len = segment_ids.shape
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim
    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do = _init_bwd_inputs(
        jax.random.PRNGKey(seed),
        batch_size,
        seq_len,
        dim_size,
        num_v_heads,
        head_v_dim,
        conv_kernel_size,
        with_bias=True,
    )
    qkv = qkv.astype(dtype)
    b = b.astype(dtype)
    a = a.astype(dtype)
    do = do.astype(dtype)

    # 1. Packed Kernel Execution (Pallas on TPU / Decoupled VJP)
    def packed_loss(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out_p, states_p = gdn_bwd_pallas.gdn_decoupled_conv1d(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=cw_in,
          conv_bias=cb_in,
          a_log=al_in,
          dt_bias=dt_in,
          conv_state=None,
          recurrent_state=None,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          conv_kernel_size=conv_kernel_size,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=True,
          compute_dtype=jnp.float32,
          segment_ids=segment_ids,
      )
      return jnp.sum(out_p.astype(jnp.float32) * do.astype(jnp.float32)), (
          out_p,
          states_p,
      )

    (_, (out_packed, states_packed)), grads_packed = jax.value_and_grad(
        packed_loss, argnums=(0, 1, 2, 3, 4, 5, 6), has_aux=True
    )(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias)

    # 2. Oracle #1: Unpacked Solo Loop Reference (segment_ids=None)
    out_o1, grads_o1 = self._run_unpacked_solo_baseline(
        qkv=qkv,
        b=b,
        a=a,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        a_log=a_log,
        dt_bias=dt_bias,
        do=do,
        segment_ids=segment_ids,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
        use_qk_norm_in_gdn=True,
    )

    np.testing.assert_allclose(
        np.asarray(out_packed, dtype=np.float32),
        out_o1,
        rtol=rtol,
        atol=atol,
        err_msg="Packed forward output diverged from Oracle #1 (Unpacked Solo Baseline)",
    )
    grad_names = ("dqkv", "db", "da", "d_conv_w", "d_conv_b", "d_a_log", "d_dt_bias")
    for g_name, g_got, g_exp in zip(grad_names, grads_packed, grads_o1):
      np.testing.assert_allclose(
          np.asarray(g_got, dtype=np.float32),
          g_exp,
          rtol=rtol,
          atol=atol,
          err_msg=f"Packed gradient {g_name} diverged from Oracle #1 (Unpacked Solo Baseline)",
      )

    # 3. Oracle #2: Step-by-Step Sequential Recurrence (C=1)
    if check_oracle2:

      def rec_loss(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
        out_r = self._step_by_step_recurrent_packed_gdn(
            qkv=qkv_in,
            b=b_in,
            a=a_in,
            conv_weight=cw_in,
            conv_bias=cb_in,
            a_log=al_in,
            dt_bias=dt_in,
            segment_ids=segment_ids,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            use_qk_norm_in_gdn=True,
        )
        return jnp.sum(out_r * do.astype(jnp.float32)), out_r

      (_, out_o2), grads_o2 = jax.value_and_grad(rec_loss, argnums=(0, 1, 2, 3, 4, 5, 6), has_aux=True)(
          qkv, b, a, conv_weight, conv_bias, a_log, dt_bias
      )
      np.testing.assert_allclose(
          np.asarray(out_packed, dtype=np.float32),
          np.asarray(out_o2, dtype=np.float32),
          rtol=rtol,
          atol=atol,
          err_msg="Packed forward output diverged from Oracle #2 (Sequential Recurrence)",
      )
      for g_name, g_got, g_exp in zip(grad_names, grads_packed, grads_o2):
        np.testing.assert_allclose(
            np.asarray(g_got, dtype=np.float32),
            np.asarray(g_exp, dtype=np.float32),
            rtol=rtol,
            atol=atol,
            err_msg=f"Packed gradient {g_name} diverged from Oracle #2 (Sequential Recurrence)",
        )

    # 4. Oracle #3: Pure-JAX Chunked Delta Rule (qwen3.jax_chunk_gated_delta_rule) + conv_state parity
    def pure_jax_loss(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out_j, states_j = gdn_bwd_pallas.pure_jax_decoupled_conv1d_gdn(
          qkv=qkv_in,
          b=b_in,
          a=a_in,
          conv_weight=cw_in,
          conv_bias=cb_in,
          a_log=al_in,
          dt_bias=dt_in,
          conv_state=None,
          recurrent_state=None,
          num_k_heads=num_k_heads,
          num_v_heads=num_v_heads,
          head_k_dim=head_k_dim,
          head_v_dim=head_v_dim,
          conv_kernel_size=conv_kernel_size,
          chunk_size=chunk_size,
          use_qk_norm_in_gdn=True,
          compute_dtype=dtype,
          segment_ids=segment_ids,
      )
      return jnp.sum(out_j.astype(jnp.float32) * do.astype(jnp.float32)), (
          out_j,
          states_j,
      )

    (_, (out_o3, states_o3)), grads_o3 = jax.value_and_grad(pure_jax_loss, argnums=(0, 1, 2, 3, 4, 5, 6), has_aux=True)(
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias
    )
    o3_atol = max(atol, 0.4) if jnp.dtype(dtype) == jnp.bfloat16 else atol
    np.testing.assert_allclose(
        np.asarray(out_packed, dtype=np.float32),
        np.asarray(out_o3, dtype=np.float32),
        rtol=rtol,
        atol=o3_atol,
        err_msg="Packed forward output diverged from Oracle #3 (Pure-JAX Chunked Delta Rule)",
    )
    np.testing.assert_allclose(
        np.asarray(states_packed[0], dtype=np.float32),
        np.asarray(states_o3[0], dtype=np.float32),
        rtol=rtol,
        atol=o3_atol,
        err_msg="Packed next_conv_state diverged from Oracle #3 (Pure-JAX Chunked Delta Rule)",
    )
    np.testing.assert_allclose(
        np.asarray(states_packed[1], dtype=np.float32),
        np.asarray(states_o3[1], dtype=np.float32),
        rtol=rtol,
        atol=o3_atol,
        err_msg="Packed next_recurrent_state diverged from Oracle #3 (Pure-JAX Chunked Delta Rule)",
    )
    for g_name, g_got, g_exp in zip(grad_names, grads_packed, grads_o3):
      np.testing.assert_allclose(
          np.asarray(g_got, dtype=np.float32),
          np.asarray(g_exp, dtype=np.float32),
          rtol=rtol,
          atol=o3_atol,
          err_msg=f"Packed gradient {g_name} diverged from Oracle #3 (Pure-JAX Chunked Delta Rule)",
      )

  def test_sequence_packing_4seq_timeline_and_strictly_interior_sequence(self):
    """Tests the 4-sequence timeline (0..63, 64..135, 136..180, 181..255) across 4 chunks of 64."""
    seg = np.zeros((1, 256), dtype=np.int32)
    seg[0, 0:64] = 1
    seg[0, 64:136] = 2
    seg[0, 136:181] = 3  # Strictly inside Chunk 2 (128..191), touching neither boundary!
    seg[0, 181:256] = 4
    self._assert_packed_matches_oracles(jnp.asarray(seg), chunk_size=64, seed=1001)

  def test_sequence_packing_conv1d_short_sequences_and_zero_leakage(self):
    """Tests ultra-short sequences (L=1, 2, 3, 5 < K=4) and zero cross-sequence perturbation leakage."""
    seg = np.zeros((1, 64), dtype=np.int32)
    seg[0, 0:1] = 1
    seg[0, 1:3] = 2
    seg[0, 3:6] = 3
    seg[0, 6:11] = 4
    seg[0, 11:35] = 5
    seg[0, 35:64] = 6
    segment_ids = jnp.asarray(seg)
    self._assert_packed_matches_oracles(segment_ids, chunk_size=64, seed=1002)

    # Adversarial perturbation check: adding +100.0 to Seq 1..3 (0..6) must cause 0.0 change on Seq 4..6 (6..64)
    hd = 128 if jax.default_backend() == "tpu" else 32
    num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel_size = 2, 4, hd, hd, 4
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim
    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do = _init_bwd_inputs(
        jax.random.PRNGKey(1003), 1, 64, dim_size, num_v_heads, head_v_dim, conv_kernel_size
    )
    do_later = do.at[:, :6].set(0.0)

    def run_and_grad(qkv_in):
      def loss_fn(q_arg):
        out, _ = gdn_bwd_pallas.gdn_decoupled_conv1d(
            q_arg,
            b,
            a,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            None,
            None,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            64,
            True,
            jnp.float32,
            None,
            segment_ids,
        )
        return jnp.sum(out * do_later), out

      (_, out_val), dqkv_val = jax.value_and_grad(loss_fn, has_aux=True)(qkv_in)
      return out_val, dqkv_val

    out_1, dqkv_1 = run_and_grad(qkv)
    qkv_perturbed = qkv.at[:, :6, :].add(100.0)
    out_2, dqkv_2 = run_and_grad(qkv_perturbed)
    np.testing.assert_allclose(
        np.asarray(out_1[:, 6:]),
        np.asarray(out_2[:, 6:]),
        rtol=1e-6,
        atol=1e-6,
        err_msg="Perturbing earlier packed sequences leaked into later sequences' forward outputs!",
    )
    np.testing.assert_allclose(
        np.asarray(dqkv_1[:, 6:]),
        np.asarray(dqkv_2[:, 6:]),
        rtol=1e-6,
        atol=1e-6,
        err_msg="Perturbing earlier packed sequences leaked into later sequences' backward gradients!",
    )

  def test_sequence_packing_mid_and_trailing_zero_padding(self):
    """Tests mid-chunk and trailing 0-padding (lengths=[40, 37, 31] in S=128 plus gapped [1..1, 0..0, 2..2])."""
    seg = np.zeros((2, 128), dtype=np.int32)
    # Row 0: 40 + 37 + 31 = 108 valid tokens + 20 trailing 0-padding tokens (108..127)
    seg[0, 0:40] = 1
    seg[0, 40:77] = 2
    seg[0, 77:108] = 3
    # Row 1: mid-chunk 0-padding between Seq 1 and Seq 2 + trailing 0-padding
    seg[1, 0:45] = 1
    seg[1, 45:72] = 0
    seg[1, 72:115] = 2
    self._assert_packed_matches_oracles(jnp.asarray(seg), chunk_size=64, seed=1004)

  def test_sequence_packing_max_16_segments_dynamic_fuzz_no_recompilation(self):
    """Packs up to 16 segments per row with irregular boundaries and verifies zero JIT recompilation."""
    hd = 128 if jax.default_backend() == "tpu" else 32
    num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel_size, chunk_size = (
        2,
        4,
        hd,
        hd,
        4,
        64,
    )
    seq_len = 256
    dim_size = num_k_heads * head_k_dim * 2 + num_v_heads * head_v_dim
    qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do = _init_bwd_inputs(
        jax.random.PRNGKey(1005), 1, seq_len, dim_size, num_v_heads, head_v_dim, conv_kernel_size
    )

    trace_count = 0

    @jax.jit
    def compiled_step(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in, do_in, seg_in):
      nonlocal trace_count
      trace_count += 1

      def loss_fn(q_a, b_a, a_a, cw_a, cb_a, al_a, dt_a):
        out_p, _ = gdn_bwd_pallas.gdn_decoupled_conv1d(
            qkv=q_a,
            b=b_a,
            a=a_a,
            conv_weight=cw_a,
            conv_bias=cb_a,
            a_log=al_a,
            dt_bias=dt_a,
            conv_state=None,
            recurrent_state=None,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            chunk_size=chunk_size,
            use_qk_norm_in_gdn=True,
            compute_dtype=jnp.float32,
            segment_ids=seg_in,
        )
        return jnp.sum(out_p * do_in), out_p

      (_, out_val), grads_val = jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6), has_aux=True)(
          qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in
      )
      return out_val, grads_val

    # Layout 1: 16 segments in S=256 including 1-token, 2-token, 3-token, and boundary-straddling segments + trailing 0-pad
    lengths_1 = [1, 2, 3, 17, 41, 15, 29, 8, 22, 11, 19, 14, 25, 9, 18, 12]  # sum = 246 + 10 pad
    seg_1 = np.zeros((1, seq_len), dtype=np.int32)
    pos = 0
    for s_id, length in enumerate(lengths_1, start=1):
      seg_1[0, pos : pos + length] = s_id
      pos += length
    seg_1_jnp = jnp.asarray(seg_1)

    out_1, grads_1 = compiled_step(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do, seg_1_jnp)
    out_o1, grads_o1 = self._run_unpacked_solo_baseline(
        qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        do,
        seg_1_jnp,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
    )
    np.testing.assert_allclose(np.asarray(out_1), out_o1, rtol=2e-3, atol=2e-3)
    for g_got, g_exp in zip(grads_1, grads_o1):
      np.testing.assert_allclose(np.asarray(g_got), g_exp, rtol=2e-3, atol=2e-3)

    # Layout 2: Different 16-segment layout invoked on the SAME compiled function -> 0 recompilation!
    lengths_2 = [13, 27, 24, 19, 1, 3, 2, 38, 16, 21, 14, 15, 11, 20, 12, 10]  # sum = 246 + 10 pad
    seg_2 = np.zeros((1, seq_len), dtype=np.int32)
    pos = 0
    for s_id, length in enumerate(lengths_2, start=1):
      seg_2[0, pos : pos + length] = s_id
      pos += length
    seg_2_jnp = jnp.asarray(seg_2)

    out_2, grads_2 = compiled_step(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias, do, seg_2_jnp)
    self.assertEqual(trace_count, 1, "Dynamic segment_ids caused an unexpected XLA recompilation!")
    out_o2, grads_o2 = self._run_unpacked_solo_baseline(
        qkv,
        b,
        a,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        do,
        seg_2_jnp,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        chunk_size=chunk_size,
    )
    np.testing.assert_allclose(np.asarray(out_2), out_o2, rtol=2e-3, atol=2e-3)
    for g_got, g_exp in zip(grads_2, grads_o2):
      np.testing.assert_allclose(np.asarray(g_got), g_exp, rtol=2e-3, atol=2e-3)

  def test_sequence_packing_qwen3_397b_geometry(self):
    """Verifies Qwen3.5-397B-A17B layer geometry (H_k=16, H_v=64, D=128, C=64, qkv_dim=12288) with sequence packing."""
    is_tpu = jax.default_backend() == "tpu"
    seg = np.zeros((1, 128), dtype=np.int32)
    seg[0, 0:45] = 1
    seg[0, 45:92] = 2
    seg[0, 92:120] = 3
    self._assert_packed_matches_oracles(
        jnp.asarray(seg),
        chunk_size=64,
        num_k_heads=16 if is_tpu else 4,
        num_v_heads=64 if is_tpu else 16,
        head_k_dim=128 if is_tpu else 64,
        head_v_dim=128 if is_tpu else 64,
        conv_kernel_size=4,
        dtype=jnp.float32,
        rtol=2e-3,
        atol=2e-3,
        check_oracle2=not is_tpu,
        seed=1006,
    )
    if is_tpu:
      self._assert_packed_matches_oracles(
          jnp.asarray(seg),
          chunk_size=64,
          num_k_heads=16,
          num_v_heads=64,
          head_k_dim=128,
          head_v_dim=128,
          conv_kernel_size=4,
          dtype=jnp.bfloat16,
          rtol=5e-2,
          atol=1.6e-1,
          check_oracle2=False,
          seed=1006,
      )


if __name__ == "__main__":
  absltest.main()

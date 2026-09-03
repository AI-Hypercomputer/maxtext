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

"""Unit tests for hybrid_bwd_analytical_pipeline with GDN kernel backward pass."""

import functools
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

try:
  from maxtext.models import hybrid_bwd_analytical_pipeline
  from maxtext.models import qwen3
except ImportError:
  from maxtext.src.maxtext.models import hybrid_bwd_analytical_pipeline
  from maxtext.src.maxtext.models import qwen3


class HybridBwdGdnKernelPipelineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    hybrid_bwd_analytical_pipeline.ensure_cpu_interpret_registered()

  def test_chunk_forward_matches_hybrid_gdn(self):
    key = jax.random.PRNGKey(42)
    chunk_size = 64
    num_kq_heads = 2
    num_v_heads = 4
    kq_head_dim = 128
    v_head_dim = 128
    repeats = num_v_heads // num_kq_heads

    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
    q = jax.random.normal(
        k1, (chunk_size, num_kq_heads, kq_head_dim), dtype=jnp.float32
    )
    k = jax.random.normal(
        k2, (chunk_size, num_kq_heads, kq_head_dim), dtype=jnp.float32
    )
    v = jax.random.normal(
        k3, (chunk_size, num_v_heads, v_head_dim), dtype=jnp.float32
    )
    b_val = jax.random.normal(k4, (chunk_size, num_v_heads), dtype=jnp.float32)
    a_val = jax.random.normal(k5, (chunk_size, num_v_heads), dtype=jnp.float32)
    a_log_val = jax.random.normal(k6, (num_v_heads,), dtype=jnp.float32)
    dt_bias_val = jax.random.normal(k7, (num_v_heads,), dtype=jnp.float32)
    state_prev = jax.random.normal(
        k8, (num_v_heads, kq_head_dim, v_head_dim), dtype=jnp.float32
    )

    out_emit, state_emit, t_inv = (
        hybrid_bwd_analytical_pipeline.chunk_forward_with_tinv(
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
    )

    self.assertEqual(t_inv.shape, (num_v_heads, chunk_size, chunk_size))

    # Reference computation using qwen3.jax_chunk_gated_delta_rule
    q_4d = q[None, :, :, :]
    k_4d = k[None, :, :, :]
    v_4d = v[None, :, :, :]
    q_rep_4d = jnp.repeat(q_4d, repeats, axis=2)
    k_rep_4d = jnp.repeat(k_4d, repeats, axis=2)
    beta_3d = jax.nn.sigmoid(b_val)[None, :, :]
    log_g_3d = (-jnp.exp(a_log_val) * jax.nn.softplus(a_val + dt_bias_val))[
        None, :, :
    ]
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
    np.testing.assert_allclose(
        state_emit, expected_state[0], rtol=5e-3, atol=5e-3
    )

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

    key = jax.random.PRNGKey(456)
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

    qkv = jax.random.normal(
        k1, (batch_size, seq_len, dim_size), dtype=jnp.float32
    )
    b = jax.random.normal(
        k2, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    a = jax.random.normal(
        k3, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(
        k4, (conv_kernel_size, 1, dim_size), dtype=jnp.float32
    )
    conv_bias = jax.random.normal(k5, (dim_size,), dtype=jnp.float32)
    a_log = jax.random.normal(k6, (num_v_heads,), dtype=jnp.float32)
    dt_bias = jax.random.normal(k7, (num_v_heads,), dtype=jnp.float32)

    qkv_conv, chunk_states, t_inv = (
        hybrid_bwd_analytical_pipeline._compute_forward_conv_and_states(
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
    )

    self.assertEqual(
        chunk_states.shape,
        (batch_size, num_chunks, num_v_heads, head_k_dim, head_v_dim),
    )
    self.assertEqual(
        t_inv.shape,
        (batch_size, num_chunks, num_v_heads, chunk_size, chunk_size),
    )

    conv_input = jnp.pad(
        qkv.astype(jnp.float32), ((0, 0), (conv_kernel_size - 1, 0), (0, 0))
    )
    expected_conv_out = jax.lax.conv_general_dilated(
        lhs=conv_input,
        rhs=conv_weight.astype(jnp.float32),
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=dim_size,
    )
    expected_conv_out = expected_conv_out + conv_bias
    expected_qkv_conv = jax.nn.silu(expected_conv_out)
    np.testing.assert_allclose(
        qkv_conv, expected_qkv_conv, rtol=1e-5, atol=1e-5
    )

  def test_fused_conv1d_gdn_kernel_gradient_against_autodiff(self):
    """Compares hybrid_fused_conv1d_gdn custom VJP against JAX autodiff on pure JAX."""
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

    key = jax.random.PRNGKey(789)
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)

    qkv = jax.random.normal(
        k1, (batch_size, seq_len, dim_size), dtype=jnp.float32
    )
    b = jax.random.normal(
        k2, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    a = jax.random.normal(
        k3, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(
        k4, (conv_kernel_size, 1, dim_size), dtype=jnp.float32
    )
    conv_bias = jax.random.normal(k5, (dim_size,), dtype=jnp.float32)
    a_log = jax.random.normal(k6, (num_v_heads,), dtype=jnp.float32)
    dt_bias = jax.random.normal(k7, (num_v_heads,), dtype=jnp.float32)
    do = jax.random.normal(
        k8, (batch_size, seq_len, num_v_heads, head_v_dim), dtype=jnp.float32
    )

    # 1. Golden Reference Gradient via Autodiff on pure JAX implementation
    def loss_pure(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out, _ = hybrid_bwd_analytical_pipeline.pure_jax_fused_conv1d_gdn(
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

    exp_dqkv, exp_db, exp_da, exp_dcw, exp_dcb, exp_dal, exp_ddt = jax.grad(
        loss_pure, argnums=(0, 1, 2, 3, 4, 5, 6)
    )(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias)

    # 2. Kernel Gradients via GDN Kernel custom VJP
    def loss_gdn_kernel(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out, _ = (
          hybrid_bwd_analytical_pipeline.hybrid_fused_conv1d_gdn(
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
      )
      return jnp.sum(out * do)

    loss_analytical = loss_gdn_kernel

    act_dqkv, act_db, act_da, act_dcw, act_dcb, act_dal, act_ddt = jax.grad(
        loss_gdn_kernel, argnums=(0, 1, 2, 3, 4, 5, 6)
    )(qkv, b, a, conv_weight, conv_bias, a_log, dt_bias)

    print(
        "\n--- GDN Kernel Custom VJP vs Pure JAX Autodiff Breakdown ---"
    )
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
      print(
          f"  {name:<28}: MaxAbsDiff = {abs_diff:.2e} | RelDiff ="
          f" {rel_diff:.2e} | {status}"
      )
      self.assertLess(
          rel_diff,
          1e-3,
          f"{name} relative difference {rel_diff:.2e} exceeds tolerance 1e-3",
      )

    print(
        "✅ All 7 parameter gradients match Pure JAX autodiff within 0.1% on"
        " CPU!"
    )

  def test_fused_conv1d_gdn_kernel_conv_bias_none(self):
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

    key = jax.random.PRNGKey(101)
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

    qkv = jax.random.normal(
        k1, (batch_size, seq_len, dim_size), dtype=jnp.float32
    )
    b = jax.random.normal(
        k2, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    a = jax.random.normal(
        k3, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(
        k4, (conv_kernel_size, 1, dim_size), dtype=jnp.float32
    )
    a_log = jax.random.normal(k5, (num_v_heads,), dtype=jnp.float32)
    dt_bias = jax.random.normal(k6, (num_v_heads,), dtype=jnp.float32)
    do = jax.random.normal(
        k7, (batch_size, seq_len, num_v_heads, head_v_dim), dtype=jnp.float32
    )

    def loss_fn(qkv_in, b_in, a_in, cw_in, al_in, dt_in):
      out, _ = (
          hybrid_bwd_analytical_pipeline.hybrid_fused_conv1d_gdn(
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
      )
      return jnp.sum(out * do)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(
        qkv, b, a, conv_weight, a_log, dt_bias
    )
    for g in grads:
      self.assertIsNotNone(g)
      self.assertFalse(np.any(np.isnan(np.array(g))))

  def test_fused_conv1d_gdn_kernel_multi_batch(self):
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

    key = jax.random.PRNGKey(202)
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)

    qkv = jax.random.normal(
        k1, (batch_size, seq_len, dim_size), dtype=jnp.float32
    )
    b = jax.random.normal(
        k2, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    a = jax.random.normal(
        k3, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(
        k4, (conv_kernel_size, 1, dim_size), dtype=jnp.float32
    )
    conv_bias = jax.random.normal(k5, (dim_size,), dtype=jnp.float32)
    a_log = jax.random.normal(k6, (num_v_heads,), dtype=jnp.float32)
    dt_bias = jax.random.normal(k7, (num_v_heads,), dtype=jnp.float32)
    do = jax.random.normal(
        k8, (batch_size, seq_len, num_v_heads, head_v_dim), dtype=jnp.float32
    )

    def loss_fn(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out, _ = (
          hybrid_bwd_analytical_pipeline.hybrid_fused_conv1d_gdn(
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
      )
      return jnp.sum(out * do)

    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))(
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias
    )
    for g in grads:
      self.assertIsNotNone(g)
      self.assertFalse(np.any(np.isnan(np.array(g))))

  def test_chunk_state_forward_with_cached_tinv_matches_chunk_forward(self):
    """Verifies chunk_state_forward_with_cached_tinv matches chunk_forward_with_tinv state."""
    key = jax.random.PRNGKey(999)
    chunk_size = 64
    num_kq_heads = 2
    num_v_heads = 4
    kq_head_dim = 128
    v_head_dim = 128
    repeats = num_v_heads // num_kq_heads

    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
    q = jax.random.normal(
        k1, (chunk_size, num_kq_heads, kq_head_dim), dtype=jnp.float32
    )
    k = jax.random.normal(
        k2, (chunk_size, num_kq_heads, kq_head_dim), dtype=jnp.float32
    )
    v = jax.random.normal(
        k3, (chunk_size, num_v_heads, v_head_dim), dtype=jnp.float32
    )
    b_val = jax.random.normal(k4, (chunk_size, num_v_heads), dtype=jnp.float32)
    a_val = jax.random.normal(k5, (chunk_size, num_v_heads), dtype=jnp.float32)
    a_log_val = jax.random.normal(k6, (num_v_heads,), dtype=jnp.float32)
    dt_bias_val = jax.random.normal(k7, (num_v_heads,), dtype=jnp.float32)
    state_prev = jax.random.normal(
        k8, (num_v_heads, kq_head_dim, v_head_dim), dtype=jnp.float32
    )

    _, expected_state, t_inv = (
        hybrid_bwd_analytical_pipeline.chunk_forward_with_tinv(
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
    )

    actual_state = (
        hybrid_bwd_analytical_pipeline.chunk_state_forward_with_cached_tinv(
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
    )

    np.testing.assert_allclose(
        actual_state, expected_state, rtol=1e-6, atol=1e-6
    )

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

    key = jax.random.PRNGKey(888)
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

    qkv = jax.random.normal(
        k1, (batch_size, seq_len, dim_size), dtype=jnp.float32
    )
    b = jax.random.normal(
        k2, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    a = jax.random.normal(
        k3, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(
        k4, (conv_kernel_size, 1, dim_size), dtype=jnp.float32
    )
    conv_bias = jax.random.normal(k5, (dim_size,), dtype=jnp.float32)
    a_log = jax.random.normal(k6, (num_v_heads,), dtype=jnp.float32)
    dt_bias = jax.random.normal(k7, (num_v_heads,), dtype=jnp.float32)

    qkv_conv_ref, chunk_states_ref, t_inv_ref = (
        hybrid_bwd_analytical_pipeline._compute_forward_conv_and_states(
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
    )

    qkv_conv_cached, chunk_states_cached, t_inv_cached = (
        hybrid_bwd_analytical_pipeline._compute_forward_conv_and_states(
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
    )

    np.testing.assert_allclose(
        qkv_conv_cached, qkv_conv_ref, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        chunk_states_cached, chunk_states_ref, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(t_inv_cached, t_inv_ref, rtol=1e-6, atol=1e-6)

  def test_fused_conv1d_gdn_kernel_bwd_with_cached_tinv_in_residuals(self):
    """Verifies _hybrid_fused_conv1d_gdn_bwd gives identical grads with cached t_inv."""
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

    key = jax.random.PRNGKey(777)
    k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)

    qkv = jax.random.normal(
        k1, (batch_size, seq_len, dim_size), dtype=jnp.float32
    )
    b = jax.random.normal(
        k2, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    a = jax.random.normal(
        k3, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(
        k4, (conv_kernel_size, 1, dim_size), dtype=jnp.float32
    )
    conv_bias = jax.random.normal(k5, (dim_size,), dtype=jnp.float32)
    a_log = jax.random.normal(k6, (num_v_heads,), dtype=jnp.float32)
    dt_bias = jax.random.normal(k7, (num_v_heads,), dtype=jnp.float32)
    do = jax.random.normal(
        k8, (batch_size, seq_len, num_v_heads, head_v_dim), dtype=jnp.float32
    )

    _, chunk_states, t_inv = (
        hybrid_bwd_analytical_pipeline._compute_forward_conv_and_states(
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

    grads_none = (
        hybrid_bwd_analytical_pipeline._hybrid_fused_conv1d_gdn_bwd(
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
    )

    grads_cached = (
        hybrid_bwd_analytical_pipeline._hybrid_fused_conv1d_gdn_bwd(
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
    )

    grads_cached_all = (
        hybrid_bwd_analytical_pipeline._hybrid_fused_conv1d_gdn_bwd(
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
    )

    for g_none, g_cached, g_cached_all in zip(
        grads_none, grads_cached, grads_cached_all
    ):
      if g_none is not None and g_cached is not None:
        np.testing.assert_allclose(g_none, g_cached, rtol=1e-5, atol=1e-5)
      if g_none is not None and g_cached_all is not None:
        np.testing.assert_allclose(g_none, g_cached_all, rtol=1e-5, atol=1e-5)

  def test_run_local_gdn_fused_fwd_returns_cached_chunk_states(self):
    """Verifies _run_local_gdn_fused_fwd returns properly shaped chunk_states."""
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

    key = jax.random.PRNGKey(101)
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

    qkv = jax.random.normal(
        k1, (batch_size, seq_len, dim_size), dtype=jnp.float32
    )
    b = jax.random.normal(
        k2, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    a = jax.random.normal(
        k3, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(
        k4, (conv_kernel_size, 1, dim_size), dtype=jnp.float32
    )
    conv_bias = jax.random.normal(k5, (dim_size,), dtype=jnp.float32)
    a_log = jax.random.normal(k6, (num_v_heads,), dtype=jnp.float32)
    dt_bias = jax.random.normal(k7, (num_v_heads,), dtype=jnp.float32)

    (out, states), t_inv, chunk_states = (
        hybrid_bwd_analytical_pipeline._run_local_gdn_fused_fwd(
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
    _, exp_chunk_states, exp_t_inv = (
        hybrid_bwd_analytical_pipeline._compute_forward_conv_and_states(
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
    )
    np.testing.assert_allclose(
        chunk_states, exp_chunk_states, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(t_inv, exp_t_inv, rtol=1e-5, atol=1e-5)

  def test_fused_conv1d_gdn_kernel_gradient_with_initial_states(self):
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
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = jax.random.split(key, 10)

    qkv = jax.random.normal(
        k1, (batch_size, seq_len, dim_size), dtype=jnp.float32
    )
    b = jax.random.normal(
        k2, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    a = jax.random.normal(
        k3, (batch_size, seq_len, num_v_heads), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(
        k4, (conv_kernel_size, 1, dim_size), dtype=jnp.float32
    )
    conv_bias = jax.random.normal(k5, (dim_size,), dtype=jnp.float32)
    a_log = jax.random.normal(k6, (num_v_heads,), dtype=jnp.float32)
    dt_bias = jax.random.normal(k7, (num_v_heads,), dtype=jnp.float32)
    do = jax.random.normal(
        k8, (batch_size, seq_len, num_v_heads, head_v_dim), dtype=jnp.float32
    )
    conv_state = jax.random.normal(
        k9, (batch_size, conv_kernel_size - 1, dim_size), dtype=jnp.float32
    )
    recurrent_state = jax.random.normal(
        k10,
        (batch_size, num_v_heads, head_k_dim, head_v_dim),
        dtype=jnp.float32,
    )

    def loss_pure(qkv_in, b_in, a_in, cw_in, cb_in, al_in, dt_in):
      out, _ = hybrid_bwd_analytical_pipeline.pure_jax_fused_conv1d_gdn(
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
      out, _ = (
          hybrid_bwd_analytical_pipeline.hybrid_fused_conv1d_gdn(
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
      )
      return jnp.sum(out * do)

    loss_analytical = loss_gdn_kernel

    exp_grads = jax.grad(loss_pure, argnums=(0, 1, 2, 3, 4, 5, 6))(
        qkv, b, a, conv_weight, conv_bias, a_log, dt_bias
    )
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
    chunk_states = jax.random.normal(
        k7, (batch_size, num_chunks, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32
    )
    t_inv = jax.random.normal(
        k8, (batch_size, num_chunks, num_v_heads, chunk_size, chunk_size), dtype=jnp.float32
    )

    # 1. Dispatch with head_tile = 4 -> 2 head groups
    dy1, db1, da1, dal1, ddt1 = (
        hybrid_bwd_analytical_pipeline.pallas_gdn_bwd_computation(
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
    )

    # 2. Dispatch with head_tile = 8 -> 1 head group
    dy2, db2, da2, dal2, ddt2 = (
        hybrid_bwd_analytical_pipeline.pallas_gdn_bwd_computation(
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
    chunk_states = jax.random.normal(
        k6, (batch_size, num_chunks, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32
    )
    t_inv = jax.random.normal(
        k7, (batch_size, num_chunks, num_v_heads, chunk_size, chunk_size), dtype=jnp.float32
    )

    # Only chunk 1 has non-zero incoming gradients; chunk 0 do is all zeros
    do_chunk1 = jax.random.normal(
        k8, (batch_size, chunk_size, num_v_heads, head_v_dim), dtype=jnp.float32
    )
    do_chunk0 = jnp.zeros((batch_size, chunk_size, num_v_heads, head_v_dim), dtype=jnp.float32)
    do = jnp.concatenate([do_chunk0, do_chunk1], axis=1)

    # 1. No document reset: gradient flows backwards from chunk 1 into chunk 0
    dy_no_reset, _, _, _, _ = (
        hybrid_bwd_analytical_pipeline.pallas_gdn_bwd_computation(
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
    )
    # Chunk 0 gradient is non-zero due to recurrent state carrying gradients from chunk 1
    self.assertGreater(float(jnp.max(jnp.abs(dy_no_reset[:, :chunk_size, :]))), 1e-4)

    # 2. With segment_ids boundary between chunk 0 (doc 0) and chunk 1 (doc 1)
    seg_doc0 = jnp.zeros((batch_size, chunk_size), dtype=jnp.int32)
    seg_doc1 = jnp.ones((batch_size, chunk_size), dtype=jnp.int32)
    segment_ids = jnp.concatenate([seg_doc0, seg_doc1], axis=1)

    dy_reset, _, _, _, _ = (
        hybrid_bwd_analytical_pipeline.pallas_gdn_bwd_computation(
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
    )
    # Chunk 0 gradient is strictly zero because boundary reset eliminated cross-document leakage!
    np.testing.assert_allclose(
        dy_reset[:, :chunk_size, :],
        jnp.zeros_like(dy_reset[:, :chunk_size, :]),
        atol=1e-6,
    )

  def test_fused_conv1d_gdn_kernel_bwd_with_head_tile(self):
    """Verifies pallas_fused_conv1d_gdn_bwd_computation forwards head_tile correctly."""
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
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = jax.random.split(key, 10)

    pre_conv_qkv = jax.random.normal(k1, (batch_size, seq_len, dim_size), dtype=jnp.float32)
    b = jax.random.normal(k2, (batch_size, seq_len, num_v_heads), dtype=jnp.float32)
    a = jax.random.normal(k3, (batch_size, seq_len, num_v_heads), dtype=jnp.float32)
    a_log = jax.random.normal(k4, (num_v_heads,), dtype=jnp.float32)
    dt_bias = jax.random.normal(k5, (num_v_heads,), dtype=jnp.float32)
    do = jax.random.normal(k6, (batch_size, seq_len, num_v_heads, head_v_dim), dtype=jnp.float32)
    chunk_states = jax.random.normal(
        k7, (batch_size, num_chunks, num_v_heads, head_k_dim, head_v_dim), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(k8, (conv_kernel_size, 1, dim_size), dtype=jnp.float32)
    conv_bias = jax.random.normal(k9, (dim_size,), dtype=jnp.float32)
    t_inv = jax.random.normal(
        k10, (batch_size, num_chunks, num_v_heads, chunk_size, chunk_size), dtype=jnp.float32
    )

    res1 = (
        hybrid_bwd_analytical_pipeline.pallas_fused_conv1d_gdn_bwd_computation(
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
    )

    res2 = (
        hybrid_bwd_analytical_pipeline.pallas_fused_conv1d_gdn_bwd_computation(
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
    )

    for g1, g2 in zip(res1, res2):
      if g1 is not None and g2 is not None:
        np.testing.assert_allclose(g1, g2, rtol=2e-2, atol=2.5e-1)


# Backwards compatibility alias for external imports
if __name__ != "__main__":
  class HybridBwdAnalyticalPipelineTest(HybridBwdGdnKernelPipelineTest):
    """Backwards compatibility alias for external imports."""
    __test__ = False


if __name__ == "__main__":
  absltest.main()



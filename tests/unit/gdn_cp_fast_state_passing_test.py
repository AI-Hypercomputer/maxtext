# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for gdn_cp_fast_state_passing (2-round cross-rank GDN state scan)."""

import functools
import os
import unittest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np

from maxtext.kernels.gdn.gdn_bwd import api
from maxtext.kernels.gdn.gdn_bwd import cp_gdn


class GdnCpFastStatePassingTest(unittest.TestCase):
  """Verifies bit-identity and collective/matmul counts of gdn_cp_fast_state_passing."""

  def setUp(self):
    super().setUp()
    if jax.device_count() < 4:
      self.skipTest("Requires at least 4 devices.")
    self.mesh = Mesh(np.array(jax.devices()[:4]), ("context",))

  def test_incoming_state_and_grad_state_bit_identical_and_counts(self):
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    a_loc = jnp.eye(16, dtype=jnp.float32)[None, None, :, :] + 0.05 * jax.random.normal(
        k1, (4, 2, 16, 16), dtype=jnp.float32
    )
    b_loc = jax.random.normal(k2, (4, 2, 16, 16), dtype=jnp.float32)
    dh_prev = jax.random.normal(k3, (4, 2, 16, 16), dtype=jnp.float32)
    h0 = jnp.zeros((1, 2, 16, 16), dtype=jnp.float32)

    def fwd_fn(a, b, h, fast):
      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(P("context", None, None, None), P("context", None, None, None), P(None, None, None, None)),
          out_specs=(P("context", None, None, None), P("context", None, None, None)),
          check_vma=False,
      )
      def _f(al, bl, hl):
        return cp_gdn.incoming_state(
            al,
            bl,
            hl,
            "context",
            has_initial_state=False,
            need_final_state=not fast,
            fast_state_passing=fast,
        )

      return _f(a, b, h)

    def bwd_fn(a, d, h, fast):
      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(P("context", None, None, None), P("context", None, None, None), P(None, None, None, None)),
          out_specs=(P("context", None, None, None), P("context", None, None, None)),
          check_vma=False,
      )
      def _f(al, dl, hl):
        return cp_gdn.incoming_grad_state(
            al,
            dl,
            hl,
            "context",
            has_recurrent_state=False,
            need_dh0_total=not fast,
            fast_state_passing=fast,
        )

      return _f(a, d, h)

    fwd_slow = jax.jit(functools.partial(fwd_fn, fast=False))
    fwd_fast = jax.jit(functools.partial(fwd_fn, fast=True))
    bwd_slow = jax.jit(functools.partial(bwd_fn, fast=False))
    bwd_fast = jax.jit(functools.partial(bwd_fn, fast=True))

    h_slow, _ = fwd_slow(a_loc, b_loc, h0)
    h_fast, _ = fwd_fast(a_loc, b_loc, h0)
    np.testing.assert_array_equal(np.asarray(h_slow), np.asarray(h_fast))

    dh_slow, _ = bwd_slow(a_loc, dh_prev, h0)
    dh_fast, _ = bwd_fast(a_loc, dh_prev, h0)
    np.testing.assert_array_equal(np.asarray(dh_slow), np.asarray(dh_fast))

    fwd_slow_txt = fwd_slow.lower(a_loc, b_loc, h0).as_text()
    fwd_fast_txt = fwd_fast.lower(a_loc, b_loc, h0).as_text()
    bwd_slow_txt = bwd_slow.lower(a_loc, dh_prev, h0).as_text()
    bwd_fast_txt = bwd_fast.lower(a_loc, dh_prev, h0).as_text()
    self.assertEqual(fwd_slow_txt.count("collective_permute"), 6)
    self.assertEqual(fwd_slow_txt.count("dot_general"), 6)
    self.assertEqual(fwd_slow_txt.count("all_reduce"), 2)
    self.assertEqual(fwd_fast_txt.count("collective_permute"), 4)
    self.assertEqual(fwd_fast_txt.count("dot_general"), 2)
    self.assertEqual(fwd_fast_txt.count("all_reduce"), 0)
    self.assertEqual(bwd_slow_txt.count("collective_permute"), 6)
    self.assertEqual(bwd_slow_txt.count("dot_general"), 6)
    self.assertEqual(bwd_slow_txt.count("all_reduce"), 2)
    self.assertEqual(bwd_fast_txt.count("collective_permute"), 4)
    self.assertEqual(bwd_fast_txt.count("dot_general"), 2)
    self.assertEqual(bwd_fast_txt.count("all_reduce"), 0)

  def test_gdn_decoupled_conv1d_fwd_bwd_bit_identical(self):
    B, T, Hk, Hv, K, V, D = 1, 512, 2, 4, 16, 16, 4
    C = 2 * Hk * K + Hv * V
    key = jax.random.PRNGKey(7)
    ks = jax.random.split(key, 8)
    qkv = jax.random.normal(ks[0], (B, T, C), dtype=jnp.bfloat16)
    # Weak decay and a small beta: each rank's state transition over its 128 tokens keeps entries up to about 0.96,
    # so the states passed on reach the ranks two and three hops away and a broken 2-round scan changes the output
    # and the gradients. With a_log, a and b drawn from N(0, 1) the transitions are about 1e-31 to 5e-29, the 2-hop
    # terms underflow to 0 in float32, and a wrong permutation in the 2-round scans left every output bit-identical.
    b = jnp.full((B, T, Hv), -4.0, dtype=jnp.bfloat16)
    a = jnp.full((B, T, Hv), -4.0, dtype=jnp.bfloat16)
    conv_w = jax.random.normal(ks[3], (D, 1, C), dtype=jnp.bfloat16) * 0.1
    conv_b = jax.random.normal(ks[4], (C,), dtype=jnp.bfloat16) * 0.1
    a_log = jnp.full((Hv,), -6.0, dtype=jnp.float32)
    dt_bias = jax.random.normal(ks[6], (Hv,), dtype=jnp.float32)
    dout = jax.random.normal(ks[7], (B, T, Hv, V), dtype=jnp.bfloat16)

    def run_step(qkv, b, a, conv_w, conv_b, a_log, dt_bias, dout, fast):
      @functools.partial(
          jax.shard_map,
          mesh=self.mesh,
          in_specs=(
              P(None, "context", None),
              P(None, "context", None),
              P(None, "context", None),
              P(None, None, None),
              P(None),
              P(None),
              P(None),
              P(None, "context", None, None),
          ),
          out_specs=(
              P(None, "context", None, None),
              P(None, "context", None),
              P(None, "context", None),
              P(None, "context", None),
              P(None, None, None),
              P(None),
              P(None),
              P(None),
          ),
          check_vma=False,
      )
      def _step(qkvl, bl, al, conv_wl, conv_bl, a_logl, dt_biasl, doutl):
        def loss_fn(qkvl, bl, al, conv_wl, conv_bl, a_logl, dt_biasl):
          out, _ = api.gdn_decoupled_conv1d(
              qkvl,
              bl,
              al,
              conv_wl,
              conv_bl,
              a_logl,
              dt_biasl,
              None,
              None,
              Hk,
              Hv,
              K,
              V,
              D,
              64,
              True,
              jnp.bfloat16,
              "context",
              None,
              cp_gdn.DEFAULT_PRECISION,
              False,
              False,
              False,
              False,
              False,
              fast,
          )
          return jnp.sum(out.astype(jnp.float32) * doutl.astype(jnp.float32)), out

        (_, out), grads = jax.value_and_grad(loss_fn, argnums=tuple(range(7)), has_aux=True)(
            qkvl, bl, al, conv_wl, conv_bl, a_logl, dt_biasl
        )
        return (out, *grads)

      return _step(qkv, b, a, conv_w, conv_b, a_log, dt_bias, dout)

    j_slow = jax.jit(functools.partial(run_step, fast=False))
    j_fast = jax.jit(functools.partial(run_step, fast=True))
    res_slow = j_slow(qkv, b, a, conv_w, conv_b, a_log, dt_bias, dout)
    res_fast = j_fast(qkv, b, a, conv_w, conv_b, a_log, dt_bias, dout)
    for x, y in zip(res_slow, res_fast):
      np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    txt_slow = j_slow.lower(qkv, b, a, conv_w, conv_b, a_log, dt_bias, dout).as_text()
    txt_fast = j_fast.lower(qkv, b, a, conv_w, conv_b, a_log, dt_bias, dout).as_text()
    # In this program (the final states are not used) the 2-round scans save 2 collective-permutes and 3 matmuls in
    # each of the forward and the backward scan. The generic fast loop saves the same permutes but only 2 matmuls,
    # so a disabled 2-round path, whose values are bit-identical, shows up in the matmul count.
    self.assertEqual(txt_slow.count("collective_permute") - txt_fast.count("collective_permute"), 4)
    self.assertEqual(txt_slow.count("dot_general") - txt_fast.count("dot_general"), 6)


if __name__ == "__main__":
  unittest.main()

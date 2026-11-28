# Copyright 2026 Google LLC
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

"""Unit, sharding, and TPU parity tests for the fused Pallas AttnRes TPU kernels."""

import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
from flax.core.spmd import logical_axis_rules
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.configs import pyconfig
from maxtext.kernels.attn_res import pallas_kernel as ar_kernel
from maxtext.layers import attn_res as attn_res_layer
from maxtext.layers.attn_res import _apply_attn_res, apply_attn_res_rms_norm
from maxtext.layers.normalizations import RMSNorm
from maxtext.utils import model_creation_utils
from tests.utils.test_helpers import get_test_config_path

AttnResKernelConfig = ar_kernel.AttnResKernelConfig
attn_res_pallas = ar_kernel.attn_res_pallas
attn_res_rms_norm_pallas = ar_kernel.attn_res_rms_norm_pallas
can_use_attn_res_pallas = ar_kernel.can_use_attn_res_pallas


def _rel_err(actual: jax.Array, desired: jax.Array) -> float:
  af = np.asarray(actual, dtype=np.float32)
  df = np.asarray(desired, dtype=np.float32)
  denom = float(np.linalg.norm(df))
  if denom == 0.0:
    return float(np.max(np.abs(af)))
  return float(np.linalg.norm(af - df) / denom)


class AttnResPallasKernelTest(parameterized.TestCase):
  """Verifies forward, custom_vjp, RMSNorm fusion, VMEM guards, and sharding parity."""

  def _cfg(self) -> AttnResKernelConfig:
    return AttnResKernelConfig(
        fwd_block_t=16,
        bwd_block_t=16,
        chunk_d=256,
        interpret=(jax.default_backend() == "cpu"),
    )

  def _assert_parity(self, actual: jax.Array, desired: jax.Array, *, dtype, max_rel_l2: float = 8e-3):
    af = np.asarray(actual, dtype=np.float32)
    df = np.asarray(desired, dtype=np.float32)
    if dtype == jnp.float32:
      np.testing.assert_allclose(af, df, rtol=1e-4, atol=1e-4)
    else:
      self.assertLess(_rel_err(actual, desired), max_rel_l2)

  @parameterized.product(
      dtype=[jnp.bfloat16, jnp.float32],
      use_pending=[False, True],
  )
  def test_pool_fwd_and_vjp_parity(self, dtype, use_pending):
    b, t, m, d = 2, 16, 5, 256
    eps = 1e-5
    k0, k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(100), 6)
    p = jax.random.normal(k0, (b, t, d), dtype=dtype)
    blks = jax.random.normal(k1, (b, t, m, d), dtype=dtype)
    ns = 1.0 + 0.1 * jax.random.normal(k2, (d,), dtype=jnp.float32)
    pr = jax.random.normal(k3, (d, 1), dtype=jnp.float32) * 0.25
    dy = jax.random.normal(k4, (b, t, d), dtype=dtype)
    pend = jax.random.normal(k5, (b, t, d), dtype=dtype)
    cfg = self._cfg()
    self.assertTrue(can_use_attn_res_pallas(p, blks, pending=(pend if use_pending else None), config=cfg))

    if use_pending:

      @jax.jit
      def _run_with_pending(pa, ba, n_comm, nsa, pra, pda, hpa, dya):
        out_k, vjp_k = jax.vjp(
            lambda x, blk, w1, w2, pd: attn_res_pallas(
                x,
                blk,
                n_comm,
                w1,
                w2,
                pending=pd,
                has_pending=hpa,
                epsilon=eps,
                config=cfg,
            ),
            pa,
            ba,
            nsa,
            pra,
            pda,
        )
        out_r, vjp_r = jax.vjp(
            lambda x, blk, w1, w2, pd: _apply_attn_res(
                x,
                blk,
                w2,
                w1,
                epsilon=eps,
                num_blocks=m,
                num_committed=n_comm,
                pending=pd,
                has_pending=hpa,
            ),
            pa,
            ba,
            nsa,
            pra,
            pda,
        )
        return out_k, out_r, vjp_k(dya), vjp_r(dya)

      for num_committed in (0, 2, 5):
        slot_mask = (jnp.arange(m) < num_committed)[None, None, :, None]
        blks_masked = jnp.where(slot_mask, blks, jnp.asarray(128.0, dtype=dtype))
        for hp in (False, True):
          out_k, out_r, grads_k, grads_r = _run_with_pending(
              p,
              blks_masked,
              jnp.int32(num_committed),
              ns,
              pr,
              pend,
              jnp.bool_(hp),
              dy,
          )
          self._assert_parity(out_k, out_r, dtype=dtype)
          for gk, gr in zip(grads_k, grads_r):
            self._assert_parity(gk, gr, dtype=dtype)
    else:

      @jax.jit
      def _run_no_pending(pa, ba, n_comm, nsa, pra, dya):
        out_k, vjp_k = jax.vjp(
            lambda x, blk, w1, w2: attn_res_pallas(x, blk, n_comm, w1, w2, epsilon=eps, config=cfg),
            pa,
            ba,
            nsa,
            pra,
        )
        out_r, vjp_r = jax.vjp(
            lambda x, blk, w1, w2: _apply_attn_res(
                x,
                blk,
                w2,
                w1,
                epsilon=eps,
                num_blocks=m,
                num_committed=n_comm,
            ),
            pa,
            ba,
            nsa,
            pra,
        )
        return out_k, out_r, vjp_k(dya), vjp_r(dya)

      for num_committed in (0, 1, 3, 5):
        slot_mask = (jnp.arange(m) < num_committed)[None, None, :, None]
        blks_masked = jnp.where(slot_mask, blks, jnp.asarray(128.0, dtype=dtype))
        out_k, out_r, grads_k, grads_r = _run_no_pending(p, blks_masked, jnp.int32(num_committed), ns, pr, dy)
        self._assert_parity(out_k, out_r, dtype=dtype)
        for gk, gr in zip(grads_k, grads_r):
          self._assert_parity(gk, gr, dtype=dtype)

  @parameterized.product(
      norm_dtype=[jnp.bfloat16, jnp.float32, None],
      weight_dtype=[jnp.float32, jnp.bfloat16],
  )
  def test_fused_rms_norm_matches_maxtext_rmsnorm(self, norm_dtype, weight_dtype):
    """Verifies fused Pallas output dtype, values, and VJP match MaxText `RMSNorm` (incl. `dtype=None`)."""
    b, t, m, d = 2, 16, 4, 256
    res_eps, norm_eps = 1e-5, 1e-6
    keys = jax.random.split(jax.random.PRNGKey(200), 6)
    p = jax.random.normal(keys[0], (b, t, d), dtype=jnp.bfloat16)
    blks = jax.random.normal(keys[1], (b, t, m, d), dtype=jnp.bfloat16)
    ns = (1.0 + 0.1 * jax.random.normal(keys[2], (d,), dtype=jnp.float32)).astype(weight_dtype)
    pr = (jax.random.normal(keys[3], (d, 1), dtype=jnp.float32) * 0.25).astype(weight_dtype)
    sub_scale = (1.0 + 0.1 * jax.random.normal(keys[4], (d,), dtype=jnp.float32)).astype(weight_dtype)
    expected_out_dtype = jnp.dtype(jnp.float32) if norm_dtype is None else jnp.dtype(norm_dtype)
    dy = jax.random.normal(keys[5], (b, t, d), dtype=expected_out_dtype)
    cfg = self._cfg()

    rms_norm = RMSNorm(
        num_features=d,
        epsilon=norm_eps,
        dtype=norm_dtype,
        weight_dtype=weight_dtype,
        rngs=nnx.Rngs(0),
    )
    norm_graphdef, norm_state = nnx.split(rms_norm)

    @jax.jit
    def _run_step(x, blk, w1, w2, ws, n_comm, dya):
      def _call(use_pal, pa, ba, nsa, pra, sa):
        norm_mod = nnx.merge(norm_graphdef, jax.tree.map(lambda _: sa, norm_state))
        return apply_attn_res_rms_norm(
            pa,
            ba,
            pra,
            nsa,
            norm_mod,
            epsilon=res_eps,
            num_blocks=m,
            num_committed=n_comm,
            use_pallas_kernel=use_pal,
            kernel_config=cfg if use_pal else None,
        )

      out_r, vjp_r = jax.vjp(lambda pa, ba, nsa, pra, sa: _call(False, pa, ba, nsa, pra, sa), x, blk, w1, w2, ws)
      out_k, vjp_k = jax.vjp(lambda pa, ba, nsa, pra, sa: _call(True, pa, ba, nsa, pra, sa), x, blk, w1, w2, ws)
      return out_k, out_r, vjp_k(dya), vjp_r(dya)

    for num_committed in (0, 2, 4):
      out_k, out_r, grads_k, grads_r = _run_step(p, blks, ns, pr, sub_scale, jnp.int32(num_committed), dy)

      self.assertEqual(out_k.dtype, out_r.dtype)
      self.assertEqual(out_k.dtype, expected_out_dtype)
      self._assert_parity(out_k, out_r, dtype=jnp.bfloat16)
      for gk, gr in zip(grads_k, grads_r):
        self.assertEqual(gk.dtype, gr.dtype)
        self._assert_parity(gk, gr, dtype=jnp.bfloat16, max_rel_l2=2e-2)

  def test_can_use_vmem_and_explicit_sharding_guards(self):
    """Checks VMEM budget fallback (e.g. 16 MiB TPU v4) and Explicit blocks/pending sharding guards."""
    b, t, m, d = 1, 64, 9, 7168
    p = jnp.zeros((b, t, d), dtype=jnp.bfloat16)
    blks = jnp.zeros((b, t, m, d), dtype=jnp.bfloat16)
    pend = jnp.zeros((b, t, d), dtype=jnp.bfloat16)
    scale = jnp.ones((d,), dtype=jnp.float32)

    # 60 MiB budget (v5p/v7x) accepts Kimi-K3; 12 MiB budget (16 MiB TPU v4) falls back cleanly.
    self.assertTrue(
        can_use_attn_res_pallas(
            p,
            blks,
            pending=pend,
            sublayer_norm_scale=scale,
            norm_dtype=jnp.bfloat16,
            config=AttnResKernelConfig(vmem_limit_bytes=60 << 20, interpret=True),
        )
    )
    self.assertFalse(
        can_use_attn_res_pallas(
            p,
            blks,
            pending=pend,
            sublayer_norm_scale=scale,
            norm_dtype=jnp.bfloat16,
            config=AttnResKernelConfig(vmem_limit_bytes=12 << 20, interpret=True),
        )
    )

    devices = jax.devices()
    if len(devices) < 4:
      return
    P = jax.sharding.PartitionSpec
    exp_mesh = jax.sharding.Mesh(
        np.array(devices[:4]).reshape(2, 2),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    with jax.set_mesh(exp_mesh):
      p_ok = jax.device_put(
          jnp.zeros((2, 16, 256), jnp.bfloat16), jax.sharding.NamedSharding(exp_mesh, P("data", None, None))
      )
      b_bad_spatial = jax.device_put(
          jnp.zeros((2, 16, 4, 256), jnp.bfloat16), jax.sharding.NamedSharding(exp_mesh, P(None, None, None, None))
      )
      b_bad_slot = jax.device_put(
          jnp.zeros((2, 16, 4, 256), jnp.bfloat16), jax.sharding.NamedSharding(exp_mesh, P("data", None, "tensor", None))
      )
      b_ok = jax.device_put(
          jnp.zeros((2, 16, 4, 256), jnp.bfloat16), jax.sharding.NamedSharding(exp_mesh, P("data", None, None, None))
      )
      pd_bad = jax.device_put(
          jnp.zeros((2, 16, 256), jnp.bfloat16), jax.sharding.NamedSharding(exp_mesh, P(None, None, None))
      )
      cfg = self._cfg()
      self.assertTrue(can_use_attn_res_pallas(p_ok, b_ok, config=cfg))
      self.assertFalse(can_use_attn_res_pallas(p_ok, b_bad_spatial, config=cfg))
      self.assertFalse(can_use_attn_res_pallas(p_ok, b_bad_slot, config=cfg))
      self.assertFalse(can_use_attn_res_pallas(p_ok, b_ok, pending=pd_bad, config=cfg))

  def test_explicit_unreduced_cotangent_parity(self):
    """Verifies Explicit backward reduces `unreduced` `grad_out` from downstream TP contractions."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 devices, found {len(devices)}")
    P = jax.sharding.PartitionSpec
    b, t, m, d, h = 2, 16, 4, 256, 128
    eps = 1e-5
    keys = jax.random.split(jax.random.PRNGKey(77), 7)
    p = jax.random.normal(keys[0], (b, t, d), dtype=jnp.float32)
    blks = jax.random.normal(keys[1], (b, t, m, d), dtype=jnp.float32)
    ns = 1.0 + 0.1 * jax.random.normal(keys[2], (d,), dtype=jnp.float32)
    pr = jax.random.normal(keys[3], (d, 1), dtype=jnp.float32) * 0.25
    sub_s = 1.0 + 0.1 * jax.random.normal(keys[4], (d,), dtype=jnp.float32)
    w_proj = jax.random.normal(keys[5], (d, h), dtype=jnp.float32) * 0.1
    dy_out = jax.random.normal(keys[6], (b, t, h), dtype=jnp.float32)
    cfg = self._cfg()

    def _ref_loss(pa, ba, nsa, pra, sa, wa):
      h_pool = _apply_attn_res(pa, ba, pra, nsa, epsilon=eps, num_blocks=m, num_committed=jnp.int32(3))
      r_inv = jax.lax.rsqrt(jnp.mean(jnp.square(h_pool), -1, keepdims=True) + eps)
      normed = (h_pool * r_inv) * sa
      proj = jnp.einsum("btd,dh->bth", normed, wa)
      return jnp.sum(proj * dy_out)

    ref_loss, ref_grads = jax.value_and_grad(_ref_loss, argnums=(0, 1, 2, 3, 4, 5))(p, blks, ns, pr, sub_s, w_proj)

    exp_mesh = jax.sharding.Mesh(
        np.array(devices[:4]).reshape(2, 2),
        ("fsdp", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    with jax.set_mesh(exp_mesh):
      p_e = jax.device_put(p, jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, None)))
      blks_e = jax.device_put(blks, jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, None, None)))
      ns_e = jax.device_put(ns, jax.sharding.NamedSharding(exp_mesh, P("fsdp")))
      pr_e = jax.device_put(pr, jax.sharding.NamedSharding(exp_mesh, P("fsdp", None)))
      sub_e = jax.device_put(sub_s, jax.sharding.NamedSharding(exp_mesh, P("fsdp")))
      # Column-parallel weight on "tensor" makes `grad_out` into AttnRes `unreduced={"tensor"}`.
      w_e = jax.device_put(w_proj, jax.sharding.NamedSharding(exp_mesh, P("fsdp", "tensor")))
      dy_e = jax.device_put(dy_out, jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, "tensor")))

      @jax.jit
      def _run_exp(pa, ba, nsa, pra, sa, wa, dya):
        def _pal_loss(x, blk, w1, w2, ws, wp):
          normed = attn_res_rms_norm_pallas(
              x,
              blk,
              3,
              w1,
              w2,
              ws,
              res_epsilon=eps,
              norm_epsilon=eps,
              norm_dtype=jnp.float32,
              config=cfg,
          )
          wp_gathered = jax.sharding.reshard(wp, jax.sharding.NamedSharding(exp_mesh, P(None, "tensor")))
          proj = jnp.einsum("btd,dh->bth", normed, wp_gathered)
          return jax.sharding.reshard(jnp.sum(proj * dya), P())

        return jax.value_and_grad(_pal_loss, argnums=(0, 1, 2, 3, 4, 5))(pa, ba, nsa, pra, sa, wa)

      pal_loss, pal_grads = _run_exp(p_e, blks_e, ns_e, pr_e, sub_e, w_e, dy_e)
      self._assert_parity(pal_loss, ref_loss, dtype=jnp.float32)
      for gp, gr in zip(pal_grads, ref_grads):
        self._assert_parity(gp, gr, dtype=jnp.float32)

  def test_sharded_auto_and_explicit_modes(self):
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 devices, found {len(devices)}")
    P = jax.sharding.PartitionSpec
    b, t, m, d = 2, 32, 4, 256
    eps = 1e-5
    keys = jax.random.split(jax.random.PRNGKey(99), 6)
    p = jax.random.normal(keys[0], (b, t, d), dtype=jnp.bfloat16)
    blks = jax.random.normal(keys[1], (b, t, m, d), dtype=jnp.bfloat16)
    ns = 1.0 + 0.1 * jax.random.normal(keys[2], (d,), dtype=jnp.float32)
    pr = jax.random.normal(keys[3], (d, 1), dtype=jnp.float32) * 0.25
    sub_s = 1.0 + 0.1 * jax.random.normal(keys[4], (d,), dtype=jnp.float32)
    dy = jax.random.normal(keys[5], (b, t, d), dtype=jnp.bfloat16)
    cfg = self._cfg()

    def _ref_fn(pa, ba, nsa, pra, sa):
      h = _apply_attn_res(pa, ba, pra, nsa, epsilon=eps, num_blocks=m, num_committed=jnp.int32(3))
      h_f32 = h.astype(jnp.float32)
      r_inv = jax.lax.rsqrt(jnp.mean(jnp.square(h_f32), -1, keepdims=True) + eps)
      y = (h_f32 * r_inv).astype(jnp.bfloat16)
      eff_s = sa.astype(jnp.float32).astype(jnp.bfloat16)
      return jnp.einsum("...d,d->...d", y, eff_s, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

    out_ref, vjp_ref = jax.vjp(_ref_fn, p, blks, ns, pr, sub_s)
    grads_ref = vjp_ref(dy)

    # 1. ShardMode.AUTO over a 2x2 (data, seq) mesh.
    auto_mesh = jax.sharding.Mesh(np.array(devices[:4]).reshape(2, 2), ("data", "seq"))
    act_pspec = P("data", "seq", None)
    p_auto = jax.device_put(p, jax.sharding.NamedSharding(auto_mesh, act_pspec))
    blks_auto = jax.device_put(blks, jax.sharding.NamedSharding(auto_mesh, P("data", "seq", None, None)))
    ns_auto = jax.device_put(ns, jax.sharding.NamedSharding(auto_mesh, P(None)))
    pr_auto = jax.device_put(pr, jax.sharding.NamedSharding(auto_mesh, P(None, None)))
    sub_auto = jax.device_put(sub_s, jax.sharding.NamedSharding(auto_mesh, P(None)))
    dy_auto = jax.device_put(dy, jax.sharding.NamedSharding(auto_mesh, act_pspec))

    @jax.jit
    def _run_auto(pa, ba, nsa, pra, sa, dya):
      out, vjp_fn = jax.vjp(
          lambda x, blk, w1, w2, ws: attn_res_rms_norm_pallas(
              x,
              blk,
              3,
              w1,
              w2,
              ws,
              res_epsilon=eps,
              norm_epsilon=eps,
              norm_dtype=jnp.bfloat16,
              mesh=auto_mesh,
              activation_pspec=act_pspec,
              config=cfg,
          ),
          pa,
          ba,
          nsa,
          pra,
          sa,
      )
      return out, vjp_fn(dya)

    out_a, grads_a = _run_auto(p_auto, blks_auto, ns_auto, pr_auto, sub_auto, dy_auto)
    self._assert_parity(out_a, out_ref, dtype=jnp.bfloat16)
    for ga, gr in zip(grads_a, grads_ref):
      self._assert_parity(ga, gr, dtype=jnp.bfloat16, max_rel_l2=1e-2)

    # 2. ShardMode.EXPLICIT over a 2x2 Explicit mesh.
    exp_mesh = jax.sharding.Mesh(
        np.array(devices[:4]).reshape(2, 2),
        ("data", "seq"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    with jax.set_mesh(exp_mesh):
      p_exp = jax.device_put(p, jax.sharding.NamedSharding(exp_mesh, act_pspec))
      blks_exp = jax.device_put(blks, jax.sharding.NamedSharding(exp_mesh, P("data", "seq", None, None)))
      ns_exp = jax.device_put(ns, jax.sharding.NamedSharding(exp_mesh, P(None)))
      pr_exp = jax.device_put(pr, jax.sharding.NamedSharding(exp_mesh, P(None, None)))
      sub_exp = jax.device_put(sub_s, jax.sharding.NamedSharding(exp_mesh, P(None)))
      dy_exp = jax.device_put(dy, jax.sharding.NamedSharding(exp_mesh, act_pspec))

      @jax.jit
      def _run_exp(pa, ba, nsa, pra, sa, dya):
        out, vjp_fn = jax.vjp(
            lambda x, blk, w1, w2, ws: attn_res_rms_norm_pallas(
                x,
                blk,
                3,
                w1,
                w2,
                ws,
                res_epsilon=eps,
                norm_epsilon=eps,
                norm_dtype=jnp.bfloat16,
                config=cfg,
            ),
            pa,
            ba,
            nsa,
            pra,
            sa,
        )
        return out, vjp_fn(dya)

      out_e, grads_e = _run_exp(p_exp, blks_exp, ns_exp, pr_exp, sub_exp, dy_exp)
      self._assert_parity(out_e, out_ref, dtype=jnp.bfloat16)
      for ge, gr in zip(grads_e, grads_ref):
        self._assert_parity(ge, gr, dtype=jnp.bfloat16, max_rel_l2=1e-2)

  @parameterized.parameters(False, True)
  def test_kimi_k3_model_fwd_and_bwd_parity(self, scan_layers: bool):
    """End-to-end 6-layer Kimi-K3 parity with `use_attn_res_pallas_kernel=True` vs `False`."""
    seq_len = 16
    emb_dim = 128
    base_kwargs = {
        "override_model_config": True,
        "base_emb_dim": emb_dim,
        "base_num_decoder_layers": 6,
        "full_attn_layers": [3, 5],
        "base_num_query_heads": 4,
        "base_num_kv_heads": 4,
        "head_dim": 32,
        "vocab_size": 128,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "base_mlp_dim": 128,
        "base_moe_mlp_dim": 64,
        "moe_intermediate_size": 64,
        "shared_intermediate_size": 64,
        "shared_experts": 1,
        "num_shared_experts": 1,
        "routed_expert_hidden_size": 64,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_expert_input_dim": 64,
        "moe_router_input_dim": emb_dim,
        "attn_res_block_size": 2,
        "megablox": False,
        "max_target_length": seq_len,
        "max_position_embeddings": seq_len,
        "original_max_position_embeddings": seq_len,
        "max_prefill_predict_length": seq_len // 2,
        "per_device_batch_size": 1,
        "dtype": "float32",
        "weight_dtype": "float32",
        "matmul_precision": "highest",
        "packing": False,
        "enable_checkpointing": False,
        "scan_layers": scan_layers,
    }
    cfg_ref = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        model_name="kimi-k3",
        use_attn_res_pallas_kernel=False,
        **base_kwargs,
    )
    cfg_pal = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        model_name="kimi-k3",
        use_attn_res_pallas_kernel=True,
        **base_kwargs,
    )
    devices = jax.devices()[:1]
    model_ref = model_creation_utils.from_config(cfg_ref, devices=devices, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    model_pal = model_creation_utils.from_config(cfg_pal, devices=devices, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))

    tokens = (jnp.arange(seq_len, dtype=jnp.int32).reshape(1, seq_len) * 7 % 113 + 1).astype(jnp.int32)
    positions = jnp.arange(seq_len, dtype=jnp.int32).reshape(1, seq_len)

    @nnx.jit
    def _step(model, tok, pos):
      def _loss_and_logits(m):
        logits = m(tok, pos, model_mode=MODEL_MODE_TRAIN, enable_dropout=False)
        loss = jnp.mean(jax.nn.log_softmax(logits.astype(jnp.float32))[..., 0])
        return loss, logits

      return nnx.value_and_grad(_loss_and_logits, has_aux=True)(model)

    (_, logits_ref), grads_ref = _step(model_ref, tokens, positions)
    test_k_cfg = AttnResKernelConfig(
        fwd_block_t=16,
        bwd_block_t=16,
        chunk_d=128,
        interpret=(jax.default_backend() == "cpu"),
    )
    with mock.patch.object(attn_res_layer, "_DEFAULT_PALLAS_CONFIG", test_k_cfg):
      with (
          mock.patch.object(ar_kernel, "attn_res_rms_norm_pallas", wraps=ar_kernel.attn_res_rms_norm_pallas) as spy_norm,
          mock.patch.object(ar_kernel, "attn_res_pallas", wraps=ar_kernel.attn_res_pallas) as spy_pool,
      ):
        (_, logits_pal), grads_pal = _step(model_pal, tokens, positions)
        self.assertGreater(spy_norm.call_count, 0)
        self.assertGreater(spy_pool.call_count, 0)

    np.testing.assert_allclose(np.asarray(logits_pal), np.asarray(logits_ref), rtol=2e-4, atol=2e-4)
    flat_ref = dict(nnx.to_flat_state(nnx.state(grads_ref, nnx.Param)))
    flat_pal = dict(nnx.to_flat_state(nnx.state(grads_pal, nnx.Param)))
    self.assertEqual(set(flat_ref.keys()), set(flat_pal.keys()))
    for path, g_ref_var in flat_ref.items():
      g_pal_val = np.asarray(flat_pal[path].get_value(), dtype=np.float32)
      g_ref_val = np.asarray(g_ref_var.get_value(), dtype=np.float32)
      np.testing.assert_allclose(g_pal_val, g_ref_val, rtol=5e-4, atol=5e-4, err_msg=f"Mismatch at {path}")

  def test_fsdp_sharded_weights_auto_and_explicit_parity(self):
    """Verifies FSDP-sharded AttnRes weights gather/reduce exact gradients in Auto and Explicit modes."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 devices, found {len(devices)}")
    P = jax.sharding.PartitionSpec
    m, d = 4, 256
    eps = 1e-5
    cfg = self._cfg()

    def _ref_fn(pa, ba, nsa, pra, sa):
      h = _apply_attn_res(pa, ba, pra, nsa, epsilon=eps, num_blocks=m, num_committed=jnp.int32(3))
      h_f32 = h.astype(jnp.float32)
      r_inv = jax.lax.rsqrt(jnp.mean(jnp.square(h_f32), -1, keepdims=True) + eps)
      y = (h_f32 * r_inv).astype(jnp.float32)
      return jnp.einsum("...d,d->...d", y, sa)

    for batch, act_pspec in [(4, P("fsdp", None, None)), (1, P(None, None, None))]:
      t = 16
      keys = jax.random.split(jax.random.PRNGKey(300 + batch), 6)
      p = jax.random.normal(keys[0], (batch, t, d), dtype=jnp.float32)
      blks = jax.random.normal(keys[1], (batch, t, m, d), dtype=jnp.float32)
      ns = 1.0 + 0.1 * jax.random.normal(keys[2], (d,), dtype=jnp.float32)
      pr = jax.random.normal(keys[3], (d, 1), dtype=jnp.float32) * 0.25
      sub_s = 1.0 + 0.1 * jax.random.normal(keys[4], (d,), dtype=jnp.float32)
      dy = jax.random.normal(keys[5], (batch, t, d), dtype=jnp.float32)
      out_ref, vjp_ref = jax.vjp(_ref_fn, p, blks, ns, pr, sub_s)
      grads_ref = vjp_ref(dy)

      # 1. ShardMode.AUTO with FSDP-sharded proj_weight P("fsdp", None) and norm weights P("fsdp",).
      auto_mesh = jax.sharding.Mesh(np.array(devices[:4]), ("fsdp",))
      blk_pspec = P(act_pspec[0], None, None, None)
      p_a = jax.device_put(p, jax.sharding.NamedSharding(auto_mesh, act_pspec))
      blks_a = jax.device_put(blks, jax.sharding.NamedSharding(auto_mesh, blk_pspec))
      ns_a = jax.device_put(ns, jax.sharding.NamedSharding(auto_mesh, P("fsdp")))
      pr_a = jax.device_put(pr, jax.sharding.NamedSharding(auto_mesh, P("fsdp", None)))
      sub_a = jax.device_put(sub_s, jax.sharding.NamedSharding(auto_mesh, P("fsdp")))
      dy_a = jax.device_put(dy, jax.sharding.NamedSharding(auto_mesh, act_pspec))

      @jax.jit
      def _run_auto(pa, ba, nsa, pra, sa, dya, _mesh=auto_mesh, _pspec=act_pspec):
        out, vjp_fn = jax.vjp(
            lambda x, blk, w1, w2, ws: attn_res_rms_norm_pallas(
                x,
                blk,
                3,
                w1,
                w2,
                ws,
                res_epsilon=eps,
                norm_epsilon=eps,
                norm_dtype=jnp.float32,
                mesh=_mesh,
                activation_pspec=_pspec,
                config=cfg,
            ),
            pa,
            ba,
            nsa,
            pra,
            sa,
        )
        return out, vjp_fn(dya)

      with jax.set_mesh(auto_mesh):
        out_a, grads_a = _run_auto(p_a, blks_a, ns_a, pr_a, sub_a, dy_a)
      self._assert_parity(out_a, out_ref, dtype=jnp.float32)
      for ga, gr in zip(grads_a, grads_ref):
        self._assert_parity(ga, gr, dtype=jnp.float32)

      # 2. ShardMode.EXPLICIT with FSDP-sharded proj_weight P("fsdp", None) and norm weights P("fsdp",).
      exp_mesh = jax.sharding.Mesh(
          np.array(devices[:4]),
          ("fsdp",),
          axis_types=(jax.sharding.AxisType.Explicit,),
      )
      with jax.set_mesh(exp_mesh):
        p_e = jax.device_put(p, jax.sharding.NamedSharding(exp_mesh, act_pspec))
        blks_e = jax.device_put(blks, jax.sharding.NamedSharding(exp_mesh, blk_pspec))
        ns_e = jax.device_put(ns, jax.sharding.NamedSharding(exp_mesh, P("fsdp")))
        pr_e = jax.device_put(pr, jax.sharding.NamedSharding(exp_mesh, P("fsdp", None)))
        sub_e = jax.device_put(sub_s, jax.sharding.NamedSharding(exp_mesh, P("fsdp")))
        dy_e = jax.device_put(dy, jax.sharding.NamedSharding(exp_mesh, act_pspec))

        @jax.jit
        def _run_exp(pa, ba, nsa, pra, sa, dya):
          out, vjp_fn = jax.vjp(
              lambda x, blk, w1, w2, ws: attn_res_rms_norm_pallas(
                  x,
                  blk,
                  3,
                  w1,
                  w2,
                  ws,
                  res_epsilon=eps,
                  norm_epsilon=eps,
                  norm_dtype=jnp.float32,
                  config=cfg,
              ),
              pa,
              ba,
              nsa,
              pra,
              sa,
          )
          return out, vjp_fn(dya)

        out_e, grads_e = _run_exp(p_e, blks_e, ns_e, pr_e, sub_e, dy_e)
        self._assert_parity(out_e, out_ref, dtype=jnp.float32)
        for ge, gr in zip(grads_e, grads_ref):
          self._assert_parity(ge, gr, dtype=jnp.float32)

  def test_can_use_rejects_tp_under_auto_jit(self):
    """Ensures `can_use_attn_res_pallas` rejects TP>1 inside `jax.jit` with an Auto mesh."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 devices, found {len(devices)}")
    P = jax.sharding.PartitionSpec
    auto_mesh = jax.sharding.Mesh(np.array(devices[:4]).reshape(2, 2), ("fsdp", "tensor"))
    p = jnp.zeros((2, 16, 256), dtype=jnp.bfloat16)
    blks = jnp.zeros((2, 16, 4, 256), dtype=jnp.bfloat16)
    seen = []

    @jax.jit
    def _check(pa, ba):
      seen.append(
          can_use_attn_res_pallas(
              pa,
              ba,
              mesh=auto_mesh,
              activation_pspec=P("fsdp", None, "tensor"),
              interpret=True,
          )
      )
      return pa

    with jax.set_mesh(auto_mesh):
      _check(p, blks)
    self.assertEqual(seen, [False])

  @parameterized.parameters(True, False)
  def test_kimi_k3_fsdp_model_dispatches_pallas_and_matches_reference(self, enter_rules_ctx: bool):
    """Verifies multi-device FSDP Kimi-K3 enters `shard_map` with exact grad parity (with or without rules ctx)."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 devices, found {len(devices)}")
    seq_len = 16
    emb_dim = 128
    base_kwargs = {
        "override_model_config": True,
        "base_emb_dim": emb_dim,
        "base_num_decoder_layers": 6,
        "full_attn_layers": [3, 5],
        "base_num_query_heads": 4,
        "base_num_kv_heads": 4,
        "head_dim": 32,
        "vocab_size": 128,
        "q_lora_rank": 32,
        "kv_lora_rank": 16,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "base_mlp_dim": 128,
        "base_moe_mlp_dim": 64,
        "moe_intermediate_size": 64,
        "shared_intermediate_size": 64,
        "shared_experts": 1,
        "num_shared_experts": 1,
        "routed_expert_hidden_size": 64,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_expert_input_dim": 64,
        "moe_router_input_dim": emb_dim,
        "attn_res_block_size": 2,
        "megablox": False,
        "max_target_length": seq_len,
        "max_position_embeddings": seq_len,
        "original_max_position_embeddings": seq_len,
        "max_prefill_predict_length": seq_len // 2,
        "per_device_batch_size": 1,
        "ici_fsdp_parallelism": 4,
        "dtype": "float32",
        "weight_dtype": "float32",
        "matmul_precision": "highest",
        "packing": False,
        "enable_checkpointing": False,
        "scan_layers": True,
    }
    cfg_ref = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        model_name="kimi-k3",
        use_attn_res_pallas_kernel=False,
        **base_kwargs,
    )
    cfg_pal = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        model_name="kimi-k3",
        use_attn_res_pallas_kernel=True,
        **base_kwargs,
    )
    devs = devices[:4]
    model_ref = model_creation_utils.from_config(cfg_ref, devices=devs, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))
    model_pal = model_creation_utils.from_config(cfg_pal, devices=devs, model_mode=MODEL_MODE_TRAIN, rngs=nnx.Rngs(0))

    tok_sharding = jax.sharding.NamedSharding(model_pal.mesh, jax.sharding.PartitionSpec("fsdp", None))
    tokens = jax.device_put(
        (jnp.arange(4 * seq_len, dtype=jnp.int32).reshape(4, seq_len) * 7 % 113 + 1).astype(jnp.int32),
        tok_sharding,
    )
    positions = jax.device_put(jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (4, seq_len)), tok_sharding)

    @nnx.jit
    def _step(model, tok, pos):
      def _loss_and_logits(m):
        logits = m(tok, pos, model_mode=MODEL_MODE_TRAIN, enable_dropout=False)
        loss = jnp.mean(jax.nn.log_softmax(logits.astype(jnp.float32))[..., 0])
        return loss, logits

      return nnx.value_and_grad(_loss_and_logits, has_aux=True)(model)

    with jax.set_mesh(model_ref.mesh), logical_axis_rules(cfg_ref.logical_axis_rules):
      (_, logits_ref), grads_ref = _step(model_ref, tokens, positions)

    test_k_cfg = AttnResKernelConfig(
        fwd_block_t=16,
        bwd_block_t=16,
        chunk_d=128,
        interpret=(jax.default_backend() == "cpu"),
    )
    with mock.patch.object(attn_res_layer, "_DEFAULT_PALLAS_CONFIG", test_k_cfg):
      with (
          mock.patch.object(ar_kernel, "attn_res_rms_norm_pallas", wraps=ar_kernel.attn_res_rms_norm_pallas) as spy_norm,
          mock.patch.object(ar_kernel, "attn_res_pallas", wraps=ar_kernel.attn_res_pallas) as spy_pool,
          mock.patch.object(ar_kernel.jax, "shard_map", wraps=ar_kernel.jax.shard_map) as spy_shmap,
      ):
        if enter_rules_ctx:
          with jax.set_mesh(model_pal.mesh), logical_axis_rules(cfg_pal.logical_axis_rules):
            (_, logits_pal), grads_pal = _step(model_pal, tokens, positions)
        else:
          with jax.set_mesh(model_pal.mesh):
            (_, logits_pal), grads_pal = _step(model_pal, tokens, positions)
        self.assertGreater(spy_norm.call_count, 0)
        self.assertGreater(spy_pool.call_count, 0)
        self.assertGreater(spy_shmap.call_count, 0)

    np.testing.assert_allclose(np.asarray(logits_pal), np.asarray(logits_ref), rtol=2e-4, atol=2e-4)
    flat_ref = dict(nnx.to_flat_state(nnx.state(grads_ref, nnx.Param)))
    flat_pal = dict(nnx.to_flat_state(nnx.state(grads_pal, nnx.Param)))
    self.assertEqual(set(flat_ref.keys()), set(flat_pal.keys()))
    for path, g_ref_var in flat_ref.items():
      g_pal_val = np.asarray(flat_pal[path].get_value(), dtype=np.float32)
      g_ref_val = np.asarray(g_ref_var.get_value(), dtype=np.float32)
      np.testing.assert_allclose(g_pal_val, g_ref_val, rtol=5e-4, atol=5e-4, err_msg=f"Mismatch at {path}")


@pytest.mark.tpu_only
class AttnResPallasKernelTpuTest(parameterized.TestCase):
  """Real TPU (`interpret=False`) Mosaic lowering, MXU stat-unpack, and sharded parity tests."""

  def setUp(self):
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Requires TPU hardware for interpret=False Mosaic compilation.")

  @parameterized.product(
      use_pending=[False, True],
      norm_dtype=[jnp.bfloat16, None],
  )
  def test_tpu_mosaic_fused_rms_norm_parity(self, use_pending: bool, norm_dtype):
    """Tests Mosaic TPU compilation (`interpret=False`), NR rsqrt, and MXU stat unpacking."""
    b, t, m, d = 1, 256, 9, 1024
    res_eps, norm_eps = 1e-5, 1e-6
    keys = jax.random.split(jax.random.PRNGKey(404), 7)
    p = jax.random.normal(keys[0], (b, t, d), dtype=jnp.bfloat16)
    blks = jax.random.normal(keys[1], (b, t, m, d), dtype=jnp.bfloat16)
    pend = jax.random.normal(keys[2], (b, t, d), dtype=jnp.bfloat16)
    ns = 1.0 + 0.1 * jax.random.normal(keys[3], (d,), dtype=jnp.float32)
    pr = jax.random.normal(keys[4], (d, 1), dtype=jnp.float32) * 0.25
    sub_scale = 1.0 + 0.1 * jax.random.normal(keys[5], (d,), dtype=jnp.float32)
    out_dt = jnp.float32 if norm_dtype is None else norm_dtype
    dy = jax.random.normal(keys[6], (b, t, d), dtype=out_dt)

    rms_norm = RMSNorm(
        num_features=d,
        epsilon=norm_eps,
        dtype=norm_dtype,
        weight_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    norm_graphdef, norm_state = nnx.split(rms_norm)
    cfg = AttnResKernelConfig(interpret=False)

    @jax.jit
    def _run(pa, ba, pda, hpa, n_comm, nsa, pra, ws, dya):
      def _call(use_pal, x, blk, pd, w1, w2, s):
        norm_mod = nnx.merge(norm_graphdef, jax.tree.map(lambda _: s, norm_state))
        return apply_attn_res_rms_norm(
            x,
            blk,
            w2,
            w1,
            norm_mod,
            epsilon=res_eps,
            num_blocks=m,
            num_committed=n_comm,
            pending=pd if use_pending else None,
            has_pending=hpa if use_pending else None,
            use_pallas_kernel=use_pal,
            kernel_config=cfg,
        )

      out_r, vjp_r = jax.vjp(lambda x, blk, pd, w1, w2, s: _call(False, x, blk, pd, w1, w2, s), pa, ba, pda, nsa, pra, ws)
      out_k, vjp_k = jax.vjp(lambda x, blk, pd, w1, w2, s: _call(True, x, blk, pd, w1, w2, s), pa, ba, pda, nsa, pra, ws)
      return out_k, out_r, vjp_k(dya), vjp_r(dya)

    for num_committed in (0, 4, 9):
      out_k, out_r, grads_k, grads_r = _run(
          p, blks, pend, jnp.bool_(True), jnp.int32(num_committed), ns, pr, sub_scale, dy
      )
      self.assertEqual(out_k.dtype, out_r.dtype)
      self.assertLess(_rel_err(out_k, out_r), 5e-3)
      for gk, gr in zip(grads_k, grads_r):
        if gk is not None and gr is not None:
          self.assertLess(_rel_err(gk, gr), 1.5e-2)

  def test_tpu_mosaic_explicit_unreduced_cotangent_parity(self):
    """Verifies real TPU (`interpret=False`) Explicit unreduced cotangent reduction."""
    devices = jax.devices()
    if len(devices) < 4:
      self.skipTest(f"Requires >= 4 TPU devices, found {len(devices)}")
    P = jax.sharding.PartitionSpec
    b, t, m, d, h = 2, 128, 4, 512, 256
    eps = 1e-5
    keys = jax.random.split(jax.random.PRNGKey(505), 7)
    p = jax.random.normal(keys[0], (b, t, d), dtype=jnp.bfloat16)
    blks = jax.random.normal(keys[1], (b, t, m, d), dtype=jnp.bfloat16)
    ns = 1.0 + 0.1 * jax.random.normal(keys[2], (d,), dtype=jnp.float32)
    pr = jax.random.normal(keys[3], (d, 1), dtype=jnp.float32) * 0.25
    sub_s = 1.0 + 0.1 * jax.random.normal(keys[4], (d,), dtype=jnp.float32)
    w_proj = jax.random.normal(keys[5], (d, h), dtype=jnp.bfloat16) * 0.1
    dy_out = jax.random.normal(keys[6], (b, t, h), dtype=jnp.bfloat16)
    cfg = AttnResKernelConfig(interpret=False)

    def _ref_loss(pa, ba, nsa, pra, sa, wa):
      h_pool = _apply_attn_res(pa, ba, pra, nsa, epsilon=eps, num_blocks=m, num_committed=jnp.int32(3))
      h_f32 = h_pool.astype(jnp.float32)
      r_inv = jax.lax.rsqrt(jnp.mean(jnp.square(h_f32), -1, keepdims=True) + eps)
      normed = (h_f32 * r_inv).astype(jnp.bfloat16) * sa.astype(jnp.bfloat16)
      proj = jnp.einsum("btd,dh->bth", normed, wa, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
      return jnp.sum(proj.astype(jnp.float32) * dy_out.astype(jnp.float32))

    ref_loss, ref_grads = jax.value_and_grad(_ref_loss, argnums=(0, 1, 2, 3, 4, 5))(p, blks, ns, pr, sub_s, w_proj)

    exp_mesh = jax.sharding.Mesh(
        np.array(devices[:4]).reshape(2, 2),
        ("fsdp", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )
    with jax.set_mesh(exp_mesh):
      p_e = jax.device_put(p, jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, None)))
      blks_e = jax.device_put(blks, jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, None, None)))
      ns_e = jax.device_put(ns, jax.sharding.NamedSharding(exp_mesh, P("fsdp")))
      pr_e = jax.device_put(pr, jax.sharding.NamedSharding(exp_mesh, P("fsdp", None)))
      sub_e = jax.device_put(sub_s, jax.sharding.NamedSharding(exp_mesh, P("fsdp")))
      w_e = jax.device_put(w_proj, jax.sharding.NamedSharding(exp_mesh, P("fsdp", "tensor")))
      dy_e = jax.device_put(dy_out, jax.sharding.NamedSharding(exp_mesh, P("fsdp", None, "tensor")))

      @jax.jit
      def _run_exp(pa, ba, nsa, pra, sa, wa, dya):
        def _pal_loss(x, blk, w1, w2, ws, wp):
          normed = attn_res_rms_norm_pallas(
              x,
              blk,
              3,
              w1,
              w2,
              ws,
              res_epsilon=eps,
              norm_epsilon=eps,
              norm_dtype=jnp.bfloat16,
              config=cfg,
          )
          wp_gathered = jax.sharding.reshard(wp, jax.sharding.NamedSharding(exp_mesh, P(None, "tensor")))
          proj = jnp.einsum("btd,dh->bth", normed, wp_gathered, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
          return jax.sharding.reshard(jnp.sum(proj.astype(jnp.float32) * dya.astype(jnp.float32)), P())

        return jax.value_and_grad(_pal_loss, argnums=(0, 1, 2, 3, 4, 5))(pa, ba, nsa, pra, sa, wa)

      pal_loss, pal_grads = _run_exp(p_e, blks_e, ns_e, pr_e, sub_e, w_e, dy_e)
      self.assertLess(_rel_err(pal_loss, ref_loss), 5e-3)
      for gp, gr in zip(pal_grads, ref_grads):
        self.assertLess(_rel_err(gp, gr), 1.5e-2)


if __name__ == "__main__":
  absltest.main()

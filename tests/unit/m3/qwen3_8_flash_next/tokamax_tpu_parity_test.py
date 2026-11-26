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

"""TPU v7x (4-chip 2x2x1, 8 TCs) parity verification for all 4 PR-4 Tokamax kernels."""

from absl.testing import absltest
from flax import nnx
import jax
from jax import numpy as jnp
from maxtext.m3.core import kernels as m3_kernels
from maxtext.m3.core.sharding import DefaultSharding, create_mesh
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next
import numpy as np


def _rel_err(a: jax.Array, b: jax.Array) -> tuple[float, float]:
  a_np = np.asarray(a, dtype=np.float32)
  b_np = np.asarray(b, dtype=np.float32)
  assert np.all(np.isfinite(a_np)), "Non-finite values in actual!"
  assert np.all(np.isfinite(b_np)), "Non-finite values in reference!"
  abs_diff = float(np.max(np.abs(a_np - b_np)))
  denom = float(np.max(np.abs(b_np))) + 1e-8
  return abs_diff, abs_diff / denom


class TokamaxTpuParityTest(absltest.TestCase):

  def setUp(self):
    """Initializes the 8-core TPU mesh for parity checks."""
    super().setUp()
    if jax.default_backend() != "tpu":
      self.skipTest("Requires TPU backend")
    self.devs = jax.devices()
    print(
        f"TPU info: backend={jax.default_backend()} n={len(self.devs)} kind={self.devs[0].device_kind}",
        flush=True,
    )
    self.assertEqual(len(self.devs), 8)
    self.mesh = create_mesh(ici={"fsdp": 2, "expert": 4}, devices=self.devs)
    self.sharding = DefaultSharding()

  def test_gdn_tokamax_mosaic_cp_and_zero_mask_finiteness(self):
    b, s, h, d = 4, 1024, 4, 128
    rng = np.random.default_rng(42)
    q0 = jnp.array(rng.standard_normal((b, s, h, d)), dtype=jnp.bfloat16)
    k0 = jnp.array(rng.standard_normal((b, s, h, d)), dtype=jnp.bfloat16)
    v0 = jnp.array(rng.standard_normal((b, s, h, d)) * 0.2, dtype=jnp.bfloat16)
    # Use realistic Qwen3.8 init gates spanning [-25.0, -0.01] (A_log = log(Uniform(0.01, 16)),
    # dt_bias = 1.0 -> g = -exp(A_log) * softplus(a + 1) reaches ~ -21.0 per token), which
    # previously overflowed fp32 exp2 under per_channel_gate=True!
    g_mild = jnp.array(-rng.uniform(0.02, 0.3, size=(b, s, h)), dtype=jnp.float32)
    g_qwen_init = jnp.array(
        -np.exp(np.log(rng.uniform(0.01, 16.0, size=(1, 1, h))))
        * np.logaddexp(0.0, rng.standard_normal((b, s, h)) * 0.5 + 1.0),
        dtype=jnp.float32,
    )
    beta0 = jnp.array(rng.uniform(0.1, 0.8, size=(b, s, h)), dtype=jnp.float32)

    for case_name, g_base, mask_tail in (
        ("unmasked_mild_g", g_mild, False),
        ("qwen38_init_steep_g_neg25", g_qwen_init, False),
        ("zero_masked_tail_steep_g", g_qwen_init, True),
    ):
      if mask_tail:
        mask = jnp.ones((b, s), dtype=bool).at[:, -128:].set(False)
        q = q0
        k = jnp.where(mask[:, :, None, None], k0, jnp.zeros((), dtype=k0.dtype))
        v = jnp.where(mask[:, :, None, None], v0, jnp.zeros((), dtype=v0.dtype))
        g = jnp.where(mask[:, :, None], g_base, jnp.zeros((), dtype=g_base.dtype))
        beta = jnp.where(mask[:, :, None], beta0, jnp.zeros((), dtype=beta0.dtype))
        out_mask = mask[:, :, None, None]
      else:
        q, k, v, g, beta = q0, k0, v0, g_base, beta0
        out_mask = jnp.ones((b, s, 1, 1), dtype=bool)

      with jax.set_mesh(self.mesh):

        @jax.jit
        def _run_tk(qi, ki, vi, gi, bi, mask_in):
          def _loss(q_, k_, v_, g_, b_):
            out = m3_kernels.gated_delta_net(
                q_,
                k_,
                v_,
                g_,
                b_,
                chunk_size=64,
                use_qk_l2norm=True,
                batch_axes=("fsdp",),
                length_axes=("expert",),
                backend="tokamax",
                mesh=self.mesh,
            )
            assert out is not None
            return jnp.sum(jnp.where(mask_in, out.astype(jnp.float32), 0.0)), out

          (loss, out), grads = jax.value_and_grad(_loss, argnums=(0, 1, 2, 3, 4), has_aux=True)(qi, ki, vi, gi, bi)
          return loss, out, grads

        @jax.jit
        def _run_ref(qi, ki, vi, gi, bi, mask_in):
          def _loss(q_, k_, v_, g_, b_):
            out = m3_kernels.context_parallel_chunk_gated_delta_rule(
                q_,
                k_,
                v_,
                g_,
                b_,
                batch_axes=("fsdp",),
                length_axes=("expert",),
                chunk_size=64,
                use_qk_l2norm_in_kernel=True,
                chunk_inverse="block_doubling",
            )
            return jnp.sum(jnp.where(mask_in, out.astype(jnp.float32), 0.0)), out

          (loss, out), grads = jax.value_and_grad(_loss, argnums=(0, 1, 2, 3, 4), has_aux=True)(qi, ki, vi, gi, bi)
          return loss, out, grads

        tk_loss, tk_out, tk_grads = _run_tk(q, k, v, g, beta, out_mask)
        ref_loss, ref_out, ref_grads = _run_ref(q, k, v, g, beta, out_mask)
        abs_o, rel_o = _rel_err(jnp.where(out_mask, tk_out, 0.0), jnp.where(out_mask, ref_out, 0.0))
        print(
            f"GDN [{case_name}] loss: tk={float(tk_loss):.4f} ref={float(ref_loss):.4f} "
            f"out_abs={abs_o:.4e} out_rel={rel_o:.4e}",
            flush=True,
        )
        self.assertLess(rel_o, 0.05)
        for name, gt, gr in zip(("dq", "dk", "dv", "dg", "dbeta"), tk_grads, ref_grads):
          gt_np = np.asarray(gt, dtype=np.float32)
          self.assertTrue(
              np.all(np.isfinite(gt_np)),
              f"GDN [{case_name}] grad {name} has non-finite values!",
          )
          abs_g, rel_g = _rel_err(gt, gr)
          print(f"  grad {name:5s}: abs={abs_g:.4e} rel={rel_g:.4e} (all finite)", flush=True)
          self.assertLess(rel_g, 0.08, f"GDN [{case_name}] grad {name} rel_err {rel_g:.4e} > 0.08")

  def test_moe_grouped_matmul_public_mosaic_v2(self):
    rng = np.random.default_rng(43)
    m_dim, k_dim, n_dim, n_exp = 512, 2560, 640, 8
    lhs = jnp.array(rng.standard_normal((m_dim, k_dim)) * 0.1, dtype=jnp.bfloat16)
    rhs = jnp.array(rng.standard_normal((n_exp, k_dim, n_dim)) * 0.1, dtype=jnp.bfloat16)
    gs = jnp.array([64, 128, 0, 96, 32, 64, 64, 64], dtype=jnp.int32)
    for backend_name in ("tokamax", "gmm_v2"):

      @jax.jit
      def _gmm_tk(l, r):
        return jax.value_and_grad(
            lambda x, w: jnp.sum(
                m3_kernels.grouped_matmul(x, w, gs, backend=backend_name, is_tpu=True).astype(jnp.float32)
            ),
            argnums=(0, 1),
        )(l, r)

      @jax.jit
      def _gmm_ref(l, r):
        return jax.value_and_grad(
            lambda x, w: jnp.sum(
                m3_kernels.grouped_matmul(x, w, gs, backend="ragged_dot", is_tpu=True).astype(jnp.float32)
            ),
            argnums=(0, 1),
        )(l, r)

      val_tk, (dl_tk, dr_tk) = _gmm_tk(lhs, rhs)
      val_ref, (dl_ref, dr_ref) = _gmm_ref(lhs, rhs)
      _, rel_dl = _rel_err(dl_tk, dl_ref)
      _, rel_dr = _rel_err(dr_tk, dr_ref)
      print(
          f"GMM [{backend_name}] loss: tk={float(val_tk):.4f} ref={float(val_ref):.4f} "
          f"dl_rel={rel_dl:.4e} dr_rel={rel_dr:.4e}",
          flush=True,
      )
      self.assertLess(rel_dl, 0.05)
      self.assertLess(rel_dr, 0.05)

  def test_upstream_splash_attention_block512(self):
    self.assertTrue(m3_kernels.splash_is_upstream(), "Expected upstream tokamax splash!")
    rng = np.random.default_rng(44)
    with jax.set_mesh(self.mesh):
      qsa_f = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
          emb_dim=256,
          num_query_heads=4,
          num_kv_heads=2,
          head_dim=128,
          partial_rotary_factor=0.25,
          rope_theta=10000.0,
          mrope_section=(8, 4, 4),
          index_n_heads=2,
          index_kv_heads=1,
          index_head_dim=64,
          token_budget=1024,
          compress_ratio=4,
          eps=1e-6,
          dtype=jnp.bfloat16,
          weight_dtype=jnp.bfloat16,
          attention_kernel="flash",
          sa_block_q=512,
          sa_block_kv=512,
          sharding=self.sharding,
          rngs=nnx.Rngs(0),
      )
      qsa_d = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
          emb_dim=256,
          num_query_heads=4,
          num_kv_heads=2,
          head_dim=128,
          partial_rotary_factor=0.25,
          rope_theta=10000.0,
          mrope_section=(8, 4, 4),
          index_n_heads=2,
          index_kv_heads=1,
          index_head_dim=64,
          token_budget=1024,
          compress_ratio=4,
          eps=1e-6,
          dtype=jnp.bfloat16,
          weight_dtype=jnp.bfloat16,
          attention_kernel="dense",
          sharding=self.sharding,
          rngs=nnx.Rngs(0),
      )
      nnx.update(qsa_d, nnx.state(qsa_f))
      x_qsa = jnp.array(rng.standard_normal((2, 1024, 256)) * 0.2, dtype=jnp.bfloat16)
      pos_qsa = jnp.broadcast_to(jnp.arange(1024, dtype=jnp.int32), (2, 1024))
      out_f, dx_f = nnx.jit(
          nnx.value_and_grad(lambda m, inp: jnp.sum(m(inp, position_ids=pos_qsa).astype(jnp.float32)), argnums=1)
      )(qsa_f, x_qsa)
      out_d, dx_d = nnx.jit(
          nnx.value_and_grad(lambda m, inp: jnp.sum(m(inp, position_ids=pos_qsa).astype(jnp.float32)), argnums=1)
      )(qsa_d, x_qsa)
      _, rel_qsa = _rel_err(dx_f, dx_d)
      print(
          f"Splash upstream: loss_f={float(out_f):.4f} loss_d={float(out_d):.4f} dx_rel={rel_qsa:.4e}",
          flush=True,
      )
      self.assertLess(rel_qsa, 0.05)

  def test_fused_linear_cross_entropy_mosaic_tpu_fsdp_expert_sharded(self):
    rng = np.random.default_rng(45)
    b_ce, s_ce, h_ce, v_ce = 4, 4096, 2560, 8192  # local_tokens = (4/2)*(4096/4) = 2048 >= 1024 -> mosaic_tpu
    h_in = jnp.array(rng.standard_normal((b_ce, s_ce, h_ce)) * 0.1, dtype=jnp.bfloat16)
    w_in = jnp.array(rng.standard_normal((h_ce, v_ce)) * 0.05, dtype=jnp.bfloat16)
    t_in = jnp.array(rng.integers(0, v_ce, size=(b_ce, s_ce)), dtype=jnp.int32)
    m_in = jnp.ones((b_ce, s_ce), dtype=bool).at[:, -256:].set(False)
    t_oob = jnp.where(m_in, t_in, -1)

    with jax.set_mesh(self.mesh):

      @jax.jit
      def _ce_tk(hi, wi):
        return jax.value_and_grad(
            lambda x, w: m3_kernels.linear_cross_entropy(
                x, t_oob, w, mask=m_in, sharding=self.sharding, mesh=self.mesh, reduction="sum", backend="tokamax"
            ),
            argnums=(0, 1),
        )(hi, wi)

      @jax.jit
      def _ce_ref(hi, wi):
        return jax.value_and_grad(
            lambda x, w: m3_kernels.linear_cross_entropy(
                x, t_oob, w, mask=m_in, sharding=self.sharding, mesh=self.mesh, reduction="sum", backend="jax"
            ),
            argnums=(0, 1),
        )(hi, wi)

      l_tk, (dh_tk, dw_tk) = _ce_tk(h_in, w_in)
      l_ref, (dh_ref, dw_ref) = _ce_ref(h_in, w_in)
      _, rel_l = _rel_err(l_tk, l_ref)
      _, rel_dh = _rel_err(dh_tk, dh_ref)
      _, rel_dw = _rel_err(dw_tk, dw_ref)
      print(
          f"Linear CE: loss_tk={float(l_tk):.4f} loss_ref={float(l_ref):.4f} "
          f"rel_l={rel_l:.4e} rel_dh={rel_dh:.4e} rel_dw={rel_dw:.4e}",
          flush=True,
      )
      self.assertLess(rel_l, 1e-3)
      self.assertLess(rel_dh, 0.05)
      self.assertLess(rel_dw, 0.05)


if __name__ == "__main__":
  absltest.main()

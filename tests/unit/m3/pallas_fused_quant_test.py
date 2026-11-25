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

"""CPU (`InterpretParams`) and TPUv7x (`TPU v7x 2x2x1`) parity & microbenchmark tests for FP8 TC-Ragged MoE."""

import dataclasses
import functools
import time
from typing import Any, cast

from absl import flags
from absl.testing import absltest
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from maxtext.configs import pyconfig  # pylint: disable=no-name-in-module
from maxtext.kernels.megablox import ops as mblx_ops  # pylint: disable=no-name-in-module
from maxtext.kernels.megablox import pallas_fused_quant  # pylint: disable=no-name-in-module
from maxtext.m3.core import moe as m3_moe  # pylint: disable=no-name-in-module
from maxtext.m3.core import sharding as m3_sharding  # pylint: disable=no-name-in-module
from maxtext.utils import globals as maxtext_globals  # pylint: disable=no-name-in-module
import numpy as np
import qwix
import qwix.pallas as qpl

_RUN_TPU_FUSED_QUANT_BENCH = flags.DEFINE_bool(
    "run_tpu_fused_quant_bench",
    False,
    "Run Mosaic TPUv7x parity, HLO checks, and 64K-local microbenchmarks on TPU v7x 2x2x1.",
)


def _rel_l2(got: jax.Array | np.ndarray, ref: jax.Array | np.ndarray) -> float:
  a = np.asarray(got, dtype=np.float32)
  b = np.asarray(ref, dtype=np.float32)
  denom = max(float(np.linalg.norm(b)), 1e-6)
  return float(np.linalg.norm(a - b)) / denom


class PallasFusedQuantTest(absltest.TestCase):
  """Verifies FP8 TC-Ragged MoE collectives, `pallas_fused_quant`, and TPUv7x parity."""

  def test_cpu_aligned_scale_ag_combine_mode_and_config_flags(self):
    """Verifies aligned 1D `scale_ag` in `_ep_combine_tokens_bwd`, `moe_ep_combine_mode`, and flags."""
    if _RUN_TPU_FUSED_QUANT_BENCH.value:
      self.skipTest("CPU test skipped when --run_tpu_fused_quant_bench=true.")

    cfg = pyconfig.initialize(
        [
            "",
            f"{maxtext_globals.MAXTEXT_CONFIGS_DIR}/base.yml",
            "model_name=qwen3.8-flash-next",
            "override_model_config=True",
            "run_name=fp8_moe_flag_test",
            "skip_jax_distributed_system=True",
            "moe_bwd_wgrad_overlap=True",
            "moe_ep_allgather_tc=True",
            "moe_fp8=True",
            "moe_tc_ragged_flatten_block_size=512",
            "moe_ep_combine_mode=tensorcore",
            "moe_row_scale_placement=sc",
            "moe_tc_3d_gmm=True",
            "moe_fused_bwd_quant=True",
        ]
    )
    self.assertTrue(cfg.moe_bwd_wgrad_overlap)
    self.assertTrue(cfg.moe_ep_allgather_tc)
    self.assertTrue(cfg.moe_fp8)
    self.assertEqual(cfg.moe_tc_ragged_flatten_block_size, 512)
    self.assertEqual(cfg.moe_ep_combine_mode, "tensorcore")
    self.assertEqual(cfg.moe_row_scale_placement, "sc")
    self.assertTrue(cfg.moe_tc_3d_gmm)
    self.assertTrue(cfg.moe_fused_bwd_quant)

    ex = m3_moe.moe_from_config(cfg, m3_sharding.EPasCP())
    self.assertTrue(ex.bwd_wgrad_overlap)
    self.assertTrue(ex.ep_allgather_tc)
    self.assertTrue(ex.moe_fp8)
    self.assertEqual(ex.tc_ragged_flatten_block_size, 512)
    self.assertEqual(ex.ep_combine_mode, "tensorcore")
    self.assertEqual(ex.row_scale_placement, "sc")
    self.assertTrue(ex.tc_3d_gmm)
    self.assertTrue(ex.fused_bwd_quant)

    if len(jax.devices()) < 2:
      return
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ("ep",))
    rng = np.random.default_rng(2026112301)

    for is_3d in (False, True):
      shape = (64, 2, 128) if is_3d else (64, 256)
      x_np = rng.standard_normal(shape).astype(np.float32) * 0.5
      cot_np = rng.standard_normal(shape).astype(np.float32) * 0.4

      with jax.set_mesh(mesh):
        in_spec = P(None, None, None) if is_3d else P(None, None)
        out_spec = P("ep", None, None) if is_3d else P("ep", None)

        def _per_shard(p_in, c_loc, fwd_on_tc_flag):
          y, vjp_fn = jax.vjp(
              lambda z: m3_moe._ep_combine_tokens(z, "ep", True, False, fwd_on_tc_flag),  # pylint: disable=protected-access
              p_in,
          )
          (dp,) = vjp_fn(c_loc)
          return y, dp

        x_dev = jax.device_put(jnp.asarray(x_np, dtype=jnp.bfloat16), jax.sharding.NamedSharding(mesh, in_spec))
        c_dev = jax.device_put(jnp.asarray(cot_np, dtype=jnp.bfloat16), jax.sharding.NamedSharding(mesh, out_spec))

        y_auto, dp_auto = jax.shard_map(
            functools.partial(_per_shard, fwd_on_tc_flag=False),
            mesh=mesh,
            in_specs=(in_spec, out_spec),
            out_specs=(out_spec, in_spec),
            check_vma=False,
        )(x_dev, c_dev)
        y_tc, dp_tc = jax.shard_map(
            functools.partial(_per_shard, fwd_on_tc_flag=True),
            mesh=mesh,
            in_specs=(in_spec, out_spec),
            out_specs=(out_spec, in_spec),
            check_vma=False,
        )(x_dev, c_dev)

        np.testing.assert_array_equal(np.asarray(y_auto), np.asarray(y_tc))
        np.testing.assert_array_equal(np.asarray(dp_auto), np.asarray(dp_tc))
        c0_qa = qpl.quantize(
            jnp.asarray(cot_np[:32], dtype=jnp.bfloat16).astype(jnp.float32),
            qtype=jnp.float8_e5m2,
            channelwise_axes=(0,),
            calibration_method="absmax",
        )
        c1_qa = qpl.quantize(
            jnp.asarray(cot_np[32:], dtype=jnp.bfloat16).astype(jnp.float32),
            qtype=jnp.float8_e5m2,
            channelwise_axes=(0,),
            calibration_method="absmax",
        )
        exp_dp = jnp.concatenate(
            [
                (c0_qa.qvalue.astype(jnp.float32) * c0_qa.scale).astype(jnp.bfloat16),
                (c1_qa.qvalue.astype(jnp.float32) * c1_qa.scale).astype(jnp.bfloat16),
            ],
            axis=0,
        )
        np.testing.assert_array_equal(np.asarray(dp_auto), np.asarray(exp_dp))

  def test_cpu_fused_bwd_prepare_and_quantize_interpret(self):
    """Verifies `fused_bwd_prepare_and_quantize` on 2D and 3D inputs with padded tail rows (`sum(gs) < M`)."""
    if _RUN_TPU_FUSED_QUANT_BENCH.value:
      self.skipTest("CPU test skipped when --run_tpu_fused_quant_bench=true.")

    rng = np.random.default_rng(2026112302)
    qrule = qwix.QtRule(bwd_qtype=jnp.float8_e5m2, bwd_calibration_method="absmax")
    for is_3d in (False, True):
      m_b = 512 if is_3d else 300
      n_b = 4096 if is_3d else 256
      g_b = 4
      gs = jnp.array([120, 140, 110, 60] if is_3d else [80, 90, 50, 30], dtype=jnp.int32)
      grad_2d = jnp.asarray(rng.standard_normal((m_b, n_b)).astype(np.float32) * 0.7, dtype=jnp.bfloat16)
      grad_in = grad_2d.reshape(m_b, n_b // 128, 128) if is_3d else grad_2d
      rscale = jnp.asarray(np.abs(rng.standard_normal((g_b, 1, n_b)).astype(np.float32)) * 0.2 + 0.02)
      lscale = jnp.asarray(
          np.abs(rng.standard_normal((m_b, 1, 1) if is_3d else (m_b, 1)).astype(np.float32)) * 0.3 + 0.03
      )
      lhs_qa = qpl.QArray(qvalue=jnp.zeros((m_b, 128), dtype=jnp.float8_e4m3fn), scale=lscale)
      rhs_qa = qpl.QArray(qvalue=jnp.zeros((g_b, 128, n_b), dtype=jnp.float8_e4m3fn), scale=rscale)

      self.assertTrue(
          pallas_fused_quant.can_fuse_bwd_prepare_and_quantize(
              grad_in,
              lhs_qa,
              rhs_qa,
              qrule,
              use_tokamax_backend=True,
              use_gmm_v2=True,
              tc_unsort_cfg=None,
          )
      )
      dlhs_pl, drhs_pl, _, _ = pallas_fused_quant.fused_bwd_prepare_and_quantize(
          grad_in, lhs_qa, rhs_qa, gs, interpret=True
      )
      dlhs_unq, drhs_unq, _, _ = mblx_ops._bwd_prepare_inputs(  # pylint: disable=protected-access
          grad_in, lhs_qa, rhs_qa, gs, True, qrule
      )
      dlhs_rf, drhs_rf = cast(
          tuple[Any, Any],
          mblx_ops._bwd_quantize_gradient(dlhs_unq, drhs_unq, qrule),  # pylint: disable=protected-access
      )

      np.testing.assert_allclose(np.asarray(dlhs_pl.scale), np.asarray(dlhs_rf.scale), rtol=1e-2, atol=1e-4)
      np.testing.assert_allclose(np.asarray(drhs_pl.scale), np.asarray(drhs_rf.scale), rtol=1e-2, atol=1e-4)
      deq_dlhs_pl = dlhs_pl.qvalue.astype(jnp.float32) * dlhs_pl.scale.astype(jnp.float32)
      deq_dlhs_rf = dlhs_rf.qvalue.astype(jnp.float32) * dlhs_rf.scale.astype(jnp.float32)
      deq_drhs_pl = drhs_pl.qvalue.astype(jnp.float32) * drhs_pl.scale.astype(jnp.float32)
      deq_drhs_rf = drhs_rf.qvalue.astype(jnp.float32) * drhs_rf.scale.astype(jnp.float32)
      self.assertLess(_rel_l2(deq_dlhs_pl, deq_dlhs_rf), 1e-2)
      self.assertLess(_rel_l2(deq_drhs_pl, deq_drhs_rf), 1e-2)

  def test_tpu_mosaic_fp8_tc_ragged_moe_and_fused_bwd_quant_parity(self):
    """Runs physical TPUv7x (`TPU v7x 2x2x1` = 8 TCs) parity, HLO checks, and 64K-local microbenchmarks."""
    if not _RUN_TPU_FUSED_QUANT_BENCH.value:
      self.skipTest("Pass --run_tpu_fused_quant_bench=true on TPU v7x 2x2x1.")
    self.assertEqual(jax.default_backend(), "tpu")
    devices = jax.devices()
    self.assertGreaterEqual(len(devices), 8)
    rng = np.random.default_rng(2026112305)

    self.assertFalse(
        pallas_fused_quant.can_use_3d_gmm_tpu(
            num_rows=204800,
            num_local_experts=4,
            d_dim=2560,
            m_tp=640,
            moe_fp8=True,
            act_dtype=jnp.bfloat16,
        )
    )
    self.assertTrue(
        pallas_fused_quant.can_use_3d_gmm_tpu(
            num_rows=204800,
            num_local_experts=4,
            d_dim=4096,
            m_tp=640,
            moe_fp8=True,
            act_dtype=jnp.bfloat16,
        )
    )

    mesh8 = jax.sharding.Mesh(np.array(devices[:8]).reshape(2, 4), ("fsdp", "expert"))
    base_exec = m3_moe.MoEExecutor(
        kind="tokamax",
        batch="fsdp",
        length="expert",
        experts="expert",
        moe_fp8=True,
        ep_allgather_tc=True,
        tc_ragged_flatten_block_size=128,
        ragged_buffer_factor=1.25,
    )
    exec_fbq = dataclasses.replace(base_exec, tc_3d_gmm=True, fused_bwd_quant=True)

    with jax.set_mesh(mesh8):
      b_m, s_m, d_m, f_m, n_exp, topk = 2, 1024, 2560, 640, 16, 4
      x_moe = jax.device_put(
          jnp.asarray(rng.standard_normal((b_m, s_m, d_m)).astype(np.float32) * 0.2, dtype=jnp.bfloat16),
          jax.sharding.NamedSharding(mesh8, P("fsdp", "expert", None)),
      )
      w_raw = np.abs(rng.standard_normal((b_m, s_m, topk)).astype(np.float32)) + 0.1
      w_raw = w_raw / np.sum(w_raw, axis=-1, keepdims=True)
      w_moe = jax.device_put(
          jnp.asarray(w_raw, dtype=jnp.bfloat16),
          jax.sharding.NamedSharding(mesh8, P("fsdp", "expert", None)),
      )
      idx_np = np.zeros((b_m, s_m, topk), dtype=np.int32)
      for bi in range(b_m):
        for si in range(s_m):
          idx_np[bi, si] = rng.choice(n_exp, size=topk, replace=False)
      idx_moe = jax.device_put(jnp.asarray(idx_np), jax.sharding.NamedSharding(mesh8, P("fsdp", "expert", None)))
      gu_moe = jax.device_put(
          jnp.asarray(rng.standard_normal((n_exp, d_m, 2, f_m)).astype(np.float32) * 0.03, dtype=jnp.bfloat16),
          jax.sharding.NamedSharding(mesh8, P("expert", "fsdp", None, None)),
      )
      dn_moe = jax.device_put(
          jnp.asarray(rng.standard_normal((n_exp, f_m, d_m)).astype(np.float32) * 0.03, dtype=jnp.bfloat16),
          jax.sharding.NamedSharding(mesh8, P("expert", None, "fsdp")),
      )
      cot_moe = jax.device_put(
          jnp.asarray(rng.standard_normal((b_m, s_m, d_m)).astype(np.float32) * 0.2, dtype=jnp.bfloat16),
          jax.sharding.NamedSharding(mesh8, P("fsdp", "expert", None)),
      )

      def _moe_loss_fn(ex_cfg):
        def _fn(xx, ww, gu, dn):
          y = m3_moe.routed_experts_fused(xx, ww, idx_moe, gu, dn, executor=ex_cfg)
          return jnp.sum(y.astype(jnp.float32) * cot_moe.astype(jnp.float32))

        return jax.jit(jax.value_and_grad(_fn, argnums=(0, 1, 2, 3)))

      fn_base = _moe_loss_fn(base_exec)
      fn_fbq = _moe_loss_fn(exec_fbq)

      hlo_fbq = fn_fbq.lower(x_moe, w_moe, gu_moe, dn_moe).as_text()
      self.assertIn("gmm_bwd_fused_dlhs_quant_amax", hlo_fbq)

      v_base, g_base = fn_base(x_moe, w_moe, gu_moe, dn_moe)
      v_fbq, g_fbq = fn_fbq(x_moe, w_moe, gu_moe, dn_moe)
      jax.block_until_ready((v_base, v_fbq, g_base, g_fbq))

      fbq_rels = {nm: _rel_l2(gg, gr) for gg, gr, nm in zip(g_fbq, g_base, ("dx", "dw", "dgu", "ddn"), strict=True)}
      print(f"[TPU_MOE_D2560_PARITY] base={float(v_base):.4f} fbq={float(v_fbq):.4f} fbq_rels={fbq_rels}")
      np.testing.assert_allclose(float(v_fbq), float(v_base), rtol=1e-4, atol=1e-4)
      self.assertLess(fbq_rels["dw"], 1e-5, msg="FusedBwdQuant dw")
      self.assertLess(fbq_rels["ddn"], 1e-5, msg="FusedBwdQuant ddn")
      self.assertLess(fbq_rels["dx"], 3e-2, msg="FusedBwdQuant dx")
      self.assertLess(fbq_rels["dgu"], 3e-2, msg="FusedBwdQuant dgu")

    mesh1 = jax.sharding.Mesh(np.array(devices[:1]), ("data",))
    with jax.set_mesh(mesh1):
      m_ulp, n_ulp, g_ulp = 5120, 1280, 4
      gs_ulp = jnp.array([300, 250, 274, 200], dtype=jnp.int32)
      grad_ulp = jnp.asarray(rng.standard_normal((m_ulp, n_ulp)).astype(np.float32) * 0.4, dtype=jnp.bfloat16)
      lhs_ulp = qpl.QArray(
          qvalue=jnp.zeros((m_ulp, 640), dtype=jnp.float8_e4m3fn),
          scale=jnp.asarray(np.abs(rng.standard_normal((m_ulp, 1)).astype(np.float32)) * 0.2 + 0.01),
      )
      rhs_ulp = qpl.QArray(
          qvalue=jnp.zeros((g_ulp, 640, n_ulp), dtype=jnp.float8_e4m3fn),
          scale=jnp.asarray(np.abs(rng.standard_normal((g_ulp, 1, n_ulp)).astype(np.float32)) * 0.2 + 0.01),
      )
      qrule = qwix.QtRule(bwd_qtype=jnp.float8_e5m2, bwd_calibration_method="absmax")
      dl_pl_u, dr_pl_u, _, _ = jax.jit(
          lambda gg: pallas_fused_quant.fused_bwd_prepare_and_quantize(gg, lhs_ulp, rhs_ulp, gs_ulp, interpret=False)
      )(grad_ulp)
      dl_unq_u, dr_unq_u, _, _ = mblx_ops._bwd_prepare_inputs(  # pylint: disable=protected-access
          grad_ulp, lhs_ulp, rhs_ulp, gs_ulp, True, qrule
      )
      dl_rf_u, dr_rf_u = cast(
          tuple[Any, Any],
          mblx_ops._bwd_quantize_gradient(dl_unq_u, dr_unq_u, qrule),  # pylint: disable=protected-access
      )
      np.testing.assert_allclose(np.asarray(dl_pl_u.scale), np.asarray(dl_rf_u.scale), rtol=1e-2, atol=1e-5)
      np.testing.assert_allclose(np.asarray(dr_pl_u.scale), np.asarray(dr_rf_u.scale), rtol=1e-2, atol=1e-5)
      for tag_u, qp_u, qr_u in (("dlhs", dl_pl_u.qvalue, dl_rf_u.qvalue), ("drhs", dr_pl_u.qvalue, dr_rf_u.qvalue)):
        bp = np.asarray(lax.bitcast_convert_type(qp_u, jnp.uint8), dtype=np.int32)
        br = np.asarray(lax.bitcast_convert_type(qr_u, jnp.uint8), dtype=np.int32)
        bp_mag = np.where((bp & 0x7F) == 0, 0, np.where((bp & 0x80) != 0, -(bp & 0x7F), bp & 0x7F))
        br_mag = np.where((br & 0x7F) == 0, 0, np.where((br & 0x80) != 0, -(br & 0x7F), br & 0x7F))
        ulp_diff = np.abs(bp_mag - br_mag)
        mismatch_frac = float(np.mean(ulp_diff > 0))
        max_ulp = int(np.max(ulp_diff))
        print(f"[TPU_BWD_QUANT_ULP_{tag_u}] mismatch_frac={mismatch_frac:.4f} max_ulp={max_ulp}")
        self.assertLessEqual(max_ulp, 1, msg=f"{tag_u} exceeded 1 e5m2 ULP")
        self.assertLess(mismatch_frac, 0.05, msg=f"{tag_u} mismatch fraction too high")

      m_prod, f_prod, d_prod, g_prod = 204800, 640, 2560, 4
      grad_2d = jnp.ones((m_prod, d_prod), dtype=jnp.bfloat16) * 0.05
      gs_p = jnp.array([51200, 51200, 51200, 51200], dtype=jnp.int32)
      lhs_p = qpl.QArray(
          qvalue=jnp.zeros((m_prod, f_prod), dtype=jnp.float8_e4m3fn),
          scale=jnp.ones((m_prod, 1), dtype=jnp.float32) * 0.02,
      )
      rhs_p = qpl.QArray(
          qvalue=jnp.zeros((g_prod, f_prod, d_prod), dtype=jnp.float8_e4m3fn),
          scale=jnp.ones((g_prod, 1, d_prod), dtype=jnp.float32) * 0.02,
      )
      fn_bq_pl = jax.jit(
          lambda gg: pallas_fused_quant.fused_bwd_prepare_and_quantize(gg, lhs_p, rhs_p, gs_p, interpret=False)[:2]
      )

      def _bq_ref(gg):
        dl, dr, _, _ = mblx_ops._bwd_prepare_inputs(gg, lhs_p, rhs_p, gs_p, True, qrule)  # pylint: disable=protected-access
        return mblx_ops._bwd_quantize_gradient(dl, dr, qrule)  # pylint: disable=protected-access

      fn_bq_rf = jax.jit(_bq_ref)
      jax.block_until_ready(fn_bq_pl(grad_2d))
      jax.block_until_ready(fn_bq_rf(grad_2d))
      t0 = time.perf_counter()
      for _ in range(5):
        o_bq_pl = fn_bq_pl(grad_2d)
      jax.block_until_ready(o_bq_pl)
      ms_bq_pl = (time.perf_counter() - t0) * 1000.0 / 5.0
      t0 = time.perf_counter()
      for _ in range(5):
        o_bq_rf = fn_bq_rf(grad_2d)
      jax.block_until_ready(o_bq_rf)
      ms_bq_rf = (time.perf_counter() - t0) * 1000.0 / 5.0
      print(
          f"[BENCH_64K_BWD_QUANT_DOWN] pallas={ms_bq_pl:.3f} ms vs ref={ms_bq_rf:.3f} ms "
          f"(48x save={(ms_bq_rf - ms_bq_pl) * 48:.1f} ms)"
      )


if __name__ == "__main__":
  absltest.main()

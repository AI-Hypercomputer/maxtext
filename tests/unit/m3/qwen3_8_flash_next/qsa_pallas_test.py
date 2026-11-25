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

"""CPU (`InterpretParams`) and TPU (`TPU v7x 2x2x1`) parity & 64K benchmark tests for `qsa_pallas`."""

import functools
import math
import time

from absl import flags
from absl.testing import absltest
from flax import nnx
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.m3.core import qsa_pallas
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next
import numpy as np

_RUN_TPU_QSA_BENCH = flags.DEFINE_bool(
    "run_tpu_qsa_bench",
    False,
    "Run Mosaic TPU parity and 8-TC CP=8 64K QSA benchmark on TPU v7x 2x2x1.",
)


def _reference_qsa_loss(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    pmask: jax.Array,
    cot: jax.Array,
    *,
    q_offset: int,
    scale: float,
) -> tuple[jax.Array, jax.Array]:
  """Exact `float32` reference QSA forward (`causal & (selected | in_tail)`) for parity checks."""
  b_loc, s_q, h_q, _ = q.shape
  _, s_kv, h_kv, _ = k.shape
  g = h_q // h_kv
  q_f = q.astype(jnp.float32)
  k_f = jnp.repeat(k.astype(jnp.float32), g, axis=2)
  v_f = jnp.repeat(v.astype(jnp.float32), g, axis=2)

  q_idx = (q_offset + jnp.arange(s_q, dtype=jnp.int32))[None, :, None]
  k_idx = jnp.arange(s_kv, dtype=jnp.int32)[None, None, :]
  blk_k = k_idx >> 2
  word_idx = blk_k >> 5
  bit_shift = (blk_k & 31).astype(jnp.uint32)
  words = jnp.take_along_axis(pmask, jnp.broadcast_to(word_idx, (b_loc, s_q, s_kv)), axis=-1)
  is_sel = ((words >> bit_shift) & jnp.uint32(1)) != jnp.uint32(0)
  in_tail = k_idx >= ((q_idx + 1) & -4)
  valid = (k_idx <= q_idx) & (is_sel | in_tail)

  scores = jnp.einsum("bqhd,bkhd->bhqk", q_f, k_f) * scale
  scores = jnp.where(valid[:, None, :, :], scores, -1e30)
  any_valid = jnp.any(valid[:, None, :, :], axis=-1, keepdims=True)
  probs = jnp.where(any_valid, jax.nn.softmax(scores, axis=-1), 0.0)
  out = jnp.einsum("bhqk,bkhd->bqhd", probs, v_f).astype(q.dtype)
  loss = jnp.sum(out.astype(jnp.float32) * cot.astype(jnp.float32))
  return loss, out


class QsaPallasTest(absltest.TestCase):
  """Verifies `qsa_pallas` `bkv` ladder, Zig-Zag CP, streaming indexer, and `Qwen3_8SparseAttention('pallas')`."""

  def test_choose_bkv_ladder(self):
    self.assertEqual(qsa_pallas.choose_bkv(65536, 0), 512)
    self.assertEqual(qsa_pallas.choose_bkv(65536, 512), 512)
    self.assertEqual(qsa_pallas.choose_bkv(65536, 1024), 1024)
    self.assertEqual(qsa_pallas.choose_bkv(65536, 2048), 2048)
    self.assertEqual(qsa_pallas.choose_bkv(1280, 1024), 128)
    self.assertEqual(qsa_pallas.choose_bkv(1536, 1024), 512)
    self.assertEqual(qsa_pallas.choose_bkv(256, 512), 128)

  def test_cpu_multi_word_block_mask_and_bkv_ladder(self):
    """Tests `s_kv = 32,768` (`256` `uint32` words = 2 Q-major 128-word blocks) under strict `InterpretParams`."""
    if _RUN_TPU_QSA_BENCH.value:
      self.skipTest("CPU interpret test skipped when --run_tpu_qsa_bench=true.")

    rng = np.random.default_rng(20261120)
    b_loc, s_q_loc, s_kv = 1, 256, 32768
    h_kv, h_q, d = 1, 2, 128
    q_offset = s_kv - s_q_loc
    scale = 1.0 / math.sqrt(float(d))
    num_words = s_kv // 128

    sbi = np.full((b_loc, s_q_loc, 12), -1, dtype=np.int32)
    for qi in range(s_q_loc):
      sbi[0, qi, 0:4] = np.array([3, 19, 127, 513], dtype=np.int32)
      sbi[0, qi, 4:8] = np.array([4096, 4129, 6001, 7900], dtype=np.int32)
      sbi[0, qi, 8:12] = rng.choice(np.arange(4200, 8100, dtype=np.int32), size=4, replace=False)
    pmask = modeling_qwen3_8_flash_next.pack_qsa_block_mask_32bit(jnp.asarray(sbi), num_words, assume_unique=True)

    q = jnp.asarray(rng.standard_normal((b_loc, s_q_loc, h_q, d)).astype(np.float32) * 0.25)
    k = jnp.asarray(rng.standard_normal((b_loc, s_kv, h_kv, d)).astype(np.float32) * 0.25)
    v = jnp.asarray(rng.standard_normal((b_loc, s_kv, h_kv, d)).astype(np.float32) * 0.25)
    cot = jnp.asarray(rng.standard_normal((b_loc, s_q_loc, h_q, d)).astype(np.float32) * 0.25)

    (ref_loss, ref_out), (ref_dq, ref_dk, ref_dv) = jax.value_and_grad(
        lambda qq, kk, vv: _reference_qsa_loss(qq, kk, vv, pmask, cot, q_offset=q_offset, scale=scale),
        argnums=(0, 1, 2),
        has_aux=True,
    )(q, k, v)

    strict_interpret = pltpu.InterpretParams()
    for bkv_q, bkv_dkv in ((128, 128), (512, 1024), (1024, 1024), (2048, 2048)):

      def _pallas_loss(qq, kk, vv, bq_v=bkv_q, bdkv_v=bkv_dkv):
        @jax.custom_vjp
        def _op(q_i, k_i, v_i):
          o, _ = qsa_pallas.qsa_pallas_fwd(
              q_i,
              k_i,
              v_i,
              pmask,
              jnp.int32(q_offset),
              scale=scale,
              interpret=strict_interpret,
              bkv=bq_v,
              disable_bounds_checks=False,
          )
          return o

        def _op_fwd(q_i, k_i, v_i):
          o, lse = qsa_pallas.qsa_pallas_fwd(
              q_i,
              k_i,
              v_i,
              pmask,
              jnp.int32(q_offset),
              scale=scale,
              interpret=strict_interpret,
              bkv=bq_v,
              disable_bounds_checks=False,
          )
          return o, (q_i, k_i, v_i, o, lse)

        def _op_bwd(res, do_i):
          q_i, k_i, v_i, o, lse = res
          return qsa_pallas.qsa_pallas_bwd(
              q_i,
              k_i,
              v_i,
              o,
              lse,
              do_i,
              pmask,
              jnp.int32(q_offset),
              scale=scale,
              interpret=strict_interpret,
              bkv=bq_v,
              bkv_dkv=bdkv_v,
              disable_bounds_checks=False,
          )

        _op.defvjp(_op_fwd, _op_bwd)
        out_p = _op(qq, kk, vv)
        return jnp.sum(out_p.astype(jnp.float32) * cot), out_p

      (pl_loss, pl_out), (pl_dq, pl_dk, pl_dv) = jax.jit(
          jax.value_and_grad(_pallas_loss, argnums=(0, 1, 2), has_aux=True)
      )(q, k, v)
      tag = f"bkv={bkv_q}, bkv_dkv={bkv_dkv}"
      np.testing.assert_allclose(float(pl_loss), float(ref_loss), rtol=1e-4, atol=1e-4, err_msg=tag)
      np.testing.assert_allclose(np.asarray(pl_out), np.asarray(ref_out), rtol=1e-4, atol=1e-4, err_msg=tag)
      np.testing.assert_allclose(np.asarray(pl_dq), np.asarray(ref_dq), rtol=1e-4, atol=1e-4, err_msg=f"dq {tag}")
      np.testing.assert_allclose(np.asarray(pl_dk), np.asarray(ref_dk), rtol=1e-4, atol=1e-4, err_msg=f"dk {tag}")
      np.testing.assert_allclose(np.asarray(pl_dv), np.asarray(ref_dv), rtol=1e-4, atol=1e-4, err_msg=f"dv {tag}")

  def test_cpu_balanced_cp_zigzag_and_pallas_fallback_on_padded_packed_inputs(self):
    """Verifies Zig-Zag `cp_size=2` `qsa_pallas` parity + `attention_kernel='pallas'` CPU/padded/packed fallback."""
    if _RUN_TPU_QSA_BENCH.value:
      self.skipTest("CPU test skipped when --run_tpu_qsa_bench=true.")
    if len(jax.devices()) < 2:
      self.skipTest("Requires >= 2 host devices.")

    rng = np.random.default_rng(77)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:2]), ("expert",))
    P = jax.sharding.PartitionSpec
    sharding_cp = m3_sharding.EPasCP()

    # 1. Direct Zig-Zag `cp_size=2` `qsa_pallas_fwd` + `qsa_pallas_bwd` under `InterpretParams` vs exact reference.
    s_cp = 1024
    h_q, h_kv, d = 2, 1, 128
    scale = 1.0 / math.sqrt(float(d))
    q_cp = jnp.asarray(rng.standard_normal((1, s_cp, h_q, d)).astype(np.float32) * 0.2)
    k_cp = jnp.asarray(rng.standard_normal((1, s_cp, h_kv, d)).astype(np.float32) * 0.2)
    v_cp = jnp.asarray(rng.standard_normal((1, s_cp, h_kv, d)).astype(np.float32) * 0.2)
    cot_cp = jnp.asarray(rng.standard_normal((1, s_cp, h_q, d)).astype(np.float32) * 0.2)
    sbi_cp = jnp.asarray(rng.integers(0, s_cp // 4, size=(1, s_cp, 8), dtype=np.int32))
    causal_blk = (jnp.arange(s_cp, dtype=jnp.int32)[None, :, None] + 1) // 4
    sbi_cp = jnp.where(sbi_cp < causal_blk, sbi_cp, -1)
    pm_cp = modeling_qwen3_8_flash_next.pack_qsa_block_mask_32bit(sbi_cp, s_cp // 128)

    (ref_loss, ref_o), (ref_dq, ref_dk, ref_dv) = jax.value_and_grad(
        lambda qq, kk, vv: _reference_qsa_loss(qq, kk, vv, pm_cp, cot_cp, q_offset=0, scale=scale),
        argnums=(0, 1, 2),
        has_aux=True,
    )(q_cp, k_cp, v_cp)

    strict_interpret = pltpu.InterpretParams()
    with jax.set_mesh(mesh):
      ns_seq = jax.sharding.NamedSharding(mesh, P(None, "expert", None, None))
      ns_pm = jax.sharding.NamedSharding(mesh, P(None, "expert", None))
      q_s = jax.device_put(q_cp, ns_seq)
      k_s = jax.device_put(k_cp, ns_seq)
      v_s = jax.device_put(v_cp, ns_seq)
      cot_s = jax.device_put(cot_cp, ns_seq)
      pm_s = jax.device_put(pm_cp, ns_pm)

      @functools.partial(
          jax.shard_map,
          mesh=mesh,
          in_specs=(
              P(None, "expert", None, None),
              P(None, "expert", None, None),
              P(None, "expert", None, None),
              P(None, "expert", None),
              P(None, "expert", None, None),
          ),
          out_specs=(
              P(),
              P(None, "expert", None, None),
              P(None, "expert", None, None),
              P(None, "expert", None, None),
              P(None, "expert", None, None),
          ),
          check_vma=False,
      )
      def _run_zigzag_cp2(q_l, k_l, v_l, pm_l, cot_l):
        def _inner(qq, kk, vv):
          b_l, s_l, h_l, d_l = kk.shape
          kv_2 = jnp.transpose(jnp.stack([kk, vv], axis=1), (0, 1, 3, 2, 4))
          kv_ag = jax.lax.all_gather(jnp.reshape(kv_2, (b_l * 2 * h_l * s_l, d_l)), "expert", axis=0, tiled=False)
          kv_bhksd = jnp.reshape(
              jnp.transpose(jnp.reshape(kv_ag, (2, b_l, 2, h_l, s_l, d_l)), (1, 2, 3, 0, 4, 5)),
              (b_l, 2, h_l, 2 * s_l, d_l),
          )
          k_bhkd, v_bhkd = kv_bhksd[:, 0], kv_bhksd[:, 1]

          @jax.custom_vjp
          def _op(q_i, kb_i, vb_i):
            o, _ = qsa_pallas.qsa_pallas_fwd(
                q_i,
                kb_i,
                vb_i,
                pm_l,
                jnp.int32(0),
                scale=scale,
                interpret=strict_interpret,
                cp_axis="expert",
                cp_size=2,
                kv_is_bhkd=True,
                bkv=512,
                disable_bounds_checks=False,
            )
            return o

          def _op_fwd(q_i, kb_i, vb_i):
            o, lse = qsa_pallas.qsa_pallas_fwd(
                q_i,
                kb_i,
                vb_i,
                pm_l,
                jnp.int32(0),
                scale=scale,
                interpret=strict_interpret,
                cp_axis="expert",
                cp_size=2,
                kv_is_bhkd=True,
                bkv=512,
                disable_bounds_checks=False,
            )
            return o, (q_i, kb_i, vb_i, o, lse)

          def _op_bwd(res, do_i):
            q_i, kb_i, vb_i, o, lse = res
            return qsa_pallas.qsa_pallas_bwd(
                q_i,
                kb_i,
                vb_i,
                o,
                lse,
                do_i,
                pm_l,
                jnp.int32(0),
                scale=scale,
                interpret=strict_interpret,
                cp_axis="expert",
                cp_size=2,
                kv_is_bhkd=True,
                bkv=512,
                bkv_dkv=512,
                disable_bounds_checks=False,
            )

          _op.defvjp(_op_fwd, _op_bwd)
          out_l = _op(qq, k_bhkd, v_bhkd)
          return jnp.sum(out_l * cot_l), out_l

        (loss_l, out_l), (dq_l, dk_l, dv_l) = jax.value_and_grad(_inner, argnums=(0, 1, 2), has_aux=True)(q_l, k_l, v_l)
        return jax.lax.psum(loss_l, "expert"), out_l, dq_l, dk_l, dv_l

      pl_loss, pl_o, pl_dq, pl_dk, pl_dv = jax.jit(_run_zigzag_cp2)(q_s, k_s, v_s, pm_s, cot_s)
      np.testing.assert_allclose(float(pl_loss), float(ref_loss), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(pl_o), np.asarray(ref_o), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(pl_dq), np.asarray(ref_dq), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(pl_dk), np.asarray(ref_dk), rtol=1e-4, atol=1e-4)
      np.testing.assert_allclose(np.asarray(pl_dv), np.asarray(ref_dv), rtol=1e-4, atol=1e-4)

      # 2. Verify `Qwen3_8SparseAttention(attention_kernel="pallas")` matches `"flash"` across short (`s <= budget`),
      #    unpacked (`s > budget`), right-padded (`attention_mask`), and packed (`segment_ids`) calls on CPU + EPasCP.
      args_qsa = (64, 4, 2, 32, 0.25, 10000.0, (2, 1, 1), 2, 1, 32, 16, 4, 1e-6, jnp.float32, jnp.float32)
      qsa_pl = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
          *args_qsa, sharding=sharding_cp, attention_kernel="pallas", pallas_bkv=512, pallas_bkv_dkv=512, rngs=nnx.Rngs(9)
      )
      qsa_fl = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
          *args_qsa, sharding=sharding_cp, attention_kernel="flash", rngs=nnx.Rngs(9)
      )
      nnx.update(qsa_pl, nnx.state(qsa_fl))
      for s_test in (16, 48):
        x_t = jnp.asarray(rng.standard_normal((2, s_test, 64), dtype=np.float32) * 0.2)
        pos_t = jnp.tile(jnp.arange(s_test, dtype=jnp.int32), (2, 1))
        mask_t = (jnp.arange(s_test, dtype=jnp.int32)[None, :] < jnp.array([[s_test], [s_test - 4]])).astype(jnp.int32)
        seg_t = jnp.ones((2, s_test), dtype=jnp.int32).at[:, s_test // 2 :].set(2)
        for am_i, sg_i in ((None, None), (mask_t, None), (None, seg_t)):
          v_p, dx_p = nnx.value_and_grad(
              lambda m, inp, pos=pos_t, am=am_i, sg=sg_i: jnp.sum(
                  m(inp, position_ids=pos, attention_mask=am, segment_ids=sg)
              ),
              argnums=(1,),
          )(qsa_pl, x_t)
          v_f, dx_f = nnx.value_and_grad(
              lambda m, inp, pos=pos_t, am=am_i, sg=sg_i: jnp.sum(
                  m(inp, position_ids=pos, attention_mask=am, segment_ids=sg)
              ),
              argnums=(1,),
          )(qsa_fl, x_t)
          np.testing.assert_allclose(float(v_p), float(v_f), rtol=1e-4, atol=1e-4)
          np.testing.assert_allclose(np.asarray(dx_p[0]), np.asarray(dx_f[0]), rtol=1e-4, atol=1e-4)

  def test_cpu_single_q_tile_ring_and_streaming_indexer_parity(self):
    """Verifies `max_act_kv == 1` 3-slot `dq` ring, `_qsa_indexer_relu_sum_scores`, and `pack_qsa_block_mask_32bit_vpu`."""
    if _RUN_TPU_QSA_BENCH.value:
      self.skipTest("CPU test skipped when --run_tpu_qsa_bench=true.")

    rng = np.random.default_rng(20261122)
    b_loc, s_q_loc, s_kv = 1, 128, 1024
    h_kv, h_q, d = 1, 2, 128
    q_offset = s_kv - s_q_loc
    scale = 1.0 / math.sqrt(float(d))
    sbi = np.broadcast_to(np.array([[[2, 35, 70, 110, 145, 180, 215, 240]]], dtype=np.int32), (b_loc, s_q_loc, 8))
    pmask = modeling_qwen3_8_flash_next.pack_qsa_block_mask_32bit(jnp.asarray(sbi), s_kv // 128, assume_unique=True)

    q = jnp.asarray(rng.standard_normal((b_loc, s_q_loc, h_q, d)).astype(np.float32) * 0.25)
    k = jnp.asarray(rng.standard_normal((b_loc, s_kv, h_kv, d)).astype(np.float32) * 0.25)
    v = jnp.asarray(rng.standard_normal((b_loc, s_kv, h_kv, d)).astype(np.float32) * 0.25)
    cot = jnp.asarray(rng.standard_normal((b_loc, s_q_loc, h_q, d)).astype(np.float32) * 0.25)

    (ref_loss, _), (ref_dq, ref_dk, ref_dv) = jax.value_and_grad(
        lambda qq, kk, vv: _reference_qsa_loss(qq, kk, vv, pmask, cot, q_offset=q_offset, scale=scale),
        argnums=(0, 1, 2),
        has_aux=True,
    )(q, k, v)

    strict_interpret = pltpu.InterpretParams(detect_races=True)
    o, lse = qsa_pallas.qsa_pallas_fwd(
        q, k, v, pmask, jnp.int32(q_offset), scale=scale, interpret=strict_interpret, bkv=128, disable_bounds_checks=False
    )
    dq_p, dk_p, dv_p = qsa_pallas.qsa_pallas_bwd(
        q,
        k,
        v,
        o,
        lse,
        cot,
        pmask,
        jnp.int32(q_offset),
        scale=scale,
        interpret=strict_interpret,
        bkv=128,
        bkv_dkv=128,
        disable_bounds_checks=False,
    )
    np.testing.assert_allclose(float(jnp.sum(o * cot)), float(ref_loss), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(dq_p), np.asarray(ref_dq), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(dk_p), np.asarray(ref_dk), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(dv_p), np.asarray(ref_dv), rtol=1e-4, atol=1e-4)

    q_idx_np = rng.standard_normal((2, 16, 4, 128)).astype(np.float32)
    bk_idx_np = rng.standard_normal((2, 32, 128)).astype(np.float32)
    for dt in (jnp.bfloat16, jnp.float32):
      q_dt = jnp.asarray(q_idx_np, dtype=dt)
      bk_dt = jnp.asarray(bk_idx_np, dtype=dt)
      got = modeling_qwen3_8_flash_next._qsa_indexer_raw_scores(q_dt, bk_dt)  # pylint: disable=protected-access
      exp = jnp.einsum("bthd,bmd->bthm", q_dt.astype(jnp.float32), bk_dt.astype(jnp.float32))
      np.testing.assert_allclose(np.asarray(got), np.asarray(exp), rtol=1e-5, atol=1e-5)
      got_relu = modeling_qwen3_8_flash_next._qsa_indexer_relu_sum_scores(q_dt, bk_dt)  # pylint: disable=protected-access
      exp_relu = jnp.sum(jax.nn.relu(exp), axis=2)
      np.testing.assert_allclose(np.asarray(got_relu), np.asarray(exp_relu), rtol=1e-5, atol=1e-5)

    bv_idx = jnp.ones((2, 32), dtype=bool)
    bseg_idx = jnp.ones((2, 32), dtype=jnp.int32)
    gstart_idx = jnp.broadcast_to((jnp.arange(32, dtype=jnp.int32) * 4)[None, :], (2, 32))
    seg_idx = jnp.ones((2, 16), dtype=jnp.int32)
    tv_tiled, ti_tiled = modeling_qwen3_8_flash_next.tiled_qsa_top_k(
        jnp.asarray(q_idx_np, dtype=jnp.bfloat16),
        jnp.asarray(bk_idx_np, dtype=jnp.bfloat16),
        bv_idx,
        bseg_idx,
        gstart_idx,
        seg_idx,
        compress_ratio=4,
        index_head_dim=128,
        k_top=8,
        block_chunk_size=16,
        q_pos_offset=128,
    )
    tv_qc, ti_qc = modeling_qwen3_8_flash_next.query_chunk_qsa_top_k(
        jnp.asarray(q_idx_np, dtype=jnp.bfloat16),
        jnp.asarray(bk_idx_np, dtype=jnp.bfloat16),
        bv_idx,
        bseg_idx,
        gstart_idx,
        seg_idx,
        compress_ratio=4,
        index_head_dim=128,
        k_top=8,
        q_chunk_size=8,
        q_pos_offset=128,
    )
    np.testing.assert_allclose(np.asarray(tv_tiled), np.asarray(tv_qc), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(ti_tiled), np.asarray(ti_qc))

    sbi_vpu = jnp.asarray(rng.integers(-1, 16384, size=(1, 1536, 16), dtype=np.int32))
    pm_vpu = modeling_qwen3_8_flash_next.pack_qsa_block_mask_32bit_vpu(sbi_vpu, 512)
    pm_ref = modeling_qwen3_8_flash_next.pack_qsa_block_mask_32bit(sbi_vpu, 512)
    np.testing.assert_array_equal(np.asarray(pm_vpu), np.asarray(pm_ref))

  def test_tpu_mosaic_parity_module_dispatch_and_64k_benchmark(self):
    """Runs Mosaic TPU parity (`Qwen3_8SparseAttention('pallas')` vs `'flash'`) and 8-TC CP=8 64K bench."""
    if not _RUN_TPU_QSA_BENCH.value:
      self.skipTest("Pass --run_tpu_qsa_bench=true on TPU v7x 2x2x1.")
    self.assertEqual(jax.default_backend(), "tpu")
    devices = jax.devices()
    self.assertGreaterEqual(len(devices), 8)

    rng = np.random.default_rng(20261121)
    P = jax.sharding.PartitionSpec

    # Part A: Full `Qwen3_8SparseAttention(attention_kernel="pallas")` vs `"flash"` parity on `EPasCP` (`fsdp=2, expert=4`).
    mesh_epcp = jax.sharding.Mesh(np.array(devices[:8]).reshape(2, 4), ("fsdp", "expert"))
    epcp = m3_sharding.EPasCP()
    b_m, s_m, emb_m = 2, 2048, 256
    x_m_np = rng.standard_normal((b_m, s_m, emb_m)).astype(np.float32) * 0.2
    cot_m_np = rng.standard_normal((b_m, s_m, emb_m)).astype(np.float32) * 0.2
    with jax.set_mesh(mesh_epcp), nnx.use_eager_sharding(True):
      mod_pallas = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
          emb_dim=emb_m,
          num_query_heads=8,
          num_kv_heads=2,
          head_dim=128,
          mrope_section=(6, 6, 4),
          index_n_heads=4,
          index_kv_heads=1,
          index_head_dim=128,
          token_budget=256,
          compress_ratio=4,
          dtype=jnp.bfloat16,
          weight_dtype=jnp.float32,
          sharding=epcp,
          attention_kernel="pallas",
          pallas_bkv=512,
          pallas_bkv_dkv=512,
          rngs=nnx.Rngs(params=17),
      )
      mod_flash = modeling_qwen3_8_flash_next.Qwen3_8SparseAttention(
          emb_dim=emb_m,
          num_query_heads=8,
          num_kv_heads=2,
          head_dim=128,
          mrope_section=(6, 6, 4),
          index_n_heads=4,
          index_kv_heads=1,
          index_head_dim=128,
          token_budget=256,
          compress_ratio=4,
          dtype=jnp.bfloat16,
          weight_dtype=jnp.float32,
          sharding=epcp,
          attention_kernel="flash",
          rngs=nnx.Rngs(params=17),
      )
      nnx.update(mod_flash, nnx.state(mod_pallas))
      m3_sharding.end_eager_placement(mod_pallas)
      m3_sharding.end_eager_placement(mod_flash)

    with jax.set_mesh(mesh_epcp):
      ns_act = jax.sharding.NamedSharding(mesh_epcp, P("fsdp", "expert", None))
      x_sh_m = jax.device_put(jnp.asarray(x_m_np, dtype=jnp.bfloat16), ns_act)
      cot_sh_m = jax.device_put(jnp.asarray(cot_m_np, dtype=jnp.bfloat16), ns_act)

      def _mod_loss(mod, xx):
        y = mod(xx)
        return jnp.sum(y.astype(jnp.float32) * cot_sh_m.astype(jnp.float32)), y

      hlo_pallas = nnx.jit(lambda m, xx: m(xx)).lower(mod_pallas, x_sh_m).as_text()
      self.assertIn("qsa_pallas_fwd", hlo_pallas)

      (lp, yp), (gp, dxp) = nnx.jit(nnx.value_and_grad(_mod_loss, argnums=(0, 1), has_aux=True))(mod_pallas, x_sh_m)
      (lf, yf), (gf, dxf) = nnx.jit(nnx.value_and_grad(_mod_loss, argnums=(0, 1), has_aux=True))(mod_flash, x_sh_m)
      np.testing.assert_allclose(float(lp), float(lf), rtol=2e-2, atol=2e-2)
      np.testing.assert_allclose(np.asarray(yp, dtype=np.float32), np.asarray(yf, dtype=np.float32), rtol=3e-2, atol=3e-2)
      np.testing.assert_allclose(
          np.asarray(dxp, dtype=np.float32), np.asarray(dxf, dtype=np.float32), rtol=4e-2, atol=4e-2
      )
      np.testing.assert_allclose(
          np.asarray(gp.q_proj.kernel[...], dtype=np.float32),
          np.asarray(gf.q_proj.kernel[...], dtype=np.float32),
          rtol=4e-2,
          atol=4e-2,
      )

      # Also verify right-padded `attention_mask` on TPU cleanly falls back to `flash_block_sparse_attention`.
      mask_m = jax.device_put(
          (jnp.arange(s_m, dtype=jnp.int32)[None, :] < jnp.array([[s_m], [s_m - 16]])).astype(jnp.int32),
          jax.sharding.NamedSharding(mesh_epcp, P("fsdp", None)),
      )
      yp_pad = nnx.jit(lambda m, xx, am: m(xx, attention_mask=am))(mod_pallas, x_sh_m, mask_m)
      yf_pad = nnx.jit(lambda m, xx, am: m(xx, attention_mask=am))(mod_flash, x_sh_m, mask_m)
      np.testing.assert_allclose(
          np.asarray(yp_pad, dtype=np.float32), np.asarray(yf_pad, dtype=np.float32), rtol=2e-2, atol=2e-2
      )

    # Part B: Exact 8-TC `CP=8` 64K QSA kernel benchmark (`b_loc=2, s_q_loc=8192, s_kv=65536, h_q=24, h_kv=2, d=256`).
    mesh8 = jax.sharding.Mesh(np.array(devices[:8]), ("cp",))
    b_b, s_loc_b, s_kv_b, h_kv_b, h_q_b, d_b = 2, 8192, 65536, 2, 24, 256
    scale64 = 1.0 / math.sqrt(float(d_b))
    with jax.set_mesh(mesh8):

      @functools.partial(
          jax.shard_map,
          mesh=mesh8,
          in_specs=(),
          out_specs=(
              P(None, "cp", None, None),
              P(None, "cp", None, None),
              P(None, "cp", None, None),
              P(None, "cp", None, None),
              P(None, "cp", None),
          ),
          check_vma=False,
      )
      def _init_64k_inputs():
        r = jax.lax.axis_index("cp")
        key = jax.random.fold_in(jax.random.PRNGKey(42), r)
        kq, kk, kv, kc, km = jax.random.split(key, 5)
        q_l = jax.random.normal(kq, (b_b, s_loc_b, h_q_b, d_b), dtype=jnp.bfloat16) * 0.2
        k_l = jax.random.normal(kk, (b_b, s_loc_b, h_kv_b, d_b), dtype=jnp.bfloat16) * 0.2
        v_l = jax.random.normal(kv, (b_b, s_loc_b, h_kv_b, d_b), dtype=jnp.bfloat16) * 0.2
        do_l = jax.random.normal(kc, (b_b, s_loc_b, h_q_b, d_b), dtype=jnp.bfloat16) * 0.2
        q_pos = r * s_loc_b + jnp.arange(s_loc_b, dtype=jnp.int32)[None, :, None]
        max_blk = jnp.maximum((q_pos + 1) // 4, 1)
        raw_sbi = jax.random.randint(km, (b_b, s_loc_b, 512), 0, s_kv_b // 4, dtype=jnp.int32) % max_blk
        sbi_l = jnp.where(q_pos >= 3, raw_sbi, -1)
        pm_l = modeling_qwen3_8_flash_next.pack_qsa_block_mask_32bit_vpu(sbi_l, s_kv_b // 128)
        return q_l, k_l, v_l, do_l, pm_l

      q64, k64, v64, do64, pm64 = jax.jit(_init_64k_inputs)()
      jax.block_until_ready((q64, k64, v64, do64, pm64))

      @functools.partial(
          jax.shard_map,
          mesh=mesh8,
          in_specs=(
              P(None, "cp", None, None),
              P(None, "cp", None, None),
              P(None, "cp", None, None),
              P(None, "cp", None),
          ),
          out_specs=(P(None, "cp", None, None), P(None, None, "cp")),
          check_vma=False,
      )
      def _fwd_64k(q_l, k_l, v_l, pm_l):
        kv_2 = jnp.transpose(jnp.stack([k_l, v_l], axis=1), (0, 1, 3, 2, 4))
        kv_ag = jax.lax.all_gather(jnp.reshape(kv_2, (b_b * 2 * h_kv_b * s_loc_b, d_b)), "cp", axis=0, tiled=False)
        kv_bhksd = jnp.reshape(
            jnp.transpose(jnp.reshape(kv_ag, (8, b_b, 2, h_kv_b, s_loc_b, d_b)), (1, 2, 3, 0, 4, 5)),
            (b_b, 2, h_kv_b, s_kv_b, d_b),
        )
        return qsa_pallas.qsa_pallas_fwd(
            q_l,
            kv_bhksd[:, 0],
            kv_bhksd[:, 1],
            pm_l,
            jnp.int32(0),
            scale=scale64,
            interpret=False,
            cp_axis="cp",
            cp_size=8,
            kv_is_bhkd=True,
            bkv=2048,
        )

      @functools.partial(
          jax.shard_map,
          mesh=mesh8,
          in_specs=(
              P(None, "cp", None, None),
              P(None, "cp", None, None),
              P(None, "cp", None, None),
              P(None, "cp", None, None),
              P(None, None, "cp"),
              P(None, "cp", None, None),
              P(None, "cp", None),
          ),
          out_specs=(P(None, "cp", None, None), P(None, "cp", None, None), P(None, "cp", None, None)),
          check_vma=False,
      )
      def _bwd_64k(q_l, k_l, v_l, o_l, lse_l, do_l, pm_l):
        kv_2 = jnp.transpose(jnp.stack([k_l, v_l], axis=1), (0, 1, 3, 2, 4))
        kv_ag = jax.lax.all_gather(jnp.reshape(kv_2, (b_b * 2 * h_kv_b * s_loc_b, d_b)), "cp", axis=0, tiled=False)
        kv_bhksd = jnp.reshape(
            jnp.transpose(jnp.reshape(kv_ag, (8, b_b, 2, h_kv_b, s_loc_b, d_b)), (1, 2, 3, 0, 4, 5)),
            (b_b, 2, h_kv_b, s_kv_b, d_b),
        )
        dq_l, dk_bhkd, dv_bhkd = qsa_pallas.qsa_pallas_bwd(
            q_l,
            kv_bhksd[:, 0],
            kv_bhksd[:, 1],
            o_l,
            lse_l,
            do_l,
            pm_l,
            jnp.int32(0),
            scale=scale64,
            interpret=False,
            cp_axis="cp",
            cp_size=8,
            kv_is_bhkd=True,
            bkv=2048,
            bkv_dkv=2048,
        )
        dkv_6d = jnp.reshape(jnp.stack([dk_bhkd, dv_bhkd], axis=1), (b_b, 2, h_kv_b, 8, s_loc_b, d_b))
        dkv_3d = jnp.reshape(jnp.transpose(dkv_6d, (3, 0, 1, 2, 4, 5)), (8, b_b * 2 * h_kv_b * s_loc_b, d_b))
        dkv_sh = jax.lax.psum_scatter(dkv_3d, "cp", scatter_dimension=0, tiled=False)
        dkv_b2shd = jnp.transpose(jnp.reshape(dkv_sh, (b_b, 2, h_kv_b, s_loc_b, d_b)), (0, 1, 3, 2, 4))
        return dq_l, dkv_b2shd[:, 0].astype(k_l.dtype), dkv_b2shd[:, 1].astype(v_l.dtype)

      j_fwd = jax.jit(_fwd_64k)
      j_bwd = jax.jit(_bwd_64k)
      o64, lse64 = j_fwd(q64, k64, v64, pm64)
      jax.block_until_ready((o64, lse64))
      dq64, dk64, dv64 = j_bwd(q64, k64, v64, o64, lse64, do64, pm64)
      jax.block_until_ready((dq64, dk64, dv64))

      fwd_times: list[float] = []
      bwd_times: list[float] = []
      for _ in range(5):
        t0 = time.perf_counter()
        o64, lse64 = j_fwd(q64, k64, v64, pm64)
        jax.block_until_ready((o64, lse64))
        fwd_times.append((time.perf_counter() - t0) * 1000.0)
        t1 = time.perf_counter()
        dq64, dk64, dv64 = j_bwd(q64, k64, v64, o64, lse64, do64, pm64)
        jax.block_until_ready((dq64, dk64, dv64))
        bwd_times.append((time.perf_counter() - t1) * 1000.0)

      fwd_ms = float(np.median(fwd_times))
      bwd_ms = float(np.median(bwd_times))
      tot_12_ms = 12.0 * (fwd_ms + bwd_ms)
      print(f"[BENCH_64K_CP8] bkv=(2048,2048): fwd={fwd_ms:.2f} ms, bwd={bwd_ms:.2f} ms, 12x={tot_12_ms:.1f} ms")
      self.assertLess(tot_12_ms, 1350.0)


if __name__ == "__main__":
  absltest.main()

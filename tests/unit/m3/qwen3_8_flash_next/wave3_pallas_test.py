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

"""CPU (`interpret=True`) and TPUv7x (`TPU v7x 2x2x1`) parity & microbenchmark tests for `hc_pallas` and `gdn_pallas`."""

import time
from typing import Any, cast

from absl import flags
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from maxtext.configs import pyconfig  # pylint: disable=no-name-in-module
from maxtext.m3.core import gdn_pallas  # pylint: disable=no-name-in-module
from maxtext.m3.core import hc_pallas  # pylint: disable=no-name-in-module
from maxtext.m3.core import sharding as m3_sharding  # pylint: disable=no-name-in-module
from maxtext.m3.infra import quantization as m3_quantization  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3_8_flash_next import modeling_qwen3_8_flash_next  # pylint: disable=no-name-in-module
from maxtext.utils import globals as maxtext_globals  # pylint: disable=no-name-in-module
import numpy as np

_RUN_TPU_WAVE3_BENCH = flags.DEFINE_bool(
    "run_tpu_wave3_bench",
    False,
    "Run Mosaic TPUv7x parity, HLO checks, and production-shape microbenchmarks on TPU v7x 2x2x1.",
)


def _rel_l2(got: jax.Array | np.ndarray, ref: jax.Array | np.ndarray) -> float:
  a = np.asarray(got, dtype=np.float32)
  b = np.asarray(ref, dtype=np.float32)
  denom = max(float(np.linalg.norm(b)), 1e-6)
  return float(np.linalg.norm(a - b)) / denom


class Wave3PallasTest(absltest.TestCase):
  """Verifies `hc_pallas` and `gdn_pallas` numerical parity, sharding, and TPUv7x dispatch."""

  def test_cpu_pyconfig_flags_and_fp8_full_wiring(self):
    """Verifies `use_hc_pallas` and `use_gdn_conv_pallas` config wiring and CPU fallback under `fp8_full`."""
    if _RUN_TPU_WAVE3_BENCH.value:
      self.skipTest("CPU test skipped when --run_tpu_wave3_bench=true.")

    cfg = pyconfig.initialize(
        [
            "",
            f"{maxtext_globals.MAXTEXT_CONFIGS_DIR}/base.yml",
            "model_name=qwen3.8-flash-next-mini",
            "override_model_config=True",
            "run_name=gdn_hc_pallas_flag_test",
            "skip_jax_distributed_system=True",
            "enable_checkpointing=False",
            "dtype=bfloat16",
            "weight_dtype=bfloat16",
            "quantization=fp8_full",
            "use_qwix_quantization=True",
            "quantize_router_proj=False",
            "use_hc_pallas=True",
            "use_gdn_conv_pallas=True",
            "use_ple_layer=False",
            "mtp_num_layers=1",
            "scan_layers=False",
        ]
    )
    self.assertTrue(cfg.use_hc_pallas)
    self.assertTrue(cfg.use_gdn_conv_pallas)

    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[: min(4, len(devices))]), ("fsdp",))
    shd = m3_sharding.DefaultSharding()
    with jax.set_mesh(mesh):
      model = modeling_qwen3_8_flash_next.create_qwen3_8_flash_next_model(cfg, mesh=mesh, sharding=shd)
      ex_tok = jax.ShapeDtypeStruct((len(mesh.devices), 16), jnp.int32)
      model = m3_quantization.maybe_quantize(model, cfg, ex_tok)
      hc_mods = [m for _, m in nnx.iter_modules(model) if isinstance(m, modeling_qwen3_8_flash_next.Qwen3_8GatedResidual)]
      gdn_mods = [
          m for _, m in nnx.iter_modules(model) if isinstance(m, modeling_qwen3_8_flash_next.Qwen3_8GatedDeltaNet)
      ]
      self.assertNotEmpty(hc_mods)
      self.assertNotEmpty(gdn_mods)
      for mod in hc_mods:
        self.assertTrue(mod.enable_hc_pallas)
        self.assertIs(type(mod.input_mix_weight_down), nnx.Linear)
        self.assertIs(type(mod.input_mix_weight_up), nnx.Linear)
      for mod in gdn_mods:
        self.assertTrue(mod.enable_conv_pallas)

      # Verify CPU cleanly falls back to pure-JAX without raising on Pallas Mosaic lowering.
      tokens = jax.device_put(
          jnp.ones((len(mesh.devices), 16), dtype=jnp.int32),
          jax.sharding.NamedSharding(mesh, P("fsdp", None)),
      )
      logits = model(tokens)
      self.assertEqual(logits.shape, (len(mesh.devices), 16, cfg.vocab_size))

  def test_cpu_hc_pallas_interpret_and_sharded_parity(self):
    """Verifies `hc_pallas` (`before_block`, `mix_input`, `combine`) fwd+bwd parity under `interpret=True`."""
    if _RUN_TPU_WAVE3_BENCH.value:
      self.skipTest("CPU test skipped when --run_tpu_wave3_bench=true.")

    rng = np.random.default_rng(2026112501)
    emb_dim, hc_count, hc_lowrank = 256, 4, 32
    hc_dim = emb_dim * hc_count

    with m3_sharding.allow_unsharded_construction():
      hc_ref = modeling_qwen3_8_flash_next.Qwen3_8GatedResidual(
          emb_dim, hc_count, hc_lowrank, use_combine=True, dtype=jnp.float32, weight_dtype=jnp.float32, rngs=nnx.Rngs(7)
      )
      sc_init = jnp.asarray(rng.standard_normal((hc_dim,)).astype(np.float32) * 0.1)
      hc_ref.hc_norm.scale[...] = sc_init

      mixer_ref = modeling_qwen3_8_flash_next.Qwen3_8GatedResidual(
          emb_dim,
          hc_count,
          hc_lowrank,
          use_combine=False,
          dtype=jnp.float32,
          weight_dtype=jnp.float32,
          rngs=nnx.Rngs(11),
      )

    # Unaligned `S = 140` exercises tail padding + masking inside `hc_before_block_local` and `hc_combine_local`.
    x_1d = jnp.asarray(rng.standard_normal((1, 140, hc_dim)).astype(np.float32) * 0.4)
    blk_1d = jnp.asarray(rng.standard_normal((1, 140, emb_dim)).astype(np.float32) * 0.4)
    cot_comb = jnp.asarray(rng.standard_normal((1, 140, hc_dim)).astype(np.float32) * 0.3)
    cot_mix = jnp.asarray(rng.standard_normal((1, 140, emb_dim)).astype(np.float32) * 0.3)

    sc_w = hc_ref.hc_norm.scale[...]
    wd_w = hc_ref.input_mix_weight_down.kernel[...]
    wu_w = hc_ref.input_mix_weight_up.kernel[...]
    assert hc_ref.block_inject_weight is not None
    wi_w = hc_ref.block_inject_weight.kernel[...]

    def _ref_hc_loss(m, xx, bb):
      mixed, inj = m.before_block(xx)
      comb = m.combine(xx, bb + mixed, inj)
      return jnp.sum(comb * cot_comb)

    def _pl_hc_loss(xx, bb, sc, wd, wu, wi):
      mixed, inj = hc_pallas.hc_before_block_local(
          xx, sc, wd, wu, wi, emb_dim=emb_dim, hc_count=hc_count, hc_lowrank=hc_lowrank, eps=1e-6, interpret=True
      )
      comb = hc_pallas.hc_combine_local(xx, bb + mixed, inj, emb_dim=emb_dim, hc_count=hc_count, interpret=True)
      return jnp.sum(comb * cot_comb)

    v_rf, (gm_rf, gx_rf, gb_rf) = nnx.value_and_grad(_ref_hc_loss, argnums=(0, 1, 2))(hc_ref, x_1d, blk_1d)
    v_pl, (gx_pl, gb_pl, gsc_pl, gwd_pl, gwu_pl, gwi_pl) = jax.jit(
        jax.value_and_grad(_pl_hc_loss, argnums=(0, 1, 2, 3, 4, 5))
    )(x_1d, blk_1d, sc_w, wd_w, wu_w, wi_w)
    np.testing.assert_allclose(float(v_pl), float(v_rf), rtol=1e-4, atol=1e-4)
    self.assertLess(_rel_l2(gx_pl, gx_rf), 1e-4)
    self.assertLess(_rel_l2(gb_pl, gb_rf), 1e-4)
    self.assertLess(_rel_l2(gsc_pl, gm_rf.hc_norm.scale[...]), 1e-4)
    self.assertLess(_rel_l2(gwd_pl, gm_rf.input_mix_weight_down.kernel[...]), 1e-4)
    self.assertLess(_rel_l2(gwu_pl, gm_rf.input_mix_weight_up.kernel[...]), 1e-4)
    assert gm_rf.block_inject_weight is not None
    self.assertLess(_rel_l2(gwi_pl, gm_rf.block_inject_weight.kernel[...]), 1e-4)

    # `mix_input` (`w_inject=None`) parity.
    sc_m = mixer_ref.hc_norm.scale[...]
    wd_m = mixer_ref.input_mix_weight_down.kernel[...]
    wu_m = mixer_ref.input_mix_weight_up.kernel[...]
    vm_rf, (gmix_rf, gx_m_rf) = nnx.value_and_grad(lambda m, xx: jnp.sum(m.mix_input(xx) * cot_mix), argnums=(0, 1))(
        mixer_ref, x_1d
    )
    vm_pl, (gx_m_pl, gsc_m_pl, gwd_m_pl, gwu_m_pl) = jax.jit(
        jax.value_and_grad(
            lambda xx, sc, wd, wu: jnp.sum(
                hc_pallas.hc_before_block_local(
                    xx,
                    sc,
                    wd,
                    wu,
                    None,
                    emb_dim=emb_dim,
                    hc_count=hc_count,
                    hc_lowrank=hc_lowrank,
                    eps=1e-6,
                    interpret=True,
                )[0]
                * cot_mix
            ),
            argnums=(0, 1, 2, 3),
        )
    )(x_1d, sc_m, wd_m, wu_m)
    np.testing.assert_allclose(float(vm_pl), float(vm_rf), rtol=1e-4, atol=1e-4)
    self.assertLess(_rel_l2(gx_m_pl, gx_m_rf), 1e-4)
    self.assertLess(_rel_l2(gsc_m_pl, gmix_rf.hc_norm.scale[...]), 1e-4)
    self.assertLess(_rel_l2(gwd_m_pl, gmix_rf.input_mix_weight_down.kernel[...]), 1e-4)
    self.assertLess(_rel_l2(gwu_m_pl, gmix_rf.input_mix_weight_up.kernel[...]), 1e-4)

    # 4-CPU `EPasCP()` `shard_map` `bfloat16` parity.
    if len(jax.devices()) >= 4:
      mesh4 = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))
      shd = m3_sharding.EPasCP()
      with jax.set_mesh(mesh4):
        hc_s_rf = modeling_qwen3_8_flash_next.Qwen3_8GatedResidual(
            emb_dim,
            hc_count,
            hc_lowrank,
            use_combine=True,
            dtype=jnp.bfloat16,
            weight_dtype=jnp.bfloat16,
            sharding=shd,
            enable_hc_pallas=False,
            rngs=nnx.Rngs(19),
        )
        ns_x = jax.sharding.NamedSharding(mesh4, P("fsdp", "expert", None))
        x_s = jax.device_put(
            jnp.asarray(rng.standard_normal((2, 256, hc_dim)).astype(np.float32) * 0.4, dtype=jnp.bfloat16), ns_x
        )
        cot_s = jax.device_put(
            jnp.asarray(rng.standard_normal((2, 256, hc_dim)).astype(np.float32) * 0.3, dtype=jnp.bfloat16), ns_x
        )

        def _sharded_ref(m, xx):
          mx, ij = m.before_block(xx)
          cb = m.combine(xx, mx, ij)
          return jnp.sum(cb.astype(jnp.float32) * cot_s.astype(jnp.float32)), cb

        def _sharded_pl(xx, sc, wd, wu, wi):
          x_spec = P("fsdp", "expert", None)

          def _per_shard(x_l, sc_l, wd_l, wu_l, wi_l):
            mx_l, ij_l = hc_pallas.hc_before_block_local(
                x_l,
                sc_l,
                wd_l,
                wu_l,
                wi_l,
                emb_dim=emb_dim,
                hc_count=hc_count,
                hc_lowrank=hc_lowrank,
                eps=1e-6,
                interpret=True,
            )
            return hc_pallas.hc_combine_local(x_l, mx_l, ij_l, emb_dim=emb_dim, hc_count=hc_count, interpret=True)

          cb = jax.shard_map(
              _per_shard,
              mesh=mesh4,
              in_specs=(x_spec, P(None), P(None, None), P(None, None), P(None, None)),
              out_specs=x_spec,
              check_vma=False,
          )(xx, sc, wd, wu, wi)
          return jnp.sum(cb.astype(jnp.float32) * cot_s.astype(jnp.float32)), cb

        assert hc_s_rf.block_inject_weight is not None
        (_, cb_rf), (_, gxs_rf) = nnx.value_and_grad(_sharded_ref, argnums=(0, 1), has_aux=True)(hc_s_rf, x_s)
        (_, cb_pl), (gxs_pl, _, _, _, _) = jax.jit(
            jax.value_and_grad(_sharded_pl, argnums=(0, 1, 2, 3, 4), has_aux=True)
        )(
            x_s,
            hc_s_rf.hc_norm.scale[...],
            hc_s_rf.input_mix_weight_down.kernel[...],
            hc_s_rf.input_mix_weight_up.kernel[...],
            hc_s_rf.block_inject_weight.kernel[...],
        )
        self.assertLess(_rel_l2(cb_pl, cb_rf), 1e-2)
        self.assertLess(_rel_l2(gxs_pl, gxs_rf), 1.5e-2)

  def test_cpu_gdn_conv_pallas_interpret_cp1_and_cp2_halo_parity(self):
    """Verifies `gdn_causal_conv1d_silu_l2norm_local` multi-tile fwd+bwd and `CP=2` `ppermute` halo parity."""
    if _RUN_TPU_WAVE3_BENCH.value:
      self.skipTest("CPU test skipped when --run_tpu_wave3_bench=true.")

    rng = np.random.default_rng(2026112502)
    nk, nv, hk, hv = 2, 4, 128, 128
    c_dim = 2 * nk * hk + nv * hv

    x_loc = jnp.asarray(rng.standard_normal((2, 384, c_dim)).astype(np.float32) * 0.4)
    halo_loc = jnp.asarray(rng.standard_normal((2, 3, c_dim)).astype(np.float32) * 0.4)
    w_loc = jnp.asarray(rng.standard_normal((4, 1, c_dim)).astype(np.float32) * 0.3)
    cot_q = jnp.asarray(rng.standard_normal((2, 384, nk, hk)).astype(np.float32) * 0.2)
    cot_k = jnp.asarray(rng.standard_normal((2, 384, nk, hk)).astype(np.float32) * 0.2)
    cot_v = jnp.asarray(rng.standard_normal((2, 384, nv, hv)).astype(np.float32) * 0.2)

    def _pl_conv_loss(xx, hh, ww):
      qn, kn, va = gdn_pallas.gdn_causal_conv1d_silu_l2norm_local(
          xx, hh, ww, num_k_heads=nk, num_v_heads=nv, head_k_dim=hk, head_v_dim=hv, eps=1e-6, interpret=True
      )
      return jnp.sum(qn * cot_q) + jnp.sum(kn * cot_k) + jnp.sum(va * cot_v)

    def _rf_conv_loss(xx, hh, ww):
      x_pad = jnp.concatenate([hh, xx], axis=1)
      s_l = xx.shape[1]
      acc = x_pad[:, 0:s_l, :] * ww[0, 0, :]
      for j in range(1, 4):
        acc = acc + x_pad[:, j : j + s_l, :] * ww[j, 0, :]
      mixed = jax.nn.silu(acc)
      kd = nk * hk
      qn = modeling_qwen3_8_flash_next.l2norm(mixed[:, :, :kd].reshape(2, s_l, nk, hk), dim=-1, eps=1e-6)
      kn = modeling_qwen3_8_flash_next.l2norm(mixed[:, :, kd : 2 * kd].reshape(2, s_l, nk, hk), dim=-1, eps=1e-6)
      va = mixed[:, :, 2 * kd :].reshape(2, s_l, nv, hv)
      return jnp.sum(qn * cot_q) + jnp.sum(kn * cot_k) + jnp.sum(va * cot_v)

    vc_pl, (dx_pl, dh_pl, dw_pl) = jax.jit(jax.value_and_grad(_pl_conv_loss, argnums=(0, 1, 2)))(x_loc, halo_loc, w_loc)
    vc_rf, (dx_rf, dh_rf, dw_rf) = jax.jit(jax.value_and_grad(_rf_conv_loss, argnums=(0, 1, 2)))(x_loc, halo_loc, w_loc)
    np.testing.assert_allclose(float(vc_pl), float(vc_rf), rtol=1e-5, atol=1e-5)
    self.assertLess(_rel_l2(dx_pl, dx_rf), 1e-5)
    self.assertLess(_rel_l2(dh_pl, dh_rf), 1e-5)
    self.assertLess(_rel_l2(dw_pl, dw_rf), 1e-5)

    if len(jax.devices()) >= 4:
      mesh4 = jax.sharding.Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("fsdp", "expert"))
      x_cp = jax.device_put(
          jnp.asarray(rng.standard_normal((2, 256, c_dim)).astype(np.float32) * 0.4, dtype=jnp.bfloat16),
          jax.sharding.NamedSharding(mesh4, P("fsdp", "expert", None)),
      )
      w_cp = jax.device_put(
          jnp.asarray(rng.standard_normal((4, 1, c_dim)).astype(np.float32) * 0.3, dtype=jnp.bfloat16),
          jax.sharding.NamedSharding(mesh4, P(None, None, None)),
      )
      cq_cp = jax.device_put(
          jnp.asarray(rng.standard_normal((2, 256, nk, hk)).astype(np.float32) * 0.2, dtype=jnp.bfloat16),
          jax.sharding.NamedSharding(mesh4, P("fsdp", "expert", None, None)),
      )

      def _cp2_pl(xx, ww):
        def _shard(x_l, k_l):
          tail = x_l[:, -3:, :]
          recv = jax.lax.ppermute(tail, "expert", [(0, 1), (1, 0)])
          sid = jax.lax.axis_index("expert")
          halo = jnp.where(sid == 0, jnp.zeros_like(recv), recv)
          q_l, _, _ = gdn_pallas.gdn_causal_conv1d_silu_l2norm_local(
              x_l, halo, k_l, num_k_heads=nk, num_v_heads=nv, head_k_dim=hk, head_v_dim=hv, eps=1e-6, interpret=True
          )
          return q_l

        q_out = jax.shard_map(
            _shard,
            mesh=mesh4,
            in_specs=(P("fsdp", "expert", None), P(None, None, None)),
            out_specs=P("fsdp", "expert", None, None),
            check_vma=False,
        )(xx, ww)
        return jnp.sum(q_out.astype(jnp.float32) * cq_cp.astype(jnp.float32)), q_out

      def _cp2_rf(xx, ww):
        x_pad = jnp.pad(xx, ((0, 0), (3, 0), (0, 0)))
        acc = x_pad[:, 0:256, :].astype(jnp.float32) * ww[0, 0, :].astype(jnp.float32)
        for j in range(1, 4):
          acc = acc + x_pad[:, j : j + 256, :].astype(jnp.float32) * ww[j, 0, :].astype(jnp.float32)
        mixed = jax.nn.silu(acc.astype(jnp.bfloat16))
        q_out = modeling_qwen3_8_flash_next.l2norm(mixed[:, :, : nk * hk].reshape(2, 256, nk, hk), dim=-1, eps=1e-6)
        return jnp.sum(q_out.astype(jnp.float32) * cq_cp.astype(jnp.float32)), q_out

      with jax.set_mesh(mesh4):
        (_, q_pl), (dx_cp_pl, dw_cp_pl) = jax.jit(jax.value_and_grad(_cp2_pl, argnums=(0, 1), has_aux=True))(x_cp, w_cp)
        (_, q_rf), (dx_cp_rf, dw_cp_rf) = jax.jit(jax.value_and_grad(_cp2_rf, argnums=(0, 1), has_aux=True))(x_cp, w_cp)
      self.assertLess(_rel_l2(q_pl, q_rf), 1e-2)
      self.assertLess(_rel_l2(dx_cp_pl, dx_cp_rf), 1.5e-2)
      self.assertLess(_rel_l2(dw_cp_pl, dw_cp_rf), 1.5e-2)

  def test_tpu_mosaic_hc_and_gdn_pallas_parity_hlo_and_microbench(self):
    """Runs physical TPUv7x (`TPU v7x 2x2x1` = 8 TCs) parity, right-padded mask checks, HLO assertions, and benchmarks."""
    if not _RUN_TPU_WAVE3_BENCH.value:
      self.skipTest("Pass --run_tpu_wave3_bench=true on TPU v7x 2x2x1.")
    self.assertEqual(jax.default_backend(), "tpu")
    devices = jax.devices()
    self.assertGreaterEqual(len(devices), 8)
    rng = np.random.default_rng(2026112503)

    mesh8 = jax.sharding.Mesh(np.array(devices[:8]).reshape(2, 4), ("fsdp", "expert"))
    # 1. Full `Qwen3_8GatedDeltaNet` on `DefaultSharding()` (`CP=1`) and `EPasCP()` (`CP=4`) with BOTH
    #    `attention_mask=None` AND right-padded `attention_mask[:, -12:] = 0`!
    for use_ep_as_cp in (False, True):
      shd = m3_sharding.EPasCP() if use_ep_as_cp else m3_sharding.DefaultSharding()
      b_sz = 2 if use_ep_as_cp else 8
      s_sz = 512 if use_ep_as_cp else 128  # s_local = 128 in both cases
      with jax.set_mesh(mesh8):
        gdn_rf = modeling_qwen3_8_flash_next.Qwen3_8GatedDeltaNet(
            emb_dim=2560,
            num_k_heads=16,
            num_v_heads=48,
            head_k_dim=128,
            head_v_dim=128,
            conv_kernel_size=4,
            chunk_size=64,
            dtype=jnp.bfloat16,
            weight_dtype=jnp.bfloat16,
            sharding=shd,
            output_gate_activation="sigmoid",
            kernel_backend="auto",
            enable_conv_pallas=False,
            rngs=nnx.Rngs(31),
        )
        gdn_pl = modeling_qwen3_8_flash_next.Qwen3_8GatedDeltaNet(
            emb_dim=2560,
            num_k_heads=16,
            num_v_heads=48,
            head_k_dim=128,
            head_v_dim=128,
            conv_kernel_size=4,
            chunk_size=64,
            dtype=jnp.bfloat16,
            weight_dtype=jnp.bfloat16,
            sharding=shd,
            output_gate_activation="sigmoid",
            kernel_backend="auto",
            enable_conv_pallas=True,
            rngs=nnx.Rngs(31),
        )
        h_spec = P("fsdp", "expert", None) if use_ep_as_cp else P(("fsdp", "expert"), None, None)
        m_spec = P("fsdp", "expert") if use_ep_as_cp else P(("fsdp", "expert"), None)
        h_in = jax.device_put(
            jnp.asarray(rng.standard_normal((b_sz, s_sz, 2560)).astype(np.float32) * 0.2, dtype=jnp.bfloat16),
            jax.sharding.NamedSharding(mesh8, h_spec),
        )
        cot_g = jax.device_put(
            jnp.asarray(rng.standard_normal((b_sz, s_sz, 2560)).astype(np.float32) * 0.2, dtype=jnp.bfloat16),
            jax.sharding.NamedSharding(mesh8, h_spec),
        )
        mask_np = np.ones((b_sz, s_sz), dtype=np.int32)
        mask_np[:, -12:] = 0
        amask = jax.device_put(jnp.asarray(mask_np), jax.sharding.NamedSharding(mesh8, m_spec))

        for mask_arg in (None, amask):
          gdef_r, ps_r, rst_r = nnx.split(gdn_rf, nnx.Param, ...)
          gdef_p, ps_p, rst_p = nnx.split(gdn_pl, nnx.Param, ...)

          def _make_gdn_step(gdef, rst, m_val, cot_val):
            @jax.jit
            def _step(ps, hh):
              def _loss(p_i, x_i):
                mod = nnx.merge(gdef, p_i, rst)
                y = mod(x_i, attention_mask=m_val)
                return jnp.sum(y.astype(jnp.float32) * cot_val.astype(jnp.float32)), y

              return jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(ps, hh)

            return _step

          step_rf = _make_gdn_step(gdef_r, rst_r, mask_arg, cot_g)
          step_pl = _make_gdn_step(gdef_p, rst_p, mask_arg, cot_g)
          hlo_pl = step_pl.lower(ps_p, h_in).as_text()
          self.assertIn("gdn_conv_silu_l2norm_fwd", hlo_pl)

          (_, y_rf), (dp_rf, dh_rf) = step_rf(ps_r, h_in)
          (_, y_pl), (dp_pl, dh_pl) = step_pl(ps_p, h_in)
          jax.block_until_ready((y_rf, y_pl, dh_rf, dh_pl))
          self.assertLess(_rel_l2(y_pl, y_rf), 1.5e-2)
          self.assertLess(_rel_l2(dh_pl, dh_rf), 2.5e-2)
          dp_pl_any, dp_rf_any = cast(Any, dp_pl), cast(Any, dp_rf)
          self.assertLess(_rel_l2(dp_pl_any["conv1d"]["kernel"][...], dp_rf_any["conv1d"]["kernel"][...]), 2.5e-2)

    # 2. Full `Qwen3_8DecoderLayer` (`EPasCP()`, `fp8_full`, `use_hc_pallas=True`, `use_gdn_conv_pallas=True`) HLO + parity!
    common_args = [
        "",
        f"{maxtext_globals.MAXTEXT_CONFIGS_DIR}/base.yml",
        "model_name=qwen3.8-flash-next",
        "override_model_config=True",
        "run_name=tpu_pr6_layer",
        "skip_jax_distributed_system=True",
        "enable_checkpointing=False",
        "dtype=bfloat16",
        "weight_dtype=bfloat16",
        "quantization=fp8_full",
        "moe_fp8=True",
        "num_experts=16",
        "num_experts_per_tok=4",
        "moe_ep_allgather_tc=True",
        "moe_tc_ragged_flatten_block_size=128",
        "moe_fused_bwd_quant=True",
        "sharding_preset=ep_as_cp",
    ]
    cfg_off = pyconfig.initialize(common_args)
    cfg_on = pyconfig.initialize(common_args + ["use_hc_pallas=True", "use_gdn_conv_pallas=True"])
    shd8 = m3_sharding.EPasCP()
    with jax.set_mesh(mesh8):
      lyr_off = modeling_qwen3_8_flash_next.Qwen3_8DecoderLayer(
          cfg_off, layer_idx=0, include_ple=False, sharding=shd8, rngs=nnx.Rngs(77)
      )
      lyr_on = modeling_qwen3_8_flash_next.Qwen3_8DecoderLayer(
          cfg_on, layer_idx=0, include_ple=False, sharding=shd8, rngs=nnx.Rngs(77)
      )
      h8 = jax.device_put(
          jnp.asarray(rng.standard_normal((2, 1024, 10240)).astype(np.float32) * 0.15, dtype=jnp.bfloat16),
          jax.sharding.NamedSharding(mesh8, P("fsdp", "expert", None)),
      )
      cot8 = jax.device_put(
          jnp.asarray(rng.standard_normal((2, 1024, 10240)).astype(np.float32) * 0.15, dtype=jnp.bfloat16),
          jax.sharding.NamedSharding(mesh8, P("fsdp", "expert", None)),
      )

      def _run_layer(layer_mod):
        gdef, p_state, rest = nnx.split(layer_mod, nnx.Param, ...)

        @jax.jit
        def _step(params, hh):
          def _loss(ps, x_in):
            m = nnx.merge(gdef, ps, rest)
            y = m(x_in)
            return jnp.sum(y.astype(jnp.float32) * cot8.astype(jnp.float32)), y

          return jax.value_and_grad(_loss, argnums=(0, 1), has_aux=True)(params, hh)

        hlo_txt = _step.lower(p_state, h8).as_text()
        compiled = _step.lower(p_state, h8).compile()
        (loss_v, y_v), (dp_v, dh_v) = compiled(p_state, h8)
        jax.block_until_ready((loss_v, y_v, dp_v, dh_v))
        return y_v, dh_v, hlo_txt

      y_off, dh_off, _ = _run_layer(lyr_off)
      y_on, dh_on, hlo_on = _run_layer(lyr_on)
      self.assertIn("hc_before_block_fwd", hlo_on)
      self.assertIn("hc_combine_fwd", hlo_on)
      self.assertIn("gdn_conv_silu_l2norm_fwd", hlo_on)
      self.assertLess(_rel_l2(y_on, y_off), 2.5e-2)
      self.assertLess(_rel_l2(dh_on, dh_off), 4e-2)

    # 3. Single-TC production-shape `[2, 8192, 10240]` microbenchmarks.
    mesh1 = jax.sharding.Mesh(np.array(devices[:1]), ("data",))
    with jax.set_mesh(mesh1):
      x_gdn = jnp.asarray(rng.standard_normal((2, 8192, 10240)).astype(np.float32) * 0.2, dtype=jnp.bfloat16)
      h_gdn = jnp.asarray(rng.standard_normal((2, 3, 10240)).astype(np.float32) * 0.2, dtype=jnp.bfloat16)
      w_gdn = jnp.asarray(rng.standard_normal((4, 1, 10240)).astype(np.float32) * 0.2, dtype=jnp.bfloat16)
      cq_g = jnp.ones((2, 8192, 16, 128), dtype=jnp.bfloat16) * 0.02
      ck_g = jnp.ones((2, 8192, 16, 128), dtype=jnp.bfloat16) * 0.02
      cv_g = jnp.ones((2, 8192, 48, 128), dtype=jnp.bfloat16) * 0.02
      fn_gdn_pl = jax.jit(
          jax.value_and_grad(
              lambda xx, hh, ww: sum(
                  jnp.sum(o.astype(jnp.float32) * c.astype(jnp.float32))
                  for o, c in zip(
                      gdn_pallas.gdn_causal_conv1d_silu_l2norm_local(
                          xx, hh, ww, num_k_heads=16, num_v_heads=48, head_k_dim=128, head_v_dim=128
                      ),
                      (cq_g, ck_g, cv_g),
                      strict=True,
                  )
              ),
              argnums=(0, 1, 2),
          )
      )
      jax.block_until_ready(fn_gdn_pl(x_gdn, h_gdn, w_gdn))
      t0 = time.perf_counter()
      for _ in range(5):
        out_g = fn_gdn_pl(x_gdn, h_gdn, w_gdn)
      jax.block_until_ready(out_g)
      print(f"[BENCH_64K_GDN_CONV_PALLAS] pallas={(time.perf_counter() - t0) * 1000.0 / 5.0:.3f} ms")


if __name__ == "__main__":
  absltest.main()

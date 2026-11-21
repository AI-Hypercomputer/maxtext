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

"""Offline TPU v7x (`tpu7x:4x4x4`) AOT compilation and HLO scan-residual tests."""

from __future__ import annotations

import json
import os
from unittest import mock

from absl.testing import absltest

from maxtext.experimental.flint40b import aot_compile
from maxtext.experimental.flint40b import flint_config


class FlintAotCompileTest(absltest.TestCase):

  def test_scan_residual_auditor_detects_leaked_gathered_weights(self) -> None:
    xla_dir = self.create_tempdir().full_path
    cfg = flint_config.flint_40b_config(
        seq_len=8192,
        global_batch_size=256,
        fsdp_size=16,
        cp_size=1,
        ep_size=8,
    )
    hlo_path = os.path.join(
        xla_dir,
        "module_0001.jit__train_step.after_optimizations_before_buffer_assignment.txt",
    )
    synthetic_hlo = "\n".join(
        [
            "HloModule jit__train_step",
            "%cond (arg: (s32[])) -> pred[] {",
            "  %c = s32[] constant(11)",
            "  ROOT %p = pred[] constant(true)",
            "}",
            ("%body (arg: (s32[], bf16[11,16,4096], bf16[11,2048,4096])) ->" " (s32[]) {"),
            ("  %p0 = (s32[], bf16[11,2,8192,2048], bf16[11,32,128,1024]," " bf16[11,32,2048,1024]) parameter(0)"),
            (
                "  %sp = bf16[2,8192,2048] custom-call(),"
                ' custom_call_target="tpu_custom_call",'
                ' backend_config="splash dkv_head_stitch: false"'
            ),
            (
                "  %rd = bf16[16384,1024] fusion(bf16[16384,2048] %x,"
                " bf16[32,2048,1024] %w, s32[32] %gs), kind=kCustom,"
                ' backend_config={"fusion_backend_config":{"is_ragged_dot":true}}'
            ),
            "  ROOT %t = (s32[], bf16[11,16,4096]) tuple(%c0, %w_sharded)",
            "}",
            "ENTRY %main () -> (s32[]) {",
            ("  %w = (s32[], bf16[11,16,4096], bf16[11,2048,4096]) while(%init)," " condition=%cond, body=%body"),
            "  ROOT %r = (s32[]) tuple(%w)",
            "}",
        ]
    )
    with open(hlo_path, "w", encoding="utf-8") as f:
      f.write(synthetic_hlo)

    audit = aot_compile.audit_flint_hlo(xla_dir, cfg)
    self.assertTrue(audit["hlo_found"])
    self.assertTrue(audit["has_splash"])
    self.assertFalse(audit["has_dkv_head_stitch"])
    self.assertTrue(audit["has_ragged_dot"])
    self.assertEqual(
        audit["gathered_weight_scan_residuals"],
        ["bf16[11,2048,4096]", "bf16[11,32,2048,1024]"],
    )

  def test_hlo_auditor_gmm_v2_copy_gib_sc_pins_and_fused_tiled_leaks(
      self,
  ) -> None:
    xla_dir = self.create_tempdir().full_path
    cfg = flint_config.flint_40b_config(
        seq_len=8192,
        global_batch_size=256,
        fsdp_size=16,
        cp_size=1,
        ep_size=8,
    )
    hlo_path = os.path.join(
        xla_dir,
        "module_0002.jit__train_step.after_optimizations_before_buffer_assignment.txt",
    )
    synthetic_hlo = "\n".join(
        [
            "HloModule jit__train_step",
            "%cond (arg: (s32[])) -> pred[] {",
            "  %c = s32[] constant(11)",
            "  ROOT %p = pred[] constant(true)",
            "}",
            "%sc0_ag (p: bf16[16,4096]) -> bf16[2048,4096] {",
            "  %p0 = bf16[16,4096] parameter(0)",
            "  ROOT %ag = bf16[2048,4096] all-gather(%p0), dimensions={0}",
            "}",
            "%sc1_rs (p: bf16[2048,4096]) -> bf16[16,4096] {",
            "  %p0 = bf16[2048,4096] parameter(0)",
            "  ROOT %rs = bf16[16,4096] reduce-scatter(%p0), dimensions={0}",
            "}",
            ("%body (arg: (s32[], bf16[11,2048,5120]," " bf16[11,32,256,8,1024])) -> (s32[], bf16[16,4096]) {"),
            ("  %p0 = (s32[], bf16[11,2048,5120]," " bf16[11,32,256,8,1024]) parameter(0)"),
            (
                "  %gmm = bf16[16384,1024] custom-call(),"
                ' custom_call_target="tpu_custom_call",'
                ' metadata={op_name="jit(_train_step)/gmm_v2/'
                "splash_mha_dkv_head_stitch/dense_gather_reduce_sc/"
                "sc_counting_sort_pass1/router_topk_fwd/"
                'pallas_qk_prologue_fwd/lb_kv_vmem_copy"}'
            ),
            "  %cp = bf16[2048,4096] copy(%x)",
            (
                "  %ags = ((bf16[16,4096]), bf16[2048,4096], u32[])"
                " async-start(%s), calls=%sc0_ag,"
                ' backend_config={"sparse_core_config":{"core_ids":[0]}}'
            ),
            (
                "  %rss = ((bf16[2048,4096]), bf16[16,4096], u32[])"
                " async-start(%g), calls=%sc1_rs,"
                ' backend_config={"sparse_core_config":{"core_ids":[1]}}'
            ),
            "  ROOT %t = (s32[], bf16[16,4096]) tuple(%c0, %w_sharded)",
            "}",
            "ENTRY %main () -> (s32[], bf16[16,4096]) {",
            ("  %w = (s32[], bf16[11,2048,5120], bf16[11,32,256,8,1024])" " while(%init), condition=%cond, body=%body"),
            "  %unpin = bf16[2048,4096] all-gather(%u0), dimensions={0}",
            "  ROOT %r = (s32[], bf16[16,4096]) tuple(%c, %u0)",
            "}",
        ]
    )
    with open(hlo_path, "w", encoding="utf-8") as f:
      f.write(synthetic_hlo)

    audit = aot_compile.audit_flint_hlo(xla_dir, cfg)
    self.assertTrue(audit["hlo_found"])
    self.assertTrue(audit["has_gmm_v2"])
    self.assertTrue(audit["has_dkv_head_stitch"])
    self.assertTrue(audit["has_sc_gather_reduce"])
    self.assertTrue(audit["has_sc_counting_sort"])
    self.assertTrue(audit["has_router_topk"])
    self.assertTrue(audit["has_qk_prologue"])
    self.assertTrue(audit["has_lb_kv_reorder"])
    self.assertGreater(audit["copy_gib_step"], 0.1)
    self.assertEqual(
        audit["sc_collectives_by_pin_step"],
        {"sc0": 11, "sc1": 11, "unpinned": 1},
    )
    self.assertEqual(
        audit["gathered_weight_scan_residuals"],
        ["bf16[11,2048,5120]", "bf16[11,32,256,8,1024]"],
    )

  def test_resolve_libtpu_args(self) -> None:
    sublane_flag = "--xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false"
    cfg_256k = flint_config.flint_40b_config(**flint_config.default_recipe_overrides())
    opt = aot_compile.resolve_libtpu_args(cfg_256k, optimized_libtpu_args=True, extra_libtpu_args="--xla_foo=1")
    self.assertEqual(
        opt,
        " ".join(
            [
                aot_compile.BASE_LIBTPU_ARGS,
                flint_config.optimized_v7x_libtpu_args(
                    global_batch_size=32,
                    grad_accum_steps=1,
                    fsdp_size=32,
                    cp_size=4,
                ),
                "--xla_foo=1",
            ]
        ),
    )
    self.assertNotIn(sublane_flag, opt.split())
    self.assertEqual(
        aot_compile.resolve_libtpu_args(
            cfg_256k,
            optimized_libtpu_args=False,
            extra_libtpu_args="--xla_foo=1",
        ),
        f"{aot_compile.BASE_LIBTPU_ARGS} --xla_foo=1",
    )
    cfg_8k = flint_config.flint_40b_config(
        **flint_config.optimized_64c_recipe_overrides(seq_len=8192, global_batch_size=1024)
    )
    self.assertIn(
        sublane_flag,
        aot_compile.resolve_libtpu_args(cfg_8k, optimized_libtpu_args=True, extra_libtpu_args="").split(),
    )
    opts = aot_compile.parse_xla_compiler_options(opt)
    self.assertEqual(opts["xla_tpu_scoped_vmem_limit_kib"], "65536")
    self.assertEqual(opts["xla_tpu_enable_sparse_core_collective_offload_all_gather"], "true")
    self.assertEqual(opts["xla_foo"], "1")

  def test_sync_compiler_and_dump_flags_exports_env(self) -> None:
    dump_dir = self.create_tempdir().full_path
    libtpu_args = "--xla_tpu_scoped_vmem_limit_kib=65536 --xla_foo=bar"
    with mock.patch.dict(os.environ):
      with mock.patch.object(aot_compile.xla_bridge, "_discover_and_register_pjrt_plugins") as discover:
        opts, _ = aot_compile.sync_compiler_and_dump_flags(libtpu_args, xla_dump_dir=dump_dir)
      self.assertEqual(os.environ["LIBTPU_INIT_ARGS"], libtpu_args)
      self.assertIn(f"--xla_dump_to={dump_dir}", os.environ["XLA_FLAGS"])
    discover.assert_called_once_with()
    self.assertEqual(opts, {"xla_tpu_scoped_vmem_limit_kib": "65536", "xla_foo": "bar"})

  def test_full_40b_64c_aot_compile_and_hlo_audit(self) -> None:
    base_dir = self.create_tempdir().full_path
    specs = [
        {
            "exp_name": "flint_64c_16k_pdb1_ep8_3d",
            "out_dir": os.path.join(base_dir, "16k_pdb1_ep8_3d"),
            "config_overrides": {
                "seq_len": 16384,
                "global_batch_size": 128,
                "grad_accum_steps": 1,
                "fsdp_size": 8,
                "cp_size": 2,
                "ep_size": 8,
            },
        },
    ]
    summaries = aot_compile.run_batch_specs(
        specs,
        default_topology="tpu7x:4x4x4",
        default_optimized_libtpu_args=False,
        default_max_hbm_gib=94.0,
        keep_hlo_dumps=False,
    )
    self.assertLen(summaries, 1)

    for spec, summary in zip(specs, summaries, strict=True):
      summary_file = os.path.join(str(spec["out_dir"]), "aot_summary.json")
      self.assertTrue(os.path.isfile(summary_file))
      self.assertFalse(os.path.exists(os.path.join(str(spec["out_dir"]), "xla")))
      with open(summary_file, "r", encoding="utf-8") as f:
        persisted = json.load(f)
      self.assertEqual(persisted["exp_name"], spec["exp_name"])
      self.assertLessEqual(summary["total_aot_hbm_gib"], 94.0)
      self.assertTrue(summary["passes_hbm_gate"])
      self.assertTrue(summary["hlo_found"])
      self.assertTrue(summary["has_splash"])
      self.assertTrue(summary["has_ragged_dot"])
      self.assertTrue(summary["tpu_comp_env_verified"])
      self.assertEqual(summary["gathered_weight_scan_residuals"], [])


if __name__ == "__main__":
  absltest.main()

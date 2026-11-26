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

"""AOT compilation & HLO verification test for 64-chip Qwen3.5-397B-A17B on TPU v7x."""

from __future__ import annotations

import json
import os
import tempfile

from absl.testing import absltest
import jax
import jax.numpy as jnp

from maxtext.experimental import qwen_35_64c as qwen35_jax
from maxtext.experimental.qwen_35_64c import aot_compile
from maxtext.experimental.qwen_35_64c.ext_kernels import rmsnorm_tc


class Qwen35AotCompileTest(absltest.TestCase):

  def test_pallas_rmsnorm_and_scatter_dw_tpu_v7x_compile(self):
    cfg = qwen35_jax.full_config()
    mesh = qwen35_jax.create_mesh(cfg, aot_topology="tpu7x:4x4x4")
    dev0 = jax.sharding.SingleDeviceSharding(mesh.devices.flat[0])

    def _fn(x_emb, x_q, x_gdn, z_gdn, s_emb, s_q, s_gdn, slots, rows, cnt):
      def _loss(xe, xq, xg, zg, se, sq, sg):
        ye = rmsnorm_tc.rms_norm_zero_centered_pallas(xe, se)
        yq = rmsnorm_tc.rms_norm_zero_centered_pallas(xq, sq)
        yg = rmsnorm_tc.rms_norm_gated_pallas(xg, zg, sg)
        return jnp.sum(ye) + jnp.sum(yq) + jnp.sum(yg)

      loss, grads = jax.value_and_grad(_loss, argnums=(0, 1, 2, 3, 4, 5, 6))(x_emb, x_q, x_gdn, z_gdn, s_emb, s_q, s_gdn)
      dw = rmsnorm_tc.scatter_dw_vmem_tc(slots, rows, jnp.int32(0), cnt, num_slots=65536)
      return loss, grads, dw

    specs = (
        jax.ShapeDtypeStruct((16384, 4096), jnp.bfloat16, sharding=dev0),
        jax.ShapeDtypeStruct((65536, 256), jnp.bfloat16, sharding=dev0),
        jax.ShapeDtypeStruct((131072, 128), jnp.bfloat16, sharding=dev0),
        jax.ShapeDtypeStruct((131072, 128), jnp.bfloat16, sharding=dev0),
        jax.ShapeDtypeStruct((4096,), jnp.float32, sharding=dev0),
        jax.ShapeDtypeStruct((256,), jnp.float32, sharding=dev0),
        jax.ShapeDtypeStruct((128,), jnp.float32, sharding=dev0),
        jax.ShapeDtypeStruct((104448,), jnp.int32, sharding=dev0),
        jax.ShapeDtypeStruct((104448,), jnp.float32, sharding=dev0),
        jax.ShapeDtypeStruct((), jnp.int32, sharding=dev0),
    )
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
      compiled = jax.jit(_fn).lower(*specs).compile()
    self.assertIsNotNone(compiled)

  def test_full_64chip_aot_compile(self):
    base_out = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or tempfile.mkdtemp(prefix="qwen35_aot_")
    out_dir = os.path.join(base_out, "qwen35_aot_64c")
    os.makedirs(out_dir, exist_ok=True)

    summary = aot_compile.run_aot_compile(
        [
            "--exp_name=qwen35_397b_64c",
            f"--out_dir={out_dir}",
            "--topology=tpu7x:4x4x4",
        ]
    )
    print("=== AOT_SUMMARY_JSON ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    self.assertTrue(
        summary.get("passes_86_5_gib_gate"),
        f"AOT HBM {summary.get('total_aot_hbm_gib')} GiB exceeds 86.5 GiB" " gate!",
    )
    self.assertTrue(
        summary.get("tpu_comp_env_verified"),
        "TpuCompilationEnvironment flags were not synced into C++ XLA!",
    )
    self.assertGreaterEqual(
        summary.get("max_overlap_stats", {}).get("modules_transformed", 0),
        1,
        f"MaxOverlapTransform did not run: {summary.get('max_overlap_stats')}",
    )
    self.assertTrue(summary.get("hlo_found"), "Scheduled HLO dump not found!")
    self.assertFalse(
        summary.get("has_lossy_const"),
        "HLO dump contains truncated '...' constant arrays!",
    )
    self.assertEqual(
        summary.get("nested_whiles"),
        [],
        f"Unexpected nested while loops in HLO: {summary.get('nested_whiles')}",
    )


if __name__ == "__main__":
  absltest.main()
